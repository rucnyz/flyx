// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A round of several sequences: each sequence drafts on its own buffers; the target's layers run on
// all the rows, the weights read once; what reads a sequence's own state (the gated delta nets' convolution and
// recurrence, rotary and the K/V append, attention, choosing and accepting tokens, the commit, the drafters'
// contexts) runs on its slice of the rows, from and into its own state.  The 1..16-row kernels give a row the same
// bits whatever the other rows are, so every sequence gets what a round of its own gives it.  When fused, each
// sequence's GDN mixer runs gdn_round straight into its slice of the shared presplit buffer before one output GEMM;
// attention writes its slice to act before the shared output GEMM (same bits as the generic path).
#include "backends/cuda/regions/layers.h"

namespace flyx::cuda {
namespace {

void embed_rows(const BatchPass& bp, const EmbedBlock& e) {
  const RowView& v = bp.v;
  for (size_t i = 0; i < bp.seq.size(); ++i) {
    const Pass& ps = bp.seq[i];
    build_rows(ps.s->P, ps.s->drafts, -1, v.b.tokens + bp.first[i], ps.rows, v.L.st);
  }
  embed_tokens(e, v.b.tokens, v.b.x, bp.rows, v.L.st);
}

void gdn_rows(const BatchPass& bp, const GdnBlock& g, bool fused) {
  const RowView& v = bp.v;
  const GdnDims gd = g.gd;
  const int H = g.hidden, R = bp.rows, C = gd.C(), vd = g.v_width();
  cudaStream_t st = v.L.st;
  norm_and_project(v, R, g.norm, g.w_in, g.n_in(), H, fused);
  if (fused) {
    const bool q8 = g.w_out.q8_inputs();
    bool all_fit = true;
    for (const Pass& ps : bp.seq) {
      const GdnBuf& sb = ps.s->gdn[size_t(g.state)];
      if (!gdn_round_fits(gd, ps.rows, sb.S, sb.s_stride, v.L, q8)) {
        all_fit = false;
        break;
      }
    }
    if (all_fit) {
      for (size_t i = 0; i < bp.seq.size(); ++i) {
        const Pass& ps = bp.seq[i];
        flyx_seq* s = ps.s;
        GdnBuf& sb = s->gdn[size_t(g.state)];
        const float* proj = v.b.proj + size_t(bp.first[i]) * v.ld_proj;
        const int next = ps.p->width == 0 ? kSlotNext : -1;
        gdn_round(proj, v.ld_proj, g.conv_w, sb.win, sb.w_stride, sb.S, sb.s_stride, next, g.A_log, g.dt_bias,
                  g.gated.w, g.gated.eps, sb.replay, s->P, ps.rows, nullptr, v.L, q8, R, bp.first[i]);
      }
      gemm_presplit(g.w_out, H, vd, v.b.x, H, R, true, v.L);
      return;
    }
  }
  for (size_t i = 0; i < bp.seq.size(); ++i) {  // each sequence's convolution and recurrence, as in a round of its own
    const Pass& ps = bp.seq[i];
    flyx_seq* s = ps.s;
    GdnBuf& sb = s->gdn[size_t(g.state)];
    const float* proj = v.b.proj + size_t(bp.first[i]) * v.ld_proj;
    const float* b = proj + C + vd;
    const float* a = b + gd.nv;
    const bool direct = ps.p->width == 0;  // plain decoding writes the next window and state in the pass
    const int next = direct ? kSlotNext : -1;
    const GdnBufs& bf = sb.replay;
    gdn_conv(proj, v.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, next, kSkipIfStopped, bf, kSlotCur, s->P,
             ps.rows, st);
    gdn_prep(bf, kSlotCur, b, a, v.ld_proj, g.A_log, g.dt_bias, s->P, ps.rows, st);
    gdn_recur(sb.S, sb.s_stride, kSlotCur, next, bf, kSlotCur, s->P, v.b.gdn_o + size_t(bp.first[i]) * vd, ps.rows,
              nullptr, direct ? kSkipIfStopped : kSkipNever, st);
  }
  gdn_out_project(v, g, R, fused);
}

void attn_rows(const BatchPass& bp, const AttnBlock& a, bool fused) {
  const RowView& v = bp.v;
  const int H = a.hidden, R = bp.rows, nh = a.nh, nkv = a.nkv, hd = a.hd, w = nh * hd;
  const int ldu = bp.seq[0].s->m->need.lora;
  cudaStream_t st = v.L.st;
  norm_and_project(v, R, a.norm, a.w_in, a.n_in(), H, fused);
  if (a.lora) {
    const int out[3] = {w, nkv * hd, nkv * hd};
    lora_in(v, a.lora, 3, out, fused ? nullptr : v.b.h, H, H, R, ldu, v.L);
  }
  for (size_t i = 0; i < bp.seq.size(); ++i) {  // rotary, the K/V append and attention: each sequence's own
    const Pass& ps = bp.seq[i];
    flyx_seq* s = ps.s;
    const KvBuf& kb = s->kv[size_t(a.kv)];
    const float* proj = v.b.proj + size_t(bp.first[i]) * v.ld_proj;
    float* q = v.b.qbuf + size_t(bp.first[i]) * w;
    attn_prep(proj, v.ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
              a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, ps.rows, s->P, kPosPosition, false, q, kb.ref(), st, 0,
              nullptr, 0, CanvasKeys{}, a.mrope());
    if (ps.reads_cache(kb)) {
      s->stage_before(kb, 0, ps.rows, st);  // all its positions in host memory
      attention(q, kb.staged(), nh, nkv, hd, ps.rows, true, 0, 1.f / std::sqrt(float(hd)), s->P,
                a.gate ? proj + w : nullptr, v.ld_proj, v.b.act + size_t(bp.first[i]) * w, w, v.b.part, kRoundSplits,
                v.L);
    }
    if (ps.reads_copy(kb)) {
      const bool stage_copy = ps.reads_copy_only(kb) && (!s->uses_shared_prefill() || s->copy_dev);
      if (stage_copy) s->copy_before(kb, ps.rows, st);  // a copy in host memory, all of it
      attention_strata(q, kb.copy_read(stage_copy), s->strata, nh, nkv, hd, ps.rows, 1.f / std::sqrt(float(hd)), s->P,
                       a.gate ? proj + w : nullptr, v.ld_proj, v.b.act + size_t(bp.first[i]) * w, w, v.b.part,
                       kRoundSplits, v.L, ps.strata == kStrataBoth ? s->strata_from : 0);
    }
  }
  gemm(v.b.act, w, a.w_out, H, w, v.b.x, H, R, true, v.L);
  if (a.lora) lora_out(v, a.lora, FLYX_LIN_O, v.b.act, w, w, H, R, ldu, v.L);
}

// The head over all the rows; each sequence's penalties and choice on its rows.  The certified head when every
// sequence chooses with it (it gives the token the full head gives), else the full head for all.
void head_rows(const BatchPass& bp, const HeadBlock& h) {
  const RowView& v = bp.v;
  flyx_seq* lead = bp.seq[0].s;
  flyx_bmodel* m = lead->m;
  const int V = h.vocab;
  cudaStream_t st = v.L.st;
  head_logits(h, v, 0, bp.rows, bp.cert, lead->cert, m->opt.kernels);
  for (size_t i = 0; i < bp.seq.size(); ++i) {
    const Pass& ps = bp.seq[i];
    flyx_seq* s = ps.s;
    float* lg = v.logits + size_t(bp.first[i]) * V;
    if (ps.pen) penalize_rows(lg, V, V, ps.rows, s->G, s->reply, s->drafts, st);
    choose_rows(lg, V, V, ps.rows, s->P, 1, s->G, s->mask, s->cz, s->ci, s->argv, s->argi,
                v.b.choice + bp.first[i], st);
  }
}

// Lockstep batched MTP drafting across all sequences drafting on the same MtpBlock: reads MTP's 5 projection
// matrices once per draft step across active sequences instead of once per sequence per step, while keeping each
// sequence's attention and 1-row draft head pick identical bit for bit to solo mtp_draft().
void mtp_draft_rows(const BatchPass& bp, const MtpBlock& b, const std::vector<size_t>& members, bool fused) {
  const RowView& v = bp.v;
  const flyx_bmodel* m = bp.seq[members[0]].s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const auto& a = block<AttnBlock>(m, b.attn);
  const auto& f = block<MlpBlock>(m, b.mlp);
  const int H = a.hidden, q_w = a.nh * a.hd;
  cudaStream_t st = v.L.st;
  const Launch& L = v.L;

  int max_w = 0;
  for (size_t idx : members) {
    const Pass& ps = bp.seq[idx];
    build_rows(ps.s->P, nullptr, 0, ps.s->mtok, 1, st);  // mtok[0] = anchor
    max_w = std::max(max_w, ps.p->nodes[0].arg0);
  }

  std::vector<size_t> active;
  active.reserve(members.size());
  for (int i = 0; i < max_w; ++i) {
    active.clear();
    for (size_t idx : members)
      if (i < bp.seq[idx].p->nodes[0].arg0) active.push_back(idx);
    const int A = static_cast<int>(active.size());
    for (int j = 0; j < A; ++j) {
      flyx_seq* s = bp.seq[active[size_t(j)]].s;
      float* hj = v.b.h + size_t(j) * H;
      if (E.blk.data) embed_blocks(s->mtok + i, E.blk, H, hj, 1, st);
      mtp_gather(s->P, kPosPosition, false, i == 0 ? nullptr : s->mg, s->ring, nullptr, 0, s->mtok + i, E.table, hj,
                 b.norm_emb.w, b.norm_hidden.w, b.norm_emb.offset, b.norm_emb.eps, H, 1,
                 v.b.mcat + size_t(j) * 2 * H, st);
    }
    gemm(v.b.mcat, 2 * H, b.fc, H, 2 * H, v.b.mx, H, A, false, L);
    norm_project(v.b.mx, v.b.h, v.b.proj, v.ld_proj, A, a.norm, a.w_in, a.n_in(), H, L, fused);
    for (int j = 0; j < A; ++j) {
      flyx_seq* s = bp.seq[active[size_t(j)]].s;
      const KvBuf& kv = s->kv[size_t(a.kv)];
      const float* proj_j = v.b.proj + size_t(j) * v.ld_proj;
      float* q_j = v.b.qbuf + size_t(j) * q_w;
      float* act_j = v.b.act + size_t(j) * q_w;
      mtp_draft_attention(s, a, kv, i - 1, proj_j, v.ld_proj, q_j, act_j, v.b.part, L);
    }
    gemm(v.b.act, q_w, a.w_out, H, q_w, v.b.mx, H, A, true, L);
    mlp_body(v.b.mx, f, A, v.b.h, v.b.proj, v.ld_proj, v.b.act, v.ld_act, L, fused);
    rmsnorm(v.b.mx, H, b.norm.w, b.norm.offset, b.norm.eps, v.b.h, H, A, H, st);
    if (A >= 2) {
      // Read the 1-bit or full draft head once across all A active sequences while keeping rough scores bit-identical
      // to 1-row solo mtp_draft() (exact_1row = true).
      draft_head_pick(head, v.b.h, A, 1, v.b.choice, nullptr, pick_scratch(bp.seq[active[0]].s, v.logits), L, true);
      for (int j = 0; j < A; ++j) {
        const Pass& ps = bp.seq[active[size_t(j)]];
        flyx_seq* s = ps.s;
        if (i + 1 < ps.p->nodes[0].arg0)
          CK(cudaMemcpyAsync(s->mg, v.b.h + size_t(j) * H, size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
        copy_ints(v.b.choice + j, s->mtok + i + 1, 1, st);
        mtp_oracle(s->P, s->oracle, i, s->mtok, st);
      }
    } else {
      const Pass& ps = bp.seq[active[0]];
      flyx_seq* s = ps.s;
      if (i + 1 < ps.p->nodes[0].arg0)
        CK(cudaMemcpyAsync(s->mg, v.b.h, size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
      draft_head_pick(head, v.b.h, 1, 1, s->mtok + i + 1, nullptr, s, L);
      mtp_oracle(s->P, s->oracle, i, s->mtok, st);
    }
  }
  for (size_t idx : members) {
    const Pass& ps = bp.seq[idx];
    copy_ints(ps.s->mtok + 1, ps.s->drafts, ps.p->nodes[0].arg0, st);
  }
}

// Batched DFlash2 drafting across sequences sharing a DFlashBlock: runs up to FLYX_MAX_ROWS / block sequences at once
// so all 5 DFlash2 layers' projection weights, the draft head (for nr >= 2), and the selector hidden projection are
// read once per group instead of once per sequence, while keeping per-sequence grouped_conv, attention, and
// selector_walk bit-identical to solo draft_block().
void draft_block_rows(const BatchPass& bp, const DFlashBlock& b, const std::vector<size_t>& members, bool fused) {
  const RowView& v = bp.v;
  const flyx_bmodel* m = bp.seq[members[0]].s->m;
  const BatchDFlash& df = m->batch->dflash;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const int B = b.block, H = b.hidden, nh = b.nh, nkv = b.nkv, hd = b.hd, I = b.inter;
  const int Tp = b.conv_taps, G = H / b.conv_group, TG = Tp * G, nqkv = (nh + 2 * nkv) * hd;
  const size_t max_group = static_cast<size_t>(FLYX_MAX_ROWS / B);
  cudaStream_t st = v.L.st;
  const Launch& L = v.L;

  for (size_t off = 0; off < members.size();) {
    const size_t n = std::min(max_group, members.size() - off);
    if (n < 2) {
      const Pass& ps = bp.seq[members[off]];
      Pass own = ps;
      own.v = rows_of(ps.s);
      own.v.L.st = st;
      ps.p->impl[0]->fn(own, ps.p->nodes[0], ps.p->block[0]);
      ++off;
      continue;
    }
    const int R = static_cast<int>(n) * B;
    for (size_t g = 0; g < n; ++g) {
      flyx_seq* s = bp.seq[members[off + g]].s;
      build_rows(s->P, nullptr, b.mask_token, df.dtok + g * B, B, st);
    }
    embed_tokens(E, df.dtok, df.dx, R, st);
    for (int j = 0; j < b.layers(); ++j) {
      const auto& l = block<DFlashLayer>(m, b.first_layer + j);
      rmsnorm(df.dx, H, l.norm.w, l.norm.offset, l.norm.eps, df.dh, H, R, H, st);
      gemm(df.dh, H, l.attn_proj, 2 * TG, H, df.ddelta, 2 * TG, R, false, L);
      for (size_t g = 0; g < n; ++g)
        grouped_conv(df.dh + g * B * H, H, df.ddelta + g * B * 2 * TG, 2 * TG, l.attn_base, H, Tp, b.conv_group,
                     df.dhc + g * B * H, H, B, false, st);
      gemm(df.dhc, H, l.w_qkv, nqkv, H, df.dproj, df.ld_dproj, R, false, L);
      for (size_t g = 0; g < n; ++g) {
        flyx_seq* s = bp.seq[members[off + g]].s;
        const KvBuf& kv = s->kv[size_t(b.kv)];
        kv16* Kj = kv.k16(j);
        kv16* Vj = kv.v16(j);
        float* q_g = df.dq + g * B * nh * hd;
        attn_prep(df.dproj + g * B * df.ld_dproj, df.ld_dproj, 0, nh * hd, (nh + nkv) * hd, l.q_norm.w, l.k_norm.w,
                  l.q_norm.offset, l.q_norm.eps, b.theta, hd, nh, nkv, hd, B, s->P, kPosPosition, false, q_g, Kj, Vj,
                  st);
        attention(q_g, Kj, Vj, nh, nkv, hd, B, false, B, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0,
                  df.dact + g * B * df.ld_dact, df.ld_dact, v.b.part, kRoundSplits, L, 0, b.window);
      }
      gemm(df.dact, df.ld_dact, l.w_o, H, nh * hd, df.dtmp, H, R, false, L);
      for (size_t g = 0; g < n; ++g)
        grouped_conv(df.dtmp + g * B * H, H, df.ddelta + g * B * 2 * TG + TG, 2 * TG, l.attn_base + size_t(Tp) * H, H,
                     Tp, b.conv_group, df.dx + g * B * H, H, B, true, st);
      rmsnorm(df.dx, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, df.dh, H, R, H, st);
      gemm(df.dh, H, l.mlp_proj, 2 * TG, H, df.ddelta, 2 * TG, R, false, L);
      for (size_t g = 0; g < n; ++g)
        grouped_conv(df.dh + g * B * H, H, df.ddelta + g * B * 2 * TG, 2 * TG, l.mlp_base, H, Tp, b.conv_group,
                     df.dhc + g * B * H, H, B, false, st);
      gemm(df.dhc, H, l.w_gu, 2 * I, H, df.dproj, df.ld_dproj, R, false, L);
      if (fused) {
        swiglu_split(df.dproj, df.ld_dproj, I, R, L, l.w_down.q8_inputs());
        gemm_presplit(l.w_down, H, I, df.dtmp, H, R, false, L);
      } else {
        swiglu(df.dproj, df.ld_dproj, I, df.dact, df.ld_dact, R, st);
        gemm(df.dact, df.ld_dact, l.w_down, H, I, df.dtmp, H, R, false, L);
      }
      for (size_t g = 0; g < n; ++g)
        grouped_conv(df.dtmp + g * B * H, H, df.ddelta + g * B * 2 * TG + TG, 2 * TG, l.mlp_base + size_t(Tp) * H, H,
                     Tp, b.conv_group, df.dx + g * B * H, H, B, true, st);
    }
    // Batch the draft head and selector hidden projection across all sequences in this group with nr >= 2
    // (solo nr >= 2 uses the tensor-core rough head k_gemm_sign_cols/k_gemm_sign, bit-identical across
    // 2..FLYX_MAX_ROWS rows; nr == 1 uses the float32 1-row table kernel and stays solo).
    size_t batched[FLYX_MAX_ROWS];
    int batch_off[FLYX_MAX_ROWS];
    size_t n_batched = 0;
    int total_nr = 0;
    for (size_t g = 0; g < n; ++g) {
      const Pass& ps = bp.seq[members[off + g]];
      const int nr = std::min(ps.p->nodes[0].arg0, B - 1);
      if (nr >= 2 && total_nr + nr <= FLYX_MAX_ROWS) {
        batched[n_batched] = g;
        batch_off[n_batched] = total_nr;
        ++n_batched;
        total_nr += nr;
      }
    }
    if (n_batched >= 2) {
      for (size_t bi = 0; bi < n_batched; ++bi) {
        const size_t g = batched[bi];
        const Pass& ps = bp.seq[members[off + g]];
        const int nr = std::min(ps.p->nodes[0].arg0, B - 1);
        rmsnorm(df.dx + (g * B + 1) * H, H, b.norm.w, b.norm.offset, b.norm.eps, df.dh + size_t(batch_off[bi]) * H, H,
                nr, H, st);
      }
      draft_head_pick(head, df.dh, total_nr, b.sel_topk, df.cand, df.unary,
                      pick_scratch(bp.seq[members[off]].s, v.logits), L);
      gemm(df.dh, H, b.sel_hid, b.sel_rank, H, df.hid, b.sel_rank, total_nr, false, L);
      for (size_t bi = 0; bi < n_batched; ++bi) {
        const size_t g = batched[bi];
        const Pass& ps = bp.seq[members[off + g]];
        flyx_seq* s = ps.s;
        const int width = ps.p->nodes[0].arg0;
        const size_t row0 = static_cast<size_t>(batch_off[bi]);
        selector_walk(s->P, df.cand + row0 * b.sel_topk, df.unary + row0 * b.sel_topk, b.sel_topk,
                      df.hid + row0 * b.sel_rank, b.sel_rank, b.sel_pred, b.sel_succ, width, s->oracle, s->drafts, st);
      }
    }
    for (size_t g = 0; g < n; ++g) {
      if (n_batched >= 2) {
        bool done = false;
        for (size_t bi = 0; bi < n_batched; ++bi)
          if (batched[bi] == g) {
            done = true;
            break;
          }
        if (done) continue;
      }
      const Pass& ps = bp.seq[members[off + g]];
      flyx_seq* s = ps.s;
      const int width = ps.p->nodes[0].arg0;
      const int nr = std::min(width, B - 1);
      rmsnorm(df.dx + (g * B + 1) * H, H, b.norm.w, b.norm.offset, b.norm.eps, s->dh, H, nr, H, st);
      draft_head_pick(head, s->dh, nr, b.sel_topk, s->cand, s->unary, s, L);
      gemm(s->dh, H, b.sel_hid, b.sel_rank, H, s->hid, b.sel_rank, nr, false, L);
      selector_walk(s->P, s->cand, s->unary, b.sel_topk, s->hid, b.sel_rank, b.sel_pred, b.sel_succ, width, s->oracle,
                    s->drafts, st);
    }
    off += n;
  }
}

// Batched MTP context update: shares b.fc and a.w_in across all bp.rows rows (same bits as solo mtp_context()).
void mtp_context_rows(const BatchPass& bp, const MtpBlock& b, bool fused) {
  const RowView& v = bp.v;
  const flyx_bmodel* m = bp.seq[0].s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const auto& a = block<AttnBlock>(m, b.attn);
  const int H = a.hidden, R = bp.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  cudaStream_t st = v.L.st;
  const Launch& L = v.L;
  rmsnorm(v.b.x, H, head.norm.w, head.norm.offset, head.norm.eps, v.b.mhn, H, R, H, st);
  if (E.blk.data) embed_blocks(v.b.tokens, E.blk, H, v.b.h, R, st);
  for (const Pass& ps : bp.seq) {
    const RowView& rv = ps.v;
    mtp_gather(ps.s->P, kPosStart, true, nullptr, ps.s->ring, rv.b.mhn, H, rv.b.tokens, E.table, rv.b.h, b.norm_emb.w,
               b.norm_hidden.w, b.norm_emb.offset, b.norm_emb.eps, H, ps.rows, rv.b.mcat, st);
    hring_put(ps.s->P, kPosStart, true, rv.b.mhn, H, ps.rows, ps.s->ring, H, st);
  }
  gemm(v.b.mcat, 2 * H, b.fc, H, 2 * H, v.b.mx, H, R, false, L);
  norm_project(v.b.mx, v.b.h, v.b.proj, v.ld_proj, R, a.norm, a.w_in, a.n_in(), H, L, fused);
  for (const Pass& ps : bp.seq) {
    const RowView& rv = ps.v;
    const KvBuf& kv = ps.s->kv[size_t(a.kv)];
    attn_prep(rv.b.proj, rv.ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
              a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, ps.rows, ps.s->P, kPosStart, true, rv.b.qbuf, kv.ref(), st,
              -1, nullptr, 0, CanvasKeys{}, a.mrope());
  }
}

// Batched DFlash2 context update: shares b.fc and b.ctx across all bp.rows rows (same bits as solo draft_context()).
void draft_context_rows(const BatchPass& bp, const DFlashBlock& b, bool fused) {
  const RowView& v = bp.v;
  const int H = b.hidden, R = bp.rows, F = b.n_taps * b.target_hidden;
  cudaStream_t st = v.L.st;
  const Launch& L = v.L;
  gemm(v.b.feats, v.ld_feats, b.fc, H, F, v.b.cx, H, R, false, L);
  norm_project(v.b.cx, v.b.h, v.b.ckv, v.ld_ckv, R, b.hidden_norm, b.ctx, b.ctx_width(), H, L, fused);
  for (const Pass& ps : bp.seq) {
    const KvBuf& kv = ps.s->kv[size_t(b.kv)];
    draft_ctx_kv(ps.v.b.ckv, ps.v.ld_ckv, b.k_norms.w, b.k_norms.eps, b.theta, b.layers(), b.nkv, b.hd, ps.rows,
                 ps.s->P, kPosStart, true, kv.k16(), kv.v16(), kv.layer_stride, st);
  }
}

}  // namespace

void run_batch(const BatchPass& bp) {
  // Drafting: batch sequences drafting on the same MtpBlock (lockstep) or DFlashBlock (grouped up to FLYX_MAX_ROWS)
  // so drafter weights are shared across sessions; other drafters and given-draft nodes run per sequence.
  const MtpBlock* mtp_blk = nullptr;
  const DFlashBlock* dflash_blk = nullptr;
  std::vector<size_t> mtp_members, dflash_members;
  std::vector<bool> has_conf(bp.seq.size(), true);
  const BatchDFlash& df = bp.seq[0].s->m->batch->dflash;
  for (size_t idx = 0; idx < bp.seq.size(); ++idx) {
    const Pass& ps = bp.seq[idx];
    const size_t ts = target_start(ps.p);
    if (ts == 1 && ps.p->nodes[0].kind == FLYX_NODE_DRAFT && ps.p->nodes[0].arg0 > 0) {
      if (const auto* t = std::get_if<MtpBlock>(ps.p->block[0]); t && (!mtp_blk || mtp_blk == t)) {
        mtp_blk = t;
        mtp_members.push_back(idx);
        continue;
      }
      // A grouped chain must use the same block size it uses alone. Noncausal attention can see later rows.
      if (const auto* d = std::get_if<DFlashBlock>(ps.p->block[0]);
          d && !d->dspine && df.dx && ps.p->nodes[0].arg0 < d->block && d->block * 2 <= FLYX_MAX_ROWS &&
          (!dflash_blk || dflash_blk == d)) {
        dflash_blk = d;
        dflash_members.push_back(idx);
        continue;
      }
    }
    for (size_t k = 0; k < ts; ++k) {
      Pass own = ps;
      own.v = rows_of(ps.s);
      own.v.L.st = bp.v.L.st;
      ps.p->impl[k]->fn(own, ps.p->nodes[k], ps.p->block[k]);
    }
  }
  if (mtp_members.size() >= 2) {
    const bool fused = !generic_impl(bp.seq[mtp_members[0]].p->impl[0]);
    mtp_draft_rows(bp, *mtp_blk, mtp_members, fused);
    for (size_t idx : mtp_members) has_conf[idx] = false;
  } else if (mtp_members.size() == 1) {
    const Pass& ps = bp.seq[mtp_members[0]];
    Pass own = ps;
    own.v = rows_of(ps.s);
    own.v.L.st = bp.v.L.st;
    ps.p->impl[0]->fn(own, ps.p->nodes[0], ps.p->block[0]);
  }
  if (dflash_members.size() >= 2) {
    const bool fused = !generic_impl(bp.seq[dflash_members[0]].p->impl[0]);
    draft_block_rows(bp, *dflash_blk, dflash_members, fused);
    for (size_t idx : dflash_members) has_conf[idx] = false;
  } else if (dflash_members.size() == 1) {
    const Pass& ps = bp.seq[dflash_members[0]];
    Pass own = ps;
    own.v = rows_of(ps.s);
    own.v.L.st = bp.v.L.st;
    ps.p->impl[0]->fn(own, ps.p->nodes[0], ps.p->block[0]);
  }
  // The target's nodes: the first sequence's plan names them (launch_batch checked the others run the same).
  const flyx_plan* p0 = bp.seq[0].p;
  const size_t t0 = target_start(p0);
  for (size_t k = t0; k < p0->nodes.size(); ++k) {
    const flyx_node& n = p0->nodes[k];
    const Block* b = p0->block[k];
    const bool fused = !generic_impl(p0->impl[k]);
    if (n.kind == FLYX_NODE_RUN) {
      if (const auto* e = std::get_if<EmbedBlock>(b)) {
        embed_rows(bp, *e);
      } else if (const auto* g = std::get_if<GdnBlock>(b)) {
        gdn_rows(bp, *g, fused);
      } else if (const auto* a = std::get_if<AttnBlock>(b)) {
        attn_rows(bp, *a, fused);
      } else if (const auto* f = std::get_if<MlpBlock>(b)) {
        Pass all = bp.seq[0];  // row by row: every row at once
        all.rows = bp.rows;
        all.v = bp.v;
        mlp_layer(all, *f, fused);
      } else if (!n.arg0) {
        head_rows(bp, std::get<HeadBlock>(*b));
      }
      continue;
    }
    if (n.kind == FLYX_NODE_TAP) {  // row by row too
      tap_copy(bp.v.b.x, bp.seq[0].s->m->hidden, bp.v.b.feats, bp.v.ld_feats, n.arg0, bp.rows, bp.v.L.st);
      continue;
    }
    // Share drafter context GEMMs across all batch rows; DV's canvas pass (a warm-up's: its trees run alone) reads
    // and writes the sequence's own canvas, a sequence at a time below.
    if (n.kind == FLYX_NODE_CONTEXT && !std::holds_alternative<DvBlock>(*b)) {
      if (const auto* t = std::get_if<MtpBlock>(b)) mtp_context_rows(bp, *t, fused);
      else draft_context_rows(bp, std::get<DFlashBlock>(*b), fused);
      continue;
    }
    // Accept, commit, DV's canvas pass: each sequence's own, by its own plan's node (accept takes its width).
    for (size_t i = 0; i < bp.seq.size(); ++i) {
      Pass ps = bp.seq[i];
      ps.draft_conf = has_conf[i];
      const size_t j = target_start(ps.p) + (k - t0);
      run_node(ps, ps.p->nodes[j], ps.p->block[j]);
    }
  }
}

}  // namespace flyx::cuda
