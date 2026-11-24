// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What each program node runs on its block: the generic implementation of every
// region; fused=true is the override regions/plan.cpp may pick.
#include "backends/cuda/regions/layers.h"

namespace flyx::cuda {

// Draft head top-k / argmax dispatch (1-bit rough pick on packed or bf16 head, else full draft logits).
void draft_head_pick(const HeadBlock& head, const float* x, int nr, int k, int* idx, float* val,
                     const DraftPickScratch& sc, const Launch& L, bool exact_1row, float* prob, float* margin) {
  const int H = head.hidden, V = head.vocab;
  if (head.coarse.bits && head.wb.data) {
    draft_pick_blocks(x, H, head.coarse, head.wb, V, H, nr, k, idx, val, sc.logits, L, exact_1row, prob, margin);
  } else if (head.coarse.bits) {
    draft_pick(x, H, head.coarse, head.w, V, H, nr, k, idx, val, sc.logits, L, exact_1row, prob, margin);
  } else {
    draft_logits(head, x, H, sc.logits, nr, L);
    if (k == 1 && !val && !prob && !margin) argmax_rows(sc.logits, V, V, idx, nr, sc.argv, sc.argi, L.st);
    else topk_rows(sc.logits, V, V, k, idx, val, nr, L.st, sc.topv, sc.topi, sc.top_rows, prob, margin);
  }
}

bool draft_head_additive_ready(const HeadBlock& head, const Lin& extra, const SignHead& extra_coarse, int K_extra) {
  return head.coarse.bits && extra_coarse.bits && (head.wb.data || head.w) &&
         (extra.w || (extra.packed() && extra.blk.size() == 1 && K_extra % 128 == 0));
}

void draft_head_pick_additive_step(const HeadBlock& head, const float* x_base, const Lin& extra,
                                   const SignHead& extra_coarse, const float* extra_bias, const float* x_extra,
                                   int K_extra, float softcap, float* cand_row, int ld, float* scratch_add, int k,
                                   int* idx_out, float* val_out, float* prob_out, float* margin_out, const Launch& L) {
  const int H = head.hidden, V = head.vocab;
  cudaStream_t st = L.st;
  gemm_sign_rough_add(x_extra, K_extra, extra_coarse, extra_bias, V, K_extra, cand_row, ld, L);
  draft_select_cands(cand_row, ld, V, 1, L);
  int* cand_i = reinterpret_cast<int*>(cand_row);
  if (head.wb.data) {
    dots_blocks(head.wb, H, cand_i, cand_i + 2 * kDraftCandCap, cand_row + kDraftCandCap, size_t(ld), kDraftCandCap,
                x_base, H, 1, st);
  } else {
    draft_score_bf16(x_base, H, head.w, nullptr, H, 1, cand_row, ld, false, 0.f, st);
  }
  if (extra.w) {
    draft_score_bf16(x_extra, K_extra, extra.w, extra_bias, K_extra, 1, cand_row, ld, true, softcap, st);
  } else {
    dots_blocks(extra.blk[0], K_extra, cand_i, cand_i + 2 * kDraftCandCap, scratch_add, 0, kDraftCandCap, x_extra,
                K_extra, 1, st);
    draft_combine_cands(scratch_add, 0, extra_bias, softcap, 1, cand_row, ld, st);
  }
  draft_pick_topk(cand_row, ld, 1, k, idx_out, val_out, st, prob_out, margin_out);
}

void norm_project(const float* x, float* h, float* out, int ld_out, int R, const Norm& n, const Lin& w, int N, int H,
                  const Launch& L, bool fused) {
  if (fused) {
    rmsnorm_split(x, H, n.w, n.offset, n.eps, R, H, L, w.q8_inputs());
    gemm_presplit(w, N, H, out, ld_out, R, false, L);
  } else {
    rmsnorm(x, H, n.w, n.offset, n.eps, h, H, R, H, L.st);
    gemm(h, H, w, N, H, out, ld_out, R, false, L);
  }
}

void norm_and_project(const RowView& v, int R, const Norm& n, const Lin& w, int N, int H, bool fused) {
  norm_project(v.b.x, v.b.h, v.b.proj, v.ld_proj, R, n, w, N, H, v.L, fused);
}

bool residual16_model(flyx_bmodel* m) {
  if (m->res16 < 0) {  // every block of the stack: the embedding a bf16 table, each residual add a W4A4 GEMM
    bool ok = m->opt.fusion && m->opt.kernels.prefill_fp4 && gemm_w4a4_supported();
    std::vector<bool> drafter_parts(m->blocks.size(), false);  // the MTP layer's blocks add to no residual here
    for (const Block& b : m->blocks)
      if (const auto* t = std::get_if<MtpBlock>(&b)) {
        if (t->attn >= 0) drafter_parts[size_t(t->attn)] = true;
        if (t->mlp >= 0) drafter_parts[size_t(t->mlp)] = true;
      }
    for (size_t i = 0; i < m->blocks.size() && ok; ++i) {
      if (drafter_parts[i]) continue;
      const Block& b = m->blocks[i];
      if (const auto* e = std::get_if<EmbedBlock>(&b)) ok = e->table != nullptr;
      else if (const auto* g = std::get_if<GdnBlock>(&b))
        ok = g->w_in.q && g->w_out.q && g->hidden % 128 == 0 && g->v_width() % 128 == 0;
      else if (const auto* a = std::get_if<AttnBlock>(&b))  // adapters (ABI 43) add float32 products: float32 rows
        ok = a->w_in.q && a->w_out.q && a->hidden % 128 == 0 && a->nh * a->hd % 128 == 0 && !a->lora;
      else if (const auto* f = std::get_if<MlpBlock>(&b))
        ok = f->w_gu.q && f->w_down.q && f->hidden % 128 == 0 && f->inter % 128 == 0 && !f->lora;
    }
    m->res16 = ok ? 1 : 0;
  }
  return m->res16 == 1;
}

bool residual16(const Pass& ps) {
  if (ps.round || ps.rows <= 16) return false;
  return residual16_model(ps.s->m) && ps.v.L.scratch && ps.v.L.scratch->af32;
}

void mlp_body(float* x, const MlpBlock& f, int R, float* h, float* proj, int ld_proj, float* act, int ld_act,
              const Launch& L, bool fused) {
  const int H = f.hidden, I = f.inter;
  const Norm& n = f.norm;
  if (fused) {
    rmsnorm_split(x, H, n.w, n.offset, n.eps, R, H, L, f.w_gu.q8_inputs());
    // SwiGLU in the gate|up GEMM's epilogue: no float32 gate|up rows (bf16 and NVFP4 weights on both sides).
    if (R > 16 && I % 8 == 0 && !f.w_gu.packed() && !f.w_down.packed()) {
      gemm_glu_presplit(f.w_gu, I, H, R, L);
      gemm_on_glu(f.w_down, H, I, x, H, R, true, L);
      return;
    }
    // A round on block weights with 8-bit inputs on both sides: the gate|up GEMM writes the SwiGLU too, beside its own
    // input (the bits of swiglu_split), which the down GEMM takes.
    if (R <= 16 && f.w_gu.packed() && f.w_down.q8_inputs() &&
        gemm_blocks_round_glu_takes(f.w_gu.blk.data(), int(f.w_gu.blk.size()), R, I, L)) {
      const RoundGlu glu{proj, ld_proj, I};
      Launch gu = L;
      gu.glu = &glu;
      gemm_presplit(f.w_gu, 2 * I, H, proj, ld_proj, R, false, gu);
      const Scratch act_q8 = glu_inputs(*L.scratch);
      Launch down = L;
      down.scratch = &act_q8;
      gemm_presplit(f.w_down, H, I, x, H, R, true, down);
      return;
    }
    gemm_presplit(f.w_gu, 2 * I, H, proj, ld_proj, R, false, L);
    swiglu_split(proj, ld_proj, I, R, L, f.w_down.q8_inputs());
    gemm_presplit(f.w_down, H, I, x, H, R, true, L);
    return;
  }
  rmsnorm(x, H, n.w, n.offset, n.eps, h, H, R, H, L.st);
  gemm(h, H, f.w_gu, 2 * I, H, proj, ld_proj, R, false, L);
  swiglu(proj, ld_proj, I, act, ld_act, R, L.st);
  gemm(act, ld_act, f.w_down, H, I, x, H, R, true, L);
}

void gdn_out_project(const RowView& v, const GdnBlock& g, int R, bool fused) {
  const GdnDims gd = g.gd;
  const int H = g.hidden, C = gd.C(), vd = g.v_width();
  const float* z = v.b.proj + C;
  if (fused) {
    gdn_gated_norm_split(v.b.gdn_o, z, v.ld_proj, g.gated.w, g.gated.eps, R, gd, v.L, g.w_out.q8_inputs());
    gemm_presplit(g.w_out, H, vd, v.b.x, H, R, true, v.L);
  } else {
    gdn_gated_norm(v.b.gdn_o, z, v.ld_proj, g.gated.w, g.gated.eps, v.b.act, R, gd, v.L.st);
    gemm(v.b.act, vd, g.w_out, H, vd, v.b.x, H, R, true, v.L);
  }
}

void head_logits(const HeadBlock& h, const RowView& v, int r0, int n, bool cert, const CertScratch& cs,
                 const KernelOptions& ko, bool x16) {
  const int H = h.hidden, V = h.vocab;
  if (x16) rmsnorm(x16_of(v) + size_t(r0) * H, H, h.norm.w, h.norm.offset, h.norm.eps, v.b.h, H, n, H, v.L.st);
  else rmsnorm(v.b.x + size_t(r0) * H, H, h.norm.w, h.norm.offset, h.norm.eps, v.b.h, H, n, H, v.L.st);
  if (h.w8 && n <= 16 && ko.head_fp8) {  // option head_fp8: the copy's scores
    gemm_fp8_skinny(v.b.h, H, h.w8, h.rs8, V, H, v.logits, V, n, false, v.L);
  } else if (h.cert && n <= 16 && cert && !ko.row_gemms) {  // option head_cert
    gemm_fp8_skinny(v.b.h, H, h.w8, h.rs8, V, H, v.logits, V, n, false, v.L);
    head_certify(v.b.h, H, h.w, h.cert, V, H, v.logits, V, n, cs.norms, cs.key, cs.cols, cs.count, v.L);
  } else if (h.w12.hi && n <= 16 && !ko.row_gemms) {
    gemm_w12_skinny(v.b.h, H, h.w12, V, H, v.logits, V, n, false, v.L);
  } else if (h.wb.data) {
    gemm_blocks(v.b.h, H, h.wb, H, v.logits, V, n, false, v.L);
  } else {
    gemm(v.b.h, H, h.w, V, H, v.logits, V, n, false, v.L);
  }
}

// DSpine (backend_abi.h flyx_dspine_block, ops/dspine/ref.h): the rows the round's drafts need (no row reads a row
// after it: a block of fewer rows drafts the same first drafts), each layer a Qwen3 layer attending causally (within
// the window) then the bridge's write; the head's candidates for the rows after the anchor; and the walk, which redoes
// the last write with the drafted token's code: every sender's message at once (the anchor's and every candidate's but
// the last row's), its scores against the next row's candidates, the chain through the picks.
void dspine_block(flyx_seq* s, const DFlashBlock& b, int width, const Launch& L, bool fused) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const KvBuf& kv = s->kv[size_t(b.kv)];
  const DSpineBridge& br = b.dspine;
  auto& d = s->ds;
  const int B = std::min(b.block, width + 1), nr = B - 1;
  const int H = b.hidden, nh = b.nh, nkv = b.nkv, hd = b.hd, I = b.inter, C = br.code, R = br.rank, K = br.topk;
  const int nqkv = (nh + 2 * nkv) * hd, senders = 1 + (nr - 1) * K;
  cudaStream_t st = L.st;
  build_rows(s->P, nullptr, b.mask_token, s->dtok, B, st);
  embed_tokens(E, s->dtok, s->dx, B, st);  // packed tables stay in host memory
  for (int j = 0; j < b.layers(); ++j) {
    const auto& l = block<DFlashLayer>(m, b.first_layer + j);
    kv16* Kj = kv.k16(j);
    kv16* Vj = kv.v16(j);
    rmsnorm(s->dx, H, l.norm.w, l.norm.offset, l.norm.eps, s->dh, H, B, H, st);
    gemm(s->dh, H, l.w_qkv, nqkv, H, s->dproj, s->ld_dproj, B, false, L);
    attn_prep(s->dproj, s->ld_dproj, 0, nh * hd, (nh + nkv) * hd, l.q_norm.w, l.k_norm.w, l.q_norm.offset,
              l.q_norm.eps, b.theta, hd, nh, nkv, hd, B, s->P, kPosPosition, false, s->dq, Kj, Vj, st);
    const bool split_o = fused && !l.w_o.q8_inputs() && (nh * hd) % 64 == 0;
    attention(s->dq, Kj, Vj, nh, nkv, hd, B, true, 0, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0,
              split_o ? nullptr : s->dact, s->ld_dact, s->own.part, kRoundSplits, L, 0, b.window);
    if (split_o) gemm_presplit(l.w_o, H, nh * hd, s->dx, H, B, true, L);
    else gemm(s->dact, s->ld_dact, l.w_o, H, nh * hd, s->dx, H, B, true, L);
    rmsnorm(s->dx, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, s->dh, H, B, H, st);
    gemm(s->dh, H, l.w_gu, 2 * I, H, s->dproj, s->ld_dproj, B, false, L);
    if (fused) {
      swiglu_split(s->dproj, s->ld_dproj, I, B, L, l.w_down.q8_inputs());
      gemm_presplit(l.w_down, H, I, s->dx, H, B, true, L);
    } else {
      swiglu(s->dproj, s->ld_dproj, I, s->dact, s->ld_dact, B, st);
      gemm(s->dact, s->ld_dact, l.w_down, H, I, s->dx, H, B, true, L);
    }
    // The bridge's write; the last one's rows before it, their rms and gate projections are what the walk redoes.
    const bool last = j == b.layers() - 1;
    dspine_normalize(s->dx, H, br.eps, B, s->dh, d.rms, st);
    gemm(s->dh, H, l.read, C, H, d.codes, C, B, false, L);
    dspine_codes(d.codes, C, br.codebook, s->P, br.eps, B, st);
    gemm(d.codes, C, br.packed, 3 * R, C, d.gates, 3 * R, B, false, L);
    dspine_message(d.gates, R, br.gate_bias, B, d.msg, last ? d.g : nullptr, st);
    gemm(d.msg, R, l.write, H, R, s->dtmp, H, B, false, L);
    dspine_apply(s->dx, s->dtmp, d.rms, H, B, last ? d.before : nullptr, st);
  }
  rmsnorm(s->dx + H, H, b.norm.w, b.norm.offset, b.norm.eps, s->dh, H, nr, H, st);
  draft_head_pick(head, s->dh, nr, K, s->cand, s->unary, s, L, s->draft_prob, s->draft_margin);
  // The walk's GEMMs, up to (block - 1) topk rows: the round kernels 64 rows at a time, each row's inputs as three bf16
  // parts (float32's precision: the scores' differences are a message's against another's; bf16 inputs moved them by
  // 5e-4 of a score, more than a near tie), the weights read once a 64 rows.
  Launch walk = L;
  walk.round_rows = true;
  walk.draft = false;
  auto rows64 = [&](const float* x, int ldx, const Lin& w, int N, int K_, float* y, int ldy, int rows) {
    for (int r0 = 0; r0 < rows; r0 += 64)
      gemm(x + size_t(r0) * ldx, ldx, w, N, K_, y + size_t(r0) * ldy, ldy, std::min(64, rows - r0), false, walk);
  };
  if (head.wb.data) embed_blocks(s->cand, head.wb, H, d.heads, nr * K, st);  // a packed head's rows as float32
  dspine_heads(s->cand, head.wb.data ? nullptr : head.w, b.norm.w, d.before, H, K, nr * K, d.heads, d.base, st);
  rows64(d.heads, H, br.write_t, R, H, d.tg, R, nr * K);
  dspine_senders(s->P, s->cand, br.codebook, C, senders, d.codes, st);
  rows64(d.codes, C, br.pv, 2 * R, C, d.gates, 2 * R, senders);
  dspine_scores(d.g, d.gates, br.gate_bias, d.tg, d.base, d.rms, R, K, senders, d.pick, st);
  dspine_walk(s->P, s->cand, d.pick, K, std::min(width, nr), s->oracle, s->drafts, st);
}

void draft_block(flyx_seq* s, const DFlashBlock& b, int width, const Launch& L, bool tree, bool fused) {
  if (b.dspine) {
    if (tree) throw std::logic_error("cuda: DSpine drafts chains, not trees");  // lower refuses its tree rounds
    return dspine_block(s, b, width, L, fused);
  }
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const KvBuf& kv = s->kv[size_t(b.kv)];
  // A chain wider than the block drafts from a wider block: width + 1 mask rows in one pass (basalt's wide block;
  // the drafter attends within its block both ways, at any count).  Trees keep the block they were made for.
  const int B = tree ? b.block : std::min(FLYX_MAX_ROWS, std::max(b.block, width + 1));
  const int H = b.hidden, nh = b.nh, nkv = b.nkv, hd = b.hd, I = b.inter;
  const int Tp = b.conv_taps, G = H / b.conv_group, TG = Tp * G, nqkv = (nh + 2 * nkv) * hd;
  cudaStream_t st = L.st;
  build_rows(s->P, nullptr, b.mask_token, s->dtok, B, st);
  embed_tokens(E, s->dtok, s->dx, B, st);  // packed tables stay in host memory
  for (int j = 0; j < b.layers(); ++j) {
    const auto& l = block<DFlashLayer>(m, b.first_layer + j);
    kv16* Kj = kv.k16(j);
    kv16* Vj = kv.v16(j);
    rmsnorm(s->dx, H, l.norm.w, l.norm.offset, l.norm.eps, s->dh, H, B, H, st);
    gemm(s->dh, H, l.attn_proj, 2 * TG, H, s->ddelta, 2 * TG, B, false, L);
    grouped_conv(s->dh, H, s->ddelta, 2 * TG, l.attn_base, H, Tp, b.conv_group, s->dhc, H, B, false, st);
    gemm(s->dhc, H, l.w_qkv, nqkv, H, s->dproj, s->ld_dproj, B, false, L);
    attn_prep(s->dproj, s->ld_dproj, 0, nh * hd, (nh + nkv) * hd, l.q_norm.w, l.k_norm.w, l.q_norm.offset,
              l.q_norm.eps, b.theta, hd, nh, nkv, hd, B, s->P, kPosPosition, false, s->dq, Kj, Vj, st);
    const bool split_o = fused && !l.w_o.q8_inputs() && (nh * hd) % 64 == 0;
    attention(s->dq, Kj, Vj, nh, nkv, hd, B, false, B, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0,
              split_o ? nullptr : s->dact, s->ld_dact, s->own.part, kRoundSplits, L, 0, b.window);
    if (split_o) gemm_presplit(l.w_o, H, nh * hd, s->dtmp, H, B, false, L);
    else gemm(s->dact, s->ld_dact, l.w_o, H, nh * hd, s->dtmp, H, B, false, L);
    grouped_conv(s->dtmp, H, s->ddelta + TG, 2 * TG, l.attn_base + size_t(Tp) * H, H, Tp, b.conv_group, s->dx, H, B,
                 true, st);
    rmsnorm(s->dx, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, s->dh, H, B, H, st);
    gemm(s->dh, H, l.mlp_proj, 2 * TG, H, s->ddelta, 2 * TG, B, false, L);
    grouped_conv(s->dh, H, s->ddelta, 2 * TG, l.mlp_base, H, Tp, b.conv_group, s->dhc, H, B, false, st);
    gemm(s->dhc, H, l.w_gu, 2 * I, H, s->dproj, s->ld_dproj, B, false, L);
    if (fused) {
      swiglu_split(s->dproj, s->ld_dproj, I, B, L, l.w_down.q8_inputs());
      gemm_presplit(l.w_down, H, I, s->dtmp, H, B, false, L);
    } else {
      swiglu(s->dproj, s->ld_dproj, I, s->dact, s->ld_dact, B, st);
      gemm(s->dact, s->ld_dact, l.w_down, H, I, s->dtmp, H, B, false, L);
    }
    grouped_conv(s->dtmp, H, s->ddelta + TG, 2 * TG, l.mlp_base + size_t(Tp) * H, H, Tp, b.conv_group, s->dx, H, B,
                 true, st);
  }
  // Rows 1..width predict tokens: unary top-k from the target head, then the
  // selector walk (or tree).  The block's layers above need all B rows (attention inside
  // the block is not causal), the head only the rows whose drafts are used.
  const int nr = std::min(width, B - 1);
  rmsnorm(s->dx + H, H, b.norm.w, b.norm.offset, b.norm.eps, s->dh, H, nr, H, st);
  draft_head_pick(head, s->dh, nr, b.sel_topk, s->cand, s->unary, s, L, s->draft_prob, s->draft_margin);
  gemm(s->dh, H, b.sel_hid, b.sel_rank, H, s->hid, b.sel_rank, nr, false, L);
  if (tree)
    selector_tree(s->P, s->cand, s->unary, b.sel_topk, s->hid, b.sel_rank, b.sel_pred, b.sel_succ, width, nr,
                  s->oracle, s->drafts, s->tree, st);
  else
    selector_walk(s->P, s->cand, s->unary, b.sel_topk, s->hid, b.sel_rank, b.sel_pred, b.sel_succ, width, s->oracle,
                  s->drafts, st);
}

// One MTP draft step's attention for one sequence, the step's row at P->position + pos_offset: its key and value
// written, then attention over what the sequence keeps of the cache (its low-bit copy, its gathered window, or all
// of it).  out null: the result stays in the split scratch for a fused output projection (gemm_presplit).
void mtp_draft_attention(flyx_seq* s, const AttnBlock& a, const KvBuf& kv, int pos_offset, const float* proj,
                         int ld_proj, float* q, float* out, float* part, const Launch& L) {
  const int nh = a.nh, nkv = a.nkv, hd = a.hd;
  const float* gate = a.gate ? proj + a.nh * a.hd : nullptr;
  attn_prep(proj, ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
            a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, 1, s->P, kPosPosition, false, q, kv.ref(), L.st, pos_offset,
            nullptr, 0, CanvasKeys{}, a.mrope());
  s->stage_before(kv, pos_offset, 1, L.st);
  if (kv.kq) {  // its copy of every position, the last ones exact
    params_at(s->P, s->P_window, pos_offset, L.st);
    attention_strata(q, kv.ref(), s->mtp_strata, nh, nkv, hd, 1, 1.f / std::sqrt(float(hd)), s->P_window, gate,
                     ld_proj, out, nh * hd, part, kRoundSplits, L, 0, nullptr, s->P);
  } else if (kv.window > 0) {  // its last positions, gathered
    kv_window(kv.ref(), s->mtp_window, nkv, hd, kv.window, s->P, s->P_window, pos_offset, L.st);
    // Not a programmatic dependent: the attention kernels read a row's position before they wait for the kernel
    // before, and that kernel (kv_window) wrote it.
    KernelOptions in_order = *L.opt;
    in_order.pdl = false;
    Launch after = L;
    after.opt = &in_order;
    attention(q, s->mtp_window, nh, nkv, hd, 1, true, 0, 1.f / std::sqrt(float(hd)), s->P_window, gate, ld_proj,
              out, nh * hd, part, kRoundSplits, after, pos_offset);
  } else {
    attention(q, kv.staged(), nh, nkv, hd, 1, true, 0, 1.f / std::sqrt(float(hd)), s->P, gate, ld_proj, out, nh * hd,
              part, kRoundSplits, L, pos_offset);
  }
}

// MTP: one layer from (target hidden, next token) pairs, see python/reference/flyx_ref/mtp.py.
// Draft step i runs at position P->position - 1 + i.
void mtp_draft(flyx_seq* s, const MtpBlock& b, int width, const Launch& L, bool fused, int start) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const auto& a = block<AttnBlock>(m, b.attn);
  const auto& f = block<MlpBlock>(m, b.mlp);
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const RowBufs& own = s->own;
  const int H = a.hidden, nh = a.nh, hd = a.hd;
  const bool split_o = fused && !a.w_out.q8_inputs() && (nh * hd) % 64 == 0;
  cudaStream_t st = L.st;
  if (start == 0) build_rows(s->P, nullptr, 0, s->mtok, 1, st);  // mtok[0] = anchor
  for (int i = start; i < width; ++i) {
    if (E.blk.data) embed_blocks(s->mtok + i, E.blk, H, own.h, 1, st);  // own.h is free until the layer's norm
    mtp_gather(s->P, kPosPosition, false, i == 0 ? nullptr : s->mg, s->ring, nullptr, 0, s->mtok + i, E.table, own.h,
               b.norm_emb.w, b.norm_hidden.w, b.norm_emb.offset, b.norm_emb.eps, H, 1, own.mcat, st);
    gemm(own.mcat, 2 * H, b.fc, H, 2 * H, own.mx, H, 1, false, L);
    norm_project(own.mx, own.h, own.proj, s->ld_proj, 1, a.norm, a.w_in, a.n_in(), H, L, fused);
    mtp_draft_attention(s, a, kv, i - 1, own.proj, s->ld_proj, own.qbuf, split_o ? nullptr : own.act, own.part, L);
    if (split_o) gemm_presplit(a.w_out, H, nh * hd, own.mx, H, 1, true, L);
    else gemm(own.act, nh * hd, a.w_out, H, nh * hd, own.mx, H, 1, true, L);
    mlp_body(own.mx, f, 1, own.h, own.proj, s->ld_proj, own.act, s->ld_act, L, fused);
    rmsnorm(own.mx, H, b.norm.w, b.norm.offset, b.norm.eps, s->mg, H, 1, H, st);
    draft_head_pick(head, s->mg, 1, 1, s->mtok + i + 1, nullptr, s, L, s->draft_prob + i, s->draft_margin + i);
    mtp_oracle(s->P, s->oracle, i, s->mtok, st);
  }
  copy_ints(s->mtok + 1 + start, s->drafts + start, width - start, st);
}

// After a pass: keep the rows' normalised target hidden and write the MTP K/V
// entries base-1 .. base+rows-2 from (hidden at j, token at j+1).  In a round
// the entries past the accepted rows hold rejected rows; they are rewritten
// (by the next draft steps or context pass) before anything attends to them.
void mtp_context(const Pass& ps, const MtpBlock& b, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const auto& a = block<AttnBlock>(m, b.attn);
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const int H = a.hidden, R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  const int sel = ps.round ? kPosStart : kPosPosition;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  if (residual16(ps)) rmsnorm(x16_of(rv), H, head.norm.w, head.norm.offset, head.norm.eps, rv.b.mhn, H, R, H, st);
  else rmsnorm(rv.b.x, H, head.norm.w, head.norm.offset, head.norm.eps, rv.b.mhn, H, R, H, st);
  if (E.blk.data) {
    embed_blocks(rv.b.tokens, E.blk, H, rv.b.h, R, st);  // rv.b.h is free until the layer's norm
    if (ps.n_embeds > 0) embed_scatter(rv.b.h, nullptr, H, ps.embeds, ps.embed_rows, ps.n_embeds, st);
  } else if (ps.n_embeds > 0) {
    embed(rv.b.tokens, E.table, H, rv.b.h, R, st);
    embed_scatter(rv.b.h, nullptr, H, ps.embeds, ps.embed_rows, ps.n_embeds, st);
  }
  const bf16* mtp_table = (E.blk.data || ps.n_embeds > 0) ? nullptr : E.table;
  mtp_gather(s->P, sel, ps.round, nullptr, s->ring, rv.b.mhn, H, rv.b.tokens, mtp_table, rv.b.h, b.norm_emb.w,
             b.norm_hidden.w, b.norm_emb.offset, b.norm_emb.eps, H, R, rv.b.mcat, st);
  if (!ps.round && s->pass_snap > 0 && s->ring_snap) {
    // Capture hidden[pass_snap - 1] before hring_put overwrites the kHring-slot ring when end - pass_snap >= kHring.
    const int snap_row = s->pass_snap - s->position;
    const float* src = snap_row == 0 ? s->ring + size_t((s->pass_snap - 1) % kHring) * size_t(H)
                                     : rv.b.mhn + size_t(snap_row - 1) * size_t(H);
    CK(cudaMemcpyAsync(s->ring_snap, src, size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
  }
  hring_put(s->P, sel, ps.round, rv.b.mhn, H, R, s->ring, H, st);
  gemm(rv.b.mcat, 2 * H, b.fc, H, 2 * H, rv.b.mx, H, R, false, L);
  norm_project(rv.b.mx, rv.b.h, rv.b.proj, rv.ld_proj, R, a.norm, a.w_in, a.n_in(), H, L, fused);
  attn_prep(rv.b.proj, rv.ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
            a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, R, s->P, sel, ps.round, rv.b.qbuf, kv.ref(), st, -1, nullptr, 0,
            CanvasKeys{}, a.mrope(ps.mtp_pos_3d));
}

// fused: the producers write the next GEMM's input already rounded (split for 1..16 rows,
// bf16 for more), so the GEMM reads it as is.
void gdn_layer(const Pass& ps, const GdnBlock& g, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  GdnBuf& sb = s->gdn[size_t(g.state)];
  const GdnDims gd = g.gd;
  const int H = g.hidden, R = ps.rows, C = gd.C(), vd = g.v_width();
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  // A prefill pass with NVFP4 inputs: the in-projection's rows in bf16 (as vLLM and SGLang hold them), read by the
  // convolution, the gates and the gated norm; the proj buffer holds them (half its bytes).
  const bool x16 = residual16(ps);
  if (x16 && !fused) throw std::logic_error("cuda: a bf16 residual's pass runs the fused producers");
  if (!ps.round && fused && R > 16 && !g.w_in.packed() && g.w_in.q) {
    if (x16) rmsnorm_split(x16_of(rv), H, g.norm.w, g.norm.offset, g.norm.eps, R, H, L, g.w_in.q8_inputs());
    else rmsnorm_split(rv.b.x, H, g.norm.w, g.norm.offset, g.norm.eps, R, H, L, g.w_in.q8_inputs());
    bf16* p16 = reinterpret_cast<bf16*>(rv.b.proj);
    if (gemm_staged_bf16(g.w_in.q, g.w_in.s, g.w_in.rs, g.n_in(), H, p16, rv.ld_proj, R, L)) {
      const GdnBufs bf{rv.b.pre.base, 0, rv.b.pre.rows_cap, gd};
      const int snap_row = s->pass_snap >= 0 ? s->pass_snap - s->position : -1;
      const int snap_sel = snap_row >= 0 ? kSlotExplicit + s->snap_slot : -1;
      gdn_conv(p16, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, kSlotCur, bf, kSlotCur, s->P, R, st, snap_sel,
               snap_row);
      gdn_prep(bf, kSlotCur, p16 + C + vd, p16 + C + vd + gd.nv, rv.ld_proj, g.A_log, g.dt_bias, s->P, R, st);
      if (!gdn_chunk(sb.S, sb.s_stride, kSlotCur, bf, s->P, s->position, rv.b.gdn_o, R, st, snap_sel, snap_row))
        gdn_recur(sb.S, sb.s_stride, kSlotCur, kSlotCur, bf, kSlotCur, s->P, rv.b.gdn_o, R, nullptr, false, st);
      gdn_gated_norm_split(rv.b.gdn_o, p16 + C, rv.ld_proj, g.gated.w, g.gated.eps, R, gd, L, g.w_out.q8_inputs());
      if (!x16) gemm_presplit(g.w_out, H, vd, rv.b.x, H, R, true, L);
      else if (!gemm_staged_bf16(g.w_out.q, g.w_out.s, g.w_out.rs, H, vd, x16_of(rv), H, R, L, true))
        throw std::logic_error("cuda: a bf16 residual's out projection is a W4A4 GEMM");
      return;
    }
    if (x16) throw std::logic_error("cuda: a bf16 residual's in-projection is a W4A4 GEMM");
    gemm_presplit(g.w_in, g.n_in(), H, rv.b.proj, rv.ld_proj, R, false, L);
  } else {
    norm_and_project(rv, R, g.norm, g.w_in, g.n_in(), H, fused);
  }
  const float* b = rv.b.proj + C + vd;
  const float* a = b + gd.nv;
  // Plain decoding (no drafts) always keeps its one row: the pass writes the
  // next window and state instead of the commit folding them (same bits, one
  // pass fewer).
  const bool direct = ps.round && ps.p->width == 0;
  const int next = direct ? kSlotNext : -1;
  const TreeRows* tree = tree_of(ps);
  if (fused && ps.round && !tree &&
      gdn_round(rv.b.proj, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, sb.S, sb.s_stride, next, g.A_log, g.dt_bias,
                g.gated.w, g.gated.eps, sb.replay, s->P, R, nullptr, L, g.w_out.q8_inputs())) {
    gemm_presplit(g.w_out, H, vd, rv.b.x, H, R, true, L);
    return;
  }
  if (ps.round) {
    const GdnBufs& bf = sb.replay;
    gdn_conv(rv.b.proj, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, next, kSkipIfStopped, bf, kSlotCur, s->P,
             R, st, -1, -1, tree);
    gdn_prep(bf, kSlotCur, b, a, rv.ld_proj, g.A_log, g.dt_bias, s->P, R, st);
    if (!tree) {
      gdn_recur(sb.S, sb.s_stride, kSlotCur, next, bf, kSlotCur, s->P, rv.b.gdn_o, R, nullptr,
                direct ? kSkipIfStopped : kSkipNever, st);
    } else if (!gdn_recur_tree(sb.S, sb.s_stride, kSlotCur, bf, kSlotCur, s->P, rv.b.gdn_o, R, tree, st)) {
      throw std::runtime_error("tree rounds: a gated delta net of key dim " + std::to_string(gd.dk) +
                               " (16 to 128 with an aligned state are taken)");
    }
  } else {
    const GdnBufs bf{rv.b.pre.base, 0, rv.b.pre.rows_cap, gd};  // the shared prefill scratch, this block's dims
    // The pass's chunk boundary for checkpoint() (runner.cpp): window and state there to the snapshot slot.
    const int snap_row = s->pass_snap >= 0 ? s->pass_snap - s->position : -1;
    const int snap_sel = snap_row >= 0 ? kSlotExplicit + s->snap_slot : -1;
    gdn_conv(rv.b.proj, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, kSlotCur, kSkipNever, bf, kSlotCur, s->P,
             R, st, snap_sel, snap_row);
    gdn_prep(bf, kSlotCur, b, a, rv.ld_proj, g.A_log, g.dt_bias, s->P, R, st);
    // Chunks on tensor cores where the shapes allow (numerics.md: P, K); s->position is the pass's first row.
    if (!gdn_chunk(sb.S, sb.s_stride, kSlotCur, bf, s->P, s->position, rv.b.gdn_o, R, st, snap_sel, snap_row))
      gdn_recur(sb.S, sb.s_stride, kSlotCur, kSlotCur, bf, kSlotCur, s->P, rv.b.gdn_o, R, nullptr, false, st);
  }
  gdn_out_project(rv, g, R, fused);
}

// A prefill's attention over a cache whose keys go past its stagings: the device part's whole 64s
// where they are, then a window of the stagings' positions at a time, each copied to a staging (ahead, beside the
// layers and the windows before it), the rows' softmax state carried between them.  The bits of one pass.
namespace {

void attn_windows(const Pass& ps, const AttnBlock& a, const KvBuf& kb, int windows, float* out, int ld_out) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  cudaStream_t st = rv.L.st;
  const float scale = 1.f / std::sqrt(float(hd));
  const float* gate = a.gate ? rv.b.proj + nh * hd : nullptr;
  const int from = s->window_from(kb), len = s->window_len(kb), end = s->position + R;
  // A pass of a round's rows or fewer (pageable caches only: windows() gives the others none) takes a round's
  // kernels, as it does without windows; attention=split has no windows at all (alloc_state).
  if (R <= FLYX_MAX_ROWS) {
    if (!rv.b.split_state) throw std::logic_error("cuda: prefill split window state buffer missing");
    // Two buffers of the rows' state, read and written in turn: a launch reads what the one before left
    // (SplitWindow::carry) while another of its blocks writes what the next will read.
    float* const planes[2] = {rv.b.split_state, rv.b.split_state + size_t(R) * s->m->need.split_state};
    SplitWindow sw;
    int launches = 0;
    auto run = [&](const KvRef& kv, int k_lo, int k_hi, bool last) {
      sw.k_lo = k_lo;
      sw.k_hi = k_hi;
      sw.last = last;
      sw.state = planes[launches & 1];
      sw.carry = planes[(launches + 1) & 1];
      attention_split(rv.b.qbuf, kv, nh, nkv, hd, R, true, 0, scale, s->P, gate, rv.ld_proj, out, ld_out, rv.b.part,
                      kPrefillSplits, rv.L, 0, 0, nullptr, {}, sw);
      sw.first = false;
      ++launches;
    };
    if (from > 0) run(kb.ref(), 0, from, false);
    for (int i = 0; i < windows; ++i) {
      const int w0 = from + i * len, w1 = std::min(end, w0 + len);
      run(s->window_before(kb, w0, w1, R, st), w0, w1, i + 1 == windows);
      s->stage_after(kb, st);
    }
    return;
  }
  if (!rv.b.attn_state) throw std::logic_error("cuda: prefill window state buffer missing");
  FlashWindow w;
  w.state = rv.b.attn_state;
  auto run = [&](const KvRef& kv, int k_lo, int k_hi, bool last) {
    w.k_lo = k_lo;
    w.k_hi = k_hi;
    w.last = last;
    w.history = k_hi <= s->position;
    if (!attention_flash(rv.b.qbuf, kv, nh, nkv, hd, R, scale, s->P, gate, rv.ld_proj, out, ld_out, rv.L, 0, w))
      throw std::logic_error("cuda: windows of keys on a head dim the tensor-core prefill does not take");
    w.first = false;
  };
  if (from > 0) run(kb.ref(), 0, from, false);
  for (int i = 0; i < windows; ++i) {
    const int w0 = from + i * len, w1 = std::min(end, w0 + len);
    run(s->window_before(kb, w0, w1, R, st), w0, w1, i + 1 == windows);
    s->stage_after(kb, st);
  }
}

}  // namespace

// A block's own adapters (ABI 43, Adapters) beside its linears, on the rows the linears read (x null: a fused
// producer's staged input, which the adapter's first GEMM reads as the linear's did): u = scale (a x), then each
// linear's y += b u.  The a's of the linears that read one input are one GEMM (q|k|v, gate|up), each b its own, every
// GEMM gemm()'s: a round's rows past 16 (Launch::round_rows) on k_gemm_skinny_mt, the weights read once for every 16-row
// tile (16 rows at a time where K is no multiple of 32), so a row has the bits it has alone.
void lora_in(const RowView& rv, const Adapters& l, int linears, const int* out, const float* x, int ldx, int K, int R,
             int ldu, const Launch& L) {
  const int r = l.rank;
  lora_down(x, ldx, l.a_in, linears * r, K, l.scale, rv.b.lora, ldu, R, L);
  for (int i = 0, col = 0; i < linears; col += out[i], ++i)
    gemm(rv.b.lora + i * r, ldu, l.b[i], out[i], r, rv.b.proj + col, rv.ld_proj, R, true, L);
}
// The last linear's (o, down), whose product goes to the residual rows.
void lora_out(const RowView& rv, const Adapters& l, int linear, const float* x, int ldx, int K, int H, int R, int ldu,
              const Launch& L) {
  lora_down(x, ldx, l.a_out, l.rank, K, l.scale, rv.b.lora, ldu, R, L);
  gemm(rv.b.lora, ldu, l.b[linear], H, l.rank, rv.b.x, H, R, true, L);
}

void attn_layer(const Pass& ps, const AttnBlock& a, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int H = a.hidden, R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  const bool x16 = residual16(ps);
  if (x16) {
    if (!fused) throw std::logic_error("cuda: a bf16 residual's pass runs the fused producers");
    rmsnorm_split(x16_of(rv), H, a.norm.w, a.norm.offset, a.norm.eps, R, H, L, a.w_in.q8_inputs());
    gemm_presplit(a.w_in, a.n_in(), H, rv.b.proj, rv.ld_proj, R, false, L);
  } else {
    norm_and_project(rv, R, a.norm, a.w_in, a.n_in(), H, fused);
  }
  const int ldu = s->m->need.lora;
  if (a.lora) {
    const int out[3] = {nh * hd, nkv * hd, nkv * hd};
    lora_in(rv, a.lora, 3, out, fused ? nullptr : rv.b.h, H, H, R, ldu, L);
  }
  const KvBuf& kb = s->kv[size_t(a.kv)];
  KvRef kv = kb.ref();
  const TreeRows* tree = tree_of(ps);
  // Stage aligned 16-row prefill Strata writes in dead device memory (rv.b.act) and flush contiguously
  // via cudaMemcpyAsync instead of scattering 4-byte writes across PCIe into host-pinned Strata buffers.
  const bool stage_strata =
      !ps.round && !tree && rv.b.act && kb.kq && kb.kq_host && kb.kq == kb.kq_host && kb.ring == 0 &&
      R >= 16 && (R & 15) == 0 && (s->position & 15) == 0;
  const bool stage_grids = kb.copy_slot >= 0;  // copy_slot < 0 keeps kqs/vqs directly on device
  const size_t q_words = size_t(R) * size_t(nkv) * size_t(kb.words);
  const size_t grid_cnt = size_t(R) * size_t(nkv);
  const size_t need_bytes =
      2 * q_words * sizeof(uint32_t) + (stage_grids ? 2 * grid_cnt * sizeof(__half2) : 0);
  const bool use_stage = stage_strata && need_bytes <= size_t(R) * size_t(rv.ld_act) * sizeof(float);
  uint32_t *stg_kq = nullptr, *stg_vq = nullptr;
  __half2 *stg_kqs = nullptr, *stg_vqs = nullptr;
  const size_t goff = size_t(s->position) * size_t(nkv);
  if (use_stage) {
    stg_kq = reinterpret_cast<uint32_t*>(rv.b.act);
    stg_vq = stg_kq + q_words;
    kv.kq = stg_kq;
    kv.vq = stg_vq;
    if (stage_grids) {
      stg_kqs = reinterpret_cast<__half2*>(stg_vq + q_words);
      stg_vqs = stg_kqs + grid_cnt;
      kv.kqs = stg_kqs;
      kv.vqs = stg_vqs;
    } else {
      kv.kqs = kb.kqs + goff;
      kv.vqs = kb.vqs + goff;
    }
  }
  attn_prep(rv.b.proj, rv.ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
            a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, R, s->P, kPosPosition, false, rv.b.qbuf, kv, st, 0, tree,
            use_stage ? s->position : 0, CanvasKeys{}, a.mrope(ps.pos_3d));
  if (use_stage) {
    const size_t woff = strata_words_at(s->position, 0, nkv, kb.words);
    CK(cudaMemcpyAsync(kb.kq + woff, stg_kq, q_words * sizeof(uint32_t), cudaMemcpyDefault, st));
    CK(cudaMemcpyAsync(kb.vq + woff, stg_vq, q_words * sizeof(uint32_t), cudaMemcpyDefault, st));
    if (stage_grids) {
      CK(cudaMemcpyAsync(kb.kqs + goff, stg_kqs, grid_cnt * sizeof(__half2), cudaMemcpyDefault, st));
      CK(cudaMemcpyAsync(kb.vqs + goff, stg_vqs, grid_cnt * sizeof(__half2), cudaMemcpyDefault, st));
    }
  }
  // The attention writes the out projection's input itself, but not in 8 bits: for such weights it stays float32.
  const bool split = fused && !a.w_out.q8_inputs();
  if (ps.reads_cache(kb)) {
    if (const int windows = ps.round ? 0 : s->windows(kb, R)) {
      attn_windows(ps, a, kb, windows, split ? nullptr : rv.b.act, nh * hd);
    } else {
      // The positions in host memory through the stagings, the rows just written among them; a small
      // pass past them, where they are (0074).
      const bool direct = s->reads_direct(kb, R, ps.round);
      if (!direct) s->stage_before(kb, 0, R, st);
      attention(rv.b.qbuf, direct ? kb.ref() : kb.staged(), nh, nkv, hd, R, true, 0, 1.f / std::sqrt(float(hd)), s->P,
                a.gate ? rv.b.proj + nh * hd : nullptr, rv.ld_proj, split ? nullptr : rv.b.act, nh * hd, rv.b.part,
                ps.round ? kRoundSplits : kPrefillSplits, L, 0, 0, tree);
      if (!direct) s->stage_after(kb, st);
    }
  }
  // A round reads the low-bit copy, a few positions the cache; about the switch, its rows replace
  // exact attention's at the switch and past it.
  if (ps.reads_copy(kb)) {
    const bool only = ps.reads_copy_only(kb);
    const cudaEvent_t full = only ? s->copy_before(kb, R, st, true) : nullptr;
    const float scale = 1.f / std::sqrt(float(hd));
    const int from = ps.strata == kStrataBoth ? s->strata_from : 0;
    if (R <= FLYX_MAX_ROWS) {
      attention_strata(rv.b.qbuf, kb.copy_read(only), s->strata, nh, nkv, hd, R, scale, s->P,
                       a.gate ? rv.b.proj + nh * hd : nullptr, rv.ld_proj, split ? nullptr : rv.b.act, nh * hd,
                       rv.b.part, kRoundSplits, L, from, full);
    } else {  // a wide round (generic: float32 out): 16 rows at a time, each as a round's rows at its own position
      const size_t q_row = size_t(nh) * hd;
      for (int a0 = 0; a0 < R; a0 += FLYX_MAX_ROWS) {
        const int k = std::min(FLYX_MAX_ROWS, R - a0);
        shift_params(s->P, s->P_shift, a0, st);
        attention_strata(rv.b.qbuf + a0 * q_row, kb.copy_read(only), s->strata, nh, nkv, hd, k, scale, s->P_shift,
                         a.gate ? rv.b.proj + size_t(a0) * rv.ld_proj + nh * hd : nullptr, rv.ld_proj,
                         rv.b.act + a0 * q_row, nh * hd, rv.b.part, kRoundSplits, L, from, a0 == 0 ? full : nullptr,
                         s->P);
      }
    }
    if (only) s->stage_after(kb, st);
  }
  if (x16) {
    if (!split || !gemm_staged_bf16(a.w_out.q, a.w_out.s, a.w_out.rs, H, nh * hd, x16_of(rv), H, R, L, true))
      throw std::logic_error("cuda: a bf16 residual's out projection is a W4A4 GEMM");
  } else if (split) {
    gemm_presplit(a.w_out, H, nh * hd, rv.b.x, H, R, true, L);
  } else {
    gemm(rv.b.act, nh * hd, a.w_out, H, nh * hd, rv.b.x, H, R, true, L);
  }
  if (a.lora) lora_out(rv, a.lora, FLYX_LIN_O, split ? nullptr : rv.b.act, nh * hd, nh * hd, H, R, ldu, L);
}

void mlp_layer(const Pass& ps, const MlpBlock& f, bool fused) {
  const RowView& rv = ps.v;
  if (residual16(ps)) {  // norm from the bf16 rows, SwiGLU in the gate|up GEMM, the down GEMM added to them
    const int H = f.hidden, I = f.inter, R = ps.rows;
    if (!fused) throw std::logic_error("cuda: a bf16 residual's pass runs the fused producers");
    rmsnorm_split(x16_of(rv), H, f.norm.w, f.norm.offset, f.norm.eps, R, H, rv.L, f.w_gu.q8_inputs());
    gemm_glu_presplit(f.w_gu, I, H, R, rv.L);
    if (!gemm_on_glu_bf16(f.w_down.q, f.w_down.s, f.w_down.rs, H, I, x16_of(rv), H, R, true, rv.L))
      throw std::logic_error("cuda: a bf16 residual's down projection is a W4A4 GEMM");
    return;
  }
  if (!f.lora) {
    mlp_body(rv.b.x, f, ps.rows, rv.b.h, rv.b.proj, rv.ld_proj, rv.b.act, rv.ld_act, rv.L, fused);
    return;
  }
  // The block's own adapters (ABI 43): the gate and up rows take their products before the SwiGLU, so the gate|up GEMM
  // and then the SwiGLU, never the SwiGLU in the GEMM's epilogue (nor block weights, which bind refuses with them).
  const int H = f.hidden, R = ps.rows, I = f.inter, ldu = ps.s->m->need.lora;
  const Norm& n = f.norm;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  const int gu[2] = {I, I};
  if (fused) {
    rmsnorm_split(rv.b.x, H, n.w, n.offset, n.eps, R, H, L, f.w_gu.q8_inputs());
    gemm_presplit(f.w_gu, 2 * I, H, rv.b.proj, rv.ld_proj, R, false, L);
    lora_in(rv, f.lora, 2, gu, nullptr, H, H, R, ldu, L);
    swiglu_split(rv.b.proj, rv.ld_proj, I, R, L, f.w_down.q8_inputs());
    gemm_presplit(f.w_down, H, I, rv.b.x, H, R, true, L);
    lora_out(rv, f.lora, 2, nullptr, I, I, H, R, ldu, L);
    return;
  }
  rmsnorm(rv.b.x, H, n.w, n.offset, n.eps, rv.b.h, H, R, H, st);
  gemm(rv.b.h, H, f.w_gu, 2 * I, H, rv.b.proj, rv.ld_proj, R, false, L);
  lora_in(rv, f.lora, 2, gu, rv.b.h, H, H, R, ldu, L);
  swiglu(rv.b.proj, rv.ld_proj, I, rv.b.act, rv.ld_act, R, st);
  gemm(rv.b.act, rv.ld_act, f.w_down, H, I, rv.b.x, H, R, true, L);
  lora_out(rv, f.lora, 2, rv.b.act, rv.ld_act, I, H, R, ldu, L);  // the SwiGLU rows, rv.ld_act apart
}

void head(const Pass& ps, const HeadBlock& h, bool last_only) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int V = h.vocab;
  const int r0 = last_only ? ps.rows - 1 : 0, n = last_only ? 1 : ps.rows;
  cudaStream_t st = rv.L.st;
  if (ps.round && n > FLYX_MAX_ROWS) {
    // A wide round: the head and the choice 16 rows at a time (the logits hold a round's 16), each row's choice what
    // a round of 16 makes.  A tree's row chooses at its depth's position and its penalties count its ancestors'
    // drafts; for the relaxed rules each row's statistics, and each draft's logit in its parent's row (child_logits:
    // what accept_tree reads of the logits), are kept as the rows go.
    const TreeRows* tree = tree_of(ps);
    for (int a = 0; a < n; a += FLYX_MAX_ROWS) {
      const int k = std::min(FLYX_MAX_ROWS, n - a);
      head_logits(h, rv, a, k, ps.cert, s->cert, s->m->opt.kernels);
      if (ps.pen) penalize_rows(rv.logits, V, V, k, s->G, s->reply, s->drafts, st, tree, a);
      choose_rows(rv.logits, V, V, k, s->P, a + 1, s->G, a == 0 ? s->mask : nullptr, s->cz, s->ci, s->argv, s->argi,
                  rv.b.choice + a, st, tree, a);  // the mask is the round's first row's
      if (ps.relaxed) {
        row_stats(rv.logits, V, V, k, s->G, s->accept_stats + 3 * a, st);
        child_logits(rv.logits, V, a, k, s->drafts, ps.p->width, tree, s->child_z, st);
      }
    }
    return;
  }
  head_logits(h, rv, r0, n, ps.cert, s->cert, s->m->opt.kernels, residual16(ps));
  // Penalties first (a request that has one): in a round, row r was given the
  // drafts before it, which count as the reply for it.
  // A tree's row: its ancestors' drafts, its depth's position.
  const TreeRows* tree = tree_of(ps);
  if (ps.pen) penalize_rows(rv.logits, V, V, n, s->G, s->reply, ps.round ? s->drafts : nullptr, st, tree);
  // Row r of the logits picks the token at position P->position + r0 + 1 + r.
  choose_rows(rv.logits, V, V, n, s->P, r0 + 1, s->G, s->mask, s->cz, s->ci, s->argv, s->argi, rv.b.choice, st, tree);
  if (ps.relaxed) row_stats(rv.logits, V, V, n, s->G, s->accept_stats, st);
}

void draft_context(const Pass& ps, const DFlashBlock& b, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const KvBuf& kv = s->kv[size_t(b.kv)];
  const int H = b.hidden, R = ps.rows, F = b.n_taps * b.target_hidden;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  gemm(rv.b.feats, rv.ld_feats, b.fc, H, F, rv.b.cx, H, R, false, L);
  norm_project(rv.b.cx, rv.b.h, rv.b.ckv, rv.ld_ckv, R, b.hidden_norm, b.ctx, b.ctx_width(), H, L, fused);
  draft_ctx_kv(rv.b.ckv, rv.ld_ckv, b.k_norms.w, b.k_norms.eps, b.theta, b.layers(), b.nkv, b.hd, R, s->P,
               ps.round ? kPosStart : kPosPosition, ps.round, kv.k16(), kv.v16(), kv.layer_stride, st);
}

// The selector's table over the slots the tree reaches (slots - 1 at most: the last slot never has candidates after
// a slide), then the tree.  A canvas that follows another round drafts DV's first tree; the test's oracle a chain.
void dv_draft(flyx_seq* s, const DvBlock& v, int width, const Launch& L) {
  flyx_seq::DvRows& d = s->dv;
  const int depths = std::min(v.slots - 1, width);
  cudaStream_t st = L.st;
  dv_selector_table(s->P, d.canvas, v.sel_ta, v.sel_tb, v.sel_rank, v.sel_alpha, v.sel_temp, depths, d.table, st);
  if (v.tree == FLYX_DV_TREE_BEAM)
    dv_tree_beam(s->P, d.canvas, d.table, width, v.beam, v.cov, depths, v.mask_token, s->oracle, s->drafts, nullptr,
                 s->tree, st);
  else
    dv_tree_pool(s->P, d.canvas, d.table, width, v.pool_ranks, depths, v.mask_token, s->oracle, s->drafts, nullptr,
                 s->tree, st);
}

// The canvas pass: slot w a row at P->start + 1 + w (P->start the round's anchor, its K/V written by the round's
// pass and never moved by the compaction), through the target's attention and MLP blocks with their base weights
// (never an adapter of the target's own: bifill-dv's canvas rows call the base linears) and the layer's canvas
// norms; each linear's adapter as two more GEMMs whose product adds to the base's (y = W x + scale b (a x)).  Its K/V
// go to the scratch slots, its attention reads the text up to the anchor and those slots (CanvasKeys), never the
// low-bit copy.  K/V in tiers: through the
// views, the text's host part staged once a layer (stage_text) and the slots written to the staging itself (only this
// pass reads them); read in place, every canvas row fetched each of the text's keys over the bus (MiniCPM5-1B, 4273
// tokens in host memory: a round of 221 ms, 35 staged).  Then fnormc, the head's weight over the slots in float32,
// what they leave, and the slide.
// The pass's GEMMs past 16 rows take bf16 inputs (Launch::draft): only drafting reads the canvas, which must be
// deterministic, not row-count invariant (numerics.md).  Past 16 slots and up to 64, on bf16 weights, each input is
// rounded to bf16 once (by its producer: rmsnorm_split, swiglu_split, the adapters' a; the attention's output by
// stage_bf16) and a layer's GEMMs on the same rows go in one launch each (gemm_draft): a base weight with its adapter's
// a, the adapters' b side by side.  A fused region as the others are: the arithmetic of each GEMM on its own (one part,
// the chunks K gives it, whatever the table), so option fusion=0 runs them one by one for the same canvas.
namespace {
MtOut draft_out(const Lin& W, int N, float* y, int ldy, bool accumulate, int x_col = 0) {
  return MtOut{W.w, y, nullptr, N, ldy, x_col, int(accumulate), 0};
}
MtOut draft_out16(const Lin& W, int N, bf16* y16, int ldy) { return MtOut{W.w, nullptr, y16, N, ldy, 0, 0, 0}; }
MtGroups draft_group(std::initializer_list<MtOut> outs) {
  MtGroups g;
  for (const MtOut& o : outs) g.g[g.n++] = o;
  return g;
}
bool bf16_lin(const Lin& W) { return W.w && !W.q && !W.packed(); }
bool drafts_grouped(const flyx_bmodel* m, const DvBlock& v) {
  if (!m->opt.fusion || v.slots <= 16 || v.slots > 64) return false;
  for (int j = 0; j < v.n_layers; ++j) {
    const DvLayer& l = block<DvLayer>(m, v.first_layer + j);
    const AttnBlock& a = block<AttnBlock>(m, l.attn);
    const MlpBlock& f = block<MlpBlock>(m, l.mlp);
    if (l.scale != 1.f) return false;  // the a's write bf16 products, no scale step (bifill-dv's adapters have none)
    for (const Lin* W : {&a.w_in, &a.w_out, &f.w_gu, &f.w_down, &l.a_qkv, &l.a_o, &l.a_gu, &l.a_down, &l.b_q, &l.b_k,
                         &l.b_v, &l.b_o, &l.b_gate, &l.b_up, &l.b_down})
      if (!bf16_lin(*W)) return false;
    if (a.hidden % 32 || a.nh * a.hd % 32 || f.inter % 32 || l.rank % 32) return false;
  }
  return true;
}
}  // namespace

void dv_context(const Pass& ps, const DvBlock& v) {
  auto* s = ps.s;
  const auto* m = s->m;
  flyx_seq::DvRows& d = s->dv;
  const auto& E = block<EmbedBlock>(m, v.embed);
  const auto& head = block<HeadBlock>(m, v.head);
  const int N = v.slots, H = v.hidden, ldp = m->need.proj, lda = m->need.act, ldl = d.ld_lora;
  Launch L{ps.v.L.st, &d.scratch, ps.v.L.opt, nullptr, ps.v.L.side};
  L.draft = true;
  cudaStream_t st = L.st;
  const CanvasKeys keys{d.s0};
  const bool grouped = drafts_grouped(m, v);
  const bf16* rows16 = d.scratch.abuf;  // where the producers leave a grouped GEMM's bf16 input
  canvas_input(s->P, d.canvas, N, v.mask_token, E.table, v.mask_emb, v.res_in.w, v.res_out.w, H, d.ug, d.x, L);
  for (int j = 0; j < v.n_layers; ++j) {
    const DvLayer& l = block<DvLayer>(m, v.first_layer + j);
    const AttnBlock& a = block<AttnBlock>(m, l.attn);
    const MlpBlock& f = block<MlpBlock>(m, l.mlp);
    const int r = l.rank, nh = a.nh, nkv = a.nkv, hd = a.hd, qw = nh * hd, kvw = nkv * hd, I = f.inter;
    const KvBuf& kb = s->kv[size_t(a.kv)];
    const KvRef kv = kb.staged();  // the views, FP8 scales past the device part in their staging
    if (grouped) {
      rmsnorm_split(d.x, H, l.norm.w, l.norm.offset, l.norm.eps, N, H, L, false);
      gemm_draft(rows16, H, H, draft_group({draft_out(a.w_in, a.n_in(), d.proj, ldp, false),
                                            draft_out16(l.a_qkv, 3 * r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_q, qw, d.proj, ldp, true),
                                                draft_out(l.b_k, kvw, d.proj + qw, ldp, true, r),
                                                draft_out(l.b_v, kvw, d.proj + qw + kvw, ldp, true, 2 * r)}), N, L);
    } else {
      rmsnorm(d.x, H, l.norm.w, l.norm.offset, l.norm.eps, d.h, H, N, H, st);
      gemm(d.h, H, a.w_in, a.n_in(), H, d.proj, ldp, N, false, L);
      lora_down(d.h, H, l.a_qkv.w, 3 * r, H, l.scale, d.lora, ldl, N, L);
      gemm(d.lora, ldl, l.b_q, qw, r, d.proj, ldp, N, true, L);
      gemm(d.lora + r, ldl, l.b_k, kvw, r, d.proj + qw, ldp, N, true, L);
      gemm(d.lora + 2 * r, ldl, l.b_v, kvw, r, d.proj + qw + kvw, ldp, N, true, L);
    }
    attn_prep(d.proj, ldp, 0, qw, qw + kvw, a.q_norm.w, a.k_norm.w, a.q_norm.offset, a.q_norm.eps, a.theta, a.rot, nh,
              nkv, hd, N, s->P, kPosStart, false, d.q, kv, st, 0, nullptr, 0, keys, a.mrope());
    s->stage_text(kb, st);
    attention(d.q, kv, nh, nkv, hd, N, v.causal, N, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0, d.act, qw, d.part,
              d.splits, L, 0, 0, nullptr, keys);
    if (grouped) {
      gemm_draft(stage_bf16(d.act, qw, N, qw, L), qw, qw,
                 draft_group({draft_out(a.w_out, H, d.x, H, true), draft_out16(l.a_o, r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_o, H, d.x, H, true)}), N, L);
      rmsnorm_split(d.x, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, N, H, L, false);
      gemm_draft(rows16, H, H, draft_group({draft_out(f.w_gu, 2 * I, d.proj, ldp, false),
                                            draft_out16(l.a_gu, 2 * r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_gate, I, d.proj, ldp, true),
                                                draft_out(l.b_up, I, d.proj + I, ldp, true, r)}), N, L);
      swiglu_split(d.proj, ldp, I, N, L, false);
      gemm_draft(rows16, I, I, draft_group({draft_out(f.w_down, H, d.x, H, true),
                                            draft_out16(l.a_down, r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_down, H, d.x, H, true)}), N, L);
      continue;
    }
    gemm(d.act, qw, a.w_out, H, qw, d.x, H, N, true, L);
    lora_down(d.act, qw, l.a_o.w, r, qw, l.scale, d.lora, ldl, N, L);
    gemm(d.lora, ldl, l.b_o, H, r, d.x, H, N, true, L);
    rmsnorm(d.x, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, d.h, H, N, H, st);
    gemm(d.h, H, f.w_gu, 2 * I, H, d.proj, ldp, N, false, L);
    lora_down(d.h, H, l.a_gu.w, 2 * r, H, l.scale, d.lora, ldl, N, L);
    gemm(d.lora, ldl, l.b_gate, I, r, d.proj, ldp, N, true, L);
    gemm(d.lora + r, ldl, l.b_up, I, r, d.proj + I, ldp, N, true, L);
    swiglu(d.proj, ldp, I, d.act, lda, N, st);
    gemm(d.act, lda, f.w_down, H, I, d.x, H, N, true, L);
    lora_down(d.act, lda, l.a_down.w, r, I, l.scale, d.lora, ldl, N, L);
    gemm(d.lora, ldl, l.b_down, H, r, d.x, H, N, true, L);
  }
  rmsnorm(d.x, H, v.norm.w, v.norm.offset, v.norm.eps, d.h, H, N, H, st);
  gemm(d.h, H, head.w, v.vocab, H, d.logits, v.vocab, N, false, L);
  canvas_stats(d.logits, v.vocab, v.vocab, E.table, d.h, H, H, v.sel_h, v.sel_rank, N, d.stats, st);
  dv_slide(s->P, d.stats, N, H, v.sel_rank, v.fill, v.mask_token, d.canvas, st);
}

// The canvas pass of a round captured as a graph, as the body of a conditional node that dv_live sets from P->noop: a
// round launched behind the one that ended the reply (it commits nothing, and its pass would leave the canvas as it
// was) skips it (the reply's last round else costs a canvas pass more).  Launched eagerly, it runs (dv_slide moves
// nothing then).
void dv_context_unless_noop(const Pass& ps, const DvBlock& v) {
  auto* s = ps.s;
  cudaStream_t st = ps.v.L.st;
  cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  size_t n_deps = 0;
  if (!s->dv.body || !s->dv.skip_noop || cudaStreamIsCapturing(st, &capturing) != cudaSuccess ||
      capturing != cudaStreamCaptureStatusActive)
    return dv_context(ps, v);
  cudaGraphConditionalHandle live{};
  CK(cudaStreamGetCaptureInfo(st, &capturing, nullptr, &graph));
  CK(cudaGraphConditionalHandleCreate(&live, graph, 0, 0));
  dv_live(live, s->P, st);
  CK(cudaStreamGetCaptureInfo(st, &capturing, nullptr, &graph, &deps, nullptr, &n_deps));
  cudaGraphNodeParams cond{};
  cond.type = cudaGraphNodeTypeConditional;
  cond.conditional.handle = live;
  cond.conditional.type = cudaGraphCondTypeIf;
  cond.conditional.size = 1;
  cudaGraphNode_t node = nullptr;
  CK(cudaGraphAddNode(&node, graph, deps, nullptr, n_deps, &cond));
  CK(cudaStreamUpdateCaptureDependencies(st, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  CK(cudaStreamBeginCaptureToGraph(s->dv.body, cond.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                   cudaStreamCaptureModeThreadLocal));
  Pass body = ps;
  body.v.L.st = s->dv.body;
  cudaGraph_t captured = nullptr;
  try {
    dv_context(body, v);
  } catch (...) {
    cudaStreamEndCapture(s->dv.body, &captured);
    throw;
  }
  CK(cudaStreamEndCapture(s->dv.body, &captured));
}

void compact_path(const Pass& ps) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const auto* m = s->m;
  cudaStream_t st = rv.L.st;
  compact_kv(s->tree_kv, s->n_tree_kv, s->tree_kv_bytes, s->P, s->tree, st);
  if (s->gdn_layers) {
    gdn_compact(s->gdn_layers, int(s->gdn.size()), s->gdn[0].replay, s->P, s->tree, st);
  } else {
    for (const auto& g : s->gdn) gdn_compact(nullptr, 1, g.replay, s->P, s->tree, st);
  }
  // What the drafters' context reads: the rows' tokens, final hidden states and taps.
  compact_rows(rv.b.tokens, 1, 1, s->P, s->tree, st);
  compact_rows(rv.b.x, size_t(m->hidden), m->hidden, s->P, s->tree, st);
  if (rv.b.feats) compact_rows(rv.b.feats, size_t(rv.ld_feats), rv.ld_feats, s->P, s->tree, st);
}

// Fold every recurrent state to the round's accepted rows.
void commit(const Pass& ps) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  if (ps.p->width == 0) return;  // plain rounds wrote the next window and state in the pass
  // Every layer in two launches when they are alike (the same arithmetic as a layer at a time).
  if (s->gdn_layers) {
    const GdnBuf& g = s->gdn[0];
    if (gdn_fold_layers(s->gdn_layers, int(s->gdn.size()), g.s_stride, g.w_stride, g.replay, kSlotLast, kSlotAfterLast,
                        kSlotLast, s->P, &s->P->last_n, rv.L.st))
      return;
  }
  for (auto& g : s->gdn) {
    gdn_recur(g.S, g.s_stride, kSlotLast, kSlotAfterLast, g.replay, kSlotLast, s->P, nullptr, 0, &s->P->last_n,
              kSkipIfNoop, rv.L.st);
    gdn_conv_fold(g.win, g.w_stride, kSlotLast, kSlotAfterLast, g.replay, kSlotLast, s->P, 0, &s->P->last_n, true,
                  rv.L.st);
  }
}

}  // namespace flyx::cuda
