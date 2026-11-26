// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Sequence state and row buffers, sized from the blocks and the state table.
#include "backends/cuda/backend.h"
#include "flyx/gemm_tables.inc"

#include <algorithm>

namespace flyx::cuda {

RowNeeds row_needs(const flyx_bmodel* m) {
  RowNeeds n;
  int k = 0;  // largest GEMM K
  auto ks = [&](std::initializer_list<int> l) { k = std::max(k, std::max(l)); };
  for (const Block& blk : m->blocks) {
    if (const auto* a = std::get_if<AttnBlock>(&blk)) {
      n.proj = std::max(n.proj, a->n_in());
      n.act = std::max(n.act, a->nh * a->hd);
      n.q = std::max(n.q, a->nh * a->hd);
      n.attn_state = std::max(n.attn_state, a->nh * kFlashStateFloats(a->hd));
      n.split_state = std::max(n.split_state, a->nh * kSplitStateFloats(a->hd));
      n.lora = std::max(n.lora, 3 * a->lora.rank);
      ks({a->hidden, a->nh * a->hd, a->lora.rank});
    } else if (const auto* g = std::get_if<GdnBlock>(&blk)) {
      n.proj = std::max(n.proj, g->n_in());
      n.act = std::max(n.act, g->v_width());
      n.gdn_o = std::max(n.gdn_o, g->v_width());
      ks({g->hidden, g->v_width()});
    } else if (const auto* f = std::get_if<MlpBlock>(&blk)) {
      n.proj = std::max(n.proj, 2 * f->inter);
      n.act = std::max(n.act, f->inter);
      if (!f->w_gu.packed() && !f->w_down.packed()) n.glu = true;
      n.lora = std::max(n.lora, 2 * f->lora.rank);
      ks({f->hidden, f->inter, f->lora.rank});
    } else if (const auto* h = std::get_if<HeadBlock>(&blk)) {
      ks({h->hidden});
    } else if (std::holds_alternative<MtpBlock>(blk)) {
      n.mtp = true;
      ks({2 * m->hidden});
    } else if (const auto* d = std::get_if<DFlashBlock>(&blk)) {
      n.feats = d->n_taps * d->target_hidden;
      n.cx = d->hidden;
      n.ckv = d->ctx_width();
      ks({d->hidden, d->inter, d->nh * d->hd, n.feats});
    }
  }
  n.k_max = (k + 63) / 64 * 64;
  return n;
}

int max_drafts(const Block& drafter) {
  // DFlash2: its block, or a round's rows drafted at once (draft_block's wide block); DSpine: the block it was made for
  if (const auto* d = std::get_if<DFlashBlock>(&drafter))
    return d->dspine ? d->block - 1 : std::max(d->block, FLYX_MAX_ROWS) - 1;
  if (const auto* v = std::get_if<DvBlock>(&drafter)) return v->tree_width;
  return std::holds_alternative<MtpBlock>(drafter) ? FLYX_MAX_ROWS - 1 : 0;
}

// Row buffers for R rows; `prefill`: the prefill set (attention partials for
// the prefill splits, GDN prefill scratch, windowed attention softmax state),
// else a session's round set.
template <class Alloc>
void alloc_rows(const flyx_bmodel* m, int R, bool prefill, RowBufs* b, Alloc&& alloc) {
  const RowNeeds& n = m->need;
  const int H = m->hidden;
  auto align_f = [](size_t floats) { return (floats + 63) / 64 * 64; };
  auto f = [&](size_t count) { return static_cast<float*>(alloc(count * sizeof(float))); };
  b->tokens = static_cast<int*>(alloc(size_t(R) * sizeof(int)));
  b->choice = static_cast<int*>(alloc(size_t(R) * sizeof(int)));
  b->x = f(size_t(R) * H);
  if (n.lora) b->lora = f(size_t(R) * n.lora);  // live beside every buffer below: a layer's linears read it
  // Prefill attention runs on tensor cores (no partials) above FLYX_MAX_ROWS rows, every head dim the backend
  // loads (attention_head_dim) included; the split kernel then only takes a last chunk of at most that many rows.
  size_t part = 0;
  for (const Block& blk : m->blocks) {
    if (const auto* a = std::get_if<AttnBlock>(&blk)) {
      const bool flash = m->opt.kernels.attention != kAttnSplit;
      part = std::max(part, prefill ? attention_partial_floats(flash ? std::min(R, FLYX_MAX_ROWS) : R, a->nh, a->hd,
                                                               kPrefillSplits)
                                    : attention_partial_floats(R, a->nh, a->hd, kRoundSplits));
    } else if (const auto* d = std::get_if<DFlashBlock>(&blk); d && !prefill) {
      part = std::max(part, attention_partial_floats(d->block, d->nh, d->hd, kRoundSplits));
    }
  }
  if (n.feats) b->feats = f(size_t(R) * n.feats);  // DFlash2 taps survive across layers

  if (prefill) {
    // Disjoint layer-local scratch regions for prefill (verified across mlp, attn, gdn, mtp_context, draft_context, head):
    //   Slot P: proj | (mcat + mhn) | ckv
    //   Slot A: act  | gdn pre      | mx | cx
    //   Slot B: h    | gdn_o        | (qbuf + max(attn_state, part))
    size_t pre_floats = 0;
    const int Rg = std::max(R, gdn_chunk_rows());
    if (!m->states.gdn.empty())
      for (const GdnDims& gd : m->states.gdn) pre_floats = std::max(pre_floats, GdnBufs::prefill(Rg, gd).floats());

    size_t slot_p = size_t(R) * n.proj;
    if (n.mtp) slot_p = std::max(slot_p, align_f(size_t(R) * 2 * H) + size_t(R) * H);
    if (n.feats) slot_p = std::max(slot_p, size_t(R) * n.ckv);
    float* base_p = f(slot_p);
    b->proj = base_p;
    if (n.mtp) {
      b->mcat = base_p;
      b->mhn = base_p + align_f(size_t(R) * 2 * H);
    }
    if (n.feats) b->ckv = base_p;

    size_t slot_a = std::max(size_t(R) * n.act, pre_floats);
    if (n.mtp) slot_a = std::max(slot_a, size_t(R) * H);
    if (n.feats) slot_a = std::max(slot_a, size_t(R) * n.cx);
    float* base_a = f(slot_a);
    b->act = base_a;
    if (!m->states.gdn.empty()) {
      b->pre = GdnBufs::prefill(Rg, m->states.gdn[0]);
      b->pre.base = base_a;
    }
    if (n.mtp) b->mx = base_a;
    if (n.feats) b->cx = base_a;

    const size_t q_off = align_f(size_t(R) * n.q);
    const size_t part_span = align_f(part);
    // Windowed split attention's state of a round's rows, twice: read from one, written to the other (SplitWindow).
    const size_t split_st = 2 * size_t(std::min(R, FLYX_MAX_ROWS)) * n.split_state;
    const size_t attn_aux = std::max(size_t(R) * n.attn_state, part_span + split_st);
    const size_t slot_b = std::max({size_t(R) * H, size_t(R) * n.gdn_o, q_off + attn_aux});
    float* base_b = f(slot_b);
    b->h = base_b;
    b->gdn_o = base_b;
    b->qbuf = base_b;
    if (n.attn_state > 0) b->attn_state = base_b + q_off;
    b->part = base_b + q_off;
    if (n.split_state > 0) b->split_state = b->part + part_span;
  } else {
    // Round buffers: mtp_draft keeps mx live across the MTP layer, so mx stays separate from act/proj/h.
    // gdn_o and qbuf belong to distinct layer types and safely alias.
    b->h = f(size_t(R) * H);
    b->proj = f(size_t(R) * n.proj);
    b->act = f(size_t(R) * n.act);
    float* oq = f(std::max(size_t(R) * n.gdn_o, size_t(R) * n.q));
    b->gdn_o = oq;
    b->qbuf = oq;
    b->part = f(part);
    if (n.feats) {
      b->cx = f(size_t(R) * n.cx);
      b->ckv = f(size_t(R) * n.ckv);
    }
    if (n.mtp) {
      b->mcat = f(size_t(R) * 2 * H);
      b->mx = f(size_t(R) * H);
      b->mhn = f(size_t(R) * H);
    }
  }
  // Both allocators give zeroed memory (the tiles' counters too), so carving is all; it does no device work, so
  // seq_bytes' dry run carves too.
  // A prefill set's rows in float32 too when NVFP4 weights take NVFP4 inputs.
  const bool f32_rows = prefill && m->opt.kernels.prefill_fp4;
  const bool need_glu = prefill && n.glu;
  const size_t scratch_size = scratch_bytes(R, n.k_max, n.q8, f32_rows, need_glu, f32_rows);
  b->scratch = scratch_carve(alloc(scratch_size), scratch_size, R, n.k_max, n.q8, f32_rows, need_glu, f32_rows);
}

template <class Alloc>
void alloc_cert_scratch(const flyx_bmodel* m, CertScratch* c, Alloc&& alloc) {
  if (!m->opt.kernels.head_cert || m->opt.kernels.head_fp8) return;
  c->norms = static_cast<float*>(alloc(2 * FLYX_MAX_ROWS * sizeof(float)));
  c->key = static_cast<int*>(alloc(FLYX_MAX_ROWS * sizeof(int)));
  c->cols = static_cast<int*>(alloc(size_t(m->vocab) * sizeof(int)));
  c->count = static_cast<int*>(alloc(sizeof(int)));
}

RowView rows_of(flyx_seq* s, RowBufs* bufs) {
  RowBufs& b = bufs ? *bufs : s->own;
  RowView v;
  v.b = b;
  v.logits = s->logits;
  v.ld_proj = s->ld_proj;
  v.ld_act = s->ld_act;
  v.ld_feats = s->ld_feats;
  v.ld_ckv = s->ld_ckv;
  v.L = Launch{s->st, &b.scratch, &s->m->opt.kernels, nullptr, &s->side};
  v.L.round_rows = bufs == nullptr;  // the sequence's own buffers hold rounds; prefill passes borrow the shared ones
  return v;
}

// The model's buffers for a multimodal pass's inputs, grown to hold embed_rows embedding rows with their row
// indices and position_rows rows of 3-D positions (two sets of three); 0 leaves a buffer as it is.  Under
// prefill_mu.
void reserve_prefill_inputs(flyx_bmodel* m, size_t embed_rows, size_t position_rows) {
  const size_t need_e = embed_rows * size_t(m->hidden);
  if (need_e > m->prefill_embeds_cap) {
    if (m->prefill_embeds) CK(cudaFree(m->prefill_embeds));
    m->prefill_embeds = nullptr;
    m->prefill_embeds_cap = 0;
    m->prefill_embeds = static_cast<float*>(device_malloc(need_e * sizeof(float)));
    m->prefill_embeds_cap = need_e;
  }
  if (embed_rows > m->prefill_embed_rows_cap) {
    if (m->prefill_embed_rows) CK(cudaFree(m->prefill_embed_rows));
    m->prefill_embed_rows = nullptr;
    m->prefill_embed_rows_cap = 0;
    m->prefill_embed_rows = static_cast<int32_t*>(device_malloc(embed_rows * sizeof(int32_t)));
    m->prefill_embed_rows_cap = embed_rows;
  }
  const size_t need_p = position_rows * 6;
  if (need_p > m->prefill_pos3d_cap) {
    if (m->prefill_pos3d) CK(cudaFree(m->prefill_pos3d));
    m->prefill_pos3d = nullptr;
    m->prefill_pos3d_cap = 0;
    m->prefill_pos3d = static_cast<int32_t*>(device_malloc(need_p * sizeof(int32_t)));
    m->prefill_pos3d_cap = need_p;
  }
}

// The prefill buffers all sessions share, for passes of up to `rows` rows: a
// wider plan replaces them; sessions only borrow them during a pass.  Allocated
// from one VMM slab when the driver supports VMM so tiered strata pools can
// share the same physical memory during decode.
int ensure_prefill_rows(flyx_bmodel* m, int rows) {
  std::lock_guard<std::mutex> lock(m->prefill_mu);
  if (rows <= m->prefill_rows) return 0;
  try {
    if (m->prefill_ev) CK(cudaEventSynchronize(m->prefill_ev));
    // Any session that mapped the previous VMM slab retains its own shared_ptr to the old handles.
    m->prefill_owner = nullptr;
    m->prefill_rows = 0;
    m->prefill_bufs = {};
    prefill_vmm_free(m);
    for (void* q : m->prefill_allocs) cudaFree(q);
    m->prefill_allocs.clear();
    m->prefill_bytes = 0;
    auto align256 = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    size_t total = 0;
    RowBufs probe{};
    alloc_rows(m, rows, true, &probe, [&](size_t bytes) {
      const size_t off = total;
      total += align256(bytes);
      return reinterpret_cast<void*>(uintptr_t(256) + off);
    });
    // Never in host memory (device_room, device_malloc): a pass reads them a layer at a time.  Where the copies of the
    // head kept for speed left no room, lowering fails and the engine has the backend give those back (trim).
    device_room(total);
    size_t actual = 0;
    void* base = prefill_vmm_alloc(m, total, &actual);
    if (!base) {
      base = device_malloc(total);
      CK(cudaMemset(base, 0, total));  // as a session's buffers (flyx_seq::alloc)
      m->prefill_allocs.push_back(base);
      actual = total;
    }
    m->prefill_bytes = actual;
    size_t off = 0;
    alloc_rows(m, rows, true, &m->prefill_bufs, [&](size_t bytes) {
      void* q = bytes ? static_cast<unsigned char*>(base) + off : nullptr;
      off += align256(bytes);
      return q;
    });
    if (m->prefill_bufs.scratch.row_scale && m->prefill_bufs.scratch.q8) {
      const uintptr_t lo = reinterpret_cast<uintptr_t>(m->prefill_bufs.scratch.row_scale);
      const uintptr_t hi = reinterpret_cast<uintptr_t>(m->prefill_bufs.scratch.q8);
      if (hi > lo) {
        m->prefill_zero_ptr = m->prefill_bufs.scratch.row_scale;
        m->prefill_zero_bytes = size_t(hi - lo);
      }
    }
    // Pre-allocate multimodal prefill staging when the model has vision blocks so memory() reports it before planning.
    bool has_vision = false;
    for (const Block& b : m->blocks)
      if (std::holds_alternative<VisionStemBlock>(b)) {
        has_vision = true;
        break;
      }
    if (has_vision) reserve_prefill_inputs(m, rows, rows);
    CK(cudaDeviceSynchronize());  // zero before any session's stream uses them: those do not wait for this one
  } catch (const std::exception& e) {
    prefill_vmm_free(m);
    for (void* q : m->prefill_allocs) cudaFree(q);
    m->prefill_allocs.clear();
    m->prefill_bytes = 0;
    m->prefill_bufs = {};
    m->prefill_rows = 0;
    return fail(std::string("prefill buffers: ") + e.what());
  }
  m->prefill_rows = rows;
  return 0;
}

void build_seq(flyx_bmodel* m, int32_t max_ctx, flyx_seq* s);

// The layout's method for the sequence: exact, strata with a low-bit copy of 2 to 4 bits, or exact
// below a switch and strata past it. The switch (unless the option sets it): kStrataHostExact
// positions past the device's part; a session whose every position is exact keeps no copy.
bool take_method(const flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq* s) {
  if (!layout) return false;
  if (layout->method == FLYX_KV_READ_ALL) return true;
  if (layout->method == FLYX_KV_AUTO) {
    const int asked = m->opt.kernels.strata_from;
    const long long past = (long long)std::max(0, layout->device_positions) + kStrataHostExact;
    const int from = asked >= 0 ? asked : int(std::min<long long>(INT32_MAX, past));
    if (from >= layout->max_ctx) return true;
    s->strata_from = from;
  } else if (layout->method != FLYX_KV_STRATA) {
    return false;
  }
  s->kv_method = FLYX_KV_STRATA;
  s->kv_bits = layout->method_arg ? layout->method_arg : 2;
  return s->kv_bits >= 2 && s->kv_bits <= 4;
}

uint64_t query_pinned_budget(const flyx_bmodel* m) {
  if (m->opt.pageable_kv == PageableKvMode::off) return UINT64_MAX;
  if (m->opt.pageable_kv == PageableKvMode::on) return 0;
  if (m->opt.pinned_host_mb >= 0) return uint64_t(m->opt.pinned_host_mb) << 20;
  return host_pinned_budget();
}

int seq_new_in(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq** out) {
  // Sample the WDDM pinned budget once so dry-run sizing and real allocation choose the identical hybrid split.
  const uint64_t pinned_budget = query_pinned_budget(m);
  // Dry-run first to measure the exact arena sizes so the real build performs one device and one host allocation
  // instead of hundreds of per-layer cudaMalloc/cudaHostAlloc calls.
  flyx_seq probe;
  if (!take_method(m, layout, &probe)) return fail("seq_new: a K/V layout this backend does not read");
  probe.dry = true;
  probe.pinned_budget = pinned_budget;
  probe.device_positions = std::max(0, layout->device_positions);
  probe.copy_host = std::max(0, layout->copy_host);
  try {
    build_seq(m, layout->max_ctx, &probe);
  } catch (const std::exception& e) {
    return fail(std::string("seq_new: ") + e.what());
  }

  auto s = std::make_unique<flyx_seq>();
  if (!take_method(m, layout, s.get())) return fail("seq_new: a K/V layout this backend does not read");
  s->pinned_budget = pinned_budget;
  s->device_positions = std::max(0, layout->device_positions);
  s->copy_host = std::max(0, layout->copy_host);
  s->arena_dev_bytes = probe.arena_dev_bytes;
  s->arena_host_bytes = probe.arena_host_bytes;
  try {
    build_seq(m, layout->max_ctx, s.get());
  } catch (const std::exception& e) {
    return fail(std::string("seq_new: ") + e.what());
  }
  *out = s.release();
  return 0;
}

int seq_new(flyx_bmodel* m, int32_t max_ctx, flyx_seq** out) {
  const flyx_seq_layout all{max_ctx, INT32_MAX, FLYX_KV_READ_ALL, 0, 0};
  return seq_new_in(m, &all, out);
}

int seq_size(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq_size* out) {
  flyx_seq s;
  if (!take_method(m, layout, &s)) return fail("a K/V layout this backend does not read");
  s.dry = true;
  s.pinned_budget = query_pinned_budget(m);
  s.device_positions = std::max(0, layout->device_positions);
  s.copy_host = std::max(0, layout->copy_host);
  size_t room = 0;  // for its round graphs: the device memory the driver takes for them
  try {
    build_seq(m, layout->max_ctx, &s);
    room = graph_room(s, true);  // plans no session captured yet sized first (graphs.cpp)
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  *out = {};
  out->device = s.dry_bytes + s.page_spill_vram + room;
  out->host = s.dry_host_bytes + s.page_spill_host;
  out->host_mapped = s.dry_mapped_bytes;
  out->device_positions = std::min(s.device_positions, layout->max_ctx);
  out->strata_from = s.kv_method == FLYX_KV_STRATA ? s.strata_from : layout->max_ctx;
  out->copies = s.copies;
  return 0;
}

int seq_bytes(flyx_bmodel* m, int32_t max_ctx, uint64_t* out) {
  const flyx_seq_layout all{max_ctx, INT32_MAX, FLYX_KV_READ_ALL, 0, 0};
  flyx_seq_size size{};
  if (const int r = seq_size(m, &all, &size); r != 0) return r;
  *out = size.device;
  return 0;
}

int memory(flyx_bmodel* m, flyx_memory_info* out) {
  *out = {};
  out->weights = m->bytes;
  out->workspace = m->prefill_bytes + m->prefill_embeds_cap * sizeof(float) +
                   m->prefill_embed_rows_cap * sizeof(int32_t) +
                   m->prefill_pos3d_cap * sizeof(int32_t);
  size_t fr = 0, tot = 0;
  if (!gpu_memory(&fr, &tot)) return fail("cudaMemGetInfo failed");
  out->free = fr;
  out->total = tot;
  out->process = gpu_process_memory();
  return 0;
}

// Strata attention's scratch for `rows` rows of nh query heads (g_hd: query heads a KV head times the head dim, at
// most) over nkv KV heads, its selection `st.how` set.
void strata_scratch(const flyx_bmodel* m, flyx_seq* s, StrataScratch& st, int rows, int nh, int nkv, int hd,
                    int g_hd = 0) {
  if (!g_hd) g_hd = nh / nkv * hd;
  st.lse_stride = ((s->cap + 1) / 2 + 7) / 8 * 8;
  st.sel_stride = std::max(1, st.how.groups(s->cap));
  st.sel = s->alloc<int>(size_t(rows) * nkv * st.sel_stride);
  st.sel_n = s->alloc<int>(size_t(rows) * nkv);
  // the selection's chunks: about kStrataSelChunk pairs each at the session's longest context
  const int per = m->opt.kernels.strata_chunk;
  st.chunks = std::clamp((st.lse_stride + per - 1) / per, 1, kStrataSelChunks);
  if ((st.lse_stride + st.chunks - 1) / st.chunks > kStrataSelChunk)  // past 4M positions
    throw std::runtime_error("flyx-cuda: strata attention takes contexts up to " +
                             std::to_string(2 * kStrataSelChunk * kStrataSelChunks) + " positions");
  size_t lse_bytes = 0;
  if (!st.how.recent_only()) {
    const size_t lse_elems = size_t(rows) * nh * st.lse_stride;
    lse_bytes = lse_elems * sizeof(__half);
    st.lse = s->alloc<__half>(lse_elems);
    if (&st == &s->strata) s->strata_lse_bytes = lse_bytes;
    const size_t ncand = size_t(rows) * nkv * st.chunks * st.sel_stride;
    st.cand = s->alloc<int>(ncand);
    st.cand_key = s->alloc<uint32_t>(ncand);
    st.cand_n = s->alloc<int>(size_t(rows) * nkv * st.chunks);
    st.gathered = s->alloc<int>(ncand);
    st.gathered_key = s->alloc<uint32_t>(ncand);
  }
  st.qr = s->alloc<float>(size_t(rows) * nh * hd);
  const size_t delta_floats = size_t(rows) * nkv * kStrataSlices * (2 * size_t(g_hd) + 8);
  // k_strata_select_part is the last reader of st.lse; k_strata_delta only writes st.delta afterward.
  if (st.lse && lse_bytes >= delta_floats * sizeof(float))
    st.delta = reinterpret_cast<float*>(st.lse);
  else
    st.delta = s->alloc<float>(delta_floats);
}

// The state the table declares, for positions [0, s->cap).
void alloc_state(const flyx_bmodel* m, flyx_seq* s) {
  // Caches in tiers: the target's, one layer's each, of a size two or more share, since the caches
  // of a size share stagings on the device (as large as one's host part): a cache alone of its size would save no
  // device memory.  A drafter's stays on the device: its draft steps read it a row at a time.
  auto row_of = [](const KvSpec& k) { return size_t(k.nkv) * k.hd * (k.fp8 ? 1 : sizeof(kv16)); };
  auto tiered = [&](const KvSpec& k) {
    if (k.layers != 1 || k.draft) return false;
    int n = 0;
    for (const KvSpec& o : m->states.kv) n += o.layers == 1 && !o.draft && row_of(o) == row_of(k);
    return n >= 2;
  };
  // Strata attention reads those caches through a low-bit copy; the query heads each one's attention
  // blocks have, for the shapes its kernels take and the scratch.
  const bool strata = s->kv_method == FLYX_KV_STRATA;
  int nh_most = 0, nkv_most = 0, hd_most = 0, g_hd_most = 0;  // the most of each, and of query heads times head dim
  for (const Block& blk : m->blocks)
    if (const auto* a = std::get_if<AttnBlock>(&blk)) {
      const KvSpec& k = m->states.kv[size_t(a->kv)];
      if (!strata || !tiered(k)) continue;
      if (!strata_supported(a->nh, a->nkv, a->hd, s->kv_bits))
        throw std::runtime_error("strata attention: not compiled for " + std::to_string(a->nh) + " query heads of " +
                                 std::to_string(a->hd) + " over " + std::to_string(a->nkv) + " KV heads at " +
                                 std::to_string(s->kv_bits) + " bits");
      nh_most = std::max(nh_most, a->nh);
      nkv_most = std::max(nkv_most, a->nkv);
      g_hd_most = std::max(g_hd_most, a->nh / a->nkv * a->hd);
      hd_most = std::max(hd_most, a->hd);
    }
  // The stagings' size: with strata attention a prefill reads keys past them a window at a time, on
  // the tensor-core kernel (the option attention=split keeps them as large as the host parts).
  if (strata && m->opt.kernels.attention != kAttnSplit) {
    s->stage_window = m->opt.kernels.stage_window;
    s->stage_reach =
        s->strata_from > 0 ? s->strata_from + std::max(FLYX_ROUND_SLOTS * FLYX_MAX_ROWS, FLYX_MAX_ROWS + m->wide_rows)
                           : 0;
  }
  std::map<size_t, int> tiered_of_row;  // caches in tiers so far, by bytes a position: their stagings alternate
  // The copies in host memory: copy_host of them, spread evenly over the layers, so that each one's
  // staging fills while the layers since the one before run; their stagings alternate too.
  s->copies = 0;
  if (strata && nh_most > 0)
    for (const KvSpec& k : m->states.kv) s->copies += tiered(k);
  const int in_host = std::min(s->copy_host, s->copies);
  // Two host copies can stage during the draft. With enough consecutive host layers, placing the resident
  // copies further inside the pass reduces waits for the next transfer.
  int edge = 0;
  if (m->mtp_block >= 0 && m->opt.stage_dma && m->opt.kernels.stage_ahead)
    for (const auto& rule : tables::kStrataPlacement) {
      if (rule.sku != m->opt.kernels.tuning || rule.copies != s->copies || rule.host != in_host ||
          rule.bits != s->kv_bits || s->max_ctx < rule.min_capacity) continue;
      if (!std::all_of(m->states.kv.begin(), m->states.kv.end(), [&](const KvSpec& k) {
            return !tiered(k) || (k.fp8 && k.nkv == rule.nkv && k.hd == rule.hd);
          })) continue;
      edge = rule.edge_host;
      if (rule.scan_halves && m->opt.kernels.strata_mma && s->cap >= kRoundSplits * kAttnMinSplitKeys) {
        // With all splits populated the 32-key-aligned chunk grows monotonically. At shorter lengths
        // with fewer splits it is at most kAttnMinSplitKeys. This covers all rows of every captured round.
        int lo;
        attn_split_range(s->cap, kRoundSplits / 2 - 1, kRoundSplits, &lo, &s->copy_scan_end);
      }
      break;
    }
  auto hosted = [&](int i) {
    if (i < edge || i >= s->copies - edge) return true;
    const int count = s->copies - 2 * edge, hosts = in_host - 2 * edge;
    return (i - edge + 1) * hosts / count > (i - edge) * hosts / count;
  };
  auto align256 = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
  // Hybrid WDDM pinned + pageable host K/V split:
  // Priority 1 (always pinned): low-bit Strata copies (streamed every round via DMA), checkpoints, arenas, bounce/drain.
  // Priority 2 (pinned up to remaining WDDM budget): as many tiered K/V layers as fit in pinned RAM (zero overhead).
  // Priority 3 (pageable RAM overflow): remaining tiered K/V layers spill cold K/V + scales to VirtualAlloc/mmap
  // (0 bytes WDDM non-local) while keeping a rolling GPU tail ring and hot-pair cache.
  int elem_most = 1;
  for (const KvSpec& k : m->states.kv)
    if (tiered(k)) elem_most = std::max(elem_most, k.fp8 ? 1 : int(sizeof(kv16)));
  StrataSel page_how{};
  page_how.denom = m->opt.kernels.strata_denom >= 0 ? m->opt.kernels.strata_denom : s->kv_bits == 2 ? 16 : 64;
  page_how.least = m->opt.kernels.strata_least;
  page_how.recent = m->opt.kernels.strata_recent;
  page_how.most = m->opt.kernels.strata_most;
  const int page_sel_stride = std::max(1, page_how.groups(s->cap));
  const size_t page_max_misses = size_t(FLYX_MAX_ROWS) * size_t(std::max(1, nkv_most)) * size_t(page_sel_stride);
  const size_t page_bounce_est =
      page_max_misses * (4 * size_t(hd_most) * size_t(elem_most) + 4 * sizeof(float) + sizeof(PageGatherReq));
  const bool page_eligible = strata && nh_most > 0 && s->copies > 0 && s->device_positions >= 0 &&
                             s->device_positions < s->cap && m->opt.pageable_kv != PageableKvMode::off &&
                             page_bounce_est <= (64ull << 20);
  // Every cache in tiers keeps the same positions on the device, whole rows up to a boundary of the mapping's
  // granularity (tiered_device_positions): a layer's scales, its pageable buffers and its tail ring are split there
  // for K and V both, and the sizes below count on it.
  if (s->device_positions >= 0 && s->device_positions < s->cap) {
    std::vector<size_t> tier_rows;
    for (const KvSpec& k : m->states.kv)
      if (tiered(k)) tier_rows.push_back(row_of(k));
    if (!tier_rows.empty())
      s->device_positions = tiered_device_positions(s->device_positions, tier_rows.data(), int(tier_rows.size()),
                                                    vmm_granularity(), page_eligible);
  }
  auto is_pageable_k = [&](int i, int k) {
    if (k <= 0) return false;
    if (k >= s->copies) return true;
    return (i + 1) * k / s->copies > i * k / s->copies;
  };
  int n_pageable = 0;
  uint64_t total_session_pinned = 0;
  if (page_eligible) {
    if (m->opt.pageable_kv == PageableKvMode::on) {
      n_pageable = s->copies;
    } else if (s->pinned_budget != UINT64_MAX) {
      const size_t gran = vmm_granularity();
      auto round_up_g = [&](size_t n) { return (n + gran - 1) / gran * gran; };
      // Priority-1 base pinned bytes (host Strata copies + checkpoints + small staging/arena overhead; bounce & drain
      // are only added if at least one layer spills to pageable RAM).
      uint64_t prio1_base = 32ull << 20;
      for (const GdnDims& gd : m->states.gdn)
        prio1_base += size_t(FLYX_CHECKPOINTS) *
                      (align256(size_t(gd.nv) * gd.dv * gd.dk * sizeof(float)) +
                       align256(size_t(gd.ck - 1) * gd.C() * sizeof(float)));
      if (m->states.ring_width)
        prio1_base += size_t(FLYX_CHECKPOINTS) * align256(size_t(m->states.ring_width) * sizeof(float));
      int ci = 0;
      std::vector<uint64_t> layer_kv_pinned, layer_drain;
      layer_kv_pinned.reserve(size_t(s->copies));
      layer_drain.reserve(size_t(s->copies));
      uint64_t all_kv_pinned = 0;
      for (const KvSpec& k : m->states.kv) {
        if (!tiered(k)) continue;
        const int idx = ci++;
        int words = strata_words(k.hd, s->kv_bits);
        if (m->opt.strata_compact && hosted(idx) && m->opt.kernels.strata_mma && k.hd == 256 && s->kv_bits == 3)
          words = 24;
        const uint64_t q_bytes = uint64_t(strata_positions(s->cap)) * k.nkv * words * sizeof(uint32_t);
        if (hosted(idx)) {
          prio1_base += 2 * q_bytes + 2 * uint64_t(s->cap) * k.nkv * sizeof(__half2);
        } else {
          // A resident copy that reaches into the stagings' or the shared prefill slab's part of the pool keeps a
          // pinned home in host memory too (pool_finish), whether or not other copies live there.  Every one is
          // counted: the split then never pins more than the budget, at worst a layer more than needed is pageable.
          prio1_base += 2 * q_bytes;
        }
        const size_t row = row_of(k);
        const size_t tot_v = round_up_g(size_t(s->cap) * row);
        const size_t dev_v = tiered_device_bytes(s->device_positions, row, gran, true);
        const uint64_t host_kv = tot_v > dev_v ? uint64_t(tot_v - dev_v) : 0;
        const uint64_t scales_b = k.fp8 ? 2 * uint64_t(s->cap) * k.nkv * sizeof(float) : 0;
        const uint64_t kv_p = 2 * host_kv + scales_b;
        const uint64_t drain_b = 2 * align256(size_t(kRing) * size_t(m->wide_rows) * row) +
                                 (k.fp8 ? 2 * align256(size_t(kRing) * size_t(m->wide_rows) * size_t(k.nkv) * sizeof(float))
                                        : 0);
        layer_kv_pinned.push_back(kv_p);
        layer_drain.push_back(drain_b);
        all_kv_pinned += kv_p;
      }
      total_session_pinned = prio1_base + all_kv_pinned;
      if (total_session_pinned <= s->pinned_budget) {
        n_pageable = 0;  // everything fits pinned without allocating bounce/drain buffers
      } else {
        // Pick the fewest pageable layers k (under the exact is_pageable_k interleaving) that fits pinned_budget.
        n_pageable = s->copies;
        for (int k = 1; k <= s->copies; ++k) {
          uint64_t need = prio1_base + page_bounce_est;
          for (int i = 0; i < s->copies; ++i)
            need += is_pageable_k(i, k) ? layer_drain[size_t(i)] : layer_kv_pinned[size_t(i)];
          if (need <= s->pinned_budget) {
            n_pageable = k;
            break;
          }
        }
      }
    }
  }
  auto is_pageable = [&](int i) { return is_pageable_k(i, n_pageable); };
  const int min_ring_safe = std::max(32, FLYX_MAX_IN_FLIGHT * m->wide_rows);
  const int want_recent = std::min(2048, std::max(256, m->opt.kernels.strata_recent + 64));
  const int page_ring_safe = s->cap >= 1024 ? std::max(want_recent, min_ring_safe) : min_ring_safe;
  const int page_ring_tail =
      (std::max({page_ring_safe + m->wide_rows + 64, s->rows_cap + 64, FLYX_ROUND_SLOTS * FLYX_MAX_ROWS + m->wide_rows}) +
       63) /
      64 * 64;

  int copy_i = 0, copy_streamed = 0;
  // The MTP layer's cache: its draft steps attend to its last mtp_window positions, which a ring holds
  // (with the rows a pass writes at once and those past the accepted ones besides).
  const int mtp_kv = m->mtp_block >= 0 ? block<AttnBlock>(m, block<MtpBlock>(m, m->mtp_block).attn).kv : -1;
  for (const KvSpec& k : m->states.kv) {
    KvBuf b;
    int positions = s->cap;
    // past 8 windows (a whole cache of fewer is small, and its every position read exactly anyway)
    const bool mtp_part = int(s->kv.size()) == mtp_kv && m->opt.kernels.mtp_window > 0 &&
                          s->cap > 8 * m->opt.kernels.mtp_window;
    // with a copy where strata attention is compiled for the MTP layer's heads; elsewhere its first positions instead
    bool mtp_copy = mtp_part && m->opt.kernels.mtp_copy > 0;
    if (mtp_copy) {
      const auto& a = block<AttnBlock>(m, block<MtpBlock>(m, m->mtp_block).attn);
      mtp_copy = strata_supported(a.nh, a.nkv, a.hd, m->opt.kernels.mtp_copy);
    }
    if (mtp_part) {
      b.window = m->opt.kernels.mtp_window;
      b.sink = mtp_copy ? 0 : m->opt.kernels.mtp_sink;  // with a copy, the copy has the first positions
      const int ring = (b.sink + std::max(b.window + 2 * FLYX_MAX_ROWS, s->rows_cap) + 63) / 64 * 64;
      if (m->opt.kernels.mtp_ring && ring < s->cap) b.ring = positions = ring;
    }
    b.layer_stride = size_t(positions) * k.nkv * k.hd;
    const size_t n = size_t(k.layers) * b.layer_stride;
    const size_t elem = k.fp8 ? 1 : sizeof(kv16);
    if (tiered(k)) {  // in tiers when the layout says
      b.row = size_t(k.nkv) * k.hd * elem;
      const int parity = m->opt.kernels.stage_ahead ? tiered_of_row[b.row]++ % 2 : 0;
      const int scale_floats = k.fp8 ? k.nkv : 0;  // the scales go where the positions go
      const int layer_i = copy_i;
      const bool use_pageable = strata && nh_most > 0 && is_pageable(layer_i);
      const flyx_seq::Tiered tk =
          s->tiered_alloc(s->cap, b.row, 0, parity, scale_floats, use_pageable, page_ring_tail, page_eligible);
      const flyx_seq::Tiered tv =
          s->tiered_alloc(s->cap, b.row, 1, parity, scale_floats, use_pageable, page_ring_tail, page_eligible);
      b.k = tk.at;
      b.vk = tk.view;
      b.v = tv.at;
      b.vv = tv.view;
      b.device_positions = std::min(tk.device_rows, tv.device_rows);
      b.stage_rows = std::min(tk.stage_rows, tv.stage_rows);
      b.vks = tk.scales;
      b.vvs = tv.scales;
      b.slot = tk.slot;
      b.cap = s->cap;
      if (use_pageable && b.vk) {
        b.pageable = true;
        b.k_page = tk.page;
        b.v_page = tv.page;
        b.page_ring_base = tk.page_ring_base;
        b.page_ring = tk.page_ring;
        b.page_safe = page_ring_safe;
        s->page_layers.push_back(int(s->kv.size()));
      }
      if (strata && nh_most > 0) {  // the low-bit copy
        b.bits = s->kv_bits;
        b.words = strata_words(k.hd, b.bits);
        b.grid = m->opt.kernels.strata_grid;
        const int i = copy_i++;
        if (m->opt.strata_compact && b.vk && hosted(i) && m->opt.kernels.strata_mma && k.hd == 256 && b.bits == 3)
          b.words = 24;
        const size_t qn = size_t(strata_positions(s->cap)) * k.nkv * b.words;  // whole tiles (0071)
        if (b.vk) {  // with a staging: in the pool the stagings share (homes set by pool_finish)
          b.q_bytes = qn * sizeof(uint32_t);
          b.q_group = size_t(k.nkv) * 16 * b.words * sizeof(uint32_t);
          if (hosted(i)) {
            b.copy_slot = m->opt.kernels.stage_ahead ? copy_streamed++ % m->opt.strata_copy_slots : 0;
            s->copy_slots = std::max(s->copy_slots, b.copy_slot + 1);
          } else {
            b.copy_off = s->copy_bytes;
            s->copy_bytes += 2 * b.q_span();
          }
        } else {  // on the device
          b.kq = s->alloc<uint32_t>(qn);
          b.vq = s->alloc<uint32_t>(qn);
        }
        if (b.copy_slot >= 0) {  // a copy in host memory keeps its grids there, staged with its codes
          b.kqs = s->host_alloc<__half2>(size_t(s->cap) * k.nkv);
          b.vqs = s->host_alloc<__half2>(size_t(s->cap) * k.nkv);
        } else {
          b.kqs = s->alloc<__half2>(size_t(s->cap) * k.nkv);
          b.vqs = s->alloc<__half2>(size_t(s->cap) * k.nkv);
        }
        const int hot = m->opt.kernels.strata_hot;
        if ((k.fp8 || b.pageable) && hot > 0 && b.device_positions < s->cap) {  // the pairs re-read lately
          while ((1 << b.hot_bits) < hot) ++b.hot_bits;
          b.hot_tags = size_t(k.nkv) << b.hot_bits;
          b.hot_tag = s->alloc<int32_t>(b.hot_tags);
          b.hot_claim = s->alloc<int32_t>(b.hot_tags);
          b.hot = s->alloc<uint8_t>(b.hot_tags * 4 * size_t(k.hd) * elem);
          if (k.fp8) b.hot_scale = s->alloc<float>(b.hot_tags * 4);  // their scales
        }
      }
    } else {
      b.k = s->alloc<uint8_t>(n * elem);
      b.v = s->alloc<uint8_t>(n * elem);
    }
    b.nkv = k.nkv;
    if (k.fp8 && b.pageable) {  // GPU scales for [0, device_positions) + tail ring; [device_positions, cap) in pageable RAM
      b.ks = s->alloc<float>(size_t(b.page_ring) * k.nkv);
      b.vs = s->alloc<float>(size_t(b.page_ring) * k.nkv);
      const size_t host_scales = size_t(s->cap - b.device_positions) * k.nkv * sizeof(float);
      b.ks_page = static_cast<float*>(s->pageable_alloc_raw(host_scales));
      b.vs_page = static_cast<float*>(s->pageable_alloc_raw(host_scales));
    } else if (k.fp8 && b.vk) {  // in tiers: the device part's scales on the device, every position's in host memory
      const size_t dev = size_t(std::max(1, b.device_positions)) * k.nkv;
      b.ks = s->alloc<float>(dev);
      b.vs = s->alloc<float>(dev);
      b.ks_host = s->host_alloc<float>(size_t(s->cap) * k.nkv);
      b.vs_host = s->host_alloc<float>(size_t(s->cap) * k.nkv);
    } else if (k.fp8) {
      b.ks = s->alloc<float>(size_t(k.layers) * positions * k.nkv);
      b.vs = s->alloc<float>(size_t(k.layers) * positions * k.nkv);
    }
    if (mtp_copy) {  // every position's copy (strata's layout), read by strata attention
      const auto& a = block<AttnBlock>(m, block<MtpBlock>(m, m->mtp_block).attn);
      b.bits = m->opt.kernels.mtp_copy;
      b.words = strata_words(k.hd, b.bits);
      b.grid = m->opt.kernels.strata_grid;
      const size_t qn = size_t(strata_positions(s->cap)) * k.nkv * b.words;
      b.kq = s->alloc<uint32_t>(qn);
      b.vq = s->alloc<uint32_t>(qn);
      b.kqs = s->alloc<__half2>(size_t(s->cap) * k.nkv);
      b.vqs = s->alloc<__half2>(size_t(s->cap) * k.nkv);
      s->P_window = s->alloc<DevParams>(1);
      StrataScratch& st = s->mtp_strata;  // the last `window` positions' pairs, and none other (the ring has no others)
      st.how.denom = 1 << 20;
      st.how.least = 0;
      st.how.recent = b.window;
      st.how.most = b.window / 2;
      strata_scratch(m, s, st, 1, a.nh, a.nkv, a.hd);
    } else if (b.window > 0) {  // the draft step's gathered positions
      const size_t w = size_t(m->opt.kernels.mtp_sink + b.window) * k.nkv;
      s->mtp_window = KvRef{s->alloc<uint8_t>(w * k.hd * elem), s->alloc<uint8_t>(w * k.hd * elem),
                            k.fp8 ? s->alloc<float>(w) : nullptr, k.fp8 ? s->alloc<float>(w) : nullptr};
      s->P_window = s->alloc<DevParams>(1);
    }
    s->kv.push_back(b);
  }
  s->pool_finish();  // the stagings and the copy's device home
  s->forget_hot();
  s->create_stage_stream();  // the copies to those caches' stagings
  if (strata && nh_most > 0) {  // the scratch of a round's strata attention, FLYX_MAX_ROWS rows
    StrataScratch& st = s->strata;
    st.how.denom = m->opt.kernels.strata_denom >= 0 ? m->opt.kernels.strata_denom : s->kv_bits == 2 ? 16 : 64;
    st.how.least = m->opt.kernels.strata_least;
    st.how.recent = m->opt.kernels.strata_recent;
    st.how.most = m->opt.kernels.strata_most;
    strata_scratch(m, s, st, FLYX_MAX_ROWS, nh_most, nkv_most, hd_most, g_hd_most);
  }
  if (!s->page_layers.empty()) {
    const size_t max_misses = size_t(FLYX_MAX_ROWS) * size_t(nkv_most) * size_t(s->strata.sel_stride);
    int32_t* miss_map = s->alloc<int32_t>(max_misses);
    s->page_drain_dev = s->alloc<PageDrainLayer>(s->page_layers.size());
    size_t off = 0;
    const size_t req_off = off;
    off += align256(max_misses * sizeof(PageGatherReq));
    const size_t cnt_off = off;
    off += 256;
    const size_t bkv_off = off;
    off += align256(max_misses * 4 * size_t(hd_most) * size_t(elem_most));
    const size_t bsc_off = off;
    off += align256(max_misses * 4 * sizeof(float));
    struct DrainOffs {
      size_t dk, dv, dks, dvs;
    };
    std::vector<DrainOffs> doffs(s->page_layers.size());
    for (size_t li = 0; li < s->page_layers.size(); ++li) {
      const KvBuf& b = s->kv[size_t(s->page_layers[li])];
      const size_t kv_sz = align256(size_t(kRing) * size_t(m->wide_rows) * b.row);
      doffs[li].dk = off;
      off += kv_sz;
      doffs[li].dv = off;
      off += kv_sz;
      if (b.ks_page) {
        const size_t sc_sz = align256(size_t(kRing) * size_t(m->wide_rows) * size_t(b.nkv) * sizeof(float));
        doffs[li].dks = off;
        off += sc_sz;
        doffs[li].dvs = off;
        off += sc_sz;
      } else {
        doffs[li].dks = doffs[li].dvs = 0;
      }
    }
    if (s->dry) {
      s->dry_host_bytes += off;
    } else {
      CK(cudaHostAlloc(&s->page_pinned_host, off, cudaHostAllocMapped));
      std::memset(s->page_pinned_host, 0, off);
      CK(cudaHostGetDevicePointer(&s->page_pinned_dev, s->page_pinned_host, 0));
      auto* hbase = static_cast<uint8_t*>(s->page_pinned_host);
      auto* dbase = static_cast<uint8_t*>(s->page_pinned_dev);
      auto* req_h = reinterpret_cast<PageGatherReq*>(hbase + req_off);
      auto* req_d = reinterpret_cast<PageGatherReq*>(dbase + req_off);
      auto* cnt_h = reinterpret_cast<int32_t*>(hbase + cnt_off);
      auto* cnt_d = reinterpret_cast<int32_t*>(dbase + cnt_off);
      uint8_t* bkv_h = hbase + bkv_off;
      uint8_t* bkv_d = dbase + bkv_off;
      auto* bsc_h = reinterpret_cast<float*>(hbase + bsc_off);
      auto* bsc_d = reinterpret_cast<float*>(dbase + bsc_off);
      s->page_drain_host.resize(s->page_layers.size());
      for (size_t li = 0; li < s->page_layers.size(); ++li) {
        const int idx = s->page_layers[li];
        KvBuf& b = s->kv[size_t(idx)];
        const KvSpec& ks = m->states.kv[size_t(idx)];
        PageGatherCtx ctx{};
        ctx.k_page = b.k_page;
        ctx.v_page = b.v_page;
        ctx.ks_page = b.ks_page;
        ctx.vs_page = b.vs_page;
        ctx.req_host = req_h;
        ctx.req_dev = req_d;
        ctx.miss_count_host = cnt_h;
        ctx.miss_count_dev = cnt_d;
        ctx.miss_map = miss_map;
        ctx.bounce_kv_host = bkv_h;
        ctx.bounce_kv_dev = bkv_d;
        ctx.bounce_scale_host = bsc_h;
        ctx.bounce_scale_dev = bsc_d;
        ctx.nkv = ks.nkv;
        ctx.hd = ks.hd;
        ctx.elem = ks.fp8 ? 1 : int(sizeof(kv16));
        ctx.max_misses = int(max_misses);
        ctx.cap = s->cap;
        ctx.device_positions = b.device_positions;
        s->page_ctxs.push_back(ctx);
        b.page_ctx = &s->page_ctxs.back();

        PageDrainLayer dl{};
        dl.k = static_cast<const uint8_t*>(b.k);
        dl.v = static_cast<const uint8_t*>(b.v);
        dl.ks = b.ks;
        dl.vs = b.vs;
        dl.drain_k = dbase + doffs[li].dk;
        dl.drain_v = dbase + doffs[li].dv;
        dl.drain_k_host = hbase + doffs[li].dk;
        dl.drain_v_host = hbase + doffs[li].dv;
        if (b.ks_page) {
          dl.drain_ks = reinterpret_cast<float*>(dbase + doffs[li].dks);
          dl.drain_vs = reinterpret_cast<float*>(dbase + doffs[li].dvs);
          dl.drain_ks_host = reinterpret_cast<float*>(hbase + doffs[li].dks);
          dl.drain_vs_host = reinterpret_cast<float*>(hbase + doffs[li].dvs);
        }
        dl.row_bytes = b.row;
        dl.nkv = ks.nkv;
        dl.device_positions = b.device_positions;
        dl.ring_base = b.page_ring_base;
        dl.ring = b.page_ring;
        s->page_drain_host[li] = dl;
      }
      CK(cudaMemcpyAsync(s->page_drain_dev, s->page_drain_host.data(),
                         s->page_drain_host.size() * sizeof(PageDrainLayer), cudaMemcpyHostToDevice, s->st));
      CK(cudaStreamSynchronize(s->st));
      if (m->verbose)
        std::fprintf(stderr, "flyx-cuda: K/V in host memory: %d of %d layers pinned, %d in pageable RAM\n",
                     s->copies - int(s->page_layers.size()), s->copies, int(s->page_layers.size()));
    }
  }
  // Windows automatic mode: seq_new_in() re-queries the live WDDM non-local budget, so concurrent sessions or other
  // processes using shared GPU memory may spill more (or fewer) layers than seq_size() saw at planning time.
  // Unconditionally reserve the worst-case VRAM, host memory, and graph room across all possible spill counts
  // 0..s->copies so real allocations never exceed what seq_size() reported to core/plan/budget.cpp (AGENTS.md).
  // Set page_spill_auto on both dry and live sessions so graph_room(live) matches the room leased by seq_size().
  s->page_spill_auto = page_eligible && m->opt.pageable_kv == PageableKvMode::automatic && s->pinned_budget != UINT64_MAX;
  if (s->dry && s->page_spill_auto) {
    const size_t gran = vmm_granularity();
    auto round_up_g = [&](size_t n) { return (n + gran - 1) / gran * gran; };
    size_t extra_vram = 0;
    size_t cur_layer_host = 0, sum_pinned_host = 0, sum_max_host = 0, cur_drain_host = 0;
    const int hot = m->opt.kernels.strata_hot;
    const size_t max_misses = size_t(FLYX_MAX_ROWS) * size_t(nkv_most) * size_t(s->strata.sel_stride);
    const size_t bounce_host = align256(max_misses * sizeof(PageGatherReq)) + 256 +
                               align256(max_misses * 4 * size_t(hd_most) * size_t(elem_most)) +
                               align256(max_misses * 4 * sizeof(float));
    for (size_t i = 0; i < s->kv.size(); ++i) {
      const KvBuf& b = s->kv[i];
      const KvSpec& k = m->states.kv[i];
      if (!b.vk || !tiered(k)) continue;
      const size_t tot_v = round_up_g(size_t(s->cap) * b.row);
      const size_t dev_v = round_up_g(size_t(b.device_positions) * b.row);
      const int ring_pad = (dev_v > 0 && (dev_v % b.row) != 0) ? 1 : 0;
      const size_t pinned_h = 2 * (tot_v - dev_v) +
                              (k.fp8 ? 2 * align256(size_t(s->cap) * size_t(k.nkv) * sizeof(float)) : 0);
      const size_t paged_h = 2 * size_t(s->cap - b.device_positions) * b.row +
                             (k.fp8 ? 2 * size_t(s->cap - b.device_positions) * size_t(k.nkv) * sizeof(float) : 0);
      const size_t drain_h = 2 * align256(size_t(kRing) * size_t(m->wide_rows) * b.row) +
                             (k.fp8 ? 2 * align256(size_t(kRing) * size_t(m->wide_rows) * size_t(k.nkv) * sizeof(float))
                                    : 0);
      cur_layer_host += b.pageable ? paged_h : pinned_h;
      if (b.pageable) cur_drain_host += drain_h;
      sum_pinned_host += pinned_h;
      sum_max_host += std::max(pinned_h, paged_h + drain_h);
      if (b.pageable) continue;
      const size_t ring_bytes = size_t(page_ring_tail) * b.row;
      const size_t ring_vmm_bytes = size_t(page_ring_tail + ring_pad) * b.row;
      extra_vram += 2 * (dev_v == 0 ? align256(ring_bytes) : round_up_g(ring_vmm_bytes));
      if (k.fp8) {
        extra_vram += 2 * align256(size_t(page_ring_tail + ring_pad) * size_t(k.nkv) * sizeof(float));
      } else if (hot > 0) {
        int hb = 0;
        while ((1 << hb) < hot) ++hb;
        const size_t tags = size_t(k.nkv) << hb;
        extra_vram += 2 * align256(tags * sizeof(int32_t)) + align256(tags * 4 * size_t(k.hd) * sizeof(kv16));
      }
    }
    if (s->page_layers.empty()) extra_vram += align256(max_misses * sizeof(int32_t));
    const size_t cur_drain_dev =
        s->page_layers.empty() ? 0 : align256(s->page_layers.size() * sizeof(PageDrainLayer));
    const size_t all_drain_dev = align256(size_t(s->copies) * sizeof(PageDrainLayer));
    if (all_drain_dev > cur_drain_dev) extra_vram += all_drain_dev - cur_drain_dev;
    const size_t cur_host = cur_layer_host + (s->page_layers.empty() ? 0 : bounce_host + cur_drain_host);
    const size_t worst_host = std::max(sum_pinned_host, bounce_host + sum_max_host);
    s->page_spill_vram = extra_vram;
    s->page_spill_host = worst_host > cur_host ? worst_host - cur_host : 0;
  }
  for (const GdnDims& gd : m->states.gdn) {
    GdnBuf b;
    b.s_stride = size_t(gd.nv) * gd.dv * gd.dk;
    b.w_stride = size_t(gd.ck - 1) * gd.C();
    b.S = s->alloc<float>(FLYX_ROUND_SLOTS * b.s_stride);
    b.win = s->alloc<float>(FLYX_ROUND_SLOTS * b.w_stride);
    b.replay = GdnBufs::round(m->wide_rows, gd);  // a round's rows: 64 where lookup rounds may be wide
    b.replay.base = s->alloc<float>(kGdnReplaySlots * b.replay.slot_stride);
    // Checkpoint host buffers are allocated lazily per slot in checkpoint.
    s->ckp_slot_bytes += align256(b.s_stride * sizeof(float)) + align256(b.w_stride * sizeof(float));
    s->gdn.push_back(b);
  }
  // The commit's table of layers (layers.cpp: commit): when every gated delta net has the first's dims, strides and
  // rows, and the lanes kernels' alignment.
  bool alike = !s->gdn.empty();
  for (const GdnBuf& g : s->gdn) {
    const GdnBuf& f = s->gdn[0];
    alike = alike && g.s_stride == f.s_stride && g.w_stride == f.w_stride &&
            g.replay.slot_stride == f.replay.slot_stride && g.replay.rows_cap == f.replay.rows_cap &&
            std::memcmp(&g.replay.d, &f.replay.d, sizeof(GdnDims)) == 0 && g.s_stride % 4 == 0 &&
            (s->dry || reinterpret_cast<uintptr_t>(g.S) % 16 == 0);
  }
  if (alike) {
    s->gdn_layers = s->alloc<GdnLayer>(s->gdn.size());
    if (!s->dry) {
      // On the session's stream, after alloc's zeroing there (a non-blocking stream: a plain copy could land first).
      std::vector<GdnLayer> host;
      for (const GdnBuf& g : s->gdn) host.push_back({g.S, g.win, g.replay.base});
      CK(cudaMemcpyAsync(s->gdn_layers, host.data(), host.size() * sizeof(GdnLayer), cudaMemcpyHostToDevice, s->st));
      CK(cudaStreamSynchronize(s->st));
    }
  }
  if (const int w = m->states.ring_width) {
    s->ring = s->alloc<float>(size_t(kHring) * w);
    s->ring_snap = s->alloc<float>(size_t(w));
    s->ckp_slot_bytes += align256(size_t(w) * sizeof(float));
  }
  if (s->dry) s->dry_host_bytes += size_t(FLYX_CHECKPOINTS) * s->ckp_slot_bytes;
}

// DV's canvas, what its pass leaves, the round's selector table and the canvas pass's rows (flyx_seq::DvRows), for
// a canvas of v.slots slots: in s->alloc's memory, so seq_bytes counts them.
void alloc_dv(const flyx_bmodel* m, const DvBlock& v, flyx_seq* s) {
  const RowNeeds& n = m->need;
  const size_t N = size_t(v.slots), H = size_t(v.hidden), R = size_t(v.sel_rank), K = size_t(kDvTop);
  flyx_seq::DvRows& d = s->dv;
  DvCanvas& c = d.canvas;
  c.cv = s->alloc<int>(N);
  c.z = s->alloc<float>(N * H);
  c.cand = s->alloc<int>(N * K);
  c.logp = s->alloc<float>(N * K);
  c.cov = s->alloc<float>(N);
  c.live = s->alloc<int>(N);
  c.hid = s->alloc<float>(N * R);
  c.next_start = s->alloc<int>(1);
  DvStats& st = d.stats;
  st.ids = s->alloc<int>(N * K);
  st.p = s->alloc<float>(N * K);
  st.lse = s->alloc<float>(N);
  st.mass = s->alloc<float>(N);
  st.zn = s->alloc<float>(N * H);
  st.hid = s->alloc<float>(N * R);
  st.part_i = s->alloc<int>(N * kDvParts * K);
  st.part_v = s->alloc<float>(N * kDvParts * K);
  d.table = s->alloc<float>(std::max<size_t>(N - 1, 1) * K * K);  // a tree reaches slots - 1 slots at most
  // The pass's rows: the target's attention and MLP blocks' widths (RowNeeds), each linear's adapter product (q|k|v's
  // three side by side the widest), the residual context's [u | g], the head's logits, the attention's partials.
  int rank = 0, nh = 0, hd = 0;
  for (int j = 0; j < v.n_layers; ++j) {
    const DvLayer& l = block<DvLayer>(m, v.first_layer + j);
    const AttnBlock& a = block<AttnBlock>(m, l.attn);
    rank = std::max(rank, l.rank);
    nh = std::max(nh, a.nh);
    hd = std::max(hd, a.hd);
  }
  d.ld_lora = 3 * rank;
  d.splits = kAttnCoresSplits;  // the float32 kernels' most: a canvas row takes no others (attention_split)
  d.x = s->alloc<float>(N * H);
  d.h = s->alloc<float>(N * H);
  d.proj = s->alloc<float>(N * size_t(n.proj));
  d.act = s->alloc<float>(N * size_t(n.act));
  d.q = s->alloc<float>(N * size_t(n.q));
  d.lora = s->alloc<float>(N * size_t(d.ld_lora));
  d.lora16 = s->alloc<bf16>(N * size_t(d.ld_lora));
  d.ug = s->alloc<float>(N * 2 * H);
  d.logits = s->alloc<float>(N * size_t(v.vocab));
  d.part = s->alloc<float>(attention_partial_floats(v.slots, nh, hd, d.splits));
  // The GEMM inputs of `slots` rows (prefill's kernels past 16 rows); K up to the adapters' rank too.
  const int k_max = std::max(n.k_max, (rank + 63) / 64 * 64);
  const size_t bytes = scratch_bytes(v.slots, k_max, n.q8, false);
  d.scratch = scratch_carve(s->alloc<unsigned char>(bytes), bytes, v.slots, k_max, n.q8, false);
}

void build_seq(flyx_bmodel* m, int32_t max_ctx, flyx_seq* s) {
  const RowNeeds& n = m->need;
  // row buffers for rounds (m->wide_rows: up to FLYX_MAX_WIDE_ROWS for a wide lookup round); prefill uses the model's
  const int H = m->hidden, V = m->vocab, R = m->wide_rows;
  const int prefill_rows = std::max(m->prefill_rows > 0 ? m->prefill_rows : 256, FLYX_MAX_ROWS);
  const DFlashBlock* d = m->dflash_block >= 0 ? &block<DFlashBlock>(m, m->dflash_block) : nullptr;
  const DvBlock* dv = m->dv_block >= 0 ? &block<DvBlock>(m, m->dv_block) : nullptr;
  s->m = m;
  s->max_ctx = max_ctx;
  s->cap = max_ctx + std::max(2 * FLYX_MAX_ROWS, m->wide_rows);
  if (dv) {  // DV's canvas K/V: scratch slots past every position a round writes
    s->dv.s0 = s->cap;
    s->cap += dv->slots;
  }
  s->rows_cap = prefill_rows;
  s->profile = m->opt.profile;
  s->graphs = m->opt.graphs;
  if (!s->dry) {
    CK(cudaStreamCreateWithFlags(&s->st, cudaStreamNonBlocking));
    create_side(&s->side);
    if (dv) CK(cudaStreamCreateWithFlags(&s->dv.body, cudaStreamNonBlocking));
    s->dv.skip_noop = m->opt.dv_conditional;
    for (auto& e : s->ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&s->idle_ev, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&s->ready_ev, cudaEventDisableTiming));
    if (s->arena_dev_bytes > 0) {
      void* p = nullptr;
      CK(cudaMalloc(&p, s->arena_dev_bytes));
      CK(cudaMemsetAsync(p, 0, s->arena_dev_bytes, s->st));
      s->allocs.push_back(p);
      s->dev_arena = static_cast<unsigned char*>(p);
      s->dev_arena_cap = s->arena_dev_bytes;
    }
    if (s->arena_host_bytes > 0) {
      void* p = nullptr;
      CK(cudaHostAlloc(&p, s->arena_host_bytes, cudaHostAllocDefault));
      std::memset(p, 0, s->arena_host_bytes);
      s->host_allocs.push_back(p);
      s->host_arena = static_cast<unsigned char*>(p);
      s->host_arena_cap = s->arena_host_bytes;
    }
  }
  s->params_host = s->pinned_alloc<DevParams>(1);
  s->gen_host = s->pinned_alloc<DevGen>(1);
  s->mask_host = s->pinned_alloc<uint32_t>(size_t((m->vocab + 31) / 32));
  s->drafts_host = s->pinned_alloc<int>(size_t(flyx_seq::kDraftSlots) * (FLYX_MAX_WIDE_ROWS + 1));
  s->tree_host = s->pinned_alloc<TreeRows>(size_t(flyx_seq::kDraftSlots));
  s->stop_host = s->pinned_alloc<flyx_stop_policy>(size_t(flyx_seq::kDraftSlots));
  s->ring_host = s->pinned_alloc<flyx_round_result>(size_t(kRing));
  s->tok_host = s->pinned_alloc<int>(size_t(prefill_rows));
  s->P = s->alloc<DevParams>(1);
  s->G = s->alloc<DevGen>(1);
  s->mask_words = (m->vocab + 31) / 32;
  s->mask = s->alloc<uint32_t>(size_t(s->mask_words));
  s->reply.count = s->alloc<int>(size_t(V));
  s->reply.seen = s->alloc<int>(size_t(std::min(s->cap, V)));
  s->reply.n_seen = s->alloc<int>(1);
  s->ring_dev = s->alloc<flyx_round_result>(kRing);
  alloc_state(m, s);

  s->ld_proj = n.proj;
  s->ld_act = n.act;
  auto seq_bytes_alloc = [&](size_t bytes) { return static_cast<void*>(s->alloc<unsigned char>(bytes)); };
  alloc_rows(m, R, false, &s->own, seq_bytes_alloc);
  s->drafts = s->alloc<int>(size_t(m->wide_rows));
  if (m->wide_rows > FLYX_MAX_ROWS) s->P_shift = s->alloc<DevParams>(1);
  const int head_rows = std::max(FLYX_MAX_ROWS, d ? d->block : 1);
  const int logit_cols = std::max(V, (d || m->mtp_block >= 0) ? kDraftPickFloats : 0);
  const int chunks = (V + 4095) / 4096;
  s->argv = s->alloc<float>(size_t(head_rows) * chunks);
  s->argi = s->alloc<int>(size_t(head_rows) * chunks);
  s->top_rows = head_rows;
  s->topv = s->alloc<float>(size_t(head_rows) * kTopkParts * 16);
  s->topi = s->alloc<int>(size_t(head_rows) * kTopkParts * 16);
  // Target strata.lse is only live inside attention_strata(); head logits and sampling sort buffers (cz, ci) are only
  // live before (drafting) or after (head / choose_rows) all target layers, so carve them from strata.lse when it fits.
  auto align256 = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
  const size_t logits_b = align256(size_t(head_rows) * logit_cols * sizeof(float));
  const size_t cz_b = align256(size_t(head_rows) * V * sizeof(float));
  const size_t ci_b = align256(size_t(head_rows) * V * sizeof(int));
  unsigned char* lse_base = reinterpret_cast<unsigned char*>(s->strata.lse);
  if (lse_base && s->strata_lse_bytes >= logits_b + cz_b + ci_b) {
    s->logits = reinterpret_cast<float*>(lse_base);
    s->cz = reinterpret_cast<float*>(lse_base + logits_b);
    s->ci = reinterpret_cast<int*>(lse_base + logits_b + cz_b);
  } else if (lse_base && s->strata_lse_bytes >= cz_b + ci_b) {
    s->logits = s->alloc<float>(size_t(head_rows) * logit_cols);
    s->cz = reinterpret_cast<float*>(lse_base);
    s->ci = reinterpret_cast<int*>(lse_base + cz_b);
  } else if (lse_base && s->strata_lse_bytes >= logits_b) {
    s->logits = reinterpret_cast<float*>(lse_base);
    s->cz = s->alloc<float>(size_t(head_rows) * V);
    s->ci = s->alloc<int>(size_t(head_rows) * V);
  } else {
    s->logits = s->alloc<float>(size_t(head_rows) * logit_cols);
    s->cz = s->alloc<float>(size_t(head_rows) * V);
    s->ci = s->alloc<int>(size_t(head_rows) * V);
  }
  if (d) {
    const int B = std::max(d->block, FLYX_MAX_ROWS), DH = d->hidden;  // a wide block's rows
    s->ld_feats = n.feats;
    s->dtok = s->alloc<int>(B);
    s->cand = s->alloc<int>(size_t(B) * d->sel_topk);
    s->unary = s->alloc<float>(size_t(B) * d->sel_topk);
    s->hid = s->alloc<float>(size_t(B) * d->sel_rank);
    s->dx = s->alloc<float>(size_t(B) * DH);
    s->dh = s->alloc<float>(size_t(B) * DH);
    s->dhc = s->alloc<float>(size_t(B) * DH);
    s->dtmp = s->alloc<float>(size_t(B) * DH);
    s->ld_dproj = std::max((d->nh + 2 * d->nkv) * d->hd, 2 * d->inter);
    s->ld_dact = std::max(d->nh * d->hd, d->inter);
    s->dproj = s->alloc<float>(size_t(B) * s->ld_dproj);
    s->dact = s->alloc<float>(size_t(B) * s->ld_dact);
    s->ddelta = s->alloc<float>(size_t(B) * 2 * d->conv_cols());
    s->dq = s->alloc<float>(size_t(B) * d->nh * d->hd);
    s->ld_ckv = n.ckv;
    if (const DSpineBridge& br = d->dspine) {
      const size_t n_rows = size_t(d->block), c = size_t(br.topk), C = size_t(br.code), Rk = size_t(br.rank);
      const size_t cands = (n_rows - 1) * c, senders = 1 + (n_rows - 2) * c;
      auto& ds = s->ds;
      ds.codes = s->alloc<float>(std::max(n_rows, senders) * C);
      ds.gates = s->alloc<float>(std::max(n_rows * 3 * Rk, senders * 2 * Rk));
      ds.msg = s->alloc<float>(n_rows * Rk);
      ds.rms = s->alloc<float>(n_rows);
      ds.before = s->alloc<float>(n_rows * size_t(DH));
      ds.g = s->alloc<float>(n_rows * Rk);
      ds.heads = s->alloc<float>(cands * size_t(DH));
      ds.tg = s->alloc<float>(cands * Rk);
      ds.base = s->alloc<float>(cands);
      ds.pick = s->alloc<int>(senders);
    }
  }
  alloc_cert_scratch(m, &s->cert, seq_bytes_alloc);
  // Drafts by position: the test oracle's and the given drafts of a round
  // (gen.drafts), which a model with no drafter takes too.
  s->oracle_cap = s->cap;
  s->oracle = s->alloc<int>(s->cap);
  if (m->mtp_block >= 0) {
    s->mg = s->alloc<float>(H);
    s->mtok = s->alloc<int>(FLYX_MAX_ROWS + 1);
  }
  // Tree rounds and relaxed acceptance (runner.cpp), last: the buffers above keep the addresses they have without them.
  s->tree = s->alloc<TreeRows>(1);
  s->tree_kv = s->alloc<KvRows>(std::max<size_t>(m->states.kv.size(), 1));
  s->accept_stats = s->alloc<float>(3 * size_t(R));  // a wide round's every row (its head 16 rows at a time)
  if (R > FLYX_MAX_ROWS) s->child_z = s->alloc<float>(FLYX_MAX_WIDE_ROWS);
  s->draft_prob = s->alloc<float>(FLYX_MAX_ROWS);
  s->draft_margin = s->alloc<float>(FLYX_MAX_ROWS);
  s->stop_dev = s->alloc<flyx_stop_policy>(1);
  s->stop_state = s->alloc<DevStopState>(1);
  if (dv) alloc_dv(m, *dv, s);
  if (!s->dry) set_params(s, 0, 0, 0);  // the device arena is already zeroed on s->st; DV's canvas: none to follow
}

void seq_free(flyx_seq* s) {
  // -v: what its rounds ran from (a run's rounds without a graph: the sum of these lines).
  if (s && s->m && s->m->verbose && s->m->opt.graphs && (s->graph_stats.from_graph || s->graph_stats.without_graph))
    std::fprintf(stderr, "flyx-cuda: a session's rounds: %llu from a graph, %llu without; %llu graphs made, %.2f MiB "
                         "held of its room's %.2f\n", (unsigned long long)s->graph_stats.from_graph,
                 (unsigned long long)s->graph_stats.without_graph, (unsigned long long)s->graph_stats.captured,
                 double(s->graph_held) / 1048576.0, graph_room(*s) / 1048576.0);
  // The graphs of batch rounds it ran in hold its buffers: they go first, once the batch's stream is done.
  if (s && s->m && s->m->batch) {
    BatchRounds& B = *s->m->batch;
    std::lock_guard<std::mutex> lock(B.mu);
    bool synced = false;
    for (auto it = B.graphs.begin(); it != B.graphs.end();) {
      bool uses = false;
      for (int i = 0; i < it->first.n; ++i)
        if (it->first.seqs[i].seq == uintptr_t(s)) {
          uses = true;
          break;
        }
      it = uses ? drop_batch_graph(B, it, &synced, true) : std::next(it);  // the others' shares back
    }
    for (auto it = B.hits.begin(); it != B.hits.end();) {
      bool uses = false;
      for (int i = 0; i < it->first.n; ++i)
        if (it->first.seqs[i].seq == uintptr_t(s)) {
          uses = true;
          break;
        }
      it = uses ? B.hits.erase(it) : std::next(it);
    }
  }
  delete s;
}

void alloc_batch(flyx_bmodel* m) {
  auto B = std::make_unique<BatchRounds>();
  auto alloc = [&](size_t bytes) {
    unsigned char* p = m->alloc<unsigned char>(bytes);
    CK(cudaMemset(p, 0, bytes));
    return static_cast<void*>(p);
  };
  const DFlashBlock* d = nullptr;
  for (const Block& blk : m->blocks)
    if (const auto* df = std::get_if<DFlashBlock>(&blk)) {
      d = df;
      break;
    }
  if (d && !d->dspine && d->block > 0 && d->block * 2 <= FLYX_MAX_ROWS) {  // DSpine drafts a sequence at a time
    const int DH = d->hidden;
    auto af = [&](size_t count) { return static_cast<float*>(alloc(count * sizeof(float))); };
    B->dflash.dtok = static_cast<int*>(alloc(size_t(FLYX_MAX_ROWS) * sizeof(int)));
    B->dflash.dx = af(size_t(FLYX_MAX_ROWS) * DH);
    B->dflash.dh = af(size_t(FLYX_MAX_ROWS) * DH);
    B->dflash.dhc = af(size_t(FLYX_MAX_ROWS) * DH);
    B->dflash.dtmp = af(size_t(FLYX_MAX_ROWS) * DH);
    B->dflash.ld_dproj = std::max((d->nh + 2 * d->nkv) * d->hd, 2 * d->inter);
    B->dflash.ld_dact = std::max(d->nh * d->hd, d->inter);
    B->dflash.dproj = af(size_t(FLYX_MAX_ROWS) * B->dflash.ld_dproj);
    B->dflash.dact = af(size_t(FLYX_MAX_ROWS) * B->dflash.ld_dact);
    B->dflash.ddelta = af(size_t(FLYX_MAX_ROWS) * 2 * d->conv_cols());
    B->dflash.dq = af(size_t(FLYX_MAX_ROWS) * d->nh * d->hd);
    B->dflash.cand = static_cast<int*>(alloc(size_t(FLYX_MAX_ROWS) * d->sel_topk * sizeof(int)));
    B->dflash.unary = af(size_t(FLYX_MAX_ROWS) * d->sel_topk);
    B->dflash.hid = af(size_t(FLYX_MAX_ROWS) * d->sel_rank);
  }
  CK(cudaStreamCreateWithFlags(&B->st, cudaStreamNonBlocking));
  create_side(&B->side);
  CK(cudaEventCreateWithFlags(&B->done, cudaEventDisableTiming));
  m->batch = std::move(B);
}

}  // namespace flyx::cuda
