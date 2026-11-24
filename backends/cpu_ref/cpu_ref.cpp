// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
//
// cpu_ref: the float64 reference backend.
//
// Implements the full backend contract (prefill, rounds, accept, commit,
// rewind, checkpoints) for the blocks of backend_abi.h with the ops in
// ops/ops.h and the same state layout ideas as the device backends: a ring
// of FLYX_ROUND_SLOTS recurrent states with the replay inputs of each
// round, so rollback inside the window is a fold.  Slow by design; used for
// tests and as the ground truth for kernels.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backends/common/snapshot_format.h"
#include "flyx/backend_abi.h"
#include "ops/ops.h"

namespace {

thread_local std::string g_error;

int fail(const std::string& msg) {
  g_error = msg;
  return -1;
}

using Vec = std::vector<double>;

struct RefModel {
  int hidden = 0, vocab = 0;
  std::vector<flyx_block> blocks;
  std::vector<flyx_state> states;
  // Per weight; every floating-point one (bf16, f16, f32, NVFP4, GGML blocks) decoded to float64 at open, so the
  // ops read doubles instead of decoding each element again in every call.
  std::vector<const flyx_tensor_view*> W;
  std::vector<std::unique_ptr<std::vector<double>>> owned;
  std::vector<std::unique_ptr<flyx_tensor_view>> owned_views;
  std::unordered_set<const flyx_tensor_view*> a16;   // inputs rounded to bf16 (NVFP4 weights)
  std::unordered_set<const flyx_tensor_view*> a16h;  // ... and held as binary16 over a row's scale (GGML blocks)
  std::unordered_set<const flyx_tensor_view*> q8;    // ... or, in every pass, as 8-bit integers with a scale per
                                                         // 32 values (FLYX_ACT_Q8)
  std::unordered_set<const flyx_tensor_view*> a4;    // NVFP4 weights whose inputs are NVFP4 in those passes
  mutable std::unordered_map<int, Vec> cache;
  int dflash = -1;  // the block drafter's block (DFlash2 or DSpine), -1: none
  int ring_width = 0;
  // Every residual add of the stack a weight with NVFP4 inputs and the embedding a bf16 table: the engine keeps a
  // prefill pass's residual in bf16 (as vLLM, SGLang and basalt do), each sum in float32 then rounded.
  bool res16 = false;
  // Option profile=1: milliseconds by kind of node, printed when the model closes.
  bool profile = false;
  mutable std::unordered_map<std::string, double> profile_ms;
  mutable std::mutex mu;  // cache and profile_ms: sessions may run on several threads

  ~RefModel() {
    if (!profile || profile_ms.empty()) return;
    std::vector<std::pair<double, std::string>> by;
    double total = 0;
    for (const auto& [what, ms] : profile_ms) by.emplace_back(ms, what), total += ms;
    std::sort(by.rbegin(), by.rend());
    std::fprintf(stderr, "cpu_ref profile: %.0f ms in the nodes\n", total);
    for (const auto& [ms, what] : by) std::fprintf(stderr, "  %9.1f ms  %s\n", ms, what.c_str());
  }
  // Adds the time since t0 to `what` when profiling.
  void timed(const char* what, std::chrono::steady_clock::time_point t0) const {
    if (!profile) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(mu);
    profile_ms[what] += ms;
  }

  const flyx_tensor_view* w(int i) const {
    if (i < 0 || i >= int(W.size())) throw std::runtime_error("weight " + std::to_string(i) + " missing");
    return W[size_t(i)];
  }
  // Small tensors (norm weights, conv kernels, gate params) as doubles.
  const Vec& vec(int i) const {  // an element's reference outlives later insertions
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(i);
    if (it != cache.end()) return it->second;
    return cache.emplace(i, flyx::ops::to_vec(w(i))).first->second;
  }
  const flyx_block& block(int i) const { return blocks[size_t(i)]; }
  // The block drafter's taps (its context features), 0 without one.
  int taps() const {
    if (dflash < 0) return 0;
    const flyx_block& b = block(dflash);
    return b.kind == FLYX_BLOCK_DSPINE ? b.dspine.n_taps : b.dflash2.n_taps;
  }
};

// x + y as the engine adds to the residual: in a pass of more than FLYX_MAX_ROWS rows of a res16 model, the sum
// (float32 in the engine) rounded to bf16.
double residual_add(const RefModel* M, int rows, double x, double y) {
  return rows > FLYX_MAX_ROWS && M->res16 ? flyx::ops::round_bf16(x + y) : x + y;
}

// A linear layer as the engine runs it: NVFP4 weights see bf16-rounded inputs, GGML block weights those inputs
// as binary16 over a row's scale, or 8-bit inputs in every pass when the weight says so; NVFP4 weights NVFP4
// inputs when the weight says so and the engine's GEMM takes more than FLYX_MAX_ROWS rows at once.  This
// reference computes some layers a row at a time: `rows` are the rows of this call, `gemm_rows` those the engine's
// GEMM takes together (the pass's, for a layer it runs over the whole pass).
void lin(const RefModel* M, int wi, const double* x, int rows, int K, int N, double* y, int gemm_rows) {
  const flyx_tensor_view* w = M->w(wi);
  if (M->q8.count(w)) flyx::ops::linear_a8(x, rows, K, w, N, y);
  else if (gemm_rows > FLYX_MAX_ROWS && M->a4.count(w)) flyx::ops::linear_a4(x, rows, K, w, N, y);
  else if (M->a16h.count(w)) flyx::ops::linear_a16h(x, rows, K, w, N, y);
  else if (M->a16.count(w)) flyx::ops::linear_a16(x, rows, K, w, N, y);
  else flyx::ops::linear(x, rows, K, w, N, y);
}

// A linear's adapter (ABI 43, flyx_lora; rank 0: none): y += scale b (a x) over the same rows, in float64.
void lora(const RefModel* M, const flyx_lora& l, const double* x, int rows, int K, int N, double* y) {
  if (l.rank > 0) flyx::ops::linear_lora(x, rows, K, M->w(l.a), M->w(l.b), l.rank, l.scale, N, y);
}

void norm(const RefModel* M, const flyx_norm& n, const double* x, int len, double* y) {
  flyx::ops::rmsnorm(x, M->vec(n.w).data(), len, n.eps, n.offset, y);
}

void vnorm(const RefModel* M, const flyx_vnorm& n, const double* x, int len, double* y) {
  if (n.w < 0 && n.b < 0) {
    if (y != x) std::copy(x, x + len, y);
    return;
  }
  const double* w = n.w >= 0 ? M->vec(n.w).data() : nullptr;
  const double* b = n.b >= 0 ? M->vec(n.b).data() : nullptr;
  if (n.kind == FLYX_NORM_LAYER) {
    flyx::ops::layernorm(x, w, b, len, n.eps, y);
  } else {
    if (!w) throw std::runtime_error("vnorm: RMSNorm with bias requires weight");
    flyx::ops::rmsnorm(x, w, len, n.eps, n.offset, y);
    if (b)
      for (int i = 0; i < len; ++i) y[i] += b[i];
  }
}

bool has_mrope(const flyx_attn_block& B) {
  return B.mrope_section[0] > 0 || B.mrope_section[1] > 0 || B.mrope_section[2] > 0;
}

// One head of an attention block's q or k as attention takes it (ops::prep_head): its norm, none where the block
// names none (weight -1), then the rotary at pos (or 3D MRoPE coordinates when pos_3d is given).
void prep_head(const RefModel* M, const flyx_attn_block& B, const flyx_norm& n, const double* x, int pos,
               double* y, const int32_t* pos_3d = nullptr) {
  const double* w = n.w < 0 ? nullptr : M->vec(n.w).data();
  if (pos_3d && has_mrope(B)) {
    flyx::ops::prep_head_3d(x, w, B.head_dim, n.eps, n.offset, pos_3d[0], pos_3d[1], pos_3d[2], B.theta, B.rot_dim,
                            B.mrope_section, B.mrope_interleaved != 0, y);
  } else {
    flyx::ops::prep_head(x, w, B.head_dim, n.eps, n.offset, pos, B.theta, B.rot_dim, y);
  }
}

struct RefPlan {
  RefModel* m;
  int phase, width, max_rows;
  std::vector<flyx_node> nodes;
};

// Per state of the description (empty where the state is of another kind).
struct Slot {
  std::vector<Vec> S, conv;                  // GDN
  bool replay_valid = false;                 // replay below describes the round that started here
  int start_pos = 0, rows = 0;               // committed rows of that round
  int rope_delta = 0;                        // decode rope_delta when this round ran
  std::vector<Vec> k, v, beta, g, mixed;     // GDN, [max_rows * ...]
};

struct Ckpt {
  bool valid = false;
  int pos = 0;
  int rope_delta = 0;
  std::vector<Vec> S, conv;
  Vec ring_h;  // the ring's vector at pos - 1 (the MTP drafter's next input)
};

struct RefSeq {
  RefModel* m;
  int max_ctx;
  std::vector<std::vector<Vec>> K, V;  // per KV state: per layer, by position
  Vec hidden_at;                       // the ring: normalised target hidden by position (MTP input)
  Slot slot[FLYX_ROUND_SLOTS];
  Ckpt ck[FLYX_CHECKPOINTS];
  int position = 0, anchor = 0, round = 0;
  int rope_delta = 0;                  // ABI 44: scalar MRoPE decode offset (0 for text-only)
  int32_t prev_pos3d[3] = {0, 0, 0};   // last prefill token's 3D pos (for cross-chunk MTP row 0)
  int prev_pos3d_at = -1;              // sequence position that prev_pos3d corresponds to (-1: none)
  Vec head_logits;  // the head's rows of the last pass, vocab each (logits())
  bool stopped = false;
  std::deque<flyx_round_result> ring;
  uint64_t seq_no = 0;
  std::vector<int32_t> oracle;
  std::vector<int> counts;   // how often each token was chosen in this reply (set_reply; for the penalties)
  flyx_gen_params gen{};  // gen.mask is the caller's, valid during prefill() / launch() only
  // per-run scratch
  int rows = 0, start = 0;
  bool noop = false, sample_last = false;
  std::vector<int32_t> tok, drafts, choice;
  // ABI 44: optional external embeddings & 3D positions for the current prefill pass only
  std::unordered_map<int, const float*> embed_override;
  const int32_t* pos_3d = nullptr;
  Vec x, feats;
  int last_n = 0, last_slot = 0;
};

using namespace flyx::ops;

int cur_slot(const RefSeq* s) { return flyx_round_slot(s->round); }

int cdim_of(const flyx_state& g) { return 2 * g.nk * g.dk + g.nv * g.dv; }

void reset_state(RefSeq* s) {
  const auto& states = s->m->states;
  for (auto& sl : s->slot) {
    sl.S.assign(states.size(), {});
    sl.conv.assign(states.size(), {});
    sl.replay_valid = false;
    for (size_t i = 0; i < states.size(); ++i)
      if (states[i].kind == FLYX_STATE_GDN) {
        const auto& g = states[i];
        sl.S[i].assign(size_t(g.nv) * g.dv * g.dk, 0.0);
        sl.conv[i].assign(size_t(g.conv - 1) * cdim_of(g), 0.0);
      }
  }
  s->position = 0;
  s->round = 0;
  s->rope_delta = 0;
  s->prev_pos3d_at = -1;
  s->stopped = false;
}

void embed_rows(RefSeq* s, const flyx_embed_block& e, const int32_t* ids, int n, Vec* x) {
  const auto* E = s->m->w(e.table);
  const int H = e.hidden;
  x->assign(size_t(n) * H, 0.0);
  for (int r = 0; r < n; ++r) {
    auto ov = s->embed_override.find(r);
    if (ov != s->embed_override.end()) {
      for (int c = 0; c < H; ++c) (*x)[size_t(r) * H + c] = double(ov->second[c]);
    } else {
      for (int c = 0; c < H; ++c) (*x)[size_t(r) * H + c] = load(E, size_t(ids[r]) * H + c);
    }
  }
}

// ---- target layers ----------------------------------------------------------

void gdn_mixer(RefSeq* s, const flyx_gdn_block& B, bool verify) {
  const auto* M = s->m;
  const int L = B.state;
  const int H = B.hidden, nk = B.nk, nv = B.nv, dk = B.dk, dv = B.dv, ck = B.conv;
  const int kd = nk * dk, cdim = 2 * kd + nv * dv, rep = nv / nk;
  Slot& sl = s->slot[cur_slot(s)];
  Vec S = sl.S[L], win = sl.conv[L];  // copies; written back only for prefill
  if (verify) {
    sl.k[L].assign(size_t(s->rows) * nv * dk, 0);
    sl.v[L].assign(size_t(s->rows) * nv * dv, 0);
    sl.beta[L].assign(size_t(s->rows) * nv, 0);
    sl.g[L].assign(size_t(s->rows) * nv, 0);
    sl.mixed[L].assign(size_t(s->rows) * cdim, 0);
  }
  const Vec& cw = M->vec(B.conv_w);
  const Vec& alog = M->vec(B.A_log);
  const Vec& dtb = M->vec(B.dt_bias);
  Vec h(H), mixed(cdim), conv(cdim), z(nv * dv), b(nv), a(nv), o(nv * dv), y(H);
  for (int r = 0; r < s->rows; ++r) {
    double* xr = &s->x[size_t(r) * H];
    norm(M, B.norm, xr, H, h.data());
    lin(M, B.qkv, h.data(), 1, H, cdim, mixed.data(), s->rows);
    lin(M, B.z, h.data(), 1, H, nv * dv, z.data(), s->rows);
    lin(M, B.b, h.data(), 1, H, nv, b.data(), s->rows);
    lin(M, B.a, h.data(), 1, H, nv, a.data(), s->rows);
    // NVFP4 inputs in a pass: the engine holds the in-projection's rows in bf16 (as vLLM and SGLang do)
    if (s->rows > FLYX_MAX_ROWS && M->a4.count(M->w(B.qkv)))
      for (Vec* v : {&mixed, &z, &b, &a})
        for (double& e : *v) e = flyx::ops::round_bf16(e);
    conv_row(win.data(), mixed.data(), cw.data(), cdim, ck, conv.data());
    conv_push(win.data(), mixed.data(), cdim, ck);
    for (int kh = 0; kh < nk; ++kh) {
      l2norm(&conv[size_t(kh) * dk], dk, 1e-6);
      l2norm(&conv[kd + size_t(kh) * dk], dk, 1e-6);
      for (int i = 0; i < dk; ++i) conv[size_t(kh) * dk + i] /= std::sqrt(double(dk));
    }
    for (int vh = 0; vh < nv; ++vh) {
      const int kh = B.head_order == FLYX_GDN_HEADS_TILED ? vh % nk : vh / rep;
      double beta = sigmoid(b[vh]);
      const double neg_exp = B.decay == FLYX_GDN_DECAY_NEG_EXP ? alog[vh] : -std::exp(alog[vh]);
      double g = neg_exp * softplus(a[vh] + dtb[vh]);
      const double* q = &conv[size_t(kh) * dk];
      const double* k = &conv[kd + size_t(kh) * dk];
      const double* v = &conv[2 * kd + size_t(vh) * dv];
      gdn_row(&S[size_t(vh) * dv * dk], q, k, v, beta, g, dk, dv, &o[size_t(vh) * dv]);
      if (verify) {
        std::copy(k, k + dk, &sl.k[L][(size_t(r) * nv + vh) * dk]);
        std::copy(v, v + dv, &sl.v[L][(size_t(r) * nv + vh) * dv]);
        sl.beta[L][size_t(r) * nv + vh] = beta;
        sl.g[L][size_t(r) * nv + vh] = g;
      }
      Vec tmp(static_cast<size_t>(dv));
      norm(M, B.out_norm, &o[size_t(vh) * dv], dv, tmp.data());
      for (int j = 0; j < dv; ++j) o[size_t(vh) * dv + j] = tmp[j] * silu(z[size_t(vh) * dv + j]);
    }
    if (verify) std::copy(mixed.begin(), mixed.end(), &sl.mixed[L][size_t(r) * cdim]);
    lin(M, B.out, o.data(), 1, nv * dv, H, y.data(), s->rows);
    for (int c = 0; c < H; ++c) xr[c] = residual_add(M, s->rows, xr[c], y[c]);
  }
  if (!verify) {
    sl.S[L] = std::move(S);
    sl.conv[L] = std::move(win);
  }
}

// Where head hh's q rows and its gate's rows start in a q projection (flyx_gate_layout).
int q_at(const flyx_attn_block& B, int hh) { return (B.gate == FLYX_GATE_PER_HEAD ? 2 * hh : hh) * B.head_dim; }
int gate_at(const flyx_attn_block& B, int hh) {
  return (B.gate == FLYX_GATE_PER_HEAD ? 2 * hh + 1 : B.n_heads + hh) * B.head_dim;
}

void attn_mixer(RefSeq* s, const flyx_attn_block& B) {
  const auto* M = s->m;
  const int H = B.hidden, nh = B.n_heads, nkv = B.n_kv, hd = B.head_dim, R = s->rows;
  const int qw = (B.gate ? 2 : 1) * nh * hd;  // the q projection's rows: q, and the gate's with an output gate
  Vec h(size_t(R) * H), qg(size_t(R) * qw), k(size_t(R) * nkv * hd), v(size_t(R) * nkv * hd);
  for (int r = 0; r < R; ++r) norm(M, B.norm, &s->x[size_t(r) * H], H, &h[size_t(r) * H]);
  lin(M, B.q, h.data(), R, H, qw, qg.data(), R);
  lin(M, B.k, h.data(), R, H, nkv * hd, k.data(), R);
  lin(M, B.v, h.data(), R, H, nkv * hd, v.data(), R);
  lora(M, B.lora[FLYX_LIN_Q], h.data(), R, H, qw, qg.data());  // never with an output gate (open)
  lora(M, B.lora[FLYX_LIN_K], h.data(), R, H, nkv * hd, k.data());
  lora(M, B.lora[FLYX_LIN_V], h.data(), R, H, nkv * hd, v.data());
  Vec& KC = s->K[size_t(B.kv)][0];
  Vec& VC = s->V[size_t(B.kv)][0];
  Vec tmp(hd);
  for (int r = 0; r < R; ++r) {
    int pos = s->start + r;
    const int32_t* p3 = s->pos_3d ? s->pos_3d + size_t(r) * 3 : nullptr;
    const int rope_pos = pos + (has_mrope(B) ? s->rope_delta : 0);
    for (int j = 0; j < nkv; ++j) {
      prep_head(M, B, B.k_norm, &k[(size_t(r) * nkv + j) * hd], rope_pos, tmp.data(), p3);
      std::copy(tmp.begin(), tmp.end(), &KC[(size_t(pos) * nkv + j) * hd]);
      std::copy(&v[(size_t(r) * nkv + j) * hd], &v[(size_t(r) * nkv + j) * hd] + hd, &VC[(size_t(pos) * nkv + j) * hd]);
    }
  }
  Vec o(size_t(nh) * hd), y(H), q(hd);
  for (int r = 0; r < R; ++r) {
    int pos = s->start + r;
    const int32_t* p3 = s->pos_3d ? s->pos_3d + size_t(r) * 3 : nullptr;
    const int rope_pos = pos + (has_mrope(B) ? s->rope_delta : 0);
    for (int hh = 0; hh < nh; ++hh) {
      prep_head(M, B, B.q_norm, &qg[size_t(r) * qw + q_at(B, hh)], rope_pos, q.data(), p3);
      attend(q.data(), KC.data(), VC.data(), pos + 1, nkv, hd, hh / (nh / nkv), 1.0 / std::sqrt(double(hd)),
             &o[size_t(hh) * hd]);
    }
    if (B.gate)
      for (int hh = 0; hh < nh; ++hh) {
        const double* gate = &qg[size_t(r) * qw + gate_at(B, hh)];
        for (int i = 0; i < hd; ++i) o[size_t(hh) * hd + i] *= sigmoid(gate[i]);
      }
    lin(M, B.o, o.data(), 1, nh * hd, H, y.data(), R);
    lora(M, B.lora[FLYX_LIN_O], o.data(), 1, nh * hd, H, y.data());
    for (int c = 0; c < H; ++c) s->x[size_t(r) * H + c] = residual_add(M, R, s->x[size_t(r) * H + c], y[c]);
  }
}

// x += down(silu(gate(h)) * up(h)), h = norm(x), for R rows of x.
void mlp(const RefModel* M, const flyx_mlp_block& B, double* x, int R) {
  const int H = B.hidden, I = B.inter;
  Vec h(H), g(I), u(I), y(H);
  for (int r = 0; r < R; ++r) {
    double* xr = &x[size_t(r) * H];
    norm(M, B.norm, xr, H, h.data());
    lin(M, B.gate, h.data(), 1, H, I, g.data(), R);
    lin(M, B.up, h.data(), 1, H, I, u.data(), R);
    lora(M, B.lora[0], h.data(), 1, H, I, g.data());
    lora(M, B.lora[1], h.data(), 1, H, I, u.data());
    for (int i = 0; i < I; ++i) g[i] = silu(g[i]) * u[i];
    // NVFP4 inputs in a pass: the engine's gate|up GEMM leaves its SwiGLU rows in bf16 (as vLLM's and SGLang's)
    if (R > FLYX_MAX_ROWS && M->a4.count(M->w(B.down)))
      for (int i = 0; i < I; ++i) g[i] = flyx::ops::round_bf16(g[i]);
    lin(M, B.down, g.data(), 1, I, H, y.data(), R);
    lora(M, B.lora[2], g.data(), 1, I, H, y.data());
    for (int c = 0; c < H; ++c) xr[c] = residual_add(M, R, xr[c], y[c]);
  }
}

bool penalized(const flyx_gen_params& g) { return g.presence_penalty != 0.f || g.frequency_penalty != 0.f; }

// A token the reply now has (accept, the first token of a reply).
void count_token(RefSeq* s, int32_t t) {
  if (!penalized(s->gen)) return;
  if (s->counts.empty()) s->counts.assign(size_t(s->m->vocab), 0);
  ++s->counts[size_t(t)];
}

// round: rows after the first read drafts, which count as the reply for the rows after them.
void head_rows(RefSeq* s, const flyx_head_block& B, int r0, int r1, bool round) {
  const int H = B.hidden, V = B.vocab;
  Vec hn(H), logits(V);
  std::vector<int> counts;
  if (penalized(s->gen)) {
    counts = s->counts;
    counts.resize(size_t(V), 0);
  }
  for (int r = r0; r < r1; ++r) {
    norm(s->m, B.norm, &s->x[size_t(r) * H], H, hn.data());
    lin(s->m, B.w, hn.data(), 1, H, V, logits.data(), r1 - r0);
    if (s->head_logits.size() < size_t(r + 1) * V) s->head_logits.resize(size_t(r + 1) * V);
    std::copy(logits.begin(), logits.end(), s->head_logits.begin() + ptrdiff_t(r) * V);
    if (round && r > 0 && !counts.empty()) ++counts[size_t(s->drafts[size_t(r) - 1])];
    // The first row the head chooses is the token this call emits first: the mask applies there only.
    s->choice[r] = sample_token(logits.data(), V, s->gen, int64_t(s->start) + r + 1, r == r0 ? s->gen.mask : nullptr,
                                counts.empty() ? nullptr : counts.data());
  }
}

// ---- DFlash2 drafter ----------------------------------------------------------

const flyx_dflash2_layer_block& dlayer(const RefModel* M, const flyx_dflash2_block& d, int j) {
  return M->block(d.first_layer + j).dflash2_layer;
}

// Drafts that replace the drafter's: this launch's (gen.drafts), else the test oracle's.
bool forced_drafts(const RefSeq* s) { return (s->gen.drafts && s->gen.n_drafts > 0) || !s->oracle.empty(); }

int32_t forced_draft(const RefSeq* s, int pos) {
  if (s->gen.drafts && s->gen.n_drafts > 0) {
    const int k = pos - s->gen.drafts_at;
    return k >= 0 && k < s->gen.n_drafts ? s->gen.drafts[k] : 0;
  }
  return pos < static_cast<int>(s->oracle.size()) ? s->oracle[pos] : 0;
}

void draft_context(RefSeq* s, const flyx_dflash2_block& d, int n_rows) {
  const auto* M = s->m;
  const int H = d.hidden, F = d.n_taps * M->hidden, hd = d.head_dim;
  Vec h(H), hn(H), k(size_t(d.n_kv) * hd), v(k.size()), tmp(hd);
  for (int r = 0; r < n_rows; ++r) {
    int pos = s->start + r;
    lin(M, d.fc, &s->feats[size_t(r) * F], 1, F, H, h.data(), n_rows);
    norm(M, d.hidden_norm, h.data(), H, hn.data());
    for (int j = 0; j < d.layers; ++j) {
      const auto& l = dlayer(M, d, j);
      lin(M, l.k, hn.data(), 1, H, d.n_kv * hd, k.data(), n_rows);
      lin(M, l.v, hn.data(), 1, H, d.n_kv * hd, v.data(), n_rows);
      for (int kh = 0; kh < d.n_kv; ++kh) {
        norm(M, l.k_norm, &k[size_t(kh) * hd], hd, tmp.data());
        rope(tmp.data(), pos, d.theta, hd);
        std::copy(tmp.begin(), tmp.end(), &s->K[size_t(d.kv)][size_t(j)][(size_t(pos) * d.n_kv + kh) * hd]);
        std::copy(&v[size_t(kh) * hd], &v[size_t(kh) * hd] + hd,
                  &s->V[size_t(d.kv)][size_t(j)][(size_t(pos) * d.n_kv + kh) * hd]);
      }
    }
  }
}

void draft_conv(const RefModel* M, int proj, int base_w, const Vec& h, int B, int H,
                const flyx_dflash2_block& d, Vec* out, Vec* delta_out) {
  int G = H / d.conv_group, T = d.conv_taps;
  Vec ker(size_t(B) * 2 * T * G);
  lin(M, proj, h.data(), B, H, 2 * T * G, ker.data(), B);
  Vec d0(size_t(B) * T * G);
  delta_out->assign(size_t(B) * T * G, 0);
  for (int r = 0; r < B; ++r)
    for (int i = 0; i < T * G; ++i) {
      d0[size_t(r) * T * G + i] = ker[size_t(r) * 2 * T * G + i];
      (*delta_out)[size_t(r) * T * G + i] = ker[size_t(r) * 2 * T * G + T * G + i];
    }
  const Vec& base = M->vec(base_w);  // [2, T, H]
  out->assign(size_t(B) * H, 0);
  grouped_conv(h.data(), d0.data(), base.data(), B, H, T, d.conv_group, B, out->data());  // the B rows: one block
}

void draft_block(RefSeq* s, const flyx_dflash2_block& d, int width) {
  const auto* M = s->m;
  // a chain wider than the block: width + 1 mask rows in one block (the CUDA backend's wide block)
  const int B = std::min(FLYX_MAX_ROWS, std::max(d.block, width + 1)), H = d.hidden, nh = d.n_heads, nkv = d.n_kv, hd = d.head_dim, I = d.inter;
  const int ctx = s->position, T = d.conv_taps;
  std::vector<int32_t> ids(B, d.mask_token);
  ids[0] = s->anchor;
  Vec x;
  embed_rows(s, M->block(d.embed).embed, ids.data(), B, &x);
  Vec h(size_t(B) * H), hc, delta, q(size_t(B) * nh * hd), k(size_t(B) * nkv * hd), v(k.size()), tmp(hd);
  for (int j = 0; j < d.layers; ++j) {
    const auto& l = dlayer(M, d, j);
    const Vec& dK = s->K[size_t(d.kv)][size_t(j)];
    const Vec& dV = s->V[size_t(d.kv)][size_t(j)];
    for (int r = 0; r < B; ++r) norm(M, l.norm, &x[size_t(r) * H], H, &h[size_t(r) * H]);
    draft_conv(M, l.attn_conv_proj, l.attn_conv_base, h, B, H, d, &hc, &delta);
    lin(M, l.q, hc.data(), B, H, nh * hd, q.data(), B);
    lin(M, l.k, hc.data(), B, H, nkv * hd, k.data(), B);
    lin(M, l.v, hc.data(), B, H, nkv * hd, v.data(), B);
    Vec KA(size_t(ctx + B) * nkv * hd), VA(KA.size());
    std::copy(dK.begin(), dK.begin() + size_t(ctx) * nkv * hd, KA.begin());
    std::copy(dV.begin(), dV.begin() + size_t(ctx) * nkv * hd, VA.begin());
    for (int r = 0; r < B; ++r) {
      for (int kh = 0; kh < nkv; ++kh) {
        norm(M, l.k_norm, &k[(size_t(r) * nkv + kh) * hd], hd, tmp.data());
        rope(tmp.data(), ctx + r, d.theta, hd);
        std::copy(tmp.begin(), tmp.end(), &KA[(size_t(ctx + r) * nkv + kh) * hd]);
        std::copy(&v[(size_t(r) * nkv + kh) * hd], &v[(size_t(r) * nkv + kh) * hd] + hd, &VA[(size_t(ctx + r) * nkv + kh) * hd]);
      }
    }
    Vec o(size_t(B) * nh * hd);
    for (int r = 0; r < B; ++r) {
      // the keys it sees: all, or those less than `window` from its position, both ways (a contiguous run)
      const int qp = ctx + r, lo = d.window > 0 ? std::max(0, qp - d.window + 1) : 0;
      const int hi = d.window > 0 ? std::min(ctx + B, qp + d.window) : ctx + B;
      for (int hh = 0; hh < nh; ++hh) {
        norm(M, l.q_norm, &q[(size_t(r) * nh + hh) * hd], hd, tmp.data());
        rope(tmp.data(), qp, d.theta, hd);
        attend(tmp.data(), KA.data() + size_t(lo) * nkv * hd, VA.data() + size_t(lo) * nkv * hd, hi - lo, nkv, hd,
               hh / (nh / nkv), 1.0 / std::sqrt(double(hd)), &o[(size_t(r) * nh + hh) * hd]);
      }
    }
    Vec ao(size_t(B) * H), fin(size_t(B) * H);
    lin(M, l.o, o.data(), B, nh * hd, H, ao.data(), B);
    const Vec& base = M->vec(l.attn_conv_base);
    grouped_conv(ao.data(), delta.data(), base.data() + size_t(T) * H, B, H, T, d.conv_group, B, fin.data());
    for (size_t i = 0; i < x.size(); ++i) x[i] += fin[i];
    for (int r = 0; r < B; ++r) norm(M, l.post_norm, &x[size_t(r) * H], H, &h[size_t(r) * H]);
    draft_conv(M, l.mlp_conv_proj, l.mlp_conv_base, h, B, H, d, &hc, &delta);
    Vec g(size_t(B) * I), u(size_t(B) * I), mo(size_t(B) * H);
    lin(M, l.gate, hc.data(), B, H, I, g.data(), B);
    lin(M, l.up, hc.data(), B, H, I, u.data(), B);
    for (size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
    lin(M, l.down, g.data(), B, I, H, mo.data(), B);
    const Vec& base2 = M->vec(l.mlp_conv_base);
    grouped_conv(mo.data(), delta.data(), base2.data() + size_t(T) * H, B, H, T, d.conv_group, B, fin.data());
    for (size_t i = 0; i < x.size(); ++i) x[i] += fin[i];
  }
  const auto& head = M->block(d.head).head;
  const int V = head.vocab, K = d.sel_topk, Rk = d.sel_rank;
  Vec hn(H), logits(V), hid(Rk), val(K);
  std::vector<int> cand(K);
  const auto* pred = M->w(d.sel_pred);
  const auto* succ = M->w(d.sel_succ);
  int prev = s->anchor;
  s->drafts.assign(width, 0);
  for (int st = 0; st < B - 1 && st < width; ++st) {
    norm(M, d.norm, &x[size_t(st + 1) * H], H, hn.data());
    linear(hn.data(), 1, H, M->w(head.w), V, logits.data());
    topk(logits.data(), V, K, cand.data(), val.data());
    lin(M, d.sel_hid, hn.data(), 1, H, Rk, hid.data(), 1);
    int best = 0;
    double best_score = -INFINITY;
    for (int c = 0; c < K; ++c) {
      double e = 0;
      for (int i = 0; i < Rk; ++i)
        e += load(pred, size_t(prev) * Rk + i) * hid[i] * load(succ, size_t(cand[c]) * Rk + i);
      double sc = val[c] + e;
      if (sc > best_score) {
        best_score = sc;
        best = c;
      }
    }
    prev = cand[best];
    s->drafts[st] = prev;
  }
  if (forced_drafts(s))
    for (int i = 0; i < width; ++i) s->drafts[i] = forced_draft(s, ctx + 1 + i);
}

// ---- DSpine drafter (python/reference/flyx_ref/dspine.py, ops/dspine/ref.h) ------

const flyx_dspine_layer_block& slayer(const RefModel* M, const flyx_dspine_block& d, int j) {
  return M->block(d.first_layer + j).dspine_layer;
}

// A weight's float64 elements, row-major (every weight is decoded to float64 at open).
const double* f64(const RefModel* M, int w) { return static_cast<const double*>(M->w(w)->data); }

// The context: DFlash2's (draft_context) with DSpine's layers.
void dspine_context(RefSeq* s, const flyx_dspine_block& d, int n_rows) {
  const auto* M = s->m;
  const int H = d.hidden, F = d.n_taps * M->hidden, hd = d.head_dim;
  Vec h(H), hn(H), k(size_t(d.n_kv) * hd), v(k.size()), tmp(hd);
  for (int r = 0; r < n_rows; ++r) {
    int pos = s->start + r;
    lin(M, d.fc, &s->feats[size_t(r) * F], 1, F, H, h.data(), n_rows);
    norm(M, d.hidden_norm, h.data(), H, hn.data());
    for (int j = 0; j < d.layers; ++j) {
      const auto& l = slayer(M, d, j);
      lin(M, l.k, hn.data(), 1, H, d.n_kv * hd, k.data(), n_rows);
      lin(M, l.v, hn.data(), 1, H, d.n_kv * hd, v.data(), n_rows);
      for (int kh = 0; kh < d.n_kv; ++kh) {
        norm(M, l.k_norm, &k[size_t(kh) * hd], hd, tmp.data());
        rope(tmp.data(), pos, d.theta, hd);
        std::copy(tmp.begin(), tmp.end(), &s->K[size_t(d.kv)][size_t(j)][(size_t(pos) * d.n_kv + kh) * hd]);
        std::copy(&v[size_t(kh) * hd], &v[size_t(kh) * hd] + hd,
                  &s->V[size_t(d.kv)][size_t(j)][(size_t(pos) * d.n_kv + kh) * hd]);
      }
    }
  }
}

// The block of the rows the round's drafts need (no row reads a row after it), its layers attending causally (within
// the window), each layer's write, then the head's candidates and the walk.
void dspine_block(RefSeq* s, const flyx_dspine_block& d, int width) {
  const auto* M = s->m;
  const int B = std::min(d.block, width + 1), H = d.hidden, nh = d.n_heads, nkv = d.n_kv, hd = d.head_dim;
  const int I = d.inter, ctx = s->position, K = d.topk;
  std::vector<int32_t> ids(B, d.mask_token);
  ids[0] = s->anchor;
  Vec x;
  embed_rows(s, M->block(d.embed).embed, ids.data(), B, &x);
  const flyx::ops::DSpineBridge br{H,           d.code,           d.rank,           d.bridge_eps, f64(M, d.codebook),
                                    f64(M, d.gate), f64(M, d.pred), f64(M, d.prev), M->vec(d.gate_bias).data()};
  Vec h(size_t(B) * H), q(size_t(B) * nh * hd), k(size_t(B) * nkv * hd), v(k.size()), tmp(hd);
  Vec before(size_t(B) * H), rms(B), g(size_t(B) * d.rank);
  for (int j = 0; j < d.layers; ++j) {
    const auto& l = slayer(M, d, j);
    const Vec& dK = s->K[size_t(d.kv)][size_t(j)];
    const Vec& dV = s->V[size_t(d.kv)][size_t(j)];
    for (int r = 0; r < B; ++r) norm(M, l.norm, &x[size_t(r) * H], H, &h[size_t(r) * H]);
    lin(M, l.q, h.data(), B, H, nh * hd, q.data(), B);
    lin(M, l.k, h.data(), B, H, nkv * hd, k.data(), B);
    lin(M, l.v, h.data(), B, H, nkv * hd, v.data(), B);
    Vec KA(size_t(ctx + B) * nkv * hd), VA(KA.size());
    std::copy(dK.begin(), dK.begin() + size_t(ctx) * nkv * hd, KA.begin());
    std::copy(dV.begin(), dV.begin() + size_t(ctx) * nkv * hd, VA.begin());
    for (int r = 0; r < B; ++r) {
      for (int kh = 0; kh < nkv; ++kh) {
        norm(M, l.k_norm, &k[(size_t(r) * nkv + kh) * hd], hd, tmp.data());
        rope(tmp.data(), ctx + r, d.theta, hd);
        std::copy(tmp.begin(), tmp.end(), &KA[(size_t(ctx + r) * nkv + kh) * hd]);
        std::copy(&v[(size_t(r) * nkv + kh) * hd], &v[(size_t(r) * nkv + kh) * hd] + hd,
                  &VA[(size_t(ctx + r) * nkv + kh) * hd]);
      }
    }
    Vec o(size_t(B) * nh * hd);
    for (int r = 0; r < B; ++r) {
      // the keys it sees: those up to its own position, within the window when it has one (a contiguous run)
      const int qp = ctx + r, lo = d.window > 0 ? std::max(0, qp - d.window + 1) : 0;
      for (int hh = 0; hh < nh; ++hh) {
        norm(M, l.q_norm, &q[(size_t(r) * nh + hh) * hd], hd, tmp.data());
        rope(tmp.data(), qp, d.theta, hd);
        attend(tmp.data(), KA.data() + size_t(lo) * nkv * hd, VA.data() + size_t(lo) * nkv * hd, qp + 1 - lo, nkv, hd,
               hh / (nh / nkv), 1.0 / std::sqrt(double(hd)), &o[(size_t(r) * nh + hh) * hd]);
      }
    }
    Vec ao(size_t(B) * H);
    lin(M, l.o, o.data(), B, nh * hd, H, ao.data(), B);
    for (size_t i = 0; i < x.size(); ++i) x[i] += ao[i];
    for (int r = 0; r < B; ++r) norm(M, l.post_norm, &x[size_t(r) * H], H, &h[size_t(r) * H]);
    Vec gt(size_t(B) * I), u(size_t(B) * I), mo(size_t(B) * H);
    lin(M, l.gate, h.data(), B, H, I, gt.data(), B);
    lin(M, l.up, h.data(), B, H, I, u.data(), B);
    for (size_t i = 0; i < gt.size(); ++i) gt[i] = silu(gt[i]) * u[i];
    lin(M, l.down, gt.data(), B, I, H, mo.data(), B);
    for (size_t i = 0; i < x.size(); ++i) x[i] += mo[i];
    const bool last = j == d.layers - 1;  // the walk redoes the last write: what it started from
    flyx::ops::dspine_write(br, f64(M, l.read), f64(M, l.write), s->anchor, B, x.data(), last ? rms.data() : nullptr,
                            last ? g.data() : nullptr, last ? before.data() : nullptr);
  }
  const auto& head = M->block(d.head).head;
  const int V = head.vocab;
  Vec hn(H), logits(V), val(K);
  std::vector<int> c(K);
  std::vector<int32_t> cand(size_t(B - 1) * K), drafts(size_t(B - 1));
  for (int st = 0; st < B - 1; ++st) {
    norm(M, d.norm, &x[size_t(st + 1) * H], H, hn.data());
    linear(hn.data(), 1, H, M->w(head.w), V, logits.data());
    topk(logits.data(), V, K, c.data(), val.data());
    std::copy(c.begin(), c.end(), cand.begin() + size_t(st) * K);
  }
  const double* hw = f64(M, head.w);
  flyx::ops::dspine_walk(br, f64(M, slayer(M, d, d.layers - 1).write), M->vec(d.norm.w).data(), before.data(),
                         rms.data(), g.data(), cand.data(), K, B - 1, s->anchor,
                         [&](int32_t t) { return hw + size_t(t) * H; }, drafts.data());
  s->drafts.assign(width, 0);
  for (int i = 0; i < width && i < B - 1; ++i) s->drafts[i] = drafts[size_t(i)];
  if (forced_drafts(s))
    for (int i = 0; i < width; ++i) s->drafts[i] = forced_draft(s, ctx + 1 + i);
}

// ---- MTP drafter (python/reference/flyx_ref/mtp.py) ----------------------------

// x = fc([norm_emb(embed(tok)), norm_hidden(h)]); the engine's GEMM takes gemm_rows rows at once.
void mtp_input(RefSeq* s, const flyx_mtp_block& B, const double* h, int tok, double* x, int gemm_rows,
               const float* emb_override = nullptr) {
  const auto* M = s->m;
  const auto& e = M->block(B.embed).embed;
  const int H = e.hidden;
  Vec cat(2 * size_t(H)), emb(H);
  if (emb_override) {
    for (int c = 0; c < H; ++c) emb[c] = double(emb_override[c]);
  } else {
    const auto* E = M->w(e.table);
    for (int c = 0; c < H; ++c) emb[c] = load(E, size_t(tok) * H + c);
  }
  norm(M, B.norm_emb, emb.data(), H, cat.data());
  norm(M, B.norm_hidden, h, H, cat.data() + H);
  lin(M, B.fc, cat.data(), 1, 2 * H, H, x, gemm_rows);
}

// Writes the K/V entry at pos from layer input x; q (nh*hd, normed and rotated) and gate if given.  The engine's
// GEMMs take gemm_rows rows at once.
void mtp_qkv(RefSeq* s, const flyx_attn_block& A, const double* x, int pos, double* q, double* gate,
             int gemm_rows, const int32_t* pos_3d = nullptr) {
  const auto* M = s->m;
  const int H = A.hidden, nh = A.n_heads, nkv = A.n_kv, hd = A.head_dim;
  const int qw = (A.gate ? 2 : 1) * nh * hd;
  const int rope_pos = pos + (has_mrope(A) ? s->rope_delta : 0);
  Vec h(H), qg(qw), k(size_t(nkv) * hd), v(size_t(nkv) * hd), tmp(hd);
  Vec& mK = s->K[size_t(A.kv)][0];
  Vec& mV = s->V[size_t(A.kv)][0];
  norm(M, A.norm, x, H, h.data());
  lin(M, A.k, h.data(), 1, H, nkv * hd, k.data(), gemm_rows);
  lin(M, A.v, h.data(), 1, H, nkv * hd, v.data(), gemm_rows);
  for (int j = 0; j < nkv; ++j) {
    prep_head(M, A, A.k_norm, &k[size_t(j) * hd], rope_pos, tmp.data(), pos_3d);
    std::copy(tmp.begin(), tmp.end(), &mK[(size_t(pos) * nkv + j) * hd]);
    std::copy(&v[size_t(j) * hd], &v[size_t(j) * hd] + hd, &mV[(size_t(pos) * nkv + j) * hd]);
  }
  if (!q) return;
  lin(M, A.q, h.data(), 1, H, qw, qg.data(), gemm_rows);
  for (int hh = 0; hh < nh; ++hh) {
    prep_head(M, A, A.q_norm, &qg[size_t(q_at(A, hh))], rope_pos, &q[size_t(hh) * hd], pos_3d);
    if (A.gate) std::copy_n(&qg[size_t(gate_at(A, hh))], hd, gate + size_t(hh) * hd);
  }
}

// Drafts `width` tokens from the anchor, one MTP step each.
void mtp_draft(RefSeq* s, const flyx_mtp_block& B, int width) {
  const auto* M = s->m;
  const auto& A = M->block(B.attn).attn;
  const auto& F = M->block(B.mlp).mlp;
  const auto& head = M->block(B.head).head;
  const int H = A.hidden, nh = A.n_heads, nkv = A.n_kv, hd = A.head_dim, I = F.inter, V = head.vocab;
  const int p = s->position;
  Vec h(s->hidden_at.begin() + size_t(p - 1) * H, s->hidden_at.begin() + size_t(p) * H);
  Vec x(H), q(size_t(nh) * hd), gate(size_t(nh) * hd), o(size_t(nh) * hd), y(H), tn(H), g(I), u(I), logits(V);
  int tok = s->anchor;
  s->drafts.assign(width, 0);
  for (int i = 0; i < width; ++i) {
    const int pos = p - 1 + i;
    mtp_input(s, B, h.data(), tok, x.data(), 1);  // a draft step: one row
    mtp_qkv(s, A, x.data(), pos, q.data(), gate.data(), 1);
    const Vec& mK = s->K[size_t(A.kv)][0];
    const Vec& mV = s->V[size_t(A.kv)][0];
    for (int hh = 0; hh < nh; ++hh)
      attend(&q[size_t(hh) * hd], mK.data(), mV.data(), pos + 1, nkv, hd, hh / (nh / nkv), 1.0 / std::sqrt(double(hd)),
             &o[size_t(hh) * hd]);
    if (A.gate)
      for (int k = 0; k < nh * hd; ++k) o[k] *= sigmoid(gate[k]);
    lin(M, A.o, o.data(), 1, nh * hd, H, y.data(), 1);
    for (int c = 0; c < H; ++c) x[c] += y[c];
    norm(M, F.norm, x.data(), H, tn.data());
    lin(M, F.gate, tn.data(), 1, H, I, g.data(), 1);
    lin(M, F.up, tn.data(), 1, H, I, u.data(), 1);
    for (int k = 0; k < I; ++k) g[k] = silu(g[k]) * u[k];
    lin(M, F.down, g.data(), 1, I, H, y.data(), 1);
    for (int c = 0; c < H; ++c) x[c] += y[c];
    norm(M, B.norm, x.data(), H, h.data());
    linear(h.data(), 1, H, M->w(head.w), V, logits.data());
    int d = argmax(logits.data(), V);
    if (forced_drafts(s)) d = forced_draft(s, p + 1 + i);
    s->drafts[i] = d;
    tok = d;
  }
}

// After a pass over `rows` rows starting at position base: keep their
// normalised hidden and write the MTP K/V entries base-1 .. base+rows-2 from
// (hidden at j, token at j+1).
void mtp_context(RefSeq* s, const flyx_mtp_block& B, int rows) {
  const auto* M = s->m;
  const auto& head = M->block(B.head).head;
  const int H = head.hidden, base = s->start;
  Vec hn(size_t(rows) * H), x(H);
  for (int r = 0; r < rows; ++r) norm(M, head.norm, &s->x[size_t(r) * H], H, &hn[size_t(r) * H]);
  for (int r = 0; r < rows; ++r) {
    const int pos = base - 1 + r;
    if (pos < 0) continue;
    const double* h = r == 0 ? &s->hidden_at[size_t(pos) * H] : &hn[size_t(r - 1) * H];
    auto ov = s->embed_override.find(r);
    const float* emb = ov != s->embed_override.end() ? ov->second : nullptr;
    const int32_t* p3 = nullptr;
    if (s->pos_3d) {
      if (r > 0) p3 = s->pos_3d + size_t(r - 1) * 3;
      else if (s->prev_pos3d_at == pos) p3 = s->prev_pos3d;
    }
    mtp_input(s, B, h, s->tok[r], x.data(), rows, emb);  // the pass's rows, as the engine runs the context
    mtp_qkv(s, M->block(B.attn).attn, x.data(), pos, nullptr, nullptr, rows, p3);
  }
  for (int r = 0; r < rows; ++r) std::copy(&hn[size_t(r) * H], &hn[size_t(r) * H] + H, &s->hidden_at[size_t(base + r) * H]);
}

// ---- round bookkeeping ----------------------------------------------------------

void accept(RefSeq* s, int width) {
  const flyx_gen_params& g = s->gen;
  const int n_stop = std::min<int>(g.n_stop_ids, FLYX_MAX_STOP_IDS);
  flyx_round_result r{};
  r.seq_no = ++s->seq_no;
  if (s->stopped) {
    s->noop = true;
    r.stopped = 1;
    r.position = s->position;
    s->ring.push_back(r);
    return;
  }
  const int n = accepted_drafts(s->drafts.data(), s->choice.data(), width, g.stop_ids, n_stop);
  r.n_tokens = n + 1;
  r.n_accepted = n;
  for (int i = 0; i <= n; ++i) {
    r.tokens[i] = s->choice[i];
    count_token(s, s->choice[i]);
  }
  r.n_drafts = width;
  for (int i = 0; i < width; ++i) r.drafts[i] = s->drafts[i], r.parents[i] = i - 1;
  s->last_n = n;
  s->last_slot = cur_slot(s);
  s->start = s->position;
  s->position += n + 1;
  s->anchor = s->choice[n];
  s->round += 1;
  s->stopped = reply_ends(s->choice[n], s->position, g.limit, s->max_ctx, g.stop_ids, n_stop);
  r.stopped = s->stopped;
  r.position = s->position;
  s->ring.push_back(r);
}

// New state = old slot state advanced by `rows` replayed rows, for every GDN state.
void fold_into(RefSeq* s, int from, int to, int rows) {
  const auto& states = s->m->states;
  Slot& a = s->slot[from];
  Slot& b = s->slot[to];
  for (size_t L = 0; L < states.size(); ++L) {
    const auto& st = states[L];
    if (st.kind != FLYX_STATE_GDN) continue;
    const int nv = st.nv, dk = st.dk, dv = st.dv, cdim = cdim_of(st);
    Vec S = a.S[L], win = a.conv[L];
    for (int r = 0; r < rows; ++r) {
      for (int vh = 0; vh < nv; ++vh)
        gdn_row(&S[size_t(vh) * dv * dk], &a.k[L][(size_t(r) * nv + vh) * dk] /*q unused*/,
                &a.k[L][(size_t(r) * nv + vh) * dk], &a.v[L][(size_t(r) * nv + vh) * dv],
                a.beta[L][size_t(r) * nv + vh], a.g[L][size_t(r) * nv + vh], dk, dv, nullptr);
      conv_push(win.data(), &a.mixed[L][size_t(r) * cdim], cdim, st.conv);
    }
    b.S[L] = std::move(S);
    b.conv[L] = std::move(win);
  }
}

void commit(RefSeq* s) {
  if (s->noop) return;
  const int from = s->last_slot, to = flyx_round_slot(from + 1);
  fold_into(s, from, to, s->last_n + 1);
  s->slot[from].replay_valid = true;
  s->slot[from].start_pos = s->start;
  s->slot[from].rows = s->last_n + 1;
  s->slot[from].rope_delta = s->rope_delta;
  s->slot[to].replay_valid = false;
}

void ensure_slot_buffers(RefSeq* s) {
  const size_t n = s->m->states.size();
  for (auto& sl : s->slot) {
    sl.k.resize(n);
    sl.v.resize(n);
    sl.beta.resize(n);
    sl.g.resize(n);
    sl.mixed.resize(n);
  }
}

// What run_nodes' profile calls a node.
const char* node_label(const flyx_node& n, const flyx_block* b, int phase) {
  const bool pre = phase == FLYX_PHASE_PREFILL;
  switch (n.kind) {
    case FLYX_NODE_RUN:
      switch (b->kind) {
        case FLYX_BLOCK_EMBED: return pre ? "prefill embed" : "round embed";
        case FLYX_BLOCK_GDN: return pre ? "prefill gdn" : "round gdn";
        case FLYX_BLOCK_ATTN: return pre ? "prefill attention" : "round attention";
        case FLYX_BLOCK_MLP: return pre ? "prefill mlp" : "round mlp";
        case FLYX_BLOCK_HEAD: return pre ? "prefill head" : "round head";
        default: return "run";
      }
    case FLYX_NODE_DRAFT:
      return b->kind == FLYX_BLOCK_MTP ? "draft mtp" : b->kind == FLYX_BLOCK_DSPINE ? "draft dspine" : "draft dflash2";
    case FLYX_NODE_CONTEXT:
      if (b->kind == FLYX_BLOCK_DV) return "context dv";
      return b->kind == FLYX_BLOCK_MTP      ? "context mtp"
             : b->kind == FLYX_BLOCK_DSPINE ? "context dspine"
                                            : "context dflash2";
    case FLYX_NODE_ACCEPT: return "accept";
    default: return "other nodes";
  }
}

void run_nodes(RefPlan* p, RefSeq* s) {
  const auto* M = s->m;
  const int H = M->hidden;
  for (const auto& n : p->nodes) {
    const flyx_block* b = n.block >= 0 ? &M->block(n.block) : nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    switch (n.kind) {
      case FLYX_NODE_RUN:
        switch (b->kind) {
          case FLYX_BLOCK_EMBED:
            if (p->phase == FLYX_PHASE_ROUND) {
              s->tok.assign(1, s->anchor);
              s->tok.insert(s->tok.end(), s->drafts.begin(), s->drafts.begin() + p->width);
              s->rows = p->width + 1;
              s->start = s->position;
            }
            embed_rows(s, b->embed, s->tok.data(), s->rows, &s->x);
            s->feats.assign(size_t(s->rows) * size_t(M->taps()) * H, 0.0);
            s->choice.assign(s->rows, 0);
            break;
          case FLYX_BLOCK_GDN: gdn_mixer(s, b->gdn, p->phase == FLYX_PHASE_ROUND); break;
          case FLYX_BLOCK_ATTN: attn_mixer(s, b->attn); break;
          case FLYX_BLOCK_MLP: mlp(M, b->mlp, s->x.data(), s->rows); break;
          case FLYX_BLOCK_HEAD:
            if (n.arg0) {
              if (s->sample_last) head_rows(s, b->head, s->rows - 1, s->rows, false);
            } else {
              head_rows(s, b->head, 0, s->rows, true);
            }
            break;
          default: throw std::runtime_error("cpu_ref: cannot run block kind " + std::to_string(b->kind));
        }
        break;
      case FLYX_NODE_TAP: {
        const int F = M->taps() * H;
        for (int r = 0; r < s->rows; ++r)
          std::copy(&s->x[size_t(r) * H], &s->x[size_t(r) * H] + H, &s->feats[size_t(r) * F + size_t(n.arg0) * H]);
        break;
      }
      case FLYX_NODE_DRAFT:
        if (b->kind == FLYX_BLOCK_MTP) mtp_draft(s, b->mtp, n.arg0);
        else if (b->kind == FLYX_BLOCK_DSPINE) dspine_block(s, b->dspine, n.arg0);
        else draft_block(s, b->dflash2, n.arg0);
        break;
      case FLYX_NODE_CONTEXT: {
        if (b->kind == FLYX_BLOCK_DV) break;  // a prefill's (lower): DV's canvas starts over, and there is none here
        const int rows = p->phase == FLYX_PHASE_PREFILL ? s->rows : s->noop ? 0 : s->last_n + 1;
        if (rows == 0) break;
        if (b->kind == FLYX_BLOCK_MTP) mtp_context(s, b->mtp, rows);
        else if (b->kind == FLYX_BLOCK_DSPINE) dspine_context(s, b->dspine, rows);
        else draft_context(s, b->dflash2, rows);
        break;
      }
      case FLYX_NODE_ACCEPT: accept(s, p->width); break;
      case FLYX_NODE_COMMIT: commit(s); break;
      case FLYX_NODE_GIVEN:
        for (int i = 0; i < n.arg0; ++i) s->drafts[size_t(i)] = forced_draft(s, s->position + 1 + i);
        break;
      default: throw std::runtime_error("cpu_ref: unknown node kind " + std::to_string(n.kind));
    }
    M->timed(node_label(n, b, p->phase), t0);
  }
}

// ---- API ------------------------------------------------------------------------

int probe(flyx_device_info* out) {
  *out = {};
  std::snprintf(out->name, sizeof(out->name), "cpu (float64 reference)");
  out->fp4_inputs = 1;  // round_nvfp4, in float64
  return 0;
}

// Backend options, "K=V,K=V": profile=0|1.
bool read_options(const char* options, RefModel* m) {
  const std::string all = options ? options : "";
  for (size_t at = 0; at < all.size();) {
    size_t end = all.find(',', at);
    if (end == std::string::npos) end = all.size();
    const std::string kv = all.substr(at, end - at);
    at = end + 1;
    if (kv == "profile=1" || kv == "profile=0") {
      m->profile = kv.back() == '1';
    } else if (!kv.empty()) {
      fail("cpu_ref: unknown option \"" + kv + "\" (it takes profile=0|1)");
      return false;
    }
  }
  return true;
}


inline RefModel* as_m(flyx_bmodel* p) { return reinterpret_cast<RefModel*>(p); }
inline const RefModel* as_m(const flyx_bmodel* p) { return reinterpret_cast<const RefModel*>(p); }
inline RefPlan* as_p(flyx_plan* p) { return reinterpret_cast<RefPlan*>(p); }
inline RefSeq* as_s(flyx_seq* p) { return reinterpret_cast<RefSeq*>(p); }
inline const RefSeq* as_s(const flyx_seq* p) { return reinterpret_cast<const RefSeq*>(p); }

int open_model(const flyx_model_desc* desc, const flyx_open_options* opts, flyx_bmodel** out) {
  auto m = std::make_unique<RefModel>();
  if (!read_options(opts ? opts->options : nullptr, m.get())) return -1;
  m->hidden = desc->hidden;
  m->vocab = desc->vocab;
  m->blocks.assign(desc->blocks, desc->blocks + desc->n_blocks);
  m->states.assign(desc->states, desc->states + desc->n_states);
  try {
    // Identify vision linear weight indices when on-the-fly vision quantization is requested.
    const int32_t vq = opts ? opts->vision_quant : FLYX_VISION_QUANT_Q8;
    std::unordered_set<int> vision_lin_w;
    if (vq == FLYX_VISION_QUANT_Q8 || vq == FLYX_VISION_QUANT_NVFP4) {
      auto mark = [&](int idx) {
        if (idx >= 0) vision_lin_w.insert(idx);
      };
      for (const flyx_block& b : m->blocks) {
        if (b.kind == FLYX_BLOCK_VIT_ATTN) {
          mark(b.vit_attn.qkv);
          mark(b.vit_attn.q);
          mark(b.vit_attn.k);
          mark(b.vit_attn.v);
          mark(b.vit_attn.o);
        } else if (b.kind == FLYX_BLOCK_VIT_MLP) {
          mark(b.vit_mlp.fc1);
          mark(b.vit_mlp.gate);
          mark(b.vit_mlp.fc2);
        } else if (b.kind == FLYX_BLOCK_VISION_MERGE) {
          mark(b.vision_merge.fc1);
          mark(b.vision_merge.fc2);
        }
      }
    }
    for (int i = 0; i < desc->n_weights; ++i) {
      const flyx_weight& w = desc->weights[i];
      if (!w.t) throw std::runtime_error("weight " + std::to_string(i) + " has no tensor");
      const int dt = w.t->dtype;
      // GGML's NVFP4 is a block type in the file and an NVFP4 weight in what it takes.
      const bool ggml_fp4 = dt == FLYX_DT_GGML_NVFP4;
      const bool blocks = flyx_block_values(dt) != 0 && !ggml_fp4, nvfp4 = dt == FLYX_DT_NVFP4 || ggml_fp4;
      const bool floats = dt == FLYX_DT_BF16 || dt == FLYX_DT_F16 || dt == FLYX_DT_F32;
      if (!blocks && !nvfp4 && !floats) {
        m->W.push_back(w.t);
        continue;
      }
      if (nvfp4 && !ggml_fp4 && (!w.scale || !w.scale2))
        throw std::runtime_error(std::string(w.t->name) + ": NVFP4 without its scales");
      auto data = std::make_unique<std::vector<double>>(nvfp4 && !ggml_fp4
                                                            ? flyx::ops::dequant_nvfp4(w.t, w.scale, w.scale2)
                                                            : flyx::ops::to_vec(w.t));
      bool q_q8 = false, q_fp4 = false;
      if (floats && w.t->ndim == 2 && vision_lin_w.count(i)) {
        const int N = int(w.t->shape[0]), K = int(w.t->shape[1]);
        std::vector<float> row_f(static_cast<size_t>(K));
        if (vq == FLYX_VISION_QUANT_Q8) {
          const int K_pad = (K + 31) / 32 * 32;
          std::vector<uint8_t> payload(static_cast<size_t>(K_pad));
          std::vector<uint16_t> scales(static_cast<size_t>(K_pad / 32));
          for (int r = 0; r < N; ++r) {
            for (int k = 0; k < K; ++k) row_f[size_t(k)] = static_cast<float>((*data)[size_t(r) * K + k]);
            flyx::ops::quantize_row_q8_0(row_f.data(), K, K_pad, payload.data(), scales.data());
            for (int k = 0; k < K; ++k)
              (*data)[size_t(r) * K + k] =
                  flyx::ops::ggml::half(scales[size_t(k / 32)]) * double(static_cast<int8_t>(payload[size_t(k)]));
          }
          q_q8 = true;
        } else if (vq == FLYX_VISION_QUANT_NVFP4) {
          const int K_pad = (K + 63) / 64 * 64;
          std::vector<uint8_t> q(static_cast<size_t>(K_pad / 2));
          std::vector<uint8_t> s(static_cast<size_t>(K_pad / 16));
          for (int r = 0; r < N; ++r) {
            for (int k = 0; k < K; ++k) row_f[size_t(k)] = static_cast<float>((*data)[size_t(r) * K + k]);
            float rs = 1.f;
            flyx::ops::quantize_row_nvfp4(row_f.data(), K, K_pad, q.data(), s.data(), &rs);
            for (int k = 0; k < K; ++k) {
              const uint8_t byte = q[size_t(k / 2)];
              const uint8_t code = (k & 1) ? (byte >> 4) : (byte & 0xF);
              (*data)[size_t(r) * K + k] =
                  flyx::ops::e2m1_to_double(code) * flyx::ops::e4m3_to_double(s[size_t(k / 16)]) * double(rs);
            }
          }
          q_fp4 = true;
        }
      }
      auto v = std::make_unique<flyx_tensor_view>(*w.t);
      v->dtype = FLYX_DT_F64;
      v->data = data->data();
      v->nbytes = data->size() * sizeof(double);
      m->W.push_back(v.get());
      if (blocks) m->a16h.insert(v.get());
      if (nvfp4 || q_fp4) m->a16.insert(v.get());
      if ((blocks && w.activations == FLYX_ACT_Q8) || q_q8) m->q8.insert(v.get());
      if (nvfp4 && w.activations == FLYX_ACT_NVFP4) m->a4.insert(v.get());
      m->owned.push_back(std::move(data));
      m->owned_views.push_back(std::move(v));
    }
    for (int i = 0; i < desc->n_blocks; ++i) {
      const flyx_block& b = desc->blocks[i];
      const int k = b.kind;
      // DV's blocks (ABI 42) are taken and never run: such a model decodes here as without DV (lower).
      if (k < FLYX_BLOCK_EMBED || k > FLYX_BLOCK_DSPINE_LAYER)
        throw std::runtime_error("block kind " + std::to_string(k) + ": not implemented by cpu_ref");
      if (k == FLYX_BLOCK_VISION_STEM && b.vision_stem.cls_token >= 0)
        throw std::runtime_error("vision_stem: cls_token is not yet implemented by cpu_ref");
      if (k == FLYX_BLOCK_DFLASH2 || k == FLYX_BLOCK_DSPINE) {
        if (m->dflash >= 0) throw std::runtime_error("more than one block drafter");
        m->dflash = i;
      }
      if (k == FLYX_BLOCK_ATTN || k == FLYX_BLOCK_MLP) {  // a target's own adapters (ABI 43)
        const flyx_lora* l = k == FLYX_BLOCK_ATTN ? b.attn.lora : b.mlp.lora;
        const int n = k == FLYX_BLOCK_ATTN ? 4 : 3;
        for (int j = 0; j < n; ++j)
          if (l[j].rank != l[0].rank || l[j].rank < 0)
            throw std::runtime_error("block " + std::to_string(i) + ": adapters on some of its linears, or of ranks "
                                     "apart (all or none, one rank)");
        if (l[0].rank && k == FLYX_BLOCK_ATTN && b.attn.gate != FLYX_GATE_NONE)
          throw std::runtime_error("block " + std::to_string(i) + ": adapters on attention with an output gate");
      }
    }
    for (int i = 0; i < desc->n_blocks; ++i) {  // an MTP layer's draft steps run its blocks without adapters
      if (desc->blocks[i].kind != FLYX_BLOCK_MTP) continue;
      const flyx_mtp_block& t = desc->blocks[i].mtp;
      if (desc->blocks[t.attn].attn.lora[0].rank || desc->blocks[t.mlp].mlp.lora[0].rank)
        throw std::runtime_error("an MTP layer with adapters");
    }
    {
      std::vector<char> drafter(size_t(desc->n_blocks), 0);  // the MTP layer's blocks add to no residual of a pass
      for (int i = 0; i < desc->n_blocks; ++i)
        if (desc->blocks[i].kind == FLYX_BLOCK_MTP) {
          drafter[size_t(desc->blocks[i].mtp.attn)] = 1;
          drafter[size_t(desc->blocks[i].mtp.mlp)] = 1;
        }
      bool ok = !m->a4.empty();
      // A block with adapters (ABI 43) adds them in float32 to float32 rows, as the CUDA backend (residual16).
      auto adapters = [](const flyx_lora* l, int n) {
        for (int i = 0; i < n; ++i)
          if (l[i].rank > 0) return true;
        return false;
      };
      for (int i = 0; i < desc->n_blocks && ok; ++i) {
        if (drafter[size_t(i)]) continue;
        const flyx_block& b = desc->blocks[i];
        if (b.kind == FLYX_BLOCK_EMBED)  // the file's table (W holds float64 copies)
          ok = desc->weights[b.embed.table].t && desc->weights[b.embed.table].t->dtype == FLYX_DT_BF16;
        else if (b.kind == FLYX_BLOCK_GDN) ok = m->a4.count(m->w(b.gdn.out)) > 0;
        else if (b.kind == FLYX_BLOCK_ATTN) ok = m->a4.count(m->w(b.attn.o)) > 0 && !adapters(b.attn.lora, 4);
        else if (b.kind == FLYX_BLOCK_MLP) ok = m->a4.count(m->w(b.mlp.down)) > 0 && !adapters(b.mlp.lora, 3);
      }
      m->res16 = ok;
      if (opts && opts->verbose)
        std::fprintf(stderr, "cpu_ref: prefill passes of more than 16 rows keep the residual in %s\n",
                     m->res16 ? "bf16" : "float32");
    }
    for (const auto& st : m->states) {
      if (st.kind < FLYX_STATE_KV || st.kind > FLYX_STATE_CANVAS)  // DV's canvas: never kept (no DV rounds here)
        throw std::runtime_error("state kind " + std::to_string(st.kind) + ": not implemented by cpu_ref");
      if (st.kind == FLYX_STATE_RING) m->ring_width = st.width;
    }
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  *out = reinterpret_cast<flyx_bmodel*>(m.release());
  return 0;
}

void close_model(flyx_bmodel* m) { delete as_m(m); }

int lower(flyx_bmodel* bm, const flyx_program* prog, flyx_plan** out) {
  if (prog->n_checkpoints > 0) return fail("adaptive rounds: not in this backend");
  RefModel* m = as_m(bm);
  if (prog->phase == FLYX_PHASE_VISION) {
    if (prog->max_rows < 1 || prog->n_nodes < 1 || !prog->nodes) return fail("vision program: empty");
    for (int i = 0; i < prog->n_nodes; ++i) {
      const flyx_node& n = prog->nodes[i];
      if (n.kind != FLYX_NODE_RUN || n.block < 0 || n.block >= int(m->blocks.size()))
        return fail("vision program: invalid node");
      const int k = m->block(n.block).kind;
      if (k < FLYX_BLOCK_VISION_STEM || k > FLYX_BLOCK_VISION_MERGE)
        return fail("vision program: non-vision block kind " + std::to_string(k));
    }
    if (m->block(prog->nodes[0].block).kind != FLYX_BLOCK_VISION_STEM)
      return fail("vision program: must begin with a vision_stem block");
    auto p = std::make_unique<RefPlan>();
    p->m = m;
    p->phase = FLYX_PHASE_VISION;
    p->width = 0;
    p->max_rows = prog->max_rows;
    p->nodes.assign(prog->nodes, prog->nodes + prog->n_nodes);
    *out = reinterpret_cast<flyx_plan*>(p.release());
    return 0;
  }
  int most = 0;  // drafts the round may verify: up to a full round of given ones, else what its drafter proposes
  for (int i = 0; i < prog->n_nodes; ++i) {
    const flyx_node& n = prog->nodes[i];
    const bool none = n.kind == FLYX_NODE_ACCEPT || n.kind == FLYX_NODE_COMMIT || n.kind == FLYX_NODE_GIVEN;
    if (!none && (n.block < 0 || n.block >= int(m->blocks.size()))) return fail("node names block " + std::to_string(n.block));
    // DV's rounds are trees (a tree round is not in this backend either), and its canvas pass is not: a model with
    // DV decodes here as without it (plain decoding and lookup rounds, plan::build), its prefill's CONTEXT (the canvas
    // starting over) a step with nothing to do.  Its rounds are checked against this backend's plain decoding on the
    // CUDA backend (docs/design/dv.md).
    if (!none && m->block(n.block).kind == FLYX_BLOCK_DV &&
        (n.kind == FLYX_NODE_DRAFT || prog->phase == FLYX_PHASE_ROUND))
      return fail("DV's rounds (its tree and its canvas pass): not in this backend");
    if (n.kind == FLYX_NODE_GIVEN) most = FLYX_MAX_ROWS - 1;
    if ((n.kind == FLYX_NODE_GIVEN || n.kind == FLYX_NODE_DRAFT) && n.arg1 != 0)
      return fail("tree rounds: not in this backend");
    if (n.kind == FLYX_NODE_DRAFT) {
      const flyx_block& b = m->block(n.block);
      most = b.kind == FLYX_BLOCK_DFLASH2  ? std::max(b.dflash2.block, FLYX_MAX_ROWS) - 1
             : b.kind == FLYX_BLOCK_DSPINE ? b.dspine.block - 1  // the rows it was made for: no wider block
             : b.kind == FLYX_BLOCK_MTP    ? FLYX_MAX_ROWS - 1
                                           : 0;
    }
  }
  if (prog->width > most) return fail("round width exceeds what the drafter can propose");
  auto p = std::make_unique<RefPlan>();
  p->m = m;
  p->phase = prog->phase;
  p->width = prog->width;
  p->max_rows = prog->max_rows;
  p->nodes.assign(prog->nodes, prog->nodes + prog->n_nodes);
  *out = reinterpret_cast<flyx_plan*>(p.release());
  return 0;
}

void plan_free(flyx_plan* p) { delete as_p(p); }

int seq_new(flyx_bmodel* bm, int32_t max_ctx, flyx_seq** out) {
  RefModel* m = as_m(bm);
  try {
    auto s = std::make_unique<RefSeq>();
    s->m = m;
    s->max_ctx = max_ctx;
    s->K.assign(m->states.size(), {});
    s->V.assign(m->states.size(), {});
    for (size_t i = 0; i < m->states.size(); ++i) {
      const auto& st = m->states[i];
      if (st.kind == FLYX_STATE_KV) {
        s->K[i].assign(size_t(st.layers), Vec(size_t(max_ctx) * st.n_kv * st.head_dim, 0.0));
        s->V[i].assign(size_t(st.layers), Vec(size_t(max_ctx) * st.n_kv * st.head_dim, 0.0));
      }
    }
    if (m->ring_width) s->hidden_at.assign((size_t(max_ctx) + FLYX_MAX_ROWS) * m->ring_width, 0.0);
    reset_state(s.get());
    ensure_slot_buffers(s.get());
    *out = reinterpret_cast<flyx_seq*>(s.release());
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

void seq_free(flyx_seq* s) { delete as_s(s); }

int prefill_ex(flyx_plan* bp, flyx_seq* bs, const flyx_prefill_input* in, const flyx_gen_params* sample) {
  RefPlan* p = as_p(bp);
  RefSeq* s = as_s(bs);
  if (p->phase != FLYX_PHASE_PREFILL) return fail("not a prefill plan");
  if (!in || !in->tokens) return fail("prefill_ex: null input");
  const int32_t n = in->n;
  if (n <= 0 || n > p->max_rows) return fail("prefill chunk size out of range");
  if (!flyx_room(s->position + n, s->max_ctx)) return fail("context is full");
  if (in->n_embeds < 0 || (in->n_embeds > 0 && (!in->embeds || !in->embed_rows)))
    return fail("prefill_ex: invalid external embeddings");
  try {
    s->tok.assign(in->tokens, in->tokens + n);
    s->embed_override.clear();
    for (int32_t i = 0; i < in->n_embeds; ++i) {
      const int32_t r = in->embed_rows[i];
      if (r < 0 || r >= n) return fail("prefill_ex: embed_rows index out of range");
      s->embed_override[r] = in->embeds + size_t(i) * size_t(s->m->hidden);
    }
    s->pos_3d = in->pos_3d;
    s->rows = n;
    s->start = s->position;
    s->sample_last = sample != nullptr;
    if (sample) s->gen = *sample;
    s->noop = false;
    // A prefill changes the current state in place: older rounds stay foldable,
    // but the current slot no longer starts a recorded round.
    s->slot[cur_slot(s)].replay_valid = false;
    run_nodes(p, s);
    if (in->pos_3d) {
      s->prev_pos3d[0] = in->pos_3d[size_t(n - 1) * 3 + 0];
      s->prev_pos3d[1] = in->pos_3d[size_t(n - 1) * 3 + 1];
      s->prev_pos3d[2] = in->pos_3d[size_t(n - 1) * 3 + 2];
      s->prev_pos3d_at = s->position + n - 1;
    } else {
      s->prev_pos3d_at = -1;
    }
    s->embed_override.clear();
    s->pos_3d = nullptr;
    if (in->set_rope_delta) s->rope_delta = in->rope_delta;
    s->position += n;
    s->stopped = false;
    if (sample) {
      s->anchor = s->choice[n - 1];
      count_token(s, s->anchor);
      flyx_round_result r{};
      r.seq_no = ++s->seq_no;
      r.n_tokens = 1;
      r.tokens[0] = s->anchor;
      r.position = s->position;
      s->ring.push_back(r);
    }
  } catch (const std::exception& e) {
    s->embed_override.clear();
    s->pos_3d = nullptr;
    return fail(e.what());
  }
  return 0;
}

int prefill(flyx_plan* bp, flyx_seq* bs, const int32_t* tokens, int32_t n, const flyx_gen_params* sample) {
  flyx_prefill_input in{};
  in.tokens = tokens;
  in.n = n;
  return prefill_ex(bp, bs, &in, sample);
}

// ---- Stateless Vision Encoder (ABI 44) --------------------------------------------------

struct VisionState {
  int rows = 0;
  int width = 0;
  Vec x;
  const int32_t* grid_thw = nullptr;
  int n_grids = 0;
  const int32_t* cu_seqlens = nullptr;
  int n_segs = 0;
};

void run_vision_stem(const RefModel* M, const flyx_vision_stem_block& B, const flyx_vision_input* in,
                     VisionState* vs) {
  if (in->patch_dim != B.in_dim)
    throw std::runtime_error("vision_stem: patch_dim mismatch (" + std::to_string(in->patch_dim) + " vs " +
                             std::to_string(B.in_dim) + ")");
  const int N = in->n_patches, H = B.hidden;
  Vec inp(size_t(N) * size_t(B.in_dim));
  for (size_t i = 0; i < inp.size(); ++i) inp[i] = double(in->patches[i]);
  vs->rows = N;
  vs->width = H;
  vs->x.assign(size_t(N) * size_t(H), 0.0);
  if (B.patch_w2 >= 0) {
    const int half = B.in_dim / 2;
    Vec p0(size_t(N) * size_t(half)), p1(size_t(N) * size_t(half)), y1(size_t(N) * size_t(H));
    for (int r = 0; r < N; ++r) {
      std::copy_n(&inp[size_t(r) * B.in_dim], half, &p0[size_t(r) * half]);
      std::copy_n(&inp[size_t(r) * B.in_dim + half], half, &p1[size_t(r) * half]);
    }
    lin(M, B.patch_w, p0.data(), N, half, H, vs->x.data(), N);
    lin(M, B.patch_w2, p1.data(), N, half, H, y1.data(), N);
    for (size_t i = 0; i < vs->x.size(); ++i) vs->x[i] += y1[i];
  } else {
    lin(M, B.patch_w, inp.data(), N, B.in_dim, H, vs->x.data(), N);
  }
  if (B.patch_b >= 0) add_bias(vs->x.data(), H, M->vec(B.patch_b).data(), N, H);
  if (B.pos_mode == FLYX_VPOS_ADD_FIXED && B.pos_w >= 0) {
    const Vec& pos = M->vec(B.pos_w);
    const int pos_rows = int(pos.size()) / H;
    if (pos_rows <= 0) throw std::runtime_error("vision_stem: pos_w has fewer than H elements");
    for (int r = 0; r < N; ++r) {
      const size_t idx = size_t(r % pos_rows) * H;
      for (int c = 0; c < H; ++c) vs->x[size_t(r) * H + c] += pos[idx + c];
    }
  } else if (B.pos_mode == FLYX_VPOS_BILINEAR_2D && B.pos_w >= 0) {
    const Vec& pos = M->vec(B.pos_w);
    if (B.pos_grid_h <= 0 || B.pos_grid_w <= 0 ||
        size_t(B.pos_grid_h) * size_t(B.pos_grid_w) * size_t(H) > pos.size())
      throw std::runtime_error("vision_stem: pos_grid_h * pos_grid_w exceeds pos_w rows");
    vision_pos_embed_bilinear_add(vs->x.data(), pos.data(), B.pos_grid_h, B.pos_grid_w, H,
                                  vs->grid_thw, vs->n_grids, B.merge_size);
  }
  if (B.pre_norm.w >= 0 || B.pre_norm.b >= 0) {
    Vec tmp(static_cast<size_t>(H));
    for (int r = 0; r < N; ++r) {
      vnorm(M, B.pre_norm, &vs->x[size_t(r) * H], H, tmp.data());
      std::copy(tmp.begin(), tmp.end(), &vs->x[size_t(r) * H]);
    }
  }
}

void run_vit_attn(const RefModel* M, const flyx_vit_attn_block& B, VisionState* vs) {
  const int N = vs->rows, H = B.hidden, nh = B.n_heads, hd = B.head_dim;
  const int proj_w = nh * hd;
  Vec h(size_t(N) * H);
  for (int r = 0; r < N; ++r) vnorm(M, B.norm, &vs->x[size_t(r) * H], H, &h[size_t(r) * H]);
  Vec Q(size_t(N) * proj_w), K(size_t(N) * proj_w), V(size_t(N) * proj_w);
  if (B.qkv >= 0) {
    Vec qkv(size_t(N) * 3 * proj_w);
    lin(M, B.qkv, h.data(), N, H, 3 * proj_w, qkv.data(), N);
    if (B.qkv_b >= 0) add_bias(qkv.data(), 3 * proj_w, M->vec(B.qkv_b).data(), N, 3 * proj_w);
    for (int r = 0; r < N; ++r) {
      const double* row = &qkv[size_t(r) * 3 * proj_w];
      std::copy_n(row, proj_w, &Q[size_t(r) * proj_w]);
      std::copy_n(row + proj_w, proj_w, &K[size_t(r) * proj_w]);
      std::copy_n(row + 2 * proj_w, proj_w, &V[size_t(r) * proj_w]);
    }
  } else {
    lin(M, B.q, h.data(), N, H, proj_w, Q.data(), N);
    lin(M, B.k, h.data(), N, H, proj_w, K.data(), N);
    lin(M, B.v, h.data(), N, H, proj_w, V.data(), N);
    if (B.q_b >= 0) add_bias(Q.data(), proj_w, M->vec(B.q_b).data(), N, proj_w);
    if (B.k_b >= 0) add_bias(K.data(), proj_w, M->vec(B.k_b).data(), N, proj_w);
    if (B.v_b >= 0) add_bias(V.data(), proj_w, M->vec(B.v_b).data(), N, proj_w);
  }
  if (B.q_norm.w >= 0 || B.k_norm.w >= 0) {
    Vec tmp(static_cast<size_t>(hd));
    for (int r = 0; r < N; ++r) {
      for (int hh = 0; hh < nh; ++hh) {
        if (B.q_norm.w >= 0) {
          vnorm(M, B.q_norm, &Q[(size_t(r) * nh + hh) * hd], hd, tmp.data());
          std::copy(tmp.begin(), tmp.end(), &Q[(size_t(r) * nh + hh) * hd]);
        }
        if (B.k_norm.w >= 0) {
          vnorm(M, B.k_norm, &K[(size_t(r) * nh + hh) * hd], hd, tmp.data());
          std::copy(tmp.begin(), tmp.end(), &K[(size_t(r) * nh + hh) * hd]);
        }
      }
    }
  }
  if (B.rope_mode == FLYX_VROPE_AXIAL_2D) {
    std::vector<int32_t> ch, cw;
    vision_spatial_coords(vs->grid_thw, vs->n_grids, B.merge_size, &ch, &cw);
    if (int(ch.size()) != N) throw std::runtime_error("vit_attn: grid_thw patch count mismatch");
    for (int r = 0; r < N; ++r)
      for (int hh = 0; hh < nh; ++hh) {
        rope_2d_axial(&Q[(size_t(r) * nh + hh) * hd], ch[size_t(r)], cw[size_t(r)], B.theta, hd);
        rope_2d_axial(&K[(size_t(r) * nh + hh) * hd], ch[size_t(r)], cw[size_t(r)], B.theta, hd);
      }
  }
  Vec out(size_t(N) * proj_w, 0.0);
  const double scale = 1.0 / std::sqrt(double(hd));
  const int n_segs = vs->n_segs > 0 ? vs->n_segs : 1;
  const int32_t fallback[2] = {0, N};
  if (vs->n_segs > 0 && !vs->cu_seqlens) throw std::runtime_error("vit_attn: n_segs > 0 with null cu_seqlens");
  const int32_t* cu = vs->n_segs > 0 ? vs->cu_seqlens : fallback;
  for (int i = 0; i <= n_segs; ++i) {
    if (cu[i] < 0 || cu[i] > N || (i > 0 && cu[i] < cu[i - 1]))
      throw std::runtime_error("vit_attn: cu_seqlens out of range or non-monotonic");
  }
  for (int seg = 0; seg < n_segs; ++seg) {
    const int s0 = cu[seg], s1 = cu[seg + 1];
    for (int r = s0; r < s1; ++r)
      for (int hh = 0; hh < nh; ++hh)
        attend_range(&Q[(size_t(r) * nh + hh) * hd], K.data(), V.data(), s0, s1, nh, hd, hh, scale,
                     &out[(size_t(r) * nh + hh) * hd]);
  }
  Vec y(size_t(N) * H);
  lin(M, B.o, out.data(), N, proj_w, H, y.data(), N);
  if (B.o_b >= 0) add_bias(y.data(), H, M->vec(B.o_b).data(), N, H);
  const double* ls = B.layer_scale >= 0 ? M->vec(B.layer_scale).data() : nullptr;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < H; ++c)
      vs->x[size_t(r) * H + c] += ls ? y[size_t(r) * H + c] * ls[c] : y[size_t(r) * H + c];
}

void run_vit_mlp(const RefModel* M, const flyx_vit_mlp_block& B, VisionState* vs) {
  const int N = vs->rows, H = B.hidden, I = B.inter;
  Vec h(size_t(N) * H);
  for (int r = 0; r < N; ++r) vnorm(M, B.norm, &vs->x[size_t(r) * H], H, &h[size_t(r) * H]);
  Vec mid(size_t(N) * I);
  lin(M, B.fc1, h.data(), N, H, I, mid.data(), N);
  if (B.fc1_b >= 0) add_bias(mid.data(), I, M->vec(B.fc1_b).data(), N, I);
  if (B.gated) {
    Vec g(size_t(N) * I);
    lin(M, B.gate, h.data(), N, H, I, g.data(), N);
    if (B.gate_b >= 0) add_bias(g.data(), I, M->vec(B.gate_b).data(), N, I);
    for (size_t i = 0; i < mid.size(); ++i) mid[i] = vact(g[i], B.act) * mid[i];
  } else {
    for (double& v : mid) v = vact(v, B.act);
  }
  Vec y(size_t(N) * H);
  lin(M, B.fc2, mid.data(), N, I, H, y.data(), N);
  if (B.fc2_b >= 0) add_bias(y.data(), H, M->vec(B.fc2_b).data(), N, H);
  const double* ls = B.layer_scale >= 0 ? M->vec(B.layer_scale).data() : nullptr;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < H; ++c)
      vs->x[size_t(r) * H + c] += ls ? y[size_t(r) * H + c] * ls[c] : y[size_t(r) * H + c];
}

void run_vision_merge(const RefModel* M, const flyx_vision_merge_block& B, VisionState* vs) {
  const int N = vs->rows, H = B.in_hidden;
  const int m = std::max(1, B.merge_size);
  const int group = m * m;
  if (!B.post_norm && (B.norm.w >= 0 || B.norm.b >= 0)) {
    Vec tmp(static_cast<size_t>(H));
    for (int r = 0; r < N; ++r) {
      vnorm(M, B.norm, &vs->x[size_t(r) * H], H, tmp.data());
      std::copy(tmp.begin(), tmp.end(), &vs->x[size_t(r) * H]);
    }
  }
  int out_rows = N;
  int merged_dim = H;
  Vec cur;
  if (B.merge_mode == FLYX_VMERGE_SPATIAL_CONCAT) {
    if (N % group != 0) throw std::runtime_error("vision_merge: patch count not divisible by merge_size^2");
    out_rows = N / group;
    merged_dim = H * group;
    cur = std::move(vs->x);  // already contiguous in spatial-merge order!
  } else if (B.merge_mode == FLYX_VMERGE_AVG_POOL_2D) {
    if (N % group != 0) throw std::runtime_error("vision_merge: patch count not divisible by merge_size^2");
    out_rows = N / group;
    merged_dim = H;
    cur.resize(size_t(out_rows) * merged_dim);
    vision_avg_pool_blocks(vs->x.data(), out_rows, group, H, cur.data());
  } else {
    cur = std::move(vs->x);
  }
  if (B.post_norm && (B.norm.w >= 0 || B.norm.b >= 0)) {
    Vec tmp(static_cast<size_t>(merged_dim));
    for (int r = 0; r < out_rows; ++r) {
      vnorm(M, B.norm, &cur[size_t(r) * merged_dim], merged_dim, tmp.data());
      std::copy(tmp.begin(), tmp.end(), &cur[size_t(r) * merged_dim]);
    }
  }
  const bool has_fc1 = B.fc1 >= 0, has_fc2 = B.fc2 >= 0;
  Vec proj;
  if (has_fc1 && has_fc2) {
    Vec mid(size_t(out_rows) * B.fc1_out);
    lin(M, B.fc1, cur.data(), out_rows, merged_dim, B.fc1_out, mid.data(), out_rows);
    if (B.fc1_b >= 0) add_bias(mid.data(), B.fc1_out, M->vec(B.fc1_b).data(), out_rows, B.fc1_out);
    for (double& v : mid) v = vact(v, B.act);
    proj.resize(size_t(out_rows) * B.out_hidden);
    lin(M, B.fc2, mid.data(), out_rows, B.fc1_out, B.out_hidden, proj.data(), out_rows);
    if (B.fc2_b >= 0) add_bias(proj.data(), B.out_hidden, M->vec(B.fc2_b).data(), out_rows, B.out_hidden);
  } else if (has_fc2) {
    proj.resize(size_t(out_rows) * B.out_hidden);
    lin(M, B.fc2, cur.data(), out_rows, merged_dim, B.out_hidden, proj.data(), out_rows);
    if (B.fc2_b >= 0) add_bias(proj.data(), B.out_hidden, M->vec(B.fc2_b).data(), out_rows, B.out_hidden);
  } else if (has_fc1) {
    proj.resize(size_t(out_rows) * B.out_hidden);
    lin(M, B.fc1, cur.data(), out_rows, merged_dim, B.out_hidden, proj.data(), out_rows);
    if (B.fc1_b >= 0) add_bias(proj.data(), B.out_hidden, M->vec(B.fc1_b).data(), out_rows, B.out_hidden);
  } else {
    proj = std::move(cur);
  }
  vs->rows = out_rows;
  vs->width = B.out_hidden;
  vs->x = std::move(proj);
}

int encode_vision(flyx_plan* bp, const flyx_vision_input* in, float* out_embeds, int32_t out_cap_rows,
                  int32_t* out_rows) {
  RefPlan* p = as_p(bp);
  if (!p || p->phase != FLYX_PHASE_VISION) return fail("encode_vision: not a vision plan");
  if (!in || !in->patches || in->n_patches <= 0 || in->patch_dim <= 0) return fail("encode_vision: invalid input");
  if (!in->grid_thw || in->n_grids <= 0) return fail("encode_vision: invalid grid_thw");
  if (!out_embeds || out_cap_rows <= 0) return fail("encode_vision: invalid output buffer");
  try {
    int merge_size = 1;
    for (const flyx_node& n : p->nodes) {
      const flyx_block& b = p->m->block(n.block);
      if (b.kind == FLYX_BLOCK_VISION_STEM) merge_size = std::max(merge_size, std::max(1, b.vision_stem.merge_size));
      else if (b.kind == FLYX_BLOCK_VIT_ATTN) merge_size = std::max(merge_size, std::max(1, b.vit_attn.merge_size));
    }
    size_t total_patches = 0;
    for (int g = 0; g < in->n_grids; ++g) {
      const int T = in->grid_thw[g * 3 + 0];
      const int H = in->grid_thw[g * 3 + 1];
      const int W = in->grid_thw[g * 3 + 2];
      if (T <= 0 || H <= 0 || W <= 0 || (H % merge_size) != 0 || (W % merge_size) != 0)
        throw std::runtime_error("encode_vision: grid_thw dimensions invalid or not divisible by merge_size");
      total_patches += size_t(T) * size_t(H) * size_t(W);
    }
    if (total_patches != size_t(in->n_patches))
      throw std::runtime_error("encode_vision: grid_thw total patches != n_patches");
    VisionState vs{};
    vs.grid_thw = in->grid_thw;
    vs.n_grids = in->n_grids;
    vs.cu_seqlens = in->cu_seqlens;
    vs.n_segs = in->n_segs;
    for (const flyx_node& n : p->nodes) {
      const flyx_block& b = p->m->block(n.block);
      switch (b.kind) {
        case FLYX_BLOCK_VISION_STEM: run_vision_stem(p->m, b.vision_stem, in, &vs); break;
        case FLYX_BLOCK_VIT_ATTN: run_vit_attn(p->m, b.vit_attn, &vs); break;
        case FLYX_BLOCK_VIT_MLP: run_vit_mlp(p->m, b.vit_mlp, &vs); break;
        case FLYX_BLOCK_VISION_MERGE: run_vision_merge(p->m, b.vision_merge, &vs); break;
        default: throw std::runtime_error("encode_vision: unsupported block kind " + std::to_string(b.kind));
      }
    }
    if (vs.rows > out_cap_rows) return fail("encode_vision: output buffer too small");
    for (size_t i = 0; i < vs.x.size(); ++i) out_embeds[i] = float(vs.x[i]);
    if (out_rows) *out_rows = vs.rows;
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

int launch(flyx_plan* bp, flyx_seq* bs, const flyx_gen_params* g) {
  RefPlan* p = as_p(bp);
  RefSeq* s = as_s(bs);
  if (p->phase != FLYX_PHASE_ROUND) return fail("not a round plan");
  if (p->width > 0 && (g->accept_min_p > 0.f || g->accept_typical > 0.f))
    return fail("relaxed acceptance: not in this backend");
  s->gen = *g;
  s->noop = false;
  try {
    if (s->stopped) {
      accept(s, p->width);
      return 0;
    }
    s->drafts.assign(p->width, 0);
    run_nodes(p, s);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  return 0;
}

int read_result(flyx_seq* bs, flyx_round_result* out) {
  RefSeq* s = as_s(bs);
  if (s->ring.empty()) return fail("no round in flight");
  *out = s->ring.front();
  s->ring.pop_front();
  return 0;
}

int rewind(flyx_seq* bs, int32_t pos, int32_t anchor) {
  RefSeq* s = as_s(bs);
  if (pos > s->position) return fail("cannot rewind forward");
  s->ring.clear();
  if (pos == 0) {
    reset_state(s);  // a failed snapshot load may have written state before advancing the position
    s->anchor = anchor;
    return 1;
  }
  if (pos == s->position) {
    s->prev_pos3d_at = -1;
    s->anchor = anchor;
    s->stopped = false;
    return 1;
  }
  for (int back = 1; back < FLYX_ROUND_SLOTS && s->round - back >= 0; ++back) {
    const int r = s->round - back, sl = flyx_round_slot(r), to = flyx_round_slot(r + 1);
    Slot& a = s->slot[sl];
    if (!a.replay_valid || pos < a.start_pos || pos > a.start_pos + a.rows) continue;
    fold_into(s, sl, to, pos - a.start_pos);
    a.rows = pos - a.start_pos;  // tokens past pos are discarded; later rewinds must not replay stale rows
    for (int k = 1; k <= back; ++k) s->slot[flyx_round_slot(r + k)].replay_valid = false;
    s->rope_delta = a.rope_delta;
    s->prev_pos3d_at = -1;
    s->round = r + 1;
    s->position = pos;
    s->anchor = anchor;
    s->stopped = false;
    return 1;
  }
  reset_state(s);
  s->anchor = anchor;
  return 0;
}

int checkpoint(flyx_seq* bs, int32_t c) {
  RefSeq* s = as_s(bs);
  if (c < 0 || c >= FLYX_CHECKPOINTS) return fail("bad checkpoint slot");
  const Slot& sl = s->slot[cur_slot(s)];
  s->ck[c] = {true, s->position, s->rope_delta, sl.S, sl.conv, {}};
  if (!s->hidden_at.empty() && s->position > 0) {
    const size_t W = size_t(s->m->ring_width);
    s->ck[c].ring_h.assign(s->hidden_at.begin() + (s->position - 1) * W, s->hidden_at.begin() + s->position * W);
  }
  return s->position;
}

int restore(flyx_seq* bs, int32_t c, int32_t anchor) {
  RefSeq* s = as_s(bs);
  if (c < 0 || c >= FLYX_CHECKPOINTS || !s->ck[c].valid) return -1;
  for (auto& sl : s->slot) sl.replay_valid = false;
  Slot& sl = s->slot[cur_slot(s)];
  sl.S = s->ck[c].S;
  sl.conv = s->ck[c].conv;
  s->position = s->ck[c].pos;
  s->rope_delta = s->ck[c].rope_delta;
  s->prev_pos3d_at = -1;
  if (!s->ck[c].ring_h.empty())
    std::copy(s->ck[c].ring_h.begin(), s->ck[c].ring_h.end(),
              s->hidden_at.begin() + size_t(s->position - 1) * s->m->ring_width);
  s->anchor = anchor;
  s->stopped = false;
  s->ring.clear();
  return s->position;
}

int logits(flyx_seq* bs, int32_t row, float* out) {
  RefSeq* s = as_s(bs);
  const size_t V = size_t(s->m->vocab);
  if (row < 0 || s->head_logits.size() < (size_t(row) + 1) * V) return fail("logits: no such row in the last pass");
  for (size_t i = 0; i < V; ++i) out[i] = float(s->head_logits[size_t(row) * V + i]);
  return 0;
}

int set_oracle(flyx_seq* bs, const int32_t* t, int32_t n) {
  RefSeq* s = as_s(bs);
  s->oracle.assign(t, t + n);
  return 0;
}

const char* last_error() { return g_error.c_str(); }

int memory(flyx_bmodel*, flyx_memory_info* out) {
  *out = {};  // host memory: no device budget to check
  return 0;
}

int seq_bytes(flyx_bmodel*, int32_t, uint64_t* out) {
  *out = 0;
  return 0;
}

// Host memory only: a layout places nothing; every position exact, with FLYX_KV_AUTO too (0066).
int seq_new_in(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq** out) {
  if (!layout || (layout->method != FLYX_KV_READ_ALL && layout->method != FLYX_KV_AUTO))
    return fail("seq_new: a K/V layout this backend does not read");
  return seq_new(m, layout->max_ctx, out);
}

int seq_size(flyx_bmodel*, const flyx_seq_layout* layout, flyx_seq_size* out) {
  *out = {};
  out->device_positions = layout ? layout->max_ctx : 0;
  out->strata_from = out->device_positions;
  return 0;
}

int32_t prefill_align(flyx_bmodel*) { return 1; }  // row by row: any pass start gives the same bits

int device_stats(flyx_device_stats*) { return -1; }  // a CPU: nothing to report

int ready(flyx_seq*) { return 1; }  // every call runs to its end: nothing is ever in flight

int set_reply(flyx_seq* bs, const int32_t* tokens, int32_t n) {
  RefSeq* s = as_s(bs);
  s->counts.assign(size_t(s->m->vocab), 0);
  for (int32_t i = 0; i < n; ++i) {
    if (tokens[i] < 0 || tokens[i] >= s->m->vocab) return fail("set_reply: a token outside the vocabulary");
    ++s->counts[size_t(tokens[i])];
  }
  return 0;
}

// ---- snapshots: a checkpoint in host memory ---------------------------------------------------------
// A header, then in doubles: per K/V state and layer the positions [begin, pos) of K and of V, per state the
// checkpoint's recurrent part (a gated delta net's state and convolution window; nothing for the others), the
// ring's vector before pos.  The header names the layout (the state table), which loading checks.
using flyx::backends::HostCursor;
using flyx::backends::kDeltaVersion;
using flyx::backends::kSnapshotMagic;
using flyx::backends::kSnapshotVersion;
using flyx::backends::pieces_hold;
using flyx::backends::SnapshotHeader;

uint64_t layout_hash(const RefModel* m) {
  uint64_t h = flyx::backends::kFnv1aOffset;  // FNV-1a over the state table, eight bytes a field
  for (const auto& st : m->states)
    for (int32_t v : {st.kind, st.layers, st.n_kv, st.head_dim, st.nk, st.nv, st.dk, st.dv, st.conv, st.width})
      flyx::backends::fnv1a64_mix(&h, uint64_t(uint32_t(v)));
  return h;
}

size_t recurrent_doubles(const flyx_state& st) {
  return st.kind == FLYX_STATE_GDN ? size_t(st.nv) * st.dv * st.dk + size_t(st.conv - 1) * cdim_of(st) : 0;
}

uint64_t snapshot_size(const RefSeq* s, int begin, int pos) {
  size_t n = size_t(s->m->ring_width);
  for (const auto& st : s->m->states) {
    if (st.kind == FLYX_STATE_KV) n += 2 * size_t(st.layers) * size_t(pos - begin) * st.n_kv * st.head_dim;
    n += recurrent_doubles(st);
  }
  return sizeof(SnapshotHeader) + n * sizeof(double);
}

int snapshot_bytes(flyx_seq* bs, int32_t c, int32_t begin, uint64_t* out) {
  RefSeq* s = as_s(bs);
  if (c < 0 || c >= FLYX_CHECKPOINTS || !s->ck[c].valid) return fail("no checkpoint in that slot");
  if (begin < 0 || begin > s->ck[c].pos) return fail("snapshot starts outside the checkpoint");
  *out = snapshot_size(s, begin, s->ck[c].pos);
  return 0;
}

int snapshot_save(flyx_seq* bs, int32_t c, int32_t begin, const flyx_host_pieces* dst, uint64_t bytes) {
  RefSeq* s = as_s(bs);
  uint64_t need = 0;
  if (snapshot_bytes(bs, c, begin, &need) != 0) return -1;
  const Ckpt& ck = s->ck[c];
  if (bytes < need || !pieces_hold(dst, need))
    return fail("a snapshot of this checkpoint takes " + std::to_string(need) + " bytes");
  const SnapshotHeader h{kSnapshotMagic, begin ? kDeltaVersion : kSnapshotVersion, layout_hash(s->m),
                         ck.pos, begin, need, ck.rope_delta, 0};
  HostCursor out{dst};
  out.put(&h, sizeof h);
  auto put = [&](const double* v, size_t n) { out.put(v, n * sizeof(double)); };
  const auto& states = s->m->states;
  for (size_t i = 0; i < states.size(); ++i)
    if (states[i].kind == FLYX_STATE_KV)
      for (int l = 0; l < states[i].layers; ++l) {
        const size_t width = size_t(states[i].n_kv) * states[i].head_dim;
        put(s->K[i][size_t(l)].data() + begin * width, (ck.pos - begin) * width);
        put(s->V[i][size_t(l)].data() + begin * width, (ck.pos - begin) * width);
      }
  for (size_t i = 0; i < states.size(); ++i)
    if (recurrent_doubles(states[i])) {
      put(ck.S[i].data(), ck.S[i].size());
      put(ck.conv[i].data(), ck.conv[i].size());
    }
  const Vec zeros(size_t(s->m->ring_width), 0.0);
  put(ck.ring_h.empty() ? zeros.data() : ck.ring_h.data(), zeros.size());
  return ck.pos;
}

int snapshot_load(flyx_seq* bs, const flyx_host_pieces* src, uint64_t bytes, int32_t anchor) {
  RefSeq* s = as_s(bs);
  SnapshotHeader h{};
  std::string err;
  if (!flyx::backends::validate_snapshot_header(
          src, bytes, layout_hash(s->m), s->max_ctx, s->position,
          [&](int begin, int pos) { return snapshot_size(s, begin, pos); }, &h, &err))
    return fail(err);
  HostCursor in{src, sizeof h};
  auto get = [&](double* v, size_t n) { in.get(v, n * sizeof(double)); };
  const auto& states = s->m->states;
  for (size_t i = 0; i < states.size(); ++i)
    if (states[i].kind == FLYX_STATE_KV)
      for (int l = 0; l < states[i].layers; ++l) {
        const size_t width = size_t(states[i].n_kv) * states[i].head_dim;
        get(s->K[i][size_t(l)].data() + h.begin * width, (h.pos - h.begin) * width);
        get(s->V[i][size_t(l)].data() + h.begin * width, (h.pos - h.begin) * width);
      }
  for (auto& sl : s->slot) sl.replay_valid = false;
  Slot& sl = s->slot[cur_slot(s)];
  for (size_t i = 0; i < states.size(); ++i)
    if (const size_t n = recurrent_doubles(states[i])) {
      sl.S[i].resize(size_t(states[i].nv) * states[i].dv * states[i].dk);
      sl.conv[i].resize(n - sl.S[i].size());
      get(sl.S[i].data(), sl.S[i].size());
      get(sl.conv[i].data(), sl.conv[i].size());
    }
  // Straight to its place: a load a few pieces at a time (ABI 31) leaves what a skipped piece would set as it is.
  if (s->m->ring_width > 0 && h.pos > 0)
    get(s->hidden_at.data() + size_t(h.pos - 1) * s->m->ring_width, size_t(s->m->ring_width));
  s->position = h.pos;
  s->rope_delta = h.rope_delta;
  s->prev_pos3d_at = -1;
  s->anchor = anchor;
  s->stopped = false;
  s->ring.clear();
  return s->position;
}

// The float64 reference runs each sequence's round in turn: what each gets alone, by definition.
int launch_batch(flyx_plan* const* plans, flyx_seq* const* seqs, const flyx_gen_params* const* gens,
                 int32_t n) {
  int rows = 0;
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t j = 0; j < i; ++j)
      if (seqs[j] == seqs[i]) return fail("launch_batch: a sequence twice");
    rows += as_p(plans[i])->width + 1;
  }
  if (n < 1 || rows > FLYX_MAX_ROWS) return fail("launch_batch: 1 to FLYX_MAX_ROWS rows");
  for (int32_t i = 0; i < n; ++i)
    if (launch(plans[i], seqs[i], gens[i]) != 0) return -1;
  return 0;
}

// The host is the device: no memory of its own to give (the snapshot store uses its own).
void* host_alloc(uint64_t) { return nullptr; }
void host_free(void*) {}

const flyx_backend_api kApi = {
    FLYX_BACKEND_ABI_VERSION, sizeof(flyx_backend_api), "cpu_ref",
    probe, open_model, close_model, lower, plan_free, seq_new, seq_free,
    prefill, launch, read_result, rewind, checkpoint, restore, set_oracle, last_error, memory, seq_bytes,
    prefill_align, device_stats, ready, set_reply, snapshot_bytes, snapshot_save, snapshot_load, launch_batch,
    host_alloc, host_free, logits, seq_new_in, seq_size, nullptr,  // trim: nothing kept for speed
    nullptr,                                                       // graph_stats: no graphs
    nullptr,  // prefill_snapshot: prefill_align 1, a pass crosses no boundary it keeps
    encode_vision,
    prefill_ex,
};

}  // namespace

#ifdef FLYX_STATIC_BACKEND
extern "C" const flyx_backend_api* flyx_backend_get_api_cpu_ref(void) {
  return &kApi;
}
#else
extern "C" FLYX_BACKEND_EXPORT const flyx_backend_api* flyx_backend_get_api(void) {
  return &kApi;
}
#endif
