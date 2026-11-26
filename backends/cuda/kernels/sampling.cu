// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Token choice (argmax, sampling as ops/sampling/sampling.h defines it, top-k) and round bookkeeping (accept, ring entries).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "ops/sampling/sampling.h"

namespace flyx::cuda {
namespace {

// ---- head, drafter, bookkeeping -----------------------------------------------------------
constexpr int kArgChunk = 4096;

// grid (rows, chunks); block 256.
__global__ void k_argmax_partial(const float* logits, int ld, int V, float* pv, int* pi, const DevGen* G,
                                 const uint32_t* mask) {
  pdl_trigger();
  if (G && G->temperature > 0.f) return;  // sampling: k_sample chooses
  __shared__ float sv[256];
  __shared__ int si[256];
  const int r = blockIdx.x, b = blockIdx.y;
  const uint32_t* mk = (G && G->mask && r == 0) ? mask : nullptr;
  const int lo = b * kArgChunk, hi = min(V, lo + kArgChunk);
  float bv = -INFINITY;
  int bi = INT32_MAX;
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    if (!ops::allowed(mk, i)) continue;
    const float v = logits[size_t(r) * ld + i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float v = sv[threadIdx.x + s];
      const int i = si[threadIdx.x + s];
      if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) {
        sv[threadIdx.x] = v;
        si[threadIdx.x] = i;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    pv[r * gridDim.y + b] = sv[0];
    pi[r * gridDim.y + b] = si[0];
  }
}

__global__ void k_argmax_final(const float* pv, const int* pi, int chunks, int* out, const DevGen* G) {
  pdl_trigger();
  if (G && G->temperature > 0.f) return;
  const int r = blockIdx.x;
  if (threadIdx.x != 0) return;
  float bv = -INFINITY;
  int bi = INT32_MAX;
  for (int b = 0; b < chunks; ++b) {
    const float v = pv[r * chunks + b];
    const int i = pi[r * chunks + b];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  out[r] = bi == INT32_MAX ? 0 : bi;
}

// ---- sampling (ops/sampling/sampling.h) -------------------------------------------------------------
constexpr int kSampleThreads = 1024;


__device__ int block_sum_int(int v) {
  __shared__ int sh[32];
  __shared__ int res;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    int t = lane < int(blockDim.x >> 5) ? sh[lane] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) res = t;
  }
  __syncthreads();
  return res;
}

// Exclusive prefix sum of one int per thread, in thread order.  *total = sum.
__device__ int block_excl_scan(int v, int* total) {
  __shared__ int sh[32];
  __shared__ int tot;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += n;
  }
  __syncthreads();
  if (lane == 31) sh[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    const int w = lane < int(blockDim.x >> 5) ? sh[lane] : 0;
    int wi = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, wi, o);
      if (lane >= o) wi += n;
    }
    sh[lane] = wi - w;  // exclusive offset of each warp
    if (lane == 31) tot = wi;
  }
  __syncthreads();
  *total = tot;
  return sh[warp] + incl - v;
}

__device__ __forceinline__ uint32_t zkey(float z) { return ops::float_key(__float_as_uint(z)); }

// Keep a precomputed reciprocal on ordinary rows. Subnormal temperatures can have an infinite reciprocal,
// even when the centered logit is zero or just one temperature below the maximum.
__device__ __forceinline__ float scaled_logit(float delta, float temperature, float inverse) {
  return isinf(inverse) ? __fdiv_rn(delta, temperature) : delta * inverse;
}

// The block's best (score, index) pair: larger score, lower index on ties; every thread gets it.
__device__ void block_best(float* best, int* bi) {
  __shared__ float sv[32];
  __shared__ int si[32];
  __shared__ float rv;
  __shared__ int ri;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float b = *best;
  int i = *bi;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (ov > b || (ov == b && oi < i)) {
      b = ov;
      i = oi;
    }
  }
  __syncthreads();
  if (lane == 0) {
    sv[warp] = b;
    si[warp] = i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < int(blockDim.x) / 32; ++w)
      if (sv[w] > b || (sv[w] == b && si[w] < i)) {
        b = sv[w];
        i = si[w];
      }
    rv = b;
    ri = i;
  }
  __syncthreads();
  *best = rv;
  *bi = ri;
}

// One block per row.  Deterministic: every sum runs in a fixed order, so the
// same logits and parameters always give the same token.
__global__ void __launch_bounds__(kSampleThreads) k_sample(const float* logits, int ld, int V, const DevParams* P,
                                                           int base_add, const DevGen* G, const uint32_t* mask,
                                                           float* cz, int* ci, int* out, const TreeRows* tree,
                                                           int tree_from) {
  pdl_trigger();
  const float T = G->temperature;
  if (!(T > 0.f)) return;
  const int r = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const uint32_t* mk = (G->mask && r == 0) ? mask : nullptr;  // filters first, then the mask
  const float* L = logits + size_t(r) * ld;
  float m = -INFINITY;
  for (int v = tid; v < V; v += nt) m = fmaxf(m, L[v]);
  m = block_max(m);
  const float invT = 1.f / T;
  int cnt = 0;
  for (int v = tid; v < V; v += nt) cnt += (scaled_logit(L[v] - m, T, invT) > ops::kSampleZMin) ? 1 : 0;
  int nc = 0;
  int off = block_excl_scan(cnt, &nc);
  float* Z = cz + size_t(r) * V;
  int* I = ci + size_t(r) * V;
  for (int v = tid; v < V; v += nt) {
    const float z = scaled_logit(L[v] - m, T, invT);
    if (z > ops::kSampleZMin) {
      Z[off] = z;
      I[off] = v;
      ++off;
    }
  }
  __syncthreads();
  uint32_t thr = 0;  // keep candidates with zkey(z) >= thr
  const int k = G->top_k;
  if (k > 0 && k < nc) {
    uint32_t lo = 0, hi = 0xFFFFFFFFu;  // largest x with count(key >= x) >= k
    while (lo < hi) {
      const uint32_t mid = uint32_t((uint64_t(lo) + hi + 1) >> 1);
      int c = 0;
      for (int i = tid; i < nc; i += nt) c += zkey(Z[i]) >= mid ? 1 : 0;
      c = block_sum_int(c);
      if (c >= k) lo = mid;
      else hi = mid - 1;
    }
    thr = lo;
  }
  if (G->top_p < 1.f) {
    float ms = 0.f;
    for (int i = tid; i < nc; i += nt)
      if (zkey(Z[i]) >= thr) ms += __expf(Z[i]);
    const float target = G->top_p * block_sum(ms);
    uint32_t lo = thr, hi = 0xFFFFFFFFu;  // largest x with mass(key >= x) >= target
    while (lo < hi) {
      const uint32_t mid = uint32_t((uint64_t(lo) + hi + 1) >> 1);
      float s = 0.f;
      for (int i = tid; i < nc; i += nt)
        if (zkey(Z[i]) >= mid) s += __expf(Z[i]);
      if (block_sum(s) >= target) lo = mid;
      else hi = mid - 1;
    }
    thr = lo;
  }
  // A tree's row: its depth's position (the round's row tree_from + r: a wide round's 16 rows at a time).
  const int64_t pos = int64_t(P->position) + base_add + tree_depth(tree, tree_from + r) - tree_from;
  float best = -INFINITY;
  int bi = INT32_MAX;
  for (int i = tid; i < nc; i += nt) {
    if (zkey(Z[i]) < thr || !ops::allowed(mk, I[i])) continue;
    const float u = ops::sample_uniform(G->seed, pos, I[i]);
    const float s = Z[i] - __logf(-__logf(u));
    if (s > best || (s == best && I[i] < bi)) {
      best = s;
      bi = I[i];
    }
  }
  block_best(&best, &bi);
  if (bi == INT32_MAX && mk) {  // no kept token is allowed: the best allowed one of all
    float allowed_max = -INFINITY;
    for (int v = tid; v < V; v += nt)
      if (ops::allowed(mk, v)) allowed_max = fmaxf(allowed_max, L[v]);
    allowed_max = block_max(allowed_max);
    best = -INFINITY;
    for (int v = tid; v < V; v += nt) {
      if (!ops::allowed(mk, v)) continue;
      const float u = ops::sample_uniform(G->seed, pos, v);
      const float s = scaled_logit(L[v] - allowed_max, T, invT) - __logf(-__logf(u));
      if (s > best || (s == best && v < bi)) {
        best = s;
        bi = v;
      }
    }
    block_best(&best, &bi);
  }
  if (tid == 0) out[r] = bi == INT32_MAX ? 0 : bi;
}

// The best (value, index) of a block's candidates, one a thread (score_better's order), and the thread holding it.
template <int NT>
__device__ __forceinline__ void block_best_of(float* v, int* i, int* who, float (*sv)[3]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int w = threadIdx.x;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, *v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, *i, o), ow = __shfl_xor_sync(0xffffffffu, w, o);
    if (score_better(ov, oi, *v, *i)) *v = ov, *i = oi, w = ow;
  }
  if (lane == 0) sv[warp][0] = *v, sv[warp][1] = __int_as_float(*i), sv[warp][2] = __int_as_float(w);
  __syncthreads();
  float bv = sv[0][0];
  int bi = __float_as_int(sv[0][1]), bw = __float_as_int(sv[0][2]);
#pragma unroll
  for (int q = 1; q < NT / 32; ++q)
    if (score_better(sv[q][0], __float_as_int(sv[q][1]), bv, bi))
      bv = sv[q][0], bi = __float_as_int(sv[q][1]), bw = __float_as_int(sv[q][2]);
  __syncthreads();  // sv is rewritten by the next round
  *v = bv, *i = bi, *who = bw;
}

// topk_rows in parallel: grid (kTopkParts, rows).  A block takes a slice of its row: each thread keeps its sorted
// top-16 in registers (compare-and-swap down the list), then 16 rounds of block argmax give the slice's top-16.
constexpr int kTopkPartThreads = 256;
__global__ void __launch_bounds__(kTopkPartThreads) k_topk_part(const float* logits, int ld, int V, float* pv,
                                                                 int* pi) {
  pdl_trigger();
  __shared__ float sv[kTopkPartThreads / 32][3];
  const int r = blockIdx.y, part = blockIdx.x, span = (V + kTopkParts - 1) / kTopkParts;
  const int lo = part * span, hi = min(V, lo + span);
  float lv[kTopMax];
  int li[kTopMax];
#pragma unroll
  for (int q = 0; q < kTopMax; ++q) lv[q] = -INFINITY, li[q] = INT32_MAX;
  const float* L = logits + size_t(r) * ld;
  for (int i = lo + threadIdx.x; i < hi; i += kTopkPartThreads) {
    float cv = L[i];
    int ci = i;
    if (!score_better(cv, ci, lv[kTopMax - 1], li[kTopMax - 1])) continue;
#pragma unroll
    for (int q = 0; q < kTopMax; ++q)
      if (score_better(cv, ci, lv[q], li[q])) {
        const float tv = lv[q];
        const int ti = li[q];
        lv[q] = cv, li[q] = ci, cv = tv, ci = ti;
      }
  }
  float* ov = pv + (size_t(r) * kTopkParts + part) * kTopMax;
  int* oi = pi + (size_t(r) * kTopkParts + part) * kTopMax;
  for (int out = 0; out < kTopMax; ++out) {
    float bv = lv[0];
    int bi = li[0], who;
    block_best_of<kTopkPartThreads>(&bv, &bi, &who, sv);
    if (threadIdx.x == 0) ov[out] = bv, oi[out] = bi;
    if (threadIdx.x == who) {  // its head is taken: the list moves up
#pragma unroll
      for (int q = 0; q + 1 < kTopMax; ++q) lv[q] = lv[q + 1], li[q] = li[q + 1];
      lv[kTopMax - 1] = -INFINITY, li[kTopMax - 1] = INT32_MAX;
    }
  }
}

// The slices' candidates of a row (block of kTopkParts * 16 threads, one each): k rounds of block argmax,
// plus optional truncated softmax p_max and top-1/top-2 margin across all kTopkParts * 16 candidates.
__global__ void __launch_bounds__(kTopkParts * kTopMax) k_topk_merge(const float* pv, const int* pi, int k, int* idx,
                                                                      float* val, float* prob, float* margin) {
  pdl_trigger();
  __shared__ float sv[kTopkParts * kTopMax / 32][3];
  __shared__ float sh_m1;
  const int r = blockIdx.x, t = threadIdx.x;
  float v = pv[size_t(r) * kTopkParts * kTopMax + t];
  int i = pi[size_t(r) * kTopkParts * kTopMax + t];
  const float v_orig = v;
  const int i_orig = i;
  const int rounds = (prob || margin) ? max(k, 2) : k;
  float m1 = -INFINITY, m2 = -INFINITY;
  for (int out = 0; out < rounds; ++out) {
    float bv = v;
    int bi = i, who;
    block_best_of<kTopkParts * kTopMax>(&bv, &bi, &who, sv);
    if (out == 0) {
      m1 = bv;
      if (t == 0) sh_m1 = bv;
    } else if (out == 1) {
      m2 = bv;
    }
    if (t == 0 && out < k) {
      idx[r * k + out] = bi == INT32_MAX ? out : bi;  // no finite score left: a valid id (argmax_final's 0)
      if (val) val[r * k + out] = bv;
    }
    if (t == who) v = -INFINITY, i = INT32_MAX;
  }
  if (prob) {
    __syncthreads();
    const float z = block_sum((i_orig != INT32_MAX && isfinite(v_orig)) ? expf(v_orig - sh_m1) : 0.f);
    if (t == 0) prob[r] = z > 0.f ? 1.f / z : 0.f;
  }
  if (margin && t == 0) margin[r] = (isfinite(m1) && isfinite(m2)) ? (m1 - m2) : 0.f;
}

// One block per row: per-thread sorted top-16 over a strided slice, then k rounds of block argmax.

__global__ void __launch_bounds__(512) k_topk(const float* logits, int ld, int V, int k, int* idx, float* val,
                                              float* prob, float* margin) {
  pdl_trigger();
  __shared__ float sv[512];
  __shared__ int si[512], st[512];
  __shared__ float sh_m1;
  const int r = blockIdx.x, tid = threadIdx.x;
  const int need = (prob || margin) ? min(V, max(k, 2)) : k;
  float lv[kTopMax];
  int li[kTopMax];
#pragma unroll
  for (int i = 0; i < kTopMax; ++i) {
    lv[i] = -INFINITY;
    li[i] = INT32_MAX;
  }
  for (int i = tid; i < V; i += blockDim.x) {
    const float v = logits[size_t(r) * ld + i];
    if (!score_better(v, i, lv[need - 1], li[need - 1])) continue;
    int p = need - 1;
    while (p > 0 && score_better(v, i, lv[p - 1], li[p - 1])) {
      lv[p] = lv[p - 1];
      li[p] = li[p - 1];
      --p;
    }
    lv[p] = v;
    li[p] = i;
  }
  int head = 0;
  float m1 = -INFINITY, m2 = -INFINITY;
  for (int out = 0; out < need; ++out) {
    sv[tid] = head < need ? lv[head] : -INFINITY;
    si[tid] = head < need ? li[head] : INT32_MAX;
    st[tid] = tid;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && score_better(sv[tid + s], si[tid + s], sv[tid], si[tid])) {
        sv[tid] = sv[tid + s];
        si[tid] = si[tid + s];
        st[tid] = st[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      if (out == 0) {
        m1 = sv[0];
        sh_m1 = sv[0];
      } else if (out == 1) {
        m2 = sv[0];
      }
      if (out < k) {
        idx[r * k + out] = si[0] == INT32_MAX ? out : si[0];  // no finite score left: a valid id
        if (val) val[r * k + out] = sv[0];
      }
    }
    if (tid == st[0]) ++head;
    __syncthreads();
  }
  if (prob) {
    float ez = 0.f;
    for (int i = tid; i < V; i += blockDim.x) ez += expf(logits[size_t(r) * ld + i] - sh_m1);
    const float z = block_sum(ez);
    if (tid == 0) prob[r] = z > 0.f ? 1.f / z : 0.f;
  }
  if (margin && tid == 0) margin[r] = (isfinite(m1) && isfinite(m2)) ? (m1 - m2) : 0.f;
}

// ---- penalties (ops/sampling/sampling.h) -------------------------------------------------------
// grid: rows; block 256.  The tokens the reply has are spread over the
// threads; the drafts before the row that the reply does not have are few
// (at most 63, a tree's depth) and go to thread 0.  The two sets are disjoint, so no logit is
// written twice, and each gets ops::penalty of its whole count in one step.
// The drafts before row r: rows 1 .. r of a chain, a tree's row's ancestors.
__global__ void k_penalize(float* logits, int ld, int V, const DevGen* G, const int* count,
                           const int* seen, const int* n_seen, const int* round_drafts, const TreeRows* tree,
                           int tree_from) {
  pdl_trigger();
  const float presence = G->presence, frequency = G->frequency;
  if (presence == 0.f && frequency == 0.f) return;
  const int r = tree_from + int(blockIdx.x), nd = round_drafts ? tree_depth(tree, r) : 0, ns = *n_seen;
  int drafts[FLYX_MAX_WIDE_ROWS];
  for (int j = 0; j < nd; ++j) drafts[j] = round_drafts[tree_row(tree, r, j + 1) - 1];
  float* L = logits + size_t(blockIdx.x) * ld;
  for (int i = threadIdx.x; i < ns; i += blockDim.x) {
    const int v = seen[i];
    int c = count[v];
    for (int j = 0; j < nd; ++j) c += drafts[j] == v ? 1 : 0;
    L[v] -= ops::penalty(presence, frequency, c);
  }
  if (threadIdx.x != 0) return;
  for (int j = 0; j < nd; ++j) {
    const int d = drafts[j];
    if (d < 0 || d >= V || count[d] > 0) continue;
    int c = 0;
    bool first = true;
    for (int k = 0; k < nd; ++k)
      if (drafts[k] == d) {
        first = first && k >= j;
        ++c;
      }
    if (first) L[d] -= ops::penalty(presence, frequency, c);
  }
}

// A token the reply now has.
__device__ void count_token(const DevGen* G, Reply reply, int t) {
  if (G->presence == 0.f && G->frequency == 0.f) return;
  if (reply.count[t]++ == 0) reply.seen[(*reply.n_seen)++] = t;
}

__device__ flyx_round_result* new_entry(DevParams* P, flyx_round_result* ring) {
  const unsigned long long seq = ++P->seq_no;
  flyx_round_result* e = ring + (seq % kRing);
  memset(e, 0, sizeof(*e));
  e->seq_no = seq;
  return e;
}

__global__ void k_accept(DevParams* P, const DevGen* G, const int* drafts, const int* choice, int width,
                         flyx_round_result* ring, Reply reply, const float* draft_prob, const float* draft_margin) {
  pdl_trigger();
  flyx_round_result* e = new_entry(P, ring);
  if (P->stopped) {
    P->noop = 1;
    e->stopped = 1;
    e->position = P->position;
    return;
  }
  P->noop = 0;
  const int n = ops::accepted_drafts(drafts, choice, width, G->stop, G->n_stop);
  e->n_tokens = n + 1;
  e->n_accepted = n;
  for (int i = 0; i <= n; ++i) {
    e->tokens[i] = choice[i];
    count_token(G, reply, choice[i]);
  }
  e->n_drafts = width;
  for (int i = 0; i < width; ++i) e->drafts[i] = drafts[i], e->parents[i] = i - 1;
  const int nc = width < FLYX_MAX_ROWS ? width : FLYX_MAX_ROWS;
  if (draft_prob)
    for (int i = 0; i < nc; ++i) e->draft_prob[i] = draft_prob[i];
  if (draft_margin)
    for (int i = 0; i < nc; ++i) e->draft_margin[i] = draft_margin[i];
  P->last_n = n;
  P->last_slot = flyx_round_slot(P->round);
  P->last_round = P->round;
  P->start = P->position;
  P->position += n + 1;
  P->anchor = choice[n];
  P->round += 1;
  P->stopped = ops::reply_ends(choice[n], P->position, G->limit, G->max_ctx, G->stop, G->n_stop);
  e->stopped = P->stopped;
  e->position = P->position;
}


// grid: rows; block 1024.  A row's distribution for the relaxed acceptance rules (ops::relaxed_qualifies): with
// z = (logit - max) / T (T = 1 when greedy), stats[3 r ..] = {max, Z = sum exp(z), H = log Z - sum(exp(z) z) / Z}.
// Off (both rules 0): nothing.
__global__ void __launch_bounds__(kSampleThreads) k_row_stats(const float* logits, int ld, int V, const DevGen* G,
                                                              float* stats) {
  pdl_trigger();
  if (!(G->accept_min_p > 0.f) && !(G->accept_typical > 0.f)) return;
  const int r = blockIdx.x;
  const float* L = logits + size_t(r) * ld;
  const float invT = G->temperature > 0.f ? 1.f / G->temperature : 1.f;
  float m = -INFINITY;
  for (int v = threadIdx.x; v < V; v += blockDim.x) m = fmaxf(m, L[v]);
  m = block_max(m);
  float z0 = 0.f, z1 = 0.f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    const float z = scaled_logit(L[v] - m, G->temperature, invT), e = __expf(z);
    z0 += e;
    if (e > 0.f) z1 += e * z;  // a zero-probability token contributes zero, including when z is -infinity
  }
  z0 = block_sum(z0);
  z1 = block_sum(z1);
  if (threadIdx.x == 0) {
    stats[3 * r] = m;
    stats[3 * r + 1] = z0;
    stats[3 * r + 2] = __logf(z0) - z1 / z0;
  }
}

// A thread a draft: each draft's logit in its parent's row, for the parents among the rows the logits hold (child_logits).
__global__ void k_child_logits(const float* logits, int ld, int from, int rows, const int* drafts, int width,
                               const TreeRows* tree, float* z) {
  pdl_trigger();
  const bool on = tree_on(tree);
  for (int c = 1 + int(threadIdx.x); c <= width; c += blockDim.x) {
    const int at = on ? int(tree->path[c][tree->depth[c] - 1]) : c - 1;
    if (at >= from && at < from + rows) z[c - 1] = logits[size_t(at - from) * ld + drafts[c - 1]];
  }
}

// A round's accept by ops::accept_tree: a tree's rows (tree non-null), else a chain; exact matching unless the
// request asks for relaxed rules (stats from k_row_stats, the logits the head chose from).
__global__ void k_accept_tree(DevParams* P, const DevGen* G, const int* drafts, const int* choice, int width,
                              flyx_round_result* ring, Reply reply, TreeRows* tree, const float* logits, int ld,
                              const float* stats, const float* draft_prob, const float* draft_margin) {
  pdl_trigger();
  flyx_round_result* e = new_entry(P, ring);
  if (tree) tree->n_taken = 0;
  if (P->stopped) {
    P->noop = 1;
    e->stopped = 1;
    e->position = P->position;
    return;
  }
  P->noop = 0;
  const bool relaxed = logits && (G->accept_min_p > 0.f || G->accept_typical > 0.f);
  const bool on = tree_on(tree);
  auto parent = [&](int c) { return on ? int(tree->path[c][tree->depth[c] - 1]) : c - 1; };
  const float invT = G->temperature > 0.f ? 1.f / G->temperature : 1.f;
  auto prob_z = [&](int at, int c) {  // ld 0: a wide round's, the child's logit gathered (child_logits)
    const float l = ld ? logits[size_t(at) * ld + drafts[c - 1]] : logits[c - 1];
    return scaled_logit(l - stats[3 * at], G->temperature, invT);
  };
  auto qualifies = [&](int at, int c) {
    return ops::relaxed_qualifies(prob_z(at, c), stats[3 * at + 1], stats[3 * at + 2], G->accept_min_p,
                                  G->accept_typical);
  };
  int taken[FLYX_MAX_WIDE_ROWS];
  int32_t tokens[FLYX_MAX_WIDE_ROWS];
  // Not past the context's end (a round of more than FLYX_MAX_ROWS rows; a round of 16 started where flyx_room holds
  // never reaches it): the round in flight behind it then writes its rows inside the K/V cap.
  const int n = ops::accept_tree(width + 1, parent, drafts, choice, relaxed, qualifies, prob_z, G->stop, G->n_stop,
                                 taken, tokens, G->max_ctx - P->position);
  e->n_tokens = n + 1;
  e->n_accepted = n;
  for (int i = 0; i <= n; ++i) {
    e->tokens[i] = tokens[i];
    count_token(G, reply, tokens[i]);
  }
  e->n_drafts = width;
  for (int i = 0; i < width; ++i) e->drafts[i] = drafts[i], e->parents[i] = parent(i + 1) - 1;
  const int nc = width < FLYX_MAX_ROWS ? width : FLYX_MAX_ROWS;
  if (draft_prob)
    for (int i = 0; i < nc; ++i) e->draft_prob[i] = draft_prob[i];
  if (draft_margin)
    for (int i = 0; i < nc; ++i) e->draft_margin[i] = draft_margin[i];
  if (tree) {
    for (int i = 0; i <= n; ++i) tree->taken[i] = taken[i];
    tree->n_taken = n;
  }
  P->last_n = n;
  P->last_slot = flyx_round_slot(P->round);
  P->last_round = P->round;
  P->start = P->position;
  P->position += n + 1;
  P->anchor = tokens[n];
  P->round += 1;
  P->stopped = ops::reply_ends(tokens[n], P->position, G->limit, G->max_ctx, G->stop, G->n_stop);
  e->stopped = P->stopped;
  e->position = P->position;
}

__global__ void k_prefill_done(DevParams* P, int n, const int* choice, int sample, flyx_round_result* ring,
                               const DevGen* G, Reply reply) {
  pdl_trigger();
  P->position += n;
  P->stopped = 0;
  P->noop = 0;
  if (sample) {
    P->anchor = choice[0];
    count_token(G, reply, choice[0]);
    flyx_round_result* e = new_entry(P, ring);
    e->n_tokens = 1;
    e->tokens[0] = P->anchor;
    e->position = P->position;
  }
}

}  // namespace

void argmax_rows(const float* logits, int ld, int V, int* out, int rows, float* scratch, int* iscratch,
                 cudaStream_t st) {
  const int chunks = ceil_div(V, kArgChunk);
  k_argmax_partial<<<dim3(rows, chunks), 256, 0, st>>>(logits, ld, V, scratch, iscratch, nullptr, nullptr);
  k_argmax_final<<<rows, 32, 0, st>>>(scratch, iscratch, chunks, out, nullptr);
}

void choose_rows(const float* logits, int ld, int V, int rows, const DevParams* P, int base_add, const DevGen* G,
                 const uint32_t* mask, float* cz, int* ci, float* scratch, int* iscratch, int* out, cudaStream_t st,
                 const TreeRows* tree, int tree_from) {
  // Both paths are enqueued so a captured round serves greedy and sampled
  // requests alike; each kernel reads G and returns early when not chosen.
  const int chunks = ceil_div(V, kArgChunk);
  k_argmax_partial<<<dim3(rows, chunks), 256, 0, st>>>(logits, ld, V, scratch, iscratch, G, mask);
  k_argmax_final<<<rows, 32, 0, st>>>(scratch, iscratch, chunks, out, G);
  k_sample<<<rows, kSampleThreads, 0, st>>>(logits, ld, V, P, base_add, G, mask, cz, ci, out, tree, tree_from);
}

void penalize_rows(float* logits, int ld, int V, int rows, const DevGen* G, Reply reply, const int* drafts,
                   cudaStream_t st, const TreeRows* tree, int tree_from) {
  k_penalize<<<rows, 256, 0, st>>>(logits, ld, V, G, reply.count, reply.seen, reply.n_seen, drafts, tree, tree_from);
}

void row_stats(const float* logits, int ld, int V, int rows, const DevGen* G, float* stats, cudaStream_t st) {
  k_row_stats<<<rows, kSampleThreads, 0, st>>>(logits, ld, V, G, stats);
}

void child_logits(const float* logits, int ld, int from, int rows, const int* drafts, int width, const TreeRows* tree,
                  float* z, cudaStream_t st) {
  k_child_logits<<<1, FLYX_MAX_WIDE_ROWS, 0, st>>>(logits, ld, from, rows, drafts, width, tree, z);
}

void topk_rows(const float* logits, int ld, int V, int k, int* idx, float* val, int rows, cudaStream_t st,
               float* part_v, int* part_i, int part_rows, float* prob, float* margin) {
  if (part_v && part_i && rows <= part_rows && k <= kTopMax && V >= kTopkParts * kTopMax) {
    k_topk_part<<<dim3(kTopkParts, rows), kTopkPartThreads, 0, st>>>(logits, ld, V, part_v, part_i);
    k_topk_merge<<<rows, kTopkParts * kTopMax, 0, st>>>(part_v, part_i, k, idx, val, prob, margin);
    return;
  }
  k_topk<<<rows, 512, 0, st>>>(logits, ld, V, k, idx, val, prob, margin);
}

void accept(DevParams* P, const DevGen* G, const int* drafts, const int* choice, int width,
            flyx_round_result* ring, Reply reply, cudaStream_t st, const float* draft_prob,
            const float* draft_margin) {
  k_accept<<<1, 1, 0, st>>>(P, G, drafts, choice, width, ring, reply, draft_prob, draft_margin);
}

void accept_tree(DevParams* P, const DevGen* G, const int* drafts, const int* choice, int width,
                 flyx_round_result* ring, Reply reply, TreeRows* tree, const float* logits, int ld,
                 const float* stats, cudaStream_t st, const float* draft_prob, const float* draft_margin) {
  k_accept_tree<<<1, 1, 0, st>>>(P, G, drafts, choice, width, ring, reply, tree, logits, ld, stats, draft_prob,
                                 draft_margin);
}

void prefill_done(DevParams* P, int n, const int* choice, int sample, flyx_round_result* ring, const DevGen* G,
                  Reply reply, cudaStream_t st) {
  k_prefill_done<<<1, 1, 0, st>>>(P, n, choice, sample, ring, G, reply);
}

}  // namespace flyx::cuda
