// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Drafting kernels: MTP inputs and hidden ring, the DFlash2 grouped convs, selector walk and tree, context K/V.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__global__ void k_grouped_conv(const float* x, int ldx, const float* delta, int ld_delta, const float* base, int H,
                               int taps, int group, float* y, int ldy, int accumulate) {
  pdl_trigger();
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  const int G = H / group;
  float acc = 0.f;
  for (int t = 0; t < taps && t <= r; ++t)
    acc += (base[size_t(t) * H + c] + delta[size_t(r) * ld_delta + t * G + c / group]) * x[size_t(r - t) * ldx + c];
  float* dst = y + size_t(r) * ldy + c;
  *dst = accumulate ? *dst + acc : acc;
}

// One block of 32*k threads: warp c scores candidate c.
__global__ void k_selector_walk(const DevParams* P, const int* cand, const float* unary, int k, const float* hid,
                                int rank, const bf16* pred, const bf16* succ, int width, const int* oracle,
                                int* drafts) {
  pdl_trigger();
  __shared__ float score[kTopMax];
  __shared__ int prev;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (P->oracle_len > 0) {
    if (threadIdx.x < width) {
      const int pos = P->position + 1 + threadIdx.x;
      drafts[threadIdx.x] = pos < P->oracle_len ? oracle[pos] : 0;
    }
    return;
  }
  if (threadIdx.x == 0) prev = P->anchor;
  __syncthreads();
  for (int s = 0; s < width; ++s) {
    if (warp < k) {
      const int cid = cand[s * k + warp];
      const bf16* pr = pred + size_t(prev) * rank;
      const bf16* sc = succ + size_t(cid) * rank;
      const float* hs = hid + size_t(s) * rank;
      float e = 0.f;
      for (int i = lane; i < rank; i += 32) e += __bfloat162float(pr[i]) * hs[i] * __bfloat162float(sc[i]);
      e = warp_sum(e);
      if (lane == 0) score[warp] = unary[s * k + warp] + e;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int best = 0;
      for (int c = 1; c < k; ++c)
        if (score[c] > score[best]) best = c;
      prev = cand[s * k + best];
      drafts[s] = prev;
    }
    __syncthreads();
  }
}

// ops/draft/ref.h's best_first_tree on the selector's scores (the walk's chain, then branches best first), the tree
// written as the round reads it (TreeRows).  One
// block of 32*k threads: warp c scores candidate c of the node being expanded (as k_selector_walk does), warp 0
// normalises them and picks the next node.  With the test's drafts (P->oracle_len), a chain of them.
__global__ void k_selector_tree(const DevParams* P, const int* cand, const float* unary, int k, const float* hid,
                                int rank, const bf16* pred, const bf16* succ, int width, int depths,
                                const int* oracle, int* drafts, TreeRows* tree) {
  pdl_trigger();
  constexpr int kRows = FLYX_MAX_ROWS;
  __shared__ float val[kRows * kTopMax];  // node n's child c at n * k + c; -inf: none, or taken
  __shared__ float score[kTopMax];
  __shared__ int row_tok[kRows], row_parent[kRows], row_depth[kRows];
  __shared__ float row_val[kRows];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < kRows * kTopMax; i += blockDim.x) val[i] = -INFINITY;
  for (int i = threadIdx.x; i < kRows * kRows; i += blockDim.x) tree->path[i / kRows][i % kRows] = 0;
  if (threadIdx.x == 0) {
    row_tok[0] = P->anchor;
    row_parent[0] = -1;
    row_depth[0] = 0;
    row_val[0] = 0.f;
  }
  __syncthreads();
  if (P->oracle_len > 0) {
    if (threadIdx.x == 0)
      for (int i = 0; i < width; ++i) {
        const int pos = P->position + 1 + i;
        drafts[i] = pos < P->oracle_len ? oracle[pos] : 0;
        row_parent[i + 1] = i;
        row_depth[i + 1] = i + 1;
      }
  } else {
    for (int n = 0; n < width; ++n) {
      const int d = row_depth[n];
      if (d < depths) {  // node n's children
        if (warp < k) {
          const int cid = cand[d * k + warp];
          const bf16* pr = pred + size_t(row_tok[n]) * rank;
          const bf16* sc = succ + size_t(cid) * rank;
          const float* hs = hid + size_t(d) * rank;
          float e = 0.f;
          for (int i = lane; i < rank; i += 32) e += __bfloat162float(pr[i]) * hs[i] * __bfloat162float(sc[i]);
          e = warp_sum(e);
          if (lane == 0) score[warp] = unary[d * k + warp] + e;
        }
        __syncthreads();
        if (warp == 0) {
          const float s = lane < k ? score[lane] : -INFINITY;
          float m = s;
          for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
          const float z = warp_sum(lane < k ? expf(s - m) : 0.f);
          if (lane < k) val[n * k + lane] = row_val[n] + (s - m - logf(z));
        }
      }
      __syncthreads();
      if (warp == 0) {  // the highest-valued child: ties to the lower index (the earlier node, the lower candidate)
        // The trunk (the walk's chain): node n's children only, node n being the trunk's last.
        const int from = n < min(width, depths) ? n * k : 0;
        float bv = -INFINITY;
        int bi = -1;
        for (int i = from + lane; i < (n + 1) * k; i += 32)
          if (val[i] > bv) bv = val[i], bi = i;
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
          const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) bv = ov, bi = oi;
        }
        if (lane == 0) {
          const int r = n + 1;
          if (bi < 0) {  // no child left: the anchor's first candidate again
            row_parent[r] = 0;
            row_tok[r] = cand[0];
            row_val[r] = -INFINITY;
          } else {
            const int u = bi / k;
            row_parent[r] = u;
            row_tok[r] = cand[row_depth[u] * k + bi % k];
            row_val[r] = bv;
            val[bi] = -INFINITY;
          }
          row_depth[r] = row_depth[row_parent[r]] + 1;
          drafts[n] = row_tok[r];
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {  // as set_tree writes a given tree
    tree->n = width + 1;
    tree->n_taken = 0;
    tree->depth[0] = 0;
    for (int r = 1; r <= width; ++r) {
      const int p = row_parent[r];
      tree->depth[r] = row_depth[r];
      for (int d = 0; d < row_depth[p] + 1; ++d) tree->path[r][d] = tree->path[p][d];
      tree->path[r][row_depth[r]] = static_cast<unsigned char>(r);
    }
  }
}

template <class TIn>
__global__ void k_tap_copy(const TIn* x, int H, float* feats, int ld, int slot) {
  pdl_trigger();
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    feats[size_t(blockIdx.x) * ld + size_t(slot) * H + i] = in_value(x + size_t(blockIdx.x) * H + i);
}

// grid (layers * nkv, rows); block hd.
__global__ void k_draft_ctx_kv(const float* kv, int ld, const float* k_norms, float eps, double theta, int nkv, int hd,
                               const DevParams* P, int pos_sel, int skip_if_noop, kv16* dK, kv16* dV,
                               size_t layer_stride, const double* rope) {
  pdl_trigger();
  extern __shared__ float sh[];
  if (skip_if_noop && P->noop) return;
  const int layer = blockIdx.x / nkv, kh = blockIdx.x - layer * nkv, r = blockIdx.y, d = threadIdx.x;
  const int pos = (pos_sel == kPosStart ? P->start : P->position) + r;
  const float* row = kv + size_t(r) * ld + size_t(layer) * 2 * nkv * hd;
  const float v = row[kh * hd + d];
  const float inv = rsqrtf(block_sum(v * v) / hd + eps);
  const float nv = v * inv * k_norms[layer * hd + d];
  sh[d] = nv;
  __syncthreads();
  const int half = hd / 2, i = d < half ? d : d - half;
  float c, s;
  if (rope) rope_cs_inv(pos, rope[i], &c, &s);  // rope_table's factor, or its pow
  else rope_cs(pos, i, hd, theta, &c, &s);
  const float out = d < half ? nv * c - sh[d + half] * s : nv * c + sh[d - half] * s;
  const size_t at = size_t(layer) * layer_stride + (size_t(pos) * nkv + kh) * hd + d;
  dK[at] = kv16_of(out);
  dV[at] = kv16_of(row[nkv * hd + kh * hd + d]);
}

constexpr int kMtpThreads = 512;
constexpr int kMtpCache = 10;  // 512 * 10 = 5120: single-pass register cache covers hidden_dim up to 5120

__global__ void __launch_bounds__(kMtpThreads) k_mtp_gather(const DevParams* P, int pos_sel, int skip_if_noop,
                                                             const float* h0, const float* hring, const float* hn,
                                                             int ldh, const int* tokens, const bf16* E,
                                                             const float* emb, const float* we, const float* wh,
                                                             float offset, float eps, int H, float* cat) {
  pdl_trigger();
  if (skip_if_noop && P->noop) return;
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int base = pos_sel == kPosStart ? P->start : P->position;
  const float* h = r > 0 ? hn + size_t(r - 1) * ldh
                 : h0    ? h0
                         : hring + size_t(((base - 1) % kHring + kHring) % kHring) * H;
  const bf16* e = E ? E + size_t(tokens[r]) * H : nullptr;
  const float* ef = emb + size_t(r) * H;  // read only without a table
  const bool cached = H <= kMtpThreads * kMtpCache;
  float ev[kMtpCache], hv[kMtpCache], wev[kMtpCache], whv[kMtpCache];
  float se = 0.f, sh = 0.f;
  if (cached) {
#pragma unroll
    for (int i = 0; i < kMtpCache; ++i) {
      const int c = tid + i * kMtpThreads;
      const bool in = c < H;
      const float ei = in ? (e ? __bfloat162float(e[c]) : ef[c]) : 0.f;
      const float hi = in ? h[c] : 0.f;
      ev[i] = ei;
      hv[i] = hi;
      wev[i] = in ? we[c] : 0.f;
      whv[i] = in ? wh[c] : 0.f;
      se += ei * ei;
      sh += hi * hi;
    }
  } else {
    for (int c = tid; c < H; c += kMtpThreads) {
      const float ei = e ? __bfloat162float(e[c]) : ef[c], hi = h[c];
      se += ei * ei;
      sh += hi * hi;
    }
  }
  __shared__ float s_e[kMtpThreads / 32], s_h[kMtpThreads / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    se += __shfl_xor_sync(0xffffffffu, se, o);
    sh += __shfl_xor_sync(0xffffffffu, sh, o);
  }
  if (lane == 0) {
    s_e[warp] = se;
    s_h[warp] = sh;
  }
  __syncthreads();
  float be = s_e[lane & 15], bh = s_h[lane & 15];
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    be += __shfl_xor_sync(0xffffffffu, be, o);
    bh += __shfl_xor_sync(0xffffffffu, bh, o);
  }
  const float ie = rsqrtf(be / H + eps);
  const float ih = rsqrtf(bh / H + eps);
  float* out = cat + size_t(r) * 2 * H;
  if (cached) {
#pragma unroll
    for (int i = 0; i < kMtpCache; ++i) {
      const int c = tid + i * kMtpThreads;
      if (c < H) {
        out[c] = ev[i] * ie * (offset + wev[i]);
        out[H + c] = hv[i] * ih * (offset + whv[i]);
      }
    }
  } else {
    for (int c = tid; c < H; c += kMtpThreads) {
      out[c] = (e ? __bfloat162float(e[c]) : ef[c]) * ie * (offset + we[c]);
      out[H + c] = h[c] * ih * (offset + wh[c]);
    }
  }
}

__global__ void k_hring_put(const DevParams* P, int pos_sel, int skip_if_noop, const float* hn, int ldh, int rows,
                            float* hring, int H) {
  pdl_trigger();
  if (skip_if_noop && P->noop) return;
  const int r = blockIdx.x;
  if (r < rows - kHring) return;  // only the last kHring rows land in the ring
  const int base = pos_sel == kPosStart ? P->start : P->position;
  float* dst = hring + size_t((base + r) % kHring) * H;
  for (int c = threadIdx.x; c < H; c += blockDim.x) dst[c] = hn[size_t(r) * ldh + c];
}

__global__ void k_mtp_oracle(const DevParams* P, const int* oracle, int step, int* mtok) {
  pdl_trigger();
  if (P->oracle_len <= 0) return;
  const int pos = P->position + 1 + step;
  mtok[step + 1] = pos < P->oracle_len ? oracle[pos] : 0;
}

__global__ void k_given_drafts(const DevParams* P, const int* oracle, int width, int* drafts) {
  pdl_trigger();
  const int i = threadIdx.x;
  if (i >= width) return;
  const int pos = P->position + 1 + i;
  drafts[i] = pos < P->oracle_len ? oracle[pos] : 0;
}

__global__ void k_copy_ints(const int* src, int* dst, int n) {
  pdl_trigger();
  if (int(threadIdx.x) < n) dst[threadIdx.x] = src[threadIdx.x];
}

}  // namespace

void mtp_gather(const DevParams* P, int pos_sel, bool skip_if_noop, const float* h0, const float* hring,
                const float* hn, int ldh, const int* tokens, const bf16* E, const float* emb, const float* we,
                const float* wh, float offset, float eps, int H, int rows, float* cat, cudaStream_t st) {
  k_mtp_gather<<<rows, kMtpThreads, 0, st>>>(P, pos_sel, skip_if_noop, h0, hring, hn, ldh, tokens, E, emb, we, wh,
                                             offset, eps, H, cat);
}

void hring_put(const DevParams* P, int pos_sel, bool skip_if_noop, const float* hn, int ldh, int rows, float* hring,
               int H, cudaStream_t st) {
  k_hring_put<<<rows, 256, 0, st>>>(P, pos_sel, skip_if_noop, hn, ldh, rows, hring, H);
}

void mtp_oracle(const DevParams* P, const int* oracle, int step, int* mtok, cudaStream_t st) {
  k_mtp_oracle<<<1, 1, 0, st>>>(P, oracle, step, mtok);
}

__global__ void k_shift_params(const DevParams* P, DevParams* dst, int add) {
  pdl_trigger();
  *dst = *P;
  dst->position += add;
}

void shift_params(const DevParams* P, DevParams* dst, int add, cudaStream_t st) {
  k_shift_params<<<1, 1, 0, st>>>(P, dst, add);
}

void given_drafts(const DevParams* P, const int* oracle, int width, int* drafts, cudaStream_t st) {
  k_given_drafts<<<1, FLYX_MAX_WIDE_ROWS, 0, st>>>(P, oracle, width, drafts);  // a wide round's drafts too
}

void copy_ints(const int* src, int* dst, int n, cudaStream_t st) { k_copy_ints<<<1, 32, 0, st>>>(src, dst, n); }

void grouped_conv(const float* x, int ldx, const float* delta, int ld_delta, const float* base, int H, int taps,
                  int group, float* y, int ldy, int rows, bool accumulate, cudaStream_t st) {
  k_grouped_conv<<<dim3(ceil_div(H, 256), rows), 256, 0, st>>>(x, ldx, delta, ld_delta, base, H, taps, group, y, ldy,
                                                               accumulate);
}

void selector_walk(const DevParams* P, const int* cand, const float* unary, int k, const float* hid, int rank,
                   const bf16* pred, const bf16* succ, int width, const int* oracle, int* drafts, cudaStream_t st) {
  k_selector_walk<<<1, 32 * (k > 1 ? k : 1), 0, st>>>(P, cand, unary, k, hid, rank, pred, succ, width, oracle, drafts);
}

void selector_tree(const DevParams* P, const int* cand, const float* unary, int k, const float* hid, int rank,
                   const bf16* pred, const bf16* succ, int width, int depths, const int* oracle, int* drafts,
                   TreeRows* tree, cudaStream_t st) {
  if (k < 1 || k > kTopMax || width < 1 || width >= FLYX_MAX_ROWS)
    throw std::runtime_error("selector_tree: candidates or width out of range");
  k_selector_tree<<<1, 32 * (k > 1 ? k : 1), 0, st>>>(P, cand, unary, k, hid, rank, pred, succ, width, depths, oracle,
                                                      drafts, tree);
}

void tap_copy(const float* x, int H, float* feats, int ld_feats, int slot, int rows, cudaStream_t st) {
  k_tap_copy<<<rows, 256, 0, st>>>(x, H, feats, ld_feats, slot);
}

void tap_copy(const bf16* x, int H, float* feats, int ld_feats, int slot, int rows, cudaStream_t st) {
  k_tap_copy<<<rows, 256, 0, st>>>(x, H, feats, ld_feats, slot);
}

void draft_ctx_kv(const float* kv, int ld, const float* k_norms, float eps, double theta, int layers, int nkv, int hd,
                  int rows, const DevParams* P, int pos_sel, bool skip_if_noop, kv16* dK, kv16* dV,
                  size_t layer_stride, cudaStream_t st) {
  k_draft_ctx_kv<<<dim3(layers * nkv, rows), hd, hd * sizeof(float), st>>>(kv, ld, k_norms, eps, theta, nkv, hd, P,
                                                                            pos_sel, skip_if_noop, dK, dV,
                                                                            layer_stride, rope_table(theta, hd));
}

}  // namespace flyx::cuda
