// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine's kernels (ops/dspine/ref.h), which regions/layers.cpp's dspine_block runs between its GEMMs: the code
// bridge's row norms, codes, messages and write, and the walk's candidate head rows and base scores, senders' codes,
// scores and chain.  Float32 throughout; a row's sums in a fixed order (block_sum, warp_sum), so a row's values are its
// own whatever else the launch holds.
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

// One block a row: rms = sqrt(mean(x^2)), xn = x / (rms + eps).
__global__ void k_dspine_normalize(const float* x, int H, float eps, float* xn, float* rms) {
  pdl_trigger();
  const int r = blockIdx.x;
  const float* xr = x + size_t(r) * H;
  float s = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) s += xr[h] * xr[h];
  s = block_sum(s);
  const float m = sqrtf(s / float(H));
  if (threadIdx.x == 0) rms[r] = m;
  const float inv = 1.f / (m + eps);
  for (int h = threadIdx.x; h < H; h += blockDim.x) xn[size_t(r) * H + h] = xr[h] * inv;
}

// One block a row: row 0 the anchor's code, sqrt(C) codebook[anchor]; row r >= 1 its read u scaled to
// sqrt(C) u / (|u| + eps), in place.
__global__ void k_dspine_codes(float* codes, int C, const bf16* codebook, const DevParams* P, float eps) {
  pdl_trigger();
  const int r = blockIdx.x;
  float* c = codes + size_t(r) * C;
  const float sq = sqrtf(float(C));
  if (r == 0) {
    const bf16* e = codebook + size_t(P->anchor) * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) c[i] = sq * __bfloat162float(e[i]);
    return;
  }
  float s = 0.f;
  for (int i = threadIdx.x; i < C; i += blockDim.x) s += c[i] * c[i];
  s = block_sum(s);
  const float f = sq / (sqrtf(s) + eps);
  for (int i = threadIdx.x; i < C; i += blockDim.x) c[i] *= f;
}

// One block a row of gates [rows][3R] (gate | pred | prev of the row's code): m_0 = 0, m_r = sigmoid(q_r + s_{r-1} +
// bias) * v_{r-1}; g (optional) each row's q.
__global__ void k_dspine_message(const float* gates, int R, const float* bias, float* msg, float* g) {
  pdl_trigger();
  const int r = blockIdx.x;
  const float* q = gates + size_t(r) * 3 * R;
  const float* sender = r > 0 ? gates + size_t(r - 1) * 3 * R : nullptr;
  for (int i = threadIdx.x; i < R; i += blockDim.x) {
    if (g) g[size_t(r) * R + i] = q[i];
    msg[size_t(r) * R + i] = sender ? sigmoidf(q[i] + sender[R + i] + bias[i]) * sender[2 * R + i] : 0.f;
  }
}

// x_r += rms_r d_r for rows r >= 1 (row 0, the anchor's, keeps its value); before (optional): x as it was.
__global__ void k_dspine_apply(float* x, const float* d, const float* rms, int H, float* before) {
  pdl_trigger();
  const int r = blockIdx.y, h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  const size_t i = size_t(r) * H + h;
  const float v = x[i];
  if (before) before[i] = v;
  if (r > 0) x[i] = v + rms[r] * d[i];
}

// One block a candidate i (row t = i / k's): heads[i] = head[cand[i]] * nw (W null: heads[i] holds the row, as
// embed_blocks gave it), base[i] = <before_{t+1}, heads[i]>.
__global__ void k_dspine_heads(const int* cand, const bf16* W, const float* nw, const float* before, int H, int k,
                               float* heads, float* base) {
  pdl_trigger();
  const int i = blockIdx.x, t = i / k;
  float* hr = heads + size_t(i) * H;
  const bf16* wr = W ? W + size_t(cand[i]) * H : nullptr;
  const float* xr = before + size_t(t + 1) * H;
  float s = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    const float v = (wr ? __bfloat162float(wr[h]) : hr[h]) * nw[h];
    hr[h] = v;
    s += xr[h] * v;
  }
  s = block_sum(s);
  if (threadIdx.x == 0) base[i] = s;
}

// One block a sender: row 0 the anchor's code, row i >= 1 that of candidate i - 1 (row (i - 1) / k's).
__global__ void k_dspine_senders(const DevParams* P, const int* cand, const bf16* codebook, int C, float* codes) {
  pdl_trigger();
  const int i = blockIdx.x;
  const int tok = i == 0 ? P->anchor : cand[i - 1];
  const bf16* e = codebook + size_t(tok) * C;
  const float sq = sqrtf(float(C));
  for (int c = threadIdx.x; c < C; c += blockDim.x) codes[size_t(i) * C + c] = sq * __bfloat162float(e[c]);
}

// One block a sender i, 32 k threads: the row after it, t = i == 0 ? 0 : 1 + (i - 1) / k, scores candidate c (warp c)
//   base[t k + c] + rms[t + 1] m . tg[t k + c],  m = sigmoid(g[t + 1] + sv_g[i] + bias) * sv_v[i]
// (sv [senders][2R]: the sender code's pred | prev); pick[i] the best candidate, the first on ties.
__global__ void k_dspine_scores(const float* g, const float* sv, const float* bias, const float* tg, const float* base,
                                const float* rms, int R, int k, int* pick) {
  pdl_trigger();
  extern __shared__ float m[];  // [R]
  __shared__ float score[kTopMax];
  const int i = blockIdx.x, t = i == 0 ? 0 : 1 + (i - 1) / k;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* gr = g + size_t(t + 1) * R;
  const float* si = sv + size_t(i) * 2 * R;
  for (int r = threadIdx.x; r < R; r += blockDim.x) m[r] = sigmoidf(gr[r] + si[r] + bias[r]) * si[R + r];
  __syncthreads();
  if (warp < k) {
    const float* tr = tg + (size_t(t) * k + warp) * R;
    float s = 0.f;
    for (int r = lane; r < R; r += 32) s += m[r] * tr[r];
    s = warp_sum(s);
    if (lane == 0) score[warp] = base[size_t(t) * k + warp] + rms[t + 1] * s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int best = 0;
    for (int c = 1; c < k; ++c)
      if (score[c] > score[best]) best = c;
    pick[i] = best;
  }
}

// The chain: draft 0 is candidate pick[0] of row 0, draft t candidate pick[1 + (t - 1) k + j] of row t, j draft
// t - 1's index among its row's candidates.  The test's tokens instead when P->oracle_len > 0.
__global__ void k_dspine_walk(const DevParams* P, const int* cand, const int* pick, int k, int width,
                              const int* oracle, int* drafts) {
  pdl_trigger();
  if (threadIdx.x != 0) return;
  if (P->oracle_len > 0) {
    for (int i = 0; i < width; ++i) {
      const int pos = P->position + 1 + i;
      drafts[i] = pos < P->oracle_len ? oracle[pos] : 0;
    }
    return;
  }
  int j = pick[0];
  drafts[0] = cand[j];
  for (int t = 1; t < width; ++t) {
    j = pick[1 + (t - 1) * k + j];
    drafts[t] = cand[size_t(t) * k + j];
  }
}

constexpr int kThreads = 256;

}  // namespace

void dspine_normalize(const float* x, int H, float eps, int rows, float* xn, float* rms, cudaStream_t st) {
  k_dspine_normalize<<<rows, kThreads, 0, st>>>(x, H, eps, xn, rms);
}

void dspine_codes(float* codes, int C, const bf16* codebook, const DevParams* P, float eps, int rows,
                  cudaStream_t st) {
  k_dspine_codes<<<rows, kThreads, 0, st>>>(codes, C, codebook, P, eps);
}

void dspine_message(const float* gates, int R, const float* bias, int rows, float* msg, float* g, cudaStream_t st) {
  k_dspine_message<<<rows, kThreads, 0, st>>>(gates, R, bias, msg, g);
}

void dspine_apply(float* x, const float* d, const float* rms, int H, int rows, float* before, cudaStream_t st) {
  k_dspine_apply<<<dim3((H + kThreads - 1) / kThreads, rows), kThreads, 0, st>>>(x, d, rms, H, before);
}

void dspine_heads(const int* cand, const bf16* W, const float* nw, const float* before, int H, int k, int n,
                  float* heads, float* base, cudaStream_t st) {
  k_dspine_heads<<<n, kThreads, 0, st>>>(cand, W, nw, before, H, k, heads, base);
}

void dspine_senders(const DevParams* P, const int* cand, const bf16* codebook, int C, int senders, float* codes,
                    cudaStream_t st) {
  k_dspine_senders<<<senders, kThreads, 0, st>>>(P, cand, codebook, C, codes);
}

void dspine_scores(const float* g, const float* sv, const float* bias, const float* tg, const float* base,
                   const float* rms, int R, int k, int senders, int* pick, cudaStream_t st) {
  k_dspine_scores<<<senders, 32 * k, size_t(R) * sizeof(float), st>>>(g, sv, bias, tg, base, rms, R, k, pick);
}

void dspine_walk(const DevParams* P, const int* cand, const int* pick, int k, int width, const int* oracle,
                 int* drafts, cudaStream_t st) {
  k_dspine_walk<<<1, 32, 0, st>>>(P, cand, pick, k, width, oracle, drafts);
}

}  // namespace flyx::cuda
