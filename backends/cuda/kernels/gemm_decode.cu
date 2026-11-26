// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GEMMs for 1..16 rows (decode, verification, drafting): CUDA-core rows, bf16 and NVFP4 on tensor cores.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "flyx/gemm_tables.inc"
#include "backends/cuda/kernels/gemm_internal.h"
#include <utility>

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

namespace flyx::cuda {
namespace {

template <int NT, bool R16, int WARPS, int UNROLL, bool W12_MODE>
__host__ __device__ constexpr bool bf16_skinny_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const int b = tables::arch_bucket(cc);
  const unsigned mask = W12_MODE ? tables::kReachW12Skinny[b][R16 ? 1 : 0] : tables::kReachBf16Skinny[b][R16 ? 1 : 0];
  bool hit = false;
#define X(i, nt, w, u) \
  if ((mask & (1u << i)) && nt == NT && w == WARPS && u == UNROLL) hit = true;
  FLYX_BF16_SKINNY(X)
#undef X
  return hit;
#endif
}

template <int SPAN, int CH, int NT, bool R16, int WARPS, int UNROLL>
__host__ __device__ constexpr bool fp4_skinny_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const unsigned mask = tables::kReachFp4Skinny[tables::arch_bucket(cc)][R16 ? 1 : 0];
  bool hit = false;
#define X(i, span, ch, nt, w, u) \
  if ((mask & (1u << i)) && span == SPAN && ch == CH && nt == NT && w == WARPS && u == UNROLL) hit = true;
  FLYX_FP4_SKINNY(X)
#undef X
  return hit;
#endif
}

// The many-row kernel (k_gemm_skinny_fp4_mt) of a reachable 9..16-row configuration's span, chunks and unrolling, one
// chunk a warp; its own column tiles (its bits are the span's and the chunks').
template <int SPAN, int CH>
__host__ __device__ constexpr bool fp4_mt_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const unsigned mask = tables::kReachFp4Skinny[tables::arch_bucket(cc)][1];
  bool hit = false;
#define X(i, span, ch, nt, w, u) \
  if ((mask & (1u << i)) && span == SPAN && ch == CH && w == CH) hit = true;
  FLYX_FP4_SKINNY(X)
#undef X
  return hit;
#endif
}

template <int NT, int CH, int UNROLL, int PARTS>
__host__ __device__ constexpr bool bf16_mt_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const unsigned mask = tables::kReachBf16Mt[tables::arch_bucket(cc)];
  bool hit = false;
#define X(i, nt, ch, u, parts) \
  if ((mask & (1u << i)) && nt == NT && ch == CH && u == UNROLL && parts == PARTS) hit = true;
  FLYX_BF16_MT(X)
#undef X
  return hit;
#endif
}

// ---- GEMM: few rows (decode, verify, drafter) ------------------------------------------
// Each warp owns COLS output columns for all rows; lanes split K in 8-element
// slices; activations are staged in shared memory one K tile at a time.
template <int ROWS>
__global__ void __launch_bounds__(128) k_gemm_rows(const float* __restrict__ x, int ldx, const bf16* __restrict__ W,
                                                   int N, int K, float* __restrict__ y, int ldy, int rows,
                                                   int accumulate) {
  pdl_trigger();
  constexpr int COLS = 4;
  constexpr int KT = ROWS <= 4 ? 1024 : (ROWS <= 8 ? 512 : 256);
  __shared__ __align__(16) float xs[ROWS][KT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * 4 + warp) * COLS;
  float acc[ROWS][COLS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int c = 0; c < COLS; ++c) acc[r][c] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KT) {
    const int kt = min(KT, K - k0);
    __syncthreads();
    for (int i = threadIdx.x; i < ROWS * KT; i += blockDim.x) {
      int r = i / KT, k = i - r * KT;
      xs[r][k] = (r < rows && k < kt) ? x[size_t(r) * ldx + k0 + k] : 0.f;
    }
    __syncthreads();
    for (int kk = lane * 8; kk < kt; kk += 256) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int n = n0 + c;
        if (n < N) {
          float w[8];
          unpack8(*reinterpret_cast<const uint4*>(W + size_t(n) * K + k0 + kk), w);
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[r][kk]);
            const float4 b = *reinterpret_cast<const float4*>(&xs[r][kk + 4]);
            acc[r][c] += a.x * w[0] + a.y * w[1] + a.z * w[2] + a.w * w[3] + b.x * w[4] + b.y * w[5] +
                         b.z * w[6] + b.w * w[7];
          }
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      float v = warp_sum(acc[r][c]);
      const int n = n0 + c;
      if (lane == 0 && r < rows && n < N) {
        float* dst = y + size_t(r) * ldy + n;
        *dst = accumulate ? *dst + v : v;
      }
    }
}

// ---- GEMM: 1..16 rows on tensor cores (decode, verify, drafter) ----------------------------
// One kernel for every row count, so a row's result never depends on how
// many rows run with it: plain decoding and verification agree bit for bit.
//
// A warp computes 16 rows x (8 * NT) columns with mma.m16n8k16; the warps of
// a block split K and add their partial tiles in a fixed order.  Each lane
// streams 16 bytes of one weight row per 32-wide K span; K is permuted the
// same way for activations and weights so those 16 bytes are exactly the
// lane's B fragments for two k16 steps (lane t holds k = 8t .. 8t+7 of the
// span: step j uses 8t+4j+{0,1} as its low pair and 8t+4j+{2,3} as its high
// pair).  Activations are float32: each is split into three bf16 parts (its
// high part, the remainder, the remainder's remainder: 24 significant bits,
// as float32), three MMAs, the smallest parts first.  The products of bf16
// weights and bf16 parts are exact, so the sums are float32's, as a
// matrix-vector kernel on float32 inputs gives.
//
// K is cut into kSkinnyChunks fixed chunks, each summed on its own and the
// chunks added in order, whatever the number of warps: every configuration
// (tile, warps, unroll) gives bit-identical results, so a tuning table can
// pick one per shape and row count freely.
constexpr int kSkinnyChunks = 16;

// The output column of a result column: itself (cols maps it for a gathered GEMM).
struct SkinnySameColumn {
  __device__ __forceinline__ int operator()(int c) const { return c; }
};

// A skinny GEMM's input: rows of K values as three bf16 planes (k_split, put()), stride K each.
struct Parts {
  const bf16 *hi, *mid, *lo;
};

// The A fragments of one row half for k16 step j (values 8t+4j+{0,1} as the low pair, +{2,3} as the high pair).
__device__ __forceinline__ void part_frags(const uint4& a, const uint4& b, int j, uint32_t (&f)[4]) {
  f[0] = j ? a.z : a.x;
  f[1] = j ? b.z : b.x;
  f[2] = j ? a.w : a.y;
  f[3] = j ? b.w : b.y;
}

// The body of k_gemm_skinny for any weight storage: load(i, k) returns the
// lane's 8 bf16 weights (as 4 bf16x2 words) of row n_base + 8 i + g at
// k + 8t .. k + 8t + 7.  Two storages that load the same bf16 values give the
// same bits.  Result column c (n_base <= c < n_base + 8 NT, c < N) goes to
// y's column cols(c); a column's sum depends only on its own weights.
template <int NT, bool R16, int WARPS, int UNROLL, class Load, class Cols = SkinnySameColumn>
__device__ __forceinline__ void skinny_body(Parts x, int ldk, int N, int K, float* y, int ldy, int rows,
                                            int accumulate, Load load, int n_base, Cols cols = {}) {
  static_assert(kSkinnyChunks % WARPS == 0, "warps must divide the chunks");
  constexpr int CPW = kSkinnyChunks / WARPS;
  __shared__ float red[kSkinnyChunks][NT][32][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int spans = K >> 5;
  const bool ra = g < rows, rb = R16 && g + 8 < rows;
  const size_t oa = size_t(g) * ldk + 8 * t, ob = size_t(g + 8) * ldk + 8 * t;
  pdl_wait();  // the activations come from the kernel before
#pragma unroll 1
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp * CPW + c;
    const int s0 = chunk * spans / kSkinnyChunks, s1 = (chunk + 1) * spans / kSkinnyChunks;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll UNROLL
    for (int s = s0; s < s1; ++s) {
      const int k = s << 5;
      uint4 w[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) w[i] = load(i, k);
      const uint4 z = make_uint4(0, 0, 0, 0);
      const uint4 ah = ra ? *reinterpret_cast<const uint4*>(x.hi + oa + k) : z;
      const uint4 am = ra ? *reinterpret_cast<const uint4*>(x.mid + oa + k) : z;
      const uint4 al = ra ? *reinterpret_cast<const uint4*>(x.lo + oa + k) : z;
      const uint4 bh = rb ? *reinterpret_cast<const uint4*>(x.hi + ob + k) : z;
      const uint4 bm = rb ? *reinterpret_cast<const uint4*>(x.mid + ob + k) : z;
      const uint4 bl = rb ? *reinterpret_cast<const uint4*>(x.lo + ob + k) : z;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // row g / row g+8, low pair then high pair (values 8t+4j+{0,1}, +{2,3})
        uint32_t hi[4], mid[4], lo[4];
        part_frags(ah, bh, j, hi);
        part_frags(am, bm, j, mid);
        part_frags(al, bl, j, lo);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const uint32_t bf[2] = {j == 0 ? w[i].x : w[i].z, j == 0 ? w[i].y : w[i].w};
          mma_bf16(acc[i], lo, bf);
          mma_bf16(acc[i], mid, bf);
          mma_bf16(acc[i], hi, bf);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[chunk][i][lane][e] = acc[i][e];
  }
  __syncthreads();
  // Add the chunks in order and store.
  for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
    const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < kSkinnyChunks; ++c) v += red[c][i][l][e];
    const int r = (l >> 2) + (e >> 1) * 8, col = n_base + i * 8 + 2 * (l & 3) + (e & 1);
    if (r < rows && col < N) {
      float* dst = y + size_t(r) * ldy + cols(col);
      *dst = accumulate ? *dst + v : v;
    }
  }
}

template <int NT, bool R16, int WARPS, int UNROLL>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_skinny(Parts x, int ldk, const bf16* __restrict__ W, int N,
                                                            int K, float* __restrict__ y, int ldy, int rows,
                                                            int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!bf16_skinny_active<NT, R16, WARPS, UNROLL, false>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  const int g = (threadIdx.x & 31) >> 2, t = threadIdx.x & 3;
  const bf16* wp[NT];
  bool wv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = blockIdx.x * 8 * NT + i * 8 + g;
    wv[i] = n < N;
    wp[i] = W + size_t(wv[i] ? n : 0) * K + 8 * t;
  }
  skinny_body<NT, R16, WARPS, UNROLL>(x, ldk, N, K, y, ldy, rows, accumulate, [&](int i, int k) {
    return wv[i] ? ldg_stream(wp[i] + k) : make_uint4(0, 0, 0, 0);
  }, blockIdx.x * 8 * NT);
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// A lane's eight float32 inputs of a span (8-byte aligned) as the three bf16 parts k_split leaves of them (split3 on
// the same pairs), or zeros.
__device__ __forceinline__ void split8(const float* p, bool ok, uint4& hi, uint4& mid, uint4& lo) {
  hi = mid = lo = make_uint4(0, 0, 0, 0);
  if (!ok) return;
  uint32_t* h = &hi.x;
  uint32_t* m = &mid.x;
  uint32_t* l = &lo.x;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 v = *reinterpret_cast<const float2*>(p + 2 * q);
    split3(v.x, v.y, h + q, m + q, l + q);
  }
}

// The same GEMM on up to 16 MT rows (a round of a large tree, docs/design/tree-verification.md phase 2): each span's
// weights loaded once and multiplied into MT tiles of 16 rows.  With the three parts and one chunk a warp
// (CH == kSkinnyChunks) a row's MMAs and sums are the ones k_gemm_skinny does for it (its chunk's spans and k16 steps
// in order, the parts smallest first, the chunks added in order; the rows beside it in a tile change nothing), so
// every row gets the bits of the 1..16-row kernel; it splits the float32 rows xf itself (split8: a third fewer bytes
// than the planes, which every block reads).  PARTS == 1 multiplies bf16 rows x.hi, over CH chunks of any count: what
// a pass only drafting reads takes (DV's canvas), deterministic but not those bits.  Several weights on the same rows
// go in one launch (MtGroups: a drafting pass's base weight and its adapter's a, or the adapters' b side by side):
// each block computes 8 NT columns of one of them, from its own columns of the input.  More than 64 rows (three parts:
// a round's, DSpine's walk) go in one launch too: blockIdx.y the block's 64 of them, every row's arithmetic the same.
template <int NT, int CH, int UNROLL, int MT, int PARTS>
__global__ void __launch_bounds__(32 * CH) k_gemm_skinny_mt(Parts x, const float* xf, int ldk, int K,
                                                            const __grid_constant__ MtGroups gs, int rows) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!bf16_mt_active<NT, CH, UNROLL, PARTS>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  static_assert(PARTS == 1 || (PARTS == 3 && CH == kSkinnyChunks), "the three parts keep k_gemm_skinny's chunks");
  __shared__ float red[CH][NT][32][4];
  int gi = 0;  // the block's weight (uniform)
  while (gi + 1 < gs.n && int(blockIdx.x) >= gs.g[gi + 1].first_block) ++gi;
  const MtOut& o = gs.g[gi];
  const int N = o.N;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = (int(blockIdx.x) - o.first_block) * 8 * NT;
  const int spans = K >> 5;
  const bf16* wp[NT];
  bool wv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = n_base + i * 8 + g;
    wv[i] = n < N;
    wp[i] = o.W + size_t(wv[i] ? n : 0) * K + 8 * t;
  }
  const size_t r0 = size_t(blockIdx.y) * 64;  // the block's first row
  rows = min(rows - int(r0), 64);
  const bf16* xh = PARTS == 1 ? x.hi + o.x_col + r0 * ldk : nullptr;
  const float* x32 = PARTS == 3 ? xf + o.x_col + r0 * ldk : nullptr;
  pdl_wait();  // the activations come from the kernel before
  const int s0 = warp * spans / CH, s1 = (warp + 1) * spans / CH;
  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[m][i][0] = acc[m][i][1] = acc[m][i][2] = acc[m][i][3] = 0.f;
#pragma unroll UNROLL
  for (int s = s0; s < s1; ++s) {
    const int k = s << 5;
    uint4 w[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) w[i] = wv[i] ? ldg_stream(wp[i] + k) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int ra_row = 16 * m + g, rb_row = ra_row + 8;
      const bool ra = ra_row < rows, rb = rb_row < rows;
      const size_t oa = size_t(ra_row) * ldk + 8 * t + k, ob = size_t(rb_row) * ldk + 8 * t + k;
      const uint4 z = make_uint4(0, 0, 0, 0);
      uint4 ah = z, am = z, al = z, bh = z, bm = z, bl = z;
      if constexpr (PARTS == 3) {
        split8(x32 + oa, ra, ah, am, al);
        split8(x32 + ob, rb, bh, bm, bl);
      } else {
        ah = ra ? *reinterpret_cast<const uint4*>(xh + oa) : z;
        bh = rb ? *reinterpret_cast<const uint4*>(xh + ob) : z;
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t hi[4], mid[4], lo[4];
        part_frags(ah, bh, j, hi);
        if constexpr (PARTS == 3) {
          part_frags(am, bm, j, mid);
          part_frags(al, bl, j, lo);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const uint32_t bf[2] = {j == 0 ? w[i].x : w[i].z, j == 0 ? w[i].y : w[i].w};
          if constexpr (PARTS == 3) {
            mma_bf16(acc[m][i], lo, bf);
            mma_bf16(acc[m][i], mid, bf);
          }
          mma_bf16(acc[m][i], hi, bf);
        }
      }
    }
  }
  // Tile by tile: the chunks' sums in order, as k_gemm_skinny adds them.
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (16 * m >= rows) break;  // uniform across the block
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[warp][i][lane][e] = acc[m][i][e];
    __syncthreads();
    for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
      const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
      float v = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c) v += red[c][i][l][e];
      const int r = 16 * m + (l >> 2) + (e >> 1) * 8, col = n_base + i * 8 + 2 * (l & 3) + (e & 1);
      if (r < rows && col < N) {
        const size_t at = (r0 + r) * o.ldy + col;
        if (o.y16) {
          o.y16[at] = __float2bfloat16_rn(v);
        } else {
          float* dst = o.y + at;
          *dst = o.accumulate ? *dst + v : v;
        }
      }
    }
    __syncthreads();  // red is written again for the next tile
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// The columns cols[0 .. *count) of k_gemm_skinny's result (the certified head's
// survivors): each with the bits k_gemm_skinny gives it, written
// to y's column cols[j].  A block takes groups of 8 NT listed columns in turn.
template <int NT, bool R16, int WARPS, int UNROLL>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_skinny_cols(Parts x, int ldk,
                                                                 const bf16* __restrict__ W, const int* cols,
                                                                 const int* count, int K, float* __restrict__ y,
                                                                 int ldy, int rows) {
  pdl_trigger();
  pdl_wait();  // the list comes from the kernel before
  const int N = *count;
  const int g = (threadIdx.x & 31) >> 2, t = threadIdx.x & 3;
  for (int base = blockIdx.x * 8 * NT; base < N; base += gridDim.x * 8 * NT) {
    __syncthreads();  // the previous group's partial sums are read
    const bf16* wp[NT];
    bool wv[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int j = base + i * 8 + g;
      wv[i] = j < N;
      wp[i] = W + size_t(wv[i] ? cols[j] : 0) * K + 8 * t;
    }
    skinny_body<NT, R16, WARPS, UNROLL>(x, ldk, N, K, y, ldy, rows, 0, [&](int i, int k) {
      return wv[i] ? ldg_stream(wp[i] + k) : make_uint4(0, 0, 0, 0);
    }, base, [&](int c) { return cols[c]; });
  }
}

// ---- W12: bf16 weights in 12 bits, lossless (SpecPilot, arXiv 2611.02501) ---------------------
// Per row n an exponent window e0[n] .. e0[n] + 15.  A weight in it is sign |
// exponent - e0 (4 bits) | mantissa (7 bits): the high plane holds sign,
// offset and the top 3 mantissa bits in a byte, the low plane the low 4
// mantissa bits (two weights a byte, the lower k in the low nibble).  Both
// planes are in lane order per 64 weights (two 32-wide spans): the 8 weights
// lane t takes from span 2p and from span 2p + 1 sit together (w12_hi_at,
// w12_lo_at), so a lane reads both spans in one 16-byte and one 8-byte load.  The
// rest (zeros, denormals, exponents outside the window) and in-window weights
// whose code is the marker (offset 15, mantissa 127) are escapes: the marker
// in the planes, the exact bits in the row's escape list (k, bf16), sorted by
// k.  Decoding gives the bf16 weights back bit for bit (make_head_w12 checks
// every one at load), so the GEMM's products and sums are those of the bf16
// kernel.  Qwen3.5-4B's head: 7.75e-5 of its weights are escapes, at most 5 a
// row; 12 bits a weight instead of 16.
// Where weight k of a row sits in its high-plane row (a byte) and low-plane row (a byte, nibble k & 1).
__host__ __device__ __forceinline__ int w12_hi_at(int k) {
  return (k & ~63) | ((k >> 3) & 3) << 4 | ((k >> 5) & 1) << 3 | (k & 7);
}
__host__ __device__ __forceinline__ int w12_lo_at(int k) { return w12_hi_at(k) >> 1; }

__device__ __forceinline__ uint32_t w12_pair(uint32_t hb2, uint32_t nib2, uint32_t E2) {
  // hb2: the two high-plane bytes in the low bytes of the two 16-bit halves; nib2 the low mantissa bits
  return ((hb2 & 0x00800080u) << 8) | (((hb2 & 0x007F007Fu) << 4) + E2) | nib2;
}

// The 8 weights k0 .. k0 + 7 of a row from 8 high-plane bytes h and 4 low-plane bytes l.
__device__ __forceinline__ uint4 w12_decode8(uint2 h, uint32_t l, uint32_t E2) {
  uint4 w;
  w.x = w12_pair(__byte_perm(h.x, 0, 0x4140), (l & 0xFu) | ((l & 0xF0u) << 12), E2);
  w.y = w12_pair(__byte_perm(h.x, 0, 0x4342), ((l >> 8) & 0xFu) | ((l & 0xF000u) << 4), E2);
  w.z = w12_pair(__byte_perm(h.y, 0, 0x4140), ((l >> 16) & 0xFu) | ((l >> 4) & 0xF0000u), E2);
  w.w = w12_pair(__byte_perm(h.y, 0, 0x4342), ((l >> 24) & 0xFu) | ((l >> 12) & 0xF0000u), E2);
  return w;
}

// Weights decoded as the marker (offset 15, mantissa 127): the exact bits from the row's escapes.
__device__ __noinline__ uint4 w12_patch(uint4 w, const W12& W, int n, int k0) {
  uint32_t* v = &w.x;
  const int a = W.esc_start[n], b = W.esc_start[n + 1];
#pragma unroll 1
  for (int j = 0; j < 8; ++j) {
    const uint32_t half = (v[j >> 1] >> (16 * (j & 1))) & 0x7FFFu;
    if (half != ((uint32_t(W.e0[n]) + 15u) << 7 | 0x7Fu)) continue;
#pragma unroll 1
    for (int e = a; e < b; ++e)
      if (W.esc_k[e] == k0 + j) {
        const int sh = 16 * (j & 1);
        v[j >> 1] = (v[j >> 1] & ~(0xFFFFu << sh)) | (uint32_t(W.esc_v[e]) << sh);
        break;
      }
  }
  return w;
}

// A high byte with offset 15 and mantissa bits 7 (escapes and the rare weights of the window's top
// binade): w12_patch looks at the whole code.
__device__ __forceinline__ bool w12_has_marker(uint2 h) {
  return (__vcmpeq4(h.x & 0x7F7F7F7Fu, 0x7F7F7F7Fu) | __vcmpeq4(h.y & 0x7F7F7F7Fu, 0x7F7F7F7Fu)) != 0;
}

template <int NT, bool R16, int WARPS, int UNROLL>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_skinny_w12(Parts x, int ldk, W12 W, int N, int K,
                                                                float* __restrict__ y, int ldy, int rows,
                                                                int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!bf16_skinny_active<NT, R16, WARPS, UNROLL, true>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  const int g = (threadIdx.x & 31) >> 2, t = threadIdx.x & 3;
  const uint8_t* hp[NT];
  const uint8_t* lp[NT];
  uint32_t E2[NT];
  int nr[NT];
  bool wv[NT];
  uint4 hc[NT];  // the lane's bytes of the current span pair
  uint2 lc[NT];
  int pc = -1;   // that pair (the same for every i)
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = blockIdx.x * 8 * NT + i * 8 + g;
    wv[i] = n < N;
    nr[i] = wv[i] ? n : 0;
    hp[i] = W.hi + size_t(nr[i]) * K + 16 * t;
    lp[i] = W.lo + size_t(nr[i]) * (K >> 1) + 8 * t;
    const uint32_t e = uint32_t(W.e0[nr[i]]) << 7;
    E2[i] = e | e << 16;
  }
  skinny_body<NT, R16, WARPS, UNROLL>(x, ldk, N, K, y, ldy, rows, accumulate, [&](int i, int k) {
    const int pair = k >> 6;
    if (pair != pc) {  // uniform: the first span of a pair, or a chunk starting at an odd span
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        hc[j] = wv[j] ? ldg_stream(hp[j] + 64 * size_t(pair)) : make_uint4(0, 0, 0, 0);
        lc[j] = wv[j] ? ldg_stream2(lp[j] + 32 * size_t(pair)) : make_uint2(0, 0);
      }
      pc = pair;
    }
    if (!wv[i]) return make_uint4(0, 0, 0, 0);
    const bool second = (k >> 5) & 1;
    const uint2 h = second ? make_uint2(hc[i].z, hc[i].w) : make_uint2(hc[i].x, hc[i].y);
    const uint32_t l = second ? lc[i].y : lc[i].x;
    const uint4 w = w12_decode8(h, l, E2[i]);
    return w12_has_marker(h) ? w12_patch(w, W, nr[i], k + 8 * t) : w;
  }, blockIdx.x * 8 * NT);
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// Encoding, one block per row (256 threads): the window with the most weights
// (the lowest e0 among equals), the planes, the row's escape count.
__global__ void k_w12_encode(const bf16* __restrict__ W, int K, uint8_t* __restrict__ hi, uint8_t* __restrict__ lo,
                             uint8_t* __restrict__ e0_out, int* __restrict__ esc_count) {
  __shared__ int hist[256];
  __shared__ int best_e0, n_esc;
  const int n = blockIdx.x, tid = threadIdx.x;
  const uint16_t* row = reinterpret_cast<const uint16_t*>(W) + size_t(n) * K;
  hist[tid] = 0;
  if (tid == 0) n_esc = 0;
  __syncthreads();
  for (int k = tid; k < K; k += blockDim.x) atomicAdd(&hist[(row[k] >> 7) & 0xFF], 1);
  __syncthreads();
  if (tid == 0) {  // exponents 1 .. 254 only: 0 (zero, denormal) and 255 (inf, nan) are escapes
    int best = -1, b0 = 1;
    for (int e0 = 1; e0 <= 239; ++e0) {
      int c = 0;
      for (int e = e0; e < e0 + 16; ++e) c += hist[e];
      if (c > best) best = c, b0 = e0;
    }
    best_e0 = b0;
    e0_out[n] = uint8_t(b0);
  }
  __syncthreads();
  const int e0 = best_e0;
  int mine = 0;
  for (int k2 = tid; k2 < K / 2; k2 += blockDim.x) {  // weights 2 k2, 2 k2 + 1: one low-plane byte
    uint8_t nib[2];
    for (int j = 0; j < 2; ++j) {
      const int k = 2 * k2 + j;
      const uint32_t b = row[k], e = (b >> 7) & 0xFF, m = b & 0x7F, off = e - e0;
      const bool in = e >= uint32_t(e0) && off < 16 && !(off == 15 && m == 0x7F);
      const uint32_t code = in ? (((b >> 8) & 0x80) | (off << 3) | (m >> 4)) : (((b >> 8) & 0x80) | 0x7F);
      hi[size_t(n) * K + w12_hi_at(k)] = uint8_t(code);
      nib[j] = uint8_t(in ? (m & 0xF) : 0xF);
      mine += !in;
    }
    lo[size_t(n) * (K / 2) + w12_lo_at(2 * k2)] = uint8_t(nib[0] | (nib[1] << 4));
  }
  atomicAdd(&n_esc, mine);
  __syncthreads();
  if (tid == 0) esc_count[n] = n_esc;
}

// The escape lists, one thread per row (few rows have any, few a row).
__global__ void k_w12_escapes(const bf16* __restrict__ W, int N, int K, const uint8_t* __restrict__ e0,
                              const int* __restrict__ esc_start, uint16_t* __restrict__ esc_k,
                              uint16_t* __restrict__ esc_v) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N || esc_start[n] == esc_start[n + 1]) return;
  const uint16_t* row = reinterpret_cast<const uint16_t*>(W) + size_t(n) * K;
  int at = esc_start[n];
  for (int k = 0; k < K; ++k) {
    const uint32_t b = row[k], e = (b >> 7) & 0xFF, m = b & 0x7F, off = e - e0[n];
    if (!(e >= e0[n] && off < 16 && !(off == 15 && m == 0x7F))) {
      esc_k[at] = uint16_t(k);
      esc_v[at] = uint16_t(b);
      ++at;
    }
  }
}

// Every weight decoded as the GEMM decodes it, against W: mismatches counted.
__global__ void k_w12_check(const bf16* __restrict__ W, int N, int K, W12 w12, unsigned* __restrict__ bad) {
  const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;  // 8 weights
  if (i >= size_t(N) * K) return;
  const int n = int(i / K), k = int(i % K);
  const size_t row_hi = size_t(n) * K, at = row_hi + w12_hi_at(k);
  const uint2 h = *reinterpret_cast<const uint2*>(w12.hi + at);
  const uint32_t l = *reinterpret_cast<const uint32_t*>(w12.lo + at / 2);
  const uint32_t e = uint32_t(w12.e0[n]) << 7;
  uint4 w = w12_decode8(h, l, e | e << 16);
  if (w12_has_marker(h)) w = w12_patch(w, w12, n, k);
  const uint4 ref = *reinterpret_cast<const uint4*>(W + i);
  if (w.x != ref.x || w.y != ref.y || w.z != ref.z || w.w != ref.w) atomicAdd(bad, 1u);
}

// ---- NVFP4 weights (W4A16): e2m1 * e4m3 block scale, exact in bf16 --------------------------
// An e2m1 value has at most 2 significant bits and an e4m3 scale at most 4,
// so their product is exact in bf16; the per-tensor scale is applied to the
// output columns after accumulation (a per-row factor of the weight matrix).

__global__ void k_fp4_decode_test(const uint32_t* in, uint32_t* out, int n) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) fp4x8_to_bf16(in[i], out + 4 * size_t(i));
}

// Skinny NVFP4 GEMM, 1..16 rows.  Like k_gemm_skinny, but:
//  * activations are plain bf16: their rounding is far below the 4-bit
//    weights' error, and it halves the MMAs;
//  * a span is SPAN (64 or 128) values of K; lane t holds SPAN/4 consecutive
//    values (8 or 16 bytes, one or two e4m3 scales); MMA step j uses the
//    lane's values 4j+{0,1} and 4j+{2,3};
//  * K is cut into CH chunks (summed in order).  Bits depend on (SPAN, CH)
//    only: every shape uses one fixed (SPAN, CH) for all its row counts and
//    configurations, so verification and plain decoding still agree exactly.
template <int SPAN, int CH, int NT, bool R16, int WARPS, int UNROLL>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_skinny_fp4(const bf16* xh, int ldk,
                                                                const uint8_t* __restrict__ Q,
                                                                const uint8_t* __restrict__ S,
                                                                const float* __restrict__ RS, int N, int K,
                                                                float* __restrict__ y, int ldy, int rows,
                                                                int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!fp4_skinny_active<SPAN, CH, NT, R16, WARPS, UNROLL>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  static_assert(CH % WARPS == 0, "warps must divide the chunks");
  // per lane: VPL values, STEPS k16 steps, WORDS fp4 words (8 values), BW bf16x2 words
  constexpr int CPW = CH / WARPS, VPL = SPAN / 4, STEPS = VPL / 4, WORDS = VPL / 8, BW = VPL / 2;
  __shared__ float red[CH][NT][32][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = blockIdx.x * 8 * NT;
  const int spans = K / SPAN;
  const bool ra = g < rows, rb = R16 && g + 8 < rows;
  const size_t oa = size_t(g) * ldk + VPL * t, ob = size_t(g + 8) * ldk + VPL * t;
  const uint8_t* qp[NT];
  const uint8_t* sp[NT];
  bool wv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = n_base + i * 8 + g;
    wv[i] = n < N;
    const size_t row = size_t(wv[i] ? n : 0);
    qp[i] = Q + row * (K >> 1) + (VPL / 2) * t;
    sp[i] = S + row * (K >> 4) + (VPL / 16) * t;
  }
  pdl_wait();  // the activations come from the kernel before
#pragma unroll 1
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp * CPW + c;
    const int s0 = chunk * spans / CH, s1 = (chunk + 1) * spans / CH;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll UNROLL
    for (int s = s0; s < s1; ++s) {
      const int k = s * SPAN;
      uint32_t wq[NT][WORDS];
      uint32_t sc[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if (SPAN == 128) {
          const uint4 v = wv[i] ? ldg_stream(qp[i] + (k >> 1)) : make_uint4(0, 0, 0, 0);
          wq[i][0] = v.x, wq[i][1] = v.y, wq[i][2] = v.z, wq[i][3] = v.w;
          sc[i] = wv[i] ? *reinterpret_cast<const uint16_t*>(sp[i] + (k >> 4)) : 0;
        } else {
          const uint2 v = wv[i] ? ldg_stream2(qp[i] + (k >> 1)) : make_uint2(0, 0);
          wq[i][0] = v.x, wq[i][1] = v.y;
          sc[i] = wv[i] ? sp[i][k >> 4] : 0;
        }
      }
      uint32_t av[BW], bv[BW];
#pragma unroll
      for (int q = 0; q < VPL / 8; ++q) {
        const uint4 z = make_uint4(0, 0, 0, 0);
        const uint4 a = ra ? *reinterpret_cast<const uint4*>(xh + oa + k + 8 * q) : z;
        const uint4 b = rb ? *reinterpret_cast<const uint4*>(xh + ob + k + 8 * q) : z;
        av[4 * q] = a.x, av[4 * q + 1] = a.y, av[4 * q + 2] = a.z, av[4 * q + 3] = a.w;
        bv[4 * q] = b.x, bv[4 * q + 1] = b.y, bv[4 * q + 2] = b.z, bv[4 * q + 3] = b.w;
      }
      uint32_t bw[NT][BW];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
#pragma unroll
        for (int q = 0; q < WORDS; ++q) fp4x8_to_bf16(wq[i][q], bw[i] + 4 * q);
#pragma unroll
        for (int b = 0; b < VPL / 16; ++b) {  // one scale per 16 values (8 bf16x2)
          const __nv_bfloat162 s2 = e4m3_bf16x2((sc[i] >> (8 * b)) & 0xFF);
#pragma unroll
          for (int e = 0; e < 8; ++e) bw[i][8 * b + e] = bf16x2_mul(bw[i][8 * b + e], s2);
        }
      }
#pragma unroll
      for (int j = 0; j < STEPS; ++j) {
        const uint32_t a[4] = {av[2 * j], bv[2 * j], av[2 * j + 1], bv[2 * j + 1]};
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const uint32_t bf[2] = {bw[i][2 * j], bw[i][2 * j + 1]};
          mma_bf16(acc[i], a, bf);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[chunk][i][lane][e] = acc[i][e];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
    const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) v += red[c][i][l][e];
    const int r = (l >> 2) + (e >> 1) * 8, col = n_base + i * 8 + 2 * (l & 3) + (e & 1);
    if (r < rows && col < N) {
      v *= RS[col];
      float* dst = y + size_t(r) * ldy + col;
      *dst = accumulate ? *dst + v : v;
    }
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// The same GEMM for 17 .. 16 MT rows (a large tree round, docs/design/tree-verification.md phase 2; a wide lookup
// round): each span's weights decoded once and multiplied into MT tiles of 16 rows, a row's MMAs and sums those
// k_gemm_skinny_fp4 does for it (the same span, chunk and step order; the rows beside it in a tile change nothing), so every row gets the
// bits of the 1..16-row kernel.  One chunk a warp (CH == WARPS): a warp holds its MT tiles' sums until the end.
template <int SPAN, int CH, int NT, int WARPS, int UNROLL, int MT>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_skinny_fp4_mt(const bf16* xh, int ldk,
                                                                   const uint8_t* __restrict__ Q,
                                                                   const uint8_t* __restrict__ S,
                                                                   const float* __restrict__ RS, int N, int K,
                                                                   float* __restrict__ y, int ldy, int rows,
                                                                   int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!fp4_mt_active<SPAN, CH>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  static_assert(CH == WARPS, "one chunk a warp");
  constexpr int VPL = SPAN / 4, WORDS = VPL / 8;
  __shared__ float red[CH][NT][32][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = blockIdx.x * 8 * NT;
  const int spans = K / SPAN;
  const uint8_t* qp[NT];
  const uint8_t* sp[NT];
  bool wv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = n_base + i * 8 + g;
    wv[i] = n < N;
    const size_t row = size_t(wv[i] ? n : 0);
    qp[i] = Q + row * (K >> 1) + (VPL / 2) * t;
    sp[i] = S + row * (K >> 4) + (VPL / 16) * t;
  }
  pdl_wait();  // the activations come from the kernel before
  const int chunk = warp;
  const int s0 = chunk * spans / CH, s1 = (chunk + 1) * spans / CH;
  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[m][i][0] = acc[m][i][1] = acc[m][i][2] = acc[m][i][3] = 0.f;
#pragma unroll UNROLL
  for (int s = s0; s < s1; ++s) {
    const int k = s * SPAN;
    uint32_t wq[NT][WORDS];
    uint32_t sc[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      if (SPAN == 128) {
        const uint4 v = wv[i] ? ldg_stream(qp[i] + (k >> 1)) : make_uint4(0, 0, 0, 0);
        wq[i][0] = v.x, wq[i][1] = v.y, wq[i][2] = v.z, wq[i][3] = v.w;
        sc[i] = wv[i] ? *reinterpret_cast<const uint16_t*>(sp[i] + (k >> 4)) : 0;
      } else {
        const uint2 v = wv[i] ? ldg_stream2(qp[i] + (k >> 1)) : make_uint2(0, 0);
        wq[i][0] = v.x, wq[i][1] = v.y;
        sc[i] = wv[i] ? sp[i][k >> 4] : 0;
      }
    }
    // A word of 8 values a lane at a time (two k16 steps): its weights decoded once for every tile, each tile's rows
    // loaded for them and multiplied into NT column tiles (the same steps in the same order as k_gemm_skinny_fp4).
#pragma unroll
    for (int q = 0; q < WORDS; ++q) {
      uint32_t bw[NT][4];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        fp4x8_to_bf16(wq[i][q], bw[i]);
        const __nv_bfloat162 s2 = e4m3_bf16x2((sc[i] >> (8 * (q >> 1))) & 0xFF);  // one scale per 16 values
#pragma unroll
        for (int e = 0; e < 4; ++e) bw[i][e] = bf16x2_mul(bw[i][e], s2);
      }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int ra_row = 16 * m + g, rb_row = ra_row + 8;
        const uint4 z = make_uint4(0, 0, 0, 0);
        const uint4 a = ra_row < rows ? *reinterpret_cast<const uint4*>(xh + size_t(ra_row) * ldk + VPL * t + k + 8 * q) : z;
        const uint4 b = rb_row < rows ? *reinterpret_cast<const uint4*>(xh + size_t(rb_row) * ldk + VPL * t + k + 8 * q) : z;
        const uint32_t a0[4] = {a.x, b.x, a.y, b.y}, a1[4] = {a.z, b.z, a.w, b.w};
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const uint32_t b0[2] = {bw[i][0], bw[i][1]}, b1[2] = {bw[i][2], bw[i][3]};
          mma_bf16(acc[m][i], a0, b0);
          mma_bf16(acc[m][i], a1, b1);
        }
      }
    }
  }
  // Tile by tile: the chunks' sums in order, as k_gemm_skinny_fp4 adds them.
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (16 * m >= rows) break;  // uniform across the block
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[chunk][i][lane][e] = acc[m][i][e];
    __syncthreads();
    for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
      const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
      float v = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c) v += red[c][i][l][e];
      const int r = 16 * m + (l >> 2) + (e >> 1) * 8, col = n_base + i * 8 + 2 * (l & 3) + (e & 1);
      if (r < rows && col < N) {
        v *= RS[col];
        float* dst = y + size_t(r) * ldy + col;
        *dst = accumulate ? *dst + v : v;
      }
    }
    __syncthreads();  // red is written again for the next tile
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- FP8 weights (e4m3 per row with a float scale): the output head, opt-in ----------------
// k_gemm_skinny with 8 bytes of e4m3 per lane and span instead of 16 of bf16:
// e4m3 values are exact in bf16, so the products are those of the bf16
// kernel on the rounded weights; the row's scale multiplies the column after
// the chunks are added.  The same fixed K chunks: every row count and
// configuration gives the same bits.
__device__ __forceinline__ void fp8x8_to_bf16(uint2 v, uint32_t out[4]) {
  uint32_t hw[4];
  e4m3x4_to_f16bits(v.x, hw);
  e4m3x4_to_f16bits(v.y, hw + 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hw[i]));
    out[i] = pack_bf16(f.x, f.y);
  }
}

template <int NT, bool R16, int WARPS, int UNROLL>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_skinny_fp8(Parts x, int ldk, const uint8_t* __restrict__ W,
                                                                const float* __restrict__ RS, int N, int K,
                                                                float* __restrict__ y, int ldy, int rows,
                                                                int accumulate) {
  pdl_trigger();
  static_assert(kSkinnyChunks % WARPS == 0, "warps must divide the chunks");
  constexpr int CPW = kSkinnyChunks / WARPS;
  __shared__ float red[kSkinnyChunks][NT][32][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = blockIdx.x * 8 * NT;
  const int spans = K >> 5;
  const bool ra = g < rows, rb = R16 && g + 8 < rows;
  const size_t oa = size_t(g) * ldk + 8 * t, ob = size_t(g + 8) * ldk + 8 * t;
  const uint8_t* wp[NT];
  bool wv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = n_base + i * 8 + g;
    wv[i] = n < N;
    wp[i] = W + size_t(wv[i] ? n : 0) * K + 8 * t;
  }
  pdl_wait();
#pragma unroll 1
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp * CPW + c;
    const int s0 = chunk * spans / kSkinnyChunks, s1 = (chunk + 1) * spans / kSkinnyChunks;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll UNROLL
    for (int s = s0; s < s1; ++s) {
      const int k = s << 5;
      uint32_t w[NT][4];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const uint2 raw = wv[i] ? ldg_stream2(wp[i] + k) : make_uint2(0, 0);
        fp8x8_to_bf16(raw, w[i]);
      }
      const uint4 z = make_uint4(0, 0, 0, 0);
      const uint4 ah = ra ? *reinterpret_cast<const uint4*>(x.hi + oa + k) : z;
      const uint4 am = ra ? *reinterpret_cast<const uint4*>(x.mid + oa + k) : z;
      const uint4 al = ra ? *reinterpret_cast<const uint4*>(x.lo + oa + k) : z;
      const uint4 bh = rb ? *reinterpret_cast<const uint4*>(x.hi + ob + k) : z;
      const uint4 bm = rb ? *reinterpret_cast<const uint4*>(x.mid + ob + k) : z;
      const uint4 bl = rb ? *reinterpret_cast<const uint4*>(x.lo + ob + k) : z;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t hi[4], mid[4], lo[4];
        part_frags(ah, bh, j, hi);
        part_frags(am, bm, j, mid);
        part_frags(al, bl, j, lo);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const uint32_t bf[2] = {w[i][2 * j], w[i][2 * j + 1]};
          mma_bf16(acc[i], lo, bf);
          mma_bf16(acc[i], mid, bf);
          mma_bf16(acc[i], hi, bf);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[chunk][i][lane][e] = acc[i][e];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
    const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < kSkinnyChunks; ++c) v += red[c][i][l][e];
    const int r = (l >> 2) + (e >> 1) * 8, col = n_base + i * 8 + 2 * (l & 3) + (e & 1);
    if (r < rows && col < N) {
      v *= RS[col];
      float* dst = y + size_t(r) * ldy + col;
      *dst = accumulate ? *dst + v : v;
    }
  }
}

// The output head in FP8, 1..16 rows, as basalt's decode window product on FP8 rows (window_gemm_sliced.cuh, MIT,
// itself after Infernix's fp8_a16_sliced_k_mma, Apache-2.0): a block owns 16 weight rows over the whole K, its 4 warps
// each a fixed 64 values of every 256 (warp w: values 64 w .. 64 w + 63 of the group), the warps' sums added in warp
// order.  Thread (g, c) holds weight rows g and g + 8 and row g of each 8-row tile of the input; in its 64 values it
// takes 16 c .. 16 c + 15 (16 bytes a weight row), step s's k-slots {2c, 2c + 1} values 16c + 4s, + 1 and {2c + 8,
// 2c + 9} values 16c + 4s + 2, + 3, and multiplies the same values of its input row: plain dot products.  The rows
// are bf16 (their hi part: the rounding is far below e4m3's, and below llama.cpp's 8-bit inputs to its head), e4m3
// is exact in bf16.  A row's sums depend on K alone: every row count gives the same bits.
// xh is written by the preceding kernel in the stream (PDL): no __restrict__ so ptxas will not hoist reads into the
// read-only cache before pdl_wait().
template <int NT>
__global__ void __launch_bounds__(128) k_gemm_fp8_sliced(const bf16* xh, int ldk,
                                                         const uint8_t* __restrict__ W, const float* __restrict__ RS,
                                                         int N, int K, float* __restrict__ y, int ldy, int rows,
                                                         int accumulate) {
  pdl_trigger();
  constexpr int KW = 4, PS = 20;  // warps (K slices); floats a column of a warp's sums (16 rows, padded)
  __shared__ float part[KW][8 * NT][PS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int row0 = blockIdx.x * 16, groups = K / 256;
  const int r0 = min(row0 + g, N - 1), r1 = min(row0 + g + 8, N - 1);  // rows past the last read the last
  const uint8_t* w0 = W + size_t(r0) * K + 64 * warp + 16 * c;
  const uint8_t* w1 = W + size_t(r1) * K + 64 * warp + 16 * c;
  constexpr int AHEAD = 4;  // groups of weights in flight: the first ones issued before the wait (they are constant)
  uint4 q0[AHEAD], q1[AHEAD];
#pragma unroll
  for (int i = 0; i < AHEAD; ++i)
    if (i < groups) q0[i] = ldg_stream(w0 + size_t(i) * 256), q1[i] = ldg_stream(w1 + size_t(i) * 256);
  pdl_wait();  // the input rows come from the kernel before
  bool live[NT];
  const bf16* xr[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    live[t] = t * 8 + g < rows;
    xr[t] = xh + size_t(live[t] ? t * 8 + g : 0) * ldk + 64 * warp + 16 * c;
  }
  float acc[NT][4] = {};
#pragma unroll 1
  for (int gb = 0; gb < groups; gb += AHEAD) {
#pragma unroll
    for (int i = 0; i < AHEAD; ++i) {
      const int grp = gb + i;
      if (grp >= groups) break;
      uint32_t a0[8], a1[8];
      fp8x8_to_bf16(make_uint2(q0[i].x, q0[i].y), a0);
      fp8x8_to_bf16(make_uint2(q0[i].z, q0[i].w), a0 + 4);
      fp8x8_to_bf16(make_uint2(q1[i].x, q1[i].y), a1);
      fp8x8_to_bf16(make_uint2(q1[i].z, q1[i].w), a1 + 4);
      if (grp + AHEAD < groups)
        q0[i] = ldg_stream(w0 + size_t(grp + AHEAD) * 256), q1[i] = ldg_stream(w1 + size_t(grp + AHEAD) * 256);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        if (t * 8 >= rows) continue;  // a tile wholly past the rows
        const uint4 z = make_uint4(0, 0, 0, 0);
        const uint4 xa = live[t] ? *reinterpret_cast<const uint4*>(xr[t] + size_t(grp) * 256) : z;
        const uint4 xb = live[t] ? *reinterpret_cast<const uint4*>(xr[t] + size_t(grp) * 256 + 8) : z;
        const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          const uint32_t a[4] = {a0[2 * st], a1[2 * st], a0[2 * st + 1], a1[2 * st + 1]};
          const uint32_t b[2] = {xw[2 * st], xw[2 * st + 1]};
          mma_bf16(acc[t], a, b);
        }
      }
    }
  }
  // acc[t][e]: weight row g + 8 (e >> 1), input row 8 t + 2 c + (e & 1)
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) part[warp][8 * t + 2 * c + (e & 1)][g + 8 * (e >> 1)] = acc[t][e];
  __syncthreads();
  for (int idx = threadIdx.x; idx < rows * 16; idx += 128) {
    const int col = idx >> 4, r = idx & 15, n = row0 + r;
    float v = part[0][col][r];
#pragma unroll
    for (int w = 1; w < KW; ++w) v += part[w][col][r];
    if (n < N) {
      v *= RS[n];
      float* dst = y + size_t(col) * ldy + n;
      *dst = accumulate ? *dst + v : v;
    }
  }
}

// grid N; block 256.  RS[n] = max|W[n]| / 448; codes e4m3(w / RS[n]), nearest.
__global__ void k_quantize_fp8_rows(const bf16* __restrict__ W, int K, uint8_t* __restrict__ Q, float* __restrict__ RS) {
  pdl_trigger();
  const int n = blockIdx.x;
  const bf16* w = W + size_t(n) * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(w[k])));
  amax = block_max(amax);
  const float rs = amax > 0.f ? amax / 448.f : 1.f;
  if (threadIdx.x == 0) RS[n] = rs;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    Q[size_t(n) * K + k] = uint8_t(__nv_cvt_float_to_fp8(__bfloat162float(w[k]) / rs, __NV_SATFINITE, __NV_E4M3));
}

// Activations for the skinny GEMMs: each float split into three bf16 parts (split3, as put() does), rows packed
// with stride K.
__global__ void k_split(const float* __restrict__ x, int ldx, int K, int total, uint32_t* __restrict__ hi,
                        uint32_t* __restrict__ mid, uint32_t* __restrict__ lo) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // one pair of values
  if (i >= total) return;
  const int r = i / (K / 2), c = 2 * (i - r * (K / 2));
  const float2 v = *reinterpret_cast<const float2*>(x + size_t(r) * ldx + c);
  split3(v.x, v.y, &hi[i], &mid[i], &lo[i]);
}

}  // namespace

// ---- launchers ------------------------------------------------------------------------------
void gemm_rows(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
               cudaStream_t st) {
  for (int r0 = 0; r0 < rows; r0 += 16) {
    const int rr = rows - r0 < 16 ? rows - r0 : 16;
    const float* xp = x + size_t(r0) * ldx;
    float* yp = y + size_t(r0) * ldy;
    dim3 grid(ceil_div(N, 16));
    if (rr <= 1) k_gemm_rows<1><<<grid, 128, 0, st>>>(xp, ldx, W, N, K, yp, ldy, rr, accumulate);
    else if (rr <= 2) k_gemm_rows<2><<<grid, 128, 0, st>>>(xp, ldx, W, N, K, yp, ldy, rr, accumulate);
    else if (rr <= 4) k_gemm_rows<4><<<grid, 128, 0, st>>>(xp, ldx, W, N, K, yp, ldy, rr, accumulate);
    else if (rr <= 8) k_gemm_rows<8><<<grid, 128, 0, st>>>(xp, ldx, W, N, K, yp, ldy, rr, accumulate);
    else k_gemm_rows<16><<<grid, 128, 0, st>>>(xp, ldx, W, N, K, yp, ldy, rr, accumulate);
  }
}

bool skinny_fits(const float* x, int ldx, const bf16* W, int K) {
  return (K % 32) == 0 && (ldx % 2) == 0 && (reinterpret_cast<uintptr_t>(x) % 8) == 0 &&
         (reinterpret_cast<uintptr_t>(W) % 16) == 0;
}

namespace {
// The scratch's three planes (split_out writes the same ones).
Parts parts_of(const Scratch& sc) { return {sc.split, sc.split + sc.split_cap, sc.split + 2 * sc.split_cap}; }

// Splits rows x K of x into the scratch; the three planes, stride K.
Parts split_rows(const Launch& L, const float* x, int ldx, int rows, int K) {
  const Scratch& sc = scratch(L);
  const size_t n = size_t(rows) * K;
  check_fits(n, sc.split_cap, "a 1..16-row GEMM");
  const int total = int(n / 2);
  const Parts p = parts_of(sc);
  k_split<<<ceil_div(total, 256), 256, 0, L.st>>>(
      x, ldx, K, total, reinterpret_cast<uint32_t*>(const_cast<bf16*>(p.hi)),
      reinterpret_cast<uint32_t*>(const_cast<bf16*>(p.mid)), reinterpret_cast<uint32_t*>(const_cast<bf16*>(p.lo)));
  return p;
}

// The input a fused producer left for x == nullptr.
Parts presplit(const Launch& L) { return parts_of(scratch(L)); }

// Rows of a round of more than 16 (gemm_fp4_rows) as bf16 in abuf: the high part split_rows() gives them (the same
// rounding), without the two parts after it, for as many rows as the prefill rows the scratch holds.
const bf16* rows_hi(const Launch& L, const float* x, int ldx, int rows, int K) {
  const Scratch& sc = scratch(L);
  check_fits(size_t(rows) * K, sc.abuf_cap, "a round's GEMM of more than 16 rows");
  rows_to_bf16(x, ldx, rows, K, sc.abuf, L.st);
  return sc.abuf;
}
}  // namespace

template <int NT, int WARPS, int UNROLL>
void launch_skinny(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                   const Launch& L) {
  const Parts p = x ? split_rows(L, x, ldx, rows, K) : presplit(L);  // x == nullptr: a producer split it
  const dim3 grid(ceil_div(N, 8 * NT));
  if (rows <= 8) {
    if constexpr (bf16_skinny_active<NT, false, WARPS, UNROLL, false>(0))
      launch_dependent(k_gemm_skinny<NT, false, WARPS, UNROLL>, grid, dim3(32 * WARPS), L, p, K, W, N, K, y, ldy, rows,
                       int(accumulate));
    else std::abort();
  } else {
    if constexpr (bf16_skinny_active<NT, true, WARPS, UNROLL, false>(0))
      launch_dependent(k_gemm_skinny<NT, true, WARPS, UNROLL>, grid, dim3(32 * WARPS), L, p, K, W, N, K, y, ldy, rows,
                       int(accumulate));
    else std::abort();
  }
}

void gemm_skinny_cfg(int cfg, const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows,
                     bool accumulate, const Launch& L) {
  switch (cfg) {
#define X(i, nt, w, u) \
  case i: launch_skinny<nt, w, u>(x, ldx, W, N, K, y, ldy, rows, accumulate, L); break;
    FLYX_BF16_SKINNY(X)
#undef X
    default: std::abort();
  }
}

void gemm_skinny_cols(const float* x, int ldx, const bf16* W, int K, const int* cols, const int* count, float* y,
                      int ldy, int rows, const Launch& L) {
  const Parts p = split_rows(L, x, ldx, rows, K);
  // 8 columns a block, 16 warps; the grid covers a few hundred survivors in one
  // pass and loops over more (any configuration gives the same bits).
  constexpr int kBlocks = 256;
  if (rows <= 8)
    k_gemm_skinny_cols<1, false, 16, 8><<<kBlocks, 512, 0, L.st>>>(p, K, W, cols, count, K, y, ldy, rows);
  else
    k_gemm_skinny_cols<1, true, 16, 8><<<kBlocks, 512, 0, L.st>>>(p, K, W, cols, count, K, y, ldy, rows);
}

// ---- 17..64 rows: k_gemm_skinny_mt -----------------------------------------------------------
namespace {
template <int NT, int CH, int UNROLL, int PARTS>
void launch_skinny_mt(Parts p, const float* xf, int ldk, int K, MtGroups gs, int rows, const Launch& L) {
  if constexpr (!bf16_mt_active<NT, CH, UNROLL, PARTS>(0)) {
    std::abort();  // no table reaches it on any GPU (a release build compiles only what they reach)
  } else {
    int blocks = 0;
    for (int i = 0; i < gs.n; ++i) {
      gs.g[i].first_block = blocks;
      blocks += ceil_div(gs.g[i].N, 8 * NT);
    }
    const dim3 grid(blocks, ceil_div(rows, 64)), block(32 * CH);
    auto run = [&](auto kernel) { launch_dependent(kernel, grid, block, L, p, xf, ldk, K, gs, rows); };
    if (rows <= 32) run(k_gemm_skinny_mt<NT, CH, UNROLL, 2, PARTS>);
    else if (rows <= 48) run(k_gemm_skinny_mt<NT, CH, UNROLL, 3, PARTS>);
    else run(k_gemm_skinny_mt<NT, CH, UNROLL, 4, PARTS>);
  }
}

void skinny_mt(int cfg, Parts p, const float* xf, int ldk, int K, const MtGroups& gs, int rows, const Launch& L) {
  switch (cfg) {
#define X(i, nt, ch, u, parts) \
  case i: launch_skinny_mt<nt, ch, u, parts>(p, xf, ldk, K, gs, rows, L); break;
    FLYX_BF16_MT(X)
#undef X
    default: std::abort();
  }
}
}  // namespace

const MtConfig kMtConfigs[] = {
#define X(i, nt, ch, u, parts) {nt, ch, u, parts},
    FLYX_BF16_MT(X)
#undef X
};
const int kMtConfigCount = FLYX_BF16_MT_COUNT;

void gemm_skinny_mt_cfg(int cfg, const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows,
                        bool accumulate, const Launch& L) {
  const bool three = cfg >= 0 && cfg < kMtConfigCount && kMtConfigs[cfg].parts == 3;
  // three parts: any rows, 64 a grid row; one (bf16 rows staged in abuf): 64 at most
  if (rows < 1 || rows > (three ? kMtMostRows : 64) || cfg < 0 || cfg >= kMtConfigCount || (three && !x)) std::abort();
  MtGroups gs;
  gs.n = 1;
  gs.g[0] = MtOut{W, y, nullptr, N, ldy, 0, int(accumulate), 0};
  // Three parts: the float32 rows as they are (the kernel splits them); one: as bf16 rows in abuf.
  if (three) return skinny_mt(cfg, Parts{}, x, ldx, K, gs, rows, L);
  skinny_mt(cfg, Parts{x ? rows_hi(L, x, ldx, rows, K) : scratch(L).abuf, nullptr, nullptr}, nullptr, K, K, gs, rows,
            L);
}

const bf16* stage_bf16(const float* x, int ldx, int rows, int K, const Launch& L) {
  return rows_hi(L, x, ldx, rows, K);
}

void gemm_draft_cfg(int cfg, const bf16* x, int ldx, int K, const MtGroups& gs, int rows, const Launch& L) {
  if (rows < 1 || rows > 64 || cfg < 0 || cfg >= kMtConfigCount || kMtConfigs[cfg].parts != 1 || gs.n < 1 ||
      gs.n > kMtGroups || K % 32 || ldx % 8 || reinterpret_cast<uintptr_t>(x) % 16)
    std::abort();
  skinny_mt(cfg, Parts{x, nullptr, nullptr}, nullptr, ldx, K, gs, rows, L);
}

template <int NT, int WARPS, int UNROLL>
void launch_skinny_w12(const float* x, int ldx, const W12& W, int N, int K, float* y, int ldy, int rows,
                       bool accumulate, const Launch& L) {
  const Parts p = x ? split_rows(L, x, ldx, rows, K) : presplit(L);
  const dim3 grid(ceil_div(N, 8 * NT));
  if (rows <= 8) {
    if constexpr (bf16_skinny_active<NT, false, WARPS, UNROLL, true>(0))
      launch_dependent(k_gemm_skinny_w12<NT, false, WARPS, UNROLL>, grid, dim3(32 * WARPS), L, p, K, W, N, K, y, ldy,
                       rows, int(accumulate));
    else std::abort();
  } else {
    if constexpr (bf16_skinny_active<NT, true, WARPS, UNROLL, true>(0))
      launch_dependent(k_gemm_skinny_w12<NT, true, WARPS, UNROLL>, grid, dim3(32 * WARPS), L, p, K, W, N, K, y, ldy,
                       rows, int(accumulate));
    else std::abort();
  }
}

void gemm_w12_skinny_cfg(int cfg, const float* x, int ldx, const W12& W, int N, int K, float* y, int ldy, int rows,
                         bool accumulate, const Launch& L) {
  switch (cfg) {
#define X(i, nt, w, u) \
  case i: launch_skinny_w12<nt, w, u>(x, ldx, W, N, K, y, ldy, rows, accumulate, L); break;
    FLYX_BF16_SKINNY(X)
#undef X
    default: std::abort();
  }
}

size_t w12_plane_bytes(int N, int K) { return size_t(N) * K + size_t(N) * K / 2; }

// Load-time helpers: CUDA errors come back as -1 / ~0u for the loader to report.
int w12_encode(const bf16* W, int N, int K, W12* out, int* esc_start_host, cudaStream_t st) {
  int* counts = out->esc_start;  // the device array (N + 1 ints) holds the counts first
  k_w12_encode<<<N, 256, 0, st>>>(W, K, out->hi, out->lo, out->e0, counts);
  if (cudaMemcpyAsync(esc_start_host, counts, sizeof(int) * N, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
      cudaStreamSynchronize(st) != cudaSuccess)
    return -1;
  int total = 0;
  for (int n = 0; n < N; ++n) {
    const int c = esc_start_host[n];
    esc_start_host[n] = total;
    total += c;
  }
  esc_start_host[N] = total;
  if (cudaMemcpyAsync(out->esc_start, esc_start_host, sizeof(int) * (N + 1), cudaMemcpyHostToDevice, st) !=
          cudaSuccess ||
      cudaStreamSynchronize(st) != cudaSuccess)
    return -1;
  return total;
}

int w12_finish(const bf16* W, int N, int K, W12* w, cudaStream_t st) {
  k_w12_escapes<<<ceil_div(N, 128), 128, 0, st>>>(W, N, K, w->e0, w->esc_start, w->esc_k, w->esc_v);
  unsigned* bad = nullptr;
  unsigned h = 0;
  if (cudaMalloc(&bad, sizeof(unsigned)) != cudaSuccess) return -1;
  const size_t groups = size_t(N) * K / 8;
  bool ok = cudaMemsetAsync(bad, 0, sizeof(unsigned), st) == cudaSuccess;
  if (ok) {
    k_w12_check<<<unsigned((groups + 255) / 256), 256, 0, st>>>(W, N, K, *w, bad);
    ok = cudaMemcpyAsync(&h, bad, sizeof(unsigned), cudaMemcpyDeviceToHost, st) == cudaSuccess &&
         cudaStreamSynchronize(st) == cudaSuccess;
  }
  cudaFree(bad);
  return ok ? int(h) : -1;
}

const Bf16Config kBf16Configs[] = {
#define X(i, nt, w, u) {nt, w, u},
    FLYX_BF16_SKINNY(X)
#undef X
};

const int kSkinnyConfigs = FLYX_BF16_SKINNY_COUNT;

bool gemm_skinny_cfg_compiled(int cfg, int rows, bool w12) {
#if FLYX_TUNE_CANDIDATES
  (void)cfg, (void)rows, (void)w12;
  return true;
#else
  const int arch = compiled_arch();
  if (arch <= 0 || cfg < 0 || cfg >= FLYX_BF16_SKINNY_COUNT) return false;
  const int b = tables::arch_bucket(arch), hi = rows > 8 ? 1 : 0;
  return ((w12 ? tables::kReachW12Skinny[b][hi] : tables::kReachBf16Skinny[b][hi]) & (1u << cfg)) != 0;
#endif
}

bool gemm_fp4_cfg_compiled(int cfg, int rows) {
#if FLYX_TUNE_CANDIDATES
  (void)cfg, (void)rows;
  return true;
#else
  const int arch = compiled_arch();
  if (arch <= 0 || cfg < 0 || cfg >= FLYX_FP4_SKINNY_COUNT) return false;
  const int b = tables::arch_bucket(arch), hi = rows > 8 ? 1 : 0;
  return (tables::kReachFp4Skinny[b][hi] & (1u << cfg)) != 0;
#endif
}

bool gemm_mt_cfg_compiled(int cfg) {
#if FLYX_TUNE_CANDIDATES
  (void)cfg;
  return true;
#else
  const int arch = compiled_arch();  // the loaded image's (a 120-virtual build's PTX on a newer GPU: 120's tables)
  if (arch <= 0 || cfg < 0 || cfg >= FLYX_BF16_MT_COUNT) return false;
  return (tables::kReachBf16Mt[tables::arch_bucket(arch)] & (1u << cfg)) != 0;
#endif
}

template <int SPAN, int CH, int NT, int WARPS, int UNROLL>
void launch_skinny_fp4(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K,
                       float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  const bf16* xh = (x ? split_rows(L, x, ldx, rows, K) : presplit(L)).hi;  // only the high part is used
  const dim3 grid(ceil_div(N, 8 * NT));
  if (rows <= 8) {
    if constexpr (fp4_skinny_active<SPAN, CH, NT, false, WARPS, UNROLL>(0))
      launch_dependent(k_gemm_skinny_fp4<SPAN, CH, NT, false, WARPS, UNROLL>, grid, dim3(32 * WARPS), L, xh, K, Q, S,
                       RS, N, K, y, ldy, rows, int(accumulate));
    else std::abort();
  } else {
    if constexpr (fp4_skinny_active<SPAN, CH, NT, true, WARPS, UNROLL>(0))
      launch_dependent(k_gemm_skinny_fp4<SPAN, CH, NT, true, WARPS, UNROLL>, grid, dim3(32 * WARPS), L, xh, K, Q, S,
                       RS, N, K, y, ldy, rows, int(accumulate));
    else std::abort();
  }
}

const Fp4Config kFp4Configs[] = {
#define X(i, span, ch, nt, w, u) {span, ch, nt, w, u},
    FLYX_FP4_SKINNY(X)
#undef X
};

const int kFp4ConfigCount = FLYX_FP4_SKINNY_COUNT;

void gemm_fp4_skinny_cfg(int cfg, const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N,
                         int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  switch (cfg) {
#define X(i, span, ch, nt, w, u) \
  case i: launch_skinny_fp4<span, ch, nt, w, u>(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate, L); break;
    FLYX_FP4_SKINNY(X)
#undef X
    default: std::abort();
  }
}

namespace {
template <int SPAN, int CH, int NT, int UNROLL, int MT>
void launch_skinny_fp4_mt(const bf16* xh, int ldk, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K,
                          float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  launch_dependent(k_gemm_skinny_fp4_mt<SPAN, CH, NT, CH, UNROLL, MT>, dim3(ceil_div(N, 8 * NT)), dim3(32 * CH), L, xh,
                   ldk, Q, S, RS, N, K, y, ldy, rows, int(accumulate));
}

// The configurations the many-row kernel is compiled for (one chunk a warp): the 27B's and the 4B's 9..16-row ones.
// Column tiles a warp, whatever the configuration's (they change no bits): each A fragment a warp loads feeds that
// many MMAs, the loads the limit with two (RTX PRO 6000, the 27B's gate and up projection: 64 rows 325 -> 157 us, 32
// rows 137 -> 92; its down projection, 16 warps of at most 128 registers, four: 64 rows 185 -> 145).
template <int MT>
bool skinny_fp4_mt(const Fp4Config& c, const bf16* xh, int ldk, const uint8_t* Q, const uint8_t* S, const float* RS,
                   int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  constexpr int kWide = MT == 2 ? 6 : 8;
#define FLYX_FP4_MT(SPAN, CH, NT)                                                                                 \
  if (c.span == SPAN && c.chunks == CH && c.warps == CH) {                                                         \
    launch_skinny_fp4_mt<SPAN, CH, NT, 1, MT>(xh, ldk, Q, S, RS, N, K, y, ldy, rows, accumulate, L);               \
    return true;                                                                                                   \
  }
  FLYX_FP4_MT(128, 8, kWide)
  FLYX_FP4_MT(128, 16, 4)
  FLYX_FP4_MT(128, 4, kWide)
#undef FLYX_FP4_MT
  return false;
}
}  // namespace

void gemm_fp4_rows(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K,
                   float* y, int ldy, int rows, bool accumulate, const Launch& L, int cfg) {
  if (rows <= 16 && cfg < 0) {
    gemm_fp4(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate, L);
    return;
  }
  // The shape's configuration for 9..16 rows: its K partition is every row count's (one partition a shape).
  if (cfg < 0) cfg = gemm_config(opts(L).tuning, 1, N, K, 16);
  const Fp4Config& c = kFp4Configs[cfg];
  if (K % 64 == 0 && ldx % 2 == 0 && reinterpret_cast<uintptr_t>(x) % 8 == 0) {
    const bf16* xh = rows <= 64 ? rows_hi(L, x, ldx, rows, K) : nullptr;
    const bool done = rows <= 32   ? skinny_fp4_mt<2>(c, xh, K, Q, S, RS, N, K, y, ldy, rows, accumulate, L)
                      : rows <= 48 ? skinny_fp4_mt<3>(c, xh, K, Q, S, RS, N, K, y, ldy, rows, accumulate, L)
                      : rows <= 64 && skinny_fp4_mt<4>(c, xh, K, Q, S, RS, N, K, y, ldy, rows, accumulate, L);
    if (done) return;
  }
  // Not compiled for this configuration (or more than 64 rows): 16 rows at a time, the same bits.
  for (int r0 = 0; r0 < rows; r0 += 16)
    gemm_fp4_skinny_cfg(cfg, x + size_t(r0) * ldx, ldx, Q, S, RS, N, K, y + size_t(r0) * ldy, ldy,
                        std::min(16, rows - r0), accumulate, L);
}

void quantize_fp8_rows(const bf16* W, int N, int K, uint8_t* Q, float* RS, cudaStream_t st) {
  k_quantize_fp8_rows<<<N, 256, 0, st>>>(W, K, Q, RS);
}

// 1..16 rows (K % 32 == 0); one configuration (16 warps, the bf16 head's): every one gives the same bits.
void gemm_fp8_skinny(const float* x, int ldx, const uint8_t* W, const float* RS, int N, int K, float* y, int ldy,
                     int rows, bool accumulate, const Launch& L) {
  const Parts p = split_rows(L, x, ldx, rows, K);
  if (gemm_fp8_skinny_rounds_input(K)) {  // the sliced kernel (bf16 rows); RTX PRO 6000, the 27B's head: 1.47 -> see the perf record
    if (rows <= 8)
      launch_dependent(k_gemm_fp8_sliced<1>, dim3(ceil_div(N, 16)), dim3(128), L, p.hi, K, W, RS, N, K, y, ldy, rows,
                       int(accumulate));
    else
      launch_dependent(k_gemm_fp8_sliced<2>, dim3(ceil_div(N, 16)), dim3(128), L, p.hi, K, W, RS, N, K, y, ldy, rows,
                       int(accumulate));
    return;
  }
  if (rows <= 8)
    launch_dependent(k_gemm_skinny_fp8<1, false, 16, 16>, dim3(ceil_div(N, 8)), dim3(512), L, p, K, W, RS, N, K, y,
                     ldy, rows, int(accumulate));
  else
    launch_dependent(k_gemm_skinny_fp8<2, true, 16, 8>, dim3(ceil_div(N, 16)), dim3(512), L, p, K, W, RS, N, K, y,
                     ldy, rows, int(accumulate));
}

void fp4_decode_test(const uint32_t* in, uint32_t* out, int n, cudaStream_t st) {
  k_fp4_decode_test<<<ceil_div(n, 128), 128, 0, st>>>(in, out, n);
}

}  // namespace flyx::cuda
