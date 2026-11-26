// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Device helpers the kernel families share (header-only: every translation
// unit compiles its own copy; no relocatable device code).
#pragma once

#include <cstdint>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

// Most candidates a row has (topk_rows, the DFlash2 selector).
constexpr int kTopMax = 16;

// A binary16 K/V cache's value (ops/dtype/ref.h round_kv16): round to nearest, saturating at the largest finite
// 65504.  A value past it would be an infinity, and one infinity in a cache makes every row that attends to it NaN
// (a drafter's block attends both ways: all of its rows).  An infinity saturates too; NaN stays NaN.
__device__ __forceinline__ __half kv16_of(float x) {
  return __float2half_rn(x > 65504.f ? 65504.f : x < -65504.f ? -65504.f : x);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over the block (blockDim.x a multiple of 32, <= 1024).  Every thread gets the result.
__device__ inline float block_sum(float v) {
  __shared__ float sh[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = lane < int(blockDim.x >> 5) ? sh[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) total = t;
  }
  __syncthreads();
  return total;
}

// N independent block sums in two barriers (bit-identical to N block_sum calls).
template <int N>
__device__ inline void block_sum_n(float (&v)[N]) {
  __shared__ float sh[N][32];
  __shared__ float total[N];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) sh[i][warp] = v[i];
  }
  __syncthreads();
  if (warp == 0) {
    const int nw = int(blockDim.x >> 5);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float t = lane < nw ? sh[i][lane] : 0.f;
      t = warp_sum(t);
      if (lane == 0) total[i] = t;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = total[i];
}

__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }
// A row value where rows are float32 or bf16 (a prefill pass's bf16 rows: residual, in-projection).
__device__ __forceinline__ float in_value(const float* p) { return *p; }
__device__ __forceinline__ float in_value(const bf16* p) { return __bfloat162float(*p); }
// A gate|up GEMM with SwiGLU in its epilogue (gemm_glu_staged) runs over 2 I
// columns whose 8-column groups alternate gate and up: column v reads weight
// row glu_row(v, I) and belongs to output j = glu_col(v), so a thread's n8
// tiles ni (gate, ni even) and ni + 1 (up) hold the same j.
__device__ __forceinline__ int glu_col(int v) { return ((v >> 4) << 3) | (v & 7); }
__device__ __forceinline__ int glu_row(int v, int I) { return ((v >> 3) & 1) * I + glu_col(v); }
__device__ __forceinline__ float sigm(float v) { return 1.f / (1.f + __expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(__expf(v)); }

__device__ __forceinline__ int slot_of(const DevParams* P, int sel) {
  switch (sel) {
    case kSlotCur: return flyx_round_slot(P->round);
    case kSlotLast: return P->last_slot;
    case kSlotAfterLast: return flyx_round_slot(P->last_slot + 1);
    case kSlotNext: return flyx_round_slot(P->round + 1);
    default: return sel - kSlotExplicit;
  }
}

__device__ __forceinline__ int replay_slot_of(const DevParams* P, int sel) {
  switch (sel) {
    case kSlotCur: return P->round % kGdnReplaySlots;
    case kSlotLast: return P->last_round % kGdnReplaySlots;
    default: return sel - kSlotExplicit;
  }
}

__device__ __forceinline__ float* slot_base(const GdnBufs& b, const DevParams* P, int sel) {
  return b.slot_stride ? b.base + size_t(replay_slot_of(P, sel)) * b.slot_stride : b.base;
}

__device__ __forceinline__ void unpack8(uint4 raw, float* w) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(h[i]);
    w[2 * i] = f.x;
    w[2 * i + 1] = f.y;
  }
}

// Rotary angle in double (positions reach 2^18), then single-precision sin/cos.  rope_inv: pair i's factor, which
// rope_table keeps (made once with this function: the same bits as the pow in a launch).
__device__ __forceinline__ double rope_inv(int i, int rot_dim, double theta) { return pow(theta, -2.0 * i / rot_dim); }
__device__ __forceinline__ void rope_cs_inv(int pos, double inv, float* c, float* s) {
  double a = fmod(double(pos) * inv, 6.283185307179586);
  sincosf(float(a), s, c);
}
__device__ __forceinline__ void rope_cs(int pos, int i, int rot_dim, double theta, float* c, float* s) {
  rope_cs_inv(pos, rope_inv(i, rot_dim, theta), c, s);
}

// ---- NVFP4 quantization (weights in quantize_nvfp4, activations in quantize_rows_nvfp4) ----
// Nearest e2m1 code of x (ties to the even code): magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
__host__ __device__ inline uint8_t e2m1_nearest(float x) {
  const float a = fabsf(x);
  const uint8_t m = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5
                  : a <= 5.f   ? 6 : 7;
  return (x < 0.f && m) ? uint8_t(m | 8) : m;
}
// An activation row's scale g = kNvfp4RowRange / max|x| keeps every block scale within e4m3.
constexpr float kNvfp4RowRange = 6.f * 448.f;
// The e4m3 scale of a 16-value block: max|block| / 6 * g, nearest.
__host__ __device__ inline uint8_t nvfp4_block_scale(float bmax, float gs) {
#ifdef __CUDA_ARCH__
  const float v = __fmul_rn(__fdiv_rn(bmax, 6.f), gs);
#else
  const float v = (bmax / 6.f) * gs;
#endif
  return uint8_t(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// Four finite e4m3 bytes -> two packed binary16x2 words (exact for all 254 finite codes, including ±0 and subnormals;
// callers only pass SATFINITE codes, never 0x7f/0xff NaN).  sm_89+ lowers to hardware cvt.rn.f16x2.e4m3x2; on Ampere
// (sm_80/86) CUDA's software __nv_cvt_fp8x2_to_halfraw2 expands to ~25 bit-manipulation instructions and spills
// registers inside k_flash_gqa.  Shifting each finite byte into a binary16 lane at exponent bias -8 and scaling by
// 2^8 (0x5c00 = 256.0h) produces identical binary16 bits in 4 branchless instructions.
__device__ __forceinline__ void e4m3x4_to_f16bits_sw(uint32_t w, uint32_t out[2]) {
  const uint32_t u0 = __byte_perm(w, 0u, 0x5140u);
  const uint32_t u1 = __byte_perm(w, 0u, 0x5342u);
  const uint32_t r0 = ((u0 & 0x00800080u) << 8) | ((u0 & 0x007f007fu) << 7);
  const uint32_t r1 = ((u1 & 0x00800080u) << 8) | ((u1 & 0x007f007fu) << 7);
  const uint32_t s256 = 0x5c005c00u;
  const __half2 h0 = __hmul2(*reinterpret_cast<const __half2*>(&r0), *reinterpret_cast<const __half2*>(&s256));
  const __half2 h1 = __hmul2(*reinterpret_cast<const __half2*>(&r1), *reinterpret_cast<const __half2*>(&s256));
  out[0] = *reinterpret_cast<const uint32_t*>(&h0);
  out[1] = *reinterpret_cast<const uint32_t*>(&h1);
}
__device__ __forceinline__ void e4m3x4_to_f16bits(uint32_t w, uint32_t out[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w & 0xFFFFu), __NV_E4M3);
  const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w >> 16), __NV_E4M3);
  out[0] = uint32_t(lo.x) | (uint32_t(lo.y) << 16);
  out[1] = uint32_t(hi.x) | (uint32_t(hi.y) << 16);
#else
  e4m3x4_to_f16bits_sw(w, out);
#endif
}

// Four e4m3 values (byte i = value i) -> float (exact through half).
__device__ __forceinline__ void e4m3x4_to_float(uint32_t w, float* f) {
  uint32_t b[2];
  e4m3x4_to_f16bits(w, b);
  const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&b[0]));
  const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&b[1]));
  f[0] = x0.x;
  f[1] = x0.y;
  f[2] = x1.x;
  f[3] = x1.y;
}
__device__ __forceinline__ float e4m3_to_float_sw(uint8_t c) {
  const uint16_t raw = uint16_t(((c & 0x80u) << 8) | ((c & 0x7fu) << 7));
  return __half2float(__ushort_as_half(raw)) * 256.f;
}
__device__ __forceinline__ float e4m3_to_float(uint8_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(c), __NV_E4M3);
  return __half2float(__half(h));
#else
  return e4m3_to_float_sw(c);
#endif
}

// The rows of a round with inputs in 8 bits as k_stage_q8_round leaves them: row r's K integers at r K, in
// spans of 32 groups of 32 values (the last span may have fewer): the span's groups' first 16 values, group
// after group, then their last 16.  So a warp whose lane l takes group l of a span reads the span in two loads
// of consecutive bytes.  The byte of value 16 half + i of group `unit` in its row:
__device__ __forceinline__ int round_q8_at(int unit, int units, int half) {
  const int span = unit >> 5, in_span = min(32, units - (span << 5));
  return (span << 10) + half * 16 * in_span + 16 * (unit & 31);
}

// Where a producer writes a value bound for a GEMM: float32, or the GEMM's
// input already rounded at element `at` of rows with stride K: three bf16
// parts for a skinny GEMM (exactly as k_split would; up to 16 rows), or the
// high part alone for the prefill GEMM (exactly as rows_to_bf16 would); or, for
// GGML block weights with 8-bit inputs (q8 non-null), the rows in 8 bits
// exactly as k_stage_q8 (a prefill pass) or k_stage_q8_round (a round:
// q8_sums non-null) leaves them, which a warp writes a group of 32 at a time
// (put_q8, put_q8_pairs).
struct Out {
  float* y;
  bf16 *hi, *mid, *lo;        // mid null: the high part only
  int8_t* q8 = nullptr;       // [rows][K] integers
  float* q8_d = nullptr;      // [rows][K / 32]: a group's scale d
  float* q8_sums = nullptr;   // a round's: d times the sum of each 16 integers, [rows][K / 16]
  int q8_k = 0;               // K
  // [rows][K] in float32: a prefill pass's rows when NVFP4 weights take NVFP4 inputs (Scratch::af32), which the
  // W4A4 kernels round once from these values
  float* f32 = nullptr;
};

// A group of 32 values in 8 bits, as the staging kernels round them (gemm_blocks.cu): d = (their largest
// magnitude) / 127 and p = the nearest integer of value / d, both in float32, written where the GEMM reads them.
// p: this lane's integer, `at` (0..31) its place in the group; s16: the sum of its half's 16 integers (the half
// at / 16); first: the lane that writes the group's d and its halves' sums.
__device__ __forceinline__ void q8_store(const Out& o, int row, int grp, int at, int p, float dg, int s16, bool first16,
                                         bool first) {
  const int units = o.q8_k / 32;
  if (o.q8_sums) {
    o.q8[size_t(row) * o.q8_k + round_q8_at(grp, units, at >> 4) + (at & 15)] = int8_t(p);
    if (first16) o.q8_sums[(size_t(row) * units + grp) * 2 + (at >> 4)] = dg * float(s16);
  } else {
    o.q8[size_t(row) * o.q8_k + 32 * grp + at] = int8_t(p);
  }
  if (first) o.q8_d[size_t(row) * units + grp] = dg;
}
__device__ __forceinline__ int q8_of(float v, float dg) {
  return dg > 0.f && dg <= 3.402823466e38f ? __float2int_rn(v / dg) : 0;
}
// A warp's 32 values, lane l value 32 grp + l of the row (every lane of the warp calls it).
__device__ __forceinline__ void put_q8(const Out& o, int row, int grp, float v) {
  const int lane = threadIdx.x & 31;
  float m = fabsf(v);
#pragma unroll
  for (int s = 16; s; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
  const float dg = m / 127.f;
  const int p = q8_of(v, dg);
  if (!o.q8_sums) {
    o.q8[size_t(row) * o.q8_k + 32 * grp + lane] = int8_t(p);
    if (!lane) o.q8_d[size_t(row) * (o.q8_k / 32) + grp] = dg;
    return;
  }
  int s16 = p;
#pragma unroll
  for (int s = 8; s; s >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, s);
  q8_store(o, row, grp, lane, p, dg, s16, !(lane & 15), !lane);
}
// A warp's 64 values in pairs, lane l values 2 l and 2 l + 1 from 64 grp2 of the row: lanes 0..15 the group
// 2 grp2, lanes 16..31 the next (every lane of the warp calls it).
__device__ __forceinline__ void put_q8_pairs(const Out& o, int row, int grp2, float a, float b) {
  const int lane = threadIdx.x & 31, l = lane & 15, grp = 2 * grp2 + (lane >> 4);
  float m = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
  for (int s = 8; s; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));  // within the half warp
  const float dg = m / 127.f;
  const int pa = q8_of(a, dg), pb = q8_of(b, dg);
  if (!o.q8_sums) {
    // Prefill q8 rows: 2*l is even and o.q8_k is a multiple of 32, so a pair writes as one 16-bit store.
    const uint16_t packed = static_cast<uint16_t>(static_cast<uint8_t>(pa) | (static_cast<uint16_t>(static_cast<uint8_t>(pb)) << 8));
    *reinterpret_cast<uint16_t*>(o.q8 + size_t(row) * o.q8_k + 32 * grp + 2 * l) = packed;
    if (!l) o.q8_d[size_t(row) * (o.q8_k / 32) + grp] = dg;
    return;
  }
  int s16 = pa + pb;  // the group's half: 8 lanes
#pragma unroll
  for (int s = 4; s; s >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, s);
  q8_store(o, row, grp, 2 * l, pa, dg, s16, !(l & 7), !l);
  q8_store(o, row, grp, 2 * l + 1, pb, dg, s16, false, false);
}

__device__ __forceinline__ void put(const Out& o, size_t at_y, size_t at_split, float v) {
  if (o.f32) {
    o.f32[at_split] = v;
  } else if (o.hi) {
    const bf16 h = __float2bfloat16_rn(v);
    o.hi[at_split] = h;
    if (o.mid) {
      const float r = v - __bfloat162float(h);
      const bf16 m = __float2bfloat16_rn(r);
      o.mid[at_split] = m;
      o.lo[at_split] = __float2bfloat16_rn(r - __bfloat162float(m));
    }
  } else {
    o.y[at_y] = v;
  }
}

__device__ inline float block_max(float v) {
  __shared__ float sh[32];
  __shared__ float res;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = lane < int(blockDim.x >> 5) ? sh[lane] : -INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (lane == 0) res = t;
  }
  __syncthreads();
  return res;
}

// N independent block maxima in two barriers (bit-identical to N block_max calls).
template <int N>
__device__ inline void block_max_n(float (&v)[N]) {
  __shared__ float sh[N][32];
  __shared__ float res[N];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v[i] = fmaxf(v[i], __shfl_xor_sync(0xffffffffu, v[i], o));
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) sh[i][warp] = v[i];
  }
  __syncthreads();
  if (warp == 0) {
    const int nw = int(blockDim.x >> 5);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float t = lane < nw ? sh[i][lane] : -INFINITY;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
      if (lane == 0) res[i] = t;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = res[i];
}

// ---- programmatic dependent launch (sm_90+) ------------------------------------------------
// A kernel launched with launch_dependent() may start while the kernel before
// it still runs (its launch and set-up overlap that kernel's tail); it calls
// pdl_wait() before it reads anything an earlier kernel wrote.  Every kernel
// calls pdl_trigger() on entry, which lets such a successor start.  Both are
// no-ops for kernels launched the ordinary way, and below sm_90.  What the
// kernel before writes is never read through a const __restrict__ pointer
// (or __ldg): the compiler takes such data as read-only for the whole kernel
// and may load it ahead of the wait (seen: k_gdn_round read stale gate
// inputs, golden_cuda_nvfp4 failed).  (Streaming
// the next GEMM's weights into L2 during the wait was measured: 4-8% slower.)
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::);
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// D (+)= A B on mma.m16n8k16: bf16 inputs, fp32 accumulators.
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D (+)= A B on mma.m16n8k16: binary16 inputs, fp32 accumulators (the K/V caches' type; bf16's rate here).
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D (+)= A B on mma.m16n8k32: e4m3 inputs, fp32 accumulators (FP8 tensor cores: sm_89 and later; callers check
// KernelOptions::fp8_mma, which leaves out sm_89, whose sums miss the op tests' tolerance).
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c, (void)a, (void)b;
  __trap();
#endif
}

// Two values as a bf16x2 word (rounded to nearest), and a split into high
// parts and remainders (the 1..16-row GEMMs' activations).
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// Two values as a binary16x2 word (rounded to nearest).
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void split2(float a, float b, uint32_t* hi, uint32_t* lo) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  float2 hf = __bfloat1622float2(h);
  *hi = *reinterpret_cast<uint32_t*>(&h);
  *lo = pack_bf16(a - hf.x, b - hf.y);
}
// Three parts (the 1..16-row GEMMs' activations): the high parts, their
// remainders and the remainders' remainders, 24 significant bits together (a
// float32's); each subtraction is exact.  put() makes the same parts.
__device__ __forceinline__ void split3(float a, float b, uint32_t* hi, uint32_t* mid, uint32_t* lo) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  const float2 hf = __bfloat1622float2(h);
  const float ra = a - hf.x, rb = b - hf.y;
  __nv_bfloat162 m = __floats2bfloat162_rn(ra, rb);
  const float2 mf = __bfloat1622float2(m);
  *hi = *reinterpret_cast<uint32_t*>(&h);
  *mid = *reinterpret_cast<uint32_t*>(&m);
  *lo = pack_bf16(ra - mf.x, rb - mf.y);
}

// Streaming loads of weights read once: not kept in L1.
__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// 8 e2m1 codes (nibble i = value i) -> 4 bf16x2 (values 2p, 2p+1 in out[p]).
__device__ __forceinline__ void fp4x8_to_bf16(uint32_t w, uint32_t out[4]) {
  // bf16 of the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6: high bytes and low bytes as byte tables
  constexpr uint32_t kHi0 = 0x3F3F3F00u, kHi1 = 0x40404040u, kLo0 = 0xC0800000u, kLo1 = 0xC0804000u;
  const uint32_t m = w & 0x77777777u;  // 3-bit selectors (bit 3 of a prmt selector would sign-extend)
  const uint32_t hb0 = __byte_perm(kHi0, kHi1, m & 0xFFFFu), lb0 = __byte_perm(kLo0, kLo1, m & 0xFFFFu);
  const uint32_t hb1 = __byte_perm(kHi0, kHi1, m >> 16), lb1 = __byte_perm(kLo0, kLo1, m >> 16);
  out[0] = __byte_perm(lb0, hb0, 0x5140);
  out[1] = __byte_perm(lb0, hb0, 0x7362);
  out[2] = __byte_perm(lb1, hb1, 0x5140);
  out[3] = __byte_perm(lb1, hb1, 0x7362);
  const uint32_t E = (w << 4) & 0x80808080u;  // byte b bit 7: sign of value 2b
  const uint32_t O = w & 0x80808080u;         // byte b bit 7: sign of value 2b + 1
  out[0] |= __byte_perm(E, O, 0x4000) & 0x80008000u;
  out[1] |= __byte_perm(E, O, 0x5010) & 0x80008000u;
  out[2] |= __byte_perm(E, O, 0x6020) & 0x80008000u;
  out[3] |= __byte_perm(E, O, 0x7030) & 0x80008000u;
}

__device__ __forceinline__ __nv_bfloat162 e4m3_bf16x2(uint32_t b) {
  // normal: 2^(e-7) (1 + m/8) -> bf16 exponent e + 120, mantissa m << 4; subnormal: m * 2^-9
  uint32_t h;
  if (b & 0x78) {
    h = ((b & 0x7Fu) << 4) + (120u << 7);
  } else {
    const __nv_bfloat16 v = __float2bfloat16(float(b & 7) * 0.001953125f);
    h = *reinterpret_cast<const uint16_t*>(&v);
  }
  h |= (b & 0x80u) << 8;
  const uint32_t two = h | (h << 16);
  return *reinterpret_cast<const __nv_bfloat162*>(&two);
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, __nv_bfloat162 s) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), s);
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t ldg_stream1(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 ldg_stream2(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

// cp.async into shared memory: 4 and 8 bytes through L1 (.ca), 16 around it (.cg).  With `pred` false the
// destination gets zeros and nothing is read.
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool pred) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem, bool pred) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(pred ? 8 : 0));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int n = pred ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

// ldmatrix_x4 transposed on the way (the A operand of O^T += V^T P^T from a value tile of keys x dims).
__device__ __forceinline__ void ldmatrix_x4_t(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

// An 8x8 matrix of 16-bit values transposed across the warp (an MMA's accumulator into a B operand).
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}

__device__ __forceinline__ uint32_t pack_halves(__half lo, __half hi) {
  return uint32_t(__half_as_ushort(lo)) | (uint32_t(__half_as_ushort(hi)) << 16);
}

// IEEE-754 float order mapped to monotonic uint32_t (larger float -> larger unsigned key, for radix select).
__device__ __forceinline__ uint32_t float_order_u32(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Tie-breaking score comparison: higher score wins; on equal scores, lower index wins.
__device__ __forceinline__ bool score_better(float va, int ia, float vb, int ib) {
  return va > vb || (va == vb && ia < ib);
}

}  // namespace flyx::cuda
