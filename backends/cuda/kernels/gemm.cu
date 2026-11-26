// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GEMM dispatch (path by rows and shape, configuration from the tuning tables), NVFP4 quantization.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "flyx/gemm_tables.inc"
#include "backends/cuda/kernels/gemm_internal.h"

namespace flyx::cuda {
namespace {

// ---- NVFP4 copy of a bf16 matrix ---------------------------------------------------------
__device__ __forceinline__ float e4m3_value(uint8_t b) {
  const int e = (b >> 3) & 0xF, m = b & 7;
  const float v = e == 0 ? m * 0.125f * 0.015625f : (1.f + m * 0.125f) * exp2f(float(e - 7));
  return (b & 0x80) ? -v : v;
}

// grid N; block 256.  Row n: RS[n] = max|W[n]| / (6 * 448) keeps the block
// scales in e4m3 range; each block of 16 gets scale e4m3(max / 6 / RS[n]) and
// codes e2m1(w / (scale * RS[n])), both rounded to nearest.
__global__ void k_quantize_nvfp4(const bf16* __restrict__ W, int K, uint8_t* __restrict__ Q, uint8_t* __restrict__ S,
                                 float* __restrict__ RS) {
  pdl_trigger();
  const int n = blockIdx.x;
  const bf16* w = W + size_t(n) * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(w[k])));
  amax = block_max(amax);
  const float rs = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
  if (threadIdx.x == 0) RS[n] = rs;
  for (int b = threadIdx.x; b < K / 16; b += blockDim.x) {
    float v[16], bmax = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      v[e] = __bfloat162float(w[b * 16 + e]);
      bmax = fmaxf(bmax, fabsf(v[e]));
    }
    const uint8_t sc = __nv_cvt_float_to_fp8(bmax / 6.f / rs, __NV_SATFINITE, __NV_E4M3);
    const float scale = e4m3_value(sc) * rs, inv = scale > 0.f ? 1.f / scale : 0.f;
    S[size_t(n) * (K / 16) + b] = sc;
#pragma unroll
    for (int e = 0; e < 16; e += 2)
      Q[size_t(n) * (K / 2) + b * 8 + e / 2] = uint8_t(e2m1_nearest(v[e] * inv) | (e2m1_nearest(v[e + 1] * inv) << 4));
  }
}

// ---- Q8_0 copy of a bf16 matrix -----------------------------------------------------------
// A thread a block of 32 weights: d = binary16(max |w| / 127), q = round(w / d) (|q| <= 127), the codes as the
// kernels hold a Q8_0 block's payload (32 bytes) and d in the scales' plane; weight = d q.
__global__ void k_quantize_q8_0(const bf16* __restrict__ W, size_t blocks, uint8_t* __restrict__ payload,
                                uint16_t* __restrict__ scales) {
  const size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const bf16* w = W + b * 32;
  float v[32], amax = 0.f;
#pragma unroll
  for (int e = 0; e < 32; ++e) {
    v[e] = __bfloat162float(w[e]);
    amax = fmaxf(amax, fabsf(v[e]));
  }
  const __half dh = __float2half_rn(amax / 127.f);
  const float d = __half2float(dh), inv = d > 0.f ? 1.f / d : 0.f;
  scales[b] = __half_as_ushort(dh);
  uint32_t words[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t word = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q = max(-127, min(127, __float2int_rn(v[4 * i + j] * inv)));
      word |= uint32_t(uint8_t(int8_t(q))) << (8 * j);
    }
    words[i] = word;
  }
  uint4* out = reinterpret_cast<uint4*>(payload + b * 32);
  out[0] = make_uint4(words[0], words[1], words[2], words[3]);
  out[1] = make_uint4(words[4], words[5], words[6], words[7]);
}

}  // namespace

void quantize_q8_0(const bf16* W, int N, int K, uint8_t* payload, uint16_t* scales, cudaStream_t st) {
  const size_t blocks = size_t(N) * (K / 32);
  k_quantize_q8_0<<<unsigned((blocks + 255) / 256), 256, 0, st>>>(W, blocks, payload, scales);
}

namespace {

// The configuration for one GEMM: the table of this GPU (or the one it
// inherits) for the shape, else the rules for shapes no table lists (this
// GPU's, else the defaults).  Never depends on anything but the table, the
// format, the shape and the row count.
// k_gemm_skinny_mt's formats (17..64 rows, a bf16_mt configuration): a round's in three parts, a drafting pass's in
// one, its chunks K's (tables::draft_chunks) whatever the table.
constexpr int kFmtBf16Rows = 3, kFmtBf16Draft = 4;

int pick_config(int sku, int fmt, int N, int K, int rows) {
  for (int s = sku; s >= 0; s = tables::kSkus[s].parent)
    for (int i = 0; i < tables::kRuleCount; ++i) {
      const auto& r = tables::kRules[i];
      if (r.sku == s && r.fmt == fmt && r.n == N && r.k == K && rows >= r.lo && rows <= r.hi) return r.cfg;
    }
  if (fmt == 1) {
    const int s = tables::rules_sku(sku, tables::kFamilyNvfp4);
    for (const auto& r : tables::kNvfp4Rules)
      if (r.sku == s) return K % 128 == 0 ? r.k128 : r.k64;
  }
  if (fmt == kFmtBf16Rows) {
    const int s = tables::rules_sku(sku, tables::kFamilyBf16Rows);
    for (const auto& r : tables::kBf16RowsRules)
      if (r.sku == s && rows >= r.lo && rows <= r.hi) return r.cfg;
    std::abort();  // the generator checks that the rules cover 17..64 rows
  }
  if (fmt == kFmtBf16Draft) {
    const int s = tables::rules_sku(sku, tables::kFamilyBf16Draft);
    for (const auto& r : tables::kBf16DraftRules)
      if (r.sku == s && rows >= r.lo && rows <= r.hi)
        return tables::draft_chunks(K) == tables::kDraftChunksLong ? r.cfg_long : r.cfg;
    std::abort();
  }
  const int s = tables::rules_sku(sku, tables::kFamilyBf16);
  for (const auto& r : tables::kBf16Rows)
    if (r.sku == s && rows >= r.lo && rows <= r.hi) return r.cfg;
  return 0;
}

}  // namespace

bool device_supported() { return compiled_arch() > 0; }

void quantize_nvfp4(const bf16* W, int N, int K, uint8_t* Q, uint8_t* S, float* RS, cudaStream_t st) {
  k_quantize_nvfp4<<<N, 256, 0, st>>>(W, K, Q, S, RS);
}

namespace {
// The pipelined kernel needs K % 32, an even row stride and room in the bf16 scratch.
bool tc_fits(const float* x, int ldx, int K) {
  return K % 32 == 0 && ldx % 2 == 0 && (reinterpret_cast<uintptr_t>(x) % 8) == 0;
}
}  // namespace

int gemm_tuning_for(const char* device_name, int cc, bool* by_arch) {
  int sku = -1;
  for (int s = 0; s < tables::kSkuCount; ++s)
    for (int d = 0; d < tables::kSkus[s].n_devices; ++d)
      if (std::strcmp(tables::kSkus[s].devices[d], device_name) == 0) sku = s;
  if (by_arch) *by_arch = false;
  for (int i = 0; sku < 0 && i < tables::kArchTableCount; ++i)
    if (tables::kArchTables[i].cc == cc) {
      sku = tables::kArchTables[i].sku;
      if (by_arch) *by_arch = true;
    }
  return sku;
}

int gemm_tuning_named(const char* name) {
  for (int s = 0; s < tables::kSkuCount; ++s)
    if (std::strcmp(tables::kSkus[s].name, name) == 0) return s;
  return -1;
}

std::string gemm_tuning_names() {
  std::string all;
  for (int s = 0; s < tables::kSkuCount; ++s) all += (s ? ", " : "") + std::string(tables::kSkus[s].name);
  return all.empty() ? "none" : all;
}

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

const char* gemm_tuning_name(int tuning) { return tuning >= 0 ? tables::kSkus[tuning].name : nullptr; }

bool gemm_tuning_compiled_for(int tuning, int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)tuning, (void)cc;
  return true;
#else
  if (tuning < 0 || tuning >= tables::kSkuCount) return true;
  const int sa = tables::kSkuArch[tuning];
  return sa == 0 || sa == cc || (cc == 80 && (sa == 86 || sa == 89));
#endif
}

bool gemm_tuning_pdl(int tuning) { return tuning >= 0 && tables::kSkus[tuning].pdl; }

int gemm_config(int tuning, int fmt, int N, int K, int rows) { return pick_config(tuning, fmt, N, K, rows); }

int gemm_draft_chunks(int K) { return tables::draft_chunks(K); }

void gemm_skinny(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                 const Launch& L) {
  gemm_skinny_cfg(pick_config(opts(L).tuning, 0, N, K, rows), x, ldx, W, N, K, y, ldy, rows, accumulate, L);
}

// The tile and unroll are the first weight's shape's; the chunks, and so every weight's sums, K's.
void gemm_draft(const bf16* x, int ldx, int K, const MtGroups& gs, int rows, const Launch& L) {
  gemm_draft_cfg(pick_config(opts(L).tuning, kFmtBf16Draft, gs.g[0].N, K, rows), x, ldx, K, gs, rows, L);
}

void gemm_w12_skinny(const float* x, int ldx, const W12& W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                     const Launch& L) {
  gemm_w12_skinny_cfg(pick_config(opts(L).tuning, 2, N, K, rows), x, ldx, W, N, K, y, ldy, rows, accumulate, L);
}

void gemm_presplit(const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  if (rows > 16)
    gemm_tc_staged(W, nullptr, nullptr, nullptr, N, K, y, ldy, rows, accumulate, L);
  else
    gemm_skinny_cfg(pick_config(opts(L).tuning, 0, N, K, rows), nullptr, K, W, N, K, y, ldy, rows, accumulate, L);
}

namespace {
// grid (rows, ceil(n / 256)); block 256: u[r][c] *= scale, one correctly rounded product.
__global__ void k_scale_rows(float* u, int ldu, int n, float scale) {
  pdl_trigger();
  pdl_wait();
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c < n) {
    float* p = u + size_t(blockIdx.x) * ldu + c;
    *p = __fmul_rn(*p, scale);
  }
}
}  // namespace

void lora_down(const float* x, int ldx, const bf16* a, int n, int K, float scale, float* u, int ldu, int rows,
               const Launch& L) {
  if (x) gemm(x, ldx, a, n, K, u, ldu, rows, false, L);
  else gemm_presplit(a, n, K, u, ldu, rows, false, L);
  if (scale != 1.f)
    launch_dependent(k_scale_rows, dim3(rows, ceil_div(n, 256)), dim3(256), L, u, ldu, n, scale);
}

void gemm_fp4_presplit(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy, int rows,
                       bool accumulate, const Launch& L) {
  if (rows > 16)
    gemm_tc_staged(nullptr, Q, S, RS, N, K, y, ldy, rows, accumulate, L);
  else
    gemm_fp4_skinny_cfg(pick_config(opts(L).tuning, 1, N, K, rows), nullptr, K, Q, S, RS, N, K, y, ldy, rows,
                        accumulate, L);
}

void gemm_fp4(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
              int ldy, int rows, bool accumulate, const Launch& L) {
  if (rows <= 16 && (K % 64) == 0 && (ldx % 2) == 0 && (reinterpret_cast<uintptr_t>(x) % 8) == 0)
    gemm_fp4_skinny_cfg(pick_config(opts(L).tuning, 1, N, K, rows), x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate,
                        L);
  else if (rows > 16 && tc_fits(x, ldx, K))
    gemm_fp4_tc(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate, L);
  else
    gemm_fp4_mma(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate, L.st);
}

void gemm(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
          const Launch& L) {
  const bool row_path = opts(L).row_gemms;
  if (rows > 16 && (L.round_rows || L.draft)) {
    // Every tile of 16 rows the 1..16-row kernel's arithmetic, the weights read once (a round's: a linear's bf16
    // weights, regions/linear.cpp, and an adapter's; numerics.md R, C), or bf16 inputs (a drafting pass's:
    // deterministic).
    // A round's of more than 64 rows (DSpine's walk) in one launch, each 64 on the configuration of 64.
    if ((rows <= 64 || (L.round_rows && rows <= kMtMostRows)) && !row_path && skinny_fits(x, ldx, W, K)) {
      const int fmt = L.round_rows ? kFmtBf16Rows : kFmtBf16Draft;
      gemm_skinny_mt_cfg(pick_config(opts(L).tuning, fmt, N, K, std::min(rows, 64)), x, ldx, W, N, K, y, ldy, rows,
                         accumulate, L);
      return;
    }
    if (L.round_rows) {  // a round's: the 1..16-row choice a tile of 16 rows at a time (numerics.md: R, C)
      for (int r0 = 0; r0 < rows; r0 += 16)
        gemm(x + size_t(r0) * ldx, ldx, W, N, K, y + size_t(r0) * ldy, ldy, std::min(16, rows - r0), accumulate, L);
      return;
    }
  }
  if (rows <= 16 && !row_path && skinny_fits(x, ldx, W, K))
    gemm_skinny(x, ldx, W, N, K, y, ldy, rows, accumulate, L);
  else if (rows <= 16 || row_path || (ldx % 4) != 0) gemm_rows(x, ldx, W, N, K, y, ldy, rows, accumulate, L.st);
  else if (tc_fits(x, ldx, K)) gemm_tc(x, ldx, W, N, K, y, ldy, rows, accumulate, L);
  else gemm_mma(x, ldx, W, N, K, y, ldy, rows, accumulate, L.st);
}

}  // namespace flyx::cuda
