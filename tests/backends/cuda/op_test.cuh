// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What the kernel tests share (op_test*.cu): inputs, transfers, the check against a reference, the scratch
// main() sets up; and the tests, one file per family of kernels.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp8.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"
#include "ops/ops.h"

namespace flyx::optest {

using namespace flyx;
using namespace flyx::cuda;
using Vec = std::vector<double>;

inline int g_fail = 0;

// Every test draws its inputs from a sequence of its own, seeded by its name and parameters (FNV-1a of them), so a
// test added, changed or reordered leaves the other tests' inputs as they were.  FLYX_OPS_SEED=S gives every
// test other inputs (a tolerance is for any inputs, not for one set).
inline std::mt19937 rng;
template <class... A>
inline void seed(const char* test, A... params) {
  std::string key = test;
  ((key += " " + std::to_string(params)), ...);
  if (const char* s = std::getenv("FLYX_OPS_SEED")) key += std::string(" seed ") + s;
  uint32_t h = 2166136261u;
  for (const unsigned char c : key) h = (h ^ c) * 16777619u;
  rng.seed(h);
}

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                    \
    }                                                                                  \
  } while (0)

inline Vec randv(size_t n, double scale = 1.0) {
  std::normal_distribution<double> d(0.0, scale);
  Vec v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

inline uint16_t to_bf16(double x) {
  float f = float(x);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}
inline double from_bf16(uint16_t h) { return ops::bf16_to_double(h); }
inline void round_bf16(Vec& v) {
  for (auto& x : v) x = from_bf16(to_bf16(x));
}
// binary16 (the K/V caches), nearest even.
inline uint16_t to_f16(double x) {
  const __half h = __float2half_rn(float(x));
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
inline double from_f16(uint16_t u) {
  __half h;
  std::memcpy(&h, &u, 2);
  return double(__half2float(h));
}
inline void round_f16(Vec& v) {
  for (auto& x : v) x = from_f16(to_f16(x));
}

template <class T>
T* dev(size_t n) {
  T* p;
  CK(cudaMalloc(&p, n * sizeof(T) + 16));
  CK(cudaMemset(p, 0, n * sizeof(T) + 16));
  return p;
}
inline float* up(const Vec& v) {
  std::vector<float> f(v.begin(), v.end());
  float* p = dev<float>(f.size());
  CK(cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
  return p;
}
inline bf16* up_bf16(const Vec& v) {
  std::vector<uint16_t> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = to_bf16(v[i]);
  bf16* p = dev<bf16>(h.size());
  CK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  return p;
}
inline kv16* up_f16(const Vec& v) {
  std::vector<uint16_t> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = to_f16(v[i]);
  kv16* p = dev<kv16>(h.size());
  CK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  return p;
}
inline Vec down(const float* p, size_t n) {
  CK(cudaDeviceSynchronize());
  std::vector<float> f(n);
  CK(cudaMemcpy(f.data(), p, n * 4, cudaMemcpyDeviceToHost));
  return Vec(f.begin(), f.end());
}
inline Vec down_f16(const kv16* p, size_t n) {
  CK(cudaDeviceSynchronize());
  std::vector<uint16_t> h(n);
  CK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
  Vec v(n);
  for (size_t i = 0; i < n; ++i) v[i] = from_f16(h[i]);
  return v;
}
inline Vec down_bf16(const bf16* p, size_t n) {
  CK(cudaDeviceSynchronize());
  std::vector<uint16_t> h(n);
  CK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
  Vec v(n);
  for (size_t i = 0; i < n; ++i) v[i] = from_bf16(h[i]);
  return v;
}
inline std::vector<int> down_i(const int* p, size_t n) {
  CK(cudaDeviceSynchronize());
  std::vector<int> v(n);
  CK(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
  return v;
}

// max |a-b| / max(|b|) for a pair of finite outputs.
inline double relative_error(const Vec& got, const Vec& want) {
  if (got.size() != want.size()) return std::numeric_limits<double>::infinity();
  double err = 0, scale = 1e-30;
  for (size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return std::numeric_limits<double>::infinity();
    err = std::max(err, std::fabs(got[i] - want[i]));
    scale = std::max(scale, std::fabs(want[i]));
  }
  return err / scale;
}
inline void expect(const std::string& what, const Vec& got, const Vec& want, double tol) {
  const double err = relative_error(got, want);
  const bool ok = err <= tol;
  std::printf("  %-4s %-48s rel err %.2e (tol %.0e)\n", ok ? "ok" : "FAIL", what.c_str(), err, tol);
  if (!ok) ++g_fail;
}

// Launches of the tests: the default stream, the scratch set up in main(), the default options.
inline Launch TL;

// GGML block weights (op_test_blocks.cu): a type and where a file block of it keeps its binary16 scales; n random
// blocks of it as a file stores them; the file's blocks on the device as the kernels read them.
struct BlockType {
  int dtype;
  const char* name;
  std::vector<int> scales_at;
};
extern const BlockType kBlockTypes[19];
const BlockType& block_type(int dtype);
std::vector<uint8_t> random_blocks(const BlockType& bt, size_t n);
BlockMat up_blocks(int dtype, const std::vector<uint8_t>& file, int rows, bool host = false);

// The test scratch's rows x K in 8 bits as a fused producer or blocks_stage left them: the integers, the scales
// and (up to 16 rows) a round's sums.
struct Q8Rows {
  std::vector<int8_t> q;
  std::vector<float> d, sums;
  bool operator==(const Q8Rows& o) const { return q == o.q && d == o.d && sums == o.sums; }
};
inline Q8Rows q8_rows(int rows, int K) {
  CK(cudaDeviceSynchronize());
  const Scratch& sc = *TL.scratch;
  Q8Rows r;
  r.q.resize(size_t(rows) * K);
  r.d.resize(size_t(rows) * K / 32);
  CK(cudaMemcpy(r.q.data(), sc.q8, r.q.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(r.d.data(), sc.q8_scale, r.d.size() * 4, cudaMemcpyDeviceToHost));
  if (rows <= 16) {
    r.sums.resize(size_t(rows) * K / 16);
    CK(cudaMemcpy(r.sums.data(), sc.q8_sums, r.sums.size() * 4, cudaMemcpyDeviceToHost));
  }
  return r;
}

inline DevParams* params(int position, int round = 0) {
  DevParams p{};
  p.position = position;
  p.round = round;
  DevParams* d = dev<DevParams>(1);
  CK(cudaMemcpy(d, &p, sizeof(p), cudaMemcpyHostToDevice));
  return d;
}

inline flyx_tensor_view view_of(const Vec& w, std::vector<uint16_t>* store, int N, int K) {
  store->resize(w.size());
  for (size_t i = 0; i < w.size(); ++i) (*store)[i] = to_bf16(w[i]);
  flyx_tensor_view t{};
  t.dtype = FLYX_DT_BF16;
  t.ndim = 2;
  t.shape[0] = N;
  t.shape[1] = K;
  t.data = store->data();
  return t;
}

void test_gemm();
void test_gemm_round();
void test_gemm_draft_tables();
void test_gemm_tables_compiled();
void test_rmsnorm();
void test_blocks();
void test_blocks_round_stacks();
void test_blocks_round_glu();
void test_blocks_round_side();
void test_attention(int hd, int rot, int nh, int nkv, int rows, int ctx, bool causal, double theta, int window = 0,
                    bool qk_norm = true);
void test_attention_fp8(int hd, int rot, int nh, int nkv, int rows, int ctx, double theta, bool qk_norm = true);
void test_attention_tree(int hd, int nh, int nkv, int ctx);
void test_kv16_saturation();
void test_strata(int hd, int nh, int nkv, int rows, int ctx, int bits, bool fp8);
void test_pageable_kv_ops();
void test_quantize_nvfp4();
void test_gdn(const GdnDims d, const int rows, const double tol);
void test_gdn_chunk(const GdnDims d, const int rows, const int pos0, const double tol);
void test_gemm_glu(int M, int I, int K);
void test_gemm_w4a4(int M, int N, int K);
void test_head_cert(int V, int K);
void test_head_cert_input_rounding();
void test_gemm_fp8(int N, int K);
void test_gemm_w12(int N, int K);
void test_draft_pick(int V, int K, bool plant, bool pick = true);
void test_draft_pick_blocks();
void test_quantize_q8_0();
void test_fused_producers();
void test_lora();
void test_grouped_conv();
void test_fp4();
void test_penalize();
void test_wide_rows_choice();
void test_sample();
void test_select();
void test_stop_decide();
void test_dv();
void test_dspine();
void test_dv_selector_tables(int vocab, int hidden, int rank);
void test_canvas_input(int slots, int hidden);
void test_canvas_stats(int slots, int vocab, int hidden, int rank);
void test_dv_slide(int slots, int hidden, int rank);
void test_dv_selector_table(int depths, int rank, int vocab);
void test_dv_trees();
void test_canvas_attention(int hd, int nh, int nkv, int start, int N, bool causal, bool fp8, double theta);
void test_layernorm();
void test_add_bias();
void test_vact();
void test_gemm_f32();
void test_vision_pos_embed();
void test_vision_rope_2d();
void test_vision_attention();
void test_vision_avg_pool();
void test_embed_scatter();
void test_mrope();

}  // namespace flyx::optest
