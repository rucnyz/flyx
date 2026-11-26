// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Storage formats as float64: bf16, f16, FP8 E4M3, NVFP4 (E2M1 elements with
// E4M3 scales), GGML block types (ggml.h); how linears round their inputs:
// to bf16 (NVFP4 weights in rounds), to binary16 over a row's scale (GGML
// block weights in rounds), and in passes of more rows as their files say:
// NVFP4 (round_nvfp4) or 8-bit integers (round_q8).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "flyx/backend_abi.h"
#include "ops/dtype/ggml.h"

namespace flyx::ops {

inline double bf16_to_double(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Element i of a stored tensor, as double (0 for types that are not numbers on their own).
inline double load(const flyx_tensor_view* t, size_t i) {
  const auto* p = static_cast<const uint8_t*>(t->data);
  switch (t->dtype) {
    case FLYX_DT_BF16: {
      uint16_t h;
      std::memcpy(&h, p + 2 * i, 2);
      return bf16_to_double(h);
    }
    case FLYX_DT_F16: {
      uint16_t h;
      std::memcpy(&h, p + 2 * i, 2);
      return ggml::half(h);
    }
    case FLYX_DT_F32: {
      float f;
      std::memcpy(&f, p + 4 * i, 4);
      return f;
    }
    case FLYX_DT_F64: {
      double d;
      std::memcpy(&d, p + 8 * i, 8);
      return d;
    }
    default: return flyx_block_values(t->dtype) ? ggml::value(t->dtype, p, i) : 0.0;
  }
}

// load() of elements [i, i + n), the type looked at once.
inline void load_run(const flyx_tensor_view* t, size_t i, int n, double* out) {
  const auto* p = static_cast<const uint8_t*>(t->data);
  switch (t->dtype) {
    case FLYX_DT_F64: std::memcpy(out, p + 8 * i, sizeof(double) * size_t(n)); return;
    case FLYX_DT_BF16:
      for (int e = 0; e < n; ++e) {
        uint16_t h;
        std::memcpy(&h, p + 2 * (i + size_t(e)), 2);
        out[e] = bf16_to_double(h);
      }
      return;
    default:
      for (int e = 0; e < n; ++e) out[e] = load(t, i + size_t(e));
  }
}

// FP8 E4M3 (no infinities) and E2M1 (NVFP4 element) decoding.
inline double e4m3_to_double(uint8_t b) {
  const int e = (b >> 3) & 0xF, m = b & 7;
  const double v = e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7);
  return (b & 0x80) ? -v : v;
}
inline double e2m1_to_double(uint8_t code) {
  static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (code & 8) ? -mag[code & 7] : mag[code & 7];
}
// An NVFP4 linear [N, K] as doubles: e2m1 * scale * scale2.
inline std::vector<double> dequant_nvfp4(const flyx_tensor_view* q, const flyx_tensor_view* s,
                                         const flyx_tensor_view* s2) {
  const size_t N = size_t(q->shape[0]), K = size_t(q->shape[1]);
  const auto* qb = static_cast<const uint8_t*>(q->data);
  const auto* sb = static_cast<const uint8_t*>(s->data);
  const double g = static_cast<const float*>(s2->data)[0];
  std::vector<double> w(N * K);
  for (size_t n = 0; n < N; ++n)
    for (size_t k = 0; k < K; ++k) {
      const uint8_t byte = qb[n * (K / 2) + k / 2];
      w[n * K + k] = e2m1_to_double((k & 1) ? byte >> 4 : byte & 0xF) * e4m3_to_double(sb[n * (K / 16) + k / 16]) * g;
    }
  return w;
}

inline std::vector<double> to_vec(const flyx_tensor_view* t) {
  size_t n = 1;
  for (int i = 0; i < t->ndim; ++i) n *= static_cast<size_t>(t->shape[i]);
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = load(t, i);
  return v;
}

// Round to bfloat16, nearest even (NVFP4 linears' inputs in rounds, and in every pass with exact activations).
inline double round_bf16(double v) {
  float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round to binary16, nearest even: 11 significant bits down to 2^-14, multiples of 2^-24 below.
inline double round_f16(double v) {
  if (v == 0 || !std::isfinite(v)) return v;
  int e;
  std::frexp(v, &e);  // |v| in [2^(e-1), 2^e)
  const int last = std::max(e - 1, -14) - 10;
  const double r = std::ldexp(std::nearbyint(std::ldexp(v, -last)), last);
  return std::fabs(r) >= 65520.0 ? std::copysign(INFINITY, v) : r;
}

// A binary16 K/V cache's value: round_f16, but the largest finite value (65504) past it instead of an infinity, an
// infinity included; NaN stays NaN.
inline double round_kv16(double v) {
  const double r = round_f16(v);
  return std::isinf(r) ? std::copysign(65504.0, v) : r;
}

// A row as the kernels of GGML block weights hold it: rounded to bf16, then
// to binary16 over the power of two that puts the row's largest magnitude in
// [2^15, 2^16) (from 2^-126 up: the scale stays a normal float32).  Nothing
// is lost of a value within 2^-32 of the largest; smaller ones round to the
// binary16 subnormals.  A row of zeros, or one with an infinity, keeps the
// scale 1.
inline void round_a16h(double* row, int K) {
  double m = 0;
  for (int k = 0; k < K; ++k) {
    row[k] = round_bf16(row[k]);
    if (std::fabs(row[k]) > m) m = std::fabs(row[k]);
  }
  double scale = 1;
  if (m > 0 && std::isfinite(m)) {
    int e;
    std::frexp(m, &e);  // m in [2^(e-1), 2^e)
    scale = std::ldexp(1.0, std::max(e - 1, -111) - 15);
  }
  for (int k = 0; k < K; ++k) row[k] = round_f16(row[k] / scale) * scale;
}

// The nearest e2m1 code of x (ties to the even code): magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6, bit 3 the sign.
inline uint8_t e2m1_nearest(float x) {
  const float a = std::fabs(x);
  const uint8_t m = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5
                  : a <= 5.f   ? 6 : 7;
  return (x < 0.f && m) ? uint8_t(m | 8) : m;
}
// The nearest FP8 E4M3 of v >= 0 (ties to even), saturating at 448: what CUDA's conversion with __NV_SATFINITE gives.
inline uint8_t e4m3_nearest(float v) {
  if (!(v > 0.f)) return 0;
  if (v >= 448.f) return 0x7E;
  int e;
  std::frexp(v, &e);
  const int E = std::max(e - 1, -6);                       // v in [2^E, 2^(E+1)), or a subnormal
  const double q = std::nearbyint(std::ldexp(double(v), 3 - E));  // in steps of 2^(E-3): 8..16, or 0..7 below 2^-6
  if (E == -6 && q < 8) return uint8_t(q);
  const int up = q >= 16, m = int(q) - 8 * (1 + up);
  return uint8_t(std::min(((E + up + 7) << 3) | m, 0x7E));
}
// Round a row as NVFP4 weights hold their inputs in passes of more than FLYX_MAX_ROWS rows (FLYX_ACT_NVFP4,
// the W4A4 kernels): from the float32 values the kernels hold, one rounding (bf16 first before, a
// second rounding, coarser than llama.cpp's); a row scale g = 6 * 448 / m (m the row's largest magnitude: every
// block scale stays within e4m3); per 16 values an e4m3 scale s = nearest((b / 6) g) (b their largest magnitude)
// and each value the nearest e2m1 code c of value * (g / s); the value is then c * s * (1 / g).  Every step in
// float32 as the kernels compute it.
inline void round_nvfp4(double* row, int K) {
  std::vector<float> a(static_cast<size_t>(K));
  float m = 0.f;
  for (int k = 0; k < K; ++k) {
    a[size_t(k)] = static_cast<float>(row[k]);
    m = std::max(m, std::fabs(a[size_t(k)]));
  }
  const float g = m > 0.f ? (6.f * 448.f) / m : 1.f, inv = 1.f / g;
  for (int b = 0; b < K / 16; ++b) {
    float bm = 0.f;
    for (int e = 0; e < 16; ++e) bm = std::max(bm, std::fabs(a[size_t(16 * b + e)]));
    const float s = static_cast<float>(e4m3_to_double(e4m3_nearest((bm / 6.f) * g))), f = s > 0.f ? g / s : 0.f;
    for (int e = 0; e < 16; ++e)
      row[16 * b + e] = e2m1_to_double(e2m1_nearest(a[size_t(16 * b + e)] * f)) * s * double(inv);
  }
}

// Round a row to 8-bit integers with a scale per 32 values: how GGML block linears hold their inputs
// (FLYX_ACT_Q8). From the float32 values the kernels hold (one rounding, as llama.cpp's): per 32
// values d = m / 127 (m their largest magnitude) and p = the nearest integer of value / d (ties to even), both in
// float32 as the kernels compute them; the value is d p.  p: K integers, d: K / 32 scales; a group with nothing finite
// to scale is all zeros.
inline void round_q8(const double* x, int K, int8_t* p, float* d) {
  for (int g = 0; g < K / 32; ++g) {
    float a[32], m = 0.f;
    for (int i = 0; i < 32; ++i) {
      a[i] = static_cast<float>(x[32 * g + i]);
      m = std::max(m, std::fabs(a[i]));
    }
    const float dg = m / 127.f;
    d[g] = dg;
    const bool scaled = dg > 0.f && std::isfinite(dg);
    for (int i = 0; i < 32; ++i) p[32 * g + i] = scaled ? static_cast<int8_t>(std::nearbyint(a[i] / dg)) : int8_t(0);
  }
}

// IEEE-754 binary16 bit encoding of a float32 (round to nearest even, matching CUDA __float2half_rn).
inline uint16_t float_to_f16_bits(float v) {
  uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  int exp = int((u >> 23) & 0xFFu) - 127;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp == 128) return uint16_t(sign | 0x7C00u | (mant ? 0x0200u : 0u));
  if (exp > 15) return uint16_t(sign | 0x7C00u);
  if (exp >= -14) {
    uint32_t m = mant + (0x0FFFu + ((mant >> 13) & 1u));
    if (m & 0x800000u) {
      m = 0;
      ++exp;
      if (exp > 15) return uint16_t(sign | 0x7C00u);
    }
    return uint16_t(sign | (uint32_t(exp + 15) << 10) | (m >> 13));
  }
  if (exp >= -25) {
    mant |= 0x800000u;
    const int shift = -14 - exp + 13;
    const uint32_t half_bit = 1u << (shift - 1);
    const uint32_t low = mant & ((1u << shift) - 1u);
    uint32_t q = mant >> shift;
    if (low > half_bit || (low == half_bit && (q & 1u))) ++q;
    return uint16_t(sign | q);
  }
  return uint16_t(sign);
}

// Quantize one weight row of length K (zero-padded to K_pad, multiple of 32) into GGML Q8_0 payload + f16 scales.
// Exact bit-for-bit formula shared by cpu_ref and the CUDA backend's on-the-fly vision quantizer.
inline void quantize_row_q8_0(const float* src, int K, int K_pad, uint8_t* payload, uint16_t* scales) {
  const int blocks = K_pad / 32;
  for (int b = 0; b < blocks; ++b) {
    float v[32], amax = 0.f;
    for (int e = 0; e < 32; ++e) {
      const int idx = 32 * b + e;
      v[e] = idx < K ? src[idx] : 0.f;
      amax = std::max(amax, std::fabs(v[e]));
    }
    const uint16_t dh_bits = float_to_f16_bits(amax / 127.f);
    const float d = static_cast<float>(ggml::half(dh_bits));
    const float inv = d > 0.f ? 1.f / d : 0.f;
    scales[b] = dh_bits;
    for (int e = 0; e < 32; ++e) {
      const int q = std::clamp(static_cast<int>(std::nearbyint(v[e] * inv)), -127, 127);
      payload[32 * b + e] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
  }
}

// Quantize one weight row of length K (zero-padded to K_pad, multiple of 64) into NVFP4 planes (Q, S, RS).
// Matches k_quantize_nvfp4: rs = max|w| / (6 * 448), block e4m3 scale per 16, e2m1 codes two per byte.
inline void quantize_row_nvfp4(const float* src, int K, int K_pad, uint8_t* Q, uint8_t* S, float* RS) {
  float amax = 0.f;
  for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(src[k]));
  const float rs = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
  *RS = rs;
  const int blocks = K_pad / 16;
  for (int b = 0; b < blocks; ++b) {
    float v[16], bmax = 0.f;
    for (int e = 0; e < 16; ++e) {
      const int idx = 16 * b + e;
      v[e] = idx < K ? src[idx] : 0.f;
      bmax = std::max(bmax, std::fabs(v[e]));
    }
    const uint8_t sc = e4m3_nearest(bmax / 6.f / rs);
    const float scale = static_cast<float>(e4m3_to_double(sc)) * rs;
    const float inv = scale > 0.f ? 1.f / scale : 0.f;
    S[b] = sc;
    for (int e = 0; e < 16; e += 2) {
      Q[b * 8 + e / 2] = static_cast<uint8_t>(e2m1_nearest(v[e] * inv) | (e2m1_nearest(v[e + 1] * inv) << 4));
    }
  }
}

}  // namespace flyx::ops
