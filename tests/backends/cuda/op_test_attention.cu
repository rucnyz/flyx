// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: attention: attn_prep, the split kernel, the tensor-core kernel, the FP8 cache, strata attention.
#include <cuda.h>
#include "tests/backends/cuda/op_test.cuh"
#include "backends/cuda/runtime/strata_snapshot.h"

namespace flyx::optest {

namespace {
// Decode and verification attention against the reference: the float32 sums over the keys; with q and p as one
// binary16 (the tensor-core kernel's default where llama.cpp rounds them so) prefill attention's.
double rows_tolerance(bool mma, int parts, int nh, int nkv, int hd, int ctx) {
  const bool one = mma && (parts == 1 || (parts == 0 && nh / nkv > 4 && hd >= 256));
  return one ? 3e-3 : ctx > 1000 ? 1e-4 : 2e-5;
}

__global__ void k_check_e4m3_sw(uint16_t* sw_bits, uint16_t* ref_bits, float* sw_f, float* ref_f) {
  const int base = threadIdx.x * 4;
  uint32_t w = 0;
  for (int i = 0; i < 4; ++i) w |= uint32_t(base + i) << (8 * i);
  uint32_t out[2];
  e4m3x4_to_f16bits_sw(w, out);
  for (int i = 0; i < 4; ++i) {
    const uint8_t b = uint8_t(base + i);
    sw_bits[base + i] = uint16_t((out[i >> 1] >> (16 * (i & 1))) & 0xFFFFu);
    ref_bits[base + i] = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3).x;
    sw_f[base + i] = e4m3_to_float_sw(b);
    ref_f[base + i] = __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
  }
}
}  // namespace

// attn_prep + attention against the reference, causal (target) and block (drafter) masks; window > 0: each row
// only the keys less than `window` from its position, both ways (a drafter's sliding window).  qk_norm false: no
// q/k norms (Llama's attention: the heads go to the rotary as projected).
// The tensor-core prefill a window of keys at a time: the first window read where the cache is, each
// later one copied alone to a staging the kernel reads at its keys' own positions, the softmax state carried between
// them: the bits of one pass.
template <class T>
void expect_windows(const std::string& what, const float* dq, const T* dK, const T* dV, const float* dKs,
                    const float* dVs, int nh, int nkv, int hd, int rows, int total, const DevParams* P,
                    const float* gate, int ld, const Vec& whole, int hb = 0) {
  const size_t row = size_t(nkv) * hd;  // values a position
  const int W = kFlashWindowAlign, first = 2 * W;
  float* state = dev<float>(size_t(rows) * nh * kFlashStateFloats(hd));
  float* dw = dev<float>(whole.size());
  T *sk = dev<T>(W * row), *sv = dev<T>(W * row);
  const size_t scales = size_t(total) * nkv;
  const int split = std::min(total, first + W + W / 2);
  float* scale_buf = dKs ? dev<float>(4 * scales) : nullptr;
  if (scale_buf) {
    // Keep only the correct side of each scale array populated. The cut falls inside a window.
    const size_t left = size_t(split) * nkv;
    for (int v = 0; v < 2; ++v) {
      const float* src = v ? dVs : dKs;
      CK(cudaMemcpy(scale_buf + v * scales, src, left * sizeof(float), cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy(scale_buf + (2 + v) * scales + left, src + left, (scales - left) * sizeof(float),
                    cudaMemcpyDeviceToDevice));
    }
  }
  const float sc = float(1.0 / std::sqrt(hd));
  int windows = 0;
  for (int w0 = 0; w0 < total; w0 = w0 == 0 ? first : w0 + W) {
    const int w1 = std::min(total, w0 == 0 ? first : w0 + W);
    KvRef kv{const_cast<T*>(dK), const_cast<T*>(dV), const_cast<float*>(dKs), const_cast<float*>(dVs)};
    if (w0 > 0) {  // staged
      CK(cudaMemcpy(sk, dK + w0 * row, (w1 - w0) * row * sizeof(T), cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy(sv, dV + w0 * row, (w1 - w0) * row * sizeof(T), cudaMemcpyDeviceToDevice));
      kv.k = sk - w0 * row;
      kv.v = sv - w0 * row;
    }
    FlashWindow fw;
    fw.k_lo = w0;
    fw.k_hi = w1;
    fw.state = state;
    fw.first = w0 == 0;
    fw.last = w1 == total;
    fw.history = w1 <= total - rows;
    if (scale_buf) {
      kv.scale_split = split;
      kv.ks = scale_buf;
      kv.vs = scale_buf + scales;
      kv.ks_far = scale_buf + 2 * scales;
      kv.vs_far = scale_buf + 3 * scales;
    }
    // The first and last windows keep the usual grouping. The candidate reads and writes the same softmax state.
    if (hb && !fw.first && !fw.last) {
      if (!attention_flash_gqa_cfg(hb, dq, kv, nh, nkv, hd, rows, sc, P, gate, ld, dw, nh * hd, TL, 0, fw)) {
        std::printf("  FAIL %s: configuration %d unavailable\n", what.c_str(), hb);
        ++g_fail;
        break;
      }
    } else {
      attention_flash(dq, kv, nh, nkv, hd, rows, sc, P, gate, ld, dw, nh * hd, TL, 0, fw);
    }
    ++windows;
  }
  std::vector<float> actual(whole.size()), expected(whole.begin(), whole.end());
  CK(cudaMemcpy(actual.data(), dw, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const bool same = std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
  std::printf("  %-4s %s in %d windows of keys: same bits\n", same ? "ok" : "FAIL", what.c_str(), windows);
  if (!same) ++g_fail;
  for (void* x : std::initializer_list<void*>{state, dw, sk, sv, scale_buf}) CK(cudaFree(x));
}

// A binary16 K/V cache holds 65504 for what is past it, an infinity included (kv16_of, ops/dtype/ref.h round_kv16):
// attn_prep's k and v (no q/k norms: k rotated as projected) and draft_ctx_kv's v, each entry the reference's, and
// attention over such a cache stays finite (with an infinity in it, every row attending to it was NaN).
void test_kv16_saturation() {
  std::printf("binary16 K/V caches saturate\n");
  seed("kv16 saturation");
  const int hd = 64, rot = 64, nh = 2, nkv = 1, rows = 4, ctx = 3, cap = ctx + rows + 16;
  const double theta = 1e4;
  const int ld = 2 * nh * hd + 2 * nkv * hd, k_off = 2 * nh * hd, v_off = k_off + nkv * hd;
  Vec qkv = randv(size_t(rows) * ld);
  qkv[size_t(1) * ld + v_off + 3] = 1e5;
  qkv[size_t(1) * ld + v_off + 7] = -3e6;
  qkv[size_t(2) * ld + k_off + 5] = 2e5;
  qkv[size_t(3) * ld + v_off + 1] = INFINITY;
  qkv[size_t(3) * ld + v_off + 2] = -INFINITY;
  const Vec zeros(size_t(cap) * nkv * hd, 0.0);
  float *dqkv = up(qkv), *dq = dev<float>(size_t(rows) * nh * hd), *dout = dev<float>(size_t(rows) * nh * hd);
  kv16 *dK = up_f16(zeros), *dV = up_f16(zeros);
  DevParams* P = params(ctx);
  attn_prep(dqkv, ld, 0, k_off, v_off, nullptr, nullptr, 1.0f, 1e-6f, theta, rot, nh, nkv, hd, rows, P, kPosPosition,
            false, dq, dK, dV, 0);
  Vec Kr = zeros, Vr = zeros;
  for (int r = 0; r < rows; ++r) {
    Vec k(hd);
    ops::prep_head(&qkv[size_t(r) * ld + k_off], nullptr, hd, 1e-6, 1.0, ctx + r, theta, rot, k.data());
    for (int d = 0; d < hd; ++d) {
      Kr[size_t(ctx + r) * hd + d] = ops::round_kv16(k[d]);
      Vr[size_t(ctx + r) * hd + d] = ops::round_kv16(qkv[size_t(r) * ld + v_off + d]);
    }
  }
  const Vec Kg = down_f16(dK, zeros.size()), Vg = down_f16(dV, zeros.size());
  bool finite = true, saturated = true;
  for (size_t i = 0; i < Kg.size(); ++i) {
    finite &= std::isfinite(Kg[i]) && std::isfinite(Vg[i]);
    if (std::fabs(Kr[i]) == 65504.0) saturated &= Kg[i] == Kr[i];
    if (std::fabs(Vr[i]) == 65504.0) saturated &= Vg[i] == Vr[i];
  }
  saturated &= Vg[size_t(ctx + 1) * hd + 3] == 65504.0 && Vg[size_t(ctx + 1) * hd + 7] == -65504.0;
  saturated &= Vg[size_t(ctx + 3) * hd + 1] == 65504.0 && Vg[size_t(ctx + 3) * hd + 2] == -65504.0;
  std::printf("  %-4s attn_prep: every cache entry finite, the ones past 65504 (infinities too) at +-65504\n",
              finite && saturated ? "ok" : "FAIL");
  g_fail += !(finite && saturated);
  expect("attn_prep V cache, saturating", Vg, Vr, 2e-3);
  expect("attn_prep K cache, saturating", Kg, Kr, 2e-3);
  const int splits = 2;
  float* part = dev<float>(attention_partial_floats(rows, nh, hd, splits));
  attention_split(dq, dK, dV, nh, nkv, hd, rows, false, rows, float(1.0 / std::sqrt(hd)), P, nullptr, 0, dout, nh * hd,
                  part, splits, TL, 0, 0);
  bool out_finite = true;
  for (double x : down(dout, size_t(rows) * nh * hd)) out_finite &= std::isfinite(x);
  std::printf("  %-4s attention over the saturated cache: finite\n", out_finite ? "ok" : "FAIL");
  g_fail += !out_finite;
  // draft_ctx_kv: a drafter's context row [layer][k | v], its v past the range
  Vec ckv = randv(size_t(2) * nkv * hd), kn = randv(hd, 0.2);
  ckv[size_t(nkv) * hd + 11] = -7e4;
  float *dckv = up(ckv), *dkn = up(kn);
  kv16 *cK = up_f16(zeros), *cV = up_f16(zeros);
  draft_ctx_kv(dckv, 2 * nkv * hd, dkn, 1e-6f, theta, 1, nkv, hd, 1, P, kPosPosition, false, cK, cV,
               size_t(cap) * nkv * hd, 0);
  const Vec cVg = down_f16(cV, zeros.size());
  bool ctx_ok = true;
  for (double x : cVg) ctx_ok &= std::isfinite(x);
  ctx_ok &= cVg[size_t(ctx) * hd + 11] == -65504.0;
  std::printf("  %-4s draft_ctx_kv: v past -65504 at -65504\n", ctx_ok ? "ok" : "FAIL");
  g_fail += !ctx_ok;
  for (void* p : std::initializer_list<void*>{dqkv, dq, dout, dK, dV, part, dckv, dkn, cK, cV}) CK(cudaFree(p));
}

void test_attention(int hd, int rot, int nh, int nkv, int rows, int ctx, bool causal, double theta, int window,
                    bool qk_norm) {
  // the inputs of the cases with norms as before the switch
  seed(qk_norm ? "attention" : "attention, no q/k norms", hd, rot, nh, nkv, rows, ctx, causal ? 1 : 0, theta, window);
  const int ld = 2 * nh * hd + 2 * nkv * hd;  // [q | gate | k | v] as in the target layout
  const int total = ctx + rows, cap = total + 16;
  Vec qkv = randv(size_t(rows) * ld), qn = randv(hd, 0.2), kn = randv(hd, 0.2);
  Vec Kc = randv(size_t(cap) * nkv * hd), Vc = randv(size_t(cap) * nkv * hd);
  round_f16(Kc);
  round_f16(Vc);
  const float offset = 1.0f;
  const double *qw = qk_norm ? qn.data() : nullptr, *kw = qk_norm ? kn.data() : nullptr;
  // reference
  Vec Kr = Kc, Vr = Vc, q(size_t(rows) * nh * hd), out(size_t(rows) * nh * hd);
  for (int r = 0; r < rows; ++r) {
    const int pos = ctx + r;
    for (int j = 0; j < nkv; ++j) {
      Vec k(hd), v(hd);
      ops::prep_head(&qkv[size_t(r) * ld + 2 * nh * hd + j * hd], kw, hd, 1e-6, offset, pos, theta, rot, k.data());
      for (int d = 0; d < hd; ++d) {
        Kr[(size_t(pos) * nkv + j) * hd + d] = from_f16(to_f16(k[d]));
        Vr[(size_t(pos) * nkv + j) * hd + d] = from_f16(to_f16(qkv[size_t(r) * ld + 2 * nh * hd + nkv * hd + j * hd + d]));
      }
    }
  }
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < nh; ++h)
      ops::prep_head(&qkv[size_t(r) * ld + h * hd], qw, hd, 1e-6, offset, ctx + r, theta, rot,
                     &q[(size_t(r) * nh + h) * hd]);
  // device
  float *dqkv = up(qkv), *dq = dev<float>(q.size()), *dout = dev<float>(out.size());
  float *dqn = qk_norm ? up(qn) : nullptr, *dkn = qk_norm ? up(kn) : nullptr;
  kv16 *dK = up_f16(Kc), *dV = up_f16(Vc);
  DevParams* P = params(ctx);
  attn_prep(dqkv, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot, nh, nkv, hd, rows, P,
            kPosPosition, false, dq, dK, dV, 0);
  const int splits = 5;
  float* part = dev<float>(attention_partial_floats(rows, nh, hd, splits));
  attention_split(dq, dK, dV, nh, nkv, hd, rows, causal, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld,
                  dout, nh * hd, part, splits, TL, 0, window);
  std::string tag = "hd " + std::to_string(hd) + " rows " + std::to_string(rows) + " ctx " + std::to_string(ctx) +
                    (causal ? " causal" : " block") + (window > 0 ? " window " + std::to_string(window) : "") +
                    (qk_norm ? "" : " no q/k norms");
  expect("q prep " + tag, down(dq, q.size()), q, 1e-5);
  if (rows > 1) {  // heads per block follow the row count: one row at a time gives the same bits
    float* dq1 = dev<float>(q.size());
    kv16 *dK1 = up_f16(Kc), *dV1 = up_f16(Vc);
    for (int r = 0; r < rows; ++r)
      attn_prep(dqkv + size_t(r) * ld, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot,
                nh, nkv, hd, 1, P, kPosPosition, false, dq1 + size_t(r) * nh * hd, dK1, dV1, 0, r);
    expect("q prep row by row: same bits " + tag, down(dq1, q.size()), down(dq, q.size()), 0);
    const size_t kv_n = size_t(cap) * nkv * hd;
    expect("K cache row by row: same bits " + tag, down_f16(dK1, kv_n), down_f16(dK, kv_n), 0);
    expect("V cache row by row: same bits " + tag, down_f16(dV1, kv_n), down_f16(dV, kv_n), 0);
  }
  expect("K cache " + tag, down_f16(dK, Kr.size()), Kr, 2e-3);
  if (!rope_table(theta, rot)) {  // the rotary factors from rope_table (made once at load) give the pow's bits
    float* dq2 = dev<float>(q.size());
    kv16 *dK2 = up_f16(Kc), *dV2 = up_f16(Vc);
    rope_register(theta, rot);
    attn_prep(dqkv, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot, nh, nkv, hd, rows,
              P, kPosPosition, false, dq2, dK2, dV2, 0);
    std::printf("  %-4s %-48s\n", rope_table(theta, rot) ? "ok" : "FAIL", ("rotary table registered " + tag).c_str());
    g_fail += !rope_table(theta, rot);
    expect("q prep, rotary factors from the table: same bits " + tag, down(dq2, q.size()), down(dq, q.size()), 0);
    expect("K cache, rotary factors from the table: same bits " + tag, down_f16(dK2, size_t(cap) * nkv * hd),
           down_f16(dK, size_t(cap) * nkv * hd), 0);
    for (void* p : std::initializer_list<void*>{dq2, dK2, dV2}) CK(cudaFree(p));
  }
  // The attention itself: the reference on the q and caches the kernel read (the prep's are checked above, and its
  // float32 q would otherwise reach the output through the scores), so the tolerance is the float32 sums over the
  // keys alone.
  {
    const Vec qk = down(dq, q.size()), Kk = down_f16(dK, Kr.size()), Vk = down_f16(dV, Vr.size());
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < nh; ++h) {
        const int p = ctx + r, all = causal ? p + 1 : total;  // the row's keys: [lo, hi)
        const int lo = window > 0 ? std::max(0, p - window + 1) : 0, hi = window > 0 ? std::min(all, p + window) : all;
        const size_t from = size_t(lo) * nkv * hd;
        double* o = &out[(size_t(r) * nh + h) * hd];
        ops::attend(&qk[(size_t(r) * nh + h) * hd], Kk.data() + from, Vk.data() + from, hi - lo, nkv, hd,
                    h / (nh / nkv), 1.0 / std::sqrt(hd), o);
        for (int d = 0; d < hd; ++d) o[d] *= ops::sigmoid(qkv[size_t(r) * ld + nh * hd + h * hd + d]);
      }
  }
  expect("attention " + tag, down(dout, out.size()), out, ctx > 1000 ? 1e-4 : 2e-5);  // float32 sums over the keys
  if (causal) {  // each row alone (decoding) gives the bits of all rows together (verification)
    const Vec all = down(dout, out.size());
    float* dout1 = dev<float>(out.size());
    for (int r = 0; r < rows; ++r)
      attention_split(dq + size_t(r) * nh * hd, dK, dV, nh, nkv, hd, 1, true, 1, float(1.0 / std::sqrt(hd)), P,
                      dqkv + size_t(r) * ld + nh * hd, ld, dout1 + size_t(r) * nh * hd, nh * hd, part, splits, TL, r,
                      window);
    expect("attention row by row: same bits " + tag, down(dout1, out.size()), all, 0);
  }
  if (window == 0) {  // the float32 kernels alone, the tensor-core one for every row, and each for half the rows
    struct Choice {
      const char* name;
      bool mma;
      int from, parts;
    };
    const Choice choices[] = {{"float32 kernels", false, 0, 0},
                              {"tensor cores", true, 0, 0},
                              {"tensor cores from the middle row", true, ctx + rows / 2 + 1, 0},
                              {"tensor cores, q and p in two parts", true, 0, 2},
                              {"tensor cores, q and p in three parts", true, 0, 3}};
    for (const Choice& c : choices) {
      KernelOptions o;
      o.rows_mma = c.mma;
      o.rows_mma_from = c.from;
      o.rows_mma_parts = c.parts;
      Launch LC = TL;
      LC.opt = &o;
      float* doutc = dev<float>(out.size());
      attention_split(dq, dK, dV, nh, nkv, hd, rows, causal, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld,
                      doutc, nh * hd, part, splits, LC, 0, window);
      expect(std::string("attention, ") + c.name + " " + tag, down(doutc, out.size()), out,
             rows_tolerance(c.mma, c.parts, nh, nkv, hd, ctx));
      if (causal) {
        const Vec all = down(doutc, out.size());
        for (int r = 0; r < rows; ++r)
          attention_split(dq + size_t(r) * nh * hd, dK, dV, nh, nkv, hd, 1, true, 1, float(1.0 / std::sqrt(hd)), P,
                          dqkv + size_t(r) * ld + nh * hd, ld, doutc + size_t(r) * nh * hd, nh * hd, part, splits,
                          LC, r, window);
        expect(std::string("attention, ") + c.name + ", row by row: same bits " + tag, down(doutc, out.size()), all,
               0);
      }
    }
  }
  if (causal && window == 0 && attention_flash(dq, dK, dV, nh, nkv, hd, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld, dout,
                                nh * hd, TL)) {
    expect("attention, tensor cores (binary16 q and p) " + tag, down(dout, out.size()), out, 3e-3);
    // key tiles sit on absolute positions: a row's bits do not depend on where its pass starts
    const int s = rows / 3;
    if (s > 0) {
      const Vec whole = down(dout, out.size());
      float* dsplit = dev<float>(out.size());
      const float sc = float(1.0 / std::sqrt(hd));
      attention_flash(dq, dK, dV, nh, nkv, hd, s, sc, P, dqkv + nh * hd, ld, dsplit, nh * hd, TL);
      attention_flash(dq + size_t(s) * nh * hd, dK, dV, nh, nkv, hd, rows - s, sc, P, dqkv + size_t(s) * ld + nh * hd,
                      ld, dsplit + size_t(s) * nh * hd, nh * hd, TL, s);
      expect("attention, tensor cores, pass split at row " + std::to_string(s) + ": same bits " + tag,
             down(dsplit, out.size()), whole, 0);
    }
    expect_windows<kv16>("attention, tensor cores " + tag, dq, dK, dV, nullptr, nullptr, nh, nkv, hd, rows, ctx + rows,
                         P, dqkv + nh * hd, ld, down(dout, out.size()));
  }
}

// FP8 K/V cache (e4m3, scale = row amax / 448): attn_prep writes it, every
// attention path reads it.  The reference attends over the cache exactly as
// stored (codes times scales, decoded in float64), so the attention
// tolerances are those of the binary16 cache; the writer is checked on its own:
// every value within half an e4m3 step of the exact K / V.  qk_norm false: no q/k norms (Llama).
double e4m3_value(uint8_t b) { return ops::e4m3_to_double(b); }

void test_attention_fp8(int hd, int rot, int nh, int nkv, int rows, int ctx, double theta, bool qk_norm) {
  static bool checked_e4m3_sw = false;
  if (!checked_e4m3_sw) {
    checked_e4m3_sw = true;
    uint16_t *d_swb = dev<uint16_t>(256), *d_refb = dev<uint16_t>(256);
    float *d_swf = dev<float>(256), *d_reff = dev<float>(256);
    k_check_e4m3_sw<<<1, 64>>>(d_swb, d_refb, d_swf, d_reff);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> swb(256), refb(256);
    std::vector<float> swf(256), reff(256);
    CK(cudaMemcpy(swb.data(), d_swb, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(refb.data(), d_refb, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(swf.data(), d_swf, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(reff.data(), d_reff, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int b = 0; b < 256; ++b) {
      if ((b & 0x7f) == 0x7f) continue;  // NaN codes are never emitted by __NV_SATFINITE
      uint32_t sw_fbits = 0, ref_fbits = 0;
      std::memcpy(&sw_fbits, &swf[b], 4);
      std::memcpy(&ref_fbits, &reff[b], 4);
      ok &= swb[b] == refb[b] && sw_fbits == ref_fbits && double(swf[b]) == e4m3_value(uint8_t(b));
    }
    std::printf("  %-4s %-48s\n", ok ? "ok" : "FAIL", "e4m3 Ampere branchless unpack (all 254 finite codes)");
    if (!ok) ++g_fail;
    cudaFree(d_swb);
    cudaFree(d_refb);
    cudaFree(d_swf);
    cudaFree(d_reff);
  }
  seed(qk_norm ? "attention fp8" : "attention fp8, no q/k norms", hd, rot, nh, nkv, rows, ctx, theta);
  const int ld = 2 * nh * hd + 2 * nkv * hd;
  const int total = ctx + rows, cap = total + 16;
  Vec qkv = randv(size_t(rows) * ld), qn = randv(hd, 0.2), kn = randv(hd, 0.2);
  const float offset = 1.0f;
  const double *qw = qk_norm ? qn.data() : nullptr, *kw = qk_norm ? kn.data() : nullptr;
  // the cache before this pass: random rows, quantized here the way attn_prep does
  std::vector<uint8_t> K8(size_t(cap) * nkv * hd, 0), V8(K8.size(), 0);
  std::vector<float> Ks(size_t(cap) * nkv, 0.f), Vs(Ks.size(), 0.f);
  for (int pos = 0; pos < ctx; ++pos)
    for (int j = 0; j < nkv; ++j)
      for (int kv = 0; kv < 2; ++kv) {
        Vec x = randv(hd);
        double amax = 0;
        for (double v : x) amax = std::max(amax, std::fabs(v));
        const float sc = float(amax) / 448.f;
        (kv ? Vs : Ks)[size_t(pos) * nkv + j] = sc;
        for (int d = 0; d < hd; ++d)
          (kv ? V8 : K8)[(size_t(pos) * nkv + j) * hd + d] = __nv_cvt_float_to_fp8(float(x[d]) / sc, __NV_SATFINITE, __NV_E4M3);
      }
  uint8_t *dK = dev<uint8_t>(K8.size()), *dV = dev<uint8_t>(V8.size());
  float *dKs = dev<float>(Ks.size()), *dVs = dev<float>(Vs.size());
  CK(cudaMemcpy(dK, K8.data(), K8.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV, V8.data(), V8.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dKs, Ks.data(), Ks.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dVs, Vs.data(), Vs.size() * 4, cudaMemcpyHostToDevice));
  const KvRef kv{dK, dV, dKs, dVs};
  float *dqkv = up(qkv), *dq = dev<float>(size_t(rows) * nh * hd), *dout = dev<float>(size_t(rows) * nh * hd);
  float *dqn = qk_norm ? up(qn) : nullptr, *dkn = qk_norm ? up(kn) : nullptr;
  DevParams* P = params(ctx);
  attn_prep(dqkv, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot, nh, nkv, hd, rows, P,
            kPosPosition, false, dq, kv, 0);
  // the cache as stored, decoded
  CK(cudaMemcpy(K8.data(), dK, K8.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(V8.data(), dV, V8.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(Ks.data(), dKs, Ks.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(Vs.data(), dVs, Vs.size() * 4, cudaMemcpyDeviceToHost));
  Vec Kr(K8.size()), Vr(V8.size());
  for (size_t i = 0; i < K8.size(); ++i) {
    Kr[i] = e4m3_value(K8[i]) * Ks[i / hd];
    Vr[i] = e4m3_value(V8[i]) * Vs[i / hd];
  }
  // the writer: the new rows within half an e4m3 step (2^-4 relative, plus the row scale's floor) of exact K / V
  double worst = 0;
  Vec q(size_t(rows) * nh * hd), out(size_t(rows) * nh * hd);
  for (int r = 0; r < rows; ++r) {
    const int pos = ctx + r;
    for (int j = 0; j < nkv; ++j) {
      Vec k(hd);
      ops::prep_head(&qkv[size_t(r) * ld + 2 * nh * hd + j * hd], kw, hd, 1e-6, offset, pos, theta, rot, k.data());
      const double* v = &qkv[size_t(r) * ld + 2 * nh * hd + nkv * hd + j * hd];
      double ka = 0, va = 0;
      for (int d = 0; d < hd; ++d) ka = std::max(ka, std::fabs(k[d])), va = std::max(va, std::fabs(v[d]));
      for (int d = 0; d < hd; ++d) {
        const size_t at = (size_t(pos) * nkv + j) * hd + d;
        worst = std::max(worst, std::fabs(Kr[at] - k[d]) / (std::fabs(k[d]) / 16 + ka / 448 / 8 + 1e-30));
        worst = std::max(worst, std::fabs(Vr[at] - v[d]) / (std::fabs(v[d]) / 16 + va / 448 / 8 + 1e-30));
      }
    }
  }
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < nh; ++h) {
      double* qr = &q[(size_t(r) * nh + h) * hd];
      ops::prep_head(&qkv[size_t(r) * ld + h * hd], qw, hd, 1e-6, offset, ctx + r, theta, rot, qr);
      ops::attend(qr, Kr.data(), Vr.data(), ctx + r + 1, nkv, hd, h / (nh / nkv), 1.0 / std::sqrt(hd),
                  &out[(size_t(r) * nh + h) * hd]);
      for (int d = 0; d < hd; ++d) out[(size_t(r) * nh + h) * hd + d] *= ops::sigmoid(qkv[size_t(r) * ld + nh * hd + h * hd + d]);
    }
  const std::string tag = "hd " + std::to_string(hd) + " rows " + std::to_string(rows) + " ctx " + std::to_string(ctx) +
                          (qk_norm ? "" : " no q/k norms");
  const bool ok = worst <= 1.0;
  std::printf("  %-4s %-48s %.2f (limit 1)\n", ok ? "ok" : "FAIL", ("fp8 K/V written, error / half step " + tag).c_str(), worst);
  if (!ok) ++g_fail;
  const int splits = 5;
  float* part = dev<float>(attention_partial_floats(rows, nh, hd, splits));
  attention_split(dq, kv, nh, nkv, hd, rows, true, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld, dout, nh * hd,
                  part, splits, TL);
  expect("attention, fp8 cache " + tag, down(dout, out.size()), out, ctx > 1000 ? 1e-4 : 2e-5);
  const Vec all = down(dout, out.size());
  float* dout1 = dev<float>(out.size());
  for (int r = 0; r < rows; ++r)
    attention_split(dq + size_t(r) * nh * hd, kv, nh, nkv, hd, 1, true, 1, float(1.0 / std::sqrt(hd)), P,
                    dqkv + size_t(r) * ld + nh * hd, ld, dout1 + size_t(r) * nh * hd, nh * hd, part, splits, TL, r);
  expect("attention, fp8 cache, row by row: same bits " + tag, down(dout1, out.size()), all, 0);
  {  // the tensor-core round kernel for every row, and for the rows from the middle one
    struct Choice {
      const char* name;
      int from, parts;
    };
    const Choice choices[] = {{"tensor cores", 0, 0},
                              {"tensor cores from the middle row", ctx + rows / 2 + 1, 0},
                              {"tensor cores, q and p in three parts", 0, 3}};
    for (const Choice& c : choices) {
      KernelOptions o;
      o.rows_mma = true;
      o.rows_mma_from = c.from;
      o.rows_mma_parts = c.parts;
      Launch LC = TL;
      LC.opt = &o;
      float* doutc = dev<float>(out.size());
      attention_split(dq, kv, nh, nkv, hd, rows, true, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld, doutc,
                      nh * hd, part, splits, LC);
      expect(std::string("attention, fp8 cache, ") + c.name + " " + tag, down(doutc, out.size()), out,
             rows_tolerance(true, c.parts, nh, nkv, hd, ctx));
      const Vec rows_all = down(doutc, out.size());
      for (int r = 0; r < rows; ++r)
        attention_split(dq + size_t(r) * nh * hd, kv, nh, nkv, hd, 1, true, 1, float(1.0 / std::sqrt(hd)), P,
                        dqkv + size_t(r) * ld + nh * hd, ld, doutc + size_t(r) * nh * hd, nh * hd, part, splits, LC, r);
      expect(std::string("attention, fp8 cache, ") + c.name + ", row by row: same bits " + tag,
             down(doutc, out.size()), rows_all, 0);
    }
  }
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  {  // q as e4m3: on the binary16 tensor cores everywhere, on the FP8 ones where they sum in float32
    // The reference with q rounded as the kernel rounds it: each query vector times 2^(8 - e), its largest |value|
    // below 2^e, to the nearest e4m3, and back.
    const Vec qk = down(dq, size_t(rows) * nh * hd);
    Vec out8(out.size()), q8(hd);
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < nh; ++h) {
        const double* x = &qk[(size_t(r) * nh + h) * hd];
        double mx = 0;
        for (int d = 0; d < hd; ++d) mx = std::max(mx, std::fabs(x[d]));
        int ex = 0;
        std::frexp(mx, &ex);
        for (int d = 0; d < hd; ++d)
          q8[d] = std::ldexp(e4m3_value(__nv_cvt_float_to_fp8(float(std::ldexp(x[d], 8 - ex)), __NV_SATFINITE,
                                                              __NV_E4M3)),
                             ex - 8);
        double* o = &out8[(size_t(r) * nh + h) * hd];
        ops::attend(q8.data(), Kr.data(), Vr.data(), ctx + r + 1, nkv, hd, h / (nh / nkv), 1.0 / std::sqrt(hd), o);
        for (int d = 0; d < hd; ++d) o[d] *= ops::sigmoid(qkv[size_t(r) * ld + nh * hd + h * hd + d]);
      }
    for (const bool fp8_mma : {false, true}) {
      if (fp8_mma && prop.major * 10 + prop.minor < 90) continue;  // as the backend sets KernelOptions::fp8_mma
      const std::string how = fp8_mma ? "q in e4m3 " : "q in e4m3 on binary16 ";
      KernelOptions o;
      o.rows_mma = true;
      o.rows_mma_from = 0;
      o.rows_mma_q8 = true;
      o.fp8_mma = fp8_mma;
      Launch LC = TL;
      LC.opt = &o;
      float* doutc = dev<float>(out.size());
      attention_split(dq, kv, nh, nkv, hd, rows, true, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld, doutc,
                      nh * hd, part, splits, LC);
      expect("attention, fp8 cache, tensor cores, " + how + tag, down(doutc, out.size()), out8,
             rows_tolerance(true, 0, nh, nkv, hd, ctx));
      const Vec rows_all = down(doutc, out.size());
      for (int r = 0; r < rows; ++r)
        attention_split(dq + size_t(r) * nh * hd, kv, nh, nkv, hd, 1, true, 1, float(1.0 / std::sqrt(hd)), P,
                        dqkv + size_t(r) * ld + nh * hd, ld, doutc + size_t(r) * nh * hd, nh * hd, part, splits, LC, r);
      expect("attention, fp8 cache, tensor cores, " + how + "row by row: same bits " + tag, down(doutc, out.size()),
             rows_all, 0);
    }
  }
  if (attention_flash(dq, kv, nh, nkv, hd, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld, dout, nh * hd, TL)) {
    expect("attention, fp8 cache, tensor cores " + tag, down(dout, out.size()), out, 3e-3);
    const int s = rows / 3;
    if (s > 0) {
      const Vec whole = down(dout, out.size());
      const float sc = float(1.0 / std::sqrt(hd));
      attention_flash(dq, kv, nh, nkv, hd, s, sc, P, dqkv + nh * hd, ld, dout1, nh * hd, TL);
      attention_flash(dq + size_t(s) * nh * hd, kv, nh, nkv, hd, rows - s, sc, P, dqkv + size_t(s) * ld + nh * hd, ld,
                      dout1 + size_t(s) * nh * hd, nh * hd, TL, s);
      expect("attention, fp8 cache, tensor cores, pass split at row " + std::to_string(s) + ": same bits " + tag,
             down(dout1, out.size()), whole, 0);
    }
    const Vec whole = down(dout, out.size());
    expect_windows<uint8_t>("attention, fp8 cache, tensor cores " + tag, dq, dK, dV, dKs, dVs, nh, nkv, hd, rows,
                            ctx + rows, P, dqkv + nh * hd, ld, whole);
    if (hd == 256 && nh == 24 && nkv == 4 && rows >= 16) {
      for (int hb : {2, 6})
        expect_windows<uint8_t>("attention, fp8 grouped, " + std::to_string(hb) + " heads/block " + tag, dq, dK, dV,
                                dKs, dVs, nh, nkv, hd, rows, ctx + rows, P, dqkv + nh * hd, ld, whole, hb);
      // Edge cases for deferred FP8 V scaling across FlashWindow boundaries:
      // zero-V window (kGqaVeUnset), tiny-scale window (1e-20), growing-scale window (1e+10),
      // and a masked future key with huge V scale (1e+35) that must never yield 0 * +INF = NaN.
      const int total_k = ctx + rows;
      std::vector<float> vs_edge(size_t(total_k) * nkv, 1e-20f);
      for (int pos = 0; pos < total_k; ++pos) {
        const float sv = pos < 64 ? 0.f : (pos < ctx + 8 ? 1e-20f : (pos < ctx + 12 ? 1e25f : 1e10f));
        for (int h = 0; h < nkv; ++h) vs_edge[size_t(pos) * nkv + h] = sv;
      }
      float* dVs_edge = dev<float>(vs_edge.size());
      CK(cudaMemcpy(dVs_edge, vs_edge.data(), vs_edge.size() * sizeof(float), cudaMemcpyHostToDevice));
      std::vector<uint8_t> v_bytes(size_t(total_k) * nkv * hd);
      CK(cudaMemcpy(v_bytes.data(), dV, v_bytes.size(), cudaMemcpyDeviceToHost));
      Vec Vr_edge(v_bytes.size());
      for (int pos = 0; pos < total_k; ++pos)
        for (int h = 0; h < nkv; ++h) {
          const double sv = vs_edge[size_t(pos) * nkv + h];
          for (int d = 0; d < hd; ++d) {
            const size_t idx = (size_t(pos) * nkv + h) * hd + d;
            Vr_edge[idx] = e4m3_value(v_bytes[idx]) * sv;
          }
        }
      const Vec q_edge = down(dq, size_t(rows) * nh * hd);
      Vec out_edge(out.size());
      for (int r = 0; r < rows; ++r)
        for (int h = 0; h < nh; ++h) {
          double* o = &out_edge[(size_t(r) * nh + h) * hd];
          ops::attend(&q_edge[(size_t(r) * nh + h) * hd], Kr.data(), Vr_edge.data(), ctx + r + 1, nkv, hd,
                      h / (nh / nkv), 1.0 / std::sqrt(hd), o);
          for (int d = 0; d < hd; ++d) o[d] *= ops::sigmoid(qkv[size_t(r) * ld + nh * hd + h * hd + d]);
        }
      const KvRef kv_edge{dK, dV, dKs, dVs_edge};
      attention_flash(dq, kv_edge, nh, nkv, hd, rows, float(1.0 / std::sqrt(hd)), P, dqkv + nh * hd, ld, dout,
                      nh * hd, TL);
      const Vec got_edge = down(dout, out_edge.size());
      expect("attention, fp8 deferred V-scale multi-binade & huge masked scale " + tag, got_edge, out_edge, 3e-3);
      for (int hb : {2, 3, 6})
        expect_windows<uint8_t>("attention, fp8 multi-binade, " + std::to_string(hb) + " heads/block " + tag, dq, dK,
                                dV, dKs, dVs_edge, nh, nkv, hd, rows, total_k, P, dqkv + nh * hd, ld, got_edge, hb);
      CK(cudaFree(dVs_edge));
    }
  }
}

// Strata attention. attn_prep writes the low-bit copy: every value within half a step of the
// rotated K / V (the grid from the head's least to its largest value, in binary16).  The tensor-core scan: the copy's
// attention (the partials combined) against float64 attention over the copy as stored, and each pair's log-sum-exp;
// with every pair refined, the cache's own attention; a row's output alone == among the round's rows, bit for bit
// (rule 4).  The first KV head's values are 30 times the others' (its copy's steps far past 4, which p times a step
// must hold in binary16).
void test_strata_layout(int hd, int nh, int nkv, int rows, int ctx, int bits, bool fp8, bool compact) {
  seed("strata", hd, nh, nkv, rows, ctx, bits, fp8 ? 1 : 0);
  const int G = nh / nkv, ld = 2 * nh * hd + 2 * nkv * hd, total = ctx + rows, cap = total + 16, slots = 64;
  const int rot = hd < 64 ? hd : 64, words = strata_words(hd, bits);
  const int stored_words = compact ? 24 : words;
  const double theta = 1e7, scale = 1.0 / std::sqrt(hd);
  const float offset = 1.0f;
  Vec qkv = randv(size_t(total) * ld), qn = randv(hd, 0.2), kn = randv(hd, 0.2);
  for (int pos = 0; pos < total; ++pos)
    for (int d = 0; d < hd; ++d) qkv[size_t(pos) * ld + 2 * nh * hd + nkv * hd + d] *= 30;
  const size_t nv = size_t(cap) * nkv * hd;
  void* dK = fp8 ? static_cast<void*>(dev<uint8_t>(nv)) : static_cast<void*>(dev<kv16>(nv));
  void* dV = fp8 ? static_cast<void*>(dev<uint8_t>(nv)) : static_cast<void*>(dev<kv16>(nv));
  float* dKs = fp8 ? dev<float>(size_t(cap) * nkv) : nullptr;
  float* dVs = fp8 ? dev<float>(size_t(cap) * nkv) : nullptr;
  const size_t ncode = size_t(strata_positions(cap)) * nkv * stored_words;
  uint32_t *dkq = dev<uint32_t>(ncode), *dvq = dev<uint32_t>(ncode);
  __half2 *dkqs = dev<__half2>(size_t(cap) * nkv), *dvqs = dev<__half2>(size_t(cap) * nkv);
  const KvRef kv{dK, dV, dKs, dVs, dkq, dvq, dkqs, dvqs, bits, stored_words, 0};
  float *dqkv = up(qkv), *dqn = up(qn), *dkn = up(kn), *dq = dev<float>(size_t(total) * nh * hd);
  attn_prep(dqkv, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot, nh, nkv, hd, total,
            params(0), kPosPosition, false, dq, kv, 0);
  CK(cudaDeviceSynchronize());
  // the copy as stored, decoded: Kq and Vq [total][nkv][hd] (rotated basis)
  std::vector<uint32_t> kq(ncode), vq(ncode);
  std::vector<__half2> kqs(size_t(cap) * nkv), vqs(kqs.size());
  CK(cudaMemcpy(kq.data(), dkq, ncode * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vq.data(), dvq, ncode * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(kqs.data(), dkqs, kqs.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vqs.data(), dvqs, vqs.size() * 4, cudaMemcpyDeviceToHost));
  KvRef native = kv;
  if (compact) {
    const size_t tiles = ncode / 384, canonical = tiles * 416;
    std::vector<uint32_t> kc(canonical), vc(canonical), baseline(canonical);
    for (size_t t = 0; t < tiles; ++t) {
      strata_snapshot_tile<true, false>(kq.data() + t * 384, kc.data() + t * 416);
      strata_snapshot_tile<false, false>(vq.data() + t * 384, vc.data() + t * 416);
    }
    kq.swap(kc);
    vq.swap(vc);
    native.words = words;
    native.k = fp8 ? static_cast<void*>(dev<uint8_t>(nv)) : static_cast<void*>(dev<kv16>(nv));
    native.v = fp8 ? static_cast<void*>(dev<uint8_t>(nv)) : static_cast<void*>(dev<kv16>(nv));
    native.ks = fp8 ? dev<float>(size_t(cap) * nkv) : nullptr;
    native.vs = fp8 ? dev<float>(size_t(cap) * nkv) : nullptr;
    native.kq = dev<uint32_t>(canonical);
    native.vq = dev<uint32_t>(canonical);
    native.kqs = dev<__half2>(kqs.size());
    native.vqs = dev<__half2>(vqs.size());
    attn_prep(dqkv, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot, nh, nkv, hd,
              total, params(0), kPosPosition, false, dq, native, 0);
    CK(cudaMemcpy(baseline.data(), native.kq, canonical * 4, cudaMemcpyDeviceToHost));
    bool equal = kq == baseline;
    CK(cudaMemcpy(baseline.data(), native.vq, canonical * 4, cudaMemcpyDeviceToHost));
    equal &= vq == baseline;
    std::vector<__half2> grids(kqs.size());
    CK(cudaMemcpy(grids.data(), native.kqs, grids.size() * 4, cudaMemcpyDeviceToHost));
    equal &= std::memcmp(kqs.data(), grids.data(), grids.size() * 4) == 0;
    CK(cudaMemcpy(grids.data(), native.vqs, grids.size() * 4, cudaMemcpyDeviceToHost));
    equal &= std::memcmp(vqs.data(), grids.data(), grids.size() * 4) == 0;
    std::printf("  %s compact3 writer matches every canonical K/V word and grid\n", equal ? "ok" : "FAIL");
    if (!equal) ++g_fail;
  }
  if (!compact && total >= 32) {
    // Staged prefill Strata write (strata_base = 16) into a garbage-filled buffer must match direct writes
    // byte-for-byte, including any unused padding words (e.g. hd=128, bits=3 where kwords < words).
    const int base = 16, nr = 16;
    const size_t sw = size_t(nr) * size_t(nkv) * size_t(stored_words), sg = size_t(nr) * size_t(nkv);
    KvRef staged = kv;
    staged.kq = dev<uint32_t>(sw);
    staged.vq = dev<uint32_t>(sw);
    staged.kqs = dev<__half2>(sg);
    staged.vqs = dev<__half2>(sg);
    CK(cudaMemset(staged.kq, 0xcd, sw * sizeof(uint32_t)));
    CK(cudaMemset(staged.vq, 0xcd, sw * sizeof(uint32_t)));
    CK(cudaMemset(staged.kqs, 0xcd, sg * sizeof(__half2)));
    CK(cudaMemset(staged.vqs, 0xcd, sg * sizeof(__half2)));
    attn_prep(dqkv + size_t(base) * ld, ld, 0, 2 * nh * hd, 2 * nh * hd + nkv * hd, dqn, dkn, offset, 1e-6f, theta, rot,
              nh, nkv, hd, nr, params(base), kPosPosition, false, dq + size_t(base) * nh * hd, staged, 0, 0, nullptr,
              base);
    std::vector<uint32_t> skq(sw), svq(sw);
    std::vector<__half2> skqs(sg), svqs(sg);
    CK(cudaMemcpy(skq.data(), staged.kq, sw * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(svq.data(), staged.vq, sw * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(skqs.data(), staged.kqs, sg * sizeof(__half2), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(svqs.data(), staged.vqs, sg * sizeof(__half2), cudaMemcpyDeviceToHost));
    const size_t woff = strata_words_at(base, 0, nkv, stored_words), goff = size_t(base) * size_t(nkv);
    bool stg_eq = std::equal(skq.begin(), skq.end(), kq.begin() + woff) &&
                  std::equal(svq.begin(), svq.end(), vq.begin() + woff) &&
                  std::memcmp(skqs.data(), kqs.data() + goff, sg * sizeof(__half2)) == 0 &&
                  std::memcmp(svqs.data(), vqs.data() + goff, sg * sizeof(__half2)) == 0;
    std::printf("  %s strata_base=%d staged write matches direct words (including padding) and grids\n",
                stg_eq ? "ok" : "FAIL", base);
    if (!stg_eq) ++g_fail;
    CK(cudaFree(staged.kq));
    CK(cudaFree(staged.vq));
    CK(cudaFree(staged.kqs));
    CK(cudaFree(staged.vqs));
  }
  const StrataLayout SL(hd, bits);
  const uint32_t mask = (1u << bits) - 1u;
  Vec Kq(size_t(total) * nkv * hd), Vq(Kq.size());
  for (int pos = 0; pos < total; ++pos)
    for (int j = 0; j < nkv; ++j) {
      const size_t at16 = strata_words_at(pos, j, nkv, words), g = size_t(pos) * nkv + j;
      const double klo = __low2float(kqs[g]), kst = __high2float(kqs[g]);
      const double vlo = __low2float(vqs[g]), vst = __high2float(vqs[g]);
      for (int d = 0; d < hd; ++d) {
        int i = 0, sh = 0;
        SL.k_code(pos & 15, d, &i, &sh);
        Kq[g * hd + d] = klo + (((kq[at16 + i] >> sh) & mask) + 0.5) * kst;
        SL.v_code(pos & 15, d, &i, &sh);
        Vq[g * hd + d] = vlo + (((vq[at16 + i] >> sh) & mask) + 0.5) * vst;
      }
    }
  auto rotate = [&](double* x) {  // x <- H x / sqrt(hd)
    for (int len = 1; len < hd; len <<= 1)
      for (int i = 0; i < hd; i += 2 * len)
        for (int k = i; k < i + len; ++k) {
          const double a = x[k], b = x[k + len];
          x[k] = a + b;
          x[k + len] = a - b;
        }
    for (int d = 0; d < hd; ++d) x[d] /= std::sqrt(double(hd));
  };
  // the writer: within half a step of the rotated K / V (and binary16's rounding of the grid's ends)
  double worst = 0;
  for (int pos = 0; pos < total; ++pos)
    for (int j = 0; j < nkv; ++j) {
      const double* vsrc = &qkv[size_t(pos) * ld + 2 * nh * hd + nkv * hd + j * hd];
      Vec k(hd), v(vsrc, vsrc + hd);
      ops::rmsnorm(&qkv[size_t(pos) * ld + 2 * nh * hd + j * hd], kn.data(), hd, 1e-6, offset, k.data());
      ops::rope(k.data(), pos, theta, rot);
      rotate(k.data());
      rotate(v.data());
      const size_t g = size_t(pos) * nkv + j;
      for (int kvi = 0; kvi < 2; ++kvi) {
        const Vec& x = kvi ? v : k;
        const double st = __high2float((kvi ? vqs : kqs)[g]);
        double xl = x[0], xh = x[0];
        for (double e : x) xl = std::min(xl, e), xh = std::max(xh, e);
        const double limit = 0.5 * st + (std::fabs(xl) + std::fabs(xh)) / 1024 + 1e-6;
        for (int d = 0; d < hd; ++d) worst = std::max(worst, std::fabs((kvi ? Vq : Kq)[g * hd + d] - x[d]) / limit);
      }
    }
  const std::string tag = "hd " + std::to_string(hd) + " G " + std::to_string(G) + " rows " + std::to_string(rows) +
                          " ctx " + std::to_string(ctx) + " " + std::to_string(bits) + "b" + (fp8 ? " fp8" : "") +
                          (compact ? " compact" : "");
  auto verdict = [&](bool ok, const std::string& what, double x) {
    std::printf("  %-4s %-48s %.2f (limit 1)\n", ok ? "ok" : "FAIL", what.c_str(), x);
    if (!ok) ++g_fail;
  };
  verdict(worst <= 1.0, "strata copy written, error / half step " + tag, worst);
  // the round: rows at ctx .. total - 1
  const Vec qall = down(dq, size_t(total) * nh * hd);
  auto setup = [&](StrataScratch& x, int n, int denom, int per_chunk = kStrataSelChunk) {
    x.how.denom = denom;
    x.how.least = 1;
    x.lse_stride = ((cap + 1) / 2 + 7) / 8 * 8;
    x.sel_stride = std::max(1, (cap + 1) / 2);
    x.chunks = std::clamp((x.lse_stride + per_chunk - 1) / per_chunk, 1, kStrataSelChunks);
    x.lse = dev<__half>(size_t(n) * nh * x.lse_stride);
    x.sel = dev<int>(size_t(n) * nkv * x.sel_stride);
    x.sel_n = dev<int>(size_t(n) * nkv);
    x.cand = dev<int>(size_t(n) * nkv * x.chunks * x.sel_stride);
    x.cand_key = dev<uint32_t>(size_t(n) * nkv * x.chunks * x.sel_stride);
    x.cand_n = dev<int>(size_t(n) * nkv * x.chunks);
    x.gathered = dev<int>(size_t(n) * nkv * x.chunks * x.sel_stride);
    x.gathered_key = dev<uint32_t>(size_t(n) * nkv * x.chunks * x.sel_stride);
    x.delta = dev<float>(size_t(n) * nkv * kStrataSlices * G * (2 * hd + 1));
    x.qr = dev<float>(size_t(n) * nh * hd);
    CK(cudaMemset(x.lse, 0x7f, size_t(n) * nh * x.lse_stride * 2));
  };
  StrataScratch sc;
  setup(sc, rows, 16);
  const size_t np = attention_partial_floats(rows, nh, hd, slots);
  float *part = dev<float>(np), *dout = dev<float>(size_t(rows) * nh * hd);
  const float* dqr = dq + size_t(ctx) * nh * hd;
  DevParams* P = params(ctx);
  attention_strata(dqr, kv, sc, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, TL);
  CK(cudaDeviceSynchronize());
  const Vec pa = down(part, np), all = down(dout, size_t(rows) * nh * hd);
  std::vector<uint16_t> lse(size_t(rows) * nh * sc.lse_stride);
  CK(cudaMemcpy(lse.data(), sc.lse, lse.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  if (compact) {
    std::vector<uint32_t> packed_out(all.size()), native_out(all.size()), packed_part(np), native_part(np);
    std::vector<uint16_t> native_lse(lse.size());
    CK(cudaMemcpy(packed_out.data(), dout, packed_out.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(packed_part.data(), part, np * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(sc.lse, 0x7f, lse.size() * sizeof(uint16_t)));
    attention_strata(dqr, native, sc, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, TL);
    CK(cudaMemcpy(native_out.data(), dout, native_out.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(native_part.data(), part, np * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(native_lse.data(), sc.lse, native_lse.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    const bool equal = packed_out == native_out && packed_part == native_part && lse == native_lse;
    verdict(equal, "compact3 native scan partials, pair scores and corrected output bits " + tag, equal ? 0 : 2);
    attention_strata(dqr, kv, sc, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, TL);
  }
  Vec got(size_t(rows) * nh * hd, 0.0), want(got.size(), 0.0);
  double lworst = 0;
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < nh; ++h) {
      const int n = ctx + r + 1, used = attn_split_used(n, slots), j = h / G;
      double* gr = &got[(size_t(r) * nh + h) * hd];
      double M = -INFINITY, Ls = 0;  // the scan's partials combined (rotated basis)
      for (int s2 = 0; s2 < used; ++s2) M = std::max(M, pa[((size_t(r) * nh + h) * slots + s2) * (hd + 2)]);
      for (int s2 = 0; s2 < used; ++s2) {
        const double* x = &pa[((size_t(r) * nh + h) * slots + s2) * (hd + 2)];
        if (std::isinf(x[0])) continue;
        const double f = std::exp(x[0] - M);
        Ls += x[1] * f;
        for (int d = 0; d < hd; ++d) gr[d] += x[2 + d] * f;
      }
      for (int d = 0; d < hd; ++d) gr[d] /= Ls;
      // float64 over the copy as stored
      Vec qr(qall.begin() + (size_t(ctx + r) * nh + h) * hd, qall.begin() + (size_t(ctx + r) * nh + h + 1) * hd);
      rotate(qr.data());
      Vec s(n);
      double mx = -INFINITY;
      for (int k = 0; k < n; ++k) {
        double dot = 0;
        for (int d = 0; d < hd; ++d) dot += qr[d] * Kq[(size_t(k) * nkv + j) * hd + d];
        s[k] = dot * scale;
        mx = std::max(mx, s[k]);
      }
      double sum = 0;
      for (int k = 0; k < n; ++k) sum += std::exp(s[k] - mx);
      double* wr = &want[(size_t(r) * nh + h) * hd];
      for (int k = 0; k < n; ++k) {
        const double p = std::exp(s[k] - mx) / sum;
        for (int d = 0; d < hd; ++d) wr[d] += p * Vq[(size_t(k) * nkv + j) * hd + d];
      }
      for (int i = 0; 2 * i < n; ++i) {
        const double a = s[2 * i], b = 2 * i + 1 < n ? s[2 * i + 1] : -INFINITY, m2 = std::max(a, b);
        const double ref = m2 + std::log(std::exp(a - m2) + (std::isinf(b) ? 0.0 : std::exp(b - m2)));
        const double x = __half2float(__ushort_as_half(lse[(size_t(r) * nh + h) * sc.lse_stride + i]));
        const double error = std::fabs(x - ref) / (std::fabs(ref) / 1024 + 4e-3);
        // The unwritten sentinel is a NaN, which std::max would silently ignore.
        lworst = std::isfinite(error) ? std::max(lworst, error) : INFINITY;
      }
    }
  expect("strata scan: the copy's attention " + tag, got, want, 3e-3);
  verdict(lworst <= 1.0, "strata scan: the pairs' log-sum-exps " + tag, lworst);
  if (hd == 256 && G == 6) {
    // Eight splits keep both launch ranges populated even in the thousand-key case.
    const int split_slots = 8;
    const size_t pn = attention_partial_floats(rows, nh, hd, split_slots), on = size_t(rows) * nh * hd;
    auto result = [&](cudaEvent_t ready) {
      CK(cudaMemset(part, ready ? 0xff : 0, pn * sizeof(float)));
      attention_strata(dqr, kv, sc, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part,
                       split_slots, TL, 0, ready);
      std::vector<uint32_t> bits_out(pn + on);
      CK(cudaMemcpy(bits_out.data(), part, pn * sizeof(float), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(bits_out.data() + pn, dout, on * sizeof(float), cudaMemcpyDeviceToHost));
      return bits_out;
    };
    const auto original = result(nullptr);
    cudaEvent_t ready;
    CK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CK(cudaEventRecord(ready));
    const bool equal = result(ready) == original;
    CK(cudaEventDestroy(ready));
    verdict(equal, "strata scan: two launches, identical partials and output " + tag, equal ? 0 : 2);
  }
  if (hd == 256 && G == 6 && rows > 1) {
    // Exercise the full gather-offset array on a moderate input, with more than 32 nonempty chunks.
    auto first_row = [&](const StrataScratch& x, int run_rows) {
      attention_strata(dqr, kv, x, nh, nkv, hd, run_rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, TL);
      const int k = x.how.groups(ctx + 1);
      std::vector<uint32_t> bits(size_t(nkv) * (k + 1) + nh * hd);
      CK(cudaMemcpy(bits.data(), x.sel_n, nkv * sizeof(int), cudaMemcpyDeviceToHost));
      for (int h = 0; h < nkv; ++h)
        CK(cudaMemcpy(bits.data() + nkv + h * k, x.sel + size_t(h) * x.sel_stride, k * sizeof(int),
                      cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(bits.data() + nkv * (k + 1), dout, nh * hd * sizeof(float), cudaMemcpyDeviceToHost));
      return bits;
    };
    const auto expected = first_row(sc, rows);
    StrataScratch many;
    setup(many, 1, 16, 1);
    const bool same = many.chunks == kStrataSelChunks && first_row(many, 1) == expected;
    verdict(same, "strata, maximum chunks vs one: pair IDs and output bits " + tag, same ? 0.0 : 2.0);
  }
  const StrataSel sparse{INT_MAX, 0, 0, 0}, longest{INT_MAX, 0, INT_MAX, INT_MAX / 2 + 1};
  const bool bounds = sparse.groups(INT_MAX) == 1 && longest.groups(INT_MAX) == INT_MAX / 2 + 1 &&
                      longest.groups(3) == 2 && longest.recent_only() && !StrataSel{1, 0, 15, 7}.recent_only() &&
                      !StrataSel{1, 9, 16, 8}.recent_only() && !StrataSel{0, 0, 16, 8}.recent_only();
  verdict(bounds, "strata pair-count bounds and recent-only eligibility", bounds ? 0.0 : 2.0);
  // Both selectors must return the same ordered pairs: the correction accumulates them in list order.
  // This window covers all pairs in the first row and part of the last row of a multirow pass.
  for (bool mma : {false, true}) {
    StrataScratch recent = sc;
    recent.how = {1 << 20, 0, 2 * (ctx / 2 + 1), ctx / 2 + 1};
    KernelOptions options;
    options.strata_mma = mma;
    Launch launch = TL;
    launch.opt = &options;
    const size_t counts = size_t(rows) * nkv, pairs = counts * recent.sel_stride, outputs = size_t(rows) * nh * hd;
    auto selected = [&](__half* scores) {
      recent.lse = scores;
      CK(cudaMemset(recent.sel, 0xff, pairs * sizeof(int)));
      attention_strata(dqr, kv, recent, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots,
                       launch);
      std::vector<uint32_t> bits(counts + pairs + outputs);
      CK(cudaMemcpy(bits.data(), recent.sel_n, counts * sizeof(int), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(bits.data() + counts, recent.sel, pairs * sizeof(int), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(bits.data() + counts + pairs, dout, outputs * sizeof(float), cudaMemcpyDeviceToHost));
      return bits;
    };
    const auto generic = selected(sc.lse);
    const bool same = recent.how.recent_only() && selected(nullptr) == generic;
    verdict(same, std::string("strata recent pairs and output bits, ") + (mma ? "MMA " : "cores ") + tag,
            same ? 0.0 : 2.0);
  }
  {  // a row alone: the same bits
    StrataScratch one;
    setup(one, 1, 16);
    float* d1 = dev<float>(size_t(nh) * hd);
    bool same = true;
    for (int r = 0; r < rows; ++r) {
      attention_strata(dqr + size_t(r) * nh * hd, kv, one, nh, nkv, hd, 1, float(scale), params(ctx + r), nullptr, 0,
                       d1, nh * hd, part, slots, TL);
      const Vec x = down(d1, size_t(nh) * hd);
      same = same && std::equal(x.begin(), x.end(), all.begin() + size_t(r) * nh * hd);
    }
    verdict(same, "strata, a row alone: the same bits " + tag, same ? 0.0 : 2.0);
  }
  {  // every pair refined: the cache's attention.  The copy's attention is taken out as the scan sums it (q and p in
     // one binary16 part, 3e-3 as attention_rows_mma's) and the pairs' copy values put back as the correction sums
     // them (float32): the scan's rounding, scaled by the copy's mass over the cache's, twice the one part's limit.
    StrataScratch every;
    setup(every, rows, 1);
    attention_strata(dqr, kv, every, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, TL);
    Vec Kr(nv), Vr(nv), exact(size_t(rows) * nh * hd);
    if (fp8) {
      std::vector<uint8_t> k8(nv), v8(nv);
      std::vector<float> ks(size_t(cap) * nkv), vs(ks.size());
      CK(cudaMemcpy(k8.data(), dK, nv, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(v8.data(), dV, nv, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ks.data(), dKs, ks.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(vs.data(), dVs, vs.size() * 4, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < nv; ++i) Kr[i] = e4m3_value(k8[i]) * ks[i / hd], Vr[i] = e4m3_value(v8[i]) * vs[i / hd];
      // Every row selects every completed pair. Four hot slots per head force collisions as well as duplicate claims.
      if (rows > 1 && ctx >= 32) {
        KvRef cached = kv;
        cached.hot_bits = 2;
        cached.host_from = 0;
        const int columns = 1 << cached.hot_bits, slots_total = nkv * columns;
        cached.hot_tag = dev<int32_t>(slots_total);
        cached.hot_claim = dev<int32_t>(slots_total);
        cached.hot = dev<uint8_t>(size_t(slots_total) * 4 * hd);
        cached.hot_scale = dev<float>(slots_total * 4);
        CK(cudaMemset(cached.hot_tag, 0xff, slots_total * sizeof(int32_t)));
        CK(cudaMemset(cached.hot_claim, 0x7f, slots_total * sizeof(int32_t)));
        std::vector<uint32_t> cold(exact.size()), held(exact.size());
        CK(cudaMemcpy(cold.data(), dout, cold.size() * sizeof(float), cudaMemcpyDeviceToHost));
        attention_strata(dqr, cached, every, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots,
                         TL);
        CK(cudaMemcpy(held.data(), dout, held.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int32_t> tags(slots_total), claims(slots_total), winners(columns, INT_MAX);
        std::vector<uint8_t> bytes(size_t(slots_total) * 4 * hd);
        std::vector<float> scales(slots_total * 4);
        CK(cudaMemcpy(tags.data(), cached.hot_tag, tags.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(claims.data(), cached.hot_claim, claims.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bytes.data(), cached.hot, bytes.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(scales.data(), cached.hot_scale, scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int q = 0; 2 * q + 1 < ctx; ++q) {
          const int sl = cached.hot_slot(q);
          winners[sl] = std::min(winners[sl], q);
        }
        bool same = cold == held;
        for (int sl = 0; sl < slots_total; ++sl) {
          const int q = winners[sl % columns], head = sl / columns;
          if (q == INT_MAX) {
            same &= tags[sl] == -1 && claims[sl] == kHotFree;
            continue;
          }
          same &= tags[sl] == q && claims[sl] == kHotFree;
          for (int p = 0; p < 4; ++p) {
            const auto& source = p < 2 ? k8 : v8;
            const size_t src = size_t(2 * q + (p & 1)) * nkv + head, dst = size_t(sl) * 4 + p;
            same &= scales[dst] == (p < 2 ? ks : vs)[src] &&
                    std::equal(bytes.data() + dst * hd, bytes.data() + (dst + 1) * hd, source.data() + src * hd);
          }
        }
        verdict(same, "strata hot-slot winners, FP8 bytes, scales and output bits " + tag, same ? 0.0 : 2.0);
        // Shifted round chunk (P past committed P_hot): pairs at or past P_hot->position must never enter kv.hot.
        KvRef shifted = cached;
        shifted.host_from = ctx - 4;
        CK(cudaMemset(shifted.hot_tag, 0xff, slots_total * sizeof(int32_t)));
        CK(cudaMemset(shifted.hot_claim, 0x7f, slots_total * sizeof(int32_t)));
        DevParams* P_hot = params(ctx - 4);
        attention_strata(dqr, shifted, every, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part,
                         slots, TL, 0, nullptr, P_hot);
        CK(cudaMemcpy(held.data(), dout, held.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tags.data(), shifted.hot_tag, tags.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(claims.data(), shifted.hot_claim, claims.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        bool clean = cold == held;
        for (int sl = 0; sl < slots_total; ++sl) clean &= tags[sl] == -1 && claims[sl] == kHotFree;
        verdict(clean, "strata shifted P_hot refuses uncommitted draft pairs " + tag, clean ? 0.0 : 2.0);
      }
    } else {
      Kr = down_f16(static_cast<kv16*>(dK), nv);
      Vr = down_f16(static_cast<kv16*>(dV), nv);
    }
    if (rows > 1 && ctx >= 64) {
      // Pageable host K/V gather (cold miss with forced hot collisions, warm hit, CUDA graph host node, and hot=null):
      // cold positions [sink, ring_lo) live ONLY in pageable RAM (absent from the GPU tail ring).
      const int elem = fp8 ? 1 : int(sizeof(kv16));
      const size_t row_bytes = size_t(nkv) * hd * elem;
      std::vector<uint8_t> k_page(size_t(cap) * row_bytes), v_page(size_t(cap) * row_bytes);
      CK(cudaMemcpy(k_page.data(), dK, k_page.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(v_page.data(), dV, v_page.size(), cudaMemcpyDeviceToHost));
      std::vector<float> ks_page, vs_page;
      if (fp8) {
        ks_page.resize(size_t(cap) * nkv);
        vs_page.resize(size_t(cap) * nkv);
        CK(cudaMemcpy(ks_page.data(), dKs, ks_page.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(vs_page.data(), dVs, vs_page.size() * sizeof(float), cudaMemcpyDeviceToHost));
      }
      const int sink = 8, ring_safe = 16, ring_tail = 32, ring_slots = sink + ring_tail;
      std::vector<uint8_t> k_ring_h(size_t(ring_slots) * row_bytes, 0x5au), v_ring_h(size_t(ring_slots) * row_bytes, 0x5au);
      std::vector<float> ks_ring_h(fp8 ? size_t(ring_slots) * nkv : 0, -999.f);
      std::vector<float> vs_ring_h(fp8 ? size_t(ring_slots) * nkv : 0, -999.f);
      auto put_ring = [&](int pos) {
        const int sl = pos < sink ? pos : sink + (pos - sink) % ring_tail;
        std::memcpy(k_ring_h.data() + size_t(sl) * row_bytes, k_page.data() + size_t(pos) * row_bytes, row_bytes);
        std::memcpy(v_ring_h.data() + size_t(sl) * row_bytes, v_page.data() + size_t(pos) * row_bytes, row_bytes);
        if (fp8) {
          std::memcpy(ks_ring_h.data() + size_t(sl) * nkv, ks_page.data() + size_t(pos) * nkv, nkv * sizeof(float));
          std::memcpy(vs_ring_h.data() + size_t(sl) * nkv, vs_page.data() + size_t(pos) * nkv, nkv * sizeof(float));
        }
      };
      for (int p = 0; p < sink; ++p) put_ring(p);
      for (int p = std::max(sink, total - ring_tail); p < total; ++p) put_ring(p);
      uint8_t* dK_ring = dev<uint8_t>(k_ring_h.size());
      uint8_t* dV_ring = dev<uint8_t>(v_ring_h.size());
      CK(cudaMemcpy(dK_ring, k_ring_h.data(), k_ring_h.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dV_ring, v_ring_h.data(), v_ring_h.size(), cudaMemcpyHostToDevice));
      float *dKs_ring = nullptr, *dVs_ring = nullptr;
      if (fp8) {
        dKs_ring = dev<float>(ks_ring_h.size());
        dVs_ring = dev<float>(vs_ring_h.size());
        CK(cudaMemcpy(dKs_ring, ks_ring_h.data(), ks_ring_h.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVs_ring, vs_ring_h.data(), vs_ring_h.size() * sizeof(float), cudaMemcpyHostToDevice));
      }
      const size_t max_misses = size_t(rows) * size_t(nkv) * size_t(every.sel_stride);
      int32_t* miss_map = dev<int32_t>(max_misses);
      auto align256 = [](size_t n) { return (n + 255) / 256 * 256; };
      const size_t req_off = 0;
      const size_t cnt_off = align256(max_misses * sizeof(PageGatherReq));
      const size_t bkv_off = cnt_off + 256;
      const size_t bsc_off = bkv_off + align256(max_misses * 4 * size_t(hd) * size_t(elem));
      const size_t pinned_bytes = bsc_off + align256(max_misses * 4 * sizeof(float));
      void *ph = nullptr, *pd = nullptr;
      CK(cudaHostAlloc(&ph, pinned_bytes, cudaHostAllocMapped));
      CK(cudaHostGetDevicePointer(&pd, ph, 0));
      auto *hb = static_cast<uint8_t*>(ph), *db = static_cast<uint8_t*>(pd);
      PageGatherCtx pctx{};
      pctx.k_page = k_page.data() + size_t(sink) * row_bytes;
      pctx.v_page = v_page.data() + size_t(sink) * row_bytes;
      pctx.ks_page = fp8 ? ks_page.data() + size_t(sink) * nkv : nullptr;
      pctx.vs_page = fp8 ? vs_page.data() + size_t(sink) * nkv : nullptr;
      pctx.req_host = reinterpret_cast<PageGatherReq*>(hb + req_off);
      pctx.req_dev = reinterpret_cast<PageGatherReq*>(db + req_off);
      pctx.miss_count_host = reinterpret_cast<int32_t*>(hb + cnt_off);
      pctx.miss_count_dev = reinterpret_cast<int32_t*>(db + cnt_off);
      pctx.miss_map = miss_map;
      pctx.bounce_kv_host = hb + bkv_off;
      pctx.bounce_kv_dev = db + bkv_off;
      pctx.bounce_scale_host = reinterpret_cast<float*>(hb + bsc_off);
      pctx.bounce_scale_dev = reinterpret_cast<float*>(db + bsc_off);
      pctx.nkv = nkv;
      pctx.hd = hd;
      pctx.elem = elem;
      pctx.max_misses = int(max_misses);
      pctx.cap = cap;
      pctx.device_positions = sink;

      std::vector<uint32_t> ref_bits(exact.size()), got_bits(exact.size());
      CK(cudaMemcpy(ref_bits.data(), dout, ref_bits.size() * sizeof(float), cudaMemcpyDeviceToHost));

      KvRef pkv = kv;
      pkv.k = dK_ring;
      pkv.v = dV_ring;
      pkv.ks = dKs_ring;
      pkv.vs = dVs_ring;
      pkv.ks_far = pkv.vs_far = nullptr;
      pkv.scale_split = INT32_MAX;
      pkv.ring = ring_slots;
      pkv.sink = sink;
      pkv.host_from = sink;
      pkv.ring_safe = ring_safe;
      pkv.page_ctx = &pctx;
      pkv.hot_bits = 2;  // 4 hot slots per KV head -> forced collisions + duplicate claims across rows
      const int cols = 1 << pkv.hot_bits, slots_total = nkv * cols;
      pkv.hot_tag = dev<int32_t>(slots_total);
      pkv.hot_claim = dev<int32_t>(slots_total);
      pkv.hot = dev<uint8_t>(size_t(slots_total) * 4 * size_t(hd) * size_t(elem));
      pkv.hot_scale = fp8 ? dev<float>(slots_total * 4) : nullptr;
      CK(cudaMemset(pkv.hot_tag, 0xff, slots_total * sizeof(int32_t)));
      CK(cudaMemset(pkv.hot_claim, 0x7f, slots_total * sizeof(int32_t)));

      cudaStream_t pst = nullptr;
      CK(cudaStreamCreateWithFlags(&pst, cudaStreamNonBlocking));
      Launch PL = TL;
      PL.st = pst;

      // 1) Cold cache miss + collisions -> bit-identical to pinned; verify hot_tag winners & cleared claims.
      attention_strata(dqr, pkv, every, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, PL);
      CK(cudaStreamSynchronize(pst));
      CK(cudaMemcpy(got_bits.data(), dout, got_bits.size() * sizeof(float), cudaMemcpyDeviceToHost));
      bool ok_page = (ref_bits == got_bits);
      std::vector<int32_t> tags(slots_total), claims(slots_total), winners(cols, INT_MAX);
      CK(cudaMemcpy(tags.data(), pkv.hot_tag, tags.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(claims.data(), pkv.hot_claim, claims.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
      const int ring_lo = std::max(sink, (ctx > ring_safe ? (ctx - ring_safe) : 0) & ~1);
      for (int q = sink / 2; 2 * q + 1 < ring_lo; ++q) {
        const int sl = pkv.hot_slot(q);
        winners[sl] = std::min(winners[sl], q);
      }
      for (int sl = 0; sl < slots_total; ++sl) {
        const int q = winners[sl % cols];
        ok_page &= (tags[sl] == (q == INT_MAX ? -1 : q)) && (claims[sl] == kHotFree);
      }
      // 2) Warm cache hit + CUDA graph capture/launch (with cudaLaunchHostFunc node).
      cudaGraph_t g = nullptr;
      cudaGraphExec_t ge = nullptr;
      CK(cudaStreamBeginCapture(pst, cudaStreamCaptureModeThreadLocal));
      attention_strata(dqr, pkv, every, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots, PL);
      CK(cudaStreamEndCapture(pst, &g));
      CK(cudaGraphInstantiate(&ge, g, 0));
      CK(cudaGraphDestroy(g));
      CK(cudaMemsetAsync(dout, 0, got_bits.size() * sizeof(float), pst));
      CK(cudaGraphLaunch(ge, pst));
      CK(cudaStreamSynchronize(pst));
      CK(cudaMemcpy(got_bits.data(), dout, got_bits.size() * sizeof(float), cudaMemcpyDeviceToHost));
      ok_page &= (ref_bits == got_bits);
      CK(cudaGraphExecDestroy(ge));
      // 3) No hot cache (hot == nullptr): every cold pair read via miss_map + bounce.
      KvRef pkv_nohot = pkv;
      pkv_nohot.hot = nullptr;
      pkv_nohot.hot_tag = pkv_nohot.hot_claim = nullptr;
      pkv_nohot.hot_scale = nullptr;
      CK(cudaMemsetAsync(dout, 0, got_bits.size() * sizeof(float), pst));
      attention_strata(dqr, pkv_nohot, every, nh, nkv, hd, rows, float(scale), P, nullptr, 0, dout, nh * hd, part, slots,
                       PL);
      CK(cudaStreamSynchronize(pst));
      CK(cudaMemcpy(got_bits.data(), dout, got_bits.size() * sizeof(float), cudaMemcpyDeviceToHost));
      ok_page &= (ref_bits == got_bits);
      verdict(ok_page, "strata pageable host gather (cold, graph+warm, no-hot) " + tag, ok_page ? 0.0 : 2.0);
      // Restore dout for the float64 reference comparison below.
      CK(cudaMemcpy(dout, ref_bits.data(), ref_bits.size() * sizeof(float), cudaMemcpyHostToDevice));

      CK(cudaStreamDestroy(pst));
      CK(cudaFreeHost(ph));
      for (void* p : std::initializer_list<void*>{dK_ring, dV_ring, dKs_ring, dVs_ring, miss_map, pkv.hot_tag,
                                                  pkv.hot_claim, pkv.hot, pkv.hot_scale})
        if (p) CK(cudaFree(p));
    }
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < nh; ++h)
        ops::attend(&qall[(size_t(ctx + r) * nh + h) * hd], Kr.data(), Vr.data(), ctx + r + 1, nkv, hd, h / G, scale,
                    &exact[(size_t(r) * nh + h) * hd]);
    expect("strata, every pair refined: the cache's attention " + tag, down(dout, exact.size()), exact, 6e-3);
  }
}

void test_strata(int hd, int nh, int nkv, int rows, int ctx, int bits, bool fp8) {
  test_strata_layout(hd, nh, nkv, rows, ctx, bits, fp8, false);
  if (hd == 256 && bits == 3) test_strata_layout(hd, nh, nkv, rows, ctx, bits, fp8, true);
}

void test_pageable_kv_ops() {
  seed("pageable_kv_ops", 0);
  // The device's part of the caches in tiers (tiered_device_bytes, tiered_device_positions): rows that divide the
  // mapping's 2 MiB end it on a row at once; rows of 768 bytes (3 KV heads of 128, binary16) asked for 13,000
  // positions end the first cache inside row 10,922 and the next, given those, at 8,192, where every cache then
  // agrees; rows of 2,560 and 1,024 bytes together agree on a position both end a row at.
  {
    const size_t gran = size_t(2) << 20;
    struct TierCase {
      int asked;
      std::vector<size_t> rows;
      int want;
    };
    bool ok_tiers = tiered_device_bytes(13000, 768, gran, false) == 4 * gran &&
                    tiered_device_bytes(10922, 768, gran, false) == 3 * gran &&
                    tiered_device_bytes(8192, 768, gran, false) == size_t(8192) * 768 &&
                    tiered_device_bytes(17000, 128, gran, true) == size_t(16384) * 128;
    for (const TierCase& tc : {TierCase{13000, {768}, 8192}, TierCase{15000, {768}, 8192}, TierCase{17000, {128}, 16384},
                               TierCase{17000, {2560}, 16384}, TierCase{20000, {2560, 1024}, 16384},
                               TierCase{5000, {2560, 768}, 0}, TierCase{100, {1024}, 0}, TierCase{0, {768}, 0}})
      for (bool even : {false, true}) {
        const int got = tiered_device_positions(tc.asked, tc.rows.data(), int(tc.rows.size()), gran, even);
        bool whole = got == tc.want && (!even || got % 2 == 0);
        for (size_t row : tc.rows) whole = whole && tiered_device_bytes(got, row, gran, even) == size_t(got) * row;
        ok_tiers = ok_tiers && whole;
      }
    std::printf("  %-4s tiered caches' device part: one position, a whole row of every cache\n", ok_tiers ? "ok" : "FAIL");
    g_fail += !ok_tiers;
  }
  // Test both aligned ring_pad=0 (nkv=4, row=1024) and unaligned ring_pad=1 (nkv=10, hd=128, binary16 -> row=2560,
  // non-divisor of 2 MiB VMM granule where slot `sink` straddles the VMM boundary and must stay untouched).
  struct RingCase {
    int nkv, hd, ring_pad;
    bool fp8;
    const char* name;
  };
  for (const RingCase rc : {RingCase{4, 128, 0, false, "binary16"}, RingCase{4, 128, 0, true, "fp8"},
                            RingCase{10, 128, 1, false, "binary16 nkv=10 ring_pad=1"},
                            RingCase{10, 128, 1, true, "fp8 nkv=10 ring_pad=1"}}) {
    const int nkv = rc.nkv, hd = rc.hd, elem = rc.fp8 ? 1 : int(sizeof(kv16));
    const bool fp8 = rc.fp8;
    const size_t row_bytes = size_t(nkv) * hd * elem;
    const int sink = 8, ring_base = sink + rc.ring_pad, tail = 32, ring = ring_base + tail, cap = 96;
    const int pos0 = 68, span = 12;  // wraps around tail boundary (68-8)%32 = 28 -> 28..31, 0..7
    std::vector<uint8_t> k_ring(size_t(ring) * row_bytes), v_ring(size_t(ring) * row_bytes);
    for (size_t i = 0; i < k_ring.size(); ++i) {
      k_ring[i] = uint8_t((i * 131 + 17 + (fp8 ? 3 : 0)) & 0xffu);
      v_ring[i] = uint8_t((i * 73 + 53 + (fp8 ? 9 : 0)) & 0xffu);
    }
    if (rc.ring_pad) {
      std::memset(k_ring.data() + size_t(sink) * row_bytes, 0xee, row_bytes);
      std::memset(v_ring.data() + size_t(sink) * row_bytes, 0xee, row_bytes);
    }
    std::vector<float> ks_ring(size_t(ring) * nkv), vs_ring(size_t(ring) * nkv);
    for (size_t i = 0; i < ks_ring.size(); ++i) {
      ks_ring[i] = 0.05f * float(i + 1);
      vs_ring[i] = 0.07f * float(i + 1);
    }
    if (rc.ring_pad) {
      for (int h = 0; h < nkv; ++h) ks_ring[size_t(sink) * nkv + h] = vs_ring[size_t(sink) * nkv + h] = -777.f;
    }
    uint8_t* dk = dev<uint8_t>(k_ring.size());
    uint8_t* dv = dev<uint8_t>(v_ring.size());
    uint8_t* dvk = dev<uint8_t>(size_t(cap) * row_bytes);
    uint8_t* dvv = dev<uint8_t>(size_t(cap) * row_bytes);
    CK(cudaMemcpy(dk, k_ring.data(), k_ring.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, v_ring.data(), v_ring.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dvk, 0, size_t(cap) * row_bytes));
    CK(cudaMemset(dvv, 0, size_t(cap) * row_bytes));
    float *dks = nullptr, *dvs = nullptr, *dvks = nullptr, *dvvs = nullptr;
    KvStageScales sc{};
    if (fp8) {
      dks = dev<float>(ks_ring.size());
      dvs = dev<float>(vs_ring.size());
      dvks = dev<float>(size_t(cap) * nkv);
      dvvs = dev<float>(size_t(cap) * nkv);
      CK(cudaMemcpy(dks, ks_ring.data(), ks_ring.size() * sizeof(float), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dvs, vs_ring.data(), vs_ring.size() * sizeof(float), cudaMemcpyHostToDevice));
      CK(cudaMemset(dvks, 0, size_t(cap) * nkv * sizeof(float)));
      CK(cudaMemset(dvvs, 0, size_t(cap) * nkv * sizeof(float)));
      sc = {dks, dvs, dvks, dvvs, nkv};
    }
    DevParams* P = params(pos0);
    kv_stage_ring(dk, dvk, dv, dvv, row_bytes, sink, ring_base, ring, pos0 - 6, cap, P, 0, span, kStageAll, 4, nullptr,
                  sc);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> gk(size_t(cap) * row_bytes), gv(size_t(cap) * row_bytes);
    CK(cudaMemcpy(gk.data(), dvk, gk.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gv.data(), dvv, gv.size(), cudaMemcpyDeviceToHost));
    std::vector<float> gks(fp8 ? size_t(cap) * nkv : 0), gvs(fp8 ? size_t(cap) * nkv : 0);
    if (fp8) {
      CK(cudaMemcpy(gks.data(), dvks, gks.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(gvs.data(), dvvs, gvs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    bool ok_stage = true;
    for (int p = pos0 - 6; p < pos0 + span; ++p) {
      const int sl = p < sink ? p : ring_base + (p - sink) % tail;
      ok_stage &= std::equal(gk.data() + size_t(p) * row_bytes, gk.data() + size_t(p + 1) * row_bytes,
                             k_ring.data() + size_t(sl) * row_bytes);
      ok_stage &= std::equal(gv.data() + size_t(p) * row_bytes, gv.data() + size_t(p + 1) * row_bytes,
                             v_ring.data() + size_t(sl) * row_bytes);
      if (fp8) {
        ok_stage &= std::equal(gks.data() + size_t(p) * nkv, gks.data() + size_t(p + 1) * nkv,
                               ks_ring.data() + size_t(sl) * nkv);
        ok_stage &= std::equal(gvs.data() + size_t(p) * nkv, gvs.data() + size_t(p + 1) * nkv,
                               vs_ring.data() + size_t(sl) * nkv);
      }
    }
    std::printf("  %-4s kv_stage_ring wrap-around copy %s\n", ok_stage ? "ok" : "FAIL", rc.name);
    g_fail += !ok_stage;

    // kv_drain_pageable: drain committed round rows (pos0 .. pos0+span-1) to mapped pinned drain ring slot.
    const int max_rows = 16;
    const unsigned long long seq_no = 11;
    const int rslot = int(seq_no % kRing);
    DevParams hp{};
    hp.position = pos0 + span;
    hp.seq_no = seq_no;
    CK(cudaMemcpy(P, &hp, sizeof(hp), cudaMemcpyHostToDevice));
    flyx_round_result hring[kRing]{};
    hring[rslot].position = pos0 + span;
    hring[rslot].n_tokens = span;
    hring[rslot].seq_no = seq_no;
    flyx_round_result* dring = dev<flyx_round_result>(kRing);
    CK(cudaMemcpy(dring, hring, sizeof(hring), cudaMemcpyHostToDevice));
    const size_t kv_sz = size_t(kRing) * max_rows * row_bytes;
    const size_t sc_sz = fp8 ? size_t(kRing) * max_rows * nkv * sizeof(float) : 0;
    void *dh = nullptr, *dd = nullptr;
    CK(cudaHostAlloc(&dh, 2 * kv_sz + 2 * sc_sz, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&dd, dh, 0));
    std::memset(dh, 0, 2 * kv_sz + 2 * sc_sz);
    auto *hb = static_cast<uint8_t*>(dh), *db = static_cast<uint8_t*>(dd);
    PageDrainLayer dl{};
    dl.k = dk;
    dl.v = dv;
    dl.ks = dks;
    dl.vs = dvs;
    dl.drain_k = db;
    dl.drain_v = db + kv_sz;
    dl.drain_ks = fp8 ? reinterpret_cast<float*>(db + 2 * kv_sz) : nullptr;
    dl.drain_vs = fp8 ? reinterpret_cast<float*>(db + 2 * kv_sz + sc_sz) : nullptr;
    dl.row_bytes = row_bytes;
    dl.device_positions = sink;
    dl.ring_base = ring_base;
    dl.ring = ring;
    dl.nkv = nkv;
    PageDrainLayer* ddl = dev<PageDrainLayer>(1);
    CK(cudaMemcpy(ddl, &dl, sizeof(dl), cudaMemcpyHostToDevice));
    kv_drain_pageable(ddl, 1, P, dring, max_rows, nullptr);
    CK(cudaDeviceSynchronize());
    bool ok_drain = true;
    const uint8_t* hk = hb;
    const uint8_t* hv = hb + kv_sz;
    const float* hks = fp8 ? reinterpret_cast<const float*>(hb + 2 * kv_sz) : nullptr;
    const float* hvs = fp8 ? reinterpret_cast<const float*>(hb + 2 * kv_sz + sc_sz) : nullptr;
    for (int r = 0; r < span; ++r) {
      const int p = pos0 + r;
      const int sl = ring_base + (p - sink) % tail;
      const size_t dst_row = size_t(rslot) * max_rows + size_t(r);
      ok_drain &= std::equal(hk + dst_row * row_bytes, hk + (dst_row + 1) * row_bytes,
                             k_ring.data() + size_t(sl) * row_bytes);
      ok_drain &= std::equal(hv + dst_row * row_bytes, hv + (dst_row + 1) * row_bytes,
                             v_ring.data() + size_t(sl) * row_bytes);
      if (fp8) {
        ok_drain &= std::equal(hks + dst_row * nkv, hks + (dst_row + 1) * nkv, ks_ring.data() + size_t(sl) * nkv);
        ok_drain &= std::equal(hvs + dst_row * nkv, hvs + (dst_row + 1) * nkv, vs_ring.data() + size_t(sl) * nkv);
      }
    }
    std::printf("  %-4s kv_drain_pageable round drain %s\n", ok_drain ? "ok" : "FAIL", rc.name);
    g_fail += !ok_drain;

    CK(cudaFreeHost(dh));
    for (void* p : std::initializer_list<void*>{dk, dv, dvk, dvv, dks, dvs, dvks, dvvs, P, dring, ddl})
      if (p) CK(cudaFree(p));
  }

  // Real VMM non-divisor regression (nkv=10, hd=128, binary16 -> row=2560, dev=4 MiB -> device_rows=1638,
  // 1638*2560 = 4,193,280 < 4,194,304): verify writing staged row `device_rows` into `vw` (which shares `hd` with
  // `va`) does NOT overwrite the wrapped ring token at slot `ring_base = 1639` in `va`.
  {
    auto get_fn = [](const char* name, auto* fn) {
      void* p = nullptr;
      cudaDriverEntryPointQueryResult q{};
      if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
          q != cudaDriverEntryPointSuccess || !p)
        return false;
      *fn = reinterpret_cast<std::remove_pointer_t<decltype(fn)>>(p);
      return true;
    };
    decltype(&cuMemGetAllocationGranularity) fn_gran = nullptr;
    decltype(&cuMemAddressReserve) fn_res = nullptr;
    decltype(&cuMemAddressFree) fn_free = nullptr;
    decltype(&cuMemCreate) fn_create = nullptr;
    decltype(&cuMemRelease) fn_rel = nullptr;
    decltype(&cuMemMap) fn_map = nullptr;
    decltype(&cuMemUnmap) fn_unmap = nullptr;
    decltype(&cuMemSetAccess) fn_acc = nullptr;
    if (get_fn("cuMemGetAllocationGranularity", &fn_gran) && get_fn("cuMemAddressReserve", &fn_res) &&
        get_fn("cuMemAddressFree", &fn_free) && get_fn("cuMemCreate", &fn_create) &&
        get_fn("cuMemRelease", &fn_rel) && get_fn("cuMemMap", &fn_map) && get_fn("cuMemUnmap", &fn_unmap) &&
        get_fn("cuMemSetAccess", &fn_acc)) {
      int dev_id = 0;
      CK(cudaGetDevice(&dev_id));
      CUmemAllocationProp dp{};
      dp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      dp.location = {CU_MEM_LOCATION_TYPE_DEVICE, dev_id};
      size_t gran = 0;
      if (fn_gran(&gran, &dp, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && gran > 0) {
        const size_t row = 10 * 128 * sizeof(kv16);  // 2560 bytes: does not divide 2 MiB
        size_t dev_b = 2 * gran;
        while (dev_b >= gran && ((dev_b / row) & 1)) dev_b -= gran;
        const int device_rows = int(dev_b / row);
        const int ring_pad = (dev_b > 0 && (dev_b % row) != 0) ? 1 : 0;
        const int ring_base = device_rows + ring_pad;
        const int tail = 64;
        const size_t ring_vmm = ((size_t(tail + ring_pad) * row + gran - 1) / gran) * gran;
        CUdeviceptr va = 0, vw = 0;
        CUmemGenericAllocationHandle hd = 0, hr = 0, hs = 0;
        CUmemAccessDesc acc{};
        acc.location = {CU_MEM_LOCATION_TYPE_DEVICE, dev_id};
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (fn_res(&va, dev_b + ring_vmm, gran, 0, 0) == CUDA_SUCCESS &&
            fn_res(&vw, dev_b + gran, gran, 0, 0) == CUDA_SUCCESS &&
            fn_create(&hd, dev_b, &dp, 0) == CUDA_SUCCESS && fn_create(&hr, ring_vmm, &dp, 0) == CUDA_SUCCESS &&
            fn_create(&hs, gran, &dp, 0) == CUDA_SUCCESS) {
          fn_map(va, dev_b, 0, hd, 0);
          fn_map(va + dev_b, ring_vmm, 0, hr, 0);
          fn_map(vw, dev_b, 0, hd, 0);
          fn_map(vw + dev_b, gran, 0, hs, 0);
          fn_acc(va, dev_b + ring_vmm, &acc, 1);
          fn_acc(vw, dev_b + gran, &acc, 1);
          std::vector<uint8_t> recent_tok(row, 0xa5u), old_tok(row, 0x3cu), got_recent(row, 0u);
          // Wrapped token at pos = device_rows + tail lands in slot `ring_base` of `va`.
          KvRef kr{};
          kr.ring = ring_base + tail;
          kr.sink = device_rows;
          kr.ring_pad = ring_pad;
          const int wrapped_slot = kr.slot_of(device_rows + tail);
          CK(cudaMemcpy(reinterpret_cast<uint8_t*>(va) + size_t(wrapped_slot) * row, recent_tok.data(), row,
                        cudaMemcpyHostToDevice));
          // Staging old token `device_rows` into `vw + device_rows * row` writes into shared `hd` tail bytes.
          CK(cudaMemcpy(reinterpret_cast<uint8_t*>(vw) + size_t(device_rows) * row, old_tok.data(), row,
                        cudaMemcpyHostToDevice));
          CK(cudaMemcpy(got_recent.data(), reinterpret_cast<uint8_t*>(va) + size_t(wrapped_slot) * row, row,
                        cudaMemcpyDeviceToHost));
          const bool ok_vmm = (ring_pad == 1) && (wrapped_slot == device_rows + 1) && (got_recent == recent_tok);
          std::printf("  %-4s VMM unaligned row=2560 view staging preserves wrapped tail ring\n",
                      ok_vmm ? "ok" : "FAIL");
          g_fail += !ok_vmm;
          fn_unmap(vw + dev_b, gran);
          fn_unmap(vw, dev_b);
          fn_unmap(va + dev_b, ring_vmm);
          fn_unmap(va, dev_b);
        }
        if (hs) fn_rel(hs);
        if (hr) fn_rel(hr);
        if (hd) fn_rel(hd);
        if (vw) fn_free(vw, dev_b + gran);
        if (va) fn_free(va, dev_b + ring_vmm);
      }
    }
  }

  // Windowed attention_split (SplitWindow): must reproduce single-pass attention_split bit-for-bit across
  // k_attention_gqa, k_attention_rows_mma (and mixed rows across rows_mma_from), and per-head k_attention.
  // A row's splits are 128 keys at ctx 400 (each window inside one split), 96 at 280 and 160 at 1100, where a split
  // ends inside a window: the split before it reads the state the launch before left while the one after it leaves
  // its own, two blocks of one launch; and windows of 384 keys at 1100, which hold a whole split between those two.
  struct WindowCase {
    int ctx, win_len;
  };
  for (const WindowCase wc : {WindowCase{400, 64}, WindowCase{280, 64}, WindowCase{1100, 64}, WindowCase{1100, 384}}) {
    const int hd = 128, nh = 8, nkv = 2, rows = 9, ctx = wc.ctx, cap = ctx + rows + 16, splits = 8;
    const int from = 64, win_len = wc.win_len, end = ctx + rows;
    const float scale = float(1.0 / std::sqrt(hd));
    const Vec q = randv(size_t(rows) * nh * hd), gate = randv(size_t(rows) * nh * hd);
    Vec Kc = randv(size_t(cap) * nkv * hd), Vc = randv(size_t(cap) * nkv * hd);
    round_f16(Kc);
    round_f16(Vc);
    float* dq = up(q);
    float* dgate = up(gate);
    float* part = dev<float>(attention_partial_floats(rows, nh, hd, splits));
    const size_t plane = size_t(rows) * nh * kSplitStateFloats(hd);  // read from one, written to the other
    float* split_st = dev<float>(2 * plane);
    float* dref = dev<float>(size_t(rows) * nh * hd);
    float* dgot = dev<float>(size_t(rows) * nh * hd);
    DevParams* P = params(ctx);
    for (bool fp8 : {false, true}) {
      const size_t elem = fp8 ? 1 : sizeof(kv16);
      const size_t row_bytes = size_t(nkv) * hd * elem;
      void *dK = nullptr, *dV = nullptr, *dKw = nullptr, *dVw = nullptr;
      float *dKs = nullptr, *dVs = nullptr, *dKsw = nullptr, *dVsw = nullptr;
      if (fp8) {
        std::vector<uint8_t> kq(size_t(cap) * nkv * hd), vq(size_t(cap) * nkv * hd);
        std::vector<float> ks(size_t(cap) * nkv), vs(size_t(cap) * nkv);
        for (int p = 0; p < cap; ++p)
          for (int h = 0; h < nkv; ++h) {
            const size_t base = (size_t(p) * nkv + h) * hd;
            double km = 0, vm = 0;
            for (int d = 0; d < hd; ++d) km = std::max(km, std::fabs(Kc[base + d])), vm = std::max(vm, std::fabs(Vc[base + d]));
            ks[size_t(p) * nkv + h] = float(km > 0 ? km / 448.0 : 1.0);
            vs[size_t(p) * nkv + h] = float(vm > 0 ? vm / 448.0 : 1.0);
            for (int d = 0; d < hd; ++d) {
              kq[base + d] = uint8_t(__nv_cvt_float_to_fp8(float(Kc[base + d] / ks[size_t(p) * nkv + h]),
                                                           __NV_SATFINITE, __NV_E4M3));
              vq[base + d] = uint8_t(__nv_cvt_float_to_fp8(float(Vc[base + d] / vs[size_t(p) * nkv + h]),
                                                           __NV_SATFINITE, __NV_E4M3));
            }
          }
        dK = dev<uint8_t>(kq.size());
        dV = dev<uint8_t>(vq.size());
        CK(cudaMemcpy(dK, kq.data(), kq.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV, vq.data(), vq.size(), cudaMemcpyHostToDevice));
        dKs = dev<float>(ks.size());
        dVs = dev<float>(vs.size());
        CK(cudaMemcpy(dKs, ks.data(), ks.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVs, vs.data(), vs.size() * sizeof(float), cudaMemcpyHostToDevice));
        dKw = dev<uint8_t>(size_t(win_len) * row_bytes);
        dVw = dev<uint8_t>(size_t(win_len) * row_bytes);
        dKsw = dev<float>(size_t(win_len) * nkv);
        dVsw = dev<float>(size_t(win_len) * nkv);
      } else {
        dK = up_f16(Kc);
        dV = up_f16(Vc);
        dKw = dev<kv16>(size_t(win_len) * nkv * hd);
        dVw = dev<kv16>(size_t(win_len) * nkv * hd);
      }
      struct Mode {
        const char* name;
        bool mma;
        int mma_from, attn;
      };
      std::vector<Mode> modes = {
          {"gqa fp32", false, 0, kAttnAuto},
          {"rows_mma all", true, 0, kAttnAuto},
          {"rows_mma+gqa mixed", true, ctx + 4, kAttnAuto},
      };
      if (!fp8) modes.push_back({"per-head", false, 0, kAttnPerHead});
      for (const Mode& md : modes) {
        KernelOptions o{};
        o.rows_mma = md.mma;
        o.rows_mma_from = md.mma_from;
        o.attention = md.attn;
        Launch L = TL;
        L.opt = &o;
        KvRef full{dK, dV, dKs, dVs};
        CK(cudaMemset(dref, 0, size_t(rows) * nh * hd * sizeof(float)));
        attention_split(dq, full, nh, nkv, hd, rows, true, 0, scale, P, dgate, nh * hd, dref, nh * hd, part, splits, L);
        std::vector<uint32_t> ref_bits(size_t(rows) * nh * hd), got_bits(size_t(rows) * nh * hd);
        CK(cudaMemcpy(ref_bits.data(), dref, ref_bits.size() * sizeof(float), cudaMemcpyDeviceToHost));

        CK(cudaMemset(part, 0xaa, attention_partial_floats(rows, nh, hd, splits) * sizeof(float)));
        CK(cudaMemset(split_st, 0x55, 2 * plane * sizeof(float)));
        CK(cudaMemset(dgot, 0, size_t(rows) * nh * hd * sizeof(float)));
        SplitWindow sw;
        int launches = 0;
        auto swap_state = [&] {  // as regions/layers.cpp attn_windows does before every launch
          sw.state = split_st + size_t(launches & 1) * plane;
          sw.carry = split_st + size_t((launches + 1) & 1) * plane;
          ++launches;
        };
        // First window: device prefix [0, from).
        sw.k_lo = 0;
        sw.k_hi = from;
        sw.first = true;
        sw.last = false;
        swap_state();
        attention_split(dq, full, nh, nkv, hd, rows, true, 0, scale, P, dgate, nh * hd, dgot, nh * hd, part, splits, L,
                        0, 0, nullptr, {}, sw);
        sw.first = false;
        const int nwin = (end - from + win_len - 1) / win_len;
        for (int wi = 0; wi < nwin; ++wi) {
          const int w0 = from + wi * win_len, w1 = std::min(end, w0 + win_len), wn = w1 - w0;
          CK(cudaMemcpy(dKw, static_cast<const uint8_t*>(dK) + size_t(w0) * row_bytes, size_t(wn) * row_bytes,
                        cudaMemcpyDeviceToDevice));
          CK(cudaMemcpy(dVw, static_cast<const uint8_t*>(dV) + size_t(w0) * row_bytes, size_t(wn) * row_bytes,
                        cudaMemcpyDeviceToDevice));
          KvRef wkv{reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dKw) - size_t(w0) * row_bytes),
                    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dVw) - size_t(w0) * row_bytes), dKs, dVs};
          if (fp8) {
            CK(cudaMemcpy(dKsw, dKs + size_t(w0) * nkv, size_t(wn) * nkv * sizeof(float), cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(dVsw, dVs + size_t(w0) * nkv, size_t(wn) * nkv * sizeof(float), cudaMemcpyDeviceToDevice));
            wkv.ks_far = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(dKsw) - size_t(w0) * nkv * sizeof(float));
            wkv.vs_far = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(dVsw) - size_t(w0) * nkv * sizeof(float));
            wkv.scale_split = from;
          }
          sw.k_lo = w0;
          sw.k_hi = w1;
          sw.last = (wi + 1 == nwin);
          swap_state();
          attention_split(dq, wkv, nh, nkv, hd, rows, true, 0, scale, P, dgate, nh * hd, dgot, nh * hd, part, splits, L,
                          0, 0, nullptr, {}, sw);
        }
        CK(cudaMemcpy(got_bits.data(), dgot, got_bits.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const bool ok_sw = (got_bits == ref_bits);
        std::printf("  %-4s SplitWindow bit-identical (%s, %s, %d keys, windows of %d)\n", ok_sw ? "ok" : "FAIL",
                    md.name, fp8 ? "fp8" : "binary16", ctx, win_len);
        g_fail += !ok_sw;
      }
      for (void* p : std::initializer_list<void*>{dK, dV, dKw, dVw, dKs, dVs, dKsw, dVsw})
        if (p) CK(cudaFree(p));
    }
    for (void* p : std::initializer_list<void*>{dq, dgate, part, split_st, dref, dgot, P}) CK(cudaFree(p));
  }
}

// A tree round of more than 16 rows (up to FLYX_MAX_WIDE_ROWS; docs/design/tree-verification.md): row r, its K/V in
// the round's slot base + r, attends to the keys before the round and to its ancestors' in depth order with the bits
// of decoding its path alone: one row at position base + depth[r] over a cache whose slots base .. base + depth[r]
// hold its path's K/V.  Through attention(), which takes the split kernels for a tree (never prefill's), with every
// kernel choice: by the row's own key count (a context about rows_attention_from: some rows on tensor cores, the
// others on the float32 kernels), the float32 kernels alone, the tensor cores for every row, per head.  Against
// float64 too (the keys a row reads).
void test_attention_tree(int hd, int nh, int nkv, int ctx) {
  std::printf("tree rows past 16 (hd %d, %d heads on %d, context %d)\n", hd, nh, nkv, ctx);
  seed("attention tree", hd, nh, nkv, ctx);
  const int most = FLYX_MAX_WIDE_ROWS, cap = ctx + most + 16, splits = 64, row = nh * hd;
  const float scale = float(1.0 / std::sqrt(hd));
  Vec Kc = randv(size_t(cap) * nkv * hd), Vc = randv(size_t(cap) * nkv * hd);
  round_f16(Kc);
  round_f16(Vc);
  kv16 *dK = up_f16(Kc), *dV = up_f16(Vc), *dKr = up_f16(Kc), *dVr = up_f16(Vc);
  const size_t slot_bytes = size_t(nkv) * hd * sizeof(kv16);
  DevParams* P = params(ctx);
  TreeRows* dtree = dev<TreeRows>(1);
  float* part = dev<float>(attention_partial_floats(most, nh, hd, splits));
  float* part1 = dev<float>(attention_partial_floats(1, nh, hd, splits));
  float *dout = dev<float>(size_t(most) * row), *d1 = dev<float>(row);
  // parent(r) of rows 1 ..: the chain (depth 31 at row 31), a random tree, a trunk of 40 rows and random branches
  struct Shape {
    const char* name;
    int rows, trunk;
  };
  for (const Shape sh : {Shape{"a chain", 32, 32}, Shape{"a random tree", 32, 1},
                         Shape{"a trunk, then branches", 64, 40}, Shape{"a random tree", 17, 1}}) {
    const int rows = sh.rows;
    TreeRows t{};
    t.n = rows;
    for (int r = 1; r < rows; ++r) {
      const int p = r < sh.trunk ? r - 1 : int(rng() % uint32_t(r));
      t.depth[r] = t.depth[p] + 1;
      for (int d = 0; d < t.depth[r]; ++d) t.path[r][d] = t.path[p][d];
      t.path[r][t.depth[r]] = static_cast<unsigned char>(r);
    }
    int deepest = 0;
    for (int r = 0; r < rows; ++r) deepest = std::max(deepest, t.depth[r]);
    CK(cudaMemcpy(dtree, &t, sizeof(t), cudaMemcpyHostToDevice));
    const Vec q = randv(size_t(rows) * row);
    float* dq = up(q);
    const std::string tag = std::string(sh.name) + " of " + std::to_string(rows) + " rows (depth " +
                            std::to_string(deepest) + ")";
    // float64: row r's keys, the context's then its path's
    {
      Vec want(size_t(rows) * row);
      for (int r = 0; r < rows; ++r) {
        const int n = ctx + t.depth[r] + 1;
        Vec Kr(size_t(n) * nkv * hd), Vr(Kr.size());
        for (int j = 0; j < n; ++j) {
          const int s = j < ctx ? j : ctx + t.path[r][j - ctx];
          std::copy_n(&Kc[size_t(s) * nkv * hd], size_t(nkv) * hd, &Kr[size_t(j) * nkv * hd]);
          std::copy_n(&Vc[size_t(s) * nkv * hd], size_t(nkv) * hd, &Vr[size_t(j) * nkv * hd]);
        }
        for (int h = 0; h < nh; ++h)
          ops::attend(&q[(size_t(r) * nh + h) * hd], Kr.data(), Vr.data(), n, nkv, hd, h / (nh / nkv), scale,
                      &want[(size_t(r) * nh + h) * hd]);
      }
      Launch LR = TL;
      LR.round_rows = true;  // a round's rows (a sequence's own buffers): never the prefill kernels
      attention(dq, KvRef{dK, dV, nullptr, nullptr}, nh, nkv, hd, rows, true, 0, scale, P, nullptr, 0, dout, row,
                part, splits, LR, 0, 0, dtree);
      expect("attention, " + tag, down(dout, want.size()), want, ctx > 1000 ? 1e-4 : 2e-5);
    }
    struct Choice {
      const char* name;
      bool mma;
      int from, attention;
    };
    const KernelOptions defaults{};
    for (const Choice c : {Choice{"by the row's keys", true, defaults.rows_mma_from, kAttnAuto},
                           Choice{"float32 kernels", false, 0, kAttnAuto}, Choice{"tensor cores", true, 0, kAttnAuto},
                           Choice{"per head", false, 0, kAttnPerHead}}) {
      KernelOptions o;
      o.rows_mma = c.mma;
      o.rows_mma_from = c.from;
      o.attention = c.attention;
      Launch L = TL;
      L.opt = &o;
      L.round_rows = true;  // a round's rows (a sequence's own buffers): never the prefill kernels
      attention(dq, KvRef{dK, dV, nullptr, nullptr}, nh, nkv, hd, rows, true, 0, scale, P, nullptr, 0, dout, row, part,
                splits, L, 0, 0, dtree);
      const Vec all = down(dout, size_t(rows) * row);
      bool same = true;
      for (int r = 0; r < rows; ++r) {
        for (int d = 0; d <= t.depth[r]; ++d) {  // the path's K/V where decoding it alone has them
          CK(cudaMemcpy(reinterpret_cast<char*>(dKr) + size_t(ctx + d) * slot_bytes,
                        reinterpret_cast<char*>(dK) + size_t(ctx + t.path[r][d]) * slot_bytes, slot_bytes,
                        cudaMemcpyDeviceToDevice));
          CK(cudaMemcpy(reinterpret_cast<char*>(dVr) + size_t(ctx + d) * slot_bytes,
                        reinterpret_cast<char*>(dV) + size_t(ctx + t.path[r][d]) * slot_bytes, slot_bytes,
                        cudaMemcpyDeviceToDevice));
        }
        attention(dq + size_t(r) * row, KvRef{dKr, dVr, nullptr, nullptr}, nh, nkv, hd, 1, true, 1, scale, P, nullptr,
                  0, d1, row, part1, splits, L, t.depth[r]);
        const Vec one = down(d1, row);
        same &= std::equal(one.begin(), one.end(), all.begin() + size_t(r) * row);
      }
      std::printf("  %-4s %s, %s: every row the bits of decoding its path alone\n", same ? "ok" : "FAIL", tag.c_str(),
                  c.name);
      g_fail += !same;
    }
    CK(cudaFree(dq));
  }
  for (void* p : std::initializer_list<void*>{dK, dV, dKr, dVr, P, dtree, part, part1, dout, d1}) CK(cudaFree(p));
}

}  // namespace flyx::optest
