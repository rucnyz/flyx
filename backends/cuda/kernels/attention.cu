// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Attention for decode, verification and drafting: K/V prep, split-K (per head or per KV-head group), combine; the dispatch to prefill attention.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include <climits>

namespace flyx::cuda {
namespace {

// ---- attention ---------------------------------------------------------------------------
// grid (ceil((nh + nkv) / hpb), rows); block = hd threads, heads hpb * blockIdx.x .. of row blockIdx.y
// (the q heads, then the k heads).  The rotation of pair i (dims i and i + half)
// is computed once per block: rope_cs is double pow and fmod, slow on this GPU
// (1/64-rate float64), so many rows give a block several heads; decode keeps one
// per block.  The barriers of block_sum separate a head's reads of sh from the
// next head's writes.
// CANVAS: a DV canvas pass's rows (CanvasKeys): row r at P->start + 1 + pos_add + r, its K/V at slot
// canvas_s0 + pos_add + r.
template <bool Compact, bool CANVAS, bool MROPE = false>
__global__ void k_attn_prep(const float* qkv, int ld, int q_off, int k_off, int v_off, const float* q_norm,
                            const float* k_norm, float norm_offset, float eps, double theta, int rot_dim, int nh,
                            int nkv, int hd, const DevParams* P, int pos_sel, int skip_if_noop, float* q_out, KvRef kv,
                            int pos_add, int hpb, const TreeRows* tree, const double* rope, int strata_base,
                            int canvas_s0, MRoPE mrope = {}) {
  pdl_trigger();
  extern __shared__ float sh[];
  if (skip_if_noop && P->noop) return;
  const int r = blockIdx.y, d = threadIdx.x;
  // A tree's row is rotated to its depth's position and written to its own row's slot.
  const int base = CANVAS ? P->start + 1 + pos_add : (pos_sel == kPosStart ? P->start : P->position) + pos_add;
  const int pos = base + tree_depth(tree, r), slot = CANVAS ? canvas_s0 + pos_add + r : base + r;
  const int half = rot_dim / 2;
  float* cs = sh + hd;
  if (d < half) {  // the pair's factor from rope_table, or its pow (a pair never registered)
    int rpos = pos;
    if constexpr (MROPE) {
      int pt, ph, pw;
      if (!CANVAS && mrope.pos_3d) {
        pt = mrope.pos_3d[3 * r + 0];
        ph = mrope.pos_3d[3 * r + 1];
        pw = mrope.pos_3d[3 * r + 2];
      } else {
        pt = ph = pw = pos + P->rope_delta;
      }
      int ax = 0;
      if (mrope.interleaved) {
        if (d % 3 == 1 && d < 3 * mrope.sec[1]) ax = 1;
        else if (d % 3 == 2 && d < 3 * mrope.sec[2]) ax = 2;
      } else {
        if (d < mrope.sec[0]) ax = 0;
        else if (d < mrope.sec[0] + mrope.sec[1]) ax = 1;
        else ax = 2;
      }
      rpos = ax == 0 ? pt : (ax == 1 ? ph : pw);
    }
    if (rope) rope_cs_inv(rpos, rope[d], &cs[d], &cs[half + d]);
    else rope_cs(rpos, d, rot_dim, theta, &cs[d], &cs[half + d]);
  }
  const int h_end = min(nh + nkv, (blockIdx.x + 1) * hpb);
  for (int hidx = blockIdx.x * hpb; hidx < h_end; ++hidx) {
    const bool is_q = hidx < nh;
    const int head = is_q ? hidx : hidx - nh;
    const float* src = qkv + size_t(r) * ld + (is_q ? q_off : k_off) + head * hd;
    const float* norm_w = is_q ? q_norm : k_norm;  // null: no norm (block-uniform, like is_q)
    float nv = src[d];
    if (norm_w) {
      const float v = nv, nw = norm_w[d];  // the weight loads with v, not after the sum
      const float inv = rsqrtf(block_sum(v * v) / hd + eps);
      nv = v * inv * (norm_offset + nw);
    } else {
      __syncthreads();  // what block_sum's barriers do otherwise: the last head's reads of sh before this one's write
    }
    sh[d] = nv;
    __syncthreads();
    float out = nv;
    if (d < rot_dim) {
      const int i = d < half ? d : d - half;
      const float c = cs[i], s = cs[half + i];
      out = d < half ? nv * c - sh[d + half] * s : nv * c + sh[d - half] * s;
    }
    if (is_q) {
      q_out[(size_t(r) * nh + head) * hd + d] = out;
    } else if (slot >= 0) {  // block-uniform, like is_q
      const int cell = kv.slot_of(slot);  // a ring's slot
      const size_t at = (size_t(cell) * nkv + head) * hd + d;
      const float vv = qkv[size_t(r) * ld + v_off + head * hd + d];
      if (kv.fp8()) {  // e4m3 with the row's scale amax / 448
        float amax[2] = {fabsf(out), fabsf(vv)};
        block_max_n(amax);
        const float ksc = amax[0] / 448.f, vsc = amax[1] / 448.f;
        static_cast<uint8_t*>(kv.k)[at] = ksc > 0.f ? __nv_cvt_float_to_fp8(out / ksc, __NV_SATFINITE, __NV_E4M3) : 0;
        static_cast<uint8_t*>(kv.v)[at] = vsc > 0.f ? __nv_cvt_float_to_fp8(vv / vsc, __NV_SATFINITE, __NV_E4M3) : 0;
        if (d == 0) {
          kv.kscales(cell)[size_t(cell) * nkv + head] = ksc;
          kv.vscales(cell)[size_t(cell) * nkv + head] = vsc;
        }
      } else {
        static_cast<kv16*>(kv.k)[at] = kv16_of(out);
        static_cast<kv16*>(kv.v)[at] = kv16_of(vv);
      }
      if (kv.kq) {  // the low-bit copy strata attention reads: rotated, quantized, packed
        // xk and xv reuse the 2*hd scratch floats (after the grid reduction they hold byte-packed codes_k/codes_v).
        float* xk = sh + hd + rot_dim;
        float* xv = xk + hd;
        // Warp-level FWHT butterflies (len = 1..16): bit-identical to shared-memory butterflies.
        float fk = out, fv = vv;
#pragma unroll
        for (int len = 1; len < 32; len <<= 1) {
          const float ok = __shfl_xor_sync(0xffffffffu, fk, len);
          const float ov = __shfl_xor_sync(0xffffffffu, fv, len);
          fk = (d & len) ? (ok - fk) : (fk + ok);
          fv = (d & len) ? (ov - fv) : (fv + ov);
        }
        if (hd > 32) {
          xk[d] = fk;
          xv[d] = fv;
          __syncthreads();
          const int half_hd = hd >> 1;
          for (int len = 32; len < half_hd; len <<= 1) {
            if (d < half_hd) {
              const int lo = ((d & ~(len - 1)) << 1) | (d & (len - 1));
              const float ak = xk[lo], bk = xk[lo + len];
              xk[lo] = ak + bk;
              xk[lo + len] = ak - bk;
              const float av = xv[lo], bv = xv[lo + len];
              xv[lo] = av + bv;
              xv[lo + len] = av - bv;
            }
            __syncthreads();
          }
          // Final butterfly stage lands directly in registers (no store or trailing barrier).
          const float sk = xk[d], pk = xk[d ^ half_hd];
          fk = (d & half_hd) ? (pk - sk) : (sk + pk);
          const float sv = xv[d], pv = xv[d ^ half_hd];
          fv = (d & half_hd) ? (pv - sv) : (sv + pv);
        }
        const float inv_s = rsqrtf(float(hd));
        const float xrk = fk * inv_s, xrv = fv * inv_s;
        const int top = (1 << kv.bits) - 1;
        float lok, stepk, lov, stepv;
        if (kv.grid) {  // the uniform step that serves a normal distribution best (Max, 1960), about the mean
          float ms[2] = {xrk, xrv};
          block_sum_n(ms);
          const float mk = ms[0] / float(hd), mv = ms[1] / float(hd);
          float vs[2] = {(xrk - mk) * (xrk - mk), (xrv - mv) * (xrv - mv)};
          block_sum_n(vs);
          const float c = kv.bits == 2 ? 0.9957f : kv.bits == 3 ? 0.5860f : 0.3352f;
          stepk = c * sqrtf(vs[0] / float(hd));
          stepv = c * sqrtf(vs[1] / float(hd));
          lok = mk - float((top + 1) / 2) * stepk;
          lov = mv - float((top + 1) / 2) * stepv;
        } else {
          float mm[4] = {xrk, -xrk, xrv, -xrv};
          block_max_n(mm);
          lok = -mm[1];
          stepk = (mm[0] - lok) / float(top + 1);
          lov = -mm[3];
          stepv = (mm[2] - lov) / float(top + 1);
        }
        const __half2 gridk = __floats2half2_rn(lok, stepk);  // kept in binary16: the codes are chosen on it
        const __half2 gridv = __floats2half2_rn(lov, stepv);
        lok = __low2float(gridk);
        stepk = __high2float(gridk);
        lov = __low2float(gridv);
        stepv = __high2float(gridv);
        uint8_t* codes_k = reinterpret_cast<uint8_t*>(xk);
        uint8_t* codes_v = codes_k + hd;
        codes_k[d] = uint8_t(stepk > 0.f ? min(top, max(0, int(floorf((xrk - lok) / stepk)))) : 0);
        codes_v[d] = uint8_t(stepv > 0.f ? min(top, max(0, int(floorf((xrv - lov) / stepv)))) : 0);
        __syncthreads();
        // StrataLayout direct inverse packing: warp 0 packs K words while warp 1 packs V words in O(cph) steps.
        const StrataLayout L(hd, kv.bits);
        const int sslot = slot - strata_base;
        const size_t woff = strata_words_at(sslot, head, nkv, kv.words);
        const int p16 = sslot & 15;
        const int kw = Compact ? 24 : L.kwords();
        const int vw = Compact ? 24 : L.vwords();
        const int vbase = hd >= 64 ? ((kv.words + 31) & ~31) : kv.words;
        if (d < kw) {
          uint32_t w = 0;
          if (d < 4 * L.kown) {
            const int t = d / L.kown, j0 = (d % L.kown) * L.cph;
            for (int s = 0; s < L.cph; ++s) {
              const int first = 8 * (j0 + s) + 2 * t;
              w |= uint32_t(codes_k[first]) << (s * kv.bits);
              w |= uint32_t(codes_k[first + 1]) << (16 + s * kv.bits);
            }
          } else {
            const int t0 = (d - 4 * L.kown) * L.kgpw, t_end = min(4, t0 + L.kgpw);
            for (int t = t0; t < t_end; ++t) {
              for (int r = 0; r < L.kleft; ++r) {
                const int s = (t - t0) * L.kleft + r;
                const int first = 8 * (L.kown * L.cph + r) + 2 * t;
                w |= uint32_t(codes_k[first]) << (s * kv.bits);
                w |= uint32_t(codes_k[first + 1]) << (16 + s * kv.bits);
              }
            }
          }
          if constexpr (Compact) {  // each half's spare high bit holds a bit of this MMA lane's last code pairs
            const int i = d % 6, bit = i % 3;
            const int first = 2 * (4 * (30 + i / 3) + d / 6);
            w |= uint32_t((codes_k[first] >> bit) & 1) << 15;
            w |= uint32_t((codes_k[first + 1] >> bit) & 1) << 31;
          }
          kv.kq[woff + L.k_index(p16, d)] = w;
        } else if (d < kv.words) {
          kv.kq[woff + d * 16 + p16] = 0u;
        } else if (d >= vbase && d < vbase + vw) {
          const int vd = d - vbase;
          uint32_t w = 0;
          if (vd < 8 * L.vown) {
            const int g = vd / L.vown, m0 = (vd % L.vown) * L.cph;
            for (int s = 0; s < L.cph; ++s) {
              const int first = 16 * (m0 + s) + g;
              w |= uint32_t(codes_v[first]) << (s * kv.bits);
              w |= uint32_t(codes_v[first + 8]) << (16 + s * kv.bits);
            }
          } else {
            const int g0 = (vd - 8 * L.vown) * L.vgpw, g_end = min(8, g0 + L.vgpw);
            for (int g = g0; g < g_end; ++g) {
              for (int r = 0; r < L.vleft; ++r) {
                const int s = (g - g0) * L.vleft + r;
                const int first = 16 * (L.vown * L.cph + r) + g;
                w |= uint32_t(codes_v[first]) << (s * kv.bits);
                w |= uint32_t(codes_v[first + 8]) << (16 + s * kv.bits);
              }
            }
          }
          if constexpr (Compact) {
            const int i = vd % 3, bit = i;
            const int first = 240 + vd / 3;
            w |= uint32_t((codes_v[first] >> bit) & 1) << 15;
            w |= uint32_t((codes_v[first + 8] >> bit) & 1) << 31;
          }
          kv.vq[woff + L.v_index(p16, vd)] = w;
        } else if (d >= vbase + vw && d < vbase + kv.words) {
          kv.vq[woff + (d - vbase) * 16 + p16] = 0u;
        }
        if (d == 0) {
          kv.kqs[size_t(sslot) * nkv + head] = gridk;
          kv.vqs[size_t(sslot) * nkv + head] = gridv;
        }
      }
    }
  }
}

// Partial attention over one split of the keys for up to RT rows of one head.
// grid (nh, splits, ceil(rows / RT)); block 128 (4 warps; warp w takes the keys
// j with j % 4 == w).  A row's keys are split by its own position (causal:
// position + 1 keys), never by the rows it shares the launch with, so
// verifying k rows gives each row the bits that decoding it alone gives.
// window > 0: a row at position p sees only the keys j with |p - j| < window.
// RT = 1 only for a tree round (its rows' round keys read from their ancestors' slots, tree_slot); RT = 4 has none of
// that code.  CANVAS (RT = 4): a DV canvas pass's rows, base P->start + 1 + pos_add, the keys after P->start read
// from canvas_s0 on (CanvasKeys).
template <int HD, int RT, bool CANVAS = false>
__global__ void __launch_bounds__(128) k_attention(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv,
                                                   int rows, int causal, int extra, float scale, const DevParams* P,
                                                   float* part, int pos_add, int window, int slots,
                                                   const TreeRows* tree, int canvas_s0, SplitWindow sw = {}) {
  pdl_trigger();
  constexpr int DPL = HD / 32;
  __shared__ float s_m[4][RT], s_l[4][RT];
  __shared__ float s_acc[4][RT][HD];
  const int h = blockIdx.x, split = blockIdx.y, splits = gridDim.y, r0 = blockIdx.z * RT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kvh = h / (nh / nkv);
  const int canvas_from = CANVAS ? P->start + 1 : 0;  // the first key in the canvas's slots
  const int base = (CANVAS ? canvas_from : P->position) + pos_add;
  const int nrows = min(RT, rows - r0);
  int ks[RT], ke[RT], lo = INT_MAX, hi = 0;  // row i: keys [ks, ke) of this split; the block walks [lo, hi)
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    const int p = base + (RT == 1 ? tree_depth(tree, r0 + i) : r0 + i), all = causal ? p + 1 : base + extra;
    const int first = window > 0 ? max(0, p - window + 1) : 0, total = window > 0 ? min(all, p + window) : all;
    const int chunk = (total - first + splits - 1) / splits;
    ks[i] = min(total, first + split * chunk);
    ke[i] = min(total, ks[i] + chunk);
    if (i < nrows && ks[i] < ke[i]) {
      lo = min(lo, ks[i]);
      hi = max(hi, ke[i]);
    }
  }
  const int orig_ks = ks[0], orig_ke = ke[0];
  if constexpr (RT == 1 && !CANVAS) {
    if (sw.state) {
      if (nrows <= 0) return;
      if (orig_ks >= orig_ke) {
        if (sw.first) {
          float* pr = part + ((size_t(r0) * nh + h) * slots + split) * (HD + 2);
          for (int idx = threadIdx.x; idx < HD + 2; idx += blockDim.x) pr[idx] = idx == 0 ? -INFINITY : 0.f;
        }
        return;
      }
      ks[0] = max(orig_ks, sw.k_lo);
      ke[0] = min(orig_ke, sw.k_hi);
      lo = ks[0];
      hi = ke[0];
      if (lo >= hi) return;
    }
  }

  float qv[RT][DPL], acc[RT][DPL], m[RT], l[RT];
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) {
      acc[i][e] = 0.f;
      qv[i][e] = i < nrows ? q[(size_t(r0 + i) * nh + h) * HD + lane * DPL + e] : 0.f;
    }
  }
  if constexpr (RT == 1 && !CANVAS) {
    if (sw.state && orig_ks < sw.k_lo) {
      const float* wp = sw.carry + (size_t(r0) * nh + h) * kSplitStateFloats(HD) + warp * (HD + 40);
      m[0] = wp[0];
      l[0] = wp[1];
#pragma unroll
      for (int e = 0; e < DPL; ++e) acc[0][e] = wp[2 + lane * DPL + e];
    }
  }
  for (int j = lo < hi ? lo + ((warp - lo) & 3) : hi; j < hi; j += 4) {
    float kf[DPL], vf[DPL];
    const int key = CANVAS ? (j < canvas_from ? j : canvas_s0 + (j - canvas_from))
                           : RT == 1 ? tree_slot(tree, r0, base, j) : j;
    const size_t at = (size_t(key) * nkv + kvh) * HD + lane * DPL;
#pragma unroll
    for (int e = 0; e < DPL; ++e) {
      kf[e] = __half2float(Kc[at + e]);
      vf[e] = __half2float(Vc[at + e]);
    }
#pragma unroll
    for (int i = 0; i < RT; ++i) {
      if (i >= nrows) break;
      if (j < ks[i] || j >= ke[i]) continue;  // uniform across the warp
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < DPL; ++e) d += qv[i][e] * kf[e];
      const float sc = warp_sum(d) * scale;
      const float mn = fmaxf(m[i], sc);
      const float corr = __expf(m[i] - mn), p = __expf(sc - mn);
      l[i] = l[i] * corr + p;
#pragma unroll
      for (int e = 0; e < DPL; ++e) acc[i][e] = acc[i][e] * corr + p * vf[e];
      m[i] = mn;
    }
  }
  if constexpr (RT == 1 && !CANVAS) {
    if (sw.state && orig_ke > sw.k_hi) {
      float* wp = sw.state + (size_t(r0) * nh + h) * kSplitStateFloats(HD) + warp * (HD + 40);
      if (lane == 0) {
        wp[0] = m[0];
        wp[1] = l[0];
      }
#pragma unroll
      for (int e = 0; e < DPL; ++e) wp[2 + lane * DPL + e] = acc[0][e];
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    if (lane == 0) {
      s_m[warp][i] = m[i];
      s_l[warp][i] = l[i];
    }
#pragma unroll
    for (int e = 0; e < DPL; ++e) s_acc[warp][i][lane * DPL + e] = acc[i][e];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < RT * HD; idx += blockDim.x) {
    const int i = idx / HD, d = idx - i * HD;
    if (i >= nrows) continue;
    float M = -INFINITY;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, s_m[w][i]);
    float L = 0.f, A = 0.f;
    if (M != -INFINITY)
      for (int w = 0; w < 4; ++w) {
        const float f = __expf(s_m[w][i] - M);
        L += s_l[w][i] * f;
        A += s_acc[w][i][d] * f;
      }
    float* pr = part + ((size_t(r0 + i) * nh + h) * slots + split) * (HD + 2);
    if (d == 0) {
      pr[0] = M;
      pr[1] = L;
    }
    pr[2 + d] = A;
  }
}

// Split attention for decode and verification, by KV head: a block takes the
// G query heads that share one KV head (G = 4 for Qwen3.5-4B) and RT rows, so
// every key and value is read once for all of them.  For the scores a lane
// owns whole keys (the dot products need no shuffles); for the values a lane
// owns HD/32 dimensions.  Keys go to warps by 32-key tile (tile t to warp
// t % 4) and a row's keys are split by its own position: a tile with none of
// a row's keys leaves that row's state exactly as it was, so verifying k rows
// gives each row the bits that decoding it alone gives.
// grid (ceil(rows / RT), nkv, splits): the blocks of a round's rows for one
// split run next to each other and share its keys and values through L2 (with
// the rows last, a round deep in a long prompt read the whole cache from DRAM
// again for every few rows).  block 128.  Same partials as k_attention.
// A tree round's row (RT == 1, tree != nullptr) reads its round keys through s_path; RT == 2 has none of that code.
// CANVAS (RT = 1, no tree): a DV canvas pass's row, base P->start + 1 + pos_add, the keys after P->start read from
// canvas_s0 on (CanvasKeys).
template <int HD, int G, int RT, bool FP8, bool CANVAS = false>
__global__ void __launch_bounds__(128) k_attention_gqa(const float* __restrict__ q, KvRef kv, int nh, int nkv, int rows,
                                                       int causal, int extra, float scale, const DevParams* P,
                                                       float* __restrict__ part, int pos_add, int below,
                                                       int slots, const TreeRows* __restrict__ tree, int canvas_s0,
                                                       SplitWindow sw = {}) {
  pdl_trigger();
  constexpr int QN = G * RT, DPL = HD / 32, CH = HD / 8;
  __shared__ __align__(16) float Qs[QN][HD];
  __shared__ float s_m[4][QN], s_l[4][QN];
  __shared__ float s_acc[4][QN][HD];
  const int kvh = blockIdx.y, split = blockIdx.z, splits = gridDim.z, r0 = blockIdx.x * RT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int canvas_from = CANVAS ? P->start + 1 : 0;  // the first key in the canvas's slots
  const int base = (CANVAS ? canvas_from : P->position) + pos_add;
  const int nrows = min(RT, rows - r0);
  static_assert(!CANVAS || RT == 1, "a canvas's rows go one a block");
  // A tree's row (RT == 1): its path's slots in shared memory (filled with Qs below), so a key's slot is no global load.
  __shared__ unsigned char s_path[RT == 1 ? FLYX_MAX_WIDE_ROWS : 1];
  auto slot = [&](int j) {  // where key j of the block's row is
    if constexpr (CANVAS) return j < canvas_from ? j : canvas_s0 + (j - canvas_from);
    else if constexpr (RT == 1) return (tree && j >= base) ? base + int(s_path[j - base]) : j;
    else return j;
  };
  int ks[RT], ke[RT], lo = INT_MAX, hi = 0;  // row i: keys [ks, ke) of this split; the block walks [lo, hi)
  bool own[RT];  // rows of fewer than `below` keys (the others are attention_rows_mma's)
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    // A row uses as many splits as its keys fill (short contexts do not pay for 32 splits), on 32-key tile
    // boundaries (each warp one tile per 128 keys).
    const int total = causal ? base + (RT == 1 && tree ? tree_depth(tree, r0 + i) : r0 + i) + 1 : base + extra;
    own[i] = i < nrows && total < below;
    attn_split_range(total, split, splits, &ks[i], &ke[i]);
    if (!own[i]) ks[i] = ke[i] = 0;
    if (own[i] && ks[i] < ke[i]) {
      lo = min(lo, ks[i]);
      hi = max(hi, ke[i]);
    }
  }
  const int orig_ks = ks[0], orig_ke = ke[0];
  if constexpr (RT == 1 && !CANVAS) {
    if (sw.state) {
      if (!own[0]) return;
      if (orig_ks >= orig_ke) {
        if (sw.first) {
          for (int idx = threadIdx.x; idx < QN * (HD + 2); idx += blockDim.x) {
            const int qi = idx / (HD + 2), e = idx - qi * (HD + 2);
            part[((size_t(r0) * nh + kvh * G + qi) * slots + split) * (HD + 2) + e] = e == 0 ? -INFINITY : 0.f;
          }
        }
        return;
      }
      ks[0] = max(orig_ks, sw.k_lo);
      ke[0] = min(orig_ke, sw.k_hi);
      lo = ks[0];
      hi = ke[0];
      if (lo >= hi) return;
    }
  }
  if (lo >= hi) {  // no keys for any of the block's rows in this split: empty partials
    for (int idx = threadIdx.x; idx < QN * (HD + 2); idx += blockDim.x) {
      const int qi = idx / (HD + 2), e = idx - qi * (HD + 2), i = qi / G, g = qi - i * G;
      if (!own[i]) continue;
      part[((size_t(r0 + i) * nh + kvh * G + g) * slots + split) * (HD + 2) + e] = e == 0 ? -INFINITY : 0.f;
    }
    return;
  }
  {  // the queries, 4 floats a load and every load before the first store (one at a time: 12 round trips, 2 us)
    constexpr int Q4 = QN * HD / 4, PER = (Q4 + 127) / 128;
    float4 qv[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int idx = int(threadIdx.x) + 128 * u, qi = 4 * idx / HD, d = 4 * idx - qi * HD, i = qi / G, g = qi - i * G;
      qv[u] = idx < Q4 && i < nrows
                  ? *reinterpret_cast<const float4*>(q + (size_t(r0 + i) * nh + kvh * G + g) * HD + d)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int idx = int(threadIdx.x) + 128 * u, qi = 4 * idx / HD, d = 4 * idx - qi * HD;
      if (idx < Q4) *reinterpret_cast<float4*>(&Qs[qi][d]) = qv[u];
    }
  }
  static_assert(FLYX_MAX_WIDE_ROWS <= 128, "a thread of the block a path entry");
  if constexpr (RT == 1)
    if (tree && threadIdx.x < FLYX_MAX_WIDE_ROWS) s_path[threadIdx.x] = tree->path[r0][threadIdx.x];
  __syncthreads();

  float acc[QN][DPL], m[QN], lp[QN];
#pragma unroll
  for (int qi = 0; qi < QN; ++qi) {
    m[qi] = -INFINITY;
    lp[qi] = 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) acc[qi][e] = 0.f;
  }
  if constexpr (RT == 1 && !CANVAS) {
    if (sw.state && orig_ks < sw.k_lo) {
#pragma unroll
      for (int qi = 0; qi < QN; ++qi) {
        const float* wp = sw.carry + (size_t(r0) * nh + kvh * G + qi) * kSplitStateFloats(HD) + warp * (HD + 40);
        m[qi] = wp[0];
        lp[qi] = wp[1 + lane];
#pragma unroll
        for (int e = 0; e < DPL; ++e) acc[qi][e] = wp[33 + lane * DPL + e];
      }
    }
  }
  if (lo < hi) {
    const int t0 = lo >> 5, t1 = (hi - 1) >> 5;
    // A lane's value slice of one key: VW 32-bit words; VB keys per batch.
    // Every load of a key row (scores) or of a batch of value slices is issued
    // before the first use, so a tile waits three round trips to DRAM (the
    // key row, two batches of values) where loads issued 8 at a time waited
    // eight; the sums run in the same order.  (32 keys per batch: slower, the
    // compiler no longer keeps them all in flight.)
    constexpr int VBYTES = DPL * (FP8 ? 1 : 2), VW = VBYTES >= 4 ? VBYTES / 4 : 1, VB = 16;
    for (int t = t0 + ((warp - t0) & 3); t <= t1; t += 4) {
      const int j = (t << 5) + lane;
      const int kn = min(32, hi - (t << 5));  // keys of the tile below hi (the same for the whole warp)
      // The first batch of value slices loads while the softmax runs.
      uint32_t vw[VB][VW];
      float vsc[FP8 ? VB : 1];
      auto load_values = [&](int k0) {
#pragma unroll
        for (int b = 0; b < VB; ++b) {
          const int kk = min(k0 + b, kn - 1);  // past kn: a valid row whose p is never used
          const int key = slot((t << 5) + kk);
          const size_t row = (size_t(key) * nkv + kvh) * HD + lane * DPL;
          const void* src = FP8 ? static_cast<const void*>(static_cast<const uint8_t*>(kv.v) + row)
                                : static_cast<const void*>(static_cast<const kv16*>(kv.v) + row);
          if constexpr (VBYTES == 16) {
            const uint4 x = __ldg(reinterpret_cast<const uint4*>(src));
            vw[b][0] = x.x, vw[b][1] = x.y, vw[b][2] = x.z, vw[b][3] = x.w;
          } else if constexpr (VBYTES == 8) {
            const uint2 x = __ldg(reinterpret_cast<const uint2*>(src));
            vw[b][0] = x.x, vw[b][1] = x.y;
          } else if constexpr (VBYTES == 4) {
            vw[b][0] = __ldg(reinterpret_cast<const unsigned int*>(src));
          } else if constexpr (VBYTES == 2) {
            vw[b][0] = __ldg(reinterpret_cast<const unsigned short*>(src));
          } else {
            vw[b][0] = __ldg(reinterpret_cast<const unsigned char*>(src));
          }
          if constexpr (FP8) vsc[b] = __ldg(kv.vscales(key) + size_t(key) * nkv + kvh);
        }
      };
      float sc[QN], ksj = 1.f;
#pragma unroll
      for (int qi = 0; qi < QN; ++qi) sc[qi] = 0.f;
      const int jslot = j < hi ? slot(j) : j;
      if (FP8 && j < hi) {  // the codes' dot product, times the key's scale below
        const uint4* kr = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(kv.k) + (size_t(jslot) * nkv + kvh) * HD);
        ksj = __ldg(kv.kscales(jslot) + size_t(jslot) * nkv + kvh);
        uint4 kq[HD / 16];
#pragma unroll
        for (int c = 0; c < HD / 16; ++c) kq[c] = __ldg(kr + c);
#pragma unroll
        for (int c = 0; c < HD / 16; ++c) {
          float kf[16];
          e4m3x4_to_float(kq[c].x, kf);
          e4m3x4_to_float(kq[c].y, kf + 4);
          e4m3x4_to_float(kq[c].z, kf + 8);
          e4m3x4_to_float(kq[c].w, kf + 12);
#pragma unroll
          for (int qi = 0; qi < QN; ++qi) {
            float s = sc[qi];
#pragma unroll
            for (int h4 = 0; h4 < 4; ++h4) {
              const float4 a = *reinterpret_cast<const float4*>(&Qs[qi][16 * c + 4 * h4]);
              s = __fmaf_rn(a.x, kf[4 * h4], s);
              s = __fmaf_rn(a.y, kf[4 * h4 + 1], s);
              s = __fmaf_rn(a.z, kf[4 * h4 + 2], s);
              s = __fmaf_rn(a.w, kf[4 * h4 + 3], s);
            }
            sc[qi] = s;
          }
        }
      } else if (!FP8 && j < hi) {
        const uint4* kr = reinterpret_cast<const uint4*>(static_cast<const kv16*>(kv.k) + (size_t(jslot) * nkv + kvh) * HD);
        uint4 kq[CH];
#pragma unroll
        for (int c = 0; c < CH; ++c) kq[c] = __ldg(kr + c);
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          const uint32_t w[4] = {kq[c].x, kq[c].y, kq[c].z, kq[c].w};
          float kf[8];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[e]));
            kf[2 * e] = f.x;
            kf[2 * e + 1] = f.y;
          }
#pragma unroll
          for (int qi = 0; qi < QN; ++qi) {
            const float4 a = *reinterpret_cast<const float4*>(&Qs[qi][8 * c]);
            const float4 b = *reinterpret_cast<const float4*>(&Qs[qi][8 * c + 4]);
            float s = sc[qi];
            s = __fmaf_rn(a.x, kf[0], s);
            s = __fmaf_rn(a.y, kf[1], s);
            s = __fmaf_rn(a.z, kf[2], s);
            s = __fmaf_rn(a.w, kf[3], s);
            s = __fmaf_rn(b.x, kf[4], s);
            s = __fmaf_rn(b.y, kf[5], s);
            s = __fmaf_rn(b.z, kf[6], s);
            s = __fmaf_rn(b.w, kf[7], s);
            sc[qi] = s;
          }
        }
      }
      load_values(0);
      float pj[QN];
#pragma unroll
      for (int qi = 0; qi < QN; ++qi) {
        const int i = qi / G;
        const bool ok = i < nrows && j >= ks[i] && j < ke[i];
        const float v = ok ? __fmul_rn(FP8 ? __fmul_rn(sc[qi], ksj) : sc[qi], scale) : -INFINITY;
        float mt = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
        const float mn = fmaxf(m[qi], mt);
        const float corr = mn == m[qi] ? 1.f : __expf(m[qi] - mn);  // exactly 1 when nothing changed
        pj[qi] = ok ? __expf(v - mn) : 0.f;
        lp[qi] = __fmaf_rn(lp[qi], corr, pj[qi]);
#pragma unroll
        for (int e = 0; e < DPL; ++e) acc[qi][e] = __fmul_rn(acc[qi][e], corr);
        m[qi] = mn;
      }
      for (int k0 = 0; k0 < kn; k0 += VB) {
        if (k0 > 0) load_values(k0);
#pragma unroll
        for (int b = 0; b < VB; ++b) {
          if (k0 + b >= kn) break;  // uniform across the warp
          float vf[DPL];
          if constexpr (FP8 && VBYTES >= 4) {
#pragma unroll
            for (int w = 0; w < VW; ++w) e4m3x4_to_float(vw[b][w], vf + 4 * w);
          } else if constexpr (FP8) {
#pragma unroll
            for (int e = 0; e < DPL; ++e) vf[e] = e4m3_to_float(uint8_t(vw[b][0] >> (8 * e)));
          } else if constexpr (VBYTES >= 4) {
#pragma unroll
            for (int w = 0; w < VW; ++w) {
              const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&vw[b][w]));
              vf[2 * w] = f.x;
              vf[2 * w + 1] = f.y;
            }
          } else {
            vf[0] = __half2float(__ushort_as_half(static_cast<unsigned short>(vw[b][0])));  // one binary16
          }
#pragma unroll
          for (int qi = 0; qi < QN; ++qi) {
            const float p = __shfl_sync(0xffffffffu, pj[qi], k0 + b);
            const float pv = FP8 ? __fmul_rn(p, vsc[b]) : p;  // the value's scale folded into p
#pragma unroll
            for (int e = 0; e < DPL; ++e) acc[qi][e] = __fmaf_rn(pv, vf[e], acc[qi][e]);
          }
        }
      }
    }
  }
  if constexpr (RT == 1 && !CANVAS) {
    if (sw.state && orig_ke > sw.k_hi) {
#pragma unroll
      for (int qi = 0; qi < QN; ++qi) {
        float* wp = sw.state + (size_t(r0) * nh + kvh * G + qi) * kSplitStateFloats(HD) + warp * (HD + 40);
        if (lane == 0) wp[0] = m[qi];
        wp[1 + lane] = lp[qi];
#pragma unroll
        for (int e = 0; e < DPL; ++e) wp[33 + lane * DPL + e] = acc[qi][e];
      }
      return;
    }
  }
  // per-lane partial sums of p -> the warp's (fixed tree), then the four warps in order
#pragma unroll
  for (int qi = 0; qi < QN; ++qi) {
    float l = lp[qi];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if (lane == 0) {
      s_m[warp][qi] = m[qi];
      s_l[warp][qi] = l;
    }
#pragma unroll
    for (int e = 0; e < DPL; ++e) s_acc[warp][qi][lane * DPL + e] = acc[qi][e];
  }
  __syncthreads();
  // The four warps' factors once a query (each value's loop below took them again: 2 us at short contexts), then
  // every value with the same products in the same order.
  __shared__ float s_f[4][QN], s_M[QN], s_L[QN];
  if (threadIdx.x < QN) {
    const int qi = threadIdx.x;
    float M = -INFINITY;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, s_m[w][qi]);
    float L = 0.f;
    for (int w = 0; w < 4; ++w) {
      float f = 0.f;
      if (M != -INFINITY) {
        f = __expf(s_m[w][qi] - M);
        L += s_l[w][qi] * f;
      }
      s_f[w][qi] = f;
    }
    s_M[qi] = M;
    s_L[qi] = L;
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < QN * HD; idx += blockDim.x) {
    const int qi = idx / HD, d = idx - qi * HD, i = qi / G, g = qi - i * G;
    if (!own[i]) continue;
    const float M = s_M[qi];
    float A = 0.f;
    if (M != -INFINITY)
      for (int w = 0; w < 4; ++w) {
        const float f = s_f[w][qi];
        A += s_acc[w][qi][d] * f;
      }
    float* pr = part + ((size_t(r0 + i) * nh + kvh * G + g) * slots + split) * (HD + 2);
    if (d == 0) {
      pr[0] = M;
      pr[1] = s_L[qi];
    }
    pr[2 + d] = A;
  }
}

// grid (nh, rows); block HD.
// The slots [0, used) a row's kernel filled: from >= 0: the grouped kernels (attention_rows_mma for rows of `from` keys
// or more, up to every slot; the float32 one up to kAttnCoresSplits), as their keys fill them; from < 0: the per-head
// kernel, every one of its grid's.  The slots past are not read (they may hold an earlier round's).
template <bool TREE>
__global__ void k_attention_combine(const float* part, int nh, int hd, int splits, const float* gate, int ld_gate,
                                    Out out, int ld_out, const DevParams* P, int causal, int extra, int pos_add,
                                    int from, const TreeRows* tree) {
  pdl_trigger();
  constexpr int kFastSplits = 128;
  __shared__ float s_m[kFastSplits], s_l[kFastSplits], s_f[kFastSplits];
  __shared__ float s_M, s_L;
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const float* pr = part + (size_t(r) * nh + h) * splits * (hd + 2);
  const float g = gate ? gate[size_t(r) * ld_gate + h * hd + d] : 0.f;  // loads with the partials
  const int cores = min(splits, kAttnCoresSplits);
  int used = cores;
  if (from >= 0) {
    const int at = TREE ? tree_depth(tree, r) : r;  // a tree's row: its depth's keys
    const int total = causal ? P->position + pos_add + at + 1 : P->position + pos_add + extra;
    used = attn_split_used(total, total >= from ? splits : cores);
  }
  float L = 0.f, A = 0.f;
  if (used > 16 && used <= kFastSplits && blockDim.x >= 32) {
    // Load every split's (m, l) in parallel across the block, then compute M and exp factors once in warp 0.
    for (int s = d; s < used; s += blockDim.x) {
      s_m[s] = pr[s * (hd + 2)];
      s_l[s] = pr[s * (hd + 2) + 1];
    }
    __syncthreads();
    if (d < 32) {
      float M = -INFINITY;
      for (int s = d; s < used; s += 32) M = fmaxf(M, s_m[s]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
      if (d == 0) s_M = M;
      if (M != -INFINITY) {
        for (int s = d; s < used; s += 32) s_f[s] = s_m[s] == -INFINITY ? 0.f : __expf(s_m[s] - M);
        __syncwarp();
        if (d == 0) {
          float ls = 0.f;
          for (int s = 0; s < used; ++s)
            if (s_m[s] != -INFINITY) ls += s_l[s] * s_f[s];
          s_L = ls;
        }
      }
    }
    __syncthreads();
    const float M = s_M;
    if (M != -INFINITY) {
      L = s_L;
#pragma unroll 4
      for (int s = 0; s < used; ++s) {
        if (s_m[s] != -INFINITY) A += pr[s * (hd + 2) + 2 + d] * s_f[s];
      }
    }
  } else {
    float M = -INFINITY;
    for (int s = 0; s < used; ++s) M = fmaxf(M, pr[s * (hd + 2)]);
    if (M != -INFINITY)
      for (int s = 0; s < used; ++s) {
        if (pr[s * (hd + 2)] == -INFINITY) continue;  // an empty split adds zeros: not read
        const float f = __expf(pr[s * (hd + 2)] - M);
        L += pr[s * (hd + 2) + 1] * f;
        A += pr[s * (hd + 2) + 2 + d] * f;
      }
  }
  float o = L > 0.f ? A / L : 0.f;
  if (gate) o *= sigm(g);
  put(out, size_t(r) * ld_out + h * hd + d, size_t(r) * nh * hd + h * hd + d, o);
}

}  // namespace

void attn_prep(const float* qkv, int ld, int q_off, int k_off, int v_off, const float* q_norm, const float* k_norm,
               float norm_offset, float eps, double theta, int rot_dim, int nh, int nkv, int hd, int rows,
               const DevParams* P, int pos_sel, bool skip_if_noop, float* q_out, KvRef kv, cudaStream_t st,
               int pos_add, const TreeRows* tree, int strata_base, CanvasKeys canvas, MRoPE mrope) {
  // Heads per block: enough blocks to fill the GPU (about 512), else fewer rotations.  Every
  // grouping gives the same bits.  A 5760-token prefill: one head per block 11.5 ms of
  // attn_prep, all 20 of a row 2.9 ms (bench/trace).
  const int H = nh + nkv, hpb = std::clamp(H * rows / 512, 1, H);
  dim3 grid((H + hpb - 1) / hpb, rows);
  const size_t shared = (hd + rot_dim + (kv.kq ? 2 * hd : 0)) * sizeof(float);
  const bool use_mrope = mrope.active();
  if (canvas.s0 >= 0) {
    if (tree || kv.kq) {  // DV's trees read every position exactly: no tree rows here, no low-bit copy to write
      std::fprintf(stderr, "flyx-cuda: attn_prep of a canvas pass with a tree or a low-bit copy\n");
      std::abort();
    }
    if (use_mrope)
      k_attn_prep<false, true, true><<<grid, hd, shared, st>>>(qkv, ld, q_off, k_off, v_off, q_norm, k_norm, norm_offset,
                                                               eps, theta, rot_dim, nh, nkv, hd, P, pos_sel, skip_if_noop,
                                                               q_out, kv, pos_add, hpb, nullptr, rope_table(theta, rot_dim),
                                                               0, canvas.s0, mrope);
    else
      k_attn_prep<false, true, false><<<grid, hd, shared, st>>>(qkv, ld, q_off, k_off, v_off, q_norm, k_norm, norm_offset,
                                                                eps, theta, rot_dim, nh, nkv, hd, P, pos_sel, skip_if_noop,
                                                                q_out, kv, pos_add, hpb, nullptr, rope_table(theta, rot_dim),
                                                                0, canvas.s0);
    return;
  }
  auto run = [&](auto compact_tag, auto mrope_tag) {
    k_attn_prep<decltype(compact_tag)::value, false, decltype(mrope_tag)::value><<<grid, hd, shared, st>>>(
        qkv, ld, q_off, k_off, v_off, q_norm, k_norm, norm_offset, eps, theta, rot_dim, nh, nkv, hd, P, pos_sel,
        skip_if_noop, q_out, kv, pos_add, hpb, tree, rope_table(theta, rot_dim), strata_base, -1, mrope);
  };
  const bool compact = hd == 256 && kv.bits == 3 && kv.words == 24;
  if (compact && use_mrope) run(std::true_type{}, std::true_type{});
  else if (compact) run(std::true_type{}, std::false_type{});
  else if (use_mrope) run(std::false_type{}, std::true_type{});
  else run(std::false_type{}, std::false_type{});
}

namespace {

__global__ void k_kv_window(KvRef src, KvRef dst, int nkv, size_t row16, int window, const DevParams* P,
                            DevParams* P_out, int pos_add) {
  pdl_trigger();
  // positions [0, head) and [from, q]: the first sink apart when the window starts past them
  const int q = P->position + pos_add, first = max(0, q - window + 1);
  const int head = first > src.sink ? src.sink : 0, from = first > src.sink ? first : 0;
  const int n = max(0, head + q - from + 1);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    DevParams p = *P;
    p.position = n - 1 - pos_add;  // the attention's row then sees keys [0, n)
    *P_out = p;
  }
  auto cell = [&](int r) { return src.slot_of(r < head ? r : from + (r - head)); };
  const size_t stride = size_t(gridDim.x) * blockDim.x, at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (size_t i = at; i < size_t(n) * row16; i += stride) {
    const size_t from = size_t(cell(int(i / row16))) * row16 + i % row16;
    static_cast<uint4*>(dst.k)[i] = static_cast<const uint4*>(src.k)[from];
    static_cast<uint4*>(dst.v)[i] = static_cast<const uint4*>(src.v)[from];
  }
  if (src.ks)
    for (size_t i = at; i < size_t(n) * nkv; i += stride) {
      const size_t from = size_t(cell(int(i / nkv))) * nkv + i % nkv;
      dst.ks[i] = src.ks[from];
      dst.vs[i] = src.vs[from];
    }
}

}  // namespace

namespace {
__global__ void k_params_at(const DevParams* P, DevParams* P_out, int pos_add) {
  pdl_trigger();
  DevParams p = *P;
  p.position += pos_add;
  *P_out = p;
}
}  // namespace

void params_at(const DevParams* P, DevParams* P_out, int pos_add, cudaStream_t st) {
  k_params_at<<<1, 1, 0, st>>>(P, P_out, pos_add);
}

void kv_window(const KvRef& src, const KvRef& dst, int nkv, int hd, int window, const DevParams* P, DevParams* P_out,
               int pos_add, cudaStream_t st) {
  const size_t row16 = size_t(nkv) * hd * (src.fp8() ? 1 : sizeof(kv16)) / 16;
  k_kv_window<<<128, 256, 0, st>>>(src, dst, nkv, row16, window, P, P_out, pos_add);
}

void attn_prep(const float* qkv, int ld, int q_off, int k_off, int v_off, const float* q_norm, const float* k_norm,
               float norm_offset, float eps, double theta, int rot_dim, int nh, int nkv, int hd, int rows,
               const DevParams* P, int pos_sel, bool skip_if_noop, float* q_out, kv16* Kc, kv16* Vc,
               cudaStream_t st, int pos_add) {
  attn_prep(qkv, ld, q_off, k_off, v_off, q_norm, k_norm, norm_offset, eps, theta, rot_dim, nh, nkv, hd, rows, P,
            pos_sel, skip_if_noop, q_out, KvRef{Kc, Vc, nullptr, nullptr}, st, pos_add);
}

// ---- the rotary factors (rope_table) ---------------------------------------------------------------------
// A pair (theta, rot_dim)'s factors in a slot of a table in device memory, made by the kernels' own rope_inv: a
// float64 pow takes about 3 us of a launch on this GPU (k_attn_prep at one row 5.5 -> 2.5 us a layer).
constexpr int kRopeSlots = 32, kRopePairs = 128;  // 32 KB; a model registers one or two pairs, the tests more
__device__ double g_rope[kRopeSlots][kRopePairs];
namespace {
struct RopeSlot {
  int dev = 0;
  double theta = 0;
  int rot = 0;
  const double* ptr = nullptr;
};
std::mutex g_rope_mu;
RopeSlot g_rope_slots[kRopeSlots];
std::atomic<int> g_rope_used{0};
__global__ void k_rope_fill(double* out, double theta, int rot_dim) {
  for (int i = threadIdx.x; i < rot_dim / 2; i += blockDim.x) out[i] = rope_inv(i, rot_dim, theta);
}
}  // namespace

void rope_register(double theta, int rot_dim) {
  if (rot_dim <= 0 || rot_dim % 2 || rot_dim / 2 > kRopePairs) return;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return;
  std::lock_guard<std::mutex> lock(g_rope_mu);
  const int used = g_rope_used.load(std::memory_order_relaxed);
  for (int i = 0; i < used; ++i)
    if (g_rope_slots[i].dev == dev && g_rope_slots[i].theta == theta && g_rope_slots[i].rot == rot_dim) return;
  if (used == kRopeSlots) return;  // the kernels take the pow themselves
  double* base = nullptr;
  if (cudaGetSymbolAddress(reinterpret_cast<void**>(&base), g_rope) != cudaSuccess) return;
  double* slot_ptr = base + size_t(used) * kRopePairs;
  k_rope_fill<<<1, kRopePairs>>>(slot_ptr, theta, rot_dim);
  if (cudaDeviceSynchronize() != cudaSuccess) return;
  g_rope_slots[used] = {dev, theta, rot_dim, slot_ptr};
  g_rope_used.store(used + 1, std::memory_order_release);
}

const double* rope_table(double theta, int rot_dim) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
  const int used = g_rope_used.load(std::memory_order_acquire);
  for (int i = 0; i < used; ++i)
    if (g_rope_slots[i].dev == dev && g_rope_slots[i].theta == theta && g_rope_slots[i].rot == rot_dim)
      return g_rope_slots[i].ptr;
  return nullptr;
}

size_t attention_partial_floats(int rows, int nh, int hd, int splits) {
  return size_t(rows) * nh * splits * (hd + 2);
}

// Every attention kernel (split, grouped, flash; binary16 or FP8 caches) has an instance for each of these and no
// other: a block with another head dim is refused at load (regions/plan.cpp), never run on the wrong strides.
bool attention_head_dim(int hd) { return hd == 32 || hd == 64 || hd == 128 || hd == 256; }

void attention(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int rows, bool causal,
               int extra, float scale, const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out,
               float* partial, int splits, const Launch& L, int pos_add, int window) {
  attention(q, KvRef{const_cast<kv16*>(Kc), const_cast<kv16*>(Vc), nullptr, nullptr}, nh, nkv, hd, rows, causal, extra,
            scale, P, gate, ld_gate, out, ld_out, partial, splits, L, pos_add, window);
}

void attention(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra, float scale,
               const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, float* partial, int splits,
               const Launch& L, int pos_add, int window, const TreeRows* tree, CanvasKeys canvas) {
  // Prefill (more causal rows than a round has): tensor cores, unless the options ask for the float32 kernel.  A
  // canvas's rows read their keys through its map, which only the float32 kernels have.
  if (window == 0 && causal && rows > FLYX_MAX_ROWS && !L.round_rows && canvas.s0 < 0 &&
      opts(L).attention != kAttnSplit &&
      attention_flash(q, kv, nh, nkv, hd, rows, scale, P, gate, ld_gate, out, ld_out, L, pos_add))
    return;
  attention_split(q, kv, nh, nkv, hd, rows, causal, extra, scale, P, gate, ld_gate, out, ld_out, partial, splits, L,
                  pos_add, window, tree, canvas);
}

template <int HD, int G, bool FP8>
void launch_gqa(const float* q, KvRef kv, int nh, int nkv, int rows, bool causal, int extra, float scale,
                const DevParams* P, float* partial, int splits, cudaStream_t st, int pos_add, int below,
                const TreeRows* tree, CanvasKeys canvas, const SplitWindow& sw) {
  // Two rows a block where their queries fit the 48 KB of static shared memory (Qs and the four warps' s_acc: five
  // floats a query and dim); else one (the 27B's 6 query heads of 256 dims a KV head), and one for a tree (rows of
  // other branches read other keys of the round), a canvas, or a split window.  A row's result does not depend on RT.
  constexpr bool kTwoRows = size_t(5) * 2 * G * HD * sizeof(float) <= 48 * 1024;
  if (canvas.s0 >= 0) {
    k_attention_gqa<HD, G, 1, FP8, true><<<dim3(rows, nkv, std::min(splits, kAttnCoresSplits)), 128, 0, st>>>(
        q, kv, nh, nkv, rows, causal, extra, scale, P, partial, pos_add, below, splits, nullptr, canvas.s0);
  } else if (tree || sw.state || rows == 1 || !kTwoRows) {
    k_attention_gqa<HD, G, 1, FP8><<<dim3(rows, nkv, std::min(splits, kAttnCoresSplits)), 128, 0, st>>>(
        q, kv, nh, nkv, rows, causal, extra, scale, P, partial, pos_add, below, splits, tree, -1, sw);
  } else if constexpr (kTwoRows) {
    k_attention_gqa<HD, G, 2, FP8><<<dim3(ceil_div(rows, 2), nkv, std::min(splits, kAttnCoresSplits)), 128, 0, st>>>(
        q, kv, nh, nkv, rows, causal, extra, scale, P, partial, pos_add, below, splits, nullptr, -1);
  }
}

namespace {
// The query heads a KV head has that the grouped kernel is compiled for: 1, 2 and 4 at every head dim; 6 (Qwen3.5's
// 27B: 24 over 4) and 8 (Qwen3's 32B: 64 over 8) at the head dims of real models.
bool gqa_compiled(int hd, int g) {
  if (g == 1 || g == 2 || g == 4) return attention_head_dim(hd);
  return (g == 6 || g == 8) && (hd == 128 || hd == 256);
}
}  // namespace

bool kv_fp8_supported(const KernelOptions& opt, int nh, int nkv, int hd) {
  if (opt.attention == kAttnPerHead || nkv <= 0 || nh % nkv != 0) return false;
  return gqa_compiled(hd, nh / nkv);
}

namespace {
// The KV-head-grouped kernel for the G = nh / nkv and head dims gqa_compiled names.
bool attention_gqa(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra, float scale,
                   const DevParams* P, float* partial, int splits, cudaStream_t st, int pos_add,
                   const KernelOptions& opt, int below, const TreeRows* tree, CanvasKeys canvas,
                   const SplitWindow& sw) {
  if (!kv_fp8_supported(opt, nh, nkv, hd)) return false;  // the shapes this kernel takes, FP8 or not
  const int g = nh / nkv;
#define FLYX_GQA(HD_, G_)                                                                                     \
  if (hd == HD_ && g == G_) {                                                                                 \
    if (kv.fp8())                                                                                             \
      launch_gqa<HD_, G_, true>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, st, pos_add,  \
                                below, tree, canvas, sw);                                                     \
    else                                                                                                      \
      launch_gqa<HD_, G_, false>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, st, pos_add, \
                                 below, tree, canvas, sw);                                                    \
    return true;                                                                                              \
  }
  FLYX_GQA(32, 1) FLYX_GQA(32, 2) FLYX_GQA(32, 4)
  FLYX_GQA(64, 1) FLYX_GQA(64, 2) FLYX_GQA(64, 4)
  FLYX_GQA(128, 1) FLYX_GQA(128, 2) FLYX_GQA(128, 4) FLYX_GQA(128, 6) FLYX_GQA(128, 8)
  FLYX_GQA(256, 1) FLYX_GQA(256, 2) FLYX_GQA(256, 4) FLYX_GQA(256, 6) FLYX_GQA(256, 8)
#undef FLYX_GQA
  return false;
}
}  // namespace

void attention_split(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int rows, bool causal,
                     int extra, float scale, const DevParams* P, const float* gate, int ld_gate, float* out,
                     int ld_out, float* partial, int splits, const Launch& L, int pos_add, int window) {
  attention_split(q, KvRef{const_cast<kv16*>(Kc), const_cast<kv16*>(Vc), nullptr, nullptr}, nh, nkv, hd, rows, causal,
                  extra, scale, P, gate, ld_gate, out, ld_out, partial, splits, L, pos_add, window);
}

void attention_split(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra, float scale,
                     const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, float* partial,
                     int splits, const Launch& L, int pos_add, int window, const TreeRows* tree, CanvasKeys canvas,
                     const SplitWindow& sw) {
  const cudaStream_t st = L.st;
  const bool on_canvas = canvas.s0 >= 0;
  if (on_canvas && (tree || window > 0)) {
    std::fprintf(stderr, "flyx-cuda: attention of a canvas pass with a tree or a window\n");
    std::abort();
  }
  const int rt = (tree || sw.state) ? 1 : 4;  // the per-head kernel's rows a block
  dim3 grid(nh, std::min(splits, kAttnCoresSplits), ceil_div(rows, rt));
  const kv16* Kc = static_cast<const kv16*>(kv.k);
  const kv16* Vc = static_cast<const kv16*>(kv.v);
  // A window (a drafter's, binary16 cache) takes the per-head kernel: its rows' keys start anywhere.  A row of `from`
  // keys or more takes the tensor-core kernel (binary16 and FP8 caches), the others the float32 ones: by its own
  // position, so decoding it alone and verifying it with others take the same kernel.  Both launch; each skips the
  // other's rows.  FP8 caches take only the shapes the grouped kernel has (kv_fp8_supported).
  // A canvas's rows take the float32 kernels at any length (their keys through its map; determinism, not class R).
  int from = INT_MAX;
  if (window == 0 && !on_canvas && opts(L).rows_mma && opts(L).attention != kAttnPerHead) {
    const int want = nkv > 0 && nh % nkv == 0 && gqa_compiled(hd, nh / nkv) ? opts(L).rows_mma_from : 0;
    if (attention_rows_mma(q, kv, nh, nkv, hd, rows, causal, extra, scale, P, partial, splits, L, pos_add, want,
                           tree, sw))
      from = want;
  }
  if (from > 0 && (window > 0 || !attention_gqa(q, kv, nh, nkv, hd, rows, causal, extra, scale, P, partial, splits,
                                                st, pos_add, opts(L), from, tree, canvas, sw))) {
    if (kv.fp8())  // the backend refuses FP8 caches of other shapes at load
      throw std::runtime_error("flyx-cuda: FP8 KV cache with an unsupported attention shape");
    auto per_head = [&](auto kernel1, auto kernel4, auto kernel_canvas) {
      if (on_canvas) kernel_canvas<<<grid, 128, 0, st>>>(q, Kc, Vc, nh, nkv, rows, causal, extra, scale, P, partial,
                                                         pos_add, window, splits, nullptr, canvas.s0, SplitWindow{});
      else if (tree || sw.state) kernel1<<<grid, 128, 0, st>>>(q, Kc, Vc, nh, nkv, rows, causal, extra, scale, P,
                                                               partial, pos_add, window, splits, tree, -1, sw);
      else kernel4<<<grid, 128, 0, st>>>(q, Kc, Vc, nh, nkv, rows, causal, extra, scale, P, partial, pos_add, window,
                                          splits, nullptr, -1, SplitWindow{});
    };
    switch (hd) {
    case 32: per_head(k_attention<32, 1>, k_attention<32, 4>, k_attention<32, 4, true>); break;
    case 64: per_head(k_attention<64, 1>, k_attention<64, 4>, k_attention<64, 4, true>); break;
    case 128: per_head(k_attention<128, 1>, k_attention<128, 4>, k_attention<128, 4, true>); break;
    case 256: per_head(k_attention<256, 1>, k_attention<256, 4>, k_attention<256, 4, true>); break;
    default:  // the backend refuses other head dims at load (attention_head_dim)
      throw std::runtime_error("flyx-cuda: attention with head dim " + std::to_string(hd));
    }
  }
  if (sw.state && !sw.last) return;
  const Out o = out ? Out{out, nullptr, nullptr} : split_out(L, rows, nh * hd);
  const int used_from = from == INT_MAX ? -1 : from;
  if (tree)
    k_attention_combine<true><<<dim3(nh, rows), hd, 0, st>>>(partial, nh, hd, splits, gate, ld_gate, o, ld_out, P,
                                                             causal, extra, pos_add, used_from, tree);
  else
    k_attention_combine<false><<<dim3(nh, rows), hd, 0, st>>>(partial, nh, hd, splits, gate, ld_gate, o, ld_out, P,
                                                              causal, extra, pos_add, used_from, nullptr);
}

}  // namespace flyx::cuda
