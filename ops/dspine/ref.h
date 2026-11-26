// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
//
// DSpine (backend_abi.h flyx_dspine_block; python/reference/flyx_ref/dspine.py): the code bridge's write after each
// of its layers and the walk that drafts from the last one, in float64.
//
// The write over a block's rows (row 0 the anchor), x_r a row's residual after the layer:
//   rms_r = sqrt(mean(x_r^2))
//   c_r   = sqrt(C) u_r / (|u_r| + eps), u_r = read (x_r / (rms_r + eps))          (r >= 1)
//   c_0   = sqrt(C) codebook[anchor]
//   m_r   = sigmoid(gate c_r + pred c_{r-1} + gate_bias) * (prev c_{r-1})           (r >= 1; m_0 = 0)
//   x_r  += rms_r write m_r
// The walk: with cand the head's candidates for the rows after the anchor (row t + 1's at cand[t K ..]), x_r, rms_r and
// g_r = gate c_r from before the last layer's write, draft t is the candidate maximising
//   score(t, e, k) = <x_{t+1} + rms_{t+1} write m(g_{t+1}, e), norm_w * head[cand[t K + k]]>
//                  = <x_{t+1}, norm_w * head[k]> + rms_{t+1} m(g_{t+1}, e) . (write^T (norm_w * head[k]))
// with e = sqrt(C) codebook[the draft before it] (the anchor for t = 0), m(g, e) = sigmoid(g + pred e + gate_bias) *
// (prev e); the first candidate on ties.  A backend computes the second form (the base, then the message against
// each candidate's head row through write), every sender at once.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ops/activation/ref.h"

namespace flyx::ops {

// The bridge's weights, row-major in float64: codebook [vocab, code], gate / pred / prev [rank, code], gate_bias
// [rank].
struct DSpineBridge {
  int hidden = 0, code = 0, rank = 0;
  double eps = 0;
  const double *codebook = nullptr, *gate = nullptr, *pred = nullptr, *prev = nullptr, *gate_bias = nullptr;
};

// y[n] = sum_k W[n K + k] x[k], n < N.
inline void dspine_mv(const double* W, int N, int K, const double* x, double* y) {
  for (int n = 0; n < N; ++n) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += W[size_t(n) * size_t(K) + size_t(k)] * x[k];
    y[n] = s;
  }
}

inline double dspine_rms(const double* x, int H) {
  double s = 0;
  for (int h = 0; h < H; ++h) s += x[h] * x[h];
  return std::sqrt(s / H);
}

// A row's code from its residual x (rms its rms), through the layer's read [code, hidden].
inline void dspine_code(const DSpineBridge& b, const double* read, const double* x, double rms, double* c) {
  std::vector<double> xn(size_t(b.hidden));
  for (int h = 0; h < b.hidden; ++h) xn[size_t(h)] = x[h] / (rms + b.eps);
  dspine_mv(read, b.code, b.hidden, xn.data(), c);
  double n2 = 0;
  for (int i = 0; i < b.code; ++i) n2 += c[i] * c[i];
  const double f = std::sqrt(double(b.code)) / (std::sqrt(n2) + b.eps);
  for (int i = 0; i < b.code; ++i) c[i] *= f;
}

// A token's code: sqrt(code) codebook[token].
inline void dspine_token_code(const DSpineBridge& b, int32_t token, double* c) {
  const double f = std::sqrt(double(b.code));
  for (int i = 0; i < b.code; ++i) c[i] = f * b.codebook[size_t(token) * size_t(b.code) + size_t(i)];
}

// The message to a row of gate projection g (gate c_r) from a sender's code e.
inline void dspine_message(const DSpineBridge& b, const double* g, const double* e, double* m) {
  std::vector<double> s(size_t(b.rank)), v(size_t(b.rank));
  dspine_mv(b.pred, b.rank, b.code, e, s.data());
  dspine_mv(b.prev, b.rank, b.code, e, v.data());
  for (int i = 0; i < b.rank; ++i) m[i] = sigmoid(g[i] + s[size_t(i)] + b.gate_bias[i]) * v[size_t(i)];
}

// A layer's write on `rows` rows x [rows, hidden] in place (read [code, hidden], write [hidden, rank] the layer's).
// Out when given: every row's rms [rows] and gate projection g [rows, rank], and x as it was [rows, hidden].
inline void dspine_write(const DSpineBridge& b, const double* read, const double* write, int32_t anchor, int rows,
                         double* x, double* rms_out = nullptr, double* g_out = nullptr, double* x_before = nullptr) {
  const int H = b.hidden, C = b.code, R = b.rank;
  std::vector<double> rms(static_cast<size_t>(rows)), code(size_t(rows) * C), g(size_t(rows) * R),
      m(static_cast<size_t>(R)), d(static_cast<size_t>(H));
  for (int r = 0; r < rows; ++r) {
    rms[size_t(r)] = dspine_rms(x + size_t(r) * H, H);
    if (r == 0) dspine_token_code(b, anchor, code.data());
    else dspine_code(b, read, x + size_t(r) * H, rms[size_t(r)], code.data() + size_t(r) * C);
    dspine_mv(b.gate, R, C, code.data() + size_t(r) * C, g.data() + size_t(r) * R);
  }
  if (x_before) std::copy(x, x + size_t(rows) * H, x_before);
  for (int r = rows - 1; r >= 1; --r) {  // each row from its sender's code, which the write does not change
    dspine_message(b, g.data() + size_t(r) * R, code.data() + size_t(r - 1) * C, m.data());
    dspine_mv(write, H, R, m.data(), d.data());
    for (int h = 0; h < H; ++h) x[size_t(r) * H + h] += rms[size_t(r)] * d[size_t(h)];
  }
  if (rms_out) std::copy(rms.begin(), rms.end(), rms_out);
  if (g_out) std::copy(g.begin(), g.end(), g_out);
}

// score(t, e, k) above for one row: x, rms and g the row's from before the last write, e the sender's code, hk the
// candidate's head row [hidden]; write the last layer's [hidden, rank], nw the final norm's weight.
inline double dspine_score(const DSpineBridge& b, const double* write, const double* nw, const double* x, double rms,
                           const double* g, const double* e, const double* hk) {
  const int H = b.hidden, R = b.rank;
  std::vector<double> m(static_cast<size_t>(R)), d(static_cast<size_t>(H));
  dspine_message(b, g, e, m.data());
  dspine_mv(write, H, R, m.data(), d.data());
  double s = 0;
  for (int h = 0; h < H; ++h) s += (x[h] + rms * d[size_t(h)]) * nw[h] * hk[h];
  return s;
}

// The walk's n drafts (rows 1..n of the block): x [n + 1, hidden], rms [n + 1] and g [n + 1, rank] from before the
// last write (row 0 the anchor's), cand [n, k]; head_row(token) the head's row of a token [hidden].  scores [n, k]
// out when given: each row's scores after the draft before it.
template <class HeadRow>
void dspine_walk(const DSpineBridge& b, const double* write, const double* nw, const double* x, const double* rms,
                 const double* g, const int32_t* cand, int k, int n, int32_t anchor, HeadRow head_row,
                 int32_t* drafts, double* scores = nullptr) {
  const int H = b.hidden, R = b.rank;
  std::vector<double> e(size_t(b.code));
  int32_t prev = anchor;
  for (int t = 0; t < n; ++t) {
    dspine_token_code(b, prev, e.data());
    int best = 0;
    double best_score = 0;
    for (int c = 0; c < k; ++c) {
      const double s = dspine_score(b, write, nw, x + size_t(t + 1) * H, rms[t + 1], g + size_t(t + 1) * R,
                                    e.data(), head_row(cand[size_t(t) * k + c]));
      if (scores) scores[size_t(t) * k + c] = s;
      if (c == 0 || s > best_score) best = c, best_score = s;
    }
    prev = cand[size_t(t) * k + best];
    drafts[t] = prev;
  }
}

}  // namespace flyx::ops
