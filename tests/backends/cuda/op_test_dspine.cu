// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: DSpine (dspine.cu) as regions/layers.cpp's dspine_block runs it, its GEMMs between the kernels, against
// ops/dspine/ref.h in float64 on the same bf16 weights: the bridge's write (the rows after it, their rms, their gate
// projections, the rows before it), and the walk (each candidate's base score and its head row through the last
// write, every sender's pick, the chain; the test's oracle tokens in its place).
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {
namespace {

struct Bridge {
  int H, C, R, V;
  Vec codebook, gate, pred, prev, bias, read, write;  // bf16 values (the backend's), bias float32
  bf16 *d_codebook, *d_packed, *d_read, *d_write, *d_write_t;
  float* d_bias;
  ops::DSpineBridge ref() const {
    return {H, C, R, 1e-6, codebook.data(), gate.data(), pred.data(), prev.data(), bias.data()};
  }
};

Bridge random_bridge(int H, int C, int R, int V) {
  Bridge b{H, C, R, V};
  b.codebook = randv(size_t(V) * C);
  b.gate = randv(size_t(R) * C, 1 / std::sqrt(double(C)));
  b.pred = randv(size_t(R) * C, 1 / std::sqrt(double(C)));
  b.prev = randv(size_t(R) * C, 1 / std::sqrt(double(C)));
  b.read = randv(size_t(C) * H, 1 / std::sqrt(double(H)));
  b.write = randv(size_t(H) * R, 1.5 / std::sqrt(double(R)));
  b.bias = randv(size_t(R), 0.5);
  for (Vec* v : {&b.codebook, &b.gate, &b.pred, &b.prev, &b.read, &b.write}) round_bf16(*v);
  for (auto& x : b.bias) x = double(float(x));
  Vec packed(b.gate);
  packed.insert(packed.end(), b.pred.begin(), b.pred.end());
  packed.insert(packed.end(), b.prev.begin(), b.prev.end());
  Vec write_t(b.write.size());
  for (int h = 0; h < H; ++h)
    for (int r = 0; r < R; ++r) write_t[size_t(r) * H + h] = b.write[size_t(h) * R + r];
  b.d_codebook = up_bf16(b.codebook);
  b.d_packed = up_bf16(packed);
  b.d_read = up_bf16(b.read);
  b.d_write = up_bf16(b.write);
  b.d_write_t = up_bf16(write_t);
  b.d_bias = up(b.bias);
  return b;
}

DevParams* anchored(int anchor, int position = 5, int oracle_len = 0) {
  DevParams p{};
  p.position = position;
  p.anchor = anchor;
  p.oracle_len = oracle_len;
  DevParams* d = dev<DevParams>(1);
  CK(cudaMemcpy(d, &p, sizeof(p), cudaMemcpyHostToDevice));
  return d;
}

// The write over `rows` rows, as dspine_block runs it (a layer's GEMMs on 1..16 rows).
void test_dspine_write(int H, int C, int R, int rows) {
  std::printf("dspine write, hidden %d, code %d, rank %d, %d rows\n", H, C, R, rows);
  seed("dspine write", H, C, R, rows);
  const int V = 300, anchor = 17;
  const Bridge b = random_bridge(H, C, R, V);
  Vec x = randv(size_t(rows) * H, 2.0);
  for (auto& v : x) v = double(float(v));  // the rows as the device holds them
  Vec want = x, rms(rows), g(size_t(rows) * R), before(x.size());
  ops::dspine_write(b.ref(), b.read.data(), b.write.data(), anchor, rows, want.data(), rms.data(), g.data(),
                    before.data());
  float *dx = up(x), *dxn = dev<float>(x.size()), *drms = dev<float>(rows), *dcodes = dev<float>(size_t(rows) * C);
  float *dgates = dev<float>(size_t(rows) * 3 * R), *dmsg = dev<float>(size_t(rows) * R), *dg = dev<float>(g.size());
  float *dd = dev<float>(x.size()), *dbefore = dev<float>(x.size());
  DevParams* P = anchored(anchor);
  dspine_normalize(dx, H, 1e-6f, rows, dxn, drms, 0);
  gemm(dxn, H, b.d_read, C, H, dcodes, C, rows, false, TL);
  dspine_codes(dcodes, C, b.d_codebook, P, 1e-6f, rows, 0);
  gemm(dcodes, C, b.d_packed, 3 * R, C, dgates, 3 * R, rows, false, TL);
  dspine_message(dgates, R, b.d_bias, rows, dmsg, dg, 0);
  gemm(dmsg, R, b.d_write, H, R, dd, H, rows, false, TL);
  dspine_apply(dx, dd, drms, H, rows, dbefore, 0);
  expect("the rows after the write", down(dx, x.size()), want, 2e-5);
  expect("their rms", down(drms, rms.size()), rms, 1e-6);
  expect("their gate projections", down(dg, g.size()), g, 2e-5);
  expect("the rows before it", down(dbefore, x.size()), before, 0);
}

// The walk over the n rows after the anchor (k candidates a row), as dspine_block runs it: the candidates' heads and
// base scores, their heads through the last write and the senders' pred | prev on the round GEMMs (three-part
// inputs), every sender's pick, the chain.  The picks are held to the float64 scores of the same senders: the best,
// or within float32's rounding of it (a near tie may go either way).
void test_dspine_walk(int H, int C, int R, int k, int n) {
  std::printf("dspine walk, hidden %d, code %d, rank %d, %d candidates, %d rows\n", H, C, R, k, n);
  seed("dspine walk", H, C, R, k, n);
  const int V = 2000, anchor = 42;
  const Bridge b = random_bridge(H, C, R, V);
  Vec W = randv(size_t(V) * H, 0.5), nw = randv(size_t(H), 0.2);
  round_bf16(W);
  for (auto& w : nw) w = double(float(1 + w));
  Vec before = randv(size_t(n + 1) * H, 2.0), g = randv(size_t(n + 1) * R), rms(size_t(n + 1));
  for (int r = 0; r <= n; ++r) rms[size_t(r)] = ops::dspine_rms(&before[size_t(r) * H], H);
  for (auto& v : before) v = double(float(v));
  for (auto& v : g) v = double(float(v));
  for (auto& v : rms) v = double(float(v));
  std::vector<int> cand(size_t(n) * k);
  std::uniform_int_distribution<int> tok(0, V - 1);
  for (int t = 0; t < n; ++t)  // distinct tokens a row, as a top-k's
    for (int c = 0; c < k; ++c) {
      int v;
      do v = tok(rng);
      while (std::find(cand.begin() + size_t(t) * k, cand.begin() + size_t(t) * k + c, v) !=
             cand.begin() + size_t(t) * k + c);
      cand[size_t(t) * k + c] = v;
    }
  const ops::DSpineBridge br = b.ref();
  auto head_row = [&](int32_t t) { return &W[size_t(t) * H]; };
  // float64: each sender's scores of the row after it
  const int senders = 1 + (n - 1) * k;
  std::vector<double> ref_scores(size_t(senders) * k);
  Vec e(static_cast<size_t>(C));
  for (int i = 0; i < senders; ++i) {
    const int t = i == 0 ? 0 : 1 + (i - 1) / k;
    ops::dspine_token_code(br, i == 0 ? anchor : cand[size_t(i) - 1], e.data());
    for (int c = 0; c < k; ++c)
      ref_scores[size_t(i) * k + c] = ops::dspine_score(br, b.write.data(), nw.data(), &before[size_t(t + 1) * H],
                                                        rms[size_t(t + 1)], &g[size_t(t + 1) * R], e.data(),
                                                        head_row(cand[size_t(t) * k + c]));
  }
  std::vector<int32_t> want(static_cast<size_t>(n));
  {
    std::vector<int32_t> c32(cand.begin(), cand.end());
    ops::dspine_walk(br, b.write.data(), nw.data(), before.data(), rms.data(), g.data(), c32.data(), k, n, anchor,
                     head_row, want.data());
  }
  int* dcand = dev<int>(cand.size());
  CK(cudaMemcpy(dcand, cand.data(), cand.size() * 4, cudaMemcpyHostToDevice));
  bf16* dW = up_bf16(W);
  float *dnw = up(nw), *dbefore = up(before), *dg = up(g), *drms = up(rms);
  float *dheads = dev<float>(size_t(n) * k * H), *dbase = dev<float>(size_t(n) * k);
  float* dtg = dev<float>(size_t(n) * k * R);
  float *dcodes = dev<float>(size_t(senders) * C), *dsv = dev<float>(size_t(senders) * 2 * R);
  int *dpick = dev<int>(size_t(senders)), *ddrafts = dev<int>(size_t(n));
  DevParams* P = anchored(anchor);
  Launch walk = TL;  // as dspine_block: a round's kernel, every row in one launch
  walk.round_rows = true;
  dspine_heads(dcand, dW, dnw, dbefore, H, k, n * k, dheads, dbase, 0);
  {  // a packed head (W null): the rows given as float32 (embed_blocks') in heads, the same bits
    std::vector<float> rows(size_t(n) * k * H);
    for (int i = 0; i < n * k; ++i)
      for (int h = 0; h < H; ++h) rows[size_t(i) * H + h] = float(W[size_t(cand[size_t(i)]) * H + h]);
    float *dheads2 = dev<float>(rows.size()), *dbase2 = dev<float>(size_t(n) * k);
    CK(cudaMemcpy(dheads2, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
    dspine_heads(dcand, nullptr, dnw, dbefore, H, k, n * k, dheads2, dbase2, 0);
    const bool same = down(dheads2, rows.size()) == down(dheads, rows.size()) &&
                      down(dbase2, size_t(n) * k) == down(dbase, size_t(n) * k);
    std::printf("  %-4s %-48s\n", same ? "ok" : "FAIL", "a packed head's rows (W null): the same bits");
    g_fail += !same;
    CK(cudaFree(dheads2));
    CK(cudaFree(dbase2));
  }
  gemm(dheads, H, b.d_write_t, R, H, dtg, R, n * k, false, walk);
  dspine_senders(P, dcand, b.d_codebook, C, senders, dcodes, 0);
  gemm(dcodes, C, b.d_packed + size_t(R) * C, 2 * R, C, dsv, 2 * R, senders, false, walk);
  dspine_scores(dg, dsv, b.d_bias, dtg, dbase, drms, R, k, senders, dpick, 0);
  dspine_walk(P, dcand, dpick, k, n, nullptr, ddrafts, 0);
  // base scores: <before_{t+1}, head * nw>
  Vec base(size_t(n) * k);
  for (int i = 0; i < n * k; ++i) {
    double s = 0;
    for (int h = 0; h < H; ++h)
      s += before[size_t(i / k + 1) * H + h] * W[size_t(cand[size_t(i)]) * H + h] * float(nw[size_t(h)]);
    base[size_t(i)] = s;
  }
  expect("the candidates' base scores", down(dbase, base.size()), base, 2e-5);
  // the picks: each the float64 best, or within rounding of it
  const std::vector<int> pick = down_i(dpick, size_t(senders)), drafts = down_i(ddrafts, size_t(n));
  double worst = 0, scale = 1e-30;
  int exact = 0;
  for (int i = 0; i < senders; ++i) {
    const double* sc = &ref_scores[size_t(i) * k];
    const double best = *std::max_element(sc, sc + k);
    for (int c = 0; c < k; ++c) scale = std::max(scale, std::fabs(sc[c]));
    worst = std::max(worst, best - sc[pick[size_t(i)]]);
    exact += sc[pick[size_t(i)]] == best;
  }
  const bool picks_ok = worst <= 1e-5 * scale;
  std::printf("  %-4s %-48s %d / %d the float64 best, the others %.1e below (scale %.1e)\n", picks_ok ? "ok" : "FAIL",
              "every sender's pick", exact, senders, worst, scale);
  if (!picks_ok) ++g_fail;
  // the chain through the picks; where it is the float64 walk's (every pick its best), its drafts
  bool chain_ok = true;
  int j = pick[0];
  chain_ok = chain_ok && drafts[0] == cand[size_t(j)];
  for (int t = 1; t < n; ++t) {
    j = pick[size_t(1 + (t - 1) * k + j)];
    chain_ok = chain_ok && drafts[size_t(t)] == cand[size_t(t) * k + j];
  }
  const bool same = std::equal(drafts.begin(), drafts.end(), want.begin());
  const bool walk_ok = chain_ok && (same || exact < senders);
  std::printf("  %-4s %-48s %s\n", walk_ok ? "ok" : "FAIL", "the chain through the picks",
              same ? "the float64 walk's drafts" : "(a near tie went the other way)");
  if (!walk_ok) ++g_fail;
  // the test's tokens in place of the walk's
  std::vector<int> oracle(64);
  for (size_t i = 0; i < oracle.size(); ++i) oracle[i] = int(i) * 7 % V;
  int* doracle = dev<int>(oracle.size());
  CK(cudaMemcpy(doracle, oracle.data(), oracle.size() * 4, cudaMemcpyHostToDevice));
  DevParams* Po = anchored(anchor, 5, int(oracle.size()));
  dspine_walk(Po, dcand, dpick, k, n, doracle, ddrafts, 0);
  const std::vector<int> od = down_i(ddrafts, size_t(n));
  bool oracle_ok = true;
  for (int i = 0; i < n; ++i) oracle_ok = oracle_ok && od[size_t(i)] == oracle[size_t(5 + 1 + i)];
  std::printf("  %-4s %-48s\n", oracle_ok ? "ok" : "FAIL", "the oracle's tokens in the walk's place");
  if (!oracle_ok) ++g_fail;
}

}  // namespace

void test_dspine() {
  test_dspine_write(64, 32, 16, 16);      // the tiny drafter's
  test_dspine_write(5120, 1024, 512, 16); // the 27B's
  test_dspine_write(4096, 1024, 512, 7);  // the 8B's, a block of fewer rows
  test_dspine_walk(64, 32, 16, 16, 15);
  test_dspine_walk(5120, 1024, 512, 16, 15);
  test_dspine_walk(4096, 1024, 512, 8, 6);
}

}  // namespace flyx::optest
