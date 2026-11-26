// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: bf16 GEMMs: every path against linear, and the SwiGLU epilogue.
#include "tests/backends/cuda/op_test.cuh"

#include <set>
#include <utility>

#include "flyx/gemm_tables.inc"

namespace flyx::optest {

void test_gemm() {
  std::printf("gemm\n");
  seed("gemm");
  struct S { int rows, N, K; };
  for (S s : {S{1, 200, 256}, S{3, 77, 64}, S{16, 300, 512}, S{17, 130, 96}, S{64, 64, 2560}, S{200, 333, 136}}) {
    Vec x = randv(size_t(s.rows) * s.K), w = randv(size_t(s.N) * s.K, 1.0 / std::sqrt(s.K)), y0 = randv(size_t(s.rows) * s.N);
    std::vector<uint16_t> store;
    auto t = view_of(w, &store, s.N, s.K);
    Vec want(size_t(s.rows) * s.N), want16(want.size());
    ops::linear(x.data(), s.rows, s.K, &t, s.N, want.data());
    ops::linear_a16(x.data(), s.rows, s.K, &t, s.N, want16.data());  // the tensor-core paths take bf16 inputs
    Vec want_acc = want, want16_acc = want16;
    for (size_t i = 0; i < want.size(); ++i) want_acc[i] += y0[i], want16_acc[i] += y0[i];
    float* dx = up(x);
    bf16* dw = up_bf16(w);
    float* dy = up(y0);
    std::string tag = std::to_string(s.rows) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
    gemm_rows(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, false, 0);
    expect("rows path " + tag, down(dy, want.size()), want, 2e-5);
    CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.end()).data(), y0.size() * 4, cudaMemcpyHostToDevice));
    gemm_rows(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, true, 0);
    expect("rows path accumulate " + tag, down(dy, want.size()), want_acc, 2e-5);
    gemm_mma(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, false, 0);
    expect("tensor-core path (bf16 inputs) " + tag, down(dy, want.size()), want16, 2e-5);
    if (s.K % 32 == 0) {
      gemm_tc(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, false, TL);
      expect("pipelined tensor-core path (bf16 inputs) " + tag, down(dy, want.size()), want16, 2e-5);
      CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.end()).data(), y0.size() * 4, cudaMemcpyHostToDevice));
      gemm_tc(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, true, TL);
      const Vec acc0 = down(dy, want.size());
      expect("pipelined accumulate " + tag, acc0, want16_acc, 2e-5);
      for (int c = 1; c < kTcConfigs; ++c) {
        CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.end()).data(), y0.size() * 4, cudaMemcpyHostToDevice));
        gemm_tc(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, true, TL, c);
        expect("pipelined configuration " + std::to_string(c) + ": same bits " + tag, down(dy, want.size()), acc0, 0);
      }
    }
    if (s.rows <= 16 && s.K % 32 == 0) {
      gemm_skinny(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, false, TL);
      expect("skinny tensor-core path " + tag, down(dy, want.size()), want, 2e-5);
      CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.end()).data(), y0.size() * 4, cudaMemcpyHostToDevice));
      gemm_skinny(dx, s.K, dw, s.N, s.K, dy, s.N, s.rows, true, TL);
      expect("skinny accumulate " + tag, down(dy, want.size()), want_acc, 2e-5);
    }
  }
  // A row's result must not depend on the other rows (plain decoding == verification).
  {
    const int N = 1000, K = 2560;
    Vec x = randv(size_t(16) * K), w = randv(size_t(N) * K, 1.0 / std::sqrt(K));
    float* dx = up(x);
    bf16* dw = up_bf16(w);
    float* d16 = dev<float>(size_t(16) * N);
    float* d1 = dev<float>(N);
    gemm_skinny(dx, K, dw, N, K, d16, N, 16, false, TL);
    bool same = true;
    for (int r : {0, 5, 9, 15}) {
      gemm_skinny(dx + size_t(r) * K, K, dw, N, K, d1, N, 1, false, TL);
      Vec a = down(d16 + size_t(r) * N, N), b = down(d1, N);
      same &= a == b;
    }
    for (int cfg = 0; cfg < kSkinnyConfigs; ++cfg) {  // every compiled tuning configuration gives the same bits
      if (!gemm_skinny_cfg_compiled(cfg, 1, false)) continue;
      gemm_skinny_cfg(cfg, dx, K, dw, N, K, d1, N, 1, false, TL);
      same &= down(d1, N) == down(d16, N);
    }
    std::printf("  %-4s skinny: a row's result does not depend on the row count or the configuration\n",
                same ? "ok" : "FAIL");
    g_fail += !same;
  }
}

// A round's GEMM of more than 16 rows (Launch::round_rows: a wide round's, up to FLYX_MAX_WIDE_ROWS; gemm() takes a
// linear's bf16 weights so, regions/linear.cpp, and an adapter's): every row the bits it has alone, plain and
// accumulating, for bf16 weights (k_gemm_skinny_mt, every configuration of three parts, and the dispatch; with option
// row_gemms the row path, which tiles itself), NVFP4 weights (gemm_fp4_rows: up to four tiles a weight decode) and
// gemm_rows called directly.  The same rows without the flag (a prefill pass: the tensor cores, bf16 inputs) differ:
// what the tiles keep a round from.  A drafting pass's (Launch::draft: bf16 inputs on k_gemm_skinny_mt) against
// linear_a16, every configuration of one part, the same bits run to run.  More than 64 rows in three parts (DSpine's
// walk: one launch, 64 a grid row): every row the bits it has alone too.
void test_gemm_round() {
  std::printf("a round's GEMM of more than 16 rows\n");
  struct S { int N, K; };
  // MiniCPM5-1B's q|k|v and down projections, and K not a multiple of 64 (bf16 only: NVFP4's skinny kernel needs it)
  for (S s : {S{2560, 1536}, S{1536, 4608}, S{333, 96}}) {
    seed("gemm round", s.N, s.K);
    const int most = 240;  // FLYX_MAX_WIDE_ROWS, and DSpine's walk past it (15 rows of 16 candidates)
    const Vec x = randv(size_t(most) * s.K), y0 = randv(size_t(most) * s.N);
    float *dx = up(x), *dy = dev<float>(size_t(most) * s.N), *d1 = dev<float>(s.N);
    bf16* dw = up_bf16(randv(size_t(s.N) * s.K, 1.0 / std::sqrt(s.K)));
    std::vector<uint8_t> Qh(size_t(s.N) * s.K / 2), Sh(size_t(s.N) * s.K / 16);
    for (auto& b : Qh) b = uint8_t(rng());
    for (auto& b : Sh) b = uint8_t(0x28 + rng() % 0x18);
    Vec rsv(s.N);
    for (auto& v : rsv) v = 0.01 + 0.01 * double(rng() % 1000) / 1000.0;
    uint8_t *dq = dev<uint8_t>(Qh.size()), *ds = dev<uint8_t>(Sh.size());
    CK(cudaMemcpy(dq, Qh.data(), Qh.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, Sh.data(), Sh.size(), cudaMemcpyHostToDevice));
    float* drs = up(rsv);
    static KernelOptions row_opts;
    row_opts.row_gemms = true;
    Launch LR = TL, LRR = TL;
    LR.round_rows = true;
    LRR.round_rows = true;
    LRR.opt = &row_opts;
    auto set_y = [&](int rows) {
      CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.begin() + size_t(rows) * s.N).data(),
                    size_t(rows) * s.N * 4, cudaMemcpyHostToDevice));
    };
    // run(rows, accumulate) of the path into dy; one(r, accumulate) of row r alone into d1 (y0's row r first).
    auto same_as_one = [&](const char* what, int rows, auto run, auto one) {
      bool same = true;
      for (bool acc : {false, true}) {
        set_y(rows);
        run(rows, acc);
        const Vec all = down(dy, size_t(rows) * s.N);
        for (int r = 0; r < rows; ++r) {
          CK(cudaMemcpy(d1, std::vector<float>(y0.begin() + size_t(r) * s.N, y0.begin() + size_t(r + 1) * s.N).data(),
                        size_t(s.N) * 4, cudaMemcpyHostToDevice));
          one(r, acc);
          const Vec a = down(d1, s.N);
          same &= std::equal(a.begin(), a.end(), all.begin() + size_t(r) * s.N);
        }
      }
      std::printf("  %-4s %s %d x %d, %d rows: every row the bits it has alone\n", same ? "ok" : "FAIL", what, s.N,
                  s.K, rows);
      g_fail += !same;
    };
    for (int rows : {17, 31, 32, 33, 48, 64}) {
      same_as_one("bf16 (the round's dispatch)", rows,
                  [&](int n, bool acc) { gemm(dx, s.K, dw, s.N, s.K, dy, s.N, n, acc, LR); },
                  [&](int r, bool acc) { gemm(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, acc, TL); });
      for (int c = 0; c < kMtConfigCount && s.K % 32 == 0; ++c) {
        if (kMtConfigs[c].parts != 3 || !gemm_mt_cfg_compiled(c)) continue;
        const std::string what = "bf16 k_gemm_skinny_mt " + std::to_string(c);
        same_as_one(what.c_str(), rows,
                    [&](int n, bool acc) { gemm_skinny_mt_cfg(c, dx, s.K, dw, s.N, s.K, dy, s.N, n, acc, TL); },
                    [&](int r, bool acc) { gemm(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, acc, TL); });
      }
      same_as_one("bf16, option row_gemms", rows,
                  [&](int n, bool acc) { gemm(dx, s.K, dw, s.N, s.K, dy, s.N, n, acc, LRR); },
                  [&](int r, bool acc) { gemm(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, acc, LRR); });
      same_as_one("gemm_rows", rows,
                  [&](int n, bool acc) { gemm_rows(dx, s.K, dw, s.N, s.K, dy, s.N, n, acc, 0); },
                  [&](int r, bool acc) { gemm_rows(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, acc, 0); });
      if (s.K % 64 == 0)
        same_as_one("nvfp4 (gemm_fp4_rows)", rows,
                    [&](int n, bool acc) { gemm_fp4_rows(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, n, acc, LR); },
                    [&](int r, bool acc) {
                      gemm_fp4(dx + size_t(r) * s.K, s.K, dq, ds, drs, s.N, s.K, d1, s.N, 1, acc, TL);
                    });
    }
    for (int rows : {65, 100, 240}) {
      if (s.K % 32) break;  // k_gemm_skinny_mt's spans
      same_as_one("bf16 (the round's dispatch, one launch)", rows,
                  [&](int n, bool acc) { gemm(dx, s.K, dw, s.N, s.K, dy, s.N, n, acc, LR); },
                  [&](int r, bool acc) { gemm(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, acc, TL); });
      for (int c = 0; c < kMtConfigCount; ++c) {
        if (kMtConfigs[c].parts != 3 || !gemm_mt_cfg_compiled(c)) continue;
        const std::string what = "bf16 k_gemm_skinny_mt " + std::to_string(c) + ", one launch";
        same_as_one(what.c_str(), rows,
                    [&](int n, bool acc) { gemm_skinny_mt_cfg(c, dx, s.K, dw, s.N, s.K, dy, s.N, n, acc, TL); },
                    [&](int r, bool acc) { gemm(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, acc, TL); });
      }
    }
    // A prefill pass of the same rows: the tensor cores (bf16 inputs), not the bits of a row alone.
    const int rows = 32;
    gemm(dx, s.K, dw, s.N, s.K, dy, s.N, rows, false, TL);
    const Vec pre = down(dy, size_t(rows) * s.N);
    int differ = 0;
    for (int r = 0; r < rows; ++r) {
      gemm(dx + size_t(r) * s.K, s.K, dw, s.N, s.K, d1, s.N, 1, false, TL);
      const Vec a = down(d1, s.N);
      differ += !std::equal(a.begin(), a.end(), pre.begin() + size_t(r) * s.N);
    }
    std::printf("  %-4s bf16 %d x %d, a prefill pass of %d rows: %d rows' bits differ from the row alone\n",
                differ > 0 ? "ok" : "FAIL", s.N, s.K, rows, differ);
    g_fail += differ == 0;
    if (s.K % 32 == 0) {  // a drafting pass: bf16 inputs, against linear_a16; each configuration the same bits again
      std::vector<uint16_t> store;
      const Vec w = down_bf16(dw, size_t(s.N) * s.K);
      auto t = view_of(w, &store, s.N, s.K);
      Launch LD = TL;
      LD.draft = true;
      for (int n : {17, 32, 48, 64}) {
        Vec want(size_t(n) * s.N);
        ops::linear_a16(x.data(), n, s.K, &t, s.N, want.data());
        const std::string tag = std::to_string(n) + " rows " + std::to_string(s.N) + "x" + std::to_string(s.K);
        gemm(dx, s.K, dw, s.N, s.K, dy, s.N, n, false, LD);
        expect("bf16 drafting pass (bf16 inputs) " + tag, down(dy, want.size()), want, 2e-5);
        bool again = true;
        for (int c = 0; c < kMtConfigCount; ++c) {
          if (kMtConfigs[c].parts != 1 || !gemm_mt_cfg_compiled(c)) continue;
          gemm_skinny_mt_cfg(c, dx, s.K, dw, s.N, s.K, dy, s.N, n, false, TL);
          const Vec a = down(dy, want.size());
          expect("  k_gemm_skinny_mt " + std::to_string(c) + " (one part) " + tag, a, want, 2e-5);
          gemm_skinny_mt_cfg(c, dx, s.K, dw, s.N, s.K, dy, s.N, n, false, TL);
          again &= a == down(dy, want.size());
        }
        std::printf("  %-4s one part, every configuration: the same bits run to run, %s\n", again ? "ok" : "FAIL",
                    tag.c_str());
        g_fail += !again;
        // Weights on the same bf16 rows in one launch (gemm_draft, as DV's canvas pass groups them): each its own
        // columns of the input, plain, accumulating and in bf16, the bits of each launched alone.
        Vec x2(size_t(n) * 2 * s.K);
        for (int r = 0; r < n; ++r)
          for (int k = 0; k < 2 * s.K; ++k)
            x2[size_t(r) * 2 * s.K + k] = x[size_t(k < s.K ? r : (r + 1) % n) * s.K + k % s.K];
        float* dx2 = up(x2);
        float *ya = dev<float>(size_t(n) * s.N), *yb = dev<float>(size_t(n) * s.N);
        bf16* yc = dev<bf16>(size_t(n) * s.N);
        auto run = [&](std::initializer_list<int> which) {
          CK(cudaMemcpy(yb, std::vector<float>(y0.begin(), y0.begin() + size_t(n) * s.N).data(), size_t(n) * s.N * 4,
                        cudaMemcpyHostToDevice));
          const MtOut outs[3] = {MtOut{dw, ya, nullptr, s.N, s.N, 0, 0, 0}, MtOut{dw, yb, nullptr, s.N, s.N, s.K, 1, 0},
                                 MtOut{dw, nullptr, yc, s.N, s.N, 0, 0, 0}};
          MtGroups g;
          for (int i : which) g.g[g.n++] = outs[i];
          gemm_draft(stage_bf16(dx2, 2 * s.K, n, 2 * s.K, TL), 2 * s.K, s.K, g, n, TL);
        };
        run({0, 1, 2});
        const Vec ga = down(ya, size_t(n) * s.N), gb = down(yb, size_t(n) * s.N), gc = down_bf16(yc, size_t(n) * s.N);
        run({0});
        bool grouped = ga == down(ya, size_t(n) * s.N);
        run({1});
        grouped &= gb == down(yb, size_t(n) * s.N);
        run({2});
        grouped &= gc == down_bf16(yc, size_t(n) * s.N);
        for (size_t i = 0; i < ga.size(); ++i) {
          const double rounded = __bfloat162float(__float2bfloat16_rn(float(ga[i])));
          grouped &= rounded == gc[i];
        }
        expect("  gemm_draft, the first weight " + tag, ga, want, 2e-5);
        std::printf("  %-4s gemm_draft: three weights in one launch, each the bits it has alone (bf16 out: its "
                    "sums rounded), %s\n", grouped ? "ok" : "FAIL", tag.c_str());
        g_fail += !grouped;
        for (void* p : std::initializer_list<void*>{dx2, ya, yb, yc}) CK(cudaFree(p));
      }
    }
    for (void* p : std::initializer_list<void*>{dx, dy, d1, dw, dq, ds, drs}) CK(cudaFree(p));
  }
}

// A drafting pass's GEMMs (DV's canvas) under every tuning table (the defaults and each GPU's): the K partition is
// K's alone (gemm_draft_chunks) at every row count of 17..64, whatever the shape; and the weights gemm_draft takes in
// one launch, of other shapes than the first (MiniCPM5-1B's canvas groups, which run in the first weight's tile and
// unroll), each give the bits gemm() of a drafting pass gives it alone (what option fusion=0 runs): plain,
// accumulating and rounded to bf16.
void test_gemm_draft_tables() {
  std::printf("a drafting pass's GEMMs under every tuning table\n");
  std::vector<int> tunings = {-1};
  const std::string names = gemm_tuning_names();
  for (size_t i = 0; i < names.size();) {
    const size_t e = std::min(names.find(", ", i), names.size());
    if (const int t = gemm_tuning_named(names.substr(i, e - i).c_str()); t >= 0) tunings.push_back(t);
    i = e + 2;
  }
  struct S { int N, K; };
  int wrong = 0, checked = 0;
  for (int t : tunings)
    for (S s : {S{2560, 1536}, S{768, 1536}, S{2048, 256}, S{256, 256}, S{1536, 2048}, S{256, 2048}, S{1536, 256},
                S{9216, 1536}, S{512, 1536}, S{4608, 256}, S{1536, 4608}, S{256, 4608}, S{130560, 1536}, S{333, 96},
                S{4096, 4096}, S{2560, 9728}})
      for (int rows = 17; rows <= 64; ++rows, ++checked) {
        const int c = gemm_config(t, 4, s.N, s.K, rows);
        wrong += c < 0 || c >= kMtConfigCount || kMtConfigs[c].parts != 1 ||
                 kMtConfigs[c].chunks != gemm_draft_chunks(s.K);
      }
  std::printf("  %-4s %zu tables, %d shapes and row counts: a drafting pass's GEMM in its K's chunks (%d, %d from "
              "K = 4096), %d not\n", wrong == 0 ? "ok" : "FAIL", tunings.size(), checked, gemm_draft_chunks(32),
              gemm_draft_chunks(4096), wrong);
  g_fail += wrong != 0;
  // The launches: under the tables whose kernels this build has for the GPU (a release build: those of its
  // architecture, gemm_tuning_compiled_for; every table otherwise).
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  const size_t every = tunings.size();
  tunings.erase(std::remove_if(tunings.begin(), tunings.end(),
                               [&](int t) { return !gemm_tuning_compiled_for(t, prop.major * 10 + prop.minor); }),
                tunings.end());

  // MiniCPM5-1B's canvas launches (layers.cpp, dv_context): a weight's out in float32, plain or accumulating, and an
  // adapter's a rounded to bf16; the adapters' b, each from its own columns of the rows.
  struct W { int N, x_col; bool acc, b16; };
  struct G { const char* what; int K, ldx; std::vector<W> w; };
  const G groups[] = {{"w_in with a_qkv", 1536, 1536, {{2560, 0, false, false}, {768, 0, false, true}}},
                      {"b of q, k and v", 256, 768, {{2048, 0, true, false}, {256, 256, true, false},
                                                     {256, 512, true, false}}},
                      {"w_out with a_o", 2048, 2048, {{1536, 0, true, false}, {256, 0, false, true}}},
                      {"w_gu with a_gu", 1536, 1536, {{9216, 0, false, false}, {512, 0, false, true}}},
                      {"b of gate and up", 256, 512, {{4608, 0, true, false}, {4608, 256, true, false}}},
                      {"w_down with a_down", 4608, 4608, {{1536, 0, true, false}, {256, 0, false, true}}}};
  std::vector<KernelOptions> per_table(tunings.size());
  for (const G& gr : groups) {
    seed("gemm draft tables", gr.K, gr.ldx, int(gr.w.size()));
    const int most = 64;
    const Vec x = randv(size_t(most) * gr.ldx);
    float* dx = up(x);
    std::vector<bf16*> dw;
    std::vector<float*> y, y1;
    std::vector<bf16*> y16;
    std::vector<Vec> y0;
    for (const W& w : gr.w) {
      dw.push_back(up_bf16(randv(size_t(w.N) * gr.K, 1.0 / std::sqrt(gr.K))));
      y0.push_back(randv(size_t(most) * w.N));
      y.push_back(dev<float>(size_t(most) * w.N));
      y1.push_back(dev<float>(size_t(most) * w.N));
      y16.push_back(dev<bf16>(size_t(most) * w.N));
    }
    bool same = true;
    for (size_t ti = 0; ti < tunings.size(); ++ti) {
      per_table[ti].tuning = tunings[ti];
      Launch L = TL, LD = TL;
      L.opt = LD.opt = &per_table[ti];
      LD.draft = true;
      for (int rows : {17, 32, 48, 64}) {
        auto start = [&](size_t i, float* p) {
          CK(cudaMemcpy(p, std::vector<float>(y0[i].begin(), y0[i].begin() + size_t(rows) * gr.w[i].N).data(),
                        size_t(rows) * gr.w[i].N * 4, cudaMemcpyHostToDevice));
        };
        MtGroups g;
        for (size_t i = 0; i < gr.w.size(); ++i) {
          const W& w = gr.w[i];
          start(i, y[i]);
          g.g[g.n++] = MtOut{dw[i], w.b16 ? nullptr : y[i], w.b16 ? y16[i] : nullptr, w.N, w.N, w.x_col, int(w.acc), 0};
        }
        gemm_draft(stage_bf16(dx, gr.ldx, rows, gr.ldx, L), gr.ldx, gr.K, g, rows, L);
        for (size_t i = 0; i < gr.w.size(); ++i) {
          const W& w = gr.w[i];
          start(i, y1[i]);
          gemm(dx + w.x_col, gr.ldx, dw[i], w.N, gr.K, y1[i], w.N, rows, w.acc, LD);
          Vec alone = down(y1[i], size_t(rows) * w.N);
          if (w.b16)
            for (double& v : alone) v = __bfloat162float(__float2bfloat16_rn(float(v)));
          const Vec grouped = w.b16 ? down_bf16(y16[i], alone.size()) : down(y[i], alone.size());
          if (grouped != alone) {
            same = false;
            std::printf("       %s, weight %zu (%d x %d), table %s, %d rows: other bits than alone\n", gr.what, i, w.N,
                        gr.K, tunings[ti] < 0 ? "defaults" : gemm_tuning_name(tunings[ti]), rows);
          }
        }
      }
    }
    std::printf("  %-4s gemm_draft, %s (K %d): each weight the bits of a drafting pass's gemm() alone, %zu of %zu "
                "tables (those compiled for this GPU), 17, 32, 48 and 64 rows\n", same ? "ok" : "FAIL", gr.what, gr.K,
                tunings.size(), every);
    g_fail += !same;
    CK(cudaFree(dx));
    for (size_t i = 0; i < gr.w.size(); ++i)
      for (void* p : std::initializer_list<void*>{dw[i], y[i], y1[i], y16[i]}) CK(cudaFree(p));
  }
}

// The W4A4 prefill GEMM: activations quantized as the host does it (same
// codes, scales and row factors, bit for bit), the product against float64 of
// the quantized operands, and (reported) against the unquantized activations.
// The prefill gate|up GEMM with SwiGLU in its epilogue against the GEMM into
// float32 and swiglu_split: the same bits for bf16 and NVFP4 weights, every
// configuration, and W4A4 when the device has it; then the next GEMM on its
// rows against the same GEMM on swiglu_split's.
void test_gemm_glu(int M, int I, int K) {
  std::printf("SwiGLU in the gate|up GEMM (%d rows, I %d, K %d)\n", M, I, K);
  seed("gemm glu", M, I, K);
  const int N = 2 * I, H = 64;
  const Scratch& sc = *TL.scratch;
  Vec x = randv(size_t(M) * K);
  std::vector<uint16_t> xb(x.size());
  for (size_t i = 0; i < x.size(); ++i) xb[i] = to_bf16(x[i]);
  auto stage = [&] { CK(cudaMemcpy(sc.abuf, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice)); };
  bf16* dw = up_bf16(randv(size_t(N) * K, 0.1));
  bf16* dw2 = up_bf16(randv(size_t(H) * I, 0.1));  // the next GEMM: [H, I]
  std::vector<uint8_t> Qh(size_t(N) * K / 2), Sh(size_t(N) * K / 16);
  for (auto& b : Qh) b = uint8_t(rng());
  for (auto& b : Sh) b = uint8_t(0x28 + rng() % 0x18);
  Vec rsv(N);
  for (auto& v : rsv) v = 0.01 + 0.01 * double(rng() % 1000) / 1000.0;
  uint8_t *dq = dev<uint8_t>(Qh.size()), *ds = dev<uint8_t>(Sh.size());
  CK(cudaMemcpy(dq, Qh.data(), Qh.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ds, Sh.data(), Sh.size(), cudaMemcpyHostToDevice));
  float* drs = up(rsv);
  float *dy = dev<float>(size_t(M) * N), *dz1 = dev<float>(size_t(M) * H), *dz2 = dev<float>(size_t(M) * H);
  const size_t n = size_t(M) * I;
  auto run = [&](const char* what, bool fp4, int cfg, const Launch& L) {
    stage();
    gemm_glu_staged(fp4 ? nullptr : dw, fp4 ? dq : nullptr, ds, drs, I, K, M, L, cfg);
    const Vec fused = down_bf16(sc.glu, n);
    stage();
    if (fp4) gemm_fp4_presplit(dq, ds, drs, N, K, dy, N, M, false, L);
    else gemm_presplit(dw, N, K, dy, N, M, false, L);
    swiglu_split(dy, N, I, M, L);  // into abuf
    expect(std::string(what) + ": same bits as the GEMM and swiglu_split", fused, down_bf16(sc.abuf, n), 0);
    gemm_on_glu(dw2, nullptr, nullptr, nullptr, H, I, dz1, H, M, false, L);  // glu rows
    gemm_presplit(dw2, H, I, dz2, H, M, false, L);                          // abuf rows, the same values
    expect(std::string(what) + ", next GEMM: same bits", down(dz1, size_t(M) * H), down(dz2, size_t(M) * H), 0);
  };
  for (int c = 0; c < kTcConfigs; ++c) {
    run(("bf16 weights, configuration " + std::to_string(c)).c_str(), false, c, TL);
    run(("nvfp4 weights, configuration " + std::to_string(c)).c_str(), true, c, TL);
  }
  if (gemm_w4a4_supported() && K % 128 == 0) {
    static KernelOptions o;
    o.prefill_fp4 = true;
    Launch L4 = TL;
    L4.opt = &o;
    run("nvfp4 weights, W4A4", true, -1, L4);
    // NVFP4 inputs from float32 rows (a scratch with af32): the gate|up GEMM leaves bf16 SwiGLU rows and their
    // maxima; the same bits as the W4A4 GEMM into float32 and swiglu_split, every configuration; the next GEMM on
    // them as on swiglu_split's rows
    const size_t bytes = scratch_bytes(M, K, true, true);
    const Scratch s4 = scratch_in(dev<unsigned char>(bytes), bytes, M, K, true, true);
    L4.scratch = &s4;
    std::vector<float> xf(x.size());
    for (size_t i = 0; i < x.size(); ++i) xf[i] = float(x[i]);
    CK(cudaMemcpy(s4.af32, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice));
    quantize_rows_nvfp4(s4.af32, K, M, K, s4.q4, s4.q4_scale, s4.q4_inv, 0);
    gemm_w4a4(s4.q4, s4.q4_scale, s4.q4_inv, dq, ds, drs, N, K, dy, N, M, false, 0);
    swiglu_split(dy, N, I, M, TL);  // into TL's abuf
    const Vec want = down_bf16(sc.abuf, n);
    Vec want_max(M, 0.0);
    for (size_t i = 0; i < n; ++i) want_max[i / I] = std::max(want_max[i / I], std::fabs(want[i]));
    for (int cfg = -1; cfg < 5; ++cfg) {
      const std::string what = "W4A4 from float32 rows, " + (cfg < 0 ? std::string("gemm_glu_staged")
                                                                      : "configuration " + std::to_string(cfg));
      if (cfg < 0) gemm_glu_staged(nullptr, dq, ds, drs, I, K, M, L4);
      else gemm_w4a4(s4.q4, s4.q4_scale, s4.q4_inv, dq, ds, drs, N, K, nullptr, 0, M, false, 0, cfg, s4.glu,
                     s4.glu_amax);
      expect(what + ": SwiGLU rows, same bits as the GEMM and swiglu_split", down_bf16(s4.glu, n), want, 0);
      expect(what + ": each row's largest magnitude", down(s4.glu_amax, M), want_max, 0);
      if (cfg >= 0) continue;
      gemm_on_glu(dw2, nullptr, nullptr, nullptr, H, I, dz1, H, M, false, L4);  // glu rows, bf16 weights
      gemm_presplit(dw2, H, I, dz2, H, M, false, TL);                           // abuf rows, the same values
      expect(what + ", next GEMM: same bits", down(dz1, size_t(M) * H), down(dz2, size_t(M) * H), 0);
    }
  }
}

// A release build (FLYX_TUNE_CANDIDATES off) compiles a GEMM configuration only for the architectures whose tables
// reach it, as the generator works them out (tools/codegen/gen_gemm_tables.py, the kReach* masks); anywhere else the
// kernel traps.  So every configuration gemm_config gives (pick_config, what the GPU runs) under each table, for the
// shapes the tables list and shapes they do not, at every row count, must be in the masks of every architecture whose
// cubin runs that table: its own (sm_80's also runs the tables of sm_86 and sm_89); the defaults and a table of no
// architecture, all.  Host data alone, whatever GPU runs the test and whichever build.
void test_gemm_tables_compiled() {
  std::printf("every GEMM configuration a tuning table gives is compiled where the table runs\n");
  namespace t = tables;
  constexpr int kBucketCc[] = {80, 86, 89, 90, 100, 120};  // t::arch_bucket's, then kArchBucketAny
  auto runs = [&](int bucket, int arch) {                  // whether a cubin of `bucket` runs a table of `arch`
    if (bucket == t::kArchBucketAny || arch == 0) return true;
    return arch == kBucketCc[bucket] || (bucket == 0 && (arch == 86 || arch == 89));
  };
  std::set<std::pair<int, int>> shapes = {{12345, 4096}, {12345, 4160}, {333, 96}};  // shapes no table lists
  for (int i = 0; i < t::kRuleCount; ++i)
    if (t::kRules[i].sku >= 0) shapes.insert({t::kRules[i].n, t::kRules[i].k});
  long checked = 0, missing = 0;
  auto has = [&](unsigned mask, int cfg, const char* what, int tuning, int bucket, std::pair<int, int> s, int rows) {
    ++checked;
    if (cfg >= 0 && cfg < 32 && (mask >> cfg & 1u)) return;
    if (++missing <= 8)
      std::printf("       %s, table %s, the cubin of bucket %d, %d x %d, %d rows: configuration %d not compiled\n",
                  what, tuning < 0 ? "defaults" : t::kSkus[tuning].name, bucket, s.first, s.second, rows, cfg);
  };
  for (int tuning = -1; tuning < t::kSkuCount; ++tuning)
    for (int b = 0; b < t::kArchBucketCount; ++b) {
      if (!runs(b, tuning < 0 ? 0 : t::kSkuArch[tuning])) continue;
      for (const auto& s : shapes) {
        const auto [N, K] = s;
        for (int rows = 1; rows <= 16; ++rows) {
          const int hi = rows > 8;
          has(t::kReachBf16Skinny[b][hi], gemm_config(tuning, 0, N, K, rows), "bf16", tuning, b, s, rows);
          has(t::kReachW12Skinny[b][hi], gemm_config(tuning, 2, N, K, rows), "w12", tuning, b, s, rows);
          has(t::kReachFp4Skinny[b][hi], gemm_config(tuning, 1, N, K, rows), "nvfp4", tuning, b, s, rows);
        }
        // gemm_fp4_rows: a round's rows past 16 on the 16-row configuration, its last slice of 1..8 rows too
        has(t::kReachFp4Skinny[b][0], gemm_config(tuning, 1, N, K, 16), "nvfp4 slice", tuning, b, s, 16);
        for (int rows = 17; rows <= 64; ++rows) {
          has(t::kReachBf16Mt[b], gemm_config(tuning, 3, N, K, rows), "bf16 round", tuning, b, s, rows);
          has(t::kReachBf16Mt[b], gemm_config(tuning, 4, N, K, rows), "bf16 drafting pass", tuning, b, s, rows);
        }
      }
    }
  std::printf("  %-4s %d tables and the defaults, %zu shapes, rows 1..64: %ld configurations given, %ld not "
              "compiled\n", missing == 0 ? "ok" : "FAIL", t::kSkuCount, shapes.size(), checked, missing);
  g_fail += missing != 0;
}

}  // namespace flyx::optest
