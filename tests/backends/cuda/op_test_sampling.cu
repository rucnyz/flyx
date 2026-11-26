// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: token choice: penalties, sampling, argmax and top-k.
#include "ops/speculation/ref.h"
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

// A request's sampling as the device holds it (DevGen).
DevGen* gen_params(float T, float top_p, int top_k, uint64_t seed, bool mask = false) {
  DevGen g{};
  g.temperature = T;
  g.top_p = top_p;
  g.top_k = top_k;
  g.seed = seed;
  g.mask = mask;
  DevGen* d = dev<DevGen>(1);
  CK(cudaMemcpy(d, &g, sizeof(g), cudaMemcpyHostToDevice));
  return d;
}

// Penalties (ops/sampling/sampling.h): the logits penalize_rows leaves and the
// token chosen from them, against the float64 definition.  A reply with
// tokens chosen once and many times; a round's drafts, some of them tokens the
// reply has, one given three times (row r counts drafts[0..r)); the pass that
// chooses a reply's second token (no drafts); and a request without penalty,
// whose logits must come back bit for bit.
void test_penalize() {
  std::printf("penalties\n");
  seed("penalties");
  const int rows = 16, V = 50000;
  Vec lg = randv(size_t(rows) * V, 3.0);
  std::vector<int> reply;
  for (int i = 0; i < 400; ++i) reply.push_back(i % 9 == 0 ? 7 : int(ops::mix64(uint64_t(i) + 99) % 3000));
  std::vector<int> count(size_t(V), 0);
  std::vector<int> seen;
  for (int t : reply)
    if (count[size_t(t)]++ == 0) seen.push_back(t);
  // Every row's largest logit is a token the reply has, half a unit above the rest: the penalties have choices to
  // change whatever the random logits (else a row's best is one of the reply's few tokens 5% of the time).
  for (int r = 0; r < rows; ++r) {
    double* row = &lg[size_t(r) * V];
    row[seen[size_t(r) * 37 % seen.size()]] = *std::max_element(row, row + V) + 0.5;
  }
  const std::vector<int> drafts = {reply[1], 40001, 40001, 7, 40002, reply[5], 40003, 40001,
                                   40004, 7,     40005, 40006, reply[9], 40007, 40008};
  Reply rep{dev<int>(size_t(V)), dev<int>(seen.size()), dev<int>(1)};
  const int n_seen = static_cast<int>(seen.size());
  CK(cudaMemcpy(rep.count, count.data(), sizeof(int) * size_t(V), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rep.seen, seen.data(), sizeof(int) * seen.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rep.n_seen, &n_seen, sizeof(int), cudaMemcpyHostToDevice));
  int* ddrafts = dev<int>(drafts.size());
  CK(cudaMemcpy(ddrafts, drafts.data(), sizeof(int) * drafts.size(), cudaMemcpyHostToDevice));
  DevParams* P = params(100);
  float* cz = dev<float>(size_t(rows) * V);
  int* ci = dev<int>(size_t(rows) * V);
  float* sv = dev<float>(rows * 16);
  int* si = dev<int>(rows * 16);
  int* out = dev<int>(rows);
  struct Cfg {
    float presence, frequency, T;
    bool round;
  } cfgs[] = {{1.5f, 0.f, 0.f, true},   {0.f, 0.3f, 0.f, true},    {1.5f, 0.25f, 0.7f, true},
              {-0.5f, 0.1f, 1.f, true}, {1.5f, 0.25f, 0.f, false}, {2.f, 2.f, 0.9f, false}};
  double worst = 0;
  int mism = 0, total = 0, untouched_wrong = 0, changed = 0;
  for (const auto& c : cfgs) {
    DevGen dg{};
    dg.temperature = c.T;
    dg.top_p = 0.9f;
    dg.top_k = 40;
    dg.seed = 31337;
    dg.presence = c.presence;
    dg.frequency = c.frequency;
    DevGen* G = dev<DevGen>(1);
    CK(cudaMemcpy(G, &dg, sizeof(dg), cudaMemcpyHostToDevice));
    float* dl = up(lg);
    penalize_rows(dl, V, V, rows, G, rep, c.round ? ddrafts : nullptr, 0);
    choose_rows(dl, V, V, rows, P, 1, G, nullptr, cz, ci, sv, si, out, 0);
    const Vec got = down(dl, size_t(rows) * V);
    const auto chosen = down_i(out, rows);
    flyx_gen_params g{};
    g.temperature = c.T;
    g.top_p = 0.9f;
    g.top_k = 40;
    g.seed = 31337;
    g.presence_penalty = c.presence;
    g.frequency_penalty = c.frequency;
    std::vector<int> counts(count.begin(), count.end());
    for (int r = 0; r < rows; ++r) {
      if (c.round && r > 0) ++counts[size_t(drafts[size_t(r) - 1])];
      const double* row = &lg[size_t(r) * V];
      for (int v = 0; v < V; ++v) {
        const double raw = double(float(row[v]));
        if (counts[size_t(v)] == 0) {
          untouched_wrong += got[size_t(r) * V + v] != raw;
          continue;
        }
        const double want = raw - ops::penalty(double(c.presence), double(c.frequency), counts[size_t(v)]);
        worst = std::max(worst, std::fabs(got[size_t(r) * V + v] - want) / std::max(1.0, std::fabs(want)));
      }
      std::vector<double> frow(row, row + V);
      for (auto& x : frow) x = double(float(x));
      const int want_token = ops::sample_token(frow.data(), V, g, 101 + r, nullptr, counts.data());
      flyx_gen_params plain = g;
      plain.presence_penalty = plain.frequency_penalty = 0;
      changed += want_token != ops::sample_token(frow.data(), V, plain, 101 + r);
      mism += chosen[size_t(r)] != want_token;
      ++total;
    }
    CK(cudaFree(dl));
    CK(cudaFree(G));
  }
  bool ok = worst <= 1e-6 && untouched_wrong == 0;  // float32: the penalty's product and one subtraction
  std::printf("  %-4s penalized logits == float64 definition (rel err %.1e; %d logits of tokens not in the reply changed)\n",
              ok ? "ok" : "FAIL", worst, untouched_wrong);
  g_fail += !ok;
  ok = mism * 50 <= total && changed > 0;  // float32 vs float64 near-ties only
  std::printf("  %-4s choice from them == float64 reference (%d of %d differ; the penalty changed %d choices)\n",
              ok ? "ok" : "FAIL", mism, total, changed);
  g_fail += !ok;
  {  // no penalty: nothing is written
    DevGen dg{};
    DevGen* G = dev<DevGen>(1);
    CK(cudaMemcpy(G, &dg, sizeof(dg), cudaMemcpyHostToDevice));
    float* dl = up(lg);
    penalize_rows(dl, V, V, rows, G, rep, ddrafts, 0);
    const Vec got = down(dl, size_t(rows) * V);
    bool same = true;
    for (size_t i = 0; i < got.size(); ++i) same = same && got[i] == double(float(lg[i]));
    std::printf("  %-4s a request without penalty: the logits bit for bit\n", same ? "ok" : "FAIL");
    g_fail += !same;
  }
  {  // The same 65,536th occurrence, as a draft before row 1 and then as a committed token before row 0.
    constexpr int token = 7;
    count[token] = 65535;
    CK(cudaMemcpy(rep.count + token, &count[token], sizeof(count[token]), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(out, &token, sizeof(token), cudaMemcpyHostToDevice));
    DevGen dg{};
    dg.frequency = 0.25f;
    DevGen* G = dev<DevGen>(1);
    CK(cudaMemcpy(G, &dg, sizeof(dg), cudaMemcpyHostToDevice));
    float* logits = dev<float>(3 * V);
    penalize_rows(logits, V, V, 2, G, rep, out, 0);
    prefill_done(P, 1, out, 1, dev<flyx_round_result>(kRing), G, rep, 0);
    penalize_rows(logits + 2 * V, V, V, 1, G, rep, nullptr, 0);
    const double want = -ops::penalty(0., double(dg.frequency), 65536);
    expect("draft and committed penalties after 65,535 repeats",
           {down(logits + V + token, 1)[0], down(logits + 2 * V + token, 1)[0]}, {want, want}, 0);
  }
}

// A wide round's head chooses 16 rows at a time (regions/layers.cpp, head): the penalties, the choice, the relaxed
// rules' statistics on rows a .. a + 15 of the round (tree_from), and each draft's logit in its parent's row gathered
// (child_logits).  Against one call over all 64 rows, bit for bit: a tree's rows at their depths, its penalties its
// ancestors' drafts, the grammar mask on the round's first row, greedy and sampled; and a chain's.
void test_wide_rows_choice() {
  std::printf("a wide round's choice 16 rows at a time\n");
  seed("wide rows choice");
  const int rows = FLYX_MAX_WIDE_ROWS, width = rows - 1, V = 6000;
  const Vec lg = randv(size_t(rows) * V, 3.0);
  std::vector<int> drafts(static_cast<size_t>(width)), count(static_cast<size_t>(V), 0), seen;
  for (int i = 0; i < width; ++i) drafts[size_t(i)] = int(ops::mix64(uint64_t(i) + 7) % 40);  // repeats among them
  for (int i = 0; i < 300; ++i) {
    const int tok = int(ops::mix64(uint64_t(i) + 1234) % 60);
    if (count[size_t(tok)]++ == 0) seen.push_back(tok);
  }
  Reply rep{dev<int>(size_t(V)), dev<int>(seen.size()), dev<int>(1)};
  const int n_seen = static_cast<int>(seen.size());
  CK(cudaMemcpy(rep.count, count.data(), sizeof(int) * size_t(V), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rep.seen, seen.data(), sizeof(int) * seen.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(rep.n_seen, &n_seen, sizeof(int), cudaMemcpyHostToDevice));
  int* ddrafts = dev<int>(drafts.size());
  CK(cudaMemcpy(ddrafts, drafts.data(), sizeof(int) * drafts.size(), cudaMemcpyHostToDevice));
  TreeRows ht{};  // a random tree: each draft continues one of the rows before it
  ht.n = rows;
  std::vector<int> parent(size_t(rows), -1);
  for (int r = 1; r < rows; ++r) {
    const int pr = int(ops::mix64(uint64_t(r) * 31 + 5) % uint64_t(r));
    parent[size_t(r)] = pr;
    ht.depth[r] = ht.depth[pr] + 1;
    for (int d = 0; d <= ht.depth[pr]; ++d) ht.path[r][d] = ht.path[pr][d];
    ht.path[r][ht.depth[r]] = static_cast<unsigned char>(r);
  }
  TreeRows* dtree = dev<TreeRows>(1);
  CK(cudaMemcpy(dtree, &ht, sizeof(ht), cudaMemcpyHostToDevice));
  std::vector<uint32_t> mask(size_t((V + 31) / 32), 0);
  for (int v = 0; v < V; v += 3) mask[size_t(v / 32)] |= 1u << (v % 32);
  uint32_t* dmask = dev<uint32_t>(mask.size());
  CK(cudaMemcpy(dmask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));
  DevParams* P = params(100);
  float* cz = dev<float>(size_t(rows) * V);
  int* ci = dev<int>(size_t(rows) * V);
  float *sv = dev<float>(size_t(rows) * 16), *stats = dev<float>(3 * size_t(rows)), *stats16 = dev<float>(3 * size_t(rows));
  int *si = dev<int>(size_t(rows) * 16), *out = dev<int>(rows), *out16 = dev<int>(rows);
  float* z = dev<float>(size_t(width));
  struct Cfg {
    const char* what;
    float T, presence, frequency;
    bool tree, masked;
  } cfgs[] = {{"greedy, penalties, a tree", 0.f, 1.5f, 0.25f, true, false},
              {"sampled, penalties, a tree", 0.9f, 0.5f, 0.3f, true, false},
              {"sampled, the grammar's mask, a tree", 1.f, 0.f, 0.f, true, true},
              {"sampled, penalties, a chain", 0.8f, 1.5f, 0.25f, false, false}};
  for (const auto& c : cfgs) {
    DevGen dg{};
    dg.temperature = c.T;
    dg.top_p = 0.95f;
    dg.top_k = 50;
    dg.seed = 777;
    dg.presence = c.presence;
    dg.frequency = c.frequency;
    dg.accept_min_p = 0.2f;
    dg.mask = c.masked;
    DevGen* G = dev<DevGen>(1);
    CK(cudaMemcpy(G, &dg, sizeof(dg), cudaMemcpyHostToDevice));
    const TreeRows* tr = c.tree ? dtree : nullptr;
    float *all = up(lg), *part = up(lg);
    penalize_rows(all, V, V, rows, G, rep, ddrafts, 0, tr);
    choose_rows(all, V, V, rows, P, 1, G, c.masked ? dmask : nullptr, cz, ci, sv, si, out, 0, tr);
    row_stats(all, V, V, rows, G, stats, 0);
    for (int a = 0; a < rows; a += FLYX_MAX_ROWS) {
      const int k = std::min(FLYX_MAX_ROWS, rows - a);
      float* lp = part + size_t(a) * V;
      penalize_rows(lp, V, V, k, G, rep, ddrafts, 0, tr, a);
      choose_rows(lp, V, V, k, P, a + 1, G, c.masked && a == 0 ? dmask : nullptr, cz, ci, sv, si, out16 + a, 0, tr,
                  a);
      row_stats(lp, V, V, k, G, stats16 + 3 * a, 0);
      child_logits(lp, V, a, k, ddrafts, width, tr, z, 0);
    }
    const Vec la = down(all, size_t(rows) * V), lb = down(part, size_t(rows) * V);
    const Vec sa = down(stats, 3 * size_t(rows)), sb = down(stats16, 3 * size_t(rows)), zg = down(z, size_t(width));
    bool gathered = true;
    for (int cc = 1; cc <= width; ++cc) {
      const int at = c.tree ? parent[size_t(cc)] : cc - 1;
      gathered = gathered && zg[size_t(cc - 1)] == la[size_t(at) * V + size_t(drafts[size_t(cc - 1)])];
    }
    const bool same = la == lb && down_i(out, rows) == down_i(out16, rows) && sa == sb && gathered;
    std::printf("  %-4s %s: penalties, choices, statistics and gathered logits those of all %d rows at once\n",
                same ? "ok" : "FAIL", c.what, rows);
    g_fail += !same;
    CK(cudaFree(all));
    CK(cudaFree(part));
    CK(cudaFree(G));
  }
}

// Token choice (ops/sampling/sampling.h): the kernel against the float64 reference, and
// the empirical distribution against the filtered softmax.
void test_sample() {
  std::printf("sampling\n");
  seed("sampling");
  struct Cfg {
    float T, top_p;
    int top_k;
  } cfgs[] = {{0.f, 1.f, 0}, {1.f, 1.f, 0}, {0.7f, 0.9f, 0}, {0.6f, 1.f, 20}, {1.2f, 0.8f, 40}, {0.3f, 0.95f, 5}};
  const int rows = 16, V = 50000;
  Vec lg = randv(size_t(rows) * V, 3.0);
  float* dl = up(lg);
  DevParams* P = params(100);
  float* cz = dev<float>(size_t(rows) * V);
  int* ci = dev<int>(size_t(rows) * V);
  float* sv = dev<float>(rows * 16);
  int* si = dev<int>(rows * 16);
  int* out = dev<int>(rows);
  int mism = 0, total = 0;
  {
    const Vec logits = {-0., 0., -1., -0., 0., -1.};
    float* tied = up(logits);
    for (const Cfg filter : {Cfg{1.f, 1.f, 1}, Cfg{1.f, 0.3f, 0}}) {
      flyx_gen_params g{};
      g.temperature = filter.T;
      g.top_p = filter.top_p;
      g.top_k = filter.top_k;
      g.seed = 1;
      DevGen* G = gen_params(g.temperature, g.top_p, g.top_k, g.seed);
      const int ref = ops::sample_token(logits.data(), 3, g, 101);
      choose_rows(tied, 3, 3, 1, P, 1, G, nullptr, cz, ci, sv, si, out, 0);
      const int plain = down_i(out, 1)[0];
      choose_rows(tied, 3, 3, 2, P, 0, G, nullptr, cz, ci, sv, si, out, 0);
      const int verify = down_i(out, 2)[1];
      const bool ok = ref == 0 && plain == ref && verify == ref;
      std::printf("  %-4s %s retains signed-zero ties: reference %d, plain %d, round %d\n", ok ? "ok" : "FAIL",
                  g.top_k ? "top-k" : "top-p", ref, plain, verify);
      g_fail += !ok;
    }
  }
  for (float T : {1e-40f, 1e-38f}) {
    flyx_gen_params g{};
    g.temperature = T;
    g.top_p = 1.f;
    g.seed = 7;
    DevGen dg{};
    dg.temperature = g.temperature;
    dg.top_p = g.top_p;
    dg.seed = g.seed;
    dg.mask = 1;
    dg.accept_min_p = 1.f;
    dg.max_ctx = 4096;
    DevGen* G = dev<DevGen>(1);
    CK(cudaMemcpy(G, &dg, sizeof(dg), cudaMemcpyHostToDevice));
    const Vec logits = {-2., 0., 4., -2. * T, -double(T), 0.};
    float* tiny = up(logits);
    const uint32_t mask = 3;  // row 0: neither allowed token is in the filtered set
    uint32_t* dmask = dev<uint32_t>(1);
    CK(cudaMemcpy(dmask, &mask, sizeof(mask), cudaMemcpyHostToDevice));
    for (const uint32_t* mk : {static_cast<const uint32_t*>(nullptr), &mask}) {
      choose_rows(tiny, 3, 3, 2, P, 1, G, mk ? dmask : nullptr, cz, ci, sv, si, out, 0);
      const auto got = down_i(out, 2);
      const bool ok = got[0] == ops::sample_token(logits.data(), 3, g, 101, mk) &&
                      got[1] == ops::sample_token(logits.data() + 3, 3, g, 102);
      std::printf("  %-4s T %.2e, %s: tokens %d, %d\n", ok ? "ok" : "FAIL", T, mk ? "mask" : "unmasked",
                  got[0], got[1]);
      g_fail += !ok;
    }
    const Vec tied = {0., 0., 4.};
    choose_rows(up(tied), 3, 3, 1, P, 1, G, dmask, cz, ci, sv, si, out, 0);
    const int tie = ops::sample_token(tied.data(), 2, g, 101);
    const bool same_tie = tie == 1 && ops::sample_token(tied.data(), 3, g, 101, &mask) == tie && down_i(out, 1)[0] == tie;
    std::printf("  %-4s T %.2e: fallback preserves sampling among tied allowed tokens\n", same_tie ? "ok" : "FAIL", T);
    g_fail += !same_tie;
    float* stats = dev<float>(6);
    row_stats(tiny, 3, 3, 2, G, stats, 0);
    const double a = std::exp(-1.), b = std::exp(-2.), z = 1. + a + b;
    expect("cold distributions: maximum, normalizer and entropy", down(stats, 6),
           {4., 1., 0., 0., z, std::log(z) + (a + 2. * b) / z}, 1e-6);
    // A maximum-probability draft qualifies for min-p 1 even if not the target's sampled token.
    const int ids[] = {2, 1, 0};
    CK(cudaMemcpy(out, ids, sizeof(ids), cudaMemcpyHostToDevice));
    auto* ring = dev<flyx_round_result>(kRing);
    accept_tree(params(100), G, out, out + 1, 1, ring, {}, nullptr, tiny + 3, 3, stats + 3, 0);
    flyx_round_result result{};
    CK(cudaMemcpy(&result, ring + 1, sizeof(result), cudaMemcpyDeviceToHost));
    const bool accepted = result.n_accepted == 1 && result.tokens[0] == 2;
    std::printf("  %-4s T %.2e: min-p 1 accepts the highest-logit draft\n", accepted ? "ok" : "FAIL", T);
    g_fail += !accepted;
  }
  {  // This seed gives token 0 the highest 24-bit sample at position 101. Its noise must still be finite.
    flyx_gen_params g{};
    g.temperature = g.top_p = 1.f;
    g.seed = 0x058e908a45e19308ULL;
    const Vec logits = {-25., 0., -25., 0.};
    float* tiny = up(logits);
    DevGen* G = gen_params(g.temperature, g.top_p, 0, g.seed);
    choose_rows(tiny, 2, 2, 1, P, 1, G, nullptr, cz, ci, sv, si, out, 0);
    const int plain = down_i(out, 1)[0];
    choose_rows(tiny, 2, 2, 2, P, 0, G, nullptr, cz, ci, sv, si, out, 0);
    const int round = down_i(out, 2)[1], ref = ops::sample_token(logits.data(), 2, g, 101);
    const bool finite = ops::sample_uniform(g.seed, 101, 0) == std::nextafter(1.f, 0.f);
    const bool ok = finite && ref == 1 && plain == ref && round == ref;
    std::printf("  %-4s largest sampling draw: reference %d, plain %d, round %d\n", ok ? "ok" : "FAIL", ref, plain, round);
    g_fail += !ok;
  }
  for (const auto& c : cfgs)
    for (uint64_t seed : {1ull, 77ull, 123456789ull}) {
      choose_rows(dl, V, V, rows, P, 1, gen_params(c.T, c.top_p, c.top_k, seed), nullptr, cz, ci, sv, si, out, 0);
      auto got = down_i(out, rows);
      flyx_gen_params g{};
      g.temperature = c.T;
      g.top_p = c.top_p;
      g.top_k = c.top_k;
      g.seed = seed;
      for (int r = 0; r < rows; ++r, ++total) mism += got[r] != ops::sample_token(&lg[size_t(r) * V], V, g, 101 + r);
    }
  bool ok = mism * 50 <= total;  // float32 vs float64 near-ties only
  std::printf("  %-4s kernel == float64 reference (%d of %d differ)\n", ok ? "ok" : "FAIL", mism, total);
  g_fail += !ok;

  // A mask on row 0 (grammar-constrained decoding): the kernel against the
  // reference, for a mask that keeps a quarter of the tokens and one that
  // allows only tokens outside every filter's kept set (the fallback).  The
  // other rows are not masked.  And the property core/grammar relies on: when
  // the unmasked choice is allowed, the masked choice is that token.
  {
    const int words = (V + 31) / 32;
    std::vector<uint32_t> quarter(size_t(words), 0), tail(size_t(words), 0);
    for (int v = 0; v < V; ++v)
      if (ops::mix64(uint64_t(v) * 7919u) % 4 == 0) quarter[size_t(v >> 5)] |= 1u << (v & 31);
    {  // the 8 lowest logits of row 0: no filter keeps them
      std::vector<int> order(V);
      for (int v = 0; v < V; ++v) order[size_t(v)] = v;
      std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return lg[size_t(a)] < lg[size_t(b)]; });
      for (int i = 0; i < 8; ++i) tail[size_t(order[size_t(i)] >> 5)] |= 1u << (order[size_t(i)] & 31);
    }
    int mm = 0, n = 0, same_when_allowed = 0, allowed_cases = 0;
    for (const auto* mask : {&quarter, &tail}) {
      uint32_t* dmask = dev<uint32_t>(size_t(words));
      CK(cudaMemcpy(dmask, mask->data(), sizeof(uint32_t) * size_t(words), cudaMemcpyHostToDevice));
      for (const auto& c : cfgs)
        for (uint64_t seed : {1ull, 77ull, 123456789ull}) {
          choose_rows(dl, V, V, rows, P, 1, gen_params(c.T, c.top_p, c.top_k, seed, true), dmask, cz, ci, sv, si, out, 0);
          auto got = down_i(out, rows);
          flyx_gen_params g{};
          g.temperature = c.T;
          g.top_p = c.top_p;
          g.top_k = c.top_k;
          g.seed = seed;
          for (int r = 0; r < rows; ++r, ++n)
            mm += got[size_t(r)] != ops::sample_token(&lg[size_t(r) * V], V, g, 101 + r, r == 0 ? mask->data() : nullptr);
          for (int r = 0; r < rows; ++r) {  // reference only: every row as if it were masked
            const int free = ops::sample_token(&lg[size_t(r) * V], V, g, 101 + r);
            if (!ops::allowed(mask->data(), free)) continue;
            ++allowed_cases;
            same_when_allowed += ops::sample_token(&lg[size_t(r) * V], V, g, 101 + r, mask->data()) == free;
          }
          if (!ops::allowed(mask->data(), got[0])) mm += 1000;  // never a disallowed token
        }
    }
    ok = mm * 50 <= n && same_when_allowed == allowed_cases;
    std::printf("  %-4s masked row 0 == float64 reference (%d of %d differ); unmasked choice kept when allowed (%d of %d)\n",
                ok ? "ok" : "FAIL", mm, n, same_when_allowed, allowed_cases);
    g_fail += !ok;
  }

  // Distribution over many positions of one small row.
  const int N = 20000, W = 12;
  Vec row = randv(W, 1.5);
  Vec many(size_t(N) * W);
  for (int r = 0; r < N; ++r) std::copy(row.begin(), row.end(), many.begin() + size_t(r) * W);
  float* dm = up(many);
  float* cz2 = dev<float>(size_t(N) * W);
  int* ci2 = dev<int>(size_t(N) * W);
  int* out2 = dev<int>(N);
  for (const auto& c : cfgs) {
    if (c.T <= 0) continue;
    choose_rows(dm, W, W, N, P, 1, gen_params(c.T, c.top_p, c.top_k, 2024), nullptr, cz2, ci2, sv, si, out2, 0);
    auto got = down_i(out2, N);
    // expected: softmax(row / T), top-k, then top-p, renormalised
    std::vector<std::pair<double, int>> z;
    double m = *std::max_element(row.begin(), row.end());
    for (int v = 0; v < W; ++v) z.push_back({(row[v] - m) / c.T, v});
    std::sort(z.begin(), z.end(), [](auto& a, auto& b) { return a.first > b.first; });
    size_t keep = c.top_k > 0 ? std::min<size_t>(c.top_k, W) : W;
    double tot = 0, cum = 0;
    for (size_t i = 0; i < keep; ++i) tot += std::exp(z[i].first);
    for (size_t i = 0; i < keep; ++i) {
      cum += std::exp(z[i].first);
      if (cum >= c.top_p * tot) {
        keep = i + 1;
        break;
      }
    }
    double kept = 0;
    for (size_t i = 0; i < keep; ++i) kept += std::exp(z[i].first);
    std::vector<double> want(W, 0.0), freq(W, 0.0);
    for (size_t i = 0; i < keep; ++i) want[z[i].second] = std::exp(z[i].first) / kept;
    for (int t : got) freq[t] += 1.0 / N;
    double worst = 0;
    for (int v = 0; v < W; ++v) worst = std::max(worst, std::fabs(freq[v] - want[v]));
    bool okd = worst < 0.015;
    std::printf("  %-4s distribution T=%.1f top_p=%.2f top_k=%d (max |freq - p| = %.4f)\n", okd ? "ok" : "FAIL", c.T,
                c.top_p, c.top_k, worst);
    g_fail += !okd;
  }
}

void test_select() {
  std::printf("argmax, top-k, candidate walk\n");
  seed("argmax");
  const int rows = 3, V = 20000, k = 16, rank = 32, width = 2;
  Vec lg = randv(size_t(rows) * V);
  lg[5] = lg[17] = 9.0;  // a tie: lower index wins
  float* dl = up(lg);
  int *di = dev<int>(rows), *ai = dev<int>(rows * 8), *ti = dev<int>(rows * k);
  float *av = dev<float>(rows * 8), *tv = dev<float>(rows * k);
  argmax_rows(dl, V, V, di, rows, av, ai, 0);
  topk_rows(dl, V, V, k, ti, tv, rows, 0);
  auto got = down_i(di, rows), tgot = down_i(ti, rows * k);
  Vec tval = down(tv, rows * k);
  bool ok = true;
  std::vector<int> want_idx(rows * k);
  Vec want_val(rows * k);
  for (int r = 0; r < rows; ++r) {
    ok &= got[r] == ops::argmax(&lg[size_t(r) * V], V);
    ops::topk(&lg[size_t(r) * V], V, k, &want_idx[r * k], &want_val[r * k]);
  }
  std::printf("  %-4s argmax (ties to the lower index)\n", ok ? "ok" : "FAIL");
  g_fail += !ok;
  std::printf("  %-4s top-%d indices\n", tgot == want_idx ? "ok" : "FAIL", k);
  g_fail += tgot != want_idx;
  {  // the parallel slices and their merge: one block a row's top-k, ties included (a tie across three slices too)
    const int V2 = 248320, k2 = 16;
    Vec l2 = randv(size_t(rows) * V2);
    l2[3] = l2[V2 / 2] = l2[V2 - 1] = 11.0;
    float* d2 = up(l2);
    int *i1 = dev<int>(rows * k2), *i2 = dev<int>(rows * k2), *pi = dev<int>(rows * kTopkParts * 16);
    float *v1 = dev<float>(rows * k2), *v2 = dev<float>(rows * k2), *pv = dev<float>(rows * kTopkParts * 16);
    topk_rows(d2, V2, V2, k2, i1, v1, rows, 0);
    topk_rows(d2, V2, V2, k2, i2, v2, rows, 0, pv, pi, rows);
    const bool same = down_i(i1, rows * k2) == down_i(i2, rows * k2) && down(v1, rows * k2) == down(v2, rows * k2);
    std::printf("  %-4s top-k over slices: one block a row's result\n", same ? "ok" : "FAIL");
    g_fail += !same;
    // A row with no finite score (a drafter's NaN rows): valid ids, 0..k-1, from both kernels (argmax_rows gives 0).
    Vec nan_rows(size_t(rows) * V2, std::nan(""));
    float* dn = up(nan_rows);
    int *n1 = dev<int>(rows * k2), *n2 = dev<int>(rows * k2), *npi = dev<int>(rows * kTopkParts * 16);
    float *nv1 = dev<float>(rows * k2), *nv2 = dev<float>(rows * k2), *npv = dev<float>(rows * kTopkParts * 16);
    topk_rows(dn, V2, V2, k2, n1, nv1, rows, 0);
    topk_rows(dn, V2, V2, k2, n2, nv2, rows, 0, npv, npi, rows);
    bool valid = true;
    for (auto* ids : {&n1, &n2}) {
      const auto got_ids = down_i(*ids, rows * k2);
      for (int j = 0; j < rows * k2; ++j) valid &= got_ids[size_t(j)] == j % k2;
    }
    std::printf("  %-4s top-k of a row without a finite score: ids 0..k-1, both kernels\n", valid ? "ok" : "FAIL");
    g_fail += !valid;
    for (void* p : std::initializer_list<void*>{dn, n1, n2, npi, nv1, nv2, npv}) CK(cudaFree(p));
    // topk_rows confidence outputs (prob & margin) for full-head drafters (both k=1/val=null merge and small-V exact).
    int* i_one = dev<int>(rows);
    float *dp = dev<float>(rows), *dm = dev<float>(rows);
    topk_rows(d2, V2, V2, 1, i_one, nullptr, rows, 0, pv, pi, rows, dp, dm);
    const auto got_i1 = down_i(i_one, rows);
    const Vec got_p = down(dp, rows), got_m = down(dm, rows), got_v2 = down(v2, rows * k2);
    const auto got_i2 = down_i(i2, rows * k2);
    Vec want_m(static_cast<size_t>(rows));
    bool k1_ok = true;
    for (int r = 0; r < rows; ++r) {
      k1_ok &= got_i1[size_t(r)] == got_i2[size_t(r) * k2] && got_p[size_t(r)] > 0.0 && got_p[size_t(r)] <= 1.0 + 1e-6;
      want_m[size_t(r)] = got_v2[size_t(r) * k2] - got_v2[size_t(r) * k2 + 1];
    }
    std::printf("  %-4s topk_rows k=1 (val=null) argmax & prob bounds\n", k1_ok ? "ok" : "FAIL");
    g_fail += !k1_ok;
    expect("topk_rows merge top-1/top-2 margin", got_m, want_m, 1e-5);
    // Small-V exact fallback (k_topk): verify exact full-vocab softmax p_max and margin against float64.
    const int Vs = 128;
    Vec ls = randv(size_t(rows) * Vs, 2.0);
    float* ds = up(ls);
    topk_rows(ds, Vs, Vs, 1, i_one, nullptr, rows, 0, nullptr, nullptr, 0, dp, dm);
    const Vec gp_s = down(dp, rows), gm_s = down(dm, rows);
    Vec wp_s(static_cast<size_t>(rows)), wm_s(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
      int ti2[2];
      double tv2[2];
      ops::topk(&ls[size_t(r) * Vs], Vs, 2, ti2, tv2);
      wm_s[size_t(r)] = tv2[0] - tv2[1];
      double z = 0.0;
      for (int v = 0; v < Vs; ++v) z += std::exp(ls[size_t(r) * Vs + v] - tv2[0]);
      wp_s[size_t(r)] = 1.0 / z;
    }
    expect("topk_rows exact softmax p_max (small V)", gp_s, wp_s, 1e-5);
    expect("topk_rows exact margin (small V)", gm_s, wm_s, 1e-5);
  }
  // walk over rows 0..width-1 of the top-k
  Vec pred = randv(size_t(V) * rank), succ = randv(size_t(V) * rank), hid = randv(size_t(rows) * rank);
  round_bf16(pred);
  round_bf16(succ);
  DevParams* P = params(10);
  DevParams hp{};
  hp.position = 10;
  hp.anchor = 1234;
  CK(cudaMemcpy(P, &hp, sizeof(hp), cudaMemcpyHostToDevice));
  int* dr = dev<int>(16);
  selector_walk(P, ti, tv, k, up(hid), rank, up_bf16(pred), up_bf16(succ), width, nullptr, dr, 0);
  auto drafts = down_i(dr, width);
  int prev = 1234;
  std::vector<int> want;
  for (int s = 0; s < width; ++s) {
    int best = 0;
    double bs = -1e300;
    for (int c = 0; c < k; ++c) {
      double e = 0;
      for (int i = 0; i < rank; ++i)
        e += pred[size_t(prev) * rank + i] * hid[size_t(s) * rank + i] * succ[size_t(want_idx[s * k + c]) * rank + i];
      double sc = want_val[s * k + c] + e;
      if (sc > bs) {
        bs = sc;
        best = c;
      }
    }
    prev = want_idx[s * k + best];
    want.push_back(prev);
  }
  std::printf("  %-4s candidate walk\n", drafts == want ? "ok" : "FAIL");
  g_fail += drafts != want;
  // The candidates as a tree (ops/draft/ref.h): drafts, parents, and the TreeRows the round reads; its trunk is the
  // walk's chain.  A tree wider than the positions (it must branch), one that fills a whole round, and one of 5 over
  // 2 positions.
  auto score = [&](int32_t prev, int d, int c) {
    double e = 0;
    for (int i = 0; i < rank; ++i)
      e += pred[size_t(prev) * rank + i] * hid[size_t(d) * rank + i] * succ[size_t(want_idx[d * k + c]) * rank + i];
    return want_val[d * k + c] + e;
  };
  TreeRows* dt = dev<TreeRows>(1);
  for (const auto& [tw, depths] : {std::pair{11, rows}, std::pair{FLYX_MAX_ROWS - 1, rows}, std::pair{5, 2}}) {
    selector_tree(P, ti, tv, k, up(hid), rank, up_bf16(pred), up_bf16(succ), tw, depths, nullptr, dr, dt, 0);
    TreeRows t{};
    CK(cudaMemcpy(&t, dt, sizeof(t), cudaMemcpyDeviceToHost));
    const auto got_drafts = down_i(dr, size_t(tw));
    std::vector<int32_t> wd(static_cast<size_t>(tw)), wp(static_cast<size_t>(tw));
    ops::best_first_tree(tw, depths, k, 1234, want_idx.data(), score, wd.data(), wp.data());
    bool same = t.n == tw + 1 && std::vector<int32_t>(got_drafts.begin(), got_drafts.end()) == wd;
    for (int i = 0; i < width; ++i) same = same && wd[size_t(i)] == want[size_t(i)] && wp[size_t(i)] == i - 1;  // trunk
    int deepest = 0, branches = 0;
    for (int i = 0; i < tw && same; ++i) {  // row i + 1's depth and path follow its parent's
      const int r = i + 1, p = wp[size_t(i)] + 1;
      same = t.depth[r] == t.depth[p] + 1 && t.path[r][t.depth[r]] == r;
      for (int d = 0; same && d <= t.depth[p]; ++d) same = t.path[r][d] == t.path[p][d];
      deepest = std::max(deepest, t.depth[r]);
      branches += p != r - 1;
    }
    same = same && deepest <= depths;
    std::printf("  %-4s candidate tree of %d drafts over %d positions (depth %d, %d off the chain)\n",
                same ? "ok" : "FAIL", tw, depths, deepest, branches);
    g_fail += !same;
  }
}

void test_stop_decide() {
  std::printf("stop_decide (sequential & block confidence stopping)\n");
  flyx_stop_policy pol{};
  pol.enabled = 1;
  pol.n_buckets = 5;
  const int ws[5] = {1, 3, 5, 7, 12};
  const float cs[5] = {17.5f, 20.7f, 23.8f, 26.8f, 34.5f};
  for (int i = 0; i < 5; ++i) {
    pol.widths[i] = ws[i];
    pol.costs[i] = cs[i];
  }
  for (int k = 0; k < FLYX_MAX_ROWS; ++k) pol.p_pos[k] = k < 7 ? 0.80f : 0.72f;
  pol.w_bias = -0.35f;
  pol.w_margin = 0.27f;
  pol.w_logit_p = 0.20f;
  pol.lambda = 0.165f;

  flyx_stop_policy* dpol = dev<flyx_stop_policy>(1);
  DevStopState* dst = dev<DevStopState>(1);
  float* dprob = dev<float>(FLYX_MAX_ROWS);
  float* dmargin = dev<float>(FLYX_MAX_ROWS);

  seed("stop_decide");
  std::uniform_real_distribution<double> u_prob(0.15, 0.9995), u_margin(0.05, 16.0);

  bool step_ok = true, block_ok = true;
  Vec got_s, want_s;
  for (int trial = 0; trial < 32; ++trial) {
    flyx_stop_policy cur = pol;
    if (trial == 0) cur.enabled = 0;
    CK(cudaMemcpy(dpol, &cur, sizeof(cur), cudaMemcpyHostToDevice));
    std::vector<float> hp(FLYX_MAX_ROWS), hm(FLYX_MAX_ROWS);
    std::vector<double> dp(FLYX_MAX_ROWS), dm(FLYX_MAX_ROWS);
    for (int k = 0; k < FLYX_MAX_ROWS; ++k) {
      if (trial == 1) {
        hp[size_t(k)] = 0.f;
        hm[size_t(k)] = 0.f;
      } else {
        hp[size_t(k)] = float(u_prob(rng));
        hm[size_t(k)] = float(u_margin(rng));
      }
      dp[size_t(k)] = hp[size_t(k)];
      dm[size_t(k)] = hm[size_t(k)];
    }
    CK(cudaMemcpy(dprob, hp.data(), sizeof(float) * FLYX_MAX_ROWS, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmargin, hm.data(), sizeof(float) * FLYX_MAX_ROWS, cudaMemcpyHostToDevice));

    double ref_s = 1.0;
    for (int b = 0; b < cur.n_buckets; ++b) {
      const int w0 = b == 0 ? 0 : cur.widths[b - 1];
      const int w1 = cur.widths[b];
      const int want_c = ops::stop_decide_step_ref(dp.data(), dm.data(), cur, b, w0, w1, &ref_s);
      stop_decide_step(dprob, dmargin, dpol, dst, b, w0, w1, 0, 0, 0);
      DevStopState got{};
      CK(cudaMemcpy(&got, dst, sizeof(got), cudaMemcpyDeviceToHost));
      got_s.push_back(double(got.survival));
      want_s.push_back(ref_s);
      step_ok = step_ok && (got.decision == want_c);
    }

    const int want_b = ops::stop_decide_block_ref(dp.data(), dm.data(), cur);
    StopCondHandles zero_h{};
    stop_decide_block(dprob, dmargin, dpol, dst, zero_h, 0);
    DevStopState got_b{};
    CK(cudaMemcpy(&got_b, dst, sizeof(got_b), cudaMemcpyDeviceToHost));
    block_ok = block_ok && (got_b.decision == want_b);
  }
  expect("survival probability against float64", got_s, want_s, 1e-5);
  std::printf("  %-4s sequential stop_decide_step matches float64 reference\n", step_ok ? "ok" : "FAIL");
  std::printf("  %-4s parallel stop_decide_block matches float64 reference\n", block_ok ? "ok" : "FAIL");
  g_fail += !step_ok + !block_ok;

  // Out-of-bounds n_buckets / widths must clamp cleanly inside FLYX_MAX_STOP_BUCKETS and FLYX_MAX_ROWS.
  flyx_stop_policy oob = pol;
  oob.enabled = 1;
  oob.n_buckets = FLYX_MAX_STOP_BUCKETS + 7;
  for (int i = 0; i < FLYX_MAX_STOP_BUCKETS; ++i) oob.widths[i] = FLYX_MAX_ROWS + 32;
  CK(cudaMemcpy(dpol, &oob, sizeof(oob), cudaMemcpyHostToDevice));
  StopCondHandles zero_h{};
  stop_decide_step(dprob, dmargin, dpol, dst, 0, 0, FLYX_MAX_ROWS + 5, 0, 0, 0);
  DevStopState got_oob_s{};
  CK(cudaMemcpy(&got_oob_s, dst, sizeof(got_oob_s), cudaMemcpyDeviceToHost));
  stop_decide_block(dprob, dmargin, dpol, dst, zero_h, 0);
  DevStopState got_oob_b{};
  CK(cudaMemcpy(&got_oob_b, dst, sizeof(got_oob_b), cudaMemcpyDeviceToHost));
  const bool oob_ok = std::isfinite(got_oob_s.survival) && std::isfinite(got_oob_b.survival) &&
                      got_oob_b.decision >= 0 && got_oob_b.decision < FLYX_MAX_STOP_BUCKETS;
  std::printf("  %-4s stop_decide clamps out-of-bounds n_buckets and widths\n", oob_ok ? "ok" : "FAIL");
  g_fail += !oob_ok;
}

}  // namespace flyx::optest
