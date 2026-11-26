// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// cuda_bench tune: this GPU's tuning table, measured (backends/cuda/variants/<arch>/tables/).
//
//   cuda_bench tune [FAMILIES] [FILE...] [NxK...] > backends/cuda/variants/sm120/tables/<sku>.toml
//
// FAMILIES, a comma-separated list (every one without): gemm (Qwen3.5-4B's bf16, NVFP4 and W12 GEMMs shape by shape,
// and from the same timings the bf16 and nvfp4 rules for shapes no table lists; with NxK arguments, those bf16 GEMMs
// alone, shape by shape, for a table that inherits the rest: a model's GEMMs as the engine runs them,
// --backend-opt log_shapes), rows (bf16 GEMMs of 17..64 rows on
// MiniCPM5-1B's layers, DV's adapters and head: bf16_rows, a round's in three parts, and bf16_draft, a drafting pass's
// in one, shape by shape and as rules), blocks, blocks_q8, blocks_q8_runs, blocks_round (the GGML block kernels'
// rules: candidates.toml's families of those names), and flash_gqa (FP8 grouped-query prefill on a full staged
// window). FILE: GGUF files; the
// block families are measured on their block matrices, each type and shape once, weighed by how many the files hold.
// Without files: every block type on Qwen3.8-27B's and Qwen3.5-4B's layers, each as many times as those models have
// it.
//
// A block family's rules: per class of rows (rounds of 1, 2, 3-4, 5-6, 7-8 and 9-16 rows; passes of 17-128 rows
// and more) and of width (up to 64, 2048, 5120, 6144, 12288 or 65536 columns, and wider), the candidate with the least
// time over the matrices there, every type together; a type's own where that saves 25 us and 3% of the type's time there;
// neighbouring widths with one candidate make one rule.  The runs of K a pass adds a layer up in decide its bits, so
// they are chosen per shape, over every pass size.  Every candidate gives the same bits as the others (the runs
// aside), so a table changes speed only.  The output's comments give the files' GEMMs' time with these rules, with
// the defaults (candidates.toml) and with the best candidate for each matrix. At one and four rows, the round tuner
// also reports the active and best configuration for each measured type and output width, weighted over its Ks.
// Separately, tune strata_placement FILE selects from full decode timings for one host-copy workload. See
// for the input and emitted table.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "bench/microbench/bench.cuh"
#include "bench/microbench/tune_files.h"

namespace flyx::bench {
namespace {

constexpr long kInf = 2147483647;  // "the rest" in the tables
// 5120: the 27B's projections back to its hidden size, whose 8-row rounds want other configurations than its GDN z
// (6144 x 5120: 8 rows of K fit in shared memory whole)
constexpr long kWidths[] = {64, 2048, 5120, 6144, 12288, 65536, kInf};
constexpr int kWidthCount = 7;
constexpr double kMinUs = 25, kMinGain = 0.03;
constexpr double kInfiniteCost = std::numeric_limits<double>::infinity();
constexpr int kNone = -1000;  // no candidate
constexpr int kWarpsKernel = -5;  // rounds: kernel 5 (round_candidates)

int width_class(long n) {
  int w = 0;
  while (n > kWidths[w]) ++w;
  return w;
}
long lower_width(int w) { return w ? kWidths[w - 1] : 0; }

// A block matrix the block families are measured on: its type and shape, and how many of it the files hold.
struct Mat {
  int dtype, N, K;
  double count;
  bool head() const { return N > 100000; }
};

// Without files: kBlockShapes, each as many times as Qwen3.8-27B (64 layers: 48 gated delta nets, 16 attention) and
// Qwen3.5-4B (32: 24 and 8) have it, so a rule's microseconds are a round's of those models: gate and up, down, the
// mixers' out, attention q (with its gate), GDN q|k|v, z, attention k and v, GDN a and b, the head; then the 4B's.
constexpr double kBlockShapeLayers[] = {128, 64, 64, 16, 48, 48, 32, 96, 1, 64, 32, 32, 32, 48};
static_assert(std::size(kBlockShapeLayers) == std::size(kBlockShapes));

std::vector<Mat> block_matrices(const std::vector<std::string>& files) {
  std::map<std::tuple<int, int, int>, double> count;
  if (files.empty())
    for (const BlockType& t : kBlockTypes)
      for (size_t i = 0; i < std::size(kBlockShapes); ++i) {
        const Shape& s = kBlockShapes[i];
        if (s.N < 100000 || t.dtype == FLYX_DT_GGML_Q4_K || t.dtype == FLYX_DT_GGML_Q6_K)
          count[{t.dtype, s.N, s.K}] += kBlockShapeLayers[i];
      }
  for (const std::string& f : files) {
    std::vector<FileMatrix> ms;
    std::string err;
    if (!file_matrices(f, &ms, &err)) {
      std::fprintf(stderr, "cuda_bench tune: %s: %s\n", f.c_str(), err.c_str());
      std::exit(2);
    }
    for (const FileMatrix& m : ms)
      if (block_type_name(m.dtype) && blocks_supported(m.dtype, m.K)) count[{m.dtype, m.N, m.K}] += 1;
  }
  std::vector<Mat> out;
  for (const auto& [k, n] : count) out.push_back({std::get<0>(k), std::get<1>(k), std::get<2>(k), n});
  return out;
}

// One GEMM's inputs (rows up to max_rows, K up to k_max), outputs and scratch; launches with this GPU's options.
struct Bed {
  int k_max;
  float *x = nullptr, *y = nullptr;
  void* mem = nullptr;
  Scratch scratch;
  Launch L;
  Bed(int max_rows, int k_max, size_t y_floats) : k_max(k_max) {
    const size_t bytes = scratch_bytes(max_rows, k_max);
    CK(cudaMalloc(&mem, bytes));
    scratch = scratch_in(mem, bytes, max_rows, k_max);
    L = Launch{0, &scratch, &g_bench_opts};
    CK(cudaMalloc(&x, size_t(max_rows) * k_max * 4));
    k_fill_f<<<64, 256>>>(x, size_t(max_rows) * k_max, 7u, -1.f, 2.f);
    CK(cudaMalloc(&y, y_floats * 4));
  }
  ~Bed() {
    for (void* p : {(void*)x, (void*)y, mem}) cudaFree(p);
  }
};

// us per matrix (its `count` copies), class of rows and candidate; no entry: not measured, infinite: cannot run.
using Costs = std::vector<std::vector<std::map<int, double>>>;

// A family's choice: per class of rows and width the candidate for every type, and per type where its own is better
// enough (its exceptions) and what is best for it everywhere it has matrices.
struct Chosen {
  std::vector<std::vector<int>> every;                                // [class][width]
  std::set<std::tuple<int, int, int>> exceptions;                     // (type, class, width)
  std::map<int, std::vector<std::vector<int>>> own;                   // type -> [class][width] (kNone: no matrices)
  int pick(int dtype, int cls, int w) const {                         // what the rules give a matrix
    return exceptions.count({dtype, cls, w}) ? own.at(dtype)[cls][w] : every[cls][w];
  }
};

// Candidates in the order ties go: dp4a configurations (and tiles) first, then the tensor-core kernels.
bool before(int a, int b) { return (a >= 0) != (b >= 0) ? a >= 0 : a >= 0 ? a < b : a > b; }

Chosen choose(const std::vector<Mat>& mats, const Costs& costs, int classes, bool typed) {
  Chosen ch;
  ch.every.assign(classes, std::vector<int>(kWidthCount, kNone));
  for (int c = 0; c < classes; ++c) {
    for (int w = 0; w < kWidthCount; ++w) {
      std::vector<size_t> group;
      for (size_t m = 0; m < mats.size(); ++m)
        if (width_class(mats[m].N) == w && !costs[m][c].empty()) group.push_back(m);
      if (group.empty()) continue;
      std::vector<int> cands;
      for (const auto& [k, us] : costs[group[0]][c]) cands.push_back(k);
      std::sort(cands.begin(), cands.end(), before);
      auto cost = [&](int k, const std::vector<size_t>& ms) {
        double t = 0;
        for (size_t m : ms) {
          const auto it = costs[m][c].find(k);
          if (it == costs[m][c].end()) return kInfiniteCost;
          t += mats[m].count * it->second;
        }
        return t;
      };
      // every: only the candidates every type has (kernel 5 takes some types alone)
      auto best_of = [&](const std::vector<size_t>& ms, bool every) {
        int best = kNone;
        for (int k : cands)
          if ((!every || k != kWarpsKernel) && std::isfinite(cost(k, ms)) &&
              (best == kNone || cost(k, ms) < cost(best, ms)))
            best = k;
        return best;
      };
      const int best = ch.every[c][w] = best_of(group, true);
      std::map<int, std::vector<size_t>> by_type;
      for (size_t m : group) by_type[mats[m].dtype].push_back(m);
      for (const auto& [t, ms] : by_type) {
        auto& own = ch.own[t];
        if (own.empty()) own.assign(classes, std::vector<int>(kWidthCount, kNone));
        const int tb = own[c][w] = best_of(ms, false);
        const double saved = cost(best, ms) - cost(tb, ms);
        if (typed && tb != best && saved >= kMinUs && saved >= kMinGain * cost(best, ms)) ch.exceptions.insert({t, c, w});
      }
    }
    // Widths without matrices take the nearest width's candidate, a wider one first: any serves them.
    for (int w = 0; w < kWidthCount; ++w)
      for (int d = 1; ch.every[c][w] == kNone && d < kWidthCount; ++d) {
        if (w + d < kWidthCount && ch.every[c][w + d] != kNone) ch.every[c][w] = ch.every[c][w + d];
        else if (w - d >= 0 && ch.every[c][w - d] != kNone) ch.every[c][w] = ch.every[c][w - d];
      }
  }
  return ch;
}

// The time the matrices' GEMMs take with a choice per matrix and class.
template <class Pick>
double total_us(const std::vector<Mat>& mats, const Costs& costs, int cls, Pick&& pick) {
  double t = 0;
  for (size_t m = 0; m < mats.size(); ++m) {
    if (costs[m][cls].empty()) continue;
    const auto it = costs[m][cls].find(pick(m));
    t += mats[m].count * (it == costs[m][cls].end() ? kInfiniteCost : it->second);
  }
  return t;
}
double best_us(const std::vector<Mat>& mats, const Costs& costs, int cls) {
  double t = 0;
  for (size_t m = 0; m < mats.size(); ++m) {
    double b = kInfiniteCost;
    for (const auto& [k, us] : costs[m][cls]) b = std::min(b, us);
    if (!costs[m][cls].empty()) t += mats[m].count * b;
  }
  return t;
}
void print_summary(const char* family, const char* what, const std::vector<Mat>& mats, const Costs& costs, int cls,
                   double rules, double defaults) {
  std::printf("# %s, %s: %.0f us with these rules, %.0f with the defaults (%+.1f%%), %.0f with the best for each\n",
              family, what, rules, defaults, 100 * (rules / defaults - 1), best_us(mats, costs, cls));
}

std::string types_list(const std::vector<int>& types) {
  std::string s = "types = [";
  for (size_t i = 0; i < types.size(); ++i) s += (i ? ", \"" : "\"") + std::string(block_type_name(types[i])) + "\"";
  return s + "], ";
}

// ---- rounds (blocks_round) ------------------------------------------------------------------------------------------
struct RowClass {
  int lo, hi, rows;  // rows lo..hi, measured at `rows`
};
const RowClass kRoundClasses[] = {{1, 1, 1}, {2, 2, 2}, {3, 4, 4}, {5, 6, 6}, {7, 8, 8}, {9, 16, 16}};
constexpr int kRoundClassCount = 6;
int held(int rows) { return rows <= 1 ? 1 : rows <= 2 ? 2 : rows <= 4 ? 4 : rows <= 8 ? 8 : 16; }

// dp4a configurations (>= 0) whose kernel holds the rows (and the 2-row kernel's for 1 row), -1 the tensor cores
// with the rows in the MMA's first operand, -2 with the columns there (8 rows at most), -3 with the rows there and
// the streams split among blocks, -4 the same with the columns there (8 rows at most), -5 the streams split among a
// block's warps (the types it takes: no rule for every type names it).
std::vector<int> round_candidates(int rows, int dtype) {
  std::vector<int> c;
  for (int i = 0; i < kBlocksRoundConfigCount; ++i)
    if (kBlocksRoundConfigs[i].rows == held(rows) || (rows == 1 && kBlocksRoundConfigs[i].rows == 2)) c.push_back(i);
  c.push_back(-1);
  if (rows <= 8) c.push_back(-2);
  c.push_back(-3);
  if (rows <= 8) c.push_back(-4);
  if (gemm_blocks_round_warps_takes(dtype)) c.push_back(kWarpsKernel);
  return c;
}

Costs time_rounds(const std::vector<Mat>& mats, Bed& bed) {
  Costs out(mats.size(), std::vector<std::map<int, double>>(kRoundClassCount));
  for (size_t m = 0; m < mats.size(); ++m) {
    const Mat& a = mats[m];
    std::vector<BlockMat> w = block_copies(a.dtype, a.N, a.K);
    for (int c = 0; c < kRoundClassCount; ++c) {
      const int rows = kRoundClasses[c].rows;
      blocks_stage(bed.x, bed.k_max, a.K, rows, bed.L, true);
      const std::vector<int> cands = round_candidates(rows, a.dtype);
      size_t i = 0;
      auto timed = [&](int k, int reps) {
        return time_ms([&] {
          BlockMat b = w[i++ % w.size()];
          b.q8 = true;
          if (k >= 0) gemm_blocks_round_q8_cfg(k, b, a.K, bed.y, a.N, rows, false, bed.L);
          else gemm_blocks_round_q8_kernel(-k, b, a.K, bed.y, a.N, rows, false, bed.L);
        }, reps);
      };
      // The clocks settle; then every candidate in order and in reverse, twice, the fastest of the four: the GPU's
      // state drifts within a run (two passes chose another rule for 8-row rounds of 2048 to 5120 columns from one run
      // to the next).
      timed(cands[0], 100);
      for (int pass = 0; pass < 4; ++pass)
        for (size_t j = 0; j < cands.size(); ++j) {
          const int k = cands[pass % 2 ? cands.size() - 1 - j : j];
          const double us = timed(k, 50) * 1e3;
          const auto it = out[m][c].find(k);
          out[m][c][k] = it == out[m][c].end() ? us : std::min(it->second, us);
        }
    }
    free_copies(w);
    std::fprintf(stderr, "rounds: %s %d x %d (%zu of %zu)\n", block_type_name(a.dtype), a.N, a.K, m + 1, mats.size());
  }
  return out;
}

void print_round_rule(const std::vector<int>& types, const RowClass& rc, long n_min, long n_max, int k) {
  std::printf("  { %srows = [%d, %d], ", types.empty() ? "" : types_list(types).c_str(), rc.lo, rc.hi);
  if (n_min) std::printf("n_min = %ld, ", n_min);
  if (n_max != kInf) std::printf("n_max = %ld, ", n_max);
  if (k < 0) {
    std::printf("tensor_cores = \"%s\" },\n",
                k == -1 ? "rows" : k == -2 ? "columns" : k == -3 ? "split" : k == -4 ? "split_columns" : "warps");
    return;
  }
  const BlocksRoundConfig& c = kBlocksRoundConfigs[k];
  if (c.rows != held(rc.hi)) std::printf("kernel = %d, ", c.rows);
  std::printf("columns = %d, ahead = %d", c.columns, c.ahead);
  if (c.warps != 8) std::printf(", warps = %d", c.warps);
  if (c.staged) std::printf(", staged = %d", c.staged);
  if (c.parts != 1) std::printf(", parts = %d", c.parts);
  std::printf(" },\n");
}

void tune_rounds(const std::vector<Mat>& mats, Bed& bed) {
  const Costs costs = time_rounds(mats, bed);
  const Chosen ch = choose(mats, costs, kRoundClassCount, true);
  // A type's exceptions: neighbouring widths with one candidate as one span, reaching down over the widths where the
  // rule for every type has that candidate; types with the same span and candidate share a rule.
  struct Span {
    std::vector<int> types;
    int cls;
    long n_min, n_max;
    int k;
  };
  std::vector<Span> spans;
  for (const auto& [t, c, w] : ch.exceptions) {  // ordered by type, class, width
    const int k = ch.own.at(t)[c][w];
    Span* last = spans.empty() ? nullptr : &spans.back();
    if (last && last->types[0] == t && last->cls == c && last->k == k && last->n_max == lower_width(w)) {
      last->n_max = kWidths[w];
      continue;
    }
    long n_min = lower_width(w);
    while (n_min && ch.every[c][width_class(n_min)] == k) n_min = lower_width(width_class(n_min));
    spans.push_back({{t}, c, n_min, kWidths[w], k});
  }
  std::vector<Span> typed;
  for (const Span& s : spans) {
    auto same = std::find_if(typed.begin(), typed.end(), [&](const Span& u) {
      return u.cls == s.cls && u.n_min == s.n_min && u.n_max == s.n_max && u.k == s.k;
    });
    if (same != typed.end()) same->types.push_back(s.types[0]);
    else typed.push_back(s);
  }
  std::stable_sort(typed.begin(), typed.end(),
                   [](const Span& a, const Span& b) { return std::tie(a.cls, a.n_min) < std::tie(b.cls, b.n_min); });
  std::printf("blocks_round = [\n");
  for (const Span& s : typed) print_round_rule(s.types, kRoundClasses[s.cls], s.n_min, s.n_max, s.k);
  for (int c = 0; c < kRoundClassCount; ++c) {
    long n_min = 0;
    for (int w = 0; w < kWidthCount; ++w) {
      if (w + 1 < kWidthCount && ch.every[c][w + 1] == ch.every[c][w]) continue;
      print_round_rule({}, kRoundClasses[c], n_min, kWidths[w], ch.every[c][w]);
      n_min = kWidths[w];
    }
  }
  std::printf("]\n");
  for (int c = 0; c < kRoundClassCount; ++c) {
    const RowClass& rc = kRoundClasses[c];
    const double rules = total_us(mats, costs, c, [&](size_t m) {
      return ch.pick(mats[m].dtype, c, width_class(mats[m].N));
    });
    const double defaults = total_us(mats, costs, c, [&](size_t m) {
      const int k = gemm_blocks_round_default_kernel(-1, mats[m].dtype, mats[m].N, rc.rows);
      return k == 0 ? gemm_blocks_round_default(-1, mats[m].dtype, mats[m].N, rc.rows) : -k;
    });
    char what[64];
    std::snprintf(what, sizeof what, "rounds of %d rows", rc.rows);
    print_summary("blocks_round", what, mats, costs, c, rules, defaults);
  }
  // Exact output widths for the common single-row and four-row launches. A table cannot distinguish K here:
  // weigh all the file's matrices of this type and width together, as the dispatch will see them.
  std::map<std::pair<int, int>, std::vector<size_t>> shapes;
  for (size_t m = 0; m < mats.size(); ++m) shapes[{mats[m].dtype, mats[m].N}].push_back(m);
  for (const auto& [shape, ms] : shapes) {
    const auto [dtype, n] = shape;
    for (int c = 0; c < kRoundClassCount; ++c) {
      const int rows = kRoundClasses[c].rows;
      if (rows != 1 && rows != 4) continue;
      std::map<int, double> us;
      for (size_t m : ms)
        for (const auto& [k, t] : costs[m][c]) us[k] += mats[m].count * t;
      const int kernel = gemm_blocks_round_default_kernel(g_bench_opts.tuning, dtype, n, rows);
      const int active = kernel ? -kernel : gemm_blocks_round_default(g_bench_opts.tuning, dtype, n, rows);
      const auto best = std::min_element(us.begin(), us.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second < b.second : before(a.first, b.first);
      });
      std::printf("# round shape %s N=%d rows=%d active=%d ", block_type_name(dtype), n, rows, active);
      const auto measured = us.find(active);
      if (measured == us.end()) std::printf("unmeasured");
      else std::printf("%.2f us", measured->second);
      std::printf(" best=%d %.2f us; K:count", best->first, best->second);
      for (size_t m : ms) std::printf(" %d:%.0f", mats[m].K, mats[m].count);
      std::printf("\n# ");
      print_round_rule({dtype}, {rows, rows, rows}, n - 1, n, best->first);
    }
  }
}

// ---- passes of more than 16 rows in 8 bits (blocks_q8, blocks_q8_runs) -----------------------------------------------
const int kPassRows[] = {32, 64, 128, 512};
constexpr int kPassRowCount = 4;
const int kRuns[] = {1, 2, 4, 8, 16};
// the classes of passes: 17-128 rows (measured at 32, 64 and 128), more (512)
constexpr int kPassClassCount = 2;
int pass_class(int rows) { return rows <= 128 ? 0 : 1; }

// us of a pass per matrix: [rows][runs][configuration] (infinite: runs past a run a window)
using PassTimes = std::vector<std::vector<std::vector<std::vector<double>>>>;

PassTimes time_passes(const std::vector<Mat>& mats, Bed& bed) {
  PassTimes out(mats.size());
  for (size_t m = 0; m < mats.size(); ++m) {
    const Mat& a = mats[m];
    out[m].assign(kPassRowCount, std::vector<std::vector<double>>(std::size(kRuns),
                                                                  std::vector<double>(kBlocksQ8ConfigCount, kInfiniteCost)));
    if (a.head()) continue;  // the head runs as rounds
    std::vector<BlockMat> w = block_copies(a.dtype, a.N, a.K);
    const int windows = (a.K + 255) / 256;
    size_t i = 0;
    for (int r = 0; r < kPassRowCount; ++r) {
      blocks_stage(bed.x, bed.k_max, a.K, kPassRows[r], bed.L, true);
      for (size_t u = 0; u < std::size(kRuns) && kRuns[u] <= windows; ++u)
        for (int c = 0; c < kBlocksQ8ConfigCount; ++c)
          out[m][r][u][c] = 1e3 * time_ms([&] {
            BlockMat b = w[i++ % w.size()];
            b.q8 = true;
            gemm_blocks_q8_cfg(c, b, a.K, bed.y, a.N, kPassRows[r], false, bed.L, -1, kRuns[u]);
          }, 10);
    }
    free_copies(w);
    std::fprintf(stderr, "passes: %s %d x %d (%zu of %zu)\n", block_type_name(a.dtype), a.N, a.K, m + 1, mats.size());
  }
  return out;
}

int runs_index(int runs) {
  for (size_t u = 0; u < std::size(kRuns); ++u)
    if (kRuns[u] >= runs) return int(u);
  return int(std::size(kRuns)) - 1;
}

// The runs of K per shape: least time over every pass size, each with its best configuration (fewer runs on a tie
// within 0.5%).  Returns runs by (N, K) and prints the rules if asked.
std::map<std::pair<int, int>, int> tune_runs(const std::vector<Mat>& mats, const PassTimes& t, bool print) {
  std::map<std::pair<int, int>, std::vector<double>> cost;  // shape -> us by runs
  for (size_t m = 0; m < mats.size(); ++m) {
    if (mats[m].head()) continue;
    auto& c = cost[{mats[m].N, mats[m].K}];
    c.resize(std::size(kRuns), 0);
    for (size_t u = 0; u < std::size(kRuns); ++u)
      for (int r = 0; r < kPassRowCount; ++r)
        c[u] += mats[m].count * *std::min_element(t[m][r][u].begin(), t[m][r][u].end());
  }
  std::map<std::pair<int, int>, int> runs;
  for (const auto& [shape, c] : cost) {
    size_t best = 0;
    for (size_t u = 1; u < c.size(); ++u)
      if (c[u] < c[best] * 0.995) best = u;
    runs[shape] = kRuns[best];
  }
  if (!print) return runs;
  // By rising n_max (the shapes' columns), then k_max: neighbouring K with one count merge; each n_max ends with a
  // rule for the rest of K, the list with one for the rest (1 run); neighbouring columns with the same rules merge.
  std::printf("blocks_q8_runs = [\n");
  std::map<int, std::vector<std::pair<int, int>>> by_n;  // N -> its rules: (k_max, runs), the last k_max the rest
  for (const auto& [shape, r] : runs) {
    auto& ks = by_n[shape.first];
    if (!ks.empty() && ks.back().second == r) ks.back().first = shape.second;
    else ks.push_back({shape.second, r});
  }
  for (auto& [n, ks] : by_n) ks.back().first = int(kInf);
  for (auto it = by_n.begin(); it != by_n.end(); ++it) {
    const auto next = std::next(it);
    if (next != by_n.end() && next->second == it->second) continue;
    if (next == by_n.end() && it->second.size() == 1 && it->second[0].second == 1) continue;  // the rest's rule
    for (const auto& [k, r] : it->second) {
      if (k == kInf) std::printf("  { n_max = %d, runs = %d },\n", it->first, r);
      else std::printf("  { n_max = %d, k_max = %d, runs = %d },\n", it->first, k, r);
    }
  }
  std::printf("  { runs = 1 },\n]\n");
  double ours = 0, defaults = 0;
  for (const auto& [shape, c] : cost) {
    ours += c[runs_index(runs[shape])];
    defaults += c[runs_index(gemm_blocks_q8_runs(-1, shape.first, shape.second))];
  }
  std::printf("# blocks_q8_runs, passes of 32, 64, 128 and 512 rows: %.0f us with these runs, %.0f with the "
              "defaults' (%+.1f%%)\n", ours, defaults, 100 * (ours / defaults - 1));
  return runs;
}

void print_q8_rule(const std::vector<int>& types, int cls, long n_max, int k, bool last) {
  std::printf("  { %s", types.empty() ? "" : types_list(types).c_str());
  if (!last) std::printf("rows = [%d, %ld], ", cls ? 129 : 17, cls ? kInf : 128);
  if (!last && n_max != kInf) std::printf("n_max = %ld, ", n_max);
  const BlocksQ8Config& c = kBlocksQ8Configs[k];
  std::printf("bn = %d, qk = %d, bm = %d },\n", c.bn, c.qk, c.bm);
}

// A list of rules from [class][width] candidates: neighbouring widths with one candidate merge; the last rule is the
// one for the rest (17 rows and more, every width), as the generator wants each list to end.
void print_q8_list(const std::vector<int>& types, const std::vector<std::vector<int>>& pick) {
  for (int c = 0; c < kPassClassCount; ++c)
    for (int w = 0; w < kWidthCount; ++w) {
      if (w + 1 < kWidthCount && pick[c][w + 1] == pick[c][w]) continue;
      print_q8_rule(types, c, kWidths[w], pick[c][w], c == kPassClassCount - 1 && w == kWidthCount - 1);
    }
}

void tune_passes(const std::vector<Mat>& mats, Bed& bed, bool want_runs, bool want_tiles) {
  const PassTimes t = time_passes(mats, bed);
  // The runs these rules run with: the ones chosen here, or this GPU's table's (else the defaults').
  const auto chosen_runs = tune_runs(mats, t, want_runs);
  if (!want_tiles) return;
  auto runs_of = [&](const Mat& a) {
    return want_runs ? chosen_runs.at({a.N, a.K}) : gemm_blocks_q8_runs(g_bench_opts.tuning, a.N, a.K);
  };
  Costs costs(mats.size(), std::vector<std::map<int, double>>(kPassClassCount));
  for (size_t m = 0; m < mats.size(); ++m) {
    if (mats[m].head()) continue;
    const int u = runs_index(runs_of(mats[m]));
    for (int r = 0; r < kPassRowCount; ++r)
      for (int c = 0; c < kBlocksQ8ConfigCount; ++c) costs[m][pass_class(kPassRows[r])][c] += t[m][r][u][c];
  }
  const Chosen ch = choose(mats, costs, kPassClassCount, true);
  // A type with an exception gets a list of its own, whole (its best everywhere; the rule for every type where it
  // has no matrices); types with the same list share it.
  std::vector<std::pair<std::vector<int>, std::vector<std::vector<int>>>> lists;
  for (const auto& [t, own] : ch.own) {
    if (std::none_of(ch.exceptions.begin(), ch.exceptions.end(),
                     [dtype = t](const auto& e) { return std::get<0>(e) == dtype; }))
      continue;
    std::vector<std::vector<int>> pick = own;
    for (int c = 0; c < kPassClassCount; ++c)
      for (int w = 0; w < kWidthCount; ++w)
        if (pick[c][w] == kNone) pick[c][w] = ch.every[c][w];
    auto same = std::find_if(lists.begin(), lists.end(), [&](const auto& l) { return l.second == pick; });
    if (same != lists.end()) same->first.push_back(t);
    else lists.push_back({{t}, pick});
  }
  std::printf("blocks_q8 = [\n");
  for (const auto& [types, pick] : lists) print_q8_list(types, pick);
  print_q8_list({}, ch.every);
  std::printf("]\n");
  for (int c = 0; c < kPassClassCount; ++c) {
    auto listed = [&](int dtype) {
      return std::any_of(lists.begin(), lists.end(), [&](const auto& l) {
        return std::find(l.first.begin(), l.first.end(), dtype) != l.first.end();
      });
    };
    const double rules = total_us(mats, costs, c, [&](size_t m) {
      const int w = width_class(mats[m].N);
      if (listed(mats[m].dtype)) {
        const int k = ch.own.at(mats[m].dtype)[c][w];
        return k == kNone ? ch.every[c][w] : k;
      }
      return ch.every[c][w];
    });
    // the defaults: their configuration with their runs, from the same timings
    double defaults = 0;
    for (size_t m = 0; m < mats.size(); ++m) {
      if (mats[m].head()) continue;
      const int u = runs_index(gemm_blocks_q8_runs(-1, mats[m].N, mats[m].K));
      for (int r = 0; r < kPassRowCount; ++r)
        if (pass_class(kPassRows[r]) == c)
          defaults += mats[m].count * t[m][r][u][gemm_blocks_q8_default(-1, mats[m].dtype, mats[m].N, kPassRows[r])];
    }
    print_summary("blocks_q8", c ? "passes of 512 rows" : "passes of 32, 64 and 128 rows", mats, costs, c, rules,
                  defaults);
  }
}

// ---- 1..16 rows with exact inputs (blocks) --------------------------------------------------------------------------
const RowClass kExactClasses[] = {{1, 1, 1}, {2, 2, 2}, {3, 4, 4}, {5, 8, 8}, {9, 16, 16}};
constexpr int kExactClassCount = 5;

void tune_exact(const std::vector<Mat>& mats, Bed& bed) {
  Costs costs(mats.size(), std::vector<std::map<int, double>>(kExactClassCount));
  for (size_t m = 0; m < mats.size(); ++m) {
    const Mat& a = mats[m];
    std::vector<BlockMat> w = block_copies(a.dtype, a.N, a.K);
    size_t i = 0;
    for (int c = 0; c < kExactClassCount; ++c) {
      const int rows = kExactClasses[c].rows;
      blocks_stage(bed.x, bed.k_max, a.K, rows, bed.L);
      for (int k = 0; k < kBlocksConfigCount; ++k) {
        bool fits = true;
        const double ms = time_ms([&] { fits = gemm_blocks_cfg(k, w[i++ % w.size()], a.K, bed.y, a.N, rows, false, bed.L); },
                                  20);
        costs[m][c][k] = fits ? ms * 1e3 : kInfiniteCost;
      }
    }
    free_copies(w);
    std::fprintf(stderr, "exact inputs: %s %d x %d (%zu of %zu)\n", block_type_name(a.dtype), a.N, a.K, m + 1,
                 mats.size());
  }
  const Chosen ch = choose(mats, costs, kExactClassCount, false);
  // By rising n_max; each width's rules cover rows 1..16 once (neighbouring classes with one tile merge); neighbouring
  // widths with the same rules merge.
  std::printf("blocks = [\n");
  for (int w = 0; w < kWidthCount; ++w) {
    bool same_next = w + 1 < kWidthCount;
    for (int c = 0; same_next && c < kExactClassCount; ++c) same_next = ch.every[c][w + 1] == ch.every[c][w];
    if (same_next) continue;
    for (int c = 0; c < kExactClassCount; ++c) {
      if (c + 1 < kExactClassCount && ch.every[c + 1][w] == ch.every[c][w]) continue;
      int lo = c;
      while (lo > 0 && ch.every[lo - 1][w] == ch.every[c][w]) --lo;
      const BlocksConfig& k = kBlocksConfigs[ch.every[c][w]];
      std::printf("  { ");
      if (kWidths[w] != kInf) std::printf("n_max = %ld, ", kWidths[w]);
      std::printf("rows = [%d, %d], nt = %d, nw = %d, kw = %d },\n", kExactClasses[lo].lo, kExactClasses[c].hi, k.nt,
                  k.nw, k.kw);
    }
  }
  std::printf("]\n");
  for (int c = 0; c < kExactClassCount; ++c) {
    const int rows = kExactClasses[c].rows;
    const double rules = total_us(mats, costs, c, [&](size_t m) { return ch.every[c][width_class(mats[m].N)]; });
    const double defaults =
        total_us(mats, costs, c, [&](size_t m) { return gemm_blocks_default(-1, mats[m].N, rows); });
    char what[64];
    std::snprintf(what, sizeof what, "%d rows", rows);
    print_summary("blocks", what, mats, costs, c, rules, defaults);
  }
}

// ---- the bf16, NVFP4 and W12 GEMMs of Qwen3.5-4B, and Qwen3.8-27B's NVFP4 ones (gemm) ---------------------------------
struct GemmShape {
  int fmt, N, K;  // fmt: 0 bf16, 1 nvfp4, 2 w12
  const char* what;
};
const GemmShape kGemmShapes[] = {
    {1, 10240, 2560, "attention q|k|v"}, {1, 12352, 2560, "gdn in (qkv|z|b|a)"}, {1, 18432, 2560, "mlp gate|up"},
    {1, 2560, 9216, "mlp down"},         {1, 2560, 4096, "attention / gdn out"},
    {0, 248320, 2560, "lm head"},        {0, 10240, 2560, "mtp q|k|v, dflash2 context k|v"},
    {0, 18432, 2560, "mtp / dflash2 mlp gate|up"}, {0, 2560, 9216, "mtp / dflash2 mlp down"},
    {0, 2560, 4096, "mtp / dflash2 attention out"}, {0, 2560, 5120, "mtp fc"}, {0, 6144, 2560, "dflash2 q|k|v"},
    {0, 640, 2560, "dflash2 conv proj"},  {0, 2560, 12800, "dflash2 fc"}, {0, 256, 2560, "dflash2 selector"},
    {2, 248320, 2560, "lm head in W12 (the same bits as bf16)"},
    // Qwen3.8-27B in NVFP4 (GGML's, repacked), its stacks as the engine forms them.
    {1, 14336, 5120, "27B attention q|k|v"}, {1, 16480, 5120, "27B gdn in (qkv|z|b|a)"},
    {1, 34816, 5120, "27B mlp gate|up"},      {1, 5120, 17408, "27B mlp down"},
    {1, 5120, 6144, "27B attention / gdn out"}};
const int kBands[4][3] = {{1, 1, 1}, {2, 4, 4}, {5, 8, 8}, {9, 16, 16}};  // lo, hi, rows measured
const char* const kFmtName[] = {"bf16", "nvfp4", "w12"};

// Per shape and row band the fastest configuration; for NVFP4 first the K partition with the best total over the
// bands (one for all bands).  Returns ms[shape][configuration][band].
std::vector<std::vector<std::array<double, 4>>> tune_gemm_shapes(const std::vector<GemmShape>& shapes) {
  std::vector<std::vector<std::array<double, 4>>> all;
  for (const GemmShape& s : shapes) {
    const bool fp4 = s.fmt == 1;
    const size_t bytes = fp4 ? size_t(s.N) * s.K / 2 + size_t(s.N) * s.K / 16
                             : s.fmt == 2 ? w12_plane_bytes(s.N, s.K) : size_t(s.N) * s.K * 2;
    const size_t copies = std::max<size_t>(1, (512ull << 20) / bytes + 1);
    std::vector<void*> wq(copies), ws(copies);
    std::vector<W12> w12(copies);
    for (size_t i = 0; i < copies; ++i) {
      const size_t wbytes = fp4 ? size_t(s.N) * s.K / 2 : size_t(s.N) * s.K * 2;
      CK(cudaMalloc(&wq[i], wbytes));
      ws[i] = nullptr;
      if (s.fmt == 2) {  // random bf16 weights (escapes included), their W12 copy
        k_fill_bf16<<<1024, 256>>>(static_cast<bf16*>(wq[i]), size_t(s.N) * s.K, 101u + uint32_t(i), -0.05f, 0.1f);
        W12& c = w12[i];
        CK(cudaMalloc(&c.hi, size_t(s.N) * s.K));
        CK(cudaMalloc(&c.lo, size_t(s.N) * s.K / 2));
        CK(cudaMalloc(&c.e0, s.N));
        CK(cudaMalloc(&c.esc_start, (size_t(s.N) + 1) * 4));
        std::vector<int> starts(size_t(s.N) + 1);
        const int esc = w12_encode(static_cast<bf16*>(wq[i]), s.N, s.K, &c, starts.data(), 0);
        CK(cudaMalloc(&c.esc_k, std::max(esc, 1) * 2));
        CK(cudaMalloc(&c.esc_v, std::max(esc, 1) * 2));
        if (w12_finish(static_cast<bf16*>(wq[i]), s.N, s.K, &c, 0) != 0) std::abort();
        CK(cudaFree(wq[i]));  // only the W12 planes are read
        wq[i] = nullptr;
        continue;
      }
      CK(cudaMemset(wq[i], 0x11, wbytes));
      if (fp4) {
        CK(cudaMalloc(&ws[i], size_t(s.N) * s.K / 16));
        CK(cudaMemset(ws[i], 0x38, size_t(s.N) * s.K / 16));
      }
    }
    float *rs, *x, *y;
    CK(cudaMalloc(&rs, size_t(s.N) * 4));
    CK(cudaMemset(rs, 0, size_t(s.N) * 4));
    CK(cudaMalloc(&x, size_t(16) * s.K * 4));
    CK(cudaMemset(x, 0, size_t(16) * s.K * 4));
    CK(cudaMalloc(&y, size_t(16) * s.N * 4));
    const int nc = fp4 ? kFp4ConfigCount : kSkinnyConfigs;
    std::vector<std::array<double, 4>> ms(nc);
    size_t it = 0;
    for (int c = 0; c < nc; ++c) {
      if (fp4 && s.K % kFp4Configs[c].span) {
        ms[c].fill(1e30);
        continue;
      }
      for (int b = 0; b < 4; ++b) {
        std::vector<double> runs;
        for (int rep = 0; rep < 5; ++rep)
          runs.push_back(time_ms([&] {
            const size_t i = it++ % copies;
            if (fp4)
              gemm_fp4_skinny_cfg(c, x, s.K, static_cast<uint8_t*>(wq[i]), static_cast<uint8_t*>(ws[i]), rs, s.N,
                                  s.K, y, s.N, kBands[b][2], false, BL(0));
            else if (s.fmt == 2)
              gemm_w12_skinny_cfg(c, x, s.K, w12[i], s.N, s.K, y, s.N, kBands[b][2], false, BL(0));
            else
              gemm_skinny_cfg(c, x, s.K, static_cast<bf16*>(wq[i]), s.N, s.K, y, s.N, kBands[b][2], false, BL(0));
          }, 10));
        std::sort(runs.begin(), runs.end());
        ms[c][b] = runs[2];
      }
    }
    // NVFP4: the partition (span, chunks) with the best total, then the best configuration per band inside it.
    auto same_part = [&](int a, int b) {
      return !fp4 || (kFp4Configs[a].span == kFp4Configs[b].span && kFp4Configs[a].chunks == kFp4Configs[b].chunks);
    };
    int part = 0;
    double part_best = 1e300;
    for (int c = 0; c < nc; ++c) {
      double tot = 0;
      for (int b = 0; b < 4; ++b) {
        double best = 1e300;
        for (int d = 0; d < nc; ++d)
          if (same_part(c, d)) best = std::min(best, ms[d][b]);
        tot += best;
      }
      if (tot < part_best) part_best = tot, part = c;
    }
    int pick[4];
    for (int b = 0; b < 4; ++b) {
      pick[b] = -1;
      for (int c = 0; c < nc; ++c)
        if (same_part(part, c) && (pick[b] < 0 || ms[c][b] < ms[pick[b]][b])) pick[b] = c;
    }
    std::printf("\n[[gemm]]  # %s\nformat = \"%s\"\nn = %d\nk = %d\nrules = [\n", s.what, kFmtName[s.fmt], s.N, s.K);
    for (int b = 0; b < 4;) {
      int e = b;
      while (e + 1 < 4 && pick[e + 1] == pick[b]) ++e;
      const int c = pick[b];
      std::printf("  { rows = [%d, %d], ", kBands[b][0], kBands[e][1]);
      if (fp4) std::printf("span = %d, chunks = %d, ", kFp4Configs[c].span, kFp4Configs[c].chunks);
      const int nt = fp4 ? kFp4Configs[c].nt : kBf16Configs[c].nt;
      const int w = fp4 ? kFp4Configs[c].warps : kBf16Configs[c].warps;
      const int u = fp4 ? kFp4Configs[c].unroll : kBf16Configs[c].unroll;
      std::printf("nt = %d, warps = %d, unroll = %d },  #", nt, w, u);
      for (int k = b; k <= e; ++k) std::printf(" r%d %.0f", kBands[k][2], bytes / 1e9 / (ms[c][k] / 1e3));
      std::printf("\n");
      b = e + 1;
    }
    std::printf("]\n");
    std::fflush(stdout);
    for (size_t i = 0; i < copies; ++i) {
      if (wq[i]) cudaFree(wq[i]);
      if (ws[i]) cudaFree(ws[i]);
      for (void* q : {(void*)w12[i].hi, (void*)w12[i].lo, (void*)w12[i].e0, (void*)w12[i].esc_start,
                      (void*)w12[i].esc_k, (void*)w12[i].esc_v})
        if (q) cudaFree(q);
    }
    cudaFree(rs);
    cudaFree(x);
    cudaFree(y);
    all.push_back(ms);
  }
  return all;
}

// The bf16 and nvfp4 rules for shapes no table lists, from the shapes' timings: per band the bf16 configuration
// least slower than each shape's best on average; for NVFP4 one configuration for every row count (one partition),
// for K % 128 == 0 and one whose span is 64 for the rest.
void print_gemm_rules(const std::vector<std::vector<std::array<double, 4>>>& ms) {
  // the sum over the shapes of a format (NVFP4 with K % 128 == 0 only, if asked) of a configuration's time over
  // the shape's best, in band b
  auto slowdown = [&](int fmt, int c, int b, bool k128_only) {
    double s = 0;
    for (size_t i = 0; i < std::size(kGemmShapes); ++i) {
      const GemmShape& g = kGemmShapes[i];
      if (g.fmt != fmt || (k128_only && g.K % 128)) continue;
      double best = 1e300;
      for (const auto& m : ms[i]) best = std::min(best, m[b]);
      s += ms[i][c][b] / best;
    }
    return s;
  };
  std::printf("bf16 = [\n");
  int pick[4];
  for (int b = 0; b < 4; ++b) {
    pick[b] = 0;
    for (int c = 1; c < kSkinnyConfigs; ++c)
      if (slowdown(0, c, b, false) < slowdown(0, pick[b], b, false)) pick[b] = c;
  }
  for (int b = 0; b < 4;) {
    int e = b;
    while (e + 1 < 4 && pick[e + 1] == pick[b]) ++e;
    const Bf16Config& c = kBf16Configs[pick[b]];
    std::printf("  { rows = [%d, %d], nt = %d, warps = %d, unroll = %d },\n", kBands[b][0], kBands[e][1], c.nt, c.warps,
                c.unroll);
    b = e + 1;
  }
  std::printf("]\n");
  for (int span : {128, 64}) {
    int best = -1;
    double best_s = kInfiniteCost;
    for (int c = 0; c < kFp4ConfigCount; ++c) {
      if ((span == 64 && kFp4Configs[c].span != 64) || 128 % kFp4Configs[c].span) continue;
      double s = 0;
      for (int b = 0; b < 4; ++b) s += slowdown(1, c, b, span == 128);
      if (s < best_s) best_s = s, best = c;
    }
    const Fp4Config& c = kFp4Configs[best];
    std::printf("nvfp4_k%d = { span = %d, chunks = %d, nt = %d, warps = %d, unroll = %d }\n", span, c.span, c.chunks,
                c.nt, c.warps, c.unroll);
  }
}

void tune_flash_gqa_rows(int rows) {
  constexpr int nh = 24, nkv = 4, hd = 256, keys = 32832;
  const size_t kv_heads = size_t(keys) * nkv, query_heads = size_t(rows) * nh;
  const size_t state_bytes = query_heads * kFlashStateFloats(hd) * sizeof(float);
  uint8_t* cache;
  float *scales, *q, *state, *out;
  DevParams* params;
  CK(cudaMalloc(&cache, 2 * kv_heads * hd));
  CK(cudaMalloc(&scales, 2 * kv_heads * sizeof(float)));
  CK(cudaMalloc(&q, query_heads * hd * sizeof(float)));
  CK(cudaMalloc(&out, query_heads * hd * sizeof(float)));
  CK(cudaMalloc(&state, state_bytes));
  CK(cudaMalloc(&params, sizeof(DevParams)));
  DevParams host{};
  host.position = keys;
  CK(cudaMemcpy(params, &host, sizeof(host), cudaMemcpyHostToDevice));
  k_fill<<<64, 256>>>(cache, 2 * kv_heads * hd, 91u, 8, 104);  // finite e4m3
  k_fill_f<<<64, 256>>>(scales, 2 * kv_heads, 37u, 0.01f, 0.02f);
  k_fill_f<<<64, 256>>>(q, query_heads * hd, 17u, -1.f, 2.f);
  KvRef kv{cache, cache + kv_heads * hd, scales, scales + kv_heads};
  FlashWindow win{0, keys, state, false, false, true};  // middle window, carries online softmax state
  constexpr int heads[] = {2, 3, 6};
  std::array<std::array<double, 6>, 3> samples{};
  std::array<int, 3> order{0, 1, 2};
  int pass = 0;
  do {
    for (int c : order) {
      CK(cudaMemset(state, 0, state_bytes));
      samples[c][pass] = time_ms([&] {
        if (!attention_flash_gqa_cfg(heads[c], q, kv, nh, nkv, hd, rows, 0.0625f, params, nullptr, 0,
                                     out, nh * hd, BL(0), 0, win)) std::exit(2);
      }, 5);
    }
    ++pass;
  } while (std::next_permutation(order.begin(), order.end()));
  int best = 0;
  std::array<double, 3> median{};
  std::printf("\n# FP8 GQA, one full middle window; six orders, five timed launches per sample (ms).\n");
  for (int c = 0; c < 3; ++c) {
    std::printf("# HB%d", heads[c]);
    for (double ms : samples[c]) std::printf(" %.4f", ms);
    std::sort(samples[c].begin(), samples[c].end());
    median[c] = (samples[c][2] + samples[c][3]) / 2;
    std::printf(" median %.4f\n", median[c]);
    if (median[c] < median[best]) best = c;
  }
  std::printf("[[flash_gqa]]\nfp8 = true\nnh = %d\nnkv = %d\nhd = %d\nrows = %d\nkeys = %d\nhb = %d\n",
              nh, nkv, hd, rows, keys, heads[best]);
  CK(cudaFree(params));
  CK(cudaFree(state));
  CK(cudaFree(out));
  CK(cudaFree(q));
  CK(cudaFree(scales));
  CK(cudaFree(cache));
}

void tune_flash_gqa() {
  for (int rows : {256, 1024}) tune_flash_gqa_rows(rows);
}

const char* const kFamilies[] = {"gemm", "rows", "blocks", "blocks_q8", "blocks_q8_runs", "blocks_round", "flash_gqa"};

// Placement overlaps PCIe transfers with a particular model's verification and draft steps. Use whole
// generations timed with that model. The file contains one shape and the warm decode times for each schedule.
int tune_strata_placement(const char* file) {
  std::ifstream input(file);
  std::array<int, 6> shape{};  // copies, hosted, bits, KV heads, head dim, tested session capacity
  std::map<std::pair<int, int>, std::vector<double>> runs;  // resident placement's host edges, scan batches
  auto invalid = [&](const char* reason) {
    std::fprintf(stderr, "cuda_bench tune strata_placement: %s: %s\n", file, reason);
    return 2;
  };
  if (!input) return invalid("cannot open measurements");
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line.substr(0, line.find('#')));
    std::string kind;
    if (!(row >> kind)) continue;
    if (kind == "shape") {
      if (shape[0]) return invalid("expected a single workload");
      for (int& dim : shape)
        if (!(row >> dim) || dim <= 0) return invalid("expected six positive shape fields");
      if (row >> kind || shape[1] >= shape[0]) return invalid("invalid host-copy shape");
    } else if (kind == "run") {
      int edges = 0, batches = 0;
      if (!shape[0] || !(row >> edges >> batches) || edges < 0 || edges > shape[1] / 2 ||
          (batches != 1 && batches != 2) || (!edges && batches != 1))
        return invalid("expected host edges and scan batches after the shape");
      double ms = 0;
      auto& times = runs[{edges, batches}];
      const auto before = times.size();
      while (row >> std::ws && !row.eof()) {
        if (!(row >> ms) || !(ms > 0 && std::isfinite(ms))) return invalid("expected a positive decode time");
        times.push_back(ms);
      }
      if (times.size() == before) return invalid("expected a warm decode time");
    } else {
      return invalid("expected shape or run");
    }
  }
  if (!input.eof() || !runs.count({0, 1}) || runs.size() < 2) return invalid("expected a baseline and candidate");
  double best = kInfiniteCost;
  std::pair<int, int> pick{};
  for (auto& [schedule, ms] : runs) {
    std::sort(ms.begin(), ms.end());
    const double median = (ms[(ms.size() - 1) / 2] + ms[ms.size() / 2]) / 2;
    std::printf("# edge_host=%d scan_batches=%d samples=%zu median_decode_ms=%.3f\n",
                schedule.first, schedule.second, ms.size(), median);
    if (median < best) {
      best = median;
      pick = schedule;
    }
  }
  if (pick.first) {
    std::printf("[[strata_placement]]\ncopies = %d\nhost = %d\nbits = %d\nnkv = %d\nhd = %d\n"
                "min_capacity = %d\nedge_host = %d\nscan_halves = %s\n",
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], pick.first, pick.second == 2 ? "true" : "false");
  }
  return 0;
}
// ---- bf16 GEMMs of 17..64 rows on k_gemm_skinny_mt (rows): bf16_rows and bf16_draft --------------------------------
// MiniCPM5-1B's layers and head, what a tree round of more than 16 rows verifies (bf16_rows), and the
// shapes the first weight of each of a DV canvas pass's launches has (bf16_draft: dv_context's gemm_draft picks its
// tile and unroll by it; the chunks are K's, gemm_draft_chunks, so only those configurations are measured).
struct MtShape {
  const char* family;
  int N, K;
  const char* what;
};
const MtShape kMtShapes[] = {
    {"bf16_rows", 2560, 1536, "MiniCPM5-1B q|k|v"},        {"bf16_rows", 1536, 2048, "MiniCPM5-1B out"},
    {"bf16_rows", 9216, 1536, "MiniCPM5-1B gate|up"},      {"bf16_rows", 1536, 4608, "MiniCPM5-1B down"},
    {"bf16_rows", 130560, 1536, "MiniCPM5-1B head"},       {"bf16_draft", 2560, 1536, "DV canvas q|k|v (and a)"},
    {"bf16_draft", 2048, 256, "DV canvas b of q, k, v"},   {"bf16_draft", 1536, 2048, "DV canvas out (and a)"},
    {"bf16_draft", 1536, 256, "DV canvas b of out, down"}, {"bf16_draft", 9216, 1536, "DV canvas gate|up (and a)"},
    {"bf16_draft", 4608, 256, "DV canvas b of gate, up"},  {"bf16_draft", 1536, 4608, "DV canvas down (and a)"},
    {"bf16_draft", 130560, 1536, "DV canvas head"}};
const int kMtBands[3][3] = {{17, 32, 32}, {33, 48, 48}, {49, 64, 64}};  // lo, hi, rows measured

// Per shape and band of rows the fastest configuration of its family's parts (a drafting pass's: of its K's chunks);
// returns ms[shape][configuration][band] (infinite where it may not serve).
std::vector<std::vector<std::array<double, 3>>> tune_mt_shapes() {
  std::vector<std::vector<std::array<double, 3>>> all;
  for (const MtShape& s : kMtShapes) {
    const int parts = std::strcmp(s.family, "bf16_rows") == 0 ? 3 : 1;
    const bool draft = parts == 1;
    const size_t bytes = size_t(s.N) * s.K * 2;
    const size_t copies = std::max<size_t>(1, (512ull << 20) / bytes + 1);
    std::vector<bf16*> w(copies);
    for (size_t i = 0; i < copies; ++i) {
      CK(cudaMalloc(&w[i], bytes));
      k_fill_bf16<<<1024, 256>>>(w[i], size_t(s.N) * s.K, 101u + uint32_t(i), -0.05f, 0.1f);
    }
    float *x, *y;
    CK(cudaMalloc(&x, size_t(64) * s.K * 4));
    k_fill_f<<<64, 256>>>(x, size_t(64) * s.K, 7u, -1.f, 2.f);
    CK(cudaMalloc(&y, size_t(64) * s.N * 4));
    std::vector<std::array<double, 3>> ms(kMtConfigCount);
    size_t it = 0;
    for (int c = 0; c < kMtConfigCount; ++c) {
      ms[c].fill(1e30);
      if (kMtConfigs[c].parts != parts || (draft && kMtConfigs[c].chunks != gemm_draft_chunks(s.K))) continue;
      for (int b = 0; b < 3; ++b) {
        std::vector<double> runs;
        for (int rep = 0; rep < 5; ++rep)
          runs.push_back(time_ms([&] {
            gemm_skinny_mt_cfg(c, x, s.K, w[it++ % copies], s.N, s.K, y, s.N, kMtBands[b][2], false, BL(0));
          }, 10));
        std::sort(runs.begin(), runs.end());
        ms[c][b] = runs[2];
      }
    }
    int pick[3];
    for (int b = 0; b < 3; ++b) {
      pick[b] = 0;
      for (int c = 1; c < kMtConfigCount; ++c)
        if (ms[c][b] < ms[pick[b]][b]) pick[b] = c;
    }
    std::printf("\n[[gemm]]  # %s\nformat = \"%s\"\nn = %d\nk = %d\nrules = [\n", s.what, s.family, s.N, s.K);
    for (int b = 0; b < 3;) {
      int e = b;
      while (e + 1 < 3 && pick[e + 1] == pick[b]) ++e;
      const MtConfig& c = kMtConfigs[pick[b]];
      if (draft)
        std::printf("  { rows = [%d, %d], nt = %d, unroll = %d },  #", kMtBands[b][0], kMtBands[e][1], c.nt, c.unroll);
      else
        std::printf("  { rows = [%d, %d], nt = %d, chunks = %d, unroll = %d, parts = %d },  #", kMtBands[b][0],
                    kMtBands[e][1], c.nt, c.chunks, c.unroll, c.parts);
      for (int k = b; k <= e; ++k)
        std::printf(" r%d %.0f (%.1f us)", kMtBands[k][2], bytes / 1e9 / (ms[pick[b]][k] / 1e3), ms[pick[b]][k] * 1e3);
      std::printf("\n");
      b = e + 1;
    }
    std::printf("]\n");
    std::fflush(stdout);
    for (bf16* p : w) cudaFree(p);
    cudaFree(x);
    cudaFree(y);
    all.push_back(ms);
  }
  return all;
}

// The configuration of bf16_mt with this tile, chunks, unroll and parts, or -1.
int mt_config(int nt, int chunks, int unroll, int parts) {
  for (int c = 0; c < kMtConfigCount; ++c)
    if (kMtConfigs[c].nt == nt && kMtConfigs[c].chunks == chunks && kMtConfigs[c].unroll == unroll &&
        kMtConfigs[c].parts == parts)
      return c;
  return -1;
}

// The bf16_rows and bf16_draft rules for shapes no table lists: per band the configuration least slower than each
// shape's best on average over the family's shapes.  A drafting pass's rule names a tile and unroll, which serve each
// shape in its K's chunks: a pair compiled in both partitions (gen_gemm_tables.py checks the same).
void print_mt_rules(const std::vector<std::vector<std::array<double, 3>>>& ms) {
  const int partitions[2] = {gemm_draft_chunks(32), gemm_draft_chunks(1 << 20)};  // a short K's, a long K's
  for (const char* family : {"bf16_rows", "bf16_draft"}) {
    const bool draft = std::strcmp(family, "bf16_draft") == 0;
    // The configuration candidate c gives shape i: c itself, or a drafting pass's c's tile and unroll in the chunks of
    // the shape's K (-1: not compiled).
    auto serves = [&](int c, size_t i) {
      const MtConfig& m = kMtConfigs[c];
      return draft ? mt_config(m.nt, gemm_draft_chunks(kMtShapes[i].K), m.unroll, 1) : c;
    };
    int pick[3];
    for (int b = 0; b < 3; ++b) {
      double best_s = 1e300;
      pick[b] = -1;
      for (int c = 0; c < kMtConfigCount; ++c) {
        const MtConfig& m = kMtConfigs[c];
        if (draft && (m.parts != 1 || m.chunks != partitions[0] || mt_config(m.nt, partitions[1], m.unroll, 1) < 0))
          continue;
        double s = 0;
        for (size_t i = 0; i < std::size(kMtShapes); ++i) {
          if (std::strcmp(kMtShapes[i].family, family)) continue;
          double best = 1e300;
          for (const auto& t : ms[i]) best = std::min(best, t[b]);
          const int ci = serves(c, i);
          s += ci < 0 ? 1e30 : ms[i][ci][b] / best;
        }
        if (s < best_s) best_s = s, pick[b] = c;
      }
    }
    std::printf("%s = [\n", family);
    for (int b = 0; b < 3;) {
      int e = b;
      while (e + 1 < 3 && pick[e + 1] == pick[b]) ++e;
      const MtConfig& c = kMtConfigs[pick[b]];
      if (draft)
        std::printf("  { rows = [%d, %d], nt = %d, unroll = %d },\n", kMtBands[b][0], kMtBands[e][1], c.nt, c.unroll);
      else
        std::printf("  { rows = [%d, %d], nt = %d, chunks = %d, unroll = %d, parts = %d },\n", kMtBands[b][0],
                    kMtBands[e][1], c.nt, c.chunks, c.unroll, c.parts);
      b = e + 1;
    }
    std::printf("]\n");
  }
}


}  // namespace

int tune(int argc, char** argv) {
  if (argc > 0 && std::strcmp(argv[0], "strata_placement") == 0) {
    if (argc == 2) return tune_strata_placement(argv[1]);
    std::fprintf(stderr, "usage: cuda_bench tune strata_placement MEASUREMENTS\n");
    return 2;
  }
  std::set<std::string> families;
  std::vector<std::string> files;
  std::vector<std::string> given;  // NxK: bf16 GEMMs of a model (their comments' text)
  std::vector<GemmShape> shapes;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    int n = 0, k = 0, used = 0;
    if (std::sscanf(a.c_str(), "%dx%d%n", &n, &k, &used) == 2 && used == int(a.size()) && n > 0 && k > 0) {
      if (k % 32) {
        std::fprintf(stderr, "cuda_bench tune: %s: K must be a multiple of 32\n", a.c_str());
        return 2;
      }
      given.push_back("bf16 " + a);
      shapes.push_back({0, n, k, nullptr});
      continue;
    }
    bool named = i == 0 && a.find('/') == std::string::npos && a.find('.') == std::string::npos;
    for (size_t p = 0; named && p < a.size();) {
      const size_t e = std::min(a.find(',', p), a.size());
      const std::string f = a.substr(p, e - p);
      if (std::find(std::begin(kFamilies), std::end(kFamilies), f) == std::end(kFamilies)) {
        std::fprintf(stderr, "cuda_bench tune: unknown family %s (gemm, rows, blocks, blocks_q8, blocks_q8_runs, "
                             "blocks_round, flash_gqa)\n", f.c_str());
        return 2;
      }
      families.insert(f);
      p = e + 1;
    }
    if (!named) files.push_back(a);
  }
  if (families.empty()) families.insert(std::begin(kFamilies), std::end(kFamilies));
  auto want = [&](const char* f) { return families.count(f) > 0; };

  cudaDeviceProp p{};
  CK(cudaGetDeviceProperties(&p, 0));
  std::string sku;
  {
    std::string n = p.name;
    for (const char* pre : {"NVIDIA GeForce ", "NVIDIA "})
      if (n.rfind(pre, 0) == 0) n = n.substr(std::strlen(pre));
    for (char c : n)
      if (c != ' ' && c != '-') sku += char(std::tolower(static_cast<unsigned char>(c)));
  }
  std::string args;
  for (int i = 0; i < argc; ++i) args += std::string(" ") + argv[i];
  std::printf("# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0\n");
  std::printf("# Tuning table for %s, written by `cuda_bench tune%s` (GB/s of weights streamed in brackets).\n", p.name,
              args.c_str());
  std::printf("sku = \"%s\"\ndevices = [\"%s\"]\n", sku.c_str(), p.name);
  std::fflush(stdout);

  if (!shapes.empty() && !want("gemm")) {
    std::fprintf(stderr, "cuda_bench tune: NxK shapes are the gemm family's: name gemm among the families\n");
    return 2;
  }
  if (want("flash_gqa")) tune_flash_gqa();
  std::vector<std::vector<std::array<double, 4>>> gemm_ms;
  if (want("gemm") && !shapes.empty()) {
    for (size_t i = 0; i < shapes.size(); ++i) shapes[i].what = given[i].c_str();
    std::printf("# The GEMMs given alone: rules and every other family from the table this one inherits.\n");
    (void)tune_gemm_shapes(shapes);
  } else if (want("gemm")) {
    gemm_ms = tune_gemm_shapes(std::vector<GemmShape>(std::begin(kGemmShapes), std::end(kGemmShapes)));
  }
  std::vector<std::vector<std::array<double, 3>>> mt_ms;
  if (want("rows")) mt_ms = tune_mt_shapes();
  const bool blocks = want("blocks") || want("blocks_q8") || want("blocks_q8_runs") || want("blocks_round");
  const std::vector<Mat> mats = blocks ? block_matrices(files) : std::vector<Mat>();
  if (gemm_ms.empty() && mt_ms.empty() && mats.empty()) return 0;
  std::printf("\n# This GPU's rules: for shapes no [[gemm]] lists, and the GGML block kernels'%s.\n"
              "[rules]\n", files.empty() ? " on Qwen3.8-27B's and Qwen3.5-4B's layers, every block type" :
                                          " on the files' block matrices");
  if (!gemm_ms.empty()) print_gemm_rules(gemm_ms);
  if (!mt_ms.empty()) print_mt_rules(mt_ms);
  if (mats.empty()) return 0;
  int k_max = 0, n_rows = 0, n_head = 0;
  for (const Mat& m : mats) {
    k_max = std::max(k_max, m.K);
    (m.head() ? n_head : n_rows) = std::max(m.head() ? n_head : n_rows, m.N);
  }
  Bed bed(512, k_max, std::max(size_t(512) * n_rows, size_t(16) * n_head));
  std::fflush(stdout);
  if (want("blocks")) tune_exact(mats, bed);
  std::fflush(stdout);
  if (want("blocks_q8") || want("blocks_q8_runs")) tune_passes(mats, bed, want("blocks_q8_runs"), want("blocks_q8"));
  std::fflush(stdout);
  if (want("blocks_round")) tune_rounds(mats, bed);
  return 0;
}

}  // namespace flyx::bench
