// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// End-to-end check against golden outputs of the Python reference.
//
//   golden_test ASSET_DIR BACKEND
//
// For every golden prompt: plain decoding, DFlash2 decoding, decoding with
// oracle drafts (forces long accepted runs), and session reuse (continue,
// rewind inside the round window, restore a checkpoint, edit the prompt).
// Every path must produce the reference's greedy tokens.  A model without a
// drafter (golden "plain_only": the tiny Qwen3, the tiny Llama) runs the plain paths.
// FLYX_TEST_DEVICE_MEMORY=1: only that sessions and engines give their
// device memory back, and that sessions' round graphs stay inside what they
// lease (device_memory, graph_memory below); =graphs, =tiers: the latter
// alone, on any model (golden_test ASSET_DIR cuda VARIANT MODEL).
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <source_location>
#include <string>
#include <thread>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/grammar/grammar.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "core/session/session.h"
#include "nlohmann/json.hpp"

using namespace flyx;
using json = nlohmann::json;

namespace {

int g_fail = 0;
// The block drafter the tiny model holds (its golden file's "block_drafter"): DFlash2, or DSpine in place of it.
std::string g_block = "dflash2";
std::string block_label() { return g_block == "dspine" ? "DSpine" : "DFlash2"; }

void check(bool ok, const std::string& what) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

// Session::init, checked.  A session the engine refuses (no room left for it, as on a GPU other programs hold, or
// seq_new failed) has no sequence: a FAIL naming this file's line and why, and false.  generate refuses such a
// session, so a caller that goes on fails cleanly as well.
bool init_checked(Session& s, std::source_location at = std::source_location::current()) {
  std::string err;
  if (s.init(&err)) return true;
  check(false, "a session at golden_test.cpp:" + std::to_string(at.line()) + ": " + err);
  return false;
}

std::string str(const std::vector<int32_t>& v, size_t n = 12) {
  std::string s;
  for (size_t i = 0; i < v.size() && i < n; ++i) s += (i ? " " : "") + std::to_string(v[i]);
  return v.size() > n ? s + " ..." : s;
}

// Where two token lists part, for a failing check's message.
std::string first_difference(const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  if (i == got.size() && i == want.size()) return "the same tokens";
  return "token " + std::to_string(i) + " of " + std::to_string(want.size()) + ": got " +
         (i < got.size() ? std::to_string(got[i]) : "nothing") + ", want " +
         (i < want.size() ? std::to_string(want[i]) : "nothing");
}

struct Out {
  std::vector<int32_t> tokens;
  flyx_usage usage{};
};

// How many of a run's drafts are the reference's (`rounds`: the reference's rounds; the run's first result is its
// prefill's).
void drafts_equal(const std::vector<flyx_round_result>& trace, const json& rounds, int* same, int* total) {
  *same = *total = 0;
  for (size_t i = 1; i < trace.size() && i - 1 < rounds.size(); ++i) {
    const auto d = rounds[i - 1]["drafts"].get<std::vector<int32_t>>();
    for (int j = 0; j < trace[i].n_drafts && j < static_cast<int>(d.size()); ++j, ++*total)
      *same += trace[i].drafts[j] == d[size_t(j)];
  }
}

// Generate n tokens, greedy unless sampling is given, ignoring end of sequence.
Out gen(Session& s, const std::vector<int32_t>& prompt, int n, std::vector<flyx_round_result>* trace = nullptr,
        const SamplingParams& sampling = {}) {
  Out o;
  GenerateRequest rq;
  rq.prompt = prompt;
  rq.max_new_tokens = n;
  rq.ignore_eos = true;
  rq.sampling = sampling;
  GenerateResult res;
  std::string err;
  s.trace = trace;
  const TextSink collect = [&](std::string_view, const int32_t* t, int nt) {
    o.tokens.insert(o.tokens.end(), t, t + nt);
    return true;
  };
  if (generate(s, rq, collect, &res, &err) != 0) {
    std::printf("  generate failed: %s\n", err.c_str());
    ++g_fail;
  }
  s.trace = nullptr;
  o.usage = res.usage;
  return o;
}

// Compare all serialized bytes: the K/V and their scales, wrapped draft caches, recurrent state and the hidden row.
// Matching the next greedy token alone would not detect a missing draft cache or a wrong offset in a delta.
void snapshot_delta(Engine* e, Session& source, Session& dest, const std::vector<int32_t>& prompt,
                    const std::string& what) {
  const auto* api = e->api();
  auto save = [&](flyx_seq* seq, int begin, int pos, std::vector<uint8_t>* out) {
    uint64_t bytes = 0;
    if (api->snapshot_bytes(seq, 0, begin, &bytes) != 0) return false;
    out->resize(size_t(bytes));
    void* p = out->data();
    flyx_host_pieces pieces{&p, 1, 0, std::max<uint64_t>(4096, bytes)};
    return api->snapshot_save(seq, 0, begin, &pieces, bytes) == pos;
  };
  auto load = [&](std::vector<uint8_t>& data, int anchor) {
    void* p = data.data();
    flyx_host_pieces pieces{&p, 1, 0, std::max<size_t>(4096, data.size())};
    return api->snapshot_load(dest.seq(), &pieces, data.size(), anchor);
  };
  auto fill = [&](int begin, int end) {
    while (begin < end) {
      const int n = std::min(end - begin, e->prefill_chunk());
      if (api->prefill(e->prefill_plan(), source.seq(), prompt.data() + begin, n, nullptr) != 0) return false;
      begin += n;
    }
    return true;
  };
  std::vector<uint8_t> parent, delta, full, restored;
  const int first = 137, last = 301;
  bool ok = api->rewind(source.seq(), 0, prompt[0]) >= 0 && fill(0, first);
  const int a = ok ? api->checkpoint(source.seq(), 0) : -1;
  ok = a > 0 && save(source.seq(), 0, a, &parent) && fill(first, last);
  const int b = ok ? api->checkpoint(source.seq(), 0) : -1;
  ok = b > a && save(source.seq(), a, b, &delta) && save(source.seq(), 0, b, &full) && delta.size() < full.size();
  ok = ok && api->rewind(dest.seq(), 0, prompt[0]) >= 0 && load(delta, prompt[b]) < 0 &&
       load(parent, prompt[a]) == a && load(delta, prompt[b]) == b && api->checkpoint(dest.seq(), 0) == b &&
       save(dest.seq(), 0, b, &restored) && restored == full;
  check(ok, what + ": a parent plus its delta restores every checkpoint byte");
}

// A checkpoint through host memory: a session's kept in a store and loaded into a fresh session,
// which then decodes what a session that prefilled everything decodes, and drafts what it drafts (the drafters'
// caches go with the snapshot); `other`, a model of another state layout, refuses it.  A prompt long enough for a
// checkpoint worth keeping.
void snapshot_through_host(Engine* e, Engine* other, const std::vector<int32_t>& seed, const std::string& what,
                           bool every_draft) {
  std::vector<int32_t> prompt;
  while (prompt.size() < 320) prompt.insert(prompt.end(), seed.begin(), seed.end());
  std::string err;
  Session a(e), loaded(e), fresh(e), foreign(other);
  init_checked(a);
  init_checked(loaded);
  init_checked(fresh);
  init_checked(foreign);
  gen(a, prompt, 4);  // its checkpoint at the end of the prompt
  SnapshotStore store(uint64_t(1) << 30);
  std::string kept;
  const int held = a.save_to(store, &kept);
  check(held >= 256 && store.size() == 1, what + ": a checkpoint kept in host memory (" + kept + ")");
  auto longer = prompt;
  longer.insert(longer.end(), seed.begin(), seed.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(8, seed.size())));
  loaded.set_store(&store);
  std::vector<flyx_round_result> ta, tb;
  const Out oa = gen(loaded, longer, 16, &ta), ob = gen(fresh, longer, 16, &tb);
  int same = 0, total = 0;
  for (size_t i = 1; i < ta.size() && i < tb.size(); ++i)
    for (int j = 0; j < ta[i].n_drafts && j < tb[i].n_drafts; ++j, ++total) same += ta[i].drafts[j] == tb[i].drafts[j];
  check(oa.tokens == ob.tokens && oa.usage.reused_tokens == held,
        what + ": loaded into a fresh session, the tokens of one that prefilled it all (" +
            std::to_string(oa.usage.reused_tokens) + " of " + std::to_string(longer.size()) + " reused)");
  check(total > 0 && (every_draft ? same == total : same * 5 >= total * 4),
        what + ":   the drafts of that session: " + std::to_string(same) + " / " + std::to_string(total));
  const SnapshotStore::Entry* s = store.find(longer, longer.size(), 0);
  const flyx_host_pieces pieces = s ? s->view() : flyx_host_pieces{};
  check(s && other->api()->snapshot_load(foreign.seq(), &pieces, s->bytes, longer[0]) < 0,
        what + ": a model of another state layout refuses it");
  // The store's memory: pieces of the backend's (page-locked) when it gives some, reused when a snapshot replaces
  // another, and the store's own before an unload (own_memory), the snapshot loading the same from there.
  const uint64_t reserved = store.reserved();
  check(s && reserved >= s->bytes && reserved <= store.budget(), what + ": the store's memory within its budget");
  a.save_to(store, &kept);
  check(store.size() == 1 && store.reserved() == reserved, what + ": the same tokens again reuse the pieces");
  store.own_memory();
  s = store.find(longer, longer.size(), 0);
  Session again(e);
  init_checked(again);
  again.set_store(&store);
  const Out oc = gen(again, longer, 16);
  check(s && !s->backend_owned() && oc.tokens == ob.tokens && oc.usage.reused_tokens == held,
        what + ": in the store's own memory, it loads the same");
  // The disk tier: a store with memory for one snapshot sends the older of two to disk; a session
  // continuing it reads it back (sending the other there) and gets the tokens of one that prefilled it all.
  {
    SnapshotStore small(SnapshotStore::kPieceBytes);
    const auto dir = std::filesystem::temp_directory_path() / "flyx-golden-disk";
    std::string why;
    check(small.set_disk(os::from_path(dir), uint64_t(1) << 30, &why), what + ": a disk tier (" + why + ")");
    auto other = prompt;
    other[0] = (other[0] + 5) % 300;
    Session sa(e), sb(e), back(e);
    init_checked(sa);
    init_checked(sb);
    init_checked(back);
    gen(sa, prompt, 4);
    gen(sb, other, 4);
    sa.save_to(small, &why);
    sb.save_to(small, &why);
    small.settle();
    check(small.size() == 2 && small.on_disk() == 1, what + ": of two snapshots, the older on disk (" +
                                                         std::to_string(small.on_disk()) + " of " +
                                                         std::to_string(small.size()) + ")");
    back.set_store(&small);
    const Out od = gen(back, longer, 16);
    small.settle();
    check(od.tokens == ob.tokens && od.usage.reused_tokens == held && small.on_disk() == 1,
          what + ": read back from disk, the same tokens (" + std::to_string(od.usage.reused_tokens) + " reused)");
  }
  // Larger than the store's memory: straight to disk a few pieces at a time (null pieces skipped,
  // ABI 31), and loaded from there the same way, the file kept.  No memory at all, and pieces of 4096 bytes: the
  // snapshot takes many calls each way.
  {
    SnapshotStore none(0, 4096);
    const auto dir = std::filesystem::temp_directory_path() / "flyx-golden-direct";
    std::string why;
    check(none.set_disk(os::from_path(dir), uint64_t(1) << 30, &why), what + ": a disk tier alone (" + why + ")");
    Session sa(e), back(e);
    init_checked(sa);
    init_checked(back);
    gen(sa, prompt, 4);
    const int on = sa.save_to(none, &why);
    const SnapshotStore::Entry* d = none.find(longer, longer.size(), 0);
    const uint64_t bytes = d ? d->bytes : 0;
    check(on == held && d && none.on_disk() == none.size() && bytes > 8 * SnapshotStore::kBounce * 4096 &&
              why.find("on disk") != std::string::npos,
          what + ": larger than the store's memory, to disk " + std::to_string(bytes / 4096 + 1) + " pieces of 4096 (" +
              why + ")");
    back.set_store(&none);
    const Out od = gen(back, longer, 16);
    check(od.tokens == ob.tokens && od.usage.reused_tokens == held && none.on_disk() == none.size() &&
              none.bytes() == 0,
          what + ": loaded from disk a few pieces at a time, the same tokens (" +
              std::to_string(od.usage.reused_tokens) + " reused)");
    if (e->backend_name() == "cpu_ref") {  // every prompt end is a distinct checkpoint on this backend
      const int next = back.save_to(none, &why);
      auto follow = longer;
      follow.push_back(seed.front());
      const auto* child = none.find(follow, follow.size() - 1, held);
      check(child && d && child->parent == d->disk_id && next > held,
            what + ": a longer disk checkpoint shares the earlier range");
      if (child) {
        // Keep the expected size so discovery succeeds, then fail the actual load after its parent has been read.
        std::fstream file(os::to_path(child->file), std::ios::binary | std::ios::in | std::ios::out);
        const int magic = file.get();
        file.seekp(0);
        file.put(static_cast<char>(magic ^ 0xff));
        file.close();
        check(magic != EOF && file.good(), what + ": the child header was changed");
        Session recover(e), cold(e);
        init_checked(recover);
        init_checked(cold);
        recover.set_store(&none);
        const Out got = gen(recover, follow, 4), want = gen(cold, follow, 4);
        check(got.usage.reused_tokens == held && got.tokens == want.tokens,
              what + ": an unreadable child falls back to its intact prefix and yields the cold tokens (" +
                  std::to_string(got.usage.reused_tokens) + " reused)");
      }
    }
  }
  snapshot_delta(e, a, loaded, prompt, what);
}

// Rounds of several sequences: sequences, each at the end of its own prompt, run rounds together
// (launch_batch) with the plans given, whose drafters and widths may differ; each gets, round after round, the ring
// entries (tokens, drafts, how many accepted, position) the same rounds give it alone.
void batch_rounds(Engine* e, const std::vector<std::vector<int32_t>>& prompts, const std::vector<size_t>& plan_of,
                  const std::string& what) {
  const auto* api = e->api();
  const size_t n = prompts.size();
  std::string err;
  std::vector<std::unique_ptr<Session>> alone, together;
  std::vector<flyx_plan*> plans;
  std::vector<flyx_seq*> seqs;
  for (size_t i = 0; i < n; ++i) {
    alone.push_back(std::make_unique<Session>(e));
    together.push_back(std::make_unique<Session>(e));
    if (!init_checked(*alone.back()) || !init_checked(*together.back())) return;
    gen(*alone.back(), prompts[i], 1);  // at the end of its prompt, its first token the next input
    gen(*together.back(), prompts[i], 1);
    plans.push_back(e->round_plan(plan_of[i]));
    seqs.push_back(together.back()->seq());
  }
  flyx_gen_params g{};
  g.top_p = 1.f;
  const std::vector<const flyx_gen_params*> gens(n, &g);
  int same = 0, total = 0;
  bool ok = true;
  for (int r = 0; r < 6 && ok; ++r) {
    std::vector<flyx_round_result> a(n), b(n);
    for (size_t i = 0; i < n; ++i)
      ok = ok && api->launch(plans[i], alone[i]->seq(), &g) == 0 && api->read(alone[i]->seq(), &a[i]) == 0;
    ok = ok && api->launch_batch(plans.data(), seqs.data(), gens.data(), static_cast<int32_t>(n)) == 0;
    for (size_t i = 0; i < n; ++i) ok = ok && api->read(seqs[i], &b[i]) == 0;
    if (!ok) {
      std::printf("  %s: %s\n", what.c_str(), api->last_error());
      break;
    }
    for (size_t i = 0; i < n; ++i, ++total) {
      const flyx_round_result &x = a[i], &y = b[i];
      same += x.n_tokens == y.n_tokens && x.n_accepted == y.n_accepted && x.position == y.position &&
              x.stopped == y.stopped && x.n_drafts == y.n_drafts &&
              std::equal(x.tokens, x.tokens + x.n_tokens, y.tokens) &&
              std::equal(x.drafts, x.drafts + x.n_drafts, y.drafts);
    }
  }
  check(ok && total > 0 && same == total, what + ": each sequence gets what it gets alone (" + std::to_string(same) +
                                              " of " + std::to_string(total) + " ring entries)");
}

using TreePlans = std::vector<std::pair<int, flyx_plan*>>;

// Given trees for tree_rounds: the engine's (Engine::tree_widths(): 3, 7 and 15 drafts; none with engine_trees
// false) and, lowered here, trees of `wider` drafts (ascending; rounds of more than FLYX_MAX_ROWS rows, on an
// engine with wide rows: CUDA's backend option wide=1): no engine lowers those of its own.  The plans lowered here are
// freed with it, so the sessions that ran them go first.
class GivenTrees {
 public:
  GivenTrees(Engine* e, const std::vector<int>& wider, bool engine_trees = true) : e_(e) {
    if (engine_trees)
      for (size_t i = 0; i < e->tree_widths().size(); ++i) plans_.emplace_back(e->tree_widths()[i], e->tree_plan(i));
    for (int w : wider) {
      auto program = e->model().round_program(w, -1, true);
      for (auto& n : program)
        if (n.kind == FLYX_NODE_GIVEN) n.arg1 = FLYX_GIVEN_TREE;
      const flyx_program rp{FLYX_PHASE_ROUND, w, w + 1, static_cast<int32_t>(program.size()), program.data()};
      flyx_plan* p = nullptr;
      if (e->api()->lower(e->bmodel(), &rp, &p) != 0) {
        refused_ += (refused_.empty() ? "" : "; ") + std::to_string(w) + " drafts: " + e->api()->last_error();
        continue;
      }
      plans_.emplace_back(w, p);
      own_.push_back(p);
    }
  }
  ~GivenTrees() {
    for (flyx_plan* p : own_) e_->api()->plan_free(p);
  }
  GivenTrees(const GivenTrees&) = delete;
  GivenTrees& operator=(const GivenTrees&) = delete;
  const TreePlans& plans() const { return plans_; }
  bool lowered() const { return refused_.empty(); }  // every one of `wider`
  // The widths the backend did not lower, and why, for a check's message: " (refused: 63 drafts: ...)"; empty: none.
  std::string refusal() const { return refused_.empty() ? "" : " (refused: " + refused_ + ")"; }

 private:
  Engine* e_;
  TreePlans plans_;
  std::vector<flyx_plan*> own_;
  std::string refused_;
};

// How tree_rounds runs its rounds, besides the request's sampling.
struct TreeRun {
  const TreePlans* trees = nullptr;   // the widths and plans to take (null: the engine's, Engine::tree_widths())
  float min_p = 0.f;                  // the relaxed rules' min-p (gen.accept_min_p)
  std::vector<int>* paths = nullptr;  // the paths' lengths, round by round: recorded, or given with path_first
  bool path_first = false;            // the path the first rows of each tree, of the lengths given
  flyx_plan* then = nullptr;          // a drafter's plan, `after` rounds of it after the trees
  int after = 0;
  // Every round launched before the one before it is read (two in flight, as the engine runs them), with its drafts
  // where that round's hidden path leaves the sequence: the ring of pinned slots a round's drafts and tree go
  // through, the drafts written at their positions behind a round not yet run, and the rows a round in flight may
  // move the sequence on by, with real tokens.
  bool two_in_flight = false;
};

// Tree rounds (docs/design/tree-verification.md): rounds that verify a tree of given drafts, plain decoding's
// continuation hidden on a random path of a random tree (the other drafts never the token decoding chooses at their
// depth), emit plain decoding's tokens and accept exactly that path, round after round; sampling and penalties
// included, as each row's choice is its path's decoded alone.  The first rounds take every tree width once, widest
// first (trees of more than FLYX_MAX_ROWS rows where run.trees has them: GEMMs and the head 16 rows at a time), the
// rounds after them trees of random widths.  With run.paths, the paths' lengths, round by round: with path_first the
// path is the first rows of each tree, of the lengths given (its rows need no moving after the accept), else they
// are recorded there.  Returns the ring entries of run.after rounds of plan run.then (a drafter's) launched after
// the trees, for comparing what the drafters see; empty on failure.
std::vector<flyx_round_result> tree_rounds(Engine* e, const std::vector<int32_t>& prompt, const SamplingParams& sp,
                                           const std::string& what, const TreeRun& run = {}) {
  std::vector<flyx_round_result> tail;
  TreePlans engine_trees;
  for (size_t i = 0; i < e->tree_widths().size(); ++i) engine_trees.emplace_back(e->tree_widths()[i], e->tree_plan(i));
  const TreePlans* trees = run.trees ? run.trees : &engine_trees;
  if (trees->empty()) {
    check(false, what + ": no tree rounds");
    return tail;
  }
  const auto* api = e->api();
  const int n_new = 64;
  std::string err;
  auto generated = [&](Session& s, int n, std::vector<int32_t>* out) {
    GenerateRequest rq;
    rq.prompt = prompt;
    rq.max_new_tokens = n;
    rq.ignore_eos = true;
    rq.sampling = sp;
    GenerateResult res;
    const TextSink collect = [&](std::string_view, const int32_t* t, int nt) {
      out->insert(out->end(), t, t + nt);
      return true;
    };
    return generate(s, rq, collect, &res, &err) == 0;
  };
  std::vector<int32_t> want, first;
  Session ref(e), s(e);
  if (!ref.init(&err) || !s.init(&err) || !generated(ref, n_new + 2 * FLYX_MAX_WIDE_ROWS, &want) ||
      !generated(s, 1, &first) || first.empty() || first[0] != want[0]) {
    check(false, what + ": plain decoding and the first token (" + err + ")");
    return tail;
  }
  flyx_gen_params g{};
  g.temperature = sp.temperature;
  g.top_p = sp.top_p;
  g.top_k = sp.top_k;
  g.seed = sp.seed;
  g.presence_penalty = sp.presence_penalty;
  g.frequency_penalty = sp.frequency_penalty;
  g.accept_min_p = run.min_p;
  const int vocab = e->model().desc().vocab;
  uint32_t rng = 12345u, shape = 777u;  // shape: the trees' widths alone, the same with path_first
  auto draw = [](uint32_t* r, uint32_t n) {
    *r = *r * 1664525u + 1013904223u;
    return (*r >> 8) % n;
  };
  auto next = [&](uint32_t n) { return draw(&rng, n); };
  std::vector<int32_t> got = {want[0]};
  int launched = 0, rounds = 0, as_walked = 0, accepted = 0, off_first = 0;
  // The rounds launched and not yet read, each one's hidden path's length; `at`: the tokens known once they are read,
  // each having accepted its path.
  std::deque<int> in_flight;
  size_t at = got.size();
  auto read_one = [&]() {
    flyx_round_result r{};
    if (api->read(s.seq(), &r) != 0) return false;
    ++rounds;
    as_walked += r.n_accepted == in_flight.front();
    accepted += r.n_accepted;
    got.insert(got.end(), r.tokens, r.tokens + r.n_tokens);
    in_flight.pop_front();
    return true;
  };
  bool ok = true;
  const size_t n_widths = trees->size();
  while (ok && at < size_t(n_new)) {
    const size_t wi = size_t(launched) < n_widths ? n_widths - 1 - size_t(launched) : draw(&shape, uint32_t(n_widths));
    const int W = (*trees)[wi].first;
    // The path to a random node of a random tree; the drafts on it are what decoding chooses there.
    const size_t nw = static_cast<size_t>(W);
    std::vector<int32_t> parents(nw), depth(nw), drafts(nw);
    const int L = run.path_first ? std::min(W, (*run.paths)[size_t(launched) % run.paths->size()]) : 0;
    for (int i = 0; i < W; ++i) parents[size_t(i)] = run.path_first && i < L ? i - 1 : int(next(uint32_t(i + 1))) - 1;
    for (int i = 0; i < W; ++i) depth[size_t(i)] = parents[size_t(i)] < 0 ? 1 : depth[size_t(parents[size_t(i)])] + 1;
    int deepest = 0;  // half the trees hide their deepest path
    for (int i = 0; i < W; ++i)
      if (depth[size_t(i)] > depth[size_t(deepest)]) deepest = i;
    const int x = run.path_first ? L - 1 : next(2) ? deepest : int(next(uint32_t(W)));  // the path's last node
    if (run.paths && !run.path_first) run.paths->push_back(depth[size_t(x)]);
    std::vector<bool> on(nw, false);
    for (int c = x; c >= 0; c = parents[size_t(c)]) on[size_t(c)] = true;
    const size_t base = in_flight.empty() ? got.size() : at;  // want[base + d - 1]: decoding's token at depth d
    for (int i = 0; i < W; ++i) {
      const int32_t truth = want[base + size_t(depth[size_t(i)]) - 1];
      int32_t t = int32_t(next(uint32_t(vocab)));
      if (t == truth) t = (t + 1) % vocab;
      drafts[size_t(i)] = on[size_t(i)] ? truth : t;
      off_first += on[size_t(i)] && i + 1 != depth[size_t(i)];
    }
    g.drafts = drafts.data();
    g.n_drafts = W;
    g.drafts_at = static_cast<int32_t>(prompt.size() + base);
    g.parents = parents.data();
    ok = api->launch((*trees)[wi].second, s.seq(), &g) == 0;
    if (ok) {
      ++launched;
      in_flight.push_back(depth[size_t(x)]);
      at = base + size_t(depth[size_t(x)]) + 1;
      if (in_flight.size() == (run.two_in_flight ? size_t(FLYX_MAX_IN_FLIGHT) : 1)) ok = read_one();
    }
  }
  while (ok && !in_flight.empty()) ok = read_one();
  if (!ok) {
    check(false, what + ": " + api->last_error());
    return tail;
  }
  got.resize(std::min(got.size(), want.size()));
  check(got == std::vector<int32_t>(want.begin(), want.begin() + std::ptrdiff_t(got.size())),
        what + ": " + std::to_string(rounds) + " tree rounds give plain decoding's tokens (" +
            first_difference(got, std::vector<int32_t>(want.begin(), want.begin() + std::ptrdiff_t(got.size()))) + ")");
  check(as_walked == rounds, what + ":   each accepts its hidden path (" + std::to_string(as_walked) + " of " +
                                 std::to_string(rounds) + ", " + std::to_string(accepted) + " drafts, " +
                                 std::to_string(off_first) + " path rows off the first chain; widths up to " +
                                 std::to_string(trees->back().first) + (run.two_in_flight ? ", two in flight" : "") +
                                 ")");
  // Rewinding into the last tree round refolds the replay rows its accept moved into place: plain rounds from there
  // go on as plain decoding (without penalties: a rewind does not take tokens back out of the reply's counts).
  size_t plain_plan = e->widths().size();
  for (size_t i = 0; i < e->widths().size(); ++i)
    if (e->widths()[i] == 0) plain_plan = i;
  if (!run.then && plain_plan < e->widths().size() && sp.presence_penalty == 0 && sp.frequency_penalty == 0 &&
      got.size() >= 2) {
    const int32_t pos = static_cast<int32_t>(prompt.size() + got.size()) - 2;
    bool rewound = api->rewind(s.seq(), pos, got[got.size() - 2]) == 1;
    flyx_gen_params pg = g;
    pg.drafts = nullptr;
    pg.n_drafts = 0;
    pg.parents = nullptr;
    std::vector<int32_t> again;
    for (int k = 0; rewound && k < 4; ++k) {
      flyx_round_result r{};
      rewound = api->launch(e->round_plan(plain_plan), s.seq(), &pg) == 0 && api->read(s.seq(), &r) == 0;
      again.insert(again.end(), r.tokens, r.tokens + r.n_tokens);
    }
    const auto from = want.begin() + std::ptrdiff_t(got.size()) - 1;
    check(rewound && again.size() == 4 && std::equal(again.begin(), again.end(), from),
          what + ":   rewound into the last tree round, plain rounds go on as plain decoding");
  }
  flyx_gen_params plain{};
  plain.temperature = sp.temperature;
  plain.top_p = sp.top_p;
  plain.top_k = sp.top_k;
  plain.seed = sp.seed;
  plain.presence_penalty = sp.presence_penalty;
  plain.frequency_penalty = sp.frequency_penalty;
  for (int i = 0; i < run.after && run.then; ++i) {
    flyx_round_result r{};
    if (api->launch(run.then, s.seq(), &plain) != 0 || api->read(s.seq(), &r) != 0) break;
    tail.push_back(r);
  }
  return tail;
}

constexpr int kReferenceWidth = -2;

std::unique_ptr<Engine> make(const std::string& path, const std::string& backend, bool spec,
                             const std::string& drafter = g_block, bool kv_fp8 = false,
                             const std::string& backend_options = "", bool lookup = false, int max_ctx = 512,
                             int width = kReferenceWidth, bool draft_trees = false, int kv_gpu = -1,
                             const std::string& kv_method = "exact", int kv_copy_gpu = -1, int prefill_chunk = 16,
                             int adaptive = -1) {
  EngineOptions o;
  if (adaptive >= 0) {  // adaptive stopping on (1) or off (0); else FLYX_ADAPTIVE's, on unless 0
    o.adaptive = adaptive == 1;
    o.adaptive_given = true;
  }
  o.kv_gpu = kv_gpu;  // K/V past it in host memory: 1 GB of it at most here
  o.kv_copy_gpu = kv_copy_gpu;  // the strata copy of the layers past it in host memory too
  o.kv_ram = kv_gpu >= 0 ? 1'000'000'000 : 0;
  o.kv_method = kv_method;
  o.lookup = lookup;  // off unless asked: the reference drafts are the drafter's own
  o.draft_trees = draft_trees;
  o.backend_options = backend_options;
  o.kv = kv_fp8 ? FLYX_KV_FP8 : FLYX_KV_16;
  o.max_ctx = max_ctx;
  o.backend = backend;
  o.speculative = spec;
  o.drafter = drafter;
  o.prefill_chunk = prefill_chunk;  // several chunks per prompt
  // fixed widths, as in the reference rounds (MTP 3, DFlash2 15), unless given (-1: chosen every round)
  o.width = width != kReferenceWidth ? width : !spec ? 0 : drafter == "mtp" ? 3 : 15;
  auto e = std::make_unique<Engine>();
  std::string err;
  if (!e->load(path, o, &err)) {
    std::printf("load failed: %s\n", err.c_str());
    std::exit(1);
  }
  return e;
}

void prefill_only_checks(Engine* e, const std::vector<int32_t>& base) {
  const int align = e->prefill_align();
  const int boundary = (64 + align - 1) / align * align;
  std::string err;
  for (int n : {1, 7, e->prefill_chunk(), boundary, boundary + 3}) {
    GenerateRequest rq;
    for (int i = 0; i < n; ++i) rq.prompt.push_back(base[size_t(i) % base.size()]);
    rq.prefill_only = true;
    rq.stop = {":"};
    for (bool extend : {false, true}) {
      Session warm(e), cold(e);
      if (!init_checked(warm) || !init_checked(cold)) continue;
      GenerateResult result;
      std::vector<flyx_round_result> trace;
      warm.trace = &trace;
      int output = 0;
      const auto sink = [&](std::string_view, const int32_t*, int count) { output += count; return true; };
      const bool ok = generate(warm, rq, sink, &result, &err) == 0;
      warm.trace = nullptr;
      check(ok && trace.empty() && output == 0 && result.usage.new_tokens == 0 && result.usage.rounds == 0 &&
                result.usage.decode_ms == 0 && result.stop_token == -1 && result.usage.finish_reason == FLYX_FINISH_LENGTH,
            "prefill only: no chosen token or decoding rounds");
      auto next = rq.prompt;
      if (extend) next.insert(next.end(), base.begin(), base.begin() + std::min(size_t(5), base.size()));
      const auto after = gen(warm, next, 8);
      const int want_reuse = extend ? n : std::max(0, n - e->prefill_chunk() - align);
      check(after.tokens == gen(cold, next, 8).tokens && after.usage.reused_tokens >= want_reuse,
            "prefill only: " + std::string(extend ? "continued" : "identical") + " prompt at " +
                std::to_string(n) + " reuses " + std::to_string(after.usage.reused_tokens) + " prompt tokens");
    }
  }
  {
    Session warm(e), cold(e);
    init_checked(warm);
    init_checked(cold);
    std::vector<int32_t> prompt;
    for (int i = 0; i < boundary; ++i) prompt.push_back(base[size_t(i) % base.size()]);
    gen(warm, prompt, 5);
    const int decoded = warm.state_pos();
    GenerateRequest warm_tail;
    warm_tail.prefill_only = true;
    warm_tail.prompt = warm.tokens();
    warm_tail.prompt.resize(size_t(decoded / align * align + e->prefill_chunk()), base.front());
    GenerateResult result;
    const bool ok = generate(warm, warm_tail, [](std::string_view, const int32_t*, int) { return true; },
                             &result, &err) == 0;
    // A one-pass prewarm continued the previous decode. Edit inside that pass, beyond the old rollback window.
    // The last checkpoint before the edit belongs to the original prompt.
    auto edit = warm_tail.prompt;
    edit[size_t(decoded + 4)] = (edit[size_t(decoded + 4)] + 1) % e->model().desc().vocab;
    const auto got = gen(warm, edit, 8), want = gen(cold, edit, 8);
    check(ok && decoded > boundary && result.usage.reused_tokens == decoded &&
              got.usage.reused_tokens == boundary && got.tokens == want.tokens,
          "a prewarm extends the live decode at " + std::to_string(decoded) + " but an edit restores the prompt at " +
              std::to_string(boundary) + " (" + std::to_string(got.usage.reused_tokens) + " reused)");
  }
  Session s(e);
  if (!init_checked(s)) return;
  GenerateRequest rq;
  rq.prefill_only = true;
  rq.prompt = base;
  std::atomic<bool> cancel{true};
  rq.cancel = &cancel;
  GenerateResult result;
  check(generate(s, rq, [](std::string_view, const int32_t*, int) { return true; }, &result, &err) == 0 &&
            result.usage.finish_reason == FLYX_FINISH_CANCEL && result.usage.new_tokens == 0,
        "a cancelled prefill-only request stops before generating");
}

// A session the engine refused (Session::init failed: no room left for it, as on a GPU other programs hold, or
// seq_new failed) has no sequence: generate refuses it with init's reason, and the backend's calls refuse a null
// sequence, none of them a crash.  The room is taken here by a lease of all the ledger has left; the reference
// backend's ledger has no budget, so its session is one never initialized.
void refused_session(Engine* e, const std::vector<int32_t>& prompt) {
  const bool budget = e->ledger().budget() > 0;
  std::string why, refused;
  Lease all = budget ? e->ledger().reserve("test", "all that is left", e->ledger().available(), &why) : Lease{};
  Session s(e);
  const bool made = budget && s.init(&refused);
  all.release();
  GenerateRequest rq;
  rq.prompt = prompt;
  rq.max_new_tokens = 4;
  rq.ignore_eos = true;
  GenerateResult res;
  std::string err;
  const bool turned_away = !made && generate(s, rq, [](std::string_view, const int32_t*, int) { return true; },
                                             &res, &err) != 0;
  check(turned_away && err.find(budget ? refused : "init was not called") != std::string::npos,
        "a session the engine refused (" + (budget ? refused : std::string("never initialized")) +
            "): generate refuses it (" + err + ")");
  if (e->backend_name() != "cuda") return;
  const auto* api = e->api();
  flyx_gen_params g{};
  g.top_p = 1.f;
  flyx_round_result r{};
  uint64_t bytes = 0;
  float logit = 0;
  flyx_plan* plans[2] = {e->round_plan(0), e->round_plan(0)};
  flyx_seq* none[2] = {nullptr, nullptr};
  const flyx_gen_params* gens[2] = {&g, &g};
  void* piece = &logit;
  const flyx_host_pieces pieces{&piece, 1, 0, sizeof logit};
  const bool every = api->prefill(e->prefill_plan(), nullptr, prompt.data(), 1, nullptr) < 0 &&
                     api->launch(plans[0], nullptr, &g) < 0 && api->launch_batch(plans, none, gens, 2) < 0 &&
                     api->read(nullptr, &r) < 0 && api->logits(nullptr, 0, &logit) < 0 && api->ready(nullptr) < 0 &&
                     api->set_reply(nullptr, prompt.data(), 1) < 0 && api->rewind(nullptr, 0, prompt[0]) < 0 &&
                     api->checkpoint(nullptr, 0) < 0 && api->restore(nullptr, 0, prompt[0]) < 0 &&
                     api->snapshot_bytes(nullptr, 0, 0, &bytes) < 0 &&
                     api->snapshot_save(nullptr, 0, 0, &pieces, sizeof logit) < 0 &&
                     api->snapshot_load(nullptr, &pieces, sizeof logit, prompt[0]) < 0 &&
                     api->set_oracle(nullptr, prompt.data(), 1) < 0;
  check(every, std::string("every backend call on no sequence refused (") + api->last_error() + ")");
}

// A tree round of more than FLYX_MAX_ROWS rows in flight before strata attention's switch (auto): a
// round launched behind it takes its phase from the rows the rounds in flight may move the sequence on by (CUDA:
// in_flight_rows(), a tree round's own rows), and a tree round past 16 rows reads no strata copy.  With the switch
// past what a round of 64 rows and one of 16 can reach and within what two of 64 can, a chain of 63 drafts (plain
// decoding's tokens) runs exact, a second one launched behind it is refused, and plain rounds after it, across the
// switch, go on as plain decoding on the same engine.
void tree_round_before_strata(const std::string& model, const std::string& backend,
                              const std::vector<int32_t>& prompt) {
  const int p = static_cast<int>(prompt.size()), W = FLYX_MAX_WIDE_ROWS - 1, after = 3 * FLYX_MAX_ROWS + 8;
  const int from = p + FLYX_MAX_WIDE_ROWS + 3 * FLYX_MAX_ROWS;
  // As make() would (plain decoding, binary16 K/V), loaded here: a model whose attention shape has no strata kernels
  // is refused at load (its session is sized there), which make() would end the test on.
  EngineOptions o;
  o.kv_gpu = o.kv_copy_gpu = -1;
  o.kv_ram = 0;
  o.kv_method = "auto";
  o.backend_options = "wide=1,strata_denom=8,strata_least=1,strata_from=" + std::to_string(from);
  o.kv = FLYX_KV_16;
  o.max_ctx = 512;
  o.backend = backend;
  o.speculative = false;
  o.prefill_chunk = 16;
  o.width = 0;
  auto e = std::make_unique<Engine>();
  std::string err;
  if (!e->load(model, o, &err)) {
    if (err.find("not compiled for") != std::string::npos) {  // the model's attention shape has no strata kernels
      std::printf("  --   a tree round before strata's switch: %s\n", err.c_str());
      return;
    }
    check(false, "a tree round before strata's switch: load: " + err);
    return;
  }
  if (e->kv_layout().method == FLYX_KV_READ_ALL) {  // a file that holds DV: every position read, no switch
    std::printf("  --   a tree round before strata's switch: the file reads every position (it holds DV)\n");
    return;
  }
  const GivenTrees chain(e.get(), {W}, false);
  Session ref(e.get()), s(e.get());
  if (!chain.lowered() || !ref.init(&err) || !s.init(&err)) {
    check(false, "a tree round before strata's switch: the plan and sessions" + chain.refusal() + " " + err);
    return;
  }
  const std::vector<int32_t> want = gen(ref, prompt, W + 2 + after).tokens, first = gen(s, prompt, 1).tokens;
  if (want.size() != size_t(W + 2 + after) || first.size() != 1 || first[0] != want[0]) {
    check(false, "a tree round before strata's switch: plain decoding and the first token");
    return;
  }
  std::vector<int32_t> drafts(want.begin() + 1, want.begin() + 1 + W), parents(drafts.size());
  for (int i = 0; i < W; ++i) parents[size_t(i)] = i - 1;
  flyx_gen_params g{};
  g.top_p = 1.f;
  g.drafts = drafts.data();
  g.n_drafts = W;
  g.drafts_at = p + 1;
  g.parents = parents.data();
  const auto* api = e->api();
  flyx_plan* tree = chain.plans()[0].second;
  flyx_round_result r{};
  const bool ran = api->launch(tree, s.seq(), &g) == 0;
  const bool refused = ran && api->launch(tree, s.seq(), &g) != 0;
  std::vector<int32_t> got = first;
  bool read = ran && api->read(s.seq(), &r) == 0;
  if (read) got.insert(got.end(), r.tokens, r.tokens + r.n_tokens);
  check(ran && refused && read && r.n_accepted == W,
        "a tree round of " + std::to_string(W + 1) + " rows before strata's switch runs (" +
            std::to_string(r.n_accepted) + " drafts accepted); one launched behind it, whose rows it may move past "
            "the switch, is refused");
  size_t plain = e->widths().size();
  for (size_t i = 0; i < e->widths().size(); ++i)
    if (e->widths()[i] == 0) plain = i;
  flyx_gen_params pg{};
  pg.top_p = 1.f;
  for (int k = 0; read && plain < e->widths().size() && k < after; ++k) {
    read = api->launch(e->round_plan(plain), s.seq(), &pg) == 0 && api->read(s.seq(), &r) == 0;
    got.insert(got.end(), r.tokens, r.tokens + r.n_tokens);
  }
  check(read && got == want, "  plain rounds after it, across the switch, give plain decoding's tokens (" +
                                 first_difference(got, want) + ")");
}

// Adaptive stopping (FLYX_ADAPTIVE) on and off, every drafter the file has, its rounds chosen every round: the same
// tokens, greedy and sampled (gate 4).  A sequential drafter's chains (MTP) take an adaptive plan where the backend
// lowers one (CUDA 12.4 and graphs); a block drafter (DFlash2) and DV, which drafts trees, take none
// (Model::adaptive_checkpoints), so beside DV alone adaptive stopping is inactive.  Off, no drafter has a plan.
void adaptive_stopping(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt) {
  auto on = make(model, backend, true, "all", false, "", false, 512, -1, false, -1, "exact", -1, 16, 1);
  auto off = make(model, backend, true, "all", false, "", false, 512, -1, false, -1, "exact", -1, 16, 0);
  int planned = 0;
  bool trees_none = true, off_none = true;
  for (int d = 0; d < on->model().drafters(); ++d) {
    planned += on->adaptive_plan(d) != nullptr;
    off_none = off_none && off->adaptive_plan(d) == nullptr;
    const auto& cps = on->adaptive_checkpoints();
    if (on->model().drafts_trees(d))
      trees_none = trees_none && !on->adaptive_plan(d) && (size_t(d) >= cps.size() || cps[size_t(d)].empty());
  }
  check(trees_none && off_none, "adaptive stopping: no plan for a drafter of trees (DV), and none with it off");
  SamplingParams sampled;
  sampled.temperature = 0.8f;
  sampled.seed = 7;
  constexpr int kTokens = 48;
  bool same = true;
  for (const SamplingParams& sp : {SamplingParams{}, sampled}) {
    Session a(on.get()), b(off.get());
    if (!init_checked(a) || !init_checked(b)) return;
    same = same && gen(a, prompt, kTokens, nullptr, sp).tokens == gen(b, prompt, kTokens, nullptr, sp).tokens;
  }
  check(same, "adaptive stopping on and off (" + std::string(planned ? "active, " : "inactive, ") +
                  std::to_string(planned) + " of " + std::to_string(on->model().drafters()) +
                  " drafters planned): the same tokens, greedy and sampled");
}

// The end of the context, the same on every backend (flyx_room): plain
// decoding of a prompt of n tokens in a sequence of max_ctx stops after
// max_ctx - FLYX_MAX_ROWS + 2 - n tokens (the prefill's token, then a round
// a token while one more round fits), and a prompt that leaves no room for a
// round is refused.
void context_end(const std::string& model, const std::string& backend, const std::vector<int32_t>& base) {
  const int max_ctx = 64;
  // CUDA's wide rows (backend option wide=1) for the tree rounds past 16 rows below; plain decoding takes none.
  auto e = make(model, backend, false, "", false, backend == "cuda" ? "wide=1" : "", false, max_ctx);
  auto prompt_of = [&](int n) {
    std::vector<int32_t> p;
    for (int i = 0; i < n; ++i) p.push_back(base[size_t(i) % base.size()]);
    return p;
  };
  std::string err;
  for (int n : {20, max_ctx - FLYX_MAX_ROWS}) {
    Session s(e.get());
    if (!init_checked(s)) continue;
    const int got = static_cast<int>(gen(s, prompt_of(n), 1000).tokens.size());
    const int want = max_ctx - FLYX_MAX_ROWS + 2 - n;
    check(got == want, "context end: a prompt of " + std::to_string(n) + " tokens in " + std::to_string(max_ctx) +
                           " decodes " + std::to_string(got) + " tokens (want " + std::to_string(want) + ")");
  }
  Session s(e.get());
  if (!init_checked(s)) return;
  GenerateRequest rq;
  rq.prompt = prompt_of(max_ctx - FLYX_MAX_ROWS + 1);
  rq.max_new_tokens = 4;
  rq.ignore_eos = true;
  GenerateResult res;
  const TextSink none = [](std::string_view, const int32_t*, int) { return true; };
  check(generate(s, rq, none, &res, &err) != 0, "context end: a prompt that leaves no room for a round is refused");
  // Tree rounds of more than FLYX_MAX_ROWS rows (31 and 63 drafts, lowered here) where the context has room for one
  // round of FLYX_MAX_ROWS rows: the chain of plain decoding's tokens each is given is accepted up to max_ctx and no
  // further (accept_tree stops at the context's end), the sequence stops there, and the round launched after it, in
  // flight, does nothing.  Both write their rows' K/V past max_ctx, the second up to max_ctx + R - 1 for R rows, in
  // the slots the backend keeps beyond it (CUDA: cap = max_ctx + its wide rows); with too few, a layer's would run
  // onto the next layer's first positions: the prompt's K/V, every layer's, must be the bytes they were before the
  // two rounds.
  if (e->tree_widths().empty()) return;  // a backend without tree rounds (cpu_ref)
  const GivenTrees wide_trees(e.get(), {2 * FLYX_MAX_ROWS - 1, FLYX_MAX_WIDE_ROWS - 1}, false);
  if (!wide_trees.lowered()) {  // a gated delta net keeps trees of FLYX_MAX_ROWS rows
    check(wide_trees.refusal().find("gated delta net") != std::string::npos,
          "context end: trees of more than " + std::to_string(FLYX_MAX_ROWS) + " rows refused only on a gated delta "
          "net" + wide_trees.refusal());
    return;
  }
  auto wide = make(model, backend, false);  // plain decoding past this context's end
  const int n = max_ctx - FLYX_MAX_ROWS, most = max_ctx - n;
  const auto* api = e->api();
  for (const auto& [W, plan] : wide_trees.plans()) {
    const std::string rows = std::to_string(W + 1) + " rows";
    Session ref(wide.get()), t(e.get());
    if (!ref.init(&err) || !t.init(&err)) {
      check(false, "context end, a tree round of " + rows + ": sessions (" + err + ")");
      return;
    }
    const std::vector<int32_t> want = gen(ref, prompt_of(n), W + 1).tokens, first = gen(t, prompt_of(n), 1).tokens;
    if (want.size() != size_t(W) + 1 || first.size() != 1 || first[0] != want[0]) {
      check(false, "context end, a tree round of " + rows + ": plain decoding and the first token");
      return;
    }
    // The prompt's K/V as a checkpoint at its end saves them (snapshot_save: every layer's, from the live caches).
    auto prompt_kv = [&](std::vector<unsigned char>* bytes) {
      uint64_t size = 0;
      if (api->snapshot_bytes(t.seq(), 0, 0, &size) != 0) return false;
      const uint64_t piece = std::max<uint64_t>(4096, (size + 4095) / 4096 * 4096);
      bytes->assign(size_t(piece), 0);
      void* at = bytes->data();
      const flyx_host_pieces dst{&at, 1, 0, piece};
      if (api->snapshot_save(t.seq(), 0, 0, &dst, size) != n) return false;
      bytes->resize(size_t(size));
      return true;
    };
    std::vector<unsigned char> kv_before, kv_after;
    const bool saved = api->checkpoint(t.seq(), 0) == n && prompt_kv(&kv_before);
    std::vector<int32_t> drafts(want.begin() + 1, want.end()), parents(drafts.size());
    for (int i = 0; i < W; ++i) parents[size_t(i)] = i - 1;
    flyx_gen_params g{};
    g.top_p = 1.f;
    g.drafts = drafts.data();
    g.n_drafts = W;
    g.drafts_at = n + 1;
    g.parents = parents.data();
    flyx_round_result r{}, after{};
    const bool ran = api->launch(plan, t.seq(), &g) == 0 && api->launch(plan, t.seq(), &g) == 0 &&
                     api->read(t.seq(), &r) == 0 && api->read(t.seq(), &after) == 0;
    check(ran && r.n_tokens == most && r.n_accepted == most - 1 && r.stopped && r.position == max_ctx &&
              std::equal(r.tokens, r.tokens + most, want.begin() + 1),
          "context end: a tree round of " + rows + " " + std::to_string(most) + " positions before it emits plain "
              "decoding's " + std::to_string(most) + " tokens and stops (" + std::to_string(r.n_tokens) +
              ", position " + std::to_string(r.position) + (ran ? "" : std::string(": ") + api->last_error()) + ")");
    check(ran && after.n_tokens == 0 && after.stopped && after.position == max_ctx,
          "context end:   the tree round in flight after it does nothing");
    check(saved && prompt_kv(&kv_after) && kv_after == kv_before,
          "context end:   the prompt's K/V, every layer's, are the bytes they were before the two rounds (" +
              std::to_string(kv_before.size()) + " bytes)");
  }
}

// A checkpoint kept inside a prefill pass (prefill_snapshot: a multiple of the alignment the pass crosses, not its
// last): the same bytes as a pass that ends there, and restored, the first token a fresh prefill chooses.  Only where
// passes keep a state at a boundary (prefill_align > 1: the chunked GDN recurrence).
void mid_pass_checkpoint(const std::string& model, const std::string& backend, const std::vector<int32_t>& base) {
  auto run_mode = [&](bool spec, const std::string& drafter, const std::string& label) {
    auto e = make(model, backend, spec, drafter, false, "", false, 512, 0, false, -1, "exact", -1, 256);
    const auto* api = e->api();
    const int align = e->prefill_align();
    if (align <= 1 || !api->prefill_snapshot) return;
    std::vector<int32_t> p;
    while (int(p.size()) < 3 * align + 5) p.insert(p.end(), base.begin(), base.end());
    const int n = int(p.size()), c = align;  // inside the pass, >= 2*align (128 > kHring) before its end
    auto save = [&](flyx_seq* seq, std::vector<uint8_t>* out) {
      uint64_t bytes = 0;
      if (api->snapshot_bytes(seq, 0, 0, &bytes) != 0) return false;
      out->resize(size_t(std::max<uint64_t>(4096, bytes)));
      void* q = out->data();
      flyx_host_pieces pieces{&q, 1, 0, out->size()};
      return api->snapshot_save(seq, 0, 0, &pieces, bytes) == c && (out->resize(size_t(bytes)), true);
    };
    flyx_gen_params g{};
    g.top_p = 1.f;
    flyx_round_result r{};
    Session fresh(e.get()), whole(e.get()), cut(e.get());
    if (!init_checked(fresh) || !init_checked(whole) || !init_checked(cut)) return;
    bool ok = api->prefill(e->prefill_plan(), fresh.seq(), p.data(), n, &g) == 0 && api->read(fresh.seq(), &r) == 0;
    const int32_t want = r.tokens[0];
    // One pass over the prompt keeping c; a pass that ends at c.
    ok = ok && api->prefill_snapshot(whole.seq(), c) == 0 &&
         api->prefill(e->prefill_plan(), whole.seq(), p.data(), n, nullptr) == 0 && api->checkpoint(whole.seq(), 0) == c;
    ok = ok && api->prefill(e->prefill_plan(), cut.seq(), p.data(), c, nullptr) == 0 && api->checkpoint(cut.seq(), 0) == c;
    std::vector<uint8_t> a, b;
    bool same = false;
    if (!spec) {
      same = ok && save(whole.seq(), &a) && save(cut.seq(), &b) && a == b;
    } else {
      // With MTP, whole populated MTP KV[c - 1] from p[c] while cut did not yet; loading whole's mid-pass snapshot
      // into a fresh session and prefilling an edited aligned chunk from c rewrites MTP KV[c - 1] from
      // (ring[c - 1], alt[0]) while keeping MTP KV[2*c - 1] untouched, proving ring_snap preserved hidden[c - 1]
      // past kHring=64 overwrites across all snapshot bytes.
      std::vector<uint8_t> mid;
      std::vector<int32_t> alt(p.begin() + c, p.begin() + 2 * c);
      alt[0] = alt[0] == 1 ? 2 : 1;
      Session dest(e.get());
      auto load_snap = [&](flyx_seq* seq, std::vector<uint8_t>& buf, int32_t anchor) {
        void* q = buf.data();
        flyx_host_pieces pieces{&q, 1, 0, buf.size()};
        return api->snapshot_load(seq, &pieces, buf.size(), anchor) == c;
      };
      auto save1 = [&](flyx_seq* seq, std::vector<uint8_t>* out) {
        uint64_t bytes = 0;
        if (api->snapshot_bytes(seq, 1, 0, &bytes) != 0) return false;
        out->resize(size_t(std::max<uint64_t>(4096, bytes)));
        void* q = out->data();
        flyx_host_pieces pieces{&q, 1, 0, out->size()};
        return api->snapshot_save(seq, 1, 0, &pieces, bytes) == 2 * c && (out->resize(size_t(bytes)), true);
      };
      const bool mtp_ok = ok && init_checked(dest) && save(whole.seq(), &mid) && load_snap(dest.seq(), mid, alt[0]) &&
                          api->restore(cut.seq(), 0, alt[0]) == c &&
                          api->prefill(e->prefill_plan(), dest.seq(), alt.data(), c, nullptr) == 0 &&
                          api->prefill(e->prefill_plan(), cut.seq(), alt.data(), c, nullptr) == 0 &&
                          api->checkpoint(dest.seq(), 1) == 2 * c && api->checkpoint(cut.seq(), 1) == 2 * c;
      same = mtp_ok && save1(dest.seq(), &a) && save1(cut.seq(), &b) && a == b;
    }
    // Restored at c, the rest of the prompt: the first token of the fresh prefill.
    ok = ok && api->restore(whole.seq(), 0, p[size_t(c)]) == c &&
         api->prefill(e->prefill_plan(), whole.seq(), p.data() + c, n - c, &g) == 0 && api->read(whole.seq(), &r) == 0;
    check(ok && same && r.tokens[0] == want,
          "checkpoint inside a prefill pass (" + label + ", at " + std::to_string(c) + " of " + std::to_string(n) +
              "): the bytes of a pass ending there, and restored, the fresh prefill's first token");
  };
  run_mode(false, "", "plain");
  run_mode(true, "mtp", "mtp");
}

// Rewinding partway into a committed round must clamp that slot's valid row count so a later prefill + rewind past the
// cut point never replays stale pre-edit GDN rows.
void rewind_prefill_rewind(const std::string& model, const std::string& backend, const std::vector<int32_t>& base) {
  auto e = make(model, backend, true, "mtp", false, "", false, 512, 4);
  const auto* api = e->api();
  if (e->widths().empty()) return;
  size_t plan_idx = 0;
  for (size_t i = 0; i < e->widths().size(); ++i)
    if (e->widths()[i] == 4) plan_idx = i;
  if (e->widths()[plan_idx] < 3) return;
  Session s(e.get()), ref(e.get());
  if (!init_checked(s) || !init_checked(ref)) return;
  const std::vector<int32_t> truth = gen(ref, base, 6).tokens;
  if (truth.size() < 5) return;
  flyx_gen_params g{};
  g.top_p = 1.f;
  flyx_round_result r{};
  const int l0 = int(base.size());
  std::vector<int32_t> oracle = base;
  oracle.insert(oracle.end(), truth.begin(), truth.end());
  bool ok = api->prefill(e->prefill_plan(), s.seq(), base.data(), l0, &g) == 0 && api->read(s.seq(), &r) == 0;
  // Force all 4 drafts to match the model's greedy continuation so meta[0] records start=l0, rows=5.
  ok = ok && api->set_oracle(s.seq(), oracle.data(), int32_t(oracle.size())) == 0 &&
       api->launch(e->round_plan(plan_idx), s.seq(), &g) == 0 && api->read(s.seq(), &r) == 0 && r.n_tokens == 5;
  // Rewind to l0 + 1 (clamping meta[0].rows from 5 to 1), then prefill 4 different tokens to l0 + 5.
  std::vector<int32_t> alt(4);
  for (int i = 0; i < 4; ++i) alt[size_t(i)] = truth[size_t(i + 1)] == 1 ? 2 : 1;
  const int rw1 = ok ? api->rewind(s.seq(), l0 + 1, alt[0]) : -1;
  ok = ok && rw1 == 1 && api->prefill(e->prefill_plan(), s.seq(), alt.data(), 4, nullptr) == 0;
  // Rewinding to l0 + 3 (inside the old [l0, l0 + 5] span, past the new l0 + 1 clamp) must reset (return 0), not fold.
  const int rw2 = ok ? api->rewind(s.seq(), l0 + 3, alt[2]) : -1;
  check(ok && rw1 == 1 && rw2 == 0,
        "rewind clamps round slot rows so prefill + second rewind past the cut refuses stale GDN replay");
}

// A chat whose last message boundary falls inside a prefill pass (the golden engines' passes are 16 rows, the boundary
// off their multiples), its checkpoint kept, then restored by the next turn, whose last message differs: plain
// decoding's tokens, the fresh prefill's first token first, and the boundary's tokens reused.  On a file with DV the
// next turn drafts with DV where the backend runs it (pinned: every round DV's), its drafts round for round those of a
// fresh session's prefill of the same turn (the canvas the prefill after the restore starts over is the fresh one's).
// A Llama's prefill_align is 1 (cuda: no GDN state; cpu_ref: row by row): Generation::prefill ends a pass at
// the boundary, and the checkpoint holds the whole state there, DV's canvas and a V-LoRA's rows among it
// (prefill_snapshot, a checkpoint kept inside a pass, is for prefill_align > 1: mid_pass_checkpoint).
void dv_chat_checkpoint(const std::string& model, const std::string& backend, const std::vector<int32_t>& base,
                        bool dv_runs) {
  int pin = 0;
  if (dv_runs) {
    auto chosen = make(model, backend, true, "dv", false, "", false, 512, -1);
    for (size_t i = 0; i < chosen->widths().size(); ++i)
      if (chosen->round_drafters()[i] >= 0) pin = std::max(pin, chosen->widths()[i]);
  }
  auto e = dv_runs ? make(model, backend, true, "dv", false, "", false, 512, pin)
                   : make(model, backend, true, "all", false, "", false, 512, -1);  // DV's blocks taken, not run
  auto plain = make(model, backend, false);
  const int chunk = e->prefill_chunk(), align = e->prefill_align(), vocab = e->model().desc().vocab;
  std::vector<int32_t> p;
  while (int(p.size()) < 3 * chunk + 5) p.insert(p.end(), base.begin(), base.end());
  p.resize(size_t(3 * chunk + 5));
  const int c0 = chunk + 3, c = 2 * chunk + 5;  // message boundaries; the last inside the pass from 2 chunk + 3
  auto q = std::vector<int32_t>(p.begin(), p.begin() + c);  // the next turn: its last message another
  for (size_t i = 0; q.size() < p.size(); ++i) q.push_back(base[base.size() - 1 - i % base.size()]);
  if (q[size_t(c)] == p[size_t(c)]) q[size_t(c)] = (q[size_t(c)] + 1) % vocab;
  const int N = 24;
  auto run = [&](Session& s, const std::vector<int32_t>& prompt, std::vector<flyx_round_result>* trace, int* reused) {
    GenerateRequest rq;
    rq.prompt = prompt;
    rq.checkpoints = {c0, c};
    rq.max_new_tokens = N;
    rq.ignore_eos = true;
    GenerateResult res;
    std::string err;
    std::vector<int32_t> t;
    s.trace = trace;
    if (generate(s, rq, [&](std::string_view, const int32_t* x, int nx) {
          t.insert(t.end(), x, x + nx);
          return true;
        }, &res, &err) != 0)
      check(false, "a chat's checkpoint inside a pass: generate: " + err);
    s.trace = nullptr;
    if (reused) *reused = res.usage.reused_tokens;
    return t;
  };
  Session s(e.get()), fresh_dv(e.get()), fresh_p(plain.get()), fresh_q(plain.get());
  if (!init_checked(s) || !init_checked(fresh_dv) || !init_checked(fresh_p) || !init_checked(fresh_q)) return;
  const auto first = run(s, p, nullptr, nullptr);
  std::vector<flyx_round_result> trace;
  int reused = -1;
  const auto got = run(s, q, &trace, &reused);
  const auto want_p = run(fresh_p, p, nullptr, nullptr), want_q = run(fresh_q, q, nullptr, nullptr);
  std::vector<flyx_round_result> fresh_trace;
  run(fresh_dv, q, &fresh_trace, nullptr);
  int drafted = 0;
  bool same_drafts = trace.size() == fresh_trace.size();
  for (size_t i = 0; i < trace.size(); ++i) {
    drafted += trace[i].n_drafts > 0;
    same_drafts = same_drafts && i < fresh_trace.size() && trace[i].n_drafts == fresh_trace[i].n_drafts &&
                  std::equal(trace[i].drafts, trace[i].drafts + trace[i].n_drafts, fresh_trace[i].drafts);
  }
  check(first == want_p && got == want_q && !got.empty() && reused == c && same_drafts && (!dv_runs || drafted > 0),
        "a chat's last message boundary inside a prefill pass (" + std::to_string(c) + " of " +
            std::to_string(p.size()) + ", passes of " + std::to_string(chunk) + ", prefill_align " +
            std::to_string(align) + "): its checkpoint restored by the next turn (" + std::to_string(reused) +
            " reused), then " +
            (dv_runs ? "DV's rounds (" + std::to_string(drafted) + " drafted, a fresh session's drafts)"
                     : std::string("plain rounds")) +
            ": plain decoding's tokens, the fresh prefill's first token first");
}

// A DFlash2 drafter whose block (8) is less than a round's rows: its chains of 15 drafts come from a wider block (the
// backends' wide block, width + 1 mask rows in one pass), at a fixed width and chosen every round; either way the
// tokens are plain decoding's.
void wide_block(const std::string& model, const std::string& backend,
                const std::vector<std::vector<int32_t>>& prompts) {
  if (!std::filesystem::exists(os::to_path(model))) {
    check(false, "wide block: " + model + " is missing (make_tiny.py)");
    return;
  }
  auto plain = make(model, backend, false);
  auto fixed = make(model, backend, true, g_block, false, "", false, 512, FLYX_MAX_ROWS - 1);
  auto chosen = make(model, backend, true, g_block, false, "", false, 512, -1);
  const auto& w = chosen->widths();
  check(std::find(w.begin(), w.end(), FLYX_MAX_ROWS - 1) != w.end(),
        "wide block: DFlash2 rounds of 15 drafts from a block of 8 among the chosen widths");
  constexpr int kTokens = 48;
  bool same = true;
  int wide = 0;
  for (const auto& p : prompts) {
    Session a(plain.get()), b(fixed.get()), c(chosen.get());
    if (!init_checked(a) || !init_checked(b) || !init_checked(c)) return;
    std::vector<flyx_round_result> trace;
    const auto ref = gen(a, p, kTokens).tokens;
    same = same && gen(b, p, kTokens, &trace).tokens == ref && gen(c, p, kTokens).tokens == ref;
    for (const auto& r : trace) wide += r.n_drafts == FLYX_MAX_ROWS - 1;
  }
  check(same && wide > 0, "wide block: DFlash2 chains of 15 drafts (" + std::to_string(wide) +
                              " rounds), fixed and chosen every round, give plain decoding's tokens");
}

// Wide lookup rounds (the backend option wide=1: up to FLYX_MAX_WIDE_ROWS rows, every row a 16-row round's bits):
// rounds of 31, 47 and 63 given drafts, plain decoding's own continuation so that every row is accepted, give its
// tokens; so do drafts that go wrong part way, rejected there.  An NVFP4 model takes the 17..64-row GEMMs, a bf16 one
// 16 rows at a time.
void wide_rounds(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt) {
  auto plain = make(model, backend, false);
  auto wide = make(model, backend, true, "none", false, "wide=1", true);  // lookup rounds only
  std::vector<size_t> ks;
  for (size_t k = 0; k < wide->lookup_widths().size(); ++k)
    if (wide->lookup_widths()[k] > FLYX_MAX_ROWS - 1) ks.push_back(k);
  check(ks.size() == plan::wide_given_widths().size(), "wide rounds: lookup rounds of 31, 47 and 63 drafts lowered");
  if (ks.empty()) return;
  constexpr int kTokens = 320;
  Session sp(plain.get());
  if (!init_checked(sp)) return;
  const std::vector<int32_t> ref = gen(sp, prompt, kTokens).tokens;
  Session s(wide.get());
  if (!init_checked(s)) return;
  gen(s, prompt, 1);  // the prompt and ref[0], the next round's anchor
  size_t at = 1;      // ref[at]: the token after the anchor
  bool same = true;
  int rounds = 0, longest = 0, cut = 0;
  for (int i = 0; same; ++i) {
    const size_t k = ks[size_t(i) % ks.size()];
    const int w = wide->lookup_widths()[k];
    if (at + size_t(w) + 1 > ref.size()) break;
    std::vector<int32_t> d(ref.begin() + std::ptrdiff_t(at), ref.begin() + std::ptrdiff_t(at) + w);
    const bool wrong = i % 4 == 3;  // one draft wrong past the first 16 rows (after a whole round of each width)
    if (wrong) {
      const size_t j = size_t(FLYX_MAX_ROWS + (i * 7) % (w - FLYX_MAX_ROWS));
      d[j] = d[j] == 1 ? 2 : 1;
    }
    flyx_gen_params g{};
    g.top_p = 1.f;
    g.drafts = d.data();
    g.n_drafts = w;
    g.drafts_at = static_cast<int32_t>(prompt.size() + at);
    flyx_round_result r{};
    if (wide->api()->launch(wide->lookup_plan(k), s.seq(), &g) != 0 || wide->api()->read(s.seq(), &r) != 0) {
      check(false, "wide rounds: launch and read");
      return;
    }
    for (int t = 0; t < r.n_tokens && same; ++t)
      same = at + size_t(t) < ref.size() && r.tokens[t] == ref[at + size_t(t)];
    if (wrong) cut += r.n_accepted < w;
    longest = std::max(longest, r.n_accepted);
    at += size_t(r.n_tokens);
    ++rounds;
  }
  check(same && rounds >= 4 && longest == FLYX_MAX_WIDE_ROWS - 1 && cut > 0,
        "wide rounds: " + std::to_string(rounds) + " rounds of 31 to 63 drafts give plain decoding's tokens" +
            (same ? "" : " (NOT: they differ at " + std::to_string(at) + ")") + " (" +
            std::to_string(at) + " of them; " + std::to_string(longest) + " accepted at most, " + std::to_string(cut) +
            " cut at a wrong draft)");
}

// K/V caches in host memory: with every position there, the reference's tokens, plain and with each
// drafter, binary16 and fp8 alike; with the GPU's part ending inside the prompt, or inside the decoded tokens, the
// tokens of a context wholly on the GPU.  The tiny model's rows are 128 bytes, so the 2 MB the driver maps at least
// make a GPU part of 16,384 positions.
void kv_in_host_memory(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
                       const std::vector<int32_t>& want) {
  const int N = static_cast<int>(want.size());
  std::string err;
  auto tokens = [&](Engine* e, const std::vector<int32_t>& p, int n) {
    Session s(e);
    if (!s.init(&err)) return std::vector<int32_t>{-1, -1};
    return gen(s, p, n).tokens;
  };
  {
    auto plain = make(model, backend, false, "", false, "", false, 512, kReferenceWidth, false, 0);
    auto spec = make(model, backend, true, g_block, false, "", false, 512, kReferenceWidth, false, 0);
    auto mtp = make(model, backend, true, "mtp", false, backend == "cuda" ? "draft_weights=exact" : "", false, 512,
                    kReferenceWidth, false, 0);
    check(backend != "cuda" || (plain->kv_layout().device_positions == 0 && plain->kv_layout().max_ctx == 512),
          "K/V all in host memory: placed so (" + std::to_string(plain->kv_layout().device_positions) + " on the GPU)");
    // The K/V's host memory is mapped for the device alone, which no process memory figure counts: the engine says
    // how much of a session's host memory that is, and how much its sessions hold of it now (ABI 39).
    const plan::Budget& b = plain->budget();
    check(backend != "cuda" || (b.session_host_mapped > 0 && b.session_host_mapped <= b.session_host_bytes),
          "K/V all in host memory: " + std::to_string(b.session_host_mapped) + " of the session's " +
              std::to_string(b.session_host_bytes) + " host bytes mapped for the device alone");
    {
      Session held(plain.get());
      check(held.init(&err) && plain->host_mapped_in_use() == b.session_host_mapped,
            "K/V all in host memory: a session holds its mapped host memory");
    }
    check(plain->host_mapped_in_use() == 0, "K/V all in host memory: given back with the session");
    check(tokens(plain.get(), prompt, N) == want && tokens(spec.get(), prompt, N) == want &&
              tokens(mtp.get(), prompt, N) == want,
          "K/V all in host memory: the reference's tokens, plain, DFlash2 and MTP");
    snapshot_through_host(spec.get(), plain.get(), prompt, "snapshot, K/V in host memory", backend == "cpu_ref");
    auto gpu8 = make(model, backend, false, "", true);
    auto plain8 = make(model, backend, false, "", true, "", false, 512, kReferenceWidth, false, 0);
    auto spec8 = make(model, backend, true, g_block, true, "", false, 512, kReferenceWidth, false, 0);
    const auto t8 = tokens(gpu8.get(), prompt, N);
    check(tokens(plain8.get(), prompt, N) == t8 && tokens(spec8.get(), prompt, N) == t8,
          "K/V all in host memory, fp8: the GPU's tokens, plain and DFlash2");
  }
  if (backend != "cuda") return;  // the reference backend places nothing
  const int ctx = 20480;
  auto gpu = make(model, backend, false, "", false, "", false, ctx);
  auto plain = make(model, backend, false, "", false, "", false, ctx, kReferenceWidth, false, 16384);
  auto spec = make(model, backend, true, g_block, false, "", false, ctx, kReferenceWidth, false, 16384);
  // The copies to the staging right before each attention, not beside the layers before it: the same tokens.
  auto inline_copies = make(model, backend, false, "", false, "stage_ahead=0", false, ctx, kReferenceWidth, false, 16384);
  const int at = plain->kv_layout().device_positions;
  check(at == 16384 && gpu->kv_layout().device_positions >= ctx,
        "K/V past 16,384 positions in host memory: placed so (" + std::to_string(at) + ")");
  const plan::Budget& part = plain->budget();
  check(gpu->budget().session_host_mapped == 0 && part.session_host_mapped > 0 &&
            part.session_host_mapped <= part.session_host_bytes,
        "K/V past 16,384 positions: its host part mapped for the device alone, none with all on the GPU");
  // One session each: the second prompt reuses the first's tokens and prefills across the boundary.
  auto two_prompts = [&](Engine* e) {
    Session s(e);
    std::vector<std::vector<int32_t>> out;
    if (!s.init(&err)) return out;
    for (int len : {at - 24, at + 40}) {
      std::vector<int32_t> p;
      for (int i = 0; i < len; ++i) p.push_back(prompt[size_t(i) % prompt.size()]);
      out.push_back(gen(s, p, 48).tokens);
    }
    return out;
  };
  const auto t = two_prompts(gpu.get()), tp = two_prompts(plain.get()), ts = two_prompts(spec.get());
  const auto ti = two_prompts(inline_copies.get());
  for (size_t i = 0; i < 2; ++i)
    check(t.size() == 2 && t[i].size() == 48 && tp.size() == 2 && tp[i] == t[i] && ts.size() == 2 && ts[i] == t[i] &&
              ti.size() == 2 && ti[i] == t[i],
          std::string("K/V's GPU part ending inside the ") + (i == 0 ? "decoded tokens" : "prompt") +
              ": the GPU's tokens, plain, DFlash2 and with the copies inline");
}

// The MTP layer's draft steps over its first and last positions, a 400-token prompt, FP8 caches: a
// window as long as the context gives the drafts of none (its positions gathered in order); its first 16 and last 64
// positions in a ring of 128 slots give the drafts of the same positions over a slot a position (the ring across its
// wraps), and plain decoding's tokens; a snapshot of the ring, loaded, the drafts of the session it was taken from.
void mtp_window_checks(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
                       const std::vector<int32_t>& want) {
  if (backend != "cuda") return;
  std::vector<int32_t> longp;
  while (longp.size() < 400) {
    longp.insert(longp.end(), prompt.begin(), prompt.end());
    longp.insert(longp.end(), want.begin(), want.end());
  }
  longp.resize(400);
  std::string err;
  auto run = [&](const std::string& opts, std::vector<int32_t>* drafts) {
    auto e = make(model, backend, true, "mtp", true, opts, false, 512);
    Session s(e.get());
    if (!s.init(&err)) return std::vector<int32_t>{-1};
    std::vector<flyx_round_result> trace;
    const auto out = gen(s, longp, 48, &trace).tokens;
    for (const flyx_round_result& r : trace) drafts->insert(drafts->end(), r.drafts, r.drafts + r.n_drafts);
    return out;
  };
  std::vector<int32_t> d_none, d_all, d_ring, d_flat;
  const auto t_none = run("mtp_window=0", &d_none), t_all = run("mtp_window=1024", &d_all);
  const auto t_ring = run("mtp_window=64,mtp_sink=16,mtp_copy=0", &d_ring), t_flat = run("mtp_window=64,mtp_sink=16,mtp_copy=0,mtp_ring=0", &d_flat);
  std::vector<int32_t> t_plain;
  {
    auto plain = make(model, backend, false, "", true, "", false, 512);
    Session s(plain.get());
    if (s.init(&err)) t_plain = gen(s, longp, 48).tokens;
  }
  check(!d_none.empty() && d_all == d_none && t_all == t_none,
        "MTP over a window as long as the context: the drafts of none (" + std::to_string(d_none.size()) + ")");
  check(!d_ring.empty() && d_ring == d_flat && t_ring == t_flat,
        "MTP over its first 16 and last 64 positions: a ring of 128 slots gives the drafts of a slot a position");
  check(t_plain.size() == 48 && t_none == t_plain && t_ring == t_plain,
        "MTP over its first 16 and last 64 positions decodes plain decoding's tokens");
  auto ring = make(model, backend, true, "mtp", true, "mtp_window=64,mtp_sink=16,mtp_copy=0", false, 512);
  auto exact = make(model, backend, true, "mtp", true, "mtp_window=0", false, 512);
  snapshot_through_host(ring.get(), exact.get(), prompt, "snapshot, MTP over a ring of 128 slots", false);
  // Over a 2- or 3-bit copy of every position, its last 64 exact from the ring (strata attention, the correction
  // reading the ring's slots): the drafts of the same over a slot a position, plain decoding's tokens.
  for (int bits : {2, 3}) {
    const std::string copy = "mtp_window=64,mtp_copy=" + std::to_string(bits);
    std::vector<int32_t> c_ring, c_flat;
    const auto tc_ring = run(copy, &c_ring), tc_flat = run(copy + ",mtp_ring=0", &c_flat);
    check(!c_ring.empty() && c_ring == c_flat && tc_ring == tc_flat && tc_ring == t_plain,
          "MTP over a " + std::to_string(bits) +
              "-bit copy, its last 64 positions exact: a ring gives the drafts of a slot a position, plain decoding's "
              "tokens");
  }
  auto copy = make(model, backend, true, "mtp", true, "mtp_window=64,mtp_copy=2", false, 512);
  snapshot_through_host(copy.get(), exact.get(), prompt, "snapshot, MTP over a 2-bit copy", false);
}

// Strata attention: a round reads a low-bit copy of every position and refines each row's most
// attended pairs from the cache.  Drafting must still not change the output (rule 4): DFlash2 and MTP rounds give
// plain decoding's tokens, with the K/V in host memory too, binary16 and FP8 caches, 2 and 3 bits; and rounds of
// several sequences give each one's tokens alone.  The selection is narrowed (1 pair in 8, at least 1) so that the
// tiny model's short contexts leave most pairs to the copy; with every pair refined it is the cache's attention.
void strata_rounds(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
                   const std::vector<int32_t>& want) {
  if (backend != "cuda") return;  // the reference backend reads every position
  const int N = static_cast<int>(want.size());
  const std::string narrow = "strata_denom=8,strata_least=1";
  std::string err;
  auto tokens = [&](Engine* e) {
    Session s(e);
    if (!s.init(&err)) return std::vector<int32_t>{-1};
    return gen(s, prompt, N).tokens;
  };
  for (const char* method : {"strata", "strata:3"})
    for (bool fp8 : {false, true}) {
      auto plain = make(model, backend, false, "", fp8, narrow, false, 512, kReferenceWidth, false, -1, method);
      auto spec = make(model, backend, true, g_block, fp8, narrow, false, 512, kReferenceWidth, false, -1, method);
      auto mtp = make(model, backend, true, "mtp", fp8, narrow, false, 512, kReferenceWidth, false, -1, method);
      auto host = make(model, backend, true, g_block, fp8, narrow, false, 512, kReferenceWidth, false, 0, method);
      // every position in host memory, plain decoding: with FP8 caches the pairs re-read are kept on the device,
      // read from there the rounds after
      auto host_plain = make(model, backend, false, "", fp8, narrow, false, 512, kReferenceWidth, false, 0, method);
      // the copy in host memory too, every layer's or one layer's on the GPU
      auto copy_host = make(model, backend, true, g_block, fp8, narrow, false, 512, kReferenceWidth, false, 0, method,
                            0);
      auto copy_one = make(model, backend, true, "mtp", fp8, narrow, false, 512, kReferenceWidth, false, 0, method, 1);
      const auto t = tokens(plain.get());
      check(t.size() == size_t(N) && tokens(spec.get()) == t && tokens(mtp.get()) == t && tokens(host.get()) == t &&
                tokens(host_plain.get()) == t,
            std::string(method) + (fp8 ? ", fp8" : ", binary16") +
                ": DFlash2 and MTP decode == plain decode, the K/V in host memory too (pairs kept on the device)");
      check(tokens(copy_host.get()) == t && tokens(copy_one.get()) == t,
            std::string(method) + (fp8 ? ", fp8" : ", binary16") +
                ": the copy in host memory, read through stagings: DFlash2 and MTP decode == plain decode");
      // the selection over chunks of 8 pairs: the pick of one chunk, so the same tokens
      auto chunked = make(model, backend, true, g_block, fp8, narrow + ",strata_chunk=8", false, 512,
                          kReferenceWidth, false, -1, method);
      check(tokens(chunked.get()) == t, std::string(method) + (fp8 ? ", fp8" : ", binary16") +
                                            ": the selection in chunks of 8 pairs picks what one chunk picks");
    }
  {
    auto all = make(model, backend, false, "", false, "strata_least=100000", false, 512, kReferenceWidth, false, -1,
                    "strata");
    check(tokens(all.get()) == want, "strata with every pair refined: the reference's tokens");
  }
  {  // auto: exact rows below the switch, the copy's from it, in the middle of the reply here. Before
     // it, the exact reference's tokens; across it, drafted rounds (some about the switch, reading both) give plain
     // decoding's tokens, rounds in flight and rounds of one row alike.
    const int from = static_cast<int>(prompt.size()) + N / 2;
    const std::string at = narrow + ",strata_from=" + std::to_string(from);
    for (bool fp8 : {false, true}) {
      auto plain = make(model, backend, false, "", fp8, at, false, 512, kReferenceWidth, false, -1, "auto");
      auto spec = make(model, backend, true, g_block, fp8, at, false, 512, kReferenceWidth, false, -1, "auto");
      auto mtp = make(model, backend, true, "mtp", fp8, at, false, 512, kReferenceWidth, false, -1, "auto");
      auto chosen = make(model, backend, true, g_block, fp8, at, false, 512, -1, false, -1, "auto");
      auto exact = make(model, backend, false, "", fp8, "", false, 512, kReferenceWidth, false, -1);
      // the K/V in host memory: the copy moves between its homes as the rounds cross the switch
      auto host = make(model, backend, true, g_block, fp8, at, false, 512, kReferenceWidth, false, 0, "auto");
      // and the copy in host memory for good: read there about the switch, through stagings past it
      auto copy_host = make(model, backend, true, g_block, fp8, at, false, 512, kReferenceWidth, false, 0, "auto",
                            0);
      const auto t = tokens(plain.get()), te = tokens(exact.get());
      check(t.size() == size_t(N) && te.size() == size_t(N) &&
                std::equal(t.begin(), t.begin() + N / 2 + 1, te.begin()),
            std::string("auto") + (fp8 ? ", fp8" : ", binary16") + ": exact attention's tokens before the switch");
      check(tokens(spec.get()) == t && tokens(mtp.get()) == t && tokens(chosen.get()) == t,
            std::string("auto") + (fp8 ? ", fp8" : ", binary16") +
                ": DFlash2, MTP and chosen widths decode == plain decode across the switch");
      check(tokens(host.get()) == t && tokens(copy_host.get()) == t,
            std::string("auto") + (fp8 ? ", fp8" : ", binary16") +
                ": the K/V (and the copy) in host memory, DFlash2 across the switch == plain decode");
      // The switch at the start: strata's tokens (2-bit default, and explicit 3-bit); past the context: exact's.
      auto first = make(model, backend, false, "", fp8, narrow + ",strata_from=0", false, 512, kReferenceWidth, false,
                        -1, "auto");
      auto strata2 = make(model, backend, false, "", fp8, narrow, false, 512, kReferenceWidth, false, -1, "strata:2");
      auto first3 = make(model, backend, false, "", fp8, narrow + ",strata_from=0", false, 512, kReferenceWidth, false,
                         -1, "auto:3");
      auto strata3 = make(model, backend, false, "", fp8, narrow, false, 512, kReferenceWidth, false, -1, "strata:3");
      auto never = make(model, backend, false, "", fp8, narrow + ",strata_from=100000", false, 512, kReferenceWidth,
                        false, -1, "auto");
      check(tokens(first.get()) == tokens(strata2.get()) && tokens(first3.get()) == tokens(strata3.get()) &&
                tokens(never.get()) == te,
            std::string("auto") + (fp8 ? ", fp8" : ", binary16") +
                ": switched at the start, strata's tokens (2-bit default and explicit 3-bit); never, exact attention's");
    }
  }
  {  // turns: a second prompt's prefill reads the caches through the stagings, over the pool the copy
     // shares; the rounds after it move back only what changed.  The last round's logits (every position's copy
     // weighs in them, where the tiny model's tokens hardly move) and the tokens of a session whose copy never moves.
    const int vocab = make(model, backend, false)->model().desc().vocab;
    for (bool fp8 : {false, true}) {
      auto host = make(model, backend, false, "", fp8, narrow + ",head_cert=0", false, 512, kReferenceWidth, false, 0,
                       "strata");
      auto dev = make(model, backend, false, "", fp8, narrow + ",head_cert=0", false, 512, kReferenceWidth, false, -1,
                      "strata");
      // the copy in host memory, every layer's
      auto copy_host = make(model, backend, false, "", fp8, narrow + ",head_cert=0", false, 512, kReferenceWidth,
                            false, 0, "strata", 0);
      auto turns = [&](Engine* e, std::vector<float>* logits) {
        Session s(e);
        if (!s.init(&err)) return std::vector<int32_t>{-1};
        auto out = gen(s, prompt, N / 2).tokens;
        auto p2 = prompt;
        p2.insert(p2.end(), out.begin(), out.end());
        p2.insert(p2.end(), prompt.begin(), prompt.begin() + std::ptrdiff_t(std::min<size_t>(prompt.size(), 40)));
        const auto more = gen(s, p2, N / 2).tokens;
        out.insert(out.end(), more.begin(), more.end());
        logits->assign(size_t(vocab), 0.f);
        if (e->api()->logits(s.seq(), 0, logits->data()) != 0) logits->clear();
        return out;
      };
      std::vector<float> lh, ld, lc;
      const auto th = turns(host.get(), &lh), td = turns(dev.get(), &ld), tc = turns(copy_host.get(), &lc);
      check(th.size() == size_t(N / 2) * 2 && th == td && !lh.empty() && lh == ld,
            std::string("strata, two turns, the K/V in host memory") + (fp8 ? ", fp8" : ", binary16") +
                ": the logits and tokens of a session whose copy never moves");
      check(copy_host->kv_layout().copy_host >= 2 && tc == td && lc == ld,
            std::string("strata, two turns, the K/V and the copy in host memory") + (fp8 ? ", fp8" : ", binary16") +
                ": the same logits and tokens");
    }
  }
  {  // stagings of 64 positions, prefill passes of 48 rows: a 400-token prompt's passes read the keys
     // past them a window at a time, the second turn's short pass where they are in host memory.  The last round's
     // logits and the tokens of a session whose K/V is on the GPU; auto with its switch at 200 (stagings that reach it,
     // 320 positions).
    const int vocab = make(model, backend, false)->model().desc().vocab;
    std::vector<int32_t> longp;
    while (longp.size() < 400) {
      longp.insert(longp.end(), prompt.begin(), prompt.end());
      longp.insert(longp.end(), want.begin(), want.end());
    }
    longp.resize(400);
    for (const char* method : {"strata", "auto"})
      for (bool fp8 : {false, true}) {
        const std::string opts =
            narrow + ",head_cert=0" + (std::string(method) == "auto" ? ",strata_from=200" : "");
        auto dev = make(model, backend, false, "", fp8, opts, false, 512, kReferenceWidth, false, -1, method, -1, 48);
        auto win = make(model, backend, false, "", fp8, opts + ",stage_window=64", false, 512, kReferenceWidth, false,
                        0, method, -1, 48);
        auto mtp = make(model, backend, true, "mtp", fp8, opts + ",stage_window=64", false, 512, kReferenceWidth, false,
                        0, method, -1, 48);
        auto turns = [&](Engine* e, std::vector<float>* logits) {
          Session s(e);
          if (!s.init(&err)) return std::vector<int32_t>{-1};
          auto out = gen(s, longp, N / 2).tokens;
          auto p2 = longp;
          p2.insert(p2.end(), out.begin(), out.end());
          p2.insert(p2.end(), longp.begin(), longp.begin() + 8);
          const auto more = gen(s, p2, N / 2).tokens;
          out.insert(out.end(), more.begin(), more.end());
          logits->assign(size_t(vocab), 0.f);
          if (e->api()->logits(s.seq(), 0, logits->data()) != 0) logits->clear();
          return out;
        };
        std::vector<float> ld, lw, lm;
        const auto td = turns(dev.get(), &ld), tw = turns(win.get(), &lw), tm = turns(mtp.get(), &lm);
        check(td.size() == size_t(N / 2) * 2 && tw == td && !ld.empty() && lw == ld && tm == td,
              std::string(method) + (fp8 ? ", fp8" : ", binary16") +
                  ": stagings of 64 positions, prefills a window of keys at a time and short ones in host memory: the "
                  "logits and tokens of the K/V on the GPU, MTP's tokens too");
      }
  }
  {  // a snapshot carries the copy with the cache: loaded, the tokens of a session that prefilled it all
    auto host = make(model, backend, true, g_block, true, narrow, false, 512, kReferenceWidth, false, 0, "strata");
    auto exact = make(model, backend, true, g_block, true, "", false, 512, kReferenceWidth, false, 0);
    snapshot_through_host(host.get(), exact.get(), prompt, "snapshot, strata", false);
  }
  // Capture with a whole prefix and a partial 16-position tile, then reuse the same sequence for a much shorter
  // edited prompt. Restoring the original conversation grows the prefix again. Tiny-model tokens alone can miss
  // a stale copy or a missing grid; compare the last round's logits on each turn too.
  for (bool drafted : {false, true}) {
    const char* method = drafted ? "strata:3" : "strata:2";
    const std::string opts = narrow + ",head_cert=0";
    auto sm = make(model, backend, drafted, "mtp", true, opts + ",stage_dma=0", false, 512, kReferenceWidth,
                   false, 0, method, 0, 64);
    auto dma = make(model, backend, drafted, "mtp", true, opts, false, 512, kReferenceWidth, false, 0, method, 0, 64);
    SnapshotStore store(uint64_t(1) << 28);
    Session control(sm.get()), candidate(dma.get());
    const std::string label = std::string(method) + (drafted ? ", sampled MTP" : ", greedy plain") + ", default DMA";
    if (!control.init(&err) || !candidate.init(&err)) {
      check(false, label + ": creating sessions: " + err);
      continue;
    }
    check(dma->kv_layout().copy_host >= 2, label + ": both staging slots in use");
    SamplingParams sampling;
    if (drafted) {
      sampling.temperature = 0.8f;
      sampling.seed = 7919;
    }
    const int vocab = sm->model().desc().vocab;
    auto compare = [&](const std::vector<int32_t>& p, const std::string& turn) {
      const Out a = gen(control, p, 21, nullptr, sampling), b = gen(candidate, p, 21, nullptr, sampling);
      std::vector<float> la(size_t(vocab), 0.f), lb(size_t(vocab), 0.f);
      check(a.tokens.size() == 21 && a.tokens == b.tokens && sm->api()->logits(control.seq(), 0, la.data()) == 0 &&
                dma->api()->logits(candidate.seq(), 0, lb.data()) == 0 && la == lb,
            label + ": " + turn + " gives the kernel copy's logits and tokens");
      return b.usage.reused_tokens;
    };
    std::vector<int32_t> history(333);
    for (size_t i = 0; i < history.size(); ++i) history[i] = int32_t((17 * i + i / 13) % size_t(vocab));
    compare(history, "long prompt");
    const int kept = candidate.save_to(store, &err);
    check(kept >= 256, label + ": a prompt checkpoint kept for restore");
    std::vector<int32_t> edit(history.begin(), history.begin() + 45);
    edit.front() = (edit.front() + 1) % vocab;
    compare(edit, "shorter edit");
    control.set_store(&store);
    candidate.set_store(&store);
    history.push_back(edit.front());
    check(compare(history, "restored prompt") == kept, label + ": reused the saved prefix");
  }
  auto both = make(model, backend, true, "all", true, narrow, false, 512, -1, false, -1, "strata");
  std::vector<size_t> pick;
  for (size_t i = 0; i < both->widths().size(); ++i)
    if (both->widths()[i] == 0) {
      pick.push_back(i);
      break;
    }
  for (int d = 0; d < 4; ++d) {
    size_t best = both->widths().size();
    for (size_t i = 0; i < both->widths().size(); ++i)
      if (both->round_drafters()[i] == d && (best == both->widths().size() || both->widths()[i] < both->widths()[best]))
        best = i;
    if (best < both->widths().size()) pick.push_back(best);
  }
  auto other = prompt;
  other[0] = (other[0] + 7) % 300;
  const std::vector<std::vector<int32_t>> prompts = {prompt, other, prompt};
  batch_rounds(both.get(), std::vector<std::vector<int32_t>>(prompts.begin(),
                                                              prompts.begin() + std::ptrdiff_t(std::min<size_t>(3, pick.size()))),
               std::vector<size_t>(pick.begin(), pick.begin() + std::ptrdiff_t(std::min<size_t>(3, pick.size()))),
               "strata: rounds of several sequences, plain and each drafter");
}

// The tensor-core round attention for every row (rows_attention_from=0: by default it takes rows of 8,192 keys or
// more, past what the tiny model reaches), over binary16 and FP8 caches: drafted rounds give plain decoding's tokens.
void rows_on_tensor_cores(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
                          const std::vector<int32_t>& want) {
  if (backend != "cuda") return;
  const int N = static_cast<int>(want.size());
  const std::string all = "rows_attention_from=0";
  std::string err;
  auto tokens = [&](Engine* e) {
    Session s(e);
    if (!s.init(&err)) return std::vector<int32_t>{-1};
    return gen(s, prompt, N).tokens;
  };
  struct Config {
    bool fp8;
    const char* extra;  // backend options beside `all`
    const char* name;
  };
  // q and p in the kernel's default parts, in one binary16 part (the default on the 27B's shape), and
  // q as e4m3 on FP8 tensor cores (binary16 q on GPUs without them)
  const Config configs[] = {{false, "", "binary16"},
                            {true, "", "fp8"},
                            {true, ",rows_attention_parts=1", "fp8, q and p in one part"},
                            {true, ",rows_attention_q=fp8", "fp8, q in e4m3"}};
  for (const Config& c : configs) {
    const std::string opt = all + c.extra;
    auto plain = make(model, backend, false, "", c.fp8, opt, false, 512, kReferenceWidth);
    auto spec = make(model, backend, true, g_block, c.fp8, opt, false, 512, kReferenceWidth);
    auto mtp = make(model, backend, true, "mtp", c.fp8, opt, false, 512, kReferenceWidth);
    auto chosen = make(model, backend, true, g_block, c.fp8, opt, false, 512, -1);
    auto host = make(model, backend, true, g_block, c.fp8, opt, false, 512, kReferenceWidth, false, 0);
    const auto t = tokens(plain.get());
    check(t.size() == size_t(N) && tokens(spec.get()) == t && tokens(mtp.get()) == t && tokens(chosen.get()) == t &&
              tokens(host.get()) == t,
          std::string("rounds' attention on tensor cores, ") + c.name +
              ": DFlash2, MTP and chosen widths decode == plain decode, the K/V in host memory too");
  }
}

// A model without a drafter: plain decoding against the reference, and the
// session paths that do not need drafts.
// The drafter's own trees (FLYX_DRAFT_TREE: DFlash2's candidates, best first, ops/draft/ref.h): its trees of every
// size taking turns with its chain give the reference's tokens; chosen every round with the chains, too; sampled,
// plain decoding's for one seed; with oracle drafts (a chain of them in a tree round) the long accepted runs; and a
// session continued from inside its last rounds.
void drafted_trees(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
                   const std::vector<int32_t>& want, const std::vector<int32_t>& oracle) {
  const int N = static_cast<int>(want.size());
  auto trees = make(model, backend, true, g_block, false, "", false, 512, -1, true);
  std::vector<size_t> turns;
  for (size_t i = 0; i < trees->widths().size(); ++i)
    if (trees->round_trees()[i]) turns.push_back(i);
  check(turns.size() == plan::draft_tree_widths().size(), "DFlash2 drafts trees of every size");
  for (size_t i = 0; i < trees->widths().size(); ++i)  // and its chains
    if (!trees->round_trees()[i] && trees->widths()[i] > 0) turns.push_back(i);
  std::string err;
  auto turn_rounds = [&](const std::vector<int32_t>* with_oracle, int* tree_accepted) {
    Session s(trees.get());
    init_checked(s);
    if (with_oracle) trees->api()->set_oracle(s.seq(), with_oracle->data(), static_cast<int32_t>(with_oracle->size()));
    s.rounds().cycle(turns);
    std::vector<flyx_round_result> trace;
    const auto out = gen(s, prompt, N, &trace).tokens;
    *tree_accepted = 0;
    for (size_t r = 1; r < trace.size(); ++r)
      if (trees->round_trees()[turns[(r - 1) % turns.size()]]) *tree_accepted += trace[r].n_accepted;
    if (with_oracle) trees->api()->set_oracle(s.seq(), nullptr, 0);
    return out;
  };
  int accepted = 0, oracle_accepted = 0;
  const bool same = turn_rounds(nullptr, &accepted) == want;
  check(same, "DFlash2 trees taking turns with its chains: the reference's tokens (" + std::to_string(accepted) +
                  " drafts accepted in tree rounds)");
  const bool same_oracle = turn_rounds(&oracle, &oracle_accepted) == want;
  check(same_oracle && oracle_accepted > N / 2,
        "  with oracle drafts: long accepted runs (" + std::to_string(oracle_accepted) + " in tree rounds, " +
            std::to_string(N) + " tokens)");
  {
    Session s(trees.get());
    init_checked(s);
    check(gen(s, prompt, N).tokens == want, "  chosen every round with its chains: the reference's tokens");
    auto p2 = prompt;
    p2.insert(p2.end(), want.begin(), want.begin() + N - 2);
    check(gen(s, p2, 2).tokens == std::vector<int32_t>(want.end() - 2, want.end()),
          "  continued from inside its last rounds");
  }
  GenerateRequest rq;
  rq.prompt = prompt;
  rq.max_new_tokens = N;
  rq.ignore_eos = true;
  rq.sampling = SamplingParams{0.8f, 1.f, 0, 11};
  auto sampled = [&](Engine* e, bool cycle) {
    Session s(e);
    init_checked(s);
    if (cycle) s.rounds().cycle(turns);
    std::vector<int32_t> toks;
    GenerateResult res;
    generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
      toks.insert(toks.end(), t, t + nt);
      return true;
    }, &res, &err);
    return toks;
  };
  auto plain = make(model, backend, false);
  const auto a = sampled(plain.get(), false);
  check(static_cast<int>(a.size()) == N && sampled(trees.get(), true) == a,
        "  sampled (T=0.8): trees taking turns == plain for one seed");
}

// DV's rounds launched one at a time on session s from the end of its prompt (its first token decoded): round k
// (0: DV's first, a star of MASKs) verifies, every `every` rounds from round `first`, the chain make_tiny.py's
// dv_chains plants there (the test's oracle: its first a drafts are plain decoding's tokens `want`, the others one past
// theirs, a from `chains` in turn, the file's own in `file_chains` where it has them), else DV's tree; until the rounds
// have emitted `want` (or `most` rounds).
std::vector<flyx_round_result> dv_chain_rounds(Engine* e, Session& s, size_t plan, size_t prompt_len,
                                               const std::vector<int32_t>& want, const json& golden,
                                               const std::string& file, int most) {
  const auto* api = e->api();
  const bool own = golden.contains("file_chains") && golden["file_chains"].contains(file);
  const auto chains = (own ? golden["file_chains"][file] : golden["chains"]).get<std::vector<int>>();
  const int every = golden["every"].get<int>(), first = golden["first"].get<int>();
  const int vocab = e->model().desc().vocab, width = e->widths()[plan];
  const int32_t P = static_cast<int32_t>(prompt_len), N = static_cast<int32_t>(want.size());
  flyx_gen_params g{};
  g.top_p = 1.f;
  std::vector<flyx_round_result> out;
  for (int k = 0, position = P; k < most && position < P + N; ++k) {
    std::vector<int32_t> oracle;  // by position; past the case's tokens 0, as the chain reads past the oracle
    if (k % every == first) {
      const int a = chains[size_t(k / every) % chains.size()];
      oracle.assign(size_t(P + N), 0);
      for (int t = 0; t < width && position + 1 + t < P + N; ++t) {
        const int32_t truth = want[size_t(position + 1 + t - P)];
        oracle[size_t(position + 1 + t)] = t < a ? truth : (truth + 1) % vocab;
      }
    }
    flyx_round_result r{};
    if (api->set_oracle(s.seq(), oracle.empty() ? nullptr : oracle.data(), static_cast<int32_t>(oracle.size())) != 0 ||
        api->launch(e->round_plan(plan), s.seq(), &g) != 0 || api->read(s.seq(), &r) != 0) {
      check(false, std::string("DV rounds one at a time: ") + api->last_error());
      break;
    }
    out.push_back(r);
    position = r.position;
  }
  api->set_oracle(s.seq(), nullptr, 0);
  return out;
}

// DV's first two trees after a fresh start at each point k of a case (make_tiny.py dv_fresh): a session of its own
// from the prompt and tokens[:k], so that the round of DV's star runs one canvas pass from MASK, the round after it
// drafts from that pass alone (first), and the one after that from the canvas it slid (second: the canvas's input
// from its guesses and expected embeddings, the slide).  A tree's start: the position its round starts at; -1 where
// the prefill and the star did not give tokens[k] and tokens[k + 1] (the text after is not the reference's).
struct DvTrees {
  std::vector<int> start;
  std::vector<std::vector<int32_t>> drafts, parents;
  void add(const std::vector<flyx_round_result>& trace, size_t i, bool ok) {
    ok = ok && trace.size() > i;
    const flyx_round_result* r = ok ? &trace[i] : nullptr;
    start.push_back(ok ? r->position - r->n_tokens : -1);
    drafts.emplace_back(ok ? std::vector<int32_t>(r->drafts, r->drafts + r->n_drafts) : std::vector<int32_t>());
    parents.emplace_back(ok ? std::vector<int32_t>(r->parents, r->parents + r->n_drafts) : std::vector<int32_t>());
  }
};
struct DvFresh {
  DvTrees first, second;
};
DvFresh dv_fresh(Engine* e, const std::vector<int32_t>& prompt, const std::vector<int32_t>& tokens, int width) {
  DvFresh f;
  for (size_t k = 0; k + 2 < tokens.size(); ++k) {
    std::vector<int32_t> p = prompt;
    p.insert(p.end(), tokens.begin(), tokens.begin() + std::ptrdiff_t(k));
    Session s(e);
    std::string err;
    init_checked(s);
    std::vector<flyx_round_result> trace;
    const Out o = gen(s, p, width + 3, &trace);  // past the second tree's round whatever the first accepts
    const bool ok = o.tokens.size() >= 2 && o.tokens[0] == tokens[k] && o.tokens[1] == tokens[k + 1];
    f.first.add(trace, 2, ok);
    f.second.add(trace, 3, ok);
  }
  return f;
}

// How many of DV's fresh trees (and their drafts) are another run's: the trees of rounds that start where the other
// run's do (in step), the same draft for draft and parent for parent, and their drafts the same place by place.
struct FreshSame {
  int samples = 0, in_step = 0, trees = 0, drafts = 0, drafts_same = 0;
  void add(const DvTrees& a, const DvTrees& b) {
    for (size_t i = 0; i < a.drafts.size() && i < b.drafts.size(); ++i) {
      ++samples;
      if (a.start[i] < 0 || a.start[i] != b.start[i]) continue;
      ++in_step;
      trees += a.drafts[i] == b.drafts[i] && a.parents[i] == b.parents[i];
      for (size_t j = 0; j < b.drafts[i].size(); ++j, ++drafts)
        drafts_same += j < a.drafts[i].size() && a.drafts[i][j] == b.drafts[i][j];
    }
  }
  // The reference's trees ({start, drafts, parents} a start).
  static DvTrees of(const json& ref, const char* which) {
    DvTrees t;
    for (const auto& r : ref) {
      const json& x = r[which];
      t.start.push_back(x["start"].get<int>());
      t.drafts.push_back(x["drafts"].get<std::vector<int32_t>>());
      t.parents.push_back(x["parents"].get<std::vector<int32_t>>());
    }
    return t;
  }
  std::string str() const {
    return std::to_string(trees) + " of " + std::to_string(in_step) + " trees (" + std::to_string(drafts_same) +
           " of " + std::to_string(drafts) + " drafts), " + std::to_string(in_step) + " of " +
           std::to_string(samples) + " in step";
  }
};

// DV's rounds chosen among plain decoding's on a backend that runs them: the rounds to choose from
// (plain decoding, DV's trees of 15 and of the file's width, its warm-up: plain decoding and the canvas pass), and, on
// a case's prompt, rounds that mix them (the test's cycles, two in flight, with graphs and without) give plain
// decoding's tokens (gate 4); a DV round after a round of another kind drafts DV's first tree (its star: the canvas
// forgotten), after its warm-up the tree a DV round after its star drafts, draft for draft (the warm-up leaves the
// canvas DV's first round leaves), in rounds of several sequences too (launch_batch: each tree after its warm-up's
// alone); a stop string, a rewind into the rounds and a checkpoint restored with rounds mixed; the controller's own
// choice (graphs on) the same tokens; sampled for one seed and with penalties, rounds mixed and chosen, plain
// decoding's sampled tokens (gate 4); and (`more`) on FP8 K/V and with the K/V in host memory (binary16 and FP8),
// rounds mixed and chosen, plain decoding's tokens; and beside lookup rounds on a text that repeats, the rounds chosen
// at the test's costs: no lookup round while DV drafts, lookup rounds after it leaves, no DV tree on a canvas a lookup
// round left behind, and on the backend a lookup round between DV's rounds leaving the canvas behind (the next tree its
// star), DV's warm-up after it bringing it along: plain decoding's tokens; beside wide lookup rounds (wide=1) the same.
// With wide=0 DV's trees past 15 drafts are refused and its trees of 15 run.
void dv_mixed(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
              const std::vector<int32_t>& want, const std::vector<int32_t>& longer, int mask, const std::string& at,
              bool more) {
  auto e = make(model, backend, true, "dv", false, "", false, 512, -1);
  auto eager = make(model, backend, true, "dv", false, "graphs=0", false, 512, -1);
  size_t plain = 0, warm = 0, narrow = 0, wide = 0;
  int widest = 0, narrowest = 1 << 30;
  bool named = true, bare = true;
  for (size_t i = 0; i < e->widths().size(); ++i) {
    const int d = e->round_drafters()[i], w = e->widths()[i];
    if (d < 0) plain = i;
    else if (w == 0) warm = i, named = named && e->round_name(i) == "dv warm-up";
    else {
      bare = bare && e->round_trees()[i] && e->round_trunks()[i] < 0;
      if (w > widest) widest = w, wide = i;
      if (w < narrowest) narrowest = w, narrow = i;
    }
  }
  const flyx_model_desc& desc = e->model().desc();
  const int file_width = std::min(FLYX_MAX_WIDE_ROWS - 1, desc.blocks[desc.n_blocks - 1].dv.tree_width);
  const std::vector<int> expect_widths = file_width > FLYX_MAX_ROWS - 1
                                             ? std::vector<int>{0, FLYX_MAX_ROWS - 1, file_width, 0}
                                             : std::vector<int>{0, file_width, 0};
  check(e->widths() == expect_widths && named && bare && e->refused().empty() && e->widths()[warm] == 0 &&
            e->round_drafters()[warm] == 0 && e->round_drafters()[plain] == -1,
        at + "the rounds to choose from: plain decoding, DV's trees of 15 and of the file's width, its warm-up (" +
            std::to_string(e->widths().size()) + " rounds)");
  // Without wide rows (backend option wide=0) a round verifies 16 rows: DV's trees of the file's width past 15 are
  // refused and said, its trees of 15 run: plain decoding's tokens.
  if (file_width > FLYX_MAX_ROWS - 1) {
    auto x0 = make(model, backend, true, "dv", false, "wide=0", false, 512, -1);
    const std::string refused = x0->refused();
    const bool said = refused.find("dv " + std::to_string(file_width) + ":") != std::string::npos;
    std::vector<int32_t> got;
    {
      Session s(x0.get());
      init_checked(s);
      got = gen(s, prompt, static_cast<int>(want.size())).tokens;
    }
    check(said && x0->widths() == std::vector<int>({0, FLYX_MAX_ROWS - 1, 0}) && got == want,
          at + "wide=0: DV's trees of " + std::to_string(file_width) + " refused (" + refused +
              "), its trees of 15 and its warm-up beside plain decoding: plain decoding's tokens");
  }
  // Their costs at the context they run at: timed at load at the prompt's context and again
  // after a longer one where the context takes it (2048 tokens: at 1024), not where it does not (512).
  if (more) {
    auto at_context = make(model, backend, true, "dv", false, "", false, 2048, -1);
    check(at_context->round_ms_again_at() > at_context->round_ms_at() && e->round_ms_again_at() == 0,
          at + "  DV's rounds timed again after " + std::to_string(at_context->round_ms_again_at()) + " tokens of "
               "context (" + std::to_string(at_context->round_ms_at()) + " first), not at a context of 512");
  }
  const int N = static_cast<int>(want.size());
  std::string err;
  // A run of the case's prompt with rounds launched as `cycle` says (empty: as the controller chooses), two in flight;
  // its ring entries and the round each was.
  struct Run {
    std::vector<int32_t> tokens;
    std::vector<flyx_round_result> trace;
    std::vector<int> rounds;
    std::string stop;
  };
  auto run = [&](Engine* x, const std::vector<size_t>& cycle, const std::vector<int32_t>& p, int n,
                 const std::vector<std::string>& stop = {}, const SamplingParams& sampling = SamplingParams{},
                 const std::function<void(RoundChooser&)>& setup = nullptr) {
    Run r;
    Session s(x);
    init_checked(s);
    s.rounds().cycle(cycle);
    if (setup) setup(s.rounds());
    s.trace = &r.trace;
    s.trace_rounds = &r.rounds;
    GenerateRequest rq;
    rq.prompt = p;
    rq.max_new_tokens = n;
    rq.ignore_eos = true;
    rq.stop = stop;
    rq.sampling = sampling;
    GenerateResult res;
    if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
          r.tokens.insert(r.tokens.end(), t, t + nt);
          return true;
        }, &res, &err) != 0)
      check(false, at + "mixed rounds: generate: " + err);
    s.trace = nullptr;
    s.trace_rounds = nullptr;
    r.stop = res.stop_string;
    return r;
  };
  auto star = [&](const flyx_round_result& r) {
    bool all = r.n_drafts > 0;
    for (int j = 0; j < r.n_drafts; ++j) all = all && r.drafts[j] == mask && r.parents[j] == -1;
    return all;
  };
  // Every DV round after the prefill or a round of another kind drafts its star; after its warm-up or its own round,
  // a tree grown from the canvas (not the star).
  auto canvas_followed = [&](const Run& r, int* stars, int* grown) {
    bool ok = r.trace.size() == r.rounds.size();
    *stars = *grown = 0;
    for (size_t i = 1; ok && i < r.trace.size(); ++i) {
      const int k = r.rounds[i], before = r.rounds[i - 1];
      if (k < 0 || e->widths()[size_t(k)] == 0 || e->round_drafters()[size_t(k)] < 0) continue;
      const bool follows = before >= 0 && e->round_drafters()[size_t(before)] >= 0;
      if (follows) ok = ok && !star(r.trace[i]), ++*grown;
      else ok = ok && star(r.trace[i]), ++*stars;
    }
    return ok;
  };
  const std::vector<std::vector<size_t>> cycles = {
      {warm, wide, wide, plain, narrow, plain, plain, warm, narrow, wide},
      {wide, plain, narrow, warm, narrow, plain, wide, wide},
      {plain, warm, narrow, wide, narrow, narrow, wide, plain, plain, plain, warm, wide}};
  int stars = 0, grown = 0;
  bool tokens = true, followed = true;
  for (const auto& cycle : cycles) {
    const Run r = run(e.get(), cycle, prompt, N);
    int s = 0, g = 0;
    tokens = tokens && r.tokens == want;
    followed = followed && canvas_followed(r, &s, &g);
    stars += s;
    grown += g;
  }
  check(tokens, at + "rounds mixed (three cycles of plain decoding, DV's trees and its warm-up, two in flight): plain "
                     "decoding's tokens");
  check(followed && stars > 0 && grown > 0,
        at + "  a DV round after another kind of round drafts its star (" + std::to_string(stars) + "), after its "
             "warm-up or its own a grown tree (" + std::to_string(grown) + ")");
  // The warm-up leaves the canvas DV's first round leaves: the same rounds with the warm-up for the star give the same
  // trees after it, round for round, and the same tokens.
  {
    const std::vector<size_t> with_warm = {plain, warm, wide, wide, narrow};
    const std::vector<size_t> with_star = {plain, wide, wide, wide, narrow};
    const Run a = run(e.get(), with_warm, prompt, N), b = run(e.get(), with_star, prompt, N);
    bool same = a.tokens == want && b.tokens == want && a.trace.size() == b.trace.size();
    int compared = 0;
    for (size_t i = 1; same && i < a.trace.size(); ++i) {
      const flyx_round_result &x = a.trace[i], &y = b.trace[i];
      same = same && x.position == y.position && x.n_tokens == y.n_tokens;
      if (!same || a.rounds[i] != b.rounds[i]) continue;  // the warm-up and the star
      if (e->round_drafters()[size_t(a.rounds[i])] < 0) continue;
      same = same && x.n_drafts == y.n_drafts && x.n_accepted == y.n_accepted &&
             std::equal(x.drafts, x.drafts + x.n_drafts, y.drafts) &&
             std::equal(x.parents, x.parents + x.n_drafts, y.parents);
      ++compared;
    }
    check(same && compared > 0, at + "  its warm-up and its star leave the same canvas: the trees after them the same, "
                                     "round for round (" + std::to_string(compared) + " trees)");
  }
  // Its warm-ups in rounds of several sequences (launch_batch, as a --cycle naming them has the scheduler launch them):
  // each sequence's canvas pass its own, a sequence at a time, so the tree each drafts after it is the one it drafts
  // after its warm-up alone, draft for draft, round after round (past BatchRounds::kGraphWarmHits from the batch's
  // graph).  The reply's tokens are plain decoding's whatever the canvas holds; the trees after it are not.
  {
    const auto* api = e->api();
    std::vector<int32_t> other = prompt;
    other.insert(other.end(), longer.begin(),
                 longer.begin() + std::min<std::ptrdiff_t>(8, std::ptrdiff_t(longer.size())));
    const std::vector<std::vector<int32_t>> prompts = {prompt, other, prompt};
    const size_t n = prompts.size();
    std::vector<std::unique_ptr<Session>> alone, together;
    std::vector<flyx_seq*> seqs;
    bool made = true;  // six sessions: an engine with room for fewer refuses one
    for (const auto& p : prompts) {
      for (auto* v : {&alone, &together}) {
        v->push_back(std::make_unique<Session>(e.get()));
        made = made && v->back()->init(&err);
        if (made) gen(*v->back(), p, 1);  // at the end of its prompt, its first token the next input
      }
      seqs.push_back(together.back()->seq());
    }
    if (!made) std::printf("  %swarm-ups in rounds of several sequences: a session: %s\n", at.c_str(), err.c_str());
    flyx_plan* const warm_plan = e->round_plan(warm);
    flyx_plan* const tree_plan = e->round_plan(narrow);
    const std::vector<flyx_plan*> warms(n, warm_plan);
    flyx_gen_params g{};
    g.top_p = 1.f;
    const std::vector<const flyx_gen_params*> gens(n, &g);
    auto same_entry = [](const flyx_round_result& x, const flyx_round_result& y) {
      return x.n_tokens == y.n_tokens && x.n_accepted == y.n_accepted && x.position == y.position &&
             x.stopped == y.stopped && x.n_drafts == y.n_drafts &&
             std::equal(x.tokens, x.tokens + x.n_tokens, y.tokens) &&
             std::equal(x.drafts, x.drafts + x.n_drafts, y.drafts) &&
             std::equal(x.parents, x.parents + x.n_drafts, y.parents);
    };
    bool ok = made;
    int trees = 0, same = 0, grown = 0;
    for (int r = 0; r < 12 && ok; ++r) {
      std::vector<flyx_round_result> a(n), b(n), ta(n), tb(n);
      for (size_t i = 0; i < n && ok; ++i)
        ok = api->launch(warm_plan, alone[i]->seq(), &g) == 0 && api->read(alone[i]->seq(), &a[i]) == 0 &&
             api->launch(tree_plan, alone[i]->seq(), &g) == 0 && api->read(alone[i]->seq(), &ta[i]) == 0;
      ok = ok && api->launch_batch(warms.data(), seqs.data(), gens.data(), static_cast<int32_t>(n)) == 0;
      for (size_t i = 0; i < n && ok; ++i) ok = api->read(seqs[i], &b[i]) == 0;
      for (size_t i = 0; i < n && ok; ++i)
        ok = api->launch(tree_plan, seqs[i], &g) == 0 && api->read(seqs[i], &tb[i]) == 0;
      if (!ok) std::printf("  %swarm-ups in rounds of several sequences: %s\n", at.c_str(), api->last_error());
      for (size_t i = 0; ok && i < n; ++i, ++trees) {
        same += same_entry(a[i], b[i]) && same_entry(ta[i], tb[i]);
        grown += !star(ta[i]) && !star(tb[i]);
      }
    }
    check(ok && trees > 0 && same == trees && grown == trees,
          at + "  its warm-ups in rounds of several sequences (launch_batch): the tree after each its warm-up's "
               "alone, grown, draft for draft (" + std::to_string(same) + " of " + std::to_string(trees) + ")");
  }
  // Graphs off (every node its own launch): the same rounds, tree for tree.
  {
    const Run a = run(e.get(), cycles[0], prompt, N), b = run(eager.get(), cycles[0], prompt, N);
    bool same = a.tokens == want && b.tokens == want && a.trace.size() == b.trace.size();
    for (size_t i = 1; same && i < a.trace.size(); ++i)
      same = a.trace[i].n_drafts == b.trace[i].n_drafts &&
             std::equal(a.trace[i].drafts, a.trace[i].drafts + a.trace[i].n_drafts, b.trace[i].drafts);
    check(same, at + "  with the backend's graphs off: the same rounds, draft for draft");
  }
  // The controller's own choice: plain decoding's tokens, and its rounds' canvas followed.
  {
    const Run r = run(e.get(), {}, prompt, N);
    int s = 0, g = 0;
    check(r.tokens == want && canvas_followed(r, &s, &g),
          at + "rounds the controller chooses: plain decoding's tokens, every DV tree on a canvas that followed (" +
              std::to_string(g) + " grown, " + std::to_string(s) + " stars)");
  }
  // Sampled for one seed, with top-p and top-k, and with penalties: the three cycles' rounds and the controller's give
  // plain decoding's sampled tokens.
  {
    auto plain_e = make(model, backend, false);
    for (const SamplingParams& sp : {SamplingParams{0.8f, 1.f, 0, 11}, SamplingParams{1.0f, 0.9f, 40, 12345},
                                     SamplingParams{1.0f, 0.9f, 40, 4711, 1.5f, 0.25f}}) {
      const Run base = run(plain_e.get(), {}, prompt, N, {}, sp);
      bool same = base.tokens.size() == size_t(N);
      int stars = 0, grown = 0;
      for (const auto& cycle : cycles) {
        const Run r = run(e.get(), cycle, prompt, N, {}, sp);
        int s = 0, g = 0;
        same = same && r.tokens == base.tokens && canvas_followed(r, &s, &g);
        stars += s, grown += g;
      }
      same = same && run(e.get(), {}, prompt, N, {}, sp).tokens == base.tokens;
      check(same && grown > 0, at + "  sampled (seed " + std::to_string(sp.seed) +
                                   (sp.presence_penalty != 0 ? ", penalties" : "") + "), rounds mixed and chosen: "
                                   "plain decoding's tokens (" + first_difference(run(e.get(), {}, prompt, N, {}, sp)
                                                                                      .tokens, base.tokens) + ")");
    }
  }
  // FP8 K/V, the K/V in host memory (binary16, and FP8 with its scales beside it), and lookup rounds among the
  // controller's: plain decoding's tokens on the same caches, rounds mixed and chosen.
  if (more) {
    const std::vector<int32_t> want8 = run(make(model, backend, false, "", true).get(), {}, prompt, N).tokens;
    // One engine at a time (the test's processes run beside the other GPU tests).
    auto mixed_on = [&](std::unique_ptr<Engine> x, const std::vector<int32_t>& tokens, const std::string& what) {
      bool same = tokens.size() == size_t(N) && run(x.get(), {}, prompt, N).tokens == tokens;
      for (const auto& cycle : cycles) same = same && run(x.get(), cycle, prompt, N).tokens == tokens;
      check(same, at + "  " + what + ", rounds mixed and chosen: plain decoding's tokens");
    };
    mixed_on(make(model, backend, true, "dv", true, "", false, 512, -1), want8, "FP8 K/V");
    mixed_on(make(model, backend, true, "dv", false, "", false, 512, -1, false, 0), want, "K/V in host memory");
    mixed_on(make(model, backend, true, "dv", true, "", false, 512, -1, false, 0), want8, "FP8 K/V in host memory");
  }
  // DV beside lookup rounds (no lookup round while DV runs or would be entered; a lookup round leaves
  // the canvas behind).  The prompt is plain decoding's continuation twice, so that the text a reply writes repeats
  // what came before (a reply with no drafter has lookup rounds), and the session's costs are the test's, so that
  // its rounds do not depend on a load's timings: plain decoding and DV's warm-up 1 ms, DV's trees 0.5 ms (a little
  // more the wider: DV is entered on the guess of its tokens and stays) or 3 ms (entered, and left after trees of
  // fewer tokens than that pays for).
  if (more) {
    std::vector<int32_t> again = longer;
    again.insert(again.end(), longer.begin(), longer.end());
    const int n = 128;
    const std::vector<int32_t> want_again = run(make(model, backend, false).get(), {}, again, n).tokens;
    auto count = [](const Run& r, int round) { return int(std::count(r.rounds.begin(), r.rounds.end(), round)); };
    auto kinds = [&](const Run& r) {  // a letter a round: L lookup, p plain, w warm-up, n and W DV's trees
      std::string k;
      for (int i : r.rounds)
        k += i == -1 ? 'L' : i < 0 ? '.' : i == int(plain) ? 'p' : i == int(warm) ? 'w' : i == int(narrow) ? 'n' : 'W';
      return k;
    };
    const Run looked = run(make(model, backend, true, "none", false, "", true).get(), {}, again, n);
    check(looked.tokens == want_again && count(looked, -1) > 0,
          at + "  the text repeats: a reply without a drafter has lookup rounds (" + std::to_string(count(looked, -1)) +
              ")");
    auto x = make(model, backend, true, "dv", false, "", true, 512, -1);
    auto costs = [&](double tree) {
      std::vector<double> ms(x->widths().size(), 1.0);
      for (size_t i = 0; i < ms.size(); ++i)
        if (x->round_drafters()[i] >= 0 && x->widths()[i] > 0) ms[i] = tree + 0.02 * x->widths()[i];
      return ms;
    };
    const std::vector<double> stays = costs(0.5), leaves = costs(3.0);
    SharedTokens shared;  // the engine's means and credit: the test's, new for each reply
    auto given = [&](const std::vector<double>* ms) {
      return std::function<void(RoundChooser&)>([&, ms](RoundChooser& c) {
        shared.resize(ms->size());
        c.configure(&x->widths(), &x->round_drafters(), ms, &x->lookup_widths(), &x->round_trunks(), nullptr,
                    nullptr, nullptr, 0, &shared);
      });
    };
    // DV stays: its warm-up the reply's first round, its trees every round after it, no lookup round.
    {
      const Run r = run(x.get(), {}, again, n, {}, SamplingParams{}, given(&stays));
      int s = 0, g = 0;
      const bool followed = canvas_followed(r, &s, &g);
      const int trees = static_cast<int>(r.rounds.size()) - 1 - count(r, int(warm));  // the prefill's entry first
      const bool ok = r.tokens == want_again && followed && s == 0 && r.rounds.size() > 2 &&
                      r.rounds[1] == int(warm) && count(r, -1) == 0 && g == trees;
      check(ok, at + "  lookup rounds on, DV drafting: plain decoding's tokens, its warm-up first and its trees after "
                     "it, no lookup round (" + std::to_string(g) + " trees" + (ok ? "" : ": " + kinds(r)) + ")");
    }
    // DV leaves: lookup rounds after it, and no DV tree on a canvas a lookup round or another left behind.
    {
      const Run r = run(x.get(), {}, again, n, {}, SamplingParams{}, given(&leaves));
      int s = 0, g = 0;
      const bool followed = canvas_followed(r, &s, &g);
      const bool ok = r.tokens == want_again && followed && s == 0 && g > 0 && r.rounds.size() > 1 &&
                      r.rounds[1] == int(warm) && count(r, -1) > 0;
      check(ok, at + "  lookup rounds on, DV left: plain decoding's tokens, lookup rounds after DV's, every DV tree "
                     "after its warm-up or its own (" + std::to_string(g) + " trees, " + std::to_string(count(r, -1)) +
                     " lookup rounds" + (ok ? "" : ": " + kinds(r)) + ")");
    }
    // DV beside wide lookup rounds (backend option wide=1: lookup rounds up to 63 drafts), DV left: after lookup rounds
    // of 15 accepted whole, lookup rounds of more (the reply alone: never batched), each leaving the canvas behind as
    // any lookup round does: plain decoding's tokens, every DV tree after its warm-up or its own.
    {
      auto xw = make(model, backend, true, "dv", false, "wide=1", true, 512, -1);
      std::vector<double> ms(xw->widths().size(), 1.0);
      for (size_t i = 0; i < ms.size(); ++i)
        if (xw->round_drafters()[i] >= 0 && xw->widths()[i] > 0) ms[i] = 3.0 + 0.02 * xw->widths()[i];
      SharedTokens sw;
      const Run r = run(xw.get(), {}, again, n, {}, SamplingParams{}, [&](RoundChooser& c) {
        sw.resize(ms.size());
        c.configure(&xw->widths(), &xw->round_drafters(), &ms, &xw->lookup_widths(), &xw->round_trunks(), nullptr,
                    nullptr, nullptr, 0, &sw);
      });
      int s = 0, g = 0, wide_lookups = 0;
      const bool followed = canvas_followed(r, &s, &g);
      for (size_t i = 0; i < r.trace.size() && i < r.rounds.size(); ++i)
        wide_lookups += r.rounds[i] == -1 && r.trace[i].n_drafts > FLYX_MAX_ROWS - 1;
      // Wide rounds follow two lookup rounds of 15 accepted whole: a reply whose text gives fewer than three lookup
      // rounds (the V-LoRA files' continuations repeat less) has none to show.
      const int lookups = count(r, -1);
      const bool ok = r.tokens == want_again && followed && s == 0 && (wide_lookups > 0 || lookups < 3) &&
                      xw->lookup_widths().back() == FLYX_MAX_WIDE_ROWS - 1;
      check(ok, at + "  wide lookup rounds on (wide=1), DV left: plain decoding's tokens, " +
                    std::to_string(wide_lookups) + " of " + std::to_string(lookups) + " lookup rounds of more than 15 "
                    "drafts, every DV tree after its warm-up or its own (" + std::to_string(g) + " trees" +
                    (ok ? "" : ": " + kinds(r)) + ")");
    }
    // On the backend, rounds launched one at a time: a lookup round leaves the canvas behind (DV's next tree its
    // star), and DV's warm-up after it brings the canvas along (the tree after it grown); plain decoding's tokens.
    {
      Session s(x.get());
      init_checked(s);
      GenerateRequest rq;
      rq.prompt = again;
      rq.max_new_tokens = 1;
      rq.ignore_eos = true;
      std::vector<int32_t> got;
      GenerateResult res;
      bool ok = generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
                  got.insert(got.end(), t, t + nt);
                  return true;
                }, &res, &err) == 0 && got.size() == 1;
      const int lookup_width = x->lookup_widths().empty() ? 0 : x->lookup_widths().back();
      std::vector<flyx_round_result> rounds;
      auto launch = [&](flyx_plan* plan, bool looked_up) {
        flyx_gen_params g{};
        std::vector<int32_t> drafts;
        if (looked_up) {  // plain decoding's next tokens, as a lookup round of the widest given width drafts them
          drafts.assign(want_again.begin() + std::ptrdiff_t(got.size()),
                        want_again.begin() + std::ptrdiff_t(got.size() + size_t(lookup_width)));
          g.drafts = drafts.data();
          g.n_drafts = lookup_width;
          g.drafts_at = static_cast<int32_t>(again.size() + got.size());
        }
        flyx_round_result r{};
        ok = ok && x->api()->launch(plan, s.seq(), &g) == 0 && x->api()->read(s.seq(), &r) == 0;
        got.insert(got.end(), r.tokens, r.tokens + r.n_tokens);
        rounds.push_back(r);
      };
      ok = ok && lookup_width > 0 && got.size() + size_t(lookup_width) + 40 < want_again.size();
      if (ok) {
        launch(x->round_plan(warm), false);
        launch(x->round_plan(narrow), false);
        launch(x->lookup_plan(x->lookup_widths().size() - 1), true);
        launch(x->round_plan(narrow), false);
        launch(x->round_plan(warm), false);
        launch(x->round_plan(narrow), false);
      }
      got.resize(std::min(got.size(), want_again.size()));
      ok = ok && rounds.size() == 6 && std::equal(got.begin(), got.end(), want_again.begin());
      check(ok && !star(rounds[1]) && rounds[2].n_accepted == lookup_width && star(rounds[3]) && !star(rounds[5]),
            at + "  rounds launched one at a time: DV's warm-up and a tree grown, a lookup round (" +
                 std::to_string(ok ? rounds[2].n_accepted : -1) + " drafts accepted), DV's tree after it its star, "
                 "after its warm-up again a grown tree: plain decoding's tokens");
    }
  }
  // A stop string ends the output before it, rounds mixed.
  {
    const auto& tk = e->tokenizer();
    const std::string full = tk.decode(want), stop = tk.decode({want.begin() + 10, want.begin() + 12});
    const Run r = run(e.get(), cycles[1], prompt, N, {stop});
    check(!stop.empty() && tk.decode(r.tokens) == full.substr(0, full.find(stop)) && r.stop == stop,
          at + "  a stop string, rounds mixed: the output before it");
  }
  // The session again, rounds mixed: continued from inside its last rounds (a rewind) and from the end of its prompt
  // (the checkpoint there restored): plain decoding's tokens.
  {
    Session s(e.get());
    init_checked(s);
    s.rounds().cycle(cycles[2]);
    auto gen_tokens = [&](const std::vector<int32_t>& p, int n) {
      GenerateRequest rq;
      rq.prompt = p;
      rq.max_new_tokens = n;
      rq.ignore_eos = true;
      std::vector<int32_t> t;
      GenerateResult res;
      if (generate(s, rq, [&](std::string_view, const int32_t* x, int nx) {
            t.insert(t.end(), x, x + nx);
            return true;
          }, &res, &err) != 0)
        check(false, at + "mixed rounds: generate: " + err);
      return std::make_pair(t, res.usage.reused_tokens);
    };
    const auto first = gen_tokens(prompt, N);
    auto p2 = prompt;
    p2.insert(p2.end(), want.begin(), want.begin() + N - 2);
    const auto cont = gen_tokens(p2, 8);
    auto p3 = prompt;
    p3.insert(p3.end(), want.begin(), want.begin() + 20);
    const auto earlier = gen_tokens(p3, N - 20);
    const bool ok2 =
        cont.first.size() == 8 && std::equal(cont.first.begin(), cont.first.end(), longer.begin() + (N - 2));
    const bool ok3 = std::equal(earlier.first.begin(), earlier.first.end(), longer.begin() + 20) &&
                     earlier.first.size() == size_t(N - 20);
    check(first.first == want && ok2 && ok3 && cont.second > int(prompt.size()),
          at + "  the session continued from inside its rounds (" + std::to_string(cont.second) + " reused) and from "
               "an earlier point (" + std::to_string(earlier.second) + "), rounds mixed: plain decoding's tokens");
  }
}

// DV's rounds (docs/design/dv.md) on a backend that runs them, the tiny Llama's DV (a beam tree, fill 0.9; a pool tree,
// fill 0.4, a causal canvas that keeps its own guesses as inputs; 32 slots, a beam of 8 with coverage; 32 slots and
// trees of 31 drafts, rounds of 32 verify rows, a beam of 8 with coverage and a pool of 7 ranks, the beam's also beside
// a V-LoRA (the target's adapters on the 32 rows); the same two trees on 16 slots from a flatter selector, trees that
// take the beam's 8 nodes a depth and the pool's 7th rank; the beam of 15 beside a V-LoRA: make_tiny.py): they give
// plain decoding's tokens (gate 4) greedy, sampled for one seed, with penalties, on FP8 K/V, with the K/V in host
// memory, with the test's chains (long accepted runs: the canvas slides past every slot); after a rewind into the last
// DV round, and to where the last slide left the canvas with another token next, and a checkpoint restored there (DV
// starts over: its first tree), a checkpoint restore, an edited prompt (drafts and all, a fresh session's), a snapshot
// through host memory, and with a stop string; and their drafts are the float64 reference's rounding what this backend
// rounds (golden_llama_dv.json, make_tiny.py: flyx_ref/dv.py on the file as stored), round for round, with chains
// planted every third round so that the canvas slides by 1 to 16 (to 32, past every slot, with trees of 31; the tree
// after a slide past every slot the reference's exactly), and the first two trees after a fresh start at every point,
// binary16 and FP8 K/V; with the K/V in host memory (binary16, and FP8 with its scales staged beside it), the trees
// drafted with them on the GPU.
void dv_rounds(const json& g, const std::string& model, const std::string& backend) {
  namespace fs = std::filesystem;
  std::ifstream rf(fs::path(model).parent_path() / "golden_llama_dv.json");
  const json golden = rf ? json::parse(rf) : json();
  const std::string file = fs::path(model).filename().string();
  const json* reference = golden.contains("files") && golden["files"].contains(file) ? &golden["files"][file] : nullptr;
  auto json_at = [&](const char* part) {
    return golden.contains(part) && golden[part].contains(file) ? &golden[part][file] : nullptr;
  };
  const json *fresh_ref = json_at("fresh"), *fp8_ref = json_at("fp8");
  // DV's rounds alone, its file's width every round (pinned: the rounds its references draft, held to them below);
  // its rounds chosen among plain decoding's: dv_mixed.
  int pin = 0;
  {
    auto chosen = make(model, backend, true, "dv", false, "", false, 512, -1);
    for (size_t i = 0; i < chosen->widths().size(); ++i)
      if (chosen->round_drafters()[i] >= 0) pin = std::max(pin, chosen->widths()[i]);
    // Prewarms after rounds chosen among plain decoding's, as a server's (its default)
    prefill_only_checks(chosen.get(), g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  }
  auto dv = make(model, backend, true, "dv", false, "", false, 512, pin);
  auto dv8 = make(model, backend, true, "dv", true, "", false, 512, pin);
  auto dv_host = make(model, backend, true, "dv", false, "", false, 512, pin, false, 0);  // K/V all in host memory
  auto dv8_host = make(model, backend, true, "dv", true, "", false, 512, pin, false, 0);  // FP8 K/V, all there too
  auto plain = make(model, backend, false);
  auto plain8 = make(model, backend, false, "", true);
  check(backend != "cuda" || (dv_host->kv_layout().device_positions == 0 &&
                               dv8_host->kv_layout().device_positions == 0),
        "DV, K/V all in host memory: placed so (" + std::to_string(dv_host->kv_layout().device_positions) + " and " +
            std::to_string(dv8_host->kv_layout().device_positions) + " FP8 on the GPU)");
  // Trees of drafts read every position: at the load's default method DV's rounds read the caches
  // exactly, and strata, which reads chains of drafts only, is refused at load.
  {
    auto dv_auto = make(model, backend, true, "dv", false, "", false, 512, -1, false, -1, "auto");
    EngineOptions o;
    o.backend = backend;
    o.drafter = "dv";
    o.kv_method = "strata";
    o.kv = FLYX_KV_16;
    o.max_ctx = 512;
    Engine strata;
    std::string why;
    const bool loaded = strata.load(model, o, &why);
    check(dv_auto->kv_layout().method == FLYX_KV_READ_ALL && !loaded &&
              why.find("strata attention reads chains of drafts only") != std::string::npos,
          "DV: the default K/V method reads every position (method " +
              std::to_string(dv_auto->kv_layout().method) + "), strata refused (" +
              (loaded ? std::string("loaded") : why) + ")");
    // The method is the file's, not the drafter's (gate 4): with the K/V past 64 positions in host memory, where auto
    // reads strata past its switch, the file decoded without DV reads every position too, so naming DV changes no
    // token.  (The tiny Llama's 8 query heads a KV head are more than strata takes, so its tokens would not tell: the
    // decision is what is checked.)
    auto dv_spilled = make(model, backend, true, "dv", false, "", false, 512, -1, false, 64, "auto");
    auto none_spilled = make(model, backend, false, "", false, "", false, 512, kReferenceWidth, false, 64, "auto");
    check(dv_spilled->kv_layout().method == FLYX_KV_READ_ALL && none_spilled->kv_layout().method == FLYX_KV_READ_ALL,
          "a DV file, K/V past 64 positions in host memory, auto: every position read with DV (method " +
              std::to_string(dv_spilled->kv_layout().method) + ") and without (" +
              std::to_string(none_spilled->kv_layout().method) + ")");
  }
  // Prompts passed with no token generated (prewarms), after DV's rounds left their tree rows past the position and
  // its canvas behind: what a cold session gives, the prompt's state reused.
  prefill_only_checks(dv.get(), g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  const int slots = dv->model_file().header()["dv"]["slots"].get<int>();
  // The generic regions (gate 2): past 16 slots the CUDA canvas pass groups its GEMMs (gemm_draft), which fusion=0
  // runs one by one; the same sums, so the same trees.
  auto dv_generic =
      backend == "cuda" && slots > 16 ? make(model, backend, true, "dv", false, "fusion=0", false, 512, pin) : nullptr;
  size_t plan = dv->widths().size();
  for (size_t i = 0; i < dv->widths().size(); ++i)
    if (dv->round_drafters()[i] >= 0) plan = i;
  check(plan < dv->widths().size() && dv->widths().size() == 1 && dv->round_trees()[plan] &&
            std::string(dv->drafter()) == "dv" && dv->refused().empty(),
        "DV: its rounds alone, a tree of " + std::to_string(plan < dv->widths().size() ? dv->widths()[plan] : 0) +
            " drafts");
  if (plan == dv->widths().size()) return;
  const int width = dv->widths()[plan];
  const int mask = dv->model_file().header()["dv"]["mask_token"].get<int>();
  const int vocab = dv->model().desc().vocab;
  std::string err;
  // A request's tokens and its rounds' ring entries (the prefill's first), the test's oracle set when given.
  auto decode = [&](Engine* e, const GenerateRequest& rq, const std::vector<int32_t>* oracle = nullptr,
                    std::vector<flyx_round_result>* trace = nullptr, Session* on = nullptr) {
    Session own(e);
    Session& s = on ? *on : own;
    if (!on && !init_checked(s)) return std::vector<int32_t>{};
    if (oracle) e->api()->set_oracle(s.seq(), oracle->data(), static_cast<int32_t>(oracle->size()));
    std::vector<int32_t> toks;
    GenerateResult res;
    s.trace = trace;
    if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
          toks.insert(toks.end(), t, t + nt);
          return true;
        }, &res, &err) != 0)
      check(false, "DV: generate: " + err);
    s.trace = nullptr;
    if (oracle) e->api()->set_oracle(s.seq(), nullptr, 0);
    return toks;
  };
  int compared = 0, in_sync = 0, trees = 0, trees_same = 0, drafts = 0, drafts_same = 0, fixed = 0, fixed_same = 0;
  int emptied = 0, emptied_same = 0;  // trees after a round that slid the canvas past every slot
  FreshSame fresh16, fresh16_second, fresh8, fresh8_second, fresh_host, fresh8_host, fresh_generic;
  int fp8_tokens = 0;
  bool host_tokens = true, host8_tokens = true;
  int k = 0;
  for (const auto& c : g["cases"]) {
    const auto prompt = c["prompt"].get<std::vector<int32_t>>();
    const auto want = c["tokens"].get<std::vector<int32_t>>();
    const int N = static_cast<int>(want.size()), P = static_cast<int>(prompt.size());
    const std::string at = "DV, case " + std::to_string(k) + ": ";
    // plain decoding past the case's tokens: what every run below must give, whatever it drafts
    GenerateRequest rq;
    rq.prompt = prompt;
    rq.max_new_tokens = N + 40;
    rq.ignore_eos = true;
    const auto longer = decode(plain.get(), rq);
    rq.max_new_tokens = N;
    if (k < 2) dv_mixed(model, backend, prompt, want, longer, mask, at, k == 0);
    std::vector<flyx_round_result> trace;
    Session sd(dv.get());
    init_checked(sd);
    const auto greedy = decode(dv.get(), rq, nullptr, &trace, &sd);
    bool drafted = trace.size() > 2, first_star = trace.size() > 1;
    int accepted = 0;
    for (size_t i = 1; i < trace.size(); ++i) {
      drafted = drafted && trace[i].n_drafts == width;
      accepted += trace[i].n_accepted;
    }
    for (int j = 0; first_star && j < width; ++j) first_star = trace[1].drafts[j] == mask && trace[1].parents[j] == -1;
    check(greedy == want && drafted && first_star,
          at + "greedy: the reference's tokens; " + std::to_string(trace.size() - 1) + " rounds of " +
              std::to_string(width) + " drafts each (" + std::to_string(accepted) +
              " accepted), the first DV's star of MASKs (" + first_difference(greedy, want) + ")");
    // Long accepted runs: the test's oracle, a chain of plain decoding's tokens every round (every draft accepted:
    // 15, or 31).
    {
      std::vector<int32_t> oracle(prompt.begin(), prompt.end());
      oracle.insert(oracle.end(), longer.begin(), longer.end());
      std::vector<flyx_round_result> to;
      const auto o = decode(dv.get(), rq, &oracle, &to);
      bool all = to.size() > 1;
      for (size_t i = 1; i < to.size(); ++i) all = all && to[i].n_accepted == width;
      check(o == want && all, at + "a chain of plain decoding's tokens every round: the tokens, " +
                                  std::to_string(width) + " accepted every round (" + std::to_string(to.size() - 1) +
                                  " rounds)");
    }
    // The reference's drafts, round for round, while the rounds start where the reference's do.
    if (reference && k < static_cast<int>(reference->size())) {
      const json& ref = (*reference)[size_t(k)];
      Session sm(dv.get());
      init_checked(sm);
      std::vector<int32_t> first;
      GenerateRequest one = rq;
      one.max_new_tokens = 1;
      first = decode(dv.get(), one, nullptr, nullptr, &sm);
      const auto run = dv_chain_rounds(dv.get(), sm, plan, prompt.size(), want, golden, file, 4 * N);
      const int every = golden["every"].get<int>(), chain = golden["first"].get<int>();
      std::vector<int32_t> got = first;  // the tokens from position P on
      for (const flyx_round_result& x : run) got.insert(got.end(), x.tokens, x.tokens + x.n_tokens);
      for (size_t r = 0; r < run.size() && r < ref.size(); ++r) {
        const flyx_round_result& x = run[r];
        const json& y = ref[r];
        ++compared;
        if (x.position - x.n_tokens != y["pos"].get<int>()) break;
        ++in_sync;
        const auto d = y["drafts"].get<std::vector<int32_t>>(), p = y["parents"].get<std::vector<int32_t>>();
        bool same = x.n_drafts == static_cast<int>(d.size()) && x.n_accepted == y["accepted"].get<int>();
        int ds = 0;
        for (size_t j = 0; j < d.size() && j < size_t(x.n_drafts); ++j) {
          ds += x.drafts[j] == d[j];
          same = same && x.drafts[j] == d[j] && x.parents[j] == p[j];
        }
        if (r > 0 && static_cast<int>(r) % every != chain) {
          ++trees;
          trees_same += same;
          drafts += static_cast<int>(d.size());
          drafts_same += ds;
          if (ref[r - 1]["accepted"].get<int>() + 1 >= slots) {
            ++emptied;
            emptied_same += same;
          }
          if (!same && trees - trees_same <= 3)
            std::printf("       %s round %zu at %d: drafts %s, the reference's %s\n", file.c_str(), r,
                        y["pos"].get<int>(), str(std::vector<int32_t>(x.drafts, x.drafts + x.n_drafts), 15).c_str(),
                        str(d, 15).c_str());
        } else {
          ++fixed;
          fixed_same += same;
        }
      }
      check(got.size() >= want.size() && std::equal(want.begin(), want.end(), got.begin()),
            at + "rounds one at a time, chains planted: the reference's tokens (" + std::to_string(run.size()) +
                " rounds)");
      // Rewound into the last of them: DV starts over (its first tree), and its rounds go on as plain decoding.
      if (!run.empty()) {
        const auto* api = dv->api();
        const int32_t pos = run.back().position - 2;
        bool ok = api->rewind(sm.seq(), pos, got[size_t(pos - P)]) == 1;
        flyx_gen_params pg{};
        pg.top_p = 1.f;
        std::vector<int32_t> again;
        bool star = true;
        for (int r = 0; ok && r < 6; ++r) {
          flyx_round_result x{};
          ok = api->launch(dv->round_plan(plan), sm.seq(), &pg) == 0 && api->read(sm.seq(), &x) == 0;
          for (int j = 0; r == 0 && j < x.n_drafts; ++j) star = star && x.drafts[j] == mask;
          again.insert(again.end(), x.tokens, x.tokens + x.n_tokens);
        }
        const size_t from = size_t(pos - P) + 1;
        ok = ok && from + again.size() <= longer.size() &&
             std::equal(again.begin(), again.end(), longer.begin() + std::ptrdiff_t(from));
        check(ok && star, at + "rewound into the last DV round: DV's first tree, then plain decoding's tokens (" +
                              std::to_string(again.size()) + ")");
        // Where the canvas's last slide left it, another token next: rewound to the position it stands at (rewind's
        // first branch), and a checkpoint taken there restored.  The canvas follows that position but not the text
        // the round would draft after, and DV starts over (set_params forgets it); else its tree grows from it.
        auto star_after = [&](const std::function<bool(int32_t position, int32_t other)>& reset) {
          flyx_round_result x{}, y{};
          if (!ok || api->launch(dv->round_plan(plan), sm.seq(), &pg) != 0 || api->read(sm.seq(), &x) != 0 ||
              !reset(x.position, (x.tokens[x.n_tokens - 1] + 1) % vocab) ||
              api->launch(dv->round_plan(plan), sm.seq(), &pg) != 0 || api->read(sm.seq(), &y) != 0)
            return false;
          bool first = y.n_drafts == width;
          for (int j = 0; j < y.n_drafts; ++j) first = first && y.drafts[j] == mask && y.parents[j] == -1;
          return first;
        };
        check(star_after([&](int32_t at_pos, int32_t other) { return api->rewind(sm.seq(), at_pos, other) == 1; }),
              at + "rewound to where the last slide left the canvas, another token next: DV's first tree");
        const int slot = FLYX_CHECKPOINTS - 1;
        check(star_after([&](int32_t at_pos, int32_t other) {
                return api->checkpoint(sm.seq(), slot) == at_pos && api->restore(sm.seq(), slot, other) == at_pos;
              }),
              at + "a checkpoint where the last slide left the canvas restored, another token next: DV's first tree");
      }
    }
    // Sampled for one seed, with and without top-p/top-k, and with penalties (greedy, sampled; with the chains of
    // plain decoding's tokens too, whose rows count the drafts before them): plain decoding's tokens.
    for (auto sp : {SamplingParams{0.8f, 1.f, 0, 11}, SamplingParams{1.0f, 0.9f, 40, 12345},
                    SamplingParams{0.f, 1.f, 0, 0, 1.5f, 0.25f}, SamplingParams{1.0f, 0.9f, 40, 4711, 1.5f, 0.25f}}) {
      GenerateRequest sr = rq;
      sr.sampling = sp;
      const auto a = decode(plain.get(), sr), b = decode(dv.get(), sr);
      std::string what = sp.temperature > 0 ? "sampled (T=" + std::to_string(sp.temperature).substr(0, 3) + ")"
                                            : std::string("greedy");
      if (sp.presence_penalty != 0) {
        what += ", penalties";
        std::vector<int32_t> oracle(prompt.begin(), prompt.end());  // plain decoding's tokens under them
        oracle.insert(oracle.end(), a.begin(), a.end());
        check(decode(dv.get(), sr, &oracle) == a, at + what + ", chains of plain decoding's tokens: the same");
      }
      check(static_cast<int>(a.size()) == N && b == a, at + what + ": DV == plain (" + first_difference(b, a) + ")");
    }
    {
      Session a8(plain8.get()), b8(dv8.get());
      init_checked(a8);
      init_checked(b8);
      const auto t8 = gen(a8, prompt, N).tokens;
      check(static_cast<int>(t8.size()) == N && gen(b8, prompt, N).tokens == t8, at + "FP8 K/V: DV == plain");
      Session c8(dv8_host.get());
      init_checked(c8);
      host8_tokens = host8_tokens && gen(c8, prompt, N).tokens == t8;
      // DV's first tree after a fresh start at each point, against the e4m3 reference's along its own tokens (FP8
      // caches may change plain decoding's; the GPU's tokens are compared with them, starts out of step not counted).
      if (fp8_ref && k < static_cast<int>(fp8_ref->size())) {
        const auto t8_ref = (*fp8_ref)[size_t(k)]["tokens"].get<std::vector<int32_t>>();
        fp8_tokens += t8 == t8_ref;
        const json& ref8 = (*fp8_ref)[size_t(k)]["fresh"];
        const DvFresh f8 = dv_fresh(dv8.get(), prompt, t8_ref, width);
        const DvFresh fh8 = dv_fresh(dv8_host.get(), prompt, t8_ref, width);
        fresh8.add(f8.first, FreshSame::of(ref8, "first"));
        fresh8_host.add(fh8.first, f8.first);
        fresh8_host.add(fh8.second, f8.second);
        fresh8_second.add(f8.second, FreshSame::of(ref8, "second"));
      }
    }
    // DV's first tree after a fresh start at each point: the reference's; with the K/V in host memory (the canvas
    // reads the text through the stagings), the tokens and every tree this GPU drafts with them on the device.
    if (fresh_ref && k < static_cast<int>(fresh_ref->size())) {
      const DvFresh f = dv_fresh(dv.get(), prompt, want, width), fh = dv_fresh(dv_host.get(), prompt, want, width);
      fresh16.add(f.first, FreshSame::of((*fresh_ref)[size_t(k)], "first"));
      fresh16_second.add(f.second, FreshSame::of((*fresh_ref)[size_t(k)], "second"));
      fresh_host.add(fh.first, f.first);
      fresh_host.add(fh.second, f.second);
      if (dv_generic) {
        const DvFresh fg = dv_fresh(dv_generic.get(), prompt, want, width);
        fresh_generic.add(fg.first, f.first);
        fresh_generic.add(fg.second, f.second);
      }
      Session sh(dv_host.get());
      init_checked(sh);
      host_tokens = host_tokens && gen(sh, prompt, N).tokens == want;
    }
    // The session again: continued (a rewind inside its last rounds), from an earlier point (the end-of-prompt
    // checkpoint), with the prompt edited; each time its tokens and DV's drafts are a fresh session's (and, continuing
    // the case, plain decoding's tokens).
    auto as_fresh = [&](const std::vector<int32_t>& p, int n, bool continues, const std::string& what) {
      std::vector<flyx_round_result> ta, tb;
      const Out oa = gen(sd, p, n, &ta);
      Session fresh(dv.get());
      init_checked(fresh);
      const Out ob = gen(fresh, p, n, &tb);
      int same = 0, total = 0;
      for (size_t i = 1; i < ta.size() && i < tb.size(); ++i)
        for (int j = 0; j < ta[i].n_drafts && j < tb[i].n_drafts; ++j, ++total)
          same += ta[i].drafts[j] == tb[i].drafts[j];
      const bool tokens = !continues || (oa.tokens.size() == size_t(n) &&
                                         std::equal(oa.tokens.begin(), oa.tokens.end(),
                                                    longer.begin() + std::ptrdiff_t(p.size() - prompt.size())));
      check(tokens && oa.tokens == ob.tokens && ta.size() == tb.size() && same == total && total > 0,
            at + what + ": " + (continues ? "plain decoding's tokens, " : "") +
                "a fresh session's tokens and drafts (" + std::to_string(same) + " / " + std::to_string(total) + ", " +
                std::to_string(oa.usage.reused_tokens) + " of " + std::to_string(p.size()) + " reused)");
    };
    auto p2 = prompt;
    p2.insert(p2.end(), want.begin(), want.begin() + N - 2);
    as_fresh(p2, 8, true, "continued from inside its last rounds");
    auto p3 = prompt;
    p3.insert(p3.end(), want.begin(), want.begin() + 20);
    as_fresh(p3, N - 20, true, "continued from an earlier point");
    auto p4 = prompt;
    p4.back() = (p4.back() + 1) % vocab;
    as_fresh(p4, 16, false, "edited prompt");
    if (k == 0) {
      // A snapshot through host memory: loaded, DV starts over as in a fresh session.
      auto other = make((fs::path(model).parent_path() / "tiny_qwen3.flyx").string(), backend, false);
      snapshot_through_host(dv.get(), other.get(), prompt, "DV, snapshot", true);
      // A stop string ends the output before it.
      const auto& tk = dv->tokenizer();
      const std::string full = tk.decode(want), stop = tk.decode({want.begin() + 10, want.begin() + 12});
      GenerateRequest sr = rq;
      sr.stop = {stop};
      Session s(dv.get());
      init_checked(s);
      std::string shown;
      GenerateResult res;
      generate(s, sr, [&](std::string_view t, const int32_t*, int) {
        shown += t;
        return true;
      }, &res, &err);
      check(!stop.empty() && shown == full.substr(0, full.find(stop)) && res.stop_string == stop &&
                res.usage.finish_reason == FLYX_FINISH_STOP,
            "DV: a stop string ends the output before it");
    }
    ++k;
  }
  std::printf("       DV's trees after fresh starts against the reference, the first and the second: binary16 K/V %s; "
              "%s; FP8 K/V %s; %s (%d of %zu cases' tokens the e4m3 reference's); K/V in host memory against on the "
              "GPU %s\n", fresh16.str().c_str(), fresh16_second.str().c_str(), fresh8.str().c_str(),
              fresh8_second.str().c_str(), fp8_tokens, fp8_ref ? fp8_ref->size() : size_t(0), fresh_host.str().c_str());
  check(host_tokens && fresh_host.samples > 0 && fresh_host.trees == fresh_host.samples,
        "DV, K/V all in host memory: the reference's tokens, and every fresh tree the one drafted with the K/V on the "
        "GPU (" + fresh_host.str() + ")");
  if (dv_generic)
    check(fresh_generic.samples > 0 && fresh_generic.trees == fresh_generic.samples,
          "DV past 16 slots, the generic regions (fusion=0: the canvas's GEMMs one by one, not grouped): every fresh "
          "tree the fused regions' (" + fresh_generic.str() + ")");
  check(host8_tokens && fresh8_host.samples > 0 && fresh8_host.trees == fresh8_host.samples,
        "DV, FP8 K/V all in host memory (the scales past the device part staged with it): plain decoding's tokens, "
        "and every fresh tree the one drafted with the K/V on the GPU (" + fresh8_host.str() + ")");
  // The reference rounds what this backend rounds (golden_llama_dv.json), so a canvas of 8 slots leaves float32's order
  // of sums between them, and a tree that turns on it now and then: the first trees after fresh starts are the
  // reference's 184 of 184 on both 8-slot files, the second ones 180 and 183 (FP8 K/V 183 and 180, 148 and 156, whose
  // codes 1/16 apart turn on it more), where coverage left out moves 9 of the beam file's first trees, and a wrong
  // canvas pass, key map, scale or selector all.  Past 16 slots trees deep in a beam of 8 over 15 depths of a random
  // canvas turn on sibling values float32 cannot tell apart: the reference in float32 keeps 108 of its 184 first
  // trees in float64 and 85% of their drafts, this backend 106 and 85% (second 74%; FP8 K/V 84% and 67%), a causal
  // canvas run as a full one 5% of the drafts, so there the drafts are held, at what float32 leaves less a margin.
  // Trees of 31 drafts over 31 depths turn on such values more: the reference in float32 keeps 55 (beam of 8) and 76
  // (pool of 7 ranks) of its 184 first trees in float64, 73% and 78% of their drafts, and 56% and 58% of the second
  // trees' drafts; this backend 72% and 77%, 54% and 57% (FP8 K/V 72% and 75%, 49% and 54%).  The beam's beside a
  // V-LoRA (tiny_llama_dv31_lora.flyx, its 6 cases' 276 trees): the reference in float32 69.4% and 52.0%, this
  // backend 67.2% and 56.2% (FP8 K/V 70.2% and 47.8%).  Run as a full canvas, a causal one keeps 6% of the first
  // trees' drafts (beam and pool), a fill of 0.4 for 0.9 2% of the second trees',
  // coverage left out of the beam 62% of the first trees'.  Those trees rarely reach the beam's nodes past the fourth
  // or the pool's ranks past the third, so the trees of 31 are held whole on 16 slots, from a flatter selector
  // (tiny_llama_dv31_wide*.flyx): the first and second trees the reference's 184 and 184 (beam), 184 and 182 (pool),
  // where a beam of 7 leaves 43 and 47 of the first and second, a beam of 4 4 and 1, coverage left out 100 and 120, a
  // pool of 6 ranks 120 and 135, of 3 ranks 24 and 34, of 1 1 and 0.  With FP8 K/V their flat tables turn on e4m3's
  // codes more: the e4m3 reference in float32 keeps 109 (beam) and 128 (pool) of the float64 one's 184 second trees,
  // 87.6% and 89.2% of their drafts, this backend 108 and 142, 89.9% and 92.3%; held there at half the trees.
  auto held = [&](const FreshSame& f, int tree_pct, int draft_pct, int draft_pct_wide, int draft_pct_deep,
                  int step_pct, const std::string& what) {
    const bool narrow = slots <= 16;
    const int dp = narrow ? draft_pct : width >= FLYX_MAX_ROWS ? draft_pct_deep : draft_pct_wide;
    check(f.samples > 0 && f.in_step * 100 >= f.samples * step_pct &&
              (!narrow || f.trees * 100 >= f.in_step * tree_pct) && f.drafts_same * 100 >= f.drafts * dp,
          what + ": " + f.str() + " (" + (narrow ? std::to_string(tree_pct) + "% of the trees, " : std::string()) +
              std::to_string(dp) + "% of the drafts, " + std::to_string(step_pct) + "% in step)");
  };
  check(fresh_ref && fp8_ref, "DV: the reference's fresh starts for " + file);
  held(fresh16, 98, 99, 70, 65, 100, "DV's first trees after fresh starts are the reference's");
  held(fresh16_second, 95, 98, 60, 45, 95, "DV's second trees after fresh starts are the reference's");
  // Beside a V-LoRA (tiny_llama_dv_lora.flyx) the FP8 trees turn on float32's rounding more: the e4m3 reference in
  // float32 keeps 254 and 217 of the float64 one's 276 first and second trees (97.2% and 92.3% of their drafts; on
  // tiny_llama_dv.flyx, 182 and 157 of 184), this backend 249 and 192 (96.2% and 86.9%); held there at what float32
  // leaves less a margin (binary16 K/V as every file: 275 and 267 of 276).
  const json& header = dv->model_file().header();
  const bool vlora = header.contains("lora") && !header["lora"].is_null();
  held(fresh8, vlora ? 85 : 95, vlora ? 94 : 98, 70, 65, 90,
       "FP8 K/V: DV's first trees after fresh starts are the e4m3 reference's");
  held(fresh8_second, width >= FLYX_MAX_ROWS ? 50 : vlora ? 60 : 70, 85, 55, 40, 90,
       "FP8 K/V: DV's second trees after fresh starts are the e4m3 reference's");
  if (reference) {
    // Round for round, a tree that differs once changes the canvas the next rounds slide from, and where it accepts
    // another number of drafts the case's rounds go on out of step: on the beam file (fill 0.9) this backend's trees
    // are the reference's in 47 of 48 rounds (713 of 720 drafts), on the pool file, whose causal canvas keeps its own
    // guesses as inputs (fill 0.4) and so carries such a change on, in 32 of 43 (544 of 645; 72 of 73 rounds in
    // step); a canvas pass, key map, slide, fill or selector that is wrong leaves 10 of 48 or fewer (mutations on the
    // beam file: the last slot unseen, causal flipped, the target's norm for ln1c, no o adapter, fill halved, alpha 0,
    // a slide one short; 451 of 720 drafts at most); the trees of 31 on 16 slots 57 and 59 of 72 (2029 and 2031 of 2232
    // drafts), a beam of 7 or a pool of 6 ranks 8 and 35.  Past 16 slots the trees differ from the first round that
    // turns on a near tie (1 of 43 the same; with trees of 31, 9 and 12 of 40), and the rounds are held to their stars
    // and chains in step, and to the trees after a slide past every slot (below).
    std::printf("       DV's trees against the float64 reference: %d of %d rounds the same (%d of %d drafts); the "
                "star and the chains %d of %d; after a slide past every slot %d of %d; %d of %d rounds in step\n",
                trees_same, trees, drafts_same, drafts, fixed_same, fixed, emptied_same, emptied, in_sync, compared);
    check(in_sync * 10 >= compared * 9 && fixed_same == fixed && trees >= 36 &&
              (slots > 16 || (trees_same * 10 >= trees * 7 && drafts_same * 100 >= drafts * 80)),
          "DV's drafts, round for round, are the float64 reference's: " + std::to_string(trees_same) + " of " +
              std::to_string(trees) + " trees, " + std::to_string(drafts_same) + " of " + std::to_string(drafts) +
              " drafts" + (slots > 16 ? " (not held past 16 slots)" : " (70% and 80% at least)") + "; " +
              std::to_string(fixed) + " stars and chains, all; " + std::to_string(in_sync) + " of " +
              std::to_string(compared) + " rounds in step (90%)");
    // A chain that slides the canvas past every slot (where the file's chains reach that far: a canvas no longer
    // than the tree's rows) leaves the next tree no candidates: every node is worth a slot past the canvas's
    // constant and drafts MASK, its shape the order of ties, so it is the reference's whatever the slots' sums, held
    // at every number of slots.  A slide that keeps a slot's candidates or guesses, or a tree that takes them, fails:
    // a slide past every slot that leaves the canvas as it was keeps none of these trees on any file whose chains
    // slide it so, and on the 32-slot files of trees of 31 fails nothing else (their own trees not held).
    check((emptied > 0 || slots > width + 1) && emptied_same == emptied,
          "DV's trees after a chain that slid the canvas past every slot are the float64 reference's: " +
              std::to_string(emptied_same) + " of " + std::to_string(emptied));
  } else {
    check(false, "DV: no reference rounds for " + file + " in golden_llama_dv.json");
  }
}

int plain_only(const json& g, const std::string& model, const std::string& backend) {
  auto plain = make(model, backend, false);
  auto plain8 = make(model, backend, false, "", true);
  prefill_only_checks(plain.get(), g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  refused_session(plain.get(), g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  // Prompt passes of up to 64 rows, as a real prompt takes them (the engines above pass 16 rows at a time, which run
  // the kernels of a round's 1..16 rows): GEMMs of more rows, the prefill MLP's own epilogue (without adapters).
  auto passes64 = make(model, backend, false, "", false, "", false, 512, kReferenceWidth, false, -1, "exact", -1, 64);
  std::printf("backend %s, model %s, no drafter\n", plain->backend_name().c_str(), model.c_str());
  // A file with a drafter of its own (the tiny Llama's DV, tiny_llama_dv.flyx, which the default does not choose:
  // models/model.cpp): with every drafter it has (--drafter all), as the engine takes them (DV's rounds where the
  // backend runs them, which the CUDA backend must, else decoding as without it: no drafter in use, and why), and
  // with --drafter none; both must give the reference's tokens.  Named (--drafter dv), a drafter the backend does not
  // run fails the load.
  std::vector<std::unique_ptr<Engine>> drafting;
  bool dv_drafts = false;  // the file's DV, whose rounds this backend runs (dv_rounds below)
  bool dv_file = false;    // the file has DV, run or not (dv_chat_checkpoint below)
  if (auto d = make(model, backend, true, "all", false, "", true, 512, -1); d->model().drafters() > 0) {
    std::string rounds;
    for (size_t i = 0; i < d->widths().size(); ++i) rounds += (i ? ", " : "") + d->round_name(i);
    const std::string mine = d->model().drafter();
    std::printf("drafter %s of the file's %s: rounds %s%s\n", d->drafter(), mine.c_str(), rounds.c_str(),
                d->refused().empty() ? "" : (" (refused: " + d->refused() + ")").c_str());
    const bool runs = std::string(d->drafter()) == mine;
    bool drafts = false;
    for (int r : d->round_drafters()) drafts |= r >= 0;
    const bool none = std::string(d->drafter()) == "none" && !drafts && !d->refused().empty();
    check(runs ? drafts : none,
          "the drafter in use is the one whose rounds run (" + std::string(d->drafter()) + ")");
    // DV's rounds are the CUDA backend's (dv_rounds below holds them to the reference): a refusal there is a failure,
    // not a file decoded as without DV.
    if (backend == "cuda" && mine == "dv")
      check(runs && d->refused().empty(), "the CUDA backend runs DV's rounds (" + std::string(d->drafter()) +
                                              (d->refused().empty() ? "" : ": " + d->refused()) + ")");
    dv_drafts = runs && mine == "dv";
    dv_file = mine == "dv";
    EngineOptions o;
    o.backend = backend;
    o.max_ctx = 512;
    o.drafter = mine;
    std::string why;
    Engine named;
    const bool loaded = named.load(model, o, &why);
    check(runs ? loaded : !loaded && why.find("does not run its rounds") != std::string::npos,
          "--drafter " + mine + (runs ? ": loads" : ": refused, the load fails (" + why + ")"));
    drafting.push_back(std::move(d));
    drafting.push_back(make(model, backend, true, "none", false, "", true, 512, -1));
    // On the GPU, DV's weights once each and as its kernels read them: the canvas's norms float, its adapters, MASK
    // embedding and residual context bf16 as stored, the selector's matrices float32 (alpha a number on the host),
    // and, made when its rounds are lowered, its token tables A E and B E, [vocab][rank] float32 each.
    flyx_memory_info with{}, without{};
    const Engine &a = *drafting[0], &b = *drafting[1];
    if (dv_file && a.api()->memory && a.api()->memory(a.bmodel(), &with) == 0 &&
        b.api()->memory(b.bmodel(), &without) == 0 && with.weights > 0) {
      uint64_t want = 0;
      const flyx_model_desc& desc = a.model().desc();
      for (int i = 0; i < desc.n_weights; ++i) {
        const flyx_tensor_view* t = desc.weights[i].t;
        const std::string name = t ? t->name : "";
        if (name.rfind("dv.", 0) != 0 || name == "dv.sel.alpha") continue;
        uint64_t n = 1;
        for (int k = 0; k < t->ndim; ++k) n *= uint64_t(t->shape[k]);
        const bool norm = t->ndim == 1 && name != "dv.mask_emb";
        want += n * (norm || t->dtype == FLYX_DT_F32 ? 4 : 2);
        if (name == "dv.sel.A" && runs) want += 2 * uint64_t(desc.vocab) * uint64_t(t->shape[0]) * 4;
      }
      check(with.weights - without.weights == want, "DV's weights on the device: " +
                                                        std::to_string(with.weights - without.weights) + " bytes, " +
                                                        std::to_string(want) + " as its kernels read them");
    }
  }
  int k = 0;
  std::string err;
  {  // The K/V cache the caller does not choose: binary16 on a Llama (FP8 moves its next tokens visibly) and where the
     // device has room for it (Engine::roomy_for_kv16), else FP8
    EngineOptions o;
    o.backend = backend;
    o.max_ctx = 512;
    o.speculative = false;
    Engine e;
    check(e.load(model, o, &err), "default K/V cache: load: " + err);
    const bool llama = e.model_file().arch() == "llama", roomy = e.roomy();
    check(e.options().kv == (llama || roomy ? FLYX_KV_16 : FLYX_KV_FP8),
          std::string("default K/V cache: ") +
              (llama ? "binary16 on a Llama" : roomy ? "binary16 with room" : "FP8 on a Qwen3"));
  }
  for (const auto& c : g["cases"]) {
    auto prompt = c["prompt"].get<std::vector<int32_t>>();
    Session sp(plain.get());
    if (!init_checked(sp)) continue;
    // A case without the reference's tokens (golden_llama_nvfp4.json: the reference has no Llama of NVFP4 weights)
    // holds the file to its own plain decoding, as every check below does (rule 4).
    const bool own = !c.contains("tokens");
    auto want = own ? gen(sp, prompt, 48).tokens : c["tokens"].get<std::vector<int32_t>>();
    const int N = static_cast<int>(want.size());
    if (own)
      std::printf("case %d: prompt %zu tokens, its own plain decoding\n", k++, prompt.size());
    else
      std::printf("case %d: prompt %zu tokens, min margin %.3g\n", k++, prompt.size(), c["min_margin"].get<double>());
    auto o = gen(sp, prompt, N);
    check(o.tokens == want, "plain decode: " + str(o.tokens));
    for (size_t i = 0; i < drafting.size(); ++i) {
      Session sd(drafting[i].get());
      init_checked(sd);
      check(gen(sd, prompt, N).tokens == want,
            std::string(i == 0 ? "--drafter all (in use: " : "--drafter none (in use: ") +
                drafting[i]->drafter() + "): the same");
    }
    {
      Session s8(plain8.get());
      init_checked(s8);
      check(static_cast<int>(gen(s8, prompt, N).tokens.size()) == N, "fp8 kv cache: decodes");
    }
    auto p2 = prompt;
    p2.insert(p2.end(), want.begin(), want.begin() + N - 2);
    o = gen(sp, p2, 2);
    check(o.tokens == std::vector<int32_t>(want.end() - 2, want.end()), "continue the session");
    check(o.usage.reused_tokens == static_cast<int>(p2.size()) - 1,
          "  reused " + std::to_string(o.usage.reused_tokens) + " of " + std::to_string(p2.size()));
    auto p3 = prompt;
    p3.insert(p3.end(), want.begin(), want.begin() + 20);
    o = gen(sp, p3, N - 20);
    check(o.tokens == std::vector<int32_t>(want.begin() + 20, want.end()), "continue from an earlier point");
    std::printf("       reused %d of %zu tokens\n", o.usage.reused_tokens, p3.size());
    auto p4 = prompt;
    p4.back() = (p4.back() + 1) % 300;
    o = gen(sp, p4, 16);
    Session fresh(plain.get());
    init_checked(fresh);
    check(o.tokens == gen(fresh, p4, 16).tokens, "edited prompt: reused session == fresh session");
    {  // the prompt and the reference's first 40 tokens in one pass of more than 16 rows, then its last 8 tokens
      const int at = N - 8;
      auto p5 = prompt;
      p5.insert(p5.end(), want.begin(), want.begin() + at);
      Session s64(passes64.get());
      init_checked(s64);
      const auto got = gen(s64, p5, N - at).tokens;
      check(got == std::vector<int32_t>(want.begin() + at, want.end()),
            "a prompt of " + std::to_string(p5.size()) + " tokens in one pass: the reference's tokens after it (" +
                first_difference(got, std::vector<int32_t>(want.begin() + at, want.end())) + ")");
    }
    if (k == 1) {  // rounds of several sequences (launch_batch: the server's, two requests or more), plain decoding
      size_t round = plain->widths().size();
      for (size_t i = 0; i < plain->widths().size(); ++i)
        if (plain->widths()[i] == 0) round = i;
      auto other = prompt;
      other[0] = (other[0] + 7) % 300;
      auto twice = prompt;
      twice.insert(twice.end(), prompt.begin(), prompt.end());
      check(round < plain->widths().size(), "rounds of several sequences: a plain round");
      if (round < plain->widths().size())
        batch_rounds(plain.get(), {prompt, other, twice}, {round, round, round},
                     "rounds of three sequences, plain decoding");
    }
    if (k == 1 && backend == "cuda") {  // a second architecture's tree rounds (attention only)
      SamplingParams greedy, sampled, penalized;
      sampled.temperature = 0.8f;
      sampled.seed = 7;
      penalized.presence_penalty = 0.5f;
      penalized.frequency_penalty = 0.3f;
      {  // without wide rows (backend option wide=0) a round verifies 16 rows at most
        auto narrow = make(model, backend, false, "", false, "wide=0");
        const GivenTrees t(narrow.get(), {2 * FLYX_MAX_ROWS - 1}, false);
        check(!t.lowered(), "wide=0: no tree round of " + std::to_string(2 * FLYX_MAX_ROWS) + " rows" + t.refusal());
      }
      // Trees of 31 drafts beside the engine's (32 rows: the GEMMs, the head and the choice 16 rows at a time), lowered
      // here on the engine's wide rows (wide=1): an engine lowers no tree of more than 16 rows of its own (plans.h,
      // tree_widths).
      auto we = make(model, backend, false, "", false, "wide=1"), we8 = make(model, backend, false, "", true, "wide=1");
      GivenTrees trees(we.get(), {2 * FLYX_MAX_ROWS - 1}), trees8(we8.get(), {2 * FLYX_MAX_ROWS - 1});
      check(trees.lowered() && trees8.lowered(),
            "tree rounds of 31 drafts: lowered" + trees.refusal() + trees8.refusal());
      tree_rounds(we.get(), prompt, greedy, "tree rounds", {.trees = &trees.plans()});
      tree_rounds(we.get(), prompt, sampled, "tree rounds, sampling", {.trees = &trees.plans()});
      tree_rounds(we8.get(), prompt, greedy, "tree rounds, fp8 kv cache", {.trees = &trees8.plans()});
      // Past 16 rows too: a row's penalties count its ancestors' drafts, the relaxed rules' statistics every row's
      // and their drafts' logits gathered 16 rows at a time.
      tree_rounds(we.get(), prompt, penalized, "tree rounds, penalties", {.trees = &trees.plans()});
      tree_rounds(we.get(), prompt, greedy, "tree rounds, min-p 1 (only the top token qualifies)",
                  {.trees = &trees.plans(), .min_p = 1.f});
      // Trees of 47 and 63 drafts, the most a round verifies (FLYX_MAX_WIDE_ROWS rows).
      const GivenTrees wide(we.get(), {FLYX_MAX_WIDE_ROWS - 17, FLYX_MAX_WIDE_ROWS - 1}, false);
      check(wide.lowered(), "tree rounds of 47 and 63 drafts: lowered" + wide.refusal());
      if (wide.lowered()) {
        tree_rounds(we.get(), prompt, greedy, "tree rounds of 47 and 63 drafts", {.trees = &wide.plans()});
        tree_rounds(we.get(), prompt, sampled, "tree rounds of 47 and 63 drafts, sampling",
                    {.trees = &wide.plans()});
        // A round launched while the one before it is in flight, its drafts behind that round's hidden path.
        tree_rounds(we.get(), prompt, sampled, "tree rounds of 47 and 63 drafts two in flight, sampling",
                    {.trees = &wide.plans(), .two_in_flight = true});
      }
      // K/V all in host memory: a tree round's rows staged after the text's for every attention;
      // every width from 3 to 63 drafts, read one at a time and two in flight.
      auto host = make(model, backend, false, "", false, "wide=1", false, 512, kReferenceWidth, false, 0);
      const GivenTrees host_trees(host.get(), {2 * FLYX_MAX_ROWS - 1, FLYX_MAX_WIDE_ROWS - 1});
      check(host_trees.lowered(),
            "tree rounds of 31 and 63 drafts, K/V in host memory: lowered" + host_trees.refusal());
      tree_rounds(host.get(), prompt, greedy, "tree rounds, K/V in host memory", {.trees = &host_trees.plans()});
      tree_rounds(host.get(), prompt, greedy, "tree rounds two in flight, K/V in host memory",
                  {.trees = &host_trees.plans(), .two_in_flight = true});
      tree_round_before_strata(model, backend, prompt);
    }
  }
  if (dv_drafts) dv_rounds(g, model, backend);
  if (dv_file)
    dv_chat_checkpoint(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>(), dv_drafts);
  context_end(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  adaptive_stopping(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  if (backend == "cuda") wide_rounds(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}

// A session's round graphs and its shares of batch graphs (the backend's graph_stats), and its
// rounds' total.
flyx_graph_stats graph_stats(Engine* e, Session& s) {
  flyx_graph_stats st{};
  if (!e->api()->graph_stats || e->api()->graph_stats(s.seq(), &st) != 0) st = {};
  return st;
}

// `rounds` rounds of several sequences (launch_batch) of plan p, every sequence's read after each.
bool batched(Engine* e, flyx_plan* p, const std::vector<Session*>& sessions, int rounds) {
  const auto* api = e->api();
  flyx_gen_params g{};
  g.top_p = 1.f;
  std::vector<flyx_plan*> plans(sessions.size(), p);
  std::vector<flyx_seq*> seqs;
  for (Session* s : sessions) seqs.push_back(s->seq());
  const std::vector<const flyx_gen_params*> gens(sessions.size(), &g);
  flyx_round_result r{};
  for (int i = 0; i < rounds; ++i) {
    if (api->launch_batch(plans.data(), seqs.data(), gens.data(), int32_t(seqs.size())) != 0) return false;
    for (flyx_seq* q : seqs)
      if (api->read(q, &r) != 0) return false;
  }
  return true;
}

// The sessions' round graphs inside what they lease.  Sessions kept alive, each having run every kind of round its
// engine has (every round plan in turn, drafted trees among them where the K/V are read exactly, the rounds the engine
// chooses on a text that repeats, lookup rounds among them; sampled, with a penalty, with relaxed acceptance: graphs
// of other heads, for which a session gives up the ones it ran least lately; rounds of two sessions batched, past the
// hits that capture their graph, a DV file's warm-ups among them; a DV file's rounds hold its canvas pass as the body
// of a conditional node, counted by the body's nodes), with the K/V on the GPU and MTP's window (mtp_window=32: its
// draft steps read the last 32 positions past 256), and with the K/V and every layer's low-bit copy in host memory,
// across auto's switch:
//   * the room what the lease with graphs has more than the lease without (the backend's own figure), and the rounds
//     the sessions ran without a graph few: one in kWithoutPart at most (a room too tight for the rounds a session
//     runs, which then gives up graphs it needs again, fails here);
//   * the device memory the sessions took more with graphs than without within the rooms they leased, give or take
//     the driver's last piece of 2 MiB and 2 MiB for what other programs did meanwhile (the driver keeps what the
//     graphs a session gave up held, for the next: what they took is the most the sessions' graphs ever held, and
//     more where pieces were left part used, which the rooms' pool part is for): a bound per graph that falls short
//     (a kernel node counted at half its bytes) fails here;
//   * not a loose one: the rooms under twice what the graphs took;
//   * every kernel's parameters known to the driver, and none past the bound per node was measured with
//     (backends/cuda/backend.h: kGraphParamBytes).
// Case `which` of the two (each in a process of its own: the driver keeps a destroyed graph's memory for the next ones,
// which would take none of the device's), one engine at a time (a real model's two would not fit beside each other).
// `tight`: the rooms under twice what the graphs took, where the model's graphs are as large as the prices are tight
// for (the tiny Llama's rounds are smaller: plain decoding's alone leased 3.9 times what they took; graph_dv_rounds
// holds DV's own graphs to it instead).
void graph_memory(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt,
                  const std::string& drafter, int which, bool tight = true) {
  constexpr uint64_t kGraphParamBound = 1024;
  constexpr int kSessions = 8, kBatched = 12;  // past BatchRounds::kGraphWarmHits (8): captured, then replayed
  constexpr int64_t kWithoutPart = 6;  // a change of head runs up to kGraphRecent (8) rounds without: 7 a session here
  constexpr int64_t kMiB = 1 << 20, kPiece = 2 * kMiB, kOthers = 2 * kMiB;
  std::vector<int32_t> twice = prompt;
  twice.insert(twice.end(), prompt.begin(), prompt.end());
  SamplingParams sampled, penalized, relaxed;
  sampled.temperature = 0.8f;
  sampled.seed = 7;
  penalized = sampled;
  penalized.presence_penalty = 0.5f;
  relaxed.accept_min_p = 0.3f;
  struct Case {
    const char* what;
    int kv_gpu;
    const char* kv_method;
    int kv_copy_gpu;
    std::string options;
    bool trees;  // drafted trees (their rounds read every position exactly: not beside strata)
  };
  const Case cases[] = {
      {"K/V on the GPU", -1, "exact", -1, "mtp_window=32", true},
      {"K/V and the copies in host memory, auto's switch at 48", 0, "auto", 0,
       "strata_denom=8,strata_least=1,strata_from=48", false},
  };
  {
    const Case& c = cases[std::clamp(which, 0, 1)];
    struct Taken {
      int64_t each = 0, device = 0, leased = 0, held = 0, room = 0, from_graph = 0, without_graph = 0;
      uint64_t params = 0, unknown = 0;
      size_t rounds = 0, lookup = 0;
    };
    auto take = [&](const std::string& graphs) {
      const std::string options = c.options + (c.options.empty() || graphs.empty() ? "" : ",") + graphs;
      auto e = make(model, backend, true, drafter, false, options, true, 512, -1, c.trees, c.kv_gpu, c.kv_method,
                    c.kv_copy_gpu);
      auto free_now = [&] {
        flyx_memory_info mi{};
        e->api()->memory(e->bmodel(), &mi);
        return int64_t(mi.free);
      };
      std::vector<size_t> every(e->widths().size());
      for (size_t i = 0; i < every.size(); ++i) every[i] = i;
      std::vector<std::unique_ptr<Session>> kept;
      Taken t;
      const int64_t free0 = free_now(), used0 = int64_t(e->ledger().used());
      for (int i = 0; i < kSessions; ++i) {
        kept.push_back(std::make_unique<Session>(e.get()));
        Session& s = *kept.back();
        if (!init_checked(s)) break;
        s.rounds().cycle(every);
        gen(s, prompt, int(2 * every.size()) + 8);  // every round plan twice
        s.rounds().cycle({});
        gen(s, twice, 32);
        gen(s, twice, 24, nullptr, sampled);
        gen(s, twice, 24, nullptr, penalized);
        gen(s, twice, 24, nullptr, relaxed);
      }
      for (size_t i = 1; i < kept.size(); i += 2)
        check(batched(e.get(), e->round_plan(0), {kept[i - 1].get(), kept[i].get()}, kBatched),
              std::string("graph memory, ") + c.what + ": rounds of two sessions batched");
      // A DV file's warm-ups batched too (each sequence's canvas pass its own, launch_batch): their batch graphs.
      for (size_t r = 0; r < e->widths().size(); ++r)
        if (e->round_name(r) == "dv warm-up")
          for (size_t i = 1; i < kept.size(); i += 2)
            check(batched(e.get(), e->round_plan(r), {kept[i - 1].get(), kept[i].get()}, kBatched),
                  std::string("graph memory, ") + c.what + ": DV's warm-ups of two sessions batched");
      for (auto& s : kept) {
        const flyx_graph_stats st = graph_stats(e.get(), *s);
        t.held += int64_t(st.held);
        t.room += int64_t(st.room);
        t.from_graph += int64_t(st.from_graph);
        t.without_graph += int64_t(st.without_graph);
        t.params = std::max(t.params, st.params_most);
        t.unknown += st.params_unknown;
      }
      t.each = int64_t(e->session_bytes());
      t.device = free0 - free_now();
      t.leased = int64_t(e->ledger().used()) - used0;
      t.rounds = e->widths().size();
      t.lookup = e->lookup_widths().size();
      return t;
    };
    const Taken off = take("graphs=0"), on = take("");
    const std::string at = std::string("graph memory, ") + c.what + ": ";
    const int64_t room = on.each - off.each, graphs = on.device - off.device;
    check(room > 0 && on.room == kSessions * room && off.room == 0 && off.from_graph == 0,
          at + "a session leases its arena (" + std::to_string(off.each >> 10) + " KiB) and room for its graphs (" +
              std::to_string(room >> 10) + " KiB: " + std::to_string(on.rounds) + " rounds, " +
              std::to_string(on.lookup) + " lookup rounds), the backend's own figure; none with graphs off");
    check(on.from_graph > 0 && on.without_graph * kWithoutPart <= on.from_graph + on.without_graph &&
              on.leased == kSessions * on.each && off.leased == kSessions * off.each,
          at + std::to_string(kSessions) + " sessions' rounds: " + std::to_string(on.from_graph) + " from a graph, " +
              std::to_string(on.without_graph) + " without (one in " + std::to_string(kWithoutPart) +
              " at most); their graphs " + std::to_string(on.held >> 10) + " KiB of " + std::to_string(on.room >> 10));
    check(graphs <= on.room + kPiece + kOthers && graphs > 0,
          at + "they took " + std::to_string(graphs >> 10) + " KiB of the device's memory past their arenas (" +
              std::to_string(off.device >> 10) + " KiB with graphs off), within the " + std::to_string(on.room >> 10) +
              " KiB of rooms they leased");
    if (tight)
      check(on.room < 2 * std::max<int64_t>(graphs, kPiece), at + "the rooms under twice what the graphs took");
    check(on.params <= kGraphParamBound && on.unknown == 0,
          at + "no round graph's kernel takes more than " + std::to_string(kGraphParamBound) +
              " bytes of parameters (" + std::to_string(on.params) + "; the driver did not say for " +
              std::to_string(on.unknown) + ")");
  }
}

// Rounds alone (launch and read, greedy) of plan p on session s.
bool alone(Engine* e, flyx_plan* p, Session& s, int rounds) {
  flyx_gen_params g{};
  g.top_p = 1.f;
  flyx_round_result r{};
  for (int i = 0; i < rounds; ++i)
    if (e->api()->launch(p, s.seq(), &g) != 0 || e->api()->read(s.seq(), &r) != 0) return false;
  return true;
}

// A session left alone after rounds batched with three others in turn: the batch graphs it was in give way to its own
// rounds' graph within the kGraphRecent (8) rounds after, judged by its own rounds however few batches run since,
// and its rounds run from that graph from then on.  Plain decoding: a room of one plan's graph and one batch share
// and the pool's part, which the three batch graphs' shares and its own graph do not fit in together.
void graph_after_batch(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt) {
  auto e = make(model, backend, false, "", false, "", false, 512, 0);
  if (!e->api()->graph_stats) return;
  flyx_plan* plan = e->round_plan(0);
  Session a(e.get()), b(e.get()), c(e.get()), d(e.get());
  if (!init_checked(a) || !init_checked(b) || !init_checked(c) || !init_checked(d)) return;
  for (Session* s : {&a, &b, &c, &d}) gen(*s, prompt, 1);
  bool ok = true;
  for (int i = 0; i < 16 && ok; ++i)
    ok = batched(e.get(), plan, {&a, &b}, 1) && batched(e.get(), plan, {&a, &c}, 1) &&
         batched(e.get(), plan, {&a, &d}, 1);
  check(ok, "graph after batches: 48 rounds of two sessions batched, three pairs in turn");
  const flyx_graph_stats before = graph_stats(e.get(), a);
  constexpr int kAlone = 40;
  std::vector<uint64_t> from;  // from_graph after each round alone
  for (int i = 0; i < kAlone && ok; ++i) {
    ok = alone(e.get(), plan, a, 1);
    from.push_back(graph_stats(e.get(), a).from_graph);
  }
  const flyx_graph_stats after = graph_stats(e.get(), a);
  const uint64_t last = ok ? from.back() - from[size_t(kAlone - 21)] : 0;  // of its last 20 rounds alone
  check(ok && before.from_graph >= 16 && last == 20 && after.held <= after.room,
        "graph after batches: of 48 batched rounds " + std::to_string(before.from_graph) +
            " from a graph; alone after them, its last 20 of " + std::to_string(kAlone) + " rounds from its graph (" +
            std::to_string(last) + "), " + std::to_string(after.held >> 10) + " KiB held of " +
            std::to_string(after.room >> 10));
}

// A session that goes from rounds alone to rounds batched with another and back, again and again (a server's long
// request while short ones come and go beside it): its room holds its own graph and its share of the batch graph
// together, so after the first time round it captures nothing more and runs every round from a graph.
void graph_switching(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt) {
  auto e = make(model, backend, false, "", false, "", false, 512, 0);
  if (!e->api()->graph_stats) return;
  flyx_plan* plan = e->round_plan(0);
  Session a(e.get()), b(e.get());
  if (!init_checked(a) || !init_checked(b)) return;
  gen(a, prompt, 1);
  gen(b, prompt, 1);
  constexpr int kTurns = 4, kRounds = 16;  // past BatchRounds::kGraphWarmHits (8) batched, past kGraphRecent alone
  bool ok = batched(e.get(), plan, {&a, &b}, kRounds) && alone(e.get(), plan, a, kRounds);
  const flyx_graph_stats first = graph_stats(e.get(), a);
  for (int i = 1; i < kTurns && ok; ++i)
    ok = batched(e.get(), plan, {&a, &b}, kRounds) && alone(e.get(), plan, a, kRounds);
  const flyx_graph_stats last = graph_stats(e.get(), a);
  const uint64_t captured = last.captured - first.captured, without = last.without_graph - first.without_graph;
  check(ok && first.captured >= 2 && captured == 0 && without == 0,
        "graph switching: a session batched and alone in turn, " + std::to_string(kTurns) + " times " +
            std::to_string(kRounds) + " rounds each: graphs captured " + std::to_string(first.captured) +
            " the first time, then " + std::to_string(captured) + "; rounds without a graph then " +
            std::to_string(without) + " (" + std::to_string(last.held >> 10) + " KiB held of " +
            std::to_string(last.room >> 10) + ")");
}

// DV's round graphs: n sessions each having run DV's rounds only (its trees and its warm-up, a graph each) on three
// engines: graphs off, the canvas pass inline in each round's graph (backend option dv_conditional=0), and the pass the
// body of a conditional node (the default: regions/layers.cpp dv_context_unless_noop).  Every engine's sessions are
// kept till the end: the driver keeps a destroyed graph's memory for the next ones, and the memory a set of sessions
// gave back made the next set's graphs look smaller (about 20 MiB of 64 sessions' here).  What each engine with
// graphs took of the device's memory past the sessions without:
//   * within what its graphs are counted at (graph_stats' held: graph_size), give or take the driver's last piece of
//     2 MiB and 2 MiB for other programs; and not a loose count: under twice what they took;
//   * the conditional nodes and their bodies on their own: what the conditional engine's graphs took past the inline
//     ones within what they are counted at past them, give or take the same.  A body counted without its nodes (most
//     of a DV round's kernels) or without its own graph (kGraphBytes) falls short of that.
// n from the device's free memory (three engines' sessions and graphs, kSpare left for the system) and each engine's
// budget, at most kMost; under kFloor, where a body counted without its own graph would fall short by less than the
// slack, it says so and checks nothing.
void graph_dv_rounds(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt) {
  constexpr int kMost = 64, kFloor = 32, kCtx = 256;
  constexpr int64_t kMiB = 1 << 20, kPiece = 2 * kMiB, kOthers = 2 * kMiB;
  constexpr int64_t kSessionMore = 4 * kMiB;  // a session's graphs and its arenas' rounding past its lease, for n
  constexpr int64_t kSpare = 1024 * kMiB;     // Windows pages GPU memory out before the GPU is full (plan/budget.cpp)
  struct Run {
    std::unique_ptr<Engine> e;
    std::vector<std::unique_ptr<Session>> kept;  // before the engine goes
    int64_t device = 0, held = 0, captured = 0, from_graph = 0, graphs = 0;
    std::string rounds;
  };
  auto engine = [&](const std::string& options) {
    return make(model, backend, true, "dv", false, options, false, kCtx, -1);
  };
  auto free_now = [](Engine* e) {
    flyx_memory_info mi{};
    e->api()->memory(e->bmodel(), &mi);
    return int64_t(mi.free);
  };
  // n sessions of t's engine, each cycling through DV's rounds; the device memory they took (the engine's own before
  // them not counted) and their graphs' figures.
  auto take = [&](Run& t, int n) {
    Engine* e = t.e.get();
    std::vector<size_t> dv;
    for (size_t r = 0; r < e->widths().size(); ++r)
      if (e->round_name(r).rfind("dv ", 0) == 0) {
        dv.push_back(r);
        t.rounds += (t.rounds.empty() ? "" : ", ") + e->round_name(r);
      }
    if (dv.empty()) return;
    t.graphs = int64_t(n) * int64_t(dv.size());
    const int64_t free0 = free_now(e);
    for (int i = 0; i < n; ++i) {
      t.kept.push_back(std::make_unique<Session>(e));
      Session& s = *t.kept.back();
      if (!init_checked(s)) break;
      s.rounds().cycle(dv);
      gen(s, prompt, int(12 * dv.size()));
    }
    for (auto& s : t.kept) {
      const flyx_graph_stats st = graph_stats(e, *s);
      t.held += int64_t(st.held);
      t.captured += int64_t(st.captured);
      t.from_graph += int64_t(st.from_graph);
    }
    t.device = free0 - free_now(e);
  };
  // The three engines first, each one's budget from the device's free memory before any session.
  Run off, flat, cond;
  off.e = engine("graphs=0");
  flat.e = engine("dv_conditional=0");
  cond.e = engine("");
  const int64_t each = int64_t(off.e->session_bytes()) + kSessionMore;
  const int n = int(std::min<int64_t>({(free_now(off.e.get()) - kSpare) / (3 * each), kMost,
                                       off.e->sessions_that_fit(), flat.e->sessions_that_fit(),
                                       cond.e->sessions_that_fit()}));
  if (n < kFloor) {
    std::printf("  skip DV's round graphs: the device's free memory holds %d sessions of %lld MiB on each of three "
                "engines, under the %d that tell a body's own graph from the slack\n",
                n, static_cast<long long>(each >> 20), kFloor);
    return;
  }
  take(off, n);
  take(flat, n);
  take(cond, n);
  const std::string at = "DV's round graphs (" + cond.rounds + "; " + std::to_string(n) + " sessions): ";
  const auto kib = [](int64_t b) { return std::to_string(b >> 10) + " KiB"; };
  check(!cond.rounds.empty() && off.held == 0 && flat.captured == flat.graphs && cond.captured == cond.graphs &&
            flat.from_graph > 0 && cond.from_graph > 0,
        at + "a graph each round, each session (" + std::to_string(flat.captured) + " inline and " +
            std::to_string(cond.captured) + " conditional made of " + std::to_string(cond.graphs) + "; " +
            std::to_string(flat.from_graph) + " and " + std::to_string(cond.from_graph) + " rounds from them)");
  for (const Run* t : {&flat, &cond}) {
    const std::string what = at + (t == &flat ? "the canvas passes inline: " : "the canvas passes conditional: ");
    const int64_t graphs = t->device - off.device;
    check(graphs > 0 && graphs <= t->held + kPiece + kOthers,
          what + "they took " + kib(graphs) + " of the device's memory past their arenas, within the " +
              kib(t->held) + " they are counted at");
    check(t->held < 2 * std::max(graphs, kPiece), what + "counted under twice what they took");
  }
  const int64_t body = cond.device - flat.device, counted = cond.held - flat.held;
  check(counted > 0 && body <= counted + kPiece + kOthers,
        at + "the conditional nodes and their bodies took " + kib(body) + " past the passes inline, within the " +
            kib(counted) + " they are counted at past them");
}

}  // namespace

// Sessions and engines give their device memory back: the device's free memory before and after 20 sessions, and
// after an engine, come and go.  That memory is the whole GPU's, so this runs as a test of its own that runs alone
// (FLYX_TEST_DEVICE_MEMORY=1; cuda_device_memory in tests/CMakeLists.txt), and the golden tests beside each other.
int device_memory(const std::string& model, const std::string& backend, const std::vector<int32_t>& prompt) {
  auto spec = make(model, backend, true);
  auto plain = make(model, backend, false);
  auto free_now = [&](Engine* e) {
    flyx_memory_info mi{};
    e->api()->memory(e->bmodel(), &mi);
    return static_cast<int64_t>(mi.free);
  };
  const int64_t slack = 64ll << 20;  // other users of the GPU (on WSL, the desktop)
  std::string err;
  const int64_t before = free_now(plain.get());
  for (int i = 0; i < 20; ++i) {
    Session t(spec.get());
    if (!init_checked(t)) break;
    gen(t, prompt, 8);
  }
  const int64_t after = free_now(plain.get());
  check(after >= before - slack,
        "20 sessions made and freed: device memory back (" + std::to_string((before - after) >> 20) + " MB apart)");
  const int64_t e0 = free_now(plain.get());
  {
    auto extra = make(model, backend, true);
    Session t(extra.get());
    init_checked(t);
    gen(t, prompt, 8);
  }
  const int64_t e1 = free_now(plain.get());
  check(e1 >= e0 - slack,
        "an engine loaded and unloaded: device memory back (" + std::to_string((e0 - e1) >> 20) + " MB apart)");
  graph_memory(model, backend, prompt, "all", 0);
  graph_after_batch(model, backend, prompt);
  graph_switching(model, backend, prompt);
  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}

int main(int argc, char** argv) {
  const auto args = os::utf8_arguments(argc, argv);
  if (args.size() < 3) {
    std::fprintf(stderr, "usage: golden_test ASSET_DIR BACKEND [VARIANT [MODEL]]  (FLYX_TEST_DEVICE_MEMORY=1: "
                         "the device memory checks alone)\n");
    return 2;
  }
  const std::string& dir = args[1];
  const std::string& backend = args[2];
  // "_nvfp4": the NVFP4 copy of the model; "_qwen3": the tiny Qwen3; "_llama": the tiny Llama (no q/k norms);
  // "_window": a drafter that attends a sliding window of 8 (shorter than its block)
  const std::string variant = args.size() > 3 ? "_" + args[3] : "";
  const std::string reference = dir + "/golden" + variant + ".json";
  std::ifstream f(os::to_path(reference), std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open golden reference %s\n", reference.c_str());
    return 1;
  }
  json g;
  try {
    g = json::parse(f);
  } catch (const json::parse_error& ex) {
    std::fprintf(stderr, "%s: %s\n", reference.c_str(), ex.what());
    return 1;
  }
  const std::string model = args.size() > 4 ? args[4] : dir + "/tiny" + variant + ".flyx";
  g_block = g.value("block_drafter", std::string("dflash2"));
  if (const char* which = std::getenv("FLYX_TEST_DEVICE_MEMORY")) {
    const auto prompt = g["cases"][0]["prompt"].get<std::vector<int32_t>>();
    const std::string mode = which;
    if (mode != "graphs" && mode != "tiers" && mode != "rooms" && mode != "dvgraphs")
      return device_memory(model, backend, prompt);
    // The round graphs alone, on any model given, its drafters as FLYX_TEST_DRAFTER names them: the K/V on the GPU
    // (graphs; rooms: without the rooms' bound from above, for a model of smaller graphs), or the K/V and the copies
    // in host memory across auto's switch (tiers); dvgraphs: a DV file's round graphs, its canvas pass conditional and
    // inline.
    const char* drafter = std::getenv("FLYX_TEST_DRAFTER");
    if (mode == "dvgraphs") graph_dv_rounds(model, backend, prompt);
    else graph_memory(model, backend, prompt, drafter ? drafter : "all", mode == "tiers" ? 1 : 0, mode != "rooms");
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
  }
  if (g.value("plain_only", false)) return plain_only(g, model, backend);
  auto spec = make(model, backend, true);
  auto plain = make(model, backend, false);
  prefill_only_checks(plain.get(), g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  refused_session(plain.get(), g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  // Drafts are compared with the reference's from engines that draft from
  // exact weights; the others draft from NVFP4 copies (the output must not
  // change, only the drafts may).
  auto spec_exact = backend == "cpu_ref" ? nullptr : make(model, backend, true, g_block, false, "draft_weights=exact");
  auto mtp = make(model, backend, true, "mtp", false, backend == "cuda" ? "draft_weights=exact" : "");
  auto mtp_q = make(model, backend, true, "mtp");
  // Every drafter the model has: rounds of each at a fixed width (MTP 7, DFlash2 15), and the
  // drafter and width chosen every round.
  auto both = make(model, backend, true, "all");
  auto both_chosen = make(model, backend, true, "all", false, "", false, 512, -1);
  // FP8 K/V cache: no reference for it, but drafting must still not change the output.
  auto plain8 = make(model, backend, false, g_block, true);
  auto spec8 = make(model, backend, true, g_block, true);
  auto mtp8 = make(model, backend, true, "mtp", true);
  std::printf("backend %s, drafts per round %d\n", spec->backend_name().c_str(), spec->width());
  // Without a drafter whose state follows its own rounds only, the rounds are timed once at load at the prompt's
  // context, a round costing the same at any context: no second timing however long the context
  // may be, no cost per token of it.
  {
    auto longer = make(model, backend, true, "all", false, "", false, 2048, -1);
    bool once = longer->widths().size() > 1 && longer->round_ms_again_at() == 0;
    for (double ms : longer->round_ms_per_token()) once = once && ms == 0;
    check(once, "the rounds of every drafter (" + std::string(longer->drafter()) + ", " +
                    std::to_string(longer->widths().size()) + " rounds) timed once at load, at the prompt's context, "
                    "at a context of 2048");
  }

  int k = 0;
  for (const auto& c : g["cases"]) {
    auto prompt = c["prompt"].get<std::vector<int32_t>>();
    auto want = c["tokens"].get<std::vector<int32_t>>();
    const int N = static_cast<int>(want.size());
    std::printf("case %d: prompt %zu tokens, min margin %.3g\n", k++, prompt.size(), c["min_margin"].get<double>());

    Session sp(plain.get());
    std::string err;
    if (!init_checked(sp)) continue;
    auto o = gen(sp, prompt, N);
    check(o.tokens == want, "plain decode: " + str(o.tokens));

    {
      Session a8(plain8.get()), b8(spec8.get()), c8(mtp8.get());
      init_checked(a8);
      init_checked(b8);
      init_checked(c8);
      const auto t8 = gen(a8, prompt, N).tokens;
      check(gen(b8, prompt, N).tokens == t8 && gen(c8, prompt, N).tokens == t8,
            "fp8 kv cache: " + block_label() + " and MTP decode == plain decode");
    }

    Session ss(spec.get());
    init_checked(ss);
    std::vector<flyx_round_result> trace;
    o = gen(ss, prompt, N, &trace);
    check(o.tokens == want, block_label() + " decode");
    int same = 0, total = 0;
    drafts_equal(trace, c["rounds"], &same, &total);
    std::printf("       drafts equal to the reference: %d / %d\n", same, total);
    if (backend == "cpu_ref") {
      check(same == total, "drafter matches the reference exactly");
    } else {  // from exact weights: float32 against float64, so nearly all (the keys each row attends, a window's too)
      Session sx(spec_exact.get());
      init_checked(sx);
      std::vector<flyx_round_result> tx;
      const bool same_tokens = gen(sx, prompt, N, &tx).tokens == want;
      drafts_equal(tx, c["rounds"], &same, &total);
      check(same_tokens && total > 0 && same * 10 >= total * 9,
            block_label() + " from exact weights: the reference's tokens, its drafts " + std::to_string(same) + " / " +
                std::to_string(total));
    }

    Session so(spec.get());
    init_checked(so);
    auto oracle = c["oracle"].get<std::vector<int32_t>>();
    spec->api()->set_oracle(so.seq(), oracle.data(), static_cast<int32_t>(oracle.size()));
    trace.clear();
    o = gen(so, prompt, N, &trace);
    check(o.tokens == want, "oracle drafts (long accepted runs)");
    std::vector<int> ns;
    for (size_t i = 1; i < trace.size(); ++i) ns.push_back(trace[i].n_accepted);
    auto want_n = c["oracle_n"].get<std::vector<int>>();
    check(std::vector<int>(ns.begin(), ns.begin() + std::min(ns.size(), want_n.size())) == want_n,
          "accepted drafts per round match the reference");

    // Continue the same session from inside the last rounds: a fold rewind.
    auto p2 = prompt;
    p2.insert(p2.end(), want.begin(), want.begin() + N - 2);
    o = gen(so, p2, 2);
    check(o.tokens == std::vector<int32_t>(want.end() - 2, want.end()), "rewind inside the round window");
    check(o.usage.reused_tokens == static_cast<int>(p2.size()) - 1,
          "  reused " + std::to_string(o.usage.reused_tokens) + " of " + std::to_string(p2.size()));
    spec->api()->set_oracle(so.seq(), nullptr, 0);

    // Far back: restore the end-of-prompt checkpoint, then prefill.
    auto p3 = prompt;
    p3.insert(p3.end(), want.begin(), want.begin() + 20);
    o = gen(ss, p3, N - 20);
    check(o.tokens == std::vector<int32_t>(want.begin() + 20, want.end()), "continue from an earlier point");
    std::printf("       reused %d of %zu tokens\n", o.usage.reused_tokens, p3.size());

    // Edited prompt: compare a reused session with a fresh one.
    auto p4 = prompt;
    p4.back() = (p4.back() + 1) % 300;
    o = gen(ss, p4, 16);
    Session fresh(spec.get());
    init_checked(fresh);
    auto o2 = gen(fresh, p4, 16);
    check(o.tokens == o2.tokens, "edited prompt: reused session == fresh session");

    if (k == 1) {  // a checkpoint through host memory, with each drafter's caches
      snapshot_through_host(spec.get(), plain.get(), prompt, "snapshot, DFlash2", backend == "cpu_ref");
      snapshot_through_host(mtp.get(), plain.get(), prompt, "snapshot, MTP", backend == "cpu_ref");
      kv_in_host_memory(model, backend, prompt, want);
      strata_rounds(model, backend, prompt, want);
      mtp_window_checks(model, backend, prompt, want);
      rows_on_tensor_cores(model, backend, prompt, want);
    }
    if (k == 1) {  // rounds of several sequences: plain decoding beside each drafter's narrowest round; one plan, thrice
      Engine* E = both_chosen.get();
      std::vector<size_t> pick;
      for (size_t i = 0; i < E->widths().size(); ++i)
        if (E->widths()[i] == 0) {
          pick.push_back(i);
          break;
        }
      for (int d = 0; d < 4; ++d) {
        size_t best = E->widths().size();
        for (size_t i = 0; i < E->widths().size(); ++i)
          if (E->round_drafters()[i] == d && (best == E->widths().size() || E->widths()[i] < E->widths()[best])) best = i;
        if (best < E->widths().size()) pick.push_back(best);
      }
      auto other = prompt;
      other[0] = (other[0] + 7) % 300;
      auto twice = prompt;
      twice.insert(twice.end(), prompt.begin(), prompt.end());
      const std::vector<std::vector<int32_t>> prompts = {prompt, other, twice, prompt};
      batch_rounds(E, std::vector<std::vector<int32_t>>(prompts.begin(), prompts.begin() + std::ptrdiff_t(pick.size())),
                   pick, "rounds of several sequences, plain and each drafter");
      batch_rounds(E, {prompt, other, twice}, {pick.back(), pick.back(), pick.back()},
                   "rounds of three sequences, one plan");
    }
    if (k == 1 && backend == "cuda") {  // tree rounds (the reference backend has none)
      SamplingParams greedy, sampled, penalized;
      sampled.temperature = 0.8f;
      sampled.top_p = 0.95f;
      sampled.seed = 7;
      penalized.presence_penalty = 0.5f;
      penalized.frequency_penalty = 0.3f;
      tree_rounds(plain.get(), prompt, greedy, "tree rounds");
      tree_rounds(plain.get(), prompt, sampled, "tree rounds, sampling");
      tree_rounds(plain.get(), prompt, penalized, "tree rounds, penalties");
      tree_rounds(plain8.get(), prompt, greedy, "tree rounds, fp8 kv cache");
      tree_rounds(plain.get(), prompt, greedy, "tree rounds, min-p 1 (only the top token qualifies)", {.min_p = 1.f});
      // Each round launched while the one before it is in flight (the gated delta net's replays included).
      tree_rounds(plain.get(), prompt, sampled, "tree rounds two in flight, sampling", {.two_in_flight = true});
      // The drafter's context after tree rounds (their accepted rows moved) is what it is after the same rounds
      // whose rows needed no moving: the MTP rounds after give the same ring entries, drafts included.
      flyx_plan* mtp_round = nullptr;
      for (size_t i = 0; i < mtp->widths().size(); ++i)
        if (mtp->widths()[i] > 0) mtp_round = mtp->round_plan(i);
      std::vector<int> paths;
      const auto moved =
          tree_rounds(mtp.get(), prompt, greedy, "tree rounds, MTP", {.paths = &paths, .then = mtp_round, .after = 6});
      const auto in_place = tree_rounds(mtp.get(), prompt, greedy, "tree rounds, MTP, paths first",
                                        {.paths = &paths, .path_first = true, .then = mtp_round, .after = 6});
      bool same = !moved.empty() && moved.size() == in_place.size();
      for (size_t i = 0; same && i < moved.size(); ++i) {
        const flyx_round_result &x = moved[i], &y = in_place[i];
        same = x.n_tokens == y.n_tokens && x.n_drafts == y.n_drafts &&
               std::equal(x.tokens, x.tokens + x.n_tokens, y.tokens) && std::equal(x.drafts, x.drafts + x.n_drafts, y.drafts);
      }
      check(same, "tree rounds, MTP: the drafter's rounds after them are the rounds after paths that did not move");
      // Relaxed acceptance in the drafters' own rounds (chains): with min-p 1 only the target's top token
      // qualifies, which greedy decoding chooses, so the tokens are plain decoding's.
      auto relaxed_gen = [&](Engine* e, float min_p, int* accepted = nullptr) {
        Session s(e);
        init_checked(s);
        GenerateRequest rq;
        rq.prompt = prompt;
        rq.max_new_tokens = 48;
        rq.ignore_eos = true;
        rq.sampling.accept_min_p = min_p;
        std::vector<int32_t> out;
        GenerateResult res;
        const TextSink collect = [&](std::string_view, const int32_t* t, int nt) {
          out.insert(out.end(), t, t + nt);
          return true;
        };
        if (generate(s, rq, collect, &res, &err) != 0) out.clear();
        if (accepted) *accepted = res.usage.accepted_drafts;
        return out;
      };
      Session sp(plain.get());
      init_checked(sp);
      const auto plain_tokens = gen(sp, prompt, 48).tokens;
      check(relaxed_gen(mtp.get(), 1.f) == plain_tokens && relaxed_gen(spec.get(), 1.f) == plain_tokens,
            "relaxed acceptance, min-p 1 (only the top token qualifies): MTP and DFlash2 give plain decoding's tokens");
      // ... and with min-p 1e-37 nearly every draft qualifies (the tiny models' distributions are sharp: random
      // drafts sit far below 1e-9 of the top): the rule is in force, more drafts are accepted.
      int exact_m = 0, exact_d = 0, loose_m = 0, loose_d = 0;
      relaxed_gen(mtp.get(), 0.f, &exact_m);
      relaxed_gen(spec.get(), 0.f, &exact_d);
      relaxed_gen(mtp.get(), 1e-37f, &loose_m);
      relaxed_gen(spec.get(), 1e-37f, &loose_d);
      check(loose_m + loose_d > exact_m + exact_d,
            "relaxed acceptance, min-p 1e-37: more drafts accepted (MTP " + std::to_string(loose_m) + " against " +
                std::to_string(exact_m) + ", DFlash2 " + std::to_string(loose_d) + " against " +
                std::to_string(exact_d) + ")");
      if (g_block == "dflash2") drafted_trees(model, backend, prompt, want, oracle);  // DSpine drafts chains only
    }

    // MTP drafter: same tokens; drafts equal to the reference's; state kept
    // right across rollback and checkpoint restore.
    {
      auto good_enough = [&](int same, int total) { return backend == "cpu_ref" ? same == total : same * 5 >= total * 4; };
      Session sm(mtp.get());
      init_checked(sm);
      std::vector<flyx_round_result> tr;
      o = gen(sm, prompt, N, &tr);
      check(o.tokens == want, "MTP decode");
      {
        Session sq(mtp_q.get());
        init_checked(sq);
        check(gen(sq, prompt, N).tokens == want, "MTP decode, NVFP4 draft head");
      }
      int same = 0, total = 0;
      drafts_equal(tr, c["mtp_rounds"], &same, &total);
      check(total > 0 && good_enough(same, total),
            "  MTP drafts equal to the reference: " + std::to_string(same) + " / " + std::to_string(total));

      Session smo(mtp.get());
      init_checked(smo);
      mtp->api()->set_oracle(smo.seq(), oracle.data(), static_cast<int32_t>(oracle.size()));
      tr.clear();
      o = gen(smo, prompt, N, &tr);
      check(o.tokens == want, "MTP with oracle drafts");
      std::vector<int> mn;
      for (size_t i = 1; i < tr.size(); ++i) mn.push_back(tr[i].n_accepted);
      auto want_mn = c["mtp_oracle_n"].get<std::vector<int>>();
      check(std::vector<int>(mn.begin(), mn.begin() + std::min(mn.size(), want_mn.size())) == want_mn,
            "  accepted MTP drafts per round match the reference");
      mtp->api()->set_oracle(smo.seq(), nullptr, 0);

      // Reused session (rollback / checkpoint) vs fresh session: same drafts.
      for (int back : {2, 20}) {
        auto pr = prompt;
        pr.insert(pr.end(), want.begin(), want.begin() + (N - back));
        std::vector<flyx_round_result> a, b;
        auto oa = gen(sm, pr, back, &a);
        Session fresh(mtp.get());
        init_checked(fresh);
        auto ob = gen(fresh, pr, back, &b);
        int sa = 0, ta = 0;
        for (size_t i = 1; i < a.size() && i < b.size(); ++i)
          for (int j = 0; j < a[i].n_drafts; ++j, ++ta) sa += a[i].drafts[j] == b[i].drafts[j];
        check(oa.tokens == ob.tokens && ta > 0 && good_enough(sa, ta),
              "  MTP after reuse (" + std::to_string(oa.usage.reused_tokens) + " reused): drafts as fresh " +
                  std::to_string(sa) + " / " + std::to_string(ta));
      }
    }

    // Both drafters at once: every round updates both drafters' contexts, so either can draft the
    // next one.  Taking turns round by round they give the reference's tokens, and each round's drafts are those a
    // fresh session with that drafter alone drafts from the same point (on the float64 backend exactly; on the GPU
    // the fresh session's prefill passes may round otherwise).  Chosen every round, they give the reference's
    // tokens, greedy, and plain decoding's, sampled.
    {
      std::vector<size_t> turns;  // the MTP layer's round, then DFlash2's
      for (const std::string name : {std::string("mtp"), g_block})
        for (size_t i = 0; i < both->widths().size(); ++i)
          if (both->round_drafters()[i] >= 0 && name == both->model().drafter_name(both->round_drafters()[i]))
            turns.push_back(i);
      Session sb(both.get());
      init_checked(sb);
      sb.rounds().cycle(turns);
      std::vector<flyx_round_result> tb;
      o = gen(sb, prompt, N, &tb);
      check(turns.size() == 2 && o.tokens == want, "both drafters taking turns: the reference's tokens");
      int same = 0, total = 0;
      size_t done = 1;  // tokens generated before the round: the prefill's, then the rounds'
      for (size_t k = 1; k < tb.size() && k <= 12 && done < want.size(); done += size_t(tb[k].n_tokens), ++k) {
        Engine* alone = (k - 1) % 2 == 0 ? mtp_q.get() : spec.get();
        auto upto = prompt;  // the sequence before the round's anchor: a fresh prefill chooses the anchor
        upto.insert(upto.end(), want.begin(), want.begin() + std::ptrdiff_t(done - 1));
        Session fresh(alone);
        init_checked(fresh);
        std::vector<flyx_round_result> tf;
        gen(fresh, upto, 2, &tf);
        if (tf.size() < 2) continue;
        for (int j = 0; j < tb[k].n_drafts && j < tf[1].n_drafts; ++j, ++total) same += tb[k].drafts[j] == tf[1].drafts[j];
      }
      check(total > 0 && (backend == "cpu_ref" ? same == total : same * 5 >= total * 4),
            "  each round's drafts as a fresh session with its drafter alone drafts them: " + std::to_string(same) +
                " / " + std::to_string(total));
      Session sc(both_chosen.get());
      init_checked(sc);
      check(gen(sc, prompt, N).tokens == want, "both drafters, chosen every round: the reference's tokens");
    }

    // Sampling: for one seed the output does not depend on speculation.
    for (auto sp_cfg : {SamplingParams{0.8f, 1.f, 0, 11}, SamplingParams{1.0f, 0.9f, 40, 12345}}) {
      GenerateRequest rq;
      rq.prompt = prompt;
      rq.max_new_tokens = N;
      rq.ignore_eos = true;
      rq.sampling = sp_cfg;
      auto run = [&](Engine* e) {
        Session s(e);
        init_checked(s);
        std::vector<int32_t> toks;
        GenerateResult res;
        generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
          toks.insert(toks.end(), t, t + nt);
          return true;
        }, &res, &err);
        if (static_cast<int>(toks.size()) != N)
          std::printf("       generated %d, reported %zu, finish %d\n", res.usage.new_tokens, toks.size(),
                      res.usage.finish_reason);
        return toks;
      };
      auto a = run(plain.get()), b = run(spec.get()), c2 = run(both_chosen.get());
      if (a != b || static_cast<int>(a.size()) != N) {
        size_t d = 0;
        while (d < a.size() && d < b.size() && a[d] == b[d]) ++d;
        std::printf("       plain %zu tokens, DFlash2 %zu, first difference at %zu\n", a.size(), b.size(), d);
      }
      check(a == b && static_cast<int>(a.size()) == N,
            "sampled (T=" + std::to_string(sp_cfg.temperature).substr(0, 3) + "): DFlash2 == plain for one seed");
      check(c2 == a, "  both drafters, chosen every round == plain for one seed");
    }

    // Penalties on the tokens a reply already has: plain
    // decoding gives the reference's tokens, and the choice stays exact under
    // speculation: every drafter, lookup drafts and oracle drafts (long
    // accepted runs, where the rows count the drafts before them) give plain
    // decoding's tokens, greedy and sampled.
    {
      const json& pen = c["penalty"];
      const float presence = pen.is_null() ? 1.5f : pen["presence"].get<float>();
      const float frequency = pen.is_null() ? 0.25f : pen["frequency"].get<float>();
      auto mtp_l = make(model, backend, true, "mtp", false, "", true);
      for (auto sp_cfg : {SamplingParams{0.f, 1.f, 0, 0, presence, frequency},
                          SamplingParams{1.0f, 0.9f, 40, 4711, presence, frequency},
                          SamplingParams{0.7f, 0.8f, 20, 99, 1.5f, 0.f}}) {
        const bool greedy = !(sp_cfg.temperature > 0);
        GenerateRequest rq;
        rq.prompt = prompt;
        rq.max_new_tokens = N;
        rq.ignore_eos = true;
        rq.sampling = sp_cfg;
        auto run = [&](Engine* e, bool with_oracle) {
          Session s(e);
          init_checked(s);
          if (with_oracle) e->api()->set_oracle(s.seq(), oracle.data(), static_cast<int32_t>(oracle.size()));
          std::vector<int32_t> toks;
          GenerateResult res;
          if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
                toks.insert(toks.end(), t, t + nt);
                return true;
              }, &res, &err) != 0)
            check(false, "penalties: generate: " + err);
          return toks;
        };
        const auto a = run(plain.get(), false);
        const std::string mode = std::string(greedy ? "greedy" : "sampled") + ", presence " +
                                 std::to_string(sp_cfg.presence_penalty).substr(0, 4) + ", frequency " +
                                 std::to_string(sp_cfg.frequency_penalty).substr(0, 4);
        if (greedy && !pen.is_null()) {
          const auto ref = pen["tokens"].get<std::vector<int32_t>>();
          check(a == ref, "penalties (" + mode + "): plain decoding == the reference" +
                              (pen["differs"].get<bool>() ? " (not the tokens without penalties)" : ""));
        }
        const auto b = run(spec.get(), false), m = run(mtp_q.get(), false), o = run(spec.get(), true),
                   mo = run(mtp.get(), true), l = run(mtp_l.get(), false);
        if (b != a || m != a || o != a || mo != a || l != a)
          std::printf("       plain   %s\n       DFlash2 %s\n       MTP     %s\n       oracle  %s\n", str(a, 24).c_str(),
                      str(b, 24).c_str(), str(m, 24).c_str(), str(o, 24).c_str());
        check(static_cast<int>(a.size()) == N && b == a && m == a && o == a && mo == a && l == a,
              "penalties (" + mode + "): DFlash2, MTP, oracle drafts and lookup == plain");
      }
    }

    // Grammar-constrained decoding: checking each round and
    // choosing a rejected token again under the grammar's mask gives exactly
    // what masking every token gives (the reference), with and without
    // drafts, greedy and sampled.  The random model breaks the grammar often,
    // so the correction path runs many times.
    {
      const std::string tag =
          R"({"type": "structural_tag", "format": {"type": "json_schema", "style": "json", "json_schema":
          {"type": "object", "properties": {"n": {"type": "integer"}, "w": {"type": "string", "enum": ["alpha", "beta"]},
          "s": {"type": "string"}}, "required": ["n", "w", "s"]}}})";
      // With penalties too: a token chosen again makes the reply shorter, and
      // what the penalties count has to follow it back.
      for (auto sp_cfg : {SamplingParams{0.f, 1.f, 0, 0}, SamplingParams{1.0f, 0.9f, 40, 77},
                          SamplingParams{0.f, 1.f, 0, 0, 1.5f, 0.25f}, SamplingParams{1.0f, 0.9f, 40, 77, 1.5f, 0.25f}}) {
        const std::string mode = std::string(sp_cfg.temperature > 0 ? "sampled" : "greedy") +
                                 (sp_cfg.presence_penalty != 0 ? ", penalties" : "");
        auto run = [&](Engine* e, bool reference, int* fixes) {
          std::string gerr;
          auto m = e->grammar().matcher(tag, &gerr);
          std::vector<int32_t> toks;
          if (!m) {
            check(false, "grammar compiles: " + gerr);
            return toks;
          }
          GenerateRequest rq;
          rq.prompt = prompt;
          rq.max_new_tokens = 40;
          rq.sampling = sp_cfg;
          rq.grammar = m.get();
          rq.grammar_reference = reference;
          Session s(e);
          init_checked(s);
          GenerateResult res;
          if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
                toks.insert(toks.end(), t, t + nt);
                return true;
              }, &res, &err) != 0)
            check(false, "constrained generate: " + err);
          if (res.stop_token >= 0) toks.push_back(res.stop_token);  // a finished reply ends with a stop token
          if (fixes) *fixes = res.grammar_fixes;
          return toks;
        };
        int fix_plain = 0, fix_spec = 0, fix_mtp = 0;
        const auto ref = run(plain.get(), true, nullptr);
        const auto a = run(plain.get(), false, &fix_plain), b = run(spec.get(), false, &fix_spec),
                   m = run(mtp.get(), false, &fix_mtp);
        std::string gerr;
        auto replay = plain->grammar().matcher(tag, &gerr);
        bool fits = replay && !ref.empty();
        for (int32_t t : ref) fits = fits && replay->accept(t);
        check(fits, "grammar (" + mode + "): every token of the reference fits it: " + str(ref));
        if (a != ref || b != ref || m != ref)
          std::printf("       reference %s\n       checked   %s\n       DFlash2   %s\n       MTP       %s\n",
                      str(ref).c_str(), str(a).c_str(), str(b).c_str(), str(m).c_str());
        check(a == ref && b == ref && m == ref && fix_plain > 0,
              "grammar (" + mode + "): checked == every token masked, plain, DFlash2 and MTP (" +
                  std::to_string(fix_plain) + ", " + std::to_string(fix_spec) + ", " + std::to_string(fix_mtp) +
                  " tokens corrected)");
      }
    }

    // Drafts looked up in the sequence itself: text that
    // recurs makes rounds take their drafts from where it appeared before;
    // its rounds verify them with no drafter run, up to 15 at once.  MTP,
    // DFlash2 and no drafter with lookup still give plain decoding's tokens,
    // greedy and sampled.
    {
      auto mtp_l = make(model, backend, true, "mtp", false, "", true);
      auto spec_l = make(model, backend, true, g_block, false, "", true);
      auto plain_l = make(model, backend, true, "none", false, "", true);  // no drafter: lookup rounds only
      std::vector<int32_t> twice = prompt;  // a prompt that repeats itself
      twice.insert(twice.end(), prompt.begin(), prompt.end());
      for (auto sp_cfg : {SamplingParams{0.f, 1.f, 0, 0}, SamplingParams{1.0f, 0.9f, 40, 77}}) {
        const std::string mode = sp_cfg.temperature > 0 ? "sampled" : "greedy";
        auto run = [&](Engine* e, int* lookups) {
          GenerateRequest rq;
          rq.prompt = twice;
          rq.max_new_tokens = 64;
          rq.sampling = sp_cfg;
          Session s(e);
          init_checked(s);
          std::vector<int32_t> toks;
          GenerateResult res;
          if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
                toks.insert(toks.end(), t, t + nt);
                return true;
              }, &res, &err) != 0)
            check(false, "lookup: generate: " + err);
          if (lookups) *lookups = res.lookup_rounds;
          return toks;
        };
        int lm = 0, ls = 0, lp = 0;
        const auto ref = run(plain.get(), nullptr), a = run(mtp_l.get(), &lm), b = run(spec_l.get(), &ls),
                   c = run(plain_l.get(), &lp);
        if (lm + ls + lp == 0) std::printf("       plain: %s\n", str(ref, 64).c_str());
        check(a == ref && b == ref && c == ref && (mode == "sampled" || (lm > 0 && ls > 0 && lp > 0)),
              "lookup drafts (" + mode + "): MTP, DFlash2 and no drafter == plain (" + std::to_string(lm) + ", " +
                  std::to_string(ls) + ", " + std::to_string(lp) + " lookup rounds)");
      }
      // The same request again on the same session: its tokens are cut back
      // to the prompt, and entries indexed past the cut must not hide the
      // prompt's own occurrences (the second request found no match at all).
      {
        Session s(mtp_l.get());
        init_checked(s);
        GenerateRequest rq;
        rq.prompt = twice;
        rq.max_new_tokens = 64;
        rq.sampling = SamplingParams{0.f, 1.f, 0, 0};
        std::vector<int32_t> out[2];
        int rounds[2] = {0, 0};
        for (int i = 0; i < 2; ++i) {
          GenerateResult res;
          if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
                out[i].insert(out[i].end(), t, t + nt);
                return true;
              }, &res, &err) != 0)
            check(false, "lookup again: generate: " + err);
          rounds[i] = res.lookup_rounds;
        }
        std::string why;
        if (out[1] != out[0]) {  // which of the two left plain decoding, and where
          Session sr(plain.get());
          init_checked(sr);
          const auto ref = gen(sr, twice, 64).tokens;
          why = "; first: " + first_difference(out[0], ref) + "; again: " + first_difference(out[1], ref);
        }
        check(out[1] == out[0] && rounds[1] > 0,
              "lookup drafts: the same request again finds its matches (" + std::to_string(rounds[0]) + ", " +
                  std::to_string(rounds[1]) + " lookup rounds" + why + ")");
      }
    }

    // A question's checkpoint is kept through the requests that follow it
    // (GenerateRequest::keep_checkpoint): six tool turns later, a prompt that
    // differs right after the question still resumes there, and gives what a
    // fresh session gives.  Without keeping it, later checkpoints replace it.
    {
      auto run = [&](Session& s, const std::vector<int32_t>& p, std::vector<int32_t> cks, int keep,
                     flyx_usage* u) {
        GenerateRequest rq;
        rq.prompt = p;
        rq.max_new_tokens = 4;
        rq.checkpoints = std::move(cks);
        rq.keep_checkpoint = keep;
        std::vector<int32_t> toks;
        GenerateResult res;
        if (generate(s, rq, [&](std::string_view, const int32_t* t, int nt) {
              toks.insert(toks.end(), t, t + nt);
              return true;
            }, &res, &err) != 0)
          check(false, "kept checkpoint: generate: " + err);
        if (u) *u = res.usage;
        return toks;
      };
      std::vector<int32_t> q = prompt;  // the question: long enough for an aligned checkpoint
      while (q.size() < 140) q.insert(q.end(), prompt.begin(), prompt.end());
      const int qn = static_cast<int>(q.size()), at = qn / plain->prefill_align() * plain->prefill_align();
      for (bool keep : {true, false}) {
        Session sk(plain.get());
        init_checked(sk);
        std::vector<int32_t> asked = q;
        asked.insert(asked.end(), {5, 6, 7});  // the generation prompt
        run(sk, asked, {qn}, keep ? qn : -1, nullptr);
        std::vector<int32_t> loop = asked;
        for (int t = 0; t < 6; ++t) {  // tool turns, two checkpoints each
          for (int j = 0; j < 9; ++j) loop.push_back(10 + (t * 9 + j) * 7 % 250);
          run(sk, loop, {static_cast<int>(loop.size()) - 4, static_cast<int>(loop.size())}, -1, nullptr);
        }
        std::vector<int32_t> next = q;  // the next question: different right after the first
        for (int j = 0; j < 12; ++j) next.push_back(20 + j * 11 % 230);
        flyx_usage u{};
        const auto a = run(sk, next, {}, -1, &u);
        Session fresh(plain.get());
        init_checked(fresh);
        const auto b = run(fresh, next, {}, -1, nullptr);
        const std::string reused = " (reused " + std::to_string(u.reused_tokens) + " of " +
                                   std::to_string(next.size()) + ", checkpoint at " + std::to_string(at) + ")";
        if (keep)
          check(a == b && at > 0 && u.reused_tokens == at,
                "a question's checkpoint is kept through six later requests" + reused);
        else
          check(a == b && u.reused_tokens < at, "without keeping it, later checkpoints replace it" + reused);
      }
    }

    // Two requests taking turns on the GPU thread: each on a
    // session of its own, advanced when the device has its next result.  Each
    // gives the tokens it gives alone: a greedy one, and a sampled one with a
    // longer prompt that arrives while the first is decoding.
    {
      GenerateRequest ra, rb;
      ra.prompt = prompt;
      ra.max_new_tokens = N;
      ra.ignore_eos = true;
      rb.prompt = prompt;
      rb.prompt.insert(rb.prompt.end(), prompt.begin(), prompt.end());
      rb.max_new_tokens = N;
      rb.ignore_eos = true;
      rb.sampling = SamplingParams{1.0f, 0.9f, 40, 12345};
      auto into = [](std::vector<int32_t>* out) {
        return [out](std::string_view, const int32_t* t, int nt) {
          out->insert(out->end(), t, t + nt);
          return true;
        };
      };
      for (Engine* e : {plain.get(), spec.get(), mtp_q.get()}) {
        std::vector<int32_t> alone_b;
        {
          Session s(e);
          init_checked(s);
          GenerateResult res;
          generate(s, rb, into(&alone_b), &res, &err);
        }
        Session sa(e), sb(e);
        init_checked(sa);
        init_checked(sb);
        std::vector<int32_t> ta, tb;
        int both = 0;  // turns in which both were running
        {
          Generation ga(&sa, ra, into(&ta));
          for (int i = 0; i < 6; ++i) ga.advance(true);
          Generation gb(&sb, rb, into(&tb));
          while (!ga.done() || !gb.done()) {
            both += !ga.done() && !gb.done();
            bool worked = false;
            for (Generation* g : {&ga, &gb})
              if (!g->done() && g->advance(false) != Generation::Progress::kBusy) worked = true;
            if (!worked) os::sleep_ms(0);
          }
          check(!ga.failed() && !gb.failed(), "two requests taking turns: both end " + ga.error() + gb.error());
        }
        check(ta == want && tb == alone_b && static_cast<int>(tb.size()) == N && both > 0,
              std::string("two requests taking turns (") + e->drafter() + "): each gives what it gives alone (" +
                  std::to_string(both) + " turns together)");
        // The sessions they ran on are free again and reusable.
        const bool free = !sa.in_use() && !sb.in_use();
        std::vector<flyx_round_result> trace;
        const Out again = gen(sa, prompt, N, &trace);
        const bool ok = free && again.tokens == want;
        std::string rounds;  // the first results the device gave, for a failure
        for (size_t i = 0; !ok && i < trace.size() && i < 4; ++i)
          rounds += " [position " + std::to_string(trace[i].position) + ", " + std::to_string(trace[i].n_tokens) +
                    " tokens, first " + std::to_string(trace[i].n_tokens > 0 ? trace[i].tokens[0] : -1) +
                    (trace[i].stopped ? ", stopped" : "") + ", seq " + std::to_string(trace[i].seq_no) + "]";
        check(ok, "  their sessions go on as usual" +
                      (ok ? std::string()
                          : " (" + std::string(free ? "free" : "still in use") + "; reused " +
                                std::to_string(again.usage.reused_tokens) + " of " + std::to_string(prompt.size()) +
                                "; " + first_difference(again.tokens, want) + ";" + rounds + ")"));
      }
      // Two threads at once, each generating on a session of its own (the C
      // API allows it): prompts of several prefill passes, which take the
      // engine's shared prefill buffers one after the other.  Each gives what
      // it gives alone.
      std::vector<int32_t> long_a, long_b;
      while (long_a.size() < 64) long_a.insert(long_a.end(), prompt.begin(), prompt.end());
      long_b = long_a;
      long_b.insert(long_b.end(), rb.prompt.begin(), rb.prompt.end());
      GenerateRequest qa = ra, qb = rb;
      qa.prompt = long_a;
      qb.prompt = long_b;
      for (Engine* e : {plain.get(), spec.get(), mtp_q.get()}) {
        std::vector<int32_t> alone[2], both[2];
        std::string why[2];
        const GenerateRequest* rq[2] = {&qa, &qb};
        for (int i = 0; i < 2; ++i) {
          Session s(e);
          init_checked(s);
          GenerateResult res;
          generate(s, *rq[i], into(&alone[i]), &res, &err);
        }
        {
          Session s0(e), s1(e);
          init_checked(s0);
          init_checked(s1);
          Session* ss[2] = {&s0, &s1};
          std::atomic<int> started{0};
          auto run = [&](int i) {
            started.fetch_add(1);
            while (started.load() < 2) {
            }
            GenerateResult res;
            if (generate(*ss[i], *rq[i], into(&both[i]), &res, &why[i]) != 0 && why[i].empty()) why[i] = "failed";
          };
          std::thread t0(run, 0), t1(run, 1);
          t0.join();
          t1.join();
        }
        check(why[0].empty() && why[1].empty() && both[0] == alone[0] && both[1] == alone[1] &&
                  static_cast<int>(alone[0].size()) == N,
              std::string("two threads at once (") + e->drafter() + "): each gives what it gives alone (" +
                  first_difference(both[0], alone[0]) + "; " + first_difference(both[1], alone[1]) + why[0] +
                  why[1] + ")");
      }
    }

    // Stop string: the output ends right before its first occurrence.
    {
      const auto& tk = spec->tokenizer();
      std::string full = tk.decode(want);
      std::string stop = tk.decode({want.begin() + 10, want.begin() + 12});
      size_t at = full.find(stop);
      GenerateRequest rq;
      rq.prompt = prompt;
      rq.max_new_tokens = N;
      rq.ignore_eos = true;
      rq.stop = {stop};
      Session s(spec.get());
      init_checked(s);
      std::string shown;
      GenerateResult res;
      generate(s, rq, [&](std::string_view t, const int32_t*, int) {
        shown += t;
        return true;
      }, &res, &err);
      check(!stop.empty() && shown == full.substr(0, at) && res.stop_string == stop &&
                res.usage.finish_reason == FLYX_FINISH_STOP,
            "stop string ends the output before it");
      // The next turn continues from what was shown, not from what was generated.
      std::string how;
      auto ids = s.tokenize(tk.decode(prompt) + shown, nullptr, &how);
      check(ids.size() >= prompt.size() && std::equal(prompt.begin(), prompt.end(), ids.begin()),
            "  history after a stop string keeps the prompt tokens (" + how + ")");
    }
  }
  // Contract: a cancel stops a generation where it is and leaves the session usable.
  if (backend == "cuda" && !g["cases"].empty()) {
    const auto& c0 = g["cases"][0];
    const auto prompt = c0["prompt"].get<std::vector<int32_t>>();
    const auto want = c0["tokens"].get<std::vector<int32_t>>();
    std::string err;
    {
      Session sc(spec.get());
      init_checked(sc);
      Out o;
      GenerateRequest rq;
      rq.prompt = prompt;
      rq.max_new_tokens = static_cast<int>(want.size());
      rq.ignore_eos = true;
      GenerateResult res;
      int pieces = 0;
      generate(sc, rq, [&](std::string_view, const int32_t* t, int nt) {
        o.tokens.insert(o.tokens.end(), t, t + nt);
        return ++pieces < 2;  // cancel after the second piece of text
      }, &res, &err);
      o.usage = res.usage;
      check(o.usage.finish_reason == FLYX_FINISH_CANCEL && o.tokens.size() < want.size() &&
                std::equal(o.tokens.begin(), o.tokens.end(), want.begin()),
            "cancel: stops early, with the reference's tokens so far (" + std::to_string(o.tokens.size()) + ")");
      check(gen(sc, prompt, static_cast<int>(want.size())).tokens == want,
            "cancel: the same session then gives the reference output");
    }
  }

  context_end(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  adaptive_stopping(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  if (backend == "cuda") wide_rounds(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  if (backend == "cuda") mid_pass_checkpoint(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  rewind_prefill_rewind(model, backend, g["cases"][0]["prompt"].get<std::vector<int32_t>>());
  if (variant.empty()) {
    std::vector<std::vector<int32_t>> prompts;
    for (const auto& c : g["cases"]) prompts.push_back(c["prompt"].get<std::vector<int32_t>>());
    wide_block(dir + "/tiny_block8.flyx", backend, prompts);
  }
  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
