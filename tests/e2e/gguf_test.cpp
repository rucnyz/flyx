// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A model from its GGUF file is the model.
//   gguf_e2e_test ASSETS BACKEND [BACKEND_OPTIONS]
// * The tiny Qwen3.5 as GGUF (float32 weights in llama.cpp's layout) gives the
//   golden outputs of the float64 reference, with and without the MTP
//   drafter; seeded sampling and edited prompts as from its .flyx file.
// * Attention dimensions the 27B family has (hidden not a multiple of heads).
// * Packed block weights (a backend other than cpu_ref): a tiny Qwen3 per GGML
//   block type and the tiny Qwen3.5 with its matrices quantized give the
//   tokens the float64 backend gives from the same file (8-bit inputs in
//   every pass, per type teacher forced, where a
//   near tie may part), through short prefill passes (the rounds' kernels)
//   and long ones; drafting changes nothing.
// * Drafters in files of their own: the MTP layer from a GGUF of its own, and
//   a DFlash2 drafter file.
// * NVFP4 weights whose inputs are NVFP4 in passes of more than 16 rows (the
//   W4A4 kernels, tiny_w4a4.flyx, a Qwen3.5 128 wide): the
//   float64 backend's tokens, teacher forced, with the same prefill passes
//   (the rounding depends on a pass's rows); asked for by name
//   (--activations nvfp4) in a file that says bf16 inputs, the same tokens.
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "core/exec/generation.h"
#include "nlohmann/json.hpp"

using namespace flyx;
using json = nlohmann::json;

namespace {
int failures = 0;
std::string tested, tested_options;  // the backend under test and its options (the float64 backend takes none)
void check(bool ok, const std::string& message) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", message.c_str());
  failures += !ok;
}
// The same tokens; else where the two part, said.
bool same_tokens(const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
  if (got == want) return true;
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::printf("  (the tokens part at %zu of %zu)\n", i, want.size());
  return false;
}

// How a check loads its engine.
struct Load {
  bool speculative = false;
  int prefill_chunk = 16;
  std::string drafter = "mtp";  // when speculative
  std::vector<std::string> drafter_files{};
  std::string options{};  // backend options, after the tested backend's own
  std::string activations{};  // EngineOptions::activations ("": as the file has it)
};

// The engine; a load that fails throws, or with `failed` says why there.
std::unique_ptr<Engine> engine(const std::string& path, const std::string& backend, const Load& how = {},
                               std::string* failed = nullptr) {
  EngineOptions options;
  options.backend = backend;
  if (backend == tested) options.backend_options = tested_options;
  if (!how.options.empty()) options.backend_options += (options.backend_options.empty() ? "" : ",") + how.options;
  options.max_ctx = 256;
  options.prefill_chunk = how.prefill_chunk;
  options.lookup = false;
  options.width = !how.speculative ? 0 : how.drafter == "mtp" ? 3 : 15;  // the reference's rounds
  options.speculative = how.speculative;
  options.drafter = how.speculative ? how.drafter : "none";
  options.drafter_files = how.drafter_files;
  options.activations = how.activations;
  options.kv = FLYX_KV_16;  // the golden tokens and drafts are binary16 K/V's
  auto result = std::make_unique<Engine>();
  std::string error;
  if (!result->load(path, options, &error)) {
    if (failed) return *failed = error, nullptr;
    throw std::runtime_error(error);
  }
  return result;
}

// Why a load is refused ("" when it is not).
std::string refusal(const std::string& path, const std::string& backend, const Load& how) {
  std::string why;
  if (engine(path, backend, how, &why)) return "";
  return why.empty() ? "refused, no reason given" : why;
}

// The inputs a file's NVFP4 matrices take here, for the float64 backend to round its the same way: as the file has
// it (""), or, on a GPU without FP4 tensor cores (before sm_120), where the engine gives them bf16 inputs by itself,
// bf16 ones ("exact", --activations exact). The file loads either way.
std::string fp4_inputs_here(const std::string& path, const std::string& backend, const std::string& what) {
  std::string why;
  const auto e = engine(path, backend, {.prefill_chunk = 64}, &why);
  check(e != nullptr, what + ": loads on this device" + (why.empty() ? "" : " (" + why + ")"));
  return e && e->device().fp4_inputs ? "" : "exact";
}

std::vector<int32_t> generate_tokens(Session& session, const std::vector<int32_t>& prompt, int n, bool sampling = false) {
  GenerateRequest request;
  request.prompt = prompt;
  request.max_new_tokens = n;
  request.ignore_eos = true;
  request.checkpoints = {int32_t(prompt.size())};
  if (sampling) request.sampling = {0.7f, 0.9f, 20, 123, 0.5f, 0.1f};
  std::string error;
  GenerateResult result;
  std::vector<int32_t> tokens;
  if (generate(session, request,
               [&](std::string_view, const int32_t* ids, int count) {
                 tokens.insert(tokens.end(), ids, ids + count);
                 return true;
               },
               &result, &error) != 0)
    throw std::runtime_error(error);
  return tokens;
}

// A fresh session of the engine on this prompt.
std::vector<int32_t> tokens_of(Engine& e, const std::vector<int32_t>& prompt, int n, bool sampling = false) {
  Session s(&e);
  std::string error;
  if (!s.init(&error)) throw std::runtime_error(error);
  return generate_tokens(s, prompt, n, sampling);
}

// The same, greedy, with every result the device gave (the prefill's, then a round's each).
std::vector<flyx_round_result> traced(Engine& e, const std::vector<int32_t>& prompt, int n,
                                      std::vector<int32_t>* tokens) {
  Session s(&e);
  std::string error;
  if (!s.init(&error)) throw std::runtime_error(error);
  std::vector<flyx_round_result> trace;
  s.trace = &trace;
  *tokens = generate_tokens(s, prompt, n);
  s.trace = nullptr;
  return trace;
}

// Teacher forced: at how many of the reference's tokens the engine, given the reference's tokens before it, picks
// the same one.
int forced_agreement(Engine& e, const std::vector<int32_t>& prompt, const std::vector<int32_t>& want) {
  int agree = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    std::vector<int32_t> p = prompt;
    p.insert(p.end(), want.begin(), want.begin() + std::ptrdiff_t(i));
    agree += tokens_of(e, p, 1)[0] == want[i];
  }
  return agree;
}

void golden(const std::string& dir, const std::string& backend) {
  auto canonical = engine(dir + "/tiny.flyx", backend);
  auto plain = engine(dir + "/tiny_gguf.gguf", backend);
  auto speculative = engine(dir + "/tiny_gguf.gguf", backend, {.speculative = true});
  std::ifstream f(dir + "/golden.json");
  const auto golden = json::parse(f);
  for (const auto& item : golden.at("cases")) {
    const auto prompt = item.at("prompt").get<std::vector<int32_t>>();
    const auto expected = item.at("tokens").get<std::vector<int32_t>>();
    Session a(canonical.get()), b(plain.get()), c(speculative.get());
    std::string error;
    if (!a.init(&error) || !b.init(&error) || !c.init(&error)) throw std::runtime_error(error);
    check(generate_tokens(b, prompt, 48) == expected, "GGUF plain output equals float64 golden");
    check(generate_tokens(c, prompt, 48) == expected, "GGUF MTP output equals float64 golden");
    check(generate_tokens(a, prompt, 32, true) == generate_tokens(b, prompt, 32, true),
          "sampling with one seed equals the .flyx model");
    check(generate_tokens(b, prompt, 32, true) == generate_tokens(c, prompt, 32, true),
          "MTP preserves seeded sampling and penalties");
    auto continued = prompt;
    continued.insert(continued.end(), expected.begin(), expected.begin() + 24);
    check(generate_tokens(b, continued, 24) == std::vector<int32_t>(expected.begin() + 24, expected.end()),
          "GGUF session continuation/rewind");
    auto edit = prompt;
    edit.back() = (edit.back() + 1) % 300;
    check(generate_tokens(a, edit, 24) == generate_tokens(b, edit, 24), "GGUF edited prompt equals the .flyx model");
  }
}

void explicit_head_dim(const std::string& dir, const std::string& backend) {
  auto native = engine(dir + "/tiny_explicit_hd.flyx", backend);
  auto imported = engine(dir + "/tiny_explicit_hd.gguf", backend);
  auto spec = engine(dir + "/tiny_explicit_hd.gguf", backend, {.speculative = true});
  const std::vector<int32_t> prompt = {20, 41, 89, 100};
  for (bool sampling : {false, true}) {
    const auto expected = tokens_of(*native, prompt, 32, sampling);
    check(tokens_of(*imported, prompt, 32, sampling) == expected,
          "explicit head dimension: GGUF equals native with hidden % heads != 0");
    check(tokens_of(*spec, prompt, 32, sampling) == expected, "explicit head dimension: MTP preserves greedy/sampled output");
  }
}

// Packed block weights against the float64 backend on the same file.
void packed(const std::string& dir, const std::string& backend) {
  std::vector<int32_t> prompt;
  for (int i = 0; i < 80; ++i) prompt.push_back(i + 20);
  for (const char* kind : {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
                           "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_NL", "IQ4_XS",
                           "NVFP4"}) {
    const std::string path = dir + "/quant_" + kind + ".gguf";
    // Every pass takes its inputs in 8 bits, the reference's too. A group of 32 values is
    // scaled by its largest, so one bf16 step upstream (the float32 norm against the float64 one) can move a whole
    // group: where the reference's two best tokens nearly tie the backend may pick the other one, and from there
    // the replies part (Q5_K's, IQ3_XXS's and IQ4_XS's tiny models do, once each).  So the reply is compared
    // teacher forced: given the reference's tokens so far, the backend picks the reference's next one at all but at
    // most two of 32 places, through short prefill passes (the rounds' kernels) and long ones.
    const std::string act = std::string(kind) == "NVFP4" ? fp4_inputs_here(path, backend, kind) : "";
    const auto want = tokens_of(*engine(path, "cpu_ref", {.activations = act}), prompt, 32);
    for (int chunk : {16, 64}) {
      const int agree =
          forced_agreement(*engine(path, backend, {.prefill_chunk = chunk, .activations = act}), prompt, want);
      check(agree >= 30, std::string(kind) + ": teacher forced, the float64 backend's tokens at " + std::to_string(agree) +
                             " of 32 places, prefill in passes of up to " + std::to_string(chunk) + " rows");
    }
  }
  // The tiny Qwen3.5 quantized: gated delta nets, gated attention, a tied head, the MTP layer, mixed types in a stack.
  const std::string q8 = dir + "/tiny_q8.gguf";
  auto ref = engine(q8, "cpu_ref"), ref_long = engine(q8, "cpu_ref", {.prefill_chunk = 64});
  auto plain = engine(q8, backend), plain_long = engine(q8, backend, {.prefill_chunk = 64});
  auto spec = engine(q8, backend, {.speculative = true});
  std::ifstream f(dir + "/golden.json");
  const json golden = json::parse(f);  // named: a loop over json::parse(f).at(...) reads a destroyed temporary
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto want = tokens_of(*ref, p, 48);
    check(same_tokens(tokens_of(*plain, p, 48), want), "quantized Qwen3.5: the float64 backend's tokens");
    check(same_tokens(tokens_of(*spec, p, 48), want), "quantized Qwen3.5: the same with the MTP drafter");
    check(tokens_of(*plain, p, 32, true) == tokens_of(*spec, p, 32, true), "quantized Qwen3.5: MTP preserves seeded sampling");
    auto longer = p;
    for (int i = 0; i < 70; ++i) longer.push_back(want[size_t(i) % want.size()]);
    check(tokens_of(*plain, longer, 24) == tokens_of(*ref, longer, 24),
          "quantized Qwen3.5: a long prompt, prefill in passes of up to 16 rows");
    check(tokens_of(*plain_long, longer, 24) == tokens_of(*ref_long, longer, 24),
          "quantized Qwen3.5: a long prompt, prefill in passes of up to 64 rows (8-bit inputs)");
  }
  // The tiny Qwen3.5 with every matrix in GGML's NVFP4: repacked into the NVFP4 planes, gated
  // attention's q | gate rows reordered as they go, the MTP layer's too.
  const std::string nv = dir + "/tiny_nvfp4.gguf";
  const std::string act = fp4_inputs_here(nv, backend, "GGML NVFP4 Qwen3.5");
  auto nref = engine(nv, "cpu_ref", {.activations = act});
  auto nref_long = engine(nv, "cpu_ref", {.prefill_chunk = 64, .activations = act});
  auto nplain = engine(nv, backend, {.activations = act});
  auto nplain_long = engine(nv, backend, {.prefill_chunk = 64, .activations = act});
  auto nspec = engine(nv, backend, {.speculative = true, .activations = act});
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto want = tokens_of(*nref, p, 48);
    check(same_tokens(tokens_of(*nplain, p, 48), want), "GGML NVFP4 Qwen3.5: the float64 backend's tokens");
    check(same_tokens(tokens_of(*nspec, p, 48), want), "GGML NVFP4 Qwen3.5: the same with the MTP drafter");
    auto longer = p;
    for (int i = 0; i < 70; ++i) longer.push_back(want[size_t(i) % want.size()]);
    check(tokens_of(*nplain_long, longer, 24) == tokens_of(*nref_long, longer, 24),
          "GGML NVFP4 Qwen3.5: a long prompt, prefill in passes of up to 64 rows");
  }
}

// The MTP layer in a file of its own (EngineOptions::drafter_files; unsloth ships the 27B's so): the quantized tiny
// Qwen3.5 from the two files gives what the one file gives, every round's drafts included; the model without the
// file has no MTP layer, and a file whose model has one takes no second.
void split_mtp(const std::string& dir, const std::string& backend) {
  const std::string whole = dir + "/tiny_q8.gguf", target = dir + "/tiny_q8_target.gguf", mtp = dir + "/tiny_q8_mtp.gguf";
  auto one = engine(whole, backend, {.speculative = true});
  auto two = engine(target, backend, {.speculative = true, .drafter_files = {mtp}});
  auto rounds_of = [](Engine& e, const std::vector<int32_t>& prompt, std::vector<int32_t>* tokens) {
    std::vector<std::vector<int32_t>> drafts;
    for (const auto& r : traced(e, prompt, 40, tokens)) drafts.emplace_back(r.drafts, r.drafts + r.n_drafts);
    return drafts;
  };
  std::ifstream f(dir + "/golden.json");
  const json golden = json::parse(f);
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    std::vector<int32_t> ta, tb;
    const auto da = rounds_of(*one, p, &ta), db = rounds_of(*two, p, &tb);
    size_t drafted = 0;
    for (const auto& d : da) drafted += d.size();
    check(ta == tb && da == db && drafted > 0, "the MTP layer from a file of its own: the tokens and every round's drafts "
                                               "of the one file (" + std::to_string(drafted) + " drafts)");
  }
  std::string why = refusal(target, backend, {.speculative = true});
  check(why.find("no MTP layer") != std::string::npos, "the model without its MTP file has no MTP layer (" + why + ")");
  why = refusal(whole, backend, {.speculative = true, .drafter_files = {mtp}});
  check(why.find("has its MTP layer") != std::string::npos, "a model with its MTP layer takes no drafter file (" + why + ")");
}

// A DFlash2 drafter in a file of its own (z-lab publishes them so), beside a model of either format.
// * tiny.flyx's drafter alone (tiny_draft.flyx, converted from z-lab's layout) beside the tiny Qwen3.5 as GGUF
//   (float32, no drafter): the golden tokens, and on the float64 backend the reference's drafts, round by round;
//   with the GGUF's MTP layer beside it, either drafter as asked.
// * The quantized GGUF without its MTP layer (a packed embedding and head, which the drafter reads): plain
//   decoding's tokens, greedy and sampled, with the drafter and with its copy that attends a sliding window of 8;
//   on another backend, from exact weights and binary16 inputs, nearly all of the float64 backend's drafts on the
//   same files.  (With 8-bit inputs, a group of 32 scaled by its largest, one float32 step can move the target's
//   features enough to turn a near tie among the drafts, and the walk through the block carries a turned draft
//   on to every later one.)
// * A drafter file is no model; a drafter made for another model is refused; a model takes one block drafter.
void dflash_file(const std::string& dir, const std::string& backend) {
  const std::string drafter = dir + "/tiny_draft.flyx", window = dir + "/tiny_draft_window.flyx";
  std::ifstream f(dir + "/golden.json");
  const json golden = json::parse(f);
  // The drafter file, and the directory z-lab publishes it as, read in place.
  for (const std::string& source : {drafter, dir + "/hf_draft_zlab"}) {
    std::printf("  the drafter from %s\n", source.substr(dir.size() + 1).c_str());
    auto dflash = engine(dir + "/tiny_gguf.gguf", backend,
                         {.speculative = true, .drafter = "dflash2", .drafter_files = {source}});
    auto mtp = engine(dir + "/tiny_gguf.gguf", backend, {.speculative = true, .drafter_files = {drafter}});
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>(), want = item.at("tokens").get<std::vector<int32_t>>();
      std::vector<int32_t> got;
      const auto trace = traced(*dflash, p, 48, &got);
      int same = 0, total = 0;
      const json& rounds = item.at("rounds");
      for (size_t i = 1; i < trace.size() && i - 1 < rounds.size(); ++i) {
        const auto d = rounds[i - 1].at("drafts").get<std::vector<int32_t>>();
        for (int j = 0; j < trace[i].n_drafts && j < int(d.size()); ++j, ++total) same += trace[i].drafts[j] == d[size_t(j)];
      }
      check(same_tokens(got, want) && total > 0 && (backend != "cpu_ref" || same == total),
            "DFlash2 from a drafter file beside a GGUF model: the golden tokens; drafts equal to the reference's " +
                std::to_string(same) + " / " + std::to_string(total));
      check(same_tokens(tokens_of(*mtp, p, 48), want), "  the GGUF's MTP layer beside it, when asked for");
    }
  }
  const std::string q8 = dir + "/tiny_q8_target.gguf";
  auto plain = engine(q8, backend);
  for (const std::string& file : {drafter, window, dir + "/hf_draft_zlab", dir + "/hf_draft_window"}) {
    const std::string which = std::string(file.find("window") != std::string::npos ? "a DFlash2 drafter with a sliding window"
                                                                                  : "a DFlash2 drafter") +
                              (file.find("/hf_") != std::string::npos ? " (its directory)" : "");
    const Load drafted = {.speculative = true, .drafter = "dflash2", .drafter_files = {file}};
    Load in16 = drafted, exact = drafted;
    in16.activations = exact.activations = "exact";
    exact.options = "draft_weights=exact";
    auto spec = engine(q8, backend, drafted);
    auto ref = backend == "cpu_ref" ? nullptr : engine(q8, "cpu_ref", in16);
    auto spec_exact = backend == "cpu_ref" ? nullptr : engine(q8, backend, exact);
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      std::vector<int32_t> got;
      int accepted = 0;
      for (const auto& r : traced(*spec, p, 48, &got)) accepted += r.n_accepted;
      check(same_tokens(got, tokens_of(*plain, p, 48)),
            which + " beside the quantized GGUF: plain decoding's tokens (" + std::to_string(accepted) +
                " drafts accepted)");
      check(tokens_of(*plain, p, 32, true) == tokens_of(*spec, p, 32, true), "  and seeded sampling's");
      if (!ref) continue;
      std::vector<int32_t> ta, tb;
      const auto a = traced(*spec_exact, p, 48, &ta), b = traced(*ref, p, 48, &tb);
      int same = 0, total = 0;
      for (size_t i = 1; i < a.size() && i < b.size(); ++i)
        for (int j = 0; j < a[i].n_drafts && j < b[i].n_drafts; ++j, ++total) same += a[i].drafts[j] == b[i].drafts[j];
      check(ta == tb && total > 0 && same * 10 >= total * 9,
            "  from exact weights and binary16 inputs, the float64 backend's drafts on the same files: " +
                std::to_string(same) + " / " + std::to_string(total));
    }
  }
  {  // The drafters a model holds unless one is named (models/model.h: choose_drafters).
    auto own = engine(dir + "/tiny.flyx", backend, {.speculative = true, .drafter = ""});
    auto all = engine(dir + "/tiny.flyx", backend, {.speculative = true, .drafter = "all"});
    auto beside = engine(dir + "/tiny_gguf.gguf", backend, {.speculative = true, .drafter = "", .drafter_files = {drafter}});
    const std::string a = own->drafter(), b = all->drafter(), c = beside->drafter();
    check(a == "mtp" && b == "mtp+dflash2" && c == "mtp+dflash2",
          "the drafters held unless named: the model file's MTP layer of its two (" + a + "), all (" + b +
              "), and with a drafter file beside it, its drafter too (" + c + ")");
  }
  std::string why = refusal(drafter, backend, {});
  check(why.find("a drafter alone, not a model") != std::string::npos, "a drafter file alone is no model (" + why + ")");
  why = refusal(dir + "/tiny_w4a4.flyx", backend, {.speculative = true, .drafter = "dflash2", .drafter_files = {drafter}});
  check(why.find("another model") != std::string::npos,
        "a drafter made for another model (twice as wide) is refused (" + why + ")");
  why = refusal(dir + "/tiny_gguf.gguf", backend,
                {.speculative = true, .drafter = "dflash2", .drafter_files = {drafter, window}});
  check(why.find("a block drafter already") != std::string::npos, "a model takes one block drafter (" + why + ")");
}

// NVFP4 weights with NVFP4 inputs in passes of more than 16 rows against the float64 backend, which rounds the same
// inputs the same way in passes of the same rows: so the reference prefills in the same passes as the backend.  A
// pass of 64 rows rounds its inputs to 4 bits, and one step upstream (a float32 norm against a float64 one) can move
// a value across a rounding boundary: where the reference's two best tokens nearly tie the backend may pick the other,
// so the reply is compared teacher forced, as the packed types are.
void nvfp4_inputs(const std::string& dir, const std::string& backend) {
  const std::string path = dir + "/tiny_w4a4.flyx";
  if (!fp4_inputs_here(path, backend, "NVFP4 inputs").empty()) {  // what this checks needs FP4 tensor cores
    // Asked for by name where the device cannot compute them: refused at open, not bf16 inputs instead.
    const std::string why = refusal(dir + "/tiny_w4a4_a16.flyx", backend, {.activations = "nvfp4"});
    check(why.find("cannot compute") != std::string::npos,
          "NVFP4 inputs asked for without FP4 tensor cores: refused (" + why + ")");
    return;
  }
  std::vector<int32_t> prompt;
  for (int i = 0; i < 80; ++i) prompt.push_back((7 * i + 20) % 700);
  for (int chunk : {16, 64}) {
    const auto want = tokens_of(*engine(path, "cpu_ref", {.prefill_chunk = chunk}), prompt, 32);
    const int agree = forced_agreement(*engine(path, backend, {.prefill_chunk = chunk}), prompt, want);
    check(agree >= 30, "NVFP4 inputs: teacher forced, the float64 backend's tokens at " + std::to_string(agree) +
                           " of 32 places, prefill in passes of up to " + std::to_string(chunk) + " rows");
  }
  auto plain = engine(path, backend, {.prefill_chunk = 64});
  auto spec = engine(path, backend, {.speculative = true, .prefill_chunk = 64});
  check(tokens_of(*plain, prompt, 32) == tokens_of(*spec, prompt, 32), "NVFP4 inputs: the MTP drafter changes nothing");
  // Asked for by name (--activations nvfp4) in a file of the same weights that says bf16 inputs: the tokens of the
  // file that says NVFP4, greedy and sampled; by default, that file's with --activations exact.
  const std::string a16 = dir + "/tiny_w4a4_a16.flyx";
  auto asked = engine(a16, backend, {.prefill_chunk = 64, .activations = "nvfp4"});
  auto as_said = engine(a16, backend, {.prefill_chunk = 64});
  auto exact = engine(path, backend, {.prefill_chunk = 64, .activations = "exact"});
  for (bool sampling : {false, true}) {
    const std::string how = sampling ? "sampled" : "greedy";
    check(tokens_of(*asked, prompt, 32, sampling) == tokens_of(*plain, prompt, 32, sampling),
          "NVFP4 inputs asked for in a file that says bf16: the tokens of the file that says NVFP4 (" + how + ")");
    check(tokens_of(*as_said, prompt, 32, sampling) == tokens_of(*exact, prompt, 32, sampling),
          "  not asked for: that file's with --activations exact (" + how + ")");
  }
  // Asked for where this backend cannot compute them, refused: 64-wide NVFP4 weights (the W4A4 kernels' tile of K is
  // 128). A model without NVFP4 weights gives what it gives without the word.
  const std::string why = refusal(dir + "/tiny_nvfp4.flyx", backend, {.activations = "nvfp4"});
  check(why.find("a multiple of 128") != std::string::npos,
        "NVFP4 inputs asked for with K = 64: refused (" + why + ")");
  const std::vector<int32_t> short_prompt(prompt.begin(), prompt.begin() + 40);
  check(tokens_of(*engine(dir + "/tiny.flyx", backend, {.prefill_chunk = 64, .activations = "nvfp4"}), short_prompt,
                  16) == tokens_of(*engine(dir + "/tiny.flyx", backend, {.prefill_chunk = 64}), short_prompt, 16),
        "NVFP4 inputs asked for without NVFP4 weights: the tokens without the word");
}
void split_shards(const std::string& dir, const std::string& backend) {
  auto single = engine(dir + "/tiny_gguf.gguf", backend);
  auto split_plain = engine(dir + "/tiny_split-00001-of-00003.gguf", backend);
  auto split_spec = engine(dir + "/tiny_split-00001-of-00003.gguf", backend, {.speculative = true});
  std::ifstream f(dir + "/golden.json");
  const auto g = json::parse(f);
  const auto prompt = g.at("cases").at(0).at("prompt").get<std::vector<int32_t>>();
  const auto want = tokens_of(*single, prompt, 24);
  check(same_tokens(tokens_of(*split_plain, prompt, 24), want),
        "3-shard split GGUF gives the single GGUF's tokens (plain)");
  check(same_tokens(tokens_of(*split_spec, prompt, 24), want),
        "3-shard split GGUF gives the single GGUF's tokens (MTP)");
  auto qsingle = engine(dir + "/tiny_q8.gguf", backend);
  auto qsplit = engine(dir + "/tiny_q8_split-00001-of-00002.gguf", backend);
  check(same_tokens(tokens_of(*qsplit, prompt, 24), tokens_of(*qsingle, prompt, 24)),
        "2-shard quantized split GGUF gives the single quantized GGUF's tokens");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  const std::string dir = argv[1], backend = argv[2];
  tested = backend;
  if (argc > 3) tested_options = argv[3];
  try {
    golden(dir, backend);
    split_shards(dir, backend);
    explicit_head_dim(dir, backend);
    split_mtp(dir, backend);
    dflash_file(dir, backend);
    if (backend != "cpu_ref") packed(dir, backend);
    if (backend != "cpu_ref") nvfp4_inputs(dir, backend);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return failures ? 1 : 0;
}
