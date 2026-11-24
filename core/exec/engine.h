// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Engine: one loaded model (its file, backend, plans, speculation cost table
// and device memory account).  Sessions are core/session/session.h.
#pragma once

#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/grammar/grammar.h"
#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "core/plan/budget.h"
#include "core/plan/plans.h"
#include "core/os/os.h"
#include "core/render/chat.h"
#include "core/speculation/controller.h"
#include "core/tokenizer/tokenizer.h"
#include "core/resources/ledger.h"
#include "core/vision/image.h"
#include "flyx/backend_abi.h"
#include "flyx/flyx.h"

namespace flyx {

// The shortest context a context the caller did not ask for shrinks to when its session does not fit (llama.cpp's
// --fit stops at the same); shorter, the load fails and says what would fit.
inline constexpr int kMinFittedCtx = 4096;
// --ctx max for a model whose file names no context of its own.
inline constexpr int kCtxMaxUnnamed = 262144;
// A context memory cut to under this is called short where the load reports it (its log, the server's telemetry): a
// coding agent's own instructions and tools take half of it or more before the first question.
inline constexpr int kShortCtx = 32768;

struct EngineOptions {
  int max_ctx = 8192;
  // max_ctx was asked for (--ctx) and stays as it is.  Otherwise a context whose session does not fit in GPU memory
  // shrinks to the longest that does, down to kMinFittedCtx tokens.
  bool ctx_given = false;
  // --ctx max: the model's own context (Model::trained_ctx; kCtxMaxUnnamed when its file names none), or the longest
  // the GPU and host memory hold together when that does not fit (max_ctx is set at load).
  bool ctx_max = false;
  std::string backend;       // empty = auto
  std::string backend_dir;   // empty = next to the executable
  bool speculative = true;
  // "mtp" | "dflash2" | "dspine" | "dv" | "all" | "none" | empty = the drafter files' drafters and the model file's
  // MTP layer, else its block drafter (DFlash2 or DSpine; models/model.h: choose_drafters); its DV drafter only when
  // named ("dv", "all").  With
  // several, the drafter is chosen every round; DV is never beside another drafter, its rounds chosen
  // against plain decoding's and lookup's every round (docs/design/dv.md).
  std::string drafter;
  // Files with drafters' weights beside the model file: a GGUF holding only the MTP
  // layer (unsloth ships Qwen3.8-27B's so: MTP/mtp-*.gguf), a DFlash2 drafter file (python/convert --draft
  // alone).  Empty: none.
  std::vector<std::string> drafter_files;
  // Optional multimodal vision projector companion GGUF (mmproj-*.gguf). Empty: none.
  std::string mmproj_file;
  // Disable loading any vision encoder even if mmproj weights are present (--no-vision).
  bool vision = true;
  // Where vision encoder weights live: "disk" (default: auto-deleted temp file on disk, 0 bytes VRAM & 0 bytes
  // locked host RAM, 3-stage streamed via OS page cache), "host" (pinned host RAM, 0 VRAM), "gpu" (GPU resident),
  // or "auto" (GPU if >= 8 GB free after weights, else disk).
  std::string vision_device = "disk";
  // On-the-fly quantization of floating-point vision linear weights: "q8" (default: Q8_0, ~46% smaller),
  // "exact" (keep file BF16/F16/F32), or "nvfp4" (NVFP4, ~70% smaller). Pre-quantized mmproj stays packed.
  std::string vision_quant = "q8";
  // Optional cap on merged visual tokens per image (0 = auto by context size and detail level).
  int vision_max_tokens = 0;
  int verbose = 0;
  // Prompt rows per prefill pass.  Larger chunks keep the GEMMs busier; the
  // buffers for one chunk are shared by all sessions.
  // 0: by the device memory the weights leave (Engine::auto_prefill_chunk).
  int prefill_chunk = 0;
  // Target K/V cache: FLYX_KV_FP8 (e4m3, a scale per position and head: half the memory and the attention reads, the
  // next-token distributions as close to binary16's as the 8-bit inputs of the weights already make them),
  // FLYX_KV_16 (binary16, llama.cpp's), or FLYX_KV_DEFAULT: binary16 where the device has room for it (Engine::load:
  // 32 GB or more left after the weights), which is finer and faster to fill, or where the model asks for it
  // (interfaces::Model::kv_fp8_by_default: binary16 on Llama, whose next tokens FP8 moves visibly), else FP8.
  int kv = FLYX_KV_DEFAULT;
  // Where the K/V caches go: GPU memory first, then page-locked host memory the GPU reads over the
  // bus (the same bits; decoding past the GPU's part reads the rest over the bus every step).  kv_gpu: positions
  // kept on the GPU at most (-1: as many as fit).  kv_ram: bytes of host memory for the rest (-1: what the machine
  // can spare, its available memory less plan::host_keep(): a sixteenth of all of it, at least 2 GiB; 0: none, and
  // a context that does not fit the GPU shrinks as before).  kv_method: how attention reads the caches ("exact":
  // every position, every step; "strata[:BITS]": a low-bit copy, "auto[:BITS]": exact, then the copy).
  int kv_gpu = -1;
  int64_t kv_ram = -1;
  std::string kv_method = "auto";
  // With the K/V in host memory, the layers whose low-bit copy stays on the GPU at most; the others' copy is in host
  // memory too, each read through a staging a round fills a layer ahead. -1: as many as fit.
  int kv_copy_gpu = -1;
  // Host memory the process takes besides, which kv_ram's auto leaves aside.  (The server's store of parked
  // conversations, is not: it takes what the sessions' K/V leave, Scheduler.)
  uint64_t host_reserved = 0;
  // Drafts verified per round: -1 = chosen each round by the speculation
  // controller from the measured cost of each width and the acceptance seen
  // so far; >= 0 pins the width (tests, benchmarks; the drafter is still
  // chosen when there are several).
  int width = -1;
  // Measurements and tests: every reply launches these rounds in turn (indices into widths()), no lookup round,
  // instead of choosing (RoundChooser::cycle); empty: chosen.  The output is the same either way.
  std::vector<size_t> cycle;
  // Backend options, "key=value,..." (backends/<name> documents them), passed through to the backend.
  std::string backend_options;
  // How linears round their inputs: "" = as each weight's file has it (GGML block weights: 8-bit inputs in
  // prefill passes); "exact" = the formats' own rounding in every pass; "nvfp4" = NVFP4 weights
  // take NVFP4 inputs in prefill passes whatever the file says (a backend that cannot compute them refuses).
  std::string activations;
  // A Jinja chat template to use instead of the one in the model file (its source, not a path).
  std::string chat_template;
  // Render with the chat format's own renderer when the template has one;
  // false: the Jinja engine always (FLYX_CHAT_NATIVE=0).
  bool chat_native = true;
  bool chat_native_given = false;
  // Drafts looked up in the sequence itself when the text just written
  // appeared before, verified 15 at a time with no drafter run;
  // FLYX_LOOKUP=0 turns it off.  The output is the same either way.
  bool lookup = true;
  bool lookup_given = false;
  // Rounds whose drafter drafts a tree of drafts (FLYX_DRAFT_TREE; DFlash2's candidates, best first) besides its
  // chains, the controller choosing among all of them by measured cost and acceptance; FLYX_DRAFT_TREES=1 turns
  // it on.  The output is the same either way.
  bool draft_trees = false;
  bool draft_trees_given = false;
  // Intra-round confidence optimal stopping on drafters that support adaptive plans; FLYX_ADAPTIVE=0 turns it off.
  bool adaptive = true;
  bool adaptive_given = false;
};

// Fills the options the caller left unset from the environment: FLYX_BACKEND,
// FLYX_BACKEND_DIR, FLYX_DRAFTER, FLYX_ACTIVATIONS; FLYX_BACKEND_OPTIONS goes before the
// caller's backend options (a key given twice takes the later value);
// FLYX_CHAT_NATIVE=0 turns chat_native off, FLYX_LOOKUP=0 lookup, FLYX_DRAFT_TREES=1 turns draft_trees on.
// The only place the engine reads the environment.
void apply_environment(EngineOptions* o);

// The backend library `o` asks for (else the first of cuda, cpu_ref that loads) and the device it found.  *lib must
// stay loaded while *api is used.
bool open_backend(const EngineOptions& o, std::unique_ptr<os::DynLib>* lib, const flyx_backend_api** api,
                  flyx_device_info* info, std::string* err);
// The device a model would run on, before one is named (flyx serve: which files fit).  The library stays loaded for
// the process: an engine opening it later shares it.
bool probe_device(const EngineOptions& o, std::string* backend, flyx_device_info* info, std::string* err);

class Engine {
 public:
  ~Engine();
  bool load(const std::string& path, const EngineOptions& opt, std::string* err, int session_limit = 1);

  const Tokenizer& tokenizer() const { return tok_; }
  // The chat template (check ok(): a file without one still runs raw prompts).
  const ChatTemplate& chat() const { return chat_; }
  const ModelFile& model_file() const { return file_; }
  const flyx_backend_api* api() const { return api_; }
  const std::string& backend_name() const { return backend_name_; }
  const flyx_device_info& device() const { return device_; }
  // The device's live figures (load, temperature, power, memory); false when the
  // backend has none.  Any thread.
  bool device_stats(flyx_device_stats* out) const {
    return api_ && api_->device_stats && api_->device_stats(out) == 0;
  }
  const interfaces::Model& model() const { return *model_; }
  // The drafters in use ("mtp+dflash2", "none"): the model's, but one whose rounds the backend runs at no width (DV on
  // a backend without its kernels: refused() says why).
  const char* drafter() const { return drafter_.c_str(); }
  // Widths of a drafter the backend did not lower, each with why ("dv 31: ..."); empty: none.
  const std::string& refused() const { return plans_.refused; }
  const EngineOptions& options() const { return opt_; }
  int width() const { return plans_.widest(); }  // the widest round
  // The rounds to choose from: round i verifies widths()[i] drafts of drafter round_drafters()[i] (-1: none), as a
  // tree when round_trees()[i], and costs round_ms()[i] at a context of round_ms_at() tokens and
  // round_ms_per_token()[i] more for every token past it (from a second timing at round_ms_again_at() tokens where a
  // drafter's state follows its own rounds only; 0 otherwise, a round costing the same at any context); round_name(i)
  // says so ("mtp 3", "0", "dflash2 tree 11", "dv tree 31", "dv warm-up": a round of no drafts of a drafter,
  // Model::restarts).  round_trunks()[i]: a tree's trunk, the drafter's chain its first drafts are (its widest chain's
  // width); 0 for a chain; -1 for a tree without one (a drafter that drafts nothing but trees: SpecController).
  const std::vector<int>& widths() const { return plans_.widths; }
  const std::vector<int>& round_drafters() const { return plans_.drafters; }
  const std::vector<bool>& round_trees() const { return plans_.drafter_trees; }
  const std::vector<int>& round_trunks() const { return round_trunks_; }
  const std::vector<double>& round_ms() const { return round_ms_; }
  const std::vector<double>& round_ms_per_token() const { return round_ms_per_token_; }
  int round_ms_at() const { return round_ms_at_; }
  int round_ms_again_at() const { return round_ms_again_at_; }
  // What the trees without a trunk of this engine's sessions gave a round: the means every reply's estimate of them
  // starts from (a session's first reply's too); and the engine's credit for probing DV (SharedTokens, SpecController).
  SharedTokens* round_tokens() const { return &round_tokens_; }
  std::string round_name(size_t i) const;
  flyx_plan* round_plan(size_t i) const { return plans_.rounds[i]; }
  // Rounds whose drafts the engine gives (EngineOptions::lookup): no drafter runs.
  const std::vector<int>& lookup_widths() const { return plans_.given_widths; }
  const std::vector<double>& lookup_ms() const { return lookup_ms_; }
  flyx_plan* lookup_plan(size_t i) const { return plans_.given[i]; }
  // Rounds that verify a tree of given drafts (gen.parents), where the backend has them; empty: none.
  const std::vector<int>& tree_widths() const { return plans_.tree_widths; }
  flyx_plan* tree_plan(size_t i) const { return plans_.trees[i]; }
  // Optional intra-round adaptive stopping plan per drafter (null if unavailable).
  flyx_plan* adaptive_plan(int d) const {
    return (d >= 0 && size_t(d) < plans_.adaptive.size()) ? plans_.adaptive[size_t(d)] : nullptr;
  }
  const std::vector<std::vector<int>>& adaptive_checkpoints() const { return plans_.adaptive_checkpoints; }
  flyx_bmodel* bmodel() const { return bm_; }
  flyx_plan* prefill_plan() const { return plans_.prefill; }
  flyx_plan* vision_plan() const { return plans_.vision; }
  bool has_vision() const {
    return model_ && model_->vision() != nullptr && plans_.vision != nullptr && api_ && api_->encode_vision != nullptr;
  }
  // Compute per-image merged visual token budget from detail level, context size, and vision_max_tokens.
  static int image_token_budget_for(int max_ctx, int vision_max_tokens, const std::string& detail);
  int image_token_budget(const std::string& detail, size_t n_images) const;
  // Decode, preprocess, and encode one image into projected embeddings (cached by SHA-256 + detail + max_tokens).
  // When client_error is non-null, sets *client_error = true for malformed input / missing vision encoder and
  // false for internal backend failures.
  bool encode_image(const interfaces::ImageInput& img, int max_tokens, std::shared_ptr<const vision::CachedVision>* out,
                    std::string* err, bool* client_error = nullptr) const;
  // Prefill passes reproduce a longer pass bit for bit when they start at a multiple of this.
  int prefill_align() const { return prefill_align_; }
  // Rows per prefill pass: the option, as a multiple of prefill_align().
  int prefill_chunk() const { return prefill_chunk_; }
  int auto_prefill_chunk() const;  // opt_.prefill_chunk == 0: by the device memory left
  bool roomy() const;  // 32 GB of GPU memory free after the weights: binary16 K/V, GPU embeddings by default
  const std::vector<int32_t>& eos() const { return eos_; }
  // The model's vocabulary as grammars see it (core/grammar), built on first use.
  Grammar& grammar();

  // Device memory: weights and workspace are leased at load; every session
  // leases session_bytes() when it is created.
  ResourceLedger& ledger() { return ledger_; }
  uint64_t session_bytes() const { return budget_.session_bytes; }
  // Host memory a session's K/V take (0: none), and what the K/V may take in all.
  uint64_t session_host_bytes() const { return budget_.session_host_bytes; }
  uint64_t host_budget() const { return budget_.host_budget; }
  // Host memory the sessions' K/V hold mapped for the device only, which no process memory figure counts
  // (flyx_seq_size.host_mapped): monitors add it to the process's own.
  uint64_t host_mapped_in_use() const { return host_sessions_.load() * budget_.session_host_mapped; }
  // The load's memory plan: what holds the GPU besides the model, and, where memory cut the context under the one
  // wanted (--ctx max: the model's own; no --ctx: the default), which memory did.
  const plan::Budget& budget() const { return budget_; }
  int wanted_ctx() const { return wanted_ctx_; }
  bool ctx_short() const { return budget_.ctx_limit != plan::CtxLimit::kNone && opt_.max_ctx < kShortCtx; }
  // Where a session's K/V caches live: all on the GPU, or its first positions and the rest in host
  // memory.
  const flyx_seq_layout& kv_layout() const { return budget_.layout; }
  int sessions_that_fit() const;  // more sessions the budget allows now (large number if unknown)
  // Charge a session's host K/V before the backend starts allocating it.
  bool reserve_host_kv();
  void release_host_kv() { if (budget_.session_host_bytes) --host_sessions_; }
  std::string memory_report() const;

 private:
  bool load_backend(std::string* err);
  bool build_and_calibrate_plans(double* t_plans, double* t_rounds, std::string* err);
  // The plans lowered and their rounds timed (build_and_calibrate_plans), again after each step of what the backend
  // keeps only to run faster (a second copy of the head: the same output, slower; e.g. an 8 GB card beside a desktop)
  // is given back, where lowering finds no room for the prefill buffers; the time over every pass in lower_ms_ and
  // timed_ms_.
  bool lower_and_time(std::string* err);
  // One step of what the backend keeps for speed given back (flyx_backend_api::trim, counted in given_back_), the
  // plans freed to be lowered again; false when nothing is left to give.
  bool give_back();
  bool plan_memory(plan::KvTiers* tiers, int session_limit, int* asked_ctx, std::string* err);
  void log_memory_plan(int asked_ctx) const;

  EngineOptions opt_;
  ModelFile file_;
  Tokenizer tok_;
  ChatTemplate chat_;
  std::unique_ptr<os::DynLib> lib_;
  const flyx_backend_api* api_ = nullptr;
  std::string backend_name_;
  flyx_device_info device_{};
  std::unique_ptr<interfaces::Model> model_;          // the file as blocks, with the drafters in use
  flyx_bmodel* bm_ = nullptr;
  int prefill_align_ = 1, prefill_chunk_ = 1024;
  plan::Plans plans_;
  std::string drafter_ = "none";            // drafter(): the model's drafters whose rounds the plans have
  std::vector<double> round_ms_;            // measured cost of each round plan (core/speculation)
  std::vector<double> lookup_ms_;           // measured cost of each lookup plan (core/speculation)
  std::vector<double> round_ms_per_token_;  // ... more for every token of context past round_ms_at_
  mutable SharedTokens round_tokens_;
  int round_ms_at_ = 0, round_ms_again_at_ = 0;
  double lower_ms_ = 0, timed_ms_ = 0;      // the load's lowering and timing of the plans, over every pass
  uint64_t given_back_ = 0;                 // bytes the backend gave back at load (trim)
  std::vector<int> round_trunks_;
  std::vector<int32_t> eos_;
  Grammar grammar_;
  std::mutex grammar_mu_;
  mutable vision::VisionCache vision_cache_;
  ResourceLedger ledger_;
  plan::Budget budget_;                     // leases from ledger_
  int wanted_ctx_ = 0;                      // the context before the budget fitted it
  std::atomic<uint64_t> host_sessions_{0};  // includes allocations in progress
};

// Registers a backend linked directly into the executable (used by standalone single-file builds).
// Looked up before dynamic libraries unless --backend-dir / FLYX_BACKEND_DIR is explicitly set.
void register_builtin_backend(const char* name, flyx_backend_entry_fn entry);

}  // namespace flyx
