// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/common/engine_flags.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "apps/common/defaults.h"
#include "apps/common/numbers.h"
#include "core/interfaces/options.h"

namespace flyx::apps {

void SamplingOverrides::apply(SamplingParams* sp) const {
  if (temperature) sp->temperature = *temperature;
  if (top_p) sp->top_p = *top_p;
  if (top_k) sp->top_k = *top_k;
  if (seed) sp->seed = *seed;
  if (presence_penalty) sp->presence_penalty = *presence_penalty;
  if (frequency_penalty) sp->frequency_penalty = *frequency_penalty;
  if (accept_min_p) sp->accept_min_p = *accept_min_p;
  if (accept_typical) sp->accept_typical = *accept_typical;
}

bool sampling_flag(const std::string& a, const std::function<std::string()>& value, SamplingOverrides* out,
                   bool cli_only) {
  if (a == "--temperature") out->temperature = float(number_flag("--temperature", value(), 0, 100));
  else if (a == "--top-p") out->top_p = float(number_flag("--top-p", value(), 0, 1));
  else if (a == "--top-k") out->top_k = whole_flag("--top-k", value(), 0, INT_MAX);
  else if (a == "--presence-penalty") out->presence_penalty = float(number_flag("--presence-penalty", value(), -2, 2));
  else if (a == "--frequency-penalty")
    out->frequency_penalty = float(number_flag("--frequency-penalty", value(), -2, 2));
  else if (cli_only && a == "--seed") out->seed = seed_flag("--seed", value());
  else if (cli_only && a == "--accept-min-p") out->accept_min_p = float(number_flag("--accept-min-p", value(), 0, 1));
  else if (cli_only && a == "--accept-typical")
    out->accept_typical = float(number_flag("--accept-typical", value(), 0, 1));
  else return false;
  return true;
}

bool engine_flag(const std::string& a, const std::function<std::string()>& value, EngineOptions* eo) {
  if (a == "--ctx") {
    const std::string v = value();
    eo->ctx_max = v == "max";
    if (!eo->ctx_max) eo->max_ctx = whole_flag("--ctx", v, 1, kMaxCtx);
    eo->ctx_given = true;
  } else if (a == "--prefill-chunk") {
    eo->prefill_chunk = whole_flag("--prefill-chunk", value(), 1, INT_MAX);
  } else if (a == "--kv") {
    const std::string v = value();
    if (v != "fp8" && v != "16") std::fprintf(stderr, "error: --kv fp8 or --kv 16\n"), std::exit(2);
    eo->kv = v == "fp8" ? FLYX_KV_FP8 : FLYX_KV_16;
  } else if (a == "--kv-gpu") {
    eo->kv_gpu = whole_flag("--kv-gpu", value(), 0, INT_MAX);
  } else if (a == "--kv-copy-gpu") {
    eo->kv_copy_gpu = whole_flag("--kv-copy-gpu", value(), 0, INT_MAX);
  } else if (a == "--kv-ram") {
    const std::string v = value();
    eo->kv_ram = v == "auto" ? -1 : int64_t(number_flag("--kv-ram", v, 0, kMaxKvRamGb) * 1e9);
  } else if (a == "--kv-method") {
    const std::string v = value();
    interfaces::KvMethodSpec spec;
    std::string err;
    if (!interfaces::parse_kv_method(v, &spec, &err))
      std::fprintf(stderr, "error: %s\n", err.c_str()), std::exit(2);
    eo->kv_method = v;
  } else if (a == "--backend-opt") {
    eo->backend_options += (eo->backend_options.empty() ? "" : ",") + value();
  } else if (a == "--backend") {
    eo->backend = value();
  } else if (a == "--no-spec") {
    eo->speculative = false;
  } else if (a == "--no-lookup") {
    eo->lookup = false;
    eo->lookup_given = true;
  } else if (a == "--draft-trees") {
    eo->draft_trees = true;
    eo->draft_trees_given = true;
  } else if (a == "--width") {
    eo->width = whole_flag("--width", value(), 0, kMaxDraftWidth);
  } else if (a == "--cycle") {
    const std::string v = value();
    eo->cycle.clear();
    for (size_t at = 0; at <= v.size();) {
      const size_t comma = std::min(v.find(',', at), v.size());
      eo->cycle.push_back(size_t(whole_flag("--cycle", v.substr(at, comma - at), 0, INT_MAX)));
      at = comma + 1;
    }
  } else if (a == "--drafter") {
    const std::string v = value();
    interfaces::DrafterMode mode;
    if (!interfaces::parse_drafter_mode(v, &mode))
      std::fprintf(stderr, "error: unknown drafter '%s' (mtp, dflash2, dspine, dv, all or none)\n", v.c_str()),
          std::exit(2);
    eo->drafter = v;
  } else if (a == "--drafter-file") {
    eo->drafter_files.push_back(value());
  } else if (a == "--mmproj") {
    eo->mmproj_file = value();
  } else if (a == "--no-vision") {
    eo->vision = false;
  } else if (a == "--vision-offload") {
    eo->vision_device = "disk";
  } else if (a == "--vision-device") {
    const std::string v = value();
    std::string err;
    if (!interfaces::parse_vision_device(v, nullptr, &eo->vision_device, &err))
      std::fprintf(stderr, "error: %s\n", err.c_str()), std::exit(2);
  } else if (a == "--vision-quant") {
    const std::string v = value();
    std::string err;
    if (!interfaces::parse_vision_quant(v, nullptr, &eo->vision_quant, &err))
      std::fprintf(stderr, "error: %s\n", err.c_str()), std::exit(2);
  } else if (a == "--vision-max-tokens") {
    eo->vision_max_tokens = whole_flag("--vision-max-tokens", value(), 16, kMaxCtx);
  } else if (a == "--activations") {
    eo->activations = value();
  } else if (a == "--chat-template") {
    const std::string path = value();
    if (!read_text_file(path, &eo->chat_template))
      std::fprintf(stderr, "error: cannot read %s\n", path.c_str()), std::exit(2);
  } else {
    return false;
  }
  return true;
}

std::string engine_flags_help(int ctx) {
  const std::string ctx_line =
      ctx > 0
          ? "  --ctx N              context per session (default " + std::to_string(ctx) + ", or the most that fits in\n" +
                "                       GPU memory, at least " + std::to_string(kMinFittedCtx) + ")\n" +
                "  --ctx max            the model's own context (its file says), or the longest the GPU and host memory\n"
                "                       hold together when that does not fit\n"
          : "  --ctx N|max          context per session (default max: the model's own context, or the longest the\n"
            "                       GPU and host memory hold together when that does not fit)\n";
  return ctx_line +
         "  --prefill-chunk N    prompt rows per prefill pass (default 1024 to 4096 by free GPU memory; more is faster)\n"
         "  --kv fp8|16          the K/V cache: fp8 (e4m3, a scale per position and head: half the memory and\n"
         "                       attention reads) or 16 (binary16, llama.cpp's); default 16 where 32 GB of GPU\n"
         "                       memory are left after the weights or on Llama models such as MiniCPM5 (where fp8\n"
         "                       moves the next tokens visibly), else fp8\n"
         "  --kv-ram GB|auto     host memory for the K/V a context's GPU part does not hold: the GPU reads it over\n"
         "                       the bus, the same bits, slower past that part (default auto: what the machine can\n"
         "                       spare, serve's --host-cache getting what it leaves; 0: none, the context shrinks\n"
         "                       to fit the GPU)\n"
         "  --kv-gpu N           keep at most N tokens' K/V on the GPU, the rest in host memory (default: as many\n"
         "                       as fit)\n"
         "  --kv-copy-gpu N      with K/V in host memory: keep the strata copy of at most N layers on the GPU, the\n"
         "                       others' in host memory, each copied to the GPU while the layers before it run\n"
         "                       (default: as many as fit)\n"
         "  --kv-method M       how rounds read the K/V: auto[:BITS] (the default: every position while the K/V\n"
         "                       is on the GPU, then strata once it spills to host memory; BITS default 2),\n"
         "                       exact (every position, every step) or strata[:BITS] (a BITS-bit\n"
         "                       copy of every position on the GPU, the positions a row attends to most read again\n"
         "                       from the cache; BITS 2 to 4, default 2)\n"
         "  --backend B          cuda | cpu_ref (default: the first that loads); cuda runs on the first GPU CUDA\n"
         "                       lists: CUDA_VISIBLE_DEVICES=N in the environment makes that GPU N\n"
         "  --backend-opt K=V    a backend option (repeatable; also FLYX_BACKEND_OPTIONS=K=V,...)\n"
         "  --no-spec            plain decoding, ignore the drafter\n"
         "  --no-lookup          no rounds drafted from the text itself where it repeats\n"
         "  --draft-trees        the drafter also drafts trees (DFlash2: its candidates, best first), chosen per\n"
         "                       round by cost and acceptance like its chains (also FLYX_DRAFT_TREES=1)\n"
         "  --drafter D          mtp | dflash2 | dspine | dv | all | none; with several, the drafter is chosen every\n"
         "                       round\n"
         "                       (default: every drafter given in a file of its own, and the model file's MTP layer\n"
         "                       if it has one, else its DFlash2 drafter; its DV drafter, which drafts alone, only\n"
         "                       when named)\n"
         "  --drafter-file F     a drafter in a file of its own (repeatable): the MTP layer from a GGUF, for a GGUF\n"
         "                       without it (unsloth's MTP/mtp-*.gguf), or a DFlash2 drafter file (python/convert\n"
         "                       --draft alone), for a model of either format\n"
         "  --mmproj F           multimodal vision projector GGUF (mmproj-*.gguf) beside the model file\n"
         "  --no-vision          do not load a vision encoder even if a projector is available\n"
         "  --vision-offload     offload vision weights to an auto-deleted temp file on disk (0 VRAM, 0 locked RAM)\n"
         "  --vision-device D    disk | host | gpu | auto (default disk: 0 VRAM & 0 locked RAM, streamed via OS page\n"
         "                       cache; host keeps pinned RAM; auto uses GPU when >= 8 GB free after weights)\n"
         "  --vision-quant Q     q8 | exact | nvfp4 (default q8: Q8_0 linear weights, ~46% smaller; exact keeps\n"
         "                       file precision; nvfp4 ~70% smaller)\n"
         "  --vision-max-tokens N cap merged visual tokens per image (default: auto by context and detail)\n"
         "  --width N            fix the model drafter's width at N, up to its maximum (default: chosen per round)\n"
         "                       prompt lookup can still verify 15 drafts from earlier text (disable with --no-lookup)\n"
         "                       but not beside a DV drafter, whose rounds then run alone: its widest of at most N\n"
         "                       every round\n"
         "  --cycle I,J,...      measurements: launch these rounds in turn (indices into the rounds -v lists),\n"
         "                       no lookup round, instead of choosing\n"
         "  --activations A      exact: every linear rounds its inputs as its format does everywhere; nvfp4: NVFP4\n"
         "                       weights take NVFP4 inputs in prefill passes whatever the file says (a GPU with FP4\n"
         "                       tensor cores); default: as the file has it (GGML block weights take 8-bit inputs,\n"
         "                       NVFP4 weights NVFP4 ones in prefill passes on a GPU with FP4 tensor cores unless\n"
         "                       the file says bf16)\n"
         "  --chat-template F    a Jinja chat template file instead of the model's\n";
}

}  // namespace flyx::apps
