# CLI, HTTP Server, and APIs

Single reference for the `flyx` CLI, `flyx serve` HTTP endpoints, environment variables, `--backend-opt` keys, and the C / Python APIs. `flyx-server` (symlink in Linux releases) aliases `flyx serve`.

---

## 1. CLI Subcommands

| Command | Synopsis | Description |
|---|---|---|
| `serve` | `flyx serve [MODEL] [serve-opts] [engine-opts]` | Start OpenAI + Anthropic HTTP API server and dashboard. Without `MODEL`, restores the last served model (`~/.config/flyx/serve.json` or `%APPDATA%\flyx\serve.json`), or starts unloaded (HTTP 503 until loaded via `/v1/load` or dashboard). |
| `run` | `flyx run MODEL (-p TEXT \| -f FILE) [--image FILE]... [--chat] [--no-think] [-n N] [opts]` | Single-shot generation. Use `-f FILE` for prompts > 128 KB; `--image` attaches local images (implies `--chat`). |
| `chat` | `flyx chat MODEL [opts]` | Interactive multi-turn terminal chat. |
| `bench` | `flyx bench MODEL [-n N] [--prompts FILE.jsonl] [--json [FILE]] [opts]` | Benchmark decode `tok/s`, tokens/round, and GPU telemetry across 3 fixed prompts, or across a prompt set (`--prompts`). `--json` emits one JSON record on stdout, or into `FILE`. |
| `tokenize` | `flyx tokenize MODEL "TEXT"` | Print token IDs for `TEXT`. |
| `template` | `flyx template MODEL MESSAGES.json [--no-think]` | Render a JSON message array through the model's chat template. |
| `monitor` | `flyx monitor [URL] [--key KEY] [--interval S] [--once [--for S]]` | Read-only terminal monitor polling `GET /v1/telemetry` (default `http://127.0.0.1:8080`; `http://` URLs only). |
| `--version` | `flyx --version` | Print `flyx X.Y.Z` (also `server.version` in `GET /v1/telemetry`). `--help` after any command prints its usage on stdout with status 0; an option the command does not know is named in one line, status 2. |

---

## 2. Engine Flags (`serve`, `run`, `chat`, `bench`)

All commands that load a model share the parser in `apps/common/engine_flags.cpp`.

| Flag | Values | Default | Behavior |
|---|---|---|---|
| `--ctx` | `N` (`1..16777216`) \| `max` | `8192` (CLI) / `max` (`serve`) | Tokens per session. `serve` defaults to `max`: the model's trained context (or `262144` if unnamed), capped to what GPU + host RAM hold. CLI defaults to `8192` (shrinking down to `4096` if VRAM is tight). A context cut by memory is logged with which memory cut it and what would give more (other programs' share of the GPU, `--kv-ram`), as a warning under 32,768 tokens; weights and a session's fixed state stay on the GPU whatever host RAM there is, so a load can be refused with RAM to spare, and the refusal then says how much GPU memory is missing. |
| `--prefill-chunk` | `N` (≥ 1) | `1024` to `4096` by free GPU memory | Prompt rows per prefill pass. Buffers are shared across all sessions. |
| `--kv` | `fp8` \| `16` | `16` where 32 GB of GPU memory remain after the weights and on Llama models, else `fp8` | Target KV cache dtype: `fp8` (`e4m3` + per-position/head float32 scale, half memory/bandwidth) or `16` (FP16). Llama models (MiniCPM5-1B) default to `16`: FP8 moved their next-token distributions visibly. |
| `--kv-ram` | `GB` \| `auto` | `auto` | Pinned host RAM budget for KV positions beyond GPU capacity. `auto` takes available RAM minus `max(1/16 of total RAM, 2 GiB)` (32 GB machines keep 2 GiB, 64 GB 4 GiB; the remainder goes to `--host-cache`). `0` disables host KV offload. |
| `--kv-gpu` | `N` (≥ 0) | auto (max that fits) | Cap on KV positions kept on GPU; overflow goes to host RAM. |
| `--kv-copy-gpu` | `N` (≥ 0) | auto (max that fits) | With host KV, max layers keeping their StrataFlux low-bit summary copy on GPU; remaining layers keep copies in host RAM and stage ahead. |
| `--kv-method` | `auto[:BITS]` \| `exact` \| `strata[:BITS]` | `auto` (`BITS=2`) | Decode KV read mode. `BITS` in `2..4` (default `2`). `auto`: exact attention up to `kv_gpu`, then StrataFlux once KV spills to host RAM; `exact`: exact attention everywhere; `strata`: StrataFlux everywhere. A file with a DV drafter reads exactly under `auto`, with `--drafter dv` or without (its trees read every position, and naming it changes no token); `strata` with DV drafting is refused. |
| `--backend` | `cuda` \| `cpu_ref` | first available (`cuda`) | Backend shared library to load. Select GPU with `CUDA_VISIBLE_DEVICES`. |
| `--backend-opt` | `K=V` (repeatable) | none | Backend-specific option (also `FLYX_BACKEND_OPTIONS=K=V,...`). Unknown keys fail load. |
| `--drafter` | `mtp` \| `dflash2` \| `dspine` \| `dv` \| `all` \| `none` | auto | Active drafters. Default enables all `--drafter-file` drafters plus the model's embedded MTP (or embedded block drafter, DFlash2 or DSpine). Multiple active drafters are selected dynamically per round. A DV drafter (in a DV `.flyx` file, [DV](design/dv.md)) drafts only when named (`dv` or `all`): its warm-up, its trees of 15 and of the file's width, plain decoding and lookup rounds are then chosen every round. |
| `--drafter-file` | `PATH` \| `HF_REPO` (repeatable) | none | Sidecar MTP GGUF (`MTP/mtp-*.gguf`) or block drafter: DFlash2 (`.flyx`, HF repo directory, or a `general.architecture=dflash` GGUF) or DSpine (`.flyx` or HF repo directory); one block drafter a model. |
| `--mmproj` | `PATH` | auto (HF Hub) / none (local) | Multimodal vision projector GGUF (`mmproj-*.gguf`, e.g., `mmproj-BF16.gguf`) paired with the model file. Auto-fetched from Hugging Face Hub repos when available unless `--no-vision` is set. |
| `--no-vision` | flag | off | Disable loading any vision encoder / `mmproj` projector even if available. |
| `--vision-offload` | flag | on (`disk`) | Shorthand for `--vision-device disk`: offload vision weights to an auto-deleted temp file on disk (0 VRAM, 0 locked host RAM) and stream via OS page cache into GPU staging. |
| `--vision-device` | `disk` \| `host` \| `gpu` \| `auto` | `disk` | Where vision encoder weights live (`disk`: auto-deleted temp file, 0 VRAM & 0 locked host RAM so host KV budget is untouched, 3-stage streamed with bit-identical outputs; `host`: pinned host RAM, 0 VRAM; `gpu`: resident on GPU; `auto`: GPU when >= 8 GB VRAM remain free after weights, else `disk`). |
| `--vision-quant` | `q8` \| `exact` \| `nvfp4` | `q8` | On-the-fly quantization of floating-point vision linear weights (`q8`: Q8_0 ~46% smaller; `exact`: keep file precision; `nvfp4`: ~70% smaller; pre-quantized GGML `mmproj` files stay packed). |
| `--vision-max-tokens` | `N` (`16..max_ctx`) | auto | Cap merged visual tokens per image across CLI, OpenAI, Responses, and Anthropic endpoints. |
| `--width` | `N` (`0..64`) | adaptive | Pin model drafter width (up to drafter max). Prompt lookup still runs unless `--no-lookup`, but not beside a DV drafter: its widest tree of at most `N` drafts then runs every round. |
| `--cycle` | `I,J,...` | off | Measurements and tests: launch these rounds in turn (indices into the rounds `-v` lists), no lookup round, instead of the controller's choice. An index past the rounds fails the load. |
| `--no-spec` | flag | off | Disable all speculative decoding (drafters + lookup). |
| `--no-lookup` | flag | off | Disable prompt-lookup rounds (15 drafts, wider with the `wide` backend option) on repeated text. |
| `--draft-trees` | flag | off | Enable DFlash2 best-first draft trees alongside chains (also `FLYX_DRAFT_TREES=1`). |
| `--activations` | `exact` \| `nvfp4` | format default | `exact` forces per-format exact activation rounding (`bf16`/`fp16`) everywhere; `nvfp4` forces NVFP4 prefill inputs on NVFP4 weights even if the file declares bf16 (requires `sm_120a` and `K % 128 == 0`). |
| `--chat-template` | `FILE` | model's template | Override model Jinja chat template from file. |
| `-v` | flag | off | Verbose logging (memory budget bill, load timings, per-request metrics, each CUDA round graph a session keeps and the size it is counted at, and as a session ends its rounds from a graph and without one). |

### Sampling & CLI-Specific Flags

| Flag | Commands | Values / Range | Default | Behavior |
|---|---|---|---|---|
| `--temperature` | all | `0..100` | `0` (CLI) / model card (`serve`) | Sampling temperature (`0` = greedy). |
| `--top-p` | all | `0..1` | `0` (CLI) / model card (`serve`) | Nucleus cutoff (`0` or `1` = off). |
| `--top-k` | all | `≥ 0` | `0` (CLI) / model card (`serve`) | Top-k cutoff (`0` = off). |
| `--presence-penalty` | all | `-2..2` | `0` (CLI) / model card (`serve`) | One-time logit penalty for tokens already in reply. |
| `--frequency-penalty` | all | `-2..2` | `0` | Per-occurrence logit penalty for tokens in reply. |
| `--seed` | CLI only | `uint64` | `0` (random) | Seed for position-keyed Gumbel-max sampling. |
| `--accept-min-p` | CLI only | `0..1` | `0` (off) | Lossy relaxed draft acceptance: accept draft with prob ≥ `P * p_max`. |
| `--accept-typical` | CLI only | `0..1` | `0` (off) | Lossy relaxed draft acceptance: accept draft with prob ≥ `min(E, sqrt(E) * exp(-H))`. |
| `-p TEXT` / `-f FILE` | `run` | string / path | — | Prompt text or file path. |
| `--image FILE` | `run` | path (repeatable) | none | Attach one or more local image files (PNG, JPEG, BMP, GIF, TGA) to the prompt (implies `--chat`; requires a loaded vision encoder via `--mmproj` or Hub auto-fetch). |
| `--chat` | `run` | flag | off | Wrap prompt as a single user message through the chat template. |
| `--no-think` | `run`, `chat`, `template`, `serve` | flag | off | Disable thinking block by default (`enable_thinking=false`); `bench` always runs without it. |
| `-n N` | `run`, `chat`, `bench` | `≥ 0` | `256` | Maximum new tokens to generate. |
| `--ignore-eos` | `run` | flag | off | Continue past EOS tokens until `-n` or context limit (`bench` always does). |
| `--json [FILE]` | `bench` | flag, or path | off | Emit structured JSON benchmark output on stdout, or into `FILE` when the next argument is not an option. |
| `--rounds FILE` | `run` | path | — | Write one JSONL record per decode round (`position`, `tokens`, `drafts`, `accepted`, and `prob`/`margin` when the drafter reports per-draft confidence). |
| `--prompts FILE` | `bench` | path | — | A prompt set, one JSON object a line (`{"question": ...}`, `{"messages": [...]}` or `{"ids": [...]}`, the prompt as tokens), each prompt in a fresh session to its end token (`-n` defaults to 512 for a prompt set); `--json` then records every prompt's tokens and every round (for `tests/e2e/check_dv.py`). |
| `--suffix S` | `bench --prompts` | string | none | Text after each question (`\n` in `S` a newline, `\\` a backslash, any other backslash itself). |
| `--limit N` | `bench --prompts` | `≥ 0` | all | Run the set's first `N` prompts. |

---

## 3. Server Flags (`flyx serve`)

| Flag | Values | Default | Behavior |
|---|---|---|---|
| `--host` | IP / hostname | `127.0.0.1` | Listen address. Use `--api-key` when binding outside loopback. |
| `--port` | `1..65535` | `8080` | Listen port. If omitted and 8080 is held by another program, picks the next free port automatically (if held by an existing `flyx serve`, Explorer launches open its dashboard). |
| `--name` | string | filename / HF name | Model ID reported in `/v1/models` and responses. |
| `--models-dir` | `DIR` (repeatable) | none | Extra local directories scanned (depth 6) for `.gguf`/`.flyx` in `/v1/hub/cached` and dashboard (never deleted by `flyx`). |
| `--sessions` | `N` (≥ 1) | `8` (capped by memory) | Target concurrent GPU sessions (allocated on demand). Planner sizes full context for 1 session first, then attempts 2 within memory budgets when tight, admitting up to `N` sessions that fit (reported as `model.max_sessions`). |
| `--host-cache` | `GB` (≥ 0) | `8` | Pinned host RAM budget for parked session checkpoints (capped by RAM remaining after `--kv-ram`). `0` disables host snapshots. |
| `--kv-disk` | `DIR[:GB]` | off | Disk directory + GB cap (default `min(64 GB, 50% free space)`) for incremental prompt-prefix snapshots and spilled host checkpoints. Files are owned by the process and cleaned up on shutdown. |
| `--no-batch` | flag | off | Advance concurrent sessions one round at a time instead of batching rounds across sessions. |
| `--keep-billing-header` | flag | off | Retain Claude Code's per-request `x-anthropic-billing-header` line in system prompt (stripped by default to preserve KV prefix reuse). |
| `--api-key` | `KEY` | `FLYX_API_KEY` / `FLYX_KEY` | Require `Authorization: Bearer KEY` or `x-api-key: KEY` on all API routes and remote management calls. |
| `--cors` | `ORIGIN` \| `*` \| `none` (repeatable) | this machine's pages | Origins whose pages a browser lets call the inference API: `https://host[:port]`, an extension's origin (`chrome-extension://ID`), `null` (a page opened from a file; any sandboxed frame has it too). `*`: any page (said aloud at the start when no key is set). `none`: not even the pages of this machine (`localhost`, `127.0.0.1`, `[::1]`, any port), which may by default. See *Browser clients* below. |
| `--queue` | `N` (≥ 1) | `16` | Maximum queued requests waiting for admission. |
| `--queue-timeout` | `S` (seconds) | `600` | Maximum queue wait before rejecting request. |
| `--idle-unload` | `S` (seconds) | `0` (never) | Free GPU sessions after `S` idle seconds (weights stay loaded). |
| `--keep-alive` | `S` (seconds) | `0` (never) | Unload model completely after `S` idle seconds; reloads on next request. |
| `--threads` | `1..4096` | `64` | HTTP worker threads. A request holds one while it is queued or running, so keep it above `--sessions` + `--queue`; connections past the workers wait (as many again), and the ones past that are closed. |
| `--no-monitor` | flag | off | Disable terminal dashboard (auto-disabled if stdout is not a TTY). |
| `--no-browser` | flag | off | Do not auto-open browser dashboard (auto-disabled in SSH/headless sessions). |
| `--no-request-log` | flag | off | Disable in-memory ring buffer of last 50 requests (`GET /v1/requests`). |
| `--no-system-log` | flag | off | Disable in-memory ring buffer of last 2000 stderr lines (`GET /v1/logs`). |
| `--dashboard-dir` | `DIR` | embedded | Serve dashboard static files from `DIR` (e.g., `apps/flyx-server/dashboard`) before embedded assets. |

> **Remembered model settings:** Starting `flyx serve` with no `MODEL` reloads the last model and its saved model-level settings (`--ctx`, drafters, KV placement, and vision options `--mmproj`, `--no-vision`, `--vision-device`, `--vision-quant`, `--vision-max-tokens`) from `serve.json`. `serve.json` also persists a Hub mirror set via `POST /v1/hub/config` (`hub_endpoint`, applied at startup when `HF_ENDPOINT` is unset). To change saved model settings at startup, pass `MODEL` explicitly. Process flags (`--sessions`, `--kv-disk`, `--kv-method`, `--backend-opt`) must be passed on each launch.

---

## 4. Environment Variables

Defined in `core/os/env.h` (plus CUDA / Python standard variables). Explicit CLI flags take precedence over environment variables.

| Variable | Values | Effect |
|---|---|---|
| `FLYX_BACKEND` | `cuda` \| `cpu_ref` | Default backend when `--backend` is unset. |
| `FLYX_BACKEND_DIR` | path | Directory searched for `libflyx-*.so` / `libflyx-*.dll`. |
| `FLYX_BACKEND_OPTIONS` | `K=V,K2=V2` | Prepended before `--backend-opt` flags. |
| `FLYX_DRAFTER` | `mtp` \| `dflash2` \| `dspine` \| `dv` \| `all` \| `none` | Default drafter mode when `--drafter` is unset. |
| `FLYX_ACTIVATIONS` | `exact` \| `nvfp4` | Default activation mode when `--activations` is unset. |
| `FLYX_LOOKUP` | `0` \| `1` | Enable/disable prompt lookup (`0` = `--no-lookup`). |
| `FLYX_DRAFT_TREES` | `0` \| `1` | Enable/disable DFlash2 draft trees (`1` = `--draft-trees`). |
| `FLYX_ADAPTIVE` | `0` \| `1` | Enable/disable intra-round confidence optimal stopping on adaptive drafter plans (`1`, default). Enabled for sequential drafters (MTP) and for block drafters on targets whose verify cost scales steeply with width; dense targets paired with block drafters (DFlash2, DSpine) verify the chooser's fixed widths. |
| `FLYX_CHAT_NATIVE` | `0` \| `1` | Force Jinja engine (`0`) instead of native chat renderer (`1`, default). |
| `FLYX_API_KEY` / `FLYX_KEY` | string | Default server API key when `--api-key` is unset. |
| `FLYX_MODELS_DIR` | paths (`:` Linux / `;` Win) | Additional local model directories appended to `--models-dir`. |
| `FLYX_LIB` | path | Explicit `libflyx.so` / `flyx.dll` path for Python `flyx.Engine`. |
| `CUDA_VISIBLE_DEVICES` | GPU index list | Selects device 0 used by the CUDA backend. |
| `HF_HOME` / `HF_HUB_CACHE` | path | Hugging Face cache root / hub cache directory. |
| `HF_TOKEN` | token | Auth token for gated HF repositories (falls back to `hf auth login` token). |
| `HF_ENDPOINT` | URL | Hugging Face Hub mirror URL (takes precedence over a saved `hub_endpoint` in `serve.json` at startup; can still be overridden at runtime via `POST /v1/hub/config`). |
| `HF_HUB_OFFLINE` | `1` | Use local HF cache only; make no network requests. |
| `NO_COLOR` / `COLUMNS` | any / int | Terminal monitor formatting overrides. |

---

## 5. Backend Options (`--backend-opt K=V`)

Passed verbatim to the backend (`backends/cuda/backend.cpp`). Unknown keys or invalid values fail load immediately.

### Execution, Fusion, and Profiling

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `profile` | `0` \| `1` | `0` | Print per-node GPU time (disables CUDA graphs). Also supported by `cpu_ref`. |
| `graphs` | `0` \| `1` | `1` | Capture and replay decode rounds as CUDA graphs (each session's memory budget includes room for its graphs; `0`: none counted). |
| `dv_conditional` | `0` \| `1` | `1` | DV's canvas pass in a round's graph as the body of a conditional node, skipped after a no-op round (`0`: inline, run every round; measurement and tests). |
| `fusion` | `0` \| `1` | `1` | Enable fused region overrides (`0` = generic ops only). |
| `gdn_round` | `0` \| `1` | `1` | Fused 1-kernel GDN round path for ≤ 4 rows. |
| `stacks` | `0` \| `1` | `1` | Combine adjacent mixed-type projection runs in one round launch. |
| `round_glu` | `0` \| `1` | `1` | Fuse SwiGLU into dp4a gate\|up GEMM on GGUF rounds. |
| `side_stream` | `0` \| `1` | `1` | Run narrow stack GEMMs concurrently on a side stream. |
| `pdl` | `0` \| `1` | GPU table | Programmatic Dependent Launch (`sm_90+`). Default set by GPU tuning table (on for RTX 5090). |
| `wait` | `auto` \| `spin` \| `poll` | `auto` (Win) / `poll` (Linux) | Host wait mode for device completion (`auto` uses a high-resolution timer sleep followed by a short spin on Windows). |
| `tuning` | `table` \| `defaults` \| `NAME` | `table` | Kernel tuning table (`defaults` disables GPU table; `NAME` forces a named table e.g. `rtx4090`). |
| `log_shapes` | `0` \| `1` | `0` | Log each unique GEMM shape once at load/run. |
| `wide` | `auto` \| `0` \| `1` | `auto` | Wide prompt-lookup rounds: 31, 47 or 63 drafts verified at once (up to 64 rows, every row the bits of a 16-row round), taken after two 15-draft lookup rounds in a row are accepted whole (the width by the expected tokens per millisecond, from how often recent lookup drafts were rejected) and dropped when they yield fewer tokens per millisecond. `auto`: on where every layer's projections are NVFP4 and 24 GB of GPU memory remain after the weights; `0`: off; `1`: on (bf16 weights read once for every 16 rows, as NVFP4's; GGML block weights: 16-row strips). Not with penalties, relaxed acceptance or a grammar. The same rows carry a DV drafter's trees past 15 drafts: under `auto` a model that holds DV has the rows of DV's widest round (32 for MiniCPM5-1B's DV files, so lookup rounds of 31 there too), and with `0` DV drafts trees of 15 (its wider ones refused, said at load). Lookup rounds of any width run only where DV does not draft, and a lookup round leaves DV's canvas behind ([DV](design/dv.md)). Each session's buffers grow by the wider rows, and its graph room by the wider plans, both in the session size the session cap divides by (QUASAR's 27B: about 0.5 GB, most of it GDN replay inputs; 23.3 -> 23.8 GB a session at 262144 tokens; MiniCPM5-1B's DV files at 2048 tokens: 0.13 GB with `0`, 0.15 with DV's 32 rows, 0.17 with `1`). |

### Weights, Head, and Precision

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `gemm` | `auto` \| `rows` | `auto` | Force CUDA-core row GEMMs (`rows`, debugging). |
| `blocks` | `packed` \| `bf16` | `packed` | `bf16` dequantizes GGUF block weights to bf16 at load (higher VRAM, outside packed contract). |
| `embed` | `auto` \| `host` \| `gpu` | `auto` | Untied embedding table in host RAM (`host`, read over PCIe) or on the GPU (`gpu`); `auto`: `gpu` where 32 GB of GPU memory remain after the weights, else `host`. |
| `vision` | `disk` \| `host` \| `gpu` \| `auto` | `disk` | Vision encoder weight placement (`disk`: auto-deleted flushed temp file, 0 VRAM & 0 locked host RAM, 3-stage streamed via OS page cache + 16 MiB lazy pinned ring; `host`: pinned host RAM, 0 VRAM; `gpu`: GPU resident; `auto`: `gpu` when >= 8 GB free after weights, else `disk`). |
| `vision_quant` | `q8` \| `exact` \| `nvfp4` | `q8` | On-the-fly quantization of floating-point vision linears (`q8`: Q8_0, `exact`: file precision, `nvfp4`: NVFP4; pre-quantized GGML blocks stay packed). |
| `head_w12` | `0` \| `1` | `1` | Lossless 12-bit packed bf16 output head for 1..16 rows (`0` saves ~0.95 GB on 4B, ~5% slower plain). |
| `head_cert` | `0` \| `1` | `1` | Certified FP8-filtered exact greedy head (bit-identical greedy tokens; `0` saves ~0.64 GB on 4B). |
| `head_fp8` | `0` \| `1` | `0` | Lossy FP8 output head (trades accuracy for decode speed). |
| `draft_head` | `pick` \| `full` | `pick` | Draft candidate scoring: 1-bit top-256 prefilter + exact score (`pick`) vs full NVFP4 head (`full`). |
| `draft_weights` | `nvfp4` \| `exact` | `nvfp4` | Quantized (`nvfp4`/Q8_0) vs exact bf16 drafter weights. |
| `draft_kv` | `fp8` \| `16` | `fp8` | MTP drafter KV cache dtype when target KV is FP8. |
| `act_group` | `32` \| `64` \| `128` \| `256` | `32` | Measurement only: 8-bit activation scale group size (use with `fusion=0,round_glu=0`). |
| `act_fp4` | `0` \| `1` \| `2` | `0` | Measurement only: round staged activations as NVFP4 (`1`) or MXFP8 (`2`). |

### Attention, Tiered KV, StrataFlux, and MTP Cache

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `attention` | `auto` \| `split` \| `head` | `auto` | Prefill attention kernel (`split` = FP32; `head` = ungrouped, disables FP8 KV). |
| `rows_attention` | `mma` \| `cores` | `mma` | Decode/verify attention on tensor cores (`mma`) or FP32 CUDA cores (`cores`). |
| `rows_attention_from` | `N` (≥ 0) | `8192` | Key count threshold for switching decode attention to tensor cores. |
| `rows_attention_parts` | `1` \| `2` \| `3` | auto (`1` or `3`) | Binary16 split precision parts for `q` and `p` in tensor-core decode attention (`1` on GQA > 4 & HD ≥ 256, else `3`). |
| `rows_attention_q` | `16` \| `fp8` | `16` | Over FP8 KV on `sm_89+`, quantize query `q` to e4m3 and score on FP8 MMA (`sm_90+`) or binary16 tensor cores (`sm_89`, avoiding Ada's 22-bit FP8 accumulator). |
| `stage_ahead` | `0` \| `1` | `1` | Prefetch host KV / hosted StrataFlux copies concurrently with preceding layers. |
| `stage_dma` | `0` \| `1` | `1` | Use async H2D DMA transfers for committed StrataFlux prefixes (single-session decode) and aligned historical FP8 prefill windows. |
| `stage_blocks` | `N` (≥ 1) | `4` | CUDA blocks allocated to kernel-based `kv_stage` copy. |
| `stage_window` | multiple of 64 | `16384` | Device staging window positions for prefill over host KV. |
| `strata_from` | `N` (≥ 0) | `kv_gpu` | Position threshold where `--kv-method auto` switches from exact attention to StrataFlux. |
| `strata_denom` | `N` (≥ 0) | `16` (2-bit) / `64` (3–4-bit) | Refine 1 key pair in `N` per row (`0` = no refinement, scan copy only). |
| `strata_least` | `N` (≥ 1) | `32` | Minimum refined key pairs per row. |
| `strata_most` | `N` (≥ 0) | `256` | Maximum refined key pairs per row (`0` = unbounded). |
| `strata_recent` | `N` (≥ 0) | `0` | Extra most-recent positions always refined. |
| `strata_hot` | `0` or pow2 (`32..1048576`) | `2048` | Device LRU cache of refined host KV pairs per KV head (`0` = disabled). |
| `strata_copy_slots` | `2` \| `3` | `2` | Device double/triple-buffer slots for hosted StrataFlux copies when `stage_ahead=1`. |
| `strata_compact` | `0` \| `1` | `0` | Pack 256-dim hosted 3-bit StrataFlux copies (24 words/pos instead of 26). |
| `strata_scan` | `mma` \| `cores` | `mma` | Tensor-core (`mma`) vs FP32 (`cores`, measurement) StrataFlux scan. |
| `pageable_kv` | `auto` \| `on` \| `off` | `auto` | Spill tiered StrataFlux K/V overflow layers past the pinned host budget (WDDM non-local cap on Windows) into OS pageable RAM (`VirtualAlloc`/`mmap`) while keeping low-bit Strata copies and within-budget K/V layers pinned. |
| `pinned_host_mb` | `N` (≥ 0) | unset (auto) | Override the pinned host K/V budget in MiB for `pageable_kv=auto` (unset: the DXGI non-local budget on Windows, unlimited on Linux). |
| `strata_grid` | `range` \| `normal` | `range` | StrataFlux low-bit quantization grid mode. |
| `strata_chunk` | `N` (≥ 1) | `16384` | Selection chunk size in pairs (test/debug knob). |
| `mtp_window` | `N` (≥ 0) | `4096` | Above `8 * N` context, MTP reads last `N` positions exact (`0` = full exact device cache). |
| `mtp_copy` | `0` \| `2..4` | `2` | StrataFlux bit-width for MTP layer beyond `mtp_window` (`0` = disabled). |
| `mtp_sink` | `N` (≥ 0) | `64` | Initial sink positions kept exact for MTP when `mtp_copy=0`. |
| `mtp_ring` | `0` \| `1` | `1` | Ring buffer storage for MTP exact window (`0` = linear slots, tests). |

---

## 6. HTTP Endpoints & Schemas

Base URLs: OpenAI clients use `http://127.0.0.1:8080/v1`; Anthropic clients use `http://127.0.0.1:8080`. Max request body: `64 MiB`.

### Inference Endpoints

| Method & Path | API Family | Supported Request Fields & Output Contract |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat | `messages` (text or array of `text` and base64 `image_url` data URI parts), `tools` (`strict: false` allows arbitrary JSON object args), `tool_choice` (`auto`, `required`, `none`, `{"type":"function","function":{"name":...}}`), `parallel_tool_calls`, `response_format` (`json_object`, `json_schema`), `stream`, `stream_options.include_usage`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty`, `accept_min_p`, `accept_typical`, `stop`, `chat_template_kwargs` (passed directly to Jinja, e.g., `enable_thinking`, `reasoning_effort`). Returns `reasoning_content`, `tool_calls`, and `usage.prompt_tokens_details.cached_tokens`. |
| `POST /v1/responses` | OpenAI Responses | `input` (string or items: messages with `input_text` / base64 `input_image`, `reasoning`, `function_call`, `function_call_output`), `instructions` (current request only; developer messages in `input` persist), `previous_response_id`, `store`, function `tools` (top-level or `namespace`), `tool_choice`, `parallel_tool_calls`, `text.format` (`json_object`, `json_schema`), `reasoning.effort` (`none` disables thinking), `max_output_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream` (`response.created` … `response.completed` / `response.incomplete`), `include: ["reasoning.encrypted_content"]`. Thinking returned as `reasoning` items (`reasoning_text`). Rejects built-in/custom tool types, `background`, `conversation`, `prompt` with HTTP 400. |
| `GET`, `DELETE /v1/responses/{id}` | OpenAI Responses | Fetch or delete a stored response. Serialized JSON store cap is `256 MiB` (FIFO eviction; responses exceeding cap are returned unstored). |
| `GET /v1/responses/{id}/input_items` | OpenAI Responses | Paginated input items: `order=asc\|desc` (default `desc`), `limit=1..100` (default `20`), `after=LAST_ID`. |
| `POST /v1/completions` | OpenAI Legacy | Raw string `prompt`. Omitting `max_tokens` (or `null`) defaults to `16` tokens. |
| `GET /v1/models` | OpenAI | Lists the active model (empty list when unloaded). |
| `POST /v1/messages` | Anthropic Messages | `system`, content blocks (`text`, base64 `image`, `tool_use`, `tool_result`, `thinking`), `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`; `disable_parallel_tool_use`), `thinking` (`enabled`/`disabled`), `output_config.format` (`json_schema`; incompatible with assistant prefill), `stop_sequences`, `stream`, `max_tokens`. Reused tokens reported in `usage.cache_read_input_tokens`. **Prewarm mode:** `max_tokens: 0` with `stream: false` prefills and caches the prompt (rejects enabled thinking, `output_config.format`, or required tool use with HTTP 400). |
| `POST /v1/messages/count_tokens` | Anthropic Messages | Returns prompt token count for a Messages payload without generating (including expanded vision tokens when a vision encoder is loaded). |

### Management, Telemetry, and Hub Endpoints

`GET /health` answers anyone who reaches the port. `GET /v1/telemetry` and `GET /v1/load` take the API key when one is set (HTTP 401 without it). The others (`/v1/requests`, `/v1/logs`, `POST`/`DELETE /v1/load`, `/v1/unload`, `/v1/hub/*`) are for this machine, or with `--api-key` for the key's holder (HTTP 403 otherwise). This machine means a connection from the loopback address that no proxy passed on: a request carrying `Forwarded`, `X-Forwarded-*`, `X-Real-IP` or `Via` is taken as coming from elsewhere, so behind a reverse proxy or an HTTP tunnel set `--api-key`.

| Method & Path | Behavior & Response Fields |
|---|---|
| `GET /health` | `{"status":"ok", "model": ID\|null, "queued": N, "loaded": bool, "load": state}` |
| `GET /v1/telemetry?recent=N` | Schema version 1 live metrics snapshot (`recent` default `20`, max `100`). See [docs/monitor.md](monitor.md). |
| `GET /v1/requests?limit=N` | Last ≤ 50 finished requests (in-memory only; supports `ETag` / `If-None-Match` → 304). Fields: `enabled`, `requests[]` (`id`, `api`, `model`, `started_at`, `finished_at`, `stream`, `status`, `error`, `usage`, `ttft_ms`, `request`, `reasoning`, `content`, `tool_calls`, optional `timing`, `cancel_reason`, `client_hints`). Clips large request/reply strings (64 KiB cap; down to 1 KiB if body > 256 KiB); of a reply's tool calls the first 64 are kept, each argument string clipped at 64 KiB. |
| `GET /v1/logs?after=SEQ&limit=N` | Last ≤ 2000 stderr lines (8 KiB/line, in-memory; default `limit=500`). Returns `enabled`, `lines[{seq, t, text}]`, `next`, `missed`. |
| `GET /v1/load` | Current load/served status: `state` (`none`, `fetching`, `loading`, `ready`, `failed`), `progress`, `error`, `fallback` (always `null`: a model that fails to load is not replaced by the one served before; kept for clients that read the field), `notes`, `drafters`, `suggested`, `vision_projector` (`{from, size}` or `null`), active options, and `remembered` saved settings. |
| `POST /v1/load` | Async model load/switch (HTTP 202; HTTP 409 if another load is active). Body: `{"model", "drafters", "ctx" (N\|"max"), "drafter", "speculative", "kv" ("fp8"\|"16"\|"auto"), "kv_ram", "kv_gpu", "width", "draft_trees", "lookup", "draft_head", "draft_weights", "vision", "mmproj", "vision_device" ("disk"\|"host"\|"gpu"\|"auto"), "vision_quant" ("q8"\|"exact"\|"nvfp4"), "vision_max_tokens", "name"}` (`"kv": "auto"`: the engine's default cache type, whatever the server's `--kv`; the dashboard's auto choice sends it). Fetches while current model serves; on load failure after unloading, stays in `state: "failed"` with no model served. `DELETE /v1/load` aborts an in-progress fetch. |
| `POST /v1/unload` | Stops active requests and unloads model (`state: "none"`, inference returns 503). Preserves `serve.json`. HTTP 409 if loading. |
| `GET /v1/device/processes?kind=vram\|ram` | On-demand list of other applications occupying GPU VRAM (`kind=vram`, default) or host RAM (`kind=ram`): `kind`, `supported`, `can_close`, `device`, `mem_total`, `mem_used`, `mem_free`, `own_bytes`, `others_bytes`, `system_bytes`, `idle_igpu`, `apps[]` (`id`, `name`, `process`, `bytes`, `local_bytes`, `processes`, `closable`, `reason`). |
| `POST /v1/device/close` | Closes (`{"id", "force": false, "kind": "vram"\|"ram"}`, default graceful `WM_CLOSE`/`SIGTERM`) or force-quits (`"force": true`) an application currently listed by `GET /v1/device/processes`. Loopback requests only (HTTP 403 otherwise); refuses system processes, ancestor processes running FLYX, and other users' processes (HTTP 403). Answers `{"status": "closed" \| "still_running" \| "no_window", "windows_signaled", "name", "snapshot"}`. |
| `POST /v1/agents/launch` | Starts a coding agent against the served model in a new Windows PowerShell window on the server's machine (body `{"agent": "claude" \| "codex", "dir"}`). For Claude Code, the optional `permission_mode` accepts `default`, `acceptEdits`, `auto`, or `bypassPermissions`. Omitting it uses Claude Code's settings. Without `dir`, the desktop's folder dialog asks for the project folder. The server writes the command itself (the Integration page's, Codex's profile as `-c` overrides, no file of the user's written); an API key reaches the window as `FLYX_KEY` in its environment, never on a command line. Answers `{"launched": true, "dir"}`, or `{"launched": false}` when the dialog is cancelled. Loopback requests only (HTTP 403 otherwise); HTTP 409 with no model served, a dialog already open, or (Codex) a model id with one of `" & \| < > ^ %`, which PowerShell or `codex.cmd`'s cmd.exe cannot pass on; Codex is run as `codex.cmd` or `codex.exe`, so a system that runs no PowerShell scripts still starts it; HTTP 501 where no window can be opened (not Windows, SSH, a service). `telemetry.server.agent_launch` lists the agents it can start. |
| `GET /v1/hub/config` | Returns the active Hugging Face Hub endpoint configuration: `{"endpoint", "default_endpoint", "source" ("setting"\|"HF_ENDPOINT"\|"default"), "offline"}`. |
| `POST /v1/hub/config` | Updates the active Hugging Face Hub endpoint (`{"endpoint": URL}` or `""` to reset), normalizes/validates the URL, persists it as `hub_endpoint` in `serve.json`, and returns the updated config JSON. HTTP 400 on invalid URLs. |
| `GET /v1/hub/search?q=TEXT&sort=S&limit=N&cursor=C&arch=A` | Lists HF repos holding GGUF files whose names match `q`, plus matching DFlash2 drafters and exact `org/name` lookups (`sort`: `trending` (default), `downloads`, `likes`, `updated`, `created`; `limit` 1-100, default 20; `cursor` from previous `next_cursor`): `{"query", "family", "architectures", "arch", "endpoint", "next_cursor", "models": [{"id", "downloads", "likes", "updated", "created", "pipeline", "gated", "cached", "architecture", "runs", "drafter"}]}`. Without `q`, browses supported architectures (`"family": true`), optionally narrowed to `arch`. HTTP 502 when the Hub cannot be reached or `HF_HUB_OFFLINE` is set. |
| `GET /v1/hub/files?repo=ORG/NAME` | Lists a HF repo's GGUF quants, sizes, cache status, `mtp` and `mmproj` defaults, full `companions` list (`[{name, path, file, kind, size, cached, disk, default}]` covering drafters and `mmproj` projectors), known DFlash2 drafters, `drafter_repo` (`true` for `general.architecture=dflash` GGUF drafter repos), `endpoint`, `info` (`downloads`, `likes`, `updated`, `license`, `pipeline`, `gated`, the GGUFs' `architecture`, `context`, `parameters`; `null` when answered from the cache), `runs` (whether this build opens that architecture; `null` if unknown) and `build_architectures`. |
| `GET /v1/hub/cached` | Lists all cached HF snapshots (`org/name:QUANT` or `org/name:QUANT@COMMIT`, with `"mmproj": "in-repo"` when the snapshot includes a vision projector) and local `--models-dir` files (including `mmproj-*.gguf` with `"kind": "mmproj"` and `mtp-*.gguf` with `"kind": "drafter"`). |
| `GET /v1/hub/dir?path=PATH` | Browses a folder of the server's machine: `{"path", "parent", "kind": "dir", "dirs": [{"name", "path"}], "models"}` with its subfolders and the `.gguf`/`.flyx` model files in it and two folders down (entries as in `/v1/hub/cached`, loadable by `name`). A model file's path answers `"kind": "file"` with that file. Without `path`: `"kind": "start"` with where to begin (`--models-dir` folders, the home directory, on Windows the disks). HTTP 404 for a path that is neither. |
| `POST /v1/hub/dir` | Opens the desktop's own folder dialog on the server's machine (body `{"title", "start"}`, both optional) and waits: answers as `GET /v1/hub/dir` for the folder chosen, `{"path": null}` when cancelled. Loopback requests only (HTTP 403 otherwise); HTTP 501 where no dialog can be shown (SSH session, no display, a service; Linux needs `zenity` or `kdialog`); HTTP 409 while one is open. |
| `DELETE /v1/hub/cached?name=NAME` | Deletes a cached HF snapshot + unreferenced blobs (`{"deleted", "freed"}`). Refuses active models (409) and `--models-dir` files (400). |
| `GET /`, `GET /dashboard` | Browser dashboard UI (non-browser clients at `/` get JSON endpoint index). |

### Semantics & Error Codes Agents Should Expect

| Condition | Behavior / HTTP Status |
|---|---|
| Unloaded server | `503` until a model is loaded. |
| Context length exceeded | `400` before stream starts: OpenAI `context_length_exceeded` (`"This model's maximum context length is ..."`), Anthropic `"prompt is too long: N tokens > M maximum"`. If queued > 10 s on a streamed request, delivered as an in-stream error event. |
| Unsupported modality / tool format | `400` for remote image URLs, images sent when no vision encoder (`--mmproj`) is loaded, audio/video/document inputs, or `tools` on a model whose template has no supported tool-call format. Request fields not listed above (`logprobs`, `logit_bias`, `repetition_penalty`, `min_p`, the legacy `functions`, ...) are ignored. |
| Auth failure | `401` (invalid/missing API key); `403` (remote client accessing local-only management endpoint without API key). |
| Request superseded (same conversation retry) | Running request cancelled (`cancel_reason: "superseded"`). Anthropic queued/non-streaming returns `409` (`conflict_error`, `x-should-retry: false`); OpenAI Responses returns `status: "incomplete"` / `response.incomplete`. |
| Client isolation headers | Claude Code session/subagent headers and Codex `thread-id` headers isolate independent threads even with identical prompt histories. |
| Default sampling (when request omits params) | Qwen3.5 model card defaults: thinking ON → `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0`; thinking OFF → `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`. For tool/code use without thinking, pass `presence_penalty: 0` (or `--presence-penalty 0`). |
| Grammar enforcement | Enabled automatically when `tools` or `response_format` / `output_config.format` is set. Compiled once per tool/schema set and cached; checked after each speculative round with rollback on rejected tokens. |
| Tools or a schema no grammar can be made from | `400` (`invalid_request_error`, the message beginning `reply grammar:`) when the request requires the grammar (`response_format`, `tool_choice` `required` or a named tool); with `tool_choice: auto` the request runs unconstrained and the server logs why. |
| Browser clients (CORS) | A page of an allowed origin (see `--cors`) may call `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/models` and `/health`: its preflight (`OPTIONS`, which carries no key) is answered `204`, Chrome's `Access-Control-Request-Private-Network` included, and answers carry `Access-Control-Allow-Origin` with that origin. A page of any other origin gets `403`, whose message and the server's log name the `--cors` value that would let it in. The model panel, the request log, the logs and the telemetry are never another origin's, whatever `--cors` says. A request without an `Origin` (an SDK, `curl`) is as before. A page of a public site also needs the browser's own leave to reach this machine: Chrome and Edge ask the user before a public page may call a loopback address (Local Network Access), whatever the server answers; a page served from this machine is not asked. |
| Request limits | Body over 64 MiB: `413` (chunked bodies included). JSON nested more than 64 levels deep: `400`. Request target over 8 KiB: `414`. A client that sends nothing for 30 s is disconnected. |
| Server stopped serving before the request ran | `503` for a request still queued when the model is unloaded or switched, or that waited in the queue past `--queue-timeout`: send it again. |

---

## 7. Coding Agent Integrations

The dashboard's **Integration** page (`http://127.0.0.1:8080/`) generates exact shell commands pre-filled with your host, port, API key, loaded model ID, and context size for **Claude Code**, **Codex CLI**, **DeepSeek Harness**, **OpenCode**, and **Pi**. On Windows, with the browser on the server's machine, **Open in PowerShell** starts Claude Code or Codex CLI in a project folder you pick (`POST /v1/agents/launch`). For any other agent, its **Others** entry gives a text to paste to the agent (base URLs, model ID, context, how the key is passed) by which it sets itself up.

The Claude Code guide defaults to `bypassPermissions`, so tool calls generally run without asking for approval. Choose `acceptEdits` to allow project edits and prompt for other commands as needed. The permission picker applies to both the copied command and **Open in PowerShell**. Choosing `auto` lets a model review commands. On a single-slot server, those reviews can displace the active conversation and cause another prefill.

Quick environment templates:

```sh
# Claude Code (Anthropic Messages API)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=local-key   # or server's --api-key
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096

# OpenAI-compatible SDKs / Codex / OpenCode
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=local-key
```

> For long Claude Code sessions on a single GPU slot, pass `--kv-disk ./flyx-cache:40` to `flyx serve`. `flyx` writes incremental prefix snapshots (honoring Anthropic `cache_control` breakpoints) so title/permission sub-requests do not force a full re-prefill.

---

## 8. Python and C APIs

### Python (`python/flyx/__init__.py`)

Zero external dependencies (`ctypes` over `libflyx.so` / `flyx.dll`, `FLYX_API_VERSION = 3`). Automatically locates the native library inside wheels or unpacked release archives; override via `lib=` or `FLYX_LIB`.

```python
import flyx

eng = flyx.Engine("~/models/qwen35-4b.flyx", lib="~/flyx-build/bin/libflyx.so")
s = eng.session()

city_schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
weather_tool = {"type": "function", "function": {"name": "get_weather", "parameters": city_schema}}

# Tool call (schema-constrained)
reply = s.chat_full(
    [{"role": "user", "content": "Look up the weather in Paris."}],
    tools=[weather_tool],
    tool_choice="required",
    parallel_tool_calls=False,
    enable_thinking=False,
    max_new_tokens=128,
)
print(reply["tool_calls"])

# Structured JSON output
reply = s.chat_full(
    [{"role": "user", "content": "Give the capital city of France as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": city_schema}},
    enable_thinking=False,
    max_new_tokens=128,
)
print(reply["content"])
```

Concurrency & lifecycle:
- Multiple sessions on one `Engine` may generate concurrently on separate threads; one `Session` serves one generation at a time.
- `s.close()` is thread-safe and callable from a stream callback (cancels active generation and releases session on return). Closing `eng` keeps existing sessions alive until they close. Operations on closed handles raise `flyx.FlyxError`.

### C API (`include/flyx/flyx.h`)

Check `flyx_api_version() == FLYX_API_VERSION` (`3u`) before using struct layouts. Core entrypoints:
- Engine lifecycle: `flyx_load(path, &opts, &err)`, `flyx_unload(engine)`, `flyx_backend_name(engine)`, `flyx_tokenize(engine, text, out, cap)`.
- Session lifecycle: `flyx_session_new(engine)`, `flyx_generate(session, &req, on_event, user, &usage, &err)`, `flyx_cancel(session)` (thread-safe), `flyx_session_free(session)`.
- Events (`flyx_event_kind`): `FLYX_EVENT_TEXT` (1), `FLYX_EVENT_DONE` (2), `FLYX_EVENT_REASONING` (3), `FLYX_EVENT_TOOL_CALL` (4). Return non-zero from `on_event` to cancel.
- Wait for `flyx_generate` and concurrent calls on a session to return before calling `flyx_session_free`; open sessions hold a reference keeping the engine loaded past `flyx_unload`.
