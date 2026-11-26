# Architecture

System architecture, repository layout, execution pipeline, memory hierarchy, and invariants of `flyx`.

---

## 1. Repository Layout and Layer Boundaries

Directory boundaries are architecture boundaries enforced by `tools/check_layering.py` (ctest `layering`). Dependencies point strictly inward: `core/` depends only on itself and `include/`; `ops/` only on `core/interfaces/` and `include/`; `models/` on `core/interfaces/` and `core/modelfile/`; `backends/` on `core/interfaces/`, `ops/`, and `include/`; `apps/` wire layers together.

```text
include/flyx/         Public C API (flyx.h, version 3) and engine <-> backend ABI (backend_abi.h, version 44)
core/                 Plain C++20 engine, zero CUDA headers
  interfaces/         Shared headers: model.h (open_model), chat.h (open_chat), options.h, vision.h
  modelfile/          Format-agnostic mmap container: .flyx (flyx_format.cpp) & GGUF container (gguf_format.cpp)
  hub/                Hugging Face Hub client (parallel range download, SHA-256, lock/resume, cache scanner)
  plan/               Program -> backend plan lowering, VRAM/RAM/StrataFlux tier planner (budget.cpp)
  session/            Session state & prefix reuse rules, PrefixCache, host/disk SnapshotStore
  exec/               Engine load/unload, unified Request pipeline (request.h), Generation state machine,
                      ReplyText (UTF-8 & stop strings), Scheduler (GPU thread & queue), Telemetry, C API
  resources/          ResourceLedger and VRAM/host memory leases
  speculation/        RoundChooser, SpecController (cost-weighted drafter/width selection), LookupDrafter
  render/             ChatTemplate (native renderer or Jinja fallback), OpenAI/Anthropic/Responses converters
  vision/             Base64/data-URI & in-memory image decoding (image.cpp), media-span prefix matching (media_span.h)
  tokenizer/          Byte-level BPE (template-only special tokens) and streaming UTF-8 decoder
  grammar/            XGrammar compiler, cache, and per-request token matcher
  os/                 OS abstraction (mmap, dynamic libraries, time, process/system stats, env.h, utf8.manifest)
ops/<op>/ref.h        Float64 reference per op; ops/sampling/sampling.h (host/device token choice);
                      ops/dtype/ggml.h (GGML block float64 semantics)
models/               Architecture descriptions (blocks, weights, state) and Programs: qwen35/, llama/ (Qwen3 and
                      Llama), vision/ (VisionAdapter preprocessing, smart-resize, 3D MRoPE positions), gguf.h
  drafters/           Drafters the architectures share: DFlash2 from a .flyx file, its authors' directory or a
                      dflash GGUF (dflash2.cpp), DSpine from a .flyx file or its authors' directory (dspine.cpp),
                      their shared Qwen3 layers and context K/V (qwen3_layer.h), claimed, attached and given their
                      program nodes by registry.cpp
  formats/            Bidirectional chat formats: detection, reply parsing (tags/calls), XGrammar structural tags,
                      and byte-identical native renderers (qwen35.cpp)
backends/cpu_ref/     Independent float64 reference backend (shared library)
backends/cuda/        CUDA 13 backend (shared library): backend.h, blocks.h, backend.cpp
  kernels/            CUDA kernels by op family (gemm*, attention*, gdn*, rows, sampling, draft, tree, vision);
                      kernels.cuh declares all public launchers; common/ holds device.cuh and launch.h
  regions/            Generic and fused region implementations (layers.cpp, linear.cpp, plan.cpp)
  runtime/            Sequence buffers & VMM KV mapping (state.cpp), CUDA-graph rounds & rollback (runner.cpp),
                      round-graph memory and each session's room for it (graphs.cpp), disk-offloaded vision
                      weight streaming (vision_disk.cpp), adaptive host wait (sleep.cpp), NVML telemetry (device_stats.cpp)
  loader/             Block binding & stacked weight layout (bind.cpp), multithreaded pinned upload (weights.cpp)
  variants/           Compiled kernel configs (candidates.toml) and per-GPU tuning tables (<arch>/tables/<sku>.toml)
apps/                 flyx-cli (unified `flyx` binary + terminal monitor), flyx-server (`serve`: openai_chat.cpp,
                      openai_responses.cpp, anthropic_messages.cpp, embedded dashboard),
                      common/ (shared engine/sampling flag parser, defaults.h, model_source.cpp)
python/               flyx/ (ctypes binding), convert/ (HF/ModelOpt -> .flyx), reference/ (PyTorch float64 ref)
tests/                Mirrors source tree + tests/e2e/ (golden, GGUF, real-model checks)
third_party/          Vendored libs (nlohmann/json, utf8proc, cpp-httplib, jinja, xgrammar, ggml, preact, htm, stb)
bench/                cuda_bench, tree_bench, ab/, external/, prompts/, quality/, replay/, trace/
tools/                codegen/ (gen_gemm_tables.py), windows/ (vsenv.ps1), ci/, check_layering.py, check_numerics.py,
                      check_docs.py
packaging/            Linux tarball + manylinux wheel (package.sh), Windows ZIP + win_amd64 wheel
                      (package_windows.ps1), smoke checks
```

---

## 2. Unified Request Pipeline

Every entrypoint (HTTP server via `Scheduler`, C API, Python binding, CLI) builds a `Request` (`core/exec/request.h`) and runs the same two functions—wire handlers never reimplement prompt framing or checkpoint rules:

1. **Render (`render_request`, stateless):** Converts OpenAI Chat messages, OpenAI Responses items, or Anthropic content blocks into `ChatMessage`s and renders them through `ChatTemplate` (native renderer when matched, otherwise vendored Jinja). Tracks exact byte spans emitted by the template vs user text so special tokens are tokenized only where the template wrote them (preventing prompt-injection turn breaks). Records conversation branch points and whether an open thinking block was started.
2. **Session selection (`PrefixCache`):** Matches a resident session whose history the prompt extends (respecting client isolation headers like Claude Code session/subagent IDs and Codex `thread-id`), else picks a free slot, else evicts a sibling sharing ≥ 256 bytes of prefix (same system prompt/tools), else LRU. Evicted sessions save checkpoints into `SnapshotStore` (pinned host memory and optional `--kv-disk`) for fast restoration. A retry or continuation of an actively generating conversation cancels the superseded run.
3. **Prepare (`prepare_request`):** Reuses existing session tokens for the unchanged prefix, then picks the longest valid starting state: live state, short-window rollback inside the last rounds (folded from saved inputs), checkpoint restore (at message boundary or from host/disk snapshot), or cold start.
4. **Prefill:** Runs chunked passes (`--prefill-chunk`, default 1024 to 4096 rows by free GPU memory) split at the last two message boundaries (taking checkpoints there and at prompt end). The final prefill pass emits the first output token.
5. **Speculative rounds:** Up to `FLYX_MAX_IN_FLIGHT` (2) rounds remain in flight per sequence. Each round replays one captured CUDA graph: draft → verify `width + 1` rows → sample per row → accept → commit (fold recurrent state to accepted rows) → update every loaded drafter's context. `RoundChooser` selects each round's drafter/width from load-time measured costs and live acceptance, or schedules a 15-draft `LookupDrafter` round when recent tokens repeat earlier sequence text (dropping to 1 in-flight round when lookup is due).
6. **Output & stop strings:** Stop token IDs terminate on the GPU; stop strings and complete UTF-8 boundaries are handled on the host by `ReplyText`. The model family's reply parser (`models/formats/`) separates `reasoning`, `content`, and structured `tool_calls`.
7. **Post-round grammar enforcement:** When `tools` or structured JSON output is active, `core/grammar` compiles and caches an XGrammar structural-tag matcher. Speculative rounds execute at full speed with no per-token masking; tokens are validated as rounds arrive. On the first rejected token, in-flight rounds drain, the session rolls back to before that position, and the next round selects row 0 under the grammar bitmask. Position-keyed Gumbel-max sampling keeps this mathematically identical to masking every step.

---

## 3. Single-GPU Scheduler and Batched Rounds

One dedicated thread (`Scheduler`) owns the GPU and drives a set of `Generation` state machines (`advance_all()` / `advance()` execute one unit of work—prefill chunk, round launch, or result drain—with zero request state kept on the C++ call stack):

- **Admission:** Scans queued jobs in order, assigns available sessions without head-of-line blocking when another job's target session is busy, and caps concurrent execution at `max_sessions`.
- **Single active request:** Uses blocking `advance(true)` with zero scheduling overhead.
- **Multiple active requests:** Polls readiness (`ready()`) without starving any stream. Sessions in their decode phase batch rounds together (`launch_batch`, up to `FLYX_MAX_ROWS = 16` total rows) so target weights are read once across concurrent sessions while each session's attention/GDN runs on its own slice—producing bit-identical outputs to solo execution (`--no-batch` reverts to turn-taking).
- **Host & disk snapshots:** Evicted sessions copy recurrent state and positional KV prefixes into reusable 32 MiB pinned host slabs (`SnapshotStore`), restoring 30K-token 27B prompts in ~0.2 s instead of ~15 s prefill. With `--kv-disk DIR[:GB]`, incremental prompt-prefix snapshots spill to disk so single-slot GPU setups survive short side requests (e.g., title/permission prompts) without losing long conversation caches.

---

## 4. Program, Regions, and Ops

Model execution is cleanly separated into four abstractions:

1. **Blocks (`flyx_model_desc`, backend ABI 45):** `models/<arch>/` describes the model purely as data—weight table, per-sequence state table (KV caches, GDN states, MTP hidden ring), and typed blocks (`embedding`, `attention`, `gdn`, `mlp`, `head`, `mtp`, `dflash2`, `dspine`, `dv`, `dv_layer`, plus `vision_stem`, `vision_pos`, `vision_block`, `vision_merge` for `FLYX_PHASE_VISION` multimodal encoders) carrying dimensions, layout flags, and indices. Backends implement block kinds without knowing model names and refuse unsupported blocks at load time. Vision models also expose a model-owned `VisionAdapter` (`core/interfaces/vision.h`, implemented in `models/vision/`) for image preprocessing and 3D MRoPE position generation, while `backends/cuda/` encodes images via `encode_vision` and splices embeddings into prefill via `prefill_ex`. Since ABI 34 the backend also exposes `trim()` so the engine can reclaim speed-only auxiliary copies (e.g., W12 / FP8-certified / 1-bit `SignHead` draft copies) under tight VRAM budgets before failing load.
2. **Program:** `models/<arch>/` emits a flat, backend-neutral node list for prefill and for each drafter/width round (`run block`, `tap residual`, `draft`, `take context`, `accept`, `commit`). `core/plan` lowers programs once at load time.
3. **Regions (`backends/cuda/regions/`):** `lower()` maps every node to either a generic op sequence or a registered fused override matching the block shape and row count (e.g., fused producer-to-GEMM staging, 1-kernel GDN rounds for ≤ 4 rows, SwiGLU epilogues). `--backend-opt fusion=0` forces generic implementations; golden tests verify fused == generic. Lowered decode plans are captured once per session as CUDA graphs.
4. **Ops (`ops/<op>/ref.h` & `backends/cuda/kernels/`):** Every CUDA kernel launcher declared in `kernels/kernels.cuh` has a float64 reference in `ops/<op>/ref.h`, unit tests in `tests/backends/cuda/op_test_*.cu`, and a machine-checked contract row in [docs/design/numerics.md](numerics.md).

---

## 5. Memory Hierarchy, Tiered KV, and StrataFlux Attention

All device and host memory is accounted for by `ResourceLedger` before allocation (`seq_size` runs allocation sizing in counting mode):

- **Contiguous virtual KV cache:** Each session's KV cache occupies one contiguous virtual address range (avoiding page tables so CUDA graph addresses remain invariant). Default dtype is FP8 where under 32 GB of GPU memory are left after the weights and FP16 above (FP8: `e4m3` with a float32 scale per position and head; `--kv fp8` or `--kv 16` chooses).
- **GPU + pinned host VMM tiering:** `core/plan/budget.cpp` places as many leading KV positions on GPU as VRAM allows (`--kv-gpu`), and maps overflow positions into pinned host memory (`--kv-ram`) within the same virtual range using CUDA virtual memory management. During prefill and decode, host-resident KV is staged ahead over PCIe concurrently with preceding layers (`stage_ahead=1`, `stage_dma=1`).
- **2-bit StrataFlux attention (`--kv-method auto` / `strata`):** When context exceeds GPU KV capacity (`strata_from`, default `kv_gpu`), reading every host KV position every decode step becomes PCIe-bound. StrataFlux maintains a compact low-bit Walsh-Hadamard summary copy (default **2-bit** for both target and MTP layers) per position:
  - A fast 1-warp tensor-core scan (`attention_strata`) scores all positions from the 2-bit copy and computes per-pair log-sum-exp importance scores.
  - Only the highest-scoring pairs per row (`strata_denom=16`, clamped to `32..256` pairs) are refined using exact FP8/FP16 KV fetched from host RAM and cached on GPU in a per-head LRU table (`strata_hot=2048`).
  - If all layers' 2-bit copies fit in VRAM, they stay GPU-resident; otherwise remaining layer copies reside in host RAM and double-buffer (`strata_copy_slots=2`) ahead of each attention layer via async H2D DMA.
- **CUDA round graphs inside `seq_bytes`:** An instantiated CUDA graph takes device memory outside the session's arena, and the driver keeps that memory in its pool after the graph is destroyed, for later graphs (`cuda_bench graphs` on an RTX 5060 Ti: 96 KiB for a graph of one kernel node; 2.7-3.5 KiB per kernel node in chains of 100 to 1000, with or without programmatic dependent launch; 11.5 KiB per node with 4 KB of kernel parameters; 5.8 KiB per device-to-device copy node; copies to or from host memory and memsets nothing past the graph's own; on an RTX 3090, sm_86, 8.5-11 KiB per kernel node whatever its parameters and 21.8 KiB per device-to-device copy node, the rest the same). A session keeps one graph per round plan, head and StrataFlux phase it ran, and batched rounds keep up to 64 more, so `seq_bytes` / `seq_size` add a room for them (`backends/cuda/runtime/graphs.cpp`):
  - One graph for every round plan lowered (trees of given drafts, which the engine lowers but never runs, excepted), priced by node kind from the captured graph (`graph_size`: a kernel node and a copy node at the GPU's architecture's price, `graph_prices`: 3.5 and 7 KiB on sm_120, 11.5 and 22 KiB on every other, measured or not; 3 bytes per parameter byte past 1 KB, read with `cuFuncGetParamInfo`; 1 KiB per memset, 16 KiB for any other node, 128 KiB per graph, and a conditional node's bodies each as a graph of its own by its nodes: DV's canvas pass is one, see [DV](dv.md#dvs-rounds-on-the-cuda-backend); above every graph measured), plus what other rounds of the plan take on the session's layout: 8 kernels for other heads, 3 per MTP draft step once the context holds more than 8 MTP windows, 2 per KV cache in tiers, 12 kernels per low-bit copy the rounds read (10 measured at most) and 4 copy nodes per copy in host memory.
  - A DV file's rounds are all among them: its trees of 15 and of the file's width are drafted trees, not given ones, and its warm-up is a round like plain decoding's (each keeps a graph, its canvas pass the body of a conditional node); with the target's V-LoRA every round's graph holds the adapters' GEMMs. On MiniCPM5-1B's DV files that is eight plans (a lookup width of 31 on their wide rows among them): 19.8 MiB of room a session on `minicpm5-1b-dv-res2.flyx`, 34.9 with the V-LoRA on `minicpm5-1b-dv-b2h.flyx` (`flyx run -v`). Rounds past 16 rows widen the session's buffers as well as its room (the backend option `wide` in [the server's options](../server.md); [DV](dv.md) for these files at each setting).
  - One share of a batch graph beside them, priced as the largest round: a batch graph of n sessions holds fewer nodes than their n rounds alone, so a session that goes from rounds alone to rounds batched with another and back keeps both graphs instead of giving one up each time.
  - A quarter more that the session's graphs may not hold, for the driver's pool, which gives the memory of graphs a session gave up only to later graphs that fit in it: in the memory tests, where sessions give up graphs for others again and again, the device lost 0.72-0.77 (K/V on the GPU) to 0.96-0.99 (K/V and StrataFlux copies in host memory) of what the graphs held at these prices, so the quarter leaves room for a driver that takes about a quarter more.
  - A plan's price is its first captured graph (the load's round timing). A plan nothing has captured yet when `seq_size` is asked is captured there, not instantiated, on a 64-token scratch sequence freed before it returns; `seq_size` fails rather than count less when that sequence cannot be made. The count depends on the plan and the kernels chosen, not on a measurement, so a model loads with the same figure every time. With `graphs=0` or `profile=1` nothing is counted.
  - A session keeps its graphs inside its room. When a new one does not fit, the graph it ran least recently goes first, its own or a batch graph it shares alike, by the session's own round count (finished, and unused in its last 8 rounds); when that is not enough the round runs without a graph and the session tries no capture for 8 rounds (`-v` says so once). A batch graph dropped on one thread gives its shares back to sessions another thread may drive, so a session's `graph_held` is atomic.
  - The device reserve (`max(384 MiB, total / 40)`, raised to at least `832 MiB` = `768 MiB` paging floor + `64 MiB` desktop cushion where Windows pages GPU memory) absorbs driver overhead and desktop fluctuations without double-stacking on larger cards.
- **What the budget says when memory is short:** The budget is what the device has free at the load plus what the model took, less the reserve; what is not in it is held by other programs, the driver, and the process's own CUDA runtime (context and kernel modules: 0.54 GB beside the 27B on an RTX 5070 Ti under Windows). `flyx_memory_info.process` (ABI 37) is the process's share of the device as the system counts it, which tells that runtime from other programs: Windows counts video memory by process (DXGI, the adapter with the CUDA device's LUID: 11,164 MiB against the "GPU Process Memory" counter's 11,168); NVML does not there nor under WSL2, and elsewhere the figure is 0 and the two are reported as one. Host memory is told the same way since ABI 39: `flyx_seq_size.host_mapped` is the part of a session's host memory mapped for the device alone (its K/V past the GPU's part), which no process memory figure counts, and `flyx_device_stats.process_host` the process's page-locked host memory as Windows counts it; telemetry's `process.ram` adds them to the resident memory.
  - Other programs' share of a desktop GPU is not a fixed figure: Windows moves idle programs' video memory out under pressure and back when the GPU empties. On that 5070 Ti they held 1.7 GiB beside a loaded 27B, 2.2 GiB a minute after it was unloaded and 3.5 GiB at a load made while the desktop was in use; a reload measures them after they have come back (0.2-0.35 GiB more than a few seconds later, on three loads). The budget takes what it measures; it does not wait for, or count on, what Windows may move out.
  - Even with every K/V position in host memory a session keeps its fixed state, its positions' scales and its stagings on the GPU (the 27B's 2-bit file with its MTP layer: 0.74 GB at 1,024 tokens, 0.91 GB at 32,768, 1.78 GB at 131,072), and language-model weights never go to host memory (vision encoder weights default to `--vision-device disk`, taking 0 VRAM and 0 locked host RAM). So a load can fail, or `--ctx max` settle for a few thousand tokens, with host memory to spare. The refusal then names that shortest session and how much is missing rather than the context asked for; a context cut short is logged with the memory that cut it (`plan::ctx_limit_note`), as a warning under `kShortCtx` (32,768 tokens), and reported in the server's telemetry (`model.context_limit`).
- **Compact GDN state & shared prefill buffers:** Each session holds `FLYX_ROUND_SLOTS = 3` recurrent state slots on GPU (supporting 2 in-flight rounds plus rollback via input replay), while message-boundary checkpoints live in pinned host RAM. Prefill row buffers (`--prefill-chunk`) are allocated once per model and borrowed across sessions.

---

## 6. Core Invariants

1. **Pure-function token choice (`ops/sampling/sampling.h`):** Token choice is a pure function of `(logits, sampling_params, seed, position, reply_counts)` using greedy argmax or Gumbel-max keyed by `(seed, position, token)`. A speculative draft is accepted if and only if it equals that exact choice, guaranteeing identical output distribution and identical tokens for plain vs speculative decoding (including under presence/frequency penalties).
2. **Row-count independence:** Every 1..16-row GEMM, attention kernel, and GDN kernel uses shape-fixed K partitions and fixed reduction trees independent of concurrent row count or kernel tuning configuration. Plain decoding, speculative verification, and batched multi-session rounds yield identical bits per row.
3. **Matched reference rounding:** Where a kernel rounds activations to match weight format contracts (e.g., 8-bit activations for GGUF blocks, NVFP4 activations for calibrated ModelOpt prefill, bf16 activations for NVFP4 decode), `ops/<op>/ref.h` and `cpu_ref` apply identical input rounding before float64 accumulation.
4. **Per-GPU specialization without math drift:** `cuda_bench tune` writes per-GPU dispatch tables under `backends/cuda/variants/<arch>/tables/<sku>.toml`, compiled by `tools/codegen/gen_gemm_tables.py` into dispatch rules (falling back to `variants/candidates.toml`). Tables select only among bit-identical configurations and GPU-appropriate PDL flags (`--backend-opt tuning=defaults` bypasses tables).
5. **Deterministic memory accounting:** `seq_bytes` reports exact device and host consumption, and a bound on the CUDA round graphs a session keeps (kept inside it); GPU memory is freed on the GPU thread after stream drain, so `cuda_device_memory` verifies zero leaked bytes across load/unload cycles.
