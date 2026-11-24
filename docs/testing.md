# Testing

This document explains how to install the test dependencies and run the tests: test-data generation, kernel checks,
GPU tracing and end-to-end verification. Read [AGENTS.md](../AGENTS.md) before changing any test.

---

## 1. Testing Setup

```sh
pip install torch transformers tokenizers safetensors gguf   # test fixtures & HF reference
ctest --preset dev --output-on-failure                       # full GPU + CPU suite
ctest --preset cpu --output-on-failure                       # CPU-only (no GPU required)
```

End-to-end verification (http request to the qwen model server):

```sh
python tests/e2e/check_real_model.py --hf ~/models/Qwen3.5-4B --model ~/models/qwen35-4b.flyx --lib ~/flyx-build/bin/libflyx.so
python tests/apps/flyx-server/check_server.py --url http://127.0.0.1:8080
```

CI scripts: `tools/ci/cpu.sh` (hosted CPU runner) and `tools/ci/gpu.sh` (self-hosted GPU runner); see [tools/ci/README.md](../tools/ci/README.md).

---

## 2. Test Suite Map

| Test(s) | Scope |
|---|---|
| `layering`, `numerics`, `docs` | Static gates: directory include boundaries (`tools/check_layering.py`), `docs/design/numerics.md` launcher/test coverage (`tools/check_numerics.py`), public doc links and hygiene (`tools/check_docs.py`). |
| `make_assets`, `make_gguf_assets` | Fixture generators: tiny Qwen3.5 + DFlash2 (BF16/NVFP4), + DFlash and + DSpine (and with a window of 8) with float64 reference golden outputs, tiny Qwen3 with `transformers` golden outputs, and GGUF fixtures across all GGML block types. |
| `reference_vs_transformers` | Checks `python/reference` float64 reference against HuggingFace `transformers`. |
| `cuda_ops` | Every CUDA kernel (`tests/backends/cuda/op_test_*.cu`; DSpine's bridge and walk in `op_test_dspine.cu`) against `ops/<op>/ref.h` and `ops/sampling/sampling.h` (seeded; override via `FLYX_OPS_SEED`). |
| `golden_cpu_ref*`, `golden_cuda*` | End-to-end token-for-token checks: plain, MTP, DFlash2 (and DFlash or DSpine in its place, `_dflash`, `_dspine`, `_dspine_window`, a drafter of one layer `_dspine1`; on the tiny Qwen3, `_qwen3_dflash`, `_qwen3_dspine`), lookup drafts, forced long accepted runs, rollback, checkpoint restore, edited prompts, seeded sampling, stop strings, penalties, grammar enforcement, tree rounds, FP8 KV, tiered host KV, 2-bit StrataFlux & `auto` switch, fused vs unfused (`fusion=0`), PDL on/off, Qwen3 second architecture, sliding-window DFlash2, adaptive stopping on and off. |
| `cuda_device_memory*` | Serial tests verifying engines and sessions release all allocated VRAM back to the device, and that sessions' CUDA round graphs (every round kind, drafted trees, MTP's window and batched rounds included) stay inside the room `seq_bytes` counts, measured by the device's free memory (`_tiers`: K/V and StrataFlux copies in host memory, across the `auto` switch), and that a session going between batched and lone rounds keeps its graphs (`_dv`, `_dv_lora`: the tiny Llama's DV files, every kind of DV round, its warm-ups batched too, beside a V-LoRA; `_dv_graphs`, `_dv_lora_graphs`: DV's round graphs within what they are counted at and under twice what they took, the canvas pass the body of a conditional node and inline (backend option `dv_conditional=0`), and the conditional nodes with their bodies on their own: the sessions as many as free memory holds for three engines, 32 to 64, or the test says why it checks nothing). |
| `gguf_reader`, `ggml_blocks`, `gguf_models`, `gguf_cpu_ref`, `gguf_cuda*` | GGUF container validation, exact per-block float64 dequant vs `gguf` Python package, layout equivalence vs `.flyx`, packed GPU execution across all block types + GGML NVFP4 + sidecar MTP/DFlash2. |
| `dflash2_models` | The shared DFlash2 drafter (`models/drafters/`): config validation before tensor binding, drafter GGUFs beside the tiny Qwen3 and Llama (block, taps, context nodes), drafter files refused for what they say or for the model they were made for. |
| `tokenizer`, `bpe_merges`, `gguf_qwen2_*` | BPE merge exactness, pre-tokenizer splits, streaming UTF-8 decode. |
| `flyx_reader`, `flag_numbers`, `cli_usage`, `hub`, `hub_concurrent`, `drafter_dir` | Malformed `.flyx` refusal, strict CLI numeric parsing, the command line where it is asked something or cannot be followed (`--version`, `--help`, an unknown option), Hub SHA-256 & multi-process lock/resume, drafter kinds of cached repos, direct DFlash2, DFlash and DSpine directory loading. |
| `chat`, `chat_qwen35`, `chat_replies`, `reply_grammars` | Template rendering vs `transformers`, native Qwen3.5 renderer vs Jinja on 3000 generated conversations, XML/JSON reply parsing, XGrammar structural tags. |
| `request`, `reply_text`, `round_chooser`, `memory_budget`, `snapshot_store`, `strata_snapshot`, `parked_edit`, `telemetry`, `response_store`, `request_log` | Core request pipeline, stop-string boundary handling, speculation width/drafter chooser, VRAM/RAM tier planner, host/disk snapshot stores, telemetry snapshot. |
| `scheduler_*`, `session_host_budget`, `capi`, `python_binding`, `convert_*` | Multi-session scheduling, concurrent generation, prewarm/superseding semantics, refused-session guards, C API & Python `ctypes` binding, converter UTF-8/CLI checks. |
| `server_hardening` | What a hostile or broken client sends a live `flyx serve`, and what a browser brings: none of it ends the process, fills its memory, holds its workers or reads what it should not (`tests/apps/flyx-server/hardening.py` lists the checks). |
| `api_compat`, `streaming_tool_calls`, `queued_request_*`, `client_conversations`, `codex_threads`, `remembered_model`, `kv_method_flags`, `cli_utf8_paths` | Live `flyx serve` and CLI integration tests (OpenAI Chat/Responses, Anthropic Messages, cancellation, session isolation, remembered `serve.json`). |
| `gemm_tables` | `tools/codegen/gen_gemm_tables.py` validation, table inheritance, and invalid-table rejection (`cuda_ops` additionally verifies compiled table reachability via `test_gemm_tables_compiled`, rows 1..64, and a drafting pass's K partition by K alone in every table via `test_gemm_draft_tables`). |
| `golden_cpu_ref_llama*`, `golden_cuda_llama*`, `tokenizer_minicpm5`, `llama_models`, `converter` | The tiny Llama (MiniCPM5-1B's layers: no q/k norms, 16 query heads on 2 KV heads of 128) against `transformers` in float64 (plain decoding, unfused, `rows_mma`, NVFP4, prompt passes of up to 64 rows, batched rounds), MiniCPM5's pre-tokenizer, Llama descriptions and refusals, the converter's Llama path. |
| `golden_cuda_llama_dv*`, `golden_cpu_ref_llama_dv*`, `dv_models`, `dv_trees`, `dv_canvas`, `dv_convert*`, `reference_dv*`, `dv_counts` | DV: plain decoding's tokens in every mode (15 and 31 drafts, beam and pool, 8/16/32 slots, causal, two rounds in flight, graphs on and off, rewind, restore, stop strings, sampled and with penalties, FP8 and host K/V, rounds mixed with lookup rounds, wide ones too (`wide=1`), and `wide=0`'s trees of 15, rounds of several sequences, a chat's checkpoint at a message boundary inside a prefill pass restored, adaptive stopping on and off, none planned for DV); DV's drafts against `flyx_ref/dv.py` in float64; the description and its refusals; the float64 tree and canvas pieces; the converter byte for byte; the Python reference against bifill-dv's functions; `check_dv.py`'s counts. |
| `golden_cuda_llama_lora*`, `golden_cuda_llama_dv31_lora*`, `lora_convert`, `dv_convert_lora`, `residual_decision` | The V-LoRA: the tiny Llama with a target LoRA against `transformers` with it merged (float64), on both backends, unfused and `rows_mma`, its NVFP4 linears, DV beside it; `.flyx` format 2 and the converter; each backend's bf16 prefill residual decision (`-v`), taken by the NVFP4 Llama with NVFP4 inputs and never with adapters (a device without FP4 tensor cores is held to its refusal of NVFP4 inputs instead). |
| `golden_cuda_quant_q4_k`, `golden_cuda_quant_q8_0`, `tree_bench_context` | Tree rounds of up to 64 rows on quantized Qwen3 GGUFs held to their own decoding; `tree_bench` timing every width from the same context. |
| `round_plans`, `plans_test`, `calibrate_test`, `round_costs_plausible`, `scheduler_cuda_dv`, `dv_choice_replay` | DV's rounds chosen per round: the plans (widths, warm-up, refusals), costs at the context and their re-timing, a DV file's rounds in requests that run at once, and the chooser's replay on the traces in `bench/replay/dv_traces/`. |
| `bench_prompts`, `same_runs` | `flyx bench --prompts` (a prompt set, its JSON record) and `bench/ab/same_runs.py` (two runs' answers and rounds compared). |

---

## 3. CUDA Graph Tracing on Windows (CUPTI)

When configured with `FLYX_TESTS=ON` and CUPTI headers + `cupti.lib` present (optional `cufilt.lib` + `nv_decode.h` for symbol demangling), CMake builds `flyx-trace`. Place the matching CUPTI runtime DLL on `PATH`:

```powershell
cmake --build $BuildDir --target flyx-trace
$env:CUDA_INJECTION64_PATH = (Resolve-Path "$BuildDir\bin\libflyx-trace.dll").Path
$env:FLYX_TRACE = Join-Path $PWD 'rounds.tsv'
& "$BuildDir\bin\flyx.exe" run $Model -p hello --no-spec --ignore-eos -n 128
Remove-Item Env:\CUDA_INJECTION64_PATH, Env:\FLYX_TRACE
python bench\trace\gaps.py rounds.tsv --sequence
```

The TSV logs GPU kernel, copy, and memset intervals (`%p` expands PID for multi-process runs). Measure end-to-end throughput separately without CUPTI injection.

---

## 4. Coding Agent & Dashboard Checks

### Dashboard Tests

```sh
# Render Integration page across all 7 languages (open + keyed) without running a server
python tests/apps/flyx-server/check_dashboard_integration.py    # Playwright + PyYAML; --browser msedge for Edge

# Replay complete, interrupted, and error SSE streams in the Chat tab, and its browser-kept history
python tests/apps/flyx-server/check_dashboard_chat.py

# Model page: load settings and what a reload posts, the command line through this system's shells,
# the library (Hugging Face, this machine's folders), a server with nothing loaded
python tests/apps/flyx-server/check_dashboard_model.py
```

### Coding Agent Integration Tests

Verified agents: **Claude Code** (Anthropic Messages), **Codex CLI** (OpenAI Responses), **DeepSeek Harness**, **OpenCode**, and **Pi** (OpenAI Chat Completions), on both Linux/WSL2 and native Windows (loopback open and `--api-key`).

To verify an agent end-to-end:
1. Start `flyx serve MODEL` and copy the client configuration from `http://127.0.0.1:8080/` → **Integration**.
2. In an empty temporary workspace, write a random string to `probe.txt` (keep the string out of the prompt).
3. Ask the agent to read `probe.txt` and return its exact content; verify both directions of the tool protocol in the agent's reply and in `GET /v1/requests`.
4. Note for Claude Code: set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096` if the client defaults to an oversized `max_tokens` for custom local model IDs. For Codex CLI on native Windows, use `-c windows.sandbox=mxc --sandbox read-only` if default shell probing is restricted.
