# Working on flyx

Mandatory rules for anyone modifying this repository (humans and coding agents alike). They keep every fast path provably correct and every benchmark reproducible.

## Directory Layering Rules

| Layer | Decides | Never |
|---|---|---|
| `core/` (plain C++, no vendor headers) | Sessions, prompt reuse, GPU thread scheduling, prefill chunks and checkpoints, round launch timing, displayed tokens, stop strings, sampling parameters, memory budget, Jinja prompt rendering, post-round grammar enforcement | Includes CUDA; knows which kernels exist; knows a model family's chat format |
| `core/interfaces/` | Shared interfaces between core, models, and backends | Holds an implementation |
| `ops/<op>/` | Op semantics: contract and float64 reference | Depends on anything outside `core/interfaces/` and `include/` |
| `models/<arch>/` | Model description (weights, per-sequence state, typed blocks with dimensions/flags), execution program (block order), and `.flyx`/GGUF tensor names and layout flags | Binds to a GPU; emits tokens; rewrites tensors |
| `models/drafters/` | Drafters the architectures share (DFlash2, DFlash, DSpine; MTP's block from an architecture's hooks): which drafter a file or a companion holds, its config checked against the target, its blocks and state, its tap and context nodes | Knows an architecture's layers or its tensor names; reads a drafter's config before it is chosen, beyond what says which model it was made for |
| `models/formats/` | Bidirectional chat formats: reply parsing (thinking, tool calls), format detection from template text, reply grammars (`tool_choice`, JSON schemas), and exact native renderers verified against Jinja | Renders a template it was not written for byte-for-byte |
| `backends/<name>/` | Block execution: kernels, fusion, weight layouts, CUDA graphs, rollback mechanics | Knows which model it is; alters token acceptance beyond program rules; allocates outside `seq_bytes` / reported memory |
| `backends/cuda/variants/` | Kernel configuration selection per shape, row count, and GPU | Changes math (tables only pick bit-identical configs or fix one K partition per shape) |
| `apps/` | Wire formats (CLI, OpenAI / Anthropic HTTP, dashboard) | Touches device state directly |
| `python/` | Weight conversion, float64 reference, `ctypes` binding | Runs inside the C++ engine |

`tools/check_layering.py` (ctest `layering`) enforces include boundaries: `core` depends only on itself and `include/`; `ops` on `core/interfaces/` and `include/`; `models` on `core/interfaces/` and `core/modelfile/`; `backends` on `core/interfaces/`, `ops/`, and `include/`; `apps` tie layers together. Tests mirror source layout under `tests/<source-path>/`, with end-to-end tests in `tests/e2e/`.

## Correctness Gates (All Required Before Landing)

1. **Every op against float64.** Each CUDA kernel is tested against its `ops/<op>/ref.h` (or `ops/sampling/sampling.h`) in `tests/backends/cuda/op_test_<family>.cu` on seeded inputs (`FLYX_OPS_SEED` varies seeds). If a kernel rounds inputs (e.g., bf16 activations for NVFP4 weights), the reference applies the identical rounding first; tolerances cover accumulation order only. Add or update its row in `docs/design/numerics.md` in the same change (`tools/check_numerics.py` enforces coverage).
2. **Fused == generic.** Every fused region override must match the generic implementation; golden tests verify both (`fusion=0`).
3. **Whole-model verification.** `golden_test` covers tiny BF16 and NVFP4 models on both backends (plain, MTP, DFlash2, DFlash, DSpine, forced long accepted runs, rollback, checkpoint restore, edited prompts, sampling, stop strings) plus tiny Qwen3 against `transformers` (adding a model must not require backend changes). `tests/e2e/check_real_model.py` validates real checkpoints against `transformers` and confirms all drafters reproduce plain decoding; DV's runs against plain decoding are `tests/e2e/check_dv.py`'s.
4. **Speculation never changes output.** Greedy decoding produces identical tokens with and without drafters. Sampled decoding produces identical tokens for a fixed seed. Any optimization making a row depend on concurrent row count (e.g., variable K split or reduction order) violates this gate and is forbidden.
5. **No obvious quality drop.** On the same model file, default rounding (inputs, caches, state, sums) must be as fine as `llama.cpp`'s default CUDA path on the same GPU, unless real-model measurements (KL divergence and top-1 agreement against the finer rounding) show no visible drop. Opt-in user flags are exempt. `docs/design/numerics.md` tracks the comparison place by place.

## Verification Commands

```sh
cmake --build --preset dev && ctest --preset dev --output-on-failure
```

Individual static checks (also run by `ctest`):

```sh
python tools/check_layering.py
python tools/check_numerics.py
python tools/check_docs.py
```

For kernel or model-path changes, also run the real-model and HTTP checks (`tools/quickstart.sh` steps `check` and `serve`).

## Performance Work

- Base every speed claim on reproducible measurements stating model, quantization, GPU, driver, prompt set, and exact command (`flyx bench` / `cuda_bench`).
- Generate per-GPU tuning tables with `cuda_bench tune`, verify end-to-end equivalence and speedups, and commit them under `backends/cuda/variants/`.

## Code and Licensing Standards

- Every source file begins with:
  `Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0`
- Ideas from other projects are credited in `docs/prior-art.md`; do not copy external code, tuned constants, or proprietary formats into our sources. Third-party libraries are vendored whole under `third_party/` with their license, listed in `NOTICE` and `docs/prior-art.md`, with any local changes documented in `README.flyx.md` beside them.
- Prefer clear names over prose comments; use comments to explain invariants and why.
- Keep commits single-topic, tests passing, with messages stating what changed and why.
