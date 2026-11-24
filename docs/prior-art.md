# Prior Art and Third-Party Code

`flyx` is written from scratch in C++/CUDA. Vendored libraries live intact under `third_party/` with their original licenses (see [NOTICE](../NOTICE)); the one adapted kernel is listed under *Adapted Code*.

---

## 1. Shipped Third-Party Code (`third_party/`)

| Library | Version / Commit | License | Usage & Local Changes |
|---|---|---|---|
| `cpp-httplib` | `0.20.0` | MIT | HTTP server/client ([local header note](../third_party/httplib/README.flyx.md)) |
| `nlohmann/json` | `3.11.3` | MIT | JSON parsing and serialization (`third_party/nlohmann/LICENSE.MIT`) |
| `utf8proc` | `2.9.0` | MIT + Unicode | Unicode normalization for BPE (`third_party/utf8proc/LICENSE.md`) |
| `llama.cpp` Jinja engine | commit `f7591a6` | MIT | Chat template execution (`common/jinja`, `common/json`, `common/unicode`; [local changes](../third_party/jinja/README.flyx.md)) |
| `ggml` quantization header | commit `f7591a6` | MIT | `ggml-common.h` block layouts and codebooks, unchanged ([note](../third_party/ggml/README.flyx.md)) |
| `XGrammar` C++ core | commit `8f9a127` | Apache-2.0 | Grammar compiler & matcher + `picojson` (BSD-2-Clause) + `dlpack` (Apache-2.0), unchanged ([note](../third_party/xgrammar/README.flyx.md)) |
| `Preact` + `htm` | `10.26.4` / `3.1.1` | MIT / Apache-2.0 | Zero-build dashboard UI components, unchanged ([Preact note](../third_party/preact/README.flyx.md), [htm note](../third_party/htm/README.flyx.md)) |
| `stb_image` | `v2.34` | MIT / Public Domain | In-memory PNG/JPEG/BMP/GIF/TGA decoding for multimodal vision inputs (`third_party/stb/LICENSE`), unchanged ([note](../third_party/stb/README.flyx.md)) |

### Adapted Code

| Source | License | Where |
|---|---|---|
| `basalt` (Jesús Gulfo) `src/prefill/fp4_dense.cu`, itself adapted from Infernix's `nvfp4_a4_tma_kernel` (Ian Ranson and the Infernix contributors) | MIT / Apache-2.0 | `backends/cuda/kernels/gemm_w4a4.cu` (`tma4`): the W4A4 prefill GEMM's TMA pipeline (three-stage mbarrier ring, a producer warpgroup beside eight consumer warps); operands and epilogue are flyx's |

### Marks of Other Products

The dashboard names coding agents and Hugging Face by their own marks (`apps/flyx-server/dashboard/index.html`, the `i-agent-*` and `i-hf` symbols). They are trademarks of their owners, shown only to name those clients and services; `flyx` is not affiliated with any of them.

| Mark | Drawing taken from |
|---|---|
| Claude Code, DeepSeek, Pi | [Simple Icons](https://github.com/simple-icons/simple-icons) `16.38.0` (`claudecode`, `deepseek`, `pi`; icon data CC0-1.0), which cites `code.claude.com`, `www.deepseek.com` and `pi.dev/favicon.svg` |
| Hugging Face | [Hugging Face Brand Assets](https://huggingface.co/brand) (`huggingface/brand-assets`, `hf-logo-pirate.svg`, transparent hollow interior for light-theme tabs) |
| Codex | [Lobe Icons](https://github.com/lobehub/lobe-icons) static SVG `1.104.0` (`codex`; MIT) |
| OpenCode | its repository's `packages/identity/mark.svg` (commit `1251a87`), brought to a 24-unit box |

Test-only data (not shipped in releases): Qwen3.5-4B, Qwen3-4B and MiniCPM5-1B chat templates (`Apache-2.0`, `tests/core/render/templates/`). `gguf` Python package is a test-only dependency.

---

## 2. Algorithmic and Architectural References

### Models and Speculative Drafters
- **Qwen3.5 / Qwen3** (Qwen team): target architectures; float64 reference (`python/reference/flyx_ref/target.py`) verified against HuggingFace `transformers`.
- **MiniCPM5** (OpenBMB, `openbmb/MiniCPM5-1B`, Apache-2.0): a Llama, the attention without q/k norms; the tiny Llama's golden tokens and `tests/e2e/check_real_model.py` compare with `transformers`' modeling code; its `tokenizer.json` states the pre-tokenizer (numbers in groups of three, then a Qwen2-like split) that `core/tokenizer` writes by hand; its chat template is a test fixture (`tests/core/render/templates/`). `llama.cpp`'s converter is why a Llama GGUF is refused: it reorders a Llama's q and k rows for its interleaved rotary pairs.
- **DV** (bifill-dv, the research code DV comes from, and an earlier internal CUDA port of DV): a canvas of N MASK slots after the anchor run by the target's base weights plus a rank-256 LoRA adapter with its own norms, a learned MASK embedding and a residual context (the slot's previous top-16 expected embedding); a bilinear path selector over each slot's top 16 (DFlash2-style); trees grown as a beam per depth (DYN, EAGLE-2-style) or taken from a fixed pool of rank tuples that leave the selector's best path once (ST); exact tree verification; and the target's own LoRA (LoRA: Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, 2021; V-LoRA: bifill-dv's `raft.py` `WithLoRA`, rank 64 beside each of a layer's seven linears, unmerged, no scale, run on every row of the target and never on the canvas's). `python/reference/flyx_ref/dv.py` follows bifill-dv's `evalgen.py` mode DV from its design and math, held to bifill-dv's own functions (`tests/python/reference/check_dv_bifill.py`); `ops/draft/ref.h` and `ops/canvas/ref.h` are its pieces in float64. The converter (`python/convert/export_dv.py`, `flyx_convert --dv`) is ported from the earlier CUDA port's (its exports read the same, byte for byte), `models/dv.h` follows its description in what it holds on flyx's own blocks and rounds, and the CUDA kernels (`backends/cuda/kernels/dv.cu`) take their design from its kernels (the pool tree scored depth by depth in one block, the selector's token tables built once, the canvas slid on the device). The canvas pass's grouped GEMMs (a base weight with its adapter's `a`, the adapters' `b` side by side) run on flyx's own multi-tile kernel. GSM8K (Cobbe et al., *"Training Verifiers to Solve Math Word Problems"*, 2021; `openai/grade-school-math`, MIT) as bifill-dv prompts and scores it (its answer extraction and accuracy rule, and its count of forwards an answer needs) is what `tests/e2e/check_dv.py` holds DV to.
- **DFlash / DFlash2** (z-lab, *"Block Diffusion for Flash Speculative Decoding"*; `vllm-project/speculators` MIT definition): block drafting conditioned on target hidden taps, grouped dynamic causal convolutions, low-rank candidate selector, vLLM inference-time tap/anchor layout, and `llama.cpp`'s GGUF layout for the drafter (`general.architecture=dflash` keys and tensor names). DFlash itself (z-lab's `DFlashDraftModel`, e.g. `z-lab/Qwen3-8B-DFlash-b16`) runs as DFlash2's block without the convolutions and selector, read against the math of z-lab's published `dflash.py` (none of its code is used).
- **DSpine** (Zhang et al., *"Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding"*, 2026; Tencent's `tencent/Qwen3-8B-DSpine-b16` and `tencent/Qwen3.8-27B-DSpine-b16`, Apache-2.0): DFlash's block drafting with causal attention inside the block, a code bridge after every layer that passes a gated low-rank message from each position to the next, and the last write redone with the drafted token's code to choose among the head's top 16. `python/reference/flyx_ref/dspine.py` and `ops/dspine/ref.h` are written from the paper and the released checkpoints' configs, read against the math of the authors' SGLang and MLX implementations (none of their code is used); the walk's every-sender scores and their chain are flyx's own kernels (`backends/cuda/kernels/dspine.cu`).
- **Prompt Lookup & Suffix Decoding**: Saxena (*"Prompt Lookup Decoding"*, 2023), `llama.cpp` lookup, Oliaro et al. (*"SuffixDecoding"*, 2024), LZ77 overlapping copy semantics, and `YASAKANI` (`yasakani-dev/yasakani`) for the copy cursor with local re-anchoring past an edit and wide copy verification passes gated by full-acceptance runs with time-based back-off.
- **Speculative Trees**: Miao et al. (*"SpecInfer"*, 2023), Cai et al. (*"Medusa"*, 2024, typical acceptance), Li et al. (*"EAGLE-2"*, 2024, best-first dynamic tree), Chen et al. (*"Sequoia"*, 2024, cost-budgeted trees), Nguyen et al. (*"Turning Up the Heat"*, 2024, min-p), and hybrid tree analyses (STree, TreeWY, Bole, LumoTree).

### Kernels, Quantization, and Attention
- **GGUF & GGML Quants**: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), `llama.cpp` metadata/tensor naming and K-quant/IQ block definitions (I. Kawrakow and ggml authors), packed block GEMMs (`mmvq`, `mmq`, `dp4a`, `get_int_from_table_16`), host-resident input embeddings, and `llama.cpp`'s precision floor (`fattn-mma-f16`, `fattn-vec`, split-bf16 chunked GDN `gated-delta-net-mma.cu`, FP4 prefill on Blackwell).
- **Stream-K & Parallel Reductions**: Osama et al. (*"Stream-K: Work-centric Parallel Decomposition for Dense Matrix Multiplication on the GPU"*, 2023) and CUDA `threadFenceReduction` counter-and-fence pattern (adapted with shape-fixed K partitions for bitwise invariance).
- **Small-Row Tensor-Core Block GEMMs**: `YASAKANI` (`yasakani-dev/yasakani`) for lane-ordered weight loads and warp-split K reductions without shared memory (adapted to GGML blocks with fixed 32-stream trees).
- **FlashAttention & Chunked GDN**: Dao (*"FlashAttention-2"*, 2023), GQA head packing (FlashAttention-3 / FlashInfer), SageAttention2 (Zhang et al., 2024, for the finding that Ada `sm_89` `mma.f32.e4m3.e4m3.f32` accumulates in 22 bits rather than float32—motivating our binary16 tensor-core path for `rows_attention_q=fp8` on `sm_89`), Yang et al. (*"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"*, NeurIPS 2024; *"Gated Delta Networks"*, ICLR 2025).
- **Tiered KV & Sparse/Low-Bit Attention**: `llama.cpp` `--no-kv-offload`, FlexGen (Sheng et al., 2023), vLLM swap space, LMCache, SGLang HiCache, Quest (Tang et al., 2024), InfiniGen (Lee et al., 2024), ShadowKV (Sun et al., 2024), StrataFlux (2026), and NVIDIA CUDA Virtual Memory Management APIs.

### Runtime, Serving, Chat, and Tooling
- **Engine & Loop Design**: `llama.cpp` loadable backends (`GGML_BACKEND_DL`), context checkpoints, `test-backend-ops`, and pinned prefetch loader; `vLLM` / `ExLlamaV2` / `MLC-LLM` single-loop stepped execution; `velox-inferer` Profile/Backend/Variant split; `lithos-metal` token ring.
- **Chat Templates & Grammars**: `llama.cpp` template format detection and input special-token marking; Ollama / vLLM / SGLang Jinja-verified native renderers; XGrammar structural tags; `llama.cpp` optimistic post-sample grammar checking (extended to full speculative rounds); OpenAI Responses API & SDK item/event schemas.
- **Hub, Monitor & Build**: `llama.cpp` `-hf org/name:quant` syntax; `huggingface_hub` cache layout and env vars; `hf_transfer` multi-range downloads; FIPS 180-4 SHA-256; `Strata` (`Niko1221/Strata`) terminal monitor, telemetry snapshot, and conversation snapshot distinction; `Basalt` (`jesdga95/basalt`) built-in server status page; `sccache` / `GGML_CCACHE` auto-detection.
