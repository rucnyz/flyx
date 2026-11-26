# Models, Drafters, and Long-Context Sizing

Reference for supported model architectures (Qwen3.5 / Qwen3.8, Qwen3, Llama), file formats, Hugging Face Hub loading, speculative drafters, and GPU/RAM/disk KV sizing.

---

## 1. Supported Models and Formats

| Format | Supported Architectures | Supported Weight Types | Refused / Unsupported |
|---|---|---|---|
| **GGUF** (native, zero conversion) | `qwen35` (Qwen3.5 / Qwen3.8 hybrid GDN + gated attention, plus multimodal vision via `--mmproj` `qwen3vl_merger` `mmproj-*.gguf`), `qwen3` (dense) | `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`–`Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, mixed quants, GGML `NVFP4`; a split file (`-00001-of-0000N.gguf`) with all its parts in one folder | `MXFP4`, MoE experts, sliding-window target attention, scaled RoPE |
| **`.flyx`** (via `python/convert`) | Qwen3.5 / Qwen3.8, Qwen3, Llama (e.g. MiniCPM5-1B) | `BF16`, NVIDIA ModelOpt `NVFP4` | Vision / MoE; a Llama GGUF |

Key execution properties:
- **Packed GGUF blocks:** GGML block weights remain packed on GPU, decoded in registers, and multiplied by 8-bit quantized activations (matching `llama.cpp` precision; `--activations exact` forces exact activations).
- **Host-resident untied embeddings:** When output head and token embedding are untied (e.g., Qwen3.8-27B's 0.42 GB embedding), the embedding stays in host RAM where under 32 GB of GPU memory are left after the weights, its rows fetched over PCIe at negligible cost; above that it goes to the GPU (`--backend-opt embed=host` or `embed=gpu` chooses).
- **Vision encoder offload and quantization:** Vision weights (`mmproj-*.gguf`, BF16/F16/F32 or pre-quantized GGML blocks) default to `--vision-device disk` + `--vision-quant q8`: floating-point vision linears are quantized on the fly to `Q8_0` (~46% smaller; 0.50 GB on Qwen3.8-27B, KL 0.00125 / 98.75% top-1 vs BF16 over 800 multimodal positions; pre-quantized `mmproj` files stay packed) and written to an auto-deleted temp file flushed clean to OS Standby/Page Cache (**0 bytes resident VRAM, 0 bytes locked host RAM at KV-budget time** — zero impact on `--kv-ram auto` context capacity). During `encode_vision`, weights stream through a 3-stage pipeline (lazy read-only OS page-cache map → 16 MiB lazy pinned host ring → double-buffered GPU staging) with **bit-identical outputs** to GPU-resident execution (~54–64 ms warm ViT on RTX 5070 Ti). Override with `--vision-device host|gpu|auto` (`host`: pinned host RAM; `auto`: GPU when >= 8 GB free after weights, else `disk`) and `--vision-quant exact|nvfp4` (`exact`: file precision; `nvfp4`: ~70% smaller, opt-in).
- **GGUF format details:** See [docs/design/gguf.md](design/gguf.md).
- **Llama (MiniCPM5-1B):** Qwen3's layers without the per-head q/k RMSNorms (`LlamaForCausalLM`), from `.flyx` only (llama.cpp's converter reorders a Llama's q and k rows, so a Llama GGUF is refused). MiniCPM5-1B's pre-tokenizer cuts numbers into groups of three before the Qwen2 split, and its `<s>` goes before every raw prompt as its tokenizer puts it; its chat template renders as transformers renders it (tool calls are not read back yet, so a request with tools is refused). Its K/V cache is binary16 by default: FP8 moved its next-token distributions visibly (KL 0.0064, top-1 97.2%, perplexity +1.4% against binary16), so a Llama keeps binary16 unless `--kv fp8`; Qwen3.5 and Qwen3 keep FP8.
- **`.flyx` format 2:** a file whose header requires a part a reader must not ignore (`"requires": ["lora"]`, the V-LoRA below) is written as format 2, so builds that read only format 1 refuse it instead of decoding the plain model; every other file stays format 1.

---

## 2. Hugging Face Hub Resolution

Any CLI or API `MODEL` or `--drafter-file` argument accepts a local path or a Hugging Face Hub specifier:

```sh
flyx run unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --chat -p "Hello"
flyx serve unsloth/Qwen3.8-27B-GGUF:UD-IQ2_S --drafter-file z-lab/Qwen3.8-27B-DFlash2
```

| Rule | Behavior |
|---|---|
| `org/repo:QUANT` | Selects the GGUF carrying `QUANT` (falls back to `UD-<QUANT>` when unique). Passing `org/repo` without `:QUANT` lists available GGUFs and sizes. |
| `org/repo:QUANT@REV` | Pins a branch or commit hash (also used to address older cached snapshots). |
| Auto MTP sidecar | If a GGUF lacks an embedded MTP layer and the repo contains `MTP/*.gguf` (e.g., `unsloth/Qwen3.8-27B-GGUF`), `flyx` fetches and attaches it automatically. |
| Auto `mmproj` vision projector | If a Hub GGUF repo contains a vision projector (`mmproj-BF16.gguf` preferred over F16/F32), `flyx` fetches and attaches it automatically unless `--no-vision` or an explicit `--mmproj` path is given. |
| Direct block-drafter loading | `z-lab/*-DFlash2`, `vllm-project/*-DFlash2` and `tencent/*-DSpine-b16` safetensors repos as well as llama.cpp `general.architecture=dflash` GGUF drafters (e.g. `z-lab/Qwen3.8-27B-DFlash2-GGUF`) load directly with no conversion step across supported target architectures (`qwen35`, `qwen3`, `llama`). |
| Downloader & cache | 16 parallel range connections, SHA-256 verified against Hub metadata, resumable across restarts, multi-process coordinated via `.flyx-lock` files in standard HF cache (`HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`, `HF_ENDPOINT`, `HF_HUB_OFFLINE=1`). Requires `libcurl` on Linux; native WinHTTP on Windows. |

---

## 3. Speculative Drafters

`flyx` supports five drafter sources and can run multiple simultaneously; a per-round controller (`RoundChooser`) chooses the drafter and width dynamically from load-time measured costs and live acceptance rates.

| Drafter | Source | Notes |
|---|---|---|
| **MTP** (Multi-Token Prediction) | Embedded in GGUF/HF checkpoint, or sidecar `--drafter-file MTP/mtp-*.gguf` | Default when present. Shares target embedding/head. At long contexts (> 32K), uses 4096-token exact ring + 2-bit StrataFlux copy (`mtp_window`, `mtp_copy`). |
| **DFlash2** (block diffusion) | HF safetensors repo (`--drafter-file z-lab/Qwen3.8-27B-DFlash2`), `arch=dflash` GGUF (BF16 or quantized GGML blocks), or converted `.flyx` | For Qwen3.5 / Qwen3.8, Qwen3, and Llama targets (a drafter made for another model is refused). Beside GGUF and BF16 targets, floating-point DFlash2 projections are quantized to Q8_0, beside NVFP4 targets to NVFP4 (pre-quantized GGUF drafters stay packed; `--backend-opt draft_weights=exact` keeps them as given). A GGUF drafter that is causal, samples from the anchor, has a confidence head, soft-capped logits or a Markov context (DSpark), or reads the target's final hidden state, is refused. Strongest on code/structured text; `--drafter dflash2` selects it exclusively. |
| **DSpine** (adjacent causal injection) | HF repo (`--drafter-file tencent/Qwen3.8-27B-DSpine-b16`, `tencent/Qwen3-8B-DSpine-b16`) or converted `.flyx`; for Qwen3.5 / Qwen3.8, Qwen3, and Llama targets (a drafter made for another model is refused) | A DFlash-style block drafter (the same context from target taps) whose layers attend causally within the block and pass a low-rank message from each row to the next (the code bridge); its drafts walk the head's top 16 with the last layer's message redone from the token actually drafted. A round of fewer drafts runs fewer rows. Chains only (no draft trees); its layer projections are quantized as DFlash2's (Q8_0 beside BF16 or GGUF weights, NVFP4 beside NVFP4), its bridge stays BF16. `--drafter dspine` selects it exclusively; a model holds one block drafter (DFlash2 or DSpine). |
| **DV** (bifill-dv's canvas drafter) | A DV `.flyx` file: a Llama with bifill-dv's canvas and selector written in (`flyx_convert --dv`) | Only when named (`--drafter dv` or `all`). A canvas pass of the target's weights with its own adapter over 32 slots after the anchor, then a selector that grows a tree of the file's width (31) or of 15, verified exactly (rounds of up to 64 rows). Its warm-up, its trees of 15 and of 31, plain decoding and lookup rounds are chosen every round by tokens per ms at the context; `--width N` pins it. CUDA backend only; see [DV](design/dv.md). |
| **Prompt Lookup** | Built-in (enabled by default) | Verifies 15 drafts copied from matching 6-token n-grams in the current sequence with zero drafter forward pass. Big speedup on agent code edits/repeats; backs off for 32 tokens after poor rounds (`--no-lookup` disables). |

Control flags:
- `--drafter mtp|dflash2|dspine|dv|all|none`
- `--drafter-file PATH_OR_HF` (repeatable)
- `--width N` (pin model drafter width; lookup remains active unless `--no-lookup`)
- `--draft-trees` (opt-in DFlash2 dynamic trees; off by default)
- `--no-spec` (plain decoding, disables all drafters and lookup)
- `--cycle I,J,...` (testing: a fixed cycle of round indices instead of the controller's choice; `-v` lists the rounds)

---

## 4. Converting Hugging Face Checkpoints to `.flyx`

GGUF files run directly without conversion. Convert HF Safetensors (BF16 or ModelOpt NVFP4) or standalone DFlash2 / DSpine checkpoints to `.flyx` using `python/convert`:

```sh
# Target + DFlash2 combined into one .flyx (BF16 or ModelOpt NVFP4)
hf download Qwen/Qwen3.5-4B --local-dir ~/models/Qwen3.5-4B
hf download vllm-project/Qwen3.5-4B-DFlash2 --local-dir ~/models/Qwen3.5-4B-DFlash2
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3.5-4B --draft ~/models/Qwen3.5-4B-DFlash2 \
    -o ~/models/qwen35-4b.flyx

# ModelOpt NVFP4 checkpoint
hf download AxionML/Qwen3.5-4B-NVFP4 --local-dir ~/models/Qwen3.5-4B-NVFP4
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3.5-4B-NVFP4 --draft ~/models/Qwen3.5-4B-DFlash2 \
    -o ~/models/qwen35-4b-nvfp4.flyx

# Standalone DFlash2 drafter file (reusable alongside any compatible GGUF)
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir ~/models/Qwen3.8-27B-DFlash2
PYTHONPATH=python/convert python -m flyx_convert \
    --draft ~/models/Qwen3.8-27B-DFlash2 -o ~/models/qwen38-27b-dflash2.flyx

# DSpine: beside a Qwen3.5/3.8 or a Qwen3 model (in place of DFlash2), or as a drafter file of its own
hf download tencent/Qwen3-8B-DSpine-b16 --local-dir ~/models/Qwen3-8B-DSpine-b16
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3-8B --draft ~/models/Qwen3-8B-DSpine-b16 -o ~/models/qwen3-8b-dspine.flyx
hf download tencent/Qwen3.8-27B-DSpine-b16 --local-dir ~/models/Qwen3.8-27B-DSpine-b16
PYTHONPATH=python/convert python -m flyx_convert \
    --draft ~/models/Qwen3.8-27B-DSpine-b16 -o ~/models/qwen38-27b-dspine.flyx
```

```sh
# MiniCPM5-1B (Llama without q/k norms), BF16
hf download openbmb/MiniCPM5-1B --local-dir ~/models/MiniCPM5-1B
PYTHONPATH=python/convert python -m flyx_convert --model ~/models/MiniCPM5-1B -o ~/models/minicpm5-1b.flyx

# MiniCPM5-1B with a DV drafter, and with its V-LoRA: bifill-dv's checkpoints on Hugging Face, panmz/bifill-dv,
# a private repository: without access to it these files cannot be made
hf download panmz/bifill-dv canvas/dva_res2.pt selector/sel_orig_w8.pt \
    canvas/dva_cd_b2h.pt selector/sel_b2h_w8.pt v_lora/lora_g_b2h.pt --local-dir ~/models/bifill-dv
B=~/models/bifill-dv
python -I python/convert/export_dv.py --canvas $B/canvas/dva_res2.pt --selector $B/selector/sel_orig_w8.pt \
    --tree beam:31:8:cov -o ~/models/dv-res2-export
PYTHONPATH=python/convert python -m flyx_convert --model ~/models/MiniCPM5-1B \
    --dv ~/models/dv-res2-export -o ~/models/minicpm5-1b-dv-res2.flyx
python -I python/convert/export_dv.py --canvas $B/canvas/dva_cd_b2h.pt --selector $B/selector/sel_b2h_w8.pt \
    --vlora $B/v_lora/lora_g_b2h.pt --tree beam:31:8:cov -o ~/models/dv-b2h-export
PYTHONPATH=python/convert python -m flyx_convert --model ~/models/MiniCPM5-1B \
    --dv ~/models/dv-b2h-export -o ~/models/minicpm5-1b-dv-b2h.flyx
```

A target's own adapters (a **V-LoRA**: LoRA beside the target's linears, e.g. bifill-dv's rank-64 `lora_g_b2h`) run unmerged on every pass, their sums in float32 on the kernels of the linears they sit beside (a bf16 merge loses about 70% of the delta). `export_dv.py --vlora` exports them with or without a DV; `flyx_convert --lora EXPORT` writes a V-LoRA alone. A file with adapters is `.flyx` format 2. Adapters on GGML block weights, on an attention with an output gate, or on an MTP layer are refused. Which checkpoints make which DV file, and what DV is held to: [DV](design/dv.md).

> **WSL2 note:** Store models on the Linux filesystem (`~/models`), not `/mnt/c`.

---

## 5. Long-Context KV Tiering and 2-bit StrataFlux

`flyx` budgets device and host memory before allocating. `-v` prints the exact memory bill at startup.

### Memory Tier Hierarchy

| Tier | Controlled By | What Lives Here | Performance Characteristic |
|---|---|---|---|
| **Tier 1: GPU VRAM** | `--kv fp8\|16`, `--kv-gpu N` | Model weights, 3 GDN round slots, prefix of KV cache up to VRAM budget, hot StrataFlux refined-pair cache (`strata_hot=2048`), StrataFlux copies (as many layers as fit) | Full GPU speed; exact reads for all GPU-resident KV |
| **Tier 2: Pinned Host RAM** | `--kv-ram GB\|auto`, `--host-cache GB` | Overflow KV positions beyond `--kv-gpu`, overflow StrataFlux layer copies (`--kv-copy-gpu`), parked session checkpoints (`--host-cache`, restored in ~0.2 s) | Staged ahead over PCIe concurrently with preceding layers (`stage_dma=1`, `stage_ahead=1`) |
| **Tier 3: Local Disk** | `--kv-disk DIR[:GB]` | Incremental prompt-prefix snapshots as conversations grow + spilled parked checkpoints | Prevents full re-prefill when switching sessions or handling Claude Code title/permission requests |

### KV Read Methods (`--kv-method`)

| Mode | Behavior | Quality vs Speed Trade-off |
|---|---|---|
| `auto` *(default, 2-bit)* | Exact attention while context ≤ `kv_gpu`; switches automatically to 2-bit StrataFlux attention once KV spills to host RAM (`strata_from`). Falls back to `exact` if StrataFlux copies don't fit alongside KV. | Exact while KV fits on GPU; fast at long contexts without scanning full host KV every step. |
| `exact` | Reads every KV position exactly at every step (including host-resident KV over PCIe). | Zero approximation; decode latency scales linearly with host-resident KV length (~0.67 ms/step per 1K host tokens on 27B over PCIe 5 x16). |
| `strata[:2\|3\|4]` | Always scans a 2-, 3-, or 4-bit Walsh-Hadamard summary copy and refines the top attended key pairs (`strata_denom`, `strata_least=32`, `strata_most=256`) from exact FP8/FP16 KV (cached on GPU via `strata_hot=2048`). | 2-bit is default (smallest footprint, fastest 1-warp scan); `:3` or `:4` uses more memory for tighter approximation. |

### Recommended GPU Sizing Recipes (Qwen3.8-27B)

| GPU VRAM | Host RAM | Recommended Quantization & Flags | Typical Capability |
|---|---|---|---|
| **16 GB** (RTX 5070 Ti / 5080 / 4080S) | 32 GB | `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` (10.9 GB) or `UD-IQ2_S` (8.4 GB); `--ctx max --sessions 1 --kv-disk ./flyx-cache:40` | Up to ~200K–311K context with 2-bit StrataFlux + MTP (~89 tok/s at 200K on RTX 5070 Ti); fits 27B comfortably with disk-backed session swaps |
| **24 GB** (RTX 3090 / 4090) | 32–64 GB | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` (17.7 GB) or `UD-IQ3_XXS`; `--ctx max` | Fast GPU-resident context + high-speed 2-bit StrataFlux overflow to host RAM |
| **32 GB** (RTX 5090) | 64+ GB | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` + `--drafter-file z-lab/Qwen3.8-27B-DFlash2 --ctx max` | 256.8 tok/s short-context (both drafters); exact FP8 KV past 256K; 500K+ context with StrataFlux |

> **8–10 GB GPUs (e.g., RTX 3080 10 GB, RTX 4060 / 5060 8 GB):** Cannot fit 27B weights alongside session recurrent state; use smaller models such as `Qwen/Qwen3.5-4B` (BF16/NVFP4/GGUF). On tight 8 GB desktop GPUs (where Windows WDDM reserves VRAM for desktop composition), `flyx` queries NVML free memory and automatically trims speed-only auxiliary copies (e.g., duplicate W12/FP8 head copies) when needed to fit prefill workspace and sessions.
>
> **GPU tuning tables:** Pre-tuned dispatch tables ship under `backends/cuda/variants/` for RTX 3080, RTX 4080 SUPER, RTX 4090, L4, L40S, RTX 5060 Ti, RTX 5070 Ti, RTX 5080, RTX 5090, A100 PCIe 40 GB, and H100 80GB HBM3 (other GPUs inherit their architecture table or defaults).
