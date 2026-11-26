# Benchmarks

Reproducible long-context coding, `flyx bench` decode, drafter, and Needle-in-a-Haystack measurements for `flyx` on consumer NVIDIA GPUs with exact hardware, models, and commands.

---

## 1. Qwen3.8-27B Long-Context Coding Benchmark

All rows below were measured on the same `release-linux` build (`flyx 0.2.0`, commit `b950329b`, CUDA `13.0`, per-GPU tuning tables) at full native context (`--ctx max` = `262,144` tokens) with built-in **MTP** speculative decoding (plus `MTP + DFlash2` on RTX 5090 `NVFP4`):

- **RTX 5090 (Blackwell native FP4):** [`akopytko/Qwen3.8-27B-NVFP4-GGUF`](https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF) `Qwen3.8-27B-NVFP4-MTP-N4_0.gguf` (`NVFP4`, `15.72 GB` file, `15.20 GB` GPU weights; 504 GGML NVFP4 tensors executing on native `sm_120a` W4A4 FP4 tensor cores in prefill + built-in MTP).
- **24 GB GPUs (RTX 4090, RTX 3090):** [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-Q4_K_XL` (`17.66 GB` file, `17.24 GB` GPU weights).
- **16 GB GPUs (RTX 5080, RTX 5070 Ti, RTX 4080):** `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` (`10.93 GB` file, `10.72 GB` GPU weights).

### Workload & Reproducible Commands

Each prompt embeds a calibration passphrase at 50% depth inside tracked C++ repository sources (`1,022` / `131,081` / `260,049` tokens) and asks the model to recall the passphrase and write a fresh 512-token pure-Python ISO 8601 date/timestamp parser with unit tests (greedy decoding; no repeated Python code in the prompt so prompt-lookup hits stay at 0–3 rounds).

```sh
# 1. Generate deterministic 1K / 128K / 256K prompts from this repository
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 1024   -o prompt_1k.txt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 131072 -o prompt_128k.txt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 260000 -o prompt_256k.txt

# 2. Run greedy coding benchmark at full native 262K context
./bin/flyx run MODEL.gguf \
  --drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos -v -f prompt_128k.txt
```

### Hardware & Automatic 262K Memory Placement

| GPU | Model | VRAM / TDP / Driver | Automatic 262K (`--ctx max`) K/V placement |
|---|---|---|---|
| **RTX 5090** | `NVFP4` (`15.72 GB`) | `32,607 MiB` · `575 W` · driver `615.71.09` | Full `262,144` tokens exact FP8 K/V on GPU (`9.64 GB` VRAM); W4A4 NVFP4 prefill |
| **RTX 5080** | `UD-IQ3_XXS` (`10.93 GB`) | `16,303 MiB` · `360 W` · driver `610.57.04` | `43,008` tokens on GPU (`4.49 GB`) + `219,136` in RAM (`8.55 GB`); exact to `75,776`, then 2-bit StrataFlux + exact hot reads |
| **RTX 5070 Ti** (WSL2) | `UD-IQ3_XXS` (`10.93 GB`) | `16,303 MiB` · `300 W` · driver `610.88` | `2,048` tokens on GPU (`2.42 GB`, `13.51 GB` WSL2/WDDM budget) + `260,096` in RAM (`10.62 GB`); exact to `34,816`, then 2-bit StrataFlux + exact hot reads |
| **RTX 4090** | `UD-Q4_K_XL` (`17.66 GB`) | `24,564 MiB` · `450 W` · driver `590.48.01` | `88,064` tokens on GPU + `174,080` in RAM; exact to `120,832`, then 2-bit StrataFlux + exact hot reads |
| **RTX 4080** | `UD-IQ3_XXS` (`10.93 GB`) | `16,376 MiB` · `340 W` · driver `580.178.04` | `43,008` tokens on GPU (`4.49 GB`) + `219,136` in RAM (`8.55 GB`); exact to `75,776`, then 2-bit StrataFlux + exact hot reads |
| **RTX 3090** | `UD-Q4_K_XL` (`17.66 GB`) | `24,576 MiB` · `300 W` cap · driver `595.58.03` | `92,160` tokens on GPU (`6.14 GB`) + `169,984` in RAM (`6.94 GB`); exact to `124,928`, then 2-bit StrataFlux + exact hot reads |

### Coding Results Across RTX 50 / 40 / 30 Series (Greedy, 512 New Tokens)

| GPU | Model & Drafter | Context (prompt tokens) | Prefill | Prefill (`tok/s`) | Decode (`tok/s`) | `tok/rnd` (`drafts/rnd`) | Needle recall |
|---|---|---|---:|---:|---:|---:|:---:|
| **RTX 5090** (32 GB) | `NVFP4` + MTP | 1K (`1,022`) | **`0.06 s`** | **`17,090.3`** | **350.2** | `5.27` (`6.8`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP | 128K (`131,081`) | **`29.70 s`** | **`4,414.0`** | **228.7** | `5.27` (`7.0`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP | 256K (`260,049`) | **`103.09 s`** | **`2,522.5`** | **164.5** | `4.96` (`6.8`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP & DFlash2 | 1K (`1,022`) | `0.10 s` | `9,883.9` | **365.0** | `5.06` (`6.7`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP & DFlash2 | 128K (`131,081`) | `30.06 s` | `4,361.1` | **273.1** | `5.74` (`7.0`) | pass |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.44 s` | `2,312.7` | **225.9** | `5.06` (`6.2`) | pass |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `100.64 s` | `1,302.4` | **161.0** | `5.27` (`6.7`) | pass |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `283.49 s` | `917.3` | **105.8** | `3.84` (`5.6`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.63 s` | `1,611.2` | **167.4** | `4.48` (`5.0`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `135.09 s` | `970.3` | **104.6** | `5.21` (`6.5`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `382.49 s` | `679.9` | **61.2** | `3.70` (`5.0`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 1K (`1,022`) | `0.37 s` | `2,763.7` | **193.6** | `5.94` (`9.0`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 128K (`131,081`) | `85.10 s` | `1,540.4` | **155.9** | `5.94` (`8.3`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 256K (`260,049`) | `241.20 s` | `1,078.2` | **122.7** | `5.06` (`6.9`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.55 s` | `1,864.3` | **172.3** | `5.11` (`6.1`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `127.00 s` | `1,032.2` | **111.0** | `4.29` (`5.9`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `358.46 s` | `725.5` | **99.0** | `4.69` (`6.4`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 1K (`1,022`) | `1.15 s` | `888.1` | **135.8** | `5.49` (`7.6`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 128K (`131,081`) | `254.71 s` | `514.6` | **97.5** | `5.55` (`7.3`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 256K (`260,049`) | `694.67 s` | `374.3` | **68.1** | `4.41` (`6.4`) | pass |

---

## 2. Standard `flyx bench` Across GPUs & Drafters (Qwen3.8-27B)

Measured on the exact same build (`b950329b`) and machines as Section 1 across `flyx bench`'s 3 fixed mixed prompts (33..79 tokens, 128 new tokens each, greedy `tok/s`; `--ctx 4096` on 16 GB GPUs):

```sh
./bin/flyx bench MODEL.gguf --no-spec -n 128 --json
./bin/flyx bench MODEL.gguf --drafter mtp -n 128 --json
./bin/flyx bench MODEL.gguf --drafter dflash2 --drafter-file z-lab/Qwen3.8-27B-DFlash2 -n 128 --json
./bin/flyx bench MODEL.gguf --drafter all     --drafter-file z-lab/Qwen3.8-27B-DFlash2 -n 128 --json
```

### RTX 5090 (32 GB, `575 W`, driver `615.71.09`): Drafter Modes (`NVFP4`)

| Model | `flyx` plain | `flyx` + MTP | `flyx` + DFlash2 | `flyx` + both | Best Speedup |
|---|---:|---:|---:|---:|---:|
| `NVFP4` (`15.72 GB`) | **94.0** | **282.7** (`4.10 tok/rnd`) | **311.6** (`4.10 tok/rnd`) | **310.3** (`4.28 tok/rnd`) | **3.3x** |

### All 6 Consumer GPUs: Plain vs Built-In MTP (`flyx bench -n 128`)

| GPU | Model | `flyx` plain (`tok/s`) | `flyx` + MTP (`tok/s`) | Speedup |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | **94.0** | **282.7** (`4.10 tok/rnd`) | **3.0x** |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` | **71.0** | **183.9** (`3.97 tok/rnd`) | **2.6x** |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` | **58.6** | **150.4** (`3.93 tok/rnd`) | **2.6x** |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | **49.6** | **154.2** (`4.43 tok/rnd`) | **3.1x** |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` | **53.9** | **140.9** (`3.97 tok/rnd`) | **2.6x** |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` | **41.0** | **108.2** (`4.23 tok/rnd`) | **2.6x** |

Important caveats:
- **Drafted speed depends on workload predictability.** Code and structured output accept more drafts per round (`5.0–5.9 tok/rnd`) than mixed/prose prompts (`3.9–4.4 tok/rnd`).
- **Single-user latency focus:** Concurrent requests share decode rounds (`1.12x` aggregate throughput at 2 requests, `1.30x` at 4 requests on 27B 4-bit), but `flyx` is optimized for single-request latency rather than high-concurrency batching.

---

## 3. Long-Context Attention Fidelity (Needle-in-a-Haystack Suite)

To verify that neither default FP8 K/V nor tiered Host-RAM + 2-bit StrataFlux attention degrades long-range retrieval, we verified single-needle recall (`5%`, `25%`, `50%`, `75%`, `95%` depths across `32K–256K` plus `1K / 128K / 256K` 50%-depth recall on every GPU) and 4-needle simultaneous retrieval at `256K` (`ALPHA` / `BETA` / `GAMMA` / `DELTA` embedded at `10%`, `35%`, `65%`, `90%` depth) on `b950329b`: **51/51 (100%) exact recall across all 6 consumer GPUs, zero failures**.

```sh
# Generate and run the 256K 4-needle NIAH prompt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 260000 --multi-needle -o niah_256k.txt
./bin/flyx run MODEL.gguf --drafter mtp --ctx max --chat --no-think -n 96 -v -f niah_256k.txt
```

| GPU | Model & K/V Attention Mode | Single-Needle Recall | Multi-Needle at 256K (4 keys) | Total |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` · Full 262K GPU exact FP8 | `3/3` (`1K / 128K / 256K` @ `50%`) | `4/4` (`b950329b`) | **7/7 (100%)** |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` · 43K GPU + 219K RAM, 2-bit StrataFlux > 75K | `5/5` (`256K` @ `5%–95%`) | — | **5/5 (100%)** |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` · 2K GPU + 260K RAM, 2-bit StrataFlux > 34K | `5/5` (`256K` @ `5%–95%`) | `4/4` (`b950329b`) | **9/9 (100%)** |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` · 88K GPU + 174K RAM, 2-bit StrataFlux > 120K | `20/20` (`32K–256K` @ `5%–95%`) | `4/4` (`b950329b`) | **24/24 (100%)** |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` · 43K GPU + 219K RAM, 2-bit StrataFlux > 75K | `3/3` (`1K / 128K / 256K` @ `50%`) | — | **3/3 (100%)** |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` · 92K GPU + 170K RAM, 2-bit StrataFlux > 124K | `3/3` (`1K / 128K / 256K` @ `50%`) | — | **3/3 (100%)** |

---

## 4. Benchmarking Tools in This Repo

| Tool | Purpose & Command |
|---|---|
| `flyx bench` | End-to-end decode speed + tokens/round + GPU clocks/power/throttling (`flyx bench MODEL -n 128 --json`). |
| `bench/prompts/make_coding_prompt.py` | Deterministic `1K–256K` long-context coding and 4-needle NIAH prompt generator from tracked repository sources. |
| `bench/ab/ab.sh` | Interleaved A/B comparison across builds or env vars: `bench/ab/ab.sh decode\|prefill MODEL A B`. |
| `bench/drafters_27b.sh` + `bench/drafters_table.py` | Multi-workload drafter comparison across `code`, `cache`, `list` and `reason` prompts. |
| `bench/prompts/` | Fixed prefill prompt fixtures (`p512.txt`, `long.txt`, `prompt_6k.txt`), and the long chat prompts and the prompt sets DV is measured on. |
| `bench/replay/agent_replay.py` | Multi-turn tool-using OpenAI replay measuring recomputed prompt tokens and TTFT p50/p95. |
| `cuda_bench` | Kernel microbenchmarks (`gemm`, `skinny`, `fp4`, `decode`, `tc`, `w4a4`, `w12`, `pick`, `gdn`, `attn`, `flash`, `mlp`, `blocks`, `q8runs`, `rounds`, `strata`, `mma`, `host_read`, `graphs`) and tuning table generator (`cuda_bench tune [FAMILIES] [FILE.gguf...] [NxK...]`; NxK: a model's bf16 GEMMs alone, for a table that inherits the rest). |
| `tree_bench` | Speculative tree verification latency and acceptance: `tree_bench MODEL [--ctx-at P] [--widths 1,15,31,63] [--drafted T]` (every width timed from the same context; trees of up to 63 drafts where the backend lowers them). |
| `flyx bench --prompts` | A prompt set (`{"question"}`, `{"messages"}` or `{"ids"}` a line), every prompt's tokens and rounds in the `--json` record: `flyx bench MODEL --prompts FILE.jsonl [--suffix S] [--limit N] --json OUT.json`; `tests/e2e/check_dv.py` scores and compares such records. |
| `bench/ab/paired_records.sh`, `bench/ab/summarize_records.py`, `bench/ab/same_runs.py` | Two builds' `flyx bench --prompts` runs in turns (waiting for a quiet GPU), their medians, and whether two runs give the same answers and rounds. |
| `bench/ab/sglang_bench.py` | A `flyx bench --prompts` record's prompts, token for token, through an SGLang server: its decode tok/s as flyx counts it, its tokens a round, and how many answers are the record's (section 6). |
| `bench/replay/dv_choice` | DV's per-round choice replayed on recorded rounds (`bench/replay/dv_traces/`), each constant set otherwise: `python -I bench/replay/dv_choice.py table BUILD/bin/dv_choice`. |
| `bench/trace/dv_round_kernels.py` | Where a DV round's time goes, from an `nsys` trace of DV's rounds with graphs off. |
| `bench/trace/` | CUPTI kernel timeline analysis (`gaps.py`, `layer_times.py`, `per_matrix.py`, `prefill_kinds.py`). |

---

## 5. MiniCPM5-1B with DV (RTX 5060 Ti 8 GB)

Hardware: RTX 5060 Ti 8 GB, driver `610.88`, CUDA `13.5`, Windows 11 (native, MSVC), the desktop holding about 3.3 GB of the GPU. Model: `openbmb/MiniCPM5-1B` in BF16, its DV files made from bifill-dv's checkpoints on Hugging Face (`panmz/bifill-dv`, a private repository: without access to it the DV rows below cannot be reproduced; [models](models.md) has the commands, [DV](design/dv.md#the-description-and-the-file-abi-42) the files), binary16 K/V (`--kv 16`). Prompts: GSM8K's first 100 test questions as bifill-dv prompts them (`gsm8k_test.jsonl`, made from the public `openai/gsm8k` as below; the instruction below after each), greedy, at most 512 new tokens. Every DV run generates plain decoding's tokens for all 100 (`tests/e2e/check_dv.py`).

`gsm8k_test.jsonl` is GSM8K's test split, one `{"question", "answer"}` object a line in the dataset's order, as bifill-dv writes it (sha256 `ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13`, its lines ending in CRLF; with LF line ends the same prompts, sha256 `752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac`):

```python
import json
from datasets import load_dataset  # pip install datasets
with open("gsm8k_test.jsonl", "w", encoding="utf-8", newline="\r\n") as f:
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
```

| File | Drafter | Decode `tok/s` | Tokens / round |
|---|---|---:|---:|
| `minicpm5-1b.flyx` | none (`--no-lookup`) | 226.7 | 1.00 |
| `minicpm5-1b-dv-res2.flyx` (beam 31:8:cov) | DV, 31 drafts every round (`--width 31`) | 320.1 | 5.02 |
| `minicpm5-1b-dv-res2.flyx` | DV, rounds chosen per round | **338.7** | 4.30 |
| `minicpm5-1b-dv-b2h.flyx` (with its V-LoRA) | none (`--drafter none --no-lookup`) | 179.2 | 1.00 |
| `minicpm5-1b-dv-b2h.flyx` | DV, 31 drafts every round (`--width 31`) | 387.2 | 7.08 |
| `minicpm5-1b-dv-b2h.flyx` | DV, rounds chosen per round | **426.1** | 6.37 |

Commands:

```sh
S='\nPlease reason step by step, and put your final answer within \boxed{}.'
C="--prompts gsm8k_test.jsonl --kv 16 --ctx 2048 --limit 100"
flyx bench minicpm5-1b.flyx $C --suffix "$S" --no-lookup --json plain.json
flyx bench minicpm5-1b-dv-res2.flyx $C --suffix "$S" --drafter dv --width 31 --json dv31.json
flyx bench minicpm5-1b-dv-res2.flyx $C --suffix "$S" --drafter dv --json dv.json
flyx bench minicpm5-1b-dv-b2h.flyx $C --suffix "$S" --drafter none --no-lookup --json b2h_plain.json
flyx bench minicpm5-1b-dv-b2h.flyx $C --suffix "$S" --drafter dv --width 31 --json b2h31.json
flyx bench minicpm5-1b-dv-b2h.flyx $C --suffix "$S" --drafter dv --json b2h.json
python -I tests/e2e/check_dv.py --drafted dv31.json --plain plain.json --same-target
python -I tests/e2e/check_dv.py --drafted dv.json --plain plain.json --same-target
python -I tests/e2e/check_dv.py --drafted b2h31.json --plain b2h_plain.json --same-target
python -I tests/e2e/check_dv.py --drafted b2h.json --plain b2h_plain.json --same-target
```

`check_dv.py` passes for each DV run against the plain run of its file's target (`plain.json` for the `-dv-res2`
file, `b2h_plain.json` for the `-dv-b2h` file): the same generated tokens for every question.

After long prompts (the three `bench/prompts/summarize_*.txt`, 3110 to 5949 tokens; 200 tokens, a new engine each) `minicpm5-1b-dv-res2.flyx`'s DV of 31 drafts every round gives 1.75 to 1.95 tokens a round and decodes at 0.31 to 0.42 times plain decoding; with its rounds chosen per round it runs plain decoding's and lookup rounds there (1.00 to 1.02 tokens a round), at about plain decoding's speed. Every text is plain decoding's. DV drafts only when named.

```sh
for p in bench/prompts/summarize_*.txt; do
  flyx run minicpm5-1b.flyx -f $p --chat --no-think -n 200 --kv 16 --ctx 8192 --no-spec --no-lookup
  flyx run minicpm5-1b-dv-res2.flyx -f $p --chat --no-think -n 200 --kv 16 --ctx 8192 --drafter dv --width 31
  flyx run minicpm5-1b-dv-res2.flyx -f $p --chat --no-think -n 200 --kv 16 --ctx 8192 --drafter dv
done
```

---

## 6. Block Drafters on Qwen3.8-27B and Qwen3-8B: DSpine, DFlash2, MTP (RTX PRO 6000)

One RTX PRO 6000 Blackwell Server Edition (96 GB, driver `620.70.06`; flyx at the DSpine change for the 27B, with
this GPU's table and DSpine's NVFP4 layer copies below for the 8B, all of the 8B's columns in one run; Release, CUDA
13.4, `FLYX_CUDA_ARCHS=120-real`), greedy, thinking off unless said, up to 1024 new tokens a prompt, each prompt in a
fresh session:

- **Qwen3.8-27B**: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (`4a22293a`, its MTP layer kept), converted once with
  `z-lab/Qwen3.8-27B-DFlash2` (`50307d4c`, block 8) and once with `tencent/Qwen3.8-27B-DSpine-b16` (`260a6c23`).
- **Qwen3-8B**: `Qwen/Qwen3-8B` (`b968826d`, BF16), converted with `tencent/Qwen3-8B-DSpine-b16` (`3b1e69d0`).
- **Prompts**: `bench/prompts/eval_sets.py` (the first 30 of GSM8K, MATH-500, HumanEval, MBPP and MT-Bench's first
  turns; the datasets at `740312ad`, `6e4ed1a2`, `7dce6050`, `4bb6404f`, `e3a795c5`) and
  `bench/prompts/chat_short.jsonl` (24). With thinking on, GSM8K and MBPP as token ids (`--think`): on the 27B the
  first 20 of each, on Qwen3-8B the first 30 (no opener: Qwen3's template writes none).

```sh
python -I bench/prompts/eval_sets.py hf/ sets/
python -I bench/prompts/eval_sets.py hf/ think/ --n 20 --think Qwen3.8-27B-QUASAR-NVFP4/tokenizer.json --opener $'<think>\n'
PYTHONPATH=python/convert python -m flyx_convert --model Qwen3.8-27B-QUASAR-NVFP4 --draft Qwen3.8-27B-DSpine-b16 -o q27-dspine.flyx
flyx bench q27-dspine.flyx --prompts sets/gsm8k.jsonl -n 1024 --json q27_gsm8k.json --drafter dspine --width 15 --no-lookup
# likewise every set, without --width (chosen widths), and --no-spec, --drafter mtp, --drafter dflash2 [--width 7
# --no-lookup] on the DFlash2 file; Qwen3-8B the same way (think8/: --think Qwen3-8B/tokenizer.json, no --opener)
flyx bench q8-dspine.flyx --prompts sets/gsm8k.jsonl -n 1024 --json plain_gsm8k.json --no-spec
```

Decode `tok/s` (tokens a round). Every drafted answer is plain decoding's, token for token, in every configuration
(the 27B: 174 prompts and 40 with thinking on; the 8B: 234).

| Qwen3.8-27B | plain | MTP | DFlash2 | DSpine | DFlash2, 7 drafts | DSpine, 15 drafts |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 81.4 | 310.4 (5.84) | **431.1** (6.83) | 387.1 (6.08) | 417.7 (6.20) | 412.7 (7.51) |
| MATH-500 | 81.7 | 301.2 (5.44) | **422.8** (6.67) | 376.6 (5.90) | 411.6 (6.08) | 382.5 (6.91) |
| HumanEval | 80.7 | 426.1 (10.67) | 593.6 (11.82) | 590.6 (11.82) | 474.4 (7.35) | **635.6** (12.48) |
| MBPP | 80.8 | 278.8 (5.41) | 390.5 (6.22) | 387.7 (6.42) | 392.5 (6.00) | **396.4** (7.47) |
| MT-Bench | 81.5 | 196.8 (3.14) | 228.2 (3.40) | 202.7 (3.02) | **228.5** (3.37) | 175.3 (3.15) |
| chat | 81.6 | 218.1 (3.53) | **263.4** (3.94) | 237.0 (3.57) | 258.4 (3.81) | 205.6 (3.69) |

- A round costs DSpine what it costs DFlash2 at each width (the engine's load-time timings: 14.02 / 14.87 / 18.01 ms
  for 3 / 7 / 15 drafts against 14.14 / 14.90 / 17.73), and DSpine's NVFP4 layer copies beside the NVFP4 target
  accept as its exact weights do (`--backend-opt draft_weights=exact`; GSM8K at 15 drafts: 7.51 against 7.49 tokens
  a round).
- DSpine at its block of 16 accepts more than DFlash2 at its block of 8, as its authors report (GSM8K: 7.51 against
  6.20 tokens a round; with thinking on, `think/gsm8k_think.jsonl`: 6.93 against 5.60, 378.5 against 376.6 tok/s).
  Its first drafts are accepted less often than DFlash2's (the first: 0.912 against 0.948; seven deep: 0.419 against
  0.567), and flyx's DFlash2 also drafts 15 from a wider block (GSM8K, `--width 15 --no-lookup`: 8.26 tokens a round,
  455.2 tok/s): the faster of the two depends on the text, DSpine on code (HumanEval, MBPP), DFlash2 on math and chat.
- Chosen every round, DSpine's widths stay mostly at 7 drafts where 15 give more (GSM8K: 387.1 against 412.7 tok/s):
  a fresh session's acceptance per position starts from the controller's prior (0.8 - 0.02 i), and a reply of a few
  hundred tokens seldom reaches positions past 7 to learn them.

| Qwen3-8B | plain | DSpine | DSpine, 15 drafts | Tencent's plugin (SGLang), 15 drafts |
|---|---:|---:|---:|---:|
| GSM8K | 87.5 | 664.7 (9.28) | **692.1** (9.92) | 659.2 (10.00) |
| MATH-500 | 87.4 | 594.7 (8.32) | **614.5** (8.63) | 592.3 (8.81) |
| HumanEval | 87.5 | 524.9 (7.26) | **555.0** (7.81) | 531.3 (7.85) |
| MBPP | 87.4 | 517.2 (7.28) | **531.9** (7.50) | 513.5 (7.58) |
| MT-Bench | 87.3 | **265.8** (3.51) | 258.5 (3.61) | 242.1 (3.58) |
| chat | 87.4 | 331.3 (4.47) | **331.8** (4.62) | 306.1 (4.57) |
| GSM8K, thinking | 87.3 | 307.6 (4.12) | **312.4** (4.34) | 294.5 (4.39) |
| MBPP, thinking | 87.3 | **272.2** (3.58) | 269.1 (3.73) | 248.5 (3.76) |

On GSM8K DSpine accepts about what its authors report (at 15 drafts 9.92 tokens a round from its NVFP4 layer copies,
10.03 from its exact weights; theirs 10.55), on MT-Bench less (3.61 against 5.63).

**Against Tencent's own implementation.** Their SGLang plugin (`Tencent/DSpine` at `d037a9b`; Python 3.12, CUDA 13.0)
on the same GPU, model and drafter snapshots, the same prompts token for token (each flyx plain record's
`prompt_ids`), greedy, one request at a time, its README's Qwen3-8B server but for the attention backends
(FlashAttention 3 does not run on sm_120):

```sh
pip install sglang==0.5.22 torch==2.14.1 flashinfer-python==0.7.1 && pip install --no-deps -e DSpine
python -m sglang.launch_server --model-path Qwen3-8B --speculative-algorithm DSPINE \
  --speculative-draft-model-path Qwen3-8B-DSpine-b16 --speculative-num-draft-tokens 16 --dtype bfloat16 \
  --attention-backend flashinfer --speculative-draft-attention-backend triton --page-size 1 \
  --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 8 --cuda-graph-bs-decode 1 2 4 8 \
  --trust-remote-code
python -I bench/ab/sglang_bench.py plain_gsm8k.json --json sglang_gsm8k.json
```

At 15 drafts flyx is 3.6-8.4% faster on every set (9.8% on MT-Bench with widths chosen every round), at about the
same tokens a round: its round is shorter. Every flyx answer is plain decoding's; SGLang's speculative answers were
its own plain decoding's in 8 of the 30 GSM8K prompts (`--same-as` a run of the server without `--speculative-*`).
What made the difference on this GPU, measured on GSM8K and HumanEval at 15 drafts (617.6 / 490.8 tok/s before):

- **This GPU's table** (`backends/cuda/variants/sm120/tables/rtxpro6000.toml`, `cuda_bench tune gemm` on the four
  bf16 GEMMs Qwen3-8B runs): the attention out and mlp down projections at 9..16 rows read 1052 and 953 GB/s on the
  RTX 5090's rules, 1255 and 1008 on their own (+2-4%; bit-identical configurations).
- **DSpine's layer copies**: beside a BF16 target the drafter was bf16, its five layers 1.9 GB a round, each layer
  the size of a target layer; Q8_0 copies +6% at the same tokens a round, NVFP4 copies 2-3% more at most 1% fewer.
- **The walk's GEMMs** in one launch of up to 240 rows (+0.5%).

The trace that found them (`CUDA_INJECTION64_PATH=.../libflyx-trace.so FLYX_TRACE=round.tsv flyx bench ...`, then
`python -I bench/trace/gaps.py round.tsv`), before them: a round 15.35 ms, its graph busy 15.2 ms of it, the GEMMs 13.3.
