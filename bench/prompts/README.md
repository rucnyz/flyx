# Benchmark Prompts

Fixed prefill prompt fixtures for reproducible `flyx` benchmarks and A/B runs.

| File | Bytes | Tokens (Qwen3.5) | MD5 | Purpose |
|---|---:|---:|---|---|
| `p512.txt` | 2,200 | 498 | `0feb2c531ef910b0ce2e6a94f07341d6` | Random common English words (~512 rows, comparable to `llama-bench -p 512`) |
| `long.txt` | 26,549 | 6,000 | `fd80b763ee1f42a62976a274b3c68fdc` | Random common English words (~6,000 rows) |
| `prompt_6k.txt` | 20,889 | 5,760 | `a2715ef912616413e720d5ad6d59bb5f` | Technical architecture prose fixture |
| `make_coding_prompt.py` | script | `1K–256K` | — | Deterministic long-context coding & NIAH prompt generator built from tracked repo sources |

Long chat prompts for DV's measurements after long prompts (`flyx run MODEL -f FILE --chat --no-think`, the file one
user message): `Summarize this document in a few paragraphs.`, a blank line, then an earlier draft of one of flyx's
design documents, byte for byte as the measurements ran them (they are prompt text, not documentation: what they say
of decisions, records and commits is part of the text).

| File | Bytes | Tokens (MiniCPM5-1B, chat) | MD5 | Purpose |
|---|---:|---:|---|---|
| `summarize_tree_rows.txt` | 8,996 | 3,110 | `a6cf941ccf82466c7a43cddaf65c039f` | A long chat prompt (a measurement write-up) |
| `summarize_llama_qk_norms.txt` | 10,850 | 3,211 | `ca428a283c9604fc59ed401b56b9a963` | A long chat prompt (a design note on Llama) |
| `summarize_tree_verification.txt` | 21,824 | 5,949 | `8a2d294caf119bc9dae992730a886c73` | A long chat prompt (the tree verification design) |

A prompt set for `flyx bench MODEL --prompts FILE` (a JSON object a line, `{"question": ...}`: one user message,
thinking off):

| File | Bytes | Prompts | MD5 | Purpose |
|---|---:|---:|---|---|
| `chat_short.jsonl` | 2,197 | 24 | `abd29de8b71aa54b54b3f3a6f611d768` | Short chat requests unlike GSM8K (code, explanations, stories, poems, a translation, an email, lists): where a drafter trained on math answers drafts little |

`eval_sets.py` writes the sets [benchmarks](../../docs/benchmarks.md)' drafter tables (DSpine, DFlash2, MTP) ran,
from public evaluation sets downloaded with `hf download`: the first 30 of GSM8K, MATH-500, HumanEval, MBPP and
MT-Bench's first turns, as questions (`python -I bench/prompts/eval_sets.py HF_DIR OUT`), and with `--think` the
first four as token ids with thinking on.

`dv_sets.py` writes the sets DV's per-round choice was tuned on from GSM8K's test split (`gsm8k_test.jsonl`, not
kept here: the public `openai/gsm8k`, config `main`, split `test`, one `{"question", "answer"}` object a line in the
dataset's order, `json.dumps(..., ensure_ascii=False)`, lines ending in CRLF as bifill-dv's `data/gsm8k_test.jsonl`,
sha256 `ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13`; [benchmarks](../../docs/benchmarks.md) has
the recipe), byte for byte, each checked against its MD5:
`python -I bench/prompts/dv_sets.py gsm8k_test.jsonl OUT` writes `fs4.jsonl` and `fs10.jsonl` (GSM8K as a 4- and
10-shot multi-turn chat), `gsm30.jsonl` (the first 30 questions), `chat_then_gsm.jsonl` (`chat_short.jsonl`, then those
30) and `gsm_long.jsonl` (the 30 with the summarize prompts after every tenth).

## Usage

```sh
flyx run MODEL -f bench/prompts/p512.txt -n 1 --no-spec -v
PROMPT_FILE=bench/prompts/prompt_6k.txt bash bench/ab/ab.sh prefill MODEL ~/bin-a ~/bin-b
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 131072 -o prompt_128k.txt
```
