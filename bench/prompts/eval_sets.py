# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Prompt sets for `flyx bench --prompts` from public evaluation sets: what docs/benchmarks.md's drafter tables ran.

    python -I bench/prompts/eval_sets.py HF_DIR OUT [--n 30] [--think TOKENIZER.json --opener TEXT]

HF_DIR holds the datasets as `hf download --repo-type dataset` writes them, one directory each:
openai/gsm8k (gsm8k/, config main), HuggingFaceH4/MATH-500 (MATH-500/), openai/openai_humaneval
(openai_humaneval/), google-research-datasets/mbpp (mbpp/, config sanitized), HuggingFaceH4/mt_bench_prompts
(mt_bench_prompts/).  OUT gets, each the test split's first N in order, a JSON object a line:
  gsm8k.jsonl, math500.jsonl   {"question": the problem + "\\nPlease reason step by step, and put your final answer
                               within \\\\boxed{}."}
  humaneval.jsonl              {"question": "Write a solution to the following problem and make sure that it passes
                               the tests:\\n```python\\n" + the prompt + "\\n```"}
  mbpp.jsonl                   {"question": the prompt + "\\nYour code should pass these tests:\\n" + its tests}
  mtbench.jsonl                {"question": the first turn}
(flyx bench renders them as one user message, thinking off).  --think: the first four again as token ids with
thinking on, `<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n` + --opener (Qwen3.5/3.8: "<think>\\n", their
template's with thinking on; Qwen3: nothing), named *_think.jsonl.  Needs pyarrow (and tokenizers for --think).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def one(pattern: str) -> str:
    found = sorted(glob.glob(pattern, recursive=True))
    if not found:
        raise SystemExit(f"nothing matches {pattern}")
    return found[0]


def write(out: str, name: str, rows) -> None:
    with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hf")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--think", help="tokenizer.json of the model, for the *_think sets")
    ap.add_argument("--opener", default="", help="what the template writes after the assistant's role, thinking on")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    os.makedirs(a.out, exist_ok=True)
    n, hf = a.n, a.hf
    rows = pq.read_table(one(f"{hf}/gsm8k/main/test-*.parquet")).to_pylist()[:n]
    write(a.out, "gsm8k", [{"question": r["question"] + SUFFIX} for r in rows])
    with open(one(f"{hf}/MATH-500/**/test.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:n]
    write(a.out, "math500", [{"question": r["problem"] + SUFFIX} for r in rows])
    rows = pq.read_table(one(f"{hf}/openai_humaneval/**/test-*.parquet")).to_pylist()[:n]
    write(a.out, "humaneval", [{"question": "Write a solution to the following problem and make sure that it passes "
                                            "the tests:\n```python\n" + r["prompt"] + "\n```"} for r in rows])
    rows = pq.read_table(one(f"{hf}/mbpp/sanitized/test-*.parquet")).to_pylist()[:n]
    write(a.out, "mbpp", [{"question": r["prompt"] + "\nYour code should pass these tests:\n" +
                                       "\n".join(r["test_list"])} for r in rows])
    with open(one(f"{hf}/mt_bench_prompts/**/question.jsonl"), encoding="utf-8") as f:
        qs = [json.loads(line) for line in f][:n]
    write(a.out, "mtbench", [{"question": (q.get("prompt") or q.get("turns"))[0]} for q in qs])
    if a.think:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.think)
        for name in ("gsm8k", "math500", "humaneval", "mbpp"):
            with open(os.path.join(a.out, name + ".jsonl"), encoding="utf-8") as f:
                qs = [json.loads(line)["question"] for line in f]
            write(a.out, name + "_think", [{"ids": tok.encode("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>"
                                                              "assistant\n" + a.opener).ids} for q in qs])
    print(f"wrote {a.out}: gsm8k, math500, humaneval, mbpp, mtbench ({n} each)" + (" and *_think" if a.think else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
