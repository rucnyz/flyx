# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Checks on a real model (run on the GPU machine): Qwen3.5, or a model without drafters (MiniCPM5-1B).

    python tests/e2e/check_real_model.py --hf models/Qwen3.5-4B --model qwen35-4b.flyx \
        --lib build/bin/libflyx.so [--tokens 128]
    python tests/e2e/check_real_model.py --hf models/MiniCPM5-1B --model minicpm5-1b.flyx \
        --lib build/bin/flyx.dll --drafters --hf-dtype float32          # no drafter; Windows
    python -I tests/e2e/check_real_model.py --hf models/MiniCPM5-1B --model minicpm5-1b-vlora-b2h.flyx \
        --lib build/bin/flyx.dll --drafters --hf-dtype float32 --lora lora_g_b2h.pt   # the file's V-LoRA

1. tokenizer: our C++ tokenizer vs the HuggingFace tokenizer on mixed text,
   alone and as a raw prompt (with the special tokens HF's post-processor
   puts before it: MiniCPM5's <s>)
2. chat template: the engine's rendering of the model's own template vs apply_chat_template
   (text, branch points, open thinking block), for the conversations of
   tests/core/render/chat_cases.py; with tools, refused where the template's
   tool calls are in no format this build reads (format "plain")
3. greedy decoding: engine (plain, and each drafter --drafters names: MTP and
   DFlash2 unless it names others, each one the file must have) vs
   transformers in bfloat16 (--hf-dtype float32: in float32; on the CPU when
   torch has no GPU).
   Two bf16 implementations drift apart eventually; the report shows where the
   first difference happens and how close the top-2 logits were there (a near
   tie is expected noise, a large gap is a bug).  --lora: bifill-dv's V-LoRA
   checkpoint (the one the file holds) on transformers' linears as its
   evalgen.py VLORA runs it: each linear plus B (A x), unmerged, A and B rounded
   to bf16 as it casts them (--lora-scale: raft.py's none, 1).
4. draft acceptance: tokens per round with each drafter (about 1.0 means the drafter is broken)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests", "core", "render"))

import chat_cases  # noqa: E402

TEXTS = [
    "Hello, world! It's a test: 123456 + 7.5e-3 = ?",
    "推理引擎需要又快又准。我们先把正确的路径写出来，再一步步变快。",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n\nprint(fib(30))\n",
    "  leading  spaces\tand\ttabs\r\nCRLF lines\n\n\n  trailing   ",
    "Émoji 🚀✨, café, naïve, e\u0301 combining, 한국어, Ελληνικά, русский, عربى",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!",
    "I'M SURE you've seen they'd done it; THEY'LL do it again — we're told.",
    "https://example.com/path?q=1&r=two#frag  user@example.org  C:\\Windows\\System32",
    # numbers, which MiniCPM5 cuts into groups of three before its split
    "A shop sells 1250 pens at $2.75 and 384 books at $14.10: 1250 * 2.75 + 384 * 14.10 = 8851.9 dollars.",
    "１２３４５ ①②③ ½¾ ٣٤٥ 1,000,000 3.14159265358979 2026-11-17T00:12:59Z 0x7fffffff  12\n\n345",
]

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome, with a short explanation.",
    "用三句话解释什么是投机解码。",
    "List the planets of the solar system in order from the sun, one per line.",
    "A train leaves at 9:40 and travels 255 km at 85 km/h, then waits 25 minutes. When does it arrive? Think step "
    "by step.",
]


def with_lora(hf, ck, scale):
    """bifill-dv's V-LoRA (raft.py: ck['lora'], per layer q, k, v, o, gate, up, down of {'A', 'B'}) on hf's linears as
    its evalgen.py VLORA puts it there (raft.py WithLoRA, unmerged: base(x) + B (A x)), A and B rounded to bf16 as it
    casts them, then to the model's dtype.  Returns the linears wrapped."""
    import torch
    import torch.nn.functional as F

    class WithLoRA(torch.nn.Module):
        def __init__(self, base, A, B):
            super().__init__()
            self.base = base
            self.A = torch.nn.Parameter(A.to(torch.bfloat16).to(base.weight), requires_grad=False)
            self.B = torch.nn.Parameter(B.to(torch.bfloat16).to(base.weight), requires_grad=False)

        def forward(self, x):
            return self.base(x) + scale * F.linear(F.linear(x, self.A), self.B)

    names = (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"), ("self_attn", "o_proj"),
             ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"))
    layers = hf.model.layers
    if len(ck["lora"]) != len(names) * len(layers):
        raise SystemExit(f"--lora: {len(ck['lora'])} adapters, the model's {len(layers)} layers take "
                         f"{len(names) * len(layers)}")
    for i, ab in enumerate(ck["lora"]):
        parent = getattr(layers[i // len(names)], names[i % len(names)][0])
        name = names[i % len(names)][1]
        setattr(parent, name, WithLoRA(getattr(parent, name), ab["A"], ab["B"]))
    return len(ck["lora"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", default=os.path.join(ROOT, "build", "bin", "libflyx.so"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--quick", action="store_true", help="tokenizer and template checks only (no GPU)")
    ap.add_argument("--kv-fp8", action="store_true", help="run the engine with an FP8 K/V cache")
    ap.add_argument("--activations", choices=("exact", "nvfp4"),
                    help="exact: bf16 inputs for every matrix; nvfp4: NVFP4 weights take NVFP4 inputs in prefill "
                         "passes whatever the file says (default: as the file has it; NVFP4 weights take bf16 ones "
                         "by themselves on a GPU without FP4 tensor cores)")
    ap.add_argument("--drafters", nargs="*", choices=("mtp", "dflash2", "dspine"), default=["mtp", "dflash2"],
                    help="drafters held to plain decoding, each one the file must have (GGUF typically has only "
                         "mtp; none named: plain alone, for a model without drafters such as MiniCPM5-1B); plain "
                         "always runs")
    ap.add_argument("--hf-dtype", choices=("bfloat16", "float32"), default="bfloat16",
                    help="transformers' weights and activations")
    ap.add_argument("--lora", help="bifill-dv's V-LoRA checkpoint (raft.py's) the model file holds: on transformers' "
                                   "linears too, unmerged (read with torch.load(weights_only=True))")
    ap.add_argument("--lora-scale", type=float, default=1.0, help="its scale (raft.py's: none, %(default)s)")
    a = ap.parse_args()
    bindir = os.path.dirname(os.path.abspath(a.lib))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import flyx

    tok = AutoTokenizer.from_pretrained(a.hf)
    ok = True

    # 1. tokenizer
    # prompt_ids: what transformers feeds for a raw prompt, the post-processor's special tokens (MiniCPM5's <s>) first.
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False), "prompt_ids": tok.encode(t)} for t in TEXTS]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    r = subprocess.run([os.path.join(bindir, "tokenizer_test"), a.model, f.name], capture_output=True, encoding="utf-8")
    print("[tokenizer]", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr)
    ok &= r.returncode == 0

    # 2. chat template.  A template whose tool calls this build does not read (format "plain") refuses tools.
    formats = set()
    for think in (True, False):
        for name, conv, tools in chat_cases.CASES:
            want = chat_cases.expected(tok, tok.chat_template, conv, tools, think)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump({"messages": conv, "tools": tools} if tools else conv, f)
            cmd = [os.path.join(bindir, "flyx"), "template", a.model, f.name] + ([] if think else ["--no-think"])
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8")
            if r.returncode != 0 and tools and formats == {"plain"} and "tool-call format" in r.stderr:
                print(f"[template] thinking={think} {name}: tools refused (format plain)")
                continue
            got = json.loads(r.stdout) if r.returncode == 0 else {"text": "ERROR " + r.stderr}
            formats.add(got.get("format"))
            same = got["text"] == want["text"]
            points = same and got["boundary_bytes"] == want["boundaries"] and \
                got["in_reasoning"] == want["in_reasoning"]
            ok &= points
            print(f"[template] thinking={think} {name}: {'ok' if points else 'DIFF (branch points)' if same else 'DIFF'}")
            if not same:
                g, w = got["text"], want["text"]
                k = next((i for i in range(min(len(g), len(w))) if g[i] != w[i]), min(len(g), len(w)))
                print("  first difference at char", k)
                print("  want:", repr(w[max(0, k - 80):k + 120]))
                print("  got: ", repr(g[max(0, k - 80):k + 120]))
            elif not points:
                print("  branch points", got["boundary_bytes"], "want", want["boundaries"],
                      "| thinking open", got["in_reasoning"], "want", want["in_reasoning"])

    if a.quick:
        print("ALL OK" if ok else "SOME CHECKS FAILED")
        sys.exit(0 if ok else 1)

    # 3. greedy decoding vs transformers (bf16).  The engine runs first and is
    # freed before transformers loads: both copies of the weights do not fit a
    # 16 GB card, and on WSL an over-full card silently spills to system memory.
    # Penalties on the tokens a reply already has must not make the output
    # depend on the drafter either: greedy with both penalties, and the model card's general
    # setting (sampled, presence 1.5) with one seed.
    PENALIZED = {"greedy, presence 1.5, frequency 0.25": dict(presence_penalty=1.5, frequency_penalty=0.25),
                 "T 0.7, top-p 0.8, top-k 20, presence 1.5": dict(temperature=0.7, top_p=0.8, top_k=20, seed=7,
                                                               presence_penalty=1.5)}
    results, penalized = {}, {}
    activations = a.activations

    def open_engine(spec):
        return flyx.Engine(a.model, lib=a.lib, speculative=bool(spec), max_ctx=4096, drafter=spec or None,
                           kv="fp8" if a.kv_fp8 else "16", activations=activations)

    for spec in (False, *a.drafters):
        eng = open_engine(spec)
        rows = []
        for p in PROMPTS:
            # Render then encode: with tokenize=True, transformers 5 returns a dict
            # (return_dict defaults to True) and 4.x returns a list.
            prompt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            s = eng.session()
            out = s.generate(tokens=ids, max_new_tokens=a.tokens, ignore_eos=True)
            rows.append((ids, out, s.usage.new_tokens, s.usage.rounds, s.usage.decode_ms))
            s.close()
            for name, kw in PENALIZED.items():
                s = eng.session()
                out = s.generate(tokens=ids, max_new_tokens=a.tokens, ignore_eos=True, **kw)
                penalized.setdefault((spec, name), []).append((out, s.usage.new_tokens, s.usage.rounds, s.usage.decode_ms))
                s.close()
        results[spec] = rows
        eng.close()
        del eng
    device ="cuda" if torch.cuda.is_available() else "cpu"
    hf = AutoModelForCausalLM.from_pretrained(a.hf, dtype=getattr(torch, a.hf_dtype)).to(device).eval()
    if a.lora:
        n = with_lora(hf, torch.load(a.lora, map_location="cpu", weights_only=True), a.lora_scale)
        print(f"[greedy] {a.lora}: a V-LoRA on {n} of transformers' linears, unmerged")
    print(f"[greedy] transformers in {a.hf_dtype} on the {device}")
    # The end-of-turn tokens: the model's own, and ChatML's and GPT-2's where the vocabulary has them.
    eos = hf.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    stops |= {i for t, i in tok.get_vocab().items() if t in ("<|im_end|>", "<|endoftext|>")}
    for i, p in enumerate(PROMPTS):
        ids, plain, *_ = results[False][i]
        with torch.no_grad():
            ref = hf.generate(torch.tensor([ids]).to(device), max_new_tokens=a.tokens, do_sample=False,
                              min_new_tokens=a.tokens)[0, len(ids):].tolist()
            # Compare up to the engine's first end-of-turn token: past it the engine
            # (ignore_eos) keeps the stop token while min_new_tokens makes
            # transformers pick the runner-up, so the two must differ there.
            n_cmp = next((k for k, t in enumerate(plain) if t in stops), len(plain))
            first = next((k for k in range(min(len(ref), n_cmp)) if ref[k] != plain[k]), None)
            gap = None
            if first is not None:
                lg = hf(torch.tensor([ids + ref[:first]]).to(device)).logits[0, -1].float()
                top2 = torch.topk(lg, 2).values
                gap = float(top2[0] - top2[1])
        print(f"[greedy] prompt {i}: plain vs transformers "
              + ("identical" if first is None else f"first difference at token {first} (top-2 gap {gap:.3f})")
              + f" over {n_cmp} tokens")
        for d in a.drafters:
            _, spec_out, n, rounds, ms = results[d][i]
            same_spec = spec_out == plain
            ok &= same_spec
            print(f"         {d:8s} vs plain {'identical' if same_spec else 'DIFFERENT'}: "
                  f"{(n - 1) / max(rounds, 1):.2f} tokens/round, {(n - 1) * 1000 / max(ms, 1e-9):.1f} tok/s")
        print("         text:", repr(tok.decode(plain[:60])))
        for name in PENALIZED:
            pen_plain = penalized[(False, name)][i][0]
            print(f"[penalties] prompt {i}, {name}: " + ("the tokens without penalties" if pen_plain == plain
                                                         else "other tokens than without penalties"))
            for d in a.drafters:
                out, n, rounds, ms = penalized[(d, name)][i]
                same = out == pen_plain
                ok &= same
                print(f"         {d:8s} vs plain {'identical' if same else 'DIFFERENT'}: "
                      f"{(n - 1) / max(rounds, 1):.2f} tokens/round, {(n - 1) * 1000 / max(ms, 1e-9):.1f} tok/s")
            print("         text:", repr(tok.decode(pen_plain[:60])))
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
