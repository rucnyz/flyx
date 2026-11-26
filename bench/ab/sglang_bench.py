# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The prompts of a `flyx bench --prompts --json` record, token for token, through an SGLang server: its decode speed
as flyx bench counts it, its tokens a round, and how many of its answers are the record's.

    python -I bench/ab/sglang_bench.py RECORD.json [--url http://127.0.0.1:30000] [--json OUT.json] [--same-as RUN.json]

Each prompt's token ids (the record's prompt_ids, the chat template as flyx rendered it) go to /generate one request
at a time, streamed, greedy, with the record's end tokens and token limit.  Decode tok/s: the tokens after the first
over the time from the first token to the last, summed over the prompts (flyx bench's decode_tok_s).  Tokens a round:
the tokens after the first over the verification passes (meta_info.spec_verify_ct; none without speculation).  Same: the
replies whose tokens are the record's answer; with --same-as, those that are another run's (an earlier --json of the
same record, the server without speculation: whether its speculation kept its own plain decoding's output).
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def generate(url, ids, eos, max_new):
    body = json.dumps({"input_ids": ids, "stream": True,
                       "sampling_params": {"temperature": 0, "max_new_tokens": max_new,
                                           "stop_token_ids": eos}}).encode()
    req = urllib.request.Request(url + "/generate", data=body, headers={"Content-Type": "application/json"})
    first = last = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            j = json.loads(payload)
            if first is None and j.get("meta_info", {}).get("completion_tokens", 0) >= 1:
                first = time.perf_counter()
            last = j
    end = time.perf_counter()
    m = last["meta_info"]
    return {"new": m["completion_tokens"], "decode_s": end - (first or end), "verify": m.get("spec_verify_ct", 0),
            "ids": last.get("output_ids"), "finish": m.get("finish_reason")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", help="a flyx bench --prompts --json record (its prompt_ids, end tokens and limit)")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--json", help="every reply's numbers and tokens")
    ap.add_argument("--same-as", help="an earlier --json of this record: how many replies are its")
    a = ap.parse_args(argv)
    rec = json.load(open(a.record, encoding="utf-8"))
    other = json.load(open(a.same_as, encoding="utf-8"))["replies"] if a.same_as else None
    eos, max_new = rec["eos_ids"], rec["max_new_tokens"]
    rows = []
    for p in rec["prompts"]:
        r = generate(a.url, p["prompt_ids"], eos, max_new)
        r["same"] = r["ids"] == p["ids"]
        rows.append(r)
        print(f"prompt {len(rows):3d}: {r['new']:5d} tokens, {r['decode_s'] * 1e3:8.1f} ms after the first"
              f"{', same' if r['same'] else ''}", file=sys.stderr)
    tok = sum(max(0, r["new"] - 1) for r in rows)
    sec = sum(r["decode_s"] for r in rows)
    verify = sum(r["verify"] for r in rows)
    out = {"record": a.record, "url": a.url, "prompts": len(rows), "decode_tok_s": tok / sec if sec else 0.0,
           "tokens_per_round": tok / verify if verify else None, "same_as_record": sum(r["same"] for r in rows),
           "record_decode_tok_s": rec.get("decode_tok_s")}
    if other is not None:
        out["same_as_run"] = sum(r["ids"] == o["ids"] for r, o in zip(rows, other))
    print(json.dumps(out))
    if a.json:
        json.dump({**out, "replies": rows}, open(a.json, "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
