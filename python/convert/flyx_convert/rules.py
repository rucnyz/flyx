# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""What the converter's parts share about a checkpoint's tensor names: a block drafter's Qwen3 layers and its fc and
norms as the engine names them, the tensors a drafter carries for the target's sake (left out), a name mapped by
rules, and the check that every tensor the engine reads is there."""
from __future__ import annotations

import re

# A block drafter's (DFlash2's, DSpine's): the context's fc and norms, and each layer's Qwen3 layer.
DRAFT_LAYER_RULES = [
    (r"fc\.weight", "fc"),
    (r"hidden_norm\.weight", "hidden_norm"),
    (r"norm\.weight", "norm"),
    (r"layers\.(\d+)\.input_layernorm\.weight", "L{0}.in_norm"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "L{0}.post_norm"),
    (r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", "L{0}.attn.{1}"),
    (r"layers\.(\d+)\.self_attn\.(q|k)_norm\.weight", "L{0}.attn.{1}_norm"),
    (r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", "L{0}.mlp.{1}"),
]
# A block drafter's copies of what the engine takes from the target (its embedding and head) and training leftovers.
DRAFT_SKIP = re.compile(r"^(embed_tokens\.|lm_head\.|verifier_norm\.|verifier_lm_head\.|t2d|d2t|mask_embedding)")
# An attention layer's tensors as the engine names them.
ATTN_LAYER_KEYS = ("in_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "attn.q_norm", "attn.k_norm",
                   "mlp.gate", "mlp.up", "mlp.down")


def map_name(name: str, rules) -> str | None:
    """The engine's name for a checkpoint's `name` by the first rule whose pattern it matches whole, else None."""
    for pat, out in rules:
        m = re.fullmatch(pat, name)
        if m:
            return out.format(*m.groups())
    return None


def check_tensors(kind: str, need: set[str], seen: set[str]) -> None:
    """Every tensor the engine reads of `kind` was written."""
    missing = need - seen
    if missing:
        raise ValueError(f"missing {kind} tensors: {sorted(missing)[:10]}")
