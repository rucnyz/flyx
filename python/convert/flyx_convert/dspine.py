# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A DSpine drafter (tencent/Qwen3-8B-DSpine-b16, tencent/Qwen3.8-27B-DSpine-b16) -> ``draft.`` tensors.

Its checkpoint is DFlash's layout (``dflash_config``; target_layer_ids the target layers whose outputs it reads)
with the code bridge under ``selected_prefix_bridge.``: the layers' read projections stacked (the last one apart),
their write projections stacked, one codebook and the gate's three projections.  The engine takes them per layer
(``draft.L<j>.bridge.read`` [code, hidden], ``draft.L<j>.bridge.write`` [hidden, rank]) and once (``draft.bridge.``
codebook [vocab, code], gate / pred / prev [rank, code], gate_bias [rank]); the stacked tensors are cut along their
first dimension, nothing else changes.  The variant both released drafters are (python/reference/flyx_ref/dspine.py)
is the one supported: causal blocks, a write after every layer, the refinement of the last write by the drafted
token's code, no terminal write, no convolutions; anything else is refused, never read as something it is not.
"""
from __future__ import annotations

import numpy as np

from .rules import ATTN_LAYER_KEYS, DRAFT_LAYER_RULES, DRAFT_SKIP, check_tensors, map_name

BRIDGE = "selected_prefix_bridge."
# Tensors that only say which variant the checkpoint is: each must hold 1.
VERSIONS = ("draft_block_causal_version", "last_write_refine_version", BRIDGE + "independent_write_version",
            BRIDGE + "terminal_disabled_version")
_BRIDGE_NAMES = {"codebook": "bridge.codebook", "gate.weight": "bridge.gate", "gate.bias": "bridge.gate_bias",
                 "predecessor_gate.weight": "bridge.pred", "prev.weight": "bridge.prev"}
LAYER_KEYS = ATTN_LAYER_KEYS + ("bridge.read", "bridge.write")


def is_dspine(sc: dict) -> bool:
    """A DSpine drafter's config: DFlash's layout with a code bridge."""
    return "DSpineDraftModel" in sc.get("architectures", []) or \
        isinstance(sc.get("dflash_config", {}).get("selected_prefix_code_bridge"), dict)


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"DSpine drafter: {what}")


def canonical_dspine_config(sc: dict, target: dict | None) -> dict:
    """The drafter's config as the engine reads it ("kind": "dspine"), after checking it is the supported variant.
    taps: the residual stream after that many target layers (target_layer_ids + 1).  window: each position attends
    to the positions less than that far before it (0: all).  topk: the candidates a row's walk chooses among."""
    dc = sc.get("dflash_config")
    _require(isinstance(dc, dict), "no dflash_config")
    layers = int(sc["num_hidden_layers"])
    _require(sc.get("model_type", "qwen3") == "qwen3", f"model_type {sc.get('model_type')!r} (qwen3 layers)")
    _require(sc.get("hidden_act", "silu") == "silu", f"hidden_act {sc.get('hidden_act')!r}")
    _require(not sc.get("attention_bias", False), "attention biases")
    _require(dc.get("draft_block_causal") is True, "draft_block_causal must be true (the block attends causally)")
    _require(dc.get("selected_prefix_terminal") is False, "a terminal write (selected_prefix_terminal)")
    _require(dc.get("selected_prefix_decode_mode") == "serial",
             f"decode mode {dc.get('selected_prefix_decode_mode')!r} (serial)")
    _require(dc.get("selected_prefix_feature_writes") is True and
             dc.get("selected_prefix_feature_write_mode") == "independent", "independent writes after the layers")
    _require(dc.get("selected_prefix_feature_layers") == list(range(1, layers + 1)),
             "a write after every layer (selected_prefix_feature_layers 1..layers)")
    _require(not dc.get("selected_prefix_terminal_residual_rank"), "a terminal residual")
    _require(not dc.get("selected_prefix_feature_next_token_rows", False), "next-token rows (DSpark's layout)")
    _require(dc.get("selected_prefix_feature_reliability") is None, "feature reliability weights")
    _require(dc.get("attention_value_scale") is None, "an attention value scale")
    for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k", "projector_type",
                "attention_sink_bias"):
        _require(not dc.get(key), f"{key} (DFlash2's parts, which a DSpine drafter has none of)")
    widths = []
    refine = dc.get("selected_prefix_last_write_refine")
    if refine is not None:
        _require(refine.get("kind") == "exact_sender" and int(refine.get("version", 0)) == 1,
                 f"last-write refinement {refine!r} (exact_sender, version 1)")
        widths.append(int(refine["head_topk"]))
    second = sc.get("code_bridge_second_pass")
    if second is not None:
        _require(second.get("mode") == "serial_table" and int(second.get("first_exact", -1)) == layers - 1,
                 f"second pass {second!r} (serial_table, the last layer's write)")
        widths.append(int(second["head_topk"]))
    _require(len(set(widths)) == 1, f"refinement widths {widths} (one, given at least once)")
    topk = widths[0]
    _require(2 <= topk <= 16, f"refinement width {topk} (2..16)")
    bridge = dc.get("selected_prefix_code_bridge")
    _require(isinstance(bridge, dict), "no selected_prefix_code_bridge")
    rp = sc.get("rope_parameters") or {}
    _require(rp.get("rope_type", rp.get("type", "default")) == "default" and sc.get("rope_scaling") is None,
             "rope scaling (default rotary only)")
    types = sc.get("layer_types") or ["full_attention"] * layers
    if all(t == "full_attention" for t in types):
        _require(not sc.get("use_sliding_window", False), "use_sliding_window without sliding layers")
        window = 0
    else:
        _require(all(t == "sliding_attention" for t in types) and sc.get("use_sliding_window") and
                 int(sc.get("max_window_layers", layers)) == layers and int(sc.get("sliding_window") or 0) > 0,
                 f"layer types {sorted(set(types))}: all full, or all sliding with one window")
        window = int(sc["sliding_window"])
    taps = [int(i) + 1 for i in dc["target_layer_ids"]]
    vocab, hidden = int(sc["vocab_size"]), int(sc["hidden_size"])
    _require(int(sc.get("draft_vocab_size") or vocab) == vocab, "a draft vocabulary of its own")
    _require(not sc.get("tie_word_embeddings", False), "tied embeddings (it reads the target's head)")
    if target is not None:
        _require(vocab == target["vocab"], "its vocabulary differs from the target's")
        _require(hidden == target["hidden"], "its width differs from the target's")
        _require(all(1 <= t <= target["layers"] for t in taps), f"taps {taps} outside the target's layers")
    hd = sc.get("head_dim") or hidden // sc["num_attention_heads"]
    theta = rp.get("rope_theta", sc.get("rope_theta"))
    _require(theta is not None, "no rope_theta")
    return {
        "kind": "dspine", "hidden": hidden, "layers": layers, "n_heads": sc["num_attention_heads"],
        "n_kv": sc["num_key_value_heads"], "head_dim": hd, "inter": sc["intermediate_size"],
        "eps": sc["rms_norm_eps"], "rope_theta": float(theta), "block": dc.get("block_size", sc.get("block_size")),
        "mask_token": dc["mask_token_id"], "taps": taps, "window": window,
        "code": int(bridge["code_size"]), "rank": int(bridge["rank"]), "bridge_eps": float(bridge["eps"]),
        "topk": topk,
        "target": {"hidden": hidden, "vocab": vocab,
                   "layers": target["layers"] if target else sc.get("num_target_layers")},
    }


def dspine_tensors(names, read, d: dict, log):
    """(name, dtype, array) of the engine's tensors, from the checkpoint's `names` read by read(name) -> (array,
    dtype); its version tensors checked, and the terminal projection it does not use checked to be zero."""
    layers = d["layers"]
    out = []
    for name in names:
        inner = name[len("model."):] if name.startswith("model.") else name
        if DRAFT_SKIP.match(inner):
            log(f"draft: ignoring {name} (the engine uses the target's embedding and head)")
            continue
        if inner in VERSIONS:
            arr, _ = read(name)
            _require(arr.size == 1 and int(arr.reshape(-1)[0]) == 1, f"{inner} {arr.reshape(-1)[:1]} (1)")
            continue
        if inner.startswith(BRIDGE):
            part = inner[len(BRIDGE):]
            arr, dt = read(name)
            if part == "out.weight":  # the terminal write's projection: none in a drafter without one
                _require(not np.any(arr.view(np.uint16) & 0x7FFF if dt == "bf16" else arr),
                         "a terminal projection that is not zero")
                continue
            if part == "read_weight":
                out.append((f"L{layers - 1}.bridge.read", dt, arr))
            elif part == "early_read_weights":
                _require(arr.shape[0] == layers - 1, f"early_read_weights {list(arr.shape)} ({layers - 1} layers)")
                out += [(f"L{j}.bridge.read", dt, arr[j]) for j in range(layers - 1)]
            elif part == "feature_write_weights":
                _require(arr.shape[0] == layers, f"feature_write_weights {list(arr.shape)} ({layers} layers)")
                out += [(f"L{j}.bridge.write", dt, arr[j]) for j in range(layers)]
            elif part in _BRIDGE_NAMES:
                out.append((_BRIDGE_NAMES[part], dt, arr))
            else:
                raise ValueError(f"unknown DSpine bridge tensor {name!r}")
            continue
        canon = map_name(inner, DRAFT_LAYER_RULES)
        if canon is None:
            raise ValueError(f"unknown draft tensor {name!r}")
        arr, dt = read(name)
        out.append((canon, dt, arr))
    return out


def check_dspine(d: dict, seen: set[str]) -> None:
    need = {"fc", "hidden_norm", "norm", *_BRIDGE_NAMES.values()}
    for j in range(d["layers"]):
        need |= {f"L{j}.{n}" for n in LAYER_KEYS}
    check_tensors("draft", need, seen)
