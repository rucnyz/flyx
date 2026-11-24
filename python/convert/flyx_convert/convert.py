# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""HuggingFace Qwen3.5 (+ optional DFlash2 or DSpine drafter), Qwen3 or Llama (+ optional DV) -> one .flyx file.

    python -m flyx_convert --model Qwen3.5-4B/ --draft Qwen3.5-4B-DFlash2/ -o qwen35-4b.flyx
    python -m flyx_convert --draft Qwen3.8-27B-DFlash2/ -o qwen38-27b-dflash2.flyx   # a drafter file
    python -m flyx_convert --draft Qwen3.8-27B-DSpine-b16/ -o qwen38-27b-dspine.flyx  # a DSpine drafter file
    python -m flyx_convert --model MiniCPM5-1B/ -o minicpm5-1b.flyx                     # Llama, no drafter
    python -m flyx_convert --model MiniCPM5-1B/ --dv minicpm5-dv/ -o minicpm5-1b-dv-res2.flyx  # Llama with DV
    python -m flyx_convert --model MiniCPM5-1B/ --lora minicpm5-vlora/ -o minicpm5-1b-vlora.flyx  # its V-LoRA

A drafter file holds a block drafter (DFlash2, DFlash or DSpine, dspine.py) alone: the engine
takes it beside a model of any format, a GGUF included (--drafter-file).

A DV drafter (bifill-dv's canvas and selector, docs/design/dv.md) comes from
the directory python/convert/export_dv.py writes, for a Llama (dv.py); the
target's own adapters (a V-LoRA, lora.py) from export_dv.py --vlora's, for a
Llama or a Qwen3: with --dv, the V-LoRA of the DV's directory when it has one
(the canvas and selector were made for that target), with --lora one alone.

Qwen3 (dense) and Llama (MiniCPM5: Llama's layers without q/k norms) convert
the same way; Qwen3's Qwen2 pre-tokenizer split, Qwen3.5's and
MiniCPM5's (numbers in groups of three first) are supported
(tokenizer_export.py).

Weights are copied as stored (BF16 stays BF16).  Tensor names become the
engine's canonical names; the only data change is a row reorder of the
Qwen3.5 attention query projection so that all query heads come before all
gate heads.  Unknown tensors are an error, not a silent skip.  The model's
own multi-token-prediction layer (``mtp.*``), when present, is kept as a
second drafter (``mtp.`` names).  The header's ``arch`` names the model
definition the engine reads it with (models/<arch>).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np

from .dspine import canonical_dspine_config, check_dspine, dspine_tensors, is_dspine
from .dv import add_dv
from .fileformat import NUMPY, Writer
from .lora import add_lora, has_lora
from .tokenizer_export import export as export_tokenizer

ST_DTYPES = {"BF16": "bf16", "F16": "f16", "F32": "f32", "F64": "f64", "I32": "i32", "I64": "i64", "U8": "u8",
             "F8_E4M3": "f8e4m3"}


class SafeTensors:
    """Minimal safetensors reader (raw bytes; no dtype conversion)."""

    def __init__(self, directory: str):
        self.files = sorted(glob.glob(os.path.join(glob.escape(directory), "*.safetensors")))
        if not self.files:
            raise FileNotFoundError(f"no *.safetensors in {directory}")
        self.index = {}
        self._handles: dict[str, object] = {}
        for path in self.files:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for name, rec in hdr.items():
                if name != "__metadata__":
                    self.index[name] = (path, 8 + n + rec["data_offsets"][0],
                                        rec["data_offsets"][1] - rec["data_offsets"][0],
                                        ST_DTYPES[rec["dtype"]], rec["shape"])

    def names(self):
        return list(self.index)

    def read(self, name: str):
        path, off, size, dtype, shape = self.index[name]
        f = self._handles.get(path)
        if f is None:
            f = open(path, "rb")
            self._handles[path] = f
        f.seek(off)
        buf = f.read(size)
        return np.frombuffer(buf, dtype=NUMPY[dtype]).reshape(shape), dtype

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _plain_rope(tc: dict) -> dict:
    """The rotary's parameters, refused unless plain: the engine turns position p by p theta^(-2i/d), unscaled.
    transformers reads the scaling's kind from `rope_type` or from the older `type` ({"type": "linear", "factor":
    4} in many Llama-family configs), in `rope_parameters` or the older `rope_scaling`: each is checked, and a
    factor other than 1 is refused whatever its kind is called."""
    given = [rp for rp in (tc.get("rope_parameters"), tc.get("rope_scaling")) if rp]
    for rp in given:
        kind = rp.get("rope_type") or rp.get("type") or "default"
        if kind != "default":
            raise ValueError(f"unsupported rope type {kind}")
        if rp.get("factor") not in (None, 1, 1.0):
            raise ValueError(f"scaled rotary (factor {rp['factor']}) is not supported")
    return given[0] if given else {}


def _common_transformer_config(tc: dict, *, default_tie: bool, default_max_pos: int) -> tuple[dict, dict, int]:
    rp = _plain_rope(tc)
    hd = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
    if tc.get("attention_bias"):
        raise ValueError("attention bias is not supported")
    base = {
        "hidden": tc["hidden_size"], "layers": tc["num_hidden_layers"],
        "vocab": tc["vocab_size"], "inter": tc["intermediate_size"],
        "n_heads": tc["num_attention_heads"], "n_kv": tc["num_key_value_heads"], "head_dim": hd,
        "rope_theta": float(rp.get("rope_theta", tc.get("rope_theta", 10000.0))),
        "eps": tc["rms_norm_eps"],
        "tie_embed": bool(tc.get("tie_word_embeddings", default_tie)),
        "max_pos": tc.get("max_position_embeddings", default_max_pos),
    }
    return base, rp, hd


def canonical_target_config(tc: dict) -> dict:
    base, rp, hd = _common_transformer_config(tc, default_tie=True, default_max_pos=262144)
    if not tc.get("attn_output_gate", True):
        raise ValueError("models without the attention output gate are not supported")
    lt = tc.get("layer_types") or [
        "full_attention" if (i + 1) % tc.get("full_attention_interval", 4) == 0 else "linear_attention"
        for i in range(tc["num_hidden_layers"])]
    return {
        **base,
        "layer_types": ["attn" if t == "full_attention" else "gdn" for t in lt],
        "rot_dim": int(hd * rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))),
        "lin_nk": tc["linear_num_key_heads"], "lin_nv": tc["linear_num_value_heads"],
        "lin_dk": tc["linear_key_head_dim"], "lin_dv": tc["linear_value_head_dim"],
        "conv_k": tc["linear_conv_kernel_dim"],
    }


# What some Llama-family checkpoints scale beyond Llama's math (MiniCPM 1 to 4's muP, Granite's multipliers): the
# engine's Llama has none of it.
_LLAMA_SCALES = ("scale_emb", "scale_depth", "dim_model_base", "embedding_multiplier", "residual_multiplier",
                 "logits_scaling", "attention_multiplier")


def canonical_dense_config(tc: dict) -> dict:
    """Qwen3 dense and Llama: every layer attention (no output gate), plain RMSNorm weights; Qwen3 normalises each
    head's q and k before the rotary, Llama does not (models/llama)."""
    base, _, hd = _common_transformer_config(tc, default_tie=False, default_max_pos=40960)
    if tc.get("mlp_bias"):
        raise ValueError("MLP bias is not supported")
    if tc.get("use_sliding_window"):
        raise ValueError("sliding-window attention is not supported")
    if tc.get("hidden_act", "silu") != "silu":
        raise ValueError(f"activation {tc.get('hidden_act')!r} is not supported (silu)")
    scaled = [k for k in _LLAMA_SCALES if tc.get(k) not in (None, 1, 1.0)]
    if scaled:
        raise ValueError(f"scaled embeddings, residuals or logits are not supported ({', '.join(scaled)})")
    return {
        **base,
        "layer_types": ["attn"] * tc["num_hidden_layers"],
        "rot_dim": hd,
    }


def arch_of(hf_cfg: dict) -> str:
    t = _text_config(hf_cfg).get("model_type") or hf_cfg.get("model_type", "")
    if t == "qwen3":
        return "qwen3"
    if t in ("llama", "minicpm5"):  # MiniCPM5 publishes LlamaForCausalLM checkpoints
        return "llama"
    if t.startswith("qwen3_5") or t.startswith("qwen3_next"):
        return "qwen35"
    raise ValueError(f"model type {t!r} is not supported (qwen3_5, qwen3, llama)")


def canonical_draft_config(sc: dict, target: dict | None) -> dict:
    """A block drafter's config: a DSpine drafter's (dspine.py), or a DFlash2 drafter's, from either way it is
    published:
      speculators' ("speculators_model_type": "dflash2"; the layers under "transformer_layer_config";
        aux_hidden_state_layer_ids: indices into the target's hidden states, 0 the embedding);
      z-lab's and Inco's ("architectures": ["DFlash2DraftModel"], "dflash_config"; target_layer_ids: the target
        layers whose outputs it reads, hidden_states[id + 1] in their code);
    or a DFlash drafter's, what DFlash2 grew from (z-lab's "architectures": ["DFlashDraftModel"], block_size beside
    dflash_config): kind "dflash", DFlash2's block without convolutions (conv_taps 0) or selector (sel_rank 0: each
    row's draft the head's best).
    taps: the residual stream after that many target layers.  window: each position attends to the positions less
    than that far from it, both ways (0: all).  target: the model's canonical config, or None for a drafter file of
    its own (the engine checks it against the model it is given with; "target" below names what it was made for)."""
    if is_dspine(sc):
        return canonical_dspine_config(sc, target)
    dflash = False  # DFlash: no convolutions, no selector
    if "dflash_config" in sc:
        archs = sc.get("architectures", [])
        dflash = "DFlashDraftModel" in archs
        if "DFlash2DraftModel" not in archs and not dflash:
            raise ValueError(f"drafter architectures {archs!r}: DFlash2DraftModel, DFlashDraftModel and DSpine's are "
                             "supported")
        dc, tl = sc["dflash_config"], sc
        later = sorted(k for k in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k") if k in dc)
        if dflash and later:
            raise ValueError(f"a DFlashDraftModel with DFlash2's {later}: which drafter it is is unclear")
        taps = [int(i) + 1 for i in dc["target_layer_ids"]]
        vocab, target_hidden = tl["vocab_size"], tl["hidden_size"]
        causal = sc.get("is_causal")  # their attention: None = causal where the layer slides
    else:
        if sc.get("speculators_model_type") != "dflash2":
            raise ValueError(f"drafter type {sc.get('speculators_model_type')!r} is not supported (need dflash2)")
        if sc.get("sample_from_anchor", False):
            raise ValueError("sample_from_anchor=True is not supported")
        dc, tl = sc, sc["transformer_layer_config"]
        taps = [int(i) for i in sc["aux_hidden_state_layer_ids"]]
        vocab = sc.get("draft_vocab_size", tl["vocab_size"])
        target_hidden = sc.get("target_hidden_size") or tl["hidden_size"]
        causal = False
    rp = tl.get("rope_parameters") or tl.get("rope_scaling") or {}
    types = tl.get("layer_types") or ["full_attention"] * tl["num_hidden_layers"]
    if all(t == "full_attention" for t in types):
        window = 0
    elif all(t == "sliding_attention" for t in types) and tl.get("sliding_window"):
        window = int(tl["sliding_window"])
    else:
        raise ValueError(f"draft layer types {sorted(set(types))}: all full or all sliding with one window")
    if causal is None:
        causal = window > 0
    if causal:
        raise ValueError("causal DFlash2 layers are not supported (the block attends both ways)")
    if target is not None:
        if vocab != target["vocab"]:
            raise ValueError("drafter vocabulary differs from the target's")
        if target_hidden != target["hidden"]:
            raise ValueError("drafter target_hidden_size does not match the model")
        if any(not 1 <= t <= target["layers"] for t in taps):
            raise ValueError(f"drafter taps {taps} outside the target's {target['layers']} layers")
    hd = tl.get("head_dim") or tl["hidden_size"] // tl["num_attention_heads"]
    if dflash:
        conv = {"conv_taps": 0, "conv_group": 0, "sel_rank": 0, "sel_topk": 1}
    else:
        conv = {"conv_taps": dc.get("conv_kernel_size", 2), "conv_group": dc.get("conv_group_size", 16),
                "sel_rank": dc.get("selector_rank", 256), "sel_topk": dc.get("selector_top_k", 16)}
    return {
        "kind": "dflash" if dflash else "dflash2", "hidden": tl["hidden_size"], "layers": tl["num_hidden_layers"],
        "n_heads": tl["num_attention_heads"], "n_kv": tl["num_key_value_heads"], "head_dim": hd,
        "inter": tl["intermediate_size"], "eps": tl["rms_norm_eps"],
        "rope_theta": float(rp.get("rope_theta", tl.get("rope_theta", 10000.0))),
        "block": dc["block_size"] if "block_size" in dc else sc["block_size"], "mask_token": dc["mask_token_id"],
        "taps": taps, **conv, "window": window,
        "target": {"hidden": target_hidden, "vocab": vocab,
                   "layers": target["layers"] if target else sc.get("num_target_layers")},
    }


_T_RULES = [
    (r"embed_tokens\.weight", "embed"),
    (r"norm\.weight", "norm"),
    (r"layers\.(\d+)\.input_layernorm\.weight", "L{0}.in_norm"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "L{0}.post_norm"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight", "L{0}.gdn.qkv"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_z\.weight", "L{0}.gdn.z"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_b\.weight", "L{0}.gdn.b"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_a\.weight", "L{0}.gdn.a"),
    (r"layers\.(\d+)\.linear_attn\.conv1d\.weight", "L{0}.gdn.conv"),
    (r"layers\.(\d+)\.linear_attn\.A_log", "L{0}.gdn.A_log"),
    (r"layers\.(\d+)\.linear_attn\.dt_bias", "L{0}.gdn.dt_bias"),
    (r"layers\.(\d+)\.linear_attn\.norm\.weight", "L{0}.gdn.norm"),
    (r"layers\.(\d+)\.linear_attn\.out_proj\.weight", "L{0}.gdn.out"),
    (r"layers\.(\d+)\.self_attn\.q_proj\.weight", "L{0}.attn.q"),
    (r"layers\.(\d+)\.self_attn\.k_proj\.weight", "L{0}.attn.k"),
    (r"layers\.(\d+)\.self_attn\.v_proj\.weight", "L{0}.attn.v"),
    (r"layers\.(\d+)\.self_attn\.o_proj\.weight", "L{0}.attn.o"),
    (r"layers\.(\d+)\.self_attn\.q_norm\.weight", "L{0}.attn.q_norm"),
    (r"layers\.(\d+)\.self_attn\.k_norm\.weight", "L{0}.attn.k_norm"),
    (r"layers\.(\d+)\.mlp\.gate_proj\.weight", "L{0}.mlp.gate"),
    (r"layers\.(\d+)\.mlp\.up_proj\.weight", "L{0}.mlp.up"),
    (r"layers\.(\d+)\.mlp\.down_proj\.weight", "L{0}.mlp.down"),
]
_D_RULES = [
    (r"fc\.weight", "fc"),
    (r"hidden_norm\.weight", "hidden_norm"),
    (r"norm\.weight", "norm"),
    (r"layers\.(\d+)\.input_layernorm\.weight", "L{0}.in_norm"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "L{0}.post_norm"),
    (r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", "L{0}.attn.{1}"),
    (r"layers\.(\d+)\.self_attn\.(q|k)_norm\.weight", "L{0}.attn.{1}_norm"),
    (r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", "L{0}.mlp.{1}"),
    (r"layers\.(\d+)\.attention_conv\.base_kernel", "L{0}.attn_conv.base"),
    (r"layers\.(\d+)\.attention_conv\.kernel_projection\.weight", "L{0}.attn_conv.proj"),
    (r"layers\.(\d+)\.mlp_conv\.base_kernel", "L{0}.mlp_conv.base"),
    (r"layers\.(\d+)\.mlp_conv\.kernel_projection\.weight", "L{0}.mlp_conv.proj"),
    (r"candidate_selector\.predecessor_codebook", "sel.pred"),
    (r"candidate_selector\.successor_codebook", "sel.succ"),
    (r"candidate_selector\.hidden_projection\.weight", "sel.hid"),
]
_MTP_RULES = [
    (r"fc\.weight", "fc"),
    (r"pre_fc_norm_embedding\.weight", "norm_emb"),
    (r"pre_fc_norm_hidden\.weight", "norm_hidden"),
    (r"norm\.weight", "norm"),
] + [(pat, out) for pat, out in _T_RULES if pat.startswith("layers")]
# Present in some checkpoints, intentionally not used by the engine.
_T_SKIP = re.compile(r"^(model\.visual\.|visual\.)")
_MTP_PREFIX = re.compile(r"^(model\.)?mtp\.")
_D_SKIP = re.compile(r"^(embed_tokens\.|lm_head\.|verifier_norm\.|verifier_lm_head\.|t2d|d2t|mask_embedding)")


def _map(name: str, rules) -> str | None:
    for pat, out in rules:
        m = re.fullmatch(pat, name)
        if m:
            return out.format(*m.groups())
    return None


def _read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_optional_config(path: str) -> dict:
    return _read_config(path) if os.path.exists(path) else {}


def _quant_config(model_dir: str, hf_cfg: dict) -> dict | None:
    """NVFP4 checkpoints from NVIDIA ModelOpt (hf_quant_config.json or quantization_config), or compressed-tensors'
    nvfp4-pack-quantized (QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4): every quantized linear NVFP4, groups of 16."""
    q = hf_cfg.get("quantization_config")
    p = os.path.join(model_dir, "hf_quant_config.json")
    if q is None and os.path.exists(p):
        q = _read_config(p).get("quantization")
    if not q:
        return None
    if (q.get("quant_method") or "").lower() == "compressed-tensors":
        groups = list((q.get("config_groups") or {}).values())
        wq = [g.get("weights") or {} for g in groups]
        if q.get("format") != "nvfp4-pack-quantized" or not groups or any(
                x.get("num_bits") != 4 or x.get("type") != "float" or x.get("group_size") != 16 for x in wq):
            raise ValueError(f"compressed-tensors format {q.get('format')!r} is not supported (nvfp4-pack-quantized is)")
        # inputs in 4 bits as well (input_global_scale): calibrated for NVFP4 inputs, as ModelOpt's below
        a4 = all((g.get("input_activations") or {}).get("num_bits") == 4 for g in groups)
        return {"format": "nvfp4", "activations": "nvfp4" if a4 else "bf16", "producer": "compressed-tensors"}
    algo = (q.get("quant_algo") or "").upper()
    if algo != "NVFP4":
        raise ValueError(f"quantization {algo or q.get('quant_method')!r} is not supported (NVFP4 from ModelOpt is)")
    # ModelOpt's NVFP4 is W4A4: the checkpoint is calibrated for NVFP4 inputs (input_scale), and the engine gives
    # its prefill passes those.
    return {"format": "nvfp4", "activations": "nvfp4", "producer": (q.get("producer") or {}).get("name", "modelopt")}


# ModelOpt NVFP4 linear: NAME.weight (u8, two e2m1 per byte), NAME.weight_scale
# (e4m3, one per 16), NAME.weight_scale_2 (f32 global), NAME.input_scale (the
# calibrated scale of the FP4 inputs; this engine takes a scale per row instead).
# compressed-tensors' the same as NAME.weight_packed, NAME.weight_scale, NAME.weight_global_scale (1 / ModelOpt's
# weight_scale_2) and NAME.input_global_scale.
_QUANT_SUFFIX = re.compile(r"^(.*)\.(weight_scale_2|weight_scale|input_scale|weight_global_scale|input_global_scale)$")
_PACKED = ".weight_packed"


def _strip_target_prefix(name: str) -> str | None:
    for p in ("model.language_model.", "language_model.model.", "model."):
        if name.startswith(p):
            return name[len(p):]
    return None


def _reorder_q(w: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """HF rows are [h0: q(d), gate(d) | h1: ...]; engine wants [all q | all gates]."""
    w = w.reshape(n_heads, 2, head_dim, -1)
    return np.ascontiguousarray(w.transpose(1, 0, 2, 3)).reshape(2 * n_heads * head_dim, -1)


def convert(model_dir: str, out: str, draft_dir: str | None = None, log=print, activations: str | None = None,
            dv_dir: str | None = None, lora_dir: str | None = None) -> None:
    """activations: what the file says its NVFP4 weights' inputs are in prefill passes ("nvfp4", "bf16"); None: what
    the checkpoint is calibrated for (ModelOpt's NVFP4: nvfp4).  dv_dir: a DV drafter (export_dv.py's directory),
    with the V-LoRA it holds; lora_dir: a V-LoRA alone (export_dv.py --vlora's)."""
    hf_cfg = _read_config(os.path.join(model_dir, "config.json"))
    tc = _text_config(hf_cfg)
    arch = arch_of(hf_cfg)
    cfg = canonical_target_config(tc) if arch == "qwen35" else canonical_dense_config(tc)
    if arch == "llama" and draft_dir:
        raise ValueError("a block drafter is only supported for Qwen3.5 (DFlash2, DFlash, DSpine) and Qwen3 (DFlash, "
                         "DSpine)")
    if arch == "qwen3" and draft_dir:
        dcfg = _read_config(os.path.join(draft_dir, "config.json"))
        if not is_dspine(dcfg) and "DFlashDraftModel" not in dcfg.get("architectures", []):
            raise ValueError("a DFlash2 drafter is only supported for Qwen3.5 (a Qwen3 model takes a DFlash or DSpine "
                             "drafter)")
    if arch != "llama" and dv_dir:
        raise ValueError("a DV drafter is only supported for Llama (MiniCPM5)")
    if lora_dir and has_lora(dv_dir):
        raise ValueError(f"two V-LoRAs: {dv_dir}'s and {lora_dir}'s")
    lora_dir = lora_dir or (dv_dir if has_lora(dv_dir) else None)
    if lora_dir and arch not in ("llama", "qwen3"):
        raise ValueError("a V-LoRA is only supported for Llama and Qwen3")
    w = Writer()
    quant = _quant_config(model_dir, hf_cfg)
    if activations is not None:
        if activations not in ("nvfp4", "bf16"):
            raise ValueError(f"activations {activations!r}: nvfp4 or bf16")
        if quant is None:
            raise ValueError("activations: the model has no NVFP4 weights")
        quant["activations"] = activations
    seen = set()
    mtp_seen = set()
    with SafeTensors(model_dir) as st:
        for name in st.names():
            if _T_SKIP.match(name):
                continue
            qm = _QUANT_SUFFIX.match(name)
            if qm:  # scales are written with their weight below
                if not quant:
                    raise ValueError(f"{name}: quantization scales in a model without a quantization config")
                continue
            # compressed-tensors' NVFP4 codes: named as the weight they are
            mapped = name[: -len(_PACKED)] + ".weight" if name.endswith(_PACKED) else name
            m = _MTP_PREFIX.match(mapped) if arch == "qwen35" else None
            if m:
                canon = _map(mapped[m.end():], _MTP_RULES)
                if canon is None:
                    raise ValueError(f"unknown MTP tensor {name!r}")
                _add_linear(w, st, name, "mtp." + canon, cfg)
                mtp_seen.add(canon)
                continue
            if name == "lm_head.weight":
                if cfg["tie_embed"]:
                    continue
                canon = "lm_head"
            else:
                inner = _strip_target_prefix(mapped)
                canon = _map(inner, _T_RULES) if inner is not None else None
            if canon is None:
                raise ValueError(f"unknown target tensor {name!r}")
            _add_linear(w, st, name, "target." + canon, cfg, reorder_q=arch == "qwen35")
            seen.add(canon)
    _check_target(cfg, seen, qk_norm=arch != "llama")
    mtp = None
    if mtp_seen:
        n_mtp = int(tc.get("mtp_num_hidden_layers", 1))
        _check_mtp(n_mtp, mtp_seen)
        mtp = {"layers": n_mtp}

    draft = _add_draft(w, draft_dir, cfg, log) if draft_dir else None

    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok_info, blobs = export_tokenizer(tok_path)
    for k, v in blobs.items():
        w.add_blob(k, v)
    tok_cfg = _read_optional_config(os.path.join(model_dir, "tokenizer_config.json"))
    template = _chat_template(model_dir, tok_cfg)
    if template:
        w.add_blob("chat_template.jinja", template.encode())
    added = {a["content"]: a["id"] for a in json.loads(blobs["tokenizer.added"])}
    eos = _eos_ids(model_dir, tc, added)
    dv = add_dv(w, dv_dir, cfg, added, log) if dv_dir else None
    lora = add_lora(w, lora_dir, cfg, log) if lora_dir else None
    w.meta = {
        "arch": arch, "config": cfg, "draft": draft, "mtp": mtp, "dv": dv, "lora": lora, "quant": quant,
        "tokenizer": tok_info,
        "generation": {"eos_ids": eos},
        # The engine reads the template (blob above) and recognises its output
        # format itself (models/formats); these are the variables it may print.
        "chat": _template_tokens(tok_cfg),
        "source": {"model": os.path.basename(os.path.normpath(model_dir)),
                   "draft": os.path.basename(os.path.normpath(draft_dir)) if draft_dir else None,
                   "dv": os.path.basename(os.path.normpath(dv_dir)) if dv_dir else None,
                   "lora": os.path.basename(os.path.normpath(lora_dir)) if lora_dir else None},
    }
    if lora:  # a reader that dropped the adapters would decode the plain model: format 2, which earlier builds refuse
        w.meta["requires"] = ["lora"]
    w.write(out)
    log(f"wrote {out}: {len(w.entries)} entries, draft={draft['kind'] if draft else 'dv' if dv else 'none'}, "
        f"mtp={mtp['layers'] if mtp else 0} layer(s), lora={lora['rank'] if lora else 'none'}, "
        f"weights={quant['format'] if quant else 'as stored'}, eos={eos}")


def _add_draft(w: Writer, draft_dir: str, target: dict | None, log) -> dict:
    """A block drafter's tensors as ``draft.`` names; its canonical config (canonical_draft_config)."""
    draft = canonical_draft_config(_read_config(os.path.join(draft_dir, "config.json")), target)
    dseen = set()
    if draft["kind"] == "dspine":
        with SafeTensors(draft_dir) as dst:
            for canon, dt, arr in dspine_tensors(dst.names(), dst.read, draft, log):
                w.add_tensor("draft." + canon, dt, arr.shape, np.ascontiguousarray(arr))
                dseen.add(canon)
        check_dspine(draft, dseen)
        return draft
    with SafeTensors(draft_dir) as dst:
        for name in dst.names():
            inner = name[len("model."):] if name.startswith("model.") else name
            if _D_SKIP.match(inner):
                log(f"draft: ignoring {name} (the engine uses the target's embedding and head)")
                continue
            canon = _map(inner, _D_RULES)
            if canon is None:
                raise ValueError(f"unknown draft tensor {name!r}")
            arr, dt = dst.read(name)
            w.add_tensor("draft." + canon, dt, arr.shape, np.ascontiguousarray(arr))
            dseen.add(canon)
    _check_draft(draft, dseen)
    return draft


def convert_draft(draft_dir: str, out: str, log=print) -> None:
    """A block drafter (DFlash2, DFlash, DSpine) on its own -> a drafter file: the engine takes it beside a model of any
    format (--drafter-file), and checks it against that model (the "target" its config names)."""
    w = Writer()
    draft = _add_draft(w, draft_dir, None, log)
    w.meta = {"arch": "drafter", "kind": "drafter", "draft": draft,
              "source": {"draft": os.path.basename(os.path.normpath(draft_dir))}}
    w.write(out)
    kind = {"dspine": "DSpine", "dflash": "DFlash"}.get(draft["kind"], "DFlash2")
    log(f"wrote {out}: {len(w.entries)} entries, a {kind} drafter of {draft['layers']} layer(s), block {draft['block']}, "
        f"taps {draft['taps']}, window {draft['window'] or 'none'}")


def _add_linear(w: Writer, st: SafeTensors, name: str, canon: str, cfg: dict, reorder_q: bool = True) -> None:
    """One tensor; an NVFP4 weight with its two scales.  Qwen3.5 query rows are reordered (weights and scales alike)."""
    arr, dt = st.read(name)
    packed = name.endswith(_PACKED)  # compressed-tensors' NVFP4
    base = name[: -len(_PACKED)] if packed else name[: -len(".weight")] if name.endswith(".weight") else None
    quantized = base is not None and dt == "u8" and (base + ".weight_scale") in st.index
    if canon.endswith("gdn.conv"):
        arr = arr.reshape(arr.shape[0], arr.shape[-1])
    reorder = reorder_q and canon.endswith("attn.q")
    if not quantized:
        if reorder:
            arr = _reorder_q(arr, cfg["n_heads"], cfg["head_dim"])
        w.add_tensor(canon, dt, arr.shape, np.ascontiguousarray(arr))
        return
    sc, sdt = st.read(base + ".weight_scale")
    if packed:  # its global scale is the reciprocal of ModelOpt's: 1 / it, rounded once to float32
        gs, _ = st.read(base + ".weight_global_scale")
        s2 = np.float32(1.0 / float(np.asarray(gs, dtype=np.float64).reshape(-1)[0]))
    else:
        s2, _ = st.read(base + ".weight_scale_2")
    if sdt != "f8e4m3" or sc.shape[1] * 8 != arr.shape[1]:
        raise ValueError(f"{name}: unexpected NVFP4 scale layout {sdt}{list(sc.shape)}")
    if reorder:
        arr = _reorder_q(arr, cfg["n_heads"], cfg["head_dim"])
        sc = _reorder_q(sc, cfg["n_heads"], cfg["head_dim"])
    N, K = arr.shape[0], arr.shape[1] * 2
    w.add_tensor(canon, "nvfp4", [N, K], np.ascontiguousarray(arr))
    w.add_tensor(canon + ".scale", "f8e4m3", sc.shape, np.ascontiguousarray(sc))
    w.add_tensor(canon + ".scale2", "f32", [1], np.ascontiguousarray(np.asarray(s2, dtype=np.float32).reshape(1)))


def _chat_template(model_dir: str, tok_cfg: dict) -> str | None:
    """chat_template.jinja, else tokenizer_config.json's chat_template (the "default" one of a list)."""
    p = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    t = tok_cfg.get("chat_template")
    if isinstance(t, list):
        t = next((x.get("template") for x in t if isinstance(x, dict) and x.get("name") == "default"), None)
    return t if isinstance(t, str) else None


def _template_tokens(tok_cfg: dict) -> dict:
    """bos_token / eos_token as tokenizer_config.json names them (a template may print them)."""
    out = {}
    for k in ("bos_token", "eos_token"):
        v = tok_cfg.get(k)
        v = v.get("content") if isinstance(v, dict) else v
        if isinstance(v, str):
            out[k] = v
    return out


def _eos_ids(model_dir: str, tc: dict, added: dict) -> list[int]:
    ids = []
    for src in (_read_optional_config(os.path.join(model_dir, "generation_config.json")), tc):
        e = src.get("eos_token_id")
        ids += e if isinstance(e, list) else ([e] if e is not None else [])
    for s in ("<|im_end|>", "<|endoftext|>"):
        if s in added:
            ids.append(added[s])
    return sorted(set(int(i) for i in ids))


_ATTN_LAYER_KEYS = ("in_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "attn.q_norm", "attn.k_norm",
                    "mlp.gate", "mlp.up", "mlp.down")
_GDN_LAYER_KEYS = ("in_norm", "post_norm", "gdn.qkv", "gdn.z", "gdn.b", "gdn.a", "gdn.conv", "gdn.A_log",
                   "gdn.dt_bias", "gdn.norm", "gdn.out", "mlp.gate", "mlp.up", "mlp.down")
_DRAFT_CONV_KEYS = ("attn_conv.base", "attn_conv.proj", "mlp_conv.base", "mlp_conv.proj")


def _check_tensors(kind: str, need: set[str], seen: set[str]) -> None:
    missing = need - seen
    if missing:
        raise ValueError(f"missing {kind} tensors: {sorted(missing)[:10]}")


def _check_target(cfg, seen, qk_norm=True):
    """Every tensor the engine reads is there; without q/k norms (Llama), none is either."""
    need = {"embed", "norm"} | ({"lm_head"} if not cfg["tie_embed"] else set())
    for i, t in enumerate(cfg["layer_types"]):
        keys = _ATTN_LAYER_KEYS if t == "attn" else _GDN_LAYER_KEYS
        need |= {f"L{i}.{n}" for n in keys if qk_norm or n not in ("attn.q_norm", "attn.k_norm")}
    _check_tensors("target", need, seen)
    unread = sorted(n for n in seen - need if n.endswith(("q_norm", "k_norm")))
    if unread:
        raise ValueError(f"q/k norms in a model without them: {unread[:4]}")


def _check_mtp(layers, seen):
    need = {"fc", "norm_emb", "norm_hidden", "norm"}
    for j in range(layers):
        need |= {f"L{j}.{n}" for n in _ATTN_LAYER_KEYS}
    _check_tensors("MTP", need, seen)


def _check_draft(d, seen):
    """DFlash2's tensors; DFlash's, its layers and fc and norms, and none of DFlash2's convolutions or selector."""
    later = d["kind"] == "dflash2"
    need = {"fc", "hidden_norm", "norm"} | ({"sel.pred", "sel.succ", "sel.hid"} if later else set())
    for j in range(d["layers"]):
        need |= {f"L{j}.{n}" for n in (*_ATTN_LAYER_KEYS, *(_DRAFT_CONV_KEYS if later else ()))}
    _check_tensors("draft", need, seen)
    if not later and seen - need:
        raise ValueError(f"a DFlash drafter with tensors it does not read: {sorted(seen - need)[:4]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flyx_convert", description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="HuggingFace Qwen3.5, Qwen3 or Llama (MiniCPM5) directory (without it: --draft "
                                    "alone, a drafter file for --drafter-file)")
    ap.add_argument("--draft", help="block drafter directory: DFlash2 (speculators' or z-lab's), DFlash (z-lab's) or "
                                    "DSpine (optional)")
    ap.add_argument("--dv", help="DV drafter directory, as python/convert/export_dv.py writes it (optional; Llama), "
                                 "with the V-LoRA it holds")
    ap.add_argument("--lora", help="a V-LoRA alone: the directory export_dv.py --vlora writes (optional; Llama, Qwen3)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--activations", choices=("nvfp4", "bf16"),
                    help="NVFP4 weights' inputs in prefill passes (default: what the checkpoint is calibrated for)")
    a = ap.parse_args(argv)

    def log(message):
        # A redirected console may use a code page that cannot print a model's path.
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding))

    if a.model:
        convert(a.model, a.out, a.draft, log=log, activations=a.activations, dv_dir=a.dv, lora_dir=a.lora)
    elif a.draft and not a.activations and not a.dv and not a.lora:
        convert_draft(a.draft, a.out, log=log)
    else:
        ap.error("--model, or --draft alone")


if __name__ == "__main__":
    sys.exit(main())
