# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""DFlash2 block drafter (reference).

Behaviour follows the published DFlash / DFlash2 description (z-lab DFlash,
vllm-project/speculators ``dflash2``): a few attention layers whose
keys/values for the context come from target hidden states (several taps,
concatenated, projected by ``fc`` and normalised), a query block of
``[anchor, mask * (B-1)]`` attending bidirectionally to itself and to the
context before the anchor (all of it, or with a sliding window the positions
less than ``window`` from the query's, both ways, as z-lab's mask has it),
token-conditioned grouped causal convolutions around attention and MLP, and a
low-rank candidate selector that re-ranks the unary top-k with a
predecessor/hidden/successor bilinear score while walking the block left to
right.

DFlash (z-lab's DFlashDraftModel, what DFlash2 grew from) is this block with
neither: conv_taps 0 (attention and MLP unwrapped) and sel_rank 0 (each row's
draft the head's best, the lower id on ties).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .target import attention, rms_norm, rope


@dataclass
class DraftConfig:
    hidden: int
    layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    inter: int
    eps: float
    rope_theta: float
    block: int
    mask_token: int
    taps: list[int]                 # target hidden after this many layers
    conv_taps: int
    conv_group: int
    sel_rank: int
    sel_topk: int
    window: int = 0                 # positions attend to positions less than this far from them (0: all)


def draft_config_from_canonical(c: dict) -> DraftConfig:
    keys = DraftConfig.__dataclass_fields__.keys()
    return DraftConfig(**{k: c[k] for k in keys if k in c})  # files from before windows have none


@dataclass
class DraftState:
    ctx: int = 0
    k: dict = field(default_factory=dict)   # layer -> [ctx, n_kv, hd] (normed, roped)
    v: dict = field(default_factory=dict)


def grouped_conv(x, delta, base, block, group):
    """x: [T, H]; delta: [T, taps, G]; base: [taps, H]. Causal within each block."""
    T, H = x.shape
    taps, G = base.shape[0], H // group
    xg = x.view(T, G, group)
    out = torch.zeros_like(xg)
    posn = torch.arange(T) % block
    for t in range(taps):
        sh = xg if t == 0 else torch.cat([torch.zeros(t, G, group, dtype=x.dtype), xg[:-t]])
        coef = base[t].view(1, G, group) + delta[:, t, :, None]
        out = out + sh * coef * (posn >= t).view(-1, 1, 1).to(x.dtype)
    return out.view(T, H)


class Draft:
    def __init__(self, cfg: DraftConfig, W: dict, dtype=torch.float64):
        self.cfg, self.dtype = cfg, dtype
        self.W = {k: v.to(dtype) for k, v in W.items() if k.startswith("draft.")}
        # Embedding and head are the target's (full-vocabulary DFlash2).
        self.embed = W["target.embed"].to(dtype)
        self.head = W.get("target.lm_head", W["target.embed"]).to(dtype)

    def w(self, name):
        return self.W["draft." + name]

    def new_state(self):
        c, st = self.cfg, DraftState()
        for j in range(c.layers):
            st.k[j] = torch.zeros(0, c.n_kv, c.head_dim, dtype=self.dtype)
            st.v[j] = torch.zeros(0, c.n_kv, c.head_dim, dtype=self.dtype)
        return st

    def append_context(self, st: DraftState, feats):
        """feats: [n, len(taps)*hidden] target taps for positions st.ctx.. (in order)."""
        c = self.cfg
        n = feats.shape[0]
        h = rms_norm(feats.to(self.dtype) @ self.w("fc").T, self.w("hidden_norm"), c.eps)
        pos = torch.arange(st.ctx, st.ctx + n)
        for j in range(c.layers):
            p = f"L{j}.attn."
            k = rms_norm((h @ self.w(p + "k").T).view(n, c.n_kv, c.head_dim), self.w(p + "k_norm"), c.eps)
            k = rope(k, pos, c.rope_theta, c.head_dim)
            v = (h @ self.w(p + "v").T).view(n, c.n_kv, c.head_dim)
            st.k[j] = torch.cat([st.k[j], k]); st.v[j] = torch.cat([st.v[j], v])
        st.ctx += n

    def _conv(self, name, x):
        c = self.cfg
        if not c.conv_taps:  # DFlash: no convolutions
            return x, None, None
        G = c.hidden // c.conv_group
        ker = (x @ self.w(name + ".proj").T).view(x.shape[0], 2, c.conv_taps, G)
        base = self.w(name + ".base")                       # [2, taps, H]
        return grouped_conv(x, ker[:, 0], base[0], c.block, c.conv_group), ker[:, 1], base[1]

    def block(self, st: DraftState, anchor: int):
        """Run one block at positions ctx..ctx+B-1. Returns (drafts [B-1], debug dict)."""
        c = self.cfg
        B = c.block
        ids = torch.tensor([anchor] + [c.mask_token] * (B - 1))
        x = self.embed[ids]
        pos = torch.arange(st.ctx, st.ctx + B)
        scale = 1.0 / math.sqrt(c.head_dim)
        # keys: the context's positions 0..ctx-1, then the block's
        mask = (pos[:, None] - torch.arange(st.ctx + B)[None, :]).abs() < c.window if c.window else None
        for j in range(c.layers):
            p = f"L{j}."
            r = x
            h = rms_norm(x, self.w(p + "in_norm"), c.eps)
            h, ker, base = self._conv(p + "attn_conv", h)
            q = rms_norm((h @ self.w(p + "attn.q").T).view(B, c.n_heads, c.head_dim), self.w(p + "attn.q_norm"), c.eps)
            k = rms_norm((h @ self.w(p + "attn.k").T).view(B, c.n_kv, c.head_dim), self.w(p + "attn.k_norm"), c.eps)
            v = (h @ self.w(p + "attn.v").T).view(B, c.n_kv, c.head_dim)
            q = rope(q, pos, c.rope_theta, c.head_dim)
            k = rope(k, pos, c.rope_theta, c.head_dim)
            K = torch.cat([st.k[j], k]); V = torch.cat([st.v[j], v])   # bidirectional (within the window)
            o = attention(q, K, V, scale, mask).reshape(B, -1) @ self.w(p + "attn.o").T
            if ker is not None:
                o = grouped_conv(o, ker, base, B, c.conv_group)
            x = r + o
            r = x
            h = rms_norm(x, self.w(p + "post_norm"), c.eps)
            h, ker, base = self._conv(p + "mlp_conv", h)
            m = (torch.nn.functional.silu(h @ self.w(p + "mlp.gate").T) * (h @ self.w(p + "mlp.up").T)) @ self.w(p + "mlp.down").T
            x = r + (grouped_conv(m, ker, base, B, c.conv_group) if ker is not None else m)
        hn = rms_norm(x, self.w("norm"), c.eps)[1:]          # rows that predict tokens
        logits = hn @ self.head.T
        unary, cand = torch.topk(logits, c.sel_topk, dim=-1)  # sorted, ties -> lower id first
        if not c.sel_rank:  # DFlash: each row's best
            best = torch.argmax(logits, dim=-1)                # the first (lowest id) of equal bests
            return [int(t) for t in best], {"cand": cand, "unary": unary, "scores": unary, "hidden": hn}
        hid = hn @ self.w("sel.hid").T                         # [B-1, rank]
        pred, succ = self.w("sel.pred"), self.w("sel.succ")
        prev, drafts, scores = anchor, [], []
        for s in range(B - 1):
            sc = unary[s] + (succ[cand[s]] * (pred[prev] * hid[s])[None, :]).sum(-1)
            idx = int(torch.argmax(sc))
            prev = int(cand[s, idx])
            drafts.append(prev); scores.append(sc)
        return drafts, {"cand": cand, "unary": unary, "scores": torch.stack(scores), "hidden": hn}
