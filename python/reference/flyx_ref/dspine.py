# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""DSpine block drafter (reference).

Behaviour follows the DSpine paper (Zhang et al., "Draft in Parallel, Condition Through Depth: Adjacent Causal
Injection for Speculative Decoding", 2026) and the configs of its released drafters (tencent/Qwen3-8B-DSpine-b16,
tencent/Qwen3.8-27B-DSpine-b16): a block drafter in DFlash's layout (context keys and values from the target's
tapped hidden states, projected by ``fc`` and normalised; a query block of ``[anchor, mask * (B-1)]``) whose layers
are plain Qwen3 layers attending causally inside the block, each followed by the code bridge's write, which passes
a low-rank message from every row to the row after it:

    rms_r = sqrt(mean(x_r^2))
    u_r   = read (x_r / (rms_r + eps_b))                      the row's feature, code-sized
    c_r   = sqrt(C) u_r / (|u_r| + eps_b)        (r >= 1)     its code; the anchor's is its token's:
    c_0   = sqrt(C) codebook[anchor]
    m_r   = sigmoid(gate c_r + pred c_{r-1} + gate_bias) * (prev c_{r-1})        (r >= 1; m_0 = 0)
    x_r  <- x_r + rms_r * write m_r

The drafts: per row r >= 1 the final norm and the head's ``topk`` best tokens, then a walk that redoes the last
layer's write with the code of the token actually drafted before the row (the anchor's for row 1) in place of the
predicted code c_{r-1}, and takes the candidate the redone row scores highest:

    e     = sqrt(C) codebook[draft_{r-1}]
    x'_r  = x_r + rms_r * write sigmoid(gate c_r + pred e + gate_bias) * (prev e)   (x_r, rms_r, c_r before that write)
    draft_r = argmax_k <x'_r * norm_w, head[cand_k]>

(the norm's 1 / rms(x'_r) is the same for every candidate of a row, so it changes no argmax).  Attention inside the
block is causal and the messages only go forward, so a row's draft depends on no row after it: a block of fewer rows
gives the same first drafts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .draft import Draft
from .target import attention, rms_norm, rope


@dataclass
class DSpineConfig:
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
    code: int                       # the bridge's code size C
    rank: int                       # its message rank R
    bridge_eps: float
    topk: int                       # the head's candidates a row's walk chooses among
    window: int = 0                 # a position attends to the positions less than this far before it (0: all)


def dspine_config_from_canonical(c: dict) -> DSpineConfig:
    keys = DSpineConfig.__dataclass_fields__.keys()
    return DSpineConfig(**{k: c[k] for k in keys if k in c})


class DSpine(Draft):
    """The context (new_state, append_context) is DFlash2's; block() is DSpine's."""

    def _bridge_write(self, j: int, x, anchor: int):
        """Layer j's write: (x after it, (x, rms, gate c) before it)."""
        c = self.cfg
        sq = math.sqrt(c.code)
        rms = torch.sqrt((x * x).mean(-1))
        u = (x / (rms + c.bridge_eps)[:, None]) @ self.w(f"L{j}.bridge.read").T
        code = sq * u / (torch.linalg.vector_norm(u, dim=-1) + c.bridge_eps)[:, None]
        code[0] = sq * self.w("bridge.codebook")[anchor]
        q = code @ self.w("bridge.gate").T
        s = code @ self.w("bridge.pred").T
        v = code @ self.w("bridge.prev").T
        m = torch.zeros_like(q)
        m[1:] = torch.sigmoid(q[1:] + s[:-1] + self.w("bridge.gate_bias")) * v[:-1]
        return x + rms[:, None] * (m @ self.w(f"L{j}.bridge.write").T), (x, rms, q)

    def block(self, st, anchor: int):
        """Run one block at positions ctx..ctx+B-1. Returns (drafts [B-1], debug dict)."""
        c = self.cfg
        B = c.block
        ids = torch.tensor([anchor] + [c.mask_token] * (B - 1))
        x = self.embed[ids]
        pos = torch.arange(st.ctx, st.ctx + B)
        keys = torch.arange(st.ctx + B)
        mask = keys[None, :] <= pos[:, None]                        # the context, and the block up to the row
        if c.window:
            mask &= pos[:, None] - keys[None, :] < c.window
        scale = 1.0 / math.sqrt(c.head_dim)
        last = None
        for j in range(c.layers):
            p = f"L{j}."
            h = rms_norm(x, self.w(p + "in_norm"), c.eps)
            q = rms_norm((h @ self.w(p + "attn.q").T).view(B, c.n_heads, c.head_dim), self.w(p + "attn.q_norm"), c.eps)
            k = rms_norm((h @ self.w(p + "attn.k").T).view(B, c.n_kv, c.head_dim), self.w(p + "attn.k_norm"), c.eps)
            v = (h @ self.w(p + "attn.v").T).view(B, c.n_kv, c.head_dim)
            q = rope(q, pos, c.rope_theta, c.head_dim)
            k = rope(k, pos, c.rope_theta, c.head_dim)
            K = torch.cat([st.k[j], k]); V = torch.cat([st.v[j], v])
            x = x + attention(q, K, V, scale, mask).reshape(B, -1) @ self.w(p + "attn.o").T
            h = rms_norm(x, self.w(p + "post_norm"), c.eps)
            x = x + (torch.nn.functional.silu(h @ self.w(p + "mlp.gate").T) * (h @ self.w(p + "mlp.up").T)) \
                @ self.w(p + "mlp.down").T
            x, last = self._bridge_write(j, x, anchor)
        hn = rms_norm(x, self.w("norm"), c.eps)[1:]                  # rows that predict tokens
        logits = hn @ self.head.T
        unary, cand = torch.topk(logits, c.topk, dim=-1)              # sorted, ties -> lower id first
        raw, rms, gq = last
        sq = math.sqrt(c.code)
        nw = self.w("norm")
        write = self.w(f"L{c.layers - 1}.bridge.write")
        prev, drafts, scores = anchor, [], []
        for t in range(B - 1):
            r = t + 1
            e = sq * self.w("bridge.codebook")[prev]
            m = torch.sigmoid(gq[r] + e @ self.w("bridge.pred").T + self.w("bridge.gate_bias")) * (e @ self.w("bridge.prev").T)
            redone = raw[r] + rms[r] * (write @ m)
            sc = self.head[cand[t]] @ (redone * nw)
            prev = int(cand[t, int(torch.argmax(sc))])
            drafts.append(prev); scores.append(sc)
        return drafts, {"cand": cand, "unary": unary, "scores": torch.stack(scores), "hidden": hn}
