// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What the block drafters' descriptions share (DFlash2's, models/drafters/dflash2.h; DSpine's, dspine.h): each layer a
// Qwen3 layer under the names python/convert gives it ("draft.L<j>.attn.q"), and the drafter's context K/V, one cache
// a layer.  What a drafter adds to its layers (DFlash2's convolutions, DSpine's bridge) its own builder binds.
#pragma once

#include <string>

#include "models/model.h"

namespace flyx::models::drafters {

// Layer j's norms and projections into y (a flyx_dflash2_layer_block or flyx_dspine_layer_block).  dn: a drafter
// tensor's name in its file, from the name python/convert gives it.
template <class Layer, class Name>
void bind_qwen3_layer(Tables& T, Layer& y, const Name& dn, int j, int hidden, int n_heads, int n_kv, int head_dim,
                      int inter, double eps) {
  const std::string p = "draft.L" + std::to_string(j) + ".";
  const int H = hidden, q = n_heads * head_dim, kv = n_kv * head_dim, I = inter;
  y.norm = T.norm(dn(p + "in_norm"), H, 0.0, eps);
  y.post_norm = T.norm(dn(p + "post_norm"), H, 0.0, eps);
  y.q_norm = T.norm(dn(p + "attn.q_norm"), head_dim, 0.0, eps);
  y.k_norm = T.norm(dn(p + "attn.k_norm"), head_dim, 0.0, eps);
  y.q = T.weight(dn(p + "attn.q"), {q, H});
  y.k = T.weight(dn(p + "attn.k"), {kv, H});
  y.v = T.weight(dn(p + "attn.v"), {kv, H});
  y.o = T.weight(dn(p + "attn.o"), {H, q});
  y.gate = T.weight(dn(p + "mlp.gate"), {I, H});
  y.up = T.weight(dn(p + "mlp.up"), {I, H});
  y.down = T.weight(dn(p + "mlp.down"), {H, I});
}

// The drafter's context K/V: one cache a layer, of the drafter's heads.
inline int draft_kv_state(Tables& T, int layers, int n_kv, int head_dim) {
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = 1;
  st.layers = layers;
  st.n_kv = n_kv;
  st.head_dim = head_dim;
  return T.state(st);
}

}  // namespace flyx::models::drafters
