// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DFlash2 and DFlash from a "draft" config of kind "dflash2" or "dflash" and its draft.* tensors, the names
// python/convert writes: per layer L<j>.{in_norm, post_norm, attn.*, mlp.*} and DFlash2's convolutions
// L<j>.{attn_conv, mlp_conv}.{proj, base}; once fc, hidden_norm, norm and DFlash2's selector sel.{pred, succ, hid}.
#include "models/dflash2.h"

#include <climits>
#include <cstdint>
#include <stdexcept>

namespace flyx::models::dflash2 {
namespace {

using nlohmann::json;

// The config's dimensions, checked before any tensor is bound (the drafter shares the target's embedding and head,
// so a width that differs is refused here); the products the backends take of them fit an int.
struct Spec {
  flyx_dflash2_block block{};
  int layers = 0;
  int tg2 = 0;  // a layer's conv coefficient rows: 2 * conv_taps * (hidden / conv_group); 0 without convolutions
  double eps = 0.0;
  std::vector<int> taps;
};

Spec spec_of(const json& j, const Target& t) {
  const std::string kind = j.value("kind", "dflash2");
  if (kind != "dflash2" && kind != "dflash") throw std::runtime_error("a block drafter of kind " + kind);
  const bool later = kind == "dflash2";  // DFlash2: convolutions and a selector; DFlash: neither
  const std::string name = later ? "DFlash2" : "DFlash";
  // the messages as they were when only DFlash2 was described here (tests/models/gguf_test.cpp reads them)
  auto expect = [&](bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(name + " " + what);
  };
  auto invalid = [&](bool ok, const std::string& detail) {
    if (!ok) throw std::runtime_error("invalid " + name + " drafter config (" + detail + ")");
  };
  auto whole = [&](const char* key) {
    const int64_t v = j.at(key).get<int64_t>();
    invalid(v >= INT_MIN && v <= INT_MAX, std::string(key) + " out of range");
    return int(v);
  };
  Spec spec{};
  flyx_dflash2_block& d = spec.block;
  d.hidden = whole("hidden");
  spec.layers = whole("layers");
  d.n_heads = whole("n_heads");
  d.n_kv = whole("n_kv");
  d.head_dim = whole("head_dim");
  d.inter = whole("inter");
  d.block = whole("block");
  d.mask_token = whole("mask_token");
  for (int64_t tap : j.at("taps").get<std::vector<int64_t>>()) {
    expect(tap >= 1 && tap <= t.layers, "tap after layer " + std::to_string(tap) + " of " + std::to_string(t.layers));
    spec.taps.push_back(int(tap));
  }
  d.conv_taps = whole("conv_taps");
  d.conv_group = whole("conv_group");
  d.sel_rank = whole("sel_rank");
  d.sel_topk = whole("sel_topk");
  d.theta = j.at("rope_theta").get<double>();
  const int64_t win = j.value("window", int64_t(0));
  expect(win >= 0 && win <= INT_MAX, "window out of range");
  d.window = int32_t(win);
  spec.eps = j.at("eps").get<double>();
  expect(d.block >= 2 && d.block <= FLYX_MAX_ROWS, "block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  expect(d.hidden == t.hidden, "hidden size does not match target (" + std::to_string(d.hidden) + ", not " +
                                   std::to_string(t.hidden) + ")");
  const json target = j.value("target", json::object());
  expect(target.value("vocab", t.vocab) == t.vocab, "drafter made for another vocabulary");
  invalid(d.hidden > 0 && spec.layers > 0 && d.n_heads > 0 && d.n_kv > 0 && d.head_dim > 0 && d.head_dim % 2 == 0 &&
             d.inter > 0 && d.n_heads % d.n_kv == 0 && d.mask_token >= 0 && d.mask_token < t.vocab &&
             spec.eps > 0 && d.theta > 0 && d.hidden <= kMaxWidth && d.inter <= kMaxWidth &&
             spec.layers <= kMaxLayers && d.n_heads <= kMaxHeads && d.head_dim <= kMaxHeadDim && !spec.taps.empty() &&
             spec.taps.size() <= size_t(INT_MAX),
         "invalid dimensions");
  if (later) {
    invalid(d.conv_taps > 0 && d.conv_taps <= kMaxHeadDim && d.sel_rank > 0 && d.sel_rank <= kMaxWidth &&
                d.sel_topk > 0 && d.sel_topk <= t.vocab,
            "convolution or selector dimensions");
    expect(d.conv_group >= 1 && d.hidden % d.conv_group == 0, "conv group does not divide hidden");
  } else {
    invalid(d.conv_taps == 0 && d.conv_group == 0 && d.sel_rank == 0 && d.sel_topk == 1,
            "convolutions or a selector: DFlash has neither");
  }
  d.n_taps = int32_t(spec.taps.size());
  const int64_t tg2 = later ? int64_t(2) * d.conv_taps * (d.hidden / d.conv_group) : 0;
  const int64_t tap_width = int64_t(d.n_taps) * t.hidden, ctx_width = int64_t(2) * spec.layers * d.n_kv * d.head_dim;
  invalid(tg2 <= INT_MAX && tap_width <= INT_MAX && ctx_width <= INT_MAX, "dimensions too large");
  spec.tg2 = int(tg2);
  return spec;
}

}  // namespace

std::vector<int> taps(const json& d, const Target& t) { return spec_of(d, t).taps; }

int describe(Tables& T, const json& j, const Target& t, const std::function<std::string(const std::string&)>& dn) {
  const Spec spec = spec_of(j, t);
  flyx_dflash2_block d = spec.block;
  const bool conv = d.conv_taps > 0, selector = d.sel_rank > 0;
  const int H = d.hidden, q = d.n_heads * d.head_dim, kv = d.n_kv * d.head_dim, I = d.inter;
  const double eps = spec.eps;
  d.embed = t.embed;
  d.head = t.head;
  d.first_layer = int32_t(T.blocks.size());
  d.layers = spec.layers;
  for (int l = 0; l < spec.layers; ++l) {
    const std::string p = dn("draft.L" + std::to_string(l) + ".");
    flyx_block b{};
    b.kind = FLYX_BLOCK_DFLASH2_LAYER;
    auto& y = b.dflash2_layer;
    y.norm = T.norm(p + "in_norm", H, 0.0, eps);
    y.post_norm = T.norm(p + "post_norm", H, 0.0, eps);
    y.q_norm = T.norm(p + "attn.q_norm", d.head_dim, 0.0, eps);
    y.k_norm = T.norm(p + "attn.k_norm", d.head_dim, 0.0, eps);
    y.q = T.weight(p + "attn.q", {q, H});
    y.k = T.weight(p + "attn.k", {kv, H});
    y.v = T.weight(p + "attn.v", {kv, H});
    y.o = T.weight(p + "attn.o", {H, q});
    y.gate = T.weight(p + "mlp.gate", {I, H});
    y.up = T.weight(p + "mlp.up", {I, H});
    y.down = T.weight(p + "mlp.down", {H, I});
    y.attn_conv_proj = y.attn_conv_base = y.mlp_conv_proj = y.mlp_conv_base = -1;  // DFlash: none
    if (conv) {
      y.attn_conv_proj = T.weight(p + "attn_conv.proj", {spec.tg2, H});
      y.attn_conv_base = T.weight(p + "attn_conv.base", {2, d.conv_taps, H});
      y.mlp_conv_proj = T.weight(p + "mlp_conv.proj", {spec.tg2, H});
      y.mlp_conv_base = T.weight(p + "mlp_conv.base", {2, d.conv_taps, H});
    }
    T.block(b);
  }
  d.fc = T.weight(dn("draft.fc"), {H, int64_t(d.n_taps) * t.hidden});
  d.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
  d.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  d.sel_pred = d.sel_succ = d.sel_hid = -1;  // DFlash: none
  if (selector) {
    d.sel_pred = T.weight(dn("draft.sel.pred"), {t.vocab, d.sel_rank});
    d.sel_succ = T.weight(dn("draft.sel.succ"), {t.vocab, d.sel_rank});
    d.sel_hid = T.weight(dn("draft.sel.hid"), {d.sel_rank, H});
  }
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = 1;
  st.layers = spec.layers;
  st.n_kv = d.n_kv;
  st.head_dim = d.head_dim;
  d.kv = T.state(st);
  flyx_block b{};
  b.kind = FLYX_BLOCK_DFLASH2;
  b.dflash2 = d;
  return T.block(b);
}

}  // namespace flyx::models::dflash2
