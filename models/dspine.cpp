// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine from a "draft" config of kind "dspine" and its draft.* tensors, the names python/convert writes
// (python/convert/flyx_convert/dspine.py): per layer L<j>.{in_norm, post_norm, attn.*, mlp.*, bridge.read,
// bridge.write}, once fc, hidden_norm, norm and bridge.{codebook, gate, pred, prev, gate_bias}.
#include "models/dspine.h"

#include <climits>
#include <cstdint>
#include <stdexcept>

namespace flyx::models::dspine {
namespace {

using nlohmann::json;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("DSpine drafter: " + what);
}

int whole(const json& j, const char* key) {
  const int64_t v = j.at(key).get<int64_t>();
  expect(v >= INT_MIN && v <= INT_MAX, std::string(key) + " out of range");
  return int(v);
}

// The block's dimensions from the config, checked; the products the backends take of them fit an int.
flyx_dspine_block dims(const json& d, const Target& t, int* layers, double* eps) {
  expect(d.value("kind", "") == "dspine", "a config of another kind");
  flyx_dspine_block b{};
  b.hidden = whole(d, "hidden");
  *layers = whole(d, "layers");
  b.n_heads = whole(d, "n_heads");
  b.n_kv = whole(d, "n_kv");
  b.head_dim = whole(d, "head_dim");
  b.inter = whole(d, "inter");
  b.block = whole(d, "block");
  b.mask_token = whole(d, "mask_token");
  b.code = whole(d, "code");
  b.rank = whole(d, "rank");
  b.topk = whole(d, "topk");
  const int64_t win = d.value("window", int64_t(0));
  expect(win >= 0 && win <= INT_MAX, "window out of range");
  b.window = int32_t(win);
  b.theta = d.at("rope_theta").get<double>();
  b.bridge_eps = d.at("bridge_eps").get<double>();
  *eps = d.at("eps").get<double>();
  expect(b.hidden == t.hidden, "made for a target of width " + std::to_string(b.hidden) + ", not " +
                                   std::to_string(t.hidden));
  expect(b.block >= 2 && b.block <= FLYX_MAX_ROWS, "block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  expect(b.topk >= 2 && b.topk <= 16, "the walk's candidates outside 2..16");
  expect(b.hidden > 0 && b.hidden <= kMaxWidth && *layers > 0 && *layers <= kMaxLayers && b.n_heads > 0 &&
             b.n_heads <= kMaxHeads && b.n_kv > 0 && b.n_heads % b.n_kv == 0 && b.head_dim > 0 &&
             b.head_dim % 2 == 0 && b.head_dim <= kMaxHeadDim && b.inter > 0 && b.inter <= kMaxWidth &&
             b.code > 0 && b.code <= kMaxWidth && b.rank > 0 && b.rank <= kMaxWidth && b.mask_token >= 0 &&
             b.mask_token < t.vocab && b.topk <= t.vocab && *eps > 0 && b.bridge_eps > 0 && b.theta > 0,
         "invalid dimensions");
  const int64_t ctx = int64_t(2) * *layers * b.n_kv * b.head_dim, qkv = int64_t(b.n_heads + 2 * b.n_kv) * b.head_dim;
  expect(ctx <= INT_MAX && qkv <= INT_MAX && int64_t(3) * b.rank <= INT_MAX, "dimensions too large");
  return b;
}

}  // namespace

std::vector<int> taps(const json& d, const Target& t) {
  int layers = 0;
  double eps = 0;
  (void)dims(d, t, &layers, &eps);
  const json target = d.value("target", json::object());
  expect(target.value("vocab", t.vocab) == t.vocab, "made for another vocabulary");
  expect(!target.contains("layers") || target["layers"].is_null() || target["layers"].get<int>() == t.layers,
         "made for a target of " + std::to_string(target.value("layers", 0)) + " layers, not " +
             std::to_string(t.layers));
  std::vector<int> out;
  for (const auto& v : d.at("taps")) {
    const int64_t x = v.get<int64_t>();
    expect(x >= 1 && x <= t.layers, "a tap after layer " + std::to_string(x) + " of " + std::to_string(t.layers));
    out.push_back(int(x));
  }
  expect(!out.empty(), "no taps");
  expect(int64_t(out.size()) * t.hidden <= INT_MAX, "too many taps");
  return out;
}

int describe(Tables& T, const json& d, const Target& t, const std::function<std::string(const std::string&)>& dn) {
  int layers = 0;
  double eps = 0;
  flyx_dspine_block b = dims(d, t, &layers, &eps);
  b.n_taps = int32_t(taps(d, t).size());
  const int H = b.hidden, q = b.n_heads * b.head_dim, kv = b.n_kv * b.head_dim, I = b.inter, C = b.code, R = b.rank;
  b.embed = t.embed;
  b.head = t.head;
  b.first_layer = int32_t(T.blocks.size());
  b.layers = layers;
  for (int l = 0; l < layers; ++l) {
    const std::string p = dn("draft.L" + std::to_string(l) + ".");
    flyx_block blk{};
    blk.kind = FLYX_BLOCK_DSPINE_LAYER;
    auto& y = blk.dspine_layer;
    y.norm = T.norm(p + "in_norm", H, 0.0, eps);
    y.post_norm = T.norm(p + "post_norm", H, 0.0, eps);
    y.q_norm = T.norm(p + "attn.q_norm", b.head_dim, 0.0, eps);
    y.k_norm = T.norm(p + "attn.k_norm", b.head_dim, 0.0, eps);
    y.q = T.weight(p + "attn.q", {q, H});
    y.k = T.weight(p + "attn.k", {kv, H});
    y.v = T.weight(p + "attn.v", {kv, H});
    y.o = T.weight(p + "attn.o", {H, q});
    y.gate = T.weight(p + "mlp.gate", {I, H});
    y.up = T.weight(p + "mlp.up", {I, H});
    y.down = T.weight(p + "mlp.down", {H, I});
    y.read = T.weight(p + "bridge.read", {C, H});
    y.write = T.weight(p + "bridge.write", {H, R});
    T.block(blk);
  }
  b.fc = T.weight(dn("draft.fc"), {H, int64_t(b.n_taps) * t.hidden});
  b.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
  b.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  b.codebook = T.weight(dn("draft.bridge.codebook"), {t.vocab, C});
  b.gate = T.weight(dn("draft.bridge.gate"), {R, C});
  b.pred = T.weight(dn("draft.bridge.pred"), {R, C});
  b.prev = T.weight(dn("draft.bridge.prev"), {R, C});
  b.gate_bias = T.weight(dn("draft.bridge.gate_bias"), {R});
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = 1;
  st.layers = layers;
  st.n_kv = b.n_kv;
  st.head_dim = b.head_dim;
  b.kv = T.state(st);
  flyx_block blk{};
  blk.kind = FLYX_BLOCK_DSPINE;
  blk.dspine = b;
  return T.block(blk);
}

}  // namespace flyx::models::dspine
