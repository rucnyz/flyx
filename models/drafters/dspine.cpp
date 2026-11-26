// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine from a "draft" config of kind "dspine" and its draft.* tensors, the names python/convert writes
// (python/convert/flyx_convert/dspine.py): per layer L<j>.{in_norm, post_norm, attn.*, mlp.*, bridge.read,
// bridge.write}, once fc, hidden_norm, norm and bridge.{codebook, gate, pred, prev, gate_bias}.
#include "models/drafters/dspine.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/modelfile/modelfile.h"
#include "models/drafters/dflash2.h"
#include "models/drafters/qwen3_layer.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("DSpine drafter: " + what);
}

}  // namespace

DSpineSpec parse_dspine_json(const json& d, const TargetSpec& t, int companion) {
  expect(d.value("kind", "") == "dspine", "a config of another kind");
  check_config_target(d, t);
  DSpineSpec spec{};
  spec.companion = companion;
  flyx_dspine_block& b = spec.block;
  b.hidden = config_int(d, "hidden", "DSpine");
  spec.layers = config_int(d, "layers", "DSpine");
  b.n_heads = config_int(d, "n_heads", "DSpine");
  b.n_kv = config_int(d, "n_kv", "DSpine");
  b.head_dim = config_int(d, "head_dim", "DSpine");
  b.inter = config_int(d, "inter", "DSpine");
  b.block = config_int(d, "block", "DSpine");
  b.mask_token = config_int(d, "mask_token", "DSpine");
  b.code = config_int(d, "code", "DSpine");
  b.rank = config_int(d, "rank", "DSpine");
  b.topk = config_int(d, "topk", "DSpine");
  const int64_t win = d.value("window", int64_t(0));
  expect(win >= 0 && win <= INT_MAX, "window out of range");
  b.window = int32_t(win);
  b.theta = d.at("rope_theta").get<double>();
  b.bridge_eps = d.at("bridge_eps").get<double>();
  spec.eps = d.at("eps").get<double>();
  const int layers = spec.layers;
  const double eps = spec.eps;
  // The block's dimensions; the products the backends take of them fit an int.
  expect(b.hidden == t.hidden, "made for a target of width " + std::to_string(b.hidden) + ", not " +
                                   std::to_string(t.hidden));
  expect(b.block >= 2 && b.block <= FLYX_MAX_ROWS, "block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  expect(b.topk >= 2 && b.topk <= FLYX_DSPINE_MAX_TOPK,
         "the walk's candidates outside 2.." + std::to_string(FLYX_DSPINE_MAX_TOPK));
  expect(b.hidden > 0 && b.hidden <= kMaxWidth && layers > 0 && layers <= kMaxLayers && b.n_heads > 0 &&
             b.n_heads <= kMaxHeads && b.n_kv > 0 && b.n_heads % b.n_kv == 0 && b.head_dim > 0 &&
             b.head_dim % 2 == 0 && b.head_dim <= kMaxHeadDim && b.inter > 0 && b.inter <= kMaxWidth &&
             b.code > 0 && b.code <= kMaxWidth && b.rank > 0 && b.rank <= kMaxWidth && b.mask_token >= 0 &&
             b.mask_token < t.vocab && b.topk <= t.vocab && eps > 0 && std::isfinite(eps) && b.bridge_eps > 0 &&
             std::isfinite(b.bridge_eps) && b.theta > 0 && std::isfinite(b.theta),
         "invalid dimensions");
  const int64_t ctx = int64_t(2) * layers * b.n_kv * b.head_dim, qkv = int64_t(b.n_heads + 2 * b.n_kv) * b.head_dim;
  expect(ctx <= INT_MAX && qkv <= INT_MAX && int64_t(3) * b.rank <= INT_MAX, "dimensions too large");
  for (const auto& v : d.at("taps")) {
    const int64_t x = v.get<int64_t>();
    expect(x >= 1 && x <= t.layers, "a tap after layer " + std::to_string(x) + " of " + std::to_string(t.layers));
    spec.taps.push_back(int(x));
  }
  expect(!spec.taps.empty(), "no taps");
  expect(int64_t(spec.taps.size()) * t.hidden <= INT_MAX, "too many taps");
  b.n_taps = int32_t(spec.taps.size());
  return spec;
}

int build_dspine(Tables& T, const DSpineSpec& spec, const TargetSpec& t) {
  flyx_dspine_block b = spec.block;
  const int H = b.hidden, C = b.code, R = b.rank;
  const double eps = spec.eps;
  // The file's name for a drafter tensor ("draft.fc"): in the model file, or in its drafter file.
  auto dn = [&](const std::string& name) {
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, name) : name;
  };
  b.embed = t.embed;
  b.head = t.head;
  b.first_layer = int32_t(T.blocks.size());
  b.layers = spec.layers;
  for (int l = 0; l < spec.layers; ++l) {
    const std::string p = "draft.L" + std::to_string(l) + ".";
    flyx_block blk{};
    blk.kind = FLYX_BLOCK_DSPINE_LAYER;
    auto& y = blk.dspine_layer;
    bind_qwen3_layer(T, y, dn, l, H, b.n_heads, b.n_kv, b.head_dim, b.inter, eps);
    y.read = T.weight(dn(p + "bridge.read"), {C, H});
    y.write = T.weight(dn(p + "bridge.write"), {H, R});
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
  b.kv = draft_kv_state(T, spec.layers, b.n_kv, b.head_dim);
  flyx_block blk{};
  blk.kind = FLYX_BLOCK_DSPINE;
  blk.dspine = b;
  return T.block(blk);
}

}  // namespace flyx::models::drafters
