// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/dflash2.h"

#include <climits>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include "models/drafters/trunk.h"
#include "models/gguf.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

int req_int(const json& j, const char* key, const std::string& name = "DFlash2") {
  const auto& v = j.at(key);
  if (!v.is_number_integer()) throw std::runtime_error("invalid " + name + " drafter config");
  const int64_t n = v.get<int64_t>();
  if (n < INT_MIN || n > INT_MAX) throw std::runtime_error("invalid " + name + " drafter config");
  return int(n);
}

std::string mismatch_msg(const TargetSpec& target) {
  return "a drafter made for another model (this one: width " + std::to_string(target.hidden) + ", vocabulary " +
         std::to_string(target.vocab) + ", " + std::to_string(target.layers) + " layers)";
}

// A canonical config ("draft"): a block drafter (DFlash2, DFlash or DSpine, models/drafters/dspine.h), made for this
// target where it says what it was made for.
void check_json_target(const json& j, const TargetSpec& target) {
  const std::string kind = j.value("kind", "");
  if (kind != "dflash2" && kind != "dflash" && kind != "dspine") throw std::runtime_error(mismatch_msg(target));
  if (!j.contains("target") || !j["target"].is_object()) return;
  const json& t = j["target"];
  const bool layers_ok = !t.contains("layers") || t["layers"].is_null() || t["layers"].get<int>() == target.layers;
  if (t.value("hidden", target.hidden) != target.hidden || t.value("vocab", target.vocab) != target.vocab ||
      !layers_ok)
    throw std::runtime_error(mismatch_msg(target));
}

// An arch "dflash" GGUF header: the drafter reads the target's embedding and head, so its width is the target's.
void check_gguf_target(const GgufMeta& cm, const TargetSpec& target) {
  if (cm.count("embedding_length") != target.hidden ||
      (cm.has("vocab_size") && cm.count("vocab_size") != target.vocab))
    throw std::runtime_error(mismatch_msg(target));
}

// A drafter tensor's name in an arch "dflash" GGUF file, from the name python/convert gives it ("draft.fc",
// "draft.L<i>.attn.q"): its layers' projections and norms are named as a dense target's (models/gguf.h).
std::string gguf_draft_name(const std::string& name) {
  static const std::map<std::string_view, const char*> top = {
      {"draft.fc", "fc.weight"},
      {"draft.hidden_norm", "enc.output_norm.weight"},
      {"draft.norm", "output_norm.weight"},
      {"draft.sel.pred", "selector_predecessor.weight"},
      {"draft.sel.succ", "selector_successor.weight"},
      {"draft.sel.hid", "selector_hidden.weight"}};
  static const std::map<std::string_view, const char*> conv = {
      {"attn_conv.proj", "attn_conv_proj.weight"},
      {"attn_conv.base", "attn_conv_base"},
      {"mlp_conv.proj", "ffn_conv_proj.weight"},
      {"mlp_conv.base", "ffn_conv_base"}};
  if (const auto t = top.find(name); t != top.end()) return t->second;
  const size_t dot = name.find('.', 7);  // "draft.L<i>."
  const std::string_view suffix = std::string_view(name).substr(dot + 1);
  const char* dense = gguf_dense_layer_name(suffix, "ffn_norm.weight");
  return "blk." + name.substr(7, dot - 7) + "." + (dense ? dense : conv.at(suffix));
}

void validate_dflash2_spec(DFlash2Spec* spec, const TargetSpec& target) {
  flyx_dflash2_block& d = spec->block;
  const std::string name = spec->dflash ? "DFlash" : "DFlash2";
  if (d.block < 2 || d.block > FLYX_MAX_ROWS)
    throw std::runtime_error(name + " block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  if (d.hidden != target.hidden) throw std::runtime_error(name + " hidden size does not match target");
  if (d.hidden <= 0 || spec->layers <= 0 || d.n_heads <= 0 || d.n_kv <= 0 || d.head_dim <= 0 || d.head_dim % 2 != 0 ||
      d.inter <= 0 || d.n_heads % d.n_kv != 0 || d.mask_token < 0 || d.mask_token >= target.vocab ||
      !(spec->eps > 0) || !std::isfinite(spec->eps) || !(d.theta > 0) || !std::isfinite(d.theta) ||
      d.hidden > kMaxWidth || d.inter > kMaxWidth || spec->layers > kMaxLayers || d.n_heads > kMaxHeads ||
      d.head_dim > kMaxHeadDim || spec->taps.empty() || spec->taps.size() > size_t(INT_MAX) || d.window < 0)
    throw std::runtime_error("invalid " + name + " drafter config");
  d.n_taps = int32_t(spec->taps.size());
  if (spec->dflash) {  // the head's best for each row: one candidate, no convolutions to size
    if (d.conv_taps != 0 || d.conv_group != 0 || d.sel_rank != 0 || d.sel_topk != 1)
      throw std::runtime_error("invalid DFlash drafter config (convolutions or a selector: DFlash has neither)");
  } else {
    if (d.conv_taps <= 0 || d.conv_taps > kMaxHeadDim || d.sel_rank <= 0 || d.sel_rank > kMaxWidth ||
        d.sel_topk <= 0 || d.sel_topk > target.vocab)
      throw std::runtime_error("invalid DFlash2 drafter config");
    if (d.conv_group < 1 || d.hidden % d.conv_group)
      throw std::runtime_error("DFlash2 conv group does not divide hidden");
  }
  const int64_t tg2 = spec->dflash ? 0 : int64_t(2) * d.conv_taps * (d.hidden / d.conv_group);
  const int64_t tap_width = int64_t(d.n_taps) * target.hidden;
  const int64_t ctx_width = int64_t(2) * spec->layers * d.n_kv * d.head_dim;
  if (tg2 > INT_MAX || tap_width > INT_MAX || ctx_width > INT_MAX)
    throw std::runtime_error("invalid " + name + " drafter config");
  spec->tg2 = int(tg2);
}

}  // namespace

bool is_dflash_companion(const json& header) {
  if (!header.is_object()) return false;
  if (header.value("kind", "") == "drafter" && header.contains("draft") && header["draft"].is_object()) return true;
  return GgufMeta::is_header(header) && header.value("arch", "") == "dflash";
}

void check_made_for(const json& header, const TargetSpec& target) {
  if (GgufMeta::is_header(header)) check_gguf_target(GgufMeta(header), target);
  else check_json_target(header.at("draft"), target);
}

DFlash2Spec parse_dflash2_json(const json& j, const TargetSpec& target, int companion) {
  check_json_target(j, target);
  const std::string kind = j.value("kind", "");
  if (kind != "dflash2" && kind != "dflash") throw std::runtime_error("not a DFlash2 or DFlash drafter config");
  DFlash2Spec spec{};
  spec.gguf = false;
  spec.dflash = kind == "dflash";
  spec.companion = companion;
  const std::string name = spec.dflash ? "DFlash" : "DFlash2";
  flyx_dflash2_block& d = spec.block;
  d.hidden = req_int(j, "hidden", name);
  spec.layers = req_int(j, "layers", name);
  d.n_heads = req_int(j, "n_heads", name);
  d.n_kv = req_int(j, "n_kv", name);
  d.head_dim = req_int(j, "head_dim", name);
  d.inter = req_int(j, "inter", name);
  d.block = req_int(j, "block", name);
  d.mask_token = req_int(j, "mask_token", name);
  const auto raw_taps = j.at("taps").get<std::vector<int64_t>>();
  spec.taps.reserve(raw_taps.size());
  for (int64_t t : raw_taps) {
    if (t < 1 || t > target.layers) throw std::runtime_error(name + " tap after layer " + std::to_string(t));
    spec.taps.push_back(int(t));
  }
  d.conv_taps = req_int(j, "conv_taps", name);
  d.conv_group = req_int(j, "conv_group", name);
  d.sel_rank = req_int(j, "sel_rank", name);
  d.sel_topk = req_int(j, "sel_topk", name);
  d.theta = j.at("rope_theta").get<double>();
  const int64_t win = j.value("window", int64_t(0));
  if (win < 0 || win > INT_MAX) throw std::runtime_error(name + " window out of range");
  d.window = int32_t(win);
  spec.eps = j.at("eps").get<double>();
  validate_dflash2_spec(&spec, target);
  return spec;
}

DFlash2Spec parse_dflash2_gguf(const ModelFile& file, const json& header, const TargetSpec& target, int companion) {
  const GgufMeta cm(header);
  const json& g = header.at("gguf");
  check_gguf_target(cm, target);
  // What the layout can say that the block here does not run: refused, never drafted as the plain block.
  auto set = [&](const char* key) { return cm.has(key) && cm.at(key).is_boolean() && cm.at(key).get<bool>(); };
  if (set("has_markov_context"))
    throw std::runtime_error("DSpark drafters (dflash.has_markov_context = true) are not supported by this build");
  if (set("attention.causal"))
    throw std::runtime_error("causal DFlash2 layers are not supported (the block attends both ways)");
  if (set("sample_from_anchor")) throw std::runtime_error("sample_from_anchor=True is not supported");
  if (set("has_confidence_head"))
    throw std::runtime_error("a DFlash2 confidence head (dflash.has_confidence_head) is not supported");
  if (cm.has("final_logit_softcapping") &&
      !(cm.at("final_logit_softcapping").is_number() && cm.at("final_logit_softcapping").get<double>() == 0))
    throw std::runtime_error("DFlash2 logit soft-capping (dflash.final_logit_softcapping) is not supported");

  DFlash2Spec spec{};
  spec.gguf = true;
  spec.companion = companion;
  flyx_dflash2_block& d = spec.block;
  d.hidden = cm.count("embedding_length");
  spec.layers = cm.count("block_count");
  d.inter = cm.count("feed_forward_length");
  d.n_heads = cm.count("attention.head_count");
  d.n_kv = cm.count("attention.head_count_kv");
  if (!cm.has("attention.key_length") && d.hidden % d.n_heads)
    throw std::runtime_error("DFlash2 GGUF hidden size is not a whole number of heads");
  d.head_dim = cm.count_or("attention.key_length", d.hidden / d.n_heads);
  if (cm.count_or("attention.value_length", d.head_dim) != d.head_dim)
    throw std::runtime_error("DFlash2 GGUF attention key and value lengths differ");
  d.theta = cm.real_or("rope.freq_base", 10000.0);
  spec.eps = cm.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  d.block = cm.count("block_size");
  d.conv_taps = cm.count_or("conv_kernel_size", 2);
  d.conv_group = cm.count_or("conv_group_size", 16);
  d.sel_rank = cm.count_or("selector_rank", 256);
  d.sel_topk = cm.count_or("selector_top_k", 16);

  int win = cm.count_or("attention.sliding_window", 0, 0);
  if (cm.has("attention.sliding_window_pattern")) {
    const json& pat = cm.at("attention.sliding_window_pattern");
    if (!pat.is_array() || int(pat.size()) != spec.layers)
      throw std::runtime_error("DFlash2 GGUF attention.sliding_window_pattern does not match block_count");
    bool any_true = false, any_false = false;
    for (const auto& flag : pat) {
      if (!flag.is_boolean())
        throw std::runtime_error("DFlash2 GGUF attention.sliding_window_pattern must be booleans");
      if (flag.get<bool>()) any_true = true;
      else any_false = true;
    }
    if (any_true && any_false)
      throw std::runtime_error(
          "DFlash2 GGUF attention.sliding_window_pattern must be uniform (all full or all sliding)");
    if (!any_true) win = 0;
    else if (win <= 0)
      throw std::runtime_error("DFlash2 GGUF sliding_window_pattern enabled without positive attention.sliding_window");
  }
  d.window = win;

  const json& raw_taps = cm.at("target_layers");
  if (!raw_taps.is_array() || raw_taps.empty())
    throw std::runtime_error("DFlash2 GGUF missing or empty dflash.target_layers");
  spec.taps.reserve(raw_taps.size());
  for (const auto& item : raw_taps) {
    if (!item.is_number_integer()) throw std::runtime_error("DFlash2 GGUF target_layers must be integers");
    const int64_t t = item.get<int64_t>();
    if (t < 1 || t > target.layers) throw std::runtime_error("DFlash2 tap after layer " + std::to_string(t));
    // The layout's indices are transformers' hidden states (the published 27B drafter's are its config's
    // target_layer_ids + 1), and the last of those is the final norm's output.  A tap reads the residual stream
    // before that norm: a drafter made on the last one is refused, never given the other.
    if (t == target.layers)
      throw std::runtime_error("DFlash2 GGUF taps the target's final hidden state (dflash.target_layers " +
                               std::to_string(t) + "), read after the output norm: not supported yet");
    spec.taps.push_back(int(t));
  }

  int mask_token = -1;
  if (cm.has("mask_token_id")) {
    mask_token = cm.count("mask_token_id", 0);
  } else if (g.contains("tokenizer.ggml.mask_token_id") && g["tokenizer.ggml.mask_token_id"].is_number_integer()) {
    const int64_t mt = g["tokenizer.ggml.mask_token_id"].get<int64_t>();
    if (mt < 0 || mt > INT_MAX) throw std::runtime_error("DFlash2 GGUF mask_token_id out of range");
    mask_token = int(mt);
  } else {
    mask_token = find_added_token(file, "<|mask|>");
  }
  if (mask_token < 0)
    throw std::runtime_error(
        "DFlash2 GGUF missing mask_token_id (expected dflash.mask_token_id or tokenizer.ggml.mask_token_id)");
  d.mask_token = mask_token;

  validate_dflash2_spec(&spec, target);
  return spec;
}

int build_dflash2(Tables& T, const DFlash2Spec& spec, const TargetSpec& target) {
  flyx_dflash2_block d = spec.block;
  const int layers = spec.layers;
  const double eps = spec.eps;
  const int H = d.hidden, TG2 = spec.tg2;
  // The file's name for a drafter tensor ("draft.fc"): in the model file, or in its drafter file.
  auto dn = [&](const std::string& name) {
    const std::string in_file = spec.gguf ? gguf_draft_name(name) : name;
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, in_file) : in_file;
  };

  d.embed = target.embed;
  d.head = target.head;
  d.first_layer = int32_t(T.blocks.size());
  d.layers = layers;
  for (int l = 0; l < layers; ++l) {
    const std::string p = "draft.L" + std::to_string(l) + ".";
    flyx_block b{};
    b.kind = FLYX_BLOCK_DFLASH2_LAYER;
    auto& y = b.dflash2_layer;
    bind_qwen3_layer(T, y, dn, l, H, d.n_heads, d.n_kv, d.head_dim, d.inter, eps);
    y.attn_conv_proj = y.attn_conv_base = y.mlp_conv_proj = y.mlp_conv_base = -1;  // DFlash: none
    if (!spec.dflash) {
      y.attn_conv_proj = T.weight(dn(p + "attn_conv.proj"), {TG2, H});
      y.attn_conv_base = T.weight(dn(p + "attn_conv.base"), {2, d.conv_taps, H});
      y.mlp_conv_proj = T.weight(dn(p + "mlp_conv.proj"), {TG2, H});
      y.mlp_conv_base = T.weight(dn(p + "mlp_conv.base"), {2, d.conv_taps, H});
    }
    T.block(b);
  }
  d.fc = T.weight(dn("draft.fc"), {H, int64_t(d.n_taps) * target.hidden});
  d.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
  d.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  d.sel_pred = d.sel_succ = d.sel_hid = -1;  // DFlash: none
  if (!spec.dflash) {
    d.sel_pred = T.weight(dn("draft.sel.pred"), {target.vocab, d.sel_rank});
    d.sel_succ = T.weight(dn("draft.sel.succ"), {target.vocab, d.sel_rank});
    d.sel_hid = T.weight(dn("draft.sel.hid"), {d.sel_rank, H});
  }
  d.kv = draft_kv_state(T, layers, d.n_kv, d.head_dim);
  flyx_block b{};
  b.kind = FLYX_BLOCK_DFLASH2;
  b.dflash2 = d;
  return T.block(b);
}

}  // namespace flyx::models::drafters
