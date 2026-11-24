// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A block drafter as its authors publish it, DFlash2's or DSpine's: config.json and *.safetensors in a directory.
// Read into what python/convert/flyx_convert makes of it as a drafter file (--draft alone): the same "draft"
// description (canonical_draft_config with no target) and the same tensors under the same names, their bytes read in
// place (a DSpine drafter's stacked bridge tensors as views of each layer's part, flyx_convert/dspine.py).
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>

#include "core/interfaces/json_depth.h"
#include "core/modelfile/readers.h"
#include "core/os/path.h"

namespace flyx::modelfile {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

using os::from_path;
using os::to_path;

int dtype_of(const std::string& s) {
  static const std::map<std::string, int> m = {{"BF16", FLYX_DT_BF16}, {"F16", FLYX_DT_F16}, {"F32", FLYX_DT_F32},
                                               {"F64", FLYX_DT_F64},   {"I32", FLYX_DT_I32}, {"I64", FLYX_DT_I64},
                                               {"U8", FLYX_DT_U8},     {"F8_E4M3", FLYX_DT_F8E4M3}};
  auto it = m.find(s);
  return it == m.end() ? -1 : it->second;
}

void dspine_require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("DSpine drafter: " + what);
}

// python/convert/flyx_convert/dspine.py's is_dspine.
bool is_dspine(const json& sc) {
  const auto archs = sc.value("architectures", json::array());
  if (std::find(archs.begin(), archs.end(), "DSpineDraftModel") != archs.end()) return true;
  return sc.contains("dflash_config") && sc["dflash_config"].is_object() &&
         sc["dflash_config"].value("selected_prefix_code_bridge", json()).is_object();
}

bool falsy(const json& v) {
  return v.is_null() || (v.is_boolean() && !v.get<bool>()) || (v.is_number() && v.get<double>() == 0) ||
         (v.is_string() && v.get<std::string>().empty()) || (v.is_array() && v.empty()) ||
         (v.is_object() && v.empty());
}

// python/convert/flyx_convert/dspine.py's canonical_dspine_config(config, target=None): the variant both released
// drafters are, anything else refused.
json dspine_config(const json& sc) {
  dspine_require(sc.contains("dflash_config") && sc["dflash_config"].is_object(), "no dflash_config");
  const json& dc = sc["dflash_config"];
  const int64_t layers = sc.at("num_hidden_layers").get<int64_t>();
  dspine_require(layers > 0 && layers <= 1024, "num_hidden_layers out of range");
  dspine_require(sc.value("model_type", "qwen3") == "qwen3", "model_type (qwen3 layers)");
  dspine_require(sc.value("hidden_act", "silu") == "silu", "hidden_act (silu)");
  dspine_require(!sc.value("attention_bias", false), "attention biases");
  dspine_require(dc.value("draft_block_causal", json()) == json(true), "draft_block_causal must be true");
  dspine_require(dc.value("selected_prefix_terminal", json()) == json(false), "a terminal write");
  dspine_require(dc.value("selected_prefix_decode_mode", "") == "serial", "decode mode (serial)");
  dspine_require(dc.value("selected_prefix_feature_writes", json()) == json(true) &&
                     dc.value("selected_prefix_feature_write_mode", "") == "independent",
                 "independent writes after the layers");
  json every = json::array();
  for (int64_t i = 1; i <= layers; ++i) every.push_back(i);
  dspine_require(dc.value("selected_prefix_feature_layers", json()) == every, "a write after every layer");
  dspine_require(falsy(dc.value("selected_prefix_terminal_residual_rank", json())), "a terminal residual");
  dspine_require(falsy(dc.value("selected_prefix_feature_next_token_rows", json())), "next-token rows");
  dspine_require(dc.value("selected_prefix_feature_reliability", json()).is_null(), "feature reliability weights");
  dspine_require(dc.value("attention_value_scale", json()).is_null(), "an attention value scale");
  for (const char* k : {"conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k", "projector_type",
                        "attention_sink_bias"})
    dspine_require(falsy(dc.value(k, json())), std::string(k) + " (DFlash2's parts)");
  std::vector<int64_t> widths;
  const json refine = dc.value("selected_prefix_last_write_refine", json());
  if (!refine.is_null()) {
    dspine_require(refine.value("kind", "") == "exact_sender" && refine.value("version", 0) == 1,
                   "last-write refinement (exact_sender, version 1)");
    widths.push_back(refine.at("head_topk").get<int64_t>());
  }
  const json second = sc.value("code_bridge_second_pass", json());
  if (!second.is_null()) {
    dspine_require(second.value("mode", "") == "serial_table" && second.value("first_exact", int64_t(-1)) == layers - 1,
                   "second pass (serial_table, the last layer's write)");
    widths.push_back(second.at("head_topk").get<int64_t>());
  }
  dspine_require(!widths.empty() && std::all_of(widths.begin(), widths.end(), [&](int64_t w) { return w == widths[0]; }),
                 "refinement widths (one, given at least once)");
  dspine_require(widths[0] >= 2 && widths[0] <= 16, "refinement width (2..16)");
  const json bridge = dc.value("selected_prefix_code_bridge", json());
  dspine_require(bridge.is_object(), "no selected_prefix_code_bridge");
  const json rp = sc.value("rope_parameters", json::object());
  dspine_require((rp.is_object() && rp.value("rope_type", rp.value("type", "default")) == "default") &&
                     sc.value("rope_scaling", json()).is_null(),
                 "rope scaling (default rotary only)");
  std::vector<std::string> types(size_t(layers), "full_attention");
  if (sc.contains("layer_types") && sc["layer_types"].is_array() && !sc["layer_types"].empty())
    types = sc["layer_types"].get<std::vector<std::string>>();
  auto all = [&](const char* t) { return std::all_of(types.begin(), types.end(), [&](const std::string& x) { return x == t; }); };
  int64_t window = 0;
  if (all("full_attention")) {
    dspine_require(!sc.value("use_sliding_window", false), "use_sliding_window without sliding layers");
  } else {
    const json w = sc.value("sliding_window", json());
    dspine_require(all("sliding_attention") && sc.value("use_sliding_window", false) &&
                       sc.value("max_window_layers", layers) == layers && w.is_number_integer() &&
                       w.get<int64_t>() > 0,
                   "layer types: all full, or all sliding with one window");
    window = w.get<int64_t>();
  }
  std::vector<int64_t> taps;
  for (const auto& i : dc.at("target_layer_ids")) taps.push_back(i.get<int64_t>() + 1);
  const int64_t vocab = sc.at("vocab_size").get<int64_t>(), hidden = sc.at("hidden_size").get<int64_t>();
  const json dvocab = sc.value("draft_vocab_size", json());
  dspine_require(falsy(dvocab) || dvocab.get<int64_t>() == vocab, "a draft vocabulary of its own");
  dspine_require(!sc.value("tie_word_embeddings", false), "tied embeddings (it reads the target's head)");
  const int64_t n_heads = sc.at("num_attention_heads").get<int64_t>();
  dspine_require(hidden > 0 && n_heads > 0, "hidden_size and num_attention_heads must be positive");
  const json hd_given = sc.value("head_dim", json());
  const int64_t hd = hd_given.is_number() && hd_given.get<int64_t>() != 0 ? hd_given.get<int64_t>() : hidden / n_heads;
  json theta = rp.value("rope_theta", sc.value("rope_theta", json()));
  dspine_require(theta.is_number(), "no rope_theta");
  json d = json::object();
  d["kind"] = "dspine";
  d["hidden"] = hidden;
  d["layers"] = layers;
  d["n_heads"] = n_heads;
  d["n_kv"] = sc.at("num_key_value_heads");
  d["head_dim"] = hd;
  d["inter"] = sc.at("intermediate_size");
  d["eps"] = sc.at("rms_norm_eps");
  d["rope_theta"] = theta.get<double>();
  d["block"] = dc.contains("block_size") ? dc["block_size"] : sc.at("block_size");
  d["mask_token"] = dc.at("mask_token_id");
  d["taps"] = taps;
  d["window"] = window;
  d["code"] = bridge.at("code_size");
  d["rank"] = bridge.at("rank");
  d["bridge_eps"] = bridge.at("eps").get<double>();
  d["topk"] = widths[0];
  d["target"] = {{"hidden", hidden}, {"vocab", vocab}, {"layers", sc.value("num_target_layers", json())}};
  return d;
}

// A DSpine drafter's tensors that only say which variant it is: each holds 1 (flyx_convert/dspine.py's VERSIONS).
bool dspine_version(const std::string& inner) {
  return inner == "draft_block_causal_version" || inner == "last_write_refine_version" ||
         inner == "selected_prefix_bridge.independent_write_version" ||
         inner == "selected_prefix_bridge.terminal_disabled_version";
}
// Its terminal write's projection, which a drafter without one holds as zeros.
constexpr const char* kTerminal = "selected_prefix_bridge.out.weight";
// Its stacked tensors: 1 the last layer's read (read_weight), 2 the other layers' (early_read_weights), 3 every
// layer's write (feature_write_weights); 0 none.
int dspine_stacked(const std::string& inner) {
  if (inner == "selected_prefix_bridge.read_weight") return 1;
  if (inner == "selected_prefix_bridge.early_read_weights") return 2;
  if (inner == "selected_prefix_bridge.feature_write_weights") return 3;
  return 0;
}

// python/convert's canonical_draft_config(config, target=None).
json draft_config(const json& sc) {
  if (is_dspine(sc)) return dspine_config(sc);
  json dc, tl;
  std::vector<int64_t> taps;
  int64_t vocab, target_hidden;
  json causal;  // null: causal where the layers slide (z-lab's code)
  if (sc.contains("dflash_config")) {
    const auto archs = sc.value("architectures", json::array());
    if (std::find(archs.begin(), archs.end(), "DFlash2DraftModel") == archs.end())
      throw std::runtime_error("drafter architectures " + archs.dump() + ": DFlash2DraftModel is supported");
    dc = sc.at("dflash_config"), tl = sc;
    for (const auto& i : dc.at("target_layer_ids")) taps.push_back(i.get<int64_t>() + 1);
    vocab = tl.at("vocab_size").get<int64_t>(), target_hidden = tl.at("hidden_size").get<int64_t>();
    causal = sc.value("is_causal", json());
  } else {
    if (sc.value("speculators_model_type", "") != "dflash2")
      throw std::runtime_error("drafter type " + sc.value("speculators_model_type", json()).dump() +
                               " is not supported (need dflash2)");
    if (sc.value("sample_from_anchor", false)) throw std::runtime_error("sample_from_anchor=True is not supported");
    dc = sc, tl = sc.at("transformer_layer_config");
    for (const auto& i : sc.at("aux_hidden_state_layer_ids")) taps.push_back(i.get<int64_t>());
    const json dv = sc.value("draft_vocab_size", json());
    vocab = dv.is_number_integer() && dv.get<int64_t>() != 0 ? dv.get<int64_t>() : tl.at("vocab_size").get<int64_t>();
    const json th = sc.value("target_hidden_size", json());
    target_hidden = th.is_number() && th.get<int64_t>() != 0 ? th.get<int64_t>() : tl.at("hidden_size").get<int64_t>();
    causal = false;
  }
  json rp = tl.value("rope_parameters", json());
  if (rp.is_null() || rp.empty()) rp = tl.value("rope_scaling", json());
  if (rp.is_null()) rp = json::object();
  const int64_t layers = tl.at("num_hidden_layers").get<int64_t>();
  if (layers <= 0 || layers > 1024) throw std::runtime_error("draft num_hidden_layers out of range");
  std::vector<std::string> types;
  if (tl.contains("layer_types") && tl["layer_types"].is_array() && !tl["layer_types"].empty())
    types = tl["layer_types"].get<std::vector<std::string>>();
  else
    types.assign(size_t(layers), "full_attention");
  int64_t window = 0;
  auto all = [&](const char* t) { return std::all_of(types.begin(), types.end(), [&](const std::string& x) { return x == t; }); };
  if (all("full_attention")) {
    window = 0;
  } else if (all("sliding_attention") && tl.value("sliding_window", json()).is_number() && tl["sliding_window"].get<int64_t>()) {
    window = tl["sliding_window"].get<int64_t>();
  } else {
    throw std::runtime_error("draft layer types: all full or all sliding with one window");
  }
  const bool is_causal = causal.is_null() ? window > 0 : causal.get<bool>();
  if (is_causal) throw std::runtime_error("causal DFlash2 layers are not supported (the block attends both ways)");
  const int64_t hidden = tl.at("hidden_size").get<int64_t>();
  const int64_t n_heads = tl.at("num_attention_heads").get<int64_t>();
  if (hidden <= 0 || n_heads <= 0) throw std::runtime_error("draft hidden_size and num_attention_heads must be positive");
  const json hd_given = tl.value("head_dim", json());
  const int64_t hd = hd_given.is_number() && hd_given.get<int64_t>() != 0 ? hd_given.get<int64_t>() : hidden / n_heads;
  if (hd <= 0) throw std::runtime_error("draft head_dim must be positive");
  double theta = 10000.0;
  if (rp.contains("rope_theta") && rp["rope_theta"].is_number()) theta = rp["rope_theta"].get<double>();
  else if (tl.contains("rope_theta") && tl["rope_theta"].is_number()) theta = tl["rope_theta"].get<double>();
  json d = json::object();
  d["kind"] = "dflash2";
  d["hidden"] = tl.at("hidden_size");
  d["layers"] = tl.at("num_hidden_layers");
  d["n_heads"] = tl.at("num_attention_heads");
  d["n_kv"] = tl.at("num_key_value_heads");
  d["head_dim"] = hd;
  d["inter"] = tl.at("intermediate_size");
  d["eps"] = tl.at("rms_norm_eps");
  d["rope_theta"] = theta;
  d["block"] = dc.at("block_size");
  d["mask_token"] = dc.at("mask_token_id");
  d["taps"] = taps;
  d["conv_taps"] = dc.value("conv_kernel_size", json(2));
  d["conv_group"] = dc.value("conv_group_size", json(16));
  d["sel_rank"] = dc.value("selector_rank", json(256));
  d["sel_topk"] = dc.value("selector_top_k", json(16));
  d["window"] = window;
  d["target"] = {{"hidden", target_hidden}, {"vocab", vocab}, {"layers", sc.value("num_target_layers", json())}};
  return d;
}

// python/convert's _D_SKIP and _D_RULES: the drafter's tensor names as the engine knows them, or "" to leave out.
bool canonical_as(const std::string& name, bool dspine, std::string* out);

bool canonical(const std::string& name, std::string* out) {
  return canonical_as(name, false, out);
}

// The same for a DSpine drafter (flyx_convert/dspine.py's _LAYER_RULES and _BRIDGE_NAMES): its layers' tensors as
// DFlash2's are named, the bridge's own; "" for a tensor that only says the variant (checked by the caller), or a
// stacked one the caller cuts (read_weight, early_read_weights, feature_write_weights).
bool canonical_dspine(const std::string& name, std::string* out) {
  static const std::map<std::string, std::string> bridge = {
      {"selected_prefix_bridge.codebook", "bridge.codebook"}, {"selected_prefix_bridge.gate.weight", "bridge.gate"},
      {"selected_prefix_bridge.gate.bias", "bridge.gate_bias"},
      {"selected_prefix_bridge.predecessor_gate.weight", "bridge.pred"},
      {"selected_prefix_bridge.prev.weight", "bridge.prev"}};
  const std::string inner = name.rfind("model.", 0) == 0 ? name.substr(6) : name;
  if (auto it = bridge.find(inner); it != bridge.end()) return *out = it->second, true;
  return canonical_as(name, true, out);
}

bool canonical_as(const std::string& name, bool dspine, std::string* out) {
  static const std::regex skip(R"(^(embed_tokens\.|lm_head\.|verifier_norm\.|verifier_lm_head\.|t2d|d2t|mask_embedding).*)");
  static const std::vector<std::pair<std::regex, std::string>> rules = {
      {std::regex(R"(fc\.weight)"), "fc"},
      {std::regex(R"(hidden_norm\.weight)"), "hidden_norm"},
      {std::regex(R"(norm\.weight)"), "norm"},
      {std::regex(R"(layers\.(\d+)\.input_layernorm\.weight)"), "L$1.in_norm"},
      {std::regex(R"(layers\.(\d+)\.post_attention_layernorm\.weight)"), "L$1.post_norm"},
      {std::regex(R"(layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight)"), "L$1.attn.$2"},
      {std::regex(R"(layers\.(\d+)\.self_attn\.(q|k)_norm\.weight)"), "L$1.attn.$2_norm"},
      {std::regex(R"(layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight)"), "L$1.mlp.$2"},
      {std::regex(R"(layers\.(\d+)\.(attention|mlp)_conv\.base_kernel)"), "L$1.$2_conv.base"},
      {std::regex(R"(layers\.(\d+)\.(attention|mlp)_conv\.kernel_projection\.weight)"), "L$1.$2_conv.proj"},
      {std::regex(R"(candidate_selector\.predecessor_codebook)"), "sel.pred"},
      {std::regex(R"(candidate_selector\.successor_codebook)"), "sel.succ"},
      {std::regex(R"(candidate_selector\.hidden_projection\.weight)"), "sel.hid"},
  };
  const std::string inner = name.rfind("model.", 0) == 0 ? name.substr(6) : name;
  if (std::regex_match(inner, skip)) return out->clear(), true;
  for (const auto& [re, fmt] : rules) {
    if (dspine && (fmt.find("conv") != std::string::npos || fmt.rfind("sel.", 0) == 0)) continue;
    std::smatch m;
    if (!std::regex_match(inner, m, re)) continue;
    std::string s = m.format(fmt);
    const size_t at = s.find("attention_conv");
    if (at != std::string::npos) s.replace(at, 14, "attn_conv");
    *out = s;
    return true;
  }
  return false;
}

}  // namespace

bool is_drafter_dir(const std::string& path) {
  std::error_code ec;
  return fs::is_directory(to_path(path), ec) && fs::is_regular_file(to_path(path) / "config.json", ec);
}

bool read_drafter_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                      ModelFile::Contents* out, std::string* err) {
  const fs::path root = to_path(dir);
  std::vector<fs::path> shards;
  std::error_code ec;
  for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
    if (it->path().extension() == ".safetensors") shards.push_back(it->path());
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) return *err = dir + ": no *.safetensors beside its config.json", false;
  try {
    json sc;
    {
      constexpr size_t kMaxConfigBytes = 16u << 20;
      std::ifstream f(root / "config.json", std::ios::binary | std::ios::ate);
      const std::streamoff fsz = f.tellg();
      if (!f || fsz < 0 || size_t(fsz) > kMaxConfigBytes)
        return *err = dir + ": cannot read config.json (or larger than 16 MiB)", false;
      f.seekg(0);
      std::string cfg_text(size_t(fsz), '\0');
      if (fsz > 0 && !f.read(cfg_text.data(), fsz)) return *err = dir + ": cannot read config.json", false;
      if (!interfaces::json_depth_within(cfg_text))
        return *err = dir + ": config.json nested too deeply", false;
      sc = json::parse(cfg_text);
    }
    const json draft = draft_config(sc);
    const bool dspine = draft.at("kind") == "dspine";
    *out = {};
    out->arch = "drafter";
    out->header = {{"arch", "drafter"}, {"kind", "drafter"}, {"draft", draft},
                   {"source", {{"draft", from_path(root.filename().empty() ? root.parent_path().filename() : root.filename())}}}};
    std::vector<std::string> seen;
    for (const fs::path& shard : shards) {
      auto map = std::make_unique<os::MappedFile>();
      if (!map->open(from_path(shard), err)) return false;
      const uint8_t* p = map->data();
      const size_t size = map->size();
      uint64_t hlen = 0;
      if (size < 8) return *err = from_path(shard) + ": not a safetensors file", false;
      std::memcpy(&hlen, p, 8);
      if (hlen > size - 8) return *err = from_path(shard) + ": truncated header", false;
      const std::string_view hdr_sv(reinterpret_cast<const char*>(p + 8), size_t(hlen));
      if (!interfaces::json_depth_within(hdr_sv))
        return *err = from_path(shard) + ": header JSON nested too deeply", false;
      const json h = json::parse(hdr_sv);
      const size_t data_start = 8 + size_t(hlen), data_size = size - data_start;
      for (auto it = h.begin(); it != h.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const std::string inner = it.key().rfind("model.", 0) == 0 ? it.key().substr(6) : it.key();
        // A DSpine drafter's tensors that only say its variant, its terminal projection (none: zero), and the
        // stacked ones cut below.
        const bool version = dspine && dspine_version(inner), terminal = dspine && inner == kTerminal;
        const int stacked = dspine ? dspine_stacked(inner) : 0;
        std::string canon;
        if (!version && !terminal && !stacked &&
            !(dspine ? canonical_dspine(it.key(), &canon) : canonical(it.key(), &canon)))
          return *err = "unknown draft tensor " + it.key(), false;
        if (!version && !terminal && !stacked && canon.empty()) continue;  // the target's embedding and head
        const json& r = it.value();
        flyx_tensor_view v{};
        v.dtype = dtype_of(r.at("dtype").get<std::string>());
        const auto shape = r.at("shape").get<std::vector<int64_t>>();
        const auto offsets = r.at("data_offsets").get<std::vector<uint64_t>>();
        if (v.dtype < 0 || shape.size() > 4 || offsets.size() != 2 || offsets[0] > offsets[1] ||
            offsets[1] > data_size)
          return *err = it.key() + ": unsupported dtype or rank, or outside the file (truncated download?)", false;
        v.ndim = int32_t(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
          if (shape[i] < 0) return *err = it.key() + ": a negative dimension", false;
          v.shape[i] = shape[i];
        }
        uint64_t want_bytes = 0;
        if (!plain_tensor_nbytes(v.dtype, shape, &want_bytes) || offsets[1] - offsets[0] != want_bytes)
          return *err = it.key() + ": byte size does not match shape and dtype", false;
        v.data = p + data_start + offsets[0];
        v.nbytes = offsets[1] - offsets[0];
        auto add = [&](const std::string& c, const flyx_tensor_view& t) {
          const std::string name = "draft." + c;
          out->index[name] = out->tensors.size();
          out->names.push_back(name);
          out->tensors.push_back(t);
          seen.push_back(c);
        };
        if (version) {
          int64_t one = 0;
          if (v.dtype != FLYX_DT_I64 || v.nbytes != 8 || (std::memcpy(&one, v.data, 8), one != 1))
            return *err = dir + ": DSpine drafter: " + inner + " is not 1", false;
        } else if (terminal) {
          const auto* b = static_cast<const uint8_t*>(v.data);
          for (uint64_t i = 0; i < v.nbytes; ++i)  // bf16 or binary16: a sign bit alone is a zero too
            if (b[i] && !(v.dtype != FLYX_DT_F32 && i % 2 == 1 && b[i] == 0x80))
              return *err = dir + ": DSpine drafter: a terminal projection that is not zero", false;
        } else if (stacked) {
          // [n, ...] -> n views of one layer each; the last layer's read stands alone (read_weight)
          const int64_t layers = draft.at("layers").get<int64_t>();
          const bool read = stacked != 3, last = stacked == 1;
          const int64_t n = last ? 1 : v.shape[0];
          if (!last && (v.ndim != 3 || n != (read ? layers - 1 : layers)))
            return *err = dir + ": DSpine drafter: " + inner + " is not one per layer", false;
          flyx_tensor_view part = v;
          if (!last) {
            part.ndim = 2;
            part.shape[0] = v.shape[1];
            part.shape[1] = v.shape[2];
            part.shape[2] = 0;
            part.nbytes = v.nbytes / uint64_t(n);
          }
          for (int64_t j = 0; j < n; ++j) {
            part.data = static_cast<const uint8_t*>(v.data) + uint64_t(j) * part.nbytes;
            add("L" + std::to_string(last ? layers - 1 : j) + (read ? ".bridge.read" : ".bridge.write"), part);
          }
        } else {
          add(canon, v);
        }
      }
      maps->push_back(std::move(map));
    }
    for (size_t i = 0; i < out->tensors.size(); ++i) out->tensors[i].name = out->names[i].c_str();
    // python/convert's _check_draft (flyx_convert/dspine.py's check_dspine)
    std::vector<std::string> need = {"fc", "hidden_norm", "norm"};
    for (const char* n : dspine ? std::vector<const char*>{"bridge.codebook", "bridge.gate", "bridge.gate_bias",
                                                           "bridge.pred", "bridge.prev"}
                                : std::vector<const char*>{"sel.pred", "sel.succ", "sel.hid"})
      need.push_back(n);
    for (int64_t j = 0; j < draft.at("layers").get<int64_t>(); ++j) {
      for (const char* n : {"in_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "attn.q_norm",
                            "attn.k_norm", "mlp.gate", "mlp.up", "mlp.down"})
        need.push_back("L" + std::to_string(j) + "." + n);
      for (const char* n : dspine ? std::vector<const char*>{"bridge.read", "bridge.write"}
                                  : std::vector<const char*>{"attn_conv.base", "attn_conv.proj", "mlp_conv.base",
                                                             "mlp_conv.proj"})
        need.push_back("L" + std::to_string(j) + "." + n);
    }
    std::sort(seen.begin(), seen.end());
    for (const std::string& n : need)
      if (!std::binary_search(seen.begin(), seen.end(), n)) return *err = dir + ": missing draft tensor " + n, false;
  } catch (const std::exception& e) {
    *err = dir + ": " + e.what();
    return false;
  }
  return true;
}

}  // namespace flyx::modelfile
