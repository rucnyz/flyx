// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Qwen3.5 from its model file: the layer math as block flags (attention with
// an output gate, (1 + w) norms and partial rotary; GDN with a plain-weight
// output norm; the DFlash2 and DSpine drafters with plain norms), and the two ways a file
// stores the same model:
//   .flyx  the config in the header, tensors under the names
//             python/convert writes ("target.L3.attn.q"), laid out for the
//             engine (the gate's rows after q's, value heads grouped by key
//             head, norm weights w, the decay as A_log)
//   GGUF      llama.cpp's metadata keys and tensor names ("blk.3.attn_q.weight"),
//             laid out as its converter writes them: norm weights as 1 + w,
//             each head's gate rows after its q rows, value heads tiled over
//             the key heads, the decay as -exp(A_log).  The description says
//             so in its flags (backend_abi.h); no tensor is rewritten.
#include "models/qwen35/qwen35.h"

#include <climits>
#include <cmath>
#include <stdexcept>

#include "models/drafters/registry.h"
#include "models/gguf.h"
#include "models/vision/preprocess.h"

namespace flyx::models::qwen35 {
namespace {

using nlohmann::json;

struct VisionCfg {
  bool enabled = false;
  int companion = -1;       // companion index holding mmproj GGUF (-1 = embedded .flyx)
  int depth = 0;
  int hidden = 0;
  int inter = 0;
  int n_heads = 0;
  int head_dim = 0;
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;
  int out_hidden = 0;
  double eps = 1e-6;
  double rope_theta = 10000.0;
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float std[3] = {0.5f, 0.5f, 0.5f};
  int64_t min_pixels = 65536;
  int64_t max_pixels = 16777216;
  int32_t image_token_id = -1;
};

struct Config {
  int hidden, layers, vocab, inter;
  int n_heads, n_kv, head_dim, rot_dim;
  int nk, nv, dk, dv, conv;
  double theta, eps;
  std::vector<bool> is_attn;
  int mrope_section[3] = {0, 0, 0};
  bool mrope_interleaved = false;
  int trained_ctx = 0;  // the context the model was made for (0: the file does not say)
  bool gguf = false;    // the file's names and layout (above)
  bool mtp = false;     // the model file itself holds an MTP layer
  VisionCfg vision;     // optional Qwen3.5/3.8 vision encoder (from mmproj companion or .flyx)
};

// A tensor's name in a GGUF file, from the name python/convert gives it; the
// MTP layer's in companion `companion` when >= 0.
std::string gguf_name(const std::string& name, int layers, int companion) {
  if (const char* top = gguf_top_name(name)) return top;
  static const std::map<std::string, std::string> mtp = {
      {"mtp.fc", "nextn.eh_proj.weight"}, {"mtp.norm_emb", "nextn.enorm.weight"},
      {"mtp.norm_hidden", "nextn.hnorm.weight"}, {"mtp.norm", "nextn.shared_head_norm.weight"}};
  static const std::map<std::string, std::string> gdn_layer = {
      {"gdn.qkv", "attn_qkv.weight"}, {"gdn.z", "attn_gate.weight"}, {"gdn.a", "ssm_alpha.weight"},
      {"gdn.b", "ssm_beta.weight"}, {"gdn.out", "ssm_out.weight"}, {"gdn.conv", "ssm_conv1d.weight"},
      {"gdn.A_log", "ssm_a"}, {"gdn.dt_bias", "ssm_dt.bias"}, {"gdn.norm", "ssm_norm.weight"}};
  // The MTP layer is the block after the target's.
  const std::string blk = "blk." + std::to_string(layers) + ".";
  const std::string mtp_block = companion >= 0 ? ModelFile::companion_name(companion, blk) : blk;
  if (auto m = mtp.find(name); m != mtp.end()) return mtp_block + m->second;
  const bool in_mtp = name.starts_with("mtp.L0.");
  const size_t dot = in_mtp ? 6 : name.find('.', 8);  // "target.L<i>."
  const std::string block = in_mtp ? mtp_block : "blk." + name.substr(8, dot - 8) + ".";
  const std::string suffix = name.substr(dot + 1);
  if (const char* mapped = gguf_dense_layer_name(suffix, "post_attention_norm.weight")) return block + mapped;
  return block + gdn_layer.at(suffix);
}

void validate_config(const Config& f) {
  if (f.hidden <= 0 || f.layers <= 0 || f.vocab <= 0 || f.inter <= 0 || f.n_heads <= 0 || f.n_kv <= 0 ||
      f.head_dim <= 0 || f.rot_dim <= 0 || f.rot_dim > f.head_dim || f.rot_dim % 2 != 0 || f.n_heads % f.n_kv != 0 ||
      f.nk <= 0 || f.nv <= 0 || f.dk <= 0 || f.dv <= 0 || f.nv % f.nk != 0 || f.conv <= 0 || !(f.eps > 0) ||
      !(f.theta > 0) || f.hidden > kMaxWidth || f.inter > kMaxWidth || f.vocab > kMaxVocab || f.layers > kMaxLayers ||
      f.n_heads > kMaxHeads || f.head_dim > kMaxHeadDim || f.nk > kMaxHeads || f.nv > kMaxHeads ||
      f.dk > kMaxHeadDim || f.dv > kMaxHeadDim || f.conv > kMaxHeadDim)
    throw std::runtime_error("invalid Qwen3.5 model config dimensions");
}

Config read_gguf_config(const ModelFile& file) {
  const GgufDecoderBase b = read_gguf_decoder_base(file, /*allow_partial_rotary=*/true);
  const GgufMeta m(file);
  Config f{};
  f.gguf = true;
  f.vocab = b.vocab;
  f.hidden = b.hidden;
  const int blocks = b.blocks, mtp = m.count_or("nextn_predict_layers", 0, 0);
  if (mtp > 1 || mtp >= blocks) throw std::runtime_error("GGUF with more than one MTP layer");
  f.mtp = mtp == 1;
  f.layers = blocks - mtp;
  f.inter = b.inter;
  f.n_heads = b.n_heads;
  f.n_kv = b.n_kv;
  f.head_dim = b.head_dim;
  f.rot_dim = b.rot_dim;
  f.theta = b.theta;
  f.eps = b.eps;
  f.trained_ctx = b.trained_ctx;
  f.nk = m.count("ssm.group_count");
  f.nv = m.count("ssm.time_step_rank");
  f.dk = m.count("ssm.state_size");
  const int inner = m.count("ssm.inner_size");
  if (f.nv % f.nk || inner % f.nv) throw std::runtime_error("GGUF gated delta net head dimensions do not divide");
  f.dv = inner / f.nv;
  f.conv = m.count("ssm.conv_kernel");
  if (m.has("attention.recurrent_layers")) {  // one flag per block, the MTP layer's too
    const auto& flags = m.at("attention.recurrent_layers");
    if (!flags.is_array() || int(flags.size()) != blocks)
      throw std::runtime_error("GGUF recurrent layer flags do not match block_count");
    for (int i = 0; i < f.layers; ++i) f.is_attn.push_back(!flags[size_t(i)].get<bool>());
  } else {
    const int interval = m.count("full_attention_interval");
    for (int i = 0; i < f.layers; ++i) f.is_attn.push_back((i + 1) % interval == 0);
  }
  if (m.has("rope.dimension_sections")) {
    const auto& sec = m.at("rope.dimension_sections");
    if (sec.is_array() && sec.size() >= 3 && sec[0].is_number_integer() && sec[1].is_number_integer() &&
        sec[2].is_number_integer()) {
      const int s0 = sec[0].get<int>(), s1 = sec[1].get<int>(), s2 = sec[2].get<int>();
      if (s0 > 0 && s1 > 0 && s2 > 0 && s0 + s1 + s2 == f.rot_dim / 2) {
        f.mrope_section[0] = s0;
        f.mrope_section[1] = s1;
        f.mrope_section[2] = s2;
        f.mrope_interleaved = true;
      }
    }
  }
  validate_config(f);
  return f;
}

bool is_mmproj_header(const json& h) {
  if (!GgufMeta::is_header(h)) return false;
  if (h.value("arch", "") != "clip") return false;
  const json& g = h.at("gguf");
  if (!g.is_object()) return false;
  if (g.value("general.type", "") == "mmproj") return true;
  return g.contains("clip.has_vision_encoder") &&
         ((g["clip.has_vision_encoder"].is_boolean() && g["clip.has_vision_encoder"].get<bool>()) ||
          (g["clip.has_vision_encoder"].is_number_integer() && g["clip.has_vision_encoder"].get<int>() != 0));
}

void read_gguf_mmproj(const json& h, int companion_idx, const Config& f, VisionCfg* out, const std::string& what) {
  const json& g = h.at("gguf");
  const std::string proj = g.value("clip.projector_type", "");
  if (proj != "qwen3vl_merger" && proj != "qwen35")
    throw std::runtime_error(what + "unsupported mmproj projector_type '" + proj + "' (need qwen3vl_merger)");
  auto req_i = [&](const char* key) -> int {
    if (!g.contains(key) || !g.at(key).is_number_integer())
      throw std::runtime_error(what + "missing mmproj key " + key);
    const int64_t v = g.at(key).get<int64_t>();
    if (v <= 0 || v > INT_MAX) throw std::runtime_error(what + "invalid mmproj key " + key);
    return int(v);
  };
  VisionCfg v{};
  v.enabled = true;
  v.companion = companion_idx;
  v.depth = req_i("clip.vision.block_count");
  v.hidden = req_i("clip.vision.embedding_length");
  v.inter = req_i("clip.vision.feed_forward_length");
  v.n_heads = req_i("clip.vision.attention.head_count");
  v.patch_size = req_i("clip.vision.patch_size");
  v.merge_size = g.contains("clip.vision.spatial_merge_size") ? req_i("clip.vision.spatial_merge_size") : 2;
  v.out_hidden = req_i("clip.vision.projection_dim");
  if (v.out_hidden != f.hidden)
    throw std::runtime_error(what + "mmproj projection_dim (" + std::to_string(v.out_hidden) +
                             ") does not match model hidden size (" + std::to_string(f.hidden) + ")");
  if (v.hidden % v.n_heads != 0) throw std::runtime_error(what + "mmproj hidden not divisible by head_count");
  v.head_dim = v.hidden / v.n_heads;
  if (v.head_dim % 4 != 0 || v.hidden > kMaxWidth || v.inter > kMaxWidth || v.depth > kMaxLayers ||
      v.patch_size > 64 || v.merge_size > 16)
    throw std::runtime_error(what + "invalid mmproj vision dimensions");
  if (g.contains("clip.vision.attention.layer_norm_epsilon") && g["clip.vision.attention.layer_norm_epsilon"].is_number())
    v.eps = g["clip.vision.attention.layer_norm_epsilon"].get<double>();
  if (!(v.eps > 0.0) || !std::isfinite(v.eps))
    throw std::runtime_error(what + "invalid mmproj layer_norm_epsilon");
  if (g.contains("clip.vision.image_mean") && g["clip.vision.image_mean"].is_array() &&
      g["clip.vision.image_mean"].size() == 3) {
    for (int c = 0; c < 3; ++c) {
      if (!g["clip.vision.image_mean"][size_t(c)].is_number())
        throw std::runtime_error(what + "invalid mmproj image_mean");
      v.mean[c] = g["clip.vision.image_mean"][size_t(c)].get<float>();
      if (!std::isfinite(v.mean[c])) throw std::runtime_error(what + "non-finite mmproj image_mean");
    }
  }
  if (g.contains("clip.vision.image_std") && g["clip.vision.image_std"].is_array() &&
      g["clip.vision.image_std"].size() == 3) {
    for (int c = 0; c < 3; ++c) {
      if (!g["clip.vision.image_std"][size_t(c)].is_number())
        throw std::runtime_error(what + "invalid mmproj image_std");
      v.std[c] = g["clip.vision.image_std"][size_t(c)].get<float>();
      if (!(v.std[c] > 0.0f) || !std::isfinite(v.std[c]))
        throw std::runtime_error(what + "invalid mmproj image_std");
    }
  }
  if (g.contains("clip.vision.is_deepstack_layers") && g["clip.vision.is_deepstack_layers"].is_array()) {
    for (const auto& flag : g["clip.vision.is_deepstack_layers"]) {
      const bool on = (flag.is_boolean() && flag.get<bool>()) ||
                      (flag.is_number_integer() && flag.get<int>() != 0);
      if (on) throw std::runtime_error(what + "mmproj deepstack visual layers are not supported yet");
    }
  }
  *out = v;
}

// Non-drafter companions (after DrafterRegistry claims MTP and block drafter files):
//   a clip/mmproj GGUF holding the Qwen3.5/3.8 vision encoder (mmproj-BF16.gguf).
void take_vision_companions(const ModelFile& file, const std::vector<int>& unclaimed, Config* f, bool load_vision) {
  for (int i : unclaimed) {
    const json& h = file.header().at("companions").at(size_t(i));
    const std::string what = "companion file " + std::to_string(i + 1) + ": ";
    if (is_mmproj_header(h)) {
      if (!load_vision) continue;
      if (f->vision.enabled) throw std::runtime_error(what + "the model already has a vision encoder");
      read_gguf_mmproj(h, i, *f, &f->vision, what);
    } else {
      throw std::runtime_error(what + "holds no drafter or vision encoder this model takes");
    }
  }
}

Config read_config(const json& h) {
  const json& c = h.at("config");
  Config f{};
  f.mtp = h.contains("mtp") && !h["mtp"].is_null();
  f.hidden = c.at("hidden");
  f.layers = c.at("layers");
  f.vocab = c.at("vocab");
  f.inter = c.at("inter");
  f.n_heads = c.at("n_heads");
  f.n_kv = c.at("n_kv");
  f.head_dim = c.at("head_dim");
  f.rot_dim = c.at("rot_dim");
  f.nk = c.at("lin_nk");
  f.nv = c.at("lin_nv");
  f.dk = c.at("lin_dk");
  f.dv = c.at("lin_dv");
  f.conv = c.at("conv_k");
  f.theta = c.at("rope_theta");
  f.eps = c.at("eps");
  f.trained_ctx = c.value("max_pos", 0);
  const auto lt = c.at("layer_types").get<std::vector<std::string>>();
  if (int(lt.size()) != f.layers) throw std::runtime_error("layer_types does not match layers");
  for (const auto& t : lt) {
    if (t != "attn" && t != "gdn") throw std::runtime_error("unknown layer_types entry: " + t);
    f.is_attn.push_back(t == "attn");
  }
  if (c.contains("mrope_section") && c["mrope_section"].is_array() && c["mrope_section"].size() >= 3 &&
      c["mrope_section"][0].is_number_integer() && c["mrope_section"][1].is_number_integer() &&
      c["mrope_section"][2].is_number_integer()) {
    const int s0 = c["mrope_section"][0].get<int>();
    const int s1 = c["mrope_section"][1].get<int>();
    const int s2 = c["mrope_section"][2].get<int>();
    if (s0 > 0 && s1 > 0 && s2 > 0 && s0 + s1 + s2 == f.rot_dim / 2) {
      f.mrope_section[0] = s0;
      f.mrope_section[1] = s1;
      f.mrope_section[2] = s2;
      f.mrope_interleaved = true;
    }
  }
  validate_config(f);
  return f;
}

class Qwen35 final : public Model {
 public:
  explicit Qwen35(const ModelFile& file) : Model(file) {}
  bool build(const ModelFile& file, const ModelOptions& o, std::string* err);
  std::vector<flyx_node> prefill_program() const override;
  std::vector<flyx_node> round_program(int width, int d, bool given) const override;

 private:
  int attention(const std::string& p, const Config& c, bool draft);
  int gdn(const std::string& p, const Config& c);
  int mlp(const std::string& p, const Config& c);
  void build_vision(const ModelFile& file, const Config& c);
  void add_layers(std::vector<flyx_node>* p) const;
  void add_contexts(std::vector<flyx_node>* p) const;  // every drafter takes the pass's rows

  // The file's name for a tensor, and the offset of its (offset + w) norms.
  std::string n(const std::string& name) const { return gguf_ ? gguf_name(name, layers_, mtp_companion_) : name; }
  Tables::NameFn name_fn() const { return [this](const std::string& name) { return n(name); }; }
  double one() const { return gguf_ ? 0.0 : 1.0; }

  bool gguf_ = false;
  int mtp_companion_ = -1;
  int layers_ = 0;
  int embed_ = -1, head_ = -1;
  std::vector<int> mixer_, mlp_;
  drafters::DrafterRegistry drafters_reg_;
};

// "<p>attn.*" (p = "target.L3." or "mtp.L0."): q carries the output gate.
int Qwen35::attention(const std::string& p, const Config& c, bool draft) {
  AttnBlockSpec spec{c.hidden, c.n_heads, c.n_kv, c.head_dim, c.rot_dim,
                     gguf_ ? FLYX_GATE_PER_HEAD : FLYX_GATE_AFTER_Q, c.theta, one(), c.eps, draft};
  for (int i = 0; i < 3; ++i) spec.mrope_section[i] = c.mrope_section[i];
  spec.mrope_interleaved = c.mrope_interleaved;
  return tables_.attn_block(p, spec, name_fn());
}

int Qwen35::gdn(const std::string& p, const Config& c) {
  Tables& T = tables_;
  const int H = c.hidden, C = 2 * c.nk * c.dk + c.nv * c.dv;
  flyx_state st{};
  st.kind = FLYX_STATE_GDN;
  st.nk = c.nk;
  st.nv = c.nv;
  st.dk = c.dk;
  st.dv = c.dv;
  st.conv = c.conv;
  st.head_order = gguf_ ? FLYX_GDN_HEADS_TILED : FLYX_GDN_HEADS_GROUPED;
  flyx_block b{};
  b.kind = FLYX_BLOCK_GDN;
  auto& g = b.gdn;
  g.hidden = H;
  g.nk = c.nk;
  g.nv = c.nv;
  g.dk = c.dk;
  g.dv = c.dv;
  g.conv = c.conv;
  g.head_order = st.head_order;
  g.decay = gguf_ ? FLYX_GDN_DECAY_NEG_EXP : FLYX_GDN_DECAY_LOG;
  g.norm = T.norm(n(p + "in_norm"), H, one(), c.eps);
  g.qkv = T.weight(n(p + "gdn.qkv"), {C, H});
  g.z = T.weight(n(p + "gdn.z"), {c.nv * c.dv, H});
  g.b = T.weight(n(p + "gdn.b"), {c.nv, H});
  g.a = T.weight(n(p + "gdn.a"), {c.nv, H});
  g.out = T.weight(n(p + "gdn.out"), {H, c.nv * c.dv});
  g.conv_w = T.weight(n(p + "gdn.conv"), {C, c.conv});
  g.A_log = T.weight(n(p + "gdn.A_log"), {c.nv});
  g.dt_bias = T.weight(n(p + "gdn.dt_bias"), {c.nv});
  g.out_norm = T.norm(n(p + "gdn.norm"), c.dv, 0.0, c.eps);
  g.state = T.state(st);
  return T.block(b);
}

int Qwen35::mlp(const std::string& p, const Config& c) {
  return tables_.mlp_block(p, c.hidden, c.inter, one(), c.eps, name_fn());
}

void Qwen35::build_vision(const ModelFile& file, const Config& c) {
  const VisionCfg& v = c.vision;
  if (!v.enabled) return;
  Tables& T = tables_;
  auto vn = [&](const std::string& name) -> std::string {
    return v.companion >= 0 ? ModelFile::companion_name(v.companion, name) : name;
  };
  const int H = v.hidden, I = v.inter, P = v.patch_size, M = v.merge_size;
  const int slice_dim = 3 * P * P;
  const bool has_t1 = T.has(vn("v.patch_embd.weight.1"));
  const int temporal = has_t1 ? 2 : 1;
  const int in_dim = temporal * slice_dim;

  const flyx_tensor_view* pos_t = file.tensor(vn("v.position_embd.weight"));
  if (!pos_t || pos_t->ndim != 2 || pos_t->shape[0] <= 0 || pos_t->shape[0] > INT_MAX || pos_t->shape[1] != H)
    throw std::runtime_error("mmproj: missing or invalid v.position_embd.weight");
  const int num_pos = int(pos_t->shape[0]);
  const int pos_side = int(std::llround(std::sqrt(double(num_pos))));
  if (pos_side <= 0 || int64_t(pos_side) * pos_side != num_pos)
    throw std::runtime_error("mmproj: v.position_embd.weight row count is not a positive square");

  std::vector<flyx_node> vnodes;
  // 1. Stem: Conv3d patch embedding (split into t=0 and t=1 in mmproj GGUF) + bias + 2D bilinear pos_embed
  {
    flyx_block b{};
    b.kind = FLYX_BLOCK_VISION_STEM;
    auto& st = b.vision_stem;
    st.in_dim = in_dim;
    st.hidden = H;
    st.patch_w = T.weight(vn("v.patch_embd.weight"), {H, 3, P, P});
    st.patch_w2 = has_t1 ? T.weight(vn("v.patch_embd.weight.1"), {H, 3, P, P}) : -1;
    st.patch_b = T.has(vn("v.patch_embd.bias")) ? T.weight(vn("v.patch_embd.bias"), {H}) : -1;
    st.pos_mode = FLYX_VPOS_BILINEAR_2D;
    st.pos_w = T.weight(vn("v.position_embd.weight"), {num_pos, H});
    st.pos_grid_h = pos_side;
    st.pos_grid_w = pos_side;
    st.merge_size = M;
    st.cls_token = -1;
    st.pre_norm = {FLYX_NORM_LAYER, -1, -1, 0, 0.0, v.eps};
    vnodes.push_back({FLYX_NODE_RUN, T.block(b), 0, 0});
  }

  // 2. ViT blocks (depth = 27 on Qwen3.8-27B, 24 on Qwen3.5-4B)
  for (int l = 0; l < v.depth; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    {
      flyx_block b{};
      b.kind = FLYX_BLOCK_VIT_ATTN;
      auto& a = b.vit_attn;
      a.hidden = H;
      a.n_heads = v.n_heads;
      a.head_dim = v.head_dim;
      a.rope_mode = FLYX_VROPE_AXIAL_2D;
      a.merge_size = M;
      a.theta = v.rope_theta;
      a.norm = T.vnorm(FLYX_NORM_LAYER, vn(p + "ln1.weight"), vn(p + "ln1.bias"), H, 0.0, v.eps);
      a.q_norm = {FLYX_NORM_LAYER, -1, -1, 0, 0.0, v.eps};
      a.k_norm = {FLYX_NORM_LAYER, -1, -1, 0, 0.0, v.eps};
      a.qkv = T.weight(vn(p + "attn_qkv.weight"), {3 * H, H});
      a.qkv_b = T.has(vn(p + "attn_qkv.bias")) ? T.weight(vn(p + "attn_qkv.bias"), {3 * H}) : -1;
      a.q = a.k = a.v = a.q_b = a.k_b = a.v_b = -1;
      a.o = T.weight(vn(p + "attn_out.weight"), {H, H});
      a.o_b = T.has(vn(p + "attn_out.bias")) ? T.weight(vn(p + "attn_out.bias"), {H}) : -1;
      a.layer_scale = -1;
      vnodes.push_back({FLYX_NODE_RUN, T.block(b), 0, 0});
    }
    {
      flyx_block b{};
      b.kind = FLYX_BLOCK_VIT_MLP;
      auto& f = b.vit_mlp;
      f.hidden = H;
      f.inter = I;
      f.gated = 0;
      f.act = FLYX_VACT_GELU_TANH;
      f.norm = T.vnorm(FLYX_NORM_LAYER, vn(p + "ln2.weight"), vn(p + "ln2.bias"), H, 0.0, v.eps);
      f.fc1 = T.weight(vn(p + "ffn_up.weight"), {I, H});
      f.fc1_b = T.has(vn(p + "ffn_up.bias")) ? T.weight(vn(p + "ffn_up.bias"), {I}) : -1;
      f.gate = f.gate_b = -1;
      f.fc2 = T.weight(vn(p + "ffn_down.weight"), {H, I});
      f.fc2_b = T.has(vn(p + "ffn_down.bias")) ? T.weight(vn(p + "ffn_down.bias"), {H}) : -1;
      f.layer_scale = -1;
      vnodes.push_back({FLYX_NODE_RUN, T.block(b), 0, 0});
    }
  }

  // 3. Spatial Patch Merger: post_ln(1152) -> 2x2 concat(4608) -> mm.0(4608) -> GELU -> mm.2(5120)
  {
    const int merged = H * M * M;
    flyx_block b{};
    b.kind = FLYX_BLOCK_VISION_MERGE;
    auto& mg = b.vision_merge;
    mg.in_hidden = H;
    mg.merge_size = M;
    mg.merge_mode = FLYX_VMERGE_SPATIAL_CONCAT;
    mg.post_norm = 0;  // v.post_ln has width in_hidden (applied before spatial concat)
    mg.fc1_out = merged;
    mg.out_hidden = v.out_hidden;
    mg.act = FLYX_VACT_GELU;
    mg.norm = T.vnorm(FLYX_NORM_LAYER, vn("v.post_ln.weight"), vn("v.post_ln.bias"), H, 0.0, v.eps);
    mg.fc1 = T.weight(vn("mm.0.weight"), {merged, merged});
    mg.fc1_b = T.has(vn("mm.0.bias")) ? T.weight(vn("mm.0.bias"), {merged}) : -1;
    mg.fc2 = T.weight(vn("mm.2.weight"), {v.out_hidden, merged});
    mg.fc2_b = T.has(vn("mm.2.bias")) ? T.weight(vn("mm.2.bias"), {v.out_hidden}) : -1;
    vnodes.push_back({FLYX_NODE_RUN, T.block(b), 0, 0});
  }

  int32_t img_tok = v.image_token_id;
  if (img_tok < 0) img_tok = find_added_token(file, "<|image_pad|>");
  if (img_tok < 0 || img_tok >= c.vocab)
    throw std::runtime_error("vision encoder requires <|image_pad|> token in the model vocabulary");

  vision::Qwen35VisionParams vp{};
  vp.patch_size = P;
  vp.temporal_patch_size = temporal;
  vp.merge_size = M;
  vp.out_hidden = v.out_hidden;
  vp.placeholder_token = img_tok;
  vp.min_pixels = v.min_pixels;
  vp.max_pixels = v.max_pixels;
  for (int i = 0; i < 3; ++i) vp.image_mean[i] = v.mean[i], vp.image_std[i] = v.std[i];
  set_vision(std::make_unique<vision::Qwen35VisionAdapter>(vp), std::move(vnodes));
}

bool Qwen35::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  const json& h = file.header();
  Config c = GgufMeta::is(file) ? read_gguf_config(file) : read_config(h);
  trained_ctx_ = c.trained_ctx;
  gguf_ = c.gguf;
  layers_ = c.layers;

  drafters::TargetSpec target{};
  target.arch = file.arch();
  target.hidden = c.hidden;
  target.vocab = c.vocab;
  target.layers = c.layers;
  target.gguf = c.gguf;

  drafters::MtpHooks mtp_hooks{};
  mtp_hooks.embedded = c.mtp;
  mtp_hooks.norm_offset = one();
  mtp_hooks.eps = c.eps;
  mtp_hooks.tensor_name = [this](const std::string& canonical, int comp) {
    return gguf_ ? gguf_name(canonical, layers_, comp) : canonical;
  };
  // The layer's tensors are named through n(), which reads mtp_companion_ (set below, before any block is built).
  mtp_hooks.build_layer = [this, &c](int) -> std::pair<int, int> {
    return {attention("mtp.L0.", c, true), mlp("mtp.L0.", c)};
  };

  std::vector<int> unclaimed;
  drafters_reg_.scan(file, target, &mtp_hooks, /*has_dv=*/false, &unclaimed);
  take_vision_companions(file, unclaimed, &c, o.vision);
  if (c.vision.enabled && !c.mrope_interleaved) {
    // Qwen3.5 / Qwen3.8 checkpoints converted to .flyx before mrope_section was serialized use rot_dim=64 [11, 11, 10].
    if (!c.gguf && c.rot_dim == 64) {
      c.mrope_section[0] = 11;
      c.mrope_section[1] = 11;
      c.mrope_section[2] = 10;
      c.mrope_interleaved = true;
    } else {
      return *err = "vision encoder requires 3D MRoPE sections (rope.dimension_sections / mrope_section)", false;
    }
  }
  mtp_companion_ = drafters_reg_.mtp_companion();
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, drafters_reg_.sources(), &drafters, err)) return false;
  drafters_reg_.validate_selected(file, drafters, target);
  if (!tables_.set_activations(o.activations, declared_activations(file), o.fp4_inputs, err)) return false;
  Tables& T = tables_;
  const auto nf = name_fn();
  embed_ = T.embed_block(c.hidden, c.vocab, nf);
  head_ = T.head_block(c.hidden, c.vocab, embed_, one(), c.eps, nf);
  target.embed = embed_;
  target.head = head_;
  for (int i = 0; i < c.layers; ++i) {
    const std::string p = "target.L" + std::to_string(i) + ".";
    mixer_.push_back(c.is_attn[size_t(i)] ? attention(p, c, false) : gdn(p, c));
    mlp_.push_back(mlp(p, c));
  }
  for (Drafter kind : drafters) {
    int max_drafts = 0;
    const int blk = drafters_reg_.attach(kind, file, T, target, &mtp_hooks, &max_drafts);
    add_drafter(kind, blk, max_drafts);
  }
  build_vision(file, c);
  finish(c.hidden, c.vocab);
  return true;
}

void Qwen35::add_layers(std::vector<flyx_node>* p) const {
  for (size_t i = 0; i < mixer_.size(); ++i) {
    p->push_back({FLYX_NODE_RUN, mixer_[i], 0, 0});
    p->push_back({FLYX_NODE_RUN, mlp_[i], 0, 0});
    drafters_reg_.emit_taps(p, int(i) + 1);
  }
}

void Qwen35::add_contexts(std::vector<flyx_node>* p) const { drafters_reg_.emit_contexts(p); }

std::vector<flyx_node> Qwen35::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  add_contexts(&p);
  p.push_back({FLYX_NODE_RUN, head_, 1, 0});
  return p;
}

std::vector<flyx_node> Qwen35::round_program(int width, int d, bool given) const {
  std::vector<flyx_node> p;
  if (width > 0) p.push_back(given ? flyx_node{FLYX_NODE_GIVEN, -1, width, 0}
                                   : flyx_node{FLYX_NODE_DRAFT, drafters_.at(size_t(d)).block, width, 0});
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  p.push_back({FLYX_NODE_RUN, head_, 0, 0});
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  add_contexts(&p);
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<Qwen35>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::qwen35
