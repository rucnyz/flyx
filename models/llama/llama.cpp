// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Llama family from its model file: every layer attention + MLP; the layer
// math as block flags: no output gate, plain-weight RMSNorms (offset 0),
// rotary on the whole head.  Qwen3 (arch "qwen3") normalises each head's q and
// k before the rotary; Llama (arch "llama": MiniCPM5) does not, and its
// attention blocks name no q/k norm.  A .flyx file has the
// config in its header and the names python/convert writes; a GGUF file
// llama.cpp's metadata keys and tensor names, the same layout otherwise (Qwen3
// only: llama.cpp's converter reorders a Llama's q and k rows).  A Llama
// .flyx file may hold a DV drafter (models/dv.h; python/convert --dv), whose
// canvas runs this model's own attention and MLP blocks, and a .flyx file of
// either may hold the target's own adapters (a V-LoRA, python/convert --lora:
// the header's "lora" section), on every linear of every layer, in every pass.
// A Qwen3 model takes a DSpine drafter (models/dspine.h): in its .flyx file
// (python/convert --draft) or in a drafter file beside a model of either format.
#include "models/llama/llama.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "models/dspine.h"
#include "models/dv.h"
#include "models/gguf.h"

namespace flyx::models::llama {
namespace {

using nlohmann::json;

struct Config {
  int hidden, layers, vocab, inter, n_heads, n_kv, head_dim;
  double theta, eps;
  int trained_ctx = 0;  // the context the model was made for (0: the file does not say)
  bool gguf = false;
};

// A tensor's name in a GGUF file, from the name python/convert gives it.
std::string gguf_name(const std::string& name) {
  if (const char* top = gguf_top_name(name)) return top;
  const size_t dot = name.find('.', 8);  // "target.L<i>."
  const std::string_view suffix = std::string_view(name).substr(dot + 1);
  const char* mapped = gguf_dense_layer_name(suffix, "ffn_norm.weight");
  if (!mapped) throw std::out_of_range("unknown Qwen3 GGUF tensor: " + name);
  return "blk." + name.substr(8, dot - 8) + "." + mapped;
}

void validate_config(const Config& f) {
  if (f.hidden <= 0 || f.layers <= 0 || f.vocab <= 0 || f.inter <= 0 || f.n_heads <= 0 || f.n_kv <= 0 ||
      f.head_dim <= 0 || f.n_heads % f.n_kv != 0 || f.head_dim % 2 != 0 || !(f.eps > 0) || !(f.theta > 0) ||
      f.hidden > kMaxWidth || f.inter > kMaxWidth || f.vocab > kMaxVocab || f.layers > kMaxLayers ||
      f.n_heads > kMaxHeads || f.head_dim > kMaxHeadDim)
    throw std::runtime_error("invalid Qwen3 or Llama model config dimensions");
}

Config read_gguf_config(const ModelFile& file) {
  const GgufDecoderBase b = read_gguf_decoder_base(file, /*allow_partial_rotary=*/false);
  Config f{};
  f.gguf = true;
  f.vocab = b.vocab;
  f.hidden = b.hidden;
  f.layers = b.blocks;
  f.inter = b.inter;
  f.n_heads = b.n_heads;
  f.n_kv = b.n_kv;
  f.head_dim = b.head_dim;
  f.theta = b.theta;
  f.eps = b.eps;
  f.trained_ctx = b.trained_ctx;
  validate_config(f);
  return f;
}

// The target's own adapters (ABI 43): the header's "lora" section, y = W x + scale b (a x) on every linear of every
// layer with one rank; rank 0: none.  The file names them among the parts a reader must not ignore ("requires",
// .flyx format 2), so that a build that would read the target without them refuses the file instead.
struct Lora {
  int rank = 0;
  float scale = 1.f;
};

Lora read_lora(const ModelFile& file) {
  const json& h = file.header();
  if (!h.is_object() || !h.contains("lora") || h.at("lora").is_null()) return {};
  const json& l = h.at("lora");
  bool required = false;
  if (h.contains("requires"))
    for (const auto& r : h.at("requires")) required = required || r == "lora";
  if (!required)
    throw std::runtime_error("lora: adapters the file does not require (\"requires\": [\"lora\"], .flyx format 2), "
                             "which a build without them would read as the plain model: convert the file again");
  Lora a;
  a.rank = l.at("rank").get<int>();
  a.scale = l.at("scale").get<float>();
  if (a.rank < 1 || !std::isfinite(a.scale) || a.scale == 0.f)
    throw std::runtime_error("lora: rank " + std::to_string(a.rank) + ", scale " + std::to_string(a.scale) +
                             " (a rank of 1 or more, a finite scale other than 0)");
  if (a.rank > kMaxHeadDim)  // bounded as a model's dimensions are (models/model.h)
    throw std::runtime_error("lora: rank " + std::to_string(a.rank) + ", past any model's (at most " +
                             std::to_string(kMaxHeadDim) + ")");
  return a;
}

Config read_config(const json& c) {
  Config f{};
  f.hidden = c.at("hidden");
  f.layers = c.at("layers");
  f.vocab = c.at("vocab");
  f.inter = c.at("inter");
  f.n_heads = c.at("n_heads");
  f.n_kv = c.at("n_kv");
  f.head_dim = c.at("head_dim");
  f.theta = c.at("rope_theta");
  f.eps = c.at("eps");
  f.trained_ctx = c.value("max_pos", 0);
  validate_config(f);
  return f;
}

class Llama final : public Model {
 public:
  explicit Llama(const ModelFile& file) : Model(file) {}
  bool build(const ModelFile& file, const ModelOptions& o, std::string* err);
  std::vector<flyx_node> prefill_program() const override;
  std::vector<flyx_node> round_program(int width, int d, bool given) const override;

 private:
  void add_layers(std::vector<flyx_node>* p) const;

  int embed_ = -1, head_ = -1;
  std::vector<int> layers_;  // per layer: attention block, MLP block
  int dv_block_ = -1;        // DV, when it drafts
  int dspine_block_ = -1;    // DSpine, when it drafts
  std::vector<int> taps_;    // its features: the residual stream after taps_[s] layers
};

// The drafter config a Qwen3 model takes from its file ("draft") or a drafter file beside it (ModelFile::add), and
// the companion that holds it (-1: the model file); null when there is none.  Throws on one it does not take.
const json* dspine_config(const ModelFile& file, bool qwen3, int* companion) {
  const json& h = file.header();
  *companion = -1;
  const json* found = nullptr;
  if (h.is_object() && h.contains("draft") && !h["draft"].is_null()) found = &h["draft"];
  for (int i = 0; i < file.companions(); ++i) {
    const json& c = h.at("companions").at(size_t(i));
    const std::string what = "drafter file " + std::to_string(i + 1) + ": ";
    if (c.value("kind", "") != "drafter" || !c.contains("draft") || !c["draft"].is_object())
      throw std::runtime_error(what + "holds no drafter this model takes (a DSpine drafter file)");
    if (found) throw std::runtime_error(what + "the model has a block drafter already");
    found = &c["draft"];
    *companion = i;
  }
  if (found && (!qwen3 || found->value("kind", "") != "dspine"))
    throw std::runtime_error(std::string("a ") + (qwen3 ? "Qwen3" : "Llama") + " model takes no " +
                             found->value("kind", std::string("unknown")) + " drafter" +
                             (qwen3 ? " (a DSpine drafter: python/convert --draft)" : ""));
  return found;
}

bool Llama::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  const bool qwen3 = file.arch() == "qwen3", qk_norm = qwen3;  // Qwen3 normalises each head's q and k
  // DV: in a Llama .flyx file of its own (python/convert --dv); DSpine: beside a Qwen3 (dspine_config).
  const bool has_dv = dv::in_file(file);
  if (has_dv && (qwen3 || GgufMeta::is(file))) return *err = "a DV drafter in a file that is not a Llama .flyx", false;
  file_drafts_trees_ = has_dv;
  // A vision projector (--mmproj) beside it: not for this family (models/qwen35 takes one).
  for (int i = 0; i < file.companions(); ++i)
    if (file.header().at("companions").at(size_t(i)).value("arch", "") == "clip")
      return *err = std::string("a ") + (qwen3 ? "Qwen3" : "Llama") +
                    " model does not support a vision projector (--mmproj)",
             false;
  int companion = -1;
  const json* draft = nullptr;
  try {
    draft = dspine_config(file, qwen3, &companion);
  } catch (const std::exception& e) {
    return *err = e.what(), false;
  }
  DrafterSources sources;
  sources.dv = has_dv;
  sources.dspine = draft != nullptr;
  sources.dspine_beside = companion >= 0;
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, sources, &drafters, err)) return false;
  if (!tables_.set_activations(o.activations, declared_activations(file), o.fp4_inputs, err)) return false;
  // MiniCPM5-1B's next tokens move visibly with an FP8 cache (KL 0.0064, top-1 97.2%): binary16 unless
  // the caller asks for FP8.
  kv_fp8_by_default_ = qwen3;
  // llama.cpp's converter permutes a Llama's q and k rows for its interleaved rotary pairs, a layout no backend
  // reads; python/convert keeps the checkpoint's.
  if (GgufMeta::is(file) && !qwen3)
    return *err = "a Llama GGUF file (q and k rows in llama.cpp's interleaved order) is not read; convert the "
                  "checkpoint with python/convert",
           false;
  const Config c = GgufMeta::is(file) ? read_gguf_config(file) : read_config(file.header().at("config"));
  const Lora lora = GgufMeta::is(file) ? Lora{} : read_lora(file);
  trained_ctx_ = c.trained_ctx;
  const int H = c.hidden, layers = c.layers, V = c.vocab, I = c.inter;
  const int nh = c.n_heads, nkv = c.n_kv, hd = c.head_dim;
  const double eps = c.eps;
  auto n = [&](const std::string& name) { return c.gguf ? gguf_name(name) : name; };  // the file's name for a tensor
  Tables& T = tables_;
  // A linear's adapter: its a [rank, in] and b [out, rank] beside the linear's weight (python/convert --lora).
  auto adapter = [&](const std::string& linear, int64_t out, int64_t in) {
    const flyx_lora l{T.weight(linear + ".lora_a", {lora.rank, in}), T.weight(linear + ".lora_b", {out, lora.rank}),
                      lora.rank, lora.scale};
    for (int w : {l.a, l.b})
      if (T.weights[size_t(w)].t->dtype != FLYX_DT_BF16)
        throw std::runtime_error(std::string(T.weights[size_t(w)].t->name) + ": an adapter that is not bf16");
    return l;
  };
  embed_ = T.embed_block(H, V, n);
  head_ = T.head_block(H, V, embed_, 0.0, c.eps, n);
  AttnBlockSpec attn_spec{H, c.n_heads, c.n_kv, c.head_dim, c.head_dim, FLYX_GATE_NONE, c.theta, 0.0, c.eps};
  attn_spec.qk_norm = qk_norm;
  for (int i = 0; i < layers; ++i) {
    const std::string p = "target.L" + std::to_string(i) + ".";
    // A Llama names no q/k norms: a file with them is a Qwen3 checkpoint converted as a Llama.
    if (!qk_norm && (T.has(n(p + "attn.q_norm")) || T.has(n(p + "attn.k_norm"))))
      throw std::runtime_error(p + "attn: q/k norms in a Llama file (a Qwen3 checkpoint converted as Llama?)");
    layers_.push_back(T.attn_block(p, attn_spec, n));
    if (lora.rank) {
      flyx_attn_block& a = T.blocks[size_t(layers_.back())].attn;
      a.lora[0] = adapter(p + "attn.q", nh * hd, H);
      a.lora[1] = adapter(p + "attn.k", nkv * hd, H);
      a.lora[2] = adapter(p + "attn.v", nkv * hd, H);
      a.lora[3] = adapter(p + "attn.o", H, nh * hd);
    } else if (!c.gguf) {  // any of the seven linears' adapters: refused, never read as none
      for (const char* l : {"attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"})
        if (T.has(p + l + ".lora_a") || T.has(p + l + ".lora_b"))
          throw std::runtime_error(p + l + ": adapters without the header's lora section");
    }
    layers_.push_back(T.mlp_block(p, H, I, 0.0, c.eps, n));
    if (lora.rank) {
      flyx_mlp_block& f = T.blocks[size_t(layers_.back())].mlp;
      f.lora[0] = adapter(p + "mlp.gate", I, H);
      f.lora[1] = adapter(p + "mlp.up", I, H);
      f.lora[2] = adapter(p + "mlp.down", H, I);
    }
  }
  for (Drafter kind : drafters) {
    if (kind != Drafter::kDSpine) continue;
    const dspine::Target t{H, V, layers, embed_, head_};
    taps_ = dspine::taps(*draft, t);
    dspine_block_ = dspine::describe(T, *draft, t, [&](const std::string& name) {
      return companion >= 0 ? ModelFile::companion_name(companion, name) : name;
    });
    add_drafter(kind, dspine_block_, draft->at("block").get<int>() - 1);
  }
  if (std::find(drafters.begin(), drafters.end(), Drafter::kDv) != drafters.end()) {  // DV
    dv::Target t;
    t.hidden = H;
    t.vocab = V;
    t.n_heads = nh;
    t.n_kv = nkv;
    t.head_dim = hd;
    t.inter = I;
    t.eps = eps;
    t.embed = embed_;
    t.head = head_;
    for (size_t i = 0; i < layers_.size(); i += 2) {
      t.attn.push_back(layers_[i]);
      t.mlp.push_back(layers_[i + 1]);
    }
    dv_block_ = dv::describe(T, file, t);
    add_drafter(Drafter::kDv, dv_block_, T.at(dv_block_).dv.tree_width);
  }
  finish(H, V);
  return true;
}

// Every layer; after the layers DSpine taps, its features (whichever drafter drafts: its context stays current).
void Llama::add_layers(std::vector<flyx_node>* p) const {
  for (size_t i = 0; i < layers_.size(); ++i) {
    p->push_back({FLYX_NODE_RUN, layers_[i], 0, 0});
    if (dspine_block_ >= 0 && i % 2 == 1)
      for (size_t s = 0; s < taps_.size(); ++s)
        if (taps_[s] == int(i / 2) + 1) p->push_back({FLYX_NODE_TAP, dspine_block_, int32_t(s), 0});
  }
}

std::vector<flyx_node> Llama::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  if (dspine_block_ >= 0) p.push_back({FLYX_NODE_CONTEXT, dspine_block_, 0, 0});
  p.push_back({FLYX_NODE_RUN, head_, 1, 0});
  if (dv_block_ >= 0) p.push_back({FLYX_NODE_CONTEXT, dv_block_, 0, 0});  // DV's canvas starts over
  return p;
}

// Plain decoding, a round of given drafts, DSpine's round, or DV's round or warm-up (width 0; models/dv.h).  Only DV's
// rounds run its canvas: a round of another kind leaves it behind, and DV's next round starts it over.  DSpine takes
// every round's accepted rows as context, whichever drafted them.
std::vector<flyx_node> Llama::round_program(int width, int d, bool given) const {
  std::vector<flyx_node> verify;
  verify.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&verify);
  verify.push_back({FLYX_NODE_RUN, head_, 0, 0});
  if (!given && d >= 0 && drafters_.at(size_t(d)).kind == Drafter::kDv)
    return dv::round_program(drafters_.at(size_t(d)).block, width, verify);
  std::vector<flyx_node> p;
  if (given && width > 0) p.push_back({FLYX_NODE_GIVEN, -1, width, 0});
  else if (width > 0 && d >= 0) p.push_back({FLYX_NODE_DRAFT, drafters_.at(size_t(d)).block, width, 0});
  p.insert(p.end(), verify.begin(), verify.end());
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  if (dspine_block_ >= 0) p.push_back({FLYX_NODE_CONTEXT, dspine_block_, 0, 0});
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<Llama>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::llama
