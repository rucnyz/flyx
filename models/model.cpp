// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/model.h"

#include <algorithm>
#include <stdexcept>

#include "core/interfaces/options.h"
#include "models/llama/llama.h"
#include "models/qwen35/qwen35.h"

namespace flyx::models {

const char* drafter_name(Drafter d) {
  switch (d) {
    case Drafter::kMtp: return "mtp";
    case Drafter::kDFlash2: return "dflash2";
    case Drafter::kDSpine: return "dspine";
    case Drafter::kDv: return "dv";
    default: return "none";
  }
}

namespace {
// Structural properties of a drafter kind; policy functions (draft_widths, adaptive_checkpoints, block_drafts)
// decide from these traits plus target properties (verify_scales_with_width) rather than per-drafter branches.
struct DrafterTraits {
  bool sequential;     // drafts one token per step; stopping early saves both draft and verify work
  bool tree_only;      // drafts trees over a persistent canvas (DV)
  bool stretch_block;  // shared mask token lets a parallel block draft past max_drafts up to FLYX_MAX_ROWS - 1
};

DrafterTraits drafter_traits(Drafter d) {
  switch (d) {
    case Drafter::kMtp: return {/*sequential=*/true, false, false};
    case Drafter::kDFlash2: return {false, false, /*stretch_block=*/true};
    case Drafter::kDSpine: return {false, false, false};  // its block at most: lowering refuses a wider round
    case Drafter::kDv: return {false, /*tree_only=*/true, false};
  }
  return {false, false, false};
}
}  // namespace

// Width ladder and intra-round adaptive stopping, driven by drafter traits + target structure:
// - Sequential drafters (MTP, future EAGLE-4): tensor-core aligned 1, 3, 5, 7, ..., max_drafts (<= 15).
// - Parallel block drafters: {3, 5, 7} when target verify cost scales steeply with width (e.g. sparse MoE),
//   {3, 7} on dense targets, plus FLYX_MAX_ROWS - 1 when stretch_block allows drafting past max_drafts.
// - Tree-only drafters (DV): up to FLYX_MAX_WIDE_ROWS - 1 (plus 15 when wider).
std::vector<int> Model::draft_widths(int d) const {
  const DrafterBlock& b = drafters_.at(size_t(d));
  if (b.max_drafts <= 0) return {};
  const DrafterTraits tr = drafter_traits(b.kind);
  if (tr.tree_only) {
    const int widest = std::min(FLYX_MAX_WIDE_ROWS - 1, b.max_drafts);
    if (widest > FLYX_MAX_ROWS - 1) return {FLYX_MAX_ROWS - 1, widest};
    return {widest};
  }
  const int widest = tr.stretch_block ? b.max_drafts : std::min(FLYX_MAX_ROWS - 1, b.max_drafts);
  const std::vector<int> ladder = tr.sequential              ? std::vector<int>{1, 3, 5, 7}
                                  : verify_scales_with_width_ ? std::vector<int>{3, 5, 7}
                                                              : std::vector<int>{3, 7};
  std::vector<int> widths;
  for (int k : ladder)
    if (k < widest) widths.push_back(k);
  widths.push_back(widest);
  if (tr.stretch_block && widest < FLYX_MAX_ROWS - 1) widths.push_back(FLYX_MAX_ROWS - 1);
  return widths;
}

std::vector<int> Model::adaptive_checkpoints(int d) const {
  const DrafterBlock& b = drafters_.at(size_t(d));
  const DrafterTraits tr = drafter_traits(b.kind);
  if (b.max_drafts <= 0 || tr.tree_only) return {};  // tree-only drafters (DV): no intra-round chain stop
  const std::vector<int> ws = draft_widths(d);
  // Intra-round GPU confidence stopping (stop_decide_step / stop_decide_block) pays whenever cutting verify width
  // saves substantial work:
  // - sequential drafters save both remaining draft steps and verify rows;
  // - targets with steep verify-cost scaling (e.g. sparse MoE active-expert fan-out) benefit on any block drafter.
  // Dense targets paired with block drafters have nearly flat verify cost across widths (RTX PRO 6000, QUASAR NVFP4
  // 27B + DFlash2: adaptive 425-439 vs fixed 443-494 tok/s on code) and avoid extra verify subgraphs in VRAM.
  if (tr.sequential || verify_scales_with_width_) return ws.size() > 1 ? ws : std::vector<int>{};
  return {};
}

int Model::block_drafts(int d) const {
  const DrafterBlock& b = drafters_.at(size_t(d));
  return drafter_traits(b.kind).sequential ? std::min(7, b.max_drafts) : b.max_drafts;
}

bool Model::drafts_trees(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).tree_only; }

bool Model::restarts(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).tree_only; }

void Model::add_drafter(Drafter kind, int block, int max_drafts) {
  drafters_.push_back({kind, block, max_drafts});
  drafter_names_.clear();
  for (const DrafterBlock& d : drafters_)
    drafter_names_ += (drafter_names_.empty() ? "" : "+") + std::string(models::drafter_name(d.kind));
}

bool choose_drafters(const ModelOptions& o, const DrafterSources& has, std::vector<Drafter>* out, std::string* err) {
  out->clear();
  interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
  if (!interfaces::parse_drafter_mode(o.drafter, &mode))
    return *err = "unknown drafter '" + o.drafter + "' (" + interfaces::kDrafterModes + ")", false;
  if (!o.speculative || mode == interfaces::DrafterMode::kNone) return true;
  bool mtp = false, dflash2 = false, dspine = false, dv = false;
  switch (mode) {
    case interfaces::DrafterMode::kMtp:
      if (!has.mtp) return *err = "the model file has no MTP layer", false;
      mtp = true;
      break;
    case interfaces::DrafterMode::kDFlash2:
      if (!has.dflash2) return *err = "the model file has no DFlash2 drafter", false;
      dflash2 = true;
      break;
    case interfaces::DrafterMode::kDSpine:
      if (!has.dspine) return *err = "the model has no DSpine drafter", false;
      dspine = true;
      break;
    case interfaces::DrafterMode::kDv:
      if (!has.dv) return *err = "the model file has no DV drafter", false;
      dv = true;
      break;
    case interfaces::DrafterMode::kAll:
      mtp = has.mtp;
      dflash2 = has.dflash2;
      dspine = has.dspine;
      dv = has.dv;
      break;
    case interfaces::DrafterMode::kAuto: {
      // Every drafter given in a file of its own; of the model file's own, the MTP layer if it has one, else DFlash2
      // (MTP was measured faster on the 4B, 462 against 341 tok/s: a file with both keeps it until choosing between
      // them every round is measured there).  DV only when named (dv, all).
      const bool own_mtp = has.mtp && !has.mtp_beside, own_dflash2 = has.dflash2 && !has.dflash2_beside;
      const bool own_dspine = has.dspine && !has.dspine_beside;
      mtp = has.mtp_beside || own_mtp;
      dflash2 = has.dflash2_beside || (own_dflash2 && !own_mtp);
      dspine = has.dspine_beside || (own_dspine && !own_mtp);
      break;
    }
    case interfaces::DrafterMode::kNone: break;
  }
  // DV's canvas follows its own rounds only, and the controller weighs its warm-up against plain decoding's and
  // lookup's rounds alone: no drafter beside it.
  if (dv && (mtp || dflash2 || dspine)) return *err = "DV drafts alone: not beside another drafter", false;
  if (mtp) out->push_back(Drafter::kMtp);
  if (dflash2) out->push_back(Drafter::kDFlash2);
  if (dspine) out->push_back(Drafter::kDSpine);
  if (dv) out->push_back(Drafter::kDv);
  return true;
}

std::string declared_activations(const ModelFile& file) {
  const auto& h = file.header();
  if (!h.is_object() || !h.contains("quant") || !h.at("quant").is_object()) return "";
  const auto& q = h.at("quant");
  return q.contains("activations") && q.at("activations").is_string() ? q.at("activations").get<std::string>() : "";
}

int Tables::weight(const std::string& name, std::vector<int64_t> shape) {
  auto it = index_.find(name);
  if (it != index_.end()) return it->second;
  const flyx_tensor_view* t = file_->tensor(name);
  if (!t) throw std::runtime_error("missing tensor " + name);
  bool same = t->ndim == int32_t(shape.size());
  for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == shape[i];
  if (!same) {
    std::string have, want;
    for (int i = 0; i < t->ndim; ++i) have += (i ? "x" : "") + std::to_string(t->shape[i]);
    for (size_t i = 0; i < shape.size(); ++i) want += (i ? "x" : "") + std::to_string(shape[i]);
    throw std::runtime_error(name + ": shape " + have + ", the config says " + want);
  }
  flyx_weight w{t, nullptr, nullptr, FLYX_ACT_FORMAT, 0};
  // In passes of more rows GGML block weights take 8-bit inputs and NVFP4 weights (ModelOpt's or GGML's) NVFP4 ones,
  // unless the file says otherwise: what their quantizations were made and measured with.
  // GGML's NVFP4 as llama.cpp reads the same file: NVFP4 inputs in prefill (rounded once from float32, as its FP4
  // ones), bf16 in rounds (finer than its 8 bits).
  const bool fp4 = t->dtype == FLYX_DT_NVFP4 || t->dtype == FLYX_DT_GGML_NVFP4;
  if (flyx_block_values(t->dtype) && !fp4 && !exact_) w.activations = FLYX_ACT_Q8;
  if (fp4 && !exact_ && nvfp4_inputs_) w.activations = FLYX_ACT_NVFP4;
  if (t->dtype == FLYX_DT_NVFP4) {
    w.scale = file_->tensor(name + ".scale");
    w.scale2 = file_->tensor(name + ".scale2");
    const bool ok_scale = w.scale && w.scale->dtype == FLYX_DT_F8E4M3 && t->ndim == 2 && t->shape[1] % 16 == 0 &&
                          w.scale->ndim == 2 && w.scale->shape[0] == t->shape[0] &&
                          w.scale->shape[1] == t->shape[1] / 16 &&
                          w.scale->nbytes == uint64_t(t->shape[0]) * uint64_t(t->shape[1] / 16);
    const bool ok_scale2 = w.scale2 && w.scale2->dtype == FLYX_DT_F32 && w.scale2->nbytes >= sizeof(float);
    if (!ok_scale || !ok_scale2)
      throw std::runtime_error(name + ": NVFP4 without valid scales");
  }
  weights.push_back(w);
  return index_[name] = int(weights.size()) - 1;
}

bool Tables::set_activations(const std::string& how, const std::string& declared, bool fp4_inputs,
                              std::string* err) {
  if (!how.empty() && how != "exact" && how != "nvfp4")
    return *err = "activations: '" + how + "' (exact, nvfp4, or as the file has it)", false;
  if (!declared.empty() && declared != "bf16" && declared != "nvfp4")
    return *err = "the file's activations: '" + declared + "' (bf16 or nvfp4)", false;
  exact_ = how == "exact";
  // A GGUF file may name tensors whose inputs are not to be rounded to 4 bits (llama.cpp's
  // general.tensor_extra.prec_a4, false for them; by default NVFP4 weights take FP4 inputs on Blackwell). The
  // backend rounds one model's NVFP4 inputs one way, so then none are: finer than llama.cpp for the others. A file
  // whose flags do not name one tensor each is refused whatever the word.
  bool flagged = false;
  const auto& h = file_->header();
  if (h.contains("gguf") && h.at("gguf").contains("general.tensor_extra.prec_a4")) {
    const auto& g = h.at("gguf");
    const auto& flags = g.at("general.tensor_extra.prec_a4");
    if (!g.contains("general.tensor_extra.name") || !flags.is_array() ||
        g.at("general.tensor_extra.name").size() != flags.size())
      return *err = "general.tensor_extra.prec_a4: not one flag a named tensor", false;
    for (const auto& f : flags) flagged = flagged || (f.is_boolean() && !f.get<bool>());
  }
  // NVFP4 inputs go through the FP4 tensor cores in tiles of 128 values of K: a GGUF file with a GGML NVFP4 matrix
  // of another K (a tiny model's 64-wide layers) keeps bf16 inputs for all of them (finer).
  bool other_k = false;
  for (const auto& name : file_->tensor_names()) {
    const flyx_tensor_view* t = file_->tensor(name);
    other_k = other_k || (t && t->dtype == FLYX_DT_GGML_NVFP4 && t->ndim == 2 && t->shape[1] % 128);
  }
  // Asked for by name, NVFP4 weights take NVFP4 inputs whatever the file says: a file converted with bf16 inputs
  // takes them without a second conversion, and neither the device nor the GGUF rules above
  // take them back. A backend that cannot compute them (no FP4 tensor cores, a K its kernels do not take) refuses the
  // model at open rather than giving bf16 ones (no silent change of contract). A model without NVFP4
  // weights is what it is without the word, as with "exact" on a bf16 model: a server that switches models keeps it.
  // Otherwise a GPU without FP4 tensor cores gives NVFP4 weights bf16 inputs: finer than the 8 bits llama.cpp gives
  // them there.
  nvfp4_inputs_ = how == "nvfp4" || (declared != "bf16" && fp4_inputs && !flagged && !other_k);
  return true;
}

int Tables::state(const flyx_state& s) {
  states.push_back(s);
  return int(states.size()) - 1;
}

int Tables::block(const flyx_block& b) {
  blocks.push_back(b);
  return int(blocks.size()) - 1;
}

int Tables::embed_block(int hidden, int vocab, const NameFn& n) {
  flyx_block e{};
  e.kind = FLYX_BLOCK_EMBED;
  e.embed = {hidden, vocab, weight(n("target.embed"), {vocab, hidden})};
  return block(e);
}

int Tables::head_block(int hidden, int vocab, int embed_block_idx, double norm_offset, double eps, const NameFn& n) {
  flyx_block hb{};
  hb.kind = FLYX_BLOCK_HEAD;
  hb.head.hidden = hidden;
  hb.head.vocab = vocab;
  hb.head.norm = norm(n("target.norm"), hidden, norm_offset, eps);
  // No output matrix: the head is the embedding table.
  hb.head.w = has(n("target.lm_head")) ? weight(n("target.lm_head"), {vocab, hidden}) : at(embed_block_idx).embed.table;
  return block(hb);
}

int Tables::attn_block(const std::string& prefix, const AttnBlockSpec& s, const NameFn& n) {
  const int H = s.hidden, q = s.n_heads * s.head_dim, kv = s.n_kv * s.head_dim;
  const int q_rows = (s.gate == FLYX_GATE_NONE ? 1 : 2) * q;
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = s.draft;
  st.layers = 1;
  st.n_kv = s.n_kv;
  st.head_dim = s.head_dim;
  flyx_block b{};
  b.kind = FLYX_BLOCK_ATTN;
  auto& a = b.attn;
  a.hidden = H;
  a.n_heads = s.n_heads;
  a.n_kv = s.n_kv;
  a.head_dim = s.head_dim;
  a.rot_dim = s.rot_dim;
  a.gate = s.gate;
  a.theta = s.theta;
  a.norm = norm(n(prefix + "in_norm"), H, s.norm_offset, s.eps);
  a.q = weight(n(prefix + "attn.q"), {q_rows, H});
  a.k = weight(n(prefix + "attn.k"), {kv, H});
  a.v = weight(n(prefix + "attn.v"), {kv, H});
  a.o = weight(n(prefix + "attn.o"), {H, q});
  if (s.qk_norm) {
    a.q_norm = norm(n(prefix + "attn.q_norm"), s.head_dim, s.norm_offset, s.eps);
    a.k_norm = norm(n(prefix + "attn.k_norm"), s.head_dim, s.norm_offset, s.eps);
  } else {
    a.q_norm = a.k_norm = flyx_norm{-1, 0, 0.0, 0.0};  // the heads go to the rotary as projected (backend_abi.h)
  }
  a.kv = state(st);
  for (int i = 0; i < 3; ++i) a.mrope_section[i] = s.mrope_section[i];
  a.mrope_interleaved = s.mrope_interleaved ? 1 : 0;
  return block(b);
}

int Tables::mlp_block(const std::string& prefix, int hidden, int inter, double norm_offset, double eps, const NameFn& n) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_MLP;
  auto& f = b.mlp;
  f.hidden = hidden;
  f.inter = inter;
  f.norm = norm(n(prefix + "post_norm"), hidden, norm_offset, eps);
  f.gate = weight(n(prefix + "mlp.gate"), {inter, hidden});
  f.up = weight(n(prefix + "mlp.up"), {inter, hidden});
  f.down = weight(n(prefix + "mlp.down"), {hidden, inter});
  return block(b);
}

void Model::finish(int hidden, int vocab) {
  desc_ = {};
  desc_.hidden = hidden;
  desc_.vocab = vocab;
  desc_.n_weights = int32_t(tables_.weights.size());
  desc_.n_states = int32_t(tables_.states.size());
  desc_.n_blocks = int32_t(tables_.blocks.size());
  desc_.weights = tables_.weights.data();
  desc_.states = tables_.states.data();
  desc_.blocks = tables_.blocks.data();
}

namespace {

// The parts a file says a reader must not ignore (a .flyx header's "requires", format 2: core/modelfile), each one
// the model of its architecture reads (`reads`); a companion's never, as no model reads them from one.
bool reads_required(const ModelFile& file, const std::vector<std::string>& reads, std::string* err) {
  const nlohmann::json& h = file.header();
  if (!h.is_object()) return true;
  if (h.contains("requires"))
    for (const auto& r : h.at("requires"))
      if (std::find(reads.begin(), reads.end(), r.get<std::string>()) == reads.end())
        return *err = "the file requires its part \"" + r.get<std::string>() + "\", which this build's " + file.arch() +
                      " model does not read (a later build may)",
               false;
  if (h.contains("companions"))
    for (const auto& c : h.at("companions"))
      if (c.is_object() && c.contains("requires"))
        return *err = "a file beside the model requires its parts " + c.at("requires").dump() +
                      ", which no model reads from a companion",
               false;
  return true;
}

struct ArchEntry {
  const char* name;
  std::unique_ptr<Model> (*open)(const ModelFile&, const ModelOptions&, std::string*);
  bool gguf;  // its GGUF files read too (a Llama's are not: llama.cpp's converter permutes its q and k rows)
  const char* reads;  // a part a file may require (format 2's "requires") that the model reads, or null
};

constexpr ArchEntry kArchitectures[] = {
    {"qwen35", qwen35::open, true, nullptr},
    {"qwen3", llama::open, true, "lora"},  // Llama's layer with q/k norms; their own adapters (a V-LoRA)
    {"llama", llama::open, false, "lora"},
};

}  // namespace

int32_t find_added_token(const ModelFile& file, std::string_view content) {
  const std::string_view raw = file.blob("tokenizer.added");
  if (raw.empty()) return -1;
  const nlohmann::json j = nlohmann::json::parse(raw.begin(), raw.end(), nullptr, false);
  if (!j.is_array()) return -1;
  for (const auto& item : j) {
    if (!item.is_object()) continue;
    if (item.contains("content") && item["content"].is_string() && item["content"].get<std::string>() == content &&
        item.contains("id") && item["id"].is_number_integer()) {
      return item["id"].get<int32_t>();
    }
  }
  return -1;
}

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  for (const ArchEntry& entry : kArchitectures) {
    if (file.arch() != entry.name) continue;
    std::vector<std::string> reads;
    if (entry.reads) reads.emplace_back(entry.reads);
    if (!reads_required(file, reads, err)) return nullptr;
    try {
      return entry.open(file, o, err);
    } catch (const std::exception& e) {
      *err = std::string("bad model file: ") + e.what();
      return nullptr;
    }
  }
  std::string known;
  for (const ArchEntry& entry : kArchitectures) known += (known.empty() ? "" : ", ") + std::string(entry.name);
  *err = "unsupported architecture '" + file.arch() + "' (this build knows " + known + ")";
  return nullptr;
}

std::vector<std::string> architectures() {
  std::vector<std::string> names;
  for (const ArchEntry& entry : kArchitectures)
    if (entry.gguf) names.emplace_back(entry.name);
  return names;
}

}  // namespace flyx::models

namespace flyx::interfaces {

std::unique_ptr<Model> open_model(const ModelFile& file, const ModelOptions& o, std::string* err) {
  return models::open(file, o, err);
}

std::vector<std::string> model_architectures() { return models::architectures(); }

}  // namespace flyx::interfaces
