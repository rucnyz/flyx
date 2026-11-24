// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A model file as the engine hands it to a backend: the
// description (weights, per-sequence state, typed blocks; backend_abi.h) and
// its programs.  models/<arch> builds one from the file; nothing outside
// models/ knows an architecture.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"

namespace flyx::models {

// The most a model file's dimensions may be: far past any model's, and small enough that the products the
// descriptions take of two of them (heads * head_dim, the GDN's heads * dims) fit an int with room.  A file that gives
// more is not a model; without the bound a product that wraps round would pass every shape check against tensors of
// the wrapped size, and a backend would index by dimensions its weights do not have.
inline constexpr int kMaxWidth = 1 << 20, kMaxVocab = 1 << 24, kMaxLayers = 1 << 12, kMaxHeads = 1 << 12,
                     kMaxHeadDim = 1 << 14;

using interfaces::ModelOptions;

// The drafters the architectures here can have.  Only models/ knows them by
// name: the engine sees drafters by number and the round widths each offers
// (draft_widths).  DFlash2 and DFlash (models/dflash2.h) and DSpine (models/dspine.h) draft a block at once;
// DV (models/dv.h) drafts trees, and its canvas follows its own rounds only
// (Model::restarts).
enum class Drafter { kNone, kMtp, kDFlash2, kDSpine, kDv, kDFlash };
const char* drafter_name(Drafter d);

// What a .flyx file says of its NVFP4 weights' inputs ("quant": {"activations": ...}; "" if nothing).
std::string declared_activations(const ModelFile& file);

// The drafters a model has: in its own file, or in drafter files beside it (ModelFile::add).
struct DrafterSources {
  bool mtp = false, dflash2 = false;                // the model has one
  bool mtp_beside = false, dflash2_beside = false;  // ... from a drafter file
  bool dv = false;                                  // the model file has a DV drafter
  bool dspine = false, dspine_beside = false;       // a DSpine drafter (where a DFlash2 one would be)
  bool dflash = false, dflash_beside = false;       // a DFlash drafter (likewise)
};

// The drafters a model holds, the MTP layer first: the one
// named; "all": every one it has; none named: every drafter given in a file of
// its own, and of the model file's own the MTP layer if it has one, else its
// block drafter, DFlash2 or DSpine (a file with both keeps the one measured faster on it, until
// choosing between them is measured there); its DV drafter only when named
// ("dv", "all"); none when the options say so.  DV drafts alone:
// beside another drafter it is refused.
bool choose_drafters(const ModelOptions& o, const DrafterSources& has, std::vector<Drafter>* out, std::string* err);

// Parameters for building a standard dense-decoder attention block via Tables::attn_block().
struct AttnBlockSpec {
  int hidden, n_heads, n_kv, head_dim, rot_dim;
  int gate;  // FLYX_GATE_NONE, FLYX_GATE_AFTER_Q, or FLYX_GATE_PER_HEAD
  double theta, norm_offset, eps;
  bool draft = false;
  bool qk_norm = true;  // false: no RMSNorm of each head's q and k (the block names none, weight -1)
  int mrope_section[3] = {0, 0, 0};  // 3D MRoPE sections (T, H, W); all 0 = standard 1D RoPE
  bool mrope_interleaved = false;
};

// The tables of a description, built from a file's tensors by name.
class Tables {
 public:
  using NameFn = std::function<std::string(const std::string&)>;

  explicit Tables(const ModelFile& file) : file_(&file) {}
  bool has(const std::string& name) const { return file_->tensor(name) != nullptr; }
  // A weight of this shape (NVFP4: the logical [N, K], with its scales).
  // Asking for the same tensor twice gives the same index.  Throws when the
  // tensor is missing or has another shape.
  int weight(const std::string& name, std::vector<int64_t> shape);
  flyx_norm norm(const std::string& name, int64_t n, double offset, double eps) {
    return {weight(name, {n}), 0, offset, eps};
  }
  flyx_vnorm vnorm(int kind, const std::string& w_name, const std::string& b_name, int64_t n, double offset,
                   double eps) {
    const int wi = w_name.empty() ? -1 : weight(w_name, {n});
    const int bi = b_name.empty() ? -1 : weight(b_name, {n});
    return {kind, wi, bi, 0, offset, eps};
  }
  int state(const flyx_state& s);
  int block(const flyx_block& b);
  flyx_block& at(int i) { return blocks[size_t(i)]; }
  // Standard dense-decoder block builders shared across architectures.
  int embed_block(int hidden, int vocab, const NameFn& n);
  int head_block(int hidden, int vocab, int embed_block_idx, double norm_offset, double eps, const NameFn& n);
  int attn_block(const std::string& prefix, const AttnBlockSpec& s, const NameFn& n);
  int mlp_block(const std::string& prefix, int hidden, int inter, double norm_offset, double eps, const NameFn& n);
  // How weights round their inputs: `how` (ModelOptions::activations) "" = as the file has it, "exact" = the
  // formats' own rounding in every pass, "nvfp4" = NVFP4 weights take NVFP4 inputs in passes of more rows whatever
  // the file says (other weights as the file has it); `declared` what the file says of its NVFP4 weights' inputs
  // ("bf16", "nvfp4", or "": NVFP4); `fp4_inputs` whether the device computes NVFP4 inputs (else bf16 ones by
  // default, asked for, NVFP4 ones, which the backend refuses). False with *err set for anything else.
  bool set_activations(const std::string& how, const std::string& declared, bool fp4_inputs, std::string* err);

  std::vector<flyx_weight> weights;
  std::vector<flyx_state> states;
  std::vector<flyx_block> blocks;

 private:
  const ModelFile* file_;
  std::map<std::string, int> index_;
  bool exact_ = false;
  bool nvfp4_inputs_ = true;  // NVFP4 weights take NVFP4 inputs in passes of more rows
};

// The engine's view of a model (core/interfaces/model.h) with the tables it is built from.
class Model : public interfaces::Model {
 public:
  const flyx_model_desc& desc() const override { return desc_; }
  int drafters() const override { return int(drafters_.size()); }
  const char* drafter_name(int d) const override { return models::drafter_name(drafters_[size_t(d)].kind); }
  const char* drafter() const override { return drafter_names_.c_str(); }
  std::vector<int> draft_widths(int d) const override;
  std::vector<int> adaptive_checkpoints(int d) const override;
  int block_drafts(int d) const override;
  int trained_ctx() const override { return trained_ctx_; }
  bool drafts_trees(int d) const override;
  bool file_drafts_trees() const override { return file_drafts_trees_; }
  bool restarts(int d) const override;
  bool kv_fp8_by_default() const override { return kv_fp8_by_default_; }
  const interfaces::VisionAdapter* vision() const override { return vision_adapter_.get(); }
  std::shared_ptr<const interfaces::VisionAdapter> vision_shared() const override { return vision_adapter_; }
  std::vector<flyx_node> vision_program() const override { return vision_nodes_; }

 protected:
  explicit Model(const ModelFile& file) : tables_(file) {}
  int trained_ctx_ = 0;  // set by the architecture from its file
  // A drafter the model holds: its block, the most drafts it proposes in a round.
  void add_drafter(Drafter kind, int block, int max_drafts);
  void set_vision(std::unique_ptr<interfaces::VisionAdapter> adapter, std::vector<flyx_node> nodes) {
    vision_adapter_ = std::move(adapter);
    vision_nodes_ = std::move(nodes);
  }
  // desc() points into the tables from here on; call once they are complete.
  void finish(int hidden, int vocab);

  struct DrafterBlock {
    Drafter kind;
    int block;       // its FLYX_BLOCK_MTP / FLYX_BLOCK_DFLASH2 / FLYX_BLOCK_DSPINE / FLYX_BLOCK_DV block
    int max_drafts;  // the most it proposes in a round
  };
  Tables tables_;
  std::vector<DrafterBlock> drafters_;
  std::shared_ptr<const interfaces::VisionAdapter> vision_adapter_;
  std::vector<flyx_node> vision_nodes_;
  bool kv_fp8_by_default_ = true;         // FP8 where a real model of the family showed no visible drop
  bool file_drafts_trees_ = false;        // the file holds DV, asked for or not
  bool verify_scales_with_width_ = false; // sparse-MoE targets whose verify cost climbs steeply with verify width

 private:
  flyx_model_desc desc_{};
  std::string drafter_names_ = "none";
};

// Look up an added/special token's ID by its exact text in the model file's tokenizer.added blob (-1 if absent).
int32_t find_added_token(const ModelFile& file, std::string_view content);

// The model in a file, by its architecture; null with *err set if the file
// cannot be read as one.
std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err);
// The architectures open() knows, as model files name theirs, whose GGUF files it reads (a Llama: .flyx only).
std::vector<std::string> architectures();

}  // namespace flyx::models
