// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/registry.h"

#include <stdexcept>
#include <string>

#include "models/gguf.h"

namespace flyx::models::drafters {

using nlohmann::json;

namespace {

std::string companion_label(int i) { return "companion file " + std::to_string(i + 1) + ": "; }

}  // namespace

void DrafterRegistry::scan(const ModelFile& file, const TargetSpec& target, const MtpHooks* mtp, bool has_dv,
                           std::vector<int>* unclaimed) {
  *this = DrafterRegistry();
  if (unclaimed) unclaimed->clear();

  const json& hdr = file.header();
  sources_.dv = has_dv;
  if (mtp && mtp->embedded) sources_.mtp = true;
  if (hdr.is_object() && hdr.contains("draft") && hdr["draft"].is_object())
    (hdr["draft"].value("kind", "") == "dspine" ? sources_.dspine : sources_.dflash2) = true;

  for (int i = 0; i < file.companions(); ++i) {
    const json& h = hdr.at("companions").at(size_t(i));
    const std::string what = companion_label(i);
    if (is_dflash_companion(h)) {
      if (sources_.dflash2 || sources_.dspine) throw std::runtime_error(what + "the model has a block drafter already");
      try {
        check_made_for(h, target);
      } catch (const std::exception& e) {
        throw std::runtime_error(what + e.what());
      }
      if (!GgufMeta::is_header(h) && h.at("draft").value("kind", "") == "dspine") {
        sources_.dspine = true;
        sources_.dspine_beside = true;
        dspine_companion_ = i;
      } else {
        sources_.dflash2 = true;
        sources_.dflash2_beside = true;
        dflash2_companion_ = i;
      }
    } else if (mtp && GgufMeta::is_header(h) && h.value("arch", "") == target.arch) {
      if (!target.gguf) throw std::runtime_error(what + "an MTP layer from a GGUF file goes with a GGUF model");
      if (sources_.mtp) throw std::runtime_error(what + "the model has its MTP layer already");
      const GgufMeta cm(h);
      if (cm.count("embedding_length") != target.hidden || cm.count_or("nextn_predict_layers", 0, 0) != 1 ||
          cm.count("block_count") != target.layers + 1)
        throw std::runtime_error(what + "no MTP layer for this model (" + target.arch + ", width " +
                                 std::to_string(target.hidden) + ", " + std::to_string(target.layers) + " layers)");
      sources_.mtp = true;
      sources_.mtp_beside = true;
      mtp_companion_ = i;
    } else {
      if (unclaimed) unclaimed->push_back(i);
    }
  }
}

const DFlash2Spec& DrafterRegistry::ensure_dflash2_spec(const ModelFile& file, const TargetSpec& target) {
  if (dflash2_spec_) return *dflash2_spec_;
  const int i = dflash2_companion_;
  if (i < 0) return dflash2_spec_.emplace(parse_dflash2_json(file.header().at("draft"), target, -1));
  const json& h = file.header().at("companions").at(size_t(i));
  try {
    dflash2_spec_ = GgufMeta::is_header(h) ? parse_dflash2_gguf(file, h, target, i)
                                           : parse_dflash2_json(h.at("draft"), target, i);
  } catch (const std::exception& e) {
    throw std::runtime_error(companion_label(i) + e.what());
  }
  return *dflash2_spec_;
}

const DSpineSpec& DrafterRegistry::ensure_dspine_spec(const ModelFile& file, const TargetSpec& target) {
  if (dspine_spec_) return *dspine_spec_;
  const int i = dspine_companion_;
  if (i < 0) return dspine_spec_.emplace(parse_dspine_json(file.header().at("draft"), target, -1));
  try {
    dspine_spec_ = parse_dspine_json(file.header().at("companions").at(size_t(i)).at("draft"), target, i);
  } catch (const std::exception& e) {
    throw std::runtime_error(companion_label(i) + e.what());
  }
  return *dspine_spec_;
}

void DrafterRegistry::validate_selected(const ModelFile& file, const std::vector<Drafter>& selected,
                                        const TargetSpec& target) {
  for (Drafter kind : selected) {
    if (kind == Drafter::kDFlash2) (void)ensure_dflash2_spec(file, target);
    if (kind == Drafter::kDSpine) (void)ensure_dspine_spec(file, target);
  }
}

int DrafterRegistry::attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target,
                            const MtpHooks* mtp, int* max_drafts_out) {
  if (kind == Drafter::kMtp) {
    if (!mtp || !mtp->build_layer || !mtp->tensor_name)
      throw std::runtime_error("MTP drafter is not supported by this model architecture");
    const auto [attn_blk, mlp_blk] = mtp->build_layer(mtp_companion_);
    flyx_block b{};
    b.kind = FLYX_BLOCK_MTP;
    auto& m = b.mtp;
    m.embed = target.embed;
    m.head = target.head;
    m.attn = attn_blk;
    m.mlp = mlp_blk;
    m.norm_emb = T.norm(mtp->tensor_name("mtp.norm_emb", mtp_companion_), target.hidden, mtp->norm_offset, mtp->eps);
    m.norm_hidden =
        T.norm(mtp->tensor_name("mtp.norm_hidden", mtp_companion_), target.hidden, mtp->norm_offset, mtp->eps);
    m.norm = T.norm(mtp->tensor_name("mtp.norm", mtp_companion_), target.hidden, mtp->norm_offset, mtp->eps);
    m.fc = T.weight(mtp->tensor_name("mtp.fc", mtp_companion_), {target.hidden, 2 * target.hidden});
    flyx_state ring{};
    ring.kind = FLYX_STATE_RING;
    ring.draft = 1;
    ring.width = target.hidden;
    m.ring = T.state(ring);
    const int blk = T.block(b);
    context_blocks_.push_back(blk);
    if (max_drafts_out) *max_drafts_out = FLYX_MAX_ROWS - 1;
    return blk;
  }
  if (kind == Drafter::kDFlash2) {
    const DFlash2Spec& spec = ensure_dflash2_spec(file, target);
    dflash2_block_ = build_dflash2(T, spec, target);
    dflash2_taps_ = spec.taps;
    context_blocks_.push_back(dflash2_block_);
    if (max_drafts_out) *max_drafts_out = spec.block.block - 1;
    return dflash2_block_;
  }
  if (kind == Drafter::kDSpine) {
    const DSpineSpec& spec = ensure_dspine_spec(file, target);
    dspine_block_ = build_dspine(T, spec, target);
    context_blocks_.push_back(dspine_block_);
    if (max_drafts_out) *max_drafts_out = spec.block.block - 1;
    return dspine_block_;
  }
  throw std::runtime_error("unsupported drafter kind in DrafterRegistry::attach");
}

void DrafterRegistry::emit_taps(std::vector<flyx_node>* p, int layer_1based) const {
  // The block drafter's features, whichever drafter drafts: its context stays current.
  auto taps = [&](int block, const std::vector<int>& after) {
    if (block < 0) return;
    for (size_t s = 0; s < after.size(); ++s)
      if (after[s] == layer_1based) p->push_back({FLYX_NODE_TAP, block, int32_t(s), 0});
  };
  taps(dflash2_block_, dflash2_taps_);
  if (dspine_spec_) taps(dspine_block_, dspine_spec_->taps);
}

void DrafterRegistry::emit_contexts(std::vector<flyx_node>* p) const {
  for (int blk : context_blocks_) p->push_back({FLYX_NODE_CONTEXT, blk, 0, 0});
}

}  // namespace flyx::models::drafters
