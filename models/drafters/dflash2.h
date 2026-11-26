// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Architecture-agnostic DFlash2 speculative drafter parser and block builder.
// Supports three sources uniformly:
//   * embedded in a .flyx target model header ("draft"),
//   * a standalone .flyx drafter file or Hugging Face safetensors directory ("kind": "drafter"),
//   * a GGUF drafter file with general.architecture = "dflash" (non-Markov DFlash2 layout).
// The blocks are built from the names python/convert gives the drafter's tensors ("draft.fc", "draft.L0.attn.q");
// a GGUF file's own names are mapped from those, as an architecture's are (models/gguf.h).
#pragma once

#include <vector>

#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

struct DFlash2Spec {
  flyx_dflash2_block block{};
  int layers = 0;
  int tg2 = 0;
  double eps = 0.0;
  std::vector<int> taps;   // 1-based layer indices (1..target.layers)
  bool gguf = false;       // the file's tensor names are llama.cpp's (arch "dflash")
  int companion = -1;      // -1: embedded in target file; >= 0: companion index
};

// True if a companion header describes a drafter file (python/convert --draft: a DFlash2 or a DSpine drafter,
// models/drafters/dspine.h) or an arch="dflash" GGUF drafter.
bool is_dflash_companion(const nlohmann::json& header);

// Throws unless the drafter such a companion holds was made for this target (its width and vocabulary, and its
// layer count where the drafter names one).  A drafter given beside another model is refused whether or not it is
// chosen; the rest of its config is read only when it drafts (parse_dflash2_*, parse_dspine_json).
void check_made_for(const nlohmann::json& header, const TargetSpec& target);

// Parse and validate a DFlash2 spec from canonical JSON ("draft" section in .flyx or drafter_dir).
DFlash2Spec parse_dflash2_json(const nlohmann::json& j, const TargetSpec& target, int companion);

// Parse and validate a DFlash2 spec from a GGUF companion header with general.architecture == "dflash".
DFlash2Spec parse_dflash2_gguf(const ModelFile& file, const nlohmann::json& header, const TargetSpec& target,
                               int companion);

// Build FLYX_BLOCK_DFLASH2_LAYER blocks, draft KV state, and FLYX_BLOCK_DFLASH2 in `T`.
// Returns the FLYX_BLOCK_DFLASH2 block index.
int build_dflash2(Tables& T, const DFlash2Spec& spec, const TargetSpec& target);

}  // namespace flyx::models::drafters
