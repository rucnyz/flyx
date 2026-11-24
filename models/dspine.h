// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine (backend_abi.h flyx_dspine_block, python/reference/flyx_ref/dspine.py) as blocks of a target: its layers
// (FLYX_BLOCK_DSPINE_LAYER: a Qwen3 layer and the code bridge's read and write after it) and the drafter itself
// (FLYX_BLOCK_DSPINE), from a "draft" config of kind "dspine" (python/convert/flyx_convert/dspine.py) and its
// draft.* tensors, in the model file or a drafter file beside it.  It reads the target's embedding and head, and the
// target's residual stream at its taps; every architecture that taps its layers takes it with no backend change.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "models/model.h"

namespace flyx::models::dspine {

// The target a drafter is for: its width, vocabulary and layer count.
struct Target {
  int hidden = 0, vocab = 0, layers = 0;
  int embed = -1, head = -1;  // its blocks the drafter reads
};

// The drafter's taps (the residual stream after that many target layers), after checking the config fits the target;
// throws on one that does not, before any tensor is bound.
std::vector<int> taps(const nlohmann::json& d, const Target& t);

// Adds the drafter's state (its context K/V), its layers and its block to the tables; returns the block.  dn: the
// name of a drafter tensor ("draft.fc") in the file that holds it.
int describe(Tables& T, const nlohmann::json& d, const Target& t, const std::function<std::string(const std::string&)>& dn);

}  // namespace flyx::models::dspine
