// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine (backend_abi.h flyx_dspine_block, python/reference/flyx_ref/dspine.py) as blocks of a target: its layers
// (FLYX_BLOCK_DSPINE_LAYER: a Qwen3 layer and the code bridge's read and write after it) and the drafter itself
// (FLYX_BLOCK_DSPINE), from a "draft" config of kind "dspine" (python/convert/flyx_convert/dspine.py) and its
// draft.* tensors, in the model file or a drafter file beside it.  It reads the target's embedding and head, and the
// target's residual stream at its taps; every architecture that taps its layers takes it with no backend change.
#pragma once

#include <vector>

#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

// The drafter's taps (the residual stream after that many target layers), after checking the config fits the target;
// throws on one that does not, before any tensor is bound.
std::vector<int> dspine_taps(const nlohmann::json& d, const TargetSpec& target);

// Adds the drafter's state (its context K/V), its layers and its block to the tables; returns the block.  companion:
// the companion file that holds its tensors, -1: the model file.
int build_dspine(Tables& T, const nlohmann::json& d, const TargetSpec& target, int companion);

}  // namespace flyx::models::drafters
