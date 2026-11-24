// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DFlash2 and DFlash (backend_abi.h flyx_dflash2_block, python/reference/flyx_ref/draft.py) as blocks of a target: its
// layers (FLYX_BLOCK_DFLASH2_LAYER) and the drafter itself (FLYX_BLOCK_DFLASH2), from a "draft" config of kind
// "dflash2" or "dflash" (python/convert) and its draft.* tensors, in the model file or a drafter file beside it.
// DFlash is DFlash2's block without convolutions (conv_taps 0) or selector (sel_rank 0: each row's draft the head's
// best).  It reads the target's embedding and head, and the target's residual stream at its taps; every architecture
// that taps its layers takes it with no backend change.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "models/model.h"

namespace flyx::models::dflash2 {

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

}  // namespace flyx::models::dflash2
