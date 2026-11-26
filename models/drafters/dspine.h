// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine (backend_abi.h flyx_dspine_block, python/reference/flyx_ref/dspine.py) as blocks of a target: its layers
// (FLYX_BLOCK_DSPINE_LAYER: a Qwen3 layer and the code bridge's read and write after it) and the drafter itself
// (FLYX_BLOCK_DSPINE), from a "draft" config of kind "dspine" (python/convert/flyx_convert/dspine.py) and its
// draft.* tensors, in the model file or a drafter file beside it.  It reads the target's embedding and head, and the
// target's residual stream at its taps; every architecture that taps its layers takes it with no backend change.
#pragma once

#include <vector>

#include "flyx/backend_abi.h"
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

struct DSpineSpec {
  flyx_dspine_block block{};
  int layers = 0;
  double eps = 0.0;
  std::vector<int> taps;  // 1-based layer indices (1..target.layers)
  int companion = -1;     // -1: in the target's file; >= 0: the companion file holding it
};

// A "draft" config of kind "dspine", checked against the target: throws on one that does not fit, before any tensor
// is bound.
DSpineSpec parse_dspine_json(const nlohmann::json& d, const TargetSpec& target, int companion);

// Adds the drafter's state (its context K/V), its layers and its block to the tables; returns the block.
int build_dspine(Tables& T, const DSpineSpec& spec, const TargetSpec& target);

}  // namespace flyx::models::drafters
