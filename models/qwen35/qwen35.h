// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Qwen3.5 family (gated delta net and gated attention layers, an MTP
// layer, a block drafter) as blocks and programs.  No device code here;
// backends decide how each block runs.
#pragma once

#include <memory>
#include <string>

#include "models/model.h"

namespace flyx::models::qwen35 {

// The model in `file` with the drafter `o` asks for (the other drafter's weights are left out).
std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err);

}  // namespace flyx::models::qwen35
