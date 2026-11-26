// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Models as people name them: a file or a directory on disk, else a Hugging Face Hub name,
// "unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS" for a GGUF of a repository or "z-lab/Qwen3.8-27B-DFlash2" for a drafter,
// fetched into the Hugging Face cache (core/hub).  The engine itself is given paths.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/hub/hub.h"

namespace flyx::apps {

// A drafter that goes with a model, or one published for it.
struct DrafterSource {
  std::string kind;            // "mtp", "dflash2" or "dspine"
  std::string from;            // "model file" (its own MTP layer), a Hub name or file of a repository, or a path
  std::optional<uint64_t> size;  // bytes on disk; none where only the Hub could tell (or inside the model's file)
};

struct Fetched {
  std::string model;                  // a path
  std::vector<std::string> drafters;  // paths: the ones named, and the MTP file of the model's repository when the
                                      // model's GGUF lacks the layer
  std::string mmproj;                 // path to multimodal vision projector GGUF (empty: none)
  std::string mmproj_source;          // display label: repo/path (Hub) or local path
  std::optional<uint64_t> mmproj_size;  // bytes on disk
  std::string name;                   // the model's name for the API: its file's, without the extension
  std::vector<std::string> notes;     // for the log: a drafter published for this model, the MTP file taken
  std::vector<DrafterSource> drafter_sources;  // what the model drafts with: its own MTP layer, the files above
  std::vector<DrafterSource> suggested;        // block drafters published for it, none given (from: the Hub name)
};

// Fetches what is missing. want_mtp: take the repository's MTP file for a GGUF without the layer.
// want_vision: take the repository's mmproj GGUF when no explicit mmproj path was given.
bool fetch_model(const std::string& model, const std::vector<std::string>& drafters, bool want_mtp,
                 const hub::ProgressFn& progress, const std::atomic<bool>* stop, Fetched* out, std::string* err,
                 const std::string& mmproj = "", bool want_vision = true);

// Whether `eo` drafts with the MTP layer when the model has one (not --no-spec, not --drafter dflash2 or none).
bool wants_mtp(const EngineOptions& eo);

// Block drafters published for models made from these base models (a model card's base_model), by repository.
std::vector<std::string> known_drafters(const std::vector<std::string>& base_models);

// Throttled Hub fetch progress logger writing to stderr, optionally forwarding every tick to `hook`.
hub::ProgressFn make_progress_logger(const char* prefix, double interval_ms, hub::ProgressFn hook = {});

// The command line's way: the model and --drafter-file arguments made paths in place, progress on standard error.
bool fetch_named(std::string* model, EngineOptions* eo, std::string* err);

// A model's name for the API from its path: the file's name without .gguf / .flyx.
// Local paths recognize those extensions in any case; Hub model IDs keep their published spelling.
std::string model_name_of(const std::string& path, bool local = true);

// The MTP layers a GGUF says it holds (<arch>.nextn_predict_layers), read from its header alone: the metadata, never
// the tensors.  -1: not a GGUF, or not readable.
int gguf_mtp_layers(const std::string& path);

// Model files in directories of the user's own (flyx serve --models-dir, FLYX_MODELS_DIR), found by name: *.gguf
// (not projectors, importance matrices or MTP layers; complete split GGUFs grouped as one entry pointing to shard 1
// with their summed size) and *.flyx,
// at most 6 directories down, hidden directories and node_modules left out.  No file is opened.
struct LocalModel {
  std::string path;   // absolute
  std::string file;   // its name
  std::string kind;   // "gguf", "flyx", or "mmproj"
  std::string quant;  // from the name, "" when it names none
  uint64_t size = 0;
  int parts = 0;  // > 1 for a complete split GGUF (path points to shard 1, size is the sum)
  std::filesystem::file_time_type modified = std::filesystem::file_time_type::min();
};
// `deepest`: how many directories down a file may be (1: only in the directory named).
std::vector<LocalModel> scan_model_dirs(const std::vector<std::string>& dirs, int deepest = 6);
// FLYX_MODELS_DIR's directories: separated by ';' on Windows, ':' elsewhere (as PATH is).
std::vector<std::string> models_dirs_from_env();
// Unix seconds for the model listing; last_write_time's error value is reported as 0.
double modified_unix(std::filesystem::file_time_type modified);

}  // namespace flyx::apps
