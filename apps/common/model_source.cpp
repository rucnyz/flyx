// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/common/model_source.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <set>

#include "core/interfaces/options.h"
#include "core/modelfile/modelfile.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace flyx::apps {

namespace {

namespace fs = std::filesystem;

using os::from_path;
using os::to_path;

bool exists_on_disk(const std::string& path) {
  std::error_code ec;
  return fs::exists(to_path(path), ec);
}

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// The bytes of a file, or of a drafter's directory (its own files, not those below it).
uint64_t bytes_on_disk(const std::string& path) {
  std::error_code ec;
  const fs::path p = to_path(path);
  if (!fs::is_directory(p, ec)) {
    const auto sz = fs::file_size(p, ec);
    return ec ? 0 : uint64_t(sz);
  }
  uint64_t n = 0;
  for (fs::directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code e2;
    if (!it->is_regular_file(e2) || e2) continue;
    const auto sz = it->file_size(e2);
    if (!e2) n += uint64_t(sz);
  }
  return n;
}

}  // namespace

int gguf_mtp_layers(const std::string& path) { return flyx::gguf_peek_mtp_layers(path); }

double modified_unix(fs::file_time_type modified) {
  if (modified == fs::file_time_type::min()) return 0;
#if __cpp_lib_chrono >= 201907L
  const auto time = std::chrono::clock_cast<std::chrono::system_clock>(modified);
#else
  // libstdc++ exposes file_clock::to_sys before the full C++20 chrono API.
  const auto time = fs::file_time_type::clock::to_sys(modified);
#endif
  return std::chrono::duration<double>(time.time_since_epoch()).count();
}

std::vector<std::string> models_dirs_from_env() {
#if defined(_WIN32)
  constexpr char kSep = ';';
#else
  constexpr char kSep = ':';
#endif
  std::vector<std::string> out;
  const std::string v = os::getenv_str(env::kModelsDir);
  for (size_t start = 0; start <= v.size();) {
    const size_t end = std::min(v.find(kSep, start), v.size());
    if (end > start) out.push_back(v.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

std::vector<LocalModel> scan_model_dirs(const std::vector<std::string>& dirs, int deepest) {
  constexpr int kMostEntries = 200000;     // a directory named by mistake (a whole disk) ends the scan early
  std::vector<LocalModel> out;
  std::set<fs::path> seen;  // a file reached from two of the directories (one inside the other) once
  struct LocalShard {
    std::string path, file;
    uint64_t size = 0;
    fs::file_time_type modified{};
  };
  struct SplitAcc {
    int count = 0;
    std::map<int, LocalShard> parts;
  };
  std::map<std::pair<std::string, int>, SplitAcc> splits;
  for (const std::string& dir : dirs) {
    std::error_code ec;
    const fs::path root = fs::absolute(to_path(dir), ec);
    if (ec) continue;
    int entries = 0;
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end && ++entries < kMostEntries; it.increment(ec)) {
      const std::string name = from_path(it->path().filename());
      std::error_code e2;
      if (it->is_directory(e2)) {
        if (name.empty() || name[0] == '.' || name[0] == '$' || name == "node_modules" ||
            name == "System Volume Information" || it.depth() + 1 >= deepest)
          it.disable_recursion_pending();
        continue;
      }
      const std::string low = lower(name);
      const bool gguf = low.ends_with(".gguf"), flyx = low.ends_with(".flyx");
      if (!gguf && !flyx) continue;
      std::string rel = from_path(it->path().lexically_relative(root));
      std::replace(rel.begin(), rel.end(), '\\', '/');
      const bool is_mmproj = gguf && hub::mmproj_gguf_name(rel);
      // A DFlash2 drafter's GGUF says what it is in its header alone (general.architecture "dflash"), and reading
      // a header means reading through the file's tokenizer tables: only the files whose names suggest a drafter
      // are opened here.  One named otherwise is listed as a model, and refused as one when it is opened.
      const bool maybe_dflash =
          gguf && !is_mmproj && (low.find("dflash") != std::string::npos || low.find("draft") != std::string::npos);
      const bool is_drafter =
          gguf && !is_mmproj &&
          (hub::mtp_gguf_name(rel) || (maybe_dflash && flyx::gguf_peek(from_path(it->path())).arch == "dflash"));
      if (gguf && !is_mmproj && !is_drafter && !hub::model_gguf_name(rel)) continue;
      const fs::path canon = fs::weakly_canonical(it->path(), e2);
      if (e2 || !it->is_regular_file(e2) || e2 || !seen.insert(canon).second) continue;
      const auto sz = it->file_size(e2);
      if (e2) continue;
      std::error_code e3;
      const auto mt = it->last_write_time(e3);
      const fs::file_time_type modified = e3 ? fs::file_time_type::min() : mt;
      hub::SplitGguf sp;
      if (gguf && !is_mmproj && !is_drafter && hub::parse_split_gguf_name(from_path(canon), &sp)) {
        auto& acc = splits[{sp.prefix, sp.count}];
        acc.count = sp.count;
        acc.parts[sp.part] = {from_path(it->path()), name, uint64_t(sz), modified};
        continue;
      }
      LocalModel m;
      m.path = from_path(it->path());
      m.file = name;
      m.kind = is_mmproj ? "mmproj" : is_drafter ? "drafter" : gguf ? "gguf" : "flyx";
      m.quant = hub::quant_in_name(name);
      m.size = uint64_t(sz);
      m.modified = modified;
      out.push_back(std::move(m));
    }
  }
  for (const auto& [_, acc] : splits) {
    if (int(acc.parts.size()) != acc.count) continue;
    bool complete = true;
    uint64_t total = 0;
    fs::file_time_type latest = fs::file_time_type::min();
    int expect = 1;
    for (const auto& [part_no, shard] : acc.parts) {
      if (part_no != expect++) {
        complete = false;
        break;
      }
      total += shard.size;
      if (shard.modified > latest) latest = shard.modified;
    }
    if (!complete) continue;
    const LocalShard& first = acc.parts.begin()->second;
    LocalModel m;
    m.path = first.path;
    m.file = first.file;
    m.kind = "gguf";
    m.quant = hub::quant_in_name(first.file);
    m.size = total;
    m.parts = acc.count;
    m.modified = latest;
    out.push_back(std::move(m));
  }
  std::sort(out.begin(), out.end(), [](const LocalModel& a, const LocalModel& b) { return a.path < b.path; });
  return out;
}

namespace {

std::string gigabytes(uint64_t n) {
  char s[32];
  std::snprintf(s, sizeof s, "%.1f GB", double(n) / 1e9);
  return s;
}

// One thing named: a path as it is, else the Hub's files for the name.  A drafter's config and safetensors
// return its directory; a single or split GGUF returns its (first) model file.
bool one(const std::string& spec, const hub::ProgressFn& progress, const std::atomic<bool>* stop, std::string* path,
         hub::Repo* repo, std::string* err) {
  if (exists_on_disk(spec)) return *path = spec, true;
  hub::Name name;
  if (!hub::parse_name(spec, &name))
    return *err = spec + ": no such file, and not a Hugging Face name (org/name, org/name:QUANT)", false;
  std::vector<hub::RepoFile> files;
  std::vector<std::string> paths;
  if (!hub::describe(name.repo, name.revision, repo, err) || !hub::select(*repo, name, &files, err) ||
      !hub::fetch(*repo, files, &paths, progress, stop, err))
    return false;
  // A drafter's config and safetensors: its directory; else the (first) model file.
  *path = files[0].path == "config.json" ? hub::snapshot_dir(*repo) : paths[0];
  return true;
}

// What kind of drafter a file or directory given beside a model holds: a GGUF's from its header when it can be read
// (arch "dflash" against MTP), else from its name; a .flyx file or a directory of config and safetensors, the block
// drafter it holds (DFlash2's or DSpine's, as its header says; a file that cannot be read: DFlash2's, which loading it
// says more of).
std::string drafter_kind(const std::string& path) {
  const std::string lp = lower(path);
  if (lp.ends_with(".gguf")) {
    const flyx::GgufPeek p = flyx::gguf_peek(path);
    if (p.valid) {
      if (p.arch == "dflash") return "dflash2";
      if (p.mtp_layers > 0) return "mtp";
    }
    return lp.find("dflash") != std::string::npos ? "dflash2" : "mtp";
  }
  ModelFile f;
  std::string ignored;
  if (f.open(path, &ignored) && f.header().contains("draft") && f.header()["draft"].is_object())
    return f.header()["draft"].value("kind", std::string("dflash2"));
  return "dflash2";
}

}  // namespace

std::string model_name_of(const std::string& path, bool local) {
  const fs::path file = to_path(path).filename();
  hub::SplitGguf sp;
  if (hub::parse_split_gguf_name(from_path(file), &sp)) return sp.prefix;
  std::string extension = from_path(file.extension());
  if (local) extension = lower(extension);
  return from_path(extension == ".gguf" || extension == ".flyx" ? file.stem() : file);
}

namespace {

// Block drafters trained for a model, by the model they were trained on (docs/models.md), and their kinds.
struct KnownDrafter {
  const char* model;
  const char* drafter;
  const char* kind;
};
constexpr KnownDrafter kKnown[] = {
    {"Qwen/Qwen3.8-27B", "z-lab/Qwen3.8-27B-DFlash2", "dflash2"},
    {"Qwen/Qwen3.8-27B", "tencent/Qwen3.8-27B-DSpine-b16", "dspine"},
    {"Qwen/Qwen3.5-4B", "vllm-project/Qwen3.5-4B-DFlash2", "dflash2"},
    {"Qwen/Qwen3-8B", "tencent/Qwen3-8B-DSpine-b16", "dspine"},
};

std::string known_kind(const std::string& drafter) {
  for (const KnownDrafter& k : kKnown)
    if (drafter == k.drafter) return k.kind;
  return "";
}

std::string drafter_label(const std::string& kind) { return kind == "dspine" ? "DSpine" : "DFlash2"; }

}  // namespace

std::vector<std::string> known_drafters(const std::vector<std::string>& base_models) {
  std::vector<std::string> out;
  for (const std::string& b : base_models)
    for (const KnownDrafter& k : kKnown)
      if (b == k.model) out.push_back(k.drafter);
  return out;
}

bool wants_mtp(const EngineOptions& eo) {
  if (!eo.speculative) return false;
  interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
  return interfaces::parse_drafter_mode(eo.drafter, &mode) &&
         (mode == interfaces::DrafterMode::kAuto || mode == interfaces::DrafterMode::kMtp ||
          mode == interfaces::DrafterMode::kAll);
}

bool fetch_model(const std::string& model, const std::vector<std::string>& drafters, bool want_mtp,
                 const hub::ProgressFn& progress, const std::atomic<bool>* stop, Fetched* out, std::string* err,
                 const std::string& mmproj, bool want_vision) {
  *out = {};
  hub::Repo repo;
  if (!one(model, progress, stop, &out->model, &repo, err)) return false;
  out->name = model_name_of(out->model, repo.id.empty());
  const int own_mtp = want_mtp ? gguf_mtp_layers(out->model) : -1;  // -1: not a GGUF, or not asked
  if (own_mtp > 0) out->drafter_sources.push_back({"mtp", "model file", std::nullopt});
  for (const std::string& d : drafters) {
    hub::Repo drepo;
    std::string path;
    if (!one(d, progress, stop, &path, &drepo, err)) return false;
    out->drafters.push_back(path);
    out->drafter_sources.push_back({drafter_kind(path), drepo.id.empty() ? path : d, bytes_on_disk(path)});
  }
  if (want_vision && !mmproj.empty()) {
    hub::Repo mrepo;
    std::string mpath;
    if (!one(mmproj, progress, stop, &mpath, &mrepo, err)) return false;
    out->mmproj = mpath;
    out->mmproj_source = mrepo.id.empty() ? mpath : mmproj;
    out->mmproj_size = bytes_on_disk(mpath);
  }
  if (repo.id.empty()) return true;  // a path: nothing more to know
  if (want_vision && out->mmproj.empty()) {
    if (const hub::RepoFile* mm = hub::mmproj_file(repo)) {
      std::vector<std::string> paths;
      if (!hub::fetch(repo, {*mm}, &paths, progress, stop, err)) return false;
      out->mmproj = paths[0];
      out->mmproj_source = repo.id + "/" + mm->path;
      out->mmproj_size = mm->size;
      out->notes.push_back("the vision projector from " + repo.id + "/" + mm->path);
    }
  }
  // unsloth keeps the MTP layer of GGUFs made without it in MTP/.
  bool has_mtp_drafter = false;
  for (const auto& ds : out->drafter_sources)
    if (ds.kind == "mtp") has_mtp_drafter = true;
  const hub::RepoFile* mtp = hub::mtp_file(repo);
  if (want_mtp && mtp && own_mtp == 0 && !has_mtp_drafter) {
    std::vector<std::string> paths;
    if (!hub::fetch(repo, {*mtp}, &paths, progress, stop, err)) return false;
    out->drafters.push_back(paths[0]);
    out->drafter_sources.push_back({"mtp", repo.id + "/" + mtp->path, bytes_on_disk(paths[0])});
    out->notes.push_back("the MTP layer from " + repo.id + "/" + mtp->path + " (the model's file has none)");
  }
  // A model takes one block drafter: beside one given, none is suggested.
  bool block_given = false;
  for (const DrafterSource& s : out->drafter_sources) block_given = block_given || s.kind != "mtp";
  for (const std::string& k : known_drafters(repo.base_models)) {
    bool given = block_given;
    for (const std::string& d : drafters) given = given || d.rfind(k, 0) == 0;
    if (given) continue;
    const std::string kind = known_kind(k);
    out->notes.push_back("a " + drafter_label(kind) + " drafter is published for this model: --drafter-file " + k);
    // Its size where the cache has it; the Hub is not asked for a suggestion.
    std::optional<uint64_t> size;
    hub::Repo cached;
    std::vector<hub::RepoFile> files;
    std::string ignored;
    if (hub::in_cache(k, "main", &cached) && hub::select(cached, {k}, &files, &ignored)) {
      size = 0;
      for (const hub::RepoFile& f : files) *size += f.size;
    }
    out->suggested.push_back({kind, k, size});
  }
  return true;
}

hub::ProgressFn make_progress_logger(const char* prefix, double interval_ms, hub::ProgressFn hook) {
  auto last = std::make_shared<double>(0.0);
  return [prefix, interval_ms, hook = std::move(hook), last](const hub::Progress& p) {
    if (hook) hook(p);
    const double now = os::now_ms();
    if (now - *last < interval_ms && p.done < p.total) return;
    *last = now;
    std::fprintf(stderr, "%s: fetching %s: %s of %s, %.0f MB/s%s\n", prefix, p.file.c_str(), gigabytes(p.done).c_str(),
                 gigabytes(p.total).c_str(), p.bytes_per_s / 1e6, p.checking ? ", checking" : "");
  };
}

bool fetch_named(std::string* model, EngineOptions* eo, std::string* err) {
  auto progress = make_progress_logger("flyx", 2000);
  Fetched f;
  if (!fetch_model(*model, eo->drafter_files, wants_mtp(*eo), progress, nullptr, &f, err, eo->mmproj_file, eo->vision))
    return false;
  if (f.model != *model) std::fprintf(stderr, "flyx: %s is %s\n", model->c_str(), f.model.c_str());
  for (size_t i = 0; i < eo->drafter_files.size() && i < f.drafters.size(); ++i)
    if (f.drafters[i] != eo->drafter_files[i])
      std::fprintf(stderr, "flyx: %s is %s\n", eo->drafter_files[i].c_str(), f.drafters[i].c_str());
  for (const std::string& n : f.notes) std::fprintf(stderr, "flyx: %s\n", n.c_str());
  *model = f.model;
  eo->drafter_files = f.drafters;
  eo->mmproj_file = f.mmproj;
  return true;
}

}  // namespace flyx::apps
