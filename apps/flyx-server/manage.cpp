// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The dashboard's model panel: what the server serves, loading another, and the Hub's files.
//
//   GET /v1/load          the load's state (ModelHost::status; also telemetry's "load")
//   POST /v1/load         {"model": PATH or HUB NAME, "drafters": [...], "ctx": N,
//                         "drafter": "mtp|dflash2|dv|all|none", "speculative": BOOL, "kv": "fp8|16", "name": ID}
//                         -> 202 and the state; 409 while one loads
//   DELETE /v1/load       stops a fetch under way (what came stays for the next try)
//   GET /v1/hub/config    current Hugging Face Hub endpoint, default endpoint, source ("setting", "HF_ENDPOINT",
//                         "default"), and offline flag
//   POST /v1/hub/config   {"endpoint": URL or ""} sets or resets the Hub endpoint (persisted to serve.json)
//   GET /v1/hub/search?q=TEXT&sort=trending|downloads|likes|updated|created&limit=N&cursor=C&arch=A   the Hub's
//                         repositories with GGUF files; without q, filtered to the architectures this build runs
//                         (optionally narrowed to arch=A), paginated via next_cursor
//   GET /v1/hub/files?repo=ORG/NAME   a repository's models: its GGUFs by quantization with their sizes (and which
//                         are in the cache already), its MTP file, the DFlash2 drafters known for it, what the Hub
//                         says of it (downloads, license, its GGUFs' architecture) and whether this build runs that
//   GET /v1/hub/dir?path=PATH   a folder of this machine: the folders in it, and the model files in it and two folders
//                         down, as names the panel can load; a model file's own path lists that file.  Without
//                         path: where to start from (the model directories, the home directory, on Windows the disks)
//   POST /v1/hub/dir      {"title": TEXT, "start": PATH} asks the person at the server's machine for a folder, with
//                         the desktop's own dialog, and answers as GET does for the one chosen ({"path": null} when
//                         they cancel).  For a browser on that machine only: 403 from another, 501 where no dialog
//                         can be shown, 409 while one is open
//   GET /v1/hub/cached    the models on this machine, as names the panel can load: every snapshot of the Hugging Face
//                         cache's repositories, and the model files in --models-dir / FLYX_MODELS_DIR
//   DELETE /v1/hub/cached?name=NAME   deletes a model the cache holds (never a file of those directories), unless it
//                         is being served or a load is under way: 409
//   POST /v1/unload       the model served goes; the settings file still names it
//   POST /v1/agents/launch  {"agent": "claude|codex", "dir": PATH} opens a window of PowerShell on the server's
//                         machine, in folder dir (without it, one the person picks in the desktop's folder dialog),
//                         running that coding agent against the model served (agent_launch.h), with an optional
//                         permission_mode for Claude Code: {"launched": true,
//                         "dir"}, or {"launched": false} when the dialog is cancelled.  For a browser on that machine
//                         only: 403 from another, 409 with no model served or a dialog open, 501 where no window can
//                         be opened (not Windows, SSH, a service)
//
// All but GET /v1/load change the machine's state or reach the Hub for it: may_manage().
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

#include "apps/flyx-server/agent_launch.h"
#include "apps/flyx-server/gpu_apps.h"
#include "apps/flyx-server/server.h"
#include "core/interfaces/model.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace flyx::server {

namespace {

// Single desktop folder dialog guard shared across POST /v1/hub/dir and POST /v1/agents/launch.
std::atomic<bool> g_dialog_open{false};

struct DialogGuard {
  bool held = false;
  bool acquire() { return (held = !g_dialog_open.exchange(true)); }
  ~DialogGuard() {
    if (held) g_dialog_open.store(false);
  }
};

namespace fs = std::filesystem;

using os::from_path;
using os::to_path;

bool on_disk(const std::string& path) {
  std::error_code ec;
  return fs::exists(to_path(path), ec);
}

void error(httplib::Response& res, int status, const std::string& message, const char* type) {
  send_json(res, status, {{"error", {{"message", message}, {"type", type}}}});
}

// The MTP layers a GGUF holds, remembered by its path, size and time: its header is read once, not on every listing.
int mtp_layers(const std::string& path, uint64_t size, fs::file_time_type modified) {
  constexpr size_t kMaxCachedMtpEntries = 1024;
  static std::mutex mu;
  static std::map<std::string, std::tuple<uint64_t, fs::file_time_type, int>> known;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto it = known.find(path);
    if (it != known.end() && std::get<0>(it->second) == size && std::get<1>(it->second) == modified)
      return std::get<2>(it->second);
  }
  const int n = apps::gguf_mtp_layers(path);
  std::lock_guard<std::mutex> lock(mu);
  if (known.size() >= kMaxCachedMtpEntries && known.find(path) == known.end()) known.clear();
  known[path] = {size, modified, n};
  return n;
}

// One model of GET /v1/hub/cached, and what deleting it takes.
struct Listed {
  ojson json;
  std::string repo, commit;          // hub: the snapshot it is in
  std::vector<std::string> files;    // hub: its files, as paths in the repository
  std::vector<std::string> on_disk;  // its files on disk
};

ojson or_null(const std::string& s) { return s.empty() ? ojson(nullptr) : ojson(s); }

// A model file of a directory, as the panel lists it: its path is the name that loads it.
Listed dir_entry(const apps::LocalModel& m) {
  Listed l;
  const bool gguf = m.kind == "gguf";
  l.json = {{"name", m.path}, {"kind", m.kind}, {"source", "dir"}, {"repo", nullptr}, {"quant", or_null(m.quant)},
            {"file", m.file}, {"path", m.path}, {"size", m.size}, {"modified", apps::modified_unix(m.modified)},
            {"revision", nullptr}, {"current", true},
            {"mtp", gguf && mtp_layers(m.path, m.size, m.modified) > 0 ? ojson("in-file") : ojson(nullptr)}};
  if (m.parts > 1) l.json["parts"] = m.parts;
  l.on_disk = {m.path};
  return l;
}

// What GET /v1/hub/cached lists: the cache's snapshots, refs/main's first, a file under one path once (refs/main's
// copy, else the newest snapshot's); then the model directories' files.
std::vector<Listed> listing(const Server& s) {
  const std::vector<hub::Cached> snapshots = hub::cached();
  std::set<std::string> repos_with_mtp;  // the repository's MTP layer is in the cache, in any snapshot
  std::set<std::pair<std::string, std::string>> snaps_with_mmproj;  // (repo, commit) holding a vision projector
  for (const hub::Cached& c : snapshots) {
    hub::Repo r;
    r.files = c.files;
    if (hub::mtp_file(r)) repos_with_mtp.insert(c.repo);
    if (hub::mmproj_file(r)) snaps_with_mmproj.insert({c.repo, c.commit});
  }
  std::vector<Listed> out;
  std::set<std::pair<std::string, std::string>> seen;  // (repository, path)
  for (const hub::Cached& c : snapshots) {
    hub::Repo r;
    r.id = c.repo, r.revision = c.current ? "main" : c.commit, r.commit = c.commit, r.files = c.files;
    r.architectures = c.architectures, r.from_cache = true;
    const std::string suffix = c.current ? "" : "@" + c.commit;
    auto disk_of = [&](const std::string& path) { return from_path((to_path(c.dir) / to_path(path)).make_preferred()); };
    auto entry = [&](const std::string& name, const char* kind, const std::string& quant, const std::string& file,
                     const std::string& disk, uint64_t size) {
      Listed l;
      l.repo = c.repo, l.commit = c.commit;
      std::error_code ec;
      const auto modified = fs::last_write_time(to_path(disk), ec);
      ojson mtp = nullptr;
      if (std::string(kind) == "gguf") {
        if (mtp_layers(disk, size, modified) > 0) mtp = "in-file";
        else if (repos_with_mtp.count(c.repo)) mtp = "in-repo";
      }
      l.json = {{"name", name + suffix}, {"kind", kind}, {"source", "hub"}, {"repo", c.repo}, {"quant", or_null(quant)},
                {"file", file}, {"path", disk}, {"size", size}, {"modified", apps::modified_unix(modified)},
                {"revision", c.commit}, {"current", c.current}, {"mtp", mtp},
                {"mmproj", std::string(kind) == "gguf" && snaps_with_mmproj.count({c.repo, c.commit})
                               ? ojson("in-repo")
                               : ojson(nullptr)}};
      return l;
    };
    const std::vector<hub::GgufChoice> ggufs = hub::gguf_choices(r);
    for (const hub::GgufChoice& g : ggufs) {
      if ((g.split && !g.complete) || !seen.insert({c.repo, g.file.path}).second) continue;
      const std::string disk = disk_of(g.file.path);
      Listed l = entry(c.repo + ":" + g.quant, "gguf", g.quant, g.file.path.substr(g.file.path.rfind('/') + 1), disk,
                       g.total_size);
      if (g.split) l.json["parts"] = int(g.parts.size());
      for (const hub::RepoFile& p : g.parts) l.files.push_back(p.path), l.on_disk.push_back(disk_of(p.path));
      out.push_back(std::move(l));
    }
    // A drafter's config and safetensors, or a lone .flyx file: what the repository's name alone loads.
    std::vector<hub::RepoFile> files;
    std::string ignored;
    if (!ggufs.empty() || !hub::select(r, {c.repo}, &files, &ignored) || !seen.insert({c.repo, files[0].path}).second)
      continue;
    const bool drafter = files[0].path == "config.json";
    Listed l;
    uint64_t size = 0;
    for (const hub::RepoFile& f : files) size += f.size;
    if (drafter) {
      l = entry(c.repo, "drafter", "", c.repo.substr(c.repo.find('/') + 1), from_path(to_path(c.dir).make_preferred()),
                size);
    } else {
      const std::string base = files[0].path.substr(files[0].path.rfind('/') + 1);
      l = entry(c.repo, "flyx", hub::quant_in_name(base), base, disk_of(files[0].path), size);
    }
    for (const hub::RepoFile& f : files) l.files.push_back(f.path), l.on_disk.push_back(disk_of(f.path));
    out.push_back(std::move(l));
  }
  for (const apps::LocalModel& m : apps::scan_model_dirs(s.models_dirs)) out.push_back(dir_entry(m));
  return out;
}

// What GET /v1/hub/dir answers for a folder (or a model file) of this machine; "" asks where to start from.
bool dir_json(const Server& s, const std::string& asked, ojson* out, std::string* err) {
  constexpr int kLevels = 3;          // a model is listed from its folder, or from one or two folders above it
  constexpr size_t kMostFolders = 2000;
  ojson dirs = ojson::array(), models = ojson::array();
  std::error_code ec;
  if (asked.empty()) {
    std::set<fs::path> seen;
    auto start = [&](const std::string& dir) {
      std::error_code e2;
      const fs::path p = fs::weakly_canonical(to_path(dir), e2);
      if (!e2 && fs::is_directory(p, e2) && seen.insert(p).second)
        dirs.push_back({{"name", from_path(p)}, {"path", from_path(p)}});
    };
    for (const std::string& d : s.models_dirs) start(d);
    start(os::home_dir());
#if defined(_WIN32)
    for (char letter = 'C'; letter <= 'Z'; ++letter) start(std::string(1, letter) + ":\\");  // the disks, where models often are
#endif
    *out = {{"path", ""}, {"parent", nullptr}, {"kind", "start"}, {"dirs", dirs}, {"models", models}};
    return true;
  }
  const fs::path given = fs::absolute(to_path(asked), ec);
  const fs::path p = ec ? fs::path{} : fs::weakly_canonical(given, ec);
  if (ec || p.empty() || !fs::exists(p, ec)) return *err = asked + ": no such folder or file on this machine", false;
  const bool is_dir = fs::is_directory(p, ec);
  const fs::path base = is_dir ? p : p.parent_path();
  if (!is_dir) {
    // The file alone: the scan of its folder knows whether it is a model's (and a split model's parts as one).
    for (const apps::LocalModel& m : apps::scan_model_dirs({from_path(base)}, 1)) {
      std::error_code e2;
      if (fs::equivalent(to_path(m.path), p, e2)) models.push_back(dir_entry(m).json);
    }
    if (models.empty()) return *err = asked + ": not a model file (.gguf, .flyx)", false;
    *out = {{"path", from_path(p)}, {"parent", from_path(base)}, {"kind", "file"}, {"dirs", dirs}, {"models", models}};
    return true;
  }
  std::vector<fs::path> subs;
  for (fs::directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = from_path(it->path().filename());
    std::error_code e2;
    if (name.empty() || name[0] == '.' || name[0] == '$' || !it->is_directory(e2)) continue;
    subs.push_back(it->path());
    if (subs.size() >= kMostFolders) break;
  }
  std::sort(subs.begin(), subs.end());
  for (const fs::path& sub : subs) dirs.push_back({{"name", from_path(sub.filename())}, {"path", from_path(sub)}});
  for (const apps::LocalModel& m : apps::scan_model_dirs({from_path(p)}, kLevels)) models.push_back(dir_entry(m).json);
  const fs::path up = p.parent_path();
  *out = {{"path", from_path(p)}, {"parent", up.empty() || up == p ? ojson(nullptr) : ojson(from_path(up))},
          {"kind", "dir"}, {"dirs", dirs}, {"models", models}};
  return true;
}

// Whether `file` is, or is under, one of the paths the served model reads (a link followed to its blob, a hard link
// to the same file).
bool in_use(const std::string& file, const std::vector<std::string>& used) {
  std::error_code ec;
  const fs::path f = fs::weakly_canonical(to_path(file), ec);
  for (const std::string& u : used) {
    std::error_code e2;
    const fs::path p = fs::weakly_canonical(to_path(u), e2);
    if (e2 || p.empty()) continue;
    if (f == p || fs::equivalent(to_path(file), to_path(u), e2)) return true;
    const fs::path under = f.lexically_relative(p), lexical = to_path(file).lexically_relative(to_path(u));
    for (const fs::path& rel : {under, lexical})
      if (fs::is_directory(p, e2) && !rel.empty() && *rel.begin() != "..") return true;
  }
  return false;
}

ojson repo_json(const hub::Repo& r) {
  const std::string snap = hub::snapshot_dir(r) + "/";
  const bool dflash_gguf_repo = r.info.gguf_architecture == "dflash";
  ojson ggufs = ojson::array();
  ojson companions = ojson::array();
  std::set<std::string> comp_paths;
  auto push_comp = [&](const hub::RepoFile& f, const std::string& kind, bool is_default) {
    if (!comp_paths.insert(f.path).second) return;
    const std::string base = f.path.substr(f.path.rfind('/') + 1);
    const bool cached = on_disk(snap + f.path);
    const std::string disk = from_path((to_path(hub::snapshot_dir(r)) / to_path(f.path)).make_preferred());
    companions.push_back({{"name", r.id + ":" + base},
                          {"path", f.path},
                          {"file", base},
                          {"kind", kind},
                          {"size", f.size},
                          {"cached", cached},
                          {"disk", cached ? ojson(disk) : ojson(nullptr)},
                          {"default", is_default}});
  };
  const std::vector<hub::GgufChoice> choices = hub::gguf_choices(r);
  if (dflash_gguf_repo) {
    for (size_t idx = 0; idx < choices.size(); ++idx)
      if (!choices[idx].split) push_comp(choices[idx].file, "dflash2", idx == 0);
  } else {
    for (const hub::GgufChoice& c : choices) {
      bool cached = c.complete;
      for (const hub::RepoFile& p : c.parts) cached = cached && on_disk(snap + p.path);
      ggufs.push_back({{"name", r.id + ":" + c.quant}, {"quant", c.quant}, {"size", c.total_size}, {"split", c.split},
                       {"parts", int(c.parts.size())}, {"complete", c.complete}, {"cached", cached}});
    }
  }
  ojson mtp = nullptr;
  if (const hub::RepoFile* f = hub::mtp_file(r))
    mtp = {{"path", f->path}, {"size", f->size}, {"cached", on_disk(snap + f->path)}};
  ojson mmproj = nullptr;
  if (const hub::RepoFile* f = hub::mmproj_file(r))
    mmproj = {{"path", f->path}, {"size", f->size}, {"cached", on_disk(snap + f->path)}};
  for (const hub::CompanionFile& c : hub::companion_files(r)) push_comp(c.file, c.kind, c.is_default);
  // A block drafter's directory: config.json naming the architecture, and safetensors; no GGUF.
  uint64_t weights = 0;
  bool config = false;
  for (const hub::RepoFile& f : r.files) {
    config = config || f.path == "config.json";
    if (f.path.find('/') == std::string::npos && f.path.size() > 12 &&
        f.path.compare(f.path.size() - 12, 12, ".safetensors") == 0)
      weights += f.size;
  }
  // What the Hub says of it; and, where it named the GGUFs' architecture, whether this build opens that one.
  const hub::RepoInfo& i = r.info;
  const std::vector<std::string> known = interfaces::model_architectures();
  const ojson runs = dflash_gguf_repo
                         ? ojson(true)
                         : i.gguf_architecture.empty()
                               ? ojson(nullptr)
                               : ojson(std::find(known.begin(), known.end(), i.gguf_architecture) != known.end());
  const ojson info = {{"downloads", i.downloads}, {"likes", i.likes}, {"updated", or_null(i.updated)},
                      {"license", or_null(i.license)}, {"pipeline", or_null(i.pipeline)}, {"gated", i.gated},
                      {"architecture", or_null(i.gguf_architecture)},
                      {"context", i.gguf_context ? ojson(i.gguf_context) : ojson(nullptr)},
                      {"parameters", i.gguf_parameters ? ojson(i.gguf_parameters) : ojson(nullptr)}};
  return {{"repo", r.id},
          {"commit", r.commit},
          {"from_cache", r.from_cache},
          {"base_models", r.base_models},
          {"ggufs", ggufs},
          {"mtp", mtp},
          {"mmproj", mmproj},
          {"companions", companions},
          {"drafter", ggufs.empty() && config && weights > 0 && hub::is_drafter(r) ? ojson({{"size", weights}}) : ojson(nullptr)},
          {"drafter_repo", dflash_gguf_repo},
          {"architectures", r.architectures},
          {"known_drafters", apps::known_drafters(r.base_models)},
          {"info", r.from_cache ? ojson(nullptr) : info},
          {"runs", runs},
          {"build_architectures", known},
          {"endpoint", hub::endpoint()}};
}

}  // namespace

void add_manage_routes(httplib::Server& http, Server* s) {
  http.Get("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    send_json(r, 200, s->host->status());
  });
  http.Post("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    const ojson body = interfaces::parse_json(q.body);
    ModelHost::Want w;
    std::string err;
    if (body.is_discarded() || !want_from_json(body, s->base, &w, &err))
      return error(r, 400, err.empty() ? "request body is not a JSON object" : err, "invalid_request_error");
    if (!s->host->start(w, &err)) return error(r, 409, err, "conflict");
    send_json(r, 202, s->host->status());
  });
  http.Delete("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    s->host->cancel();
    send_json(r, 200, s->host->status());
  });
  auto hub_config_json = []() -> ojson {
    return {{"endpoint", hub::endpoint()},
            {"default_endpoint", "https://huggingface.co"},
            {"source", hub::endpoint_source()},
            {"offline", hub::offline()}};
  };
  http.Get("/v1/hub/config", [s, hub_config_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "configuring Hugging Face")) return;
    send_json(r, 200, hub_config_json());
  });
  http.Post("/v1/hub/config", [s, hub_config_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "configuring Hugging Face")) return;
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    std::string raw;
    if (body.contains("endpoint") && !body["endpoint"].is_null()) {
      if (!body["endpoint"].is_string())
        return error(r, 400, "\"endpoint\": a URL such as https://hf-mirror.com, or empty to reset", "invalid_request_error");
      raw = body["endpoint"].get<std::string>();
    }
    std::string normalized, err;
    if (!hub::normalize_endpoint(raw, &normalized, &err)) return error(r, 400, err, "invalid_request_error");
    // Clear the saved override when emptied, or when explicitly set back to huggingface.co with no HF_ENDPOINT env.
    const std::string saved =
        normalized.empty() || (normalized == "https://huggingface.co" && os::getenv_str(env::kHfEndpoint).empty())
            ? std::string()
            : normalized;
    if (!hub::set_endpoint(saved, &err)) return error(r, 400, err, "invalid_request_error");
    if (!ModelHost::save_hub_endpoint(saved, &err)) return error(r, 500, err, "server_error");
    std::fprintf(stderr, "flyx-server: Hugging Face endpoint set to %s\n", hub::endpoint().c_str());
    send_json(r, 200, hub_config_json());
  });
  http.Get("/v1/hub/files", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    hub::Name name;
    if (!q.has_param("repo") || !hub::parse_name(q.get_param_value("repo"), &name))
      return error(r, 400, "repo: a Hugging Face repository, org/name", "invalid_request_error");
    hub::Repo repo;
    std::string err;
    if (!hub::describe(name.repo, name.revision, &repo, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, repo_json(repo));
  });
  http.Get("/v1/hub/search", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    const std::string asked = q.get_param_value("q");
    const std::string sort = q.get_param_value("sort");
    const std::string cursor = q.get_param_value("cursor");
    const std::string arch_filter = q.get_param_value("arch");
    const int limit = query_int(q, "limit", 20, 1, 100);
    const std::vector<std::string> known = interfaces::model_architectures();
    std::vector<hub::Found> found;
    std::string next_cursor, err;
    if (!asked.empty()) {
      if (!hub::search(asked, sort, limit, &found, &err, cursor, &next_cursor, /*gguf_only=*/true))
        return error(r, 502, err, "hub_error");
      if (cursor.empty()) {
        std::set<std::string> seen_ids;
        for (const auto& f : found) seen_ids.insert(f.id);
        std::string low = asked;
        for (char& c : low) c = char(std::tolower(static_cast<unsigned char>(c)));
        const bool drafter_query = low.find("dflash") != std::string::npos || low.find("dspine") != std::string::npos ||
                                   low.find("drafter") != std::string::npos || low.find("mtp") != std::string::npos;
        // Include block drafters' safetensors repos (which filter=gguf omits) when searching drafters or when no GGUF
        // matches.
        if (found.empty() || drafter_query) {
          std::vector<hub::Found> extra;
          std::string e2;
          if (hub::search(asked, sort, limit, &extra, &e2, "", nullptr, /*gguf_only=*/false)) {
            for (auto& x : extra) {
              if (x.is_drafter && seen_ids.insert(x.id).second) found.push_back(std::move(x));
            }
          }
        }
        // If the query names an exact repository (org/name), look it up directly so safetensors drafters or newly
        // published repos are always found even when Hub search omits them.
        hub::Name exact;
        if (hub::parse_name(asked, &exact) && seen_ids.insert(exact.repo).second) {
          hub::Repo repo;
          std::string e2;
          if (hub::describe(exact.repo, exact.revision, &repo, &e2)) {
            hub::Found f;
            f.id = repo.id;
            f.downloads = repo.info.downloads;
            f.likes = repo.info.likes;
            f.updated = repo.info.updated;
            f.pipeline = repo.info.pipeline;
            f.architecture = repo.info.gguf_architecture;
            f.gated = repo.info.gated;
            f.is_drafter = hub::is_drafter(repo) || repo.info.gguf_architecture == "dflash";
            found.insert(found.begin(), std::move(f));
          }
        }
      }
    } else {
      // Without a query, browse GGUF repositories of every architecture this build runs (or one chosen via `arch`),
      // resuming cleanly across HF batches with a "<skip>:<hf_cursor>" cursor so nothing is skipped or duplicated.
      std::set<std::string> allow;
      if (!arch_filter.empty() && std::find(known.begin(), known.end(), arch_filter) != known.end())
        allow.insert(arch_filter);
      else
        allow.insert(known.begin(), known.end());
      size_t skip = 0;
      std::string hf_cursor;
      if (!cursor.empty()) {
        const size_t colon = cursor.find(':');
        if (colon != std::string::npos) {
          try {
            skip = std::stoull(cursor.substr(0, colon));
          } catch (const std::exception&) {
            skip = 0;
          }
          hf_cursor = cursor.substr(colon + 1);
        } else {
          hf_cursor = cursor;
        }
      }
      constexpr int kMaxBatches = 4;
      for (int step = 0; step < kMaxBatches && int(found.size()) < limit; ++step) {
        const std::string batch_start = hf_cursor;
        std::vector<hub::Found> batch;
        std::string batch_next;
        if (!hub::search("", sort, 100, &batch, &err, batch_start, &batch_next)) {
          if (found.empty()) return error(r, 502, err, "hub_error");
          break;
        }
        std::vector<hub::Found> matched;
        for (auto& f : batch)
          if (allow.count(f.architecture) > 0) matched.push_back(std::move(f));
        size_t i = skip;
        for (; i < matched.size() && int(found.size()) < limit; ++i) found.push_back(std::move(matched[i]));
        if (int(found.size()) >= limit) {
          if (i < matched.size()) next_cursor = std::to_string(i) + ":" + batch_start;
          else if (!batch_next.empty()) next_cursor = "0:" + batch_next;
          break;
        }
        skip = 0;
        hf_cursor = batch_next;
        if (hf_cursor.empty()) break;
        if (step + 1 == kMaxBatches) next_cursor = "0:" + hf_cursor;
      }
    }
    std::set<std::string> here;  // repositories with a snapshot in the cache
    for (const hub::Cached& c : hub::cached()) here.insert(c.repo);
    const std::set<std::string> known_set(known.begin(), known.end());
    ojson models = ojson::array();
    for (const hub::Found& f : found) {
      const bool drafter = f.is_drafter || f.architecture == "dflash";
      const ojson runs = drafter ? ojson(true)
                                 : f.architecture.empty() ? ojson(nullptr) : ojson(known_set.count(f.architecture) > 0);
      models.push_back({{"id", f.id}, {"downloads", f.downloads}, {"likes", f.likes}, {"updated", or_null(f.updated)},
                        {"created", or_null(f.created)}, {"pipeline", or_null(f.pipeline)}, {"gated", f.gated},
                        {"cached", here.count(f.id) > 0}, {"architecture", or_null(f.architecture)}, {"runs", runs},
                        {"drafter", drafter}});
    }
    send_json(r, 200,
              {{"query", asked},
               {"family", asked.empty()},
               {"architectures", known},
               {"arch", or_null(arch_filter)},
               {"endpoint", hub::endpoint()},
               {"next_cursor", or_null(next_cursor)},
               {"models", models}});
  });
  http.Get("/v1/hub/dir", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    ojson out;
    std::string err;
    if (!dir_json(*s, q.get_param_value("path"), &out, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, out);
  });
  http.Post("/v1/hub/dir", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    // The dialog opens on this machine's desktop: of use, and polite, only to someone sitting at it.
    if (!from_this_machine(q))
      return error(r, 403, "the folder dialog opens on the server's machine: give the folder's path instead", "permission_error");
    DialogGuard dialog;
    if (!dialog.acquire()) return error(r, 409, "a folder dialog is already open on the server's machine", "conflict");
    const ojson body = interfaces::parse_json(q.body);
    auto text = [&](const char* key) {
      return body.is_object() && body.contains(key) && body[key].is_string() ? body[key].get<std::string>() : std::string();
    };
    const std::string title = text("title");
    std::string path, err;
    const os::Picked picked = os::pick_folder(title.empty() ? "Choose a folder of model files" : title, text("start"), &path, &err);
    if (picked == os::Picked::kUnavailable) return error(r, 501, err, "not_supported");
    if (picked == os::Picked::kCancelled) return send_json(r, 200, {{"path", nullptr}});
    ojson out;
    if (!dir_json(*s, path, &out, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, out);
  });
  http.Get("/v1/hub/cached", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    ojson models = ojson::array();
    for (Listed& l : listing(*s)) models.push_back(std::move(l.json));
    send_json(r, 200, {{"cache", hub::cache_dir()}, {"cache_from", hub::cache_source()}, {"dirs", s->models_dirs},
                       {"models", models}});
  });
  http.Delete("/v1/hub/cached", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    const std::string name = q.get_param_value("name");
    if (name.empty()) return error(r, 400, "name: a model as GET /v1/hub/cached lists it", "invalid_request_error");
    std::vector<Listed> all = listing(*s);
    const auto it = std::find_if(all.begin(), all.end(), [&](const Listed& l) { return l.json["name"] == name; });
    if (it == all.end()) return error(r, 400, name + " is not among the cache's models", "invalid_request_error");
    if (it->json["source"] == "dir")
      return error(r, 400, name + " is a file of a model directory of yours: flyx deletes only from the Hugging Face cache",
                   "invalid_request_error");
    // When deleting the last model in a snapshot, also reclaim any unreferenced mmproj files in that snapshot
    // (unless currently mounted by the served model).
    int models_in_snap = 0;
    for (const Listed& l : all)
      if (l.json["source"] == "hub" && l.repo == it->repo && l.commit == it->commit) ++models_in_snap;
    std::vector<std::pair<std::string, std::string>> orphan_mmproj;  // (repo-relative path, disk path)
    if (models_in_snap == 1) {
      for (const hub::Cached& c : hub::cached()) {
        if (c.repo != it->repo || c.commit != it->commit) continue;
        for (const hub::RepoFile& f : c.files) {
          if (hub::mmproj_gguf_name(f.path)) {
            const std::string disk = from_path((to_path(c.dir) / to_path(f.path)).make_preferred());
            orphan_mmproj.emplace_back(f.path, disk);
          }
        }
      }
    }
    bool served = false, ok = false;
    uint64_t freed = 0;
    std::string err;
    const bool idle = s->host->with_files_in_use(
        [&](const std::vector<std::string>& used) {
          for (const std::string& f : it->on_disk) served = served || in_use(f, used);
          if (!served) {
            std::vector<std::string> to_remove = it->files;
            for (const auto& [rel, disk] : orphan_mmproj)
              if (!in_use(disk, used)) to_remove.push_back(rel);
            ok = hub::remove_cached(it->repo, it->commit, to_remove, &freed, &err);
          }
        },
        &err);
    if (!idle) return error(r, 409, err, "conflict");
    if (served) return error(r, 409, name + " is being served: load another model, or unload it, first", "conflict");
    if (!ok) return error(r, 500, err, "server_error");
    std::fprintf(stderr, "flyx-server: deleted %s from the cache (%.1f GB)\n", name.c_str(), double(freed) / 1e9);
    send_json(r, 200, {{"deleted", name}, {"freed", freed}});
  });
  http.Post("/v1/unload", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    std::string err;
    if (!s->host->unload(&err)) return error(r, 409, err, "conflict");
    send_json(r, 200, s->host->status());
  });
  http.Post("/v1/agents/launch", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "starting coding agents")) return;
    // The window opens on this machine's desktop, as the folder dialog does: for someone sitting at it.
    if (!from_this_machine(q))
      return error(r, 403, "the agent starts in a window on the server's machine: run its command where you are",
                   "permission_error");
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    auto text = [&](const char* key) {
      return body.contains(key) && body[key].is_string() ? body[key].get<std::string>() : std::string();
    };
    const std::string agent = text("agent");
    const auto& agents = launch_agents();
    if (std::find(agents.begin(), agents.end(), agent) == agents.end())
      return error(r, 400, "agent must be one of: claude, codex", "invalid_request_error");
    const std::string mode = text("permission_mode");
    if (body.contains("permission_mode") &&
        (agent != "claude" || (mode != "default" && mode != "acceptEdits" && mode != "auto" && mode != "bypassPermissions")))
      return error(r, 400, "permission_mode is for Claude Code and must be default, acceptEdits, auto or bypassPermissions",
                   "invalid_request_error");
    if (!os::shell_window_supported())
      return error(r, 501, "opening a terminal window is supported on Windows only", "not_supported");
    const ModelHost::Served served = s->host->served();
    if (!served.sched) return error(r, 409, "no model is served: load one first", "conflict");
    // The server as this request reached it: a loopback address, the agent being on the same machine.
    std::string host = q.local_addr.empty() ? "127.0.0.1" : q.local_addr;
    if (host.find(':') != std::string::npos) host = "[" + host + "]";
    const AgentTarget target{"http://" + host + ":" + std::to_string(q.local_port), served.name,
                             served.sched->max_ctx(), !s->cfg.api_key.empty(), mode};
    const std::string command = agent_powershell(agent, target);
    if (command.empty())
      return error(r, 409, "the model id has one of \" & | < > ^ % in it, which cannot be passed to Codex from "
                           "PowerShell: run the command on the page", "conflict");
    std::string dir = text("dir"), err;
    if (dir.empty()) {
      DialogGuard dialog;
      if (!dialog.acquire()) return error(r, 409, "a folder dialog is already open on the server's machine", "conflict");
      const os::Picked picked = os::pick_folder(
          agent == "claude" ? "Choose the project folder for Claude Code" : "Choose the project folder for Codex", "", &dir,
          &err);
      if (picked == os::Picked::kUnavailable) return error(r, 501, err, "not_supported");
      if (picked == os::Picked::kCancelled) return send_json(r, 200, {{"launched", false}});
    }
    std::error_code ec;
    if (!fs::is_directory(to_path(dir), ec)) return error(r, 404, "no such folder: " + dir, "not_found");
    std::vector<std::pair<std::string, std::string>> env;
    if (target.keyed) env.emplace_back("FLYX_KEY", s->cfg.api_key);
    if (!os::open_shell_window(dir, command, env, &err)) return error(r, 501, err, "not_supported");
    send_json(r, 200, {{"launched", true}, {"dir", dir}});
  });
  auto device_hint = [s]() -> std::string {
    const ModelHost::Served served = s->host->served();
    if (served.sched) {
      TelemetrySnapshot t;
      flyx_device_stats d{};
      bool ok = false;
      served.sched->telemetry(&t, &d, &ok);
      if (!t.model.device.empty()) return t.model.device;
    }
    if (s->device_probe.is_object() && s->device_probe.contains("name") && s->device_probe["name"].is_string())
      return s->device_probe["name"].get<std::string>();
    return "";
  };
  auto gpu_snapshot_json = [](const GpuAppSnapshot& snap, const char* kind, bool can_close) {
    ojson apps = ojson::array();
    for (const auto& a : snap.apps) {
      apps.push_back({{"id", a.id},
                      {"name", a.name},
                      {"process", a.process_name},
                      {"bytes", a.bytes},
                      {"local_bytes", a.local_bytes},
                      {"processes", a.processes},
                      {"closable", a.closable},
                      {"reason", a.reason.empty() ? ojson(nullptr) : ojson(a.reason)}});
    }
    return ojson{{"kind", kind},
                 {"supported", snap.supported},
                 {"can_close", can_close},
                 {"device", snap.device_name.empty() ? ojson(nullptr) : ojson(snap.device_name)},
                 {"mem_total", snap.mem_total},
                 {"mem_used", snap.mem_used},
                 {"mem_free", snap.mem_free},
                 {"own_bytes", snap.own_bytes},
                 {"others_bytes", snap.others_bytes},
                 {"system_bytes", snap.system_bytes},
                 {"idle_igpu", snap.idle_igpu.empty() ? ojson(nullptr) : ojson(snap.idle_igpu)},
                 {"apps", std::move(apps)}};
  };
  http.Get("/v1/device/processes", [s, device_hint, gpu_snapshot_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "inspecting memory usage")) return;
    const std::string kind = q.has_param("kind") ? q.get_param_value("kind") : "vram";
    if (kind != "vram" && kind != "ram") return error(r, 400, "kind must be vram or ram", "invalid_request_error");
    r.set_header("Cache-Control", "no-store");
    const bool ram = kind == "ram";
    const GpuAppSnapshot snap = ram ? list_ram_apps() : list_gpu_apps(device_hint());
    if (!ram && snap.mem_total > 0) s->probe_mem_used.store(snap.mem_used, std::memory_order_relaxed);
    send_json(r, 200, gpu_snapshot_json(snap, ram ? "ram" : "vram", from_this_machine(q)));
  });
  http.Post("/v1/device/close", [s, device_hint, gpu_snapshot_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "closing applications")) return;
    if (!from_this_machine(q))
      return error(r, 403, "closing applications is only allowed from this machine", "permission_error");
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    if (!body.contains("id") || !body["id"].is_string() || body["id"].get<std::string>().empty())
      return error(r, 400, "id: an application id from GET /v1/device/processes", "invalid_request_error");
    bool force = false;
    if (body.contains("force") && !body["force"].is_null()) {
      if (!body["force"].is_boolean()) return error(r, 400, "force: true or false", "invalid_request_error");
      force = body["force"].get<bool>();
    }
    std::string kind = "vram";
    if (body.contains("kind") && !body["kind"].is_null()) {
      if (!body["kind"].is_string()) return error(r, 400, "kind must be vram or ram", "invalid_request_error");
      kind = body["kind"].get<std::string>();
      if (kind != "vram" && kind != "ram") return error(r, 400, "kind must be vram or ram", "invalid_request_error");
    }
    const bool ram = kind == "ram";
    const GpuCloseResult res =
        ram ? close_ram_app(body["id"].get<std::string>(), force) : close_gpu_app(device_hint(), body["id"].get<std::string>(), force);
    if (!ram && res.snapshot.mem_total > 0) s->probe_mem_used.store(res.snapshot.mem_used, std::memory_order_relaxed);
    if (res.status == CloseStatus::kUnsupported)
      return error(r, 501, "closing applications is not supported on this system", "not_supported");
    if (res.status == CloseStatus::kNotFound)
      return error(r, 404, "that application is no longer listed", "not_found");
    if (res.status == CloseStatus::kForbidden)
      return error(r, 403, "that application is protected and cannot be closed from FLYX", "permission_error");
    if (res.status == CloseStatus::kAccessDenied)
      return error(r, 403, "permission denied: close that application manually", "permission_error");
    const char* status = res.status == CloseStatus::kClosed   ? "closed"
                       : res.status == CloseStatus::kNoWindow ? "no_window"
                                                              : "still_running";
    send_json(r, 200, {{"status", status},
                       {"windows_signaled", res.windows_signaled},
                       {"name", res.app_name},
                       {"snapshot", gpu_snapshot_json(res.snapshot, ram ? "ram" : "vram", true)}});
  });
}

}  // namespace flyx::server
