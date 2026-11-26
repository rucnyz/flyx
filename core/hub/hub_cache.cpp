// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Hugging Face Hub cache layout (models--<org>--<name>/{blobs,snapshots,refs}): inspection, placement, and removal.
#include "core/hub/hub_internal.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <mutex>

#include "core/modelfile/readers.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"

namespace flyx::hub {

namespace fs = std::filesystem;
using os::from_path;
using os::to_path;

namespace {

std::string token() {
  std::string t = os::getenv_str(env::kHfToken);
  if (t.empty()) {
    std::ifstream f(to_path(detail::hf_home() + "/token"));
    std::getline(f, t);
  }
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
  return t;
}

// The files of a snapshot on disk (links followed), sizes only.
std::vector<RepoFile> snapshot_files(const fs::path& snap) {
  std::vector<RepoFile> files;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(snap, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code e2;
    if (!fs::is_regular_file(it->path(), e2)) continue;  // follows the link; a broken one is no file
    RepoFile f;
    f.path = from_path(it->path().lexically_relative(snap));  // fs::relative would follow the link to the blob
    std::replace(f.path.begin(), f.path.end(), '\\', '/');
    f.size = fs::file_size(it->path(), e2);
    if (!e2) files.push_back(std::move(f));
  }
  std::sort(files.begin(), files.end(), [](const RepoFile& a, const RepoFile& b) { return a.path < b.path; });
  return files;
}

// A snapshot's config.json's architectures, if it has one; a DSpine drafter published before its architecture had that
// name (DFlashDraftModel) reads as DSpineDraftModel, as the drafter directory reader takes it.
std::vector<std::string> architectures_of(const fs::path& snap) {
  std::vector<std::string> out;
  std::ifstream f(snap / "config.json", std::ios::binary);
  if (!f) return out;
  const auto j = nlohmann::json::parse(f, nullptr, false);
  if (j.is_object() && j.contains("architectures") && j["architectures"].is_array())
    for (const auto& a : j["architectures"])
      if (a.is_string()) out.push_back(a.get<std::string>());
  if (j.is_object() && modelfile::is_dspine_config(j))
    std::replace(out.begin(), out.end(), std::string("DFlashDraftModel"), std::string("DSpineDraftModel"));
  return out;
}

// Whether `p` is `root` or lies under it, both made absolute with their links followed.
bool inside(const fs::path& p, const fs::path& root) {
  std::error_code ec, e2;
  const fs::path a = fs::weakly_canonical(p, ec), r = fs::weakly_canonical(root, e2);
  if (ec || e2 || r.empty()) return false;
  const fs::path rel = a.lexically_relative(r);
  return !rel.empty() && *rel.begin() != "..";
}

// A standalone copy has no link back to its blob. Verify the contents before reclaiming either file.
bool same_contents(const fs::path& a, const fs::path& b, uint64_t size) {
  constexpr uint64_t kChunk = uint64_t(1) << 20;
  std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
  if (!fa || !fb) return false;
  std::vector<char> x(size_t(std::min(size, kChunk))), y(x.size());
  while (size) {
    const size_t n = size_t(std::min(size, kChunk));
    if (!fa.read(x.data(), n) || !fb.read(y.data(), n) || !std::equal(x.begin(), x.begin() + n, y.begin())) return false;
    size -= n;
  }
  return true;
}

}  // namespace

namespace detail {

std::string hf_home() {
  const std::string h = os::getenv_str(env::kHfHome);
  if (!h.empty()) return h;
  const std::string xdg = os::getenv_str(env::kXdgCacheHome);
  return (xdg.empty() ? os::home_dir() + "/.cache" : xdg) + "/huggingface";
}

std::vector<std::string> auth_headers() {
  const std::string t = token();
  return t.empty() ? std::vector<std::string>{} : std::vector<std::string>{"Authorization: Bearer " + t};
}

fs::path repo_dir(const std::string& repo) {
  const size_t slash = repo.find('/');
  return to_path(cache_dir()) / to_path("models--" + repo.substr(0, slash) + "--" + repo.substr(slash + 1));
}

std::string read_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}

bool has_size(const fs::path& path, uint64_t size) {
  std::error_code ec;
  return fs::is_regular_file(path, ec) && fs::file_size(path, ec) == size && !ec;
}

bool from_cache(const std::string& repo, const std::string& revision, Repo* out) {
  if (!valid_repo(repo) || !valid_word(revision)) return false;
  const fs::path dir = repo_dir(repo);
  std::error_code ec;
  std::string commit = is_commit(revision) && fs::exists(dir / "snapshots" / revision, ec)
                           ? revision
                           : read_text(dir / "refs" / to_path(revision));
  if (!is_commit(commit) || !fs::is_directory(dir / "snapshots" / commit, ec)) return false;
  out->commit = commit;
  out->files = snapshot_files(dir / "snapshots" / commit);
  out->architectures = architectures_of(dir / "snapshots" / commit);
  out->from_cache = true;
  return true;
}

void write_ref(const Repo& r) {
  if (r.revision.empty() || r.revision == r.commit || !valid_repo(r.id) || !valid_word(r.revision) ||
      !is_commit(r.commit))
    return;
  const fs::path refs = repo_dir(r.id) / "refs";
  std::error_code ec;
  fs::create_directories(refs, ec);
  std::ofstream(refs / to_path(r.revision), std::ios::binary | std::ios::trunc) << r.commit;
}

// Places a fetched blob at its snapshot path: a link to it as huggingface_hub makes, else (Windows without the right
// to make links) a hard link, else a copy.
bool place(const fs::path& blob, const fs::path& dst, std::string* err) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  fs::remove(dst, ec);
  fs::create_symlink(fs::relative(blob, dst.parent_path(), ec), dst, ec);
  if (!ec) return true;
  ec.clear();
  fs::create_hard_link(blob, dst, ec);
  if (!ec) return true;
  ec.clear();
  fs::copy_file(blob, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) return *err = from_path(dst) + ": " + ec.message(), false;
  return true;
}

}  // namespace detail

std::string cache_dir() {
  std::string c = os::getenv_str(env::kHfHubCache);
  if (c.empty()) c = detail::hf_home() + "/hub";
#if defined(_WIN32)
  std::replace(c.begin(), c.end(), '/', '\\');  // as Windows shows paths (huggingface_hub's too)
#endif
  return c;
}

std::string cache_source() {
  if (!os::getenv_str(env::kHfHubCache).empty()) return env::kHfHubCache;
  if (!os::getenv_str(env::kHfHome).empty()) return env::kHfHome;
  if (!os::getenv_str(env::kXdgCacheHome).empty()) return env::kXdgCacheHome;
  return "default";
}

namespace {

std::mutex& endpoint_mutex() {
  static std::mutex mu;
  return mu;
}
std::string& endpoint_override() {
  static std::string ep;
  return ep;
}

}  // namespace

bool normalize_endpoint(const std::string& raw, std::string* normalized, std::string* err) {
  size_t a = 0, b = raw.size();
  while (a < b && std::isspace(static_cast<unsigned char>(raw[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(raw[b - 1]))) --b;
  std::string s = raw.substr(a, b - a);
  if (s.empty()) {
    if (normalized) normalized->clear();
    return true;
  }
  for (unsigned char c : s) {
    if (c <= 0x20 || c == 0x7f || c == '?' || c == '#' || c == '\\' || c == '@') {
      if (err) *err = "endpoint: invalid URL character";
      return false;
    }
  }
  const size_t scheme_sep = s.find("://");
  std::string scheme, rest;
  if (scheme_sep != std::string::npos) {
    scheme = detail::lower(s.substr(0, scheme_sep));
    if (scheme != "https" && scheme != "http") {
      if (err) *err = "endpoint: URL must use http:// or https://";
      return false;
    }
    rest = s.substr(scheme_sep + 3);
  } else {
    if (s.find(':') != std::string::npos) {
      // Allow host:port without scheme, reject other scheme-like prefixes (e.g. javascript:...).
      const size_t colon = s.find(':');
      const size_t slash = s.find('/');
      if (slash == std::string::npos || colon < slash) {
        const std::string port = s.substr(colon + 1, slash == std::string::npos ? std::string::npos : slash - colon - 1);
        const bool numeric = !port.empty() && std::all_of(port.begin(), port.end(),
                                                          [](unsigned char c) { return std::isdigit(c) != 0; });
        if (!numeric) {
          if (err) *err = "endpoint: URL must use http:// or https://";
          return false;
        }
      }
    }
    scheme = "https";
    rest = s;
  }
  while (!rest.empty() && rest.back() == '/') rest.pop_back();
  const size_t slash = rest.find('/');
  const std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
  if (hostport.empty() || hostport.front() == '.' || hostport.front() == ':' || hostport.find("..") != std::string::npos) {
    if (err) *err = "endpoint: missing or invalid host name";
    return false;
  }
  if (normalized) *normalized = scheme + "://" + rest;
  return true;
}

bool set_endpoint(const std::string& url, std::string* err) {
  std::string norm;
  if (!normalize_endpoint(url, &norm, err)) return false;
  std::lock_guard<std::mutex> lock(endpoint_mutex());
  endpoint_override() = std::move(norm);
  return true;
}

std::string endpoint() {
  {
    std::lock_guard<std::mutex> lock(endpoint_mutex());
    if (!endpoint_override().empty()) return endpoint_override();
  }
  std::string e = os::getenv_str(env::kHfEndpoint, "https://huggingface.co");
  while (!e.empty() && e.back() == '/') e.pop_back();
  return e.empty() ? "https://huggingface.co" : e;
}

std::string endpoint_source() {
  {
    std::lock_guard<std::mutex> lock(endpoint_mutex());
    if (!endpoint_override().empty()) return "setting";
  }
  if (!os::getenv_str(env::kHfEndpoint).empty()) return env::kHfEndpoint;
  return "default";
}

bool offline() { return os::env_bool(env::kHfHubOffline).value_or(false); }

std::string snapshot_dir(const Repo& r) { return from_path(detail::repo_dir(r.id) / "snapshots" / r.commit); }

bool in_cache(const std::string& repo, const std::string& revision, Repo* out) {
  Repo r;
  r.id = repo, r.revision = revision.empty() ? "main" : revision;
  if (!detail::from_cache(repo, r.revision, &r)) return false;
  *out = std::move(r);
  return true;
}

std::vector<Cached> cached() {
  std::vector<Cached> out;
  std::error_code ec;
  for (fs::directory_iterator it(to_path(cache_dir()), ec), end; !ec && it != end; it.increment(ec)) {
    const std::string dir = from_path(it->path().filename());
    if (!dir.starts_with("models--")) continue;
    const std::string rest = dir.substr(8);
    const size_t sep = rest.find("--");
    if (sep == std::string::npos) continue;
    const std::string repo = rest.substr(0, sep) + "/" + rest.substr(sep + 2);
    if (!detail::valid_repo(repo)) continue;
    const fs::path snaps = it->path() / "snapshots";
    const std::string main = detail::read_text(it->path() / "refs" / "main");
    struct Snapshot {
      std::string commit;
      fs::file_time_type made;
    };
    std::vector<Snapshot> found;
    std::error_code e2;
    for (fs::directory_iterator s(snaps, e2), send; !e2 && s != send; s.increment(e2)) {
      std::error_code e3;
      const std::string commit = from_path(s->path().filename());
      if (detail::is_commit(commit) && s->is_directory(e3)) found.push_back({commit, s->last_write_time(e3)});
    }
    std::sort(found.begin(), found.end(), [&](const Snapshot& a, const Snapshot& b) {
      if ((a.commit == main) != (b.commit == main)) return a.commit == main;
      return a.made > b.made;
    });
    for (const Snapshot& s : found) {
      Cached c;
      c.repo = repo, c.commit = s.commit, c.current = s.commit == main;
      c.dir = from_path(snaps / to_path(s.commit));
      c.files = snapshot_files(snaps / to_path(s.commit));
      c.architectures = architectures_of(snaps / to_path(s.commit));
      if (!c.files.empty()) out.push_back(std::move(c));
    }
  }
  std::stable_sort(out.begin(), out.end(), [](const Cached& a, const Cached& b) { return a.repo < b.repo; });
  return out;
}

bool remove_cached(const std::string& repo, const std::string& commit, const std::vector<std::string>& paths,
                   uint64_t* freed, std::string* err) {
  *freed = 0;
  if (!detail::valid_repo(repo) || !detail::is_commit(commit))
    return *err = "not a cached snapshot: " + repo + "@" + commit, false;
  const fs::path dir = detail::repo_dir(repo), snaps = dir / "snapshots", snap = snaps / commit, blobs = dir / "blobs";
  std::error_code ec;
  // A file's last name going frees its bytes; a link, or one name of a file that has others, frees none.
  auto remove_one = [&](const fs::path& p) {
    std::error_code e;
    const bool link = fs::is_symlink(fs::symlink_status(p, e));
    const uint64_t bytes = link ? 0 : fs::file_size(p, e);
    const bool last = !link && !e && fs::hard_link_count(p, e) == 1;
    if (!fs::remove(p, e) || e) return *err = from_path(p) + ": " + (e ? e.message() : "not removed"), false;
    if (last) *freed += bytes;
    return true;
  };
  // Whether an entry of any of the repository's snapshots still reads `blob`: a link to it, or a hard link.
  auto referenced = [&](const fs::path& blob) {
    std::error_code e;
    const fs::path target = fs::weakly_canonical(blob, e);
    for (fs::recursive_directory_iterator it(snaps, e), end; !e && it != end; it.increment(e)) {
      std::error_code e2;
      const fs::file_status st = it->symlink_status(e2);
      if (fs::is_symlink(st) ? fs::weakly_canonical(it->path(), e2) == target
                             : fs::is_regular_file(st) && fs::equivalent(it->path(), blob, e2))
        return true;
    }
    return false;
  };
  for (const std::string& path : paths) {
    if (!detail::safe_path(path)) return *err = "not a repository's file: " + path, false;
    const fs::path entry = snap / to_path(path);
    const fs::file_status st = fs::symlink_status(entry, ec);
    if (ec || !fs::exists(st) || fs::is_directory(st))
      return *err = repo + "@" + commit + ": " + path + " is not in the cache", false;
    if (!inside(entry.parent_path(), snap)) return *err = from_path(entry) + " is outside the cache", false;
    // The blob behind the entry: what the link names; else the blob that is the same file (a hard link); else the
    // blob the entry was copied from (place() copies where it may make neither).
    fs::path blob;
    const uint64_t size = fs::file_size(entry, ec);
    if (fs::is_symlink(st)) {
      blob = fs::weakly_canonical(entry, ec);
      std::error_code bec;
      const fs::path blobs_canon = fs::weakly_canonical(blobs, bec);
      const std::string id = from_path(blob.filename());
      std::error_code st_ec;
      if (ec || bec || blob == blobs_canon || blob.parent_path() != blobs_canon || !inside(blob, blobs) ||
          !fs::is_regular_file(fs::symlink_status(blob, st_ec)) || st_ec || (id.size() != 40 && id.size() != 64) ||
          !detail::hex_id(id))
        blob.clear();  // only delete an existing hashed regular blob directly inside blobs/
    } else {
      const bool hard = fs::hard_link_count(entry, ec) > 1;
      for (fs::directory_iterator b(blobs, ec), bend; !ec && b != bend && blob.empty(); b.increment(ec)) {
        std::error_code e2;
        const std::string id = from_path(b->path().filename());
        if ((id.size() != 40 && id.size() != 64) || !detail::hex_id(id) ||
            !fs::is_regular_file(b->symlink_status(e2)) || b->file_size(e2) != size)
          continue;
        const bool same = fs::equivalent(entry, b->path(), e2);
        if (hard ? same : !same && same_contents(entry, b->path(), size)) blob = b->path();
      }
    }
    ec.clear();
    if (!remove_one(entry)) return false;
    if (!blob.empty() && !referenced(blob) && !remove_one(blob)) return false;
    for (fs::path d = entry.parent_path(); d != snap && inside(d, snap); d = d.parent_path())
      if (!fs::is_empty(d, ec) || ec || !fs::remove(d, ec)) break;  // the directories it leaves empty
  }
  return true;
}

}  // namespace flyx::hub
