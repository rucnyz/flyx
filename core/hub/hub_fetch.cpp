// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Hugging Face Hub HTTP API queries (describe) and multi-connection resumable blob transfers (fetch).
// A file being fetched is blobs/<id>.flyx-part with a map of the pieces that came (<part>.map, a byte a piece),
// so a fetch cut short goes on where it stopped; huggingface_hub's own partial files (<id>.incomplete) are untouched.
#include "core/hub/hub_internal.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

#include "core/hub/sha256.h"
#include "core/os/http.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"

namespace flyx::hub {

namespace fs = std::filesystem;
using os::from_path;
using os::to_path;

namespace {

// 16 connections of 32 MB pieces: 734 MB/s from the Hub to a 9 Gbit/s machine where huggingface_hub fetched 430 MB/s;
// 32 connections, 928 MB/s, ask a lot of a home router for the last 25%.
constexpr uint64_t kPiece = uint64_t(32) << 20;
constexpr int kConnections = 16;
constexpr int kAttempts = 5;

struct Meter {  // bytes of all the files of one fetch, for its progress reports
  ProgressFn fn;
  std::atomic<uint64_t> done{0};
  uint64_t total = 0;
  std::string file;
  double last_ms = 0, rate = 0;
  uint64_t last_bytes = 0;
  void report(bool checking, bool force = false) {
    const double now = os::now_ms();
    if (!fn || (!force && now - last_ms < 250)) return;
    const uint64_t d = done.load();
    if (last_ms > 0 && now > last_ms) {
      const double r = double(d - std::min(d, last_bytes)) * 1000 / (now - last_ms);
      rate = rate > 0 ? 0.7 * rate + 0.3 * r : r;
    }
    last_ms = now, last_bytes = d;
    fn(Progress{file, d, total, rate, checking});
  }
};

// Other flyx processes may be using the same partial file. The owner holds this blob's lock through verification
// and publication; a waiting caller can cancel without changing what the owner has downloaded.
bool lock_blob(os::FileLock& lock, const std::atomic<bool>* stop, std::string* err) {
  for (;;) {
    if (stop && stop->load()) return *err = "stopped", false;
    const auto result = lock.try_lock(err);
    if (result != os::FileLock::Result::kBusy) return result == os::FileLock::Result::kLocked;
    os::sleep_ms(100);
  }
}

struct PartialBlob {
  PartialBlob(const fs::path& blob, uint64_t size)
      : data(fs::path(blob).concat(".flyx-part")), map(fs::path(blob).concat(".flyx-part.map")),
        bytes(size), pieces(size ? size_t((size - 1) / kPiece + 1) : 1) {}
  fs::path data, map;
  uint64_t bytes;
  size_t pieces;
  bool can_resume() const { return detail::has_size(data, bytes) && detail::has_size(map, pieces); }
  uint64_t completed(const std::vector<char>& flags) const {
    uint64_t kept = 0;
    for (size_t i = 0; i < flags.size(); ++i)
      if (flags[i]) kept += std::min(kPiece, bytes - uint64_t(i) * kPiece);
    return kept;
  }
  uint64_t remaining() const {
    if (!can_resume()) return bytes;
    std::vector<char> flags(pieces);
    std::ifstream in(map, std::ios::binary);
    return in.read(flags.data(), std::streamsize(flags.size())) ? bytes - completed(flags) : bytes;
  }
};

struct Missing {
  const RepoFile* file;
  fs::path blob;
};

// Check on the first blob we have to fetch. A concurrent download may already have reserved its disk space.
bool has_room(const Repo& r, const fs::path& blobs, const std::vector<Missing>& missing, std::string* err) {
  uint64_t need = 0;
  for (const auto& [file, blob] : missing) {
    if (detail::has_size(blob, file->size)) continue;
    const uint64_t rem = PartialBlob(blob, file->size).remaining();
    need = rem > UINT64_MAX - need ? UINT64_MAX : need + rem;
  }
  std::error_code ec;
  const auto room = fs::space(blobs, ec);
  if (!ec && room.available != static_cast<uintmax_t>(-1) && room.available < need)
    return *err = "fetching " + r.id + " needs " + detail::format_gb(need) + " in " + cache_dir() + ", which has " +
                  detail::format_gb(room.available) + " free (HF_HUB_CACHE moves the cache)", false;
  return true;
}

// The Hub answers a file's address with a redirect to its storage, a signed address: asked for once here, the
// pieces then go there directly (two hops a piece were half a small piece's time), without the token, which is
// the Hub's own.
std::string probe_direct_url(const std::string& url, const std::vector<std::string>& headers) {
  os::HttpClient probe;
  os::HttpGet g;
  g.url = url, g.headers = headers, g.ranged = true, g.first = 0, g.last = 0;
  int status = 0;
  std::string e, where;
  if (probe.get(g, [](int, const char*, size_t) { return true; }, &status, &e, &where) && status == 206 && where != url)
    return where;
  return "";
}

// Multi-connection piece download with pipelined sequential SHA-256 verification.
struct BlobTransfer {
  BlobTransfer(const RepoFile& f, std::string u, std::vector<std::string> h, size_t pieces, Meter* m,
               const std::atomic<bool>* s)
      : file(f), url(std::move(u)), headers(std::move(h)), done(pieces, 0), meter(m), stop(s) {}

  const RepoFile& file;
  std::string url;
  std::vector<std::string> headers;
  std::string direct;
  os::RandomAccessFile out, map;
  std::vector<char> done;
  Meter* meter = nullptr;
  const std::atomic<bool>* stop = nullptr;

  std::atomic<bool> direct_ok{false};
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::string first_error;
  std::string digest;

  bool cancelled() const { return stop && stop->load(); }
  bool stopped() const { return failed.load() || cancelled(); }

  void fail(const std::string& e) {
    std::lock_guard<std::mutex> lock(mu);
    if (!failed.exchange(true)) first_error = e;
    cv.notify_all();
  }

  void download_piece(os::HttpClient& client, size_t i) {
    const uint64_t a = uint64_t(i) * kPiece, b = std::min(file.size, a + kPiece) - 1;
    std::string last;
    bool ok = file.size == 0;
    for (int attempt = 0; attempt < kAttempts && !ok && !stopped(); ++attempt) {
      if (attempt > 0)
        for (int t = 0; t < (5 << attempt) && !stopped(); ++t) os::sleep_ms(200);
      const bool to_storage = direct_ok.load();
      os::HttpGet g;
      g.url = to_storage ? direct : url, g.ranged = true, g.first = a, g.last = b;
      if (!to_storage) g.headers = headers;
      uint64_t at = a;
      int status = 0;
      std::string e, werr;
      const bool whole = a == 0 && b + 1 == file.size;
      const bool got = client.get(
          g,
          [&](int code, const char* data, size_t n) {
            if (code != 206 && !(code == 200 && whole)) return true;  // an error page: not written
            if (cancelled() || at + n > b + 1 || !out.write_at(at, data, n, &werr)) return false;
            at += n;
            meter->done += n;
            return true;
          },
          &status, &e);
      if (got && (status == 206 || (status == 200 && whole)) && at == b + 1) {
        ok = true;
        break;
      }
      meter->done -= at - a;  // what this attempt counted
      if (!werr.empty()) return fail(werr);
      if (to_storage && got && (status == 401 || status == 403 || status == 410)) {  // the signature ran out
        direct_ok = false;
        --attempt;
        continue;
      }
      last = got ? file.path + ": the Hub answered " + std::to_string(status) : file.path + ": " + e;
      if (got && (status == 401 || status == 403 || status == 404 || status == 200)) break;  // no point repeating
    }
    if (!ok) return stopped() ? void() : fail(last);
    static const char one = 1;
    std::string merr;
    if (!map.write_at(i, &one, 1, &merr)) return fail(merr);
    std::lock_guard<std::mutex> lock(mu);
    done[i] = 1;
    cv.notify_all();
  }

  void run_worker() {
    os::HttpClient client;  // its connection kept from piece to piece
    for (size_t i; (i = next++) < done.size() && !stopped();) {
      {
        std::lock_guard<std::mutex> lock(mu);
        if (done[i]) continue;
      }
      download_piece(client, i);
    }
  }

  void run_hasher() {
    Sha256 h;
    std::vector<char> buf(size_t(8) << 20);
    for (size_t i = 0; i < done.size(); ++i) {
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return done[i] || failed.load() || cancelled(); });
        if (!done[i]) return;
      }
      const uint64_t a = uint64_t(i) * kPiece, e = std::min(file.size, a + kPiece);
      for (uint64_t at = a; at < e;) {
        if (stopped()) return;
        const size_t n = size_t(std::min<uint64_t>(buf.size(), e - at));
        std::string rerr;
        if (!out.read_at(at, buf.data(), n, &rerr)) return fail(rerr);
        h.update(buf.data(), n);
        at += n;
      }
    }
    digest = h.hex();
  }
};

// One file into blobs/<id>: its pieces over several connections, each written where it belongs, the hash following
// the pieces in order as they complete.
bool fetch_blob(const Repo& r, const RepoFile& f, const fs::path& blob, Meter* meter, const std::atomic<bool>* stop,
                std::string* err) {
  const PartialBlob partial(blob, f.size);
  std::error_code ec;
  if (!partial.can_resume()) fs::remove(partial.data, ec), fs::remove(partial.map, ec);  // unknown state: start over

  BlobTransfer tx(f, endpoint() + "/" + r.id + "/resolve/" + r.commit + "/" + detail::url_path(f.path),
                  detail::auth_headers(), partial.pieces, meter, stop);
  if (!tx.out.open(from_path(partial.data), f.size, err) ||
      !tx.map.open(from_path(partial.map), partial.pieces, err) ||
      !tx.map.read_at(0, tx.done.data(), partial.pieces, err))
    return false;
  const size_t missing = size_t(std::count(tx.done.begin(), tx.done.end(), 0));
  meter->done += partial.completed(tx.done);
  if (missing && f.size > kPiece && !tx.cancelled()) {
    tx.direct = probe_direct_url(tx.url, tx.headers);
    tx.direct_ok = !tx.direct.empty();
  }

  std::vector<std::thread> pool;
  for (size_t k = 0; k < std::min<size_t>(kConnections, missing); ++k) pool.emplace_back([&] { tx.run_worker(); });
  std::thread hashing;
  if (!f.sha256.empty()) hashing = std::thread([&] { tx.run_hasher(); });
  for (bool all_in = false; !all_in;) {
    {
      std::unique_lock<std::mutex> lock(tx.mu);
      all_in = tx.cv.wait_for(lock, std::chrono::milliseconds(250), [&] {
        return tx.stopped() || std::all_of(tx.done.begin(), tx.done.end(), [](char d) { return d != 0; });
      });
    }
    meter->report(false);
  }
  for (auto& t : pool) t.join();
  {
    std::lock_guard<std::mutex> lock(tx.mu);
    tx.cv.notify_all();
  }
  meter->report(true, true);
  if (hashing.joinable()) hashing.join();
  if (tx.failed) return *err = tx.first_error, false;
  if (tx.cancelled()) return *err = "stopped", false;
  if (!f.sha256.empty() && tx.digest != detail::lower(f.sha256)) {
    tx.out.close(), tx.map.close();
    fs::remove(partial.data, ec), fs::remove(partial.map, ec);
    meter->done -= f.size;
    return *err = f.path + ": what came does not match the Hub's sha256 (deleted; fetch it again)", false;
  }
  tx.out.close(), tx.map.close();
  fs::rename(partial.data, blob, ec);
  if (ec) return *err = from_path(blob) + ": " + ec.message(), false;
  fs::remove(partial.map, ec);
  return true;
}

// A field of one of the Hub's JSON answers, or `dflt` when it is absent or of another type.
std::string str_field(const nlohmann::json& o, const char* k, const std::string& dflt = "") {
  const auto it = o.find(k);
  return (it != o.end() && it->is_string()) ? it->get<std::string>() : dflt;
}
uint64_t u64_field(const nlohmann::json& o, const char* k, uint64_t dflt = 0) {
  const auto it = o.find(k);
  if (it == o.end()) return dflt;
  if (it->is_number_unsigned()) return it->get<uint64_t>();
  if (it->is_number_integer() && it->get<int64_t>() >= 0) return static_cast<uint64_t>(it->get<int64_t>());
  return dflt;
}
// "gated" is false, or the way the repository's terms are accepted ("auto", "manual").
bool gated_field(const nlohmann::json& o) {
  const auto it = o.find("gated");
  return it != o.end() && (it->is_string() || (it->is_boolean() && it->get<bool>()));
}

}  // namespace

bool describe(const std::string& repo, const std::string& revision, Repo* out, std::string* err) {
  if (!detail::valid_repo(repo)) return *err = "not a Hub repository name: " + repo, false;
  Repo r;
  r.id = repo, r.revision = revision.empty() ? "main" : revision;
  if (!detail::valid_word(r.revision)) return *err = "not a Hub revision name: " + r.revision, false;
  std::string net;
  if (!offline()) {
    os::HttpGet g;
    g.url = endpoint() + "/api/models/" + repo + "/revision/" + detail::url_path(r.revision) + "?blobs=true";
    g.headers = detail::auth_headers();
    std::string body;
    int status = 0;
    if (os::http_get_text(g, &body, &status, &net)) {
      // A repository the Hub no longer shows this machine (gone, made private, a token expired) is still served
      // from the cache when it has the revision.
      if ((status == 401 || status == 403 || status == 404) && detail::from_cache(repo, r.revision, &r)) {
        *out = std::move(r);
        return true;
      }
      if (status == 401 || status == 403)  // the Hub answers a name it does not show anyone this way too
        return *err = repo + ": not on the Hub, or gated or private: check the name; for a gated model accept its "
                             "terms on huggingface.co, then set HF_TOKEN or run `hf auth login`", false;
      if (status == 404)
        return *err = repo + (r.revision == "main" ? "" : "@" + r.revision) +
                      ": not on the Hub (a typo, or a private repository: set HF_TOKEN)", false;
      if (status == 200) {
        try {
          const auto j = nlohmann::json::parse(body);
          if (!j.is_object()) return *err = repo + ": the Hub's answer is not a JSON object", false;
          r.commit = str_field(j, "sha");
          r.info.downloads = u64_field(j, "downloads"), r.info.likes = u64_field(j, "likes");
          r.info.updated = str_field(j, "lastModified"), r.info.pipeline = str_field(j, "pipeline_tag");
          r.info.gated = gated_field(j);
          if (const auto g = j.find("gguf"); g != j.end() && g->is_object()) {
            r.info.gguf_architecture = str_field(*g, "architecture");
            r.info.gguf_context = u64_field(*g, "context_length"), r.info.gguf_parameters = u64_field(*g, "total");
          }
          if (const auto sib = j.find("siblings"); sib != j.end() && sib->is_array()) {
            for (const auto& s : *sib) {
              if (!s.is_object()) continue;
              RepoFile f;
              f.path = str_field(s, "rfilename");
              if (!detail::safe_path(f.path)) continue;
              f.size = u64_field(s, "size", 0);
              f.blob_id = str_field(s, "blobId");
              if (const auto lfs = s.find("lfs"); lfs != s.end() && lfs->is_object()) {
                f.sha256 = str_field(*lfs, "sha256");
                f.size = u64_field(*lfs, "size", f.size);
              }
              r.files.push_back(std::move(f));
            }
          }
          if (const auto cfg = j.find("config"); cfg != j.end() && cfg->is_object()) {
            if (const auto arch = cfg->find("architectures"); arch != cfg->end() && arch->is_array())
              for (const auto& a : *arch)
                if (a.is_string()) r.architectures.push_back(a.get<std::string>());
          }
          if (const auto card = j.find("cardData"); card != j.end() && card->is_object()) {
            r.info.license = str_field(*card, "license");
            if (const auto bm = card->find("base_model"); bm != card->end()) {
              if (bm->is_string()) r.base_models.push_back(bm->get<std::string>());
              else if (bm->is_array())
                for (const auto& x : *bm)
                  if (x.is_string()) r.base_models.push_back(x.get<std::string>());
            }
          }
        } catch (const std::exception& e) {
          return *err = repo + ": the Hub's answer: " + e.what(), false;
        }
        if (!detail::is_commit(r.commit)) return *err = repo + ": the Hub named no commit", false;
        *out = std::move(r);
        return true;
      }
      net = "the Hub answered " + std::to_string(status);
    }
  }
  if (detail::from_cache(repo, r.revision, &r)) {
    *out = std::move(r);
    return true;
  }
  *err = repo + ": " + (offline() ? std::string("not in the cache (") + cache_dir() + ") and HF_HUB_OFFLINE is set"
                                  : "cannot reach the Hub (" + net + ") and not in the cache (" + cache_dir() + ")");
  return false;
}

// Pulls the `cursor` parameter from a Link header entry with rel="next" (RFC 8288), or empty when absent.
std::string extract_next_cursor(const std::string& link) {
  size_t pos = 0;
  while (pos < link.size()) {
    const size_t lt = link.find('<', pos);
    if (lt == std::string::npos) break;
    const size_t gt = link.find('>', lt + 1);
    if (gt == std::string::npos) break;
    const size_t comma = link.find(',', gt + 1);
    const std::string url = link.substr(lt + 1, gt - lt - 1);
    const std::string params = link.substr(gt + 1, comma == std::string::npos ? std::string::npos : comma - gt - 1);
    pos = comma == std::string::npos ? link.size() : comma + 1;
    if (params.find("rel=\"next\"") == std::string::npos && params.find("rel=next") == std::string::npos) continue;
    const size_t q = url.find('?');
    if (q == std::string::npos) return "";
    size_t p = q + 1;
    while (p < url.size()) {
      const size_t amp = url.find('&', p);
      const std::string part = url.substr(p, amp == std::string::npos ? std::string::npos : amp - p);
      if (part.starts_with("cursor=")) return detail::url_decode(part.substr(7));
      if (amp == std::string::npos) break;
      p = amp + 1;
    }
    return "";
  }
  return "";
}

bool search(const std::string& query, const std::string& sort, int limit, std::vector<Found>* out, std::string* err,
            const std::string& cursor, std::string* next_cursor, bool gguf_only) {
  out->clear();
  if (next_cursor) next_cursor->clear();
  if (offline()) return *err = "HF_HUB_OFFLINE is set: the Hub is not asked", false;
  const char* by = sort == "downloads" ? "downloads" : sort == "likes" ? "likes" : sort == "updated" ? "lastModified"
                 : sort == "created" ? "createdAt" : "trendingScore";
  os::HttpGet g;
  // expand[]: returns GGUF architecture, timestamps, counts and gated status without full sibling file lists.
  g.url = endpoint() + "/api/models?" + (gguf_only ? "filter=gguf&" : "") + "direction=-1&sort=" + by + "&limit=" +
          std::to_string(std::clamp(limit, 1, 100)) +
          "&expand[]=gguf&expand[]=downloads&expand[]=likes&expand[]=lastModified&expand[]=createdAt"
          "&expand[]=pipeline_tag&expand[]=gated" +
          (gguf_only ? "" : "&expand[]=config") +
          (query.empty() ? "" : "&search=" + detail::url_query(query)) +
          (cursor.empty() ? "" : "&cursor=" + detail::url_query(cursor));
  g.headers = detail::auth_headers();
  g.connect_timeout_s = 10, g.stall_timeout_s = 20;
  std::string body, link, net;
  int status = 0;
  if (!os::http_get_text(g, &body, &status, &net, 64u << 20, next_cursor ? &link : nullptr))
    return *err = "cannot reach the Hub (" + net + ")", false;
  if (status != 200) return *err = "the Hub answered " + std::to_string(status), false;
  const auto j = nlohmann::json::parse(body, nullptr, false);
  if (!j.is_array()) return *err = "the Hub's answer is not a list of repositories", false;
  for (const auto& m : j) {
    if (!m.is_object()) continue;
    Found f;
    f.id = str_field(m, "id");
    if (!detail::valid_repo(f.id)) continue;
    f.downloads = u64_field(m, "downloads"), f.likes = u64_field(m, "likes");
    f.updated = str_field(m, "lastModified"), f.created = str_field(m, "createdAt");
    f.pipeline = str_field(m, "pipeline_tag"), f.gated = gated_field(m);
    if (const auto gg = m.find("gguf"); gg != m.end() && gg->is_object()) f.architecture = str_field(*gg, "architecture");
    if (f.architecture == "dflash") f.is_drafter = true;
    if (const auto cfg = m.find("config"); cfg != m.end() && cfg->is_object()) {
      if (const auto archs = cfg->find("architectures"); archs != cfg->end() && archs->is_array()) {
        std::vector<std::string> names;
        for (const auto& a : *archs)
          if (a.is_string()) names.push_back(a.get<std::string>());
        if (!drafter_kind(names).empty()) f.is_drafter = true;
      }
    }
    out->push_back(std::move(f));
  }
  if (next_cursor) *next_cursor = extract_next_cursor(link);
  return true;
}

bool fetch(const Repo& r, const std::vector<RepoFile>& files, std::vector<std::string>* paths, const ProgressFn& progress,
           const std::atomic<bool>* stop, std::string* err) {
  paths->clear();
  if (!detail::valid_repo(r.id)) return *err = "not a Hub repository name: " + r.id, false;
  if (!detail::is_commit(r.commit)) return *err = r.id + ": invalid Hub commit: " + r.commit, false;
  if (!r.revision.empty() && !detail::valid_word(r.revision))
    return *err = "not a Hub revision name: " + r.revision, false;
  if (stop && stop->load()) return *err = "stopped", false;
  const fs::path snap = to_path(snapshot_dir(r)), blobs = detail::repo_dir(r.id) / "blobs";
  Meter meter;
  meter.fn = progress;
  std::vector<Missing> missing;
  for (const RepoFile& f : files) {
    if (!detail::safe_path(f.path)) return *err = r.id + ": unsafe file path: " + f.path, false;
    const fs::path dst = snap / to_path(f.path);
    paths->push_back(from_path(dst));
    meter.total += f.size;
    if (detail::has_size(dst, f.size)) {
      meter.done += f.size;
      meter.file = f.path;
      continue;
    }
    if (r.from_cache) return *err = r.id + ": " + f.path + " is not in the cache, and the Hub is not reached", false;
    const std::string id = f.sha256.empty() ? f.blob_id : detail::lower(f.sha256);
    if (!detail::hex_id(id)) return *err = r.id + ": " + f.path + " has no id on the Hub", false;
    const fs::path blob = blobs / id;
    missing.push_back({&f, blob});
  }
  if (!missing.empty()) {
    std::error_code ec;
    fs::create_directories(blobs, ec);
    if (ec) return *err = from_path(blobs) + ": " + ec.message(), false;
  }
  bool checked_space = false;
  for (const auto& [f, blob] : missing) {
    os::FileLock lock(from_path(blob) + ".flyx-lock");
    if (!lock_blob(lock, stop, err)) return false;
    meter.file = f->path;
    const auto dst = snap / to_path(f->path);
    if (detail::has_size(dst, f->size)) {  // the owner already placed this revision's file
      meter.done += f->size;
      continue;
    }
    if (detail::has_size(blob, f->size)) meter.done += f->size;
    else {
      if (!checked_space) {
        if (!has_room(r, blobs, missing, err)) return false;
        checked_space = true;
      }
      if (!fetch_blob(r, *f, blob, &meter, stop, err)) return false;
    }
    if (!detail::place(blob, dst, err)) return false;
  }
  if (!files.empty()) meter.report(false, true);
  detail::write_ref(r);
  return true;
}

}  // namespace flyx::hub
