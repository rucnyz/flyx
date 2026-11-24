// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// core/hub: SHA-256 against FIPS 180-4's examples on both of its paths; Hub names; which files a name
// picks in a repository; a cache with two snapshots of a repository (both listed, the old commit's file loaded by
// name@commit offline, files deleted with their blobs only where no other snapshot reads them); and fetching from a
// fake Hub on this machine (HF_ENDPOINT): the files in huggingface_hub's cache layout, found again without a request and offline, a fetch cut short going on with the pieces it lacks, a
// corrupted piece refused by the hash, a storage address that stops working.  Skipped (77) where libcurl is missing.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "core/hub/hub.h"
#include "core/hub/sha256.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using flyx::hub::Sha256;
using flyx::os::from_path;
using flyx::os::to_path;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  failures += !ok;
}

struct TestDirectory {
  fs::path path;
  TestDirectory() {
    const auto base = fs::canonical(fs::temp_directory_path());
    const auto prefix = "flyx-hub-test-" + std::to_string(flyx::os::process_id()) + "-";
    for (unsigned i = 0;; ++i) {
      path = base / (prefix + std::to_string(i));
      std::error_code ec;
      if (fs::create_directory(path, ec)) return;  // only a directory we created will be removed
      if (ec && ec != std::errc::file_exists) throw fs::filesystem_error("create test directory", path, ec);
    }
  }
  TestDirectory(const TestDirectory&) = delete;
  TestDirectory& operator=(const TestDirectory&) = delete;
  ~TestDirectory() {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec) std::fprintf(stderr, "could not remove test directory %s\n", from_path(path).c_str());
  }
};

void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _wputenv_s(to_path(k).c_str(), to_path(v).c_str());
#else
  if (v.empty()) unsetenv(k);
  else setenv(k, v.c_str(), 1);
#endif
}

std::string sha(const std::string& s, bool portable) {
  Sha256 h(portable);
  h.update(s.data(), s.size());
  return h.hex();
}

void hashes() {
  const std::pair<std::string, const char*> known[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
      {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}};
  for (bool portable : {true, false})
    for (const auto& [in, want] : known)
      check(sha(in, portable) == want, std::string("sha256 of ") + std::to_string(in.size()) + " bytes, " +
                                           (portable ? "plain C++" : Sha256::accelerated() ? "SHA instructions" : "default"));
  std::mt19937 rng(7);
  std::string data(3 << 20, '\0');
  for (char& c : data) c = char(rng());
  Sha256 pieces;
  for (size_t at = 0, n = 1; at < data.size(); at += n, n = n * 3 + 1) pieces.update(data.data() + at, std::min(n, data.size() - at));
  check(pieces.hex() == sha(data, true) && sha(data, false) == sha(data, true),
        "3 MB in uneven pieces, both paths: one digest");
}

void names() {
  flyx::hub::Name n;
  check(flyx::hub::parse_name("unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS", &n) && n.repo == "unsloth/Qwen3.8-27B-GGUF" &&
            n.quant == "UD-IQ3_XXS" && n.revision == "main",
        "org/name:QUANT");
  check(flyx::hub::parse_name("https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2", &n) && n.repo == "z-lab/Qwen3.8-27B-DFlash2" &&
            n.quant.empty(),
        "a huggingface.co address");
  check(flyx::hub::parse_name("hf.co/a-b/c.d_e:Q4_K_M@v1.2", &n) && n.repo == "a-b/c.d_e" && n.quant == "Q4_K_M" &&
            n.revision == "v1.2",
        "hf.co/org/name:QUANT@revision");
  for (const char* path : {"/home/me/model.gguf", "C:\\models\\x.gguf", "D:/models/x.gguf", "model.gguf", "a/b/c",
                           "../x", "org/..", "./x", "", "org/name:", "org/na me"})
    check(!flyx::hub::parse_name(path, &n), std::string("not a Hub name: \"") + path + "\"");
  // Endpoint normalization, override, source reporting, and mirror URL parsing.
  std::string norm, ep_err;
  check(flyx::hub::normalize_endpoint("hf-mirror.com/", &norm, &ep_err) && norm == "https://hf-mirror.com",
        "normalize_endpoint adds https:// and strips trailing slash");
  check(flyx::hub::normalize_endpoint("http://127.0.0.1:8080/hf/", &norm, &ep_err) &&
            norm == "http://127.0.0.1:8080/hf",
        "normalize_endpoint keeps explicit http:// and path prefix");
  check(flyx::hub::normalize_endpoint("http://[::1]:8080/", &norm, &ep_err) && norm == "http://[::1]:8080",
        "normalize_endpoint keeps IPv6 literal host");
  check(flyx::hub::normalize_endpoint("   ", &norm, &ep_err) && norm.empty(),
        "normalize_endpoint treats blank as reset");
  for (const char* bad : {"ftp://hf-mirror.com", "javascript:alert(1)", "https://hf-mirror.com?a=1",
                          "https://hf-mirror.com#x", "https://user@hf-mirror.com", "https://", "https://bad host"})
    check(!flyx::hub::normalize_endpoint(bad, &norm, &ep_err),
          std::string("normalize_endpoint rejects invalid URL: ") + bad);
  check(flyx::hub::set_endpoint("hf-mirror.com", &ep_err) && flyx::hub::endpoint() == "https://hf-mirror.com" &&
            flyx::hub::endpoint_source() == "setting",
        "set_endpoint overrides endpoint() and reports source=setting");
  check(flyx::hub::parse_name("https://hf-mirror.com/org/model-GGUF:Q4_K_M", &n) && n.repo == "org/model-GGUF" &&
            n.quant == "Q4_K_M",
        "parse_name strips configured custom mirror URL");
  check(flyx::hub::parse_name("hf-mirror.com/org/model-GGUF:Q8_0", &n) && n.repo == "org/model-GGUF" &&
            n.quant == "Q8_0",
        "parse_name strips configured custom mirror host without scheme");
  check(flyx::hub::set_endpoint("", &ep_err) && flyx::hub::endpoint_source() != "setting",
        "set_endpoint(\"\") clears custom override");
}

flyx::hub::RepoFile file(const std::string& path, uint64_t size, const std::string& sha256 = "ab") {
  return {path, size, sha256, "cd"};
}

void choices() {
  flyx::hub::Repo r;
  r.id = "unsloth/Qwen3.8-27B-GGUF";
  r.files = {file("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", 50), file("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", 5),
             file("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 2), file("Qwen3.8-27B-Q4_0.gguf", 16),
             file("Qwen3.8-27B-UD-IQ3_XXS.gguf", 11), file("Qwen3.8-27B-IQ3_XXS.gguf", 10), file("mmproj-F16.gguf", 1),
             file("imatrix_unsloth.gguf", 1), file("README.md", 1, "")};
  std::map<std::string, flyx::hub::GgufChoice> got;
  for (const auto& c : flyx::hub::gguf_choices(r)) got[c.quant] = c;
  check(got.size() == 4 && got.count("Q4_0") && got.count("UD-IQ3_XXS") && got.count("IQ3_XXS") && got["BF16"].split &&
            got["BF16"].complete && got["BF16"].parts.size() == 2 && got["BF16"].total_size == 55,
        "GGUF choices: Q4_0, UD-IQ3_XXS, IQ3_XXS and the complete 2-part split BF16 (total size 55); no projector, matrix or MTP file");
  std::vector<flyx::hub::RepoFile> files;
  std::string err;
  flyx::hub::Name n{r.id, "IQ3_XXS"};
  check(flyx::hub::select(r, n, &files, &err) && files[0].path == "Qwen3.8-27B-IQ3_XXS.gguf", ":IQ3_XXS, the exact one");
  n.quant = "ud-iq3_xxs";
  check(flyx::hub::select(r, n, &files, &err) && files[0].path == "Qwen3.8-27B-UD-IQ3_XXS.gguf", ":ud-iq3_xxs, any case");
  n.quant = "Q8_0";
  check(!flyx::hub::select(r, n, &files, &err) && err.find(":UD-IQ3_XXS") != std::string::npos,
        "a quantization it lacks: the ones it has listed");
  n.quant = "BF16";
  check(flyx::hub::select(r, n, &files, &err) && files.size() == 2 &&
            files[0].path == "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf" &&
            files[1].path == "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
        "a complete split model selects all shards in order");
  // Incomplete split model (missing part 2 of 3) is refused with an informative error.
  flyx::hub::Repo inc;
  inc.id = "org/partial-GGUF";
  inc.files = {file("model-Q8_0-00001-of-00003.gguf", 10), file("model-Q8_0-00003-of-00003.gguf", 10)};
  const auto inc_choices = flyx::hub::gguf_choices(inc);
  check(inc_choices.size() == 1 && inc_choices[0].split && !inc_choices[0].complete,
        "incomplete split GGUF is marked incomplete");
  check(!flyx::hub::select(inc, {inc.id, "Q8_0"}, &files, &err) && err.find("incomplete") != std::string::npos,
        "incomplete split model is refused");
  // Single complete split model in a repo can be selected without :QUANT.
  flyx::hub::Repo lone_split;
  lone_split.id = "org/split-only";
  lone_split.files = {file("m-BF16-00002-of-00002.gguf", 5), file("m-BF16-00001-of-00002.gguf", 50)};
  check(flyx::hub::select(lone_split, {lone_split.id}, &files, &err) && files.size() == 2 &&
            files[0].path == "m-BF16-00001-of-00002.gguf" && files[1].path == "m-BF16-00002-of-00002.gguf",
        "a lone split GGUF is selected in shard order without naming a quant");
  n.quant = "";
  check(!flyx::hub::select(r, n, &files, &err) && err.find("several") != std::string::npos, "no quantization named: which?");
  check(flyx::hub::mtp_file(r) && flyx::hub::mtp_file(r)->path == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "the MTP file");
  check(flyx::hub::mmproj_file(r) && flyx::hub::mmproj_file(r)->path == "mmproj-F16.gguf", "the mmproj file");
  {
    const auto comps = flyx::hub::companion_files(r);
    check(comps.size() == 2 && comps[0].kind == "mtp" && comps[0].is_default &&
              comps[0].file.path == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" && comps[1].kind == "mmproj" &&
              comps[1].is_default && comps[1].file.path == "mmproj-F16.gguf",
          "companion_files lists MTP and mmproj with default flags");
    check(flyx::hub::select(r, {r.id, "mmproj-F16.gguf"}, &files, &err) && files.size() == 1 &&
              files[0].path == "mmproj-F16.gguf",
          "select resolves repo:mmproj-F16.gguf companion directly");
    check(flyx::hub::select(r, {r.id, "mtp-Qwen3.8-27B-Q4_0.gguf"}, &files, &err) && files.size() == 1 &&
              files[0].path == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
          "select resolves repo:mtp-*.gguf companion by basename");
    flyx::hub::Repo df;
    df.id = "org/model-DFlash2-GGUF";
    df.info.gguf_architecture = "dflash";
    df.files = {file("Model-DFlash2-Draft-BF16.gguf", 5)};
    check(flyx::hub::select(df, {df.id, "Model-DFlash2-Draft-BF16.gguf"}, &files, &err) && files.size() == 1 &&
              files[0].path == "Model-DFlash2-Draft-BF16.gguf",
          "select resolves repo:file.gguf directly for dflash GGUF drafter repos");
  }
  // mmproj_gguf_name edge cases (subfolders, imatrix, MTP, negative qualifiers) and precision ranking.
  check(flyx::hub::mmproj_gguf_name("mmproj-BF16.gguf") && flyx::hub::mmproj_gguf_name("mmproj/vision.gguf") &&
            flyx::hub::mmproj_gguf_name("qwen-mmproj-f16.gguf") &&
            !flyx::hub::mmproj_gguf_name("imatrix_mmproj.gguf") && !flyx::hub::mmproj_gguf_name("MTP/mmproj.gguf") &&
            !flyx::hub::mmproj_gguf_name("Qwen3-no-mmproj-Q4_K_M.gguf") &&
            !flyx::hub::mmproj_gguf_name("Qwen3-without-mmproj-Q8_0.gguf"),
        "mmproj_gguf_name handles subfolders and excludes imatrix/MTP/no-mmproj");
  check(flyx::hub::model_gguf_name("Qwen3-no-mmproj-Q4_K_M.gguf") && !flyx::hub::model_gguf_name("mmproj/vision.gguf"),
        "model_gguf_name keeps no-mmproj models and excludes mmproj/ subfolders");
  {
    flyx::hub::Repo vr;
    vr.id = "org/vlm-GGUF";
    vr.files = {file("mmproj-Q8_0.gguf", 1), file("mmproj-F32.gguf", 4), file("mmproj/mmproj-F16.gguf", 2),
                file("mmproj-BF16.gguf", 2)};
    check(flyx::hub::mmproj_file(vr) && flyx::hub::mmproj_file(vr)->path == "mmproj-BF16.gguf",
          "mmproj_file prefers BF16 over F16/F32/Q8_0");
    vr.files = {file("mmproj-Q8_0.gguf", 1), file("mmproj-F32.gguf", 4), file("mmproj/mmproj-F16.gguf", 2)};
    check(flyx::hub::mmproj_file(vr) && flyx::hub::mmproj_file(vr)->path == "mmproj/mmproj-F16.gguf",
          "mmproj_file prefers F16 over F32/Q8_0 (including mmproj/ subfolder)");
    vr.files = {file("mmproj-Q8_0.gguf", 1), file("mmproj-F32.gguf", 4)};
    check(flyx::hub::mmproj_file(vr) && flyx::hub::mmproj_file(vr)->path == "mmproj-F32.gguf",
          "mmproj_file prefers F32 over Q8_0");
  }
  flyx::hub::Repo one;
  one.id = "x/y";
  one.files = {file("llama-2-7b.Q4_K_M.gguf", 4)};
  check(flyx::hub::gguf_choices(one)[0].quant == "Q4_K_M" && flyx::hub::select(one, {one.id}, &files, &err),
        "a lone GGUF, its quantization after a dot; named without one");
  flyx::hub::Repo d;
  d.id = "z-lab/Qwen3.8-27B-DFlash2";
  d.architectures = {"DFlash2DraftModel"};
  d.files = {file("config.json", 1, ""), file("model.safetensors", 3), file("assets/figure.png", 1), file("README.md", 1, "")};
  check(flyx::hub::select(d, {d.id}, &files, &err) && files.size() == 2 && files[0].path == "config.json" &&
            files[1].path == "model.safetensors",
        "a drafter's repository: its config and safetensors");
  d.id = "Qwen/Qwen3-0.6B", d.architectures = {"Qwen3ForCausalLM"};
  const bool picked = flyx::hub::select(d, {d.id}, &files, &err);
  check(!picked && err.find("transformers checkpoint") != std::string::npos,
        "a transformers checkpoint is no drafter: says what flyx reads (" + err + ")");
}

std::string read_all(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- a cache with two snapshots of one repository --------------------------------------------------------------
// As the cache holds them: the blob by its id, an entry for it in each snapshot that has it (a link, else a hard link,
// else a copy, as hub::fetch places them where links are not allowed).
void put(const fs::path& repo, const std::string& commit, const std::string& path, const std::string& content) {
  const fs::path blob = repo / "blobs" / sha(content, true), entry = repo / "snapshots" / commit / path;
  fs::create_directories(blob.parent_path());
  fs::create_directories(entry.parent_path());
  if (!fs::exists(blob)) std::ofstream(blob, std::ios::binary) << content;
  std::error_code ec;
  fs::create_symlink(fs::relative(blob, entry.parent_path()), entry, ec);
  if (ec) ec.clear(), fs::create_hard_link(blob, entry, ec);
  if (ec) fs::copy_file(blob, entry);
}

void cache_snapshots(const fs::path& tmp) {
  check(flyx::hub::quant_in_name("Qwen3.8-27B-UD-IQ2_S.gguf") == "UD-IQ2_S" && flyx::hub::quant_in_name("model.gguf").empty() &&
            flyx::hub::quant_in_name("x-Q4_K_M-00001-of-00002.gguf") == "Q4_K_M",
        "a quantization from a file's name, none where it names none");
  check(flyx::hub::model_gguf_name("a/x-Q4_0.gguf") && !flyx::hub::model_gguf_name("MTP/x.gguf") &&
            !flyx::hub::model_gguf_name("a/mtp/x.gguf") && !flyx::hub::model_gguf_name("mtp-x.gguf") &&
            !flyx::hub::model_gguf_name("mmproj-F16.gguf") && flyx::hub::split_gguf_name("x-00001-of-00002.gguf"),
        "model GGUFs by name: no MTP layers, projectors; split parts told");
  const std::string old_c = "1111111111111111111111111111111111111111", new_c = "2222222222222222222222222222222222222222";
  const fs::path repo = tmp / "models--acme--two-GGUF";
  const std::string q4(5000, 'q'), iq2(3000, 'i'), q8(9000, '8');
  put(repo, old_c, "two-Q4_K_M.gguf", q4);
  put(repo, old_c, "two-IQ2_S.gguf", iq2);  // only in the old snapshot: the repository moved on
  put(repo, new_c, "two-Q4_K_M.gguf", q4);
  put(repo, new_c, "two-Q8_0.gguf", q8);
  fs::create_directories(repo / "refs");
  std::ofstream(repo / "refs" / "main", std::ios::binary) << new_c;
  set_env("HF_HUB_CACHE", from_path(tmp));
  const auto snaps = flyx::hub::cached();
  check(snaps.size() == 2 && snaps[0].commit == new_c && snaps[0].current && snaps[1].commit == old_c &&
            !snaps[1].current && snaps[1].files.size() == 2 && snaps[0].files.size() == 2,
        "every snapshot: refs/main's first, then the old one with the file only it has");
  flyx::hub::Repo r;
  std::string err;
  set_env("HF_HUB_OFFLINE", "1");
  flyx::hub::Name n;
  std::vector<flyx::hub::RepoFile> files;
  std::vector<std::string> paths;
  const bool old_ok = flyx::hub::parse_name("acme/two-GGUF:IQ2_S@" + old_c, &n) &&
                      flyx::hub::describe(n.repo, n.revision, &r, &err) && flyx::hub::select(r, n, &files, &err) &&
                      flyx::hub::fetch(r, files, &paths, nullptr, nullptr, &err);
  check(old_ok && r.commit == old_c && read_all(to_path(paths[0])) == iq2,
        "org/name:QUANT@commit: the old snapshot's file, offline" + (old_ok ? "" : ": " + err));
  check(flyx::hub::describe("acme/two-GGUF", "main", &r, &err) && r.commit == new_c && flyx::hub::gguf_choices(r).size() == 2,
        "  and main is the new snapshot");
  set_env("HF_HUB_OFFLINE", "");
  uint64_t freed = 0;
  bool ok = flyx::hub::remove_cached("acme/two-GGUF", old_c, {"two-Q4_K_M.gguf"}, &freed, &err);
  check(ok && !fs::exists(fs::symlink_status(repo / "snapshots" / old_c / "two-Q4_K_M.gguf")) &&
            read_all(repo / "snapshots" / new_c / "two-Q4_K_M.gguf") == q4,
        "deleting a file the newer snapshot has too: the newer one still reads it" + (ok ? "" : ": " + err));
  ok = flyx::hub::remove_cached("acme/two-GGUF", old_c, {"two-IQ2_S.gguf"}, &freed, &err);
  check(ok && freed >= iq2.size() && !fs::exists(repo / "blobs" / sha(iq2, true)),
        "deleting a file only one snapshot has: its blob goes, its bytes freed (" + std::to_string(freed) + ")");  // a copy and its blob: twice
  check(!flyx::hub::remove_cached("acme/two-GGUF", new_c, {"../../../x.gguf"}, &freed, &err) &&
            !flyx::hub::remove_cached("acme/two-GGUF", new_c, {"none.gguf"}, &freed, &err),
        "nothing outside the snapshot, nothing it lacks");
  {
    // A snapshot may contain a standalone copy. Two model files can share their size and both outer megabytes.
    const fs::path copy = repo / "snapshots" / new_c / "copied.gguf";
    const std::string data((2u << 20) + 1, 'a');
    std::string other = data;
    other[1u << 20] = 'b';
    const fs::path other_blob = repo / "blobs" / sha(other, true);
    const fs::path partial = repo / "blobs" / (sha(data, true) + ".incomplete");
    std::ofstream(copy, std::ios::binary) << data;
    std::ofstream(other_blob, std::ios::binary) << other;
    ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"copied.gguf"}, &freed, &err);
    check(ok && !fs::exists(copy) && freed == data.size() && fs::exists(other_blob),
          "deleting a standalone copy keeps a different blob with identical ends");
    std::ofstream(copy, std::ios::binary) << data;
    std::ofstream(partial, std::ios::binary) << data;
    ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"copied.gguf"}, &freed, &err);
    check(ok && freed == data.size() && fs::exists(partial), "a partial download with matching bytes is kept");
    const fs::path own_blob = repo / "blobs" / sha(data, true);
    std::ofstream(copy, std::ios::binary) << data;
    std::ofstream(own_blob, std::ios::binary) << data;
    ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"copied.gguf"}, &freed, &err);
    check(ok && !fs::exists(own_blob) && fs::exists(other_blob) && freed == 2 * data.size(),
          "a standalone copy and its matching unreferenced blob are both reclaimed");
  }
  {
    // Dangling symlinks and symlinks to partial/directory entries inside blobs/ must not abort removal or delete
    // partial files when removing cached snapshot entries.
    std::error_code sec;
    const fs::path snap = repo / "snapshots" / new_c;
    const fs::path missing_blob = repo / "blobs" / std::string(64, 'e');
    const fs::path part_blob = repo / "blobs" / (std::string(64, 'f') + ".flyx-part");
    std::ofstream(part_blob, std::ios::binary) << "partial";
    const fs::path dangling = snap / "dangling.gguf";
    const fs::path sibling = snap / "sibling.gguf";
    const fs::path part_link = snap / "part_link.gguf";
    std::ofstream(sibling, std::ios::binary) << "sibling";
    fs::create_symlink(fs::path("../../blobs") / missing_blob.filename(), dangling, sec);
    if (!sec) {
      fs::create_symlink(fs::path("../../blobs") / part_blob.filename(), part_link, sec);
      err.clear();
      ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"dangling.gguf", "sibling.gguf"}, &freed, &err);
      check(ok && !fs::exists(fs::symlink_status(dangling)) && !fs::exists(fs::symlink_status(sibling)),
            "dangling symlink does not abort remove_cached or leave remaining shards: " + err);
      err.clear();
      ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"part_link.gguf"}, &freed, &err);
      check(ok && !fs::exists(fs::symlink_status(part_link)) && fs::exists(part_blob),
            "symlink to .flyx-part only removes symlink and keeps partial blob: " + err);
    }
    fs::remove(part_blob, sec);
    fs::remove(sibling, sec);
  }
  {
    // Path traversal / invalid revision validation across parse_name(), describe(), and fetch().
    flyx::hub::Name bad_n;
    check(!flyx::hub::parse_name("acme/two-GGUF@.", &bad_n) && !flyx::hub::parse_name("acme/two-GGUF@.hidden", &bad_n),
          "parse_name rejects '.' and leading-dot revisions");
    flyx::hub::Repo bad_r;
    err.clear();
    check(!flyx::hub::describe("acme/two-GGUF", "../../other", &bad_r, &err) &&
              err.find("revision") != std::string::npos,
          "describe rejects path-traversal revision: " + err);
    const fs::path escape_target = repo.parent_path() / "escaped_ref";
    std::ofstream(escape_target, std::ios::binary) << "keep";
    flyx::hub::Repo evil;
    evil.id = "acme/two-GGUF";
    evil.commit = new_c;
    evil.revision = "../../escaped_ref";
    std::vector<std::string> out_paths;
    check(!flyx::hub::fetch(evil, {}, &out_paths, nullptr, nullptr, &err) && read_all(escape_target) == "keep",
          "fetch rejects path-traversal revision without overwriting files outside cache");
    evil.revision = "main";
    check(!flyx::hub::fetch(evil, {file("../escape.gguf", 4, std::string(64, '1'))}, &out_paths, nullptr, nullptr, &err),
          "fetch rejects unsafe RepoFile path");
    evil.id = "../bad/repo";
    check(!flyx::hub::fetch(evil, {}, &out_paths, nullptr, nullptr, &err), "fetch rejects invalid repo id");
    evil.id = "acme/two-GGUF";
    evil.commit = "not-a-commit";
    check(!flyx::hub::fetch(evil, {}, &out_paths, nullptr, nullptr, &err), "fetch rejects invalid commit");
    std::error_code sec;
    fs::remove(escape_target, sec);
  }
}

// ---- a drafter's kind from a cached config ----------------------------------------------------------------------
// tencent's DSpine drafters were first published as DFlashDraftModel with a code bridge in their dflash_config: read
// as DSpine.  DFlash's own DFlashDraftModel (no bridge) is no drafter flyx reads.
void cached_drafter_kinds(const fs::path& tmp) {
  const std::string commit = "3333333333333333333333333333333333333333";
  const std::pair<const char*, const char*> cases[] = {
      {"models--acme--old-DSpine",
       R"({"architectures": ["DFlashDraftModel"], "dflash_config": {"selected_prefix_code_bridge": {"code_dim": 8}}})"},
      {"models--acme--new-DSpine", R"({"architectures": ["DSpineDraftModel"], "dflash_config": {}})"},
      {"models--acme--DFlash", R"({"architectures": ["DFlashDraftModel"], "dflash_config": {"mask_token_id": 1}})"},
  };
  for (const auto& [dir, config] : cases) {
    put(tmp / dir, commit, "config.json", config);
    fs::create_directories(tmp / dir / "refs");
    std::ofstream(tmp / dir / "refs" / "main", std::ios::binary) << commit;
  }
  set_env("HF_HUB_CACHE", from_path(tmp));
  flyx::hub::Repo old_r, new_r, dflash;
  check(flyx::hub::in_cache("acme/old-DSpine", "main", &old_r) && flyx::hub::drafter_kind(old_r) == "dspine",
        "a DSpine drafter named DFlashDraftModel, its code bridge in its config: DSpine");
  check(flyx::hub::in_cache("acme/new-DSpine", "main", &new_r) && flyx::hub::drafter_kind(new_r) == "dspine",
        "a DSpineDraftModel: DSpine");
  check(flyx::hub::in_cache("acme/DFlash", "main", &dflash) && !flyx::hub::is_drafter(dflash),
        "DFlash's DFlashDraftModel, no code bridge: no drafter flyx reads");
}

// ---- a fake Hub ------------------------------------------------------------------------------------------------
struct FakeHub {
  const std::string commit = "0123456789abcdef0123456789abcdef01234567";
  std::map<std::string, std::string> files;  // path -> content
  httplib::Server http;
  int port = 0;
  std::thread serving;
  std::atomic<int> storage{0};        // requests for file bytes
  std::atomic<bool> fail_late{false};  // pieces past the first: 404
  std::atomic<bool> corrupt{false};    // one byte of the second piece flipped
  std::atomic<int> sig{0};             // the signature the last redirect carried
  std::atomic<bool> expire_first{false};  // signature 1 stops working after the first request

  FakeHub() {
    std::mt19937 rng(11);
    std::string big(70u << 20, '\0');  // three pieces of 32 MB: 32, 32, 6
    for (char& c : big) c = char(rng());
    files["tiny-Q4_K_M.gguf"] = big;
    files["MTP/mtp-tiny-Q4_0.gguf"] = std::string(1000, 'm');
    files["config.json"] = "{}";
    http.Get("/api/models/acme/tiny-GGUF/revision/main", [this](const httplib::Request&, httplib::Response& res) {
      nlohmann::json sib = nlohmann::json::array();
      for (const auto& [path, content] : files) {
        nlohmann::json f = {{"rfilename", path}, {"size", content.size()}, {"blobId", "00112233445566778899aabbccddeeff00112233"}};
        if (path != "config.json") f["lfs"] = {{"sha256", sha(content, true)}, {"size", content.size()}};
        sib.push_back(f);
      }
      res.set_content(nlohmann::json({{"sha", commit}, {"siblings", sib}, {"cardData", {{"base_model", "Qwen/Qwen3.8-27B"}}}}).dump(),
                      "application/json");
    });
    http.Get("/api/models/acme/nullable-GGUF/revision/main", [this](const httplib::Request&, httplib::Response& res) {
      res.set_content(nlohmann::json({{"sha", commit},
                                      {"config", nullptr},
                                      {"cardData", nullptr},
                                      {"siblings", nlohmann::json::array({
                                                       {{"rfilename", "m-Q4_0.gguf"},
                                                        {"size", nullptr},
                                                        {"blobId", nullptr},
                                                        {"lfs", {{"sha256", nullptr}, {"size", 1234}}}},
                                                       "non-object",
                                                   })}})
                          .dump(),
                      "application/json");
    });
    http.Get(R"(/api/models/acme/none/revision/main)", [](const httplib::Request&, httplib::Response& res) { res.status = 404; });
    http.Get("/api/models", [this](const httplib::Request& q, httplib::Response& res) {
      const std::string cursor = q.get_param_value("cursor");
      if (cursor.empty()) {
        res.set_header("Link", "<http://127.0.0.1:" + std::to_string(port) +
                                   "/api/models?filter=gguf&cursor=page%2F2%2Btok%3D%3D>; rel=\"next\"");
        res.set_content(nlohmann::json::array({
                            {{"id", "acme/Inkling-2-Mini-GGUF"},
                             {"downloads", 9000},
                             {"likes", 120},
                             {"lastModified", "2026-11-20T00:00:00.000Z"},
                             {"gguf", {{"architecture", "inkling2_loop"}}}},
                            {{"id", "acme/Qwen3-27B-GGUF"},
                             {"downloads", 8000},
                             {"likes", 90},
                             {"gguf", {{"architecture", "qwen35"}}}},
                        })
                            .dump(),
                        "application/json");
      } else if (cursor == "page/2+tok==") {
        res.set_content(nlohmann::json::array({
                            {{"id", "acme/Third-Model-GGUF"},
                             {"downloads", 500},
                             {"likes", 10},
                             {"gguf", {{"architecture", "qwen3"}}}},
                        })
                            .dump(),
                        "application/json");
      } else {
        res.status = 400;
      }
    });
    http.Get(R"(/acme/tiny-GGUF/resolve/([0-9a-f]+)/(.*))", [this](const httplib::Request& q, httplib::Response& res) {
      res.status = 302;  // to the storage, an address relative to this host as the Hub gives
      res.set_header("Location", "/storage/" + q.matches[2].str() + "?sig=" + std::to_string(++sig));
    });
    http.Get(R"(/storage/(.*))", [this](const httplib::Request& q, httplib::Response& res) {
      ++storage;
      const auto it = files.find(q.matches[1].str());
      if (it == files.end()) return void(res.status = 404);
      unsigned long long a = 0, b = it->second.size() - 1;
      if (std::sscanf(q.get_header_value("Range").c_str(), "bytes=%llu-%llu", &a, &b) != 2) a = 0, b = it->second.size() - 1;
      if (expire_first && q.get_param_value("sig") == "1" && !(a == 0 && b == 0)) return void(res.status = 403);
      if (fail_late && a >= (32u << 20)) {  // after the first piece is under way
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return void(res.status = 404);
      }
      // The whole file: httplib answers the request's range of it (206).
      if (!corrupt) return res.set_content(it->second, "application/octet-stream");
      std::string body = it->second;
      body[(32u << 20) + 100] ^= 1;  // in the second piece
      res.set_content(body, "application/octet-stream");
    });
    port = http.bind_to_any_port("127.0.0.1");
    serving = std::thread([this] { http.listen_after_bind(); });
    http.wait_until_ready();
  }
  ~FakeHub() {
    http.stop();
    serving.join();
  }
};

bool fetch_q4(const std::string& cache, std::string* path, std::string* err) {
  set_env("HF_HUB_CACHE", cache);
  flyx::hub::Repo r;
  std::vector<flyx::hub::RepoFile> files;
  std::vector<std::string> paths;
  if (!flyx::hub::describe("acme/tiny-GGUF", "main", &r, err) || !flyx::hub::select(r, {r.id, "Q4_K_M"}, &files, err) ||
      !flyx::hub::fetch(r, files, &paths, nullptr, nullptr, err))
    return false;
  *path = paths[0];
  return true;
}

void fetching(const fs::path& tmp) {
  FakeHub hub;
  set_env("HF_ENDPOINT", "http://127.0.0.1:" + std::to_string(hub.port));
  set_env("HF_TOKEN", "");
  set_env("HF_HUB_OFFLINE", "");
  set_env("HF_HOME", from_path(tmp / "home"));
  std::string path, err;
  const std::string cache = from_path(tmp / "a");
  bool ok = fetch_q4(cache, &path, &err);
  check(ok && read_all(to_path(path)) == hub.files["tiny-Q4_K_M.gguf"],
        "a 70 MB file in three pieces, as the Hub holds it" + (ok ? "" : ": " + err));
  const fs::path repo = tmp / "a" / "models--acme--tiny-GGUF";
  const fs::path blob = repo / "blobs" / sha(hub.files["tiny-Q4_K_M.gguf"], true);
  std::error_code ec;
  const auto snapshot = to_path(path);
  const bool linked = fs::equivalent(snapshot, blob, ec);
  const bool copied = !linked && !ec && !fs::is_symlink(snapshot) && read_all(blob) == hub.files["tiny-Q4_K_M.gguf"];
  check(read_all(repo / "refs" / "main") == hub.commit &&
            snapshot == repo / "snapshots" / hub.commit / "tiny-Q4_K_M.gguf" && (linked || copied),
        "huggingface_hub's layout: refs/main, the blob by its sha256, the same file or a copy in the snapshot");
  check(hub.storage == 4, "one request for where it is, then three for the pieces: " + std::to_string(hub.storage.load()));
  int before = hub.storage;
  check(fetch_q4(cache, &path, &err) && hub.storage == before, "again: from the cache, no file requested");
  {
    flyx::hub::Repo next;
    next.id = "acme/tiny-GGUF";
    next.commit = std::string(40, 'a');
    const auto shared = file("tiny-Q4_K_M.gguf", hub.files["tiny-Q4_K_M.gguf"].size(), from_path(blob.filename()));
    std::vector<std::string> linked_paths;
    flyx::hub::Progress last;
    check(flyx::hub::fetch(next, {shared}, &linked_paths, [&](const auto& p) { last = p; }, nullptr, &err) &&
              last.total == shared.size && last.done == last.total && last.file == shared.path && hub.storage == before,
          "another revision reuses a cached blob and reports all of its bytes");
    int reports = 0;
    auto hit = [&](const flyx::hub::Progress& p) { ++reports; last = p; };
    check(flyx::hub::fetch(next, {shared}, &linked_paths, hit, nullptr, &err) && reports == 1 &&
              last.file == shared.path && last.total == shared.size && last.done == last.total &&
              !last.checking && last.bytes_per_s == 0 && hub.storage == before,
          "an existing snapshot reports completion once without a transfer");
    fs::remove(to_path(flyx::hub::snapshot_dir(next)) / shared.path);
    fs::remove(to_path(flyx::hub::snapshot_dir(next)));
  }
  set_env("HF_HUB_OFFLINE", "1");
  flyx::hub::Repo r;
  std::vector<std::string> paths;
  check(flyx::hub::describe("acme/tiny-GGUF", "main", &r, &err) && r.from_cache && r.commit == hub.commit &&
            flyx::hub::gguf_choices(r).size() == 1 && flyx::hub::gguf_choices(r)[0].quant == "Q4_K_M",
        "offline: the repository from the cache, its file by its own name");
  check(!flyx::hub::describe("acme/other", "main", &r, &err) && err.find("HF_HUB_OFFLINE") != std::string::npos,
        "offline, a repository not in the cache: says so");
  set_env("HF_HUB_OFFLINE", "");
  ok = flyx::hub::describe("acme/none", "main", &r, &err);
  check(!ok && err.find("not on the Hub") != std::string::npos, "a repository the Hub does not have: " + err);

  const std::string cut = from_path(tmp / "b");
  hub.fail_late = true;
  ok = fetch_q4(cut, &path, &err);
  check(!ok, "the pieces after the first fail: the fetch fails (" + err + ")");
  hub.fail_late = false;
  before = hub.storage;
  check(fetch_q4(cut, &path, &err) && read_all(to_path(path)) == hub.files["tiny-Q4_K_M.gguf"] &&
            hub.storage - before == 3,
        "again: on from where it stopped, the two pieces it lacked (" + std::to_string(hub.storage - before - 1) + ")");

  {
    const fs::path resume = tmp / "cancel";
    set_env("HF_HUB_CACHE", from_path(resume));
    flyx::hub::Repo request;
    request.id = "acme/tiny-GGUF";
    request.commit = hub.commit;
    const auto model = file("tiny-Q4_K_M.gguf", hub.files["tiny-Q4_K_M.gguf"].size(), from_path(blob.filename()));
    std::atomic<bool> stop{true};
    before = hub.storage;
    check(!flyx::hub::fetch(request, {model}, &paths, nullptr, &stop, &err) && err == "stopped" &&
              hub.storage == before && !fs::exists(resume), "a cancelled fetch makes no requests or cache files");
    stop = false;
    auto cancel_check = [&](const flyx::hub::Progress& p) { if (p.checking) stop = true; };
    check(!flyx::hub::fetch(request, {model}, &paths, cancel_check, &stop, &err) && err == "stopped",
          "a cancellation during verification preserves the completed pieces");
    stop = false;
    before = hub.storage;
    check(flyx::hub::fetch(request, {model}, &paths, nullptr, &stop, &err) && hub.storage == before &&
              read_all(to_path(paths[0])) == hub.files[model.path],
          "verified locally on retry, without requesting the storage URL");
    const std::string mtp = "MTP/mtp-tiny-Q4_0.gguf";
    const auto layer = file(mtp, hub.files[mtp].size(), sha(hub.files[mtp], true));
    const auto partial = resume / "models--acme--tiny-GGUF" / "blobs" / (layer.sha256 + ".flyx-part");
    std::ofstream(partial, std::ios::binary) << std::string(layer.size, 'x');
    std::ofstream(fs::path(partial).concat(".map"), std::ios::binary).put(1).put(1);  // two flags for a one-piece file
    before = hub.storage;
    check(flyx::hub::fetch(request, {layer}, &paths, nullptr, nullptr, &err) && hub.storage == before + 1 &&
              read_all(to_path(paths[0])) == hub.files[mtp],
          "a wrong-sized bitmap counts no saved pieces and is fetched again");
  }

  hub.corrupt = true;
  check(!fetch_q4(from_path(tmp / "c"), &path, &err) && err.find("sha256") != std::string::npos,
        "a piece changed on the way: refused by the hash");
  hub.corrupt = false;
  check(!fs::exists(fs::path(tmp / "c" / "models--acme--tiny-GGUF" / "blobs" /
                             (sha(hub.files["tiny-Q4_K_M.gguf"], true) + ".flyx-part"))),
        "  and what came is deleted");
  hub.sig = 0;
  hub.expire_first = true;
  ok = fetch_q4(from_path(tmp / "d"), &path, &err);
  check(ok && read_all(to_path(path)) == hub.files["tiny-Q4_K_M.gguf"],
        "the storage address stops working: the pieces go back to the Hub's address" + (ok ? "" : ": " + err));
  hub.expire_first = false;
  set_env("HF_HUB_CACHE", from_path(tmp / "a"));
  fs::create_directories(tmp / "a" / "models----bad" / "snapshots" / hub.commit);
  const auto cached = flyx::hub::cached();
  check(cached.size() == 1 && cached[0].repo == "acme/tiny-GGUF" && cached[0].files.size() == 1 &&
            cached[0].files[0].path == "tiny-Q4_K_M.gguf" && cached[0].current,
        "the cache's models: the repository and its file (malformed dirs skipped)");
  flyx::hub::Repo nullable;
  check(flyx::hub::describe("acme/nullable-GGUF", "main", &nullable, &err) && nullable.commit == hub.commit &&
            nullable.files.size() == 1 && nullable.files[0].path == "m-Q4_0.gguf" && nullable.files[0].size == 1234,
        "describe accepts null config/cardData/sibling fields: " + err);
  {
    std::vector<flyx::hub::Found> page1, page2;
    std::string next1, next2;
    const bool ok1 = flyx::hub::search("", "trending", 20, &page1, &err, "", &next1);
    check(ok1 && page1.size() == 2 && page1[0].id == "acme/Inkling-2-Mini-GGUF" &&
              page1[0].architecture == "inkling2_loop" && page1[1].architecture == "qwen35" && next1 == "page/2+tok==",
          "search parses gguf.architecture and Link rel=next cursor: " + err);
    const bool ok2 = flyx::hub::search("", "trending", 20, &page2, &err, next1, &next2);
    check(ok2 && page2.size() == 1 && page2[0].id == "acme/Third-Model-GGUF" && page2[0].architecture == "qwen3" &&
              next2.empty(),
          "search round-trips percent-encoded cursor to page 2: " + err);
  }
  {
    // Sum of missing file sizes overflowing uint64_t must saturate and be rejected by has_room() before any download.
    set_env("HF_HUB_CACHE", from_path(tmp / "overflow_room"));
    flyx::hub::Repo huge;
    huge.id = "acme/tiny-GGUF";
    huge.commit = hub.commit;
    const uint64_t half_plus = (UINT64_MAX / 2) + 1024;
    const auto f1 = file("huge1.gguf", half_plus, std::string(64, 'a'));
    const auto f2 = file("huge2.gguf", half_plus, std::string(64, 'b'));
    before = hub.storage;
    err.clear();
    check(!flyx::hub::fetch(huge, {f1, f2}, &paths, nullptr, nullptr, &err) &&
              err.find("free") != std::string::npos && hub.storage == before,
          "has_room saturates uint64_t sum instead of wrapping around: " + err);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 7 && std::string(argv[1]) == "--fetch") {  // child of check_concurrent_fetch.py
    flyx::hub::Repo r;
    r.id = "acme/parallel";
    r.commit = argv[2];
    flyx::hub::RepoFile f;
    f.path = argv[3];
    f.size = std::stoull(argv[5]);
    const std::string id = argv[4];
    if (id.size() == 64) f.sha256 = id;
    else f.blob_id = id;
    std::atomic<bool> stop{false};
    std::thread timer;
    const int cancel_ms = std::atoi(argv[6]);
    if (cancel_ms > 0) timer = std::thread([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(cancel_ms));
      stop = true;
    });
    std::vector<std::string> paths;
    std::string err;
    auto progress = [&](const flyx::hub::Progress& p) {
      if (cancel_ms == -1 && p.checking) stop = true;
    };
    const bool ok = flyx::hub::fetch(r, {f}, &paths, progress, &stop, &err);
    if (timer.joinable()) timer.join();
    if (!ok) std::fprintf(stderr, "%s\n", err.c_str());
    return ok ? 0 : 1;
  }
  hashes();
  names();
  choices();
  const TestDirectory tmp;
  cache_snapshots(tmp.path / "snapshots");
  cached_drafter_kinds(tmp.path / "kinds");
#if !defined(_WIN32)
  void* curl = dlopen("libcurl.so.4", RTLD_NOW);
  if (!curl) curl = dlopen("libcurl.4.dylib", RTLD_NOW);
  if (!curl) {
    std::printf("skip: no libcurl here, fetching not tested\n");
    return failures ? 1 : 77;
  }
#endif
  fetching(tmp.path / "fetch");
  std::printf("%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
