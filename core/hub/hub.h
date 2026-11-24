// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Models named as the Hugging Face Hub names them: "org/name:quant" for one GGUF file of a
// repository, "org/name" for a repository with one model in it (a drafter's config and safetensors, a single GGUF).
// Files land in the Hugging Face cache in huggingface_hub's own layout, so what `hf download` fetched is found here
// and the other way round; they come over several connections at once, are checked against the sha256 the Hub
// lists, and are found there again without a network.  Settings as huggingface_hub reads them: HF_ENDPOINT,
// HF_HOME / HF_HUB_CACHE, HF_TOKEN (or the token `hf auth login` saved), HF_HUB_OFFLINE.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace flyx::hub {

struct Name {
  std::string repo;                 // "org/name"
  std::string quant{};              // after ':', empty when no quantization was named
  std::string revision = "main";    // after '@'
};
// "org/name", "org/name:quant", "org/name@revision", with "hf.co/", "huggingface.co/" or "https://huggingface.co/" in
// front or not.  false: not a Hub name (a path, most likely).
bool parse_name(const std::string& s, Name* out);

struct RepoFile {
  std::string path;     // in the repository, '/' between directories
  uint64_t size = 0;
  std::string sha256;   // of the content, for files kept in LFS (the big ones); empty otherwise
  std::string blob_id;  // git's id of the file, for the others
};

// What the Hub says of a repository beside its files; nothing of it is known from the cache alone.
struct RepoInfo {
  uint64_t downloads = 0, likes = 0;
  std::string updated;             // when it last changed (ISO 8601, as the Hub writes it)
  std::string license, pipeline;   // its model card's license; the task the Hub files it under ("text-generation")
  std::string gguf_architecture;   // its GGUFs' general.architecture, as the Hub read it from one of them
  uint64_t gguf_context = 0;       // their context length
  uint64_t gguf_parameters = 0;    // their parameter count
  bool gated = false;              // its files come only after its terms are accepted
};

struct Repo {
  std::string id, revision, commit;
  std::vector<RepoFile> files;
  std::vector<std::string> base_models;  // what its model card says it was made from
  std::vector<std::string> architectures;  // its config.json's (a transformers checkpoint, a block drafter)
  RepoInfo info;
  bool from_cache = false;               // the Hub was not asked (offline) or not reached
};

// A repository's files at a revision, from the Hub; from the cache when offline, or when the Hub cannot be reached
// or no longer shows the repository and the cache has the revision.
bool describe(const std::string& repo, const std::string& revision, Repo* out, std::string* err);
// The cache's snapshot of a revision (a branch its refs/ names, or a commit), without the Hub.
bool in_cache(const std::string& repo, const std::string& revision, Repo* out);

// A repository as the Hub lists it in a search.
struct Found {
  std::string id;  // "org/name"
  uint64_t downloads = 0, likes = 0;
  std::string updated, created;  // ISO 8601
  std::string pipeline;
  std::string architecture;      // GGUF general.architecture from expand[]=gguf, empty when absent
  bool gated = false;
  bool is_drafter = false;       // a block drafter's safetensors (drafter_kind) or a GGUF dflash drafter
};
// The Hub's repositories holding GGUF files (when `gguf_only` is true) whose names match `query` (every one when it is
// empty), at most `limit`, ordered by `sort`: "trending", "downloads", "likes", "updated" or "created", the most first.
// `cursor` resumes from a previous call's *next_cursor (from the Hub's Link rel="next" header).  Only the Hub knows:
// false, with *err saying why, offline or when it cannot be reached.
bool search(const std::string& query, const std::string& sort, int limit, std::vector<Found>* out, std::string* err,
            const std::string& cursor = "", std::string* next_cursor = nullptr, bool gguf_only = true);

// The GGUF files of a repository a model can be read from, each with the quantization its name carries ("UD-IQ3_XXS"
// for Qwen3.8-27B-UD-IQ3_XXS.gguf).  Left out: projectors, importance matrices, files under MTP/; split models are
// grouped into one choice whose `parts` hold all shards in order (1..N) and `total_size` sums their bytes.
struct GgufChoice {
  RepoFile file;                 // single file, or shard 1 of a split model
  std::vector<RepoFile> parts;   // every shard in order (size 1 for unsplit, N for split)
  uint64_t total_size = 0;       // summed byte size across all parts
  std::string quant;
  bool split = false;
  bool complete = true;          // false if split and any shard 1..N is missing
};
std::vector<GgufChoice> gguf_choices(const Repo& r);
// Whether a GGUF at `path` ('/' between directories) can hold a model by its name: not a projector, an importance
// matrix or an MTP layer of its own (mtp-*.gguf, anything under an MTP directory).  Split parts count.
bool model_gguf_name(const std::string& path);
// Whether a GGUF at `path` ('/' between directories) is a vision projector (mmproj) file by name/directory.
bool mmproj_gguf_name(const std::string& path);
// Whether a GGUF at `path` ('/' between directories) is an MTP drafter file by name/directory.
bool mtp_gguf_name(const std::string& path);
// A companion GGUF in a repository (vision projector or drafter) listed alongside its main models.
struct CompanionFile {
  RepoFile file;
  std::string kind;      // "mmproj" or "mtp"
  bool is_default = false;
};
std::vector<CompanionFile> companion_files(const Repo& r);
// Parsed shard suffix of a split GGUF ("<prefix>-00001-of-00003.gguf").
struct SplitGguf {
  std::string prefix;  // path before "-00001-of-00003.gguf"
  std::string ext;     // ".gguf" (preserving case)
  int part = 0;        // 1-based shard index (1..count)
  int count = 0;       // total shard count (>= 2)
  int digits = 5;      // digit width
};
bool parse_split_gguf_name(const std::string& path, SplitGguf* out);
// Whether a file's name is one part of a split GGUF ("-00001-of-00003.gguf").
bool split_gguf_name(const std::string& file_name);
// The quantization a file's name carries ("UD-IQ3_XXS" for Qwen3.8-27B-UD-IQ3_XXS.gguf), or "" when no word of it
// names one.
std::string quant_in_name(const std::string& file_name);

// Whether the repository holds a block drafter, not a transformers model; and which, from its config's architectures
// (DFlash2DraftModel: "dflash2", DFlashDraftModel: "dflash", DSpineDraftModel: "dspine"; "" none).
bool is_drafter(const Repo& r);
std::string drafter_kind(const Repo& r);
std::string drafter_kind(const std::vector<std::string>& architectures);
// The files `name` asks for: the GGUF of its quant (or the repository's only one), else a block drafter's
// config.json and safetensors, else the repository's only .flyx file.
bool select(const Repo& r, const Name& name, std::vector<RepoFile>* files, std::string* err);
// The MTP layer in a GGUF of its own beside the models (unsloth: MTP/mtp-*.gguf), or nullptr.
const RepoFile* mtp_file(const Repo& r);
// The multimodal vision projector in a GGUF beside the models (e.g. mmproj-BF16.gguf), or nullptr.
const RepoFile* mmproj_file(const Repo& r);

struct Progress {
  std::string file;                  // the one being fetched
  uint64_t done = 0, total = 0;      // bytes of all the files asked for
  double bytes_per_s = 0;
  bool checking = false;             // hashing what came
};
using ProgressFn = std::function<void(const Progress&)>;

// The files on disk, in the cache's snapshot of r.commit, fetched when missing (all of them: `paths` in the same
// order).  `stop`, when set, ends a fetch early; what came stays for the next try.  progress: about four times a second.
bool fetch(const Repo& r, const std::vector<RepoFile>& files, std::vector<std::string>* paths,
           const ProgressFn& progress, const std::atomic<bool>* stop, std::string* err);
// Where r.commit's files are (or will be): <cache>/models--org--name/snapshots/<commit>.
std::string snapshot_dir(const Repo& r);

// The snapshots in the cache, every one of every repository: a repository's snapshot its main ref names first, then
// the others, newest first.  A file kept under the same path in several snapshots is the first one's to show.
struct Cached {
  std::string repo, commit;
  std::string dir;              // the snapshot's directory
  bool current = false;         // the snapshot refs/main names
  std::vector<RepoFile> files;  // sizes only
  std::vector<std::string> architectures;  // its config.json's, when it has one
};
std::vector<Cached> cached();

// Deletes files of a snapshot (paths in the repository): the snapshot's entries, and the blobs behind them that no
// other entry of the repository's snapshots still refers to, by a link, a hard link or as the blob a copy was made
// of.  Nothing outside the repository's directory in the cache is touched.  *freed: the bytes given back.
bool remove_cached(const std::string& repo, const std::string& commit, const std::vector<std::string>& paths,
                   uint64_t* freed, std::string* err);

std::string cache_dir();
std::string cache_source();  // "HF_HUB_CACHE" | "HF_HOME" | "XDG_CACHE_HOME" | "default"
// Normalizes and validates a Hub endpoint URL (e.g. "hf-mirror.com" -> "https://hf-mirror.com", trimming trailing
// slashes). Empty input clears any override (falling back to HF_ENDPOINT or https://huggingface.co).
bool normalize_endpoint(const std::string& raw, std::string* normalized, std::string* err);
// Sets or clears a process-wide Hub endpoint override; empty restores HF_ENDPOINT / https://huggingface.co.
bool set_endpoint(const std::string& url, std::string* err = nullptr);
std::string endpoint();
std::string endpoint_source();  // "setting" | "HF_ENDPOINT" | "default"
bool offline();

}  // namespace flyx::hub
