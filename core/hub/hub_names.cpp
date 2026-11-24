// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Hub name parsing, quantization extraction, file selection, and shared string/path validation helpers.
#include "core/hub/hub_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
#include <string_view>

namespace flyx::hub {

namespace detail {

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

namespace {
bool name_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.'; }
}  // namespace

bool valid_repo(const std::string& r) {
  const size_t slash = r.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= r.size() || r.find('/', slash + 1) != std::string::npos)
    return false;
  if (r.size() > 200 || r.find("..") != std::string::npos || r[0] == '.' || r[slash + 1] == '.') return false;
  return std::all_of(r.begin(), r.end(), [](char c) { return c == '/' || name_char(c); });
}

bool valid_word(const std::string& w) {
  return !w.empty() && w.size() <= 100 && w[0] != '.' && w.find("..") == std::string::npos &&
         std::all_of(w.begin(), w.end(), name_char);
}

bool safe_path(const std::string& p) {
  if (p.empty() || p[0] == '/' || p.find('\\') != std::string::npos || p.find(':') != std::string::npos) return false;
  size_t start = 0;
  while (start <= p.size()) {
    const size_t end = std::min(p.find('/', start), p.size());
    const std::string part = p.substr(start, end - start);
    if (part.empty() || part == "." || part == "..") return false;
    start = end + 1;
  }
  return true;
}

std::string url_path(const std::string& p) {
  static const char* hex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : p) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') out += char(c);
    else out += '%', out += hex[c >> 4], out += hex[c & 15];
  }
  return out;
}

bool hex_id(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); });
}

bool is_commit(const std::string& s) { return s.size() == 40 && hex_id(s); }

std::string base_name(const std::string& path) { return path.substr(path.rfind('/') + 1); }

std::string format_gb(uint64_t n) {
  char s[32];
  std::snprintf(s, sizeof s, "%.1f GB", double(n) / 1e9);
  return s;
}

}  // namespace detail

namespace {

constexpr std::string_view kSep = "-._";

// Whether a quantization's name starts at s ("Q4_K_M", "IQ3_XXS", "UD-Q4_K_XL", "BF16", ...), any case.
bool quant_word(std::string_view s) {
  const std::string w = detail::lower(std::string(s));
  if (w.empty()) return false;
  auto digit = [&](size_t i) { return i < w.size() && std::isdigit(static_cast<unsigned char>(w[i])); };
  if (w.starts_with("ud-")) return quant_word(s.substr(3));
  if ((w[0] == 'q' && digit(1)) || (w.starts_with("iq") && digit(2)) || (w.starts_with("tq") && digit(2))) return true;
  for (const std::string_view f : {"bf16", "f16", "f32", "fp16", "fp8", "mxfp4", "nvfp4"})
    if (w == f || (w.starts_with(f) && kSep.find(w[f.size()]) != std::string_view::npos)) return true;
  return false;
}

bool parse_split(const std::string& path, SplitGguf* out) {
  if (path.size() < 18) return false;
  const std::string ext = path.substr(path.size() - 5);
  if (detail::lower(ext) != ".gguf") return false;
  const std::string stem = path.substr(0, path.size() - 5);
  const size_t slash = stem.find_last_of("/\\");
  const size_t base_start = slash == std::string::npos ? 0 : slash + 1;
  const size_t of = stem.rfind("-of-");
  if (of == std::string::npos || of <= base_start) return false;
  const size_t count_start = of + 4;
  const size_t digits = stem.size() - count_start;
  if (digits < 5 || digits > 9 || of < base_start + digits + 1 || stem[of - digits - 1] != '-') return false;
  auto digits_val = [&](size_t from, int* v) {
    int n = 0;
    for (size_t i = 0; i < digits; ++i) {
      const char c = stem[from + i];
      if (c < '0' || c > '9') return false;
      n = n * 10 + (c - '0');
    }
    *v = n;
    return true;
  };
  int part = 0, count = 0;
  if (!digits_val(of - digits, &part) || !digits_val(count_start, &count)) return false;
  if (count < 2 || part < 1 || part > count) return false;
  if (out) {
    out->prefix = stem.substr(0, of - digits - 1);
    out->ext = ext;
    out->part = part;
    out->count = count;
    out->digits = int(digits);
  }
  return true;
}

std::string strip_model_stem(const std::string& path) {
  SplitGguf sp;
  if (parse_split(path, &sp)) return detail::base_name(sp.prefix);
  std::string stem = detail::base_name(path);
  const std::string low = detail::lower(stem);
  if (low.ends_with(".gguf") || low.ends_with(".flyx")) stem.resize(stem.size() - 5);
  return stem;
}

std::string find_quant_in_stem(const std::string& stem) {
  for (size_t i = 0; i < stem.size(); ++i)
    if ((i == 0 || kSep.find(stem[i - 1]) != std::string_view::npos) && quant_word(std::string_view(stem).substr(i)))
      return stem.substr(i);
  return "";
}

// The quantization a GGUF's file name carries: from its first word that names one ("UD-IQ3_XXS" in
// "Qwen3.8-27B-UD-IQ3_XXS"), else what follows the name the repository's files share, else the whole name.
std::string quant_label(const std::string& stem, const std::string& shared) {
  if (std::string q = find_quant_in_stem(stem); !q.empty()) return q;
  if (!shared.empty() && stem.size() > shared.size()) return stem.substr(shared.size());
  return stem;
}

}  // namespace

bool parse_name(const std::string& s, Name* out) {
  std::string t = s;
  for (const std::string_view p : {"https://huggingface.co/", "http://huggingface.co/", "huggingface.co/",
                                   "https://hf.co/", "hf.co/", "hf://"}) {
    if (t.starts_with(p)) {
      t = t.substr(p.size());
      break;
    }
  }
  Name n;
  const size_t at = t.find('@');
  if (at != std::string::npos) {
    n.revision = t.substr(at + 1);
    t.resize(at);
    if (!detail::valid_word(n.revision)) return false;
  }
  const size_t colon = t.find(':');
  if (colon != std::string::npos) {
    n.quant = t.substr(colon + 1);
    t.resize(colon);
    if (!detail::valid_word(n.quant)) return false;
  }
  if (!detail::valid_repo(t)) return false;
  n.repo = t;
  *out = n;
  return true;
}

bool mmproj_gguf_name(const std::string& path) {
  const std::string p = detail::lower(path), base = detail::base_name(p);
  if (!base.ends_with(".gguf") || base.find("imatrix") != std::string::npos || base.starts_with("mtp-") ||
      p.starts_with("mtp/") || p.find("/mtp/") != std::string::npos)
    return false;
  if (p.starts_with("mmproj/") || p.find("/mmproj/") != std::string::npos) return true;
  for (size_t pos = 0; (pos = base.find("mmproj", pos)) != std::string::npos; pos += 6) {
    const bool left_ok = (pos == 0 || kSep.find(base[pos - 1]) != std::string_view::npos);
    const size_t after = pos + 6;
    const bool right_ok = (after == base.size() || kSep.find(base[after]) != std::string_view::npos);
    if (!left_ok || !right_ok) continue;
    const std::string_view prefix(base.data(), pos);
    if (prefix.ends_with("no-") || prefix.ends_with("no_") || prefix.ends_with("without-") ||
        prefix.ends_with("without_"))
      continue;
    return true;
  }
  return false;
}

bool model_gguf_name(const std::string& path) {
  const std::string p = detail::lower(path), base = detail::base_name(p);
  if (!base.ends_with(".gguf") || mmproj_gguf_name(path) ||
      base.find("imatrix") != std::string::npos || base.starts_with("mtp-"))
    return false;
  return !p.starts_with("mtp/") && p.find("/mtp/") == std::string::npos;
}

bool parse_split_gguf_name(const std::string& path, SplitGguf* out) { return parse_split(path, out); }

bool split_gguf_name(const std::string& file_name) { return parse_split(file_name, nullptr); }

std::string quant_in_name(const std::string& file_name) { return find_quant_in_stem(strip_model_stem(file_name)); }

std::vector<GgufChoice> gguf_choices(const Repo& r) {
  std::vector<const RepoFile*> ggufs;
  for (const RepoFile& f : r.files)
    if (model_gguf_name(f.path)) ggufs.push_back(&f);
  // The name the files share, cut where a word ends ("Qwen3.8-27B-").
  std::string shared;
  if (ggufs.size() > 1) {
    shared = strip_model_stem(ggufs[0]->path);
    for (const RepoFile* f : ggufs) {
      const std::string s = strip_model_stem(f->path);
      size_t k = 0;
      while (k < shared.size() && k < s.size() && shared[k] == s[k]) ++k;
      shared.resize(k);
    }
    while (!shared.empty() && kSep.find(shared.back()) == std::string_view::npos) shared.pop_back();
  }
  struct SplitGroup {
    int count = 0;
    std::string quant;
    std::map<int, RepoFile> by_part;
  };
  std::map<std::pair<std::string, int>, SplitGroup> splits;
  std::vector<GgufChoice> out;
  for (const RepoFile* f : ggufs) {
    SplitGguf sp;
    const std::string q = quant_label(strip_model_stem(f->path), shared);
    if (parse_split(f->path, &sp)) {
      auto& g = splits[{sp.prefix, sp.count}];
      g.count = sp.count;
      g.quant = q;
      g.by_part[sp.part] = *f;
      continue;
    }
    GgufChoice c;
    c.file = *f;
    c.parts = {*f};
    c.total_size = f->size;
    c.quant = q;
    c.split = false;
    c.complete = true;
    out.push_back(std::move(c));
  }
  for (auto& [_, g] : splits) {
    GgufChoice c;
    c.quant = g.quant;
    c.split = true;
    c.complete = int(g.by_part.size()) == g.count;
    int expect = 1;
    for (auto& [part_no, rf] : g.by_part) {
      if (part_no != expect++) c.complete = false;
      c.total_size += rf.size;
      c.parts.push_back(rf);
    }
    c.file = c.parts.front();
    out.push_back(std::move(c));
  }
  std::sort(out.begin(), out.end(), [](const GgufChoice& a, const GgufChoice& b) { return a.total_size < b.total_size; });
  return out;
}

const RepoFile* mtp_file(const Repo& r) {
  for (const RepoFile& f : r.files)
    if (f.path.starts_with("MTP/") && detail::lower(f.path).ends_with(".gguf")) return &f;
  return nullptr;
}

const RepoFile* mmproj_file(const Repo& r) {
  auto rank = [](const std::string& path) -> int {
    if (!mmproj_gguf_name(path)) return -1;
    const std::string base = detail::lower(detail::base_name(path));
    if (base.find("bf16") != std::string::npos) return 5;
    if (base.find("f16") != std::string::npos || base.find("fp16") != std::string::npos) return 4;
    if (base.find("f32") != std::string::npos || base.find("fp32") != std::string::npos) return 3;
    if (base.find("q8_0") != std::string::npos || base.find("q8") != std::string::npos) return 2;
    return 1;
  };
  const RepoFile* best = nullptr;
  int best_rank = -1;
  for (const RepoFile& f : r.files) {
    const int k = rank(f.path);
    if (k < 0) continue;
    if (k > best_rank || (k == best_rank && best && f.path.size() < best->path.size())) {
      best = &f;
      best_rank = k;
    }
  }
  return best;
}

bool select(const Repo& r, const Name& name, std::vector<RepoFile>* files, std::string* err) {
  files->clear();
  const std::vector<GgufChoice> choices = gguf_choices(r);
  auto listing = [&] {
    std::string s;
    for (const GgufChoice& c : choices) {
      std::string note;
      if (c.split) note = c.complete ? ", " + std::to_string(c.parts.size()) + " parts" : ", split: incomplete";
      s += "\n  " + r.id + ":" + c.quant + "  (" + detail::format_gb(c.total_size) + note + ")";
    }
    return s;
  };
  if (!name.quant.empty()) {
    const std::string want = detail::lower(name.quant);
    const GgufChoice* pick = nullptr;
    for (const GgufChoice& c : choices)
      if (detail::lower(c.quant) == want) pick = &c;
    if (!pick)  // a tag the name ends with: "IQ3_XXS" for a lone UD-IQ3_XXS
      for (const GgufChoice& c : choices) {
        const std::string q = detail::lower(c.quant);
        if (q.ends_with(want) &&
            (q.size() == want.size() || kSep.find(q[q.size() - want.size() - 1]) != std::string_view::npos) &&
            (!pick || c.quant.size() < pick->quant.size()))
          pick = &c;
      }
    if (!pick) return *err = r.id + " has no GGUF file for \"" + name.quant + "\"; it has:" + listing(), false;
    if (pick->split && !pick->complete)
      return *err = r.id + ":" + pick->quant + " is an incomplete split GGUF (" + std::to_string(pick->parts.size()) +
                    " parts found)", false;
    *files = pick->parts;
    return true;
  }
  if (choices.size() == 1 && choices[0].complete) {
    *files = choices[0].parts;
    return true;
  }
  if (choices.size() == 1 && choices[0].split && !choices[0].complete)
    return *err = r.id + ":" + choices[0].quant + " is an incomplete split GGUF (" +
                  std::to_string(choices[0].parts.size()) + " parts found)", false;
  if (choices.size() > 1) return *err = r.id + " holds several models: name one (" + r.id + ":QUANT):" + listing(), false;
  // No GGUF: a drafter's config and safetensors, or a .flyx file.
  bool safetensors = false;
  const RepoFile* config = nullptr;
  std::vector<const RepoFile*> flyx;
  for (const RepoFile& f : r.files) {
    if (f.path == "config.json") config = &f;
    if (f.path.find('/') == std::string::npos &&
        (f.path.ends_with(".safetensors") || f.path.ends_with(".safetensors.index.json"))) {
      files->push_back(f);
      safetensors = safetensors || f.path.ends_with(".safetensors");
    }
    if (detail::lower(f.path).ends_with(".flyx")) flyx.push_back(&f);
  }
  if (safetensors && config && is_drafter(r)) {
    files->insert(files->begin(), *config);
    return true;
  }
  if (safetensors && config)
    return *err = r.id + " is a transformers checkpoint (" +
                  (r.architectures.empty() ? std::string("its config names no architecture") : r.architectures[0]) +
                  "): flyx reads GGUF files (a GGUF repository of it, org/name-GGUF:QUANT) and block drafters (DFlash2, "
                  "DFlash, DSpine)",
           false;
  files->clear();
  if (flyx.size() == 1) {
    files->push_back(*flyx[0]);
    return true;
  }
  return *err = r.id + ": no model flyx reads there (a GGUF file, a drafter's config.json and safetensors, a .flyx file)",
         false;
}

bool is_drafter(const Repo& r) { return !drafter_kind(r).empty(); }

std::string drafter_kind(const Repo& r) {
  auto has = [&](const char* a) {
    return std::find(r.architectures.begin(), r.architectures.end(), a) != r.architectures.end();
  };
  // DFlashDraftModel: DFlash (a DSpine drafter published under that name reads as DSpineDraftModel, hub_cache.cpp)
  return has("DSpineDraftModel")    ? "dspine"
         : has("DFlash2DraftModel") ? "dflash2"
         : has("DFlashDraftModel")  ? "dflash"
                                    : "";
}

}  // namespace flyx::hub
