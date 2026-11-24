// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/model_host.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "apps/common/defaults.h"
#include "core/interfaces/options.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace flyx::server {

namespace {

namespace fs = std::filesystem;

using os::to_path;

ojson want_json(const ModelHost::Want& w) {
  return {{"model", w.model},
          {"drafters", w.drafters},
          {"ctx", w.eo.ctx_given ? (w.eo.ctx_max ? ojson("max") : ojson(w.eo.max_ctx)) : ojson(nullptr)},
          {"drafter", w.eo.drafter},
          {"speculative", w.eo.speculative},
          {"kv", w.eo.kv == FLYX_KV_FP8 ? ojson("fp8") : w.eo.kv == FLYX_KV_16 ? ojson("16") : ojson(nullptr)},
          {"kv_ram", w.eo.kv_ram < 0 ? ojson("auto") : ojson(double(w.eo.kv_ram) / 1e9)},
          {"kv_gpu", w.eo.kv_gpu < 0 ? ojson(nullptr) : ojson(w.eo.kv_gpu)},
          {"width", w.eo.width < 0 ? ojson(nullptr) : ojson(w.eo.width)},
          {"draft_trees", w.eo.draft_trees},
          {"lookup", w.eo.lookup},
          {"draft_head", w.draft_head.empty() ? ojson(nullptr) : ojson(w.draft_head)},
          {"draft_weights", w.draft_weights.empty() ? ojson(nullptr) : ojson(w.draft_weights)},
          {"mmproj", w.eo.mmproj_file.empty() ? ojson(nullptr) : ojson(w.eo.mmproj_file)},
          {"vision", w.eo.vision},
          {"vision_device", w.eo.vision_device.empty() ? ojson("disk") : ojson(w.eo.vision_device)},
          {"vision_quant", w.eo.vision_quant.empty() ? ojson("q8") : ojson(w.eo.vision_quant)},
          {"vision_max_tokens", w.eo.vision_max_tokens > 0 ? ojson(w.eo.vision_max_tokens) : ojson(nullptr)},
          {"name", w.name}};
}

ojson read_settings() {
  std::ifstream f(to_path(ModelHost::settings_path()), std::ios::binary);
  if (!f) return nullptr;
  try {
    return ojson::parse(f);
  } catch (const std::exception&) {
    return nullptr;
  }
}

// A model's requests end, and its memory comes back once the last of them lets go of it.
void retire(std::shared_ptr<Scheduler> old) {
  if (!old) return;
  old->close();
  std::weak_ptr<Scheduler> gone = old;
  old.reset();
  for (int i = 0; !gone.expired(); ++i) {
    if (i == 100) std::fprintf(stderr, "flyx-server: waiting for the last model's requests to end\n");
    os::sleep_ms(50);
  }
}

ojson drafters_json(const std::vector<apps::DrafterSource>& ds, const char* where) {
  ojson out = ojson::array();
  for (const apps::DrafterSource& d : ds)
    out.push_back({{"kind", d.kind}, {where, d.from}, {"size", d.size ? ojson(*d.size) : ojson(nullptr)}});
  return out;
}

void save_settings(const ModelHost::Want& w) {
  const fs::path p = to_path(ModelHost::settings_path());
  if (p.empty()) return;
  // A local model may be relaunched from another directory. Keep Hub names as requested
  // so their revision and quantization still go through the normal fetch path.
  auto remember_path = [](const std::string& spec) {
    const fs::path path = to_path(spec);
    std::error_code ec;
    if (path.is_absolute() || !fs::exists(path, ec)) return spec;
    const auto full = fs::absolute(path, ec);
    return ec ? spec : os::from_path(full);
  };
  ojson last = want_json(w);
  last["model"] = remember_path(w.model);
  for (size_t i = 0; i < w.drafters.size(); ++i) last["drafters"][i] = remember_path(w.drafters[i]);
  if (!w.eo.mmproj_file.empty()) last["mmproj"] = remember_path(w.eo.mmproj_file);
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  fs::path tmp = p;
  tmp += "." + std::to_string(os::process_id()) + ".tmp";
  std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
  f << ojson({{"last", std::move(last)}}).dump(2) << "\n";
  f.close();
  if (f) fs::rename(tmp, p, ec);
  if (!f || ec) {
    std::fprintf(stderr, "flyx-server: could not save the last model to %s\n", ModelHost::settings_path().c_str());
    fs::remove(tmp, ec);
  }
}

}  // namespace

bool want_from_json(const ojson& j, const EngineOptions& base, ModelHost::Want* w, std::string* err) {
  if (!j.is_object() || !j.contains("model") || !j["model"].is_string() || j["model"].get<std::string>().empty())
    return *err = "\"model\": a path or a Hugging Face name (org/name:QUANT) is needed", false;
  ModelHost::Want out;
  out.model = j["model"].get<std::string>();
  out.eo = base;
  out.eo.drafter_files.clear();
  if (j.contains("drafters") && !j["drafters"].is_null()) {
    if (!j["drafters"].is_array()) return *err = "\"drafters\": a list of paths or Hub names", false;
    for (const auto& d : j["drafters"]) {
      if (!d.is_string()) return *err = "\"drafters\": a list of paths or Hub names", false;
      if (!d.get<std::string>().empty()) out.drafters.push_back(d.get<std::string>());
    }
  }
  if (j.contains("ctx")) {
    if (j["ctx"].is_null()) {
      out.eo.ctx_max = true;
      out.eo.ctx_given = false;
    } else if (j["ctx"] == "max") {  // the model's own, or the longest the GPU and host memory hold
      out.eo.ctx_max = out.eo.ctx_given = true;
    } else if (!j["ctx"].is_number_integer() || j["ctx"].get<int64_t>() < 0 ||
               j["ctx"].get<int64_t>() > apps::kMaxCtx) {
      return *err = "\"ctx\": tokens, a whole number, or \"max\"", false;
    } else if (j["ctx"].get<int>() > 0) {
      out.eo.ctx_max = false;
      out.eo.max_ctx = j["ctx"].get<int>();
      out.eo.ctx_given = true;
    } else {
      out.eo.ctx_max = true;
      out.eo.ctx_given = false;
    }
  }
  if (j.contains("drafter") && !j["drafter"].is_null()) {
    if (!j["drafter"].is_string()) return *err = "\"drafter\": mtp, dflash2, dspine, dv, all or none", false;
    const std::string d = j["drafter"].get<std::string>();
    interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
    if (!interfaces::parse_drafter_mode(d, &mode))
      return *err = "\"drafter\": mtp, dflash2, dspine, dv, all or none", false;
    out.eo.drafter = d;
  }
  if (j.contains("speculative") && !j["speculative"].is_null()) {
    if (!j["speculative"].is_boolean()) return *err = "\"speculative\": true or false", false;
    out.eo.speculative = j["speculative"].get<bool>();
  }
  if (j.contains("kv") && !j["kv"].is_null()) {
    if (!j["kv"].is_string()) return *err = "\"kv\": fp8, 16 or auto", false;
    const std::string kv = j["kv"].get<std::string>();
    if (!kv.empty()) {  // "auto": the engine's default (Engine::load), whatever the server's --kv
      if (kv != "fp8" && kv != "16" && kv != "auto") return *err = "\"kv\": fp8, 16 or auto", false;
      out.eo.kv = kv == "fp8" ? FLYX_KV_FP8 : kv == "16" ? FLYX_KV_16 : FLYX_KV_DEFAULT;
    }
  }
  // Host memory for the K/V past the GPU's part: GB, or "auto"; absent: the server's own setting.
  if (j.contains("kv_ram") && !j["kv_ram"].is_null()) {
    if (j["kv_ram"].is_string() && j["kv_ram"].get<std::string>() == "auto") {
      out.eo.kv_ram = -1;
    } else {
      if (!j["kv_ram"].is_number() || j["kv_ram"].get<double>() < 0 || j["kv_ram"].get<double>() > apps::kMaxKvRamGb)
        return *err = "\"kv_ram\": GB of host memory, or \"auto\"", false;
      out.eo.kv_ram = int64_t(j["kv_ram"].get<double>() * 1e9);
    }
  }
  if (j.contains("kv_gpu") && !j["kv_gpu"].is_null()) {
    if (!j["kv_gpu"].is_number_integer() || j["kv_gpu"].get<int64_t>() < 0 || j["kv_gpu"].get<int64_t>() > INT32_MAX)
      return *err = "\"kv_gpu\": positions, a whole number", false;
    out.eo.kv_gpu = j["kv_gpu"].get<int>();
  }
  if (j.contains("name") && !j["name"].is_null()) {
    if (!j["name"].is_string()) return *err = "\"name\": model id string", false;
    out.name = j["name"].get<std::string>();
  }
  bool mmproj_given = false;
  if (j.contains("mmproj")) {
    if (j["mmproj"].is_null()) {
      out.eo.mmproj_file.clear();
    } else {
      if (!j["mmproj"].is_string()) return *err = "\"mmproj\": path to vision projector GGUF", false;
      out.eo.mmproj_file = j["mmproj"].get<std::string>();
      mmproj_given = !out.eo.mmproj_file.empty();
    }
  }
  const bool vision_given = j.contains("vision") && !j["vision"].is_null();
  if (vision_given) {
    if (!j["vision"].is_boolean()) return *err = "\"vision\": true or false", false;
    out.eo.vision = j["vision"].get<bool>();
  } else if (mmproj_given) {
    out.eo.vision = true;
  }
  if (!out.eo.vision) out.eo.mmproj_file.clear();
  if (j.contains("vision_device") && !j["vision_device"].is_null()) {
    if (!j["vision_device"].is_string()) return *err = "\"vision_device\": disk, host, gpu or auto", false;
    const std::string vd = j["vision_device"].get<std::string>();
    if (!interfaces::parse_vision_device(vd, nullptr, &out.eo.vision_device, err)) {
      *err = "\"vision_device\": disk, host, gpu or auto";
      return false;
    }
  }
  if (j.contains("vision_quant") && !j["vision_quant"].is_null()) {
    if (!j["vision_quant"].is_string()) return *err = "\"vision_quant\": q8, exact or nvfp4", false;
    const std::string vq = j["vision_quant"].get<std::string>();
    if (!interfaces::parse_vision_quant(vq, nullptr, &out.eo.vision_quant, err)) {
      *err = "\"vision_quant\": q8, exact or nvfp4";
      return false;
    }
  }
  if (j.contains("vision_max_tokens")) {
    if (j["vision_max_tokens"].is_null()) {
      out.eo.vision_max_tokens = 0;
    } else if (!j["vision_max_tokens"].is_number_integer()) {
      return *err = "\"vision_max_tokens\": whole number >= 16, or null for auto", false;
    } else {
      const int64_t n = j["vision_max_tokens"].get<int64_t>();
      if (n != 0 && (n < 16 || n > apps::kMaxCtx))
        return *err = "\"vision_max_tokens\": 16 to " + std::to_string(apps::kMaxCtx) + ", or null for auto", false;
      out.eo.vision_max_tokens = int(n);
    }
  }
  // Drafting, as --width, --draft-trees and --no-lookup set it; absent or null: the server's own setting.
  if (j.contains("width") && !j["width"].is_null()) {
    if (!j["width"].is_number_integer() || j["width"].get<int64_t>() < 0 ||
        j["width"].get<int64_t>() > apps::kMaxDraftWidth)
      return *err = "\"width\": drafts verified a round, 0 to 64, or null to choose each round", false;
    out.eo.width = j["width"].get<int>();
  }
  if (j.contains("draft_trees") && !j["draft_trees"].is_null()) {
    if (!j["draft_trees"].is_boolean()) return *err = "\"draft_trees\": true or false", false;
    out.eo.draft_trees = j["draft_trees"].get<bool>();
    out.eo.draft_trees_given = true;
  }
  if (j.contains("lookup") && !j["lookup"].is_null()) {
    if (!j["lookup"].is_boolean()) return *err = "\"lookup\": true or false", false;
    out.eo.lookup = j["lookup"].get<bool>();
    out.eo.lookup_given = true;
  }
  // The drafters' CUDA backend options (--backend-opt draft_head=..., draft_weights=...), after the command line's.
  // Skip appending CUDA-only options when running on cpu_ref, which rejects unknown backend options.
  auto drafting_option = [&](const char* key, const char* a, const char* b, std::string* to) {
    if (!j.contains(key) || j[key].is_null()) return true;
    const std::string v = j[key].is_string() ? j[key].get<std::string>() : "";
    if (v != a && v != b) return *err = std::string("\"") + key + "\": " + a + " or " + b, false;
    *to = v;
    if (out.eo.backend != "cpu_ref")
      out.eo.backend_options += (out.eo.backend_options.empty() ? "" : ",") + std::string(key) + "=" + v;
    return true;
  };
  if (!drafting_option("draft_head", "pick", "full", &out.draft_head)) return false;
  if (!drafting_option("draft_weights", "nvfp4", "exact", &out.draft_weights)) return false;
  *w = std::move(out);
  return true;
}

std::string ModelHost::settings_path() {
  const std::string base = os::config_dir();
  return base.empty() ? "" : base + "/flyx/serve.json";
}

bool ModelHost::remembered(Want* w, const EngineOptions& base) {
  const ojson s = read_settings();
  std::string err;
  return s.is_object() && s.contains("last") && want_from_json(s["last"], base, w, &err);
}

ModelHost::~ModelHost() {
  stop_ = true;
  if (worker_.joinable()) worker_.join();
}

ModelHost::Served ModelHost::served() const {
  std::lock_guard<std::mutex> lock(mu_);
  return served_;
}

void ModelHost::cancel() { stop_ = true; }

bool ModelHost::try_begin_locked(std::string* err) {
  if (busy_) return *err = "a model is being loaded (" + target_ + ")", false;
  busy_ = true;
  return true;
}

void ModelHost::reset_load_state_locked(const std::string& state, const std::string& target) {
  state_ = state;
  target_ = target;
  error_.clear();
  fallback_.clear();
  notes_.clear();
  progress_ = {};
  since_ms_ = os::now_ms();
}

bool ModelHost::unload(std::string* err) {
  std::shared_ptr<Scheduler> old;
  std::string name;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!try_begin_locked(err)) return false;
    old = std::move(served_.sched);
    name = served_.name;
    served_ = {}, fetched_ = {};
    reset_load_state_locked("none", "");
  }
  BusyGuard guard(this);
  if (old) std::fprintf(stderr, "flyx-server: unloading %s\n", name.c_str());
  retire(std::move(old));
  return true;
}

bool ModelHost::with_files_in_use(const std::function<void(const std::vector<std::string>&)>& fn, std::string* err) {
  std::vector<std::string> in_use;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!try_begin_locked(err)) return false;
    if (served_.sched) {
      in_use.push_back(fetched_.model);
      in_use.insert(in_use.end(), fetched_.drafters.begin(), fetched_.drafters.end());
      if (!fetched_.mmproj.empty()) in_use.push_back(fetched_.mmproj);
    }
  }
  BusyGuard guard(this);
  fn(in_use);
  return true;
}

std::shared_ptr<Scheduler> ModelHost::open(const apps::Fetched& f, const EngineOptions& options, double* load_s,
                                           std::string* err) const {
  EngineOptions eo = options;
  eo.drafter_files = f.drafters;
  eo.mmproj_file = f.mmproj;
  std::fprintf(stderr, "flyx-server: loading %s\n", f.model.c_str());
  const double t0 = os::now_ms();
  std::shared_ptr<Scheduler> sched;
  try {
    sched = std::make_shared<Scheduler>(f.model, eo, so_);
  } catch (const std::exception& e) {
    // A model file no check foresaw: a load that failed with its reason, not the end of the server.
    *err = e.what();
    std::fprintf(stderr, "flyx-server: %s: %s\n", f.model.c_str(), err->c_str());
    return nullptr;
  }
  if (!sched->ok(err)) {
    std::fprintf(stderr, "flyx-server: %s: %s\n", f.model.c_str(), err->c_str());
    return nullptr;
  }
  *load_s = (os::now_ms() - t0) / 1000;
  return sched;
}

bool ModelHost::start(const Want& w, std::string* err) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!try_begin_locked(err)) return false;
  stop_ = false;
  reset_load_state_locked("fetching", w.model);
  if (worker_.joinable()) worker_.join();  // the last one is done: it let go of busy_ as it ended
  worker_ = std::thread([this, w] {
    BusyGuard guard(this);
    std::string e;
    try {
      work(w, &e);
    } catch (const std::exception& ex) {  // e.g. no memory for what a Hub listing declared
      std::fprintf(stderr, "flyx-server: %s: %s\n", w.model.c_str(), ex.what());
      std::lock_guard<std::mutex> lock(mu_);
      error_ = w.model + ": " + ex.what(), state_ = served_.sched ? "ready" : "failed";
    }
  });
  return true;
}

bool ModelHost::run(const Want& w, std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!try_begin_locked(err)) return false;
    stop_ = false;
    reset_load_state_locked("fetching", w.model);
  }
  BusyGuard guard(this);
  return work(w, err);
}

bool ModelHost::work(const Want& w, std::string* err) {
  apps::Fetched f;
  auto logged = apps::make_progress_logger("flyx-server", 5000, [this](const hub::Progress& p) {
    std::lock_guard<std::mutex> lock(mu_);
    progress_ = p;
  });
  if (!apps::fetch_model(w.model, w.drafters, apps::wants_mtp(w.eo), logged, &stop_, &f, err, w.eo.mmproj_file,
                         w.eo.vision)) {
    std::fprintf(stderr, "flyx-server: %s: %s\n", w.model.c_str(), err->c_str());
    std::lock_guard<std::mutex> lock(mu_);
    error_ = *err, state_ = served_.sched ? "ready" : "failed";
    return false;
  }
  for (const std::string& n : f.notes) std::fprintf(stderr, "flyx-server: %s\n", n.c_str());
  // The model served now, kept to load again should the new one fail: memory for both at once is not to be had.
  Served old;
  Want old_want;
  apps::Fetched old_files;
  {
    std::lock_guard<std::mutex> lock(mu_);
    state_ = "loading", notes_ = f.notes;
    old = std::move(served_), old_want = last_, old_files = std::move(fetched_);
    served_ = {}, fetched_ = {};
  }
  const bool switching = old.sched != nullptr;
  retire(std::move(old.sched));
  std::string why;
  double load_s = 0;
  auto sched = open(f, w.eo, &load_s, &why);
  if (!sched) {
    if (!switching) {
      std::lock_guard<std::mutex> lock(mu_);
      *err = error_ = why, state_ = "failed";
      return false;
    }
    std::fprintf(stderr, "flyx-server: %s did not load: serving %s again\n", w.model.c_str(), old.name.c_str());
    std::string again;
    auto back = open(old_files, old_want.eo, &load_s, &again);
    if (back) log_loaded(*back, old.name, load_s);
    std::lock_guard<std::mutex> lock(mu_);
    if (back) {
      served_ = {back, old.name}, fetched_ = std::move(old_files);
      notes_ = fetched_.notes, state_ = "ready", fallback_ = old.name, error_ = w.model + ": " + why;
    } else {
      std::fprintf(stderr, "flyx-server: %s did not load again either: none served\n", old.name.c_str());
      state_ = "failed", error_ = w.model + ": " + why + "; " + old.name + ", loaded again: " + again;
    }
    *err = error_;
    return false;
  }
  const std::string name = !w.name.empty() ? w.name : f.name;
  log_loaded(*sched, name, load_s);
  {
    std::lock_guard<std::mutex> lock(mu_);
    served_ = {sched, name}, fetched_ = std::move(f);
    state_ = "ready", last_ = w;
  }
  save_settings(w);
  return true;
}

ojson ModelHost::status() const {
  // A slow settings drive must not hold up requests for the model already in memory.
  const ojson s = read_settings();
  std::lock_guard<std::mutex> lock(mu_);
  ojson j = {{"state", state_},
             {"model", served_.sched ? ojson(served_.name) : ojson(nullptr)},
             {"target", target_.empty() ? ojson(nullptr) : ojson(target_)},
             {"error", error_.empty() ? ojson(nullptr) : ojson(error_)},
             {"notes", notes_},
             {"fallback", fallback_.empty() ? ojson(nullptr) : ojson(fallback_)},
             {"drafters", drafters_json(served_.sched ? fetched_.drafter_sources : std::vector<apps::DrafterSource>{}, "from")},
             {"suggested", drafters_json(served_.sched ? fetched_.suggested : std::vector<apps::DrafterSource>{}, "name")},
             {"vision_projector",
              served_.sched && served_.sched->has_vision() && !fetched_.mmproj_source.empty()
                  ? ojson({{"from", fetched_.mmproj_source},
                           {"size", fetched_.mmproj_size ? ojson(*fetched_.mmproj_size) : ojson(nullptr)}})
                  : ojson(nullptr)}};
  if (busy_) j["elapsed_s"] = (os::now_ms() - since_ms_) / 1000;
  if (state_ == "fetching")
    j["progress"] = {{"file", progress_.file}, {"done", progress_.done}, {"total", progress_.total},
                     {"bytes_per_s", progress_.bytes_per_s}, {"checking", progress_.checking}};
  if (served_.sched) j["loaded"] = want_json(last_);
  j["remembered"] = s.is_object() && s.contains("last") ? s["last"] : ojson(nullptr);
  return j;
}

void log_loaded(const Scheduler& sched, const std::string& name, double load_s) {
  TelemetrySnapshot snap;
  flyx_device_stats dev{};
  bool dev_ok = false;
  sched.telemetry(&snap, &dev, &dev_ok);
  const ModelInfo& m = snap.model;
  std::fprintf(stderr, "flyx-server: %s on %s (%s), context %d, drafter %s (up to %d per round), loaded in %.1f s\n",
               name.c_str(), m.device.c_str(), m.backend.c_str(), m.max_ctx, m.drafter.c_str(), m.width, load_s);
  std::fprintf(stderr, "flyx-server: memory: %s; up to %d session%s\n", m.memory.c_str(),
               m.max_sessions, m.max_sessions == 1 ? "" : "s");
  if (m.vision) {
    std::fprintf(stderr, "flyx-server: vision: %s (%s)%s\n", m.vision_device.c_str(), m.vision_quant.c_str(),
                 m.vision_max_tokens > 0 ? (" max " + std::to_string(m.vision_max_tokens) + " tok/img").c_str() : "");
  }
  std::fprintf(stderr, "flyx-server: chat template: %s\n",
               !m.chat_ok ? "none (chat endpoints answer 400)"
               : m.chat_native
                   ? (m.chat_format + " (its own renderer)").c_str()
                   : (m.chat_format + " (Jinja engine)").c_str());
}

}  // namespace flyx::server
