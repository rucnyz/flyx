// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Typed configuration option enums and canonical string parsers shared by core, models, and apps.
#pragma once

#include <charconv>
#include <string>
#include <string_view>

namespace flyx::interfaces {

enum class DrafterMode { kAuto, kMtp, kDFlash2, kDSpine, kDv, kAll, kNone };

inline bool parse_drafter_mode(std::string_view value, DrafterMode* out) {
  if (value.empty()) {
    *out = DrafterMode::kAuto;
    return true;
  }
  if (value == "mtp") {
    *out = DrafterMode::kMtp;
    return true;
  }
  if (value == "dflash2") {
    *out = DrafterMode::kDFlash2;
    return true;
  }
  if (value == "dspine") {
    *out = DrafterMode::kDSpine;
    return true;
  }
  if (value == "dv") {
    *out = DrafterMode::kDv;
    return true;
  }
  if (value == "all") {
    *out = DrafterMode::kAll;
    return true;
  }
  if (value == "none") {
    *out = DrafterMode::kNone;
    return true;
  }
  return false;
}

enum class KvMethodKind { kExact, kAuto, kStrata };

struct KvMethodSpec {
  KvMethodKind kind = KvMethodKind::kExact;
  int bits = 0;  // 2, 3, or 4 when kind != kExact
};

// Parses "exact", "strata[:BITS]", or "auto[:BITS]" (BITS in 2..4; default: 2).
inline bool parse_kv_method(std::string_view value, KvMethodSpec* out, std::string* err) {
  if (value == "exact") {
    *out = {KvMethodKind::kExact, 0};
    return true;
  }
  const size_t colon = value.find(':');
  const std::string_view method = value.substr(0, colon);
  const bool strata = method == "strata";
  if (!strata && method != "auto") {
    if (err) *err = "--kv-method " + std::string(value) + ": exact, strata[:BITS] or auto[:BITS]";
    return false;
  }
  int bits = 2;
  if (colon != std::string_view::npos) {
    const char* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data() + colon + 1, end, bits);
    if (parsed.ec != std::errc{} || parsed.ptr != end || bits < 2 || bits > 4) {
      if (err) *err = "--kv-method " + std::string(value) + ": 2, 3 or 4 bits";
      return false;
    }
  }
  *out = {strata ? KvMethodKind::kStrata : KvMethodKind::kAuto, bits};
  return true;
}

enum class VisionDeviceMode { kDisk, kHost, kGpu, kAuto };

inline bool parse_vision_device(std::string_view value, VisionDeviceMode* out, std::string* canonical,
                                std::string* err) {
  std::string lo(value);
  for (char& c : lo)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  if (lo.empty() || lo == "disk" || lo == "file") {
    if (out) *out = VisionDeviceMode::kDisk;
    if (canonical) *canonical = "disk";
    return true;
  }
  if (lo == "host" || lo == "cpu") {
    if (out) *out = VisionDeviceMode::kHost;
    if (canonical) *canonical = "host";
    return true;
  }
  if (lo == "gpu") {
    if (out) *out = VisionDeviceMode::kGpu;
    if (canonical) *canonical = "gpu";
    return true;
  }
  if (lo == "auto") {
    if (out) *out = VisionDeviceMode::kAuto;
    if (canonical) *canonical = "auto";
    return true;
  }
  if (err) *err = "unknown --vision-device '" + std::string(value) + "' (disk, host, gpu, or auto)";
  return false;
}

enum class VisionQuantMode { kQ8, kExact, kNvfp4 };

inline bool parse_vision_quant(std::string_view value, VisionQuantMode* out, std::string* canonical,
                               std::string* err) {
  std::string lo(value);
  for (char& c : lo)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  if (lo.empty() || lo == "q8" || lo == "q8_0") {
    if (out) *out = VisionQuantMode::kQ8;
    if (canonical) *canonical = "q8";
    return true;
  }
  if (lo == "exact" || lo == "bf16") {
    if (out) *out = VisionQuantMode::kExact;
    if (canonical) *canonical = "exact";
    return true;
  }
  if (lo == "nvfp4" || lo == "fp4") {
    if (out) *out = VisionQuantMode::kNvfp4;
    if (canonical) *canonical = "nvfp4";
    return true;
  }
  if (err) *err = "unknown --vision-quant '" + std::string(value) + "' (q8, exact, or nvfp4)";
  return false;
}

inline bool normalize_image_detail(std::string_view value, std::string* canonical, std::string* err) {
  std::string lo(value);
  for (char& c : lo)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  if (lo.empty() || lo == "auto") {
    if (canonical) *canonical = "auto";
    return true;
  }
  if (lo == "low" || lo == "high") {
    if (canonical) *canonical = std::move(lo);
    return true;
  }
  if (err) *err = "invalid image detail '" + std::string(value) + "' (expected auto, low, or high)";
  return false;
}

}  // namespace flyx::interfaces
