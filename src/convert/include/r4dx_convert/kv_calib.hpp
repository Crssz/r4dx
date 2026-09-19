// r4dx_convert::ResolveKvDescale -- consumes tools/reference/kv_calibrate.py's merged calibration
// JSON ({"<layer_idx>": {"k_amax": [kv_heads floats], "v_amax": [kv_heads floats], ...}}) into the
// container's fp32[kv_heads] `text.layers.{i}.attn.k_descale` / `.v_descale` tensors
// (docs/container-format.md "KV descale tables").
//
// Descale convention (exact, matches src/kernels/src/r4dx_kernels.hip's
// r4dx_kv_write_paged_fp8_hnd): stored_fp8 = fp8e4m3(real_bf16_value / descale[head]) -- the
// kernel DIVIDES by descale before the fp8 cast on write -- so the inverse, dequant = fp8_value *
// descale[head], is exactly what the attention kernel's own dequant must (and does) use. Hence
// descale = amax / 448.0, where 448.0 is the OCP e4m3fn finite max magnitude and `amax` is the
// per-kv-head absolute max kv_calibrate.py measured (K: post-rope; V: raw v_proj, no rope).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx_convert {

constexpr float kFp8E4M3Max = 448.0f;

struct KvDescaleResult {
  std::vector<float> values;   // fp32[kv_heads]
  bool from_calibration = false;
  std::string warning;  // non-empty iff calibration was expected but unusable (missing layer,
                         // missing/malformed amax array, or wrong length) -- caller should log it.
};

// `calib` is tools/reference/kv_calibrate.py's merged JSON object (ignored, may be default-
// constructed, when `have_calib` is false). `have_calib` should be false (not just an empty/absent
// `calib`) for any layer this calibration pass never covers by construction (e.g. the MTP layer --
// kv_calibrate.py only ever calibrates the 64 text-layer indices) so a legitimately-uncalibrated
// layer falls back to descale=1.0 silently instead of emitting a spurious "no entry" warning.
// `kind` is "k" or "v"; `layer_idx` is the text-layer index kv_calibrate.py's --layer used.
inline KvDescaleResult ResolveKvDescale(const nlohmann::json& calib, bool have_calib, int layer_idx,
                                         int kv_heads, const std::string& kind) {
  KvDescaleResult r;
  r.values.assign(static_cast<size_t>(kv_heads), 1.0f);
  if (!have_calib) return r;

  const std::string key = std::to_string(layer_idx);
  const std::string amax_key = kind + "_amax";
  if (!calib.contains(key)) {
    r.warning = "kv-calib has no entry for layer " + key + " -- falling back to descale=1.0";
    return r;
  }
  const auto& entry = calib.at(key);
  if (!entry.contains(amax_key)) {
    r.warning =
        "kv-calib layer " + key + " is missing " + amax_key + " -- falling back to descale=1.0";
    return r;
  }
  std::vector<float> amax;
  try {
    amax = entry.at(amax_key).get<std::vector<float>>();
  } catch (const std::exception& e) {
    r.warning = "kv-calib layer " + key + " " + amax_key + " is malformed (" +
                std::string(e.what()) + ") -- falling back to descale=1.0";
    return r;
  }
  if (static_cast<int>(amax.size()) != kv_heads) {
    r.warning = "kv-calib layer " + key + " " + amax_key + " has " + std::to_string(amax.size()) +
                " entries, expected " + std::to_string(kv_heads) + " -- falling back to descale=1.0";
    return r;
  }
  for (int h = 0; h < kv_heads; ++h) r.values[static_cast<size_t>(h)] = amax[static_cast<size_t>(h)] / kFp8E4M3Max;
  r.from_calibration = true;
  return r;
}

}  // namespace r4dx_convert
