// r4dx::model trellis weights -- the loader half of `__metadata__.quant.trellis`
// (docs/trellis-kernel.md sections 2.3 and 2.5). The converter half is
// src/convert/include/r4dx_convert/trellis_import.hpp (`r4dx-convert --trellis-from`,
// TrellisSource::QuantMetadata / RunMetadata).
//
// HIP-free and header-only, like rotation_meta.h and w4a16_group_meta.h: Container::Load and
// Container::LoadShard (src/model/container.cpp) are the production callers, and nothing here
// touches tensor bytes or asks the kernel anything -- whether this build's r4d_gemm_trellis_nt_m64
// runs a linear's rate is CheckTrellisRate's job (quant_linear.h), called by the loader per linear.
//
// The contract, in the loader's words:
//   - no `quant.trellis`: not a trellis container (ParseTrellisMetadata returns nullopt);
//   - a block whose format, version, codebook (and its constants), state width, tail-biting flag,
//     position order, bitstream, tile grid or Hadamard is anything but the one value this runtime
//     implements is refused -- its bits would decode to other numbers with no error;
//   - `linears` = {"<container base>": {"bits": 4|5, "parts": [n0, n1]?, "prescale_log2": s?}}
//     names every trellis linear; `parts` only on a fused pair with two input transforms
//     (mlp.gate_up: gate's rows then up's);
//   - the container must say its reconstruction check passed:
//     r4dx_convert_run.trellis.verify.result starts with "pass", and `failed` (when recorded) is 0
//     and `checked` (when recorded) equals the HF tensors the linears stand for (a two-part linear
//     is two). "pending" (the conversion died before the check finished), "FAILED" and "not run"
//     (--trellis-verify none, a debug build) are refused (docs/trellis-kernel.md 10.3, "For M4").
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx::model {

// The prescale range the converter writes (--trellis-prescale-log2, [-16, 16]); the input transform
// itself accepts |s| <= 24.
inline constexpr int kTrellisMaxPrescaleLog2 = 16;
// The Hadamard block of both transforms, and so the granularity of every trellis K, N, part and
// tensor-parallel rank range (docs/trellis-kernel.md 2.4).
inline constexpr int64_t kTrellisBlock = 128;

struct TrellisLinearSpec {
  int bits = 0;                 // KB: trellis bits per weight
  std::vector<int64_t> parts;   // output rows per part (global); empty = one part of all N rows
  int prescale_log2 = 0;        // the container's, or this linear's own override
  int Parts() const { return parts.empty() ? 1 : static_cast<int>(parts.size()); }
};

struct TrellisSpec {
  int prescale_log2 = 0;
  std::map<std::string, TrellisLinearSpec> linears;
  // The HF tensors the linears stand for: one per part (verify.checked must equal it).
  int64_t HfTensors() const {
    int64_t n = 0;
    for (const auto& kv : linears) n += kv.second.Parts();
    return n;
  }
  const TrellisLinearSpec* Find(const std::string& base) const {
    const auto it = linears.find(base);
    return it == linears.end() ? nullptr : &it->second;
  }
};

// `metadata`: the container's whole `__metadata__`. Returns nullopt when there is no
// `quant.trellis` block. Throws std::runtime_error naming `path` for any field this runtime does
// not implement, a malformed `linears` entry, and a container whose reconstruction check did not
// pass.
inline std::optional<TrellisSpec> ParseTrellisMetadata(const nlohmann::json& metadata,
                                                       const std::string& path) {
  if (!metadata.is_object() || !metadata.contains("quant")) return std::nullopt;
  const nlohmann::json& quant = metadata.at("quant");
  if (!quant.is_object() || !quant.contains("trellis")) return std::nullopt;
  const auto fail = [&](const std::string& what) {
    throw std::runtime_error("r4dx::model::Container: " + path + ": __metadata__.quant.trellis " +
                             what + " -- refusing to load: its weights would decode to other "
                             "numbers than the ones converted (docs/trellis-kernel.md 2.3, 2.5)");
  };
  const nlohmann::json& t = quant.at("trellis");
  if (!t.is_object()) fail("is not an object");
  const auto str_is = [&](const nlohmann::json& obj, const char* key, const char* want,
                          const std::string& where) {
    if (!obj.contains(key) || !obj.at(key).is_string() || obj.at(key).get<std::string>() != want) {
      fail("\"" + where + key + "\" is " + (obj.contains(key) ? obj.at(key).dump() : "missing") +
           ", this runtime implements \"" + want + "\" only");
    }
  };
  const auto int_is = [&](const nlohmann::json& obj, const char* key, int64_t want,
                          const std::string& where) {
    if (!obj.contains(key) || !obj.at(key).is_number_integer() ||
        obj.at(key).get<int64_t>() != want) {
      fail("\"" + where + key + "\" is " + (obj.contains(key) ? obj.at(key).dump() : "missing") +
           ", this runtime implements " + std::to_string(want) + " only");
    }
  };
  str_is(t, "format", "r4dx-trellis", "");
  int_is(t, "version", 1, "");
  str_is(t, "codebook", "mul1", "");
  if (!t.contains("codebook_consts") || !t.at("codebook_consts").is_object()) {
    fail("has no \"codebook_consts\" object");
  }
  const nlohmann::json& cc = t.at("codebook_consts");
  str_is(cc, "mult", "0x83dcd12d", "codebook_consts.");
  str_is(cc, "k_inv_f16", "0x1eee", "codebook_consts.");
  str_is(cc, "k_bias_f16", "0xc931", "codebook_consts.");
  int_is(t, "state_bits", 16, "");
  if (!t.contains("tail_biting") || !t.at("tail_biting").is_boolean() ||
      !t.at("tail_biting").get<bool>()) {
    fail("\"tail_biting\" is not true");
  }
  str_is(t, "position_order", "exl3-tensor-core", "");
  str_is(t, "bitstream", "ring-u32-msb-first", "");
  str_is(t, "tile_grid", "n32-pairs-k-major", "");
  if (!t.contains("hadamard") || !t.at("hadamard").is_object()) fail("has no \"hadamard\" object");
  const nlohmann::json& had = t.at("hadamard");
  int_is(had, "block", kTrellisBlock, "hadamard.");
  str_is(had, "order", "sylvester-natural", "hadamard.");
  str_is(had, "scale", "1/sqrt(128)", "hadamard.");
  str_is(had, "input", "x*suh then H", "hadamard.");
  str_is(had, "output", "H then *svh", "hadamard.");

  const auto prescale = [&](const nlohmann::json& obj, const std::string& where) {
    if (!obj.at("prescale_log2").is_number_integer()) fail("\"" + where + "\" is not an integer");
    const int64_t s = obj.at("prescale_log2").get<int64_t>();
    if (s < -kTrellisMaxPrescaleLog2 || s > kTrellisMaxPrescaleLog2) {
      fail("\"" + where + "\" = " + std::to_string(s) + " is outside [-16, 16]");
    }
    return static_cast<int>(s);
  };
  TrellisSpec spec;
  if (!t.contains("prescale_log2")) fail("has no \"prescale_log2\"");
  spec.prescale_log2 = prescale(t, "prescale_log2");

  if (!t.contains("linears") || !t.at("linears").is_object() || t.at("linears").empty()) {
    fail("has no non-empty \"linears\" object");
  }
  const nlohmann::json& lin = t.at("linears");
  for (auto it = lin.begin(); it != lin.end(); ++it) {
    const std::string& base = it.key();
    const std::string where = "linears." + base + ".";
    if (base.empty()) fail("\"linears\" has an empty base name");
    const nlohmann::json& e = it.value();
    if (!e.is_object()) fail("\"" + where.substr(0, where.size() - 1) + "\" is not an object");
    for (auto f = e.begin(); f != e.end(); ++f) {
      if (f.key() != "bits" && f.key() != "parts" && f.key() != "prescale_log2") {
        fail("\"" + where + f.key() + "\" is not a field this runtime knows (bits, parts, "
             "prescale_log2)");
      }
    }
    TrellisLinearSpec L;
    if (!e.contains("bits") || !e.at("bits").is_number_integer()) {
      fail("\"" + where + "bits\" is missing or not an integer");
    }
    const int64_t bits = e.at("bits").get<int64_t>();
    if (bits != 4 && bits != 5) {
      fail("\"" + where + "bits\" = " + std::to_string(bits) +
           " is not a rate of this format (4, 5)");
    }
    L.bits = static_cast<int>(bits);
    if (e.contains("parts")) {
      const nlohmann::json& p = e.at("parts");
      if (!p.is_array() || p.size() != 2) fail("\"" + where + "parts\" is not an array of two");
      for (const auto& v : p) {
        if (!v.is_number_integer() || v.get<int64_t>() <= 0 ||
            v.get<int64_t>() % kTrellisBlock != 0) {
          fail("\"" + where + "parts\" = " + p.dump() + " is not two positive multiples of 128");
        }
        L.parts.push_back(v.get<int64_t>());
      }
    }
    L.prescale_log2 = e.contains("prescale_log2") ? prescale(e, where + "prescale_log2")
                                                  : spec.prescale_log2;
    spec.linears.emplace(base, std::move(L));
  }

  // The converter's own verdict on the bytes it wrote (docs/trellis-kernel.md 3.3 step 10, 10.3).
  const auto unverified = [&](const std::string& why) {
    throw std::runtime_error("r4dx::model::Container: " + path +
                             " is a trellis container whose reconstruction check did not pass (" +
                             why + ") -- refusing to load it (r4dx-convert --trellis-verify full "
                             "writes a verified container; docs/trellis-kernel.md 10.3)");
  };
  const nlohmann::json* verify = nullptr;
  if (metadata.contains("r4dx_convert_run") && metadata.at("r4dx_convert_run").is_object()) {
    const nlohmann::json& run = metadata.at("r4dx_convert_run");
    if (run.contains("trellis") && run.at("trellis").is_object() &&
        run.at("trellis").contains("verify") && run.at("trellis").at("verify").is_object()) {
      verify = &run.at("trellis").at("verify");
    }
  }
  if (verify == nullptr) unverified("no r4dx_convert_run.trellis.verify record");
  if (!verify->contains("result") || !verify->at("result").is_string()) {
    unverified("verify.result is missing");
  }
  const std::string result = verify->at("result").get<std::string>();
  if (result.rfind("pass", 0) != 0) unverified("verify.result is \"" + result + "\"");
  if (verify->contains("failed") &&
      (!verify->at("failed").is_number_integer() || verify->at("failed").get<int64_t>() != 0)) {
    unverified("verify.failed is " + verify->at("failed").dump());
  }
  if (verify->contains("checked") &&
      (!verify->at("checked").is_number_integer() ||
       verify->at("checked").get<int64_t>() != spec.HfTensors())) {
    unverified("verify.checked is " + verify->at("checked").dump() + ", the linears stand for " +
               std::to_string(spec.HfTensors()) + " HF tensors");
  }
  return spec;
}

}  // namespace r4dx::model
