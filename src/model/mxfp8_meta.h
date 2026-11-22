// r4dx::model mxfp8 weights -- the loader half of `__metadata__.quant.mxfp8` (docs/inkling2-plan.md 4.1). The literals
// both sides spell live in src/convert/include/r4dx_convert/mxfp8_format.hpp (the single source of truth, fixed); the
// converter writes the block, this header reads it. Mirrors w8a16_meta.h's contract (docs/w8a16.md 6.3):
//   - no `quant.mxfp8`: not an mxfp8 container (ParseMxfp8Metadata returns nullopt);
//   - a block whose version, group, elem, scale or fragment permutation is anything but the one value this runtime
//     implements, or that carries a field this runtime does not know, is refused (its bytes would decode to other
//     numbers with no error);
//   - `--layout mxfp8` on a container without the block is refused before any upload (every linear would otherwise
//     fall back to bf16);
//   - unlike w8a16 v1, mxfp8 shards at TP = 2 (whole 16-row tiles along N, whole 32-K groups along K: tp_shard's
//     kW8Wq / kMxfp8Ws parts), so tp_world is not a refusal.
// The block (version 1):
//   "quant": { "mxfp8": { "version": 1, "group": 32, "elem": "e4m3fn", "scale": "e8m0",
//                         "fragment_permutation": "wmma16x16x16_f16_b_lane16_k32block" } }
// HIP-free and header-only, so tests/model/test_mxfp8_meta.cpp runs every refusal without a device; the kernel half
// (the group r4dx_gemm_mxfp8_nt reads) is passed in by quant_linear.h's CheckMxfp8Meta.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"
#include "r4dx_convert/mxfp8_format.hpp"

namespace r4dx::model {

struct Mxfp8Spec {
  int version = 0;
  int group = 0;
};

// The distinct type of every mxfp8 format refusal (a std::runtime_error, as W8a16FormatMismatch).
struct Mxfp8FormatMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// `metadata`: the container's whole `__metadata__`. nullopt without a `quant.mxfp8` block; throws Mxfp8FormatMismatch
// naming `path` and the field otherwise.
inline std::optional<Mxfp8Spec> ParseMxfp8Metadata(const nlohmann::json& metadata, const std::string& path) {
  namespace f = r4dx_convert::mxfp8;
  if (!metadata.is_object() || !metadata.contains(f::kMetaQuantKey)) return std::nullopt;
  const nlohmann::json& quant = metadata.at(f::kMetaQuantKey);
  if (!quant.is_object() || !quant.contains(f::kMetaBlockKey)) return std::nullopt;
  const auto fail = [&](const std::string& what) {
    throw Mxfp8FormatMismatch("r4dx::model::Container: " + path + ": __metadata__.quant.mxfp8 " + what +
                              " -- refusing to load: its weights would decode to other numbers than the ones converted "
                              "(docs/inkling2-plan.md 4.1)");
  };
  const nlohmann::json& b = quant.at(f::kMetaBlockKey);
  if (!b.is_object()) fail("is not an object");
  for (auto it = b.begin(); it != b.end(); ++it) {
    const std::string& k = it.key();
    if (k != "version" && k != "group" && k != "elem" && k != "scale" && k != "fragment_permutation") {
      fail("has the field \"" + k + "\", which this runtime does not know (version, group, elem, scale, "
           "fragment_permutation)");
    }
  }
  const auto int_field = [&](const char* key) -> int64_t {
    if (!b.contains(key) || !b.at(key).is_number_integer()) {
      fail("\"" + std::string(key) + "\" is " + (b.contains(key) ? b.at(key).dump() : "missing") + ", not an integer");
    }
    return b.at(key).get<int64_t>();
  };
  const auto str_is = [&](const char* key, const char* want) {
    if (!b.contains(key) || !b.at(key).is_string() || b.at(key).get<std::string>() != want) {
      fail("\"" + std::string(key) + "\" is " + (b.contains(key) ? b.at(key).dump() : "missing") +
           ", this runtime implements \"" + want + "\" only");
    }
  };
  Mxfp8Spec s;
  const int64_t version = int_field("version");
  if (version != f::kMetaVersion) {
    fail("\"version\" is " + std::to_string(version) + ", this runtime implements " + std::to_string(f::kMetaVersion) +
         " only");
  }
  s.version = static_cast<int>(version);
  const int64_t group = int_field("group");
  if (group != f::kGroup) fail("\"group\" is " + std::to_string(group) + ", this format's group is " + std::to_string(f::kGroup));
  s.group = static_cast<int>(group);
  str_is("elem", f::kMetaElem);
  str_is("scale", f::kMetaScale);
  str_is("fragment_permutation", f::kMetaFragmentPermutation);
  return s;
}

// The kernel half: the container's group must be the one r4dx_gemm_mxfp8_nt reads (the value passed in).
inline void CheckMxfp8MetaAgainst(int group, int kernel_group, const std::string& what) {
  if (group != kernel_group) {
    throw Mxfp8FormatMismatch("r4dx::model: " + what + " packs mxfp8 at group " + std::to_string(group) +
                              " but this build's r4dx_gemm_mxfp8_nt reads group " + std::to_string(kernel_group));
  }
}

// `--layout mxfp8` (body or a head) on a container without the block: refused before any upload.
inline void RefuseMxfp8Choice(const std::optional<Mxfp8Spec>& spec, bool requested, const std::string& path) {
  if (!requested || spec) return;
  throw Mxfp8FormatMismatch("r4dx::model::Container: --layout mxfp8 needs an mxfp8 container (__metadata__.quant.mxfp8), "
                            "and " + path + " has none -- refusing before any upload (every linear would fall back "
                            "to bf16)");
}

// The shape rules of one mxfp8 linear [N, K] and its two tensors' byte sizes (`.mxfp8.wq` N*K, `.mxfp8.ws` N*K/32).
// "" when they hold, else the reason naming the tensor. Pure, so the CPU test checks it without a container.
inline std::string Mxfp8ShapeProblem(const std::string& base, int64_t N, int64_t K, uint64_t wq_bytes, uint64_t ws_bytes) {
  namespace f = r4dx_convert::mxfp8;
  const std::string who = "mxfp8 linear '" + base + "' [" + std::to_string(N) + ", " + std::to_string(K) + "]: ";
  if (N <= 0 || K <= 0 || N % 16 != 0 || K % f::kGroup != 0) {
    return who + "N must be a positive multiple of 16 and K of 32";
  }
  const uint64_t nk = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
  if (wq_bytes != nk) {
    return who + "'" + base + f::kWqSuffix + "' is " + std::to_string(wq_bytes) + " bytes, expected " + std::to_string(nk);
  }
  const uint64_t want_ws = nk / static_cast<uint64_t>(f::kGroup);
  if (ws_bytes != want_ws) {
    return who + "'" + base + f::kWsSuffix + "' is " + std::to_string(ws_bytes) + " bytes, expected " +
           std::to_string(want_ws);
  }
  return std::string();
}

}  // namespace r4dx::model
