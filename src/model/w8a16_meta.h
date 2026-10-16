// r4dx::model w8a16 weights -- the loader half of `__metadata__.quant.w8a16` (docs/w8a16.md 3.1, 6.1, 6.3).
// The converter half is src/convert/main.cpp's BuildQuantMetadata; the literals both sides spell live in
// src/convert/include/r4dx_convert/w8a16_format.hpp (the single source of truth).
//
// HIP-free and header-only, like trellis_meta.h: everything here is a pure function of the metadata, the
// requested layout and tensor byte spans, so tests/model/test_w8a16_meta.cpp runs every refusal without a
// device. What needs the kernel (the instantiated groups and the bias it undoes) is passed in by the caller:
// quant_linear.h's CheckW8a16Meta asks r4dx_gemm_w8a16_has_group / _scale_log2_bias and calls
// CheckW8a16MetaAgainst below.
//
// The contract, in the loader's words:
//   - no `quant.w8a16`: not a w8a16 container (ParseW8a16Metadata returns nullopt);
//   - a block whose version, scale format, code, fragment permutation or bias is anything but the one value
//     this runtime implements, whose group is not one the format defines, or that carries a field this
//     runtime does not know (a v1 loader must not ignore a future per-linear `groups` map) is refused -- its
//     bytes would decode to other numbers with no error;
//   - `--layout w8a16` on a container without the block is refused before any upload (6.3 rule 1: every
//     linear would otherwise fall back to bf16 and one card would try to hold a 48 GB bf16 body);
//   - w8a16 runs at TP = 1 only in v1 (the TP = 2 shard parts are a later deliverable).
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"
#include "r4dx_convert/w8a16_format.hpp"

namespace r4dx::model {

// The parsed block. Only `group` varies between valid v1 containers; the rest are format constants,
// kept so a log or a test can print what was read.
struct W8a16Spec {
  int version = 0;
  int group = 0;
  int scale_log2_bias = 0;
};

// The distinct type of every w8a16 format refusal (metadata or kernel mismatch), a std::runtime_error so
// every existing catch site is unaffected; a caller that tolerates OTHER load failures must not swallow
// this one (as W4a16GroupMismatch).
struct W8a16FormatMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// `metadata`: the container's whole `__metadata__`. Returns nullopt without a `quant.w8a16` block. Throws
// W8a16FormatMismatch naming `path` and the field otherwise.
inline std::optional<W8a16Spec> ParseW8a16Metadata(const nlohmann::json& metadata, const std::string& path) {
  namespace f = r4dx_convert::w8a16;
  if (!metadata.is_object() || !metadata.contains(f::kMetaQuantKey)) return std::nullopt;
  const nlohmann::json& quant = metadata.at(f::kMetaQuantKey);
  if (!quant.is_object() || !quant.contains(f::kMetaBlockKey)) return std::nullopt;
  const auto fail = [&](const std::string& what) {
    throw W8a16FormatMismatch("r4dx::model::Container: " + path + ": __metadata__.quant.w8a16 " + what +
                              " -- refusing to load: its weights would decode to other numbers than the ones "
                              "converted (docs/w8a16.md 3.1, 6.3)");
  };
  const nlohmann::json& b = quant.at(f::kMetaBlockKey);
  if (!b.is_object()) fail("is not an object");
  for (auto it = b.begin(); it != b.end(); ++it) {
    const std::string& k = it.key();
    if (k != "version" && k != "group" && k != "scale" && k != "scale_log2_bias" && k != "code" &&
        k != "fragment_permutation") {
      fail("has the field \"" + k + "\", which this runtime does not know (version, group, scale, "
           "scale_log2_bias, code, fragment_permutation)");
    }
  }
  const auto int_field = [&](const char* key) -> int64_t {
    if (!b.contains(key) || !b.at(key).is_number_integer()) {
      fail("\"" + std::string(key) + "\" is " + (b.contains(key) ? b.at(key).dump() : "missing") +
           ", not an integer");
    }
    return b.at(key).get<int64_t>();
  };
  const auto str_is = [&](const char* key, const char* want) {
    if (!b.contains(key) || !b.at(key).is_string() || b.at(key).get<std::string>() != want) {
      fail("\"" + std::string(key) + "\" is " + (b.contains(key) ? b.at(key).dump() : "missing") +
           ", this runtime implements \"" + want + "\" only");
    }
  };
  W8a16Spec s;
  const int64_t version = int_field("version");
  if (version != f::kMetaVersion) {
    fail("\"version\" is " + std::to_string(version) + ", this runtime implements " +
         std::to_string(f::kMetaVersion) + " only");
  }
  s.version = static_cast<int>(version);
  const int64_t group = int_field("group");
  if (group < 1 || group > 1 << 20 || !f::IsGroupSupported(static_cast<int>(group))) {
    fail("\"group\" is " + std::to_string(group) + ", not a group of this format (64, 128)");
  }
  s.group = static_cast<int>(group);
  str_is("scale", f::kMetaScaleFormat);
  const int64_t bias = int_field("scale_log2_bias");
  if (bias != f::kScaleLog2Bias) {
    fail("\"scale_log2_bias\" is " + std::to_string(bias) + ", this format's bias is " +
         std::to_string(f::kScaleLog2Bias));
  }
  s.scale_log2_bias = static_cast<int>(bias);
  str_is("code", f::kMetaCode);
  str_is("fragment_permutation", f::kMetaFragmentPermutation);
  return s;
}

// docs/w8a16.md 6.3 rule 2, the kernel half: the container's group must be one the kernel instantiates and
// its bias the one the kernel undoes (the kernel's values passed in: quant_linear.h's CheckW8a16Meta).
inline void CheckW8a16MetaAgainst(int group, int scale_log2_bias, bool kernel_has_group, int kernel_bias,
                                  const std::string& what) {
  if (!kernel_has_group) {
    throw W8a16FormatMismatch("r4dx::model: " + what + " packs w8a16 at group " + std::to_string(group) +
                              ", which this build's r4dx_gemm_w8a16_nt_m64 does not instantiate");
  }
  if (scale_log2_bias != kernel_bias) {
    throw W8a16FormatMismatch("r4dx::model: " + what + " stores w8a16 scales biased by 2^" +
                              std::to_string(scale_log2_bias) + " but this build's r4dx_gemm_w8a16_nt_m64 undoes 2^" +
                              std::to_string(kernel_bias) + " -- every output would be off by a power of two");
  }
}

// docs/w8a16.md 6.3 rule 1 and the v1 TP rule, for a load that `requested` w8a16 for any linear (body or a
// head). `spec` is ParseW8a16Metadata's result. A load that did not request w8a16 is never refused here.
inline void RefuseW8a16Choice(const std::optional<W8a16Spec>& spec, bool requested, int tp_world,
                              const std::string& path) {
  if (!requested) return;
  if (tp_world > 1) {
    throw std::invalid_argument("r4dx::model::Container: " + path + " -- the w8a16 layout runs at TP=1 only (--tp 1 "
                                "--layout w8a16); tensor parallelism of w8a16 is not part of v1 (docs/w8a16.md 6.2)");
  }
  if (!spec) {
    throw W8a16FormatMismatch("r4dx::model::Container: --layout w8a16 needs a w8a16 container "
                              "(__metadata__.quant.w8a16), and " + path + " has none -- refusing before any "
                              "upload (every linear would fall back to bf16; docs/w8a16.md 6.3 rule 1)");
  }
}

// docs/w8a16.md 3.3 / 6.3 rule 3: the shape rules of one w8a16 linear [N, K] at `group` and its two tensors'
// byte sizes (`.w8a16.wq` N*K, `.w8a16.ws` N*K/group*2). Returns "" when they hold, else the reason, naming
// the tensor. Pure, so the CPU test checks the real container's spans with it.
inline std::string W8a16ShapeProblem(const std::string& base, int64_t N, int64_t K, int group, uint64_t wq_bytes,
                                     uint64_t ws_bytes) {
  namespace f = r4dx_convert::w8a16;
  const std::string who = "w8a16 linear '" + base + "' [" + std::to_string(N) + ", " + std::to_string(K) +
                          "] at group " + std::to_string(group) + ": ";
  if (N <= 0 || K <= 0 || group <= 0 || N % f::kTileRows != 0 || K % group != 0 || K % f::kKBlock != 0) {
    return who + "N must be a multiple of 16 and K of the group (and of 32)";
  }
  const uint64_t nk = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
  if (wq_bytes != nk) {
    return who + "'" + base + f::kWqSuffix + "' is " + std::to_string(wq_bytes) + " bytes, expected " +
           std::to_string(nk);
  }
  const uint64_t want_ws = nk / static_cast<uint64_t>(group) * 2;
  if (ws_bytes != want_ws) {
    return who + "'" + base + f::kWsSuffix + "' is " + std::to_string(ws_bytes) + " bytes, expected " +
           std::to_string(want_ws);
  }
  return std::string();
}

}  // namespace r4dx::model
