// tests/model/test_w8a16_meta.cpp -- CPU only (no HIP call; HIP_VISIBLE_DEVICES=-1 hides every device from the runtime that
// linking r4dx_model_linear loads). docs/w8a16.md 7.1 `model_w8a16_meta`:
//   (a) Layout::kW8a16 is pinned at 5 and round-trips through LayoutName / LayoutFromName;
//   (b) w8a16_meta.h's strict parse of `__metadata__.quant.w8a16`: the converter's block (from w8a16_format.hpp's constants)
//       parses, no block is nullopt, and every field this runtime does not implement is refused naming the field (version,
//       group, scale, scale_log2_bias, code, fragment_permutation, an unknown field, a non-object block);
//   (c) the 6.3 refusals that need no device: rule 1 (--layout w8a16 without the block), the v1 TP=1 rule, rule 2 against
//       this build's kernel exports (CheckW8a16Meta) and against injected kernel values (CheckW8a16MetaAgainst), rule 3
//       (W8a16ShapeProblem: N % 16, K % group, wrong .wq / .ws byte sizes, naming the tensor);
//   (d) W1's tiny w8a16 container (R4DX_LFM2_TINY_W8A16_DIR, default <models root>/r4dx/lfm2/tiny_w8a16/tiny_w8a16.r4dx;
//       SKIP of this section when absent): InspectLfm2Container accepts the table, has_w8a16, the block parses at group
//       128 / bias 12 and passes the kernel check, and EVERY body linear and the head carries both w8a16 tensors whose byte
//       spans pass W8a16ShapeProblem at the config's shapes (what Lfm2Container::Load will check before each upload), and
//       no linear carries a bf16 form (a --no-bf16 container: `--layout bf16` on it is refused by construction, 6.3 rule 5).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_config.h"
#include "lfm2_container_info.h"
#include "nlohmann/json.hpp"
#include "quant_linear.h"
#include "r4dx/models_root.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/w8a16_format.hpp"
#include "w8a16_meta.h"

using nlohmann::json;
using namespace r4dx::model;
namespace fmt = r4dx_convert::w8a16;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

template <class F>
bool Throws(F&& f, const std::string& needle, std::string* msg = nullptr) {
  try {
    f();
  } catch (const std::exception& e) {
    if (msg) *msg = e.what();
    return needle.empty() || std::string(e.what()).find(needle) != std::string::npos;
  }
  if (msg) *msg = "(no exception)";
  return false;
}

template <class E, class F>
bool ThrowsType(F&& f) {
  try {
    f();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// The block BuildQuantMetadata writes (docs/w8a16.md 3.1), spelled from the shared constants.
json GoodMeta(int group = 128) {
  json m;
  m["quant"]["w8a16"] = {{"version", fmt::kMetaVersion},
                         {"group", group},
                         {"scale", fmt::kMetaScaleFormat},
                         {"scale_log2_bias", fmt::kScaleLog2Bias},
                         {"code", fmt::kMetaCode},
                         {"fragment_permutation", fmt::kMetaFragmentPermutation}};
  return m;
}

void TestLayoutEnum() {
  Check(static_cast<int>(Layout::kW8a16) == 5, "Layout::kW8a16 == 5 (pinned; not the retired 1 or 3)");
  Check(LayoutFromName("w8a16") == Layout::kW8a16, "LayoutFromName(\"w8a16\") == kW8a16");
  Check(std::string(LayoutName(Layout::kW8a16)) == "w8a16", "LayoutName(kW8a16) == \"w8a16\"");
  Check(LayoutFromName("bf16") == Layout::kBf16 && LayoutFromName("w4a16") == Layout::kW4a16 &&
            LayoutFromName("trellis") == Layout::kTrellis,
        "the other layout names are unchanged");
  Check(Throws([] { (void)LayoutFromName("w8a8"); }, "unrecognized layout"), "an unknown layout name still throws");
}

void TestParse() {
  const std::string path = "x.r4dx";
  {
    const std::optional<W8a16Spec> s = ParseW8a16Metadata(GoodMeta(), path);
    Check(s && s->version == 1 && s->group == 128 && s->scale_log2_bias == 12, "the converter's g128 block parses");
    const std::optional<W8a16Spec> s64 = ParseW8a16Metadata(GoodMeta(64), path);
    Check(s64 && s64->group == 64, "a g64 block parses");
  }
  Check(!ParseW8a16Metadata(json::object(), path), "no quant block: nullopt");
  {
    json m;
    m["quant"]["w4a16"] = {{"group", 64}};
    Check(!ParseW8a16Metadata(m, path), "a quant block without w8a16 (a w4a16 / bf16 container): nullopt");
  }
  struct Bad {
    const char* what;
    std::function<void(json&)> edit;
    const char* needle;
  };
  const std::vector<Bad> bad = {
      {"version 2", [](json& b) { b["version"] = 2; }, "\"version\""},
      {"version missing", [](json& b) { b.erase("version"); }, "\"version\""},
      {"version as a string", [](json& b) { b["version"] = "1"; }, "\"version\""},
      {"group 96", [](json& b) { b["group"] = 96; }, "\"group\""},
      {"group 256", [](json& b) { b["group"] = 256; }, "\"group\""},
      {"group missing", [](json& b) { b.erase("group"); }, "\"group\""},
      {"scale bf16", [](json& b) { b["scale"] = "bf16"; }, "\"scale\""},
      {"scale_log2_bias 13", [](json& b) { b["scale_log2_bias"] = 13; }, "\"scale_log2_bias\""},
      {"scale_log2_bias 0", [](json& b) { b["scale_log2_bias"] = 0; }, "\"scale_log2_bias\""},
      {"code int8_sym128", [](json& b) { b["code"] = "int8_sym128"; }, "\"code\""},
      {"fragment_permutation other", [](json& b) { b["fragment_permutation"] = "wmma16x16x16_lane16_koff"; },
       "\"fragment_permutation\""},
      {"an unknown field (a v2 per-linear groups map)", [](json& b) { b["groups"] = json::object(); }, "\"groups\""},
  };
  for (const Bad& c : bad) {
    json m = GoodMeta();
    c.edit(m["quant"]["w8a16"]);
    std::string msg;
    const bool ok = Throws([&] { (void)ParseW8a16Metadata(m, path); }, c.needle, &msg) &&
                    ThrowsType<W8a16FormatMismatch>([&] { (void)ParseW8a16Metadata(m, path); });
    Check(ok && msg.find(path) != std::string::npos, std::string("refused, naming the field and the path: ") + c.what +
                                                         " -> " + msg);
  }
  {
    json m;
    m["quant"]["w8a16"] = 128;
    Check(Throws([&] { (void)ParseW8a16Metadata(m, path); }, "not an object"), "a non-object block is refused");
  }
}

void TestRefusals() {
  const std::string path = "c.r4dx";
  const std::optional<W8a16Spec> spec = ParseW8a16Metadata(GoodMeta(), path);
  std::string msg;
  // Rule 1.
  Check(Throws([&] { RefuseW8a16Choice(std::nullopt, true, 1, path); }, "needs a w8a16 container", &msg) &&
            msg.find(path) != std::string::npos,
        "rule 1: --layout w8a16 on a container without quant.w8a16 is refused before any upload: " + msg);
  Check(ThrowsType<W8a16FormatMismatch>([&] { RefuseW8a16Choice(std::nullopt, true, 1, path); }),
        "rule 1 throws W8a16FormatMismatch");
  Check(!Throws([&] { RefuseW8a16Choice(std::nullopt, false, 1, path); }, ""),
        "a load that does not request w8a16 is never refused (a bf16 / w4a16 / trellis container)");
  Check(!Throws([&] { RefuseW8a16Choice(spec, false, 2, path); }, ""),
        "not requested: TP=2 on a container that carries the block is not this rule's business");
  Check(!Throws([&] { RefuseW8a16Choice(spec, true, 1, path); }, ""), "requested, block present, TP=1: accepted");
  // v1 TP rule.
  Check(Throws([&] { RefuseW8a16Choice(spec, true, 2, path); }, "TP=1 only", &msg),
        "w8a16 at TP=2 is refused (v1): " + msg);
  // Rule 2, against this build's kernel exports.
  Check(r4dx_gemm_w8a16_scale_log2_bias() == fmt::kScaleLog2Bias,
        "the kernel's scale bias export equals w8a16_format.hpp's kScaleLog2Bias (12)");
  Check(r4dx_gemm_w8a16_has_group(64) == 1 && r4dx_gemm_w8a16_has_group(128) == 1 && r4dx_gemm_w8a16_has_group(96) == 0,
        "the kernel instantiates exactly the format's groups (IsGroupSupported: 64, 128)");
  for (int g : {32, 64, 96, 128, 256}) {
    Check((r4dx_gemm_w8a16_has_group(g) != 0) == fmt::IsGroupSupported(g),
          "kernel has_group(" + std::to_string(g) + ") agrees with w8a16_format.hpp IsGroupSupported");
  }
  Check(r4dx_gemm_w8a16_max_m() == 64, "the kernel's max M is 64 (ApplyLinear's chunk)");
  Check(!Throws([&] { CheckW8a16Meta(128, 12, path); }, ""), "CheckW8a16Meta(128, 12) passes on this build");
  Check(!Throws([&] { CheckW8a16Meta(64, 12, path); }, ""), "CheckW8a16Meta(64, 12) passes on this build");
  Check(ThrowsType<W8a16FormatMismatch>([&] { CheckW8a16Meta(96, 12, path); }), "CheckW8a16Meta refuses group 96");
  Check(Throws([&] { CheckW8a16Meta(128, 11, path); }, "power of two", &msg), "CheckW8a16Meta refuses bias 11: " + msg);
  Check(Throws([&] { CheckW8a16MetaAgainst(128, 12, /*kernel_has_group=*/false, 12, path); }, "does not instantiate"),
        "a kernel without the group: refused");
  Check(Throws([&] { CheckW8a16MetaAgainst(128, 12, true, /*kernel_bias=*/13, path); }, "2^13"),
        "a kernel undoing another bias: refused, naming both");
  // Rule 3.
  const auto prob = [](int64_t N, int64_t K, int g, uint64_t wq, uint64_t ws) {
    return W8a16ShapeProblem("text.layers.2.moe.experts.0.down", N, K, g, wq, ws);
  };
  Check(prob(2048, 1536, 128, 2048ull * 1536, 2048ull * 1536 / 128 * 2).empty(), "expert down [2048, 1536] g128 sizes pass");
  Check(prob(3072, 2048, 64, 3072ull * 2048, 3072ull * 2048 / 64 * 2).empty(), "expert gate_up [3072, 2048] g64 sizes pass");
  Check(prob(2048, 11776, 128, 2048ull * 11776, 2048ull * 11776 / 128 * 2).empty(), "dense down K 11776 (92 groups) passes");
  const std::string ws_g64 = prob(2048, 1536, 128, 2048ull * 1536, 2048ull * 1536 / 64 * 2);
  Check(ws_g64.find(".w8a16.ws' is") != std::string::npos && ws_g64.find("experts.0.down") != std::string::npos,
        "rule 3: a .ws written at another group is refused, naming the tensor: " + ws_g64);
  const std::string wq_short = prob(2048, 1536, 128, 2048ull * 1536 - 16, 2048ull * 1536 / 128 * 2);
  Check(wq_short.find(".w8a16.wq' is") != std::string::npos, "rule 3: a short .wq is refused: " + wq_short);
  Check(!prob(2040, 1536, 128, 2040ull * 1536, 2040ull * 1536 / 128 * 2).empty(), "N % 16 != 0 is refused");
  Check(!prob(2048, 1600, 128, 2048ull * 1600, 2048ull * 1600 / 128 * 2).empty(), "K % group != 0 is refused");
}

void TestTinyContainer() {
  const char* env = std::getenv("R4DX_LFM2_TINY_W8A16_DIR");
  const std::string dir = env != nullptr && *env != '\0' ? std::string(env) : r4dx::ModelsPath("r4dx/lfm2/tiny_w8a16");
  const std::string path = dir + "/tiny_w8a16.r4dx";
  {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      std::printf("SKIP: %s absent (lane W1's tiny w8a16 container) -- section (d) not run\n", path.c_str());
      return;
    }
    std::fclose(f);
  }
  const Lfm2ContainerInfo info = InspectLfm2Container(path);
  Check(true, "InspectLfm2Container accepts the tiny w8a16 container's tensor table");
  Check(info.has_w8a16 && !info.has_trellis, "Inspect: has_w8a16, not trellis");
  const std::optional<W8a16Spec> spec = ParseW8a16Metadata(info.metadata, path);
  Check(spec && spec->group == fmt::kDefaultGroup && spec->scale_log2_bias == fmt::kScaleLog2Bias,
        "the container's quant.w8a16 parses: group 128, bias 12");
  if (!spec) return;
  Check(!Throws([&] { CheckW8a16Meta(spec->group, spec->scale_log2_bias, path); }, ""),
        "the container's format passes this build's kernel check");
  Check(!Throws([&] { RefuseW8a16Choice(spec, true, 1, path); }, ""), "--layout w8a16 --tp 1 is accepted");

  const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(path));
  const Lfm2Config& c = info.config;
  const int64_t Hd = c.hidden_size;
  int checked = 0, bad = 0, bf16_forms = 0;
  std::string first_bad;
  const auto lin = [&](const std::string& base, int64_t N, int64_t K) {
    ++checked;
    if (r.Has(base + ".bf16.w")) ++bf16_forms;
    if (!r.Has(base + fmt::kWqSuffix) || !r.Has(base + fmt::kWsSuffix)) {
      ++bad;
      if (first_bad.empty()) first_bad = base + ": a w8a16 tensor is missing";
      return;
    }
    const auto span = [&](const char* sfx) {
      const auto& m = r.Meta(base + sfx);
      return static_cast<uint64_t>(m.end - m.begin);
    };
    const std::string why = W8a16ShapeProblem(base, N, K, spec->group, span(fmt::kWqSuffix), span(fmt::kWsSuffix));
    if (!why.empty()) {
      ++bad;
      if (first_bad.empty()) first_bad = why;
    }
  };
  for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
    const std::string b = Lfm2LayerBase(i);
    if (c.IsAttentionLayer(i)) {
      lin(b + "attn.q", c.QDim(), Hd);
      lin(b + "attn.k", c.KvDim(), Hd);
      lin(b + "attn.v", c.KvDim(), Hd);
      lin(b + "attn.o", Hd, c.QDim());
    } else {
      lin(b + "conv.in_proj", c.ConvInProjRows(), Hd);
      lin(b + "conv.out_proj", Hd, c.conv_dim);
    }
    if (c.IsDenseLayer(i)) {
      lin(b + "mlp.gate_up", 2 * c.intermediate_size, Hd);
      lin(b + "mlp.down", Hd, c.intermediate_size);
    } else {
      for (int64_t e = 0; e < c.num_experts; ++e) {
        lin(Lfm2ExpertBase(i, e, "gate_up"), c.ExpertGateUpRows(), Hd);
        lin(Lfm2ExpertBase(i, e, "down"), Hd, c.moe_intermediate_size);
      }
    }
  }
  lin("lm_head", c.vocab_size, Hd);
  Check(bad == 0, "every body linear and the head (" + std::to_string(checked) +
                      ") carries .w8a16.wq / .ws at the loader's byte sizes" + (first_bad.empty() ? "" : ": " + first_bad));
  Check(bf16_forms == 0, "a --no-bf16 container: no linear carries a .bf16.w form (" + std::to_string(bf16_forms) + ")");
  Check(r.Has("text.embed_tokens") && !r.Has("text.embed_tokens" + std::string(fmt::kWqSuffix)),
        "the embedding stays bf16 (docs/w8a16.md 5.2)");
}

}  // namespace

int main() {
  try {
    TestLayoutEnum();
    TestParse();
    TestRefusals();
    TestTinyContainer();
  } catch (const std::exception& e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures != 0) {
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
