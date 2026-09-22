// tests/vision/test_vision_tower.cpp -- r4dx::vision::VisionTower against the real checkpoint's
// own forward pass (tools/reference/vision_golden.py), on HIP device 1.
//
// Three cases, all three of the golden's: one 448x448 square image (grid 28x28, 784 patches, the
// deep case with every encoder block's output plus block 0's internal chain), one 613x409 ->
// 608x416 non-square image (26x38 patches -- the case where an h/w transposition in the position
// or rope index math actually shows), and both images in ONE forward (two `cu_seqlens` segments
// with different grids, which is the only thing that exercises the per-image attention scoping).
//
// The input is the golden's OWN `pixel_values`, not a re-preprocessed image: preprocessing already
// has its own BIT-EXACT test (tests/vision/test_preprocess.cpp), and feeding the golden's tensor
// here means a failure in this file is a device-side failure, never a preprocessing one.
//
// This test loads ONLY the container's vision.* tensors (~0.90 GiB), not the 27B text model -- it
// is a test of the tower, and a full Model::Load would cost ~18 s and ~16 GiB for no extra
// coverage. SKIPs (77) when either the container or the golden is absent.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "../model/test_container_path.h"  // r4dx_test::ProductionTargetPath / RunGuardedMain
#include "vision_test_common.h"
#include "vision_tower.h"
#include "vision_weights.h"

using namespace r4dx_vision_test;
using r4dx::vision::GridThw;
using r4dx::vision::VisionEncodeStats;
using r4dx::vision::VisionTower;
using r4dx::vision::VisionWeights;

namespace {

// The production container matching this build's w4a16 group (v6 at 64, v3 at 128 --
// tests/model/test_container_path.h). The vision.* tensors this test reads are bf16 and
// group-independent, so either container would do numerically; the group-matched one is what the
// server/CLI defaults would load next to it.
const char* kContainerPath = r4dx_test::ProductionTargetPath();

// R4DX_VISION_GOLDEN_DIR overrides where the golden tensors are read from. Used to diff this
// implementation against a SECOND reference run -- `vision_golden.py --attn-impl sdpa` into a
// scratch directory -- which is how docs/vision.md's "the disagreement is the tower's own bf16
// conditioning, not an r4dx error" claim was measured. Unset in every ctest run, which always
// reads the committed eager golden.
std::string GoldenPath(const char* name) {
  // The MSVC CRT headers mark getenv deprecated (thread-safety of the returned pointer); this is
  // one read of one variable on the main thread, exactly like src/model/linear.cpp's and
  // src/cli/main.cpp's own getenv call sites.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  const char* dir = std::getenv("R4DX_VISION_GOLDEN_DIR");
#pragma clang diagnostic pop
  if (dir != nullptr && dir[0] != '\0') return std::string(dir) + "/" + name;
  return std::string(R4DX_SOURCE_DIR) + "/tools/reference/golden_out/" + name;
}

// The container's `__metadata__` block, which SafetensorsReader deliberately discards -- the same
// small, deliberate re-parse src/model/container.cpp's own ReadMetadata does.
nlohmann::json ReadContainerMetadata(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  std::string header_json(static_cast<size_t>(header_len), '\0');
  f.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!f) throw std::runtime_error("truncated header in " + path);
  return nlohmann::json::parse(header_json).at("__metadata__");
}

double RelL2(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return std::sqrt(num) / std::max(1e-12, std::sqrt(den));
}

// max |got - ref|, plus what the two values actually were there and how big the reference's own
// largest magnitude is. A bare max_abs is unreadable on this tower: the vision encoder develops
// "massive activation" channels in the thousands by block 26, where ONE bf16 ulp is already tens
// or hundreds, so "max_abs = 2496" means nothing until you know whether the reference value there
// was 3 or 300000.
struct AbsStats {
  double max_abs = 0.0;
  double got_at = 0.0, ref_at = 0.0;
  double ref_absmax = 0.0;
  double ulps_at = 0.0;  // max_abs expressed in bf16 ulps of the reference value there
};

AbsStats MaxAbs(const std::vector<float>& got, const std::vector<float>& ref) {
  AbsStats s;
  for (size_t i = 0; i < ref.size(); ++i) {
    s.ref_absmax = std::max(s.ref_absmax, std::abs(static_cast<double>(ref[i])));
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    if (d > s.max_abs) {
      s.max_abs = d;
      s.got_at = got[i];
      s.ref_at = ref[i];
    }
  }
  // One bf16 ulp at |ref_at|: bf16 keeps 8 total mantissa bits, so ulp = 2^(exponent-7).
  if (s.ref_at != 0.0) {
    const double e = std::floor(std::log2(std::abs(s.ref_at)));
    s.ulps_at = s.max_abs / std::pow(2.0, e - 7.0);
  }
  return s;
}

// One golden case: run the tower over the golden's pixel_values with a trace attached, then diff
// every tensor the golden carries that the tower emitted.
struct CaseResult {
  bool ran = false;
  int compared = 0;
};

CaseResult RunCase(const VisionWeights& weights, VisionTower* tower, const char* golden_file,
                   const char* label) {
  CaseResult res;
  const std::string path = GoldenPath(golden_file);
  if (!FileExists(path)) {
    std::fprintf(stderr, "[SKIP-CASE] %s not present\n", path.c_str());
    return res;
  }
  r4dx_convert::SafetensorsReader golden(r4dx_convert::Utf8ToWide(path));

  // grid_thw, then pixel_values, both straight out of the golden.
  const std::vector<int64_t> grid_raw = ReadInt(golden, "image_grid_thw");
  std::vector<GridThw> grids;
  for (size_t i = 0; i + 2 < grid_raw.size(); i += 3) {
    GridThw g;
    g.t = grid_raw[i];
    g.h = grid_raw[i + 1];
    g.w = grid_raw[i + 2];
    grids.push_back(g);
  }
  const std::vector<float> pixel_values = ReadFloat(golden, "pixel_values");
  const int64_t patch_dim = weights.config.PatchDim();
  const int64_t total_patches = static_cast<int64_t>(pixel_values.size()) / patch_dim;

  std::map<std::string, std::vector<float>> got;
  r4dx::vision::VisionTrace trace = [&got](const std::string& name, const void* data, int64_t n,
                                            bool is_fp32) {
    std::vector<float> v(static_cast<size_t>(n));
    if (is_fp32) {
      std::memcpy(v.data(), data, static_cast<size_t>(n) * 4);
    } else {
      const auto* src = static_cast<const uint16_t*>(data);
      for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = r4dx::core::Bf16ToFloat(src[i]);
    }
    got[name] = std::move(v);
  };

  r4dx::core::DeviceBuffer<uint16_t> out;
  VisionEncodeStats stats;
  tower->Encode(weights, pixel_values.data(), total_patches, grids, &out, &stats, &trace);

  std::printf("\n[%s] grids=%d patches=%lld merged=%lld segments=%lld max_seqlen=%lld "
              "encode=%.1f ms scratch=%.1f MiB\n",
              label, static_cast<int>(grids.size()), static_cast<long long>(stats.total_patches),
              static_cast<long long>(stats.merged_tokens),
              static_cast<long long>(stats.num_segments),
              static_cast<long long>(stats.max_seqlen), stats.encode_ms,
              static_cast<double>(stats.scratch_bytes) / (1024.0 * 1024.0));
  res.ran = true;

  // ---- tolerance policy (docs/vision.md "Why the deep-block disagreement is not an r4dx error")
  //
  // Three bands, and the wide one is MEASURED, not chosen:
  //
  //  * fp32 index tables (pos_embeds, rope_cos/sin): computed in fp32/double on the host against
  //    the reference's own fp32, so they are held to 1e-6 relative / 1e-5 absolute. pos_embeds is
  //    in fact exact.
  //  * the front end and block 0's whole internal chain, plus block_00_output: 1e-2. Nothing has
  //    amplified yet at this depth, so a real bug cannot hide under bf16 noise here -- and the
  //    measured worst is 4.4e-3, i.e. one to two bf16 ulps.
  //  * blocks 1..26, last_hidden_state and merger_output: 0.12. That is not slack. The reference's
  //    OWN two attention implementations disagree with EACH OTHER by 7.8e-2 at block_26_output and
  //    6.7e-2 at merger_output on this exact image (`vision_golden.py --attn-impl sdpa` into a
  //    scratch dir, then R4DX_VISION_GOLDEN_DIR at it -- docs/vision.md carries the full table),
  //    because transformers' `eager_attention_forward` rounds the attention scores to bf16 before
  //    the softmax and the probabilities to bf16 before the P@V matmul, while both SDPA and
  //    r4d_attn_vit_h72_bf16 keep them in fp32/f16. r4dx lands INSIDE that spread -- 6.7e-2 from
  //    the eager golden, 5.3e-2 from the sdpa one, i.e. closest to the fp32-accumulating
  //    implementation, which is what its own kernel is. A tighter bound here would not measure
  //    correctness, it would measure which bf16 attention kernel the golden happened to use.
  //
  // The real regression detector at depth is PerBlockLocalization below, which runs each block
  // from the REFERENCE's own input so no amplification is in the number at all.
  struct Tol {
    const char* name;
    double rel_l2;
    double max_abs;  // 0 = not checked
  };
  const Tol tolerances[] = {
      {"pos_embeds", 1e-6, 0.0},
      {"rope_cos", 0.0, 1e-5},
      {"rope_sin", 0.0, 1e-5},
  };
  constexpr double kShallowRelL2 = 1e-2;
  constexpr double kDeepRelL2 = 0.12;

  // Compare in a deterministic, meaningful order: front end, then block 0's chain, then the block
  // outputs in depth order, then the merger. Anything the golden has that this list misses would
  // be silently unchecked, so the loop below reports the count and main() asserts it.
  std::vector<std::string> order = {"patch_embed_out", "pos_embeds",      "block_input",
                                     "rope_cos",       "rope_sin",        "block0_norm1_out",
                                     "block0_attn_qkv_raw", "attn_q_post_rope", "attn_k_post_rope",
                                     "block0_attn_proj_out", "block0_norm2_out",
                                     "block0_mlp_fc1_out", "block0_mlp_fc2_out"};
  for (int b = 0; b < 27; ++b) {
    char name[32];
    std::snprintf(name, sizeof(name), "block_%02d_output", b);
    order.push_back(name);
  }
  order.push_back("last_hidden_state");
  order.push_back("merger_output");

  for (const std::string& name : order) {
    if (!golden.Has(name)) continue;  // the two shallow cases only dump the front end + merger
    if (got.find(name) == got.end()) {
      std::fprintf(stderr, "  %-22s golden has it, the tower never emitted it\n", name.c_str());
      ++g_failures;
      continue;
    }
    const std::vector<float> ref = ReadFloat(golden, name);
    const std::vector<float>& g = got[name];
    if (g.size() != ref.size()) {
      std::fprintf(stderr, "  %-22s size mismatch: got %zu, golden %zu\n", name.c_str(), g.size(),
                   ref.size());
      ++g_failures;
      continue;
    }
    const double rel = RelL2(g, ref);
    const AbsStats abs_err = MaxAbs(g, ref);
    // "Deep" = anything whose input has already been through an attention block: blocks 1..26 and
    // the two tensors downstream of them.
    const bool deep = (name == "last_hidden_state") || (name == "merger_output") ||
                       (name.rfind("block_", 0) == 0 && name != "block_00_output");
    double rel_limit = deep ? kDeepRelL2 : kShallowRelL2;
    double abs_limit = 0.0;
    for (const Tol& t : tolerances) {
      if (name == t.name) {
        rel_limit = t.rel_l2;
        abs_limit = t.max_abs;
      }
    }
    const bool ok = (rel_limit > 0.0 ? rel <= rel_limit : true) &&
                     (abs_limit > 0.0 ? abs_err.max_abs <= abs_limit : true);
    std::printf("  %-22s rel_l2=%.3e max_abs=%.3e (got %.4g vs ref %.4g, %.1f bf16 ulp; "
                "ref |max|=%.4g)  %s\n",
                name.c_str(), rel, abs_err.max_abs, abs_err.got_at, abs_err.ref_at,
                abs_err.ulps_at, abs_err.ref_absmax, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
    ++res.compared;
  }
  return res;
}

// Per-block localization. The plain run above measures each block output against the reference
// with EVERY earlier block's error already baked into its input, so a slowly rising rel_l2 there
// cannot distinguish "one block is wrong" from "27 blocks each contribute their own bf16 noise".
// This re-runs the same encode with the residual stream overwritten by the reference's own
// block_{b-1}_output before every block, so each printed number is that ONE block's own error.
void PerBlockLocalization(const VisionWeights& weights, VisionTower* tower) {
  const std::string path = GoldenPath("vision_tower.safetensors");
  r4dx_convert::SafetensorsReader golden(r4dx_convert::Utf8ToWide(path));
  const std::vector<int64_t> grid_raw = ReadInt(golden, "image_grid_thw");
  std::vector<GridThw> grids;
  for (size_t i = 0; i + 2 < grid_raw.size(); i += 3) {
    grids.push_back(GridThw{grid_raw[i], grid_raw[i + 1], grid_raw[i + 2]});
  }
  const std::vector<float> pixel_values = ReadFloat(golden, "pixel_values");
  const int64_t total_patches =
      static_cast<int64_t>(pixel_values.size()) / weights.config.PatchDim();

  // The reference input of every block, as raw bf16 (block_input for block 0, block_{b-1}_output
  // after that) -- injected verbatim, so no re-rounding happens on the way in.
  std::vector<std::vector<uint16_t>> block_in(static_cast<size_t>(weights.config.depth));
  for (int64_t b = 0; b < weights.config.depth; ++b) {
    char name[32];
    if (b == 0) {
      std::snprintf(name, sizeof(name), "block_input");
    } else {
      std::snprintf(name, sizeof(name), "block_%02d_output", static_cast<int>(b - 1));
    }
    const auto& m = golden.Meta(name);
    const auto* src = reinterpret_cast<const uint16_t*>(golden.Data(name));
    block_in[static_cast<size_t>(b)].assign(src, src + m.ElemCount());
  }

  std::map<std::string, std::vector<float>> got;
  r4dx::vision::VisionTrace trace = [&got](const std::string& name, const void* data, int64_t n,
                                            bool is_fp32) {
    if (is_fp32) return;
    std::vector<float> v(static_cast<size_t>(n));
    const auto* src = static_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = r4dx::core::Bf16ToFloat(src[i]);
    got[name] = std::move(v);
  };
  r4dx::vision::VisionPreBlock inject = [&block_in](int64_t b, void* dev_x, int64_t elems) {
    const std::vector<uint16_t>& src = block_in[static_cast<size_t>(b)];
    if (static_cast<int64_t>(src.size()) != elems) return false;
    R4DX_HIP_CHECK(hipMemcpy(dev_x, src.data(), src.size() * 2, hipMemcpyHostToDevice));
    return true;
  };

  r4dx::core::DeviceBuffer<uint16_t> out;
  tower->Encode(weights, pixel_values.data(), total_patches, grids, &out, nullptr, &trace, &inject);

  std::printf("\n[per-block localization] each block run from the REFERENCE's own input\n");
  double worst = 0.0;
  int worst_block = -1;
  for (int b = 0; b < static_cast<int>(weights.config.depth); ++b) {
    char name[32];
    std::snprintf(name, sizeof(name), "block_%02d_output", b);
    const std::vector<float> ref = ReadFloat(golden, name);
    const double rel = RelL2(got[name], ref);
    const AbsStats a = MaxAbs(got[name], ref);
    std::printf("  block %02d own rel_l2=%.3e  max_abs=%.3e (%.1f bf16 ulp, ref |max|=%.4g)\n", b,
                rel, a.max_abs, a.ulps_at, a.ref_absmax);
    if (rel > worst) {
      worst = rel;
      worst_block = b;
    }
  }
  std::printf("[per-block localization] worst single block: %02d at rel_l2=%.3e\n", worst_block,
              worst);
  // No single block may be materially worse than its peers: that is the signature of a real bug
  // (a wrong constant, a wrong activation, a mis-scoped attention) as opposed to bf16 noise, which
  // is uniform across identically-shaped blocks. Measured worst against the committed eager
  // golden: 1.18e-2 (block 02); against an sdpa reference run: 7.1e-3 -- the same eager score
  // rounding the tolerance comment in RunCase explains. 1.5e-2 is that worst case plus margin.
  CHECK(worst < 1.5e-2);
}

}  // namespace

static int RunTest() {
  if (!FileExists(kContainerPath)) return SkipMissing(kContainerPath);
  if (!FileExists(GoldenPath("vision_tower.safetensors"))) {
    return SkipMissing(GoldenPath("vision_tower.safetensors"));
  }
  R4DX_HIP_CHECK(hipSetDevice(0));  // HIP_VISIBLE_DEVICES=1 remaps physical device 1 to index 0

  const nlohmann::json metadata = ReadContainerMetadata(kContainerPath);
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(kContainerPath));
  if (!r4dx::vision::HasVisionTensors(reader)) {
    std::fprintf(stderr, "[SKIP] %s carries no vision.* tensors\n", kContainerPath);
    return kSkipReturnCode;
  }
  const VisionWeights weights =
      r4dx::vision::LoadVisionWeights(reader, metadata.at("model_config").at("vision_config"));
  std::printf("[test_vision_tower] loaded %lld vision tensors, %.4f GiB, from %s\n",
              static_cast<long long>(weights.tensor_count),
              static_cast<double>(weights.bytes) / (1024.0 * 1024.0 * 1024.0), kContainerPath);
  CHECK(weights.tensor_count == 333);

  VisionTower tower;
  int total_compared = 0;
  const CaseResult a = RunCase(weights, &tower, "vision_tower.safetensors", "square 448x448");
  const CaseResult b =
      RunCase(weights, &tower, "vision_tower_nonsquare.safetensors", "non-square 613x409");
  const CaseResult c =
      RunCase(weights, &tower, "vision_tower_two_image.safetensors", "two images, 2 segments");
  PerBlockLocalization(weights, &tower);
  total_compared = a.compared + b.compared + c.compared;
  CHECK(a.ran);
  CHECK(b.ran);
  CHECK(c.ran);
  // 42 tensors for the deep case (13 front-end/block0 + 27 block outputs + last_hidden_state +
  // merger_output) and 6 for each shallow case. Asserted so a golden regenerated without the
  // block captures cannot quietly turn this into a two-tensor test.
  std::printf("\n[test_vision_tower] %d tensors compared across 3 cases\n", total_compared);
  CHECK(total_compared >= 50);

  if (g_failures != 0) {
    std::fprintf(stderr, "[test_vision_tower] FAILED (%d checks)\n", g_failures);
    return 1;
  }
  std::printf("[test_vision_tower] OK\n");
  return 0;
}

// An exception escaping RunTest (a container the loader refuses, most often) is a FAIL with its
// message, not a 0xc0000409 crash -- see RunGuardedMain in tests/model/test_container_path.h.
int main() { return r4dx_test::RunGuardedMain("test_vision_tower", RunTest); }
