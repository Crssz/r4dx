// tests/model/test_gemma_rot_trellis.cpp -- docs/gemma4-plan.md M1-31: a ROTATED TRELLIS Gemma container through
// GemmaModel, against the same model unrotated bf16. GPU test (HIP device 1 only). NOT RUN BY ITS AUTHOR (no GPU work
// was allowed while it was written); exits 77 (SKIPPED) while its inputs are missing.
//
// Inputs (both written on the CPU by convert_gemma_trellis, which keeps its work dir when R4DX_GEMMA_TRELLIS_KEEP is
// set -- the CTest fixture does that):
//   <keep>/r4dx_test_gemma_trellis/ckpt-<kind>/       the tiny 6-layer gemma4_unified checkpoint (real head dims 256 / 512)
//   <keep>/r4dx_test_gemma_trellis/rot-<kind>.r4dx    its rotated trellis container (random ring words; the weights in the
//                                                      checkpoint are the UNFOLDED, 1%-perturbed reconstruction of them)
// The reference container is made here by the real r4dx-convert (R4DX_CONVERT_EXE): the checkpoint, unrotated, bf16 body
// and bf16 head. The two models therefore differ only by the ~1% per-tile weight perturbation, the bf16 rounding of the
// unfolded weights, and the w4a16 g128 tied head of the trellis container (heads are never trellis) -- so the gates are
// a similarity of the logits, not equality:
//   * the loader accepted the pair: HasRotation() && HasTrellis(), q2ab: Hadamard() (the o / down folds);
//   * default KV mode is bf16 (GemmaModelOptions{}.kv, R4DX_GEMMA_KV unset) and the load log says so;
//   * prefill + 4 decode steps: cosine(logits_rot_trellis, logits_ref) >= 0.95 on every row, all finite;
//   * KV fp8 (the container's placeholder descales 1.0): finite logits, cosine to the bf16-KV run >= 0.9 -- a smoke only,
//     the real fp8 gate is the M1-30 KL gate with the --kv-calib descales.
// The tolerances are unmeasured guesses (nothing here ran): if a row fails, look at the cosine of the ROTATED BF16 model
// against the unrotated one first (the rotation runtime alone), then the trellis kernel's own test_trellis_* results.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gemma_model.h"
#include "test_common.h"

#ifndef R4DX_CONVERT_EXE
#define R4DX_CONVERT_EXE ""
#endif

namespace fs = std::filesystem;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  return ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
}

bool Finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

// The logits rows of one model: Prefill(tokens[0..P)) then DecodeStep over the next kSteps tokens.
std::vector<std::vector<float>> Rows(GemmaModel& m, const std::vector<int32_t>& ids, int64_t P) {
  std::vector<std::vector<float>> rows;
  rows.push_back(m.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + P)));
  for (size_t i = static_cast<size_t>(P); i < ids.size(); ++i) rows.push_back(m.DecodeStep(ids[i]));
  return rows;
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_gemma_rot_trellis", []() -> int {
    const char* keep = std::getenv("R4DX_GEMMA_TRELLIS_KEEP");
    if (keep == nullptr || *keep == '\0') return r4dx_test::SkipMissing("R4DX_GEMMA_TRELLIS_KEEP (convert_gemma_trellis's kept dir)");
    const std::string convert_exe = R4DX_CONVERT_EXE;
    if (convert_exe.empty()) return r4dx_test::SkipMissing("r4dx-convert (not built in this configuration)");
    const fs::path dir = fs::path(keep) / "r4dx_test_gemma_trellis";

    // Token ids < 64 (the fixture's vocab); BOS-free, the model does not care.
    std::vector<int32_t> ids;
    for (int i = 0; i < 12; ++i) ids.push_back(static_cast<int32_t>((i * 7 + 3) % 64));
    const int64_t P = 8;

    Check(GemmaModelOptions{}.kv == GemmaKvMode::kBf16, "the default KV mode is bf16 (fp8 only when R4DX_GEMMA_KV=fp8)");

    for (const char* kind : {"q2ab", "q2a"}) {
      const fs::path ckpt = dir / (std::string("ckpt-") + kind);
      const fs::path rot = dir / (std::string("rot-") + kind + ".r4dx");
      if (!fs::exists(ckpt) || !fs::exists(rot)) return r4dx_test::SkipMissing(rot.string());
      const fs::path ref = dir / (std::string("ref-") + kind + ".r4dx");
      {
        std::error_code ec;
        fs::remove(ref, ec);
        const std::string cmd = "\"\"" + convert_exe + "\" --input \"" + ckpt.u8string() + "\" --output \"" + ref.u8string() +
                                "\" --threads 2 --layouts bf16 --lm-head bf16 > \"" + (dir / "ref.log").u8string() + "\" 2>&1\"";
        if (std::system(cmd.c_str()) != 0) {
          std::fprintf(stderr, "[FAIL] r4dx-convert of the unrotated bf16 reference failed (see %s)\n", (dir / "ref.log").string().c_str());
          return 1;
        }
      }
      std::printf("---- %s: rotated trellis vs the unrotated bf16 model ----\n", kind);

      std::vector<std::vector<float>> ref_rows;
      {
        GemmaModelOptions o;
        o.container_path = ref.u8string();
        o.layout = Layout::kBf16;
        o.max_ctx = 256;
        GemmaModel m = GemmaModel::Load(o);
        Check(!m.GetContainer().HasRotation() && !m.GetContainer().HasTrellis(), std::string(kind) + ": the reference is unrotated bf16");
        ref_rows = Rows(m, ids, P);
      }

      for (const bool fp8 : {false, true}) {
        GemmaModelOptions o;
        o.container_path = rot.u8string();
        o.layout = Layout::kTrellis;
        o.max_ctx = 256;
        if (fp8) o.kv = GemmaKvMode::kFp8;  // default (unset) is bf16
        GemmaModel m = GemmaModel::Load(o);
        if (!fp8) {
          Check(m.GetContainer().HasRotation() && m.GetContainer().HasTrellis(),
                std::string(kind) + ": GemmaModel accepted the rotated trellis container");
          Check(m.GetContainer().Rotation().spec.Hadamard() == (std::string(kind) == "q2ab"), std::string(kind) + ": option-A Hadamard iff q2ab");
          Check(m.KvMode() == GemmaKvMode::kBf16, std::string(kind) + ": KV mode is bf16 by default");
        }
        const std::vector<std::vector<float>> rows = Rows(m, ids, P);
        bool fin = true, cos_ok = true;
        double worst = 1.0;
        for (size_t r = 0; r < rows.size(); ++r) {
          fin = fin && Finite(rows[r]);
          const double c = Cosine(rows[r], ref_rows[r]);
          worst = std::min(worst, c);
          cos_ok = cos_ok && c >= (fp8 ? 0.90 : 0.95);
        }
        std::printf("  KV %s: %zu rows, worst cosine to the unrotated bf16 model %.4f\n", fp8 ? "fp8" : "bf16", rows.size(), worst);
        Check(fin, std::string(kind) + (fp8 ? " fp8" : " bf16") + " KV: every logit is finite");
        Check(cos_ok, std::string(kind) + (fp8 ? " fp8" : " bf16") + " KV: logits track the unrotated bf16 model");
        // Reset + the same prefill reproduces row P-1 (the trellis tickets are re-zeroed by Reset).
        if (!fp8) {
          m.Reset();
          const std::vector<float> again = m.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + P));
          Check(Cosine(again, rows[0]) > 0.9999, std::string(kind) + ": Reset() then the same prefill reproduces the first row");
        }
      }
    }
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
