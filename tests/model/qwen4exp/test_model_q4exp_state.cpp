// Qwen4ExpModel sequence behaviour on the tiny container (docs/qwen4exp-plan.md M1-16 / M1-18). Links r4dx_model.
//
//   1. prefill + decode == one shot: Prefill(first 70 tokens of the T300 golden prompt) then 230 teacher-forced
//      DecodeSteps (crossing the EOS segment boundaries at 120 / 201, QSA block boundaries and the 67-token dense-exact
//      limit of the tiny indexer) reproduce the logits rows 69..299 of the HF golden, and of this model's own one-shot
//      Prefill(300), within bf16 noise.
//   2. chunk splits: the same 300 tokens fed as 100+100+100, 255+45, 1+299 (separate Prefill calls) give every row's logits
//      within noise of the one-shot run and of the golden.
//   3. prefix reuse: Prefill(A), SaveCheckpoint, Prefill(B) + decode steps, RestoreCheckpoint, Prefill(C) is BIT-IDENTICAL to
//      a cold Reset, Prefill(A), Prefill(C) (same chunking), for A ending mid QSA block and right after an EOS token (the
//      n-gram history), restored twice from the same checkpoint; and within noise of the one-shot cold Prefill(A + C).
//   4. Reset reproduces a prefill bit for bit; DecodeStepGreedy / DecodeStepSampled agree with DecodeStep + the canonical
//      sampler.
// Bounds are max|err| / rms(ref) of the logits rows, set from the noise observed (see kRowTol and friends).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "q4exp_model_util.h"
#include "r4dx/kernels/sampler.hpp"
#include "r4dx/kernels/summary_sampler.hpp"

using namespace q4m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

constexpr int64_t kVocab = 4096;

// Row-by-row statistics over a [rows, vocab] logits comparison.
struct RowAgg {
  double max_norm = 0, mean_norm_sum = 0, max_abs = 0;
  int rows = 0, argmax_diff = 0;
  void Add(const FStats& s, bool argmax_equal) {
    max_norm = std::max(max_norm, s.max_norm);
    max_abs = std::max(max_abs, s.max_abs);
    mean_norm_sum += s.mean_norm;
    ++rows;
    argmax_diff += argmax_equal ? 0 : 1;
  }
  double mean_norm() const { return rows > 0 ? mean_norm_sum / rows : 0; }
  void Print(const char* what) const {
    std::printf("  %-44s rows %4d  max|err|/rms %.4f  mean|err|/rms %.5f  max|err| %.4g  argmax differs in %d rows\n", what,
                rows, max_norm, mean_norm(), max_abs, argmax_diff);
  }
};

// The model's logits for rows [r0, r1) of a Prefill with the capture installed, as fp32 [rows, vocab].
std::vector<float> RowsOf(const CaptureStore& s, int r0, int r1) {
  const Captured& c = s.Get("logits");
  return WidenBf16(c.bf16.data() + static_cast<int64_t>(r0) * kVocab, static_cast<int64_t>(r1 - r0) * kVocab);
}

bool Identical(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int RunTest();
int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}

int RunTest() {
  Unbuffer();
  if (!Exists(ContainerPath("bf16")) || !Exists(GoldenPath(300)) || !Exists(GoldenPath(70))) {
    std::printf("SKIP: tiny fixtures missing under %s\n", TinyDir().c_str());
    return kSkip;
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  const Golden g300(300, GoldenPath(300)), g70(70, GoldenPath(70));
  const std::vector<int32_t>& ids = g300.ids();
  int64_t n = 0;
  const uint16_t* gl = g300.Bf16("logits", &n);
  Check(n == 300 * kVocab, "golden logits shape");
  const auto golden_row = [&](int r) { return WidenBf16(gl + static_cast<int64_t>(r) * kVocab, kVocab); };

  std::printf("device %d, bf16 container, GDN prefill through the recurrent kernels (the accurate reference path)\n", TestDevice());
  auto model = LoadTiny("bf16", /*checkpoint=*/true, {}, 0, 4096, /*gdn_recurrent=*/true);

  // ---- the one-shot baseline (captured logits of all rows) ---------------------------------------------------------------
  CaptureStore oneshot;
  model->Reset();
  model->SetCapture(oneshot.Fn());
  const std::vector<float> oneshot_last = model->Prefill(ids);
  model->SetCapture(nullptr);
  const auto oneshot_ms = model->LastTimings().total_ms;
  std::printf("one-shot Prefill(300): %.1f ms\n", oneshot_ms);
  Check(model->PositionCount() == 300, "position after the one-shot prefill");
  {
    RowAgg a;
    for (int r = 0; r < 300; ++r) {
      const std::vector<float> got = RowsOf(oneshot, r, r + 1);
      const std::vector<float> ref = golden_row(r);
      a.Add(CompareF32(got.data(), ref.data(), kVocab), ArgMax(got.data(), kVocab) == ArgMax(ref.data(), kVocab));
    }
    a.Print("one-shot Prefill(300) vs golden");
  }

  // ---- 1. prefill 70 + 230 decode steps -------------------------------------------------------------------------------------
  {
    std::printf("\n[1] Prefill(70) + 230 teacher-forced DecodeSteps\n");
    model->Reset();
    const std::vector<int32_t> head(ids.begin(), ids.begin() + 70);
    const std::vector<float> last = model->Prefill(head);
    RowAgg vs_golden, vs_oneshot;
    {
      const std::vector<float> ref = golden_row(69);
      vs_golden.Add(CompareF32(last.data(), ref.data(), kVocab), ArgMax(last.data(), kVocab) == ArgMax(ref.data(), kVocab));
      const std::vector<float> os = RowsOf(oneshot, 69, 70);
      vs_oneshot.Add(CompareF32(last.data(), os.data(), kVocab), ArgMax(last.data(), kVocab) == ArgMax(os.data(), kVocab));
    }
    double decode_ms = 0;
    for (int j = 70; j < 300; ++j) {
      const std::vector<float> got = model->DecodeStep(ids[static_cast<size_t>(j)]);
      decode_ms += model->LastTimings().total_ms;
      const std::vector<float> ref = golden_row(j);
      vs_golden.Add(CompareF32(got.data(), ref.data(), kVocab), ArgMax(got.data(), kVocab) == ArgMax(ref.data(), kVocab));
      const std::vector<float> os = RowsOf(oneshot, j, j + 1);
      vs_oneshot.Add(CompareF32(got.data(), os.data(), kVocab), ArgMax(got.data(), kVocab) == ArgMax(os.data(), kVocab));
    }
    Check(model->PositionCount() == 300, "position after the decode steps");
    vs_golden.Print("prefill(70)+decode vs golden rows 69..299");
    vs_oneshot.Print("prefill(70)+decode vs one-shot rows 69..299");
    std::printf("  decode: %.2f ms/step over 230 steps (last step: stage0 %.2f ms, ple wait %.3f ms, head %.2f ms)\n",
                decode_ms / 230.0, model->LastTimings().stage_ms[0], model->LastTimings().ple_wait_ms, model->LastTimings().head_ms);
  }

  // ---- 2. chunk splits -------------------------------------------------------------------------------------------------------
  {
    std::printf("\n[2] chunk splits of the 300 tokens (separate Prefill calls)\n");
    const std::vector<std::vector<int>> splits = {{100, 100, 100}, {255, 45}, {1, 299}, {256, 44}, {7, 64, 129, 100}};
    for (const auto& split : splits) {
      model->Reset();
      CaptureStore s;
      model->SetCapture(s.Fn());
      int off = 0;
      std::vector<float> last;
      for (int len : split) {
        const std::vector<int32_t> part(ids.begin() + off, ids.begin() + off + len);
        last = model->Prefill(part);
        off += len;
      }
      model->SetCapture(nullptr);
      Check(off == 300 && model->PositionCount() == 300, "split covers the prompt");
      std::string name;
      for (int len : split) name += (name.empty() ? "" : "+") + std::to_string(len);
      RowAgg vs_one, vs_golden;
      for (int r = 0; r < 300; ++r) {
        const std::vector<float> got = RowsOf(s, r, r + 1);
        const std::vector<float> os = RowsOf(oneshot, r, r + 1);
        const std::vector<float> ref = golden_row(r);
        vs_one.Add(CompareF32(got.data(), os.data(), kVocab), ArgMax(got.data(), kVocab) == ArgMax(os.data(), kVocab));
        vs_golden.Add(CompareF32(got.data(), ref.data(), kVocab), ArgMax(got.data(), kVocab) == ArgMax(ref.data(), kVocab));
      }
      vs_one.Print((name + " vs one-shot").c_str());
      vs_golden.Print((name + " vs golden").c_str());
    }
  }

  // ---- 3. prefix reuse -------------------------------------------------------------------------------------------------------
  {
    std::printf("\n[3] prefix reuse (SaveCheckpoint / RestoreCheckpoint)\n");
    const std::vector<int32_t>& other = g70.ids();
    const std::vector<int32_t> C1(other.begin(), other.begin() + 60), C2(other.begin() + 10, other.begin() + 50);
    // A ends mid QSA block (102), on a block boundary (100) and right after the EOS token at position 120 (121)
    for (int a_len : {100, 102, 121}) {
      const std::vector<int32_t> A(ids.begin(), ids.begin() + a_len);
      const std::vector<int32_t> B(ids.begin() + a_len, ids.begin() + a_len + 80);
      const auto cold = [&](const std::vector<int32_t>& C, std::vector<float>* decoded) {
        model->Reset();
        model->Prefill(A);
        std::vector<float> last = model->Prefill(C);
        for (int i = 0; i < 5; ++i) {
          const std::vector<float> d = model->DecodeStep(ids[static_cast<size_t>(200 + i)]);
          decoded->insert(decoded->end(), d.begin(), d.end());
        }
        return last;
      };
      std::vector<float> cold1_dec, cold2_dec;
      const std::vector<float> cold1 = cold(C1, &cold1_dec);
      const std::vector<float> cold2 = cold(C2, &cold2_dec);

      model->Reset();
      model->Prefill(A);
      model->SaveCheckpoint();
      for (int rep = 0; rep < 2; ++rep) {
        // continue away from the checkpoint, then come back
        model->Prefill(B);
        for (int i = 0; i < 3; ++i) model->DecodeStep(ids[static_cast<size_t>(150 + i)]);
        model->RestoreCheckpoint();
        Check(model->PositionCount() == a_len, "position restored to the end of A");
        const std::vector<float>& want = rep == 0 ? cold1 : cold2;
        const std::vector<float>& want_dec = rep == 0 ? cold1_dec : cold2_dec;
        std::vector<float> dec;
        const std::vector<float> got = model->Prefill(rep == 0 ? C1 : C2);
        for (int i = 0; i < 5; ++i) {
          const std::vector<float> d = model->DecodeStep(ids[static_cast<size_t>(200 + i)]);
          dec.insert(dec.end(), d.begin(), d.end());
        }
        const std::string tag = "A=" + std::to_string(a_len) + " restore #" + std::to_string(rep + 1);
        std::printf("  %-24s Prefill(C) after restore vs cold: %s; 5 decode steps: %s\n", tag.c_str(),
                    Identical(got, want) ? "bit-identical" : "DIFFERENT", Identical(dec, want_dec) ? "bit-identical" : "DIFFERENT");
        Check(Identical(got, want), tag + ": logits after the restore differ from the cold run");
        Check(Identical(dec, want_dec), tag + ": decode after the restore differs from the cold run");
        // the next restore comes from the same checkpoint: Restore must not consume it
      }
      // one-shot cold of A + C1 (a different chunking: within noise)
      model->Reset();
      std::vector<int32_t> AC = A;
      AC.insert(AC.end(), C1.begin(), C1.end());
      const std::vector<float> one = model->Prefill(AC);
      const FStats fs = CompareF32(cold1.data(), one.data(), kVocab);
      std::printf("  A=%d: Prefill(A)+Prefill(C) vs one-shot Prefill(A+C): max/rms %.4f mean/rms %.5f\n", a_len, fs.max_norm,
                  fs.mean_norm);
    }
    // a checkpoint is refused right after a decode step and before any prefill
    model->Reset();
    bool threw = false;
    try {
      model->SaveCheckpoint();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "SaveCheckpoint before any token must throw");
    model->Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + 10));
    model->DecodeStep(5);
    threw = false;
    try {
      model->SaveCheckpoint();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "SaveCheckpoint after a decode step must throw");
  }

  // ---- 4. Reset, greedy, sampled ------------------------------------------------------------------------------------------------
  {
    std::printf("\n[4] Reset determinism, greedy and sampled steps\n");
    const std::vector<int32_t>& p70 = g70.ids();
    model->Reset();
    const std::vector<float> first = model->Prefill(p70);
    model->DecodeStep(11);
    model->Reset();
    const std::vector<float> second = model->Prefill(p70);
    std::printf("  Reset + the same prefill: %s\n", Identical(first, second) ? "bit-identical" : "DIFFERENT");
    Check(Identical(first, second), "Reset must reproduce a prefill bit for bit");

    model->SaveCheckpoint();
    const int32_t tok = 1234;
    const std::vector<float> logits = model->DecodeStep(tok);
    const int64_t want = ArgMax(logits.data(), kVocab);
    model->RestoreCheckpoint();
    const int32_t greedy = model->DecodeStepGreedy(tok);
    std::printf("  DecodeStepGreedy %d, argmax of DecodeStep %lld\n", greedy, static_cast<long long>(want));
    Check(greedy == want, "DecodeStepGreedy must equal the argmax of DecodeStep");

    // sampled: with the same seed the summary path must give the canonical sampler's token for the step's own logits
    const r4dx::kernels::SampleParams configs[] = {
        {0.0f, 0, 1.0f, 0.0f, 0}, {1.0f, 0, 1.0f, 0.0f, 0}, {0.7f, 40, 0.9f, 0.0f, 0}, {1.3f, 0, 1.0f, 0.05f, 0}, {1.0f, 1, 1.0f, 0.0f, 0}};
    int sampled_checked = 0;
    for (const auto& params : configs) {
      for (uint64_t seed : {1ull, 2ull, 3ull, 4ull, 5ull, 6ull}) {
        model->RestoreCheckpoint();
        std::mt19937_64 rng(seed);
        std::mt19937_64 rng_copy = rng;
        const int32_t got = model->DecodeStepSampled(tok, params, rng);
        int32_t expect;
        if (params.temperature <= 0.0f) {
          expect = static_cast<int32_t>(want);
        } else {
          const double u = r4dx::kernels::DrawUniform01(rng_copy);
          expect = r4dx::kernels::SampleCanonical(logits.data(), kVocab, params, u);
        }
        Check(got == expect, "DecodeStepSampled (T=" + std::to_string(params.temperature) + ", seed " + std::to_string(seed) +
                                 ") returned " + std::to_string(got) + ", the canonical sampler " + std::to_string(expect));
        ++sampled_checked;
      }
    }
    std::printf("  %d sampled steps equal the canonical sampler on the step's own logits (%lld unresolved summary row(s))\n",
                sampled_checked, static_cast<long long>(model->SampledFallbackRows()));
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
