// The layer-split runner (docs/qwen4exp-plan.md M1-17): a two-stage Qwen4ExpModel (layers [0, k) on one device with the host
// embedding / n-gram rows / PLE, layers [k, L) + the final mixer + lm_head on the other, the residual handed over through
// pinned host memory) must produce, bit for bit, the logits of the one-stage run: the hand-off copies bytes. Links r4dx_model.
//
// Variants (the tiny model has 4 layers: k in {1, 2, 3}):
//   * both stages on the test device (always; the emulation of the two-card run -- ctest sets HIP_VISIBLE_DEVICES=1);
//   * stage 0 on HIP ordinal 0 and stage 1 on ordinal 1 and the reverse, when two devices are visible (run the exe with
//     HIP_VISIBLE_DEVICES unset, under tools\with_lock.ps1 -Name gpu). With one device visible they are reported as not run.
// For every variant and every golden prompt (T 1, 7, 70, 300): the logits of EVERY row (capture), the final mixer output, the
// last-row Prefill logits, and 30 DecodeSteps after a Prefill(70), against the one-stage model on the test device.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "q4exp_model_util.h"

using namespace q4m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

bool SameBytes(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
}
bool SameBytes(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

struct Run {
  std::vector<std::vector<float>> prefill_last;       // per prompt
  std::vector<std::vector<uint16_t>> logits_all;      // per prompt, captured rows
  std::vector<std::vector<uint16_t>> mixer_all;       // per prompt
  std::vector<std::vector<float>> decode;             // 30 steps, concatenated
  std::vector<double> prefill_ms, stage0_ms, stage1_ms, handoff_ms;
  double decode_ms = 0, decode_stage0_ms = 0, decode_stage1_ms = 0, decode_handoff_ms = 0, decode_head_ms = 0;
};

Run Exercise(Qwen4ExpModel& model, const std::vector<Golden>& goldens) {
  Run r;
  for (const Golden& g : goldens) {
    CaptureStore s;
    model.Reset();
    model.SetCapture(s.Fn());
    r.prefill_last.push_back(model.Prefill(g.ids()));
    model.SetCapture(nullptr);
    r.logits_all.push_back(s.Get("logits").bf16);
    r.mixer_all.push_back(s.Get("mixer_out").bf16);
    const auto& t = model.LastTimings();
    r.prefill_ms.push_back(t.total_ms);
    r.stage0_ms.push_back(t.stage_ms[0]);
    r.stage1_ms.push_back(t.stage_ms[1]);
    r.handoff_ms.push_back(t.handoff_ms);
  }
  // decode: 30 steps after Prefill(first 70 of the last golden)
  const std::vector<int32_t>& ids = goldens.back().ids();
  model.Reset();
  model.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + 70));
  std::vector<float> all;
  for (int j = 70; j < 100; ++j) {
    const std::vector<float> d = model.DecodeStep(ids[static_cast<size_t>(j)]);
    const auto& t = model.LastTimings();
    r.decode_ms += t.total_ms;
    r.decode_stage0_ms += t.stage_ms[0];
    r.decode_stage1_ms += t.stage_ms[1];
    r.decode_handoff_ms += t.handoff_ms;
    r.decode_head_ms += t.head_ms;
    r.decode.push_back(d);
  }
  return r;
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
  const int prompts[] = {1, 7, 70, 300};
  if (!Exists(ContainerPath("bf16"))) {
    std::printf("SKIP: %s missing\n", ContainerPath("bf16").c_str());
    return kSkip;
  }
  std::vector<Golden> goldens;
  for (int T : prompts) {
    if (!Exists(GoldenPath(T))) {
      std::printf("SKIP: %s missing\n", GoldenPath(T).c_str());
      return kSkip;
    }
    goldens.emplace_back(T, GoldenPath(T));
  }
  const int visible = VisibleDevices();
  if (visible < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  const int dev = TestDevice();
  std::printf("%d HIP device(s) visible, test device %d\n", visible, dev);

  std::printf("\n== one stage, device %d ==\n", dev);
  Run ref;
  {
    auto model = LoadTiny("bf16");
    Check(model->NumStages() == 1, "the reference is one stage");
    ref = Exercise(*model, goldens);
    std::printf("  prefill T=300 %.1f ms, T=70 %.1f ms; decode %.2f ms/step (head %.2f ms)\n", ref.prefill_ms[3], ref.prefill_ms[2],
                ref.decode_ms / 30.0, ref.decode_head_ms / 30.0);
  }

  struct Variant {
    std::vector<int> devices;
    int64_t k;
    bool real;  // two distinct devices
  };
  std::vector<Variant> variants = {{{dev, dev}, 1, false}, {{dev, dev}, 2, false}, {{dev, dev}, 3, false}};
  int not_run = 0;
  if (visible >= 2) {
    variants.push_back({{0, 1}, 2, true});
    variants.push_back({{1, 0}, 2, true});
    variants.push_back({{0, 1}, 1, true});
    variants.push_back({{0, 1}, 3, true});
    variants.push_back({{0, 1}, 0, true});  // k from ChooseSplit
  } else {
    not_run = 5;
  }

  for (const Variant& v : variants) {
    std::printf("\n== two stages, devices {%d, %d}, split %s ==\n", v.devices[0], v.devices[1],
                v.k == 0 ? "auto (ChooseSplit)" : std::to_string(v.k).c_str());
    auto model = LoadTiny("bf16", false, v.devices, v.k);
    Check(model->NumStages() == 2, "two stages");
    if (v.k > 0) Check(model->SplitLayer() == v.k, "split layer");
    Check(model->StageDevice(0) == v.devices[0] && model->StageDevice(1) == v.devices[1], "stage devices");
    Check(model->StageEndLayer(0) == model->SplitLayer() && model->StageFirstLayer(1) == model->SplitLayer(), "stage ranges");
    std::printf("  k = %lld; stage weight bytes %.3f / %.3f MiB\n", static_cast<long long>(model->SplitLayer()),
                static_cast<double>(model->StageWeightBytes(0)) / (1 << 20), static_cast<double>(model->StageWeightBytes(1)) / (1 << 20));
    const Run got = Exercise(*model, goldens);
    bool all_same = true;
    for (size_t i = 0; i < goldens.size(); ++i) {
      const bool a = SameBytes(got.prefill_last[i], ref.prefill_last[i]);
      const bool b = SameBytes(got.logits_all[i], ref.logits_all[i]);
      const bool c = SameBytes(got.mixer_all[i], ref.mixer_all[i]);
      all_same = all_same && a && b && c;
      Check(a && b && c, "T" + std::to_string(goldens[i].T()) + ": two-stage Prefill differs from the one-stage run (last row " +
                             (a ? "same" : "DIFFERENT") + ", all-row logits " + (b ? "same" : "DIFFERENT") + ", mixer " +
                             (c ? "same" : "DIFFERENT") + ")");
    }
    bool dec_same = got.decode.size() == ref.decode.size();
    for (size_t i = 0; dec_same && i < got.decode.size(); ++i) dec_same = SameBytes(got.decode[i], ref.decode[i]);
    Check(dec_same, "the 30 decode steps differ from the one-stage run");
    std::printf("  Prefill T 1/7/70/300 (last row, every row's logits, mixer): %s; 30 decode steps: %s\n",
                all_same ? "bit-identical" : "DIFFERENT", dec_same ? "bit-identical" : "DIFFERENT");
    std::printf("  prefill T=300: total %.1f ms = stage0 %.1f + hand-off %.1f + stage1 %.1f (+ embed / head)\n", got.prefill_ms[3],
                got.stage0_ms[3], got.handoff_ms[3], got.stage1_ms[3]);
    std::printf("  prefill T=70 : total %.1f ms = stage0 %.1f + hand-off %.1f + stage1 %.1f\n", got.prefill_ms[2], got.stage0_ms[2],
                got.handoff_ms[2], got.stage1_ms[2]);
    std::printf("  decode (mean of 30): total %.2f ms = stage0 %.2f + hand-off %.3f + stage1 %.2f + head %.2f\n", got.decode_ms / 30.0,
                got.decode_stage0_ms / 30.0, got.decode_handoff_ms / 30.0, got.decode_stage1_ms / 30.0, got.decode_head_ms / 30.0);
  }
  if (not_run > 0) {
    std::printf("\n(%d real two-device variant(s) not run: only %d HIP device visible; run with HIP_VISIBLE_DEVICES unset)\n",
                not_run, visible);
  }
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
