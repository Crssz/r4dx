// LoadTextModel on a qwen4_exp container (docs/qwen4exp-plan.md 4.1, M1-16): the dispatch in tp_model.cpp, the TextModel wrapper
// (Qwen4ExpLocalTextModel) and MakeQwen4ExpModelOptions. Links r4dx_model.
//
//   * LoadTextModel(tiny bf16 container) returns a Qwen4ExpLocalTextModel whose Config / ModelId / TpWorld / NumLoadedLayers /
//     Vram / capabilities describe the model, and whose Prefill / DecodeStep / DecodeStepGreedy / DecodeStepSampled /
//     Save / Restore / Reset behave like the underlying Qwen4ExpModel (bit-identical logits);
//   * MTP, DFlash, vision and the step profilers throw std::runtime_error naming the feature;
//   * the requests this arch refuses (--tp 2, --pp 2, MTP, a drafter, --vision on, a partial layer load) are refused before
//     any weight loads, with a message that names the request.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "qwen4exp_local_text_model.h"
#include "q4exp_model_util.h"
#include "text_model.h"

using namespace q4m;
using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
  std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what.c_str());
}

template <class Ex, class F>
bool ThrowsWith(F&& f, const std::string& substr) {
  try {
    f();
  } catch (const Ex& e) {
    const bool ok = std::string(e.what()).find(substr) != std::string::npos;
    if (!ok) std::fprintf(stderr, "  (threw: %s)\n", e.what());
    return ok;
  } catch (...) {
    std::fprintf(stderr, "  (threw a different exception type)\n");
    return false;
  }
  return false;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
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
  if (!Exists(ContainerPath("bf16")) || !Exists(GoldenPath(70))) {
    std::printf("SKIP: tiny fixtures missing under %s\n", TinyDir().c_str());
    return kSkip;
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  const Golden g70(70, GoldenPath(70));
  const std::vector<int32_t>& ids = g70.ids();

  ModelOptions o;
  o.container_path = ContainerPath("bf16");
  o.layout = Layout::kBf16;
  o.max_ctx = 4096;
  o.prompt_checkpoint = true;

  // ---- refusals (no weight is loaded) -----------------------------------------------------------------------------------
  {
    TpOptions tp;
    tp.world = 2;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(o, tp); }, "--tp 2"), "--tp 2 is refused by name");
    ModelOptions pp = o;
    pp.pp = 2;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(pp, TpOptions{}); }, "--pp 2"), "--pp 2 is refused by name");
    ModelOptions mtp = o;
    mtp.mtp_draft_k = 2;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(mtp, TpOptions{}); }, "MTP"), "MTP is refused by name");
    ModelOptions df = o;
    df.dflash_container = "x.r4dx";
    df.dflash_draft_k = 4;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(df, TpOptions{}); }, "DFlash"), "a DFlash drafter is refused by name");
    ModelOptions vis = o;
    vis.vision = ModelOptions::VisionMode::kOn;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(vis, TpOptions{}); }, "vision"), "--vision on is refused by name");
    ModelOptions ll = o;
    ll.layer_limit = 2;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(ll, TpOptions{}); }, "layer_limit"), "layer_limit is refused by name");
    TpOptions odd;
    odd.submit_layers = 7;
    Check(ThrowsWith<std::invalid_argument>([&] { LoadTextModel(o, odd); }, "TpOptions"), "tensor-parallel knobs at world 1 are refused");
    // a layout the container does not have
    ModelOptions w4 = o;
    w4.layout = Layout::kTrellis;
    Check(ThrowsWith<std::exception>([&] { LoadTextModel(w4, TpOptions{}); }, "trellis"), "layout trellis on a bf16 container is refused, naming trellis");
  }

  // ---- the wrapper -----------------------------------------------------------------------------------------------------------
  std::unique_ptr<TextModel> tm = LoadTextModel(o, TpOptions{});
  auto* local = dynamic_cast<Qwen4ExpLocalTextModel*>(tm.get());
  Check(local != nullptr, "LoadTextModel returns a Qwen4ExpLocalTextModel");
  if (local == nullptr) return 1;
  Check(tm->Config().arch == Arch::kQwen4Exp, "Config().arch == kQwen4Exp");
  Check(tm->Config().num_hidden_layers == 4 && tm->Config().vocab_size == 4096 && tm->Config().hidden_size == 2560,
        "Config() geometry of the tiny model");
  Check(tm->Config().layer_types.size() == 4 && tm->Config().layer_types[3] == "full_attention", "Config().layer_types");
  Check(!tm->ModelId().empty(), "ModelId() is set (" + tm->ModelId() + ")");
  Check(tm->TpWorld() == 1 && tm->NumLoadedLayers() == 4, "TpWorld() 1, NumLoadedLayers() 4");
  Check(!tm->HasVision() && !tm->MtpEnabled() && !tm->DflashEnabled() && !tm->MtpUsingReducedVocabDraft() && !tm->HasAudio(),
        "no vision / MTP / DFlash / audio");
  Check(tm->ImageTokenId() == -1 && tm->ImageBoiTokenId() == -1 && tm->ImageEoiTokenId() == -1, "no image tokens");
  {
    const std::vector<VramReport> v = tm->Vram();
    Check(!v.empty() && v[0].buffers_gib > 0.0 && v[0].total_gib > 0.0, "Vram() reports the device(s)");
    for (const VramReport& r : v) std::printf("  stage %d device %d: %.3f GiB resident of this process, %.1f / %.1f GiB used\n", r.rank, r.device,
                                              r.buffers_gib, r.used_gib, r.total_gib);
  }
  std::printf("  %d stage(s): layers split at %lld\n", local->model().NumStages(), static_cast<long long>(local->model().SplitLayer()));

  // Prefill / Decode through the wrapper equal the same calls on a directly loaded model (same devices).
  {
    Qwen4ExpModelOptions mo = MakeQwen4ExpModelOptions(o);
    auto direct = Qwen4ExpModel::Load(mo);
    const std::vector<float> a = tm->Prefill(ids);
    const std::vector<float> b = direct->Prefill(ids);
    Check(Same(a, b), "Prefill through the TextModel equals the direct model, bit for bit");
    Check(tm->PositionCount() == 70 && direct->PositionCount() == 70, "PositionCount() after Prefill(70)");
    const std::vector<float> da = tm->DecodeStep(17), db = direct->DecodeStep(17);
    Check(Same(da, db), "DecodeStep through the TextModel equals the direct model");
    Check(tm->PositionCount() == 71, "PositionCount() after a decode step");
  }

  // greedy / sampled / checkpoint
  {
    tm->Reset();
    Check(tm->PositionCount() == 0, "Reset() -> PositionCount() 0");
    tm->Prefill(ids);
    tm->SaveCheckpoint();
    const std::vector<float> logits = tm->DecodeStep(321);
    const int64_t want = ArgMax(logits.data(), 4096);
    tm->RestoreCheckpoint();
    Check(tm->PositionCount() == 70, "RestoreCheckpoint() rewinds the position");
    const int32_t greedy = tm->DecodeStepGreedy(321);
    Check(greedy == want, "DecodeStepGreedy == argmax of DecodeStep (" + std::to_string(greedy) + ")");
    tm->RestoreCheckpoint();
    std::mt19937_64 rng(42);
    r4dx::kernels::SampleParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 20;
    const int32_t s = tm->DecodeStepSampled(321, sp, rng);
    Check(s >= 0 && s < 4096, "DecodeStepSampled returns a token id (" + std::to_string(s) + ")");
    tm->RestoreCheckpoint();
    bool overlap_ran = false;
    const int32_t og = tm->DecodeStepGreedyOverlap(321, [&] { overlap_ran = true; });
    Check(overlap_ran && og == want, "DecodeStepGreedyOverlap runs the callback and returns the greedy token");
  }

  // unsupported features
  {
    std::mt19937_64 rng(1);
    r4dx::kernels::SampleParams sp;
    Check(ThrowsWith<std::runtime_error>([&] { tm->DecodeStepMtpGreedy(1, 2); }, "MTP"), "DecodeStepMtpGreedy throws naming MTP");
    Check(ThrowsWith<std::runtime_error>([&] { tm->DecodeStepMtpSampled(1, 2, sp, rng); }, "MTP"), "DecodeStepMtpSampled throws naming MTP");
    Check(ThrowsWith<std::runtime_error>([&] { tm->DecodeStepDflashGreedy(1, 2, 0.0f, 0); }, "DFlash"), "DecodeStepDflashGreedy throws naming DFlash");
    Check(ThrowsWith<std::runtime_error>([&] { tm->DecodeStepDflashSampled(1, 2, 0.0f, 0, sp, rng); }, "DFlash"),
          "DecodeStepDflashSampled throws naming DFlash");
    Check(ThrowsWith<std::runtime_error>([&] { tm->SetDflashInjectionEnabled(true); }, "DFlash"), "SetDflashInjectionEnabled(true) throws");
    tm->SetDflashInjectionEnabled(false);
    std::vector<ImageSpan> images(1);
    Check(ThrowsWith<std::runtime_error>([&] { tm->PrefillMultimodal(ids, images); }, "vision"), "PrefillMultimodal with images throws naming vision");
    Check(ThrowsWith<std::runtime_error>([&] { tm->EncodeImages(nullptr, 0, {}, nullptr); }, "vision"), "EncodeImages throws naming vision");
    Check(ThrowsWith<std::runtime_error>([&] { tm->DecodeStepProfiled(1); }, "profile"), "DecodeStepProfiled throws");
    Check(ThrowsWith<std::runtime_error>([&] { tm->PrefillProfiled(ids); }, "profile"), "PrefillProfiled throws");
    tm->Reset();
    const std::vector<float> plain = tm->Prefill(ids);
    tm->Reset();
    const std::vector<float> mm = tm->PrefillMultimodal(ids, {});
    Check(Same(plain, mm), "PrefillMultimodal without images is Prefill");
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
