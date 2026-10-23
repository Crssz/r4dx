// Qwen35MoeTpModel (the --tp 2 TextModel) through LoadTextModel, on the tiny qwen3_5_moe containers (docs/qwen35moe-tp-design.md P3).
//
//   * dispatch: a tiny container + TpOptions{world 2} is a Qwen35MoeTpModel (TpWorld() 2, arch kQwen35Moe); --pp 2, --dflash and a partial layer load are
//     refused by name; the layer split (world 1) is untouched;
//   * emulate mode (two ranks on ONE device) against the layer-split model: prefill logits at T = 1 / 7 / 70 / 300, full-row decode logits, 24 greedy and 8 sampled
//     steps -- close (the facade compares the ranks' results on every call, so rank divergence would already have thrown);
//   * checkpoint / restore, Reset determinism (the same feed after Reset() gives identical bytes), PositionCount bookkeeping;
//   * fault injection (emulate): a throwing all-reduce puts the group in kNeedsRecovery, device-work calls refuse until Reset(), recovery works and the
//     next run equals the pre-fault one byte for byte;
//   * REAL mode (R4DX_TP2GPU=1, both cards visible): bit-identical to emulate on every output above (G6' on the tiny model: the real mailbox all-reduce adds the same
//     two bf16 partials the emulation adds on the host).
// Fixtures as test_model_q35moe_tp_tiny.cpp. Exit code 77 = a fixture is missing.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

#include "arch.h"
#include "model.h"
#include "q35moe_model_util.h"
#include "qwen35moe_tp_model.h"
#include "text_model.h"

using namespace q35m;
using r4dx::model::Arch;
using r4dx::model::ModelOptions;
using r4dx::model::PpOptions;
using r4dx::model::Qwen35MoeTpModel;
using r4dx::model::TextModel;
using r4dx::model::TpOptions;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
}

constexpr int kTs[] = {1, 7, 70, 300};

std::vector<int32_t> Ids(int T) { return Golden(T, GoldenPath(T)).ids(); }

struct Out {
  std::vector<std::vector<float>> prefill, decode;
  std::vector<int32_t> greedy, sampled;
  bool operator==(const Out& o) const { return prefill == o.prefill && decode == o.decode && greedy == o.greedy && sampled == o.sampled; }
};

Out Scenario(TextModel& m) {
  Out o;
  for (int T : kTs) {
    m.Reset();
    o.prefill.push_back(m.Prefill(Ids(T)));
    Check(m.PositionCount() == T, "PositionCount == " + std::to_string(T) + " after the prefill");
  }
  m.Reset();
  (void)m.Prefill(Ids(70));
  int32_t tok = static_cast<int32_t>(ArgMax(o.prefill[2].data(), static_cast<int64_t>(o.prefill[2].size())));
  const int32_t first = tok;
  for (int i = 0; i < 24; ++i) {
    tok = m.DecodeStepGreedy(tok);
    o.greedy.push_back(tok);
  }
  Check(m.PositionCount() == 70 + 24, "PositionCount after 24 decode steps");
  m.Reset();
  (void)m.Prefill(Ids(70));
  {
    std::mt19937_64 rng(2024);
    r4dx::kernels::SampleParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 40;
    int32_t t = first;
    for (int i = 0; i < 8; ++i) {
      t = m.DecodeStepSampled(t, sp, rng);
      o.sampled.push_back(t);
    }
  }
  m.Reset();
  (void)m.Prefill(Ids(70));
  int32_t t = first;
  for (int i = 0; i < 3; ++i) {
    o.decode.push_back(m.DecodeStep(t));
    t = static_cast<int32_t>(ArgMax(o.decode.back().data(), static_cast<int64_t>(o.decode.back().size())));
  }
  m.Reset();
  return o;
}

void CompareClose(const Out& ref, const Out& got, const std::string& tag) {
  bool ok = true;
  for (size_t i = 0; i < ref.prefill.size(); ++i) {
    const FStats s = CompareF32(got.prefill[i].data(), ref.prefill[i].data(), static_cast<int64_t>(ref.prefill[i].size()));
    std::printf("  %s prefill T=%d logits mean/rms %.5f\n", tag.c_str(), kTs[i], s.mean_norm);
    ok = ok && !s.nan && s.mean_norm <= 0.05 && ArgMax(got.prefill[i].data(), static_cast<int64_t>(got.prefill[i].size())) == ArgMax(ref.prefill[i].data(), static_cast<int64_t>(ref.prefill[i].size()));
  }
  for (size_t i = 0; i < ref.decode.size(); ++i) {
    const FStats s = CompareF32(got.decode[i].data(), ref.decode[i].data(), static_cast<int64_t>(ref.decode[i].size()));
    ok = ok && !s.nan && s.mean_norm <= 0.05;
  }
  Check(ok, tag + ": prefill / decode logits within bf16 noise of the layer split, argmax equal");
  int agree = 0;
  while (agree < 24 && ref.greedy[static_cast<size_t>(agree)] == got.greedy[static_cast<size_t>(agree)]) ++agree;
  std::printf("  %s greedy: %d of 24 leading tokens agree\n", tag.c_str(), agree);
  Check(agree >= 20, tag + ": greedy tokens agree with the layer split (>= 20 of 24)");
}

ModelOptions Options(const std::string& path, const std::string& layout) {
  ModelOptions o;
  o.container_path = path;
  o.layout = r4dx::model::LayoutFromName(layout);
  o.max_ctx = 4096;
  o.prompt_checkpoint = true;
  return o;
}

TpOptions TpOpts(TpOptions::Mode mode) {
  TpOptions tp;
  tp.world = 2;
  tp.mode = mode;
  return tp;
}

void RunLayout(const std::string& path, const std::string& layout, bool real) {
  std::printf("\n===== %s (%s) =====\n", layout.c_str(), path.c_str());
  if (!Exists(path)) {
    std::printf("SKIP: %s not found\n", path.c_str());
    return;
  }
  const ModelOptions o = Options(path, layout);
  Out one;
  {
    TpOptions tp1;
    auto m = r4dx::model::LoadTextModel(o, tp1);
    Check(m->TpWorld() == 1 && m->Config().arch == Arch::kQwen35Moe, "world 1 is still the layer split");
    one = Scenario(*m);
  }
  Out emu;
  {
    auto m = r4dx::model::LoadTextModel(o, TpOpts(TpOptions::Mode::kEmulate));
    Check(m->TpWorld() == 2 && m->Config().arch == Arch::kQwen35Moe && m->Config().vocab_size == 4096, "world 2 emulate is a tensor-parallel Qwen35 MoE model");
    auto* tpm = dynamic_cast<Qwen35MoeTpModel*>(m.get());
    Check(tpm != nullptr && tpm->NumRankThreads() == 2, "it is a Qwen35MoeTpModel with two rank threads");
    emu = Scenario(*m);
    CompareClose(one, emu, layout + " TP2 emulate vs layer split");

    // Reset determinism and checkpoint / restore
    m->Reset();
    (void)m->Prefill(Ids(70));
    const std::vector<float> a = m->DecodeStep(Ids(70).back());
    m->Reset();
    (void)m->Prefill(Ids(70));
    const std::vector<float> b = m->DecodeStep(Ids(70).back());
    Check(a == b, layout + ": the same feed after Reset() gives identical logits");
    m->Reset();
    (void)m->Prefill(Ids(300));
    m->SaveCheckpoint();
    const std::vector<float> c1 = m->DecodeStep(5);
    const std::vector<float> c2 = m->DecodeStep(6);
    m->RestoreCheckpoint();
    Check(m->PositionCount() == 300, layout + ": RestoreCheckpoint rewinds PositionCount");
    const std::vector<float> d1 = m->DecodeStep(5);
    const std::vector<float> d2 = m->DecodeStep(6);
    Check(c1 == d1 && c2 == d2, layout + ": RestoreCheckpoint + the same steps reproduce the logits bit for bit");

    // fault injection: a throwing all-reduce -> kNeedsRecovery -> Reset() recovers
    if (tpm != nullptr) {
      m->Reset();
      tpm->ArmFaultInjection(1, 5, /*kind=*/0);
      bool threw = false;
      try {
        (void)m->Prefill(Ids(70));
      } catch (const std::exception&) {
        threw = true;
      }
      Check(threw && tpm->GetState() == Qwen35MoeTpModel::State::kNeedsRecovery, layout + ": a faulted all-reduce throws and leaves kNeedsRecovery");
      Check(Throws([&] { (void)m->DecodeStepGreedy(1); }, "call Reset() first"), layout + ": device-work calls refuse until Reset()");
      m->Reset();
      Check(tpm->GetState() == Qwen35MoeTpModel::State::kReady, layout + ": Reset() recovers the group");
      m->Reset();
      const Out after = Scenario(*m);
      Check(after == emu, layout + ": a full scenario after recovery equals the pre-fault one byte for byte");
    }
  }
  if (real) {
    auto m = r4dx::model::LoadTextModel(o, TpOpts(TpOptions::Mode::kReal));
    const Out re = Scenario(*m);
    Check(re == emu, layout + ": REAL two-GPU mode is bit-identical to the emulation (prefill/decode logits, greedy and sampled tokens)");
    auto* tpm = dynamic_cast<Qwen35MoeTpModel*>(m.get());
    if (tpm != nullptr) std::printf("  %s\n", tpm->StatsLine().c_str());
  }
}

}  // namespace

int main() {
  Unbuffer();
  if (!Exists(GoldenPath(70))) {
    std::printf("SKIP: golden fixtures missing under %s\n", TinyDir().c_str());
    return kSkip;
  }
  try {
    const bool real = std::getenv("R4DX_TP2GPU") != nullptr && std::string(std::getenv("R4DX_TP2GPU")) == "1" && VisibleDevices() >= 2;
    std::printf("devices visible: %d, real two-GPU part %s\n", VisibleDevices(), real ? "ON" : "off (set R4DX_TP2GPU=1 with both cards visible)");
    // dispatch refusals
    {
      const std::string p = ContainerPath("bf16");
      if (Exists(p)) {
        ModelOptions o = Options(p, "bf16");
        ModelOptions pp = o;
        pp.pp = 1;
        PpOptions ppo;
        Check(Throws([&] { (void)r4dx::model::LoadTextModel(pp, TpOpts(TpOptions::Mode::kEmulate), ppo); }, "--pp 2 does not support qwen3_5_moe"),
              "--tp 2 --pp 2 stays refused for qwen3_5_moe");
        ModelOptions df = o;
        df.dflash_container = "x.r4dx";
        df.dflash_draft_k = 4;
        Check(Throws([&] { (void)r4dx::model::LoadTextModel(df, TpOpts(TpOptions::Mode::kEmulate)); }, "P6"), "--dflash under --tp 2 is refused by name");
        ModelOptions lim = o;
        lim.layer_limit = 2;
        Check(Throws([&] { (void)r4dx::model::LoadTextModel(lim, TpOpts(TpOptions::Mode::kEmulate)); }, "layer_limit"), "a partial layer load under --tp 2 is refused");
        TpOptions three;
        three.world = 3;
        Check(Throws([&] { (void)r4dx::model::LoadTextModel(o, three); }, "qwen3_5_moe"), "--tp 3 is refused");
      }
    }
    const char* e = std::getenv("R4DX_Q35MOE_TRELLIS");
    const std::string trellis = e != nullptr && e[0] != 0 ? e : std::string("E:\\models\\r4dx\\q35moe-tiny\\tiny-mix.r4dx");
    RunLayout(ContainerPath("bf16"), "bf16", real);
    RunLayout(trellis, "trellis", real);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", ex.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
