// tests/server/test_engine_recovery.cpp -- Engine::RunRequest's error path (docs/tp.md 2.4, 8.4), end
// to end through engine.cpp itself, against CPU fakes of the two TextModel kinds. No GPU work: the
// fake never calls HIP (the binary links what r4dx-server links only because engine.cpp does). Needs
// the real tokenizer directory (R4DX_TOKENIZER_MODEL_DIR, the same cache variable tests/tokenizer
// uses) and skips (77) without it.
//
// The fake (FakeTextModel) has two modes, one per half of 8.4:
//   - TP (TpModel's contract, not its mechanics): after any failed command the group is
//     kNeedsRecovery, every device-work call then throws until Reset() recovers it, and the host-only
//     calls (SetDflashInjectionEnabled, the cached accessors) stay legal (docs/tp.md 2.4). A request
//     that reached a forward call without a Reset() fails -- the permanent-500 half.
//   - TP=1 (LocalTextModel's: no state machine): a failed command leaves its tokens in the state and
//     every later call succeeds, so a request that skipped the Reset() runs on the failed request's
//     leftovers -- the silent-wrong-output half.
// In both, the "logits" are a pure function of every token fed since the last Reset(), so leftovers
// change the text: the text checks below are what catch the TP=1 half.
//
// Checked in both modes, with and without a (fake) DFlash2 drafter configured:
//   1. a request that faults mid-decode answers 500 (TP: and leaves the fake in kNeedsRecovery);
//   2. the next request (the same conversation) answers 200 through exactly one Reset() after the
//      fault, with the text a fresh engine produces for it;
//   3. a fault in the first forward after a Reset() (the prefill) fails that request, and the one
//      after it resets again (PrefixState's Invalidate() survives the Reset() of the failed
//      request, prefix_state.h) and reproduces the pre-fault reference text;
//   4. with a drafter, the per-request injection toggle is called after the fault and before the
//      Reset(), i.e. while a TP group awaits recovery -- which is why TpModel must keep it host-only
//      (docs/tp.md 2.4).
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "model_config.h"
#include "model_types.h"
#include "openai_types.h"
#include "response_sink.h"
#include "text_model.h"

#ifndef R4DX_TOKENIZER_MODEL_DIR
#define R4DX_TOKENIZER_MODEL_DIR "C:/AI/models/Qwen3.8-27B"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                   \
      std::fprintf(stderr, "\n");                                          \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

using r4dx::model::ImageRows;
using r4dx::model::ImageSpan;
using r4dx::model::StepProfile;
using r4dx::model::VramReport;

constexpr int64_t kVocab = 248320;  // the checkpoint's vocabulary: the engine samples over logits.size()

class FakeTextModel final : public r4dx::model::TextModel {
 public:
  enum class State { kReady, kNeedsRecovery };

  // tp: TpModel's state machine (see the file comment); false: LocalTextModel's TP=1 behaviour.
  FakeTextModel(bool dflash, bool tp) : dflash_(dflash), tp_(tp) {}

  // The n-th forward command from now fails, after it has fed its tokens (a real mid-forward failure
  // leaves the KV/GDN state dirty too).
  void ArmFault(int64_t nth_forward) { fault_in_ = nth_forward; }
  State GetState() const { return state_; }
  bool Dirty() const { return dirty_; }  // a fault's tokens are in the state, no Reset() since
  int recoveries = 0;                    // Reset() calls that cleared a fault's leftovers
  int toggles_after_fault = 0;           // SetDflashInjectionEnabled calls between a fault and its Reset()

  // ---- host-only: legal in every state (docs/tp.md 2.4) ----------------------------------------
  const r4dx::model::ModelConfig& Config() const override { return cfg_; }
  const std::string& ModelId() const override { return id_; }
  int64_t ImageTokenId() const override { return 248056; }
  int64_t VisionMergeSize() const override { return 2; }
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return dflash_; }
  int64_t SampledFallbackRows() const override { return 0; }
  int64_t PositionCount() const override { return static_cast<int64_t>(fed_.size()); }
  int64_t NumLoadedLayers() const override { return 4; }
  int TpWorld() const override { return tp_ ? 2 : 1; }
  std::vector<VramReport> Vram() const override { return {}; }
  void SetDflashInjectionEnabled(bool) override {
    if (dirty_) ++toggles_after_fault;
  }

  // ---- sequence state ---------------------------------------------------------------------------
  void Reset() override {
    if (dirty_) ++recoveries;
    dirty_ = false;
    state_ = State::kReady;
    fed_.clear();
  }

  // ---- device work: TpStateError's message in kNeedsRecovery (TP mode) ---------------------------
  void EncodeImages(const float*, int64_t, const std::vector<r4dx::vision::GridThw>&, ImageRows*,
                    r4dx::vision::VisionEncodeStats*) override {
    throw std::logic_error("FakeTextModel: no vision");
  }
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override {
    Forward(token_ids);
    return Row();
  }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>&) override {
    return Prefill(token_ids);
  }
  std::vector<float> DecodeStep(int32_t token_id) override {
    Forward({token_id});
    return Row();
  }
  int32_t DecodeStepGreedy(int32_t token_id) override {
    Forward({token_id});
    return Next();
  }
  int32_t DecodeStepSampled(int32_t, const r4dx::kernels::SampleParams&, std::mt19937_64&) override {
    throw std::logic_error("FakeTextModel: greedy requests only");
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t, int64_t) override {
    throw std::logic_error("FakeTextModel: no MTP head");
  }
  std::vector<int32_t> DecodeStepMtpSampled(int32_t, int64_t, const r4dx::kernels::SampleParams&,
                                            std::mt19937_64&) override {
    throw std::logic_error("FakeTextModel: no MTP head");
  }
  // A round that accepts no draft: feeds the anchor and returns the bonus token.
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t, float, int64_t,
                                              int64_t* walk_len_out) override {
    if (!dflash_) throw std::logic_error("FakeTextModel: no drafter");
    Forward({token_id});
    if (walk_len_out != nullptr) *walk_len_out = 0;
    return {Next()};
  }
  std::vector<int32_t> DecodeStepDflashSampled(int32_t, int64_t, float, int64_t, const r4dx::kernels::SampleParams&,
                                               std::mt19937_64&, int64_t*) override {
    throw std::logic_error("FakeTextModel: greedy requests only");
  }
  StepProfile DecodeStepProfiled(int32_t) override { throw std::logic_error("FakeTextModel: no profiling"); }
  StepProfile PrefillProfiled(const std::vector<int32_t>&) override {
    throw std::logic_error("FakeTextModel: no profiling");
  }

 private:
  void Forward(const std::vector<int32_t>& tokens) {
    if (state_ != State::kReady) throw std::runtime_error("tp: group aborted by an earlier error; call Reset() first");
    fed_.insert(fed_.end(), tokens.begin(), tokens.end());
    if (fault_in_ > 0 && --fault_in_ == 0) {
      dirty_ = true;
      if (tp_) state_ = State::kNeedsRecovery;  // TP=1: the next call simply runs on the leftovers
      throw std::runtime_error(tp_ ? "tp fault injection" : "fault injection");
    }
  }
  // FNV-1a over everything fed since the last Reset(), mapped into ordinary (non-special, non-EOS)
  // token ids.
  int32_t Next() const {
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : fed_) h = (h ^ static_cast<uint32_t>(t)) * 1099511628211ull;
    return static_cast<int32_t>(1000 + h % 20000);
  }
  std::vector<float> Row() const {
    std::vector<float> row(static_cast<size_t>(kVocab), 0.0f);
    row[static_cast<size_t>(Next())] = 1.0f;
    return row;
  }

  const bool dflash_;
  const bool tp_;
  r4dx::model::ModelConfig cfg_;
  std::string id_ = "fake/tp";
  State state_ = State::kReady;
  bool dirty_ = false;
  int64_t fault_in_ = 0;
  std::vector<int32_t> fed_;
};

struct Harness {
  std::unique_ptr<r4dx::server::Engine> engine;
  FakeTextModel* fake = nullptr;
};

Harness MakeEngine(const std::string& tokenizer_dir, bool dflash, bool tp) {
  Harness e;
  r4dx::server::EngineOptions opts;
  opts.tokenizer_dir = tokenizer_dir;
  opts.model_opts.max_ctx = 4096;
  opts.log_level = "warn";
  if (dflash) {
    // Only the engine's own "is a drafter configured" signal; the fake stands in for it.
    opts.model_opts.dflash_container = "fake-drafter.r4dx";
    opts.model_opts.dflash_draft_k = 7;
  }
  FakeTextModel** slot = &e.fake;
  opts.model_loader = [slot, dflash, tp](const r4dx::model::ModelOptions&, const r4dx::model::TpOptions&) {
    auto m = std::make_unique<FakeTextModel>(dflash, tp);
    *slot = m.get();
    return std::unique_ptr<r4dx::model::TextModel>(std::move(m));
  };
  e.engine = std::make_unique<r4dx::server::Engine>(std::move(opts));
  e.engine->LoadAndStart();
  return e;
}

r4dx::server::ChatMessage Msg(const std::string& role, const std::string& content) {
  r4dx::server::ChatMessage m;
  m.role = role;
  m.content = content;
  return m;
}

std::shared_ptr<r4dx::server::BufferingSink> Run(Harness& e, const std::vector<r4dx::server::ChatMessage>& messages,
                                                 int64_t max_tokens) {
  auto req = std::make_shared<r4dx::server::PendingRequest>();
  req->kind = r4dx::server::RequestKind::kChat;
  req->request_id = r4dx::server::GenerateRequestId("chatcmpl-");
  req->messages = messages;
  req->sampling.temperature = 0.0f;
  req->max_tokens = max_tokens;
  auto sink = std::make_shared<r4dx::server::BufferingSink>();
  req->sink = sink;
  if (!e.engine->Submit(req)) throw std::runtime_error("queue full");
  sink->Wait();
  return sink;
}

void Scenario(const std::string& tokenizer_dir, bool dflash, bool tp) {
  const std::string tag_s = std::string(tp ? "tp2 " : "tp1 ") + (dflash ? "dflash" : "plain");
  const char* tag = tag_s.c_str();
  const int failures_before = g_failures;
  const std::vector<r4dx::server::ChatMessage> hello = {Msg("user", "Say hello in one short sentence.")};

  Harness e = MakeEngine(tokenizer_dir, dflash, tp);
  const auto ref = Run(e, hello, 8);
  CHECK(!ref->errored && !ref->text.empty() && ref->completion_tokens == 8, "[%s] reference request: errored=%d '%s'",
        tag, static_cast<int>(ref->errored), ref->error_message.c_str());

  // 1. A conversation, then its continuation faults mid-decode (the third forward: prefill, then two
  //    decode steps).
  const auto turn1 = Run(e, {Msg("user", "What is 2 plus 2?")}, 6);
  CHECK(!turn1->errored, "[%s] turn 1 failed: %s", tag, turn1->error_message.c_str());
  const std::vector<r4dx::server::ChatMessage> turn2 = {Msg("user", "What is 2 plus 2?"), Msg("assistant", turn1->text),
                                                         Msg("user", "And 3 plus 3?")};
  e.fake->ArmFault(3);
  const auto a = Run(e, turn2, 16);
  CHECK(a->errored && a->error_status == 500, "[%s] the fault request answered errored=%d status=%d", tag,
        static_cast<int>(a->errored), a->error_status);
  CHECK(e.fake->Dirty() && (!tp || e.fake->GetState() == FakeTextModel::State::kNeedsRecovery),
        "[%s] the fault did not leave its leftovers (TP: the group needing recovery)", tag);

  // 2. The same conversation again: one Reset() after the fault, and a fresh engine's text for it.
  //    At TP=1 a skipped Reset() would not throw -- only this text check would see it.
  const auto b = Run(e, turn2, 16);
  CHECK(!b->errored, "[%s] the request after the fault failed: %s", tag, b->error_message.c_str());
  CHECK(e.fake->recoveries == 1, "[%s] recoveries after the first fault: %d, want 1", tag, e.fake->recoveries);
  {
    Harness fresh = MakeEngine(tokenizer_dir, dflash, tp);
    const auto want = Run(fresh, turn2, 16);
    CHECK(!want->errored && b->text == want->text && b->completion_tokens == want->completion_tokens,
          "[%s] after recovery the text differs from a fresh engine's ('%s' vs '%s')", tag, b->text.c_str(),
          want->text.c_str());
    fresh.engine->Shutdown();
  }

  // 3. A fault in the first forward after a Reset() -- the prefill of a new conversation -- and then
  //    the pre-fault reference request again.
  e.fake->ArmFault(1);
  const auto c = Run(e, hello, 8);
  CHECK(c->errored && c->error_status == 500, "[%s] the prefill-fault request answered errored=%d status=%d", tag,
        static_cast<int>(c->errored), c->error_status);
  const auto d = Run(e, hello, 8);
  CHECK(!d->errored && d->text == ref->text && d->completion_tokens == ref->completion_tokens,
        "[%s] after the second recovery: errored=%d text '%s', want '%s'", tag, static_cast<int>(d->errored),
        d->text.c_str(), ref->text.c_str());
  CHECK(e.fake->recoveries == 2, "[%s] recoveries after the second fault: %d, want 2", tag, e.fake->recoveries);

  // 4. The injection toggle ran before the Reset() both times (dflash only).
  if (dflash) {
    CHECK(e.fake->toggles_after_fault == 2, "[%s] injection toggles between a fault and its Reset(): %d, want 2", tag,
          e.fake->toggles_after_fault);
  } else {
    CHECK(e.fake->toggles_after_fault == 0, "[%s] the toggle was called without a drafter", tag);
  }
  if (g_failures == failures_before) {
    std::printf("[%s] two faults -> 500 each; the request after each answered 200 through a Reset() (%d in all), "
                "with the fresh-engine / pre-fault text\n",
                tag, e.fake->recoveries);
  }
  e.engine->Shutdown();
}

}  // namespace

int main() {
  const std::string tokenizer_dir = R4DX_TOKENIZER_MODEL_DIR;
  if (!std::filesystem::exists(std::filesystem::path(tokenizer_dir) / "tokenizer.json")) {
    std::fprintf(stderr, "SKIP: %s/tokenizer.json not found (set -DR4DX_TOKENIZER_MODEL_DIR=...)\n",
                 tokenizer_dir.c_str());
    return 77;
  }
  try {
    for (bool tp : {true, false}) {
      Scenario(tokenizer_dir, /*dflash=*/false, tp);
      Scenario(tokenizer_dir, /*dflash=*/true, tp);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all engine recovery checks passed\n");
  return 0;
}
