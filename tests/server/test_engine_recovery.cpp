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
// CheckpointScenario covers --prompt-checkpoint's reuse path the same way (see its own comment).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "dialect.h"
#include "engine.h"
#include "model_config.h"
#include "model_types.h"
#include "openai_types.h"
#include "response_sink.h"
#include "text_model.h"
#include "tokenizer.h"

#include "r4dx/models_root.h"

#ifndef R4DX_TOKENIZER_MODEL_DIR
#define R4DX_TOKENIZER_MODEL_DIR (r4dx::ModelsPath("Huihui-Qwen3.8-27B-abliterated"))
#endif
// The assembled Gemma 4 tokenizer directory (tests/tokenizer's cache variable of the same name); the
// Gemma scenario below is skipped (not failed) without it.
#ifndef R4DX_LFM2_TOKENIZER_DIR
#define R4DX_LFM2_TOKENIZER_DIR (r4dx::ModelsPath("LiquidAI/LFM2.5-24B-A2B"))
#endif
#ifndef R4DX_GEMMA_TOKENIZER_DIR
#define R4DX_GEMMA_TOKENIZER_DIR (r4dx::ModelsPath("Huihui-gemma-4-12B-it-abliterated-tok"))
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

// The engine samples over logits.size(). 262144 covers both vocabularies (Qwen 248320, Gemma 262144), so a
// scripted reply token of either tokenizer is always in range.
constexpr int64_t kVocab = 262144;

class FakeTextModel final : public r4dx::model::TextModel {
 public:
  enum class State { kReady, kNeedsRecovery };

  // tp: TpModel's state machine (see the file comment); false: LocalTextModel's TP=1 behaviour.
  FakeTextModel(bool dflash, bool tp, r4dx::model::Arch arch = r4dx::model::Arch::kQwen35)
      : dflash_(dflash), tp_(tp) {
    cfg_.arch = arch;  // TextModel::Config().arch: what the engine cross-checks its dialect against
  }

  // Every Prefill call's row count since construction, and the first prefill's tokens after the last
  // Reset(): the Gemma scenario reads the checkpoint split (ckpt_back) and the BOS off them.
  std::vector<size_t> prefill_sizes;
  std::vector<int32_t> first_prefill;

  // The n-th forward command from now fails, after it has fed its tokens (a real mid-forward failure
  // leaves the KV/GDN state dirty too).
  // The logits row is `floor` everywhere and 1.0 at the scripted token (default floor 0: exactly the row every
  // other scenario has always seen). A floor above 0 makes a repetition penalty on the scripted token visible:
  // 1.0 / p drops below the floor, so greedy decoding moves to the lowest-index token that was not seen.
  void SetRowFloor(float floor) { row_floor_ = floor; }
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
    ckpt_.reset();  // Model::Reset() drops the checkpoint too
  }

  // Model::SaveCheckpoint/RestoreCheckpoint: the "state" is fed_, so a checkpoint is a copy of it and
  // a restore truncates back to it. Device work: refused in kNeedsRecovery like a forward.
  int saves = 0;
  int restores = 0;
  void SaveCheckpoint() override {
    RequireReady();
    ckpt_ = fed_;
    ++saves;
  }
  void RestoreCheckpoint() override {
    RequireReady();
    if (!ckpt_) throw std::logic_error("FakeTextModel: no checkpoint since the last Reset()");
    fed_ = *ckpt_;
    ++restores;
  }

  // The next prompt makes the model answer exactly `reply` (the last token is meant to be an EOS):
  // the state after that request's LAST prefill call -- the engine may split a prompt around its
  // checkpoint -- answers reply[0], and so on; any other state keeps the hash. Still a pure function
  // of fed_, so a state that is not exactly the new prompt still changes the text.
  void ScriptReplyToNextPrompt(std::vector<int32_t> reply) { pending_reply_ = std::move(reply); }

  // ---- device work: TpStateError's message in kNeedsRecovery (TP mode) ---------------------------
  void EncodeImages(const float*, int64_t, const std::vector<r4dx::vision::GridThw>&, ImageRows*,
                    r4dx::vision::VisionEncodeStats*) override {
    throw std::logic_error("FakeTextModel: no vision");
  }
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override {
    prefill_sizes.push_back(token_ids.size());
    if (fed_.empty()) first_prefill = token_ids;
    Forward(token_ids);
    if (!pending_reply_.empty()) {  // armed until the first decode step (Decode below)
      script_prompt_ = fed_;
      script_reply_ = pending_reply_;
    }
    return Row();
  }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>&) override {
    return Prefill(token_ids);
  }
  std::vector<float> DecodeStep(int32_t token_id) override {
    Decode(token_id);
    return Row();
  }
  int32_t DecodeStepGreedy(int32_t token_id) override {
    Decode(token_id);
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
    Decode(token_id);
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
  void RequireReady() const {
    if (state_ != State::kReady) throw std::runtime_error("tp: group aborted by an earlier error; call Reset() first");
  }
  void Decode(int32_t token_id) {
    pending_reply_.clear();  // the scripted prompt is final once generation starts
    Forward({token_id});
  }
  void Forward(const std::vector<int32_t>& tokens) {
    RequireReady();
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
    if (!script_reply_.empty() && fed_.size() >= script_prompt_.size() &&
        std::equal(script_prompt_.begin(), script_prompt_.end(), fed_.begin())) {
      const size_t i = fed_.size() - script_prompt_.size();
      if (i < script_reply_.size() &&
          std::equal(script_reply_.begin(), script_reply_.begin() + static_cast<ptrdiff_t>(i),
                     fed_.begin() + static_cast<ptrdiff_t>(script_prompt_.size()))) {
        return script_reply_[i];
      }
    }
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : fed_) h = (h ^ static_cast<uint32_t>(t)) * 1099511628211ull;
    return static_cast<int32_t>(1000 + h % 20000);
  }
  std::vector<float> Row() const {
    std::vector<float> row(static_cast<size_t>(kVocab), row_floor_);
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
  float row_floor_ = 0.0f;
  std::vector<int32_t> fed_;
  std::optional<std::vector<int32_t>> ckpt_;
  std::vector<int32_t> pending_reply_, script_prompt_, script_reply_;
};

struct Harness {
  std::unique_ptr<r4dx::server::Engine> engine;
  FakeTextModel* fake = nullptr;
};

Harness MakeEngine(const std::string& tokenizer_dir, bool dflash, bool tp, bool checkpoint = false,
                   bool thinking = false, const std::string& request_log_path = "",
                   r4dx::model::Arch arch = r4dx::model::Arch::kQwen35,
                   std::optional<r4dx::server::DialectKind> dialect = std::nullopt,
                   const std::function<void(r4dx::server::EngineOptions&)>& opts_hook = nullptr) {
  Harness e;
  r4dx::server::EngineOptions opts;
  if (!request_log_path.empty()) {  // --request-log
    std::string err;
    opts.request_log = r4dx::server::RequestLog::Open(request_log_path, &err);
    if (!opts.request_log) throw std::runtime_error(err);
  }
  opts.tokenizer_dir = tokenizer_dir;
  opts.dialect = dialect;  // nullopt = auto, from the tokenizer directory
  opts.model_opts.max_ctx = 4096;
  opts.model_opts.prompt_checkpoint = checkpoint;  // --prompt-checkpoint
  opts.default_thinking = thinking;                // --think on
  opts.log_level = "warn";
  if (dflash) {
    // Only the engine's own "is a drafter configured" signal; the fake stands in for it.
    opts.model_opts.dflash_container = "fake-drafter.r4dx";
    opts.model_opts.dflash_draft_k = 7;
  }
  FakeTextModel** slot = &e.fake;
  opts.model_loader = [slot, dflash, tp, arch](const r4dx::model::ModelOptions&, const r4dx::model::TpOptions&) {
    auto m = std::make_unique<FakeTextModel>(dflash, tp, arch);
    *slot = m.get();
    return std::unique_ptr<r4dx::model::TextModel>(std::move(m));
  };
  if (opts_hook) opts_hook(opts);
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
                                                 int64_t max_tokens, bool thinking = false,
                                                 const nlohmann::json& tools = nlohmann::json::array(),
                                                 bool raw_completion = false, const std::string& raw_prompt = "",
                                                 const std::function<void(r4dx::server::PendingRequest&)>& tweak = nullptr) {
  auto req = std::make_shared<r4dx::server::PendingRequest>();
  req->kind = raw_completion ? r4dx::server::RequestKind::kCompletion : r4dx::server::RequestKind::kChat;
  req->request_id = r4dx::server::GenerateRequestId("chatcmpl-");
  req->messages = messages;
  req->raw_prompt = raw_prompt;
  req->tools = tools;
  req->sampling.temperature = 0.0f;  // greedy unless the tweak asks otherwise
  if (tweak) tweak(*req);
  req->max_tokens = max_tokens;
  // `thinking`: split the reply into `text` (the answer) and `reasoning_text`, as http_server.cpp
  // builds the sink for a thinking request. The engine's dialect goes to the sink the way http_server.cpp
  // hands it (a Qwen dialect leaves the sink exactly as before; a Gemma one always splits).
  const bool chat = !raw_completion;
  auto sink = std::make_shared<r4dx::server::BufferingSink>(thinking && chat, true,
                                                            chat ? &e.engine->Dialect() : nullptr);
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

// --prompt-checkpoint (docs/server.md "Prefix cache") through RunRequest itself, on the 2026-09-26
// smoke failure: turn 1 answers " yes" (leading space) then EOS, the client replays it, the chat
// template's |trim renders "yes", so turn 2 does not extend the committed sequence.
//   - without a checkpoint: a full re-prefill (the bug, kept as the reference behaviour);
//   - with one: exactly one restore, only the tokens after turn 1's prompt are prefilled, and the
//     text equals a fresh engine's for the same conversation -- the fake's text hashes its whole
//     state, so a restore to the wrong position, or a stale tail, would change it;
//   - a fault in the prefill right after a restore: that request answers 500, and the next one
//     resets rather than restoring (Invalidate() drops the checkpoint with everything else) and
//     produces a fresh engine's text.
void CheckpointScenario(const std::string& tokenizer_dir, bool tp) {
  const char* tag = tp ? "tp2 checkpoint" : "tp1 checkpoint";
  const int failures_before = g_failures;
  r4dx::Tokenizer::Options topt;
  topt.allow_unimplemented_normalizer = true;  // as Engine::LoadAndStart
  const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tokenizer_dir, topt);
  const std::vector<r4dx::TokenId> space_yes = tok.encode(" yes");
  const std::vector<r4dx::TokenId> im_end = tok.encode("<|im_end|>", /*parse_special=*/true);
  const auto& eos = tok.eos_ids();
  if (space_yes.size() != 1 || im_end.size() != 1 || std::find(eos.begin(), eos.end(), im_end[0]) == eos.end()) {
    CHECK(false, "[%s] ' yes' and <|im_end|> must be single tokens and <|im_end|> an EOS", tag);
    return;
  }
  const std::string question = "Is there a circle in this image? Reply with exactly one word: yes or no.";
  const std::vector<r4dx::server::ChatMessage> turn1 = {Msg("user", question)};

  auto fresh_text = [&](const std::vector<r4dx::server::ChatMessage>& messages, int64_t max_tokens) {
    Harness fresh = MakeEngine(tokenizer_dir, /*dflash=*/false, tp);
    const auto want = Run(fresh, messages, max_tokens);
    fresh.engine->Shutdown();
    return want->errored ? std::string("<errored>") : want->text;
  };

  for (bool checkpoint : {false, true}) {
    Harness e = MakeEngine(tokenizer_dir, /*dflash=*/false, tp, checkpoint);
    e.fake->ScriptReplyToNextPrompt({space_yes[0], im_end[0]});
    const auto a = Run(e, turn1, 4);
    CHECK(!a->errored && a->text == " yes", "[%s] turn 1 answered '%s', want ' yes'", tag, a->text.c_str());
    const std::vector<r4dx::server::ChatMessage> turn2 = {
        Msg("user", question), Msg("assistant", a->text), Msg("user", "Now tell me what color the background is.")};
    const auto b = Run(e, turn2, 8);
    CHECK(!b->errored, "[%s] turn 2 failed: %s", tag, b->error_message.c_str());
    CHECK(b->prompt_tokens > a->prompt_tokens, "[%s] turn 2 is not longer than turn 1", tag);
    if (!checkpoint) {
      CHECK(b->timings.prompt_n == b->prompt_tokens && e.fake->restores == 0,
            "[%s] off: turn 2 prefilled %lld of %lld tokens, restores %d (want everything, 0)", tag,
            static_cast<long long>(b->timings.prompt_n), static_cast<long long>(b->prompt_tokens), e.fake->restores);
      e.engine->Shutdown();
      continue;
    }
    CHECK(e.fake->saves == 2 && e.fake->restores == 1, "[%s] saves %d restores %d, want 2 and 1", tag, e.fake->saves,
          e.fake->restores);
    CHECK(b->timings.prompt_n == b->prompt_tokens - a->prompt_tokens,
          "[%s] turn 2 prefilled %lld tokens, want %lld (everything after turn 1's prompt)", tag,
          static_cast<long long>(b->timings.prompt_n), static_cast<long long>(b->prompt_tokens - a->prompt_tokens));
    const std::string want2 = fresh_text(turn2, 8);
    CHECK(b->text == want2, "[%s] turn 2 text '%s', a fresh engine's '%s'", tag, b->text.c_str(), want2.c_str());

    // Turn 3 extends turn 2's prompt but not its reply (the client edited it), so it restores; the
    // restored state's first forward faults.
    std::vector<r4dx::server::ChatMessage> turn3 = turn2;
    turn3.push_back(Msg("assistant", "An edited answer."));
    turn3.push_back(Msg("user", "Thanks."));
    e.fake->ArmFault(1);
    const auto c = Run(e, turn3, 8);
    CHECK(c->errored && c->error_status == 500 && e.fake->restores == 2,
          "[%s] the faulting turn 3: errored=%d status=%d restores=%d (want 500 after a restore)", tag,
          static_cast<int>(c->errored), c->error_status, e.fake->restores);
    const auto d = Run(e, turn3, 8);
    CHECK(!d->errored && e.fake->restores == 2 && e.fake->recoveries == 1 && d->timings.prompt_n == d->prompt_tokens,
          "[%s] after the fault: errored=%d restores=%d recoveries=%d prefilled %lld of %lld (want a Reset() and a "
          "full prefill, no restore)",
          tag, static_cast<int>(d->errored), e.fake->restores, e.fake->recoveries,
          static_cast<long long>(d->timings.prompt_n), static_cast<long long>(d->prompt_tokens));
    const std::string want3 = fresh_text(turn3, 8);
    CHECK(d->text == want3, "[%s] turn 3 text '%s', a fresh engine's '%s'", tag, d->text.c_str(), want3.c_str());
    e.engine->Shutdown();
  }
  if (g_failures == failures_before) {
    std::printf("[%s] a trimmed ' yes' reply: off -> full re-prefill; on -> one restore, only the new tail "
                "prefilled, fresh-engine text; a fault after a restore -> reset, not restore\n",
                tag);
  }
}

// --prompt-checkpoint with thinking on (--think on): the prompt ends "<think>\n", and the client
// replays turn 1's answer WITHOUT its reasoning, which renders "<think>\n\n</think>\n\n<answer>" --
// "\n\n" is one token, so the replay diverges at the prompt's LAST token. The engine checkpoints one
// token early (engine.cpp's ckpt_back): turn 2 must restore and prefill everything from there, and
// match a fresh engine -- also after a regenerate of turn 1 in between, which restores to that
// checkpoint with only the held-back token left and must keep it.
void CheckpointThinkingScenario(const std::string& tokenizer_dir, bool tp) {
  const char* tag = tp ? "tp2 checkpoint+think" : "tp1 checkpoint+think";
  const int failures_before = g_failures;
  r4dx::Tokenizer::Options topt;
  topt.allow_unimplemented_normalizer = true;
  const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tokenizer_dir, topt);
  std::vector<r4dx::TokenId> reply = tok.encode("Okay.\n</think>\n\nYes.", /*parse_special=*/true);
  const std::vector<r4dx::TokenId> im_end = tok.encode("<|im_end|>", /*parse_special=*/true);
  reply.push_back(im_end.at(0));
  const std::vector<r4dx::server::ChatMessage> turn1 = {Msg("user", "Is 7 a prime number?")};

  Harness e = MakeEngine(tokenizer_dir, /*dflash=*/false, tp, /*checkpoint=*/true, /*thinking=*/true);
  e.fake->ScriptReplyToNextPrompt(std::vector<int32_t>(reply.begin(), reply.end()));
  const auto a = Run(e, turn1, 16, /*thinking=*/true);
  CHECK(!a->errored && a->text == "Yes." && a->reasoning_text == "Okay.",
        "[%s] turn 1: answer '%s' reasoning '%s', want 'Yes.' / 'Okay.'", tag, a->text.c_str(), a->reasoning_text.c_str());
  // A regenerate (the same turn 1 again): it restores the checkpoint and feeds only the held-back "\n",
  // and must KEEP that checkpoint rather than save a new one at the prompt's end -- or the next turn,
  // below, could no longer reuse it.
  const auto regen = Run(e, turn1, 16, /*thinking=*/true);
  CHECK(!regen->errored && regen->text == a->text && e.fake->restores == 1 && e.fake->saves == 1 &&
            regen->timings.prompt_n == 1,
        "[%s] regenerate: errored=%d text '%s' restores=%d saves=%d prompt_n=%lld (want a's text, 1, 1, 1)", tag,
        static_cast<int>(regen->errored), regen->text.c_str(), e.fake->restores, e.fake->saves,
        static_cast<long long>(regen->timings.prompt_n));
  // The client drops the reasoning, as most OpenAI clients do.
  const std::vector<r4dx::server::ChatMessage> turn2 = {Msg("user", "Is 7 a prime number?"), Msg("assistant", a->text),
                                                         Msg("user", "And 9?")};
  const auto b = Run(e, turn2, 8, /*thinking=*/true);
  CHECK(!b->errored && e.fake->restores == 2, "[%s] turn 2: errored=%d restores=%d (want a second restore)", tag,
        static_cast<int>(b->errored), e.fake->restores);
  CHECK(b->timings.prompt_n == b->prompt_tokens - (a->prompt_tokens - 1),
        "[%s] turn 2 prefilled %lld tokens, want %lld (everything from one token before turn 1's prompt end)", tag,
        static_cast<long long>(b->timings.prompt_n), static_cast<long long>(b->prompt_tokens - (a->prompt_tokens - 1)));
  {
    Harness fresh = MakeEngine(tokenizer_dir, /*dflash=*/false, tp, /*checkpoint=*/false, /*thinking=*/true);
    const auto want = Run(fresh, turn2, 8, /*thinking=*/true);
    CHECK(!want->errored && b->text == want->text && b->reasoning_text == want->reasoning_text,
          "[%s] turn 2 differs from a fresh engine's ('%s' vs '%s')", tag, b->text.c_str(), want->text.c_str());
    fresh.engine->Shutdown();
  }
  e.engine->Shutdown();
  if (g_failures == failures_before) {
    std::printf("[%s] a regenerate keeps the checkpoint; reasoning dropped from the replay: a restore from one "
                "token before turn 1's prompt end, fresh-engine text\n",
                tag);
  }
}

// --request-log through RunRequest's failure path (docs/server.md "Request log"): a request that
// faults mid-decode still gets exactly one line -- status 500, what was known when it failed, no
// verdict -- and the request after it, which takes the recovery Reset(), logs full_reset + reset_ms.
// The speculative mode is the engine's ("dflash" here). Numbers only, in every line.
void RequestLogFaultScenario(const std::string& tokenizer_dir) {
  const int failures_before = g_failures;
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "r4dx_engine_request_log_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string log_path = (dir / "requests.jsonl").string();
  {
    Harness e = MakeEngine(tokenizer_dir, /*dflash=*/true, /*tp=*/false, false, false, log_path);
    const auto turn1 = Run(e, {Msg("user", "What is 2 plus 2?")}, 6);
    CHECK(!turn1->errored, "turn 1 failed: %s", turn1->error_message.c_str());
    const std::vector<r4dx::server::ChatMessage> turn2 = {Msg("user", "What is 2 plus 2?"), Msg("assistant", turn1->text),
                                                           Msg("user", "And 3 plus 3?")};
    e.fake->ArmFault(3);
    const auto a = Run(e, turn2, 16);
    CHECK(a->errored && a->error_status == 500, "the fault request answered errored=%d", static_cast<int>(a->errored));
    const auto b = Run(e, turn2, 16);
    CHECK(!b->errored, "the request after the fault failed: %s", b->error_message.c_str());
    e.engine->Shutdown();
  }
  std::ifstream in(log_path, std::ios::binary);
  std::vector<nlohmann::json> lines;
  for (std::string l; std::getline(in, l);) lines.push_back(nlohmann::json::parse(l));
  in.close();
  CHECK(lines.size() == 3, "%zu log lines for 3 requests", lines.size());
  if (lines.size() == 3) {
    CHECK(lines[0]["http_status"] == 200 && lines[0]["speculative"] == "dflash" && lines[0]["finish_reason"] == "length" &&
              lines[0]["cached_tokens"] == 0 && lines[0]["completion_tokens"] == 6 && lines[0]["draft_n"].is_number(),
          "%s", lines[0].dump().c_str());
    CHECK(lines[1]["http_status"] == 500 && lines[1]["error_status"] == 500 && lines[1]["finish_reason"].is_null() &&
              lines[1]["completion_tokens"].is_null() && lines[1]["prompt_tokens"].is_number() &&
              lines[1]["cancelled"] == false,
          "%s", lines[1].dump().c_str());
    // turn 2 extended turn 1's state, so the faulting request had cached tokens when it died.
    CHECK(lines[1]["cached_tokens"].get<int64_t>() > 0 && lines[1]["full_reset"] == false, "%s", lines[1].dump().c_str());
    CHECK(lines[2]["http_status"] == 200 && lines[2]["full_reset"] == true && lines[2]["reset_ms"].is_number() &&
              lines[2]["cached_tokens"] == 0 && lines[2]["prompt_n"] == lines[2]["prompt_tokens"],
          "the request after a fault resets and prefills everything: %s", lines[2].dump().c_str());
  }
  std::filesystem::remove_all(dir);
  if (g_failures == failures_before) {
    std::printf("[request log] a mid-decode fault logs one 500 line with no verdict; the recovery request logs "
                "full_reset + reset_ms; dflash mode and draft_n recorded\n");
  }
}

// The Gemma 4 dialect through RunRequest itself (docs/gemma4-plan.md 5.3, task M1-14), against a CPU fake
// whose Config().arch is kGemma4 and the assembled Gemma tokenizer directory. Every reply is scripted
// (ScriptReplyToNextPrompt), so what is checked is the engine's own wiring:
//   * dialect auto-detected from the tokenizer dir, and a forced dialect that contradicts the model's
//     architecture fails the load;
//   * generation stops on EOS 106 and on 50 (the dialect's [1, 106, 50]; neither is decoded);
//   * thinking off, direct answer: no reasoning; thinking on: "<|channel>thought\n...<channel|>" is split
//     into reasoning_content / content (the sink's start state comes from OnStart);
//   * a parsed call stops on 50 with finish_reason "tool_calls";
//   * a tool message with no earlier assistant tool_calls is a 400; reasoning_effort is ignored;
//   * a raw /v1/completions prompt gets exactly one BOS, a chat prompt the template's one;
//   * --prompt-checkpoint holds back the whole "<|channel>thought\n<channel|>" suffix (4 tokens) with
//     thinking off, and nothing with thinking on.
void GemmaScenario(const std::string& dir) {
  using r4dx::model::Arch;
  using r4dx::server::DialectKind;
  const int failures_before = g_failures;
  r4dx::Tokenizer::Options topt;
  topt.allow_unimplemented_normalizer = true;
  const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(dir, topt);
  auto ids = [&](const std::string& s) {
    const std::vector<r4dx::TokenId> v = tok.encode(s, /*parse_special=*/true);
    return std::vector<int32_t>(v.begin(), v.end());
  };
  auto with = [](std::vector<int32_t> v, int32_t last) {
    v.push_back(last);
    return v;
  };
  auto thinking_on = [](r4dx::server::PendingRequest& r) { r.thinking.enabled = true; };
  auto thinking_off = [](r4dx::server::PendingRequest& r) { r.thinking.enabled = false; };
  const std::vector<r4dx::server::ChatMessage> user = {Msg("user", "Say hi.")};
  const nlohmann::json no_tools = nlohmann::json::array();

  {
    Harness e = MakeEngine(dir, /*dflash=*/false, /*tp=*/false, false, false, "", Arch::kGemma4);
    CHECK(e.engine->Dialect().kind == DialectKind::kGemma4, "[gemma] dialect not auto-detected as gemma4");

    // thinking off, direct answer, stop on <turn|> (106)
    e.fake->ScriptReplyToNextPrompt(with(ids("Hi there."), 106));
    const auto a = Run(e, user, 16, false, no_tools, false, "", thinking_off);
    CHECK(!a->errored && a->text == "Hi there." && a->reasoning_text.empty() && a->finish_reason == "stop" &&
              a->completion_tokens == static_cast<int64_t>(ids("Hi there.").size()),
          "[gemma] direct answer: errored=%d '%s' reasoning '%s' finish '%s' n=%lld", static_cast<int>(a->errored),
          a->text.c_str(), a->reasoning_text.c_str(), a->finish_reason.c_str(), static_cast<long long>(a->completion_tokens));
    CHECK(e.fake->first_prefill.size() > 1 && e.fake->first_prefill[0] == 2 && e.fake->first_prefill[1] != 2,
          "[gemma] a chat prompt must start with exactly the template's one <bos> (first ids %d, %d)",
          e.fake->first_prefill.size() > 0 ? e.fake->first_prefill[0] : -1,
          e.fake->first_prefill.size() > 1 ? e.fake->first_prefill[1] : -1);

    // thinking on: the model opens the span itself
    e.fake->ScriptReplyToNextPrompt(with(ids("<|channel>thought\nPlan it.<channel|>Done."), 106));
    const auto b = Run(e, user, 32, true, no_tools, false, "", thinking_on);
    CHECK(!b->errored && b->text == "Done." && b->reasoning_text == "Plan it." && b->reasoning_tokens > 0,
          "[gemma] thinking on: errored=%d text '%s' reasoning '%s' reasoning_tokens %lld", static_cast<int>(b->errored),
          b->text.c_str(), b->reasoning_text.c_str(), static_cast<long long>(b->reasoning_tokens));

    // EOS 50 (<|tool_response>) with a well-formed call: finish_reason tool_calls
    const nlohmann::json tools = nlohmann::json::parse(
        R"([{"type":"function","function":{"name":"get_weather","description":"Current weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])");
    e.fake->ScriptReplyToNextPrompt(
        with(ids("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"), 50));
    const auto c = Run(e, {Msg("user", "What is the weather in Paris?")}, 48, false, tools, false, "", thinking_off);
    CHECK(!c->errored && c->tool_calls.size() == 1 && c->tool_calls[0].name == "get_weather" &&
              c->tool_calls[0].arguments_json.find("Paris") != std::string::npos && c->finish_reason == "tool_calls",
          "[gemma] tool call: errored=%d calls=%zu finish '%s' text '%s'", static_cast<int>(c->errored), c->tool_calls.size(),
          c->finish_reason.c_str(), c->text.c_str());

    // a tool message nobody asked for: 400
    r4dx::server::ChatMessage tool_msg = Msg("tool", "{\"temp\": 20}");
    tool_msg.tool_call_id = "call_1";
    const auto d = Run(e, {Msg("user", "hi"), tool_msg}, 8, false, no_tools, false, "", thinking_off);
    CHECK(d->errored && d->error_status == 400, "[gemma] tool message without tool_calls: errored=%d status=%d",
          static_cast<int>(d->errored), d->error_status);

    // reasoning_effort (field and kwarg) is ignored, not an error
    const auto f = Run(e, user, 4, false, no_tools, false, "", [](r4dx::server::PendingRequest& r) {
      r.thinking.enabled = false;
      r.thinking.template_effort = "low";
      r.chat_template_kwargs = nlohmann::json{{"reasoning_effort", "low"}};
    });
    CHECK(!f->errored, "[gemma] reasoning_effort must be ignored: %s", f->error_message.c_str());

    // raw completion: one BOS in front of the plain tokens
    const auto g = Run(e, {}, 4, false, no_tools, /*raw_completion=*/true, "Hello world");
    const std::vector<r4dx::TokenId> plain = tok.encode("Hello world", /*parse_special=*/false);
    CHECK(!g->errored && e.fake->first_prefill.size() == plain.size() + 1 && e.fake->first_prefill[0] == 2 &&
              std::equal(plain.begin(), plain.end(), e.fake->first_prefill.begin() + 1),
          "[gemma] raw prompt: %zu prefill ids, want BOS + %zu", e.fake->first_prefill.size(), plain.size());
    e.engine->Shutdown();
  }

  // --prompt-checkpoint: the closed-span suffix is held back whole
  {
    const size_t suffix_tokens = ids("<|channel>thought\n<channel|>").size();
    CHECK(suffix_tokens == 4, "[gemma] the closed reasoning span is %zu tokens, expected 4", suffix_tokens);
    Harness e = MakeEngine(dir, false, false, /*checkpoint=*/true, false, "", Arch::kGemma4);
    e.fake->ScriptReplyToNextPrompt(with(ids("Ok."), 106));
    const auto a = Run(e, user, 8, false, no_tools, false, "", thinking_off);
    CHECK(!a->errored && e.fake->prefill_sizes.size() == 2 && e.fake->prefill_sizes[1] == suffix_tokens &&
              e.fake->saves == 1,
          "[gemma] thinking off: prefill split %zu calls (last %zu), saves %d; want 2 calls ending in %zu tokens, 1 save",
          e.fake->prefill_sizes.size(), e.fake->prefill_sizes.empty() ? size_t{0} : e.fake->prefill_sizes.back(),
          e.fake->saves, suffix_tokens);
    e.engine->Shutdown();
    Harness t = MakeEngine(dir, false, false, /*checkpoint=*/true, true, "", Arch::kGemma4);
    t.fake->ScriptReplyToNextPrompt(with(ids("<|channel>thought\nHm.<channel|>Ok."), 106));
    const auto b = Run(t, user, 16, true, no_tools, false, "", thinking_on);
    CHECK(!b->errored && t.fake->prefill_sizes.size() == 1 && t.fake->saves == 1,
          "[gemma] thinking on: prefill calls %zu (want 1), saves %d (want 1)", t.fake->prefill_sizes.size(), t.fake->saves);
    t.engine->Shutdown();
  }

  // a forced dialect that contradicts the model's architecture fails the load, before any request
  {
    bool threw = false;
    try {
      Harness e = MakeEngine(dir, false, false, false, false, "", Arch::kGemma4, DialectKind::kQwen35);
      e.engine->Shutdown();
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "[gemma] --dialect qwen35 against a gemma4 model must fail LoadAndStart");
  }
  if (g_failures == failures_before) {
    std::printf("[gemma] dialect auto-detected and arch-checked; EOS 106/50; direct and thought-wrapped replies "
                "split; tool call on EOS 50; tool-without-call 400; BOS once; checkpoint holds back 4 tokens\n");
  }
}

// HF repetition penalty through RunRequest itself (docs/lfm2-plan.md 5.4), on the Qwen tokenizer: the penalty is
// not dialect-specific. The fake's logits row is `floor` (0.5) everywhere and 1.0 at the scripted token, so a
// penalty p on a SEEN scripted token (1.0 / 3 < 0.5) moves greedy decoding to the lowest-index unseen token
// (id 0, "!"). Checked: a generated token is penalized from its second occurrence; a PROMPT token is seen from
// the first step; the sampled path (temperature > 0, top_k 1) penalizes the row too; penalty 1 changes nothing
// (and is what every request without the field carries); a speculative engine refuses penalty != 1 with a 400.
void RepetitionPenaltyScenario(const std::string& dir) {
  const int failures_before = g_failures;
  r4dx::Tokenizer::Options topt;
  topt.allow_unimplemented_normalizer = true;
  const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(dir, topt);
  const int32_t eos = static_cast<int32_t>(tok.eos_ids().front());
  const int32_t X = static_cast<int32_t>(tok.encode("x", /*parse_special=*/false).front());
  const std::string xs = tok.decode({X}, true);
  const std::vector<r4dx::server::ChatMessage> user = {Msg("user", "Say hi.")};
  const nlohmann::json no_tools = nlohmann::json::array();
  auto pen = [](float p, float temperature = 0.0f) {
    return [p, temperature](r4dx::server::PendingRequest& r) {
      r.sampling.repetition_penalty = p;
      r.sampling.temperature = temperature;
      r.sampling.top_k = temperature > 0.0f ? 1 : 0;  // a sampled request that is still deterministic
      r.sampling.has_seed = true;
      r.sampling.seed = 5;
    };
  };
  auto script = [&](Harness& e, std::vector<int32_t> reply) {
    reply.push_back(eos);
    e.fake->ScriptReplyToNextPrompt(std::move(reply));
  };
  auto fresh = [&](bool dflash = false) {
    Harness e = MakeEngine(dir, dflash, /*tp=*/false);
    e.fake->SetRowFloor(0.5f);
    return e;
  };

  for (float temperature : {0.0f, 0.7f}) {
    const char* tag = temperature > 0.0f ? "sampled" : "greedy";
    // (penalty 1 on the sampled path runs the device summary step, which the fake does not model: greedy only)
    if (temperature == 0.0f) {  // penalty 1: the scripted reply, untouched
      Harness e = fresh();
      script(e, {X, X, X});
      const auto a = Run(e, user, 8, false, no_tools, false, "", pen(1.0f, temperature));
      CHECK(!a->errored && a->text == xs + xs + xs && a->finish_reason == "stop",
            "[reppen %s] penalty 1: errored=%d text '%s'", tag, static_cast<int>(a->errored), a->text.c_str());
      e.engine->Shutdown();
    }
    {  // penalty 3: x is fine once, then 1/3 < 0.5 and the lowest unseen id takes over
      Harness e = fresh();
      script(e, {X, X, X});
      const auto b = Run(e, user, 8, false, no_tools, false, "", pen(3.0f, temperature));
      CHECK(!b->errored && b->text.rfind(xs, 0) == 0 && b->text != xs + xs + xs && b->completion_tokens >= 2,
            "[reppen %s] penalty 3: errored=%d text '%s' n=%lld", tag, static_cast<int>(b->errored), b->text.c_str(),
            static_cast<long long>(b->completion_tokens));
      e.engine->Shutdown();
    }
    {  // a prompt token is in the seen set from the very first step
      int32_t prompt_tok = -1;
      {
        Harness probe = fresh();
        script(probe, {X});
        Run(probe, user, 4, false, no_tools, false, "", pen(1.0f, temperature));
        CHECK(probe.fake->first_prefill.size() > 2, "[reppen %s] probe prefill too short", tag);
        if (probe.fake->first_prefill.size() > 2) prompt_tok = probe.fake->first_prefill[1];
        probe.engine->Shutdown();
      }
      const std::string piece = tok.decode({prompt_tok}, /*skip_special_tokens=*/false);
      if (temperature == 0.0f) {
        Harness e1 = fresh();
        script(e1, {prompt_tok});
        const auto a = Run(e1, user, 4, false, no_tools, false, "", pen(1.0f, temperature));
        CHECK(!a->errored && a->text == piece, "[reppen %s] prompt token, penalty 1: '%s' want '%s'", tag,
              a->text.c_str(), piece.c_str());
        e1.engine->Shutdown();
      }
      Harness e2 = fresh();
      script(e2, {prompt_tok});
      const auto b = Run(e2, user, 4, false, no_tools, false, "", pen(3.0f, temperature));
      CHECK(!b->errored && b->text != piece && b->text.rfind("!", 0) == 0,
            "[reppen %s] prompt token, penalty 3 must move to id 0: '%s'", tag, b->text.c_str());
      e2.engine->Shutdown();
    }
  }
  {  // a speculative engine cannot hook the penalty: 400, before anything runs
    Harness e = fresh(/*dflash=*/true);
    script(e, {X, X});
    const auto a = Run(e, user, 4, false, no_tools, false, "", pen(1.2f));
    CHECK(a->errored && a->error_status == 400, "[reppen dflash] penalty != 1: errored=%d status=%d",
          static_cast<int>(a->errored), a->error_status);
    const auto b = Run(e, user, 4, false, no_tools, false, "", pen(1.0f));
    CHECK(!b->errored, "[reppen dflash] penalty 1 still runs: %s", b->error_message.c_str());
    e.engine->Shutdown();
  }
  if (g_failures == failures_before) {
    std::printf("[reppen] generated and prompt tokens penalized (greedy and sampled); penalty 1 untouched; "
                "dflash + penalty -> 400\n");
  }
}

#ifdef R4DX_TEST_LFM2_ARCH
// The LFM2 dialect through RunRequest itself (docs/lfm2-plan.md 5.3, L2-5), against a CPU fake whose Config().arch
// is kLfm2Moe and the real LFM2.5 tokenizer directory (this scenario needs Arch::kLfm2Moe from lane C: it is
// compiled only when src/model/arch.h has it). Every reply is scripted, so what is checked is the engine wiring:
//   * dialect auto-detected from the tokenizer dir, a contradicting forced dialect fails the load, and the
//     dialect's own generation_config defaults (temperature 0.1, top_k 50, repetition_penalty 1.05) fill only
//     the --default-* fields the operator left alone;
//   * the model always thinks: the request's enable_thinking is ignored, `reasoning</think>answer` splits into
//     reasoning_content / content with reasoning_tokens reported, thinking requested off included;
//   * generation stops on <|im_end|> (124900); a Pythonic call stops it with finish_reason "tool_calls";
//   * a chat prompt carries the template's one <|startoftext|>, a raw prompt exactly one, prepended;
//   * --prompt-checkpoint holds back the one `<think>` token;
//   * a template error (a tool call whose arguments are a non-empty JSON string) is a 400.
void Lfm2Scenario(const std::string& dir) {
  using r4dx::model::Arch;
  using r4dx::server::DialectKind;
  const int failures_before = g_failures;
  const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(dir);
  auto ids = [&](const std::string& s) {
    const std::vector<r4dx::TokenId> v = tok.encode(s, /*parse_special=*/true);
    return std::vector<int32_t>(v.begin(), v.end());
  };
  auto with = [](std::vector<int32_t> v, int32_t last) {
    v.push_back(last);
    return v;
  };
  constexpr int32_t kImEnd = 124900, kBos = 124894, kThink = 124901;
  const std::vector<r4dx::server::ChatMessage> user = {Msg("user", "Say hi.")};
  const nlohmann::json no_tools = nlohmann::json::array();

  {
    Harness e = MakeEngine(dir, false, false, false, false, "", Arch::kLfm2Moe);
    CHECK(e.engine->Dialect().kind == DialectKind::kLfm2, "[lfm2] dialect not auto-detected as lfm2");
    CHECK(e.engine->DefaultThinking(), "[lfm2] /v1/models must report thinking on by default");
    const r4dx::server::SamplingParams& sd = e.engine->SamplingDefaults();
    CHECK(sd.temperature == 0.1f && sd.top_k == 50 && sd.repetition_penalty == 1.05f && sd.top_p == 1.0f &&
              sd.min_p == 0.0f,
          "[lfm2] dialect sampling defaults: t=%g k=%d p=%g rp=%g", sd.temperature, sd.top_k, sd.top_p,
          sd.repetition_penalty);
    CHECK(e.engine->ResolveThinking(r4dx::server::ThinkingControls{}) &&
              e.engine->ResolveThinking([] { r4dx::server::ThinkingControls t; t.enabled = false; return t; }()),
          "[lfm2] thinking must be forced on, whatever the request says");

    // a direct reply with reasoning, stop on <|im_end|>
    e.fake->ScriptReplyToNextPrompt(with(ids("Plan it.</think>Hi there."), kImEnd));
    const auto a = Run(e, user, 32, true, no_tools, false, "", [](r4dx::server::PendingRequest& r) {
      r.thinking.enabled = false;  // ignored
    });
    CHECK(!a->errored && a->text == "Hi there." && a->reasoning_text == "Plan it." && a->finish_reason == "stop" &&
              a->reasoning_tokens > 0,
          "[lfm2] reply: errored=%d text '%s' reasoning '%s' finish '%s' reasoning_tokens %lld",
          static_cast<int>(a->errored), a->text.c_str(), a->reasoning_text.c_str(), a->finish_reason.c_str(),
          static_cast<long long>(a->reasoning_tokens));
    CHECK(e.fake->first_prefill.size() > 2 && e.fake->first_prefill[0] == kBos && e.fake->first_prefill[1] != kBos &&
              e.fake->first_prefill.back() == kThink,
          "[lfm2] chat prompt: one <|startoftext|> first, an open <think> last (first %d, last %d)",
          e.fake->first_prefill.empty() ? -1 : e.fake->first_prefill.front(),
          e.fake->first_prefill.empty() ? -1 : e.fake->first_prefill.back());

    // a Pythonic call
    const nlohmann::json tools = nlohmann::json::parse(
        R"([{"type":"function","function":{"name":"get_weather","description":"Current weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])");
    e.fake->ScriptReplyToNextPrompt(with(
        ids("need it</think><|tool_call_start|>[get_weather(city='Paris', days=3)]<|tool_call_end|>"), kImEnd));
    const auto c = Run(e, {Msg("user", "Weather in Paris?")}, 64, true, tools);
    CHECK(!c->errored && c->tool_calls.size() == 1 && c->tool_calls[0].name == "get_weather" &&
              c->tool_calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":3}" && c->finish_reason == "tool_calls" &&
              c->reasoning_text == "need it",
          "[lfm2] tool call: errored=%d calls=%zu finish '%s' text '%s'", static_cast<int>(c->errored),
          c->tool_calls.size(), c->finish_reason.c_str(), c->text.c_str());

    // a tool round trip: the assistant turn's call (arguments decoded to a mapping) and the tool result render
    r4dx::server::ChatMessage asst = Msg("assistant", "");
    asst.content = std::nullopt;
    r4dx::server::ToolCallOut tc;
    tc.id = "call_1";
    tc.name = "get_weather";
    tc.arguments_json = "{\"city\":\"Paris\"}";
    asst.tool_calls.push_back(tc);
    r4dx::server::ChatMessage tool_msg = Msg("tool", "{\"temp\": 20}");
    tool_msg.tool_call_id = "call_1";
    e.fake->ScriptReplyToNextPrompt(with(ids("done</think>It is 20."), kImEnd));
    const auto d = Run(e, {Msg("user", "Weather in Paris?"), asst, tool_msg}, 32, true, tools);
    CHECK(!d->errored && d->text == "It is 20.", "[lfm2] tool round trip: errored=%d '%s' (%s)",
          static_cast<int>(d->errored), d->text.c_str(), d->error_message.c_str());

    // a template error is a 400, not a 500
    r4dx::server::ChatMessage bad = asst;
    bad.tool_calls[0].arguments_json = "\"not an object\"";
    const auto f = Run(e, {Msg("user", "x"), bad}, 8, true, no_tools);
    CHECK(f->errored && f->error_status == 400, "[lfm2] template error: errored=%d status=%d",
          static_cast<int>(f->errored), f->error_status);

    // a raw prompt: exactly one BOS in front of the plain tokens
    const auto g = Run(e, {}, 4, false, no_tools, /*raw_completion=*/true, "Hello world");
    const std::vector<r4dx::TokenId> plain = tok.encode("Hello world", /*parse_special=*/false);
    CHECK(!g->errored && e.fake->first_prefill.size() == plain.size() + 1 && e.fake->first_prefill[0] == kBos &&
              std::equal(plain.begin(), plain.end(), e.fake->first_prefill.begin() + 1),
          "[lfm2] raw prompt: %zu prefill ids, want BOS + %zu", e.fake->first_prefill.size(), plain.size());
    e.engine->Shutdown();
  }

  // the dialect's sampling defaults yield to the operator's explicit --default-* flags
  {
    Harness e = MakeEngine(dir, false, false, false, false, "", Arch::kLfm2Moe, std::nullopt,
                           [](r4dx::server::EngineOptions& o) {
                             o.sampling_defaults.temperature = 0.6f;
                             o.sampling_given.temperature = true;
                             o.sampling_defaults.repetition_penalty = 1.0f;
                             o.sampling_given.repetition_penalty = true;
                           });
    const r4dx::server::SamplingParams& sd = e.engine->SamplingDefaults();
    CHECK(sd.temperature == 0.6f && sd.repetition_penalty == 1.0f && sd.top_k == 50,
          "[lfm2] explicit flags must win: t=%g rp=%g k=%d", sd.temperature, sd.repetition_penalty, sd.top_k);
    e.engine->Shutdown();
  }

  // --prompt-checkpoint: only the single <think> token is held back
  {
    Harness e = MakeEngine(dir, false, false, /*checkpoint=*/true, false, "", Arch::kLfm2Moe);
    e.fake->ScriptReplyToNextPrompt(with(ids("Hm.</think>Ok."), kImEnd));
    const auto a = Run(e, user, 16, true, no_tools);
    CHECK(!a->errored && e.fake->prefill_sizes.size() == 2 && e.fake->prefill_sizes[1] == 1 && e.fake->saves == 1,
          "[lfm2] checkpoint: %zu prefill calls (last %zu), saves %d; want 2 calls ending in 1 token, 1 save",
          e.fake->prefill_sizes.size(), e.fake->prefill_sizes.empty() ? size_t{0} : e.fake->prefill_sizes.back(),
          e.fake->saves);
    e.engine->Shutdown();
  }

  // a forced dialect that contradicts the model's architecture fails the load, in both directions
  for (int dir_of_mismatch = 0; dir_of_mismatch < 2; ++dir_of_mismatch) {
    bool threw = false;
    try {
      Harness e = dir_of_mismatch == 0
                      ? MakeEngine(dir, false, false, false, false, "", Arch::kLfm2Moe, DialectKind::kQwen35)
                      : MakeEngine(dir, false, false, false, false, "", Arch::kQwen35, DialectKind::kLfm2);
      e.engine->Shutdown();
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "[lfm2] a dialect/arch mismatch (%s) must fail LoadAndStart", dir_of_mismatch == 0 ? "qwen35 on lfm2" : "lfm2 on qwen");
  }
  if (g_failures == failures_before) {
    std::printf("[lfm2] dialect auto-detected and arch-checked; thinking forced; reasoning split; Pythonic call -> "
                "tool_calls; BOS once; checkpoint holds back <think>; template error 400; sampling defaults honoured\n");
  }
}
#endif  // R4DX_TEST_LFM2_ARCH

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
      CheckpointScenario(tokenizer_dir, tp);
      CheckpointThinkingScenario(tokenizer_dir, tp);
    }
    RequestLogFaultScenario(tokenizer_dir);
    RepetitionPenaltyScenario(tokenizer_dir);
#ifdef R4DX_TEST_LFM2_ARCH
    const std::string lfm2_dir = R4DX_LFM2_TOKENIZER_DIR;
    if (std::filesystem::exists(std::filesystem::path(lfm2_dir) / "tokenizer.json")) {
      Lfm2Scenario(lfm2_dir);
    } else {
      std::fprintf(stderr, "NOTE: %s/tokenizer.json not found; the LFM2 scenario was skipped\n", lfm2_dir.c_str());
    }
#else
    std::fprintf(stderr, "NOTE: src/model/arch.h has no Arch::kLfm2Moe yet (lane C); the LFM2 scenario is not compiled\n");
#endif
    const std::string gemma_dir = R4DX_GEMMA_TOKENIZER_DIR;
    if (std::filesystem::exists(std::filesystem::path(gemma_dir) / "tokenizer.json")) {
      GemmaScenario(gemma_dir);
    } else {
      std::fprintf(stderr, "NOTE: %s/tokenizer.json not found; the Gemma scenario was skipped\n", gemma_dir.c_str());
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
