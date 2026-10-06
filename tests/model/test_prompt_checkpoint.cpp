// tests/model/test_prompt_checkpoint.cpp -- Model::SaveCheckpoint / RestoreCheckpoint (docs/server.md
// "Prefix cache", r4dx-server --prompt-checkpoint) on the 4-layer MTP test container.
//
// The contract: after Prefill(P); SaveCheckpoint(); <any generation>; RestoreCheckpoint(), the model
// is exactly where Prefill(P) left it, so Prefill(T) returns logits BIT-IDENTICAL to a model that ran
// Prefill(P); Prefill(T) with nothing in between, and the greedy continuation is the same. (Not
// compared with one Prefill(P + T): the chunk boundaries differ, which is the extend path's
// behaviour too, and not what the checkpoint is for.) Checked, per layout:
//   1. plain greedy decode in between, on a Model without speculation (window 1) and on one sized
//      for --mtp 3 (a plain decode step there threads num_accepted = 1);
//   2. speculative rounds that commit several tokens at once, restored straight after -- oracle
//      drafts through VerifyWindow + CommitVerifiedWindow, as test_mtp.cpp's
//      CheckChatMultiTurnMidRoundStop does, because this container's MTP head never agrees with its
//      truncated target. The live GDN state then sits in window slot n-1 and at conv history offset
//      n-1 (gdn_state.h), not where the checkpoint was copied from, so a restore that missed either
//      (or left the num_accepted thread armed) shows up here; and real MTP rounds (the seed row);
//   3. a second restore of the same checkpoint (the model allows it; the server replaces its
//      checkpoint on every request);
//   4. the preconditions: no checkpoint storage, a decode step or a bare verify window since the
//      prefill, Reset() dropping the checkpoint;
//   5. tensor parallel (--tp-mode emulate, both ranks on device 1): the same contract through TpModel.
// Plus CheckExtendAfterMultiTokenRound, the regression test for a bug on the OTHER reuse path that
// this test found (see its comment).
//
// Device 1 (HIP_VISIBLE_DEVICES=1, from CMake), SKIP 77 without the container.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "local_text_model.h"
#include "model.h"
#include "test_common.h"
#include "text_model.h"

namespace {

using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::TextModel;
using r4dx::model::TpOptions;

const char* kContainerPath = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
constexpr int64_t kDraftK = 3;
constexpr int kContinue = 8;  // greedy tokens compared after Prefill(T)

int g_failures = 0;

#define CHECK(cond, ...)                                                          \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);      \
      std::fprintf(stderr, __VA_ARGS__);                                          \
      std::fprintf(stderr, "\n");                                                 \
      ++g_failures;                                                               \
    }                                                                             \
  } while (0)

// Deterministic in-vocabulary ids; P is two prefill chunks and T crosses a chunk boundary of its own.
std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 100 + (i * 41 + salt * 977) % 5000;
  return v;
}
const std::vector<int32_t> kP = Tokens(100, 1);
const std::vector<int32_t> kT = Tokens(70, 2);

int32_t Argmax(const std::vector<float>& logits) {
  size_t best = 0;
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<int32_t>(best);
}

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    m = std::max(m, d < 0 ? -d : d);
  }
  return m;
}

template <class M>
std::vector<int32_t> Continue(M& m, const std::vector<float>& logits) {
  std::vector<int32_t> out;
  int32_t tok = Argmax(logits);
  for (int i = 0; i < kContinue; ++i) {
    out.push_back(tok);
    tok = m.DecodeStepGreedy(tok);
  }
  return out;
}

bool Throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

ModelOptions Options(Layout layout, int64_t mtp_k, bool checkpoint) {
  ModelOptions o;
  o.container_path = kContainerPath;
  o.layout = layout;
  o.max_ctx = 512;
  o.layer_limit = 4;
  o.mtp_draft_k = mtp_k;
  o.vision = ModelOptions::VisionMode::kOff;
  o.prompt_checkpoint = checkpoint;
  return o;
}

// The greedy continuation after P, fed one token at a time -- the oracle drafts, and the sequential
// twin of the oracle rounds.
std::vector<int32_t> Oracle(Model& m, int n) {
  m.Reset();
  std::vector<int32_t> seq;
  int32_t tok = Argmax(m.Prefill(kP));
  for (int i = 0; i < n; ++i) {
    seq.push_back(tok);
    tok = m.DecodeStepGreedy(tok);
  }
  return seq;
}

// `rounds` rounds of k+1 candidates drawn from `oracle`, each committed as far as the real preds
// accept it. Returns how many tokens were committed; `last_committed` gets the final round's count.
int64_t OracleRounds(Model& m, const std::vector<int32_t>& oracle, int rounds, int64_t* last_committed) {
  size_t at = 0;
  for (int r = 0; r < rounds; ++r) {
    std::vector<int32_t> cand(oracle.begin() + static_cast<ptrdiff_t>(at),
                              oracle.begin() + static_cast<ptrdiff_t>(at + kDraftK + 1));
    const std::vector<int32_t> preds = m.VerifyWindow(cand);
    int64_t accepted = 0;
    while (accepted < kDraftK && preds[static_cast<size_t>(accepted)] == cand[static_cast<size_t>(accepted + 1)]) {
      ++accepted;
    }
    m.CommitVerifiedWindow(accepted + 1);
    *last_committed = accepted + 1;
    at += static_cast<size_t>(accepted + 1);
  }
  return static_cast<int64_t>(at);
}

// 1-3 for one Model configuration. `generate` runs whatever comes between the save and the restore.
void CheckRestoreExact(Model& m, const char* what, const std::function<void(Model&)>& generate) {
  m.Reset();
  m.Prefill(kP);
  const std::vector<float> want = m.Prefill(kT);
  const std::vector<int32_t> want_cont = Continue(m, want);

  m.Reset();
  m.Prefill(kP);
  m.SaveCheckpoint();
  CHECK(m.CheckpointPosition() == static_cast<int64_t>(kP.size()), "[%s] checkpoint at %lld, want %zu", what,
        static_cast<long long>(m.CheckpointPosition()), kP.size());
  generate(m);
  for (int restore = 1; restore <= 2; ++restore) {
    m.RestoreCheckpoint();
    CHECK(m.PositionCount() == static_cast<int64_t>(kP.size()), "[%s] restore %d: position %lld, want %zu", what,
          restore, static_cast<long long>(m.PositionCount()), kP.size());
    const std::vector<float> got = m.Prefill(kT);
    CHECK(SameBits(got, want), "[%s] restore %d: Prefill(T) logits differ (max |d| %.3g)", what, restore,
          MaxAbsDiff(got, want));
    const std::vector<int32_t> got_cont = Continue(m, got);
    CHECK(got_cont == want_cont, "[%s] restore %d: greedy continuation differs", what, restore);
  }
  std::fprintf(stderr, "[check] %s: restore -> Prefill(T) bit-identical to Prefill(P); Prefill(T), twice\n", what);
}

// Not the checkpoint: the EXTEND path (PrefixState's fed() reuse, which the checkpoint only falls
// back from; r4dx-cli --chat's reuse too). After a speculative round that committed n > 1 tokens,
// the live GDN state is in window slot n-1 / conv history offset n-1, while a Prefill reads window 0
// / offset 0 -- Model::CollapseSpeculativeWindow moves it there first. Compares
//   Prefill(P); oracle rounds committing G (the last one committing k+1); Prefill(T)
// with the sequential twin
//   Prefill(P); DecodeStepGreedy over G; Prefill(T)
// -- equal bit for bit when the hand-over is right (VerifyWindow's rows equal sequential decode's,
// test_mtp.cpp's CheckVerifyMatchesSequential). Found with this check, 2026-09-26: without the move,
// a 1-token T after a 4-token round was rel L2 6.4e-3 off, with a different argmax on w4a16.
double RelL2(const std::vector<float>& ref, const std::vector<float>& got) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

void CheckExtendAfterMultiTokenRound(Model& m, const std::vector<int32_t>& oracle, const std::vector<int32_t>& tail,
                                     const char* what) {
  m.Reset();
  m.Prefill(kP);
  int64_t last = 0;
  const int64_t n = OracleRounds(m, oracle, 3, &last);
  CHECK(last > 1, "[%s] the last oracle round committed %lld token(s); the case needs > 1", what,
        static_cast<long long>(last));
  const std::vector<float> spec = m.Prefill(tail);

  m.Reset();
  m.Prefill(kP);
  for (int64_t i = 0; i < n; ++i) m.DecodeStepGreedy(oracle[static_cast<size_t>(i)]);
  const std::vector<float> seq = m.Prefill(tail);
  std::fprintf(stderr,
               "[check] %s: extend by %zu token(s) after a %lld-token round vs the sequential twin: max |d| %.3g, "
               "rel L2 %.3g, argmax %d vs %d\n",
               what, tail.size(), static_cast<long long>(last), MaxAbsDiff(spec, seq), RelL2(seq, spec), Argmax(spec),
               Argmax(seq));
  CHECK(SameBits(spec, seq), "[%s] extend path: Prefill of %zu token(s) after a multi-token round is not bit-identical",
        what, tail.size());
}

void CheckLayout(Layout layout) {
  const std::string tag = LayoutName(layout);
  {
    // 1 (window 1) + 4.
    Model m = Model::Load(Options(layout, /*mtp_k=*/0, /*checkpoint=*/true));
    CheckRestoreExact(m, (tag + " mtp0 plain decode").c_str(), [](Model& x) {
      int32_t tok = 1234;
      for (int i = 0; i < 20; ++i) tok = x.DecodeStepGreedy(tok);
    });
    m.Reset();
    CHECK(m.CheckpointPosition() == -1 && Throws([&] { m.RestoreCheckpoint(); }),
          "[%s] Reset() did not drop the checkpoint", tag.c_str());
  }
  {
    Model m = Model::Load(Options(layout, /*mtp_k=*/0, /*checkpoint=*/false));
    m.Prefill(kP);
    CHECK(Throws([&] { m.SaveCheckpoint(); }), "[%s] SaveCheckpoint without storage did not throw", tag.c_str());
  }
  {
    Model m = Model::Load(Options(layout, kDraftK, /*checkpoint=*/true));
    const std::vector<int32_t> oracle = Oracle(m, 3 * static_cast<int>(kDraftK + 1) + 1);
    // 1 (window k+1, plain steps) and 2 (multi-token rounds, then real MTP rounds on top).
    CheckRestoreExact(m, (tag + " mtp3 plain decode").c_str(), [](Model& x) {
      int32_t tok = 1234;
      for (int i = 0; i < 20; ++i) tok = x.DecodeStepGreedy(tok);
    });
    // The restore runs straight after the oracle rounds, so the live state is in window slot n-1 at
    // that moment (and Prefill's CollapseSpeculativeWindow would move it, were the restore to leave
    // the num_accepted thread armed).
    CheckRestoreExact(m, (tag + " mtp3 multi-token rounds").c_str(), [&](Model& x) {
      int64_t last = 0;
      OracleRounds(x, oracle, 3, &last);
      CHECK(last > 1, "[%s] the last oracle round committed %lld token(s); the case needs > 1", tag.c_str(),
            static_cast<long long>(last));
    });
    CheckRestoreExact(m, (tag + " mtp3 real MTP rounds").c_str(), [&](Model& x) {
      int32_t tok = 1234;
      for (int r = 0; r < 4; ++r) tok = x.DecodeStepMtpGreedy(tok, kDraftK).back();
    });
    // 4: a decode step, or a bare verify window, since the prefill.
    m.Reset();
    m.Prefill(kP);
    m.DecodeStepGreedy(1234);
    CHECK(Throws([&] { m.SaveCheckpoint(); }), "[%s] SaveCheckpoint after a decode step did not throw", tag.c_str());
    m.Reset();
    m.Prefill(kP);
    m.VerifyWindow({oracle[0], oracle[1]});
    CHECK(Throws([&] { m.SaveCheckpoint(); }), "[%s] SaveCheckpoint after a verify window did not throw",
          tag.c_str());
    CheckExtendAfterMultiTokenRound(m, oracle, kT, (tag + " mtp3").c_str());
    CheckExtendAfterMultiTokenRound(m, oracle, {kT[0]}, (tag + " mtp3").c_str());
  }
}

// 5: the same contract through TpModel, both ranks on device 1.
void CheckTpEmulation(Layout layout) {
  TpOptions tp;
  tp.world = 2;
  tp.mode = TpOptions::Mode::kEmulate;
  std::unique_ptr<TextModel> m = r4dx::model::LoadTextModel(Options(layout, kDraftK, /*checkpoint=*/true), tp);
  m->Reset();
  m->Prefill(kP);
  const std::vector<float> want = m->Prefill(kT);
  const std::vector<int32_t> want_cont = Continue(*m, want);

  m->Reset();
  m->Prefill(kP);
  m->SaveCheckpoint();
  int32_t tok = 1234;
  for (int r = 0; r < 4; ++r) tok = m->DecodeStepMtpGreedy(tok, kDraftK).back();
  for (int i = 0; i < 8; ++i) tok = m->DecodeStepGreedy(tok);
  m->RestoreCheckpoint();
  CHECK(m->PositionCount() == static_cast<int64_t>(kP.size()), "[tp2 %s] position %lld after restore, want %zu",
        LayoutName(layout), static_cast<long long>(m->PositionCount()), kP.size());
  const std::vector<float> got = m->Prefill(kT);
  CHECK(SameBits(got, want), "[tp2 %s] Prefill(T) after a restore differs (max |d| %.3g)", LayoutName(layout),
        MaxAbsDiff(got, want));
  CHECK(Continue(*m, got) == want_cont, "[tp2 %s] greedy continuation after a restore differs", LayoutName(layout));
  std::fprintf(stderr, "[check] tp2 emulate %s: restore -> Prefill(T) bit-identical\n", LayoutName(layout));
}

int RunTest() {
  if (!r4dx_test::FileExists(kContainerPath)) return r4dx_test::SkipMissing(kContainerPath);
  for (Layout layout : {Layout::kBf16, Layout::kW4a16}) CheckLayout(layout);
  CheckTpEmulation(Layout::kW4a16);
  if (g_failures != 0) {
    std::fprintf(stderr, "test_prompt_checkpoint: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "[PASS] test_prompt_checkpoint\n");
  return 0;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_prompt_checkpoint", RunTest); }
