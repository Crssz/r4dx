// tests/model/test_gdn_write_once_model.cpp -- model level, GPU (device 1): the write-once GDN state
// (ModelOptions::gdn_write_once, docs/gdn-write-once.md) against the window-slot path, on the 4-layer MTP test
// container. Two Models (or two TpModels in emulate mode) are loaded in one process, gdn_write_once = 0 and 1,
// and driven through the same scripts; every observable must be BITWISE equal between them, after every event:
//   logits (verify windows and decode steps), predicted / emitted tokens, the live GDN state of every layer
//   (Model::DebugLiveGdnDigest), and everything else the per-sequence state digest covers except the window-slot
//   allocation itself (conv state, KV caches, MTP KV, position).
// Scripts (each with the digests taken at the end only, and again after EVERY event -- a digest taken while a
// committed prefix is pending flushes it, which must be invisible to the continuing run):
//   rounds     verify + commit n for T in {8, 4, 5, 1, 3} and many n (1, the anchor alone, is the case a naive
//              "store row 0" scheme gets wrong; 8 is the full window), plain decode steps in between, a Prefill
//              extend after a multi-token round, after an n = 1 round of T > 1 (the case that needs a
//              materialise although legacy has nothing to move) and after a plain step;
//   checkpoint Save, rounds, Restore, Prefill: bit-identical to Save, nothing, Prefill (test_prompt_checkpoint's
//              cases 1 to 3 with write-once);
//   reset      rounds, Reset(), Prefill: equal to a fresh run;
//   mtp        real MTP rounds (k = 3, window 4), the committed count is whatever the head gets accepted;
//   bare       two VerifyWindow calls with no commit between, after a round that committed n > 1: the write-once
//              rows are equal (the legacy path's are not: the latent bug this design fixes), then the commit;
//   bounds     CommitVerifiedWindow(n) with n > the last verify's rows is refused by the write-once Model;
//   window 1   a Model without a speculative window ignores the request (the window-slot manager, one slot);
//   vram       the write-once Model's GDN allocation is smaller (printed).
// Part TP: the same scripts through TpModel (emulate, both ranks on device 1) with the switch 0 and 1; the
// per-rank live digests are compared per rank.
//
// Device 1 (HIP_VISIBLE_DEVICES=1, from CMake), SKIP 77 without the 4-layer container. Links the
// R4DX_TP_TESTING library variant (DebugStateDigest / DebugLiveGdnDigest / RunCollectiveForTest).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "test_common.h"
#include "tp_model.h"

namespace {

using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;

const char* kContainerPath = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");

int g_failures = 0;
#define CHECK(cond, ...)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                     \
      std::fprintf(stderr, "\n");                                            \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 100 + (i * 41 + salt * 977) % 5000;
  return v;
}
const std::vector<int32_t> kP = Tokens(100, 1);
const std::vector<int32_t> kT = Tokens(70, 2);

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
uint64_t Fnv(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}
int32_t Argmax(const std::vector<float>& logits) {
  size_t best = 0;
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<int32_t>(best);
}

using Digest = std::vector<std::pair<std::string, uint64_t>>;

// ---- an engine: one Model, or one TpModel (both ranks), behind the calls the scripts make -------------------
class Engine {
 public:
  virtual ~Engine() = default;
  virtual std::vector<float> Prefill(const std::vector<int32_t>& ids) = 0;
  virtual std::vector<float> DecodeStep(int32_t tok) = 0;
  virtual std::vector<int32_t> Verify(const std::vector<int32_t>& cands, std::vector<float>* logits) = 0;
  virtual void Commit(int64_t n) = 0;
  virtual std::vector<int32_t> MtpRound(int32_t tok, int64_t k) = 0;
  virtual void Reset() = 0;
  virtual void Save() = 0;
  virtual void Restore() = 0;
  // The state the two modes must agree on: per rank the live GDN digests and the state digest without the
  // recurrent allocations ("gdn.rec.*" differs by construction).
  virtual Digest State() = 0;
  virtual bool WriteOnce() = 0;
  virtual int64_t GdnBytes() = 0;
  virtual bool CommitRefused(int64_t n) = 0;
};

Digest Comparable(Model& m) {
  Digest out;
  for (auto& kv : m.DebugStateDigest()) {
    if (kv.first.rfind("gdn.rec.", 0) == 0) continue;
    out.push_back(kv);
  }
  for (auto& kv : m.DebugLiveGdnDigest()) out.push_back(kv);
  return out;
}

class SingleEngine : public Engine {
 public:
  explicit SingleEngine(const ModelOptions& o) : m_(Model::Load(o)) {}
  std::vector<float> Prefill(const std::vector<int32_t>& ids) override { return m_.Prefill(ids); }
  std::vector<float> DecodeStep(int32_t tok) override { return m_.DecodeStep(tok); }
  std::vector<int32_t> Verify(const std::vector<int32_t>& c, std::vector<float>* l) override { return m_.VerifyWindow(c, l); }
  void Commit(int64_t n) override { m_.CommitVerifiedWindow(n); }
  std::vector<int32_t> MtpRound(int32_t tok, int64_t k) override { return m_.DecodeStepMtpGreedy(tok, k); }
  void Reset() override { m_.Reset(); }
  void Save() override { m_.SaveCheckpoint(); }
  void Restore() override { m_.RestoreCheckpoint(); }
  Digest State() override { return Comparable(m_); }
  bool WriteOnce() override { return m_.GdnWriteOnceEnabled(); }
  int64_t GdnBytes() override { return m_.GdnStateBytes(); }
  bool CommitRefused(int64_t n) override {
    try {
      m_.CommitVerifiedWindow(n);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  }

 private:
  Model m_;
};

class TpEngine : public Engine {
 public:
  explicit TpEngine(const ModelOptions& o) {
    TpOptions t;
    t.world = 2;
    t.mode = TpOptions::Mode::kEmulate;
    m_ = TpModel::Load(o, t);
  }
  std::vector<float> Prefill(const std::vector<int32_t>& ids) override { return m_->Prefill(ids); }
  std::vector<float> DecodeStep(int32_t tok) override { return m_->DecodeStep(tok); }
  std::vector<int32_t> Verify(const std::vector<int32_t>& c, std::vector<float>* l) override {
    std::vector<std::vector<int32_t>> preds(2);
    std::vector<std::vector<float>> logits(2);
    m_->RunCollectiveForTest([&](Model& m, int rank) {
      preds[static_cast<size_t>(rank)] = m.VerifyWindow(c, &logits[static_cast<size_t>(rank)]);
    });
    CHECK(preds[0] == preds[1] && SameBits(logits[0], logits[1]), "the two ranks' verify results differ");
    if (l != nullptr) *l = logits[0];
    return preds[0];
  }
  void Commit(int64_t n) override {
    m_->RunCollectiveForTest([&](Model& m, int) { m.CommitVerifiedWindow(n); });
  }
  std::vector<int32_t> MtpRound(int32_t tok, int64_t k) override { return m_->DecodeStepMtpGreedy(tok, k); }
  void Reset() override { m_->Reset(); }
  void Save() override { m_->SaveCheckpoint(); }
  void Restore() override { m_->RestoreCheckpoint(); }
  Digest State() override {
    std::vector<Digest> per(2);
    m_->RunCollectiveForTest([&](Model& m, int rank) { per[static_cast<size_t>(rank)] = Comparable(m); });
    Digest out;
    for (int r = 0; r < 2; ++r) {
      for (auto& kv : per[static_cast<size_t>(r)]) out.emplace_back("rank" + std::to_string(r) + "/" + kv.first, kv.second);
    }
    return out;
  }
  bool WriteOnce() override {
    bool on[2] = {false, false};
    m_->RunCollectiveForTest([&](Model& m, int rank) { on[rank] = m.GdnWriteOnceEnabled(); });
    return on[0] && on[1];
  }
  int64_t GdnBytes() override {
    int64_t b[2] = {0, 0};
    m_->RunCollectiveForTest([&](Model& m, int rank) { b[rank] = m.GdnStateBytes(); });
    return b[0];
  }
  bool CommitRefused(int64_t n) override {
    bool refused[2] = {false, false};
    m_->RunCollectiveForTest([&](Model& m, int rank) {
      try {
        m.CommitVerifiedWindow(n);
      } catch (const std::runtime_error&) {
        refused[rank] = true;
      }
    });
    return refused[0] && refused[1];
  }

 private:
  std::unique_ptr<TpModel> m_;
};

ModelOptions Options(Layout layout, int64_t mtp_k, int64_t dflash_k, int gdn_wo, bool checkpoint) {
  ModelOptions o;
  o.container_path = kContainerPath;
  o.layout = layout;
  o.max_ctx = 1024;
  o.layer_limit = 4;
  o.mtp_draft_k = mtp_k;
  o.dflash_draft_k = dflash_k;  // sizes the verify window; no drafter is loaded
  o.vision = ModelOptions::VisionMode::kOff;
  o.prompt_checkpoint = checkpoint;
  o.gdn_write_once = gdn_wo;
  return o;
}

// ---- scripts -------------------------------------------------------------------------------------------------
enum class K { kVerify, kPlain, kPrefill, kBare, kMtp, kSave, kRestore, kReset };
struct Ev {
  K kind;
  int T = 0, n = 0;
  int salt = 0;
};

struct Trace {
  std::vector<std::pair<std::string, uint64_t>> obs;     // logits / tokens, in order
  std::vector<std::pair<std::string, Digest>> states;    // taken at the digest points
};

int32_t g_tok = 1234;  // the "last emitted token" the plain / MTP steps feed

void Run(Engine& e, const std::vector<Ev>& script, bool digest_each, Trace* tr, const char* tag) {
  e.Reset();
  {
    const std::vector<float> lg = e.Prefill(kP);
    tr->obs.emplace_back(std::string(tag) + "/prefill/P", Fnv(lg.data(), lg.size() * sizeof(float)));
  }
  g_tok = 1234;
  int step = 0;
  for (const Ev& ev : script) {
    const std::string label = std::string(tag) + "/" + std::to_string(step) + "/";
    switch (ev.kind) {
      case K::kVerify: {
        const std::vector<int32_t> cands = Tokens(ev.T, 40 + ev.salt + step);
        std::vector<float> lg;
        const std::vector<int32_t> preds = e.Verify(cands, &lg);
        tr->obs.emplace_back(label + "verify" + std::to_string(ev.T) + "/logits", Fnv(lg.data(), lg.size() * sizeof(float)));
        tr->obs.emplace_back(label + "verify/preds", Fnv(preds.data(), preds.size() * sizeof(int32_t)));
        e.Commit(ev.n);
        break;
      }
      case K::kPlain: {
        const std::vector<float> lg = e.DecodeStep(g_tok);
        tr->obs.emplace_back(label + "plain/logits", Fnv(lg.data(), lg.size() * sizeof(float)));
        g_tok = Argmax(lg);
        break;
      }
      case K::kPrefill: {
        const std::vector<int32_t> ids = ev.T == 1 ? std::vector<int32_t>{kT[0]} : std::vector<int32_t>(kT.begin(), kT.begin() + ev.T);
        const std::vector<float> lg = e.Prefill(ids);
        tr->obs.emplace_back(label + "prefill/logits", Fnv(lg.data(), lg.size() * sizeof(float)));
        g_tok = Argmax(lg);
        break;
      }
      case K::kMtp: {
        const std::vector<int32_t> r = e.MtpRound(g_tok, ev.T);
        tr->obs.emplace_back(label + "mtp/tokens", Fnv(r.data(), r.size() * sizeof(int32_t)));
        g_tok = r.back();
        break;
      }
      case K::kSave: e.Save(); break;
      case K::kRestore: e.Restore(); break;
      case K::kReset: {
        e.Reset();
        const std::vector<float> lg = e.Prefill(kP);
        tr->obs.emplace_back(label + "reset/prefill", Fnv(lg.data(), lg.size() * sizeof(float)));
        break;
      }
      case K::kBare: break;  // handled by CheckBare
    }
    if (digest_each) tr->states.emplace_back(label + "state", e.State());
    ++step;
  }
  tr->states.emplace_back(std::string(tag) + "/end", e.State());
}

bool SameTrace(const Trace& a, const Trace& b, const char* what, bool compare_states = true) {
  bool ok = a.obs.size() == b.obs.size();
  CHECK(ok, "[%s] the two traces have different lengths (%zu vs %zu)", what, a.obs.size(), b.obs.size());
  for (size_t i = 0; ok && i < a.obs.size(); ++i) {
    if (a.obs[i].first != b.obs[i].first || a.obs[i].second != b.obs[i].second) {
      CHECK(false, "[%s] observable %zu (%s) differs between the window-slot and the write-once Model", what, i, a.obs[i].first.c_str());
      ok = false;
    }
  }
  if (compare_states) {
    CHECK(a.states.size() == b.states.size(), "[%s] state trace lengths differ", what);
    for (size_t i = 0; i < a.states.size() && i < b.states.size(); ++i) {
      const Digest& x = a.states[i].second;
      const Digest& y = b.states[i].second;
      CHECK(x.size() == y.size(), "[%s] %s: digest entry counts differ (%zu vs %zu)", what, a.states[i].first.c_str(), x.size(), y.size());
      for (size_t j = 0; j < x.size() && j < y.size(); ++j) {
        if (x[j] != y[j]) {
          CHECK(false, "[%s] %s: state entry %s differs", what, a.states[i].first.c_str(), x[j].first.c_str());
          ok = false;
        }
      }
    }
  }
  return ok;
}

// the scripts
const std::vector<Ev> kRounds = {
    {K::kVerify, 8, 3},  {K::kVerify, 8, 8},  {K::kVerify, 8, 1},  {K::kVerify, 4, 2},  {K::kPlain},
    {K::kVerify, 5, 5},  {K::kVerify, 8, 6},  {K::kVerify, 1, 1},  {K::kVerify, 8, 2},  {K::kPlain},
    {K::kPlain},         {K::kVerify, 3, 3},  {K::kPrefill, 70},   // extend after a multi-token round
    {K::kPlain},         {K::kVerify, 8, 4},  {K::kVerify, 8, 1},  {K::kPrefill, 1},    // extend after an n = 1 round of T > 1
    {K::kPlain},         {K::kVerify, 6, 6},  {K::kPlain},         {K::kPrefill, 33},   // extend after a plain step
    {K::kVerify, 8, 7},  {K::kVerify, 2, 1},  {K::kVerify, 8, 8},
};
const std::vector<Ev> kCheckpoint = {
    {K::kSave},          {K::kVerify, 8, 3}, {K::kVerify, 8, 8}, {K::kPlain},       {K::kVerify, 8, 1},
    {K::kRestore},       {K::kPrefill, 70},  {K::kPlain},        {K::kVerify, 8, 5}, {K::kRestore},
    {K::kVerify, 8, 2},  {K::kRestore},      {K::kPrefill, 70},  {K::kVerify, 4, 4}, {K::kPlain},
};
const std::vector<Ev> kReset = {
    {K::kVerify, 8, 5}, {K::kVerify, 8, 8}, {K::kReset}, {K::kVerify, 8, 2}, {K::kPlain}, {K::kReset},
    {K::kPlain},        {K::kVerify, 5, 1}, {K::kPrefill, 40},
};
const std::vector<Ev> kMtp = {
    {K::kMtp, 3}, {K::kMtp, 3}, {K::kPlain}, {K::kMtp, 3}, {K::kPlain}, {K::kMtp, 3},
    {K::kMtp, 2}, {K::kPlain}, {K::kPrefill, 40}, {K::kMtp, 3}, {K::kMtp, 3},
};

void CheckScript(Engine& a, Engine& b, const std::vector<Ev>& script, const char* what) {
  Trace ta, tb, ta_each, tb_each;
  Run(a, script, /*digest_each=*/false, &ta, "end");
  Run(b, script, false, &tb, "end");
  const bool same = SameTrace(ta, tb, what);
  // a digest after every event: the flush of a pending prefix is invisible, and the states agree at every point
  Run(a, script, /*digest_each=*/true, &ta_each, "each");
  Run(b, script, true, &tb_each, "each");
  const bool same_each = SameTrace(ta_each, tb_each, what);
  // and neither engine's own observables moved because digests were taken (the flush is invisible)
  for (int side = 0; side < 2; ++side) {
    const Trace& plain = side == 0 ? ta : tb;
    const Trace& each = side == 0 ? ta_each : tb_each;
    bool ok = plain.obs.size() == each.obs.size();
    for (size_t i = 0; ok && i < plain.obs.size(); ++i) ok = plain.obs[i].second == each.obs[i].second;
    CHECK(ok, "[%s] %s: taking a state digest after every event changed an observable", what, side == 0 ? "window-slot" : "write-once");
  }
  if (same && same_each) std::printf("[ok] %s: %zu events, observables and state bitwise equal (digest at the end and after every event)\n", what, script.size());
}

// two VerifyWindow calls with no commit between, after a round that committed n > 1
void CheckBare(Engine& legacy, Engine& wo, const char* what) {
  for (Engine* e : {&legacy, &wo}) {
    e->Reset();
    (void)e->Prefill(kP);
    std::vector<float> l;
    (void)e->Verify(Tokens(8, 90), &l);
    e->Commit(3);
  }
  const std::vector<int32_t> w = Tokens(8, 91);
  std::vector<float> lw1, lw2, ll1, ll2;
  (void)wo.Verify(w, &lw1);
  (void)wo.Verify(w, &lw2);
  (void)legacy.Verify(w, &ll1);
  (void)legacy.Verify(w, &ll2);
  CHECK(SameBits(lw1, lw2), "[%s] the write-once Model's second bare verify differs from its first", what);
  CHECK(SameBits(lw1, ll1), "[%s] the first bare verify differs between the two modes", what);
  std::printf("[ok] %s: bare double verify -- write-once rows %s; window-slot path's second call %s its first (the latent bug)\n",
              what, SameBits(lw1, lw2) ? "equal" : "DIFFER", SameBits(ll1, ll2) ? "equals" : "differs from");
  // the commit after the second bare call, then a round: both modes agree again from the commit on? No: the
  // window-slot path has lost the committed state. The write-once Model must still equal a fresh replay of the
  // same history -- compare it with a clean write-once run of (round, commit 3, verify w, commit 2):
  wo.Commit(2);
  const std::vector<float> after = wo.DecodeStep(1234);
  wo.Reset();
  (void)wo.Prefill(kP);
  std::vector<float> l;
  (void)wo.Verify(Tokens(8, 90), &l);
  wo.Commit(3);
  (void)wo.Verify(w, &l);
  wo.Commit(2);
  const std::vector<float> clean = wo.DecodeStep(1234);
  CHECK(SameBits(after, clean), "[%s] a plain step after two bare verifies + commit differs from the clean history", what);
}

void CheckBounds(Engine& wo, const char* what) {
  wo.Reset();
  (void)wo.Prefill(kP);
  std::vector<float> l;
  (void)wo.Verify(Tokens(3, 92), &l);
  CHECK(wo.CommitRefused(5), "[%s] CommitVerifiedWindow(5) after a 3-row verify was accepted", what);
  wo.Commit(2);  // the refusal changed nothing: a valid commit still works
  const std::vector<float> lg = wo.DecodeStep(77);
  CHECK(!lg.empty(), "[%s] no logits after a refused then a valid commit", what);
}

void CheckLayout(Layout layout) {
  const std::string tag = LayoutName(layout);
  {
    // window 8 (the DFlash k = 7 sizing): the verify / plain / extend scripts
    SingleEngine a(Options(layout, 0, 7, /*gdn_wo=*/0, /*ckpt=*/true));
    SingleEngine b(Options(layout, 0, 7, /*gdn_wo=*/1, /*ckpt=*/true));
    CHECK(!a.WriteOnce() && b.WriteOnce(), "[%s] gdn_write_once 0 / 1 did not select the manager", tag.c_str());
    CHECK(b.GdnBytes() < a.GdnBytes(), "[%s] the write-once Model's GDN allocation (%lld B) is not smaller than the window-slot one (%lld B)",
          tag.c_str(), static_cast<long long>(b.GdnBytes()), static_cast<long long>(a.GdnBytes()));
    std::printf("[info] %s window 8: GDN state + logs %.2f MiB window-slot vs %.2f MiB write-once (%.2f MiB freed on 3 layers)\n",
                tag.c_str(), a.GdnBytes() / 1048576.0, b.GdnBytes() / 1048576.0, (a.GdnBytes() - b.GdnBytes()) / 1048576.0);
    CheckScript(a, b, kRounds, (tag + " rounds").c_str());
    CheckScript(a, b, kCheckpoint, (tag + " checkpoint").c_str());
    CheckScript(a, b, kReset, (tag + " reset").c_str());
    CheckBare(a, b, (tag + " bare").c_str());
    CheckBounds(b, (tag + " bounds").c_str());
  }
  {
    // real MTP rounds, window 4
    SingleEngine a(Options(layout, 3, 0, 0, false));
    SingleEngine b(Options(layout, 3, 0, 1, false));
    CheckScript(a, b, kMtp, (tag + " mtp k=3").c_str());
  }
  {
    // no speculative window: nothing to save, the request is moot and the Models are the same thing
    SingleEngine a(Options(layout, 0, 0, 0, false));
    SingleEngine b(Options(layout, 0, 0, 1, false));
    CHECK(!b.WriteOnce(), "[%s] a window-1 Model ran the write-once state", tag.c_str());
    CHECK(a.GdnBytes() == b.GdnBytes(), "[%s] window 1: the allocations differ", tag.c_str());
    const std::vector<Ev> plain = {{K::kPlain}, {K::kPlain}, {K::kPrefill, 40}, {K::kPlain}};
    CheckScript(a, b, plain, (tag + " window 1").c_str());
  }
}

void CheckTp(Layout layout) {
  const std::string tag = std::string("tp2 ") + LayoutName(layout);
  {
    // two TpModels (four rank Models) at a time, never four TpModels
    TpEngine a(Options(layout, 0, 7, 0, true));
    TpEngine b(Options(layout, 0, 7, 1, true));
    CHECK(!a.WriteOnce() && b.WriteOnce(), "[%s] gdn_write_once 0 / 1 did not select the manager on both ranks", tag.c_str());
    CHECK(b.GdnBytes() < a.GdnBytes(), "[%s] the write-once rank's GDN allocation is not smaller", tag.c_str());
    std::printf("[info] %s: per-rank GDN state + logs %.2f MiB window-slot vs %.2f MiB write-once\n", tag.c_str(),
                a.GdnBytes() / 1048576.0, b.GdnBytes() / 1048576.0);
    CheckScript(a, b, kRounds, (tag + " rounds").c_str());
    CheckScript(a, b, kCheckpoint, (tag + " checkpoint").c_str());
    CheckBounds(b, (tag + " bounds").c_str());
  }
  {
    TpEngine ma(Options(layout, 3, 0, 0, false));
    TpEngine mb(Options(layout, 3, 0, 1, false));
    CheckScript(ma, mb, kMtp, (tag + " mtp k=3").c_str());
  }
}

int RunTest() {
  if (!r4dx_test::FileExists(kContainerPath)) return r4dx_test::SkipMissing(kContainerPath);
  for (Layout layout : {Layout::kBf16, Layout::kW4a16}) CheckLayout(layout);
  CheckTp(Layout::kW4a16);
  if (g_failures != 0) {
    std::fprintf(stderr, "test_gdn_write_once_model: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "[PASS] test_gdn_write_once_model\n");
  return 0;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_gdn_write_once_model", RunTest); }
