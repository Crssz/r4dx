// tests/model/test_tp_hybrid_identity.cpp -- the TWO-GPU gate of TpModel's hybrid mode (`--tp 2 --pp 2`; docs/pp-tp2-hybrid.md 4, 15): the real
// production path -- two rank threads on two cards, a stage Model next to each rank, the pinned host ring between the cards -- against a TP=1
// reference Model on the headless card. test_hybrid_emulate_identity proves the pieces on one device; this proves TpModel's orchestration of them.
//
//   G-H1  After every pipelined call: TpModel::Prefill's logits equal the TP=1 reference's byte for byte (stage Y's epilogue), and each rank's
//         Model::DebugLiveStateDigest equals the digest of hybrid::ReshardRef(reference state, rank). After a call that follows TP-only work (a TP=2
//         decode, a TP-path prefill, a checkpoint restore) the reference is first loaded with the ranks' GATHERED state, which is the TP=1 oracle of
//         "this prefill from THIS state" (the warm gather, the MTP seed sync and the primed-block rule are all inside it).
//   G-H2  16 greedy tokens (plain, then --mtp 3 rounds, rewound by the checkpoint) decoded at TP=2 by the hybrid-prefilled ranks equal those decoded by
//         the same ranks loaded with the reference image (DebugImportFullState), and the ranks' states afterwards agree.
//   Dispatch   a call below --pp-min-rows and a call over the stage-KV capacity run the TP=2 prefill (counted in the stats, reasons named) and leave a
//         state the next pipelined call can continue from (G-H1 on that call).
//   Recovery   an injected failure on either card, in each phase (the gather, the stage prefill, the reshard, the adoption): the other card is freed at
//         once, the injected error is the one reported, the group is kNeedsRecovery, Reset() heals it, and the next pipelined call is identical to the reference.
//
// Configs: l4/bf16 (the 4-layer selftest container, k = 2) and real16 (the production trellis container with layer_limit 16, k = 8: real shapes, real
// shard rules; ~25 GiB over the two cards with the reference). R4DX_TEST_ONLY=<substring> selects configurations / scenarios ("l4/", "real16/warm"...).
// Both cards must be visible (CMake sets HIP_VISIBLE_DEVICES=0,1): rank 0 = device 1 (the reference lives there too), rank 1 = device 0 (the desktop card).
// Output: "[PASS] ..." / "FAIL ..." lines and a final "test_tp_hybrid_identity: PASS (N configurations)"; exit 0 only when every check passed, 1 on a failure,
// 77 when a container or a device is missing. Written, NOT run by its author (CPU-only session). Links the R4DX_TP_TESTING variant.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "live_digest.h"
#include "live_state.h"
#include "model.h"
#include "r4dx/core/tp_comm.hpp"
#include "reshard_plan.h"
#include "test_common.h"
#include "tp_model.h"

#ifndef R4DX_TP_TESTING
#error "test_tp_hybrid_identity needs the R4DX_TP_TESTING hooks: link r4dx_model_tptest (src/model/CMakeLists.txt)"
#endif

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::PpOptions;
using r4dx::model::StageSyncState;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace hybrid = r4dx::model::hybrid;

namespace {

constexpr int kDecodeTokens = 16;
constexpr int kMtpK = 3;
constexpr int64_t kStageCtx = 16384;  // --hybrid-ctx: the smallest capacity the mode accepts
constexpr int64_t kMaxCtx = 20000;    // the ranks and the reference hold more than the stages (the "over S" case needs the room)

int g_fails = 0;
int g_configs = 0;

bool Only(const std::string& name) {
  const char* only = std::getenv("R4DX_TEST_ONLY");
  if (only == nullptr || *only == '\0') return true;
  return name.find(only) != std::string::npos;
}

class Group {
 public:
  explicit Group(std::string name) : name_(std::move(name)) {}
  void Ok(bool cond, const std::string& what) {
    ++checks_;
    if (cond) return;
    ++fails_;
    ++g_fails;
    std::fprintf(stderr, "FAIL %s: %s\n", name_.c_str(), what.c_str());
  }
  void Done(const std::string& note = "") {
    if (fails_ == 0) std::fprintf(stderr, "[PASS] %s: %d checks%s%s\n", name_.c_str(), checks_, note.empty() ? "" : " -- ", note.c_str());
  }
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  int checks_ = 0, fails_ = 0;
};

uint64_t HashLogits(const std::vector<float>& x) { return hybrid::FnvUpdate(hybrid::kFnvInit, x.data(), x.size() * sizeof(float)); }
int32_t Argmax(const std::vector<float>& v) {
  size_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) best = i;
  }
  return static_cast<int32_t>(best);
}
std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 100 + (i * 41 + salt * 977) % 5000;
  return v;
}

struct Cfg {
  std::string name, path;
  Layout layout = Layout::kBf16;
  int64_t layers = -1;
  int64_t split = 2;
};

struct DecodeOutcome {
  std::vector<int32_t> tok[2];
  hybrid::DigestList dig[2][2];  // [mode][rank]
};

class Rig {
 public:
  Cfg cfg;
  std::unique_ptr<TpModel> tpm;
  std::unique_ptr<Model> r;  // the TP=1 reference, on rank 0's card
  hybrid::StateGeometry g;
  hybrid::DigestShape rank_shape[2];
  std::string state_file;
  int ref_device = 0;

  explicit Rig(Cfg c) : cfg(std::move(c)) {
    state_file = (std::filesystem::temp_directory_path() / "r4dx_tp_hybrid_identity.state").string();
    ModelOptions o;
    o.container_path = cfg.path;
    o.layout = cfg.layout;
    o.max_ctx = kMaxCtx;
    o.layer_limit = cfg.layers;
    o.mtp_draft_k = kMtpK;
    o.vision = ModelOptions::VisionMode::kOff;
    o.prompt_checkpoint = true;
    o.pp = 2;  // --pp 2
    TpOptions tp;
    tp.world = 2;
    tp.mode = TpOptions::Mode::kReal;  // --tp 2, two cards, devices auto: rank 0 = ordinal 1, rank 1 = ordinal 0
    PpOptions pp;
    pp.split = static_cast<int>(cfg.split);
    pp.min_rows = 1;  // pipeline every call (the dispatch cases move it)
    pp.min_rows_given = true;
    pp.hybrid = 1;
    pp.hybrid_ctx = kStageCtx;
    std::fprintf(stderr, "[tp-hybrid] %s: loading the two-GPU TpModel with the hybrid\n", cfg.name.c_str());
    tpm = TpModel::Load(o, tp, pp);
    if (!tpm->HybridEngaged()) throw std::runtime_error("the hybrid did not engage: " + tpm->HybridRefusal());
    ref_device = tpm->Vram().at(0).device;  // rank 0's card
    if (hipSetDevice(ref_device) != hipSuccess) throw std::runtime_error("hipSetDevice failed for the reference");
    std::fprintf(stderr, "[tp-hybrid] %s: loading the TP=1 reference on HIP device %d\n", cfg.name.c_str(), ref_device);
    ModelOptions ro = o;
    ro.pp = 0;
    ro.prompt_checkpoint = false;
    r = std::make_unique<Model>(Model::Load(ro));
    r4dx::model::ModelConfig gc = r->GlobalConfig();
    gc.num_hidden_layers = r->GetContainer().NumLoadedLayers();
    gc.layer_types.resize(static_cast<size_t>(gc.num_hidden_layers));
    g = hybrid::StateGeometry::FromRules(gc, r->PpKvBlockSize());
    for (int k = 0; k < 2; ++k) rank_shape[k] = {g.block_tokens, g.ranks[static_cast<size_t>(k)].kv_heads, g.KvTokenHeadBytes(), g.conv_live};
    std::fprintf(stderr, "[tp-hybrid] %s: ready (split %lld, S %lld, min rows %lld)\n", cfg.name.c_str(), static_cast<long long>(tpm->HybridSplit()),
                 static_cast<long long>(tpm->HybridStageCtx()), static_cast<long long>(tpm->HybridMinRows()));
  }

  // `read_only`: the closure only reads the shards (the hybrid tracker keeps its idea of what the stages hold)
  void OnRanks(const std::function<void(Model&, int)>& fn, bool read_only = false) { tpm->RunCollectiveForTest(fn, read_only); }
  std::vector<hybrid::DigestList> RankDigests() {
    std::vector<hybrid::DigestList> out(2);
    OnRanks([&](Model& m, int rank) { out[static_cast<size_t>(rank)] = m.DebugLiveStateDigest(); }, /*read_only=*/true);
    return out;
  }
  std::vector<hybrid::LiveState> SnapshotRanks() {
    std::vector<hybrid::LiveState> rs(2);
    OnRanks(
        [&](Model& m, int rank) {
          m.ReshardCollapse();
          m.ReshardSync();
          rs[static_cast<size_t>(rank)] = m.DebugExportLiveState();
        },
        /*read_only=*/true);
    return rs;
  }

  // ---- one TpModel::Prefill and its reference -------------------------------------------------------------------------------------------
  struct CallOut {
    bool pipelined = false;
    std::vector<float> logits;
    std::vector<float> ref_logits;
  };
  // `fresh`: the first call of a conversation (the reference starts from nothing); `moved`: the ranks did TP-only work since the last call (the
  // reference is loaded with their gathered state). `expect`: -1 any route, 1 must be pipelined, 0 must be the TP prefill.
  CallOut DoCall(const std::vector<int32_t>& ids, bool fresh, bool moved, int expect, Group& grp, const std::string& tag) {
    CallOut out;
    const int64_t p0 = tpm->PositionCount();
    const int64_t n = static_cast<int64_t>(ids.size());
    std::vector<hybrid::LiveState> rs;
    if (!fresh && moved) rs = SnapshotRanks();
    const TpModel::HybridStats before = tpm->GetHybridStats();
    out.logits = tpm->Prefill(ids);
    const TpModel::HybridStats after = tpm->GetHybridStats();
    out.pipelined = after.pipelined_calls == before.pipelined_calls + 1;
    grp.Ok(out.pipelined || after.tp_prefill_calls == before.tp_prefill_calls + 1, tag + ": the call is counted as pipelined or as a TP prefill");
    if (expect >= 0) grp.Ok(out.pipelined == (expect == 1), tag + ": the dispatch rule chose the " + (expect == 1 ? "pipeline" : "TP prefill"));
    grp.Ok(tpm->PositionCount() == p0 + n, tag + ": the position is " + std::to_string(p0 + n));
    if (!out.pipelined) return out;

    // the reference: the same prefill on the TP=1 Model
    if (fresh) {
      r->Reset();
    } else if (!rs.empty()) {
      hybrid::LiveState full;
      full.image = hybrid::GatherRef(g, rs[0].image, rs[1].image);
      full.scalars = rs[0].scalars;
      r->DebugImportLiveState(full);
    }
    out.ref_logits = r->Prefill(ids);
    const hybrid::LiveState ref = r->DebugExportLiveState();
    grp.Ok(HashLogits(out.logits) == HashLogits(out.ref_logits) && out.logits.size() == out.ref_logits.size(),
           tag + ": the pipelined prefill's logits equal the TP=1 reference's byte for byte");
    const std::vector<hybrid::DigestList> got = RankDigests();
    for (int k = 0; k < 2; ++k) {
      const hybrid::DigestList want = hybrid::DigestLiveImage(hybrid::ReshardRef(g, ref.image, k), rank_shape[k], ref.scalars.pos);
      const std::string why = hybrid::DiffDigests(want, got[static_cast<size_t>(k)]);
      grp.Ok(why.empty(), tag + ": G-H1 rank " + std::to_string(k) + "'s live-state digest equals ReshardRef(reference state): " + why);
    }
    return out;
  }

  // ---- G-H2 -------------------------------------------------------------------------------------------------------------------------------
  DecodeOutcome DecodeModes(int32_t first) {
    tpm->SaveCheckpoint();
    DecodeOutcome o;
    for (int mode = 0; mode < 2; ++mode) {
      if (mode > 0) tpm->RestoreCheckpoint();
      int32_t tok = first;
      while (static_cast<int>(o.tok[mode].size()) < kDecodeTokens) {
        if (mode == 0) {
          tok = tpm->DecodeStepGreedy(tok);
          o.tok[0].push_back(tok);
        } else {
          const std::vector<int32_t> round = tpm->DecodeStepMtpGreedy(tok, kMtpK);
          o.tok[1].insert(o.tok[1].end(), round.begin(), round.end());
          tok = round.back();
        }
      }
      const std::vector<hybrid::DigestList> d = RankDigests();
      o.dig[mode][0] = d[0];
      o.dig[mode][1] = d[1];
    }
    return o;
  }
  void CheckDecode(int32_t first, Group& grp) {
    r->DebugExportFullState(state_file);  // the reference image of the last call
    const DecodeOutcome h = DecodeModes(first);
    tpm->Reset();
    OnRanks([&](Model& m, int) { m.DebugImportFullState(state_file); });
    const DecodeOutcome ref = DecodeModes(first);
    for (int mode = 0; mode < 2; ++mode) {
      const char* name = mode == 0 ? "plain" : "--mtp 3";
      grp.Ok(h.tok[mode].size() >= static_cast<size_t>(kDecodeTokens), std::string("G-H2 ") + name + ": decoded " + std::to_string(h.tok[mode].size()) + " tokens");
      grp.Ok(h.tok[mode] == ref.tok[mode], std::string("G-H2 ") + name + ": the tokens decoded at TP=2 after the hybrid prefill equal those decoded from the reference image");
      grp.Ok(hybrid::DiffDigests(h.dig[mode][0], ref.dig[mode][0]).empty() && hybrid::DiffDigests(h.dig[mode][1], ref.dig[mode][1]).empty(),
             std::string("G-H2 ") + name + ": both ranks' live state after the decode equals the reference-image run's");
    }
  }

  // ---- scenarios ------------------------------------------------------------------------------------------------------------------------------
  enum class K { kCall, kDecode, kSave, kRestore };
  struct Step {
    K kind = K::kCall;
    int n = 0, salt = 1;
    bool mtp = false;
  };
  static Step Call(int n, int salt = 1) { return {K::kCall, n, salt, false}; }
  static Step Decode(int n, bool mtp) { return {K::kDecode, n, 0, mtp}; }
  static Step Save() { return {K::kSave, 0, 0, false}; }
  static Step Restore() { return {K::kRestore, 0, 0, false}; }

  void RunScenario(const std::string& name, const std::vector<Step>& steps, bool check_decode) {
    const std::string full = cfg.name + "/" + name;
    if (!Only(full)) return;
    Group grp(full);
    try {
      tpm->Reset();
      bool fresh = true, moved = false;
      int32_t next = 0;
      std::vector<hybrid::DigestList> saved;
      int64_t saved_pos = 0;
      int32_t saved_token = 0;
      int calls = 0;
      for (const Step& st : steps) {
        switch (st.kind) {
          case K::kCall: {
            const CallOut c = DoCall(Tokens(st.n, st.salt), fresh, moved, 1, grp, full + " call " + std::to_string(calls++));
            next = Argmax(c.ref_logits.empty() ? c.logits : c.ref_logits);
            fresh = moved = false;
            break;
          }
          case K::kDecode: {
            int32_t tok = next;
            int got = 0;
            while (got < st.n) {
              if (!st.mtp) {
                tok = tpm->DecodeStepGreedy(tok);
                ++got;
              } else {
                const std::vector<int32_t> round = tpm->DecodeStepMtpGreedy(tok, kMtpK);
                got += static_cast<int>(round.size());
                tok = round.back();
              }
            }
            next = tok;
            moved = true;
            break;
          }
          case K::kSave:
            tpm->SaveCheckpoint();
            saved = RankDigests();
            saved_pos = tpm->PositionCount();
            saved_token = next;
            break;
          case K::kRestore: {
            tpm->RestoreCheckpoint();
            const std::vector<hybrid::DigestList> d = RankDigests();
            grp.Ok(tpm->PositionCount() == saved_pos, "RestoreCheckpoint returned the ranks to the saved position");
            for (int k = 0; k < 2; ++k) {
              grp.Ok(hybrid::DiffDigests(saved[static_cast<size_t>(k)], d[static_cast<size_t>(k)]).empty(),
                     "rank " + std::to_string(k) + "'s digest after RestoreCheckpoint equals the one at SaveCheckpoint");
            }
            next = saved_token;
            moved = true;
            break;
          }
        }
      }
      if (check_decode) CheckDecode(next, grp);
      grp.Done(std::to_string(calls) + " call(s)" + (check_decode ? ", G-H2 plain + --mtp 3" : ""));
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
    }
  }

  // ---- dispatch ------------------------------------------------------------------------------------------------------------------------------
  void RunDispatch() {
    const std::string name = cfg.name + "/dispatch";
    if (!Only(name)) return;
    Group grp(name);
    try {
      tpm->Reset();
      tpm->SetHybridMinRows(1000);
      // below the threshold: the TP prefill, counted
      TpModel::HybridStats s0 = tpm->GetHybridStats();
      CallOut a = DoCall(Tokens(300, 1), /*fresh=*/true, false, /*expect=*/0, grp, name + " below min rows");
      TpModel::HybridStats s1 = tpm->GetHybridStats();
      grp.Ok(s1.declined[2] == s0.declined[2] + 1, "the decline is counted as 'below min rows'");
      // ... and the next, long enough call continues from the TP-prefilled state through the warm gather (everything stale: TpOnly(0))
      CallOut b = DoCall(Tokens(1100, 2), /*fresh=*/false, /*moved=*/true, /*expect=*/1, grp, name + " pipelined after a TP prefill");
      tpm->SetHybridMinRows(1);
      // over the stage-KV capacity: the TP prefill again (the ranks hold 20000 tokens, the stages 16384)
      tpm->Reset();
      s0 = tpm->GetHybridStats();
      CallOut c = DoCall(Tokens(static_cast<int>(kStageCtx) + 600, 3), /*fresh=*/true, false, /*expect=*/0, grp, name + " over S");
      s1 = tpm->GetHybridStats();
      grp.Ok(s1.declined[3] == s0.declined[3] + 1, "the decline is counted as 'over the stage-KV capacity'");
      const int32_t tok = tpm->DecodeStepGreedy(Argmax(c.logits));  // the TP state is usable
      grp.Ok(tok >= 0, "decode continues after the TP prefill");
      (void)a;
      (void)b;
      // the stats line
      const std::string line = tpm->HybridStatsLine();
      std::fprintf(stderr, "[stats] %s\n", line.c_str());
      grp.Ok(line.rfind("hybrid:", 0) == 0, "the hybrid stats line is there");
      grp.Done("min-rows and over-S fallbacks counted, the pipeline continues from a TP-prefilled state");
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
    }
    tpm->SetHybridMinRows(1);
  }

  // ---- recovery ------------------------------------------------------------------------------------------------------------------------------
  void RunFault(int rank, TpModel::HybridFaultPhase phase, const char* phase_name) {
    const std::string name = cfg.name + "/fault-rank" + std::to_string(rank) + "-" + phase_name;
    if (!Only(name)) return;
    Group grp(name);
    try {
      tpm->Reset();
      // a conversation with TP-only work in it, so the faulted call is a warm one that has to gather
      (void)tpm->Prefill(Tokens(300, 1));
      int32_t tok = 1000;
      for (int i = 0; i < 4; ++i) tok = tpm->DecodeStepGreedy(tok);
      tpm->ArmHybridFault(rank, phase);
      bool threw = false;
      std::string what;
      try {
        (void)tpm->Prefill(Tokens(400, 2));
      } catch (const std::exception& e) {
        threw = true;
        what = e.what();
      }
      grp.Ok(threw, "the faulted call throws");
      grp.Ok(what.find("injected hybrid fault") != std::string::npos, "the injected error is the one reported (the other card's poison is not): '" + what + "'");
      grp.Ok(tpm->GetState() == TpModel::State::kNeedsRecovery, "the group is kNeedsRecovery");
      bool state_error = false;
      try {
        (void)tpm->Prefill(Tokens(300, 3));
      } catch (const r4dx::core::TpStateError&) {
        state_error = true;
      } catch (...) {
      }
      grp.Ok(state_error, "a device-work call is refused with TpStateError until Reset()");
      tpm->Reset();
      grp.Ok(tpm->GetState() == TpModel::State::kReady && tpm->PositionCount() == 0, "Reset() heals the group (both ranks, both stages, the channel, the tracker)");
      (void)DoCall(Tokens(300, 4), /*fresh=*/true, false, /*expect=*/1, grp, name + " after recovery");
      grp.Done("freed at once, root cause reported, healed");
    } catch (const std::exception& e) {
      grp.Ok(false, std::string("uncaught exception: ") + e.what());
    }
  }
};

Cfg L4Cfg() {
  Cfg c;
  c.name = "l4/bf16";
  c.path = ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
  c.layout = Layout::kBf16;
  c.layers = 4;
  c.split = 2;
  return c;
}
Cfg Real16Cfg() {
  Cfg c;
  c.name = "real16";
  c.path = ProductionTargetPath();
  c.layout = r4dx::model::LayoutFromName(ProductionLayoutName());
  c.layers = 16;
  c.split = 8;
  return c;
}

void RunConfig(const Cfg& cfg) {
  if (!Only(cfg.name + "/")) return;
  if (!FileExists(cfg.path)) {
    std::fprintf(stderr, "[SKIP] %s: %s not found\n", cfg.name.c_str(), cfg.path.c_str());
    return;
  }
  ++g_configs;
  std::unique_ptr<Rig> rig;
  try {
    rig = std::make_unique<Rig>(cfg);
  } catch (const std::exception& e) {
    ++g_fails;
    std::fprintf(stderr, "FAIL %s: the rig did not load: %s\n", cfg.name.c_str(), e.what());
    return;
  }
  // G-H1 / G-H2 cold
  rig->RunScenario("cold17", {Rig::Call(17)}, true);
  rig->RunScenario("cold300", {Rig::Call(300)}, true);
  rig->RunScenario("cold1000", {Rig::Call(1000)}, false);
  rig->RunScenario("cold2125", {Rig::Call(2048 + 77)}, true);
  // prefix-reuse shapes: consecutive pipelined calls (the MTP-block-only gather)
  rig->RunScenario("prefix-reuse", {Rig::Call(300, 1), Rig::Call(333, 2), Rig::Call(257, 3), Rig::Call(1, 4)}, true);
  // warm turns: TP=2 decode between two pipelined calls (the full gather, the MTP seed sync, the primed-block rule)
  rig->RunScenario("warm-plain", {Rig::Call(300, 1), Rig::Decode(20, false), Rig::Call(517, 2)}, true);
  rig->RunScenario("warm-mtp3", {Rig::Call(300, 1), Rig::Decode(20, true), Rig::Call(517, 2)}, true);
  // checkpoint save / restore around a decode, then a warm pipelined turn from the restored state
  rig->RunScenario("checkpoint", {Rig::Call(300, 1), Rig::Save(), Rig::Decode(12, false), Rig::Restore(), Rig::Call(400, 2)}, true);
  rig->RunDispatch();
  using P = TpModel::HybridFaultPhase;
  rig->RunFault(1, P::kStagePrefill, "stage");
  rig->RunFault(0, P::kStagePrefill, "stage");
  rig->RunFault(1, P::kReshard, "reshard");
  rig->RunFault(0, P::kReshard, "reshard");
  rig->RunFault(0, P::kGather, "gather");
  rig->RunFault(1, P::kAdopt, "adopt");
}

}  // namespace

static int RunTest() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 2) {
    std::fprintf(stderr, "[SKIP] the two-GPU hybrid needs two visible HIP devices (%d visible)\n", devices);
    return kSkipReturnCode;
  }
  RunConfig(L4Cfg());
  RunConfig(Real16Cfg());
  if (g_configs == 0) return SkipMissing(L4Cfg().path);
  if (g_fails != 0) {
    std::fprintf(stderr, "test_tp_hybrid_identity: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_tp_hybrid_identity: PASS (%d configurations)\n", g_configs);
  return 0;
}

int main() { return RunGuardedMain("test_tp_hybrid_identity", [] { return RunTest(); }); }
