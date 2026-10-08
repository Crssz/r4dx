// tests/model/test_reshard_exec.cpp -- the device half of the hybrid mode's P2 part 1 (docs/pp-tp2-hybrid.md 3, 4, 9): the reshard EXECUTOR
// (Model::ReshardExport / ReshardImport / ReshardCopyLocal, driven by the copy ops of hybrid::ScatterPlan / GatherPlan) and the
// live-state test hooks (DebugLiveStateDigest, DebugExportLiveState / DebugImportLiveState / DebugExportFullState /
// DebugImportFullState), on the 4-layer MTP container, ONE device (device 1): a TP=1 Model R plays both pipeline stages (it holds every
// layer and the full MTP head; split k = 2 gives the plan its stage A / stage B layer ownership), and a TpModel in emulate mode
// supplies the two TP rank Models, reached through RunCollectiveForTest on their own threads.
//   1. scatter: R prefills 300 tokens; the plan's ops move its live state into the rank Models (a) every op through the host ring
//      (D2H on R, H2D on the rank), (b) with the same-card ops as device-to-device copies; TpAdoptPrefill carries the scalars.
//      Gate (G-H1): each rank's DebugLiveStateDigest == the digest of ReshardRef(R's exported image, rank) -- the same records, hashed
//      by the same code from a host image -- for (a) and (b); and the rank loaded through DebugImportFullState (a file R dumped) has
//      the same digest.
//   2. G-H2 lite: the ranks decode 4 tokens from the scattered state and from the imported state: identical tokens, identical digests.
//   3. warm gather: R (stale at 300) is brought to the ranks' state at 304 by GatherPlan (tracker plan: KV rows [300, 304), all GDN
//      state, the MTP block of row 303) + the scalars and the seed of rank 0: its digest == the digest of GatherRef(the two ranks' exports).
//   4. NEGATIVE CONTROLS (each must change a rank digest): a skipped KV op, skipped MTP ops, a dropped conv segment, swapped head halves,
//      a KV source one block off; a gather without its MTP block / KV rows. GUARDS: a plan for another conv pitch, the wrong rank, the
//      wrong side, a TP rank asked for the full side, a missing conv scratch each throw.
//   5. speculative state: after VerifyWindow + CommitVerifiedWindow(3) the live GDN state sits in a window slot / behind a pending prefix
//      (window-slot AND write-once): DebugLiveStateDigest follows it there, a recurrent / conv op is refused until ReshardCollapse, and the
//      collapse leaves the digest unchanged.
//   6. the executor allocates nothing inside the rank closures (TpCollectiveScope allocation counter).
// Device 1 (HIP_VISIBLE_DEVICES=1, from CMake), SKIP 77 without the 4-layer container. Links the R4DX_TP_TESTING library variant.
// Written, NOT run by its author (CPU-only session).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "test_common.h"
#include "tp_model.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::StageSyncState;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace hybrid = r4dx::model::hybrid;

namespace {

const char* kL4Container = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
constexpr int64_t kSplit = 2;
constexpr int64_t kPrompt = 300;  // block 18 holds rows 288..303: the warm gather's one MTP block covers rows 299..302
constexpr int64_t kDecode = 4;
constexpr uint64_t kSliceBudget = 64 << 10;  // pieces of the host hop: several slices per KV op, one per recurrent op

int g_fails = 0;
void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fails;
  }
}
template <class E = std::exception, class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = 100 + (i * 41 + salt * 977) % 5000;
  return v;
}

ModelOptions Base(int gdn_write_once = -1) {
  ModelOptions o;
  o.container_path = kL4Container;
  o.layout = Layout::kBf16;
  o.max_ctx = 2048;
  o.layer_limit = 4;
  o.mtp_draft_k = 3;
  o.vision = ModelOptions::VisionMode::kOff;
  o.gdn_write_once = gdn_write_once;
  return o;
}

std::string Why(const hybrid::DigestList& a, const hybrid::DigestList& b) { return hybrid::DiffDigests(a, b); }

// ---- the fixture: R (both stages), the ranks, the geometry and plan parameters -------------------------------------------------------
struct Fix {
  std::unique_ptr<Model> r;
  std::unique_ptr<TpModel> tpm;
  hybrid::StateGeometry g;
  hybrid::PlanParams p;
  hybrid::DigestShape full_shape, rank_shape[2];
  r4dx::core::PinnedBuffer<uint8_t> host;

  // fn(model, rank) on both rank threads as one command
  void OnRanks(const std::function<void(Model&, int)>& fn) { tpm->RunCollectiveForTest(fn); }
  void ZeroRanks() {
    tpm->Reset();
    OnRanks([](Model& m, int) { m.DebugZeroKvState(); });
  }
  std::vector<hybrid::DigestList> RankDigests() {
    std::vector<hybrid::DigestList> out(2);
    OnRanks([&](Model& m, int rank) { out[static_cast<size_t>(rank)] = m.DebugLiveStateDigest(); });
    return out;
  }
};

// One op of a plan, executed. host == false: the same-card ops as D2D (when the plan says local); true: everything through the host hop.
void ScatterOp(Fix& f, const hybrid::CopyOp& op, bool force_host) {
  if (op.local && !force_host) {
    f.OnRanks([&](Model& m, int rank) {
      if (rank != op.rank) return;
      for (const hybrid::OpSlice& s : hybrid::WholeOp(op)) Model::ReshardCopyLocal(*f.r, m, op, s, hybrid::Dir::kStageToRank);
      m.ReshardSync();
    });
    return;
  }
  for (const hybrid::OpSlice& s : hybrid::SplitOp(op, kSliceBudget)) {
    f.r->ReshardExport(op, s, hybrid::Side::kFull, f.host.data());
    f.r->ReshardSync();
    f.OnRanks([&](Model& m, int rank) {
      if (rank != op.rank) return;
      m.ReshardImport(op, s, hybrid::Side::kRank, f.host.data());
      m.ReshardSync();
    });
  }
}
void GatherOp(Fix& f, const hybrid::CopyOp& op, bool force_host) {
  if (op.local && !force_host) {
    f.OnRanks([&](Model& m, int rank) {
      if (rank != op.rank) return;
      for (const hybrid::OpSlice& s : hybrid::WholeOp(op)) Model::ReshardCopyLocal(*f.r, m, op, s, hybrid::Dir::kRankToStage);
      f.r->ReshardSync();
    });
    return;
  }
  for (const hybrid::OpSlice& s : hybrid::SplitOp(op, kSliceBudget)) {
    f.OnRanks([&](Model& m, int rank) {
      if (rank != op.rank) return;
      m.ReshardExport(op, s, hybrid::Side::kRank, f.host.data());
      m.ReshardSync();
    });
    f.r->ReshardImport(op, s, hybrid::Side::kFull, f.host.data());
    f.r->ReshardSync();
  }
}

enum class Fault { kNone, kSkipKv, kSkipMtp, kDropConvSegment, kSwapHeads, kKvOneBlockOff };
const char* FaultName(Fault x) {
  switch (x) {
    case Fault::kSkipKv: return "a skipped KV op";
    case Fault::kSkipMtp: return "skipped MTP ops";
    case Fault::kDropConvSegment: return "a dropped conv segment";
    case Fault::kSwapHeads: return "swapped head halves";
    case Fault::kKvOneBlockOff: return "a KV source one block off";
    default: return "no fault";
  }
}

// The scatter of the whole plan with at most one fault injected; `fault_rank` is the rank whose op is damaged.
void Scatter(Fix& f, const hybrid::ReshardPlan& plan, bool force_host, Fault fault = Fault::kNone, int fault_rank = 0) {
  bool done = false;
  for (const hybrid::CopyOp& op : plan.ops) {
    hybrid::CopyOp use = op;
    const bool mine = op.rank == fault_rank;
    if (fault == Fault::kSkipKv && op.kind == hybrid::StateKind::kKv && mine && !done) {
      done = true;
      continue;
    }
    if (fault == Fault::kSkipMtp && op.kind == hybrid::StateKind::kMtpKv && mine) continue;
    if (fault == Fault::kDropConvSegment && op.kind == hybrid::StateKind::kGdnConv && mine && !done) {
      done = true;
      use.runs.pop_back();  // the v segment: the biggest of the three
    }
    if (fault == Fault::kSwapHeads && op.kind == hybrid::StateKind::kKv && mine && !done) {
      done = true;
      for (const hybrid::CopyOp& other : plan.ops) {
        if (other.kind == op.kind && other.layer == op.layer && other.stage == op.stage && other.rank != op.rank) {
          use.runs[0].full_off = other.runs[0].full_off;  // read the OTHER rank's head half
        }
      }
    }
    if (fault == Fault::kKvOneBlockOff && op.kind == hybrid::StateKind::kKv && mine && !done) {
      done = true;
      use.runs[0].full_off += use.runs[0].full_pitch;
      use.runs[0].height -= 1;  // blocks 1.. of the source land on blocks 0..
    }
    ScatterOp(f, use, force_host);
  }
}

uint64_t MaxSliceBytes(const hybrid::ReshardPlan& plan) {
  uint64_t m = 0;
  for (const hybrid::CopyOp& op : plan.ops) {
    for (const hybrid::OpSlice& s : hybrid::SplitOp(op, kSliceBudget)) m = std::max(m, hybrid::SliceBytes(op, s));
  }
  return m;
}

}  // namespace

static int RunTest() {
  if (!FileExists(kL4Container)) return SkipMissing(kL4Container);
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::fprintf(stderr, "[SKIP] no HIP device\n");
    return kSkipReturnCode;
  }
  const std::filesystem::path state_file = std::filesystem::temp_directory_path() / "r4dx_test_reshard_exec_full.state";

  Fix f;
  const ModelOptions opt = Base();
  f.r = std::make_unique<Model>(Model::Load(opt));
  TpOptions emu;
  emu.world = 2;
  emu.mode = TpOptions::Mode::kEmulate;
  f.tpm = TpModel::Load(opt, emu);
  f.r->ReshardInit();
  int64_t rank_pitch[2] = {0, 0};
  f.OnRanks([&](Model& m, int rank) {
    m.ReshardInit();
    rank_pitch[rank] = m.ReshardConvPitch();
  });
  Check(rank_pitch[0] == rank_pitch[1] && rank_pitch[0] == 2 + 4 && f.r->ReshardConvPitch() == 2 + 4,
        "ReshardConvPitch: conv_width - 2 + window = 6 entries per channel on R and on both ranks (mtp_draft_k 3)");

  // The 4-layer file's config still describes 64 layers (only 0..3 are on disk and loaded): the plan must cover the loaded ones only.
  r4dx::model::ModelConfig gc = f.r->GlobalConfig();
  gc.num_hidden_layers = f.r->GetContainer().NumLoadedLayers();
  gc.layer_types.resize(static_cast<size_t>(gc.num_hidden_layers));
  Check(gc.num_hidden_layers == 4 && !gc.IsGdnLayer(3) && gc.IsGdnLayer(2), "the loaded layers are 0..3, attention at 3");
  f.g = hybrid::StateGeometry::FromRules(gc, f.r->PpKvBlockSize());
  f.p = hybrid::PlanParams::ForSplit(f.g, kSplit, rank_pitch[0]);
  f.p.stage_conv_pitch = f.r->ReshardConvPitch();  // R's own lines (R is both stages; stage B's differs from A's only with a real stage A)
  f.full_shape = {f.g.block_tokens, f.g.kv_heads_full, f.g.KvTokenHeadBytes(), f.g.conv_live};
  for (int r = 0; r < 2; ++r) f.rank_shape[r] = {f.g.block_tokens, f.g.ranks[static_cast<size_t>(r)].kv_heads, f.g.KvTokenHeadBytes(), f.g.conv_live};

  // ---- 1. scatter -------------------------------------------------------------------------------------------------------------
  (void)f.r->Prefill(Tokens(static_cast<int>(kPrompt), 1));
  f.r->DebugExportFullState(state_file.string());
  const hybrid::LiveState ref = f.r->DebugExportLiveState();
  const int64_t pos = ref.scalars.pos;
  Check(pos == kPrompt && ref.scalars.started && ref.scalars.mtp_seed_valid && !ref.image.mtp_kv.empty(), "R: pos 300, started, a valid MTP seed, MTP rows primed");
  {
    const hybrid::LiveState from_file = hybrid::ReadLiveState(state_file.string());
    Check(from_file.image == ref.image && from_file.scalars.pos == ref.scalars.pos && from_file.scalars.mtp_seed == ref.scalars.mtp_seed,
          "DebugExportFullState: the dump file reads back as the exported image");
  }
  hybrid::LiveImage rank_img[2] = {hybrid::ReshardRef(f.g, ref.image, 0), hybrid::ReshardRef(f.g, ref.image, 1)};
  hybrid::DigestList want[2];
  for (int r = 0; r < 2; ++r) want[r] = hybrid::DigestLiveImage(rank_img[r], f.rank_shape[r], pos);
  {
    const hybrid::DigestList full_digest = hybrid::DigestLiveImage(ref.image, f.full_shape, pos);
    Check(Why(full_digest, f.r->DebugLiveStateDigest()).empty(), "R: DebugLiveStateDigest == the digest of its own exported image (device state and host image hash alike)");
  }
  const hybrid::ReshardPlan plan = hybrid::ScatterPlan(f.g, f.p, 0, pos);
  f.host = r4dx::core::PinnedBuffer<uint8_t>(static_cast<size_t>(std::max<uint64_t>(MaxSliceBytes(plan), 1)));
  {
    int local = 0, cross = 0;
    for (const hybrid::CopyOp& op : plan.ops) (op.local ? local : cross)++;
    Check(local > 0 && cross > 0, "the plan has both same-card and cross-card ops");
  }

  hybrid::DigestList scattered[2];
  for (const bool force_host : {true, false}) {
    f.ZeroRanks();
    const uint64_t allocs0 = r4dx::core::g_tp_collective_allocs.load();
    Scatter(f, plan, force_host);
    f.OnRanks([&](Model& m, int) { m.TpAdoptPrefill(ref.scalars); });
    const std::vector<hybrid::DigestList> got = f.RankDigests();
    for (int r = 0; r < 2; ++r) {
      const std::string why = Why(want[r], got[static_cast<size_t>(r)]);
      if (!why.empty()) std::fprintf(stderr, "  rank %d (%s): %s\n", r, force_host ? "all through the host" : "same-card ops D2D", why.c_str());
      Check(why.empty(), std::string("scatter ") + (force_host ? "(every op through the host hop)" : "(same-card ops device-to-device)") + ": rank " + std::to_string(r) +
                             "'s live-state digest equals the digest of ReshardRef(R's state)");
      if (force_host) scattered[r] = got[static_cast<size_t>(r)];
    }
    Check(r4dx::core::g_tp_collective_allocs.load() == allocs0, "the executor allocated nothing inside the rank commands");
  }

  // the rank loaded through DebugImportFullState (the file) is the same state
  f.ZeroRanks();
  f.OnRanks([&](Model& m, int) { m.DebugImportFullState(state_file.string()); });
  {
    const std::vector<hybrid::DigestList> got = f.RankDigests();
    for (int r = 0; r < 2; ++r) Check(Why(want[r], got[static_cast<size_t>(r)]).empty(), "DebugImportFullState: rank " + std::to_string(r) + " loaded from the dump has the reference digest");
  }

  // ---- 2. G-H2 lite: decode from the scattered state and from the imported state -----------------------------------------------------
  std::vector<int32_t> dec_scatter, dec_import;
  hybrid::DigestList after_scatter[2], after_import[2];
  const auto decode = [&](std::vector<int32_t>* out) {
    int32_t tok = 777;
    for (int64_t i = 0; i < kDecode; ++i) {
      tok = f.tpm->DecodeStepGreedy(tok);
      out->push_back(tok);
    }
  };
  decode(&dec_import);
  {
    const std::vector<hybrid::DigestList> d = f.RankDigests();
    after_import[0] = d[0];
    after_import[1] = d[1];
  }
  f.ZeroRanks();
  Scatter(f, plan, /*force_host=*/false);
  f.OnRanks([&](Model& m, int) { m.TpAdoptPrefill(ref.scalars); });
  decode(&dec_scatter);
  {
    const std::vector<hybrid::DigestList> d = f.RankDigests();
    after_scatter[0] = d[0];
    after_scatter[1] = d[1];
  }
  Check(dec_scatter == dec_import && dec_scatter.size() == static_cast<size_t>(kDecode), "G-H2 lite: 4 greedy tokens decoded from the resharded state == from the imported state");
  for (int r = 0; r < 2; ++r) Check(Why(after_scatter[r], after_import[r]).empty(), "G-H2 lite: rank " + std::to_string(r) + "'s live state after the decode is identical for the two loads");
  {
    int64_t p_after[2] = {0, 0};
    f.OnRanks([&](Model& m, int rank) { p_after[rank] = m.PositionCount(); });
    Check(p_after[0] == kPrompt + kDecode && p_after[1] == kPrompt + kDecode, "the ranks decoded from position 300 to 304");
  }

  // ---- 3. warm gather: R is stale at 300, the ranks are at 304 -------------------------------------------------------------------------
  {
    hybrid::TpMasterTracker tracker(f.g.block_tokens, true);
    tracker.AfterPipelined(pos);
    tracker.TpOnly(pos);
    const hybrid::TpMasterTracker::SyncPlan sync = tracker.PlanSync(pos + kDecode);
    Check(sync.stage[0].gdn && sync.stage[1].gdn && sync.stage[0].kv_row0 == pos && sync.stage[0].kv_row1 == pos + kDecode && sync.seed && sync.mtp_block == (pos + kDecode - 1) / 16,
          "the tracker asks for KV rows [300, 304), all GDN state, the seed and the MTP block of row 303 after four TP-only steps");
    const hybrid::ReshardPlan gplan = hybrid::GatherPlan(f.g, f.p, sync);
    // the ranks' exports (their own geometry) -> the host gather reference
    hybrid::LiveState rank_state[2];
    StageSyncState seed_src;
    f.OnRanks([&](Model& m, int rank) {
      m.ReshardCollapse();
      m.ReshardSync();
      rank_state[rank] = m.DebugExportLiveState();
      if (rank == 0) seed_src = m.StageGetSyncState();
    });
    const hybrid::LiveImage gathered_ref = hybrid::GatherRef(f.g, rank_state[0].image, rank_state[1].image);
    const hybrid::DigestList gathered_want = hybrid::DigestLiveImage(gathered_ref, f.full_shape, pos + kDecode);

    const auto restore_stale = [&] { f.r->DebugImportLiveState(ref); };
    for (const int variant : {0, 1, 2, 3}) {  // 0: the real gather (host hop); 1: with same-card ops D2D; 2: without the MTP block; 3: without the KV rows
      restore_stale();
      f.host = r4dx::core::PinnedBuffer<uint8_t>(static_cast<size_t>(std::max<uint64_t>(MaxSliceBytes(gplan), 1)));
      for (const hybrid::CopyOp& op : gplan.ops) {
        if (variant == 2 && op.kind == hybrid::StateKind::kMtpKv) continue;
        if (variant == 3 && op.kind == hybrid::StateKind::kKv) continue;
        GatherOp(f, op, /*force_host=*/variant == 0);
      }
      f.r->TpAdoptPrefill(seed_src);
      const hybrid::DigestList got = f.r->DebugLiveStateDigest();
      const std::string why = Why(gathered_want, got);
      if (variant <= 1) {
        if (!why.empty()) std::fprintf(stderr, "  gather variant %d: %s\n", variant, why.c_str());
        Check(why.empty(), std::string("warm gather ") + (variant == 0 ? "(host hop)" : "(same-card ops D2D)") +
                               ": R's digest equals the digest of GatherRef(the two ranks' exports) -- KV rows, GDN state, the MTP block, the scalars");
      } else {
        Check(!why.empty(), std::string("NEGATIVE CONTROL: a gather ") + (variant == 2 ? "without the MTP block" : "without the KV rows") + " leaves R different from the ranks");
      }
    }
  }

  // ---- 4. negative controls on the scatter ---------------------------------------------------------------------------------------------
  f.r->DebugImportLiveState(ref);  // the gather above moved R on to position 304: the scatters below read its 300-token state again
  for (const Fault fault : {Fault::kSkipKv, Fault::kSkipMtp, Fault::kDropConvSegment, Fault::kSwapHeads, Fault::kKvOneBlockOff}) {
    for (int fr = 0; fr < 2; ++fr) {
      f.host = r4dx::core::PinnedBuffer<uint8_t>(static_cast<size_t>(std::max<uint64_t>(MaxSliceBytes(plan), 1)));
      f.ZeroRanks();
      Scatter(f, plan, /*force_host=*/true, fault, fr);
      f.OnRanks([&](Model& m, int) { m.TpAdoptPrefill(ref.scalars); });
      const std::vector<hybrid::DigestList> got = f.RankDigests();
      Check(!Why(want[fr], got[static_cast<size_t>(fr)]).empty(), std::string("NEGATIVE CONTROL: ") + FaultName(fault) + " on rank " + std::to_string(fr) + " changes its digest");
      Check(Why(want[1 - fr], got[static_cast<size_t>(1 - fr)]).empty(), std::string("... and the other rank is untouched by ") + FaultName(fault));
    }
  }
  // skipped scalars: the digest carries "pos" (a rank that never adopted the scalars reports 0)
  {
    f.ZeroRanks();
    Scatter(f, plan, true);
    const std::vector<hybrid::DigestList> got = f.RankDigests();
    Check(!Why(want[0], got[0]).empty(), "NEGATIVE CONTROL: without TpAdoptPrefill the rank's position (and so its digest) is wrong");
  }

  // ---- guards ---------------------------------------------------------------------------------------------------------------------------
  {
    const hybrid::CopyOp* conv_op = nullptr;
    const hybrid::CopyOp* kv_op = nullptr;
    for (const hybrid::CopyOp& op : plan.ops) {
      if (op.kind == hybrid::StateKind::kGdnConv && conv_op == nullptr && op.rank == 0) conv_op = &op;
      if (op.kind == hybrid::StateKind::kKv && kv_op == nullptr && op.rank == 0) kv_op = &op;
    }
    Check(conv_op != nullptr && kv_op != nullptr, "guards: the plan has a conv and a KV op for rank 0");
    if (conv_op != nullptr && kv_op != nullptr) {
      // a plan built for another rank conv pitch
      hybrid::PlanParams wrong = f.p;
      wrong.rank_conv_pitch = rank_pitch[0] + 1;
      const hybrid::ReshardPlan bad = hybrid::ScatterPlan(f.g, wrong, 0, pos);
      const hybrid::CopyOp* bad_conv = nullptr;
      for (const hybrid::CopyOp& op : bad.ops) {
        if (op.kind == hybrid::StateKind::kGdnConv && op.rank == 0) {
          bad_conv = &op;
          break;
        }
      }
      bool refused = false;
      f.OnRanks([&](Model& m, int rank) {
        if (rank != 0) return;
        refused = Throws<std::logic_error>([&] { m.ReshardImport(*bad_conv, hybrid::WholeOp(*bad_conv)[0], hybrid::Side::kRank, f.host.data()); });
      });
      Check(refused, "guard: a plan built for another conv line pitch is refused by the rank (logic_error), not mis-addressed");
      // the wrong rank / the wrong side
      bool wrong_rank = false;
      f.OnRanks([&](Model& m, int rank) {
        if (rank != 1) return;
        wrong_rank = Throws<std::logic_error>([&] { m.ReshardExport(*kv_op, hybrid::WholeOp(*kv_op)[0], hybrid::Side::kRank, f.host.data()); });
      });
      Check(wrong_rank, "guard: rank 1 refuses an op addressed to rank 0");
      Check(Throws<std::logic_error>([&] { f.r->ReshardExport(*kv_op, hybrid::WholeOp(*kv_op)[0], hybrid::Side::kRank, f.host.data()); }),
            "guard: a TP=1 Model refuses the rank side of an op");
      bool full_on_rank = false;
      f.OnRanks([&](Model& m, int rank) {
        if (rank != 0) return;
        full_on_rank = Throws<std::logic_error>([&] { m.ReshardExport(*kv_op, hybrid::WholeOp(*kv_op)[0], hybrid::Side::kFull, f.host.data()); });
      });
      Check(full_on_rank, "guard: a TP rank refuses the full-head side of an op");
      Check(Throws<std::logic_error>([&] { Model::ReshardCopyLocal(*f.r, *f.r, *kv_op, hybrid::WholeOp(*kv_op)[0], hybrid::Dir::kStageToRank); }),
            "guard: ReshardCopyLocal refuses a TP=1 Model as the rank side (and an op that crosses cards)");
      // an op for a layer the Model has no state for
      hybrid::CopyOp ghost = *kv_op;
      ghost.layer = 0;  // layer 0 is a GDN layer: no KV cache
      Check(Throws<std::logic_error>([&] { f.r->ReshardExport(ghost, hybrid::WholeOp(ghost)[0], hybrid::Side::kFull, f.host.data()); }), "guard: an op for a layer without that state is refused");
      // a slice that reaches past the buffer
      hybrid::CopyOp wide = *kv_op;
      wide.runs[0].height = 100000;
      Check(Throws<std::exception>([&] { f.r->ReshardExport(wide, {0, 0, 100000}, hybrid::Side::kFull, f.host.data()); }), "guard: a slice past the end of the cache is refused");
    }
    // a Model that never called ReshardInit cannot move a pitched conv line
    Model bare = Model::Load(Base());
    const hybrid::CopyOp* any_conv = nullptr;
    for (const hybrid::CopyOp& op : plan.ops) {
      if (op.kind == hybrid::StateKind::kGdnConv) {
        any_conv = &op;
        break;
      }
    }
    if (any_conv != nullptr) {
      (void)bare.Prefill(Tokens(20, 2));
      Check(Throws<std::logic_error>([&] { bare.ReshardExport(*any_conv, hybrid::WholeOp(*any_conv)[0], hybrid::Side::kFull, f.host.data()); }),
            "guard: a pitched conv export without ReshardInit is refused (no allocation in the hot path)");
    }
  }

  // ---- 5. speculative state: the digest follows it, the executor refuses it until collapsed --------------------------------------------------
  for (const int wo : {0, 1}) {
    Model m = Model::Load(Base(wo));
    m.ReshardInit();
    (void)m.Prefill(Tokens(100, 3));
    const std::vector<int32_t> cands = Tokens(4, 9);
    (void)m.VerifyWindow(cands);
    m.CommitVerifiedWindow(3);  // the live GDN state is now in window slot 2 / behind a pending prefix, conv history at offset 2
    const hybrid::DigestList speculative = m.DebugLiveStateDigest();
    const hybrid::LiveState export_before = m.DebugExportLiveState();
    const hybrid::ReshardPlan one = hybrid::ScatterPlan(f.g, f.p, 0, m.PositionCount());
    const hybrid::CopyOp* rec_op = nullptr;
    const hybrid::CopyOp* conv_op = nullptr;
    for (const hybrid::CopyOp& op : one.ops) {
      if (op.kind == hybrid::StateKind::kGdnRecurrent && rec_op == nullptr) rec_op = &op;
      if (op.kind == hybrid::StateKind::kGdnConv && conv_op == nullptr) conv_op = &op;
    }
    if (rec_op != nullptr && conv_op != nullptr) {
      const std::string tag = wo != 0 ? " (write-once)" : " (window slots)";
      Check(Throws<std::logic_error>([&] { m.ReshardExport(*rec_op, hybrid::WholeOp(*rec_op)[0], hybrid::Side::kFull, f.host.data()); }),
            "a recurrent op on a Model with a committed speculative window is refused" + tag);
      Check(Throws<std::logic_error>([&] { m.ReshardExport(*conv_op, hybrid::WholeOp(*conv_op)[0], hybrid::Side::kFull, f.host.data()); }),
            "a conv op on a Model with a committed speculative window is refused" + tag);
      m.ReshardCollapse();
      m.ReshardSync();
      const hybrid::DigestList collapsed = m.DebugLiveStateDigest();
      Check(Why(speculative, collapsed).empty(), "ReshardCollapse leaves the live-state digest unchanged (the digest follows the window slot / conv offset)" + tag);
      Check(m.DebugExportLiveState().image == export_before.image, "the exported host image of the speculative state equals that of the collapsed one" + tag);
      // after the collapse the same ops are accepted (an exception here fails the test)
      m.ReshardExport(*rec_op, hybrid::WholeOp(*rec_op)[0], hybrid::Side::kFull, f.host.data());
      m.ReshardExport(*conv_op, hybrid::WholeOp(*conv_op)[0], hybrid::Side::kFull, f.host.data());
      m.ReshardSync();
    }
  }

  std::filesystem::remove(state_file);
  if (g_fails != 0) {
    std::fprintf(stderr, "test_reshard_exec: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_reshard_exec: PASS\n");
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_reshard_exec", RunTest); }
