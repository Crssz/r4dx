// tests/model/test_hybrid_emulate_identity.cpp -- the ONE-DEVICE identity harness of the hybrid serving mode (docs/pp-tp2-hybrid.md 4, 9 "P2 one
// device (emulate)", gates G-H1 / G-H2, the negative controls of section 4, Gate C; section 13 of that file describes this test).
//
// Everything runs on ONE device (HIP_VISIBLE_DEVICES=1 from CMake) in one process:
//   R   a TP=1 Model        -- the reference: the prefill the hybrid must reproduce byte for byte;
//   X,Y the two stage Models (ModelOptions::stage_only front / back, hole-prefix, the shared pinned embedding and R's borrowed device mirror), attached
//       as pipeline stages WITHOUT the KV payload (PpStageSetup::carry_kv = false) and driven the way the real two-card prefill will drive them: stage X
//       on its own thread, stage Y on this one, a 3-slot pinned StageChannel between them;
//   TP  a TpModel in emulate mode -- the two TP=2 rank Models on their own threads, reached through RunCollectiveForTest.
// A hybrid CALL is: [warm: the TP-master gather (hybrid_sync.h TpMasterTracker + GatherPlan: scalars + MTP seed into both stages, stale KV rows and GDN
// state of the shards into the stages)], the stage prefill, the scatter (ScatterPlan: the reshard executor moves the stages' live state into the two rank
// Models, same-card ops device-to-device, cross-card ops through the pinned host ring), TpAdoptPrefill on both ranks.
//
//   G-H1  after every call each rank's Model::DebugLiveStateDigest equals the digest of hybrid::ReshardRef(R's exported live state, rank) -- the same
//         records hashed by the same code from a host image; and Y's logits equal R's byte for byte. After a warm turn R is first loaded with the ranks'
//         gathered state (GatherRef -> DebugImportLiveState: the TP=1 oracle of "this prefill from THIS state") and then prefills the same tokens.
//   G-H2  from the final state N = 16 greedy tokens (plain, then --mtp 3 rounds, rewound by the TpModel checkpoint) decoded at TP=2 by the hybrid ranks
//         equal those decoded by ranks loaded with the reference image through DebugImportFullState (the dump file), and the ranks' states afterwards agree.
//   Scenarios: row counts 1, 17, 256, 300, 1000, 2048 + 77; prefix-reuse shapes (300 + 333, 64 + 511, 257 + 1, 1 + 255 + 257: consecutive hybrid calls, the
//         MTP-block-only warm gather); warm turns (prefill, 20 tokens decoded at TP=2 plain / --mtp 3, second hybrid prefill with the warm gather
//         incl. the MTP seed sync into stage Y); checkpoint save / restore around a decode and a warm turn after the restore; image rows (synthetic merger rows
//         spliced by PrefillMultimodal, mrope carried into the next turn) where the container has a vision config.
//   Configs (each one "configuration" in the final count): l4/bf16 (the 4-layer selftest container, split 2, window-slot GDN), l4/bf16-writeonce,
//         l4/w4a16, real16 (the production trellis container with layer_limit 16, split 8: real shapes, real shard rules). ~20 GiB resident for real16.
//   NEGATIVE CONTROLS (design section 4; each must change a digest or a token comparison, printed "[PASS] negative control NCk: ..."):
//         NC1 swap the head halves   NC2 drop a conv segment   NC3 skip the MTP half   NC5 off-by-one KV block   NC6 wrong rank mapping (GDN v-heads)
//         NC7 stale mirror (no gather / no GDN gather / no KV-rows gather on a warm turn)   NC8 skip the scalars (and drop the mrope delta)
//         NC9 skip the TP -> Y MTP seed sync on a warm turn   NC10 the stage prefill under the TP2 tuning flag (the entry guard throws; with the
//         DebugSetTp2Tuning hook the digest changes)   NC12 MTP KV copied from block 0 instead of the primed range on a warm turn.
//         NC4 (skip the DFlash tail) and NC11 (drop the image rope rows in the tail injection) are in Gate C, which needs the production drafter.
//   Gate C (drafters only, ~4 GiB + the 4-layer target for the captured features): the tail-fed drafter ring == the fully fed one under the TP-thread
//         tuning flag, features captured by Model::StageArmDflashTail, rope rows from PrefillMultimodal's rope_rows_out through TemporalRopeRows.
//   Gate B (full depth, 64 layers; separate mode, see below): the reference image is dumped by an EARLIER process; the emulated hybrid runs with the embedding
//         gathered from the host. Cold scenarios only (no TP=1 Model fits next to the ranks and stages at that depth).
//
// Modes:   test_hybrid_emulate_identity                       Gates A + C (what CTest runs)
//          test_hybrid_emulate_identity --gate-b-dump <dir>   process 1 of Gate B: the TP=1 reference prefills, state files + logits hashes into <dir>
//          test_hybrid_emulate_identity --gate-b <dir>        process 2 of Gate B: ranks + stages only, --embed-device-resident off, compares with <dir>
//   (env R4DX_HYBRID_GATE_B=dump|verify + R4DX_HYBRID_GATE_B_DIR=<dir> are the same switches; --layers N / --split K change the Gate B depth.)
//   R4DX_TEST_ONLY=<substring> runs only the configurations / scenarios whose "cfg/name" contains it.
// Output: "[PASS] ..." / "FAIL ..." lines on stderr and a final "test_hybrid_emulate_identity: PASS (N configurations)"; exit 0 only when every required
// case passed, 1 on a failure, 77 when there is no device or no container at all. Device 1 (HIP_VISIBLE_DEVICES=1), links the R4DX_TP_TESTING variant.
// Written, NOT run by its author (CPU-only session).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dflash_draft.h"
#include "dflash_draft_weights.h"
#include "dflash_tail.h"
#include "dflash_tail_plan.h"
#include "linear.h"
#include "model.h"
#include "pp_channel.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "test_common.h"
#include "tp_model.h"

#ifndef R4DX_TP_TESTING
#error "test_hybrid_emulate_identity needs the R4DX_TP_TESTING hooks: link r4dx_model_tptest (src/model/CMakeLists.txt)"
#endif

using namespace r4dx_test;
using r4dx::core::DeviceBuffer;
using r4dx::core::PinnedBuffer;
using r4dx::model::Container;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::StageSyncState;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace hybrid = r4dx::model::hybrid;
namespace pp = r4dx::model::pp;
namespace stage = r4dx::model::stage;

namespace {

constexpr uint64_t kSliceBudget = 1u << 20;  // bytes per host-hop slice: several slices per KV op, one per recurrent op
constexpr size_t kRingBytes = 4u << 20;      // the pinned ring: the widest row of any op is the 1.5 MiB recurrent run
constexpr int kDecodeTokens = 16;            // G-H2: tokens per mode
constexpr int kMtpK = 3;                     // --mtp 3
constexpr int kStageTimeoutMs = 120000;      // the first call of a process loads every kernel module

// ---- bookkeeping --------------------------------------------------------------------------------------------------------------------
int g_fails = 0;
int g_configs = 0;
std::set<int> g_nc_done;

bool Only(const std::string& name) {
  const char* only = std::getenv("R4DX_TEST_ONLY");
  return only == nullptr || *only == '\0' || name.find(only) != std::string::npos;
}

// One scenario's checks: silent while they pass, one "[PASS]" line at Done(), a "FAIL" line per failing check.
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
  void Fail(const std::string& what) { Ok(false, what); }
  void Done(const std::string& note = "") {
    if (fails_ == 0) std::fprintf(stderr, "[PASS] %s: %d checks%s%s\n", name_.c_str(), checks_, note.empty() ? "" : " -- ", note.c_str());
  }
  bool Failed() const { return fails_ != 0; }
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  int checks_ = 0, fails_ = 0;
};

void Skip(const std::string& what, const std::string& why, bool required) {
  std::fprintf(stderr, "[SKIP] %s: %s%s\n", what.c_str(), why.c_str(), required ? " (REQUIRED: counted as a failure)" : "");
  if (required) ++g_fails;
}

void NegativeControl(int nc, bool detected, const std::string& what, const std::string& observable) {
  if (detected) {
    std::fprintf(stderr, "[PASS] negative control NC%d: %s -- %s\n", nc, what.c_str(), observable.c_str());
    g_nc_done.insert(nc);
  } else {
    ++g_fails;
    std::fprintf(stderr, "FAIL negative control NC%d: %s was NOT detected (%s)\n", nc, what.c_str(), observable.c_str());
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

uint64_t Fnv(const void* p, size_t n) { return hybrid::FnvUpdate(hybrid::kFnvInit, p, n); }
uint64_t HashLogits(const std::vector<float>& x) { return Fnv(x.data(), x.size() * sizeof(float)); }
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

// ---- synthetic image rows (test_pp_emulate_identity's: the vision tower's output replaced by deterministic bf16 rows) -----------------
struct ImageRows {
  DeviceBuffer<uint16_t> dev;
  int64_t tokens = 0;
  int64_t side = 0;  // patch grid h == w
};

ImageRows MakeImageRows(int64_t merge, int64_t tokens_side, int64_t hidden, int salt) {
  ImageRows r;
  r.side = tokens_side * merge;
  r.tokens = tokens_side * tokens_side;
  std::vector<uint16_t> host(static_cast<size_t>(r.tokens * hidden));
  uint32_t x = 12345u + static_cast<uint32_t>(salt);
  for (auto& v : host) {
    x = x * 1664525u + 1013904223u;
    v = r4dx::core::FloatToBf16((static_cast<float>(x >> 9) / static_cast<float>(1u << 23) - 0.5f) * 0.5f);
  }
  r.dev = DeviceBuffer<uint16_t>(host.size());
  r.dev.CopyFromHost(host);
  return r;
}

// ---- faults (the negative controls) -------------------------------------------------------------------------------------------------
enum class Fault {
  kNone,
  // scatter-time (the stage -> rank reshard at the end of a call)
  kSwapKvHeads,       // NC1  rank r's first KV op reads the OTHER rank's head half
  kDropConvSegment,   // NC2  one conv segment (the v channels) of one layer is not copied
  kSkipMtp,           // NC3  the MTP-KV ops of one rank are skipped
  kKvOneBlockOff,     // NC5  a KV op's source starts one block late
  kWrongRankGdn,      // NC6  rank r's first recurrent op reads the other rank's v-heads
  kMtpFromBlock0,     // NC12 the MTP KV is copied from block 0 instead of the primed range (warm turn)
  kSkipAdopt,         // NC8  TpAdoptPrefill is skipped
  kDropMrope,         // NC8b the adopted scalars lose the mrope delta
  kTuningFlag,        // NC10 the stage prefill runs under the TP2 tuning flag (needs the DebugSetTp2Tuning hook)
  // gather-time (the TP-master sync before a warm call)
  kSkipGather,        // NC7a nothing is gathered (stale mirror)
  kSkipGatherGdn,     // NC7b the GDN state is not gathered
  kSkipGatherKv,      // NC7c the KV rows are not gathered
  kStaleSeed,         // NC9  stage Y keeps its own (stale) MTP seed instead of rank 0's
};
bool IsGatherFault(Fault f) { return f == Fault::kSkipGather || f == Fault::kSkipGatherGdn || f == Fault::kSkipGatherKv; }
struct FaultSpec {
  Fault f = Fault::kNone;
  int rank = 0;  // the rank an op-level fault damages
};

// ---- scenario description -------------------------------------------------------------------------------------------------------------
enum class StepKind { kCall, kDecode, kSave, kRestore };
struct Step {
  StepKind kind = StepKind::kCall;
  int n = 0;                  // call: text rows; decode: tokens
  int salt = 1;
  int image = 0;              // call: 0 none, 1 big image (256 tokens), 2 small (144)
  int before = 0, after = 0;  // call with an image: text rows before / after it
  bool mm = false;            // call: PrefillMultimodal with no image (the multi-turn continuation case)
  bool mtp = false;           // decode: --mtp 3 rounds instead of plain steps
  bool same_as_prev = false;  // decode: the tokens must equal the previous decode's (after a restore)
};
Step Call(int n, int salt = 1) {
  Step s;
  s.n = n;
  s.salt = salt;
  return s;
}
Step ImageCall(int image, int before, int after, int salt = 1) {
  Step s;
  s.image = image;
  s.before = before;
  s.after = after;
  s.salt = salt;
  return s;
}
Step Decode(int n, bool mtp, bool same_as_prev = false) {
  Step s;
  s.kind = StepKind::kDecode;
  s.n = n;
  s.mtp = mtp;
  s.same_as_prev = same_as_prev;
  return s;
}
Step Save() {
  Step s;
  s.kind = StepKind::kSave;
  return s;
}
Step Restore() {
  Step s;
  s.kind = StepKind::kRestore;
  return s;
}
struct Scenario {
  std::string name;
  std::vector<Step> steps;
  bool host_hop = false;      // every reshard op through the host ring (else same-card ops device-to-device)
  bool check_decode = true;   // G-H2 at the end
  bool needs_vision = false;
};

// ---- the per-call reference, what a call returns -------------------------------------------------------------------------------------------
struct RefState {
  hybrid::LiveState st;
  uint64_t logits_hash = 0;
  int32_t first_token = 0;
};
struct CallResult {
  bool logits_equal = false;
  bool scalars_equal = false;
  bool digest_equal[2] = {false, false};
  std::string why[2];
  int64_t p0 = 0, n = 0;
};
struct DecodeOutcome {
  std::vector<int32_t> tok[2];            // [mode: 0 plain, 1 mtp]
  hybrid::DigestList dig[2][2];           // [mode][rank]
};
bool CallOk(const CallResult& c) { return c.logits_equal && c.scalars_equal && c.digest_equal[0] && c.digest_equal[1]; }
struct ScenarioOut {
  std::vector<CallResult> calls;          // every call in order; the last one is the one a fault was injected into
  bool decode_checked = false;
  bool tokens_equal[2] = {true, true};
  bool digests_equal[2] = {true, true};
  const CallResult& Last() const { return calls.back(); }
  bool EarlierCallsOk() const {
    for (size_t i = 0; i + 1 < calls.size(); ++i) {
      if (!CallOk(calls[i])) return false;
    }
    return true;
  }
};

struct Cfg {
  std::string name, path;
  Layout layout = Layout::kBf16;
  int64_t layers = -1;  // layer_limit
  int64_t split = 2;
  int gdn_wo = -1;
  bool vision = false;      // R and the stages parse the vision config (image scenarios)
  bool gate_b = false;      // the verify process of Gate B: no R, the embedding gathered from the host
  bool want_r = true;       // load the TP=1 reference
  bool want_hybrid = true;  // load the emulated ranks and the two stages
  int64_t max_ctx = 4096;   // R, the ranks AND the stage-KV capacity S
  bool nc10_must_flip = false;
};

// ======================================================================================================================================
// The rig
// ======================================================================================================================================
class Rig {
 public:
  Cfg cfg;
  std::unique_ptr<Model> r, x, y;
  std::unique_ptr<TpModel> tpm;
  hybrid::StateGeometry g;
  hybrid::PlanParams p;
  hybrid::DigestShape full_shape, rank_shape[2];
  std::unique_ptr<hybrid::TpMasterTracker> tracker;
  PinnedBuffer<uint8_t> ring;
  std::vector<PinnedBuffer<uint8_t>> slots;
  std::unique_ptr<pp::StageChannel> channel;
  std::shared_ptr<const PinnedBuffer<uint16_t>> embed_host;
  int device = 0;
  int64_t call_id = 0;
  int64_t hidden = 0;
  std::unique_ptr<ImageRows> big, small;
  std::string state_file;

  explicit Rig(Cfg c) : cfg(std::move(c)) {
    if (hipGetDevice(&device) != hipSuccess) throw std::runtime_error("hipGetDevice failed");
    state_file = (std::filesystem::temp_directory_path() / ("r4dx_hybrid_emu_" + Sanitized(cfg.name) + ".state")).string();
    ModelOptions base;
    base.container_path = cfg.path;
    base.layout = cfg.layout;
    base.max_ctx = cfg.max_ctx;
    base.layer_limit = cfg.layers;
    base.mtp_draft_k = kMtpK;
    base.gdn_write_once = cfg.gdn_wo;
    base.vision = cfg.vision ? ModelOptions::VisionMode::kAuto : ModelOptions::VisionMode::kOff;
    base.embed_device_resident = !cfg.gate_b;
    if (cfg.want_r) {
      std::fprintf(stderr, "[hybrid-emu] %s: loading the TP=1 reference\n", cfg.name.c_str());
      r = std::make_unique<Model>(Model::Load(base));
    }
    if (!cfg.want_hybrid) return;

    // The TP rank Models (emulate: two threads, one device). Vision stays off on the ranks: they never prefill an image, the mrope delta arrives
    // with the adopted scalars.
    std::fprintf(stderr, "[hybrid-emu] %s: loading the two emulated TP ranks\n", cfg.name.c_str());
    ModelOptions to = base;
    to.vision = ModelOptions::VisionMode::kOff;
    to.prompt_checkpoint = true;
    TpOptions emu;
    emu.world = 2;
    emu.mode = TpOptions::Mode::kEmulate;
    tpm = TpModel::Load(to, emu);

    // The stage Models: the process's shared pinned embedding, R's device mirror borrowed (none in the host-gather mode of Gate B).
    std::fprintf(stderr, "[hybrid-emu] %s: loading the stage Models (split %lld)\n", cfg.name.c_str(), static_cast<long long>(cfg.split));
    embed_host = Container::LoadEmbedTokensHost(cfg.path);
    std::shared_ptr<const r4dx::model::EmbedMirrorLease> lease;
    if (r) lease = r->GetContainer().EmbedMirrorLeaseHandle();
    ModelOptions yo = base;
    yo.stage_only.role = stage::Role::kBack;
    yo.stage_only.split = cfg.split;
    yo.stage_only.shared_embed_host = embed_host;
    yo.stage_only.borrowed_embed_mirror = lease;
    yo.stage_only.embed_device_resident_decided = cfg.gate_b ? 0 : -1;
    ModelOptions xo = yo;
    xo.stage_only.role = stage::Role::kFront;
    xo.layer_limit = cfg.split + 1;
    xo.mtp_draft_k = 0;
    x = std::make_unique<Model>(Model::Load(xo));
    y = std::make_unique<Model>(Model::Load(yo));
    x->CheckStageKv(cfg.max_ctx);
    y->CheckStageKv(cfg.max_ctx);
    if (x->PrefillChunkRows() != y->PrefillChunkRows() || x->PrefillInt8Enabled() != y->PrefillInt8Enabled() ||
        (r && (r->PrefillChunkRows() != y->PrefillChunkRows() || r->PrefillInt8Enabled() != y->PrefillInt8Enabled()))) {
      throw std::runtime_error("the stages and the reference chose different prefill chunk sizes / int8 prefill: the bits would differ");
    }
    hidden = y->Config().hidden_size;

    // Geometry and plan parameters. The 4-layer file's config still describes 64 layers: the plan covers the loaded ones only.
    r4dx::model::ModelConfig gc = y->GlobalConfig();
    gc.num_hidden_layers = y->GetContainer().NumLoadedLayers();
    gc.layer_types.resize(static_cast<size_t>(gc.num_hidden_layers));
    g = hybrid::StateGeometry::FromRules(gc, y->PpKvBlockSize());
    x->ReshardInit();
    y->ReshardInit();
    int64_t rank_pitch[2] = {0, 0};
    OnRanks([&](Model& m, int rank) {
      m.ReshardInit();
      rank_pitch[rank] = m.ReshardConvPitch();
    });
    if (rank_pitch[0] != rank_pitch[1]) throw std::runtime_error("the two ranks disagree on the conv line pitch");
    p = hybrid::PlanParams::ForSplit(g, cfg.split, rank_pitch[0]);
    p.stage_conv_pitch = x->ReshardConvPitch();
    p.stage_b_conv_pitch = y->ReshardConvPitch();
    full_shape = {g.block_tokens, g.kv_heads_full, g.KvTokenHeadBytes(), g.conv_live};
    for (int k = 0; k < 2; ++k) rank_shape[k] = {g.block_tokens, g.ranks[static_cast<size_t>(k)].kv_heads, g.KvTokenHeadBytes(), g.conv_live};
    tracker = std::make_unique<hybrid::TpMasterTracker>(g.block_tokens, true);
    ring = PinnedBuffer<uint8_t>(kRingBytes);

    // The channel between the stages. No KV payload and no DFlash columns: the carry only.
    const size_t cap = pp::MaxSlotBytes(y->PrefillChunkRows(), hidden, /*dfl_cols=*/0, /*attn_layers=*/0, y->PpKvBlockSize(), y->PpKvBlockStrideBytes());
    std::vector<uint8_t*> ptrs;
    for (int i = 0; i < 3; ++i) {
      slots.emplace_back(cap, hipHostMallocPortable);
      ptrs.push_back(slots.back().data());
    }
    channel = std::make_unique<pp::StageChannel>(ptrs, cap);
    x->PpAttach(StageSetup(Model::PpRole::kStageA));
    y->PpAttach(StageSetup(Model::PpRole::kStageB));
    std::fprintf(stderr, "[hybrid-emu] %s: ready (%lld layers, attention %zu / GDN %zu, conv pitch rank %lld, stage X %lld, stage Y %lld)\n", cfg.name.c_str(),
                 static_cast<long long>(g.num_layers), g.attn_layers.size(), g.gdn_layers.size(), static_cast<long long>(p.rank_conv_pitch),
                 static_cast<long long>(p.stage_conv_pitch), static_cast<long long>(p.StageConvPitch(hybrid::kStageB)));
  }

  static std::string Sanitized(std::string s) {
    for (char& c : s) {
      if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
  }

  Model::PpStageSetup StageSetup(Model::PpRole role) {
    Model::PpStageSetup s;
    s.role = role;
    s.split = cfg.split;
    s.channel = channel.get();
    s.timeout_ms = kStageTimeoutMs;
    s.carry_kv = false;
    return s;
  }

  // ---- helpers on the ranks ---------------------------------------------------------------------------------------------------------
  void OnRanks(const std::function<void(Model&, int)>& fn) { tpm->RunCollectiveForTest(fn); }
  void ZeroRanks() {
    tpm->Reset();
    OnRanks([](Model& m, int) { m.DebugZeroKvState(); });
  }
  void ResetStages() {
    x->Reset();
    y->Reset();
    x->DebugZeroKvState();
    y->DebugZeroKvState();
    tracker->Reset();
  }
  std::vector<hybrid::DigestList> RankDigests() {
    std::vector<hybrid::DigestList> out(2);
    OnRanks([&](Model& m, int rank) { out[static_cast<size_t>(rank)] = m.DebugLiveStateDigest(); });
    return out;
  }
  // The ranks' live state as host images (after collapsing any speculative window), one LiveState per rank.
  std::vector<hybrid::LiveState> SnapshotRanks() {
    std::vector<hybrid::LiveState> rs(2);
    OnRanks([&](Model& m, int rank) {
      m.ReshardCollapse();
      m.ReshardSync();
      rs[static_cast<size_t>(rank)] = m.DebugExportLiveState();
    });
    return rs;
  }
  Model& StageOf(int s) { return s == hybrid::kStageA ? *x : *y; }

  // ---- the stage prefill: X on its own thread, Y on this one ---------------------------------------------------------------------------
  static bool IsPoison(const std::exception_ptr& e) {
    try {
      std::rethrow_exception(e);
    } catch (const pp::ChannelPoisoned&) {
      return true;
    } catch (...) {
    }
    return false;
  }
  std::vector<float> StagePrefill(const std::vector<int32_t>& ids, const std::vector<Model::ImageSpan>* images, bool tp_flag) {
    channel->BeginCall(++call_id);
    std::exception_ptr xerr, yerr;
    std::vector<float> logits;
    pp::StageChannel* const ch = channel.get();
    std::thread tx([&] {
      try {
        if (hipSetDevice(device) != hipSuccess) throw std::runtime_error("hipSetDevice failed on the stage X thread");
        r4dx::model::Tp2TuningScope scope(tp_flag);
        if (images != nullptr) {
          (void)x->PrefillMultimodal(ids, *images);
        } else {
          (void)x->Prefill(ids);
        }
      } catch (...) {
        xerr = std::current_exception();
        ch->Poison("stage X failed");
      }
    });
    try {
      r4dx::model::Tp2TuningScope scope(tp_flag);
      y->PpSetActive(true);
      logits = images != nullptr ? y->PrefillMultimodal(ids, *images) : y->Prefill(ids);
    } catch (...) {
      yerr = std::current_exception();
      ch->Poison("stage Y failed");
    }
    tx.join();
    try {
      y->PpSetActive(false);
    } catch (...) {
    }
    if (xerr || yerr) {
      ch->Reset();
      const std::exception_ptr root = (xerr && (!yerr || IsPoison(yerr))) ? xerr : yerr;
      std::rethrow_exception(root);
    }
    return logits;
  }

  // ---- the reshard executor, driven by the plan's ops ------------------------------------------------------------------------------------
  void ScatterOp(const hybrid::CopyOp& op, bool force_host) {
    Model& st = StageOf(op.stage);
    if (op.local && !force_host) {
      OnRanks([&](Model& m, int rank) {
        if (rank != op.rank) return;
        for (const hybrid::OpSlice& s : hybrid::WholeOp(op)) Model::ReshardCopyLocal(st, m, op, s, hybrid::Dir::kStageToRank);
        m.ReshardSync();
      });
      return;
    }
    for (const hybrid::OpSlice& s : hybrid::SplitOp(op, kSliceBudget)) {
      if (hybrid::SliceBytes(op, s) > ring.bytes()) throw std::logic_error("a slice does not fit the pinned ring");
      st.ReshardExport(op, s, hybrid::Side::kFull, ring.data());
      st.ReshardSync();
      OnRanks([&](Model& m, int rank) {
        if (rank != op.rank) return;
        m.ReshardImport(op, s, hybrid::Side::kRank, ring.data());
        m.ReshardSync();
      });
    }
  }
  void GatherOp(const hybrid::CopyOp& op, bool force_host) {
    Model& st = StageOf(op.stage);
    if (op.local && !force_host) {
      OnRanks([&](Model& m, int rank) {
        if (rank != op.rank) return;
        for (const hybrid::OpSlice& s : hybrid::WholeOp(op)) Model::ReshardCopyLocal(st, m, op, s, hybrid::Dir::kRankToStage);
        st.ReshardSync();
      });
      return;
    }
    for (const hybrid::OpSlice& s : hybrid::SplitOp(op, kSliceBudget)) {
      if (hybrid::SliceBytes(op, s) > ring.bytes()) throw std::logic_error("a slice does not fit the pinned ring");
      OnRanks([&](Model& m, int rank) {
        if (rank != op.rank) return;
        m.ReshardExport(op, s, hybrid::Side::kRank, ring.data());
        m.ReshardSync();
      });
      st.ReshardImport(op, s, hybrid::Side::kFull, ring.data());
      st.ReshardSync();
    }
  }

  // The fault applied to one op; false = the op is skipped. `done` makes a "first op of its kind" fault fire once.
  bool MutateOp(const hybrid::ReshardPlan& plan, const hybrid::CopyOp& op, const FaultSpec& fs, bool gather, hybrid::CopyOp* use, bool* done) {
    using hybrid::StateKind;
    if (IsGatherFault(fs.f) != gather || fs.f == Fault::kNone) return true;
    const bool mine = op.rank == fs.rank;
    switch (fs.f) {
      case Fault::kSkipMtp:
        if (op.kind == StateKind::kMtpKv && mine) return false;
        break;
      case Fault::kDropConvSegment:
        if (op.kind == StateKind::kGdnConv && mine && !*done) {
          *done = true;
          use->runs.pop_back();  // the v segment: the biggest of the three
        }
        break;
      case Fault::kSwapKvHeads:
        if (op.kind == StateKind::kKv && mine && !*done) {
          *done = true;
          for (const hybrid::CopyOp& other : plan.ops) {
            if (other.kind == op.kind && other.layer == op.layer && other.stage == op.stage && other.rank != op.rank) use->runs[0].full_off = other.runs[0].full_off;
          }
        }
        break;
      case Fault::kKvOneBlockOff:
        if (op.kind == StateKind::kKv && mine && !*done && use->runs[0].height > 1) {
          *done = true;
          use->runs[0].full_off += use->runs[0].full_pitch;
          use->runs[0].height -= 1;  // blocks 1.. of the source land on blocks 0..
        }
        break;
      case Fault::kWrongRankGdn:
        if (op.kind == StateKind::kGdnRecurrent && mine && !*done) {
          *done = true;
          for (const hybrid::CopyOp& other : plan.ops) {
            if (other.kind == op.kind && other.layer == op.layer && other.stage == op.stage && other.rank != op.rank) use->runs[0].full_off = other.runs[0].full_off;
          }
        }
        break;
      case Fault::kMtpFromBlock0:
        if (op.kind == StateKind::kMtpKv) {
          const hybrid::Run2D& run = op.runs[0];
          const int64_t first = static_cast<int64_t>(run.rank_off / run.rank_pitch);
          *use = hybrid::detail::KvOp(g, StateKind::kMtpKv, -1, hybrid::kStageB, op.rank, hybrid::BlockRange{0, first + static_cast<int64_t>(run.height) - 1});
        }
        break;
      case Fault::kSkipGather:
        return false;
      case Fault::kSkipGatherGdn:
        if (op.kind == StateKind::kGdnRecurrent || op.kind == StateKind::kGdnConv) return false;
        break;
      case Fault::kSkipGatherKv:
        if (op.kind == StateKind::kKv) return false;
        break;
      default:
        break;
    }
    return true;
  }
  void ExecPlan(const hybrid::ReshardPlan& plan, const FaultSpec& fs, bool gather, bool force_host) {
    bool done = false;
    for (const hybrid::CopyOp& op : plan.ops) {
      hybrid::CopyOp use = op;
      if (!MutateOp(plan, op, fs, gather, &use, &done)) continue;
      if (gather) {
        GatherOp(use, force_host);
      } else {
        ScatterOp(use, force_host);
      }
    }
  }

  // ---- the TP-master gather before a warm call (design 5) ----------------------------------------------------------------------------------
  void WarmSync(int64_t p0, const FaultSpec& fs, bool force_host) {
    const hybrid::TpMasterTracker::SyncPlan sync = tracker->PlanSync(p0);
    StageSyncState s;
    OnRanks([&](Model& m, int rank) {
      m.ReshardCollapse();
      m.ReshardSync();
      if (rank == 0) s = m.StageGetSyncState();
    });
    x->StageSetSyncState(s);
    StageSyncState sy = s;
    if (fs.f == Fault::kStaleSeed) {  // NC9: stage Y keeps the seed its own last call left
      const StageSyncState old = y->StageGetSyncState();
      sy.mtp_seed_valid = old.mtp_seed_valid;
      sy.mtp_seed = old.mtp_seed;
    }
    y->StageSetSyncState(sy);
    ExecPlan(hybrid::GatherPlan(g, p, sync), fs, /*gather=*/true, force_host);
    tracker->AfterSync(p0);
  }

  // ---- images ---------------------------------------------------------------------------------------------------------------------------------
  bool VisionOk() const { return cfg.vision && y && y->VisionSpliceEnabled() && (!r || r->VisionSpliceEnabled()); }
  void EnsureImages() {
    if (big) return;
    const int64_t merge = y->GetContainer().VisionCfg().spatial_merge_size;
    big = std::make_unique<ImageRows>(MakeImageRows(merge, 16, hidden, 1));    // 256 tokens
    small = std::make_unique<ImageRows>(MakeImageRows(merge, 12, hidden, 2));  // 144 tokens
  }
  // The call's tokens and image spans.
  void BuildCall(const Step& st, std::vector<int32_t>* ids, std::vector<Model::ImageSpan>* spans, bool* multimodal) {
    spans->clear();
    *multimodal = st.image != 0 || st.mm;
    if (st.image == 0) {
      *ids = Tokens(st.n, st.salt);
      return;
    }
    EnsureImages();
    const ImageRows& rows = st.image == 1 ? *big : *small;
    const int32_t image_id = static_cast<int32_t>(y->GetContainer().ImageTokenId());
    const std::vector<int32_t> a = Tokens(st.before, st.salt + 3), b = Tokens(st.after, st.salt + 5);
    ids->assign(a.begin(), a.end());
    Model::ImageSpan sp;
    sp.offset = static_cast<int64_t>(ids->size());
    sp.tokens = rows.tokens;
    sp.grid.t = 1;
    sp.grid.h = rows.side;
    sp.grid.w = rows.side;
    sp.embeds = rows.dev.data();
    sp.embeds_on_host = false;
    spans->push_back(sp);
    ids->insert(ids->end(), static_cast<size_t>(rows.tokens), image_id);
    ids->insert(ids->end(), b.begin(), b.end());
  }

  // ---- one hybrid call + its reference -------------------------------------------------------------------------------------------------------
  struct Ctx {
    bool fresh = true;
    bool ranks_moved = false;
    int calls_left = 0;
    int call_index = 0;
    int fault_call = -1;  // ordinal of the call the fault is injected into (-1: the last)
    int32_t next_token = 0;
    RefState ref;
    CallResult last;
    std::vector<std::vector<int32_t>> decodes;
    std::vector<hybrid::DigestList> saved_digest;
    int64_t saved_pos = 0;
    int32_t saved_token = 0;
    std::string ref_file;  // Gate B verify: the dumped reference of this call
  };

  RefState ReferenceCall(Ctx& ctx, const std::vector<int32_t>& ids, const std::vector<Model::ImageSpan>* images, const std::vector<hybrid::LiveState>* rs) {
    RefState ref;
    if (ctx.fresh) {
      r->Reset();
      r->DebugZeroKvState();
    } else if (rs != nullptr) {
      // The ranks moved on without R (TP=2 decode, a restore): R becomes the gathered state of the ranks, then does what the stages are about to do.
      hybrid::LiveState full;
      full.image = hybrid::GatherRef(g, (*rs)[0].image, (*rs)[1].image);
      full.scalars = (*rs)[0].scalars;
      r->DebugImportLiveState(full);
    }
    const std::vector<float> logits = images != nullptr ? r->PrefillMultimodal(ids, *images) : r->Prefill(ids);
    ref.st = r->DebugExportLiveState();
    ref.logits_hash = HashLogits(logits);
    ref.first_token = Argmax(logits);
    return ref;
  }

  RefState LoadReference(const std::string& file) {
    RefState ref;
    ref.st = hybrid::ReadLiveState(file);
    const auto h = ref.st.extra.find("logits_hash");
    const auto t = ref.st.extra.find("first_token");
    if (h == ref.st.extra.end() || h->second.size() != 8 || t == ref.st.extra.end() || t->second.size() != 4) {
      throw std::runtime_error("the reference file '" + file + "' carries no logits hash / first token (not written by --gate-b-dump)");
    }
    std::memcpy(&ref.logits_hash, h->second.data(), 8);
    std::memcpy(&ref.first_token, t->second.data(), 4);
    return ref;
  }

  CallResult DoCall(const Step& st, Ctx& ctx, const FaultSpec& fs_in, Group* grp, bool host_hop) {
    const bool is_fault_call = ctx.fault_call == ctx.call_index || (ctx.fault_call < 0 && ctx.calls_left == 1);
    const FaultSpec fs = is_fault_call ? fs_in : FaultSpec{};
    std::vector<int32_t> ids;
    std::vector<Model::ImageSpan> spans;
    bool multimodal = false;
    BuildCall(st, &ids, &spans, &multimodal);
    const int64_t n = static_cast<int64_t>(ids.size());
    const int64_t p0 = ctx.fresh ? 0 : tpm->PositionCount();
    const std::string tag = grp != nullptr ? grp->name() + " call " + std::to_string(ctx.call_index) : std::string("control");
    CallResult res;
    res.p0 = p0;
    res.n = n;

    // ---- the reference ----
    std::vector<hybrid::LiveState> rs;
    if (!ctx.fresh && ctx.ranks_moved && r) rs = SnapshotRanks();
    RefState ref;
    if (r) {
      ref = ReferenceCall(ctx, ids, multimodal ? &spans : nullptr, rs.empty() ? nullptr : &rs);
    } else {
      ref = LoadReference(ctx.ref_file);
    }
    ctx.ref = ref;
    if (ctx.calls_left == 1 && r) r->DebugExportFullState(state_file);  // the dump file the G-H2 import path reads
    const int64_t pos = ref.st.scalars.pos;
    if (grp != nullptr) grp->Ok(pos == p0 + n, tag + ": the reference ended at position " + std::to_string(p0 + n) + " (got " + std::to_string(pos) + ")");

    // ---- the hybrid ----
    if (ctx.fresh) {
      ResetStages();
    } else {
      WarmSync(p0, fs, host_hop);
    }
    const bool flag = fs.f == Fault::kTuningFlag;
    if (flag) {
      x->DebugSetTp2Tuning(true);
      y->DebugSetTp2Tuning(true);
    }
    std::vector<float> ylogits;
    try {
      ylogits = StagePrefill(ids, multimodal ? &spans : nullptr, flag);
    } catch (...) {
      x->DebugSetTp2Tuning(false);
      y->DebugSetTp2Tuning(false);
      throw;
    }
    x->DebugSetTp2Tuning(false);
    y->DebugSetTp2Tuning(false);
    res.logits_equal = HashLogits(ylogits) == ref.logits_hash;
    const StageSyncState end = y->StageGetSyncState();
    res.scalars_equal = end.pos == ref.st.scalars.pos && end.started == ref.st.scalars.started && end.mrope_active == ref.st.scalars.mrope_active &&
                        end.mrope_delta == ref.st.scalars.mrope_delta && end.mtp_seed_valid == ref.st.scalars.mtp_seed_valid &&
                        end.mtp_seed == ref.st.scalars.mtp_seed;
    if (grp != nullptr) {
      grp->Ok(res.logits_equal, tag + ": stage Y's logits equal the TP=1 reference's byte for byte");
      grp->Ok(res.scalars_equal, tag + ": stage Y's end scalars (pos, mrope, MTP seed) equal the reference's");
      grp->Ok(x->PositionCount() == end.pos, tag + ": stage X is at the same position");
    }
    const hybrid::ReshardPlan plan = hybrid::ScatterPlan(g, p, p0, p0 + n);
    ExecPlan(plan, fs, /*gather=*/false, host_hop);
    if (fs.f != Fault::kSkipAdopt) {
      StageSyncState adopt = end;
      if (fs.f == Fault::kDropMrope) {
        adopt.mrope_active = false;
        adopt.mrope_delta = 0;
      }
      OnRanks([&](Model& m, int) { m.TpAdoptPrefill(adopt); });
    }
    tracker->AfterPipelined(p0 + n);

    // ---- G-H1 ----
    hybrid::DigestList want[2];
    for (int k = 0; k < 2; ++k) want[k] = hybrid::DigestLiveImage(hybrid::ReshardRef(g, ref.st.image, k), rank_shape[k], pos);
    const std::vector<hybrid::DigestList> got = RankDigests();
    for (int k = 0; k < 2; ++k) {
      res.why[k] = hybrid::DiffDigests(want[k], got[static_cast<size_t>(k)]);
      res.digest_equal[k] = res.why[k].empty();
      if (grp != nullptr) grp->Ok(res.digest_equal[k], tag + ": G-H1 rank " + std::to_string(k) + "'s live-state digest equals ReshardRef(R's state): " + res.why[k]);
    }
    ctx.fresh = false;
    ctx.ranks_moved = false;
    ctx.next_token = ref.first_token;
    ctx.last = res;
    return res;
  }

  // ---- G-H2: decode from the current ranks vs from ranks loaded with the reference image -----------------------------------------------------
  DecodeOutcome DecodeModes(int32_t first_token, int n_tokens) {
    tpm->SaveCheckpoint();
    DecodeOutcome o;
    for (int mode = 0; mode < 2; ++mode) {
      if (mode > 0) tpm->RestoreCheckpoint();
      int32_t tok = first_token;
      while (static_cast<int>(o.tok[mode].size()) < n_tokens) {
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
  void CheckDecode(Ctx& ctx, const std::string& file, Group* grp, ScenarioOut* out) {
    const DecodeOutcome h = DecodeModes(ctx.next_token, kDecodeTokens);
    ZeroRanks();
    OnRanks([&](Model& m, int) { m.DebugImportFullState(file); });
    const DecodeOutcome ref = DecodeModes(ctx.next_token, kDecodeTokens);
    out->decode_checked = true;
    for (int mode = 0; mode < 2; ++mode) {
      const char* name = mode == 0 ? "plain" : "--mtp 3";
      out->tokens_equal[mode] = h.tok[mode] == ref.tok[mode];
      out->digests_equal[mode] = hybrid::DiffDigests(h.dig[mode][0], ref.dig[mode][0]).empty() && hybrid::DiffDigests(h.dig[mode][1], ref.dig[mode][1]).empty();
      if (grp != nullptr) {
        grp->Ok(h.tok[mode].size() >= static_cast<size_t>(kDecodeTokens), std::string("G-H2 ") + name + ": decoded " + std::to_string(h.tok[mode].size()) + " tokens");
        grp->Ok(out->tokens_equal[mode], std::string("G-H2 ") + name + ": the " + std::to_string(kDecodeTokens) + "+ greedy tokens decoded at TP=2 after the hybrid prefill equal those decoded from the reference image");
        grp->Ok(out->digests_equal[mode], std::string("G-H2 ") + name + ": both ranks' live state after the decode equals the reference-image run's");
      }
    }
  }

  // ---- a whole scenario -----------------------------------------------------------------------------------------------------------------------------
  void RunScenario(const Scenario& sc, const FaultSpec& fs, Group* grp, ScenarioOut* out, const std::string& ref_file = "") {
    ZeroRanks();
    ResetStages();
    Ctx ctx;
    ctx.ref_file = ref_file;
    for (const Step& st : sc.steps) {
      if (st.kind == StepKind::kCall) ++ctx.calls_left;
    }
    for (const Step& st : sc.steps) {
      switch (st.kind) {
        case StepKind::kCall: {
          out->calls.push_back(DoCall(st, ctx, fs, grp, sc.host_hop));
          ++ctx.call_index;
          --ctx.calls_left;
          break;
        }
        case StepKind::kDecode: {
          const int64_t before = tpm->PositionCount();
          std::vector<int32_t> toks;
          int32_t tok = ctx.next_token;
          while (static_cast<int>(toks.size()) < st.n) {
            if (!st.mtp) {
              tok = tpm->DecodeStepGreedy(tok);
              toks.push_back(tok);
            } else {
              const std::vector<int32_t> round = tpm->DecodeStepMtpGreedy(tok, kMtpK);
              toks.insert(toks.end(), round.begin(), round.end());
              tok = round.back();
            }
          }
          if (st.same_as_prev && grp != nullptr) {
            grp->Ok(!ctx.decodes.empty() && toks == ctx.decodes.back(), "the tokens decoded after the restore equal the ones decoded before it");
          }
          ctx.decodes.push_back(toks);
          ctx.next_token = tok;
          ctx.ranks_moved = true;
          tracker->TpOnly(before);
          break;
        }
        case StepKind::kSave:
          tpm->SaveCheckpoint();
          ctx.saved_digest = RankDigests();
          ctx.saved_pos = tpm->PositionCount();
          ctx.saved_token = ctx.next_token;
          break;
        case StepKind::kRestore: {
          tpm->RestoreCheckpoint();
          tracker->TpRestored();
          const std::vector<hybrid::DigestList> d = RankDigests();
          if (grp != nullptr) {
            grp->Ok(tpm->PositionCount() == ctx.saved_pos, "RestoreCheckpoint returned the ranks to the saved position");
            for (int k = 0; k < 2; ++k) {
              grp->Ok(hybrid::DiffDigests(ctx.saved_digest[static_cast<size_t>(k)], d[static_cast<size_t>(k)]).empty(),
                      "rank " + std::to_string(k) + "'s live-state digest after RestoreCheckpoint equals the one at SaveCheckpoint");
            }
          }
          ctx.next_token = ctx.saved_token;
          ctx.ranks_moved = true;
          break;
        }
      }
    }
    if (sc.check_decode) {
      const std::string file = !ref_file.empty() ? ref_file : state_file;
      CheckDecode(ctx, file, grp, out);
    }
  }
};

// ======================================================================================================================================
// Scenarios
// ======================================================================================================================================
Scenario Cold(int n, bool host_hop = false) {
  Scenario s;
  s.name = "cold" + std::to_string(n);
  s.steps = {Call(n)};
  s.host_hop = host_hop;
  return s;
}
Scenario Multi(const std::string& name, const std::vector<int>& calls, bool host_hop = false) {
  Scenario s;
  s.name = name;
  int salt = 1;
  for (const int c : calls) s.steps.push_back(Call(c, salt++));
  s.host_hop = host_hop;
  return s;
}
// turn 1, `decode` tokens at TP=2 (plain or --mtp 3), turn 2 through the warm gather
Scenario Warm(const std::string& name, int first, int decode, bool mtp, int second, bool host_hop = false) {
  Scenario s;
  s.name = name;
  s.steps = {Call(first, 1), Decode(decode, mtp), Call(second, 2)};
  s.host_hop = host_hop;
  return s;
}
Scenario Checkpoint() {
  Scenario s;
  s.name = "checkpoint";
  // save at the end of the prompt, decode, restore (the digest must be the saved one, the decode repeatable), restore again after a second decode, then a
  // warm hybrid turn from the restored state (the tracker's TpRestored rule: GDN state and seed are gathered again)
  s.steps = {Call(300, 1), Save(), Decode(12, false), Restore(), Decode(12, false, /*same_as_prev=*/true), Restore(), Call(400, 2)};
  return s;
}
std::vector<Scenario> ImageScenarios() {
  std::vector<Scenario> v;
  Scenario a;
  a.name = "image-mid";  // text 100 + image 256 + text 200
  a.steps = {ImageCall(1, 100, 200)};
  a.needs_vision = true;
  v.push_back(a);
  Scenario b;
  b.name = "image-first";  // the image at position 0
  b.steps = {ImageCall(2, 0, 300)};
  b.needs_vision = true;
  v.push_back(b);
  Scenario c;
  c.name = "image-conversation";  // an image prompt, a text continuation (mrope delta carried), TP=2 decode, a text-only multimodal call
  Step mm = Call(270, 9);
  mm.mm = true;
  c.steps = {ImageCall(2, 60, 100), Call(300, 7), Decode(10, false), mm};
  c.needs_vision = true;
  c.host_hop = true;
  v.push_back(c);
  return v;
}

// ---- running a list of scenarios on a rig ------------------------------------------------------------------------------------------------------
void RunScenarioChecked(Rig& rig, const Scenario& sc) {
  const std::string name = rig.cfg.name + "/" + sc.name;
  if (!Only(name)) return;
  if (sc.needs_vision && !rig.VisionOk()) {
    Skip(name, "the container has no vision config (no image rows can be spliced)", /*required=*/false);
    return;
  }
  Group grp(name);
  ScenarioOut out;
  try {
    rig.RunScenario(sc, FaultSpec{}, &grp, &out);
    grp.Done(std::to_string(out.calls.size()) + " call(s), " + (sc.check_decode ? "G-H2 plain + --mtp 3" : "no decode check"));
  } catch (const std::exception& e) {
    grp.Fail(std::string("uncaught exception: ") + e.what());
  }
}

// ---- the negative controls -------------------------------------------------------------------------------------------------------------------------
std::string Why(const CallResult& c) {
  std::string s;
  if (!c.logits_equal) s += "logits differ; ";
  if (!c.scalars_equal) s += "scalars differ; ";
  for (int k = 0; k < 2; ++k) {
    if (!c.digest_equal[k]) s += "rank " + std::to_string(k) + ": " + c.why[k] + "; ";
  }
  return s.empty() ? "nothing changed" : s;
}

void RunControls(Rig& rig, bool full) {
  const std::string base = rig.cfg.name + "/nc";
  if (!Only(base)) return;
  const auto run = [&](const Scenario& sc, const FaultSpec& fs, ScenarioOut* out, const char* what) {
    try {
      rig.RunScenario(sc, fs, nullptr, out);
      return !out->calls.empty();
    } catch (const std::exception& e) {
      ++g_fails;
      std::fprintf(stderr, "FAIL %s: the control run (%s) threw: %s\n", base.c_str(), what, e.what());
      return false;
    }
  };

  // -- scatter-time faults on a cold 300-row call (every op through the host hop) --
  Scenario cold;
  cold.name = "nc-cold300";
  cold.steps = {Call(300)};
  cold.check_decode = false;
  cold.host_hop = true;
  struct ColdCase {
    int nc;
    Fault f;
    const char* what;
  };
  const ColdCase cold_cases[] = {
      {1, Fault::kSwapKvHeads, "the KV head halves swapped"},
      {2, Fault::kDropConvSegment, "a conv segment dropped"},
      {3, Fault::kSkipMtp, "the MTP half skipped"},
      {5, Fault::kKvOneBlockOff, "a KV block off by one"},
      {6, Fault::kWrongRankGdn, "the wrong rank's v-heads (GDN rank mapping)"},
  };
  for (const ColdCase& cc : cold_cases) {
    for (int rank = 0; rank < 2; ++rank) {
      if (!full && rank != cc.nc % 2) continue;  // the short list: one rank per control
      ScenarioOut out;
      if (!run(cold, FaultSpec{cc.f, rank}, &out, cc.what)) continue;
      const CallResult& c = out.Last();
      const bool detected = !c.digest_equal[rank] && c.digest_equal[1 - rank];
      NegativeControl(cc.nc, detected, std::string(cc.what) + " on rank " + std::to_string(rank),
                      detected ? "that rank's live-state digest changed, the other rank's did not" : Why(c));
    }
  }
  {  // NC8: the scalars are not adopted
    ScenarioOut out;
    if (run(cold, FaultSpec{Fault::kSkipAdopt, 0}, &out, "skip TpAdoptPrefill")) {
      const CallResult& c = out.Last();
      NegativeControl(8, !c.digest_equal[0] && !c.digest_equal[1], "TpAdoptPrefill skipped", "both ranks' digests (position and every row count) changed");
    }
  }

  // -- NC10: the stage prefill under the TP2 tuning flag --
  {
    rig.ZeroRanks();
    rig.ResetStages();
    const std::vector<int32_t> ids = Tokens(300, 1);
    const bool refused = Throws<std::logic_error>([&] { (void)rig.StagePrefill(ids, nullptr, /*tp_flag=*/true); });
    NegativeControl(10, refused, "the stage prefill run with the thread's TP2 tuning flag true", "Model::CheckTuningScope refuses it (std::logic_error) before any state changes");
    ScenarioOut out;
    if (run(cold, FaultSpec{Fault::kTuningFlag, 0}, &out, "tuning flag")) {
      const CallResult& c = out.Last();
      const bool flipped = !CallOk(c);
      if (flipped) {
        NegativeControl(10, true, "the stage prefill driven under the TP2 tuning flag (DebugSetTp2Tuning bypasses the guard)", Why(c));
      } else if (rig.cfg.nc10_must_flip) {
        NegativeControl(10, false, "the stage prefill driven under the TP2 tuning flag", "on the production container the digest must change on a rank-equal shape");
      } else {
        std::fprintf(stderr, "[INFO] NC10 digest part: on %s the TP2 tuning table picks the same bytes (no change); the production container is the one that must flip\n",
                     rig.cfg.name.c_str());
      }
    }
  }

  // -- gather-time faults and the MTP block on a warm turn (turn 1 decoded with --mtp 3 so the shards' MTP rows differ from the stage's) --
  Scenario warm = Warm("nc-warm", 300, 20, /*mtp=*/true, 517, /*host_hop=*/true);
  warm.check_decode = false;
  struct WarmCase {
    int nc;
    Fault f;
    const char* what;
  };
  const WarmCase warm_cases[] = {
      {7, Fault::kSkipGather, "a stale mirror (nothing gathered)"},
      {7, Fault::kSkipGatherGdn, "a stale mirror (GDN state not gathered)"},
      {7, Fault::kSkipGatherKv, "a stale mirror (KV rows not gathered)"},
      {9, Fault::kStaleSeed, "the TP -> Y MTP seed sync skipped"},
      {12, Fault::kMtpFromBlock0, "the MTP KV copied from block 0 instead of the primed range"},
  };
  {
    ScenarioOut out;  // the unfaulted warm turn: the controls mean nothing unless it is right
    if (run(warm, FaultSpec{}, &out, "warm turn without a fault")) {
      if (!CallOk(out.Last()) || !out.EarlierCallsOk()) {
        ++g_fails;
        std::fprintf(stderr, "FAIL %s: the unfaulted warm turn of the controls is not identical to the reference (%s)\n", base.c_str(), Why(out.Last()).c_str());
      }
    }
  }
  for (const WarmCase& wc : warm_cases) {
    if (!full && wc.f != Fault::kSkipGather && wc.f != Fault::kStaleSeed) continue;
    ScenarioOut out;
    if (!run(warm, FaultSpec{wc.f, 0}, &out, wc.what) || !out.EarlierCallsOk()) continue;
    const bool detected = !CallOk(out.Last());
    NegativeControl(wc.nc, detected, wc.what, detected ? Why(out.Last()) : "the second turn is identical to the reference");
  }

  // -- NC8b: the mrope delta lost at the adoption (an image prompt: the prefill digests do not contain the delta, the TP=2 decode does) --
  if (rig.VisionOk()) {
    Scenario sc = ImageScenarios()[0];
    sc.name = "nc-image-mid";
    ScenarioOut out;
    if (run(sc, FaultSpec{Fault::kDropMrope, 0}, &out, "mrope dropped")) {
      const bool differ = !out.tokens_equal[0] || !out.tokens_equal[1] || !out.digests_equal[0] || !out.digests_equal[1];
      NegativeControl(8, out.decode_checked && CallOk(out.Last()) && differ, "the mrope delta dropped at the adoption (image prompt)",
                      "the prefill digests still agree (the delta is not in them) but the TP=2 decode's tokens / state differ from the reference-image run's");
    }
  } else if (full) {
    Skip(base + "/nc8b", "no vision config in this container", false);
  }
}

// ---- the configurations ---------------------------------------------------------------------------------------------------------------------
Cfg L4Cfg(const std::string& name, Layout layout, int gdn_wo, bool vision) {
  Cfg c;
  c.name = name;
  c.path = ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
  c.layout = layout;
  c.layers = 4;
  c.split = 2;
  c.gdn_wo = gdn_wo;
  c.vision = vision;
  return c;
}

void RunConfig(const Cfg& cfg, const std::vector<Scenario>& scenarios, bool controls, bool full_controls) {
  bool any = Only(cfg.name + "/nc") && controls;
  for (const Scenario& sc : scenarios) any = any || Only(cfg.name + "/" + sc.name);
  if (!any) return;
  if (!FileExists(cfg.path)) {
    Skip(cfg.name, cfg.path + " not found", /*required=*/false);
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
  for (const Scenario& sc : scenarios) RunScenarioChecked(*rig, sc);
  if (controls) {
    try {
      RunControls(*rig, full_controls);
    } catch (const std::exception& e) {
      ++g_fails;
      std::fprintf(stderr, "FAIL %s/nc: uncaught exception: %s\n", cfg.name.c_str(), e.what());
    }
  }
}

// ======================================================================================================================================
// Gate C: the DFlash tail, drafters only
// ======================================================================================================================================
std::vector<uint16_t> RandomFeatures(int64_t rows, int64_t cols, uint64_t seed) {
  uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
  std::vector<uint16_t> v(static_cast<size_t>(rows * cols));
  for (uint16_t& x : v) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    const float f = (static_cast<float>((s >> 33) % 2001) - 1000.0f) / 2000.0f;  // [-0.5, 0.5]
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    x = static_cast<uint16_t>(bits >> 16);
  }
  return v;
}

struct DrafterWindow {
  int64_t lo = 0, injected = 0;
  std::vector<uint64_t> k, v;
  bool operator==(const DrafterWindow& o) const { return lo == o.lo && injected == o.injected && k == o.k && v == o.v; }
};
DrafterWindow WindowOf(const r4dx::model::DflashDraft& d, r4dx::core::Stream& stream) {
  const r4dx::model::Dflash2Config& cfg = d.Config();
  DrafterWindow w;
  int64_t hi = 0;
  hybrid::DflashWindow(d.InjectedCount(), d.ValidFrom(), cfg.attention.sliding_window, &w.lo, &hi);
  w.injected = d.InjectedCount();
  const int64_t row_elems = cfg.attention.head_count_kv * cfg.attention.key_length;
  for (int64_t l = 0; l < cfg.block_count; ++l) {
    const std::vector<uint16_t> k = d.DebugStoreK(stream, l, w.lo, hi - w.lo), v = d.DebugStoreV(stream, l, w.lo, hi - w.lo);
    w.k.push_back(hybrid::DigestDflashRows(k.data(), hi - w.lo, row_elems, w.lo));
    w.v.push_back(hybrid::DigestDflashRows(v.data(), hi - w.lo, row_elems, w.lo));
  }
  return w;
}

// The captured target features have the capture's columns (3 layers of the 4-layer container); the drafter wants 5 target layers' worth: the blocks of
// `hidden` columns are repeated cyclically (the content only has to be real activations, the layer identity does not matter to the injection).
std::vector<uint16_t> ExpandCols(const uint16_t* src, int64_t rows, int64_t src_cols, int64_t dst_cols, int64_t hid) {
  std::vector<uint16_t> out(static_cast<size_t>(rows * dst_cols));
  const int64_t sb = src_cols / hid, db = dst_cols / hid;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < db; ++b) {
      std::memcpy(out.data() + static_cast<size_t>(r * dst_cols + b * hid), src + static_cast<size_t>(r * src_cols + (b % sb) * hid), static_cast<size_t>(hid) * sizeof(uint16_t));
    }
  }
  return out;
}

void RunGateC() {
  const std::string name = "gate-c/dflash-tail";
  if (!Only(name)) return;
  const std::string drafter_path = ProductionDrafterPath();
  const std::string l4 = ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
  if (!FileExists(drafter_path)) {
    Skip(name, drafter_path + " not found: NC4 (skip the DFlash tail) and NC11 (drop the image rope rows) are not exercised", false);
    return;
  }
  if (!FileExists(l4)) {
    Skip(name, l4 + " not found (the captured features come from it)", false);
    return;
  }
  ++g_configs;
  Group grp(name);
  try {
    // 1. the captured features: layers {0, 1, 3} of the 4-layer target, every row of two calls (an image inside the first when the container has a
    //    vision config), the tail of each call armed through StageArmDflashTail on a second run, the rope rows out of PrefillMultimodal.
    ModelOptions o;
    o.container_path = l4;
    o.layout = Layout::kBf16;
    o.max_ctx = 4096;
    o.layer_limit = 4;
    o.vision = ModelOptions::VisionMode::kAuto;
    Model m = Model::Load(o);
    m.AttachDflashFeatureCapture({0, 1, 3});
    const int64_t src_cols = m.DflashFeatureCols();
    const int64_t hid = m.Config().hidden_size;
    constexpr int64_t n1 = 2600, n2 = 900, n = n1 + n2;
    const bool vision = m.VisionSpliceEnabled();
    std::vector<int32_t> ids1;
    std::vector<Model::ImageSpan> spans1;
    std::unique_ptr<ImageRows> img;
    if (vision) {
      img = std::make_unique<ImageRows>(MakeImageRows(m.GetContainer().VisionCfg().spatial_merge_size, 16, hid, 1));
      const std::vector<int32_t> a = Tokens(1000, 3), b = Tokens(static_cast<int>(n1 - 1000 - img->tokens), 5);
      ids1 = a;
      Model::ImageSpan sp;
      sp.offset = static_cast<int64_t>(ids1.size());
      sp.tokens = img->tokens;
      sp.grid.t = 1;
      sp.grid.h = img->side;
      sp.grid.w = img->side;
      sp.embeds = img->dev.data();
      ids1.insert(ids1.end(), static_cast<size_t>(img->tokens), static_cast<int32_t>(m.GetContainer().ImageTokenId()));
      ids1.insert(ids1.end(), b.begin(), b.end());
      spans1.push_back(sp);
    } else {
      ids1 = Tokens(static_cast<int>(n1), 3);
    }
    const std::vector<int32_t> ids2 = Tokens(static_cast<int>(n2), 4);
    grp.Ok(static_cast<int64_t>(ids1.size()) == n1, "call 1 has " + std::to_string(n1) + " rows");

    std::vector<uint16_t> all(static_cast<size_t>(n * src_cols), 0);
    std::vector<int32_t> rope3_1, rope3_2;
    m.Reset();
    m.DebugZeroKvState();
    m.SetDflashCaptureObserver([&all, src_cols](const uint16_t* features, int64_t rows, int64_t start) {
      if (hipMemcpy(all.data() + static_cast<size_t>(start * src_cols), features, static_cast<size_t>(rows * src_cols) * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("reference capture: D2H failed");
      }
    });
    (void)m.PrefillMultimodal(ids1, spans1, nullptr, &rope3_1);
    (void)m.PrefillMultimodal(ids2, {}, nullptr, &rope3_2);
    m.ClearDflashCaptureObserver();

    const int64_t s1 = hybrid::DflashTailStart(0, n1), s2 = hybrid::DflashTailStart(n1, n);
    PinnedBuffer<uint16_t> tail1(static_cast<size_t>((n1 - s1 + 64) * src_cols)), tail2(static_cast<size_t>((n - s2 + 64) * src_cols));
    m.Reset();
    m.DebugZeroKvState();
    m.StageArmDflashTail(tail1.data(), n1 - s1 + 64, s1, n1);
    (void)m.PrefillMultimodal(ids1, spans1);
    grp.Ok(m.StageDisarmDflashTail() == n1 - s1, "the armed capture holds the tail [" + std::to_string(s1) + ", " + std::to_string(n1) + ") of call 1");
    m.StageArmDflashTail(tail2.data(), n - s2 + 64, s2, n);
    (void)m.PrefillMultimodal(ids2, {});
    grp.Ok(m.StageDisarmDflashTail() == n - s2, "the armed capture holds the tail [" + std::to_string(s2) + ", " + std::to_string(n) + ") of call 2");
    grp.Ok(std::memcmp(tail1.data(), all.data() + static_cast<size_t>(s1 * src_cols), static_cast<size_t>((n1 - s1) * src_cols) * sizeof(uint16_t)) == 0,
           "call 1's captured tail equals the observer's rows byte for byte");
    grp.Ok(std::memcmp(tail2.data(), all.data() + static_cast<size_t>(s2 * src_cols), static_cast<size_t>((n - s2) * src_cols) * sizeof(uint16_t)) == 0,
           "call 2's captured tail equals the observer's rows byte for byte");
    grp.Ok(hybrid::CheckTailInject(0, n1, s1, n1 - s1).empty() && hybrid::CheckTailInject(n1, n, s2, n - s2).empty(), "the tail plans pass CheckTailInject");

    // The temporal rope rows: whole-call rows for the reference, the tail's slice of them for the hybrid; text-only conversations pass nullptr.
    const bool use_rope = vision;
    std::vector<int32_t> t1, t2, tt1, tt2;
    if (use_rope) {
      t1 = hybrid::TemporalRopeRows(rope3_1, n1, 0, n1);
      t2 = hybrid::TemporalRopeRows(rope3_2, n2, 0, n2);
      tt1 = hybrid::TemporalRopeRows(rope3_1, n1, s1, n1 - s1);
      tt2 = hybrid::TemporalRopeRows(rope3_2, n2, s2 - n1, n - s2);
      grp.Ok(t1[1500] != 1500 && t1[100] == 100, "the captured rope rows carry the image (position 1500 is not 1500, position 100 is)");
    }

    // 2. the drafters, under the TP-thread tuning flag (the rank drafters' numerics)
    r4dx::model::Tp2TuningScope tp_thread(true);
    const r4dx::model::DflashDraftWeights peek = r4dx::model::DflashDraftWeights::Open(drafter_path);
    r4dx::model::DflashDraftOptions dopt;
    dopt.container_path = drafter_path;
    dopt.layout = r4dx::model::LayoutFromName(peek.Config().layout);
    r4dx::model::DflashDraft full = r4dx::model::DflashDraft::Load(dopt);
    r4dx::model::DflashDraft tail = r4dx::model::DflashDraft::Load(dopt);
    r4dx::core::Stream stream;
    r4dx::core::Arena arena;
    arena.Reserve(256ull << 20);
    const int64_t cols = full.FeatureCols();
    grp.Ok(cols % hid == 0, "the drafter's feature width is a whole number of hidden-wide blocks");
    DeviceBuffer<uint16_t> staging(static_cast<size_t>(64 * cols));
    const std::vector<uint16_t> poison = RandomFeatures(2100, cols, 999);
    const auto prepare = [&](r4dx::model::DflashDraft& d) {  // a poisoned window, then reset: a slot the tail failed to write cannot hide behind an earlier equal write
      d.Reset();
      r4dx::model::InjectFeatureRowsFromHost(d, stream, arena, staging.data(), 64, poison.data(), 2100, 0, nullptr);
      d.Reset();
    };
    const auto inject = [&](r4dx::model::DflashDraft& d, const std::vector<uint16_t>& feat, int64_t rows, int64_t start, const std::vector<int32_t>& rope) {
      r4dx::model::InjectFeatureRowsFromHost(d, stream, arena, staging.data(), 64, feat.data(), rows, start, rope.empty() ? nullptr : rope.data());
    };
    const std::vector<uint16_t> f1 = ExpandCols(all.data(), n1, src_cols, cols, hid), f2 = ExpandCols(all.data() + static_cast<size_t>(n1 * src_cols), n2, src_cols, cols, hid);
    const std::vector<uint16_t> g1 = ExpandCols(tail1.data(), n1 - s1, src_cols, cols, hid), g2 = ExpandCols(tail2.data(), n - s2, src_cols, cols, hid);

    prepare(full);  // the reference: every row of both calls
    inject(full, f1, n1, 0, t1);
    const DrafterWindow ref1 = WindowOf(full, stream);
    inject(full, f2, n2, n1, t2);
    const DrafterWindow ref2 = WindowOf(full, stream);
    grp.Ok(ref1.injected == n1 && ref2.injected == n, "the fully fed drafter's frontier is n after each call");

    prepare(tail);  // the hybrid: only the tail of each call
    inject(tail, g1, n1 - s1, s1, tt1);
    const DrafterWindow w1 = WindowOf(tail, stream);
    grp.Ok(w1 == ref1, "Gate C call 1: the tail-fed drafter ring (K and V of every draft layer over its window, frontier, lower bound) equals the fully fed one");
    inject(tail, g2, n - s2, s2, tt2);
    const DrafterWindow w2 = WindowOf(tail, stream);
    grp.Ok(w2 == ref2, "Gate C call 2 (warm): the tail-fed ring equals the fully fed one");

    // negative controls
    prepare(tail);
    inject(tail, g1, n1 - s1, s1, tt1);
    NegativeControl(4, !(WindowOf(tail, stream) == ref2), "the DFlash tail of call 2 skipped", "the drafter's frontier and window differ from the fully fed ring's");
    if (use_rope) {
      prepare(tail);
      inject(tail, g1, n1 - s1, s1, std::vector<int32_t>());  // the rope rows dropped: the drafter ropes at start + t (+ its delta 0)
      NegativeControl(11, !(WindowOf(tail, stream) == ref1), "the image rope rows dropped from the tail injection", "the ring differs (the image run sits inside the tail window)");
    } else {
      std::fprintf(stderr, "[INFO] NC11 is not exercised: the 4-layer container has no vision config, so there are no image rope rows\n");
    }
    // informational: does the thread's tuning flag change the drafter's bytes at these shapes?
    {
      prepare(full);
      inject(full, f1, n1, 0, t1);
      const DrafterWindow with_flag = WindowOf(full, stream);
      DrafterWindow without;
      {
        r4dx::model::Tp2TuningScope tp1(false);
        prepare(full);
        inject(full, f1, n1, 0, t1);
        without = WindowOf(full, stream);
      }
      std::fprintf(stderr, "[INFO] gate-c: the drafter's injected window %s with the TP2 tuning flag vs without\n", with_flag == without ? "does NOT change" : "CHANGES");
    }
    grp.Done(std::string(vision ? "with" : "without") + " an image run in call 1; Model::TpInjectDflashTail itself runs in test_dflash_tail part c");
  } catch (const std::exception& e) {
    grp.Fail(std::string("uncaught exception: ") + e.what());
  }
}

// ======================================================================================================================================
// Gate B: full depth, the reference dumped by an earlier process
// ======================================================================================================================================
struct GateBCase {
  const char* name;
  int rows;
};
const GateBCase kGateB[] = {{"cold17", 17}, {"cold300", 300}, {"cold1000", 1000}, {"cold2125", 2048 + 77}};

Cfg GateBCfg(int64_t layers, int64_t split) {
  Cfg c;
  c.name = "gate-b";
  c.path = ProductionTargetPath();
  c.layout = r4dx::model::LayoutFromName(ProductionLayoutName());
  c.layers = layers;
  c.split = split;
  return c;
}
std::string GateBFile(const std::string& dir, const char* name) { return (std::filesystem::path(dir) / (std::string("ghb_") + name + ".state")).string(); }

int RunGateBDump(const std::string& dir, int64_t layers, int64_t split) {
  Cfg c = GateBCfg(layers, split);
  if (!FileExists(c.path)) return SkipMissing(c.path);
  std::filesystem::create_directories(dir);
  c.want_hybrid = false;
  Rig rig(c);  // loads the TP=1 reference only
  Model& r = *rig.r;
  Group grp("gate-b/dump");
  for (const GateBCase& gc : kGateB) {
    r.Reset();
    r.DebugZeroKvState();
    const std::vector<float> logits = r.Prefill(Tokens(gc.rows, 1));
    hybrid::LiveState st = r.DebugExportLiveState();
    const uint64_t h = HashLogits(logits);
    const int32_t first = Argmax(logits);
    st.extra["logits_hash"].assign(reinterpret_cast<const uint8_t*>(&h), reinterpret_cast<const uint8_t*>(&h) + 8);
    st.extra["first_token"].assign(reinterpret_cast<const uint8_t*>(&first), reinterpret_cast<const uint8_t*>(&first) + 4);
    hybrid::WriteLiveState(GateBFile(dir, gc.name), st);
    grp.Ok(st.scalars.pos == gc.rows, std::string(gc.name) + ": dumped a state at position " + std::to_string(gc.rows));
  }
  // the host-gather vs device-gather embedding bit-equality: the device mirror's rows are the host copy's rows, byte for byte
  {
    const Container& ct = r.GetContainer();
    const int64_t vocab = r.GlobalConfig().vocab_size, hid = r.Config().hidden_size;
    grp.Ok(ct.EmbedTokensDeviceResident(), "the reference process mirrors the embedding on the device");
    bool same = true;
    std::vector<uint16_t> row(static_cast<size_t>(hid));
    for (int64_t k = 0; k < 997 && same; ++k) {
      const int64_t id = (k * 7919 + 13) % vocab;
      if (hipMemcpy(row.data(), ct.EmbedTokensDevice() + id * hid, row.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess) same = false;
      same = same && std::memcmp(row.data(), ct.EmbedTokensHost() + id * hid, row.size() * sizeof(uint16_t)) == 0;
    }
    grp.Ok(same, "997 sampled embedding rows: device mirror == pinned host copy, byte for byte");
  }
  grp.Done("state files in " + dir);
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_emulate_identity: gate-b dump FAILED\n");
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_emulate_identity: gate-b dump DONE (%zu states in %s)\n", sizeof(kGateB) / sizeof(kGateB[0]), dir.c_str());
  return 0;
}

int RunGateBVerify(const std::string& dir, int64_t layers, int64_t split) {
  Cfg c = GateBCfg(layers, split);
  if (!FileExists(c.path)) return SkipMissing(c.path);
  for (const GateBCase& gc : kGateB) {
    if (!FileExists(GateBFile(dir, gc.name))) {
      std::fprintf(stderr, "FAIL gate-b: %s is missing -- run --gate-b-dump %s first\n", GateBFile(dir, gc.name).c_str(), dir.c_str());
      return 1;
    }
  }
  c.gate_b = true;
  c.want_r = false;
  ++g_configs;
  Rig rig(c);
  for (const GateBCase& gc : kGateB) {
    Scenario sc;
    sc.name = gc.name;
    sc.steps = {Call(gc.rows, 1)};
    Group grp("gate-b/" + std::string(gc.name));
    ScenarioOut out;
    try {
      rig.RunScenario(sc, FaultSpec{}, &grp, &out, GateBFile(dir, gc.name));
      grp.Done("host-gather embedding, reference from the dump");
    } catch (const std::exception& e) {
      grp.Fail(std::string("uncaught exception: ") + e.what());
    }
  }
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_emulate_identity: gate-b %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_emulate_identity: PASS (%d configurations)\n", g_configs);
  return 0;
}

}  // namespace

static int RunTest(int argc, char** argv) {
  std::string gate_b, gate_b_dir;
  int64_t layers = -1, split = 32;
  if (const char* e = std::getenv("R4DX_HYBRID_GATE_B")) gate_b = e;
  if (const char* e = std::getenv("R4DX_HYBRID_GATE_B_DIR")) gate_b_dir = e;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
      return argv[++i];
    };
    if (a == "--gate-b-dump") {
      gate_b = "dump";
      gate_b_dir = next();
    } else if (a == "--gate-b") {
      gate_b = "verify";
      gate_b_dir = next();
    } else if (a == "--layers") {
      layers = std::stoll(next());
    } else if (a == "--split") {
      split = std::stoll(next());
    } else {
      throw std::invalid_argument("unknown argument " + a);
    }
  }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::fprintf(stderr, "[SKIP] no HIP device\n");
    return kSkipReturnCode;
  }
  if (!gate_b.empty()) {
    if (gate_b_dir.empty()) throw std::invalid_argument("Gate B needs a directory (--gate-b <dir> / R4DX_HYBRID_GATE_B_DIR)");
    if (gate_b == "dump") return RunGateBDump(gate_b_dir, layers, split);
    if (gate_b == "verify") return RunGateBVerify(gate_b_dir, layers, split);
    throw std::invalid_argument("R4DX_HYBRID_GATE_B must be dump or verify");
  }

  const std::string l4 = ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
  const char* real = ProductionTargetPath();
  if (!FileExists(l4) && !FileExists(real) && !FileExists(ProductionDrafterPath())) return SkipMissing(l4);

  // ---- Gate A: the 4-layer selftest container ----
  {
    std::vector<Scenario> sc;
    for (const int n : {1, 17, 256, 300, 1000, 2048 + 77}) sc.push_back(Cold(n, /*host_hop=*/n == 17 || n == 1000));
    sc.push_back(Multi("multi300+333", {300, 333}));
    sc.push_back(Multi("multi64+511", {64, 511}, true));
    sc.push_back(Multi("multi257+1", {257, 1}));
    sc.push_back(Multi("multi1+255+257", {1, 255, 257}));
    sc.push_back(Warm("warm-plain", 300, 20, /*mtp=*/false, 517));
    sc.push_back(Warm("warm-mtp3", 300, 20, /*mtp=*/true, 517));
    sc.push_back(Warm("warm-mtp3-unaligned", 700, 17, /*mtp=*/true, 300, /*host_hop=*/true));
    sc.push_back(Checkpoint());
    for (const Scenario& s : ImageScenarios()) sc.push_back(s);
    RunConfig(L4Cfg("l4/bf16", Layout::kBf16, /*gdn_wo=*/-1, /*vision=*/true), sc, /*controls=*/true, /*full_controls=*/true);
  }
  {  // write-once GDN state: the ranks' live state sits behind a pending prefix after --mtp 3 rounds (ReshardCollapse before every gather)
    std::vector<Scenario> sc = {Cold(300), Multi("multi300+333", {300, 333}), Warm("warm-mtp3", 300, 20, true, 517), Warm("warm-plain", 300, 20, false, 517, true)};
    RunConfig(L4Cfg("l4/bf16-writeonce", Layout::kBf16, /*gdn_wo=*/1, /*vision=*/false), sc, /*controls=*/false, false);
  }
  {
    std::vector<Scenario> sc = {Cold(17, true), Cold(300), Cold(2048 + 77), Warm("warm-mtp3", 300, 20, true, 517)};
    RunConfig(L4Cfg("l4/w4a16", Layout::kW4a16, -1, false), sc, /*controls=*/false, false);
  }

  // ---- Gate A: the real container, 16 layers, k = 8 ----
  {
    Cfg c;
    c.name = "real16";
    c.path = real;
    c.layout = r4dx::model::LayoutFromName(ProductionLayoutName());
    c.layers = 16;
    c.split = 8;
    c.nc10_must_flip = true;
    std::vector<Scenario> sc = {Cold(17, true),
                                Cold(300),
                                Cold(1000),
                                Cold(2048 + 77, true),
                                Multi("multi300+333", {300, 333}),
                                Warm("warm-plain", 300, 20, false, 517),
                                Warm("warm-mtp3", 300, 20, true, 517, true),
                                Checkpoint()};
    RunConfig(c, sc, /*controls=*/true, /*full_controls=*/false);
  }

  // ---- Gate C ----
  RunGateC();

  // ---- summary ----
  {
    std::string done, missing;
    for (const int nc : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) (g_nc_done.count(nc) != 0 ? done : missing) += std::to_string(nc) + " ";
    std::fprintf(stderr, "[hybrid-emu] negative controls detected: %s\n", done.empty() ? "(none)" : done.c_str());
    if (!missing.empty()) std::fprintf(stderr, "[hybrid-emu] negative controls NOT exercised in this run: %s(filtered out, or the container they need is missing)\n", missing.c_str());
  }
  if (g_configs == 0) return SkipMissing(l4);
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_emulate_identity: %d FAILED (%d configurations)\n", g_fails, g_configs);
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_emulate_identity: PASS (%d configurations)\n", g_configs);
  return 0;
}

int main(int argc, char** argv) { return r4dx_test::RunGuardedMain("test_hybrid_emulate_identity", [&] { return RunTest(argc, argv); }); }
