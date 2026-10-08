// tests/model/test_stage_load.cpp -- the device half of the hybrid mode's P0 prerequisites (docs/pp-tp2-hybrid.md 9): a stage-only
// Model (ModelOptions::stage_only) next to a full Model on ONE card, the way a rank worker will hold a stage Model next to its TP
// rank Model. Device 1, the 4-layer MTP container (bf16); SKIP 77 without it. The CPU policy is test_stage_load_cpu.
//
//   * back stage Y (split 2 of 4): hole-prefix container -- NumLoadedLayers() stays 4, layers 0 and 1 are holes (Layer(i) throws,
//     LayerLoaded false, no GDN state), the head and the MTP head are whole, the KV cache has exactly S = 1024 tokens, a monolithic
//     Prefill is refused by the range assertion at its entry (Prefill / RunChunk check before the window collapse, the prologue's id
//     upload and embedding gather; the test can observe only the position, not at_prefill_end_), PpAttach as stage A is refused (its range holds holes);
//   * front stage X (split 2): layers 0, 1 full + layer 2 as its input_layernorm only (LayerInputNorm works, Layer(2) throws), no
//     lm_head / MTP / vision, no GDN state for layer 2 (the hand-off wire is two layers' worth, the full Model's three), a monolithic
//     Prefill and PpAttach as stage B are refused, a drafter target at the split layer is refused;
//   * the embedding: both stages take the process's shared pinned host copy and BORROW the full Model's device mirror (same
//     pointer), and use less VRAM than a full Model; the lease OWNS the mirror, so destroying the uploading Model first is fine (the
//     stage still reads it);
//   * StageSetSyncState on both roles and TpAdoptPrefill on a rank: pos / started / mrope and the MTP seed travel, a wrong-width seed
//     is refused before anything is written, a stale `mtp_seed_valid` on the stage is overwritten by an invalid one, stage A (no MTP
//     head) ignores the seed, a stage and a pos-0 state cannot be adopted, an adopted rank can checkpoint (at_prefill_end_);
//   * the stage-KV check: CheckStageKv(S) holds for S and fails for another S, on Y and on a full Model.
// Written, NOT run by its author (CPU-only session).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "pp_channel.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Container;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::StageSyncState;
namespace stage = r4dx::model::stage;

namespace {

const char* kL4Container = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");
constexpr int64_t kSplit = 2;
constexpr int64_t kStageCtx = 1024;  // S: a whole number of 16-token blocks

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

// Runs fn and returns true iff it threw an exception of type E.
template <class E>
bool Throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

size_t FreeBytes() {
  size_t f = 0, t = 0;
  (void)hipMemGetInfo(&f, &t);
  return f;
}

std::vector<int32_t> Tokens(int n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 200 + (i * 53 + (i / 7) * 3) % 5000;
  return ids;
}

// A stage channel over host memory: PpAttach needs one; nothing is ever published through it here.
struct DummyChannel {
  std::vector<std::vector<uint8_t>> mem;
  std::unique_ptr<r4dx::model::pp::StageChannel> ch;
  DummyChannel() {
    std::vector<uint8_t*> ptrs;
    mem.assign(3, std::vector<uint8_t>(4096));
    for (auto& m : mem) ptrs.push_back(m.data());
    ch = std::make_unique<r4dx::model::pp::StageChannel>(ptrs, 4096);
  }
};

Model::PpStageSetup Setup(Model::PpRole role, int64_t split, DummyChannel& c) {
  Model::PpStageSetup s;
  s.role = role;
  s.split = split;
  s.channel = c.ch.get();
  s.timeout_ms = 1000;
  return s;
}

bool SameState(const StageSyncState& a, const StageSyncState& b) {
  return a.pos == b.pos && a.started == b.started && a.mrope_active == b.mrope_active && a.mrope_delta == b.mrope_delta &&
         a.mtp_seed_valid == b.mtp_seed_valid && a.mtp_seed == b.mtp_seed;
}

}  // namespace

static int RunTest() {
  if (!FileExists(kL4Container)) return SkipMissing(kL4Container);
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::fprintf(stderr, "[SKIP] no HIP device\n");
    return kSkipReturnCode;
  }

  ModelOptions base;
  base.container_path = kL4Container;
  base.layout = Layout::kBf16;
  base.max_ctx = 2048;
  base.layer_limit = 4;
  base.mtp_draft_k = 3;
  base.prompt_checkpoint = true;

  // ---- the "rank": a full Model that owns the embedding mirror ----------------------------------------------------------------
  const size_t free0 = FreeBytes();
  auto full = std::make_unique<Model>(Model::Load(base));
  const size_t free1 = FreeBytes();
  const int64_t hidden = full->Config().hidden_size;
  auto lease = full->GetContainer().EmbedMirrorLeaseHandle();
  Check(lease != nullptr && full->GetContainer().EmbedTokensDeviceResident(), "the full Model mirrors the embedding on device and offers a lease");
  const auto host = Container::LoadEmbedTokensHost(kL4Container);
  Check(full->GetContainer().AllLayersLoaded() && full->GetContainer().LayerLoaded(0) && full->GetContainer().LayerLoaded(3), "a full Model has every layer resident");
  Check(full->GetContainer().HasLmHead() && full->GetContainer().HasMtp(), "a full Model has its heads");

  // ---- back stage Y ---------------------------------------------------------------------------------------------------------
  ModelOptions yo = base;
  yo.max_ctx = kStageCtx;
  yo.stage_only.role = stage::Role::kBack;
  yo.stage_only.split = kSplit;
  yo.stage_only.shared_embed_host = host;
  yo.stage_only.borrowed_embed_mirror = lease;
  auto y = std::make_unique<Model>(Model::Load(yo));
  const size_t free2 = FreeBytes();
  {
    const Container& c = y->GetContainer();
    Check(c.NumLoadedLayers() == 4, "Y: NumLoadedLayers() stays the full count (layers_ stays globally indexed)");
    Check(!c.LayerLoaded(0) && !c.LayerLoaded(1) && c.LayerLoaded(2) && c.LayerLoaded(3), "Y: layers 0, 1 are holes, 2, 3 resident");
    Check(!c.AllLayersLoaded(), "Y: not a complete container");
    Check(Throws<std::logic_error>([&] { (void)c.Layer(0); }) && Throws<std::logic_error>([&] { (void)c.LayerInputNorm(1); }),
          "Y: Layer(hole) and LayerInputNorm(hole) throw");
    Check(c.HasLmHead() && c.HasMtp() && c.LmHead().N == full->GetContainer().LmHead().N, "Y: the full lm_head and the full MTP head");
    Check(!c.HasVision() && y->VisionSpliceEnabled() == full->VisionSpliceEnabled(), "Y: no vision tower, the same splice capability as the full Model (geometry only)");
    Check(c.EmbedTokensDevice() == full->GetContainer().EmbedTokensDevice() && c.EmbedTokensDeviceResident(), "Y: borrows the full Model's device mirror (same pointer)");
    Check(c.EmbedTokensHost() == host->data(), "Y: uses the shared pinned host embedding");
    Check(y->StageOnlyRole() == stage::Role::kBack && y->StageOnlySplit() == kSplit, "Y: role and split are recorded");
    Check(y->PpGdnWireBytes(4) > 0 && y->PpGdnWireBytes(2) == 0, "Y: GDN state only for the layers above the split (none below)");
    Check(y->PpAttnLayers(4).size() == 1, "Y: the one attention layer has its KV cache");
  }
  Check(y->StageKvCapacityTokens() == kStageCtx, "Y: the KV cache holds exactly S tokens (max_ctx = S)");
  y->CheckStageKv(kStageCtx);
  Check(Throws<std::logic_error>([&] { y->CheckStageKv(kStageCtx + 16); }), "Y: CheckStageKv refuses another S");
  full->CheckStageKv(base.max_ctx);
  Check(Throws<std::logic_error>([&] { full->CheckStageKv(kStageCtx); }), "a full Model's caches are not S = 1024 either");
  Check(Throws<std::logic_error>([&] { (void)y->Prefill(Tokens(10)); }), "Y: a monolithic Prefill is refused by the range assertion");
  Check(y->PositionCount() == 0, "Y: ... before it changed the position");
  Check(Throws<std::logic_error>([&] { (void)y->DecodeStepGreedy(5); }), "Y: a decode step is refused too");
  {
    DummyChannel ch;
    Check(Throws<std::logic_error>([&] { y->PpAttach(Setup(Model::PpRole::kStageA, kSplit, ch)); }), "Y: PpAttach as stage A is refused (holes in [0, split))");
    y->PpAttach(Setup(Model::PpRole::kStageB, kSplit, ch));
    Check(y->PpStageRole() == Model::PpRole::kStageB, "Y: PpAttach as stage B works");
    y->PpDetach();
  }
  Check(y->GetContainer().EmbedMirrorLeaseHandle() != nullptr, "Y offers the borrowed lease on");

  // ---- front stage X ---------------------------------------------------------------------------------------------------------
  ModelOptions xo = base;
  xo.layer_limit = kSplit + 1;
  xo.mtp_draft_k = 0;
  xo.prompt_checkpoint = false;
  xo.stage_only.role = stage::Role::kFront;
  xo.stage_only.split = kSplit;
  xo.stage_only.shared_embed_host = host;
  xo.stage_only.borrowed_embed_mirror = lease;
  auto x = std::make_unique<Model>(Model::Load(xo));
  const size_t free3 = FreeBytes();
  {
    const Container& c = x->GetContainer();
    Check(c.NumLoadedLayers() == kSplit + 1, "X: layers [0, k] are in the container");
    Check(c.LayerLoaded(0) && c.LayerLoaded(1) && !c.LayerLoaded(2), "X: layers 0, 1 resident, the split layer is not (norm-only)");
    Check(Throws<std::logic_error>([&] { (void)c.Layer(2); }), "X: Layer(split) throws");
    Check(!c.LayerInputNorm(2).empty() && c.LayerInputNorm(2).size() == static_cast<size_t>(hidden), "X: the split layer's input_layernorm is there");
    Check(!c.HasLmHead() && !c.HasMtp() && !c.HasVision() && x->VisionSpliceEnabled() == full->VisionSpliceEnabled(),
          "X: no lm_head, no MTP head, no vision tower, the full Model's splice capability");
    Check(c.EmbedTokensDevice() == full->GetContainer().EmbedTokensDevice() && c.EmbedTokensHost() == host->data(), "X: shared host copy and borrowed mirror");
    Check(x->PpGdnWireBytes(kSplit + 1) == x->PpGdnWireBytes(kSplit) && x->PpGdnWireBytes(kSplit) > 0,
          "X: no GDN state for the split layer (the wire does not grow past k)");
    Check(full->PpGdnWireBytes(kSplit + 1) > full->PpGdnWireBytes(kSplit), "... where a full Model's does");
  }
  Check(Throws<std::out_of_range>([&] { x->AttachDflashFeatureCapture({1, 2}); }), "X: a drafter target at the split layer is refused");
  x->AttachDflashFeatureCapture({0, 1});
  x->DetachDflashFeatureCapture();
  Check(Throws<std::logic_error>([&] { (void)x->Prefill(Tokens(10)); }), "X: a monolithic Prefill is refused (its range holds the norm-only layer)");
  Check(x->PositionCount() == 0, "X: ... before it changed the position");
  {
    DummyChannel ch;
    Check(Throws<std::logic_error>([&] { x->PpAttach(Setup(Model::PpRole::kStageB, kSplit, ch)); }), "X: PpAttach as stage B is refused");
    x->PpAttach(Setup(Model::PpRole::kStageA, kSplit, ch));
    Check(x->PpStageRole() == Model::PpRole::kStageA, "X: PpAttach as stage A works");
    x->PpDetach();
  }

  // refused option combinations
  {
    ModelOptions bad = xo;
    bad.mtp_draft_k = 3;
    Check(Throws<std::invalid_argument>([&] { (void)Model::Load(bad); }), "a front stage with mtp_draft_k > 0 is refused");
    bad = xo;
    bad.layer_limit = 4;
    Check(Throws<std::invalid_argument>([&] { (void)Model::Load(bad); }), "a front stage with layer_limit != split + 1 is refused");
    bad = yo;
    bad.stage_only.split = 4;
    Check(Throws<std::invalid_argument>([&] { (void)Model::Load(bad); }), "a back stage with nothing above its split is refused");
    bad = yo;
    bad.tp.world = 2;
    Check(Throws<std::invalid_argument>([&] { (void)Model::Load(bad); }), "a stage is a TP=1 Model");
    r4dx::model::ContainerLoadOptions co;
    co.tp_world = 2;
    co.first_layer = 1;
    Check(Throws<std::invalid_argument>([&] { (void)Container::Load(kL4Container, co); }), "the shard loader refuses the stage-only fields");
  }

  // ---- VRAM: the stages are lighter than a full Model, and the borrowed mirror is not uploaded again ----------------------
  const double gib = 1024.0 * 1024.0 * 1024.0;
  const double used_full = static_cast<double>(free0 - free1) / gib, used_y = static_cast<double>(free1 - free2) / gib,
               used_x = static_cast<double>(free2 - free3) / gib;
  std::fprintf(stderr, "  VRAM: full %.2f GiB, back stage %.2f GiB, front stage %.2f GiB (hipMemGetInfo deltas)\n", used_full, used_y, used_x);
  Check(used_y < used_full && used_x < used_full, "both stages use less device memory than a full Model (no second embedding mirror, fewer layers)");

  // ---- the sync state ---------------------------------------------------------------------------------------------------------
  (void)full->Prefill(Tokens(100));
  StageSyncState s = full->StageGetSyncState();
  Check(s.pos == 100 && s.started && s.mtp_seed_valid && static_cast<int64_t>(s.mtp_seed.size()) == hidden, "the rank's state: pos 100, a valid 5120-wide seed");
  {
    DummyChannel ch;
    y->PpAttach(Setup(Model::PpRole::kStageB, kSplit, ch));
    x->PpAttach(Setup(Model::PpRole::kStageA, kSplit, ch));

    y->StageSetSyncState(s);
    Check(SameState(y->StageGetSyncState(), s), "Y: the sync state (scalars + seed) reads back as written");
    Check(y->PositionCount() == 100, "Y: position set");
    StageSyncState stale = s;
    stale.mtp_seed_valid = false;
    stale.mtp_seed.clear();
    y->StageSetSyncState(stale);
    Check(!y->StageGetSyncState().mtp_seed_valid, "Y: an invalid seed in the state clears the stage's stale flag (design 5)");
    y->StageSetSyncState(s);
    StageSyncState narrow = s;
    narrow.pos = 999;
    narrow.mtp_seed.resize(static_cast<size_t>(hidden) - 1);
    Check(Throws<std::invalid_argument>([&] { y->StageSetSyncState(narrow); }), "Y: a wrong-width seed is refused");
    Check(y->PositionCount() == 100, "Y: ... without writing anything");

    x->StageSetSyncState(s);
    const StageSyncState xs = x->StageGetSyncState();
    Check(xs.pos == 100 && xs.started && !xs.mtp_seed_valid && xs.mtp_seed.empty(), "X: pos and started travel, the seed is ignored (no MTP head)");
    x->PpSetSyncState({77, true, false, 0});
    Check(x->PositionCount() == 77, "X: PpSetSyncState (stage A's PpModel path) still works");
    Check(Throws<std::logic_error>([&] { y->PpSetSyncState({1, true, false, 0}); }), "Y: PpSetSyncState stays stage-A only");
    y->PpDetach();
    x->PpDetach();
    Check(Throws<std::logic_error>([&] { y->StageSetSyncState(s); }), "a Model that is not attached as a stage refuses StageSetSyncState");
  }

  // ---- TpAdoptPrefill ---------------------------------------------------------------------------------------------------------
  {
    auto rank = std::make_unique<Model>(Model::Load(base));
    StageSyncState adopt = s;
    adopt.mrope_active = true;
    adopt.mrope_delta = -37;
    rank->TpAdoptPrefill(adopt);
    Check(SameState(rank->StageGetSyncState(), adopt), "TpAdoptPrefill: scalars, mrope and the MTP seed land on the rank");
    Check(rank->PositionCount() == 100 && rank->MropeActive() && rank->MropeDelta() == -37, "TpAdoptPrefill: position and mrope");
    rank->SaveCheckpoint();  // at_prefill_end_ is true, so the prompt checkpoint is allowed
    Check(rank->PositionCount() == 100, "TpAdoptPrefill: a rank that adopted a prefill can checkpoint it");
    StageSyncState nopos;
    Check(Throws<std::logic_error>([&] { rank->TpAdoptPrefill(nopos); }), "TpAdoptPrefill: a state that is not a finished prefill is refused");
    Check(Throws<std::logic_error>([&] { y->TpAdoptPrefill(s); }), "TpAdoptPrefill: refused on a stage-only Model");
    rank.reset();
  }

  // ---- the lease owns the mirror: no destruction order is required -------------------------------------------------------------
  // The owner dies FIRST while the option structs, this test's copy and a stage Container still hold the lease; the stage must
  // keep gathering from live memory (the mirror's first row still equals the shared host copy's).
  Check(lease.use_count() >= 5, "the lease is held by the owner, this test, the two option structs and the two stage Containers");
  y.reset();
  full.reset();
  {
    const uint16_t* dev = x->GetContainer().EmbedTokensDevice();
    std::vector<uint16_t> row(static_cast<size_t>(hidden));
    const hipError_t err = hipMemcpy(row.data(), dev, row.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
    Check(err == hipSuccess && std::memcmp(row.data(), host->data(), row.size() * sizeof(uint16_t)) == 0,
          "the borrowed mirror outlives the Model that uploaded it (the lease owns the buffer)");
  }
  x.reset();
  yo.stage_only.borrowed_embed_mirror.reset();
  xo.stage_only.borrowed_embed_mirror.reset();
  lease.reset();
  std::fprintf(stderr, "  destroyed the owner of the borrowed mirror before its borrowers\n");

  if (g_fails != 0) {
    std::fprintf(stderr, "test_stage_load: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_stage_load: PASS\n");
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_stage_load", RunTest); }
