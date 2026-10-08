// tests/model/test_stage_load_cpu.cpp -- CPU-only checks of the HIP-free policy of the hybrid mode's stage-only loads
// (docs/pp-tp2-hybrid.md 9 "P0 prerequisites"): src/model/stage_load.h (which layers a stage holds, the request validation, the
// "is this layer range resident" check the stage entry points assert, the ModelOptions::stage_only -> container request),
// src/model/embed_mirror_lease.h (the borrowed embedding mirror and its destruction-order rule) and src/model/stage_sync.h (the
// MTP seed rule of StageSetSyncState / TpAdoptPrefill). The device half -- Container's hole-prefix and norm-only layers, the
// state allocation, the sync primitives -- is tests/model/test_stage_load.cpp (GPU). No HIP call; always runs.
//
// Negative controls: a checker that treats norm-only layers as loaded, one that forgets the hole prefix, a lease counter that
// forgets the borrower, and a seed rule that ignores the model's lack of an MTP head -- each is built here and must be caught by
// the oracle, otherwise the oracle proves nothing.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "embed_mirror_lease.h"
#include "stage_load.h"
#include "stage_sync.h"

using namespace r4dx::model;
using namespace r4dx::model::stage;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
bool Contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

// ---- layer kinds ---------------------------------------------------------------------------------------------------------
void Kinds() {
  // Back stage Y of the real model: holes [0, 32), full [32, 64).
  const std::vector<LayerKind> y = LayerKinds(64, 32, -1);
  Check(y.size() == 64, "Y keeps all 64 slots (layers_ stays globally indexed)");
  Check(y[0] == LayerKind::kHole && y[31] == LayerKind::kHole && y[32] == LayerKind::kFull && y[63] == LayerKind::kFull, "Y: holes below k, full from k");
  // Front stage X: layers [0, 32] loaded, 32 norm-only.
  const std::vector<LayerKind> x = LayerKinds(33, 0, 32);
  Check(x.size() == 33 && x[0] == LayerKind::kFull && x[31] == LayerKind::kFull && x[32] == LayerKind::kNormOnly, "X: full below k, norm-only at k");
  const std::vector<LayerKind> all = LayerKinds(64, 0, -1);
  bool every_full = true;
  for (LayerKind k : all) every_full = every_full && k == LayerKind::kFull;
  Check(every_full, "the defaults (first_layer 0, no norm-only) are every layer full: no behaviour change for any other Model");
  Check(LayerKinds(0, 0, -1).empty(), "no layers, no kinds");
}

void RequestValidation() {
  Check(ValidateLayerRequest(64, 0, -1).empty(), "defaults are valid");
  Check(ValidateLayerRequest(64, 32, -1).empty() && ValidateLayerRequest(64, 63, -1).empty(), "first_layer 32 / 63 valid");
  Check(!ValidateLayerRequest(64, 64, -1).empty(), "first_layer == num_layers leaves no layer");
  Check(!ValidateLayerRequest(64, -1, -1).empty(), "negative first_layer");
  Check(ValidateLayerRequest(33, 0, 32).empty(), "norm-only tail at the last loaded layer");
  Check(!ValidateLayerRequest(33, 0, 33).empty(), "norm_only_from == num_layers is not a layer");
  Check(!ValidateLayerRequest(33, 5, 5).empty(), "norm-only from first_layer leaves no full layer");
  Check(!ValidateLayerRequest(33, 0, -2).empty(), "norm_only_from < -1");
  Check(!ValidateLayerRequest(0, 0, -1).empty(), "no layers");
}

// The oracle for the range check: a range is loaded iff it lies inside [first_layer, norm_only_from or N).
bool Oracle(int64_t n, int64_t first_layer, int64_t norm_only_from, int64_t a, int64_t b) {
  const int64_t hi = norm_only_from >= 0 ? norm_only_from : n;
  return a >= 0 && a <= b && b <= n && (a == b || (a >= first_layer && b <= hi));
}

void RangeCheck() {
  const std::vector<LayerKind> y = LayerKinds(64, 32, -1);
  Check(CheckRangeLoaded(y, 32, 64).empty(), "Y runs [32, 64)");
  const std::string hole = CheckRangeLoaded(y, 0, 64);
  Check(!hole.empty() && Contains(hole, "layer 0 ") && Contains(hole, "a hole"), "Y refuses a monolithic [0, 64) and names layer 0 as a hole");
  Check(Contains(CheckRangeLoaded(y, 31, 40), "layer 31 "), "Y refuses a range that starts one layer low and names it");
  const std::vector<LayerKind> x = LayerKinds(33, 0, 32);
  Check(CheckRangeLoaded(x, 0, 32).empty(), "X runs [0, 32)");
  const std::string norm = CheckRangeLoaded(x, 0, 33);
  Check(!norm.empty() && Contains(norm, "layer 32 ") && Contains(norm, "norm-only"), "X refuses a range that includes its norm-only layer and names it");
  Check(!CheckRangeLoaded(x, 0, 34).empty() && !CheckRangeLoaded(x, -1, 3).empty() && !CheckRangeLoaded(x, 5, 3).empty(), "ranges outside the container or reversed are refused");
  Check(CheckRangeLoaded(x, 7, 7).empty(), "an empty range is loaded");

  // Exhaustive against the oracle over a few shapes, then random shapes.
  int64_t checked = 0;
  std::mt19937 rng(7);
  for (int round = 0; round < 400; ++round) {
    const int64_t n = 2 + static_cast<int64_t>(rng() % 70);
    const int64_t first = static_cast<int64_t>(rng() % static_cast<uint32_t>(n));
    int64_t norm_from = -1;
    if (rng() % 2 == 0 && n - first >= 2) norm_from = first + 1 + static_cast<int64_t>(rng() % static_cast<uint32_t>(n - first - 1));
    Check(ValidateLayerRequest(n, first, norm_from).empty(), "oracle shapes are valid requests");
    const std::vector<LayerKind> k = LayerKinds(n, first, norm_from);
    for (int64_t a = -1; a <= n + 1; ++a) {
      for (int64_t b = a; b <= n + 1; ++b) {
        ++checked;
        if (CheckRangeLoaded(k, a, b).empty() != Oracle(n, first, norm_from, a, b)) {
          std::fprintf(stderr, "  mismatch n=%lld first=%lld norm=%lld [%lld,%lld)\n", static_cast<long long>(n), static_cast<long long>(first),
                       static_cast<long long>(norm_from), static_cast<long long>(a), static_cast<long long>(b));
          Check(false, "CheckRangeLoaded disagrees with the oracle");
          return;
        }
      }
    }
  }
  std::fprintf(stderr, "  range check: %lld (shape, range) pairs agree with the oracle\n", static_cast<long long>(checked));

  // Negative controls: two broken checkers must disagree with the oracle somewhere.
  const auto mutant_norm_as_full = [](const std::vector<LayerKind>& k, int64_t a, int64_t b) {
    for (int64_t i = a; i < b; ++i) {
      if (k[static_cast<size_t>(i)] == LayerKind::kHole) return false;
    }
    return true;
  };
  const auto mutant_no_hole = [](const std::vector<LayerKind>& k, int64_t a, int64_t b) {
    for (int64_t i = a; i < b; ++i) {
      if (k[static_cast<size_t>(i)] == LayerKind::kNormOnly) return false;
    }
    return true;
  };
  bool caught_norm = false, caught_hole = false;
  for (int64_t a = 0; a <= 33; ++a) {
    for (int64_t b = a; b <= 33; ++b) {
      caught_norm = caught_norm || mutant_norm_as_full(x, a, b) != Oracle(33, 0, 32, a, b);
    }
  }
  for (int64_t a = 0; a <= 64; ++a) {
    for (int64_t b = a; b <= 64; ++b) {
      caught_hole = caught_hole || mutant_no_hole(y, a, b) != Oracle(64, 32, -1, a, b);
    }
  }
  Check(caught_norm, "negative control: a checker that treats norm-only as loaded is caught");
  Check(caught_hole, "negative control: a checker that forgets the hole prefix is caught");
}

// ---- ModelOptions::stage_only -> the container request -------------------------------------------------------------------
void Roles() {
  RoleRequest q;
  Check(ValidateRole(q).empty(), "kOff is always valid");
  q.role = Role::kFront;
  q.split = 32;
  Check(ValidateRole(q).empty(), "front, split 32, layer_limit -1");
  q.layer_limit = 33;
  Check(ValidateRole(q).empty(), "front with layer_limit == split + 1");
  q.layer_limit = 40;
  Check(!ValidateRole(q).empty(), "front with another layer_limit is refused");
  q.layer_limit = -1;
  q.mtp_draft_k = 3;
  Check(Contains(ValidateRole(q), "no speculation"), "front runs no speculation (mtp)");
  q.mtp_draft_k = 0;
  q.dflash_draft_k = 7;
  Check(Contains(ValidateRole(q), "no speculation"), "front runs no speculation (dflash)");
  q.dflash_draft_k = 0;
  q.dflash_container = true;
  Check(Contains(ValidateRole(q), "drafter"), "no drafter on a stage");
  q.dflash_container = false;
  q.tp_world = 2;
  Check(Contains(ValidateRole(q), "TP=1"), "a stage is a TP=1 Model");
  q.tp_world = 1;
  q.split = 0;
  Check(!ValidateRole(q).empty(), "split 0 is refused");

  RoleRequest y;
  y.role = Role::kBack;
  y.split = 32;
  y.mtp_draft_k = 3;  // the back stage keeps its MTP head and its sizing
  Check(ValidateRole(y).empty(), "back with MTP is fine");
  y.layer_limit = 4;
  y.split = 2;
  Check(ValidateRole(y).empty(), "back, 4-layer test container, split 2");
  y.layer_limit = 2;
  Check(!ValidateRole(y).empty(), "back with layer_limit <= split holds nothing");

  RoleRequest fx;
  fx.role = Role::kFront;
  fx.split = 32;
  ContainerArgs a = ContainerArgsFor(fx);
  Check(a.first_layer == 0 && a.norm_only_from == 32 && a.layer_limit == 33 && a.skip_heads, "front -> layers [0, k], norm-only k, no heads");
  RoleRequest by;
  by.role = Role::kBack;
  by.split = 32;
  a = ContainerArgsFor(by);
  Check(a.first_layer == 32 && a.norm_only_from == -1 && a.layer_limit == -1 && !a.skip_heads, "back -> hole prefix k, full head");
  by.layer_limit = 4;
  Check(ContainerArgsFor(by).layer_limit == 4, "back keeps the test container's layer_limit");
  a = ContainerArgsFor(RoleRequest{});
  Check(a.first_layer == 0 && a.norm_only_from == -1 && a.layer_limit == -1 && !a.skip_heads, "kOff -> the defaults");
}

// ---- the borrowed embedding mirror ----------------------------------------------------------------------------------------
void Lease() {
  std::shared_ptr<EmbedMirrorLease> owner;
  Check(EmbedMirrorLeaseViolation(owner).empty() && EmbedMirrorBorrowers(owner) == 0, "no lease, nothing to violate");
  owner = std::make_shared<EmbedMirrorLease>();
  static const uint16_t kMirror[4] = {1, 2, 3, 4};
  owner->data = kMirror;
  owner->elems = 4;
  owner->device = 1;
  Check(EmbedMirrorLeaseViolation(owner).empty(), "an owner with no borrower may be destroyed");
  {
    std::shared_ptr<const EmbedMirrorLease> stage = owner;  // what Container::EmbedMirrorLeaseHandle hands the stage's Load
    Check(EmbedMirrorBorrowers(owner) == 1, "one borrower");
    const std::string why = EmbedMirrorLeaseViolation(owner);
    Check(Contains(why, "destroy the stage Model before the rank Model"), "destroying the owner under a live borrower names the rule");
    std::shared_ptr<const EmbedMirrorLease> second = stage;
    Check(EmbedMirrorBorrowers(owner) == 2, "two borrowers");

    // Negative control: a counter that forgets the borrower (compares the count to 1 the wrong way) is caught.
    const bool mutant_sees_borrower = owner.use_count() > 2;  // off by one
    Check(mutant_sees_borrower, "negative control: the off-by-one counter still sees TWO borrowers ...");
    stage.reset();
    Check(EmbedMirrorBorrowers(owner) == 1 && !(owner.use_count() > 2), "... but misses the last one, which the real counter reports");
  }
  Check(EmbedMirrorLeaseViolation(owner).empty(), "stage destroyed first, then the owner: no violation");

  EmbedMirrorLease l;
  l.data = kMirror;
  l.elems = 4;
  l.device = 1;
  Check(CheckEmbedMirrorBorrow(l, 1, 4).empty(), "borrow on the owner's device, same size");
  Check(Contains(CheckEmbedMirrorBorrow(l, 0, 4), "device"), "a borrower on the other card is refused (no peer access)");
  Check(Contains(CheckEmbedMirrorBorrow(l, 1, 5), "elements"), "a mirror of another size is refused");
  l.data = nullptr;
  Check(Contains(CheckEmbedMirrorBorrow(l, 1, 4), "empty"), "an empty lease is refused");
}

// ---- the sync state's seed rule ------------------------------------------------------------------------------------------
void Seed() {
  constexpr int64_t kHidden = 5120;
  StageSyncState s;
  s.pos = 4096;
  s.started = true;
  s.mtp_seed_valid = true;
  s.mtp_seed.assign(kHidden, 7);

  SeedPlan p = PlanSeed(s, /*model_has_mtp=*/true, kHidden);
  Check(p.error.empty() && p.valid && p.copy, "an MTP model takes a valid seed and copies it");
  p = PlanSeed(s, false, kHidden);
  Check(p.error.empty() && !p.valid && !p.copy, "a model without an MTP head (stage A) ignores the seed");
  s.mtp_seed_valid = false;
  p = PlanSeed(s, true, kHidden);
  Check(p.error.empty() && !p.valid && !p.copy, "an invalid source seed clears the flag and copies nothing (a decode-free Reset'd rank)");
  s.mtp_seed_valid = true;
  s.mtp_seed.resize(kHidden - 1);
  p = PlanSeed(s, true, kHidden);
  Check(Contains(p.error, "hidden_size"), "a seed of the wrong width is refused before anything is written");
  s.mtp_seed.clear();
  Check(!PlanSeed(s, true, kHidden).error.empty(), "a valid flag with no bytes is refused");
  Check(PlanSeed(s, false, kHidden).error.empty(), "... unless the model has no MTP head to write it into");

  // The warm-turn hazard of design section 5: the stage's old flag is stale after a decode; the plan always assigns it.
  s.mtp_seed_valid = false;
  s.mtp_seed.clear();
  Check(!PlanSeed(s, true, kHidden).valid, "negative control: with `valid` taken from the source, a stale true on the stage cannot survive");

  StageSyncState a;
  Check(!CheckAdoptable(a).empty(), "a fresh state (pos 0) is not an adoptable prefill");
  a.pos = 100;
  Check(!CheckAdoptable(a).empty(), "pos > 0 but not started");
  a.started = true;
  Check(CheckAdoptable(a).empty(), "a finished prefill is adoptable");
}

}  // namespace

int main() {
  Kinds();
  RequestValidation();
  RangeCheck();
  Roles();
  Lease();
  Seed();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_stage_load_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_stage_load_cpu: PASS\n");
  return 0;
}
