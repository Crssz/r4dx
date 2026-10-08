// tests/model/test_hybrid_sync_cpu.cpp -- CPU-only checks of src/model/hybrid_sync.h, the stale-state rules of the hybrid
// serving mode's warm turns (docs/pp-tp2-hybrid.md 5): PP-2 prefill + TP-2 decode, the TP shards are the truth and BOTH
// pipeline stages are mirrors. The TpMasterTracker's per-stage numbers (kv_valid_to, gdn_pos), the stage-B MTP seed flag and
// the MTP primed-block rule are run against an abstract model of every holder's contents -- the two TP ranks' shards (one
// merged image per stage's layers), stage A, stage B with its MTP head KV and seed -- under tens of thousands of random
// operation sequences (pipelined prefill with its warm gather and block-granular reshard, decode / speculative round / short
// TP prefill, checkpoint save / restore, Reset), the way test_pp_sync_cpu does for the one-mirror tracker. An ORACLE (a single
// device that runs every operation directly) is the reference: after every operation the TP shards' live state must equal
// it, whatever path the operation took. Then NEGATIVE CONTROLS: ten deliberately wrong rules (a forgotten event, an
// off-by-one, the MTP boundary block, the seed, one stage's sync skipped, ...) each of which the same walk must catch.
// A stale mirror silently produces wrong tokens; no HIP call, no container, always runs.
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "hybrid_sync.h"

using namespace r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

constexpr int64_t kRows = 2048, kB = 16, kLimit = 1750;

// ---- the abstract contents ------------------------------------------------------------------------------------------------
// A version stamp per KV row (what was written there last) and one stamp for the whole GDN state (the history it was last
// advanced by); the MTP head has its own per-row stamps and the seed one stamp + a valid flag. Two holders agree on a row /
// the state iff the stamps match. New stamps are a function of (op, row[, part]) only: the abstract "model" is deterministic,
// so a stage that started from the right state produces exactly the oracle's stamps.
struct Part {  // the KV rows and GDN state of one stage's layers
  std::vector<int64_t> kv = std::vector<int64_t>(kRows, -1);
  int64_t gdn = 0;
};
struct Truth {  // the TP shards (both ranks, merged) and the oracle: the state of ALL layers + the MTP head + the scalars
  Part a, b;  // stage A's layers, stage B's layers
  std::vector<int64_t> mtp = std::vector<int64_t>(kRows, -1);
  int64_t seed = 0;
  bool seed_valid = false;
  int64_t pos = 0;
};
struct StageModel {  // one pipeline stage's Model; stage B alone uses mtp / seed
  Part p;
  std::vector<int64_t> mtp = std::vector<int64_t>(kRows, -1);
  int64_t seed = 0;
  bool seed_valid = false;
  int64_t pos = 0;
};

int64_t Mix(int64_t a, int64_t b, int64_t c) {
  uint64_t h = static_cast<uint64_t>(a) * 6364136223846793005ull + 1442695040888963407ull;
  h ^= static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h ^= static_cast<uint64_t>(c) * 0xC2B2AE3D27D4EB4Full + (h << 7) + (h >> 3);
  return static_cast<int64_t>(h >> 1);
}

// Rows [p0, p0 + n) written, the GDN state advanced.
void AdvancePart(Part* p, int64_t p0, int64_t n, int64_t op, int part) {
  for (int64_t r = p0; r < p0 + n; ++r) p->kv[static_cast<size_t>(r)] = Mix(op, r, 10 + part);
  p->gdn = Mix(p->gdn, op, 20 + part);
}
// Model::ChunkEpilogue's MTP priming (model.cpp "MTP lockstep KV priming"): the boundary row p0-1 when the seed is valid (it
// pairs the PREVIOUS call's last hidden state -- the seed -- with this call's first token), then rows [p0, p0+n-2]; the call's
// last row stays dangling and becomes the new seed.
void AdvanceMtp(std::vector<int64_t>* mtp, int64_t* seed, bool* seed_valid, int64_t p0, int64_t n, int64_t op) {
  if (p0 > 0 && *seed_valid) (*mtp)[static_cast<size_t>(p0 - 1)] = Mix(op, p0 - 1, *seed + 30);
  for (int64_t r = p0; r + 1 < p0 + n; ++r) (*mtp)[static_cast<size_t>(r)] = Mix(op, r, 31);
  *seed = Mix(op, p0 + n - 1, 32);
  *seed_valid = true;
}
void TruthAdvance(Truth* t, int64_t n, int64_t op) {
  AdvancePart(&t->a, t->pos, n, op, 0);
  AdvancePart(&t->b, t->pos, n, op, 1);
  AdvanceMtp(&t->mtp, &t->seed, &t->seed_valid, t->pos, n, op);
  t->pos += n;
}

enum class Fault {
  kNone,
  kSkipTpOnlyEvent,       // the tracker is never told about a decode / spec round / short TP prefill
  kTpOnlyLate,            // ... told with from_pos + 1: row from_pos is rewritten but still claimed valid
  kSkipRestoreEvent,      // the tracker is never told about a checkpoint restore (the seed goes stale)
  kNoSeedSync,            // the plan's seed copy into stage B is not applied
  kMtpGatherWrongBlock,   // the warm gather loads the MTP block of row p0 instead of row p0-1
  kMtpScatterWrongBlock,  // the reshard writes the MTP blocks from the block of p0 instead of p0-1
  kKvScatterOffByOne,     // the reshard writes the layer KV blocks from block(p0) + 1
  kNoGdnGather,           // the plan's GDN copy into the stages is not applied
  kSkipStageBSync,        // stage B's part of the warm gather is skipped altogether
};

class World {
 public:
  explicit World(Fault f) : fault(f), t(kB, /*mtp=*/true) {}

  void Reset() {  // Model::Reset: GDN zeroed, pos 0, no seed; KV rows keep their stale bytes
    for (Truth* x : {&tp, &oracle}) {
      x->a.gdn = x->b.gdn = 0;
      x->pos = 0;
      x->seed = 0;
      x->seed_valid = false;
    }
    for (StageModel* s : {&sa, &sb}) {
      s->p.gdn = 0;
      s->pos = 0;
      s->seed = 0;
      s->seed_valid = false;
    }
    t.Reset();
    ckpt_valid = false;
  }

  // The tracker's plan applied: the warm gather, TP shards -> stages (exact rows for layer KV, whole blocks for MTP KV).
  void Sync() {
    const int64_t p0 = tp.pos;
    const TpMasterTracker::SyncPlan plan = t.PlanSync(p0);
    for (int s = 0; s < 2; ++s) {
      if (fault == Fault::kSkipStageBSync && s == 1) continue;
      StageModel& st = s == 0 ? sa : sb;
      const Part& src = s == 0 ? tp.a : tp.b;
      if (plan.stage[s].gdn && fault != Fault::kNoGdnGather) st.p.gdn = src.gdn;
      for (int64_t r = plan.stage[s].kv_row0; r < plan.stage[s].kv_row1; ++r) st.p.kv[static_cast<size_t>(r)] = src.kv[static_cast<size_t>(r)];
      st.pos = p0;
    }
    if (plan.seed && fault != Fault::kNoSeedSync) {
      sb.seed = tp.seed;
      sb.seed_valid = tp.seed_valid;
    }
    if (plan.mtp_block >= 0) {
      const int64_t blk = fault == Fault::kMtpGatherWrongBlock ? p0 / kB : plan.mtp_block;
      for (int64_t r = blk * kB; r < (blk + 1) * kB && r < kRows; ++r) sb.mtp[static_cast<size_t>(r)] = tp.mtp[static_cast<size_t>(r)];
    }
    t.AfterSync(p0);
  }
  // Right after a sync both stages must equal the TP shards on everything the call reads: rows [0, p0), the GDN state, the
  // scalars, and (stage B) the seed and the MTP block of the boundary row.
  bool EqualAfterSync() const {
    const int64_t p0 = tp.pos;
    for (int64_t r = 0; r < p0; ++r) {
      if (sa.p.kv[static_cast<size_t>(r)] != tp.a.kv[static_cast<size_t>(r)]) return false;
      if (sb.p.kv[static_cast<size_t>(r)] != tp.b.kv[static_cast<size_t>(r)]) return false;
    }
    if (sa.p.gdn != tp.a.gdn || sb.p.gdn != tp.b.gdn || sa.pos != p0 || sb.pos != p0) return false;
    if (sb.seed_valid != tp.seed_valid || (tp.seed_valid && sb.seed != tp.seed)) return false;
    if (p0 > 0) {
      const int64_t blk = (p0 - 1) / kB;
      for (int64_t r = blk * kB; r < (blk + 1) * kB; ++r) {
        if (sb.mtp[static_cast<size_t>(r)] != tp.mtp[static_cast<size_t>(r)]) return false;
      }
    }
    return true;
  }
  // The claims are never stronger than the truth, at any time.
  bool Consistent() const {
    for (int s = 0; s < 2; ++s) {
      const StageModel& st = s == 0 ? sa : sb;
      const Part& src = s == 0 ? tp.a : tp.b;
      for (int64_t r = 0; r < t.KvValidTo(s); ++r) {
        if (st.p.kv[static_cast<size_t>(r)] != src.kv[static_cast<size_t>(r)]) return false;
      }
      if (t.GdnPos(s) >= 0 && t.GdnPos(s) == tp.pos && st.p.gdn != src.gdn) return false;
    }
    if (t.SeedSynced() && (sb.seed_valid != tp.seed_valid || (tp.seed_valid && sb.seed != tp.seed))) return false;
    return true;
  }
  // The TP shards' LIVE state equals the oracle's: positions, GDN, KV rows below pos, MTP rows below pos-1 (the last row is
  // dangling), the seed.
  bool MatchesOracle() const {
    if (tp.pos != oracle.pos || tp.a.gdn != oracle.a.gdn || tp.b.gdn != oracle.b.gdn) return false;
    if (tp.seed_valid != oracle.seed_valid || (tp.seed_valid && tp.seed != oracle.seed)) return false;
    for (int64_t r = 0; r < tp.pos; ++r) {
      if (tp.a.kv[static_cast<size_t>(r)] != oracle.a.kv[static_cast<size_t>(r)]) return false;
      if (tp.b.kv[static_cast<size_t>(r)] != oracle.b.kv[static_cast<size_t>(r)]) return false;
    }
    for (int64_t r = 0; r + 1 < tp.pos; ++r) {
      if (tp.mtp[static_cast<size_t>(r)] != oracle.mtp[static_cast<size_t>(r)]) return false;
    }
    return true;
  }

  void PipelinedPrefill(int64_t n) {
    const int64_t op = ++op_id;
    const int64_t p0 = tp.pos, n_end = p0 + n;
    Sync();
    if (!EqualAfterSync()) ++sync_failures;
    // both stages run their layers from their own state
    AdvancePart(&sa.p, p0, n, op, 0);
    AdvancePart(&sb.p, p0, n, op, 1);
    AdvanceMtp(&sb.mtp, &sb.seed, &sb.seed_valid, p0, n, op);
    sa.pos = sb.pos = n_end;
    // the end-of-call reshard: whole blocks, stage -> TP shards
    BlockRange kb = KvScatterBlocks(p0, n_end, kB);
    if (fault == Fault::kKvScatterOffByOne) ++kb.first;
    for (int64_t blk = kb.first; blk <= kb.last; ++blk) {
      for (int64_t r = blk * kB; r < (blk + 1) * kB && r < kRows; ++r) {
        tp.a.kv[static_cast<size_t>(r)] = sa.p.kv[static_cast<size_t>(r)];
        tp.b.kv[static_cast<size_t>(r)] = sb.p.kv[static_cast<size_t>(r)];
      }
    }
    tp.a.gdn = sa.p.gdn;
    tp.b.gdn = sb.p.gdn;
    BlockRange mb = MtpScatterBlocks(p0, n_end, kB);
    if (fault == Fault::kMtpScatterWrongBlock) mb.first = p0 / kB;
    for (int64_t blk = mb.first; blk <= mb.last; ++blk) {
      for (int64_t r = blk * kB; r < (blk + 1) * kB && r < kRows; ++r) tp.mtp[static_cast<size_t>(r)] = sb.mtp[static_cast<size_t>(r)];
    }
    tp.seed = sb.seed;  // both ranks adopt stage B's seed (Model::TpAdoptPrefill)
    tp.seed_valid = sb.seed_valid;
    tp.pos = n_end;
    t.AfterPipelined(n_end);
    TruthAdvance(&oracle, n, op);
  }
  void TpAdvance(int64_t n) {  // decode / speculative round / short TP prefill: the shards move alone
    const int64_t op = ++op_id, p0 = tp.pos;
    TruthAdvance(&tp, n, op);
    TruthAdvance(&oracle, n, op);
    if (fault == Fault::kSkipTpOnlyEvent) return;
    t.TpOnly(fault == Fault::kTpOnlyLate ? p0 + 1 : p0);
  }
  void Save() {
    ckpt = tp;
    ockpt = oracle;
    ckpt_valid = true;
  }
  void Restore() {  // RestoreCheckpoint: the GDN state, the position and the seed come back; KV and MTP rows keep what they hold
    if (!ckpt_valid) return;
    const auto restore = [](Truth* x, const Truth& c) {
      x->a.gdn = c.a.gdn;
      x->b.gdn = c.b.gdn;
      x->pos = c.pos;
      x->seed = c.seed;
      x->seed_valid = c.seed_valid;
    };
    restore(&tp, ckpt);
    restore(&oracle, ockpt);
    if (fault != Fault::kSkipRestoreEvent) t.TpRestored();
  }

  Fault fault;
  Truth tp, oracle, ckpt, ockpt;
  StageModel sa, sb;
  TpMasterTracker t;
  bool ckpt_valid = false;
  int64_t op_id = 0;
  int64_t sync_failures = 0;
};

struct WalkResult {
  int64_t pipelined = 0, warm_pipelined = 0, sync_failures = 0, claims_violated = 0, oracle_mismatch = 0;
  int64_t boundary_aligned = 0;  // pipelined calls whose p0 is a multiple of the block size (the boundary row is in the previous block)
  int64_t Detected() const { return sync_failures + claims_violated + oracle_mismatch; }
};

WalkResult Walk(Fault fault, int sequences, uint64_t seed) {
  std::mt19937_64 rng(seed);
  WalkResult res;
  for (int seq = 0; seq < sequences; ++seq) {
    World w(fault);
    w.Reset();
    for (int step = 0; step < 14; ++step) {
      const int op = static_cast<int>(rng() % 11);
      int64_t n = 1 + static_cast<int64_t>(rng() % 200);
      if (rng() % 4 == 0) n = 1 + static_cast<int64_t>(rng() % 20);
      if (w.tp.pos + n + 300 >= kLimit) {
        w.Reset();
        continue;
      }
      switch (op) {
        case 0:
          w.Reset();
          break;
        case 1:
        case 2:
        case 3:
        case 4: {
          const int64_t p0 = w.tp.pos;
          const int64_t failures_before = w.sync_failures;
          w.PipelinedPrefill(n);
          ++res.pipelined;
          if (p0 > 0) ++res.warm_pipelined;
          if (p0 > 0 && p0 % kB == 0) ++res.boundary_aligned;
          if (w.sync_failures != failures_before) ++res.sync_failures;
          break;
        }
        case 5:
        case 6:
          w.TpAdvance(std::min<int64_t>(n, 12));  // decode / a speculative round
          break;
        case 7:
          w.TpAdvance(n);  // a short TP prefill
          break;
        case 8:
          w.Save();
          break;
        case 9:
          w.Restore();
          break;
        default:
          w.TpAdvance(1);
          break;
      }
      if (!w.Consistent()) ++res.claims_violated;
      if (!w.MatchesOracle()) ++res.oracle_mismatch;
    }
  }
  return res;
}

void RandomSequences() {
  const WalkResult r = Walk(Fault::kNone, 20000, 20261008);
  Check(r.pipelined > 40000, "the random walk exercised many pipelined prefills");
  Check(r.warm_pipelined > 20000, "... most of them warm (p0 > 0)");
  Check(r.boundary_aligned > 1000, "... with the boundary row in the previous block often enough (p0 a multiple of the block size)");
  Check(r.sync_failures == 0, "after the planned warm gather both stages equal the TP shards on every row, the GDN state, the seed and the MTP block");
  Check(r.claims_violated == 0, "the tracker never claims more than the mirrors hold");
  Check(r.oracle_mismatch == 0, "after every operation the TP shards' live state equals the oracle's (KV, GDN, MTP rows, seed, position)");
  std::fprintf(stderr, "  real rules: %lld pipelined calls (%lld warm, %lld with p0 %% 16 == 0), 0 violations\n",
               static_cast<long long>(r.pipelined), static_cast<long long>(r.warm_pipelined), static_cast<long long>(r.boundary_aligned));
}

void NegativeControls() {
  struct Control {
    Fault f;
    const char* name;
  };
  const Control controls[] = {
      {Fault::kSkipTpOnlyEvent, "NEGATIVE CONTROL: a forgotten TpOnly event is caught"},
      {Fault::kTpOnlyLate, "NEGATIVE CONTROL: TpOnly(from_pos + 1) is caught"},
      {Fault::kSkipRestoreEvent, "NEGATIVE CONTROL: a forgotten TpRestored event (stale stage-B seed) is caught"},
      {Fault::kNoSeedSync, "NEGATIVE CONTROL: not syncing the MTP seed into stage B is caught"},
      {Fault::kMtpGatherWrongBlock, "NEGATIVE CONTROL: gathering the MTP block of row p0 instead of p0-1 is caught"},
      {Fault::kMtpScatterWrongBlock, "NEGATIVE CONTROL: writing back the MTP blocks from block(p0) instead of block(p0-1) is caught"},
      {Fault::kKvScatterOffByOne, "NEGATIVE CONTROL: an off-by-one first KV block in the reshard is caught"},
      {Fault::kNoGdnGather, "NEGATIVE CONTROL: not copying the GDN state into the stages is caught"},
      {Fault::kSkipStageBSync, "NEGATIVE CONTROL: skipping stage B's part of the warm gather is caught"},
  };
  for (const Control& c : controls) {
    const WalkResult r = Walk(c.f, 4000, 77 + static_cast<uint64_t>(c.f));
    Check(r.Detected() > 0, c.name);
    std::fprintf(stderr, "  %-90s detected %lld (sync %lld, claims %lld, oracle %lld)\n", c.name, static_cast<long long>(r.Detected()),
                 static_cast<long long>(r.sync_failures), static_cast<long long>(r.claims_violated),
                 static_cast<long long>(r.oracle_mismatch));
  }
}

void Rules() {
  TpMasterTracker t;  // 16-row blocks, MTP on
  TpMasterTracker::SyncPlan p = t.PlanSync(0);
  Check(!p.Any() && p.scalars && p.mtp_block == -1, "a fresh pair needs no sync for a cold prompt, and there is no boundary row at p0 == 0");
  t.AfterPipelined(1000);
  p = t.PlanSync(1000);
  Check(!p.stage[0].Any() && !p.stage[1].Any() && !p.seed, "right after a pipelined call, the next call at the same position copies no layer state and no seed");
  Check(p.mtp_block == 62, "... but always loads the MTP block of row p0-1 (999 / 16 = 62)");
  t.TpOnly(1000);  // decode from 1000
  p = t.PlanSync(1030);
  Check(p.stage[0].gdn && p.stage[0].kv_row0 == 1000 && p.stage[0].kv_row1 == 1030, "after 30 decoded tokens: stage A needs the GDN state and rows [1000, 1030)");
  Check(p.stage[1].gdn && p.stage[1].kv_row0 == 1000 && p.stage[1].kv_row1 == 1030, "... and so does stage B");
  Check(p.seed, "... and stage B needs the MTP seed (any forward moves it)");
  t.AfterSync(1030);
  p = t.PlanSync(1030);
  Check(!p.stage[0].Any() && !p.stage[1].Any() && !p.seed && t.SeedSynced(), "synced");
  t.TpOnly(1030);
  t.TpRestored();
  p = t.PlanSync(1000);  // restore to a checkpoint at 1000
  Check(p.stage[0].gdn && p.stage[1].gdn && p.seed, "a restore always forces the GDN copy and the seed");
  Check(p.stage[0].kv_row1 <= p.stage[0].kv_row0 || p.stage[0].kv_row0 == 1000, "no KV rows below the restored position are copied twice");
  // restore alone (no decode in between) still invalidates the seed: the checkpoint's seed is not the stage's
  t.AfterPipelined(2000);
  Check(t.SeedSynced(), "a pipelined call leaves the seed synced (stage B produced it, both ranks adopt it)");
  t.TpRestored();
  Check(!t.SeedSynced() && t.GdnPos(0) == -1 && t.GdnPos(1) == -1 && t.KvValidTo(0) == 2000, "a restore invalidates the GDN position and the seed, not the KV prefix");
  // rewinding below the valid prefix never copies rows
  t.AfterPipelined(500);
  p = t.PlanSync(300);
  Check(p.stage[0].kv_row0 == 300 && p.stage[0].kv_row1 == 300 && p.stage[0].gdn, "a rewind copies no KV rows");
  // TP-only writes invalidate exactly from their start
  t.AfterPipelined(2000);
  t.TpOnly(1500);
  Check(t.KvValidTo(0) == 1500 && t.KvValidTo(1) == 1500 && t.GdnPos(0) == -1 && t.GdnPos(1) == -1 && !t.SeedSynced(),
        "a TP-only write at 1500 shortens both stages' valid prefix to 1500");
  t.TpOnly(1800);
  Check(t.KvValidTo(0) == 1500, "a later TP-only write does not lengthen it");
  t.Reset();
  Check(t.KvValidTo(0) == 0 && t.GdnPos(1) == 0 && t.SeedSynced(), "Reset: zero state at position 0 everywhere, no seed anywhere");
  // the MTP boundary block
  Check(t.PlanSync(1).mtp_block == 0 && t.PlanSync(16).mtp_block == 0 && t.PlanSync(17).mtp_block == 1 && t.PlanSync(1024).mtp_block == 63 &&
            t.PlanSync(1025).mtp_block == 64,
        "the MTP block is floor((p0-1)/16): a call starting exactly on a block boundary reloads the PREVIOUS block");
  // no MTP head: no seed, no block
  TpMasterTracker plain(16, /*mtp=*/false);
  plain.TpOnly(10);
  p = plain.PlanSync(100);
  Check(!p.seed && p.mtp_block == -1 && p.stage[0].gdn, "without an MTP head the plan carries neither the seed nor the MTP block");
  // block helpers
  Check(MtpFirstBlock(0, 16) == 0 && MtpFirstBlock(1, 16) == 0 && MtpFirstBlock(16, 16) == 0 && MtpFirstBlock(17, 16) == 1, "MtpFirstBlock");
  BlockRange b = KvScatterBlocks(100, 164, 16);  // rows [100, 164): blocks 6 .. 10
  Check(b.first == 6 && b.last == 10 && b.Blocks() == 5, "layer KV scatter blocks [floor(p0/16), floor((n_end-1)/16)]");
  b = MtpScatterBlocks(112, 130, 16);  // p0 = 112 = 7 * 16: the boundary row 111 is in block 6
  Check(b.first == 6 && b.last == 8, "MTP scatter blocks start at the block of row p0-1 (block 6), layer KV would start at 7");
  b = MtpScatterBlocks(0, 40, 16);
  Check(b.first == 0 && b.last == 2, "MTP scatter blocks at p0 == 0 start at block 0");
  Check(KvScatterBlocks(5, 5, 16).Blocks() == 0 && MtpScatterBlocks(5, 5, 16).Blocks() == 0, "an empty call has no blocks");
}

}  // namespace

int main() {
  Rules();
  RandomSequences();
  NegativeControls();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_sync_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_sync_cpu: PASS\n");
  return 0;
}
