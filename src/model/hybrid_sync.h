// HIP-free policy of the hybrid serving mode's warm turns (docs/pp-tp2-hybrid.md 5): PP-2 prefill + TP-2 decode, where
// the TP shards are the truth and BOTH pipeline stages are mirrors. pp::MirrorTracker (pp_sync.h) tracks one mirror of a
// decode Model; this is its generalisation to two stage mirrors of a pair of TP ranks, plus the MTP bookkeeping that
// only exists in the hybrid (the stage-B seed and the primed-block rule). Header-only so a CPU test
// (tests/model/test_hybrid_sync_cpu.cpp) checks the stale-state rules -- a stale mirror silently produces wrong tokens --
// against an abstract model of every holder's contents, with no GPU.
#pragma once

#include <algorithm>
#include <cstdint>

namespace r4dx::model::hybrid {

// ---- block ranges (shared with reshard_plan.h) ------------------------------------------------------------------------
struct BlockRange {
  int64_t first = 0, last = -1;  // inclusive; empty when last < first
  int64_t Blocks() const { return last >= first ? last - first + 1 : 0; }
};
// The blocks that hold rows [row0, row1).
inline BlockRange BlocksOfRows(int64_t row0, int64_t row1, int64_t block_tokens) {
  BlockRange b;
  if (row1 > row0) {
    b.first = row0 / block_tokens;
    b.last = (row1 - 1) / block_tokens;
  }
  return b;
}
// What a pipelined call over rows [p0, n_end) hands back to the TP shards. Layer KV: the call wrote rows [p0, n_end); the
// first block is copied whole, which is right because the stage's rows below p0 were made equal to the shards' by the warm
// gather. MTP head KV: the stage's copy holds ONLY the rows the call primed, i.e. the boundary row p0-1 (when there is a
// seed) and rows [p0, n_end-2], so the copy starts at the block of row p0-1 -- and the warm gather loaded that block into
// stage B first (TpMasterTracker::SyncPlan::mtp_block), otherwise the block's older rows would be overwritten with stale
// stage bytes. No boundary row at p0 == 0.
inline int64_t MtpFirstBlock(int64_t p0, int64_t block_tokens) { return p0 > 0 ? (p0 - 1) / block_tokens : 0; }
inline BlockRange KvScatterBlocks(int64_t p0, int64_t n_end, int64_t block_tokens) {
  return BlocksOfRows(p0, n_end, block_tokens);
}
inline BlockRange MtpScatterBlocks(int64_t p0, int64_t n_end, int64_t block_tokens) {
  BlockRange b = BlocksOfRows(p0, n_end, block_tokens);
  if (b.Blocks() > 0) b.first = MtpFirstBlock(p0, block_tokens);
  return b;
}

// ---- the tracker ------------------------------------------------------------------------------------------------------
// Stage A (the desktop card X, TP rank 1) owns layers [0, k), stage B (the headless card Y, TP rank 0) layers [k, N) plus
// the MTP head (docs/pp-tp2-hybrid.md 0, 1). Each stage keeps, for ITS layers, the full-head live state; the TP ranks hold
// the same state head-split. A pipelined call reads the stages and writes the TP shards back at its end; decode, speculative
// rounds, a restore, a Reset and any short TP prefill move the TP shards alone. Per stage the tracker keeps
//   kv_valid_to  the stage's KV rows [0, kv_valid_to) equal the TP shards' (all the stage's attention layers)
//   gdn_pos      the position at which the stage's GDN state equals the TP shards', or -1 (unknown / stale)
// and, for stage B only,
//   seed_synced  stage B's MTP seed (valid flag + the 10 KiB hidden row) equals TP rank 0's
// Every rule only ever makes these smaller or exact, never guesses.
//
// MTP KV is deliberately NOT a cumulative mirror: stage B's MTP head KV holds only the rows ITS calls primed, so each warm
// call loads the one block that holds row p0-1 from the TP shards before it runs (SyncPlan::mtp_block) and writes back the
// blocks [MtpFirstBlock(p0), last] after it (MtpScatterBlocks above; the copy ops are reshard_plan.h's). Nothing about it
// needs remembering between calls.
class TpMasterTracker {
 public:
  static constexpr int kStages = 2;  // 0 = stage A (layers < k), 1 = stage B (layers >= k, MTP head)

  // `block_tokens`: KV rows per block (the attention kernel's block size, 16). `mtp`: the Model has an MTP head.
  explicit TpMasterTracker(int64_t block_tokens = 16, bool mtp = true) : block_tokens_(block_tokens), mtp_(mtp) { Reset(); }

  // Every Model freshly reset (or just loaded): zero state at position 0 everywhere, no seed anywhere.
  void Reset() {
    for (Stage& s : st_) s = Stage{0, 0};
    seed_synced_ = true;
  }

  // What must be copied TP shards -> stage before a pipelined Prefill call that starts at position `p0`.
  struct StageSync {
    bool gdn = false;                  // the GDN state of the stage's layers (recurrent live slot + live conv entries)
    int64_t kv_row0 = 0, kv_row1 = 0;  // KV rows [kv_row0, kv_row1) of the stage's attention layers (empty when equal)
    bool Any() const { return gdn || kv_row1 > kv_row0; }
  };
  struct SyncPlan {
    StageSync stage[kStages];
    bool scalars = true;     // pos_/started_/mrope into both stages: host-only, tens of bytes, so always
    bool seed = false;       // MTP seed (flag + hidden row) TP rank 0 -> stage B
    int64_t mtp_block = -1;  // MTP KV block of row p0-1, TP shards -> stage B; -1 when there is no boundary row (p0 == 0)
    bool Any() const { return stage[0].Any() || stage[1].Any() || seed || mtp_block >= 0; }
  };
  SyncPlan PlanSync(int64_t p0) const {
    SyncPlan p;
    for (int s = 0; s < kStages; ++s) {
      p.stage[s].gdn = st_[s].gdn_pos != p0;
      p.stage[s].kv_row0 = std::min(st_[s].kv_valid_to, p0);
      p.stage[s].kv_row1 = p0;
    }
    if (mtp_) {
      p.seed = !seed_synced_;
      p.mtp_block = p0 > 0 ? MtpFirstBlock(p0, block_tokens_) : -1;
    }
    return p;
  }
  // The plan was applied (both stages now equal the TP shards on everything the next call reads, at position p0).
  void AfterSync(int64_t p0) {
    for (Stage& s : st_) {
      s.gdn_pos = p0;
      s.kv_valid_to = p0;
    }
    seed_synced_ = true;
  }
  // The pipelined call completed and its reshard wrote the TP shards (and the seed to both ranks): everything agrees
  // through `pos_end`.
  void AfterPipelined(int64_t pos_end) {
    for (Stage& s : st_) {
      s.gdn_pos = pos_end;
      s.kv_valid_to = pos_end;
    }
    seed_synced_ = true;
  }
  // The TP shards alone ran something that changes their GDN state and writes KV rows from `from_pos` on: a decode step, a
  // speculative round, a short TP prefill (rows at positions >= from_pos only; rows below are untouched). Any forward also
  // moves the MTP seed.
  void TpOnly(int64_t from_pos) {
    for (Stage& s : st_) {
      s.kv_valid_to = std::min(s.kv_valid_to, from_pos);
      s.gdn_pos = -1;
    }
    seed_synced_ = false;
  }
  // The TP ranks' state was replaced by a checkpoint (RestoreCheckpoint): the GDN state and the MTP seed are the saved ones;
  // KV is unchanged below the restored position, and rows at or above it will be rewritten before they are read.
  void TpRestored() {
    for (Stage& s : st_) s.gdn_pos = -1;
    seed_synced_ = false;
  }

  int64_t KvValidTo(int stage) const { return st_[stage].kv_valid_to; }
  int64_t GdnPos(int stage) const { return st_[stage].gdn_pos; }
  bool SeedSynced() const { return seed_synced_; }

 private:
  struct Stage {
    int64_t kv_valid_to;
    int64_t gdn_pos;
  };
  Stage st_[kStages];
  bool seed_synced_ = true;
  int64_t block_tokens_;
  bool mtp_;
};

}  // namespace r4dx::model::hybrid
