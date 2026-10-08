// The per-sequence scalars a pipelined prefill call hands between a TP rank Model and the two stage Models of the hybrid mode
// (docs/pp-tp2-hybrid.md 3 last row, 5): what Model::StageSetSyncState writes into a stage before a call (both roles, replacing
// the stage-A-only PpSetSyncState for the hybrid) and Model::TpAdoptPrefill writes into a rank after it. HIP-free; the seed
// rule is a pure function so tests/model/test_stage_load_cpu.cpp covers it.
//
// The MTP seed (mtp_seed_hidden_, [hidden] bf16, 10 KiB, plus mtp_seed_valid_) travels with the scalars: without it a stage Y
// whose last call predates a decode has a stale `mtp_seed_valid_`, `have_boundary` (model.cpp ChunkEpilogue) is wrong and the
// MTP KV row of the call's first position is skipped or primed from a wrong hidden state (design section 5, critique item 5).
// It is host bytes in the struct (not a device pointer) because the two cards have no peer access and the copy is 10 KiB.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r4dx::model {

struct StageSyncState {
  int64_t pos = 0;
  bool started = false;
  bool mrope_active = false;
  int64_t mrope_delta = 0;
  bool mtp_seed_valid = false;
  std::vector<uint16_t> mtp_seed;  // [hidden] bf16 when mtp_seed_valid and the source has an MTP head, else empty
};

// What a Model with / without an MTP head does with a sync state's seed.
struct SeedPlan {
  bool valid = false;  // the Model's mtp_seed_valid_ afterwards (only written when the Model has an MTP head)
  bool copy = false;   // upload s.mtp_seed into mtp_seed_hidden_
  std::string error;
};
inline SeedPlan PlanSeed(const StageSyncState& s, bool model_has_mtp, int64_t hidden) {
  SeedPlan p;
  if (!model_has_mtp || !s.mtp_seed_valid) return p;
  if (static_cast<int64_t>(s.mtp_seed.size()) != hidden) {
    p.error = "the MTP seed holds " + std::to_string(s.mtp_seed.size()) + " elements, hidden_size is " + std::to_string(hidden);
    return p;
  }
  p.valid = true;
  p.copy = true;
  return p;
}

// A state a rank adopts after a pipelined prefill call: the call fed at least one row.
inline std::string CheckAdoptable(const StageSyncState& s) {
  if (s.pos <= 0 || !s.started) return "a pipelined prefill leaves pos > 0 and started (got pos " + std::to_string(s.pos) + ")";
  return "";
}

}  // namespace r4dx::model
