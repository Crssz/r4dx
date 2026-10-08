// HIP-free policy of the real pipeline-parallel prefill (docs/pp-prefill.md, Phase 2): what the stage-A mirror
// (device 0) is known to hold relative to the decode Model (stage B, device 1), the switches that turn the pipeline
// on, the default split and the device-order rule. Header-only so a CPU test (tests/model/test_pp_sync_cpu.cpp)
// checks the stale-state rules -- the dangerous part of the design: a stale mirror silently produces wrong tokens --
// against an abstract model of both devices' contents, with no GPU.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace r4dx::model::pp {

// ---- the mirror ----------------------------------------------------------------------------------------------------
// Stage A holds, for its own layers [0, split): the KV rows of its attention layers and the live GDN state. Stage B
// (the decode Model) holds all of it and is the truth. After a pipelined Prefill call both agree through the call's
// end position; decode, speculative rounds, a restore, a Reset and any B-only prefill then move B alone. The tracker
// keeps two numbers:
//   kv_valid_to  A's KV rows [0, kv_valid_to) equal B's
//   gdn_pos      the position at which A's GDN state equals B's, or -1 (unknown / stale)
// and nothing else: every rule below only ever makes these smaller or exact, never guesses.
class MirrorTracker {
 public:
  // Both Models freshly reset (or just loaded): zero state at position 0 on both.
  void Reset() {
    kv_valid_to_ = 0;
    gdn_pos_ = 0;
  }

  // What must be copied B -> A before a pipelined Prefill call that starts at B's position `p0`.
  struct SyncPlan {
    bool gdn = false;            // the GDN states of A's layers (B's live slot / conv history)
    int64_t kv_row0 = 0, kv_row1 = 0;  // KV rows [kv_row0, kv_row1) of A's attention layers (empty when equal)
    bool Any() const { return gdn || kv_row1 > kv_row0; }
  };
  SyncPlan PlanSync(int64_t p0) const {
    SyncPlan p;
    p.gdn = gdn_pos_ != p0;
    p.kv_row0 = std::min(kv_valid_to_, p0);
    p.kv_row1 = p0;
    return p;
  }
  // The plan was applied (A now equals B on everything A's next call reads, at position p0).
  void AfterSync(int64_t p0) {
    gdn_pos_ = p0;
    kv_valid_to_ = p0;
  }
  // The pipelined call completed (A's chunks and the GDN hand-off): both agree through `pos_end`.
  void AfterPipelined(int64_t pos_end) {
    gdn_pos_ = pos_end;
    kv_valid_to_ = pos_end;
  }
  // B alone ran something that changes its GDN state and writes KV rows from `from_pos` on: a decode step, a
  // speculative round, a B-only prefill. (Rows written at positions >= from_pos only; rows below are untouched.)
  void BOnly(int64_t from_pos) {
    kv_valid_to_ = std::min(kv_valid_to_, from_pos);
    gdn_pos_ = -1;
  }
  // B's GDN state was replaced by a checkpoint (RestoreCheckpoint). B's KV is unchanged below its new position,
  // and rows at or above it will be rewritten before they are read (Model::Reset's argument).
  void BRestored() { gdn_pos_ = -1; }

  int64_t KvValidTo() const { return kv_valid_to_; }
  int64_t GdnPos() const { return gdn_pos_; }

 private:
  int64_t kv_valid_to_ = 0;
  int64_t gdn_pos_ = 0;
};

// ---- switches -------------------------------------------------------------------------------------------------------
// R4DX_PP (ModelOptions::pp / --pp): unset, empty, "0", "off", "false" = off (the default); "1", "on", "true" or
// "2" = on (a 2-stage pipeline); anything else = -1 (the caller refuses it).
inline int ParsePpEnable(const char* e) {
  if (e == nullptr || *e == '\0') return 0;
  const std::string s = e;
  if (s == "0" || s == "off" || s == "false") return 0;
  if (s == "1" || s == "on" || s == "true" || s == "2") return 1;
  return -1;
}

// The default split layer k (docs/pp-prefill.md 5): 33 without a drafter, 35 with one (it moves the DFlash injection's
// cost off stage B); both keep 8 attention layers per stage. Clamped into [1, layers - 1] for a shorter container.
inline int64_t DefaultSplit(bool dflash, int64_t num_layers) {
  const int64_t k = dflash ? 35 : 33;
  return std::max<int64_t>(1, std::min<int64_t>(k, num_layers - 1));
}

// Whether a Prefill call of `rows` rows goes through the pipeline: only when it is long enough for the fill (one stage-A
// chunk) and the warm-turn sync-back to pay (docs/pp-prefill.md 6.3). `min_rows` 1 pipelines every call (tests).
inline bool ShouldPipeline(int64_t rows, int64_t min_rows) { return rows >= std::max<int64_t>(1, min_rows); }

// HIP_VISIBLE_DEVICES must be "1,0": ordinal 0 of the process is then the physical headless card (decode's, stage B,
// what every existing code path already uses) and ordinal 1 the desktop card (stage A's thread selects it with
// hipSetDevice). Whitespace around the entries is tolerated. Returns "" when fine, else the reason.
inline std::string CheckDeviceOrder(const std::string& hip_visible_devices) {
  std::vector<std::string> parts;
  std::string cur;
  for (const char ch : hip_visible_devices) {
    if (ch == ',') {
      parts.push_back(cur);
      cur.clear();
    } else if (ch != ' ' && ch != '\t') {
      cur.push_back(ch);
    }
  }
  parts.push_back(cur);
  if (parts.size() == 2 && parts[0] == "1" && parts[1] == "0") return "";
  return "pipeline-parallel prefill needs HIP_VISIBLE_DEVICES=1,0 (process ordinal 0 = the headless card that decodes, "
         "ordinal 1 = the desktop card that runs stage A), got '" +
         (hip_visible_devices.empty() ? std::string("<unset>") : hip_visible_devices) + "'";
}

}  // namespace r4dx::model::pp
