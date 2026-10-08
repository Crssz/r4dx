// HIP-free policy of the real pipeline-parallel prefill (docs/pp-prefill.md, Phase 2): what the stage-A mirror
// is known to hold relative to the decode Model (stage B), the switches that turn the pipeline on, the default split
// and the device-placement rule. Header-only so a CPU test (tests/model/test_pp_sync_cpu.cpp)
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

// The default split layer k (docs/pp-prefill.md 5), measured with the corrected placement (stage A on the desktop card, which
// is the slower stage; 2026-10-08, E:\models\r4dx\hybrid_p-1\split and hybrid_p0\dsweep): 29 without a drafter (8k 1.835 s,
// 32k 7.856 s vs 2.049 / 8.744 s at k = 32; stage B idles 76 / 148 ms per call vs 506 / 1918 ms), 30 with one (8k 1.844 s,
// 32k 8.221 s vs 2.243 / 9.620 s at k = 35). Stage A then holds 7 attention layers, stage B 9. Clamped into [1, layers - 1]
// for a shorter container.
inline int64_t DefaultSplit(bool dflash, int64_t num_layers) {
  const int64_t k = dflash ? 30 : 29;
  return std::max<int64_t>(1, std::min<int64_t>(k, num_layers - 1));
}

// Whether a Prefill call of `rows` rows goes through the pipeline: only when it is long enough for the fill (one stage-A
// chunk) and the warm-turn sync-back to pay (docs/pp-prefill.md 6.3). `min_rows` 1 pipelines every call (tests).
inline bool ShouldPipeline(int64_t rows, int64_t min_rows) { return rows >= std::max<int64_t>(1, min_rows); }

// ---- device placement -----------------------------------------------------------------------------------------------
// Which process-visible HIP ordinals the two stages run on. Nothing here depends on HIP_VISIBLE_DEVICES reordering the
// cards (on this ROCm / Windows runtime "1,0" is NOT a reorder: it exposes the same two cards in the natural order), so
// the placement is explicit: `--pp-devices B,A` / R4DX_PP_DEVICES (PpOptions::devices), else the auto rule.
//
// Auto = stage B (decode, the full Model) on the LAST visible ordinal, stage A on the one before it -- the same rule as
// `--tp-devices auto` (docs/tp.md 9.2): with HIP_VISIBLE_DEVICES unset, ordinal 1 = physical device 1 = the headless card
// (pci bus 07) that every TP=1 run of r4dx-cli / r4dx-server uses (the habit HIP_VISIBLE_DEVICES=1), and ordinal 0 = the
// desktop card (pci bus 03) runs stage A. Both cards must be visible: HIP_VISIBLE_DEVICES=1 alone is refused.

// "auto" or "" -> *out cleared (auto placement); "b,a" -> {b, a}. Whitespace around the entries is tolerated. Returns "" when
// fine, else the reason (the caller prefixes the flag / variable name).
inline std::string ParseDevicePair(const std::string& text, std::vector<int>* out) {
  out->clear();
  std::string t;
  for (const char ch : text) {
    if (ch != ' ' && ch != '\t') t.push_back(ch);
  }
  if (t.empty() || t == "auto") return "";
  const std::string what = "expects 'auto' or 'B,A' (two HIP ordinals: stage B = decode card, stage A), got '" + text + "'";
  std::vector<int> v;
  size_t start = 0;
  while (start <= t.size()) {
    const size_t comma = t.find(',', start);
    const std::string item = t.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (item.empty() || item.size() > 4) return what;
    int d = 0;
    for (const char ch : item) {
      if (ch < '0' || ch > '9') return what;
      d = d * 10 + (ch - '0');
    }
    v.push_back(d);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (v.size() != 2) return what;
  if (v[0] == v[1]) return "stage B and stage A must be different ordinals, got '" + text + "'";
  *out = v;
  return "";
}

struct Placement {
  int stage_b = -1;     // process-visible HIP ordinal of the decode Model
  int stage_a = -1;     // ... and of the half-weight prefill stage
  std::string error;    // non-empty: refused, with the reason
  bool Ok() const { return error.empty(); }
};

// `requested`: PpOptions::devices (empty = auto, else {B, A}); `visible`: hipGetDeviceCount; `hip_visible_devices`: the
// environment value, for the message only.
inline Placement ResolvePlacement(const std::vector<int>& requested, int visible, const std::string& hip_visible_devices) {
  Placement p;
  const std::string hvd = hip_visible_devices.empty() ? std::string("<unset>") : hip_visible_devices;
  if (visible < 2) {
    p.error = "pipeline-parallel prefill needs two visible HIP devices (decode card + desktop card), but HIP_VISIBLE_DEVICES=" +
              hvd + " exposes " + std::to_string(std::max(visible, 0)) + ". Unset it (or expose both cards).";
    return p;
  }
  if (requested.empty()) {
    p.stage_b = visible - 1;
    p.stage_a = visible - 2;
    return p;
  }
  if (requested.size() != 2) {
    p.error = "pipeline-parallel prefill needs exactly two ordinals (stage B, stage A), got " + std::to_string(requested.size());
    return p;
  }
  const int b = requested[0], a = requested[1];
  if (b < 0 || a < 0 || b >= visible || a >= visible) {
    p.error = "pipeline-parallel prefill devices " + std::to_string(b) + "," + std::to_string(a) +
              " are not both inside the " + std::to_string(visible) + " visible HIP devices (HIP_VISIBLE_DEVICES=" + hvd + ")";
    return p;
  }
  if (a == b) {
    p.error = "pipeline-parallel prefill needs two different devices, got " + std::to_string(b) + "," + std::to_string(a);
    return p;
  }
  p.stage_b = b;
  p.stage_a = a;
  return p;
}

}  // namespace r4dx::model::pp
