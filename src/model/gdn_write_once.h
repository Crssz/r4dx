// r4dx::model write-once GDN state (docs/gdn-write-once.md): R4DX_GDN_WRITE_ONCE and the decision of when a
// Model keeps ONE recurrent state per sequence plus a per-row log of the speculative verify window instead of
// one state slot per candidate row.
//
// Header-only and free of HIP so the parser and the decision have a CPU unit test
// (tests/model/test_gdn_write_once_cpu.cpp); Model::Load is the only caller.
//
// R4DX_GDN_WRITE_ONCE (read once per process, like R4DX_PREFILL_INT8). ON BY DEFAULT since the GPU gates of the
// document passed on 2026-10-08 (E:\models\r4dx\gdnwo; kGdnWriteOnceDefault below is the one line that flips it):
//   - "1" or "on": the write-once state, where this Model can use it (DecideGdnWriteOnce): a speculative verify
//     call stores a 0.26 MiB-per-layer log of its rows and one state per layer, instead of one 3 MiB state per
//     candidate row; the accepted prefix of the log is applied inside the NEXT call's seed load. Every output is
//     bit-identical to the window-slot path (logits, tokens, the materialised state); the window-slot VRAM
//     (about 0.96 GiB per rank at k = 7, TP = 1) is not allocated;
//   - "0" or "off": the window-slot path, today's behaviour byte for byte (a build without the feature);
//   - anything else: a warning on stderr, then the default.
// R4DX_DECODE_LEGACY=gdnwo is the same kill switch through the decode-legacy mask (core/decode_legacy.hpp): it
// forces the window-slot path when ModelOptions::gdn_write_once follows the environment, whatever
// R4DX_GDN_WRITE_ONCE says -- the switch that stays meaningful after the default flips.
// ModelOptions::gdn_write_once = 0 / 1 forces the choice whatever the environment says (the A/B tests load a Model
// of each in one process).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace r4dx::model {

// What an unset / empty / unreadable R4DX_GDN_WRITE_ONCE means. Flip to true to make the write-once state the
// default (the kill switches above keep restoring the window-slot path).
inline constexpr bool kGdnWriteOnceDefault = true;

// "1" / "on" -> true, "0" / "off" -> false, unset / empty -> `dflt`; anything else warns and is `dflt`.
inline bool ParseGdnWriteOnce(const char* e, bool dflt = kGdnWriteOnceDefault, bool warn = true) {
  if (e == nullptr || *e == '\0') return dflt;
  if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) return true;
  if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return false;
  if (warn) {
    std::fprintf(stderr, "r4dx: R4DX_GDN_WRITE_ONCE='%s' not recognized (0|off|1|on); using the default (%s)\n", e,
                 dflt ? "on" : "off");
  }
  return dflt;
}
inline bool GdnWriteOnceRequest() {
  static const bool v = ParseGdnWriteOnce(std::getenv("R4DX_GDN_WRITE_ONCE"));
  return v;
}

inline bool ValidGdnWriteOnceOption(int opt) { return opt == -1 || opt == 0 || opt == 1; }

struct GdnWriteOnceInputs {
  int option = -1;                 // ModelOptions::gdn_write_once: -1 follows the environment, 0 / 1 force
  bool env_request = false;        // GdnWriteOnceRequest()
  bool decode_legacy_gdnwo = false;  // R4DX_DECODE_LEGACY has the gdnwo token (or "all")
  int64_t window = 1;              // 1 + max(mtp_draft_k, dflash_draft_k): the speculative verify window
};

// Whether this Model runs the write-once state, and why not when it does not. A window of 1 (no speculation)
// has no per-row states to save: the manager is the window-slot one with a single slot either way, and the
// request is moot.
inline bool DecideGdnWriteOnce(const GdnWriteOnceInputs& in, const char** why = nullptr) {
  auto no = [&](const char* r) {
    if (why != nullptr) *why = r;
    return false;
  };
  if (in.window <= 1) return no("no speculative window (nothing to save)");
  if (in.option == 0) return no("ModelOptions::gdn_write_once = 0");
  if (in.option == 1) return true;
  if (in.decode_legacy_gdnwo) return no("R4DX_DECODE_LEGACY has gdnwo");
  if (!in.env_request) return no("R4DX_GDN_WRITE_ONCE is off");
  return true;
}

// ---- the host-side pieces of the design, pure so a CPU test can pin them ---------------------------------

// Slot arithmetic of one layer's recurrent allocation. Slot 0 is NULL_BLOCK_ID (the kernels skip a slot <= 0).
// Window-slot path: `window` slots per sequence, one per candidate row of a verify call. Write-once: ONE slot per
// sequence (the state B); `window` is then only the depth of the logs and of the conv history.
struct GdnSlotPlan {
  int64_t max_seqs = 1;
  int64_t window = 1;
  bool write_once = false;

  int64_t SlotsPerSeq() const { return write_once ? 1 : window; }
  int64_t RecurrentSlots() const { return max_seqs * SlotsPerSeq() + 1; }
  int32_t SlotForSeq(int32_t seq) const { return 1 + seq * static_cast<int32_t>(SlotsPerSeq()); }
  // True when `slot` is a slot this plan allocates (slot 0, the null block, is not).
  bool ValidSlot(int64_t slot) const { return slot >= 1 && slot < RecurrentSlots(); }
};

// The two log buffers of a layer: the call that replays the previous log writes the other half.
// In() is the log of the last launched T > 1 call (the one a pending commit refers to), Out() the one the next
// T > 1 call writes; Flip() runs once per such launch.
struct GdnLogRing {
  int parity = 0;
  int In() const { return parity; }
  int Out() const { return parity ^ 1; }
  void Flip() { parity ^= 1; }
};

// Floats in one log buffer for a window of `depth` rows (libr4d's r4d_gdn_wo_log_bytes / 4, which a test checks
// against the kernel's own answer): u [depth][H][V], kk [depth][Hg][K], eg [depth][H], rounded up to 64 floats.
inline int64_t GdnLogFloats(int64_t H, int64_t Hg, int64_t V, int64_t K, int64_t depth) {
  const int64_t f = depth * H * V + depth * Hg * K + depth * H;
  return (f + 63) / 64 * 64;
}

// Whether the logical GDN state of a write-once Model is "B plus the first n rows of the last verify call's
// log" (pending) or just B -- the whole rollback story (docs/gdn-write-once.md 2.1). `n` is the count the
// commit wrote to mtp_num_accepted_dev_, the Model's existing acceptance thread, which the kernels read as the
// replay length.
//
//   event                                          pending after
//   Prefill / Reset / RestoreCheckpoint            false  (a pending state was materialised first)
//   plain decode, T == 1                           false  (the kernel replayed it and wrote B)
//   verify call launched (any T)                   false  (it replayed the previous log into B; its own
//                                                          rows are not committed yet)
//   commit n after a verify with T > 1             true, count n
//   commit n after a verify with T == 1            false  (the T == 1 kernel stored the state directly)
//
// Rejecting draft rows is "do not mark them pending": nothing is rolled back and nothing is copied.
class GdnPendingBook {
 public:
  bool Pending() const { return pending_; }
  int64_t LastVerifyT() const { return last_verify_t_; }
  // The replay length of the pending state (valid while Pending()).
  int64_t Count() const { return count_; }

  void OnVerifyLaunched(int64_t T) {
    pending_ = false;
    last_verify_t_ = T;
    count_ = 0;
  }
  // Returns false when `n` rows cannot have been logged by the last verify call (a host bug: the legacy path
  // would read a window slot that call never wrote).
  bool OnCommit(int64_t n) {
    if (n < 1 || n > last_verify_t_) return false;
    pending_ = last_verify_t_ > 1;
    count_ = n;
    return true;
  }
  void OnPlainDecode() { Clear(); }
  // After the state was materialised into B (a flush, or a Prefill's collapse), or discarded (Reset, Restore).
  void Clear() {
    pending_ = false;
    last_verify_t_ = 1;
    count_ = 0;
  }

 private:
  bool pending_ = false;
  int64_t last_verify_t_ = 1;
  int64_t count_ = 0;
};

}  // namespace r4dx::model
