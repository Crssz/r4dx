// r4dx::model::tp::SubmitBounder -- bounded GPU submission for tensor parallelism on a desktop card
// (docs/tp.md Appendix B N56, N57; Appendix C question 2).
//
// HIP device 0 drives the display and cannot preempt compute (ComputePreemptionSupported = 0). The
// user keeps the display there, so TP keeps every GPU submission short and the work queued ahead of
// the GPU small instead. What that does and does not buy (docs/tp.md Appendix B N64): it bounds
// where submissions happen and how much work is queued; only K == 1 (below) also leaves the GPU
// idle for a moment at every unit, while the host enqueues the next one. It is NOT by itself
// evidence that a long device-0 run is TDR-safe: P3's decode-pattern stress, which TDR'd twice in
// ~200 s (N44), already ran inside tighter bounds (<= 129-command submissions, ~50 ms queued) with
// the GPU never idle. Long device-0 runs go through tools/tp/tdr_watch.psm1, which stops the run at
// the first TDR.
//
// What the runtime does (tests/kernels/tool_tp_submit_probe.cpp, N56, both cards): a kernel launch
// is NOT submitted when it is enqueued. The runtime submits on its own only every 129 commands; the
// rest waits -- indefinitely (> 200 ms measured with no HIP call; 700 ms in N43) -- until
// hipEventRecord, hipEventQuery or a synchronize. A hipMemcpyAsync or hipMemsetAsync does not submit.
// GPU_FLUSH_ON_EXECUTION=1 submits every launch instead, at ~26-50 us of GPU time per submission.
// And ONE hipEventRecord on a stream -- timing or not, even after the event is destroyed -- makes
// every later dispatch on that stream ~0.39 us slower on the GPU for the stream's whole life; a
// synchronize, a copy or a memset does not (N56 case G). Model's stream_ therefore never sees an
// event record in the default configuration.
//
// A SubmitBounder marks the end of each UNIT of work its caller enqueues on one stream (N57, N64:
// every UnitLayersForContext(`TpRankOptions::submit_layers`, chunk end) layers of a prefill chunk;
// every `--flush-every` all-reduces in tool_tp_ar_stress). EndUnit(stream) forces the runtime to
// submit the unit now, so the GPU never has more than K units queued (before unit i is enqueued,
// unit i-K has finished):
//   K == 1 (the default under TP)  hipStreamSynchronize: submits and waits for everything so far,
//                                  with no event, so the stream stays untainted. The GPU then
//                                  idles until the host has enqueued (and the runtime submitted)
//                                  the next unit -- the only mode with an idle gap per unit.
//   K == 0 or K >= 2               records the next of a ring of hipEventDisableTiming events and
//                                  queries it once (the runtime submits), then -- K >= 2 -- polls
//                                  (30 s watchdog, never hipStreamQuery: N33) until at most K-1
//                                  recorded units are unfinished. K == 0 is the forced submission
//                                  alone. Both taint the stream (above): the price of K != 1. And
//                                  neither idles the GPU: K >= 2 keeps a unit queued at all times.
// The events are created on the current device at construction (never inside a collective command:
// docs/tp.md 6.3.7) and destroyed with the object, on the thread that owns it (a TP rank thread). A
// default-constructed SubmitBounder is inert and makes no HIP call, so a TP=1 Model carries one at
// no cost.
#pragma once

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <vector>

namespace r4dx::model::tp {

// Polls hipEventQuery(ev) until it reports complete (true) or `limit` passes (false): yields for the
// first 200,000 polls, then sleeps 1 ms at a time (tp_bench's wait_event_wd; docs/tp.md Appendix B
// N33). Clears the thread's last HIP error (hipErrorNotReady); throws core::HipError on any other.
bool WaitEventWithWatchdog(hipEvent_t ev, std::chrono::milliseconds limit);

// The unit size, in layers, of a prefill chunk that ends at position `ctx_end` (docs/tp.md
// Appendix B N64): `submit_layers` (0 = off) up to 16k context, then at most 16, 8 and 4 layers up
// to 64k, 128k and beyond. Estimated (not measured past 4k): a 64-row chunk costs each rank ~40 ms
// plus ~1.4 ms per 1,000 positions of context -- N57's 2k chunk plus the attention slope of
// docs/perf.md's TP=1 prefill curve, which TP does not split (8.6) -- so every unit stays at
// <= ~33 ms of GPU time up to 262k, where a fixed 32 layers would queue ~200 ms. Both ranks run the
// same positions (lockstep, H1), so they cut the same units.
inline int64_t UnitLayersForContext(int64_t submit_layers, int64_t ctx_end) {
  if (submit_layers <= 0) return 0;
  const int64_t cap = ctx_end <= 16384 ? submit_layers : ctx_end <= 65536 ? 16 : ctx_end <= 131072 ? 8 : 4;
  return submit_layers < cap ? submit_layers : cap;
}

class SubmitBounder {
 public:
  struct Stats {
    uint64_t units = 0;       // EndUnit calls (== forced submissions)
    uint64_t waits = 0;       // EndUnit calls that waited for the cap (K == 1: every one)
    double wait_us_total = 0;
    double wait_us_max = 0;
  };

  SubmitBounder() = default;  // inert: Active() false, no events, no HIP call
  // max_inflight K >= 0 (0 = no cap). K == 1 needs no events; otherwise creates max(K, 1) + 1
  // hipEventDisableTiming events on the current device.
  explicit SubmitBounder(int max_inflight);
  ~SubmitBounder();
  SubmitBounder(SubmitBounder&& o) noexcept;
  SubmitBounder& operator=(SubmitBounder&& o) noexcept;
  SubmitBounder(const SubmitBounder&) = delete;
  SubmitBounder& operator=(const SubmitBounder&) = delete;

  bool Active() const { return active_; }
  int MaxInflight() const { return cap_; }

  // The caller has just enqueued one unit on `stream` (one stream per SubmitBounder between two
  // Reset()s): submit it and wait for the cap (above). K != 1: throws core::TpTimeoutError if a unit
  // is still unfinished after 30 s.
  void EndUnit(hipStream_t stream);
  // Every recorded unit is known finished (the caller synchronized the stream) or abandoned (an
  // exception): forget them. Host-only.
  void Reset() { recorded_ = complete_ = 0; }

  const Stats& GetStats() const { return stats_; }
  void ResetStats() { stats_ = Stats{}; }

 private:
  void Destroy() noexcept;
  void NoteWait(std::chrono::steady_clock::time_point t0);

  bool active_ = false;
  std::vector<hipEvent_t> ring_;  // K != 1 only
  int cap_ = 0;
  uint64_t recorded_ = 0;  // units recorded since Reset()
  uint64_t complete_ = 0;  // of those, known finished
  Stats stats_;
};

}  // namespace r4dx::model::tp
