// r4dx::model::DebugProbe -- two debug-only measurements of a real decode
// (docs/trellis-kernel.md 6 "Benches" mode 5, 7 M5), off unless an environment variable turns them
// on:
//
//   R4DX_CLOCK_PROBE=1      The in-model clock probe (Q-g3): after every Mlp of the layer stack
//                           (RunChunk, VerifyWindow) one wave runs 8 x 400 dependent v_add between
//                           clock64() / wall_clock64() reads -- tool_trellis_gemm_bench's probe,
//                           the same device code (kernels/clock_probe_device.h) at the bench's
//                           default --probe-iters -- so the sustained shader clock inside the
//                           model can be set against the replay's. It also records the round
//                           composition of MTP and DFlash (Q-g8): per round the drafter, the
//                           verify window and (DFlash) the feature injection, each as a GPU span,
//                           and the rest of the round's host wall time.
//   R4DX_PROFILE_LINEARS=1  Per-linear-class GPU time: wall_clock64 stamps around every "gemm:"
//                           span of the layer code (profile_span.h's ProfiledCall with no hipEvent
//                           profiler attached: every body linear, the bf16 gdn.in_proj_a/b, and
//                           FinalLmHead's lm_head) and one inside ApplyLinear between the input
//                           stage (w4a16's f16 cast, trellis's input transform) and the GEMM;
//                           summed per (call kind, rows, class, layer). `all` stamps every
//                           ProfiledCall span, not only the "gemm:" ones.
//   R4DX_PROBE_JSON=<file>  Also write everything as JSON at exit (otherwise a summary on stderr
//                           only; every summary line starts with "[probe]").
//   R4DX_PROBE_TIMELINE=<file>  Also write every span as it is collected, one CSV line each: call
//                           kind, T, call serial, span name, layer, start (us after the call's
//                           first stamp), duration (us). What found docs/trellis-kernel.md 10.6's
//                           pattern: which layers' GEMMs ran slow, step after step.
//
// Every model call that runs the layer stack is a "call" with its own GPU span (a stamp before its
// first kernel and one after its last) and host times: prefill (a RunChunk prefill chunk), decode
// (a RunChunk decode step), verify (VerifyWindow), mtp_prime (RunChunk's MTP KV priming), and the
// speculative rounds mtp_round / dflash_round with their mtp_draft / dflash_draft / dflash_inject
// parts. Calls nest; the clock probes and spans are attributed to the innermost one.
//
// TIMING is the bench's (tool_trellis_gemm_bench.cpp's Timer): one-thread kernels write
// wall_clock64() -- hip events proved unreliable for short intervals on this stack -- the REALTIME
// rate is calibrated against host time when the probe starts (Get()'s first call, which
// Model::Load makes once the device is idle), and every interval has the empty bracket's cost
// subtracted. A stamp between two kernels adds one launch to the timeline, so a profiled step runs
// somewhat slower than an unprofiled one; the per-class times are what the kernels take, not what
// the probe adds.
//
// Zero cost when unset: Model holds a null DebugProbe* (one test per hook), and ProfiledCall and
// ApplyLinear read one static pointer. TP = 1 only: a tensor-parallel rank never starts it. Stamps
// are read back only where the model has already synchronized its stream (Collect), so apart from
// the start-up calibration the probe adds no synchronize. The summary is printed and the JSON
// written at normal process exit; calls whose stamps were never read back by then (a DFlash round
// ends without a synchronize, so the last one) are counted as unfinished and left out. Not
// thread-safe: one model on one thread, like Model itself.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace r4dx::model {

class DebugProbe {
 public:
  // nullptr unless R4DX_CLOCK_PROBE or R4DX_PROFILE_LINEARS is set. The first call reads the
  // environment once, allocates the probe's device ring and calibrates the clocks (a private
  // stream, synchronized: call it where the device is idle).
  static DebugProbe* Get();
  // The started probe when R4DX_PROFILE_LINEARS is on, else nullptr: the one static pointer
  // ProfiledCall and ApplyLinear test.
  static DebugProbe* Linears() { return linears_; }

  // ---- calls (host-side stack) ------------------------------------------------------------------
  // Begin stamps the call's start on `s` and returns its id; End(s) stamps its end -- call it after
  // the call's last kernel and before its stream synchronize. `host_done`: the call returns now (no
  // synchronize follows, e.g. a DFlash round); otherwise its host end is taken at the next Collect.
  int Begin(hipStream_t s, const char* kind, int64_t T);
  void End(hipStream_t s, bool host_done = false);
  // Drops call `id` (and anything opened inside it) unreported: a call left by an exception.
  void Abandon(int id);
  // Tokens the innermost open call committed (the speculative rounds).
  void SetCommitted(int64_t tokens);
  // The layer the next clock probe and spans belong to (-1: outside the layer stack).
  void SetLayer(int64_t layer) { layer_ = layer; }
  // After a layer's Mlp: one clock probe, when R4DX_CLOCK_PROBE is on.
  void AfterMlp(hipStream_t s);
  // `s` must hold every stamp so far and be synchronized (or about to be: Collect synchronizes it,
  // which is free where the caller just did). Reads the new stamps back and aggregates every
  // finished record.
  void Collect(hipStream_t s);

  // ---- spans (profile_span.h's ProfiledCall; ApplyLinear) --------------------------------------
  bool WantsSpan(const char* name) const;
  void BeginSpan(hipStream_t s, const char* name);
  // ApplyLinear, after its input stage and before its GEMM (first chunk only).
  void MarkInput(hipStream_t s);
  void EndSpan(hipStream_t s);

  DebugProbe(const DebugProbe&) = delete;
  DebugProbe& operator=(const DebugProbe&) = delete;
  ~DebugProbe();  // the summary (stderr) and R4DX_PROBE_JSON; no HIP call

 private:
  DebugProbe(bool clock, int linears, std::string json_path);
  void Start();       // device ring, calibration
  uint64_t Alloc(uint64_t n);
  uint64_t Stamp(hipStream_t s);
  bool Valid(uint64_t slot) const { return slot < copied_ && slot + cap_ >= copied_; }
  double Ms(uint64_t from, uint64_t to) const;   // stamp to stamp, less the empty bracket
  void Report() const;                           // stderr
  void WriteJson(const std::string& path) const;

  struct Call {
    std::string kind;
    int64_t T = 0;
    uint64_t serial = 0;
    int parent = -1;
    uint64_t b = 0, e = 0;
    double host_b = 0, host_e = 0;  // steady-clock ms; host_e < 0: taken at the next Collect
    bool closed = false;
    int64_t committed = 0;
    std::map<std::string, double> child_ms;  // children's GPU ms by kind
    std::map<std::string, int64_t> child_T;  // the last child's T by kind
  };
  struct Span {
    int call;
    const char* name;  // ProfiledCall's literal
    int64_t layer;
    uint64_t b, mid, e;
    bool has_mid;
  };
  struct ProbeRec {
    int call;
    int64_t layer;
    uint64_t slot;
  };

  // Aggregates, by (call kind, T).
  using Key = std::pair<std::string, int64_t>;
  struct CallAgg {
    std::vector<double> gpu_ms, host_ms, gap_ms;
  };
  struct RoundRow {
    double host_ms, gap_ms, drafter_ms, verify_ms, inject_ms;
    int64_t verify_T, committed;
  };
  struct SpanAgg {
    int64_t n = 0;
    double in_ms = 0, gemm_ms = 0, total_ms = 0;
    int64_t with_input = 0;
    std::vector<double> total_us;
    std::map<int64_t, std::pair<int64_t, double>> by_layer;  // layer -> (n, total ms)
  };
  struct ClockSample {
    uint64_t serial;
    int64_t layer;
    double mhz, us, cycles_per_add;
  };

  bool clock_;
  int linears_mode_;  // 0 off, 1 "gemm:" spans, 2 every span
  std::string json_path_;
  FILE* timeline_ = nullptr;  // R4DX_PROBE_TIMELINE
  int probe_iters_ = 400;

  unsigned long long* dev_ = nullptr;  // the stamp ring (never freed: see ~DebugProbe)
  uint64_t cap_ = uint64_t{1} << 16;
  uint64_t next_ = 0, copied_ = 0;
  std::vector<unsigned long long> mirror_;

  // Calibration.
  double wall_mhz_ = 100.0, wall_mhz_attr_ = 100.0;
  double bracket_ms_ = 0.0;
  double idle_probe_mhz_ = 0.0, idle_probe_us_ = 0.0, idle_cycles_per_add_ = 0.0;

  int64_t layer_ = -1;
  uint64_t serial_ = 0;
  std::vector<Call> calls_;  // pool; cleared whenever nothing is open or pending
  std::vector<int> stack_, closed_;
  std::vector<Span> spans_;
  std::vector<ProbeRec> probes_;
  bool span_open_ = false;
  Span cur_{};
  double last_top_done_ = -1.0;
  std::string last_top_kind_;

  std::map<Key, CallAgg> call_agg_;
  std::map<Key, std::vector<RoundRow>> rounds_;
  std::map<std::tuple<std::string, int64_t, std::string>, SpanAgg> span_agg_;
  std::map<Key, std::vector<ClockSample>> clock_agg_;
  int64_t dropped_ = 0, abandoned_ = 0, overflows_ = 0;

  inline static DebugProbe* linears_ = nullptr;
};

// Begin/End as a scope: End() where the call's GPU work ends; a scope left without End() (an
// exception) abandons its call. Inert with a null probe.
class ProbeScope {
 public:
  ProbeScope(DebugProbe* p, hipStream_t s, const char* kind, int64_t T)
      : p_(p), s_(s), id_(p != nullptr ? p->Begin(s, kind, T) : -1) {}
  void End(bool host_done = false) {
    if (p_ != nullptr) p_->End(s_, host_done);
    p_ = nullptr;
  }
  ~ProbeScope() {
    if (p_ != nullptr) p_->Abandon(id_);
  }
  ProbeScope(const ProbeScope&) = delete;
  ProbeScope& operator=(const ProbeScope&) = delete;

 private:
  DebugProbe* p_;
  hipStream_t s_;
  int id_;
};

}  // namespace r4dx::model
