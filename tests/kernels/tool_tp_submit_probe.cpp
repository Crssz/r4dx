// tests/kernels/tool_tp_submit_probe.cpp -- when does the HIP runtime actually SUBMIT an enqueued
// kernel to the GPU? (docs/tp.md Appendix B N56: the evidence behind the device-0 submission
// bounding of N57.)
//
// A marker kernel (r4dx_tp_test_marker) release-stores a value into mapped pinned host memory when it
// STARTS; the host polls that memory with plain loads -- no HIP call -- so it sees when a kernel
// really began running, not when some HIP call reported it complete. Each experiment enqueues, then
// observes for --observe-ms with no HIP call at all, then (if the marker never started) flushes with
// one event query and drains with an event-polled wait (never hipStreamQuery, N33):
//
//   A  marker alone                                   does a lone launch start by itself?
//   B  marker + hipEventRecord                        does recording an event submit?
//   C  marker + hipEventRecord + ONE hipEventQuery    does one query submit? (and how fast)
//   D  ~5 ms busy kernel + marker + record + 1 query  the whole chain runs in order after one query
//   E  --markers markers with value = index, the host polling after every launch: the launch
//      indices at which the runtime submits on its own (auto-flush points), then the tail's fate
//   E2 the same, paced by a 20 us host spin per launch so the GPU is idle at every submit: the
//      observed value jumps straight to each auto-submit's launch index (the batch size)
//   F  flush cost: --units units of (a ~0.2 ms busy kernel + 8 tiny kernels), an event record + one
//      query after every Nth unit, N in {0 = never, 1, 4, 16}: GPU time between two timing events
//      (hipEventElapsedTime) and host time, min of three interleaved rounds
//   G  does a stream stay slower once an event was recorded on it? Fresh streams, 4096 tiny kernels
//      and a marker carried through an auto-submit (no HIP wait): launch-to-marker time on a stream
//      that never saw an event vs after one DisableTiming / timing event record (and after one
//      whose event was then destroyed), a synchronize, a D2H copy, a memset; min of three rounds
//   H  does a small hipMemcpyAsync / hipMemsetAsync submit the marker queued before it?
//
// ONE device: whatever HIP_VISIBLE_DEVICES exposes as ordinal --device (default 0). Every experiment
// queues at most ~15 ms of GPU work, so it is safe on the desktop card (device 0) too. Prints the
// GPU_FLUSH_ON_EXECUTION environment variable (a ROCclr setting, read at runtime start) so the two
// runs -- with and without it -- are told apart in the log. Built, never add_test()'d.
//
//   tool_tp_submit_probe.exe [--device 0] [--observe-ms 200] [--markers 2048] [--units 64]
//                            [--json path]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/tp_kernels.h"
#include "tp/tp_group.h"
#include "tp_ar_harness.h"

using namespace tp_harness;

namespace {

using Clock = std::chrono::steady_clock;

double UsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Args {
  int device = 0;
  int observe_ms = 200;
  int markers = 2048;
  int units = 64;
  std::string json;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "%s\nusage: tool_tp_submit_probe.exe [--device N] [--observe-ms N] [--markers N] [--units N] "
               "[--json path]\n",
               why.c_str());
  std::exit(1);
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(s + " needs a value");
      return argv[++i];
    };
    try {
      if (s == "--device") a.device = std::stoi(next());
      else if (s == "--observe-ms") a.observe_ms = std::stoi(next());
      else if (s == "--markers") a.markers = std::stoi(next());
      else if (s == "--units") a.units = std::stoi(next());
      else if (s == "--json") a.json = next();
      else Usage("unknown option " + s);
    } catch (const std::exception&) {
      Usage("bad value for " + s);
    }
  }
  if (a.observe_ms < 1 || a.observe_ms > 2000) Usage("--observe-ms must be in [1, 2000]");
  if (a.markers < 1 || a.markers > 8192) Usage("--markers must be in [1, 8192]");
  if (a.units < 1 || a.units > 256) Usage("--units must be in [1, 256]");
  return a;
}

// The host's view of the marker words: [0] value (release-stored last), [1] device wall clock.
struct Mailbox {
  volatile uint64_t* host = nullptr;
  int64_t dev = 0;
  uint64_t Value() const { return __atomic_load_n(host, __ATOMIC_ACQUIRE); }
  void Clear() {
    __atomic_store_n(host, 0ull, __ATOMIC_RELEASE);
    host[1] = 0;
  }
};

// Polls until the marker value reaches `want` or `ms` pass, with plain loads only. Returns the
// microseconds from `t0` at which it was seen, or -1.
double PollFor(const Mailbox& mb, uint64_t want, Clock::time_point t0, int ms) {
  const auto end = t0 + std::chrono::milliseconds(ms);
  for (;;) {
    if (mb.Value() >= want) return UsSince(t0);
    if (Clock::now() >= end) return -1.0;
    _mm_pause();
  }
}

// One query of `ev`; clears the thread's last error (hipErrorNotReady).
void QueryOnce(hipEvent_t ev) {
  const hipError_t e = hipEventQuery(ev);
  if (e != hipSuccess && e != hipErrorNotReady) {
    (void)hipGetLastError();
    throw r4dx::core::HipError(e, "hipEventQuery", __FILE__, __LINE__);
  }
  (void)hipGetLastError();
}

void Drain(hipStream_t st, hipEvent_t ev) {
  if (!tp::SyncWithWatchdogEvent(st, ev)) DieStuck("submit probe drain");
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  const std::string flush_env = GetEnv("GPU_FLUSH_ON_EXECUTION");
  if (a.device < 0 || a.device >= VisibleDevices()) {
    std::fprintf(stderr, "HIP device %d is not visible (%d visible)\n", a.device, VisibleDevices());
    return 1;
  }
  R4DX_HIP_CHECK(hipSetDevice(a.device));
  std::printf("[probe] %s, GPU_FLUSH_ON_EXECUTION=%s, observe %d ms\n", DeviceLabel(a.device).c_str(),
              flush_env.empty() ? "(unset)" : flush_env.c_str(), a.observe_ms);

  void* h = nullptr;
  R4DX_HIP_CHECK(hipHostMalloc(&h, 4096, hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
  void* dp = nullptr;
  R4DX_HIP_CHECK(hipHostGetDevicePointer(&dp, h, 0));
  Mailbox mb;
  mb.host = static_cast<volatile uint64_t*>(h);
  mb.dev = reinterpret_cast<int64_t>(dp);
  mb.Clear();

  r4dx::core::Stream stream;
  const hipStream_t st = stream.get();
  const int64_t s64 = reinterpret_cast<int64_t>(st);
  r4dx::core::DeviceBuffer<uint64_t> scratch(2);
  const int64_t scratch64 = reinterpret_cast<int64_t>(scratch.data());
  hipEvent_t ev = nullptr, t0e = nullptr, t1e = nullptr;
  R4DX_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  R4DX_HIP_CHECK(hipEventCreate(&t0e));
  R4DX_HIP_CHECK(hipEventCreate(&t1e));

  // ~8.3 us per iteration of r4dx_tp_clock_probe's s_sleep loop (tp_group.cpp's CheckWallClockRate).
  const auto busy = [&](int iters) { r4dx_tp_clock_probe(scratch64, iters, s64); };

  // Warm-up: first touch of both code objects, then an idle stream.
  r4dx_tp_test_marker(mb.dev, 1, s64);
  busy(1);
  Drain(st, ev);
  mb.Clear();

  std::string json = "{\n  \"device\": " + JsonStr(DeviceLabel(a.device)) + ",\n  \"gpu_flush_on_execution\": " +
                     JsonStr(flush_env) + ",\n  \"observe_ms\": " + std::to_string(a.observe_ms) + ",\n";

  // ---- A / B / C: one marker; nothing / an event record / a record + one query ------------------
  const char* names[3] = {"A marker alone", "B marker + hipEventRecord", "C marker + record + 1 hipEventQuery"};
  const char* keys[3] = {"A", "B", "C"};
  uint64_t v = 1;
  for (int x = 0; x < 3; ++x) {
    ++v;
    r4dx_tp_test_marker(mb.dev, v, s64);
    const auto t0 = Clock::now();
    if (x >= 1) R4DX_HIP_CHECK(hipEventRecord(ev, st));
    if (x == 2) QueryOnce(ev);
    const double seen = PollFor(mb, v, t0, a.observe_ms);
    double after_query = -1.0;
    if (seen < 0) {
      // Never started on its own: flush with one query and time that.
      if (x == 0) R4DX_HIP_CHECK(hipEventRecord(ev, st));
      const auto tq = Clock::now();
      QueryOnce(ev);
      after_query = PollFor(mb, v, tq, 2000);
    }
    Drain(st, ev);
    if (seen >= 0) {
      std::printf("[probe] %-40s started %.1f us after the launch returned\n", names[x], seen);
    } else {
      std::printf("[probe] %-40s NOT started after %d ms; one hipEventQuery then started it in %.1f us\n", names[x],
                  a.observe_ms, after_query);
    }
    json += std::string("  \"") + keys[x] + "\": {\"started_us\": " + JsonNum(seen) +
            ", \"after_one_query_us\": " + JsonNum(after_query) + "},\n";
  }

  // ---- D: a ~5 ms busy kernel, then the marker; record + ONE query ------------------------------
  {
    ++v;
    busy(600);
    r4dx_tp_test_marker(mb.dev, v, s64);
    R4DX_HIP_CHECK(hipEventRecord(ev, st));
    const auto t0 = Clock::now();
    QueryOnce(ev);
    const double seen = PollFor(mb, v, t0, a.observe_ms);
    Drain(st, ev);
    std::printf("[probe] %-40s marker started %.1f us after the query (busy kernel ~5 ms ahead of it)\n",
                "D busy(~5 ms) + marker + 1 query", seen);
    json += "  \"D\": {\"marker_after_query_us\": " + JsonNum(seen) + "},\n";
  }

  // ---- E: markers 1..N, polled after every launch -------------------------------------------------
  {
    mb.Clear();
    struct Transition {
      int launch;       // launches enqueued when the change was seen (1-based)
      uint64_t seen;    // highest marker value observed running
      double us;        // since the first launch
    };
    std::vector<Transition> tr;
    uint64_t last = 0;
    const auto t0 = Clock::now();
    for (int i = 1; i <= a.markers; ++i) {
      r4dx_tp_test_marker(mb.dev, static_cast<uint64_t>(i), s64);
      const uint64_t now_v = mb.Value();
      if (now_v != last) {
        tr.push_back({i, now_v, UsSince(t0)});
        last = now_v;
      }
    }
    const double enqueue_us = UsSince(t0);
    // The tail: observe with no HIP call.
    const auto te = Clock::now();
    const double tail_seen = PollFor(mb, static_cast<uint64_t>(a.markers), te, a.observe_ms);
    const uint64_t tail_value = mb.Value();
    R4DX_HIP_CHECK(hipEventRecord(ev, st));
    const auto tq = Clock::now();
    QueryOnce(ev);
    const double tail_after_query = tail_seen >= 0 ? -1.0 : PollFor(mb, static_cast<uint64_t>(a.markers), tq, 2000);
    Drain(st, ev);
    std::printf("[probe] E %d markers enqueued in %.0f us (%.2f us/launch); %zu auto-submit point(s) seen:\n",
                a.markers, enqueue_us, enqueue_us / a.markers, tr.size());
    for (size_t k = 0; k < tr.size() && k < 24; ++k) {
      std::printf("[probe]     at launch %5d: markers up to %5llu running (%.0f us)\n", tr[k].launch,
                  static_cast<unsigned long long>(tr[k].seen), tr[k].us);
    }
    if (tr.size() > 24) std::printf("[probe]     ... %zu more\n", tr.size() - 24);
    if (tail_seen >= 0) {
      std::printf("[probe]   the tail (marker %d) started on its own %.1f us after the last launch\n", a.markers,
                  tail_seen);
    } else {
      std::printf("[probe]   the tail: after %d ms with no HIP call only markers up to %llu had run; one "
                  "hipEventQuery ran marker %d %.1f us later\n",
                  a.observe_ms, static_cast<unsigned long long>(tail_value), a.markers, tail_after_query);
    }
    json += "  \"E\": {\"markers\": " + std::to_string(a.markers) + ", \"enqueue_us\": " + JsonNum(enqueue_us) +
            ", \"tail_started_us\": " + JsonNum(tail_seen) + ", \"tail_value_before_query\": " +
            std::to_string(tail_value) + ", \"tail_after_one_query_us\": " + JsonNum(tail_after_query) +
            ", \"transitions\": [";
    for (size_t k = 0; k < tr.size(); ++k) {
      json += (k ? ", " : "") + std::string("[") + std::to_string(tr[k].launch) + ", " + std::to_string(tr[k].seen) +
              ", " + JsonNum(tr[k].us) + "]";
    }
    json += "]},\n";
  }

  // ---- E2: the same, paced (a 20 us host spin after every launch), so the GPU is idle whenever the
  // runtime submits: the observed value then jumps straight to the launch index of each auto-submit.
  {
    mb.Clear();
    std::vector<int> points;  // launch indices at which the observed value moved
    std::vector<uint64_t> values;
    uint64_t last = 0;
    const int n = std::min(a.markers, 1024);
    for (int i = 1; i <= n; ++i) {
      r4dx_tp_test_marker(mb.dev, static_cast<uint64_t>(i), s64);
      const auto g0 = Clock::now();
      while (UsSince(g0) < 20.0) _mm_pause();
      const uint64_t now_v = mb.Value();
      if (now_v != last) {
        points.push_back(i);
        values.push_back(now_v);
        last = now_v;
      }
    }
    const double tail_seen = PollFor(mb, static_cast<uint64_t>(n), Clock::now(), a.observe_ms);
    const uint64_t tail_value = mb.Value();
    Drain(st, ev);
    std::printf("[probe] E2 %d paced markers: the observed value moved at %zu launch(es):", n, points.size());
    for (size_t k = 0; k < points.size() && k < 16; ++k) {
      std::printf(" %d(->%llu)", points[k], static_cast<unsigned long long>(values[k]));
    }
    std::printf("%s\n[probe]   tail: %s (value %llu after %d ms with no HIP call)\n", points.size() > 16 ? " ..." : "",
                tail_seen >= 0 ? "started on its own" : "NOT started", static_cast<unsigned long long>(tail_value),
                a.observe_ms);
    json += "  \"E2\": {\"markers\": " + std::to_string(n) + ", \"tail_value_before_drain\": " +
            std::to_string(tail_value) + ", \"points\": [";
    for (size_t k = 0; k < points.size(); ++k) {
      json += (k ? ", " : "") + std::string("[") + std::to_string(points[k]) + ", " + std::to_string(values[k]) + "]";
    }
    json += "]},\n";
  }

  // ---- F: flush cost ------------------------------------------------------------------------------
  // Interleaved, three rounds, after a ~30 ms clock warm-up (the first configuration otherwise runs
  // at an idle GPU's low clock and looks slowest); min over the rounds.
  {
    for (int i = 0; i < 3; ++i) busy(1200);
    Drain(st, ev);
    json += "  \"F\": [";
    const int intervals[4] = {0, 1, 4, 16};
    double best_gpu[4], best_host[4];
    int flushes_of[4] = {0, 0, 0, 0};
    for (int x = 0; x < 4; ++x) best_gpu[x] = best_host[x] = 1e30;
    for (int round = 0; round < 3; ++round) {
      for (int x = 0; x < 4; ++x) {
        const int every = intervals[x];
        Drain(st, ev);  // idle stream first, so every configuration starts the same way
        int flushes = 0;
        const auto h0 = Clock::now();
        R4DX_HIP_CHECK(hipEventRecord(t0e, st));
        for (int u = 0; u < a.units; ++u) {
          busy(24);  // ~0.2 ms
          for (int k = 0; k < 8; ++k) busy(0);
          if (every > 0 && (u + 1) % every == 0) {
            R4DX_HIP_CHECK(hipEventRecord(ev, st));
            QueryOnce(ev);
            ++flushes;
          }
        }
        R4DX_HIP_CHECK(hipEventRecord(t1e, st));
        const double host_us = UsSince(h0);
        Drain(st, ev);
        float ms = 0.0f;
        R4DX_HIP_CHECK(hipEventElapsedTime(&ms, t0e, t1e));
        best_gpu[x] = std::min(best_gpu[x], static_cast<double>(ms));
        best_host[x] = std::min(best_host[x], host_us / 1000.0);
        flushes_of[x] = flushes;
      }
    }
    for (int x = 0; x < 4; ++x) {
      std::printf("[probe] F %d units, flush every %-2d (%3d flushes): GPU %.3f ms, host enqueue %.3f ms (min of 3)\n",
                  a.units, intervals[x], flushes_of[x], best_gpu[x], best_host[x]);
      json += std::string(x ? ", " : "") + "{\"every\": " + std::to_string(intervals[x]) + ", \"flushes\": " +
              std::to_string(flushes_of[x]) + ", \"gpu_ms\": " + JsonNum(best_gpu[x]) +
              ", \"host_ms\": " + JsonNum(best_host[x]) + "}";
    }
    json += "],\n";
  }

  // ---- G: is a stream slower once an event has been recorded on it? ------------------------------
  // Fresh streams, never synchronized or queried before the measurement (only the marker poll and the
  // runtime's own 129-command auto-submits move them): 4096 tiny kernels, then a marker, then 130
  // more to carry the marker through an auto-submit; time from the first launch to the marker
  // starting, and the enqueue time alone. Stream kinds: no event ever recorded on it; one
  // hipEventDisableTiming event recorded (and waited for) first; one timing event recorded first.
  {
    constexpr int kKinds = 7;
    const char* kinds[kKinds] = {"no event ever recorded",
                                 "after one DisableTiming event record",
                                 "after one timing event record",
                                 "after a DisableTiming record, event destroyed",
                                 "after one hipStreamSynchronize",
                                 "after one hipMemcpyAsync D2H (pinned)",
                                 "after one hipMemsetAsync"};
    double best_done[kKinds], best_enq[kKinds];
    for (int x = 0; x < kKinds; ++x) best_done[x] = best_enq[x] = 1e30;
    constexpr int kKernels = 4096;
    uint64_t* pinned8 = const_cast<uint64_t*>(mb.host) + 64;  // a spare line of the mapped buffer
    for (int round = 0; round < 3; ++round) {
      for (int x = 0; x < kKinds; ++x) {
        r4dx::core::Stream s2;
        const int64_t s2_64 = reinterpret_cast<int64_t>(s2.get());
        if (x == 1 || x == 2) {
          hipEvent_t e = x == 1 ? ev : t0e;
          R4DX_HIP_CHECK(hipEventRecord(e, s2.get()));
          if (!tp::WaitEventWithWatchdog(e, std::chrono::milliseconds(30000))) DieStuck("probe G event");
        } else if (x == 3) {
          hipEvent_t e = nullptr;
          R4DX_HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
          R4DX_HIP_CHECK(hipEventRecord(e, s2.get()));
          if (!tp::WaitEventWithWatchdog(e, std::chrono::milliseconds(30000))) DieStuck("probe G event");
          R4DX_HIP_CHECK(hipEventDestroy(e));
        } else if (x == 4) {
          r4dx_tp_clock_probe(scratch64, 0, s2_64);
          R4DX_HIP_CHECK(hipStreamSynchronize(s2.get()));
        } else if (x == 5) {
          R4DX_HIP_CHECK(hipMemcpyAsync(pinned8, scratch.data(), 8, hipMemcpyDeviceToHost, s2.get()));
          R4DX_HIP_CHECK(hipStreamSynchronize(s2.get()));
        } else if (x == 6) {
          R4DX_HIP_CHECK(hipMemsetAsync(scratch.data(), 0, 8, s2.get()));
          R4DX_HIP_CHECK(hipStreamSynchronize(s2.get()));
        }
        // Warm-up without a HIP wait: one kernel + a marker carried through an auto-submit.
        ++v;
        r4dx_tp_clock_probe(scratch64, 0, s2_64);
        r4dx_tp_test_marker(mb.dev, v, s2_64);
        for (int i = 0; i < 130; ++i) r4dx_tp_clock_probe(scratch64, 0, s2_64);
        if (PollFor(mb, v, Clock::now(), 2000) < 0) DieStuck("probe G warm-up");
        ++v;
        const auto t0 = Clock::now();
        for (int i = 0; i < kKernels; ++i) r4dx_tp_clock_probe(scratch64, 0, s2_64);
        const double enq = UsSince(t0);
        r4dx_tp_test_marker(mb.dev, v, s2_64);
        for (int i = 0; i < 130; ++i) r4dx_tp_clock_probe(scratch64, 0, s2_64);
        const double done = PollFor(mb, v, t0, 2000);
        if (done < 0) DieStuck("probe G");
        best_done[x] = std::min(best_done[x], done);
        best_enq[x] = std::min(best_enq[x], enq);
        // ~Stream (hipStreamDestroy) drains the rest.
      }
    }
    json += "  \"G\": [";
    for (int x = 0; x < kKinds; ++x) {
      std::printf("[probe] G %-38s %d kernels: marker after %.0f us (%.3f us/kernel), enqueue %.0f us (min of 3)\n",
                  kinds[x], kKernels, best_done[x], best_done[x] / kKernels, best_enq[x]);
      json += std::string(x ? ", " : "") + "{\"kind\": " + JsonStr(kinds[x]) + ", \"marker_us\": " +
              JsonNum(best_done[x]) + ", \"enqueue_us\": " + JsonNum(best_enq[x]) + "}";
    }
    json += "],\n";
  }

  // ---- H: does a small copy or memset submit what is queued before it? (fresh streams) ----------
  {
    uint64_t* pinned8 = const_cast<uint64_t*>(mb.host) + 64;
    const char* names_h[2] = {"H marker + hipMemcpyAsync D2H 8 B (pinned)", "H marker + hipMemsetAsync 8 B"};
    json += "  \"H\": [";
    for (int x = 0; x < 2; ++x) {
      r4dx::core::Stream s3;
      ++v;
      r4dx_tp_test_marker(mb.dev, v, reinterpret_cast<int64_t>(s3.get()));
      const auto t0 = Clock::now();
      if (x == 0) {
        R4DX_HIP_CHECK(hipMemcpyAsync(pinned8, scratch.data(), 8, hipMemcpyDeviceToHost, s3.get()));
      } else {
        R4DX_HIP_CHECK(hipMemsetAsync(scratch.data(), 0, 8, s3.get()));
      }
      const double seen = PollFor(mb, v, t0, a.observe_ms);
      R4DX_HIP_CHECK(hipStreamSynchronize(s3.get()));
      if (seen >= 0) {
        std::printf("[probe] %-44s started %.1f us after the call\n", names_h[x], seen);
      } else {
        std::printf("[probe] %-44s NOT started after %d ms\n", names_h[x], a.observe_ms);
      }
      json += std::string(x ? ", " : "") + "{\"case\": " + JsonStr(names_h[x]) + ", \"started_us\": " + JsonNum(seen) + "}";
    }
    json += "]\n}\n";
  }

  (void)hipEventDestroy(ev);
  (void)hipEventDestroy(t0e);
  (void)hipEventDestroy(t1e);
  (void)hipHostFree(h);
  if (!a.json.empty() && !WriteFile(a.json, json)) {
    std::fprintf(stderr, "cannot write %s\n", a.json.c_str());
    return 1;
  }
  return 0;
}
