#include "tp/tp_submit.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <string>
#include <thread>
#include <utility>

#include "r4dx/core/error.hpp"
#include "r4dx/core/tp_comm.hpp"

namespace r4dx::model::tp {

namespace {
constexpr std::chrono::milliseconds kUnitWatchdog{30000};
}  // namespace

bool WaitEventWithWatchdog(hipEvent_t ev, std::chrono::milliseconds limit) {
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t spins = 0;; ++spins) {
    const hipError_t e = hipEventQuery(ev);
    if (e == hipSuccess) break;
    if (e != hipErrorNotReady) {
      (void)hipGetLastError();
      throw core::HipError(e, "hipEventQuery", __FILE__, __LINE__);
    }
    if ((spins & 255u) == 0u && std::chrono::steady_clock::now() - t0 > limit) {
      (void)hipGetLastError();
      return false;
    }
    // tp_bench's wait: yield for the first 200k polls (a sleep on Windows rounds up to the timer
    // tick, which would add milliseconds to every short wait), then 1 ms sleeps for a genuinely long
    // wait.
    if (spins < 200000u) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  (void)hipGetLastError();  // hipEventQuery leaves hipErrorNotReady as the thread's last error
  return true;
}

SubmitBounder::SubmitBounder(int max_inflight) : active_(true), cap_(std::max(0, max_inflight)) {
  if (cap_ == 1) return;  // synchronize mode: no events
  // K + 1 slots: after the cap wait at most K - 1 recorded units are unfinished, and the next
  // record must not reuse a slot whose unit might still be pending.
  const size_t n = static_cast<size_t>(std::max(cap_, 1)) + 1;
  ring_.reserve(n);
  try {
    for (size_t i = 0; i < n; ++i) {
      hipEvent_t e = nullptr;
      R4DX_HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
      ring_.push_back(e);
    }
  } catch (...) {
    Destroy();
    throw;
  }
}

SubmitBounder::~SubmitBounder() { Destroy(); }

SubmitBounder::SubmitBounder(SubmitBounder&& o) noexcept
    : active_(o.active_),
      ring_(std::move(o.ring_)),
      cap_(o.cap_),
      recorded_(o.recorded_),
      complete_(o.complete_),
      stats_(o.stats_) {
  o.ring_.clear();
  o.active_ = false;
}

SubmitBounder& SubmitBounder::operator=(SubmitBounder&& o) noexcept {
  if (this != &o) {
    Destroy();
    active_ = o.active_;
    ring_ = std::move(o.ring_);
    o.ring_.clear();
    o.active_ = false;
    cap_ = o.cap_;
    recorded_ = o.recorded_;
    complete_ = o.complete_;
    stats_ = o.stats_;
  }
  return *this;
}

void SubmitBounder::Destroy() noexcept {
  for (hipEvent_t e : ring_) (void)hipEventDestroy(e);
  ring_.clear();
  active_ = false;
}

void SubmitBounder::NoteWait(std::chrono::steady_clock::time_point t0) {
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  ++stats_.waits;
  stats_.wait_us_total += us;
  stats_.wait_us_max = std::max(stats_.wait_us_max, us);
}

void SubmitBounder::EndUnit(hipStream_t stream) {
  if (!active_) return;
  ++stats_.units;
  if (cap_ == 1) {
    // Submit and wait for everything enqueued so far -- unit i-1 finished before unit i is
    // enqueued -- without recording an event on the caller's stream (docs/tp.md Appendix B N56 G).
    // Bounded like Model::RunChunk's own end-of-chunk synchronize: a spinning all-reduce gives up
    // after its timeout (500 ms), so this returns.
    const auto t0 = std::chrono::steady_clock::now();
    R4DX_HIP_CHECK(hipStreamSynchronize(stream));
    NoteWait(t0);
    return;
  }
  const size_t n = ring_.size();
  hipEvent_t ev = ring_[static_cast<size_t>(recorded_ % n)];
  R4DX_HIP_CHECK(hipEventRecord(ev, stream));
  ++recorded_;
  // One query: the runtime submits every command enqueued so far (docs/tp.md Appendix B N56). Not
  // ready is the expected answer.
  const hipError_t q = hipEventQuery(ev);
  if (q != hipSuccess && q != hipErrorNotReady) {
    (void)hipGetLastError();
    throw core::HipError(q, "hipEventQuery", __FILE__, __LINE__);
  }
  (void)hipGetLastError();
  if (cap_ == 0) return;
  // Before the caller enqueues unit `recorded_`, unit `recorded_ - cap_` must have finished.
  const auto t0 = std::chrono::steady_clock::now();
  bool waited = false;
  while (recorded_ - complete_ >= static_cast<uint64_t>(cap_)) {
    const hipError_t c = hipEventQuery(ring_[static_cast<size_t>(complete_ % n)]);
    if (c != hipSuccess) {
      (void)hipGetLastError();
      if (c != hipErrorNotReady) throw core::HipError(c, "hipEventQuery", __FILE__, __LINE__);
      waited = true;
      if (!WaitEventWithWatchdog(ring_[static_cast<size_t>(complete_ % n)], kUnitWatchdog)) {
        throw core::TpTimeoutError("tp: a submission unit is still unfinished after 30 s (a kernel is stuck)");
      }
    }
    ++complete_;
  }
  if (waited) NoteWait(t0);
}

}  // namespace r4dx::model::tp
