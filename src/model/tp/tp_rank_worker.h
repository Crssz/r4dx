// r4dx::model::tp::RankWorker -- one persistent thread per tensor-parallel rank, and the facade
// side of the hand-off (docs/tp.md 2.2).
//
// TpModel owns one RankWorker per rank; each rank's Model, TpComm endpoint and every DeviceBuffer
// they own are built, used and destroyed ONLY on that rank's thread (HIP's current device is per
// thread; the thread's first action is TpModel's `thread_init`, i.e. hipSetDevice). One facade
// thread (the CLI main thread / the server's Engine worker) posts the same command to every rank
// and waits for all of them -- lockstep at the command level.
//
// This header knows nothing about HIP or Model, so the whole hand-off -- including the progress
// watchdog -- is CPU-tested (tests/model/test_tp_rank_worker.cpp).
//
// The hand-off, C++17 (no std::atomic::wait), so every blocking hand-off publishes under the
// mutex its waiter checks the predicate under -- that rules out the lost wakeup where the waiter
// tests the predicate, the publisher stores and notifies, and only then does the waiter block:
//   1. Post (facade):  { lock mu_; require done_ == posted_; cmd_ = cmd; posted_ += 1 (release) }
//                      cv_.notify_one()
//   2. Rank wait:      next = done_ + 1; spin (_mm_pause, yield every 64) while posted_ < next for
//                      at most timing.rank_spin; then
//                      { lock mu_; cv_.wait(exit_ || posted_ >= next) } -- mid-generation
//                      commands arrive every ~18 ms and stay off the condvar, an idle server
//                      blocks and burns no CPU.
//   3. Run:            try { cmd_() } catch (...) { err_ = current_exception() }; the closure is
//                      destroyed here, on the rank thread.
//   4. Complete:       { lock group.mu; done_ = next (release) } group.cv.notify_all()
//   5. Facade wait:    WaitAllIdle below -- spin up to timing.facade_spin, then group.cv.wait_for
//                      in 1 s slices, running the ProgressWatchdog at every wake-up.
#pragma once

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace r4dx::model::tp {

struct WorkerTiming {
  std::chrono::microseconds rank_spin{20000};  // rank spins this long before blocking
  std::chrono::microseconds facade_spin{200};  // facade spins this long before blocking
};

// Shared by the facade and every worker: the facade blocks here, a worker publishes each completion
// under `mu` and notifies `cv`.
class CompletionGroup {
 public:
  std::mutex mu;
  std::condition_variable cv;
};

class RankWorker {
 public:
  // Starts the thread. Its first action is `thread_init()` (TpModel passes
  // [dev]{ R4DX_HIP_CHECK(hipSetDevice(dev)); SetThreadDescription(L"r4dx-tp-rank<r>"); }). If
  // thread_init throws, the worker stays up but runs no command: every command completes at once
  // with that exception as its TakeError(), so the facade sees it on its first command.
  RankWorker(int rank, std::function<void()> thread_init, CompletionGroup* done_group,
             WorkerTiming timing = {})
      : rank_(rank), group_(done_group), timing_(timing) {
    if (group_ == nullptr) throw std::invalid_argument("RankWorker: null CompletionGroup");
    th_ = std::thread([this, init = std::move(thread_init)]() mutable { ThreadMain(init); });
  }
  RankWorker(const RankWorker&) = delete;
  RankWorker& operator=(const RankWorker&) = delete;

  // docs/tp.md 2.6: never posts. Waits (bounded 30 s) until the worker has finished whatever it is
  // running -- cmd_ is never written while the rank may still be executing it -- then sets exit_,
  // wakes the thread and joins it (bounded 30 s). If either wait expires the process logs and
  // quick_exit(3)s rather than free memory a kernel may still touch.
  ~RankWorker() {
    const auto deadline = std::chrono::steady_clock::now() + kShutdownWait;
    {
      std::unique_lock<std::mutex> lk(group_->mu);
      if (!group_->cv.wait_until(lk, deadline, [&] { return Idle(); })) {
        lk.unlock();
        DieStuck();
      }
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      exit_ = true;
    }
    cv_.notify_all();
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (!cv_.wait_until(lk, std::chrono::steady_clock::now() + kShutdownWait,
                          [&] { return exited_; })) {
        lk.unlock();
        DieStuck();
      }
    }
    th_.join();
  }

  // Facade only. Precondition Idle(); a post to a busy worker throws std::logic_error instead of
  // overwriting the command the rank may be executing.
  void Post(std::function<void()> cmd) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      const uint64_t posted = posted_.load(std::memory_order_relaxed);
      if (done_.load(std::memory_order_acquire) != posted) {
        throw std::logic_error("RankWorker::Post: rank " + std::to_string(rank_) +
                               " is still running its previous command");
      }
      cmd_ = std::move(cmd);
      posted_.store(posted + 1, std::memory_order_release);
    }
    cv_.notify_one();
  }

  bool Idle() const {
    return done_.load(std::memory_order_acquire) == posted_.load(std::memory_order_acquire);
  }
  uint64_t Posted() const { return posted_.load(std::memory_order_acquire); }
  uint64_t Done() const { return done_.load(std::memory_order_acquire); }
  int Rank() const { return rank_; }

  // The exception of the last completed command, or null; clears it. Facade only, while Idle().
  std::exception_ptr TakeError() { return std::exchange(err_, nullptr); }

  // Bumped (relaxed fetch_add) by the rank's own code at every all-reduce enqueue and every host
  // exchange entry/exit; read by the facade's ProgressWatchdog.
  std::atomic<uint64_t>& Heartbeat() { return heartbeat_; }

 private:
  static constexpr std::chrono::seconds kShutdownWait{30};

  void ThreadMain(const std::function<void()>& thread_init) {
    std::exception_ptr init_err;
    try {
      if (thread_init) thread_init();
    } catch (...) {
      init_err = std::current_exception();
    }
    for (;;) {
      const uint64_t next = done_.load(std::memory_order_relaxed) + 1;
      if (!WaitPosted(next)) break;
      err_ = init_err;
      if (!init_err) {
        try {
          cmd_();
        } catch (...) {
          err_ = std::current_exception();
        }
      }
      cmd_ = nullptr;  // destroy the closure (and what it captured) on this thread
      {
        std::lock_guard<std::mutex> lk(group_->mu);
        done_.store(next, std::memory_order_release);
      }
      group_->cv.notify_all();
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      exited_ = true;
    }
    cv_.notify_all();
  }

  // Step 2. True when command `next` is posted, false when the worker must exit.
  bool WaitPosted(uint64_t next) {
    if (timing_.rank_spin.count() > 0) {
      const auto t0 = std::chrono::steady_clock::now();
      for (uint32_t spins = 1; posted_.load(std::memory_order_acquire) < next; ++spins) {
        _mm_pause();
        if ((spins & 63) == 0) {
          std::this_thread::yield();
          if (std::chrono::steady_clock::now() - t0 >= timing_.rank_spin) break;
        }
      }
    }
    if (posted_.load(std::memory_order_acquire) >= next) return true;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return exit_ || posted_.load(std::memory_order_acquire) >= next; });
    return posted_.load(std::memory_order_acquire) >= next;
  }

  [[noreturn]] void DieStuck() const {
    std::fprintf(stderr, "[r4dx-tp] rank %d did not finish in 30 s at shutdown\n", rank_);
    std::fflush(stderr);
    std::quick_exit(3);
  }

  int rank_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;  // the rank blocks here (and ~RankWorker, for exited_)
  std::atomic<uint64_t> posted_{0}, done_{0}, heartbeat_{0};
  std::function<void()> cmd_;
  std::exception_ptr err_;
  bool exit_ = false;
  bool exited_ = false;
  CompletionGroup* group_;
  WorkerTiming timing_;
};

// The progress watchdog (docs/tp.md 2.2 step 6). A per-command timeout would kill a legitimate
// 100k-token prefill, so the facade instead watches the ranks' heartbeats: it fires only when NO
// rank's heartbeat has moved for `stall_limit` (60 s in production). Every real wait inside a
// command is already bounded (500 ms all-reduce spin, 30 s host exchange, 30 s stream-sync
// watchdog), so this only fires on a thread stuck inside a HIP call.
//
// The clock is a parameter so tests can drive it with a fake one.
//
// A command that moves no heartbeat at all (Model::Load, the pinned embed load, the teardown
// closure -- docs/tp.md 2.9) must be waited on with kNoStallLimit, which never fires, or with a
// limit sized for that command.
class ProgressWatchdog {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::milliseconds kNoStallLimit = std::chrono::milliseconds::max();

  explicit ProgressWatchdog(std::chrono::milliseconds stall_limit = std::chrono::seconds(60))
      : stall_limit_(stall_limit) {}

  // Starts watching one wait: `heartbeats` = every rank's Heartbeat() now.
  void Arm(const std::vector<uint64_t>& heartbeats, Clock::time_point now) {
    last_ = heartbeats;
    last_move_ = now;
  }
  // One facade wake-up that found a rank still busy: true iff no rank's heartbeat has changed for
  // stall_limit. Compares per rank, so a lagging rank catching up counts as progress even while
  // the maximum over ranks stands still.
  bool Stalled(const std::vector<uint64_t>& heartbeats, Clock::time_point now) {
    if (heartbeats != last_) {
      last_ = heartbeats;
      last_move_ = now;
      return false;
    }
    // In whole ms: `now - last_move_ >= stall_limit_` would convert the limit to the clock's
    // nanoseconds, which overflows for any limit past ~292 years (kNoStallLimit: fires at once).
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - last_move_) >= stall_limit_;
  }
  std::chrono::milliseconds StallLimit() const { return stall_limit_; }

 private:
  std::chrono::milliseconds stall_limit_;
  std::vector<uint64_t> last_;
  Clock::time_point last_move_{};
};

enum class WaitResult { kDone, kStalled };

// Step 5, the facade wait: returns kDone once every worker in `workers` is Idle() (has finished
// everything posted to it). Spins up to timing.facade_spin, then blocks on group.cv in `wake`
// slices (1 s in production), running `watchdog` each time a slice ends with a worker still busy.
// kStalled: the watchdog fired. The workers are left untouched -- the caller (TpModel) sets kFatal,
// logs every rank's Posted/Done/Heartbeat, throws TpTimeoutError and never posts to a worker that
// is not Idle() again. `now` (optional) replaces the watchdog's clock.
inline WaitResult WaitAllIdle(const std::vector<RankWorker*>& workers, CompletionGroup& group,
                              const WorkerTiming& timing, ProgressWatchdog& watchdog,
                              std::chrono::milliseconds wake = std::chrono::seconds(1),
                              const std::function<ProgressWatchdog::Clock::time_point()>& now =
                                  nullptr) {
  const auto all_idle = [&] {
    for (const RankWorker* w : workers) {
      if (!w->Idle()) return false;
    }
    return true;
  };
  if (timing.facade_spin.count() > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t spins = 1; !all_idle(); ++spins) {
      _mm_pause();
      if ((spins & 63) == 0 && std::chrono::steady_clock::now() - t0 >= timing.facade_spin) break;
    }
  }
  if (all_idle()) return WaitResult::kDone;

  const auto clock = [&] { return now ? now() : ProgressWatchdog::Clock::now(); };
  const auto heartbeats = [&] {
    std::vector<uint64_t> h;
    h.reserve(workers.size());
    for (RankWorker* w : workers) h.push_back(w->Heartbeat().load(std::memory_order_relaxed));
    return h;
  };
  watchdog.Arm(heartbeats(), clock());
  std::unique_lock<std::mutex> lk(group.mu);
  while (!group.cv.wait_for(lk, wake, all_idle)) {
    if (watchdog.Stalled(heartbeats(), clock())) return WaitResult::kStalled;
  }
  return WaitResult::kDone;
}

}  // namespace r4dx::model::tp
