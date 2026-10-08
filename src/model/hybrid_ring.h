// HIP-free host-side handshake of the hybrid mode's two-card reshard (docs/pp-tp2-hybrid.md 3, 7; P3 step 2): the cross-card bytes of
// a reshard plan leave one card through a pinned host ring and enter the other card from it, and the two rank threads (one per card,
// no TpComm between them) drive that with nothing but these primitives.
//
//   RingPipe    one direction of the ring: the SOURCE card's thread fills slot (batch % slots) with a D2H, then Publish(batch); the
//               DESTINATION card's thread waits for it, drains it with an H2D, then Consume(batch), which frees the slot for batch
//               + slots. FIFO, bounded waits, poison.
//   HostGate    a one-shot "this host data is ready" flag between the threads (stage Y's end state, the DFlash tail capture).
//   RunLockstep the per-card loop of a reshard: step i exports batch i (the card's own outgoing bytes) and imports batch i - 1 (the
//               peer's), so the D2H of one batch overlaps the H2D of the previous one and, because a card's two Models own two streams,
//               the card's own D2H overlaps its own H2D too. Both threads run the same loop with the roles swapped; it cannot deadlock
//               (a thread's step i needs only the peer's publish of batch i - 1 and the peer's consume of batch i - slots, both of
//               which the peer finishes in steps before the ones this step waits on) -- the CPU test checks that under random shapes.
//
// Any failure poisons everything the failing thread's call shares (HyPoison in the caller), which makes every wait on the other
// thread throw RingPoisoned at once; a peer that never comes (it hangs in a HIP call) ends in RingTimeout. Header-only and HIP-free
// so tests/model/test_hybrid_ring_cpu.cpp runs the protocol with fake copies.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>

#include "pp_channel.h"

namespace r4dx::model::hybrid {

class RingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class RingPoisoned : public RingError {
 public:
  using RingError::RingError;
};
class RingTimeout : public RingError {
 public:
  using RingError::RingError;
};

// True when `e` is only the consequence of the OTHER thread having failed (a poisoned ring or stage channel): the owner reports the
// peer's error instead (TpModel turns these into core::TpAbortedError, which Run()'s root-cause rule ranks last).
inline bool IsPoisonConsequence(const std::exception_ptr& e) {
  try {
    std::rethrow_exception(e);
  } catch (const RingPoisoned&) {
    return true;
  } catch (const pp::ChannelPoisoned&) {
    return true;
  } catch (...) {
    return false;
  }
}

// ---- one direction of the ring ---------------------------------------------------------------------------------------------------
class RingPipe {
 public:
  explicit RingPipe(int slots = 3) : slots_(slots < 1 ? 1 : slots) {}
  RingPipe(const RingPipe&) = delete;
  RingPipe& operator=(const RingPipe&) = delete;

  int Slots() const { return slots_; }
  // Between phases, once both threads are out of the pipe: counters back to zero, the poison cleared.
  void Reset() {
    std::lock_guard<std::mutex> lk(mu_);
    published_ = consumed_ = 0;
    poisoned_ = false;
    reason_.clear();
  }
  // Idempotent; the first reason sticks. Wakes every waiter.
  void Poison(const std::string& why) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!poisoned_) {
        poisoned_ = true;
        reason_ = why;
      }
    }
    cv_.notify_all();
  }
  bool Poisoned() const {
    std::lock_guard<std::mutex> lk(mu_);
    return poisoned_;
  }

  // ---- source side
  // Blocks until slot (batch % slots) is free, i.e. the destination consumed batch - slots.
  void WaitSlotFree(int64_t batch, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    Wait(lk, [&] { return poisoned_ || batch - consumed_ < slots_; }, timeout, "WaitSlotFree");
  }
  // Batch `batch` is complete in its slot (the D2H finished). Batches are published in order.
  void Publish(int64_t batch) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (poisoned_) throw RingPoisoned("RingPipe::Publish: poisoned (" + reason_ + ")");
      if (batch != published_) throw std::logic_error("RingPipe::Publish: batches are published in order");
      ++published_;
    }
    cv_.notify_all();
  }

  // ---- destination side
  void WaitPublished(int64_t batch, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    Wait(lk, [&] { return poisoned_ || published_ > batch; }, timeout, "WaitPublished");
  }
  // The H2D of batch `batch` finished: its slot may be refilled. Consumed in order, never ahead of the publish.
  void Consume(int64_t batch) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (poisoned_) throw RingPoisoned("RingPipe::Consume: poisoned (" + reason_ + ")");
      if (batch != consumed_ || batch >= published_) throw std::logic_error("RingPipe::Consume: out of order or not published");
      ++consumed_;
    }
    cv_.notify_all();
  }

  int64_t Published() const {
    std::lock_guard<std::mutex> lk(mu_);
    return published_;
  }
  int64_t Consumed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return consumed_;
  }

 private:
  template <class Pred>
  void Wait(std::unique_lock<std::mutex>& lk, Pred ready, std::chrono::milliseconds timeout, const char* what) {
    if (!ready() && !cv_.wait_for(lk, timeout, ready)) {
      throw RingTimeout(std::string("RingPipe::") + what + ": no progress for " + std::to_string(timeout.count()) + " ms");
    }
    if (poisoned_) throw RingPoisoned(std::string("RingPipe::") + what + ": poisoned (" + reason_ + ")");
  }

  const int slots_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  int64_t published_ = 0, consumed_ = 0;
  bool poisoned_ = false;
  std::string reason_;
};

// ---- a one-shot flag between the two threads -------------------------------------------------------------------------------------
class HostGate {
 public:
  void Reset() {
    std::lock_guard<std::mutex> lk(mu_);
    open_ = poisoned_ = false;
    reason_.clear();
  }
  void Open() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      open_ = true;
    }
    cv_.notify_all();
  }
  void Poison(const std::string& why) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!poisoned_) {
        poisoned_ = true;
        reason_ = why;
      }
    }
    cv_.notify_all();
  }
  bool IsOpen() const {
    std::lock_guard<std::mutex> lk(mu_);
    return open_;
  }
  // Returns once Open() was called (data published under the mutex is visible); throws RingPoisoned / RingTimeout otherwise.
  void Wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto ready = [&] { return open_ || poisoned_; };
    if (!ready() && !cv_.wait_for(lk, timeout, ready)) {
      throw RingTimeout("HostGate::Wait: no progress for " + std::to_string(timeout.count()) + " ms");
    }
    if (poisoned_) throw RingPoisoned("HostGate::Wait: poisoned (" + reason_ + ")");
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool open_ = false, poisoned_ = false;
  std::string reason_;
};

// ---- the per-card loop ----------------------------------------------------------------------------------------------------------
struct LockstepCallbacks {
  std::function<void(int64_t)> export_enqueue;  // enqueue the D2H of own batch i into its ring slot (async on the source Model's stream)
  std::function<void(int64_t)> export_sync;     // wait for it (the source Model's stream): the slot is complete
  std::function<void(int64_t)> import_enqueue;  // enqueue the H2D of the peer's batch i from its slot (async on the destination Model's stream)
  std::function<void(int64_t)> import_sync;     // wait for it: the slot may be refilled
  std::function<void()> on_step;                // progress (the rank worker's heartbeat); optional
};
// Steps needed for n_out outgoing and n_in incoming batches.
inline int64_t LockstepSteps(int64_t n_out, int64_t n_in) { return std::max<int64_t>(n_out, n_in > 0 ? n_in + 1 : 0); }

// One phase (the warm gather, the end-of-call scatter) of a card's reshard. A pipe lives for the whole call and its batch numbers keep
// counting across phases: `out_base` / `in_base` are the batches earlier phases already put through the outgoing / incoming pipe, so a
// phase never refills a ring slot the peer may still be reading from the previous phase (the two threads have no barrier between phases:
// a card that has nothing to gather runs its prefill, and its scatter, while the peer is still draining the gather). The callbacks get
// the phase-local batch index; the ring slot is (base + i) % slots.
inline void RunLockstep(RingPipe& out, RingPipe& in, int64_t n_out, int64_t n_in, int64_t out_base, int64_t in_base, const LockstepCallbacks& cb,
                        std::chrono::milliseconds timeout) {
  const int64_t steps = LockstepSteps(n_out, n_in);
  for (int64_t i = 0; i < steps; ++i) {
    const bool exporting = i < n_out;
    const int64_t imp = i - 1;
    const bool importing = imp >= 0 && imp < n_in;
    if (exporting) {
      out.WaitSlotFree(out_base + i, timeout);
      cb.export_enqueue(i);
    }
    if (importing) {
      in.WaitPublished(in_base + imp, timeout);
      cb.import_enqueue(imp);
    }
    if (exporting) {
      cb.export_sync(i);
      out.Publish(out_base + i);
    }
    if (importing) {
      cb.import_sync(imp);
      in.Consume(in_base + imp);
    }
    if (cb.on_step) cb.on_step();
  }
}
inline void RunLockstep(RingPipe& out, RingPipe& in, int64_t n_out, int64_t n_in, const LockstepCallbacks& cb, std::chrono::milliseconds timeout) {
  RunLockstep(out, in, n_out, n_in, 0, 0, cb, timeout);
}

}  // namespace r4dx::model::hybrid
