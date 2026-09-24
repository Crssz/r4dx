// r4dx::core::HostExchange -- the in-process host rendezvous of the tensor-parallel ranks
// (docs/tp.md 6.6): a blocking all-gather of small host buffers between the rank threads of one
// process, used for every cross-rank merge that is not a device all-reduce (lockstep fingerprints,
// vocab-split argmax / row-summary / top-16 merges, full-row gathers -- docs/tp.md 6.2 H1-H7) and
// as EmulatedComm's barrier.
//
// Header-only and CPU-only (no HIP call). One instance is shared by every rank of a TpGroup; rank r
// only ever calls it from its own rank thread.
//
// Protocol, call g of rank r (g counts this rank's calls since construction / the last Reset()):
//   aborted -> TpAbortedError at once, before any of the below;
//   g = ++gen[r]; write mine into slot[g & 1][r]; arrive[r].store(g, release);
//   wait until arrive[q] >= g for every q -- spin (_mm_pause) for up to 50 ms, then yield, checking
//   the abort flag every iteration; past `timeout` -> Abort() + TpTimeoutError;
//   copy slot[g & 1][q] of every q to out, in rank order.
// No second barrier is needed to make the slot reuse safe: rank r rewrites slot[g & 1][r] only at
// call g + 2, after passing call g + 1's wait, which needs every peer to have ARRIVED at g + 1,
// i.e. to have finished copying call g's slots. The same parity argument backs EmulatedComm (6.5).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/core/tp_comm.hpp"

namespace r4dx::core {

class HostExchange {
 public:
  explicit HostExchange(int world, std::chrono::milliseconds timeout = std::chrono::seconds(30))
      : world_(world), timeout_(timeout) {
    if (world < 1) throw std::invalid_argument("HostExchange: world must be >= 1");
    ranks_.reset(new PerRank[static_cast<size_t>(world)]);
  }
  HostExchange(const HostExchange&) = delete;
  HostExchange& operator=(const HostExchange&) = delete;

  // out[q*bytes .. (q+1)*bytes) = rank q's `mine`, for every q, in rank order. Every rank passes
  // the same `bytes` (a mismatch aborts the group and throws TpDivergenceError). Throws
  // TpAbortedError if the group is (or becomes) aborted while waiting, TpTimeoutError (after
  // aborting the group) if a peer does not arrive within the timeout.
  void AllGather(int rank, const void* mine, size_t bytes, void* out) {
    AllGatherFor(rank, mine, bytes, out, timeout_);
  }

  void Barrier(int rank) { AllGather(rank, nullptr, 0, nullptr); }

  // The same with this call's own wait bound instead of the constructor's. EmulatedComm's
  // all-reduce barrier waits at most the all-reduce spin timeout, exactly as long as the device
  // kernel it stands in for would spin (docs/tp.md 6.5; Appendix B N45).
  void BarrierFor(int rank, std::chrono::milliseconds timeout) {
    AllGatherFor(rank, nullptr, 0, nullptr, timeout);
  }

  void AllGatherFor(int rank, const void* mine, size_t bytes, void* out,
                    std::chrono::milliseconds timeout) {
    CheckRank(rank);
    PerRank& me = ranks_[rank];
    Bump(me);
    // Before touching the slot: a call on an aborted group never gets past the wait below, so it
    // must not write slot[g & 1] either -- a slower peer may still be copying out of it.
    if (aborted_.load(std::memory_order_acquire)) {
      Bump(me);
      throw TpAbortedError(AbortMessage());
    }
    const uint64_t g = ++me.gen;
    std::vector<uint8_t>& my_slot = me.slot[g & 1];
    my_slot.resize(bytes);
    if (bytes > 0) std::memcpy(my_slot.data(), mine, bytes);
    me.arrive.store(g, std::memory_order_release);

    const auto t0 = std::chrono::steady_clock::now();
    bool spinning = true;
    for (uint64_t it = 0;; ++it) {
      if (aborted_.load(std::memory_order_acquire)) {
        Bump(me);
        throw TpAbortedError(AbortMessage());
      }
      int q = 0;
      while (q < world_ && ranks_[q].arrive.load(std::memory_order_acquire) >= g) ++q;
      if (q == world_) break;
      if (spinning) {
        _mm_pause();
        if ((it & 63) != 63) continue;  // read the clock every 64 polls while spinning
      }
      const auto waited = std::chrono::steady_clock::now() - t0;
      if (waited >= timeout) {
        const std::string why =
            "tp: host exchange timed out: rank " + std::to_string(rank) + " waited " +
            std::to_string(timeout.count()) + " ms at call " + std::to_string(g) + " for rank " +
            std::to_string(q);
        Abort(why);
        Bump(me);
        throw TpTimeoutError(why);
      }
      if (waited >= kSpin) {
        spinning = false;
        std::this_thread::yield();
      }
    }

    for (int q = 0; q < world_; ++q) {
      const std::vector<uint8_t>& s = ranks_[q].slot[g & 1];
      if (s.size() != bytes) {
        const std::string why = "tp: host exchange size mismatch at call " + std::to_string(g) +
                                ": rank " + std::to_string(rank) + " gathers " +
                                std::to_string(bytes) + " B, rank " + std::to_string(q) +
                                " sent " + std::to_string(s.size()) + " B";
        Abort(why);
        Bump(me);
        throw TpDivergenceError(why);
      }
      if (bytes > 0) std::memcpy(static_cast<uint8_t*>(out) + q * bytes, s.data(), bytes);
    }
    Bump(me);
  }

  // Poisons the exchange: every current and future waiter throws TpAbortedError until Reset().
  // The first reason is kept (the flag is set inside the same critical section that tests it).
  void Abort(const std::string& why) noexcept {
    try {
      std::lock_guard<std::mutex> lk(why_mu_);
      if (!aborted_.load(std::memory_order_relaxed)) why_ = why;
      aborted_.store(true, std::memory_order_release);
      return;
    } catch (...) {
    }
    aborted_.store(true, std::memory_order_release);
  }
  bool Aborted() const noexcept { return aborted_.load(std::memory_order_acquire); }

  // Recovery only (docs/tp.md 2.5 step 6), with NO rank inside AllGather/Barrier: zeroes every
  // rank's arrive word AND call counter -- an asymmetric abort (one rank threw before its ++gen,
  // the other after) otherwise leaves the two counters one apart, and the next call would read the
  // peer's stale slot -- zero-fills both slot generations and clears the abort flag.
  void Reset() {
    for (int r = 0; r < world_; ++r) {
      PerRank& p = ranks_[r];
      p.arrive.store(0, std::memory_order_relaxed);
      p.gen = 0;
      for (auto& s : p.slot) std::fill(s.begin(), s.end(), uint8_t{0});
    }
    {
      std::lock_guard<std::mutex> lk(why_mu_);
      why_.clear();
    }
    aborted_.store(false, std::memory_order_release);
  }

  // Optional: `hb` is bumped (relaxed) on entry to and exit from every wait of `rank`, which is
  // what the facade's progress watchdog watches (docs/tp.md 2.2 step 6). nullptr = none.
  void SetHeartbeat(int rank, std::atomic<uint64_t>* hb) {
    CheckRank(rank);
    ranks_[rank].heartbeat = hb;
  }

 private:
  static constexpr std::chrono::milliseconds kSpin{50};

  // One rank's words, on their own 128-B lines, so the arrive words the peers poll never
  // false-share.
  struct alignas(128) PerRank {
    std::atomic<uint64_t> arrive{0};
    uint64_t gen = 0;  // written only by this rank's thread (and by Reset, with no rank inside)
    std::vector<uint8_t> slot[2];
    std::atomic<uint64_t>* heartbeat = nullptr;
  };

  void CheckRank(int rank) const {
    if (rank < 0 || rank >= world_) {
      throw std::invalid_argument("HostExchange: rank " + std::to_string(rank) + " outside [0, " +
                                  std::to_string(world_) + ")");
    }
  }
  static void Bump(PerRank& p) {
    if (p.heartbeat != nullptr) p.heartbeat->fetch_add(1, std::memory_order_relaxed);
  }
  std::string AbortMessage() const {
    std::lock_guard<std::mutex> lk(why_mu_);
    return "tp: group aborted: " + (why_.empty() ? std::string("(no reason given)") : why_);
  }

  int world_;
  std::chrono::milliseconds timeout_;
  std::unique_ptr<PerRank[]> ranks_;
  std::atomic<bool> aborted_{false};
  mutable std::mutex why_mu_;
  std::string why_;
};

}  // namespace r4dx::core
