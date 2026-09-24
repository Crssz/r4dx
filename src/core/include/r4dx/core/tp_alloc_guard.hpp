// r4dx::core -- the tensor-parallel allocation guard (docs/tp.md 2.7, 6.3.7 L4).
//
// hipMalloc / hipFree may imply a device synchronization. Inside a tensor-parallel COLLECTIVE
// command (TpModel::RunCollective: every rank runs the same forward, its all-reduces paired by
// sequence number across ranks) such a hidden sync on one rank stalls it behind its own queued
// all-reduce kernels, which is exactly the kind of skew the 500 ms all-reduce spin timeout turns
// into an abort. The rule is therefore: no device allocation or free inside a collective command
// after warm-up. This guard does not enforce the rule (it cannot: the free path runs in a
// destructor); it COUNTS violations, so tests and the soak can require the count to be 0:
//
//   * TpModel::RunCollective wraps the closure it runs on each rank in a TpCollectiveScope
//     (g_tp_collective_depth++ / --, per thread -- only the rank thread running the closure counts);
//   * core::DeviceBuffer's allocating constructor, Resize() and the free path call
//     TpNoteDeviceAlloc(), which counts (and logs once per call site) whenever the depth is > 0.
//
// At TP=1 the depth is always 0: one thread-local load per allocation, no behaviour change. Header-
// only and HIP-free.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace r4dx::core {

// > 0 while this thread runs a tensor-parallel collective command.
inline thread_local int g_tp_collective_depth = 0;
// Device allocations and frees seen inside collective commands, over the whole process.
inline std::atomic<uint64_t> g_tp_collective_allocs{0};

class TpCollectiveScope {
 public:
  TpCollectiveScope() noexcept { ++g_tp_collective_depth; }
  ~TpCollectiveScope() { --g_tp_collective_depth; }
  TpCollectiveScope(const TpCollectiveScope&) = delete;
  TpCollectiveScope& operator=(const TpCollectiveScope&) = delete;
};

// Called by DeviceBuffer on every real hipMalloc / hipFree. `site` names the call site and
// `logged` is that site's own once-flag. Never throws (the free path is a destructor).
inline void TpNoteDeviceAlloc(const char* site, std::atomic<bool>& logged) noexcept {
  if (g_tp_collective_depth <= 0) return;
  g_tp_collective_allocs.fetch_add(1, std::memory_order_relaxed);
  if (!logged.exchange(true, std::memory_order_relaxed)) {
    std::fprintf(stderr,
                 "[r4dx-tp] WARNING: device %s inside a tensor-parallel collective command "
                 "(docs/tp.md 6.3.7: an implicit device sync); counted in g_tp_collective_allocs\n",
                 site);
  }
}

}  // namespace r4dx::core
