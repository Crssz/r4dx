// r4dx_convert::ParallelFor -- the CPU multithreading primitive every quantizer/packer in this
// component uses (task requirement: "Quantizers (CPU, multithreaded with std::thread)"). Splits
// [begin, end) into `nthreads` contiguous chunks (row-tile-sized work, so false sharing between
// threads is never an issue -- each thread owns disjoint rows/tiles start to finish) and joins.
//
// Deliberately NOT a persistent thread pool: r4dx-convert's hot loops are "quantize/pack one
// tensor" calls a few hundred times total, not a fine-grained task graph, so the ~microsecond
// thread-creation cost per call is noise next to the milliseconds of per-tensor work, and a
// spin-up/join-down helper is far less code than a work-stealing pool.
#pragma once

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace r4dx_convert {

// Calls body(lo, hi) once per chunk, on up to `nthreads` worker threads, then blocks until all
// chunks finish. `end <= begin` is a no-op. `nthreads <= 1` (or a range too small to split
// usefully) runs on the calling thread with no thread creation at all.
template <typename Body>
void ParallelFor(int64_t begin, int64_t end, int nthreads, Body&& body) {
  if (end <= begin) return;
  const int64_t total = end - begin;
  nthreads = std::max(1, nthreads);
  nthreads = static_cast<int>(std::min<int64_t>(nthreads, total));

  if (nthreads <= 1) {
    body(begin, end);
    return;
  }

  const int64_t chunk = (total + nthreads - 1) / nthreads;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(nthreads));
  for (int t = 0; t < nthreads; ++t) {
    const int64_t lo = begin + static_cast<int64_t>(t) * chunk;
    const int64_t hi = std::min(end, lo + chunk);
    if (lo >= hi) break;
    workers.emplace_back([lo, hi, &body]() { body(lo, hi); });
  }
  for (auto& w : workers) w.join();
}

}  // namespace r4dx_convert
