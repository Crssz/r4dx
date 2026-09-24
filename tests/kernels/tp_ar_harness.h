// tests/kernels/tp_ar_harness.h -- shared by the tensor-parallel all-reduce test and tools
// (docs/tp.md 10.1: test_tp_allreduce_2gpu, tool_tp_ar_stress, tool_tp_ar_latency).
//
//   * RankPool: one tp::RankWorker per rank (hipSetDevice first), RunAll / RunOne, and the
//     TpGroup::RankRunner that TpGroup::Recover drives.
//   * RunDriver: tools/tp_bench's verified batch runner (run_plan / ar_rank) ported onto a TpComm
//     endpoint. Per batch of K all-reduces it generates tp_bench's hash pattern (call c, element j,
//     rank r) into K ring rows, all-reduces them IN PLACE through the endpoint (optionally with a
//     VRAM filler before every all-reduce: tp_bench's decode pattern), verifies every output element
//     on the device, and votes with the peer through HostAllGather, so both ranks stop at the same
//     batch and in-flight batches always pair up. kInflight batches are enqueued ahead; the next
//     batch is enqueued BEFORE any host-side work (progress printing). With conditions == 3 the
//     batches interleave tp_bench's decode conditions (a) all-reduce, (b) a local stand-in on the
//     same grid, (c) fillers only, hipEvent-timed around the body only (gen and verify excluded).
//   * DecodeStats: tp_bench's decode_stats (paired per-token differences).
//   * AutoDevices / PreflightVram (docs/tp.md 9.2).
//
// Every wait is bounded: stream/event polls under a 30 s watchdog (then the process _Exit(3)s, like
// tp_bench's g_stuck -- a kernel may still touch the buffers), host rendezvous under the
// HostExchange's 30 s, all-reduce spins under the endpoint's timeout (<= 1500 ms).
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/tp_kernels.h"
#include "tp/tp_group.h"
#include "tp/tp_rank_worker.h"

namespace tp_harness {

namespace tp = r4dx::model::tp;
namespace core = r4dx::core;

inline double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

[[noreturn]] inline void DieStuck(const char* where) {
  std::fprintf(stderr, "FATAL: a stream is still busy after 30 s (%s); exiting without freeing device memory\n", where);
  std::fflush(stderr);
  std::fflush(stdout);
  std::_Exit(3);
}

// The environment variable `name`, or "" (_dupenv_s: getenv is deprecated under the MSVC CRT).
inline std::string GetEnv(const char* name) {
#ifdef _MSC_VER
  char* v = nullptr;
  size_t len = 0;
  std::string out;
  if (_dupenv_s(&v, &len, name) == 0 && v != nullptr) out = v;
  std::free(v);
  return out;
#else
  const char* v = std::getenv(name);
  return v != nullptr ? std::string(v) : std::string();
#endif
}

// ---- devices ------------------------------------------------------------------------------------

inline int VisibleDevices() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) n = 0;
  (void)hipGetLastError();
  return n;
}

// docs/tp.md 9.2 `--tp-devices auto`, real mode: descending ordinals from the last visible one, so
// with HIP_VISIBLE_DEVICES unset rank 0 = physical device 1 (headless), rank 1 = device 0.
inline std::vector<int> AutoDevices() {
  const int n = VisibleDevices();
  if (n < 2) return {};
  return {n - 1, n - 2};
}

// "a,b" -> {a, b}; "" or "auto" -> AutoDevices().
inline std::vector<int> ParseDevices(const std::string& s) {
  if (s.empty() || s == "auto") return AutoDevices();
  std::vector<int> d;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) d.push_back(std::stoi(item));
  return d;
}

// docs/tp.md 9.2 pre-flight: exit 1 unless every device has need_gib free.
inline void PreflightVram(const std::vector<int>& devices, double need_gib) {
  for (int d : devices) {
    R4DX_HIP_CHECK(hipSetDevice(d));
    size_t free_b = 0, total_b = 0;
    R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
    if (free_gib < need_gib) {
      std::fprintf(stderr, "need %.1f GiB free on HIP device %d, have %.2f GiB -- is the production server running?\n",
                   need_gib, d, free_gib);
      std::exit(1);
    }
  }
}

inline std::string DeviceLabel(int d) {
  hipDeviceProp_t p;
  if (hipGetDeviceProperties(&p, d) != hipSuccess) {
    (void)hipGetLastError();
    return "HIP device " + std::to_string(d);
  }
  char b[256];
  std::snprintf(b, sizeof b, "HIP device %d (%s, %s, pci %02x:%02x)", d, p.name, p.gcnArchName, p.pciBusID,
                p.pciDeviceID);
  return b;
}

// ---- rank threads -------------------------------------------------------------------------------

class RankPool {
 public:
  explicit RankPool(const std::vector<int>& devices) : devices_(devices) {
    for (size_t r = 0; r < devices.size(); ++r) {
      const int dev = devices[r];
      workers_.emplace_back(
          std::make_unique<tp::RankWorker>(static_cast<int>(r), [dev] { R4DX_HIP_CHECK(hipSetDevice(dev)); }, &done_));
    }
  }
  int World() const { return static_cast<int>(workers_.size()); }
  int Device(int r) const { return devices_[static_cast<size_t>(r)]; }
  std::atomic<uint64_t>& Heartbeat(int r) { return workers_[static_cast<size_t>(r)]->Heartbeat(); }

  // Called on the rank thread when a command throws, before the error is recorded. Tests abort the
  // rank's endpoint here, so a peer blocked in a comm call wakes up (docs/tp.md 2.2 step 3).
  std::function<void(int rank, const std::string& what)> on_error;

  // Runs fn(r) on every rank's thread; rethrows the root cause (docs/tp.md 2.4: the lowest rank's
  // exception that is not a TpAbortedError, else the lowest rank's).
  void RunAll(const std::function<void(int)>& fn) {
    for (int r = 0; r < World(); ++r) {
      workers_[static_cast<size_t>(r)]->Post([this, &fn, r] { Guard(r, [&] { fn(r); }); });
    }
    WaitAndRethrow();
  }
  void RunOne(int rank, const std::function<void()>& fn) {
    workers_[static_cast<size_t>(rank)]->Post([this, &fn, rank] { Guard(rank, fn); });
    WaitAndRethrow();
  }
  tp::TpGroup::RankRunner Runner() {
    tp::TpGroup::RankRunner rr;
    rr.run_all = [this](const std::function<void(int)>& f) { RunAll(f); };
    rr.run_one = [this](int r, const std::function<void()>& f) { RunOne(r, f); };
    return rr;
  }

 private:
  void Guard(int r, const std::function<void()>& fn) {
    try {
      fn();
    } catch (const std::exception& e) {
      if (on_error) on_error(r, e.what());
      throw;
    } catch (...) {
      if (on_error) on_error(r, "unknown exception");
      throw;
    }
  }
  void WaitAndRethrow() {
    std::vector<tp::RankWorker*> ptrs;
    for (auto& w : workers_) ptrs.push_back(w.get());
    tp::ProgressWatchdog wd(tp::ProgressWatchdog::kNoStallLimit);
    tp::WaitAllIdle(ptrs, done_, tp::WorkerTiming{}, wd);
    std::exception_ptr root, first_aborted;
    for (auto& w : workers_) {
      std::exception_ptr e = w->TakeError();
      if (!e) continue;
      bool aborted = false;
      try {
        std::rethrow_exception(e);
      } catch (const core::TpAbortedError&) {
        aborted = true;
      } catch (...) {
      }
      if (!aborted && !root) root = e;
      if (aborted && !first_aborted) first_aborted = e;
    }
    if (root) std::rethrow_exception(root);
    if (first_aborted) std::rethrow_exception(first_aborted);
  }

  std::vector<int> devices_;
  tp::CompletionGroup done_;
  std::vector<std::unique_ptr<tp::RankWorker>> workers_;
};

// ---- the verified batch driver ------------------------------------------------------------------

enum { kCondAr = 0, kCondStandin = 1, kCondFillers = 2 };

struct DriverConfig {
  std::vector<size_t> sizes = {10240};  // all-reduce with pattern call index c uses sizes[c % n] bytes
  int K = 50;                           // all-reduces per batch (ring rows)
  int64_t batches = 1;                  // total batches, all conditions
  int conditions = 1;                   // 1: all-reduce only; 3: tp_bench decode (a)/(b)/(c)
  int discard = 0;                      // leading batches not timed
  bool fillers = false;                 // a VRAM filler before every slot (tp_bench's decode pattern)
  size_t filler_rd_bytes = size_t{60} << 20;
  size_t filler_dirty_bytes = size_t{4} << 20;
  size_t filler_ring_bytes = size_t{512} << 20;  // >= 2 x rd, larger than the 64 MiB Infinity Cache
  int nb[2] = {4, 4};                   // the group's channel geometry (the stand-in copies its grid)
  // Bounded submission (tp::SubmitBounder, docs/tp.md Appendix B N57): force a submission after
  // every `flush_every` slots of a batch body (a slot = [filler +] all-reduce / stand-in; 0 = off),
  // and keep at most `max_inflight_units` such units queued ahead of the GPU (0 = no cap). With
  // max_inflight_units == 1 (a hipStreamSynchronize per unit) `idle_us` > 0 then keeps the host --
  // and so the GPU, which has nothing queued -- idle that long after each unit's synchronize before
  // the next unit is enqueued: an explicit idle gap for device-0 endurance runs (N64).
  int flush_every = 0;
  int max_inflight_units = 0;
  int idle_us = 0;
  uint64_t call0 = 0;                   // pattern call index of the first all-reduce
  std::function<void(int64_t calls_verified)> on_batch;  // after each checked batch, after the next enqueue
};

struct DriverResult {
  bool ok = false;               // every planned batch verified, no abort, both ranks agreed
  std::string error;             // host exception on this rank
  std::string abort_message;     // CheckHealthy()'s message when an abort word was set
  bool stopped_by_peer = false;  // this rank's checks passed but the peer's did not
  int64_t batches_ok = 0;
  int64_t calls_verified = 0;
  uint64_t next_call = 0;        // pattern call index after the last enqueued all-reduce
  R4dxTpStatus verify{};         // mismatches / min_key (the test's record)
  R4dxTpStatus comm_status{};    // the endpoint's VRAM Status at the end
  std::vector<double> ms[3];     // timed batch-body ms per condition
  double seconds = 0;
  int fgrid = 0;
  tp::SubmitBounder::Stats bound;  // DriverConfig::flush_every's forced submissions and cap waits
};

inline std::string DescribeMismatch(const R4dxTpStatus& v) {
  if (v.mismatches == 0) return "";
  const uint64_t key = v.min_key;
  const uint64_t call = key >> 36;
  const uint32_t elem = static_cast<uint32_t>((key >> 16) & 0xFFFFFu);
  const uint32_t got = static_cast<uint32_t>(key & 0xFFFFu);
  const uint32_t exp = r4dx_tp_test_expected_bits(call, elem);
  char b[256];
  std::snprintf(b, sizeof b,
                "%u mismatching elements; first at call %llu element %u (16-B word %u): expected 0x%04x got 0x%04x",
                v.mismatches, static_cast<unsigned long long>(call), elem, elem / 8, exp, got);
  return b;
}

// Polls an event under a wall-clock watchdog; false = still pending after limit_s.
inline bool WaitEvent(hipEvent_t ev, double limit_s = 30.0) {
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t spins = 0;; ++spins) {
    const hipError_t e = hipEventQuery(ev);
    if (e == hipSuccess) break;
    if (e != hipErrorNotReady) {
      (void)hipGetLastError();
      throw core::HipError(e, "hipEventQuery", __FILE__, __LINE__);
    }
    if ((spins & 255u) == 0u && SecondsSince(t0) > limit_s) {
      (void)hipGetLastError();
      return false;
    }
    if (spins < 200000u) std::this_thread::yield();  // tp_bench's wait_event_wd
    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  (void)hipGetLastError();  // hipEventQuery leaves hipErrorNotReady as the last error
  return true;
}

// On the rank's thread, both ranks with the same cfg (except on_batch). `st` is the stream every
// kernel goes on; it must outlive the endpoint (the endpoint syncs every stream it enqueued an
// all-reduce on -- TpSetup::streams). Throws only for setup failures; run-time failures (mismatch,
// abort, a host exception in the loop) land in the result.
inline DriverResult RunDriver(tp::TpEndpoint& comm, int rank, hipStream_t st, const DriverConfig& cfg) {
  constexpr int kInflight = 3, kSlots = kInflight + 1;
  DriverResult R;
  const int K = cfg.K, C = cfg.conditions;
  if (K < 1 || (C != 1 && C != 3) || cfg.batches < 1 || cfg.sizes.empty()) {
    throw std::invalid_argument("driver: bad config");
  }
  size_t max_bytes = 0;
  bool uniform = true;
  for (size_t s : cfg.sizes) {
    if (s == 0 || s % 16 != 0 || s > tp::kMaxAllReduceBytes) throw std::invalid_argument("driver: bad size");
    max_bytes = std::max(max_bytes, s);
    uniform = uniform && s == cfg.sizes[0];
  }
  const size_t row_elems = max_bytes / 2;
  int dev = 0;
  R4DX_HIP_CHECK(hipGetDevice(&dev));
  int cus = 0;
  R4DX_HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));

  // ---- VRAM, events (all allocated before the start barrier) ----
  const int64_t s64 = reinterpret_cast<int64_t>(st);
  core::DeviceBuffer<uint16_t> ring(static_cast<size_t>(K) * row_elems);
  core::DeviceBuffer<uint16_t> local(row_elems);
  core::DeviceBuffer<uint8_t> rec(sizeof(R4dxTpStatus));
  core::PinnedBuffer<R4dxTpStatus> rec_host(kSlots);
  hipEvent_t ev_s[kSlots] = {}, ev_e[kSlots] = {}, ev_d[kSlots] = {};
  struct EventGuard {
    hipEvent_t* v[3];
    int n;
    ~EventGuard() {
      for (hipEvent_t* a : v)
        for (int i = 0; i < n; ++i)
          if (a[i] != nullptr) (void)hipEventDestroy(a[i]);
    }
  } ev_guard{{ev_s, ev_e, ev_d}, kSlots};
  for (int i = 0; i < kSlots; ++i) {
    R4DX_HIP_CHECK(hipEventCreate(&ev_s[i]));
    R4DX_HIP_CHECK(hipEventCreate(&ev_e[i]));
    R4DX_HIP_CHECK(hipEventCreateWithFlags(&ev_d[i], hipEventDisableTiming));
  }

  core::DeviceBuffer<uint8_t> fring, fdirty;
  core::DeviceBuffer<uint32_t> fsink(256);
  uint64_t rd_words = 0, dirty_words = 0;
  std::vector<uint64_t> foff(static_cast<size_t>(K), 0);
  const int fgrid = std::max(1, cus) * r4dx_tp_test_filler_occupancy();
  R.fgrid = fgrid;
  if (cfg.fillers) {
    const size_t ring_bytes = (std::max(cfg.filler_ring_bytes, 2 * cfg.filler_rd_bytes) + 4095) / 4096 * 4096;
    fring = core::DeviceBuffer<uint8_t>(ring_bytes);
    fdirty = core::DeviceBuffer<uint8_t>(std::max<size_t>(cfg.filler_dirty_bytes, 16));
    rd_words = cfg.filler_rd_bytes / 16;
    dirty_words = cfg.filler_dirty_bytes / 16;
    r4dx_tp_test_fill_hash(reinterpret_cast<int64_t>(fring.data()), ring_bytes / 16, 0x5A5A0000u + static_cast<uint32_t>(rank),
                           std::max(1, cus) * 4, s64);
    R4DX_HIP_CHECK(hipMemsetAsync(fdirty.data(), 0, fdirty.bytes(), st));
    const uint64_t span = ring_bytes - cfg.filler_rd_bytes;
    for (int p = 0; p < K; ++p) {
      foff[static_cast<size_t>(p)] =
          span ? ((static_cast<uint64_t>(p) * cfg.filler_rd_bytes) % span) / 4096 * 4096 / 16 : 0;
    }
  }
  R4DX_HIP_CHECK(hipMemsetAsync(ring.data(), 0, ring.bytes(), st));
  R4DX_HIP_CHECK(hipMemsetAsync(rec.data(), 0, rec.bytes(), st));
  R4DX_HIP_CHECK(hipMemsetAsync(rec.data() + offsetof(R4dxTpStatus, min_key), 0xFF, sizeof(uint64_t), st));
  r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(local.data()), 0xABCDEFull, static_cast<uint32_t>(row_elems), 1,
                        rank, s64);
  if (!tp::SyncWithWatchdog(st)) DieStuck("driver setup");

  const auto size_of = [&](uint64_t call) { return cfg.sizes[call % cfg.sizes.size()]; };
  const auto row = [&](int k) { return ring.data() + static_cast<size_t>(k) * row_elems; };
  // Pattern call index of batch n's first all-reduce: only condition-(a) batches advance it.
  const auto base_of = [&](int64_t n) {
    const int64_t ar_before = C == 1 ? n : (n + C - 1) / C;
    return cfg.call0 + static_cast<uint64_t>(ar_before) * static_cast<uint64_t>(K);
  };
  const auto gen = [&](uint64_t base) {
    if (uniform) {
      r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(row(0)), base, static_cast<uint32_t>(row_elems), K, rank, s64);
      return;
    }
    for (int k = 0; k < K; ++k) {
      r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(row(k)), base + k, static_cast<uint32_t>(size_of(base + k) / 2),
                            1, rank, s64);
    }
  };
  const auto verify = [&](uint64_t base) {
    if (uniform) {
      r4dx_tp_test_verify_bf16(reinterpret_cast<int64_t>(row(0)), base, static_cast<uint32_t>(row_elems), K,
                               reinterpret_cast<int64_t>(rec.data()), s64);
      return;
    }
    for (int k = 0; k < K; ++k) {
      r4dx_tp_test_verify_bf16(reinterpret_cast<int64_t>(row(k)), base + k,
                               static_cast<uint32_t>(size_of(base + k) / 2), 1, reinterpret_cast<int64_t>(rec.data()),
                               s64);
    }
  };
  if (cfg.idle_us < 0 || (cfg.idle_us > 0 && (cfg.flush_every <= 0 || cfg.max_inflight_units != 1))) {
    throw std::invalid_argument("driver: idle_us needs flush_every > 0 and max_inflight_units == 1");
  }
  tp::SubmitBounder bound;
  if (cfg.flush_every > 0) bound = tp::SubmitBounder(cfg.max_inflight_units);
  // A yield loop, not a sleep: a Windows sleep rounds up to the timer tick (N33's wait).
  const auto idle = [&] {
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(cfg.idle_us);
    while (std::chrono::steady_clock::now() < until) std::this_thread::yield();
  };
  const auto body = [&](int cond, uint64_t base) {
    for (int k = 0; k < K; ++k) {
      if (bound.Active() && k > 0 && k % cfg.flush_every == 0) {
        bound.EndUnit(st);
        if (cfg.idle_us > 0) idle();
      }
      if (cfg.fillers) {
        r4dx_tp_test_filler(reinterpret_cast<int64_t>(fring.data()) +
                                static_cast<int64_t>(foff[static_cast<size_t>(k)] * 16),
                            rd_words, reinterpret_cast<int64_t>(fdirty.data()), dirty_words,
                            reinterpret_cast<int64_t>(fsink.data()), fgrid, s64);
      }
      const size_t bytes = size_of(base + k);
      if (cond == kCondAr) {
        comm.AllReduceSumBf16(row(k), static_cast<int64_t>(bytes / 2), st);
      } else if (cond == kCondStandin) {
        const tp::CallChunks g = tp::ChunkCall(bytes, cfg.nb[tp::MailboxLayout::ChannelFor(bytes)]);
        r4dx_tp_test_standin_bf16(reinterpret_cast<int64_t>(row(k)), reinterpret_cast<int64_t>(local.data()), g.n16,
                                  g.nb, g.w, tp::kArThreads, s64);
      }
    }
  };
  const auto enqueue = [&](int64_t n) {
    const int slot = static_cast<int>(n % kSlots);
    const int cond = static_cast<int>(n % C);
    const uint64_t base = base_of(n);
    gen(base);  // before EVERY batch: identical pre-token work in every condition
    R4DX_HIP_CHECK(hipEventRecord(ev_s[slot], st));
    body(cond, base);
    R4DX_HIP_CHECK(hipEventRecord(ev_e[slot], st));
    if (cond == kCondAr) verify(base);
    R4DX_HIP_CHECK(hipMemcpyAsync(&rec_host[static_cast<size_t>(slot)], rec.data(), sizeof(R4dxTpStatus),
                                  hipMemcpyDeviceToHost, st));
    R4DX_HIP_CHECK(hipEventRecord(ev_d[slot], st));
  };
  // Both ranks vote; true iff every rank's vote was true. Throws if the exchange is aborted.
  const auto vote = [&](bool ok) {
    const int32_t mine = ok ? 1 : 0;
    std::vector<int32_t> all(static_cast<size_t>(comm.World()), 0);
    comm.HostAllGather(&mine, sizeof mine, all.data());
    bool every = true;
    for (int32_t v : all) every = every && v == 1;
    return every;
  };

  // ---- start barrier, then the pipelined plan ----
  int64_t enq = 0, n = 0;
  try {
    if (!vote(true)) throw core::TpError("driver: the peer failed its setup");
    const auto t0 = std::chrono::steady_clock::now();
    for (; enq < std::min<int64_t>(kInflight, cfg.batches); ++enq) enqueue(enq);
    for (n = 0; n < cfg.batches; ++n) {
      const int slot = static_cast<int>(n % kSlots);
      if (!WaitEvent(ev_d[slot])) DieStuck("driver batch");
      if (n >= cfg.discard) {
        float t = 0;
        R4DX_HIP_CHECK(hipEventElapsedTime(&t, ev_s[slot], ev_e[slot]));
        R.ms[n % C].push_back(t);
      }
      const R4dxTpStatus v = rec_host[static_cast<size_t>(slot)];
      const bool ok = v.mismatches == 0 && !comm.Aborted();
      if (!vote(ok)) {
        R.stopped_by_peer = ok;
        break;
      }
      ++R.batches_ok;
      if (n % C == kCondAr) R.calls_verified += K;
      if (enq < cfg.batches) enqueue(enq++);  // enqueue FIRST (both ranks keep the same work queued) ...
      if (cfg.on_batch) cfg.on_batch(R.calls_verified);  // ... then host work that might block
    }
    R.seconds = SecondsSince(t0);
  } catch (const std::exception& e) {
    R.error = e.what();
    comm.Abort(core::kAbortHost, std::string("driver: rank ") + std::to_string(rank) + ": " + e.what());
  }
  // ---- drain every enqueued batch (in-flight all-reduces pair up, or bail on the abort word) ----
  if (!tp::SyncWithWatchdog(st)) DieStuck("driver drain");
  R.next_call = base_of(enq);
  R4DX_HIP_CHECK(hipMemcpyAsync(&R.verify, rec.data(), sizeof R.verify, hipMemcpyDeviceToHost, st));
  if (!tp::SyncWithWatchdog(st)) DieStuck("driver final read");
  try {
    R.comm_status = comm.ReadStatus();
    comm.CheckHealthy();
  } catch (const std::exception& e) {
    R.abort_message = e.what();
  }
  R.ok = R.error.empty() && R.abort_message.empty() && !R.stopped_by_peer && R.batches_ok == cfg.batches &&
         R.verify.mismatches == 0;
  R.bound = bound.GetStats();
  return R;
}

// A one-line summary of a failed driver result ("" if ok).
inline std::string DescribeFailure(const DriverResult& R) {
  if (R.ok) return "";
  std::string s;
  if (!R.error.empty()) s += "error: " + R.error + "; ";
  if (!R.abort_message.empty()) s += R.abort_message + "; ";
  if (R.verify.mismatches) s += "MISMATCH: " + DescribeMismatch(R.verify) + "; ";
  if (R.stopped_by_peer) s += "stopped by the peer's failed check; ";
  s += "batches verified " + std::to_string(R.batches_ok);
  return s;
}

// tp_bench's decode_stats: per-token means of the three conditions and the paired differences.
struct DecodeStats {
  size_t n = 0;
  double ta = NAN, tb = NAN, tc = NAN;       // mean per-token ms per condition
  double L = NAN, L_se = NAN;                // vs stand-in: mean(a_i - b_i) / ars_per_token, us
  double Lnok = NAN, Lnok_se = NAN;          // vs fillers only: mean(a_i - c_i) / ars_per_token, us
  std::vector<double> L_quintiles;           // vs no AR kernel, over 5 contiguous fifths (drift check)
};
inline DecodeStats Decode(const DriverResult& X, int ars_per_token) {
  DecodeStats d;
  const size_t n = std::min(X.ms[kCondAr].size(), std::min(X.ms[kCondStandin].size(), X.ms[kCondFillers].size()));
  d.n = n;
  if (n == 0) return d;
  const double k = 1000.0 / ars_per_token;
  const auto mean_sd = [](const std::vector<double>& v, double* sd) {
    double s = 0;
    for (double x : v) s += x;
    const double m = s / static_cast<double>(v.size());
    double ss = 0;
    for (double x : v) ss += (x - m) * (x - m);
    *sd = v.size() > 1 ? std::sqrt(ss / static_cast<double>(v.size() - 1)) : 0.0;
    return m;
  };
  std::vector<double> a(X.ms[kCondAr].begin(), X.ms[kCondAr].begin() + static_cast<std::ptrdiff_t>(n));
  std::vector<double> b(X.ms[kCondStandin].begin(), X.ms[kCondStandin].begin() + static_cast<std::ptrdiff_t>(n));
  std::vector<double> c(X.ms[kCondFillers].begin(), X.ms[kCondFillers].begin() + static_cast<std::ptrdiff_t>(n));
  std::vector<double> dab(n), dac(n);
  for (size_t i = 0; i < n; ++i) {
    dab[i] = a[i] - b[i];
    dac[i] = a[i] - c[i];
  }
  double sd = 0;
  d.ta = mean_sd(a, &sd);
  d.tb = mean_sd(b, &sd);
  d.tc = mean_sd(c, &sd);
  d.L = mean_sd(dab, &sd) * k;
  d.L_se = sd / std::sqrt(static_cast<double>(n)) * k;
  d.Lnok = mean_sd(dac, &sd) * k;
  d.Lnok_se = sd / std::sqrt(static_cast<double>(n)) * k;
  if (n >= 5) {
    for (int q = 0; q < 5; ++q) {
      const size_t lo = n * static_cast<size_t>(q) / 5, hi = n * static_cast<size_t>(q + 1) / 5;
      std::vector<double> part(dac.begin() + static_cast<std::ptrdiff_t>(lo), dac.begin() + static_cast<std::ptrdiff_t>(hi));
      d.L_quintiles.push_back(mean_sd(part, &sd) * k);
    }
  }
  return d;
}

// ---- JSON helpers (the tools' --json output) ------------------------------------------------------

inline std::string JsonStr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char ch : s) {
    if (ch == '"') o += "\\\"";
    else if (ch == '\\') o += "\\\\";
    else if (ch == '\n') o += "\\n";
    else if (ch < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", ch);
      o += b;
    } else o += static_cast<char>(ch);
  }
  return o + "\"";
}
inline std::string JsonNum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, "%.10g", v);
  return b;
}
inline std::string JsonArr(const std::vector<double>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + JsonNum(v[i]);
  return o + "]";
}
inline bool WriteFile(const std::string& path, const std::string& text) {
  FILE* f = nullptr;
#ifdef _MSC_VER
  if (fopen_s(&f, path.c_str(), "wb") != 0) f = nullptr;
#else
  f = std::fopen(path.c_str(), "wb");
#endif
  if (f == nullptr) return false;
  const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  return std::fclose(f) == 0 && ok;
}

// ---- a TP group on a RankPool -------------------------------------------------------------------

// Creates the group, allocates + zeroes the mailbox on rank 0's thread, checks each device's
// WallClockRate (before any spinning kernel), and creates one endpoint and one stream per rank on
// its own thread. Endpoints and streams are destroyed on their rank threads (DestroyEndpoints)
// before the group; the streams outlive the endpoints (the endpoints sync them).
// Declare it AFTER the RankPool it is created on: if a test unwinds past Destroy(), ~TpSetup still
// tears down on the rank threads through that pool; if even that fails it leaks the endpoints,
// streams and group rather than free, on the wrong thread, memory a kernel may still touch.
struct TpSetup {
  std::unique_ptr<tp::TpGroup> group;
  std::vector<std::unique_ptr<tp::TpEndpoint>> eps;
  std::vector<std::unique_ptr<core::Stream>> streams;
  std::vector<double> clock_khz;
  RankPool* pool_ = nullptr;
  hipStream_t Stream(int r) const { return streams[static_cast<size_t>(r)]->get(); }

  TpSetup() = default;
  TpSetup(const TpSetup&) = delete;
  TpSetup& operator=(const TpSetup&) = delete;
  ~TpSetup() {
    if (!group || pool_ == nullptr) return;
    try {
      Destroy(*pool_);
    } catch (...) {
      std::fprintf(stderr, "[tp_harness] teardown on the rank threads failed; leaking the TP group\n");
      for (auto& e : eps) (void)e.release();
      for (auto& s : streams) (void)s.release();
      (void)group.release();
    }
  }

  void Create(RankPool& pool, const tp::TpGroup::Geometry& geo, int timeout_ms,
              uint32_t seq_base = tp::kInitialSeqBase) {
    pool_ = &pool;
    group = tp::TpGroup::Create(tp::TpGroup::Mode::kReal, pool.World(), geo, timeout_ms, seq_base);
    eps.resize(static_cast<size_t>(pool.World()));
    streams.resize(static_cast<size_t>(pool.World()));
    clock_khz.assign(static_cast<size_t>(pool.World()), 0.0);
    pool.RunOne(0, [&] { group->AllocateMailbox(); });
    pool.RunAll([&](int r) {
      clock_khz[static_cast<size_t>(r)] = tp::CheckWallClockRate();
      streams[static_cast<size_t>(r)] = std::make_unique<core::Stream>();
      eps[static_cast<size_t>(r)] = group->CreateEndpoint(r, &pool.Heartbeat(r));
    });
  }
  void DestroyEndpoints(RankPool& pool) {
    pool.RunAll([&](int r) {
      eps[static_cast<size_t>(r)].reset();
      streams[static_cast<size_t>(r)].reset();
    });
  }
  void Destroy(RankPool& pool) {
    if (!group) return;
    DestroyEndpoints(pool);
    group.reset();
  }
};

}  // namespace tp_harness
