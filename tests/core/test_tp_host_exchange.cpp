// tests/core/test_tp_host_exchange.cpp -- CPU-only (threads, no GPU). docs/tp.md 6.6 / 10.1.
//
// r4dx::core::HostExchange, the in-process rendezvous of the tensor-parallel rank threads:
//   * 2 threads x 1,000,000 all-gathers of varying sizes (0 B .. 536 B, plus a full vocab row of
//     496,640 B every 10,000th call, with a Barrier every 1,000th), every byte of every gathered
//     slot verified; heartbeats bumped on entry and exit of every wait;
//   * Abort() wakes a waiter with TpAbortedError (and later calls throw at once);
//   * a peer that never arrives -> TpTimeoutError after the timeout, and the group is aborted;
//     BarrierFor / AllGatherFor honour their own per-call bound (EmulatedComm, docs/tp.md N45);
//   * Reset() zeroes arrive[] AND the per-rank generations: after an asymmetric abort -- rank 1
//     already counted the call, rank 0 never made it -- Reset() gives correct contents on the next
//     1,000 gathers;
//   * a size mismatch is a TpDivergenceError; world 1; bad rank.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/core/tp_host_exchange.hpp"

using r4dx::core::HostExchange;
using r4dx::core::TpAbortedError;
using r4dx::core::TpDivergenceError;
using r4dx::core::TpTimeoutError;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

// The payload word j of rank r's call i.
uint64_t Word(int r, uint64_t i, size_t j) {
  uint64_t x = ((i + 1) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(r) << 56) ^
               (j * 0xD6E8FEB86659FD93ull);
  x ^= x >> 31;
  return x * 0xBF58476D1CE4E5B9ull;
}

// Bytes of call i: 0..536 in steps of 8, a 496,640-B full vocab row every 10,000th call.
size_t CallBytes(uint64_t i) {
  if (i % 10000 == 9999) return 496640;
  return static_cast<size_t>((i * 2654435761ull) % 68) * 8;
}

// Runs `calls` all-gathers on both ranks of `ex` (starting at call index `first`), verifying every
// word; returns the number of bad gathers seen by either rank.
int64_t RunGathers(HostExchange& ex, uint64_t first, uint64_t calls, bool barriers) {
  std::atomic<int64_t> bad{0};
  std::atomic<bool> threw{false};
  const auto rank_main = [&](int r) {
    std::vector<uint64_t> mine(496640 / 8), out(2 * 496640 / 8);
    try {
      for (uint64_t i = first; i < first + calls; ++i) {
        const size_t bytes = CallBytes(i), words = bytes / 8;
        for (size_t j = 0; j < words; ++j) mine[j] = Word(r, i, j);
        ex.AllGather(r, mine.data(), bytes, out.data());
        bool ok = true;
        for (int q = 0; q < 2 && ok; ++q) {
          for (size_t j = 0; j < words; ++j) {
            if (out[q * words + j] != Word(q, i, j)) {
              ok = false;
              break;
            }
          }
        }
        if (!ok) bad.fetch_add(1);
        if (barriers && i % 1000 == 999) ex.Barrier(r);
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "rank %d threw: %s\n", r, e.what());
      threw = true;
    }
  };
  std::thread t0(rank_main, 0), t1(rank_main, 1);
  t0.join();
  t1.join();
  return threw ? -1 : bad.load();
}

void TestMillionGathers() {
  HostExchange ex(2);
  std::atomic<uint64_t> hb0{0}, hb1{0};
  ex.SetHeartbeat(0, &hb0);
  ex.SetHeartbeat(1, &hb1);
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t calls = 1000000;
  const int64_t bad = RunGathers(ex, 0, calls, /*barriers=*/true);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  1,000,000 all-gathers + 1,000 barriers on 2 threads: %.2f s (%.2f us each)\n", s,
              s * 1e6 / static_cast<double>(calls));
  Check(bad == 0, "2 threads x 1,000,000 all-gathers of varying sizes: every byte correct");
  const uint64_t waits = calls + calls / 1000;
  Check(hb0.load() == 2 * waits && hb1.load() == 2 * waits,
        "the heartbeat is bumped on entry and exit of every wait (" + std::to_string(hb0.load()) +
            " == " + std::to_string(2 * waits) + ")");
  Check(!ex.Aborted(), "no abort after a clean run");
}

void TestAbortWakesWaiter() {
  HostExchange ex(2);
  std::string caught;
  std::chrono::steady_clock::time_point woke;
  std::thread waiter([&] {
    uint64_t v = 42, out[2];
    try {
      ex.AllGather(1, &v, 8, out);  // rank 0 never arrives
    } catch (const TpAbortedError& e) {
      caught = e.what();
    } catch (const std::exception& e) {
      caught = std::string("WRONG TYPE: ") + e.what();
    }
    woke = std::chrono::steady_clock::now();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));  // past the 50 ms spin phase
  const auto aborted_at = std::chrono::steady_clock::now();
  ex.Abort("rank 0 threw: test abort");
  waiter.join();
  // Both times on one clock, taken on either side of Abort(): immune to a late thread start.
  const double woke_after_ms =
      std::chrono::duration<double, std::milli>(woke - aborted_at).count();
  Check(caught.find("test abort") != std::string::npos,
        "Abort() wakes a blocked waiter with TpAbortedError carrying the reason [" + caught + "]");
  Check(woke_after_ms >= 0.0 && woke_after_ms < 5000.0,
        "the waiter blocked until the abort, then woke (" + std::to_string(woke_after_ms) +
            " ms after it)");
  Check(ex.Aborted(), "Aborted() after Abort()");
  bool again = false;
  try {
    uint64_t v = 1, out[2];
    ex.AllGather(0, &v, 8, out);
  } catch (const TpAbortedError&) {
    again = true;
  }
  Check(again, "every later call throws TpAbortedError until Reset()");
}

void TestTimeout() {
  HostExchange ex(2, std::chrono::milliseconds(200));
  const auto t0 = std::chrono::steady_clock::now();
  bool timed_out = false;
  try {
    uint64_t v = 7, out[2];
    ex.AllGather(0, &v, 8, out);  // rank 1 never arrives
  } catch (const TpTimeoutError&) {
    timed_out = true;
  }
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Check(timed_out && ms >= 200.0 && ms < 3000.0,
        "a peer that never arrives -> TpTimeoutError after the 200 ms timeout (" +
            std::to_string(ms) + " ms)");
  Check(ex.Aborted(), "the timeout aborts the group");
}

// BarrierFor / AllGatherFor wait at most their own bound, not the constructor's (the 30 s default
// here): EmulatedComm's all-reduce barrier is bounded by the all-reduce timeout (docs/tp.md N45).
void TestPerCallBound() {
  HostExchange ex(2);  // 30 s constructor timeout
  const auto t0 = std::chrono::steady_clock::now();
  bool timed_out = false;
  try {
    ex.BarrierFor(0, std::chrono::milliseconds(150));  // rank 1 never arrives
  } catch (const TpTimeoutError&) {
    timed_out = true;
  }
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Check(timed_out && ms >= 150.0 && ms < 3000.0,
        "BarrierFor(150 ms) times out at its own bound, not the 30 s constructor timeout (" +
            std::to_string(ms) + " ms)");
  Check(ex.Aborted(), "a BarrierFor timeout aborts the group");

  // After Reset(), AllGatherFor with both ranks present gathers normally.
  ex.Reset();
  uint64_t got[2][2] = {};
  std::thread t1([&] {
    const uint64_t v = 11;
    ex.AllGatherFor(1, &v, 8, got[1], std::chrono::milliseconds(5000));
  });
  const uint64_t v0 = 10;
  ex.AllGatherFor(0, &v0, 8, got[0], std::chrono::milliseconds(5000));
  t1.join();
  Check(got[0][0] == 10 && got[0][1] == 11 && got[1][0] == 10 && got[1][1] == 11,
        "AllGatherFor with both ranks present gathers both slots");
}

void TestResetAfterAsymmetricAbort() {
  HostExchange ex(2);
  Check(RunGathers(ex, 0, 10, false) == 0, "10 clean gathers before the fault");
  // Rank 1 enters call 11 (its generation is now 11, its arrive word 11); rank 0 "threw before
  // its increment" and never calls. The group aborts; rank 1 wakes with TpAbortedError.
  bool aborted = false;
  std::thread r1([&] {
    uint64_t v = 5, out[2];
    try {
      ex.AllGather(1, &v, 8, out);
    } catch (const TpAbortedError&) {
      aborted = true;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ex.Abort("rank 0: injected fault before its exchange");
  r1.join();
  Check(aborted, "asymmetric abort: rank 1 (already counted) wakes with TpAbortedError");

  ex.Reset();
  Check(!ex.Aborted(), "Reset() clears the abort flag");
  // Without the generation reset rank 0 would now run call 11 against rank 1's call 12, see
  // arrive[1] = 11 >= 11 at once and read rank 1's stale call-11 slot.
  Check(RunGathers(ex, 1000, 1000, false) == 0,
        "after an asymmetric abort + Reset(), the next 1,000 gathers are all correct");
}

void TestSizeMismatchAndWorldOne() {
  HostExchange ex(2);
  bool div0 = false, div1 = false;
  std::thread t0([&] {
    uint64_t v[2] = {1, 2}, out[4];
    try {
      ex.AllGather(0, v, 16, out);
    } catch (const TpDivergenceError&) {
      div0 = true;
    } catch (const TpAbortedError&) {
      div0 = true;  // the peer detected it first and aborted the group
    }
  });
  std::thread t1([&] {
    uint64_t v = 3, out[2];
    try {
      ex.AllGather(1, &v, 8, out);
    } catch (const TpDivergenceError&) {
      div1 = true;
    } catch (const TpAbortedError&) {
      div1 = true;
    }
  });
  t0.join();
  t1.join();
  Check(div0 && div1 && ex.Aborted(),
        "ranks gathering different sizes both fail (TpDivergenceError) and the group aborts");

  HostExchange solo(1);
  const uint32_t mine[3] = {7, 8, 9};
  uint32_t out[3] = {};
  solo.AllGather(0, mine, sizeof(mine), out);
  solo.Barrier(0);
  Check(std::memcmp(mine, out, sizeof(mine)) == 0, "world 1: AllGather copies mine to out");

  bool bad_rank = false;
  try {
    ex.Barrier(2);
  } catch (const std::invalid_argument&) {
    bad_rank = true;
  }
  Check(bad_rank, "a rank outside [0, world) is refused");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  TestMillionGathers();
  TestAbortWakesWaiter();
  TestTimeout();
  TestPerCallBound();
  TestResetAfterAsymmetricAbort();
  TestSizeMismatchAndWorldOne();
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
