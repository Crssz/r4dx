// tests/kernels/test_tp_allreduce_2gpu.cpp -- HostMailboxComm end to end on both GPUs
// (docs/tp.md 6.3, 10.1; ctest LABEL tp2gpu, OPT-IN).
//
// Opt-in (docs/tp.md 10.1): unless R4DX_TP2GPU is exactly "1" it prints SKIP and exits 77; with
// fewer than two visible HIP devices it exits 77 too; then the per-device free-VRAM pre-flight
// (exit 1, not a skip). `tests\run_tests.ps1 -TwoGpu` is the one sanctioned way to run it: it
// unsets HIP_VISIBLE_DEVICES and sets R4DX_TP2GPU=1. Devices: `--tp-devices auto` (9.2), i.e. rank
// 0 = the last visible ordinal (physical device 1 with HIP_VISIBLE_DEVICES unset), rank 1 = the one
// before it.
//
// Cases:
//   1. load-time SelfTest (6.3.9) on both endpoints;
//   2. 1,000,000 all-reduces, sizes cycling over both channels, a VRAM filler kernel that dirties L2
//      before every all-reduce, every output element verified bit-exact every batch (tp_bench's
//      hash pattern; tests/kernels/tp_ar_harness.h);
//   3. abort propagation: rank 0 enqueues an all-reduce rank 1 never joins; 50 ms later rank 1
//      calls Abort() -> rank 0's kernel exits in < 5 ms, and both ranks' CheckHealthy() throw
//      TpAbortedError;
//   4. TpGroup::Recover() (2.5: seq rebase above every value used, mailbox zeroed, counters reset,
//      lockstep check, SelfTest), then 10,000 more clean verified all-reduces;
//   5. a DEVICE timeout: fault injection kind 1 stalls rank 1 for 700 ms before its 50th
//      all-reduce, a kernel of rank 0 times out (ABORT[0] = kAbortTimeout, Status claimed with
//      channel 0, later calls skipped), rank 1 bails on the abort word, both CheckHealthy() throw
//      TpAbortedError; then Recover() (new base = old + max advance + 64) and 10,000 clean ones.
// Every spin is bounded (500 ms all-reduce timeout, sticky abort); device 0 drives the desktop.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "tp_ar_harness.h"

using namespace tp_harness;

namespace {

int g_failures = 0;
void Expect(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  std::fflush(stdout);
  if (!ok) ++g_failures;
}

const std::vector<size_t> kMixedSizes = {10240, 81920, 174080, 184320, 655360, 20480, 40960, 122880};

DriverConfig MixedConfig(int64_t calls, uint64_t call0) {
  DriverConfig c;
  c.sizes = kMixedSizes;
  c.K = 50;
  c.batches = (calls + c.K - 1) / c.K;
  c.fillers = true;  // dirty L2 between all-reduces
  c.filler_rd_bytes = size_t{2} << 20;
  c.filler_dirty_bytes = size_t{1} << 20;
  c.filler_ring_bytes = size_t{64} << 20;
  c.call0 = call0;
  return c;
}

// Runs one verified stress on both ranks; returns the pattern call index after it.
uint64_t Stress(RankPool& pool, TpSetup& T, const char* label, int64_t calls, uint64_t call0) {
  std::vector<DriverResult> res(2);
  const auto t0 = std::chrono::steady_clock::now();
  auto last = t0;
  pool.RunAll([&](int r) {
    DriverConfig c = MixedConfig(calls, call0);
    if (r == 0) {
      c.on_batch = [&](int64_t done) {
        if (SecondsSince(last) >= 5.0) {
          last = std::chrono::steady_clock::now();
          std::printf("    [%s] %lld / %lld all-reduces verified, %.0f s\n", label, static_cast<long long>(done),
                      static_cast<long long>(calls), SecondsSince(t0));
          std::fflush(stdout);
        }
      };
    }
    res[static_cast<size_t>(r)] = RunDriver(*T.eps[static_cast<size_t>(r)], r, T.Stream(r), c);
  });
  const double secs = SecondsSince(t0);
  for (int r = 0; r < 2; ++r) {
    const DriverResult& R = res[static_cast<size_t>(r)];
    Expect(R.ok, std::string(label) + " rank " + std::to_string(r) + ": " + std::to_string(R.calls_verified) +
                     " all-reduces verified bit-exact in " + std::to_string(secs) + " s" +
                     (R.ok ? "" : " -- " + DescribeFailure(R)));
  }
  return res[0].next_call;
}

void AbortPropagation(RankPool& pool, TpSetup& T) {
  std::atomic<int64_t> t_abort_ns{0};
  double exit_ms = -1.0;
  bool finished_before_abort = false;
  std::string msg[2];
  uint32_t bailed = 0;
  const auto now_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  pool.RunAll([&](int r) {
    tp::TpEndpoint& ep = *T.eps[static_cast<size_t>(r)];
    const hipStream_t st = T.Stream(r);
    core::DeviceBuffer<uint16_t> buf(5120);  // 10 KiB, channel 0
    R4DX_HIP_CHECK(hipMemsetAsync(buf.data(), 0, buf.bytes(), st));
    if (!tp::SyncWithWatchdog(st)) DieStuck("abort case setup");
    ep.HostAllGather(nullptr, 0, nullptr);  // both ready
    if (r == 0) {
      hipEvent_t done = nullptr;
      R4DX_HIP_CHECK(hipEventCreateWithFlags(&done, hipEventDisableTiming));
      ep.AllReduceSumBf16(buf.data(), 5120, st);  // rank 1 never joins: the kernel spins
      R4DX_HIP_CHECK(hipEventRecord(done, st));
      ep.HostAllGather(nullptr, 0, nullptr);  // tell rank 1 it is enqueued
      const auto t0 = std::chrono::steady_clock::now();
      for (;;) {  // an event poll, never hipStreamQuery (docs/tp.md Appendix B N33)
        const hipError_t e = hipEventQuery(done);
        if (e == hipSuccess) break;
        if (e != hipErrorNotReady) throw core::HipError(e, "hipEventQuery", __FILE__, __LINE__);
        if (SecondsSince(t0) > 30.0) DieStuck("abort case");
      }
      const int64_t t1 = now_ns();
      (void)hipGetLastError();
      (void)hipEventDestroy(done);
      for (int i = 0; i < 1000000 && t_abort_ns.load() == 0; ++i) std::this_thread::yield();
      const int64_t ta = t_abort_ns.load();
      finished_before_abort = ta == 0 || t1 < ta;
      exit_ms = static_cast<double>(t1 - ta) / 1e6;
      bailed = ep.ReadStatus().n_abort_exits;
    } else {
      ep.HostAllGather(nullptr, 0, nullptr);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));  // rank 0's kernel is spinning now
      t_abort_ns.store(now_ns());
      ep.Abort(core::kAbortHost, "test: rank 1 aborts");
    }
    try {
      ep.CheckHealthy();
    } catch (const core::TpAbortedError& e) {
      msg[r] = e.what();
    }
  });
  Expect(!finished_before_abort && exit_ms >= 0.0 && exit_ms < 5.0,
         "rank 1 Abort() -> rank 0's spinning kernel exited " + std::to_string(exit_ms) + " ms later (limit 5 ms)" +
             (finished_before_abort ? " -- the kernel had finished BEFORE the abort" : ""));
  Expect(bailed >= 1, "rank 0's Status counts " + std::to_string(bailed) + " blocks that bailed on the abort word");
  for (int r = 0; r < 2; ++r) {
    Expect(!msg[r].empty(), "rank " + std::to_string(r) + " CheckHealthy() throws TpAbortedError: \"" + msg[r] + "\"");
  }
}

// Recovers and checks the new base: old base + max over ranks of SeqAdvance() + 64 (2.5 step 3).
// Returns true if Recover() did not throw and the base is as expected.
bool RecoverAndCheck(RankPool& pool, TpSetup& T, const char* label) {
  uint32_t adv[2] = {0, 0};
  pool.RunAll([&](int r) { adv[r] = T.eps[static_cast<size_t>(r)]->SeqAdvance(); });
  const uint32_t base_before = T.group->SeqBase();
  T.group->Recover(pool.Runner());
  const uint32_t base_after = T.group->SeqBase();
  const uint32_t want = base_before + std::max(adv[0], adv[1]) + 64u;
  char line[256];
  std::snprintf(line, sizeof line,
                "%s: Recover(): seq base 0x%08x -> 0x%08x (advance rank 0 %u, rank 1 %u; want 0x%08x), abort "
                "words and exchange cleared, SelfTest passed",
                label, base_before, base_after, adv[0], adv[1], want);
  const bool ok = base_after == want && static_cast<int32_t>(base_after - base_before) > 0 &&
                  static_cast<int32_t>(base_after - (base_before + adv[0])) > 0 && !T.eps[0]->Aborted() &&
                  !T.eps[1]->Aborted();
  Expect(ok, line);
  return ok;
}

// Case 5: a DEVICE timeout (fault injection kind 1, 2.5's most likely production failure). Rank 1
// sleeps 700 ms before enqueuing its 50th all-reduce, so a kernel of rank 0 (device 1) spins its
// 500 ms timeout: it release-stores ABORT[0] = kAbortTimeout, claims its Status and sets sticky, and
// every later call of rank 0 skips while still bumping its seq. Rank 1's call at that seq then
// completes (rank 0's flag was posted before its spin) and its next one bails on ABORT[0].
// Which call times out is up to the runtime: measured, it is rank 0's FIRST, because rank 1's 49
// launches before the sleep were not submitted to its GPU until after it (docs/tp.md N43). No
// assertion depends on which. Returns the pattern call index after the run.
uint64_t DeviceTimeout(RankPool& pool, TpSetup& T, uint64_t call0) {
  constexpr int kAt = 50;
  pool.RunOne(1, [&] { T.eps[1]->ArmFaultInjection(kAt, tp::kFaultStall); });
  std::vector<DriverResult> res(2);
  pool.RunAll([&](int r) {
    DriverConfig c;
    c.sizes = {10240};
    c.K = 50;
    c.batches = 4;  // up to 200 all-reduces; the driver stops at the first failed batch
    c.call0 = call0;
    res[static_cast<size_t>(r)] = RunDriver(*T.eps[static_cast<size_t>(r)], r, T.Stream(r), c);
  });
  const uint32_t ab0 = tp::HostU32Load(
      reinterpret_cast<const volatile uint32_t*>(T.group->MailboxHost() + tp::MailboxLayout::kAbortOff[0]));
  const uint32_t ab1 = tp::HostU32Load(
      reinterpret_cast<const volatile uint32_t*>(T.group->MailboxHost() + tp::MailboxLayout::kAbortOff[1]));
  Expect(!res[0].ok && !res[1].ok && res[0].error.empty() && res[1].error.empty(),
         "both drivers stopped on the failure without a host error (rank 0: " + DescribeFailure(res[0]) +
             "; rank 1: " + DescribeFailure(res[1]) + ")");
  Expect(ab0 == core::kAbortTimeout && ab1 == 0,
         "ABORT words: rank 0 " + std::to_string(ab0) + " (want kAbortTimeout = 1), rank 1 " + std::to_string(ab1) +
             " (want 0)");
  const R4dxTpStatus& s0 = res[0].comm_status;
  const R4dxTpStatus& s1 = res[1].comm_status;
  Expect(s0.claimed == 1 && s0.code == core::kAbortTimeout && s0.word == 0 && s0.phase == kR4dxTpPhaseFlagWait &&
             s0.sticky == 1 && s0.n_skipped > 0,
         "rank 0 Status: claimed " + std::to_string(s0.claimed) + ", code " + std::to_string(s0.code) + ", channel " +
             std::to_string(s0.word) + ", block " + std::to_string(s0.block) + ", seq " + std::to_string(s0.seq) +
             ", timed-out blocks " + std::to_string(s0.n_timeouts) + ", skipped blocks " + std::to_string(s0.n_skipped));
  Expect(s1.claimed == 0 && s1.n_abort_exits >= 1 && s1.sticky == 1,
         "rank 1 Status: claimed " + std::to_string(s1.claimed) + ", " + std::to_string(s1.n_abort_exits) +
             " blocks bailed on the abort word, " + std::to_string(s1.n_skipped) + " skipped (sticky " +
             std::to_string(s1.sticky) + ")");
  std::string msg[2];
  pool.RunAll([&](int r) {
    try {
      T.eps[static_cast<size_t>(r)]->CheckHealthy();
    } catch (const core::TpAbortedError& e) {
      msg[r] = e.what();
    }
  });
  Expect(msg[0].find("timeout at channel 0") != std::string::npos,
         "rank 0 CheckHealthy() throws TpAbortedError: \"" + msg[0] + "\"");
  Expect(!msg[1].empty(), "rank 1 CheckHealthy() throws TpAbortedError: \"" + msg[1] + "\"");
  return std::max(res[0].next_call, res[1].next_call);
}

}  // namespace

int main() {
  if (GetEnv("R4DX_TP2GPU") != "1") {
    std::printf("SKIP: two-GPU test; run tests\\run_tests.ps1 -TwoGpu\n");
    return 77;
  }
  const int visible = VisibleDevices();
  if (visible < 2) {
    std::printf("SKIP: two-GPU test needs two visible HIP devices, HIP_VISIBLE_DEVICES exposes %d\n", visible);
    return 77;
  }
  const std::vector<int> devices = AutoDevices();
  PreflightVram(devices, 1.0);
  try {
    RankPool pool(devices);
    TpSetup T;
    pool.on_error = [&](int r, const std::string& what) {
      if (T.eps.size() > static_cast<size_t>(r) && T.eps[static_cast<size_t>(r)]) {
        T.eps[static_cast<size_t>(r)]->Abort(core::kAbortHost, what);
      }
    };
    T.Create(pool, tp::TpGroup::Geometry{}, tp::kArTimeoutDefaultMs);
    std::printf("test_tp_allreduce_2gpu: rank 0 -> %s, rank 1 -> %s; WallClockRate checks %.0f / %.0f kHz; mailbox "
                "%zu B\n",
                DeviceLabel(devices[0]).c_str(), DeviceLabel(devices[1]).c_str(), T.clock_khz[0], T.clock_khz[1],
                T.group->Layout().bytes);

    std::printf("1. SelfTest\n");
    pool.RunAll([&](int r) { T.eps[static_cast<size_t>(r)]->SelfTest(); });
    Expect(true, "SelfTest passed on both ranks (channels 0 and 1, rows 1/8/17/18/64)");

    std::printf("2. 1,000,000 all-reduces, mixed sizes, fillers between them\n");
    uint64_t call = Stress(pool, T, "stress", 1000000, 0);

    std::printf("3. abort propagation\n");
    AbortPropagation(pool, T);

    std::printf("4. Recover, then 10,000 clean all-reduces\n");
    RecoverAndCheck(pool, T, "host abort");
    call = Stress(pool, T, "after recovery", 10000, call);
    auto c0 = T.eps[0]->CallCounts(), c1 = T.eps[1]->CallCounts();
    Expect(c0 == c1, "both endpoints report the same CallCounts() since recovery (" + std::to_string(c0[0]) + " / " +
                         std::to_string(c0[1]) + ")");

    std::printf("5. device timeout (fault injection: rank 1 stalls 700 ms), Recover, 10,000 clean all-reduces\n");
    call = DeviceTimeout(pool, T, call);
    RecoverAndCheck(pool, T, "device timeout");
    call = Stress(pool, T, "after timeout recovery", 10000, call);
    c0 = T.eps[0]->CallCounts();
    c1 = T.eps[1]->CallCounts();
    const core::TpCommStats st0 = T.eps[0]->Stats();
    Expect(c0 == c1 && st0.ar_calls[0] == c0[0] && st0.ar_calls[1] == c0[1],
           "both endpoints report the same CallCounts() since recovery (" + std::to_string(c0[0]) + " / " +
               std::to_string(c0[1]) + "), equal to rank 0's Stats().ar_calls");
    T.Destroy(pool);
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
