// tests/kernels/test_tp_allreduce_cpu_peer.cpp -- the engine's tensor-parallel all-reduce kernel
// (docs/tp.md 6.3, 10.1) on ONE GPU against a CPU thread acting as rank 1 through a real pinned
// TpGroup mailbox.
//
// Rank 0 is the real HostMailboxComm endpoint (TpGroup::CreateEndpoint) on HIP device 0 of the
// process (ctest sets HIP_VISIBLE_DEVICES=1, so physical device 1). Rank 1 is a CPU thread that
// speaks the protocol with the same MailboxLayout, ChunkCall and HostU32Load/Store the endpoint
// uses: per block b of call s it writes its words into rank 0's receive slot (s & 1), release-
// stores rank 0's flag line b = s, waits (wrap-safe) for rank 0's flag >= s, and checks the words
// rank 0's GPU pushed into its own receive slot. Cases:
//   1. 100,000 all-reduces cycling through {10240, 81920, 174080, 655360} B (both channels), every
//      output element verified on the device, every word the GPU pushed verified on the CPU:
//      bit-exact.
//   2. Session seq base 0xFFFFFF00: 1,200 all-reduces carry every block's per-call seq across the
//      2^32 wrap on both channels, bit-exact.
//   3. Fault injection (kind 0) throws before enqueuing; then with the CPU peer silent, one
//      all-reduce times out within timeout + 10% (100 ms timeout), rank 0's ABORT word reads
//      kAbortTimeout, the VRAM Status names channel/block/phase, the next call is skipped (sticky:
//      returns in < 5 ms, buffer untouched), and CheckHealthy() throws TpAbortedError.
//   4. r4dx_tp_add_bf16 (EmulatedComm's add, same add_bf16x8) is the exact sum of both patterns.
//
// Data: tp_bench's hash pattern (tp_kernels.h), with the pattern call index = call % 13 so the CPU
// can keep every rank's rows precomputed (13 is odd: data stale by 2 calls -- the only staleness the
// slot protocol could produce -- still differs from the expected row, as does any wrong offset).
#include <hip/hip_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

namespace tp = r4dx::model::tp;
namespace core = r4dx::core;

namespace {

constexpr int kPhases = 13;
constexpr size_t kMaxElems = tp::kMaxAllReduceBytes / 2;

int g_failures = 0;
void Expect(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

double Since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// rows[r][p][j]: rank r's bf16 bits for pattern index p, element j.
std::vector<uint16_t> g_rows[2];
const uint16_t* Row(int rank, int phase) {
  return g_rows[rank].data() + static_cast<size_t>(phase) * kMaxElems;
}
void BuildRows() {
  for (int r = 0; r < 2; ++r) {
    g_rows[r].resize(static_cast<size_t>(kPhases) * kMaxElems);
    for (int p = 0; p < kPhases; ++p) {
      uint16_t* dst = g_rows[r].data() + static_cast<size_t>(p) * kMaxElems;
      for (uint32_t j = 0; j < kMaxElems; ++j) dst[j] = r4dx_tp_test_int_bf16(r4dx_tp_test_pattern(p, j, r));
    }
  }
}

// Rank 1 on the CPU. Keeps its own per-block seq counters, like the GPU's VRAM ones.
class CpuPeer {
 public:
  CpuPeer(tp::TpGroup& g, const std::atomic<bool>* stop) : L_(g.Layout()), host_(g.MailboxHost()), stop_(stop) {
    for (auto& ch : seq_)
      for (uint32_t& s : ch) s = g.SeqBase();
    abort_[0] = reinterpret_cast<volatile uint32_t*>(host_ + tp::MailboxLayout::kAbortOff[0]);
    abort_[1] = reinterpret_cast<volatile uint32_t*>(host_ + tp::MailboxLayout::kAbortOff[1]);
  }

  // Rank 1's side of the all-reduce of pattern call `call` (phase call % kPhases), `bytes` long.
  bool Run(uint64_t call, size_t bytes, std::string* err) {
    const int ch = tp::MailboxLayout::ChannelFor(bytes);
    const tp::MailboxLayout::Channel& C = L_.ch[ch];
    const tp::CallChunks g = tp::ChunkCall(bytes, C.nb);
    const int phase = static_cast<int>(call % kPhases);
    uint32_t s[kR4dxTpNbMax];
    // PUSH every block's words into rank 0's slot, then release its flag line.
    for (int b = 0; b < g.nb; ++b) {
      s[b] = ++seq_[ch][b];
      const uint32_t slot = s[b] & 1u;
      const int start = b * g.w, end = std::min(start + g.w, g.n16);
      uint8_t* dst = host_ + C.mbox_off[0] + static_cast<size_t>(slot) * C.slot_stride +
                     static_cast<size_t>(b) * C.blk_stride;
      std::memcpy(dst, Row(1, phase) + static_cast<size_t>(start) * 8, static_cast<size_t>(end - start) * 16);
      tp::HostU32Store(Flag(C.flags_off[0], b), s[b]);
    }
    // WAIT for rank 0's flags, then check what its GPU pushed into my slot.
    for (int b = 0; b < g.nb; ++b) {
      if (!WaitFlag(Flag(C.flags_off[1], b), s[b], call, ch, b, err)) return false;
      const uint32_t slot = s[b] & 1u;
      const int start = b * g.w, end = std::min(start + g.w, g.n16);
      const uint16_t* got = reinterpret_cast<const uint16_t*>(host_ + C.mbox_off[1] +
                                                              static_cast<size_t>(slot) * C.slot_stride +
                                                              static_cast<size_t>(b) * C.blk_stride);
      const uint16_t* want = Row(0, phase) + static_cast<size_t>(start) * 8;
      const size_t n = static_cast<size_t>(end - start) * 8;
      if (std::memcmp(got, want, n * 2) != 0) {
        size_t i = 0;
        while (i < n && got[i] == want[i]) ++i;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "CPU peer: rank 0's pushed data wrong at call %llu (%zu B, channel %d) block %d element %zu: "
                      "0x%04x, want 0x%04x",
                      static_cast<unsigned long long>(call), bytes, ch, b, static_cast<size_t>(start) * 8 + i, got[i],
                      want[i]);
        *err = buf;
        return false;
      }
    }
    return true;
  }

 private:
  volatile uint32_t* Flag(size_t off, int b) const {
    return reinterpret_cast<volatile uint32_t*>(host_ + off + static_cast<size_t>(b) * kR4dxTpLineBytes);
  }
  bool WaitFlag(volatile uint32_t* f, uint32_t want, uint64_t call, int ch, int b, std::string* err) {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t spins = 1;; ++spins) {
      const uint32_t v = tp::HostU32Load(f);
      const int d = static_cast<int>(v - want);
      if (d >= 0) {
        if (d >= 2) {
          *err = "CPU peer: protocol violation: rank 0's flag " + std::to_string(v) + " is >= 2 ahead of " +
                 std::to_string(want) + " at call " + std::to_string(call);
          return false;
        }
        return true;
      }
      if ((spins & 1023u) == 0u) {
        if (stop_->load(std::memory_order_relaxed)) {
          *err = "CPU peer: stopped by the GPU side";
          return false;
        }
        if (tp::HostU32Load(abort_[0]) != 0) {
          *err = "CPU peer: rank 0's ABORT word is set (" + std::to_string(tp::HostU32Load(abort_[0])) + ")";
          return false;
        }
        if (Since(t0) > 3.0) {
          *err = "CPU peer: timed out (3 s) waiting for rank 0's flag at call " + std::to_string(call) + " channel " +
                 std::to_string(ch) + " block " + std::to_string(b) + " (want " + std::to_string(want) + ", have " +
                 std::to_string(v) + ")";
          return false;
        }
      }
      _mm_pause();
    }
  }

  tp::MailboxLayout L_;
  uint8_t* host_;
  const std::atomic<bool>* stop_;
  volatile uint32_t* abort_[2];
  uint32_t seq_[2][kR4dxTpNbMax];
};

// One group, rank 0 on the GPU and rank 1 on the CPU: `count` all-reduces, sizes cycling through
// `sizes`, every output verified. Returns true when every call verified on both sides.
bool RunBitExact(const char* label, uint32_t seq_base, const std::vector<size_t>& sizes, int64_t count) {
  std::printf("%s: %lld all-reduces, sizes", label, static_cast<long long>(count));
  for (size_t s : sizes) std::printf(" %zu", s);
  std::printf(" B, seq base 0x%08x\n", seq_base);
  // The production 500 ms spin bound: a CPU peer descheduled for longer trips the GPU's timeout,
  // and that is a test FAILURE, never a reason to lengthen the device spin (docs/tp.md N43).
  auto group = tp::TpGroup::Create(tp::TpGroup::Mode::kReal, 2, {}, tp::kArTimeoutDefaultMs, seq_base);
  group->AllocateMailbox();
  core::Stream stream;  // outlives the endpoint (declared first)
  const hipStream_t st = stream.get();
  const int64_t s64 = reinterpret_cast<int64_t>(st);
  std::unique_ptr<tp::TpEndpoint> ep = group->CreateEndpoint(0, nullptr);

  constexpr int K = 50, kInflight = 3, kSlots = kInflight + 1;
  core::DeviceBuffer<uint16_t> ring(static_cast<size_t>(K) * kMaxElems);
  core::DeviceBuffer<uint8_t> rec(sizeof(R4dxTpStatus));
  core::PinnedBuffer<R4dxTpStatus> rec_host(kSlots);
  hipEvent_t ev[kSlots];
  for (hipEvent_t& e : ev) R4DX_HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  R4DX_HIP_CHECK(hipMemsetAsync(rec.data(), 0, rec.bytes(), st));
  R4DX_HIP_CHECK(hipMemsetAsync(rec.data() + offsetof(R4dxTpStatus, min_key), 0xFF, 8, st));
  if (!tp::SyncWithWatchdog(st)) throw std::runtime_error("stream stuck");

  std::atomic<bool> stop{false};
  std::string peer_err;
  std::atomic<int64_t> peer_done{0};
  CpuPeer peer(*group, &stop);
  std::thread cpu([&] {
    for (int64_t c = 0; c < count; ++c) {
      if (!peer.Run(static_cast<uint64_t>(c), sizes[static_cast<size_t>(c) % sizes.size()], &peer_err)) {
        // Rank 1's host error: its ABORT word, so rank 0's spinning kernels bail at once.
        tp::HostU32Store(reinterpret_cast<volatile uint32_t*>(group->MailboxHost() + tp::MailboxLayout::kAbortOff[1]),
                         core::kAbortHost);
        return;
      }
      peer_done.store(c + 1, std::memory_order_relaxed);
    }
  });

  const int64_t batches = (count + K - 1) / K;
  const auto enqueue = [&](int64_t n) {
    for (int k = 0; k < K; ++k) {
      const int64_t c = n * K + k;
      if (c >= count) break;
      const size_t bytes = sizes[static_cast<size_t>(c) % sizes.size()];
      uint16_t* row = ring.data() + static_cast<size_t>(k) * kMaxElems;
      const uint64_t phase = static_cast<uint64_t>(c % kPhases);
      r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(row), phase, static_cast<uint32_t>(bytes / 2), 1, 0, s64);
      ep->AllReduceSumBf16(row, static_cast<int64_t>(bytes / 2), st);
      r4dx_tp_test_verify_bf16(reinterpret_cast<int64_t>(row), phase, static_cast<uint32_t>(bytes / 2), 1,
                               reinterpret_cast<int64_t>(rec.data()), s64);
    }
    const int slot = static_cast<int>(n % kSlots);
    R4DX_HIP_CHECK(hipMemcpyAsync(&rec_host[static_cast<size_t>(slot)], rec.data(), sizeof(R4dxTpStatus),
                                  hipMemcpyDeviceToHost, st));
    R4DX_HIP_CHECK(hipEventRecord(ev[slot], st));
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::string gpu_err;
  int64_t n = 0, enq = 0;
  try {
    for (; enq < std::min<int64_t>(kInflight, batches); ++enq) enqueue(enq);
    for (n = 0; n < batches; ++n) {
      const int slot = static_cast<int>(n % kSlots);
      const auto tw = std::chrono::steady_clock::now();
      for (;;) {
        const hipError_t e = hipEventQuery(ev[slot]);
        if (e == hipSuccess) break;
        if (e != hipErrorNotReady) throw core::HipError(e, "hipEventQuery", __FILE__, __LINE__);
        if (Since(tw) > 30.0) {
          std::fprintf(stderr, "FATAL: stream stuck > 30 s\n");
          std::_Exit(3);
        }
        std::this_thread::yield();
      }
      (void)hipGetLastError();
      const R4dxTpStatus v = rec_host[static_cast<size_t>(slot)];
      if (v.mismatches != 0) {
        char b[160];
        std::snprintf(b, sizeof b, "device verify: %u mismatching output elements (first: phase %llu element %u got 0x%04x)",
                      v.mismatches, static_cast<unsigned long long>(v.min_key >> 36),
                      static_cast<unsigned>((v.min_key >> 16) & 0xFFFFFu), static_cast<unsigned>(v.min_key & 0xFFFFu));
        gpu_err = b;
        break;
      }
      if (ep->Aborted()) {
        gpu_err = "an ABORT word is set";
        break;
      }
      if (enq < batches) enqueue(enq++);
    }
  } catch (const std::exception& e) {
    gpu_err = std::string("GPU side: ") + e.what();
  }
  stop.store(true);
  if (!tp::SyncWithWatchdog(st)) {
    const uint8_t* h = group->MailboxHost();
    std::fprintf(stderr, "FATAL: stream stuck > 30 s while draining; gpu_err '%s', peer_err '%s', peer_done %lld, "
                 "batches %lld, abort words %u %u\n",
                 gpu_err.c_str(), peer_err.c_str(), static_cast<long long>(peer_done.load()),
                 static_cast<long long>(n),
                 tp::HostU32Load(reinterpret_cast<const volatile uint32_t*>(h + tp::MailboxLayout::kAbortOff[0])),
                 tp::HostU32Load(reinterpret_cast<const volatile uint32_t*>(h + tp::MailboxLayout::kAbortOff[1])));
    for (int c = 0; c < 2; ++c)
      for (int r = 0; r < 2; ++r) {
        std::fprintf(stderr, "  flags ch%d rank%d:", c, r);
        for (int b = 0; b < 4; ++b)
          std::fprintf(stderr, " %08x",
                       tp::HostU32Load(reinterpret_cast<const volatile uint32_t*>(
                           h + group->Layout().ch[c].flags_off[r] + static_cast<size_t>(b) * kR4dxTpLineBytes)));
        std::fprintf(stderr, "\n");
      }
    std::_Exit(3);
  }
  cpu.join();
  const double secs = Since(t0);
  std::string health;
  try {
    ep->CheckHealthy();
  } catch (const std::exception& e) {
    health = e.what();
  }
  for (hipEvent_t e : ev) (void)hipEventDestroy(e);
  const bool ok = gpu_err.empty() && peer_err.empty() && health.empty() && n == batches &&
                  peer_done.load() == count;
  char line[512];
  std::snprintf(line, sizeof line, "%s: %lld/%lld calls verified by the CPU peer, %lld batches on the GPU, %.1f s%s%s%s%s%s%s",
                label, static_cast<long long>(peer_done.load()), static_cast<long long>(count),
                static_cast<long long>(n), secs, gpu_err.empty() ? "" : "; ", gpu_err.c_str(),
                peer_err.empty() ? "" : "; ", peer_err.c_str(), health.empty() ? "" : "; ", health.c_str());
  Expect(ok, line);
  if (ok) {
    const core::TpCommStats stats = ep->Stats();
    Expect(stats.ar_calls[0] + stats.ar_calls[1] == static_cast<uint64_t>(count) &&
               ep->CallCounts()[0] == stats.ar_calls[0] && ep->CallCounts()[1] == stats.ar_calls[1],
           std::string(label) + ": endpoint counted " + std::to_string(stats.ar_calls[0]) + " channel-0 and " +
               std::to_string(stats.ar_calls[1]) + " channel-1 all-reduces");
  }
  ep.reset();
  return ok;
}

// Case 3: fault injection, then the timeout path with the CPU peer silent.
void RunTimeoutCase() {
  constexpr int kTimeoutMs = 100;
  std::printf("timeout: CPU peer silent, all-reduce timeout %d ms\n", kTimeoutMs);
  auto group = tp::TpGroup::Create(tp::TpGroup::Mode::kReal, 2, {}, kTimeoutMs);
  group->AllocateMailbox();
  core::Stream stream;
  const hipStream_t st = stream.get();
  std::unique_ptr<tp::TpEndpoint> ep = group->CreateEndpoint(0, nullptr);
  const int64_t n = 5120;  // 10 KiB: channel 0, 4 blocks
  core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(n));
  r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(buf.data()), 7, static_cast<uint32_t>(n), 1, 0,
                        reinterpret_cast<int64_t>(st));
  std::vector<uint16_t> before(static_cast<size_t>(n)), after(static_cast<size_t>(n));
  R4DX_HIP_CHECK(hipMemcpyAsync(before.data(), buf.data(), static_cast<size_t>(n) * 2, hipMemcpyDeviceToHost, st));
  if (!tp::SyncWithWatchdog(st)) throw std::runtime_error("stream stuck");

  // Fault injection kind 0: throws before enqueuing anything.
  ep->ArmFaultInjection(1, tp::kFaultThrow);
  bool threw = false;
  try {
    ep->AllReduceSumBf16(buf.data(), n, st);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()) == "tp fault injection";
  }
  Expect(threw && ep->Stats().ar_calls[0] == 0, "fault injection kind 0 throws \"tp fault injection\" before enqueuing");

  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));
  R4DX_HIP_CHECK(hipEventRecord(e0, st));
  ep->AllReduceSumBf16(buf.data(), n, st);  // the CPU peer never answers
  R4DX_HIP_CHECK(hipEventRecord(e1, st));
  const auto h0 = std::chrono::steady_clock::now();
  if (!tp::SyncWithWatchdog(st)) throw std::runtime_error("stream stuck after the timed-out all-reduce");
  const double host_ms = Since(h0) * 1000.0;
  float ms = 0;
  R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  Expect(ms >= kTimeoutMs * 0.95 && ms <= kTimeoutMs * 1.10,
         "timed-out all-reduce took " + std::to_string(ms) + " ms on the device (" + std::to_string(host_ms) +
             " ms host wait); timeout " + std::to_string(kTimeoutMs) + " ms, limit +10%");
  const uint32_t ab0 = tp::HostU32Load(
      reinterpret_cast<volatile uint32_t*>(group->MailboxHost() + tp::MailboxLayout::kAbortOff[0]));
  Expect(ab0 == core::kAbortTimeout, "rank 0's ABORT word = " + std::to_string(ab0) + " (want kAbortTimeout = 1)");
  const R4dxTpStatus s1 = ep->ReadStatus();
  Expect(s1.claimed == 1 && s1.code == core::kAbortTimeout && s1.phase == kR4dxTpPhaseFlagWait && s1.word == 0 &&
             s1.sticky == 1 && s1.block < 4,
         "VRAM Status: claimed " + std::to_string(s1.claimed) + ", code " + std::to_string(s1.code) + ", channel " +
             std::to_string(s1.word) + ", block " + std::to_string(s1.block) + ", phase " + std::to_string(s1.phase) +
             ", sticky " + std::to_string(s1.sticky) + ", timed-out blocks " + std::to_string(s1.n_timeouts));

  // The next call is skipped entirely (sticky): no push, no wait, no reduce. Timed on the host,
  // launch to completion (hipEvent timestamps around a microsecond-long kernel can come out
  // slightly out of order on this runtime).
  const auto k0 = std::chrono::steady_clock::now();
  ep->AllReduceSumBf16(buf.data(), n, st);
  if (!tp::SyncWithWatchdog(st)) throw std::runtime_error("stream stuck");
  ms = static_cast<float>(Since(k0) * 1000.0);
  const R4dxTpStatus s2 = ep->ReadStatus();
  R4DX_HIP_CHECK(hipMemcpyAsync(after.data(), buf.data(), static_cast<size_t>(n) * 2, hipMemcpyDeviceToHost, st));
  if (!tp::SyncWithWatchdog(st)) throw std::runtime_error("stream stuck");
  Expect(ms < 5.0 && s2.n_skipped >= 4 && before == after,
         "next call skipped (sticky): " + std::to_string(ms) + " ms host launch-to-done, " + std::to_string(s2.n_skipped) +
             " blocks skipped, buffer " + (before == after ? "untouched" : "CHANGED"));

  std::string msg;
  try {
    ep->CheckHealthy();
  } catch (const core::TpAbortedError& e) {
    msg = e.what();
  }
  Expect(msg.find("timeout at channel 0") != std::string::npos && msg.find("phase flag-wait") != std::string::npos,
         "CheckHealthy throws TpAbortedError: \"" + msg + "\"");
  (void)hipEventDestroy(e0);
  (void)hipEventDestroy(e1);
  ep.reset();
}

// Case 4: r4dx_tp_add_bf16 (EmulatedComm's add, docs/tp.md 6.5) -- the same add_bf16x8 as the
// all-reduce: inout = rank 0's pattern, peer = rank 1's, the result must be the exact sum.
void RunAddCase() {
  std::printf("r4dx_tp_add_bf16\n");
  core::Stream stream;
  const int64_t s64 = reinterpret_cast<int64_t>(stream.get());
  const uint32_t n = static_cast<uint32_t>(kMaxElems);
  core::DeviceBuffer<uint16_t> a(n), b(n);
  core::DeviceBuffer<uint8_t> rec(sizeof(R4dxTpStatus));
  R4DX_HIP_CHECK(hipMemsetAsync(rec.data(), 0, rec.bytes(), stream.get()));
  R4DX_HIP_CHECK(hipMemsetAsync(rec.data() + offsetof(R4dxTpStatus, min_key), 0xFF, 8, stream.get()));
  uint32_t mismatches = 0;
  for (uint64_t call = 0; call < 8; ++call) {
    r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(a.data()), call, n, 1, 0, s64);
    r4dx_tp_test_gen_bf16(reinterpret_cast<int64_t>(b.data()), call, n, 1, 1, s64);
    r4dx_tp_add_bf16(reinterpret_cast<int64_t>(a.data()), reinterpret_cast<int64_t>(b.data()), n, s64);
    r4dx_tp_test_verify_bf16(reinterpret_cast<int64_t>(a.data()), call, n, 1, reinterpret_cast<int64_t>(rec.data()), s64);
  }
  R4dxTpStatus r;
  R4DX_HIP_CHECK(hipMemcpyAsync(&r, rec.data(), sizeof r, hipMemcpyDeviceToHost, stream.get()));
  if (!tp::SyncWithWatchdog(stream.get())) throw std::runtime_error("stream stuck in the add case");
  mismatches = r.mismatches;
  Expect(mismatches == 0, "8 x 327680-element adds bit-exact (" + std::to_string(mismatches) + " mismatches)");
}

}  // namespace

int main() {
  try {
    R4DX_HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t p;
    R4DX_HIP_CHECK(hipGetDeviceProperties(&p, 0));
    const double khz = tp::CheckWallClockRate();
    std::printf("test_tp_allreduce_cpu_peer on %s (pci %02x), WallClockRate check %.0f kHz\n", p.name, p.pciBusID, khz);
    // R4DX_TP_TEST_CASES (debugging aid): a subset of "1234"; default all four.
    std::string cases = "1234";
#ifdef _MSC_VER
    char* env = nullptr;
    size_t env_len = 0;
    if (_dupenv_s(&env, &env_len, "R4DX_TP_TEST_CASES") == 0 && env != nullptr && *env != '\0') cases = env;
    std::free(env);
#endif
    BuildRows();
    if (cases.find('1') != std::string::npos)
      RunBitExact("bit-exact", tp::kInitialSeqBase, {10240, 81920, 174080, 655360}, 100000);
    if (cases.find('2') != std::string::npos)
      RunBitExact("seq wrap", 0xFFFFFF00u, {10240, 655360, 81920, 184320}, 1200);
    if (cases.find('3') != std::string::npos) RunTimeoutCase();
    if (cases.find('4') != std::string::npos) RunAddCase();
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
