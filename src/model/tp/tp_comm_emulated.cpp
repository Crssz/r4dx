// EmulatedComm -- the single-device TpComm (docs/tp.md 6.5, `--tp-mode emulate`): both ranks of a
// TP=2 group run on ONE device, each on its own rank thread with its own streams, and every
// all-reduce is a host-synchronized exact add. Per call c of rank r (c counts this endpoint's
// all-reduces since creation / the last ResetCounters):
//
//   1. hipMemcpyAsync(xchg[r][c & 1], buf, bytes, D2D, stream); wait for `stream` (a marker event,
//      never hipStreamQuery -- docs/tp.md Appendix B N33);
//   2. HostExchange barrier, bounded by the all-reduce timeout (Appendix B N45);
//   3. r4dx_tp_add_bf16(buf, xchg[peer][c & 1], n, stream) -- the SAME add_bf16x8 the mailbox
//      kernel reduces with, operand order (mine, peer). fp32 addition is commutative, so both
//      ranks write identical bits, exactly as the real kernel's two ranks do.
//
// No second barrier: rank r rewrites xchg[r][p] only at call c + 2, after passing barrier c + 1,
// which needs the peer to have synchronized its stream at c + 1 -- i.e. after the peer's call-c
// add (the only reader of xchg[r][p]) finished. That holds only if the peer's call-c add is on a
// stream the peer drained before barrier c + 1: when call c + 1 comes on a DIFFERENT stream than
// call c's add, step 1 drains the add's stream too (Appendix B N53), so the invariant does not
// depend on every all-reduce of a rank sharing one stream. Everything else (HostAllGather, CheckLockstep,
// Abort, the per-channel accounting, fault injection, SelfTest) behaves as in HostMailboxComm, so
// an emulated run executes the same Model code, kernels, tunings and fp32 add as a real one.
//
// A barrier that times out plays the part of the device kernel's spin timeout (docs/tp.md 6.3.5):
// the rank stores kAbortTimeout in its (host) abort word, records the call in its Status, poisons
// the exchange and throws core::TpAbortedError -- so fault injection kind 1 (a 700 ms stall on the
// peer) reaches the caller the way it would on two GPUs.
//
// Everything runs on the endpoint's rank thread, except Abort()/Aborted(), which touch only host
// words and the HostExchange (TpModel's shutdown may call them from the facade thread, 2.6).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/tp_kernels.h"
#include "tp/tp_group.h"

namespace r4dx::model::tp {

namespace {

constexpr int kHidden = 5120;  // SelfTest rows are hidden-sized (docs/tp.md 6.3.9)
constexpr int kSelfTestIters = 16;
constexpr int kSelfTestRows[2][3] = {{1, 8, 17}, {18, 64, 0}};
constexpr size_t kMaxTrackedStreams = 8;

const char* AbortCodeName(uint32_t c) {
  switch (c) {
    case core::kAbortNone: return "none";
    case core::kAbortTimeout: return "timeout";
    case core::kAbortProtocol: return "protocol-violation";
    case core::kAbortHost: return "host-error";
    case core::kAbortShutdown: return "shutdown";
    default: return "unknown";
  }
}

}  // namespace

class EmulatedComm final : public TpEndpoint {
 public:
  EmulatedComm(TpGroup* group, int rank, std::atomic<uint64_t>* heartbeat)
      : g_(group),
        rank_(rank),
        world_(group->World()),
        hb_(heartbeat),
        selftest_(kMaxAllReduceBytes / 2),
        selftest_host_(kMaxAllReduceBytes / 2) {
    if (world_ != 2) throw std::invalid_argument("tp::EmulatedComm: world must be 2");
    R4DX_HIP_CHECK(hipGetDevice(&device_));
    int rate = 0;
    R4DX_HIP_CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, device_));
    khz_ = static_cast<double>(rate);
    R4DX_HIP_CHECK(hipEventCreateWithFlags(&marker_, hipEventDisableTiming));
    for (int p = 0; p < 2; ++p) {
      xchg_[p] = core::DeviceBuffer<uint16_t>(kMaxAllReduceBytes / 2);
      R4DX_HIP_CHECK(hipMemsetAsync(xchg_[p].data(), 0, xchg_[p].bytes(), stream_.get()));
    }
    SyncOwn("EmulatedComm construction");
    std::memset(&status_, 0, sizeof status_);
    status_.abort_seq_min = 0xFFFFFFFFu;
    SetAllReduceTimeoutMs(g_->ArTimeoutMs());
    base_ = g_->SeqBase();
    for (int p = 0; p < 2; ++p) g_->emu_xchg_[rank_][p].store(xchg_[p].data(), std::memory_order_release);
    if (hb_ != nullptr) g_->Exchange().SetHeartbeat(rank_, hb_);
    g_->Register(rank_, this);  // last: nothing below can throw
  }

  ~EmulatedComm() override {
    g_->Unregister(rank_, this);
    try {
      g_->Exchange().SetHeartbeat(rank_, nullptr);
    } catch (...) {
    }
    for (auto& x : g_->emu_xchg_[rank_]) x.store(nullptr, std::memory_order_release);
    // Only the endpoint's own stream: the streams all-reduces were enqueued on belong to the Model,
    // which is destroyed first (docs/tp.md 2.6). TpModel drains every rank's streams in a separate
    // step before tearing any rank down, so no peer add can still be reading xchg_ here.
    bool idle = false;
    try {
      idle = SyncWithWatchdogEvent(stream_.get(), marker_);
      if (!idle) g_->MarkStuck();
    } catch (...) {
      idle = false;
    }
    if (!idle || g_->Stuck()) {
      // The peer's add, or this rank's own queued work, may still read these: leak rather than free
      // (docs/tp.md 2.6, the same rule HostMailboxComm follows).
      std::fprintf(stderr, "[r4dx-tp] rank %d: emulated endpoint buffers and stream not freed (%s)\n", rank_,
                   idle ? "the group is marked stuck" : "a stream is stuck or in error");
      (void)new core::Stream(std::move(stream_));
      for (auto& x : xchg_) (void)new core::DeviceBuffer<uint16_t>(std::move(x));
      (void)new core::DeviceBuffer<uint16_t>(std::move(selftest_));
      (void)new core::PinnedBuffer<uint16_t>(std::move(selftest_host_));
      return;  // the marker event leaks with them
    }
    (void)hipEventDestroy(marker_);
  }

  int Rank() const override { return rank_; }
  int World() const override { return world_; }
  int Device() const override { return device_; }
  double WallClockKhz() const override { return khz_; }
  uint32_t SeqBase() const override { return base_; }

  // docs/tp.md 6.5 (see the file comment). Blocks the host: this rank's device work up to here is
  // drained at every call, which is what serializes the two ranks' kernels on the shared device.
  void AllReduceSumBf16(uint16_t* buf, int64_t n, hipStream_t stream) override {
    const int64_t bytes = n * 2;
    if (buf == nullptr || n <= 0 || bytes % 16 != 0 || static_cast<size_t>(bytes) > kMaxAllReduceBytes) {
      throw std::invalid_argument("tp::EmulatedComm::AllReduceSumBf16: n = " + std::to_string(n) +
                                  " bf16 elements (need a non-null buffer and 0 < 2n <= " +
                                  std::to_string(kMaxAllReduceBytes) + ", 2n % 16 == 0)");
    }
    const int ch = MailboxLayout::ChannelFor(static_cast<size_t>(bytes));
    if (fault_at_ > 0 && !fault_fired_ && ++fault_calls_ == fault_at_) {
      fault_fired_ = true;
      std::fprintf(stderr, "[r4dx-tp] rank %d: fault injection (kind %d) at all-reduce #%lld\n", rank_, fault_kind_,
                   static_cast<long long>(fault_at_));
      if (fault_kind_ == kFaultThrow) throw std::runtime_error("tp fault injection");
      std::this_thread::sleep_for(std::chrono::milliseconds(700));
    }
    if (stream != last_stream_) TrackStream(stream);
    if (hb_ != nullptr) hb_->fetch_add(1, std::memory_order_relaxed);

    const uint64_t c = call_index_++;
    const int parity = static_cast<int>(c & 1u);
    // 1. publish this rank's partial and drain the stream (every kernel that produced it is done) --
    //    and, on a stream switch, the stream of the previous call's add, which reads the peer's
    //    xchg[peer][(c - 1) & 1] that the peer may rewrite once it passes this call's barrier.
    if (add_pending_ && add_stream_ != stream && !SyncWithWatchdogEvent(add_stream_, marker_)) {
      g_->MarkStuck();
      throw core::TpTimeoutError("tp: rank " + std::to_string(rank_) +
                                 ": the previous all-reduce's stream stuck > 30 s before emulated all-reduce");
    }
    add_pending_ = false;
    R4DX_HIP_CHECK(hipMemcpyAsync(xchg_[parity].data(), buf, static_cast<size_t>(bytes), hipMemcpyDeviceToDevice,
                                  stream));
    if (!SyncWithWatchdogEvent(stream, marker_)) {
      g_->MarkStuck();
      throw core::TpTimeoutError("tp: rank " + std::to_string(rank_) + ": stream stuck > 30 s before emulated all-reduce");
    }
    // 2. both partials are published.
    try {
      g_->Exchange().BarrierFor(rank_, std::chrono::milliseconds(timeout_ms_));
    } catch (const core::TpTimeoutError&) {
      // The device kernel's spin timeout, emulated: claim the Status, set this rank's abort word
      // (the exchange is already poisoned by the timed-out wait), report like CheckHealthy would.
      if (status_.claimed == 0) {
        status_.claimed = 1;
        status_.code = core::kAbortTimeout;
        status_.block = 0;
        status_.seq = static_cast<uint32_t>(c);
        status_.phase = kR4dxTpPhaseFlagWait;
        status_.word = static_cast<uint32_t>(ch);
        ++status_.n_timeouts;
      }
      status_.sticky = 1;
      StoreAbort(core::kAbortTimeout);
      char msg[256];
      std::snprintf(msg, sizeof msg,
                    "tp: rank %d timeout at channel %d block 0 seq %llu phase flag-wait (emulated all-reduce "
                    "waited %d ms for rank %d); peer abort word %u (%s)",
                    rank_, ch, static_cast<unsigned long long>(c), timeout_ms_, 1 - rank_, PeerAbortWord(),
                    AbortCodeName(PeerAbortWord()));
      throw core::TpAbortedError(msg);
    }
    // 3. buf = bf16(f32(mine) + f32(peer)): the mailbox kernel's add, element for element.
    uint16_t* peer = g_->emu_xchg_[1 - rank_][parity].load(std::memory_order_acquire);
    if (peer == nullptr) throw std::logic_error("tp::EmulatedComm: the peer rank has no endpoint");
    r4dx_tp_add_bf16(reinterpret_cast<int64_t>(buf), reinterpret_cast<int64_t>(peer), n,
                     reinterpret_cast<int64_t>(stream));
    add_pending_ = true;
    add_stream_ = stream;
    ++stats_.ar_calls[ch];
    stats_.ar_bytes[ch] += static_cast<uint64_t>(bytes);
    ++calls_[static_cast<size_t>(ch)];
  }

  void HostAllGather(const void* mine, size_t bytes, void* out) override {
    const auto t0 = std::chrono::steady_clock::now();
    g_->Exchange().AllGather(rank_, mine, bytes, out);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ++stats_.host_exchanges;
    stats_.host_exchange_wait_us_max = std::max(stats_.host_exchange_wait_us_max, us);
  }

  void CheckLockstep(const uint64_t fingerprint[4]) override {
    std::vector<uint64_t> all(static_cast<size_t>(world_) * 4);
    HostAllGather(fingerprint, 4 * sizeof(uint64_t), all.data());
    for (int q = 0; q < world_; ++q) {
      if (std::memcmp(all.data() + static_cast<size_t>(q) * 4, fingerprint, 4 * sizeof(uint64_t)) != 0) {
        char buf[512];
        const uint64_t* o = all.data() + static_cast<size_t>(q) * 4;
        std::snprintf(buf, sizeof buf,
                      "tp: lockstep divergence: rank %d fingerprint {%llx, %llx, %llx, %llx}, rank %d "
                      "{%llx, %llx, %llx, %llx}",
                      rank_, (unsigned long long)fingerprint[0], (unsigned long long)fingerprint[1],
                      (unsigned long long)fingerprint[2], (unsigned long long)fingerprint[3], q,
                      (unsigned long long)o[0], (unsigned long long)o[1], (unsigned long long)o[2],
                      (unsigned long long)o[3]);
        throw core::TpDivergenceError(buf);
      }
    }
  }

  // docs/tp.md 6.3.5, emulated: the two abort words and the exchange flag; if any is set, drain
  // this rank's streams (watchdog) and throw TpAbortedError naming the recorded failure.
  void CheckHealthy() override {
    const uint32_t a0 = g_->emu_abort_[0].load(std::memory_order_acquire);
    const uint32_t a1 = g_->emu_abort_[1].load(std::memory_order_acquire);
    const bool ex = g_->Exchange().Aborted();
    if (a0 == 0 && a1 == 0 && !ex) return;
    SyncStreams();
    const uint32_t mine = rank_ == 0 ? a0 : a1, peer = rank_ == 0 ? a1 : a0;
    char buf[512];
    if (status_.claimed != 0) {
      std::snprintf(buf, sizeof buf,
                    "tp: rank %d %s at channel %u block %u seq %u phase flag-wait (emulated); peer abort word "
                    "%u (%s)",
                    rank_, AbortCodeName(status_.code), status_.word, status_.block, status_.seq, peer,
                    AbortCodeName(peer));
    } else {
      std::snprintf(buf, sizeof buf,
                    "tp: rank %d aborted: own abort word %u (%s), peer abort word %u (%s), host exchange %s "
                    "(emulated)",
                    rank_, mine, AbortCodeName(mine), peer, AbortCodeName(peer), ex ? "aborted" : "ok");
    }
    throw core::TpAbortedError(buf);
  }

  void Abort(uint32_t code, const std::string& why) noexcept override {
    StoreAbort(code == 0 ? static_cast<uint32_t>(core::kAbortHost) : code);
    g_->Exchange().Abort(why);
    aborts_.fetch_add(1, std::memory_order_relaxed);
  }
  bool Aborted() const noexcept override {
    return g_->emu_abort_[0].load(std::memory_order_acquire) != 0 ||
           g_->emu_abort_[1].load(std::memory_order_acquire) != 0 || g_->Exchange().Aborted();
  }

  size_t MaxAllReduceBytes() const override { return kMaxAllReduceBytes; }

  // docs/tp.md 6.3.9, through the emulated all-reduce: per channel, rows {1, 8, 17} / {18, 64}, 16
  // iterations k: rank r fills bf16(r == 0 ? k % 64 : 64 + k % 64), all-reduces, reads back and
  // checks every element == bf16(64 + 2 * (k % 64)), then CheckHealthy(). Collective.
  void SelfTest() override {
    SyncStreams();
    for (int c = 0; c < 2; ++c) {
      for (int rows : kSelfTestRows[c]) {
        if (rows == 0) continue;
        const int64_t n = static_cast<int64_t>(rows) * kHidden;
        for (int k = 0; k < kSelfTestIters; ++k) {
          const int mine = rank_ == 0 ? k % 64 : 64 + k % 64;
          const uint16_t want = core::FloatToBf16(static_cast<float>(64 + 2 * (k % 64)));
          R4DX_HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(selftest_.data()),
                                           core::FloatToBf16(static_cast<float>(mine)), static_cast<size_t>(n),
                                           stream_.get()));
          AllReduceSumBf16(selftest_.data(), n, stream_.get());
          R4DX_HIP_CHECK(hipMemcpyAsync(selftest_host_.data(), selftest_.data(), static_cast<size_t>(n) * 2,
                                        hipMemcpyDeviceToHost, stream_.get()));
          SyncOwn("SelfTest");
          CheckHealthy();
          for (int64_t i = 0; i < n; ++i) {
            if (selftest_host_[static_cast<size_t>(i)] != want) {
              char buf[256];
              std::snprintf(buf, sizeof buf,
                            "tp: self-test FAILED on rank %d (emulated): channel %d, %d rows, iteration %d, "
                            "element %lld = 0x%04x, want 0x%04x",
                            rank_, c, rows, k, static_cast<long long>(i), selftest_host_[static_cast<size_t>(i)],
                            want);
              Abort(core::kAbortHost, buf);
              throw core::TpError(buf);
            }
          }
        }
      }
    }
  }

  core::TpCommStats Stats() const override {
    core::TpCommStats s = stats_;
    s.aborts = aborts_.load(std::memory_order_relaxed);
    return s;
  }

  void SetAllReduceTimeoutMs(int ms) override { timeout_ms_ = ClampArTimeoutMs(ms); }

  // docs/tp.md 2.5 step 6: the call index c (both parities restart from xchg[r][0]), the per-channel
  // call counters, the fault-injection counter and the Stats() deltas restart at 0, and both
  // exchange buffers are zero-filled; Stats().aborts stays cumulative. The recorded Status is
  // re-initialised in Rebase (step 5), like the VRAM Status of the real transport.
  void ResetCounters() override {
    call_index_ = 0;
    add_pending_ = false;  // recovery step 1 drained every stream
    calls_ = {0, 0};
    fault_calls_ = 0;
    stats_ = core::TpCommStats{};
    for (auto& x : xchg_) R4DX_HIP_CHECK(hipMemsetAsync(x.data(), 0, x.bytes(), stream_.get()));
    SyncOwn("ResetCounters");
  }
  std::array<uint64_t, 2> CallCounts() const override { return calls_; }

  // ---- TpEndpoint -------------------------------------------------------------------------------
  void SyncStreams() override {
    for (hipStream_t s : AllStreams()) {
      if (!SyncWithWatchdog(s)) {
        g_->MarkStuck();
        throw core::TpTimeoutError("tp: rank " + std::to_string(rank_) + ": a stream is still busy after 30 s");
      }
    }
  }

  // No device seq counters in emulation: the base only moves (by recovery's + 64).
  uint32_t SeqAdvance() override {
    SyncStreams();
    return 0;
  }

  void Rebase(uint32_t new_base) override {
    SyncStreams();
    std::memset(&status_, 0, sizeof status_);
    status_.abort_seq_min = 0xFFFFFFFFu;
    base_ = new_base;
  }

  R4dxTpStatus ReadStatus() override {
    SyncStreams();
    return status_;
  }

  void ArmFaultInjection(int64_t at_allreduce, int kind) override {
    if (kind != kFaultThrow && kind != kFaultStall) {
      throw std::invalid_argument("tp: fault kind must be 0 (throw) or 1 (stall), got " + std::to_string(kind));
    }
    fault_at_ = at_allreduce;
    fault_kind_ = kind;
    fault_calls_ = 0;
    fault_fired_ = false;
    if (at_allreduce > 0) {
      std::fprintf(stderr, "[r4dx-tp] rank %d: FAULT INJECTION ARMED: kind %d (%s) at all-reduce #%lld\n", rank_, kind,
                   kind == kFaultThrow ? "throw" : "stall 700 ms", static_cast<long long>(at_allreduce));
    }
  }

 private:
  void StoreAbort(uint32_t code) noexcept {
    uint32_t expected = 0;
    (void)g_->emu_abort_[rank_].compare_exchange_strong(expected, code, std::memory_order_acq_rel);
  }
  uint32_t PeerAbortWord() const noexcept { return g_->emu_abort_[1 - rank_].load(std::memory_order_acquire); }

  std::vector<hipStream_t> AllStreams() const {
    std::vector<hipStream_t> all(streams_);
    all.push_back(stream_.get());
    return all;
  }

  void SyncOwn(const char* where) {
    if (!SyncWithWatchdogEvent(stream_.get(), marker_)) {
      g_->MarkStuck();
      throw core::TpTimeoutError(std::string("tp: rank ") + std::to_string(rank_) + ": stream stuck > 30 s in " +
                                 where);
    }
  }

  // Never evicts (docs/tp.md N43): SyncStreams must wait on every stream an all-reduce was queued on.
  void TrackStream(hipStream_t s) {
    if (std::find(streams_.begin(), streams_.end(), s) == streams_.end()) {
      if (streams_.size() >= kMaxTrackedStreams) {
        throw std::logic_error("tp: rank " + std::to_string(rank_) + ": all-reduces on more than " +
                               std::to_string(kMaxTrackedStreams) +
                               " distinct streams; recovery could not sync them all (docs/tp.md N43)");
      }
      streams_.push_back(s);
    }
    last_stream_ = s;
  }

  TpGroup* g_;
  int rank_, world_;
  int device_ = -1;
  std::atomic<uint64_t>* hb_;
  double khz_ = 0;
  int timeout_ms_ = kArTimeoutDefaultMs;
  uint32_t base_ = 0;
  core::Stream stream_;                   // the endpoint's own (SelfTest, ResetCounters)
  hipEvent_t marker_ = nullptr;           // SyncWithWatchdogEvent's marker, every stream
  core::DeviceBuffer<uint16_t> xchg_[2];  // xchg[rank][parity], 640 KiB each
  core::DeviceBuffer<uint16_t> selftest_;
  core::PinnedBuffer<uint16_t> selftest_host_;  // pinned: the D2H is truly async (N43)
  R4dxTpStatus status_{};                 // host-side: emulation has no device Status
  std::vector<hipStream_t> streams_;      // streams all-reduces were enqueued on
  hipStream_t last_stream_ = reinterpret_cast<hipStream_t>(~uintptr_t{0});
  uint64_t call_index_ = 0;               // c (docs/tp.md 6.5)
  bool add_pending_ = false;              // the last call's add was enqueued on add_stream_ ...
  hipStream_t add_stream_ = nullptr;      // ... and not yet known to be finished
  core::TpCommStats stats_;
  std::atomic<uint64_t> aborts_{0};
  std::array<uint64_t, 2> calls_ = {0, 0};
  int64_t fault_at_ = -1;
  int fault_kind_ = kFaultThrow;
  int64_t fault_calls_ = 0;
  bool fault_fired_ = false;
};

std::unique_ptr<TpEndpoint> MakeEmulatedComm(TpGroup* group, int rank, std::atomic<uint64_t>* heartbeat) {
  return std::make_unique<EmulatedComm>(group, rank, heartbeat);
}

}  // namespace r4dx::model::tp
