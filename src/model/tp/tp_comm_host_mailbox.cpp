// HostMailboxComm -- the real two-GPU TpComm (docs/tp.md 6.3): tools/tp_bench's flag(drain 3,
// acq 0) all-reduce through the pinned host mailbox of a TpGroup, nb = 4, nt = 256, two channels
// with their own flags, slots and VRAM seq counters (6.3.2), a 500 ms kernel spin timeout with a
// sticky local abort (6.3.5), and the load-time / post-recovery SelfTest (6.3.9).
//
// Everything here runs on the endpoint's rank thread, except Abort()/Aborted(), which only touch
// host words and the HostExchange and may be called from any thread (TpModel's shutdown, 2.6).
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
const char* PhaseName(uint32_t p) { return p == kR4dxTpPhaseFlagWait ? "flag-wait" : "none"; }

}  // namespace

class HostMailboxComm final : public TpEndpoint {
 public:
  HostMailboxComm(TpGroup* group, int rank, std::atomic<uint64_t>* heartbeat)
      : g_(group),
        rank_(rank),
        world_(group->World()),
        hb_(heartbeat),
        L_(group->Layout()),
        seq_(2 * static_cast<size_t>(kR4dxTpNbMax)),
        status_(sizeof(R4dxTpStatus)),
        selftest_(kMaxAllReduceBytes / 2),
        selftest_host_(kMaxAllReduceBytes / 2),
        seq_host_(2 * static_cast<size_t>(kR4dxTpNbMax)),
        status_host_(1) {
    R4DX_HIP_CHECK(hipGetDevice(&device_));
    int rate = 0;
    R4DX_HIP_CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, device_));
    if (rate <= 0) throw core::TpError("tp: hipDeviceAttributeWallClockRate is 0 on HIP device " + std::to_string(device_));
    khz_ = static_cast<double>(rate);
    host_ = g_->MailboxHost();
    void* dp = nullptr;
    R4DX_HIP_CHECK(hipHostGetDevicePointer(&dp, host_, 0));  // this device's view (measured == host_)
    dev_ = static_cast<char*>(dp);
    abort_host_[0] = reinterpret_cast<volatile uint32_t*>(host_ + MailboxLayout::kAbortOff[0]);
    abort_host_[1] = reinterpret_cast<volatile uint32_t*>(host_ + MailboxLayout::kAbortOff[1]);
    const int peer = 1 - rank_;
    for (int c = 0; c < 2; ++c) {
      const MailboxLayout::Channel& ch = L_.ch[c];
      R4dxTpArArgs& a = args_[c];
      std::memset(&a, 0, sizeof a);
      a.peer_mbox = reinterpret_cast<int64_t>(dev_ + ch.mbox_off[peer]);
      a.my_mbox = reinterpret_cast<int64_t>(dev_ + ch.mbox_off[rank_]);
      a.peer_flags = reinterpret_cast<int64_t>(dev_ + ch.flags_off[peer]);
      a.my_flags = reinterpret_cast<int64_t>(dev_ + ch.flags_off[rank_]);
      a.my_abort = reinterpret_cast<int64_t>(dev_ + MailboxLayout::kAbortOff[rank_]);
      a.peer_abort = reinterpret_cast<int64_t>(dev_ + MailboxLayout::kAbortOff[peer]);
      a.seq = reinterpret_cast<int64_t>(seq_.data() + static_cast<size_t>(c) * kR4dxTpNbMax);
      a.status = reinterpret_cast<int64_t>(status_.data());
      a.blk_stride = ch.blk_stride;
      a.slot_stride = ch.slot_stride;
      a.drain = 3;  // production: flag(d3, a0), relaxed polls + one acquire fence
      a.acq = 0;
      a.spin_acq = 0;
      a.channel = c;
    }
    SetAllReduceTimeoutMs(g_->ArTimeoutMs());
    base_ = g_->SeqBase();
    InitDeviceState(base_);
    if (hb_ != nullptr) g_->Exchange().SetHeartbeat(rank_, hb_);
    g_->Register(rank_, this);  // last: nothing below can throw
  }

  ~HostMailboxComm() override {
    g_->Unregister(rank_, this);
    try {
      g_->Exchange().SetHeartbeat(rank_, nullptr);
    } catch (...) {
    }
    bool idle = false;
    try {
      idle = SyncWithWatchdog(stream_.get());
      if (!idle) g_->MarkStuck();
    } catch (...) {
    }
    if (!idle || g_->Stuck()) {
      // A kernel may still touch these (a stuck self-test all-reduce and the D2H queued behind it):
      // leak them rather than free them, like the mailbox (2.6; tp_bench leaked everything on
      // g_stuck). Moving each into a never-deleted heap object keeps its destructor from running.
      std::fprintf(stderr,
                   "[r4dx-tp] rank %d: endpoint VRAM, pinned buffers and stream not freed (%s)\n", rank_,
                   idle ? "the group is marked stuck" : "its stream is stuck or in error");
      (void)new core::Stream(std::move(stream_));
      (void)new core::DeviceBuffer<uint32_t>(std::move(seq_));
      (void)new core::DeviceBuffer<uint8_t>(std::move(status_));
      (void)new core::DeviceBuffer<uint16_t>(std::move(selftest_));
      (void)new core::PinnedBuffer<uint16_t>(std::move(selftest_host_));
      (void)new core::PinnedBuffer<uint32_t>(std::move(seq_host_));
      (void)new core::PinnedBuffer<R4dxTpStatus>(std::move(status_host_));
    }
  }

  int Rank() const override { return rank_; }
  int World() const override { return world_; }
  int Device() const override { return device_; }
  double WallClockKhz() const override { return khz_; }
  uint32_t SeqBase() const override { return base_; }

  // docs/tp.md 6.3.4. Never blocks the host.
  void AllReduceSumBf16(uint16_t* buf, int64_t n, hipStream_t stream) override {
    const int64_t bytes = n * 2;
    if (buf == nullptr || n <= 0 || bytes % 16 != 0 || static_cast<size_t>(bytes) > kMaxAllReduceBytes) {
      throw std::invalid_argument("tp::HostMailboxComm::AllReduceSumBf16: n = " + std::to_string(n) +
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
    const CallChunks g = ChunkCall(static_cast<size_t>(bytes), L_.ch[ch].nb);
    R4dxTpArArgs a = args_[ch];
    a.buf = reinterpret_cast<int64_t>(buf);
    a.n16 = g.n16;
    a.nb = g.nb;
    a.w = g.w;
    if (stream != last_stream_) TrackStream(stream);  // before the launch: it throws when the set is full
    r4dx_tp_ar_flag_bf16(&a, kArThreads, reinterpret_cast<int64_t>(stream));
    ++stats_.ar_calls[ch];
    stats_.ar_bytes[ch] += static_cast<uint64_t>(bytes);
    ++calls_[static_cast<size_t>(ch)];
    if (hb_ != nullptr) hb_->fetch_add(1, std::memory_order_relaxed);
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

  // docs/tp.md 6.3.5: acquire-load both ABORT words and the exchange flag; if any is set, sync this
  // rank's streams (watchdog), read the VRAM Status and throw TpAbortedError.
  void CheckHealthy() override {
    const uint32_t a0 = HostU32Load(abort_host_[0]), a1 = HostU32Load(abort_host_[1]);
    const bool ex = g_->Exchange().Aborted();
    if (a0 == 0 && a1 == 0 && !ex) return;
    SyncStreams();
    const R4dxTpStatus s = ReadStatusNoSync();
    const uint32_t mine = rank_ == 0 ? a0 : a1, peer = rank_ == 0 ? a1 : a0;
    char buf[512];
    if (s.claimed != 0) {
      std::snprintf(buf, sizeof buf, "tp: rank %d %s at channel %u block %u seq %u phase %s; peer abort word %u (%s)",
                    rank_, AbortCodeName(s.code), s.word, s.block, s.seq, PhaseName(s.phase), peer,
                    AbortCodeName(peer));
    } else {
      std::snprintf(buf, sizeof buf,
                    "tp: rank %d aborted: own abort word %u (%s), peer abort word %u (%s), host exchange %s; "
                    "%u blocks bailed on an abort word, %u skipped (sticky)",
                    rank_, mine, AbortCodeName(mine), peer, AbortCodeName(peer), ex ? "aborted" : "ok",
                    s.n_abort_exits, s.n_skipped);
    }
    throw core::TpAbortedError(buf);
  }

  void Abort(uint32_t code, const std::string& why) noexcept override {
    volatile uint32_t* mine = abort_host_[rank_];
    if (HostU32Load(mine) == 0) HostU32Store(mine, code == 0 ? static_cast<uint32_t>(core::kAbortHost) : code);
    g_->Exchange().Abort(why);
    aborts_.fetch_add(1, std::memory_order_relaxed);
  }
  bool Aborted() const noexcept override {
    return HostU32Load(abort_host_[0]) != 0 || HostU32Load(abort_host_[1]) != 0 || g_->Exchange().Aborted();
  }

  size_t MaxAllReduceBytes() const override { return kMaxAllReduceBytes; }

  // docs/tp.md 6.3.9: per channel, rows {1, 8, 17} (ch0) / {18, 64} (ch1), 16 iterations k: rank r
  // fills its buffer with bf16(r == 0 ? k % 64 : 64 + k % 64), all-reduces, reads back and checks
  // every element == bf16(64 + 2 * (k % 64)) exactly, then CheckHealthy(). Collective.
  void SelfTest() override {
    SyncStreams();  // every earlier all-reduce of this rank is done (the seq counters are stream-ordered)
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
          // Pinned target: a D2H into pageable memory is synchronous in HIP and would block this
          // thread in the driver behind the spinning kernel, past the watchdog below.
          R4DX_HIP_CHECK(hipMemcpyAsync(selftest_host_.data(), selftest_.data(), static_cast<size_t>(n) * 2,
                                        hipMemcpyDeviceToHost, stream_.get()));
          if (!SyncWithWatchdog(stream_.get())) {
            g_->MarkStuck();
            throw core::TpTimeoutError("tp: rank " + std::to_string(rank_) + ": stream stuck > 30 s in SelfTest");
          }
          CheckHealthy();
          for (int64_t i = 0; i < n; ++i) {
            if (selftest_host_[static_cast<size_t>(i)] != want) {
              char buf[256];
              std::snprintf(buf, sizeof buf,
                            "tp: self-test FAILED on rank %d: channel %d, %d rows, iteration %d, element %lld = "
                            "0x%04x, want 0x%04x",
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

  void SetAllReduceTimeoutMs(int ms) override {
    timeout_ms_ = ClampArTimeoutMs(ms);
    const uint64_t ticks = static_cast<uint64_t>(timeout_ms_) * static_cast<uint64_t>(khz_);
    args_[0].timeout_ticks = ticks;
    args_[1].timeout_ticks = ticks;
  }

  // 2.5 step 6: the call counters, the fault-injection counter and the Stats() deltas (per-channel
  // calls/bytes, host exchanges and their max wait) restart at 0; Stats().aborts stays cumulative
  // over the endpoint's life (N43).
  void ResetCounters() override {
    calls_ = {0, 0};
    fault_calls_ = 0;
    stats_ = core::TpCommStats{};
  }
  std::array<uint64_t, 2> CallCounts() const override { return calls_; }

  // ---- TpEndpoint -------------------------------------------------------------------------------
  void SyncStreams() override {
    std::vector<hipStream_t> all(streams_);
    all.push_back(stream_.get());
    for (hipStream_t s : all) {
      if (!SyncWithWatchdog(s)) {
        g_->MarkStuck();
        throw core::TpTimeoutError("tp: rank " + std::to_string(rank_) + ": a stream is still busy after 30 s");
      }
    }
  }

  uint32_t SeqAdvance() override {
    SyncStreams();
    R4DX_HIP_CHECK(hipMemcpyAsync(seq_host_.data(), seq_.data(), seq_.bytes(), hipMemcpyDeviceToHost, stream_.get()));
    SyncOwn("SeqAdvance");
    uint32_t adv = 0;
    for (uint32_t v : seq_host_) adv = std::max(adv, static_cast<uint32_t>(v - base_));
    return adv;
  }

  void Rebase(uint32_t new_base) override {
    SyncStreams();
    InitDeviceState(new_base);
    base_ = new_base;
  }

  R4dxTpStatus ReadStatus() override {
    SyncStreams();
    return ReadStatusNoSync();
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
  // Seq counters of both channels := base (zeroed first, docs/tp.md 6.3.3), Status zeroed with
  // abort_seq_min = 0xffffffff (2.5 step 5), synchronized.
  void InitDeviceState(uint32_t base) {
    R4DX_HIP_CHECK(hipMemsetAsync(seq_.data(), 0, seq_.bytes(), stream_.get()));
    R4DX_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(seq_.data()), static_cast<int>(base),
                                     seq_.size(), stream_.get()));
    R4DX_HIP_CHECK(hipMemsetAsync(status_.data(), 0, sizeof(R4dxTpStatus), stream_.get()));
    R4DX_HIP_CHECK(hipMemsetAsync(status_.data() + offsetof(R4dxTpStatus, abort_seq_min), 0xFF, sizeof(uint32_t),
                                  stream_.get()));
    SyncOwn("InitDeviceState");
  }

  R4dxTpStatus ReadStatusNoSync() {
    R4DX_HIP_CHECK(hipMemcpyAsync(status_host_.data(), status_.data(), sizeof(R4dxTpStatus), hipMemcpyDeviceToHost,
                                  stream_.get()));
    SyncOwn("ReadStatus");
    return status_host_[0];
  }

  void SyncOwn(const char* where) {
    if (!SyncWithWatchdog(stream_.get())) {
      g_->MarkStuck();
      throw core::TpTimeoutError(std::string("tp: rank ") + std::to_string(rank_) + ": stream stuck > 30 s in " +
                                 where);
    }
  }

  // Never evicts: SyncStreams (recovery step 1, CheckHealthy, SelfTest) must wait on EVERY stream an
  // all-reduce may still be queued on, so a ninth distinct stream is a caller bug, not a cache miss.
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
  MailboxLayout L_;
  double khz_ = 0;
  int timeout_ms_ = kArTimeoutDefaultMs;
  uint8_t* host_ = nullptr;
  char* dev_ = nullptr;
  volatile uint32_t* abort_host_[2] = {nullptr, nullptr};
  uint32_t base_ = 0;
  core::Stream stream_;                  // the endpoint's own (SelfTest, Status/seq reads, rebase)
  core::DeviceBuffer<uint32_t> seq_;     // u32[64] per channel
  core::DeviceBuffer<uint8_t> status_;   // R4dxTpStatus
  core::DeviceBuffer<uint16_t> selftest_;
  // D2H targets are pinned, so every copy is truly async and SyncWithWatchdog does the waiting.
  core::PinnedBuffer<uint16_t> selftest_host_;
  core::PinnedBuffer<uint32_t> seq_host_;
  core::PinnedBuffer<R4dxTpStatus> status_host_;
  R4dxTpArArgs args_[2];                 // per channel, everything but buf / n16 / nb / w
  std::vector<hipStream_t> streams_;     // streams all-reduces were enqueued on
  hipStream_t last_stream_ = reinterpret_cast<hipStream_t>(~uintptr_t{0});
  core::TpCommStats stats_;
  std::atomic<uint64_t> aborts_{0};
  std::array<uint64_t, 2> calls_ = {0, 0};
  int64_t fault_at_ = -1;
  int fault_kind_ = kFaultThrow;
  int64_t fault_calls_ = 0;
  bool fault_fired_ = false;
};

std::unique_ptr<TpEndpoint> MakeHostMailboxComm(TpGroup* group, int rank, std::atomic<uint64_t>* heartbeat) {
  return std::make_unique<HostMailboxComm>(group, rank, heartbeat);
}

}  // namespace r4dx::model::tp
