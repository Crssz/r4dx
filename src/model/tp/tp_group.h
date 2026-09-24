// r4dx::model::tp::TpGroup -- the shared state of one tensor-parallel group (docs/tp.md 6.1, 6.3):
// the pinned host mailbox both GPUs map, the HostExchange, the session seq base and recovery (2.5).
// It creates one TpEndpoint (a core::TpComm) per rank; for Mode::kReal that endpoint is the
// HostMailboxComm of tp_comm_host_mailbox.cpp, the port of tools/tp_bench's flag(drain 3, acq 0).
//
// Threading (docs/tp.md 2.1): each endpoint is created, used and destroyed on its rank's thread
// (hipSetDevice first). The group itself is created and destroyed on the facade thread; its
// mailbox is allocated on rank 0's thread (AllocateMailbox) and freed by ~TpGroup, which must run
// only after every endpoint is gone and no kernel can still touch the region (2.6 step 3).
//
// Shared region (one hipHostMalloc(Coherent | Mapped | Portable)), docs/tp.md 6.3.2, offsets for
// the default nb = 4 on both channels (MailboxLayout::Make computes every offset from the geometry):
//   0x000000  ABORT[0]           128-B line, written only by rank 0 (device on timeout, host on error)
//   0x000080  ABORT[1]           written only by rank 1
//   0x000800  MIRROR[0]          64 B Status mirror of rank 0 (written by rank 0's device)
//   0x000C00  MIRROR[1]
//   0x001000  FLAGS[ch0][rank0]  64 lines x 128 B, written ONLY by rank 1
//   0x003000  FLAGS[ch0][rank1]  written ONLY by rank 0
//   0x005000  FLAGS[ch1][rank0]
//   0x007000  FLAGS[ch1][rank1]
//   0x010000  MBOX[ch0][rank0]   2 slots, 64 KiB-aligned size (0x60000), written ONLY by rank 1
//   0x070000  MBOX[ch0][rank1]
//   0x0D0000  MBOX[ch1][rank0]   (0x140000)
//   0x210000  MBOX[ch1][rank1]
//   0x350000  end
// No 128-B line is ever written by both devices.
#pragma once

#include <hip/hip_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "r4dx/core/tp_comm.hpp"
#include "r4dx/core/tp_host_exchange.hpp"
#include "r4dx/kernels/tp_kernels.h"

namespace r4dx::model::tp {

// ---- host-side atomics on the shared region (docs/tp.md 6.3.5) ---------------------------------
// The engine builds as C++17, so tp_bench's std::atomic_ref is not available; clang-cl provides the
// GCC builtins. Here rather than in tp_comm_host_mailbox.cpp (N31) so TpGroup and the CPU-peer test
// use the very same helpers.
inline uint32_t HostU32Load(const volatile uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void HostU32Store(volatile uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

// ---- protocol constants (docs/tp.md 6.3.2 - 6.3.5) ----------------------------------------------
constexpr size_t kChannel0MaxBytes = 174080;   // decode T=1, DFlash verify T<=8, MTP verify T<=17
constexpr size_t kMaxAllReduceBytes = 655360;  // one 64-row prefill chunk
constexpr int kArThreads = 256;                // nt
constexpr uint32_t kInitialSeqBase = 0x00001000;
constexpr int kArTimeoutMinMs = 10, kArTimeoutMaxMs = 1500, kArTimeoutDefaultMs = 500;
constexpr std::chrono::milliseconds kStreamWatchdog{30000};  // SyncWithWatchdog in recovery etc.
constexpr uint64_t kRecoveryTag = 0x5245434F56455259ull;     // "RECOVERY" (2.5 step 7)
constexpr int kFaultThrow = 0, kFaultStall = 1;              // TpOptions::fault_kind

int ClampArTimeoutMs(int ms);

// Byte offsets of everything in the shared region, for one channel geometry.
struct MailboxLayout {
  static constexpr size_t kAbortOff[2] = {0x0000, 0x0080};
  static constexpr size_t kMirrorOff[2] = {0x0800, 0x0C00};
  static constexpr size_t kFlagsBase = 0x1000;     // FLAGS[ch][r] = kFlagsBase + (2*ch + r) * 0x2000
  static constexpr size_t kFlagsBytes = 0x2000;    // kR4dxTpNbMax lines of 128 B
  static constexpr size_t kMboxBase = 0x10000;
  static constexpr size_t kMboxAlign = 0x10000;
  struct Channel {
    int nb = 4;                 // blocks per call (the --tp-ar-nb / --tp-ar-nb-large knob)
    size_t max_bytes = 0;       // largest message on this channel
    uint32_t blk_stride = 0;    // round_up(ceil(max_n16 / nb) * 16, 128)
    uint32_t slot_stride = 0;   // nb * blk_stride
    size_t mbox_bytes = 0;      // one rank's two slots, rounded up to 64 KiB
    size_t flags_off[2] = {0, 0};  // rank r's flags (written by the peer, read by r)
    size_t mbox_off[2] = {0, 0};   // rank r's receive mailbox (written by the peer)
  };
  Channel ch[2];
  size_t bytes = 0;  // whole region

  // Throws std::invalid_argument unless both nb are in [1, 64].
  static MailboxLayout Make(int nb_small, int nb_large);
  // Channel of a message: 0 iff bytes <= kChannel0MaxBytes. Depends on the byte count only (L5).
  static int ChannelFor(size_t bytes) { return bytes <= kChannel0MaxBytes ? 0 : 1; }
};

// Per-call chunking (docs/tp.md 6.3.4; tp_bench's make_geom): n16 = bytes / 16 payload words,
// w = ceil(n16 / nb) words per block, and the grid is the nb_eff = ceil(n16 / w) blocks that own
// words (== nb for every engine size). A function of the byte count and the channel's nb only.
struct CallChunks {
  int n16 = 0, nb = 0, w = 0;
};
CallChunks ChunkCall(size_t bytes, int nb_channel);

// Waits until everything enqueued on `stream` so far has finished (true) or `limit` passes (false):
// records a marker event and polls hipEventQuery -- never hipStreamQuery, which on this box's HIP
// runtime does not see a stream ending in an event record go idle while it is being polled (N33).
// Never blocks in the driver, so a stuck kernel cannot hang the caller. Clears the thread's last
// HIP error (hipErrorNotReady) before returning. Throws core::HipError on any other error.
bool SyncWithWatchdog(hipStream_t stream, std::chrono::milliseconds limit = kStreamWatchdog);

// docs/tp.md 2.9 step 4 (tp_bench's validate_wallclock): on the CURRENT device, times a busy kernel
// bounded by an iteration count (r4dx_tp_clock_probe) with the host clock and compares the
// wall_clock64 ticks it read against hipDeviceAttributeWallClockRate. Every all-reduce timeout
// trusts that rate. Returns the measured kHz; throws core::TpError if it is more than 5% off.
// Must run before the first spinning kernel on the device.
double CheckWallClockRate();

// A rank's endpoint: the TpComm plus the per-rank pieces of recovery (docs/tp.md 2.5), each called
// on this endpoint's rank thread with none of its calls in flight.
class TpEndpoint : public core::TpComm {
 public:
  // Step 1 (the part the endpoint can do): SyncWithWatchdog on its own stream and on every stream
  // an AllReduceSumBf16 was enqueued on. A stream still busy after 30 s marks the group stuck (its
  // mailbox is then never freed) and throws core::TpTimeoutError. Those streams must outlive the
  // endpoint's use of them (a Model's stream_ does: the model is destroyed first, 2.6). At most 8
  // distinct streams: an all-reduce on a ninth throws std::logic_error before enqueuing (N43).
  virtual void SyncStreams() = 0;
  // Step 2: D2H of both channels' VRAM seq counters; returns max over them of (seq - base), in
  // wrap-safe uint32 arithmetic.
  virtual uint32_t SeqAdvance() = 0;
  // Step 5: seq counters of both channels := new_base, VRAM Status re-initialised (zero, then
  // abort_seq_min = 0xffffffff), synchronized.
  virtual void Rebase(uint32_t new_base) = 0;
  // D2H snapshot of the VRAM Status (synchronizes the endpoint's streams first).
  virtual R4dxTpStatus ReadStatus() = 0;
  // Test-only fault injection (TpOptions::fault_*): fires ONCE, at the `at_allreduce`-th
  // AllReduceSumBf16 counted from this call (1-based). kFaultThrow: throws std::runtime_error("tp
  // fault injection") before enqueuing; kFaultStall: sleeps 700 ms before enqueuing (the peer's
  // kernel times out). at_allreduce <= 0 disarms.
  virtual void ArmFaultInjection(int64_t at_allreduce, int kind) = 0;
  virtual int Device() const = 0;       // HIP ordinal the endpoint was created on
  virtual double WallClockKhz() const = 0;
  virtual uint32_t SeqBase() const = 0;
};

class HostMailboxComm;

class TpGroup {
 public:
  enum class Mode { kReal, kEmulate };
  struct Geometry {
    int nb_small = 4;  // channel 0 (--tp-ar-nb)
    int nb_large = 4;  // channel 1 (--tp-ar-nb-large)
  };
  // How Recover() reaches the rank threads: run_all runs fn(r) on every rank's thread and returns
  // when all finished (rethrowing a rank's exception); run_one runs fn on one rank's thread.
  // TpModel passes its RunAll/RunOne; tests pass RankWorker-based ones.
  struct RankRunner {
    std::function<void(const std::function<void(int)>&)> run_all;
    std::function<void(int, const std::function<void()>&)> run_one;
  };

  // docs/tp.md 2.9 step 7. kReal: world must be 2 (1.2: TP > 2 is a non-goal). kEmulate arrives in
  // P2b and throws core::TpUnsupportedError until then. ar_timeout_ms is clamped to [10, 1500].
  // seq_base: the first session base (kInitialSeqBase; tests start near the 2^32 wrap).
  static std::unique_ptr<TpGroup> Create(Mode mode, int world, const Geometry& geometry, int ar_timeout_ms,
                                         uint32_t seq_base = kInitialSeqBase);
  ~TpGroup();
  TpGroup(const TpGroup&) = delete;
  TpGroup& operator=(const TpGroup&) = delete;

  // kReal, on rank 0's thread after its hipSetDevice: hipHostMalloc(Coherent | Mapped | Portable)
  // of the whole region, then reset IMMEDIATELY (ResetMailbox with the session base): hipHostMalloc
  // does not promise zeroed memory, and a garbage flag "ahead" of the base would be a protocol
  // violation on the first all-reduce.
  void AllocateMailbox();
  // On rank `rank`'s thread (hipSetDevice done): that rank's endpoint -- its own
  // hipHostGetDevicePointer of the region, VRAM seq counters (both channels, set to the session
  // base) and Status. `heartbeat` (may be null) is bumped at every AllReduceSumBf16 and registered
  // with the HostExchange (docs/tp.md 2.2 step 6). One endpoint per rank at a time.
  std::unique_ptr<TpEndpoint> CreateEndpoint(int rank, std::atomic<uint64_t>* heartbeat);

  // docs/tp.md 2.5 steps 1-7, with every endpoint registered and no rank inside a comm call
  // (TpModel sets state_ = kReady, step 8). Step 1 syncs the streams the endpoints know; the
  // caller syncs any other stream it owns before calling. Throws on failure (TpModel -> kFatal).
  void Recover(const RankRunner& run);

  // Host memset of the whole region, then every flag line of both channels and ranks := seq_base,
  // then a seq_cst fence. Only while no kernel on any device can touch it (after allocation, and in
  // recovery step 4). The flags hold the base rather than 0 because the first call's s = base + 1
  // must see its peer's unwritten flag as "not yet" (base, one behind): a 0 is "behind" only while
  // the base is below 2^31, and a session base in the upper half -- the 2^32 wrap test, or any
  // recovery after ~2^31 calls -- would read a zeroed flag as ahead of s, a protocol violation
  // (docs/tp.md Appendix B N32).
  void ResetMailbox(uint32_t seq_base);

  Mode GetMode() const { return mode_; }
  int World() const { return world_; }
  int ArTimeoutMs() const { return ar_timeout_ms_; }
  const MailboxLayout& Layout() const { return layout_; }
  uint8_t* MailboxHost() const { return host_; }  // null until AllocateMailbox
  uint32_t SeqBase() const;
  core::HostExchange& Exchange() { return exchange_; }
  // A stream-sync watchdog fired: a kernel may still touch the mailbox, so ~TpGroup leaks it.
  void MarkStuck() { stuck_.store(true, std::memory_order_release); }
  bool Stuck() const { return stuck_.load(std::memory_order_acquire); }

 private:
  friend class HostMailboxComm;
  TpGroup(Mode mode, int world, const Geometry& geometry, int ar_timeout_ms, uint32_t seq_base);
  void Register(int rank, TpEndpoint* ep);
  void Unregister(int rank, TpEndpoint* ep) noexcept;
  std::vector<TpEndpoint*> Endpoints();

  Mode mode_;
  int world_;
  Geometry geometry_;
  int ar_timeout_ms_;
  MailboxLayout layout_;
  core::HostExchange exchange_;
  uint8_t* host_ = nullptr;
  std::atomic<bool> stuck_{false};
  mutable std::mutex mu_;       // guards endpoints_ and seq_base_
  std::vector<TpEndpoint*> endpoints_;
  uint32_t seq_base_;
};

}  // namespace r4dx::model::tp
