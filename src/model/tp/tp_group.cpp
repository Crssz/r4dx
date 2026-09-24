#include "tp/tp_group.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model::tp {

// Defined in tp_comm_host_mailbox.cpp / tp_comm_emulated.cpp.
std::unique_ptr<TpEndpoint> MakeHostMailboxComm(TpGroup* group, int rank, std::atomic<uint64_t>* heartbeat);
std::unique_ptr<TpEndpoint> MakeEmulatedComm(TpGroup* group, int rank, std::atomic<uint64_t>* heartbeat);

namespace {

size_t RoundUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

void CheckNb(const char* what, int nb) {
  if (nb < 1 || nb > static_cast<int>(kR4dxTpNbMax)) {
    throw std::invalid_argument(std::string("tp: ") + what + " = " + std::to_string(nb) + " (need 1..64)");
  }
}

}  // namespace

int ClampArTimeoutMs(int ms) { return std::min(std::max(ms, kArTimeoutMinMs), kArTimeoutMaxMs); }

MailboxLayout MailboxLayout::Make(int nb_small, int nb_large) {
  CheckNb("nb (channel 0, --tp-ar-nb)", nb_small);
  CheckNb("nb (channel 1, --tp-ar-nb-large)", nb_large);
  MailboxLayout L;
  const int nbs[2] = {nb_small, nb_large};
  const size_t max_bytes[2] = {kChannel0MaxBytes, kMaxAllReduceBytes};
  size_t off = kMboxBase;
  for (int c = 0; c < 2; ++c) {
    Channel& ch = L.ch[c];
    ch.nb = nbs[c];
    ch.max_bytes = max_bytes[c];
    const size_t max_n16 = max_bytes[c] / 16;
    const size_t w_max = (max_n16 + static_cast<size_t>(ch.nb) - 1) / static_cast<size_t>(ch.nb);
    ch.blk_stride = static_cast<uint32_t>(RoundUp(w_max * 16, kR4dxTpLineBytes));
    ch.slot_stride = static_cast<uint32_t>(static_cast<size_t>(ch.nb) * ch.blk_stride);
    ch.mbox_bytes = RoundUp(2 * static_cast<size_t>(ch.slot_stride), kMboxAlign);
    for (int r = 0; r < 2; ++r) {
      ch.flags_off[r] = kFlagsBase + static_cast<size_t>(2 * c + r) * kFlagsBytes;
      ch.mbox_off[r] = off;
      off += ch.mbox_bytes;
    }
  }
  L.bytes = off;
  return L;
}

CallChunks ChunkCall(size_t bytes, int nb_channel) {
  CallChunks g;
  g.n16 = static_cast<int>(bytes / 16);
  const int nb = std::max(1, std::min(nb_channel, g.n16));
  g.w = (g.n16 + nb - 1) / nb;
  g.nb = (g.n16 + g.w - 1) / g.w;
  return g;
}

bool SyncWithWatchdog(hipStream_t stream, std::chrono::milliseconds limit) {
  // Polls a marker EVENT, never hipStreamQuery (docs/tp.md Appendix B N33): on this box's HIP
  // runtime, while hipStreamQuery is being polled -- tight or with 1 ms sleeps -- a stream whose
  // last command is an event record never reports idle, even long after its kernels finished;
  // hipEventQuery polling (tp_bench's wait_event_wd) completes normally.
  hipEvent_t ev = nullptr;
  R4DX_HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  struct EventGuard {
    hipEvent_t e;
    ~EventGuard() { (void)hipEventDestroy(e); }
  } guard{ev};
  return SyncWithWatchdogEvent(stream, ev, limit);
}

bool SyncWithWatchdogEvent(hipStream_t stream, hipEvent_t ev, std::chrono::milliseconds limit) {
  R4DX_HIP_CHECK(hipEventRecord(ev, stream));
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t spins = 0;; ++spins) {
    const hipError_t e = hipEventQuery(ev);
    if (e == hipSuccess) break;
    if (e != hipErrorNotReady) {
      (void)hipGetLastError();
      throw core::HipError(e, "hipEventQuery", __FILE__, __LINE__);
    }
    if ((spins & 255u) == 0u && std::chrono::steady_clock::now() - t0 > limit) {
      (void)hipGetLastError();
      return false;
    }
    // tp_bench's wait: yield for the first 200k polls (a sleep on Windows rounds up to the timer
    // tick, which would add milliseconds to every short wait and wreck CheckWallClockRate's host
    // timing), then 1 ms sleeps for a genuinely long wait.
    if (spins < 200000u) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  (void)hipGetLastError();  // hipEventQuery leaves hipErrorNotReady as the thread's last error
  return true;
}

double CheckWallClockRate() {
  int dev = 0;
  R4DX_HIP_CHECK(hipGetDevice(&dev));
  int rate = 0;
  R4DX_HIP_CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, dev));
  if (rate <= 0) throw core::TpError("tp: hipDeviceAttributeWallClockRate of HIP device " + std::to_string(dev) + " is 0");
  const double reported = static_cast<double>(rate);
  core::Stream st;
  core::DeviceBuffer<uint64_t> d(2);
  const auto run = [&](int iters, double* host_s) -> uint64_t {
    const auto a = std::chrono::steady_clock::now();
    r4dx_tp_clock_probe(reinterpret_cast<int64_t>(d.data()), iters, reinterpret_cast<int64_t>(st.get()));
    if (!SyncWithWatchdog(st.get())) throw core::TpTimeoutError("tp: stream stuck in the WallClockRate check");
    *host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    uint64_t v[2] = {0, 0};
    R4DX_HIP_CHECK(hipMemcpyAsync(v, d.data(), sizeof v, hipMemcpyDeviceToHost, st.get()));
    if (!SyncWithWatchdog(st.get())) throw core::TpTimeoutError("tp: stream stuck in the WallClockRate check");
    return v[1] - v[0];
  };
  double hs = 0, ovh = INFINITY;
  (void)run(0, &hs);  // warm-up (code-object load)
  for (int i = 0; i < 3; ++i) {
    (void)run(0, &hs);
    ovh = std::min(ovh, hs);
  }
  int iters = 1200;  // ~10 ms at ~1 GHz+; grown if the interval is too short to be precise
  uint64_t ticks = 0;
  double busy_s = 0;
  for (int attempt = 0; attempt < 4; ++attempt) {
    ticks = run(iters, &hs);
    busy_s = hs - ovh;
    if (busy_s >= 0.005 || iters >= 16000) break;
    iters *= 4;
  }
  const double measured = busy_s > 0 ? static_cast<double>(ticks) / (busy_s * 1000.0) : NAN;
  const double rel = std::fabs(measured - reported) / reported;
  if (!std::isfinite(rel) || rel > 0.05) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "tp: hipDeviceAttributeWallClockRate of HIP device %d is %.0f kHz but the host clock "
                  "measures %.0f kHz (%.1f%% off, limit 5%%): every all-reduce timeout depends on it",
                  dev, reported, measured, rel * 100.0);
    throw core::TpError(buf);
  }
  return measured;
}

// ---- TpGroup -------------------------------------------------------------------------------------

std::unique_ptr<TpGroup> TpGroup::Create(Mode mode, int world, const Geometry& geometry, int ar_timeout_ms,
                                         uint32_t seq_base) {
  if (world != 2) {
    throw std::invalid_argument(std::string("tp: TpGroup ") + (mode == Mode::kReal ? "real" : "emulate") +
                                " mode needs world == 2, got " + std::to_string(world));
  }
  return std::unique_ptr<TpGroup>(new TpGroup(mode, world, geometry, ar_timeout_ms, seq_base));
}

TpGroup::TpGroup(Mode mode, int world, const Geometry& geometry, int ar_timeout_ms, uint32_t seq_base)
    : mode_(mode),
      world_(world),
      geometry_(geometry),
      ar_timeout_ms_(ClampArTimeoutMs(ar_timeout_ms)),
      layout_(MailboxLayout::Make(geometry.nb_small, geometry.nb_large)),
      exchange_(world),
      endpoints_(static_cast<size_t>(world), nullptr),
      seq_base_(seq_base) {
  for (int r = 0; r < 2; ++r) {
    emu_abort_[r].store(0, std::memory_order_relaxed);
    for (int p = 0; p < 2; ++p) emu_xchg_[r][p].store(nullptr, std::memory_order_relaxed);
  }
}

TpGroup::~TpGroup() {
  if (host_ == nullptr) return;
  bool live = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (TpEndpoint* ep : endpoints_) live = live || ep != nullptr;
  }
  if (live || Stuck()) {
    // A kernel may still read or write the region: leaking it is the only safe choice (2.6).
    std::fprintf(stderr, "[r4dx-tp] mailbox not freed (%s)\n",
                 live ? "an endpoint is still alive" : "a stream-sync watchdog fired");
    return;
  }
  (void)hipHostFree(host_);
  host_ = nullptr;
}

void TpGroup::AllocateMailbox() {
  if (mode_ != Mode::kReal) throw std::logic_error("tp: TpGroup::AllocateMailbox: emulate mode has no mailbox");
  if (host_ != nullptr) throw std::logic_error("tp: TpGroup::AllocateMailbox called twice");
  void* h = nullptr;
  R4DX_HIP_CHECK(hipHostMalloc(&h, layout_.bytes, hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
  host_ = static_cast<uint8_t*>(h);
  ResetMailbox(SeqBase());
}

void TpGroup::ResetMailbox(uint32_t seq_base) {
  if (mode_ == Mode::kEmulate) {
    // No shared region: the emulated ABORT words are the only rank-visible state to clear. The
    // exchange buffers need no reset here -- each endpoint zero-fills its own in ResetCounters
    // (docs/tp.md 2.5 step 6).
    for (auto& w : emu_abort_) w.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return;
  }
  if (host_ == nullptr) throw std::logic_error("tp: TpGroup mailbox not allocated");
  std::memset(host_, 0, layout_.bytes);
  for (const MailboxLayout::Channel& ch : layout_.ch) {
    for (size_t off : ch.flags_off) {
      for (uint32_t b = 0; b < kR4dxTpNbMax; ++b) {
        *reinterpret_cast<uint32_t*>(host_ + off + static_cast<size_t>(b) * kR4dxTpLineBytes) = seq_base;
      }
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

std::unique_ptr<TpEndpoint> TpGroup::CreateEndpoint(int rank, std::atomic<uint64_t>* heartbeat) {
  if (rank < 0 || rank >= world_) {
    throw std::invalid_argument("tp: CreateEndpoint rank " + std::to_string(rank) + " outside [0, " +
                                std::to_string(world_) + ")");
  }
  if (mode_ == Mode::kEmulate) return MakeEmulatedComm(this, rank, heartbeat);
  if (host_ == nullptr) throw std::logic_error("tp: CreateEndpoint before AllocateMailbox");
  return MakeHostMailboxComm(this, rank, heartbeat);
}

uint32_t TpGroup::SeqBase() const {
  std::lock_guard<std::mutex> lk(mu_);
  return seq_base_;
}

void TpGroup::Register(int rank, TpEndpoint* ep) {
  std::lock_guard<std::mutex> lk(mu_);
  if (endpoints_[static_cast<size_t>(rank)] != nullptr) {
    throw std::logic_error("tp: rank " + std::to_string(rank) + " already has an endpoint");
  }
  endpoints_[static_cast<size_t>(rank)] = ep;
}

void TpGroup::Unregister(int rank, TpEndpoint* ep) noexcept {
  std::lock_guard<std::mutex> lk(mu_);
  if (endpoints_[static_cast<size_t>(rank)] == ep) endpoints_[static_cast<size_t>(rank)] = nullptr;
}

std::vector<TpEndpoint*> TpGroup::Endpoints() {
  std::lock_guard<std::mutex> lk(mu_);
  return endpoints_;
}

void TpGroup::Recover(const RankRunner& run) {
  if (!run.run_all || !run.run_one) throw std::invalid_argument("tp: Recover needs run_all and run_one");
  // Every step closure co-owns the state it touches (docs/tp.md Appendix B N53): TpModel's progress
  // watchdog may give up on a step while a rank is still inside it, and this frame then unwinds.
  const auto eps = std::make_shared<const std::vector<TpEndpoint*>>(Endpoints());
  for (int r = 0; r < world_; ++r) {
    if ((*eps)[static_cast<size_t>(r)] == nullptr) {
      throw core::TpStateError("tp: Recover: rank " + std::to_string(r) + " has no endpoint");
    }
  }
  // 1. every stream the endpoints enqueued on is idle (bounded 30 s; spinning kernels are bounded by
  //    the all-reduce timeout, so this terminates unless a kernel is truly stuck -> TpTimeoutError).
  run.run_all([eps](int r) { (*eps)[static_cast<size_t>(r)]->SyncStreams(); });
  // 2. each rank's VRAM seq counters; 3. a base above every value used, in wrap-safe arithmetic.
  const auto advance = std::make_shared<std::vector<uint32_t>>(static_cast<size_t>(world_), 0u);
  run.run_all([eps, advance](int r) {
    (*advance)[static_cast<size_t>(r)] = (*eps)[static_cast<size_t>(r)]->SeqAdvance();
  });
  const uint32_t old_base = SeqBase();
  const uint32_t new_base = old_base + *std::max_element(advance->begin(), advance->end()) + 64u;
  // 4. zero the whole region (ABORT words, mirrors, flags, slots), flag lines := new_base (N32):
  //    both GPUs are idle.
  run.run_one(0, [this, new_base] { ResetMailbox(new_base); });
  // 5. seq counters := new_base, Status re-initialised, on each rank.
  run.run_all([eps, new_base](int r) { (*eps)[static_cast<size_t>(r)]->Rebase(new_base); });
  {
    std::lock_guard<std::mutex> lk(mu_);
    seq_base_ = new_base;
  }
  // 6. host-side counters, with no rank inside any comm call.
  exchange_.Reset();
  run.run_all([eps](int r) { (*eps)[static_cast<size_t>(r)]->ResetCounters(); });
  // 7. both ranks agree on the new base and their (zeroed) call counters, then the self-test.
  run.run_all([eps, new_base](int r) {
    TpEndpoint* ep = (*eps)[static_cast<size_t>(r)];
    const std::array<uint64_t, 2> calls = ep->CallCounts();
    const uint64_t fp[4] = {kRecoveryTag, new_base, calls[0], calls[1]};
    ep->CheckLockstep(fp);
    ep->SelfTest();
  });
}

}  // namespace r4dx::model::tp
