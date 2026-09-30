// r4dx::core::TpComm -- the tensor-parallel communicator interface (docs/tp.md 6.1), plus the error
// types every TP component throws.
//
// Pure virtual and header-only so it can live in src/core: the three all-reduce call sites include
// the header-only AttentionLayer (r4dx_model_attention), which cannot link r4dx_model, so they
// reach the communicator through this interface. The implementations -- NoopComm, EmulatedComm,
// HostMailboxComm -- live in src/model/tp/ (target r4dx_tp) and are created by TpGroup.
//
// Contract shared by every implementation: all ranks make the SAME sequence of collective calls
// with the same sizes (docs/tp.md 6.3.6 "Why the sequences stay in lockstep"); any failure poisons
// the whole group (Abort) and only TpGroup::Recover() heals it.
#pragma once

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace r4dx::core {

struct TpError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct TpAbortedError : TpError {  // the group is poisoned (this or another rank aborted)
  using TpError::TpError;
};
struct TpTimeoutError : TpError {  // a host-side wait expired
  using TpError::TpError;
};
struct TpDivergenceError : TpError {  // the ranks disagreed (lockstep fingerprint, results, rng)
  using TpError::TpError;
};
struct TpStateError : TpError {  // call Reset() first / fatal, restart the process
  using TpError::TpError;
};
struct TpUnsupportedError : TpError {
  using TpError::TpError;
};

enum TpAbortCode : uint32_t {
  kAbortNone = 0,
  kAbortTimeout = 1,
  kAbortProtocol = 2,
  kAbortHost = 3,
  kAbortShutdown = 5,
};

struct TpCommStats {
  uint64_t ar_calls[2] = {0, 0}, ar_bytes[2] = {0, 0};  // per channel
  uint64_t host_exchanges = 0;
  double host_exchange_wait_us_max = 0;
  uint64_t aborts = 0;
};

class TpComm {
 public:
  virtual ~TpComm() = default;
  virtual int Rank() const = 0;
  virtual int World() const = 0;
  // In place: buf[i] = bf16_rne(f32(buf_rank0[i]) + f32(buf_rank1[i])) on every rank, bit-identical
  // across ranks. Enqueued on `stream`. HostMailbox: never blocks the host. n*2 must be a multiple
  // of 16 and <= MaxAllReduceBytes(). Every rank must make the same sequence of calls with the same
  // n.
  virtual void AllReduceSumBf16(uint16_t* buf, int64_t n, hipStream_t stream) = 0;
  // The same sum for a [rows, row_elems] activation of ANY number of rows: one AllReduceSumBf16 call per
  // slice of whole rows that fits MaxAllReduceBytes() (64 rows of a 5120-wide row), in order. A 64-row
  // prefill chunk or a decode step is one call, exactly as before; a 256-row prefill super-chunk
  // (docs/prefill.md) is four, the four calls its four 64-row chunks would make. The result is
  // bit-identical however the rows are sliced -- each element is the sum of the same two ranks' values --
  // and every rank makes the same calls (lockstep), so the mailbox, its 500 ms spin timeout and its abort
  // protocol are untouched: each call is an ordinary one.
  void AllReduceSumBf16Rows(uint16_t* buf, int64_t rows, int64_t row_elems, hipStream_t stream) {
    const int64_t max_rows = (static_cast<int64_t>(MaxAllReduceBytes()) / 2) / row_elems;
    const int64_t per = max_rows > 0 ? max_rows : 1;  // a row wider than the mailbox is refused by the call
    for (int64_t r0 = 0; r0 < rows; r0 += per) {
      const int64_t n = rows - r0 < per ? rows - r0 : per;
      AllReduceSumBf16(buf + r0 * row_elems, n * row_elems, stream);
    }
  }
  // Blocking host rendezvous: out[r*bytes .. (r+1)*bytes) = rank r's `mine`, for every r, in rank
  // order. Same `bytes` on every rank. Call only with this rank's device work synchronized.
  virtual void HostAllGather(const void* mine, size_t bytes, void* out) = 0;
  // Throws TpDivergenceError if `fingerprint` differs between ranks (one HostAllGather of 32 B).
  virtual void CheckLockstep(const uint64_t fingerprint[4]) = 0;
  // Throws TpAbortedError if any rank aborted (reads host ABORT words / exchange flag; cheap).
  virtual void CheckHealthy() = 0;
  virtual void Abort(uint32_t code, const std::string& why) noexcept = 0;
  virtual bool Aborted() const noexcept = 0;
  virtual size_t MaxAllReduceBytes() const = 0;  // 655360
  virtual void SelfTest() = 0;                   // docs/tp.md 6.3.9; throws on failure
  virtual TpCommStats Stats() const = 0;
  // Spin timeout for later AllReduceSumBf16 calls (a kernel argument; no device work). Warm-up
  // raises it to 1500 ms and restores the configured value (docs/tp.md 2.9 step 9). Clamped to
  // [10, 1500].
  virtual void SetAllReduceTimeoutMs(int ms) = 0;
  // Recovery only (docs/tp.md 2.5 step 6), with no call of this endpoint in flight: zero every
  // host-side per-endpoint counter (call index, fault counter) and, for EmulatedComm, the xchg
  // buffers.
  virtual void ResetCounters() = 0;
  // Calls per channel since the last ResetCounters(); both ranks must agree (docs/tp.md 2.5
  // step 7).
  virtual std::array<uint64_t, 2> CallCounts() const = 0;
};

}  // namespace r4dx::core
