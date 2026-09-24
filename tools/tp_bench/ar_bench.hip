// =================================================================================================
// ar_bench.hip -- TP=2 all-reduce feasibility benchmark for r4dx (two R9700 / gfx1201, no P2P)
// =================================================================================================
//
// WHY. r4dx is a single-GPU engine. TP=2 across the box's two R9700s pays off only if the in-engine
// all-reduce is cheap: each decode token needs 128 all-reduces of ~10 KiB (64 layers x 2), and 1.5x
// needs L(10 KiB) <= ~7-12 us (break-even ~80 us); DFlash2 verify needs L(80 KiB) <= ~15 us; prefill
// chunks need 640 KiB. The cards have NO peer access (hipDeviceCanAccessPeer == 0, isLargeBar 0), so
// the only fast option is a kernel-driven, zero-copy, one-shot PUSH all-reduce through PINNED HOST
// MEMORY that both GPUs map. This tool measures that path hop by hop and in an emulated decode step.
// hipMemcpyPeerAsync is never used (it is mis-ordered against the consuming kernel on this box).
//
// MODES (see README.md): probe, loopback, pingpong, allreduce, stress, sweep.
//
// -------------------------------------------------------------------------------------------------
// SHARED HOST REGION (one pinned allocation, mapped into both devices; each item on its own 128-B line)
//   0x00000  ABORT[0]     written only by rank 0 (its GPU on timeout / protocol error; its host thread
//                         on a host-side error). Read by everyone.
//   0x00080  ABORT[1]     written only by rank 1.
//   0x00800  MIRROR[0]    rank 0's Status copy (a publish kernel on rank 0's GPU writes it; the host
//   0x00C00  MIRROR[1]    reads it). Written only by that rank's device.
//   0x01000  FLAGS[0][b]  rank 0's per-block flags, b < 64, one 128-B line each: written ONLY by rank 1.
//   0x03000  FLAGS[1][b]  rank 1's per-block flags: written ONLY by rank 0.
//   0x10000  MBOX[0]      rank 0's mailbox: 2 slots (seq parity) x nb block regions, each region
//                         starting on a 128-B line: written ONLY by rank 1 (data), read by rank 0.
//   +region  MBOX[1]      rank 1's mailbox: written ONLY by rank 0.
//   => no 128-B line is ever written by both devices. VRAM holds each rank's inputs/outputs, its
//      per-block seq counters and its Status (diagnostics + verify record).
//
// THE ALL-REDUCE (both variants; rank r, block b of nb, call with per-block sequence number s)
//   * s = ++seq[b] (uint32 in VRAM, incremented by the kernel => hipGraph replay-safe). Both ranks run
//     the same sequence of all-reduce launches with the same nb (SPMD), so seq[b] stays in lockstep.
//   * slot = s & 1 (double buffering). Block b owns payload 16-B words [b*w, min((b+1)*w, n16)).
//   * PUSH: block b writes its chunk of MY input into the PEER's mailbox, slot `slot`, region b.
//   * WAIT until the peer's chunk for call s is in MY mailbox, slot `slot`, region b.
//   * REDUCE: out = bf16_rne(f32(in) + f32(peer)). fp32 add is commutative, so both ranks produce
//     bit-identical outputs.
//
//   Variant FLAG (libr4d's r4d_ar_oneshot_2rank_exact protocol, hardened):
//     push with plain 16-B stores; DRAIN; __syncthreads; tid0: system-scope RELEASE store
//     peer.FLAGS[b] = s; tid0: spin until (int)(FLAGS[b] - s) >= 0 (--spin rlx, the default: system-
//     scope RELAXED polls, then ONE system-scope acquire fence after success; --spin acq: an ACQUIRE
//     load on every poll); __syncthreads; optional all-thread __threadfence_system (--acq 1); reduce
//     from own mailbox.
//     Memory-ordering argument:
//       - drain=1: every thread executes __threadfence_system() (seq_cst, system scope: on gfx12 a
//         global_wb scope:SCOPE_SYS + s_wait_storecnt 0 + global_inv), so every pushed store is
//         visible at system scope before the thread reaches the barrier.
//         drain=3: an explicit per-wave s_wait_storecnt 0. NOTE: this build runs in WGP mode, where
//         __syncthreads' workgroup release already emits s_wait_loadcnt 0 + s_wait_storecnt 0 in every
//         wave, so drain=3 is ISA-identical to "no drain". Its correctness comes from tid0's release
//         store below: global_wb scope:SCOPE_SYS is L2-wide, so it writes back the other waves' pushed
//         lines too (the barrier already made them L2-visible). Under CU mode (-mcumode) the barrier
//         lowering differs and the explicit wait would matter.
//       - The workgroup barrier orders all of the block's pushes before tid0's release store. A
//         system-scope release is cumulative over everything that happens-before it (the barrier's
//         workgroup release/acquire makes the other waves' stores happen-before it), so the peer
//         cannot observe FLAGS[b] == s before it can observe the whole chunk.
//       - The peer's poll that observes s, followed by the acquire (fence or acquire load), synchron-
//         izes-with that release (a relaxed load that reads the released value, sequenced before an
//         acquire fence, is enough in the C++/HSA model); its __syncthreads extends the ordering to all
//         its waves (workgroup acquire also invalidates L0 in WGP mode). acq=1 adds a full system
//         fence in every thread (belt and braces).
//       - Drain modes 0 (none) and 2 (tid0-only fence) are NOT offered (spec: unsafe in general).
//       - Legal flag values at call s are s-1 (not yet), s, s+1 (peer already pushed call s+1 into the
//         OTHER slot). A value >= s+2 is only possible if the peer pushed after an abort; the kernel
//         then re-reads the abort words and either bails (abort set) or reports PROTOCOL_VIOLATION.
//   Variant LL (flag-in-data, NCCL-LL style):
//     every 16-B wire word = {data32, s, data32, s}; the writer emits each wire word with ONE 16-B
//     store carrying scope:SCOPE_SYS (write-through to system memory -- inline asm, because there is
//     no 16-B atomic); the reader polls each 8-B half {data32, seq} with a 64-bit system-scope RELAXED
//     atomic load until the seq half of every 8-B half of the word equals s, then reduces. Each thread
//     issues the loads for up to 4 of its payload words in one pass and re-polls only the words that
//     are not ready yet (memory-level parallelism on the PCIe read path).
//     Memory-ordering argument:
//       - An aligned 8-B load is single-copy atomic, and each 8-B half was produced by one store, so
//         a half whose seq field reads s carries the data32 written together with that s. Validity
//         travels with the data: no flag, no drain, no release/acquire fence is needed. It relies
//         only on 8-B single-copy atomicity (the stress mode is what validates that on this box).
//       - Wire traffic is 2x the payload.
//       - Stale data: a slot of parity p holds seq s-2 (or an older session's seq, or 0 after zeroing)
//         until the peer overwrites it with s; none equals s, so stale halves are never accepted. The
//         peer can never be ahead by 2 (see slot reuse below), so the only legal values are "behind"
//         and s; any half "ahead" of s is checked against the abort words and, if none is set,
//         reported as a PROTOCOL_VIOLATION (a correctness failure, exit 2), never accepted.
//   Slot reuse (both variants): rank A's block b writes B's slot p = s&1 at call s. B's block b last
//     read slot p at call s-2. A reaches call s only after its call s-1 finished, which needed B's
//     call s-1 data (FLAG: B's flag >= s-1; LL: B's words tagged s-1), which B pushes only after its
//     call s-2 kernel completed (same-stream order). So a slot is never overwritten while it is still
//     being read, and never read before it is written. This argument needs every push to follow the
//     protocol -- hence the sticky local abort below (an aborted rank pushes nothing more).
//
// SAFETY / LIVENESS
//   * Every spin loop: __builtin_amdgcn_s_sleep, wrap-safe compare (int)(cur - want) >= 0, and a
//     wall_clock64() timeout (--timeout-ms, hard cap 500 ms, far below the 2 s Windows TDR). Every 16
//     polls it also reads BOTH abort words (in the same load clause as the poll); if either is set it
//     returns at once. On timeout a block release-stores its rank's ABORT word, records
//     block/seq/phase in its Status, and returns WITHOUT reducing.
//   * Sticky abort: a block that times out, violates the protocol or bails on a set abort word also
//     sets its rank's VRAM `sticky` word. Every later all-reduce kernel of that rank reads it at entry
//     (one VRAM load, issued together with the seq load) and then SKIPS push, release, wait and reduce.
//     So an aborted rank never pushes again: in-flight batches cannot overwrite a slot the stalled peer
//     has yet to read, and secondary failures cannot masquerade as protocol violations.
//   * The host checks both ABORT words and the Status mirror after every batch; a set abort word ends
//     the run (exit 3; exit 2 for a protocol violation) with a report of which rank/block/call failed.
//     Mismatches at or after the first failed call are labelled post-abort (consequential).
//   * Every host wait polls hipEventQuery / hipStreamQuery against a wall-clock watchdog (30 s); a
//     stuck stream is reported (exit 3) instead of hanging. Host threads never wait on the peer's
//     stream; they only meet at host barriers (which time out; a barrier timeout is an error, never a
//     silent early "ok"). hipGraphs are captured, instantiated and uploaded BEFORE the start barrier,
//     and each rank enqueues its next batch before any host-side printing, so a host stall idles both
//     GPUs instead of leaving one rank's work unpaired.
//   * No hipMalloc / hipFree / hipHostFree / hipDeviceSynchronize inside timed loops.
//   * The shared region is zeroed (host memset, both GPUs idle) before each session, AND each session
//     starts its per-block seq counters above every seq value any earlier session in this process
//     used, so a stale flag or LL seq half (e.g. a CPU memset that a GPU did not snoop) is always
//     "behind" and never accepted.
//   * Loopback / pingpong kernels: a 100-round calibration launch, then launches of ~40 ms
//     (calibrated, 16..20000 rounds), AND an in-kernel budget: the initiator starts no new round once
//     kPingBudgetMs of wall_clock64 have passed since kernel entry (the responder: +50 ms), and fails
//     with ABORT_BUDGET instead. A launch therefore lasts at most budget + one timeout (< 1 s).
//   * hipDeviceAttributeWallClockRate is validated against the host clock once per device before the
//     first spinning kernel (exit 1 if it is off by more than 5%): every timeout and every in-kernel ns
//     figure depends on it.
//
// CORRECTNESS DATA: rank r's input for call c, element j is (hash(c, j, r) % 97) - 48 in bf16 (a
// 32-bit mixer of the full call index), so the fp32 sum is exact in bf16 and a stale 16-B word matches
// the expected one with probability ~97^-8. A verify kernel checks every output of every batch (<= 50
// calls per batch in isolated/stress, 128 per decode token); gen/verify run OUTSIDE the hipEvent-timed
// region (excluded, not subtracted).
// =================================================================================================

#define _CRT_SECURE_NO_WARNINGS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

// =================================================================================================
// Constants
// =================================================================================================
enum ExitCode { EXIT_OK = 0, EXIT_ERR = 1, EXIT_MISMATCH = 2, EXIT_TIMEOUT = 3 };

constexpr size_t kLine = 128;          // every flag / abort word / block region starts on its own line
constexpr int kFlagStride = 32;        // u32 per flag line
constexpr int kNbMax = 64;             // max blocks per all-reduce
constexpr size_t kOffAbort[2] = {0x0000, 0x0080};
constexpr size_t kOffMirror[2] = {0x0800, 0x0C00};
constexpr size_t kOffFlags[2] = {0x1000, 0x3000};  // kNbMax lines (8 KiB) each
constexpr size_t kOffMbox0 = 0x10000;
constexpr int kTimeoutCapMs = 500;
constexpr double kWatchdogS = 30.0;    // host wall-clock limit per batch / launch
constexpr double kBarrierS = 120.0;    // host barrier limit
constexpr int kIsoK = 50;              // all-reduces per verified batch (isolated, stress)
constexpr int kIsoDiscard = 4;         // isolated: leading batches (200 ARs) not timed (clock ramp)
constexpr int kLayers = 64;
constexpr int kDecodeK = 2 * kLayers;  // all-reduces per decode token
constexpr int kInflight = 3;           // batches enqueued ahead of the one being checked
constexpr int kEvSlots = kInflight + 1;
constexpr int kMaxChunk = 20000;       // max ping rounds per launch
constexpr int kMinChunk = 16;          // min ping rounds per launch (after calibration)
constexpr int kCalibRounds = 100;      // first (calibration) ping launch
constexpr double kChunkTargetMs = 40.0;
constexpr int kPingBudgetMs = 250;     // in-kernel budget per ping launch (initiator; responder +50 ms)
constexpr int kPingRespExtraMs = 50;
constexpr double kClockTolerance = 0.05;  // WallClockRate vs host clock
constexpr size_t kMaxBytes = 1u << 20; // --bytes cap (keeps the mismatch key packing valid)
// Projection inputs from the TP=2 plan (per-rank decode step at zero comm cost; single-GPU step)
constexpr double kTokMsZeroCommLo = 17.0, kTokMsZeroCommHi = 17.6, kTokMsSingleGpu = 27.8;

// ABORT word values / Status.code
enum : u32 { ABORT_NONE = 0, ABORT_TIMEOUT = 1, ABORT_PROTOCOL = 2, ABORT_HOST = 3, ABORT_BUDGET = 4 };
// Status.phase
enum : u32 { PH_NONE = 0, PH_FLAG_WAIT = 1, PH_LL_WAIT = 2, PH_PING_INIT = 3, PH_PING_RESP = 4, PH_PING_BUDGET = 5 };

// Per-rank diagnostics + verify record. Lives in VRAM; a publish kernel copies it (16 u32) to the
// rank's MIRROR line in host memory after every batch.
struct Status {
  u32 claimed;        // 0 until the first failing thread claims the record
  u32 code;           // ABORT_TIMEOUT / ABORT_PROTOCOL / ABORT_BUDGET
  u32 block;
  u32 seq;            // per-block seq of the failing call (call index = seq - 1 - session seq base)
  u32 phase;
  u32 word;           // payload 16-B word (LL) or 0xffffffff
  u32 got;            // flag value / seq half observed at failure
  u32 n_timeouts;     // failing threads (LL: per thread; FLAG: per block)
  u32 n_abort_exits;  // blocks/threads that bailed out because an abort word was set
  u32 mismatches;     // verify: mismatching elements
  u32 sticky;         // != 0 once any block of this rank failed or bailed: later AR kernels skip
  u32 abort_seq_min;  // min seq at which a block of this rank failed / bailed / skipped (init ~0)
  u64 min_key;        // verify: min over mismatches of (call << 36 | elem << 16 | got_bits)
  u32 n_skipped;      // AR blocks that skipped a whole call because `sticky` was set
  u32 resp_started;   // loopback: the responder block started (co-scheduling diagnostic)
};
static_assert(sizeof(Status) == 64, "Status must be 16 u32");

static const char* abort_code_name(u32 c) {
  switch (c) {
    case ABORT_NONE: return "none";
    case ABORT_TIMEOUT: return "timeout";
    case ABORT_PROTOCOL: return "protocol-violation (peer flag/seq ahead by >= 2 calls with no abort set)";
    case ABORT_HOST: return "host-error";
    case ABORT_BUDGET: return "ping-launch-budget-exceeded";
    default: return "unknown";
  }
}
static const char* phase_name(u32 p) {
  switch (p) {
    case PH_FLAG_WAIT: return "flag-wait";
    case PH_LL_WAIT: return "ll-wait";
    case PH_PING_INIT: return "ping-initiator-wait";
    case PH_PING_RESP: return "ping-responder-wait";
    case PH_PING_BUDGET: return "ping-launch-budget";
    default: return "none";
  }
}

// =================================================================================================
// Host utilities: errors, JSON, stats, host barrier, watchdog waits
// =================================================================================================
struct HipError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
static std::string hip_msg(hipError_t e, const char* expr, const char* file, int line) {
  char buf[768];
  snprintf(buf, sizeof buf, "%s -> %s (%d) at %s:%d", expr, hipGetErrorString(e), (int)e, file, line);
  return buf;
}
#define HIPC(expr)                                                                   \
  do {                                                                               \
    hipError_t _e = (expr);                                                          \
    if (_e != hipSuccess) throw HipError(hip_msg(_e, #expr, __FILE__, __LINE__));    \
  } while (0)

// hipEventQuery/hipStreamQuery leave hipErrorNotReady as the thread's last error; ignore it here.
static void check_launch(const char* what) {
  hipError_t e = hipGetLastError();
  if (e != hipSuccess && e != hipErrorNotReady)
    throw HipError(std::string("kernel launch (") + what + ") -> " + hipGetErrorString(e));
}

static std::atomic<bool> g_stuck{false};  // a watchdog fired: kernels may still run -> leak and _Exit

static size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
static double secs_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
static std::string hex64(u64 v) {
  char b[32];
  snprintf(b, sizeof b, "0x%llx", v);
  return b;
}
static int worse_code(int a, int b) {
  // priority: host error (1) > timeout (3) > mismatch (2) > ok (0)
  auto rank = [](int c) { return c == EXIT_ERR ? 3 : c == EXIT_TIMEOUT ? 2 : c == EXIT_MISMATCH ? 1 : 0; };
  return rank(a) >= rank(b) ? a : b;
}

// ---- minimal ordered JSON value ----------------------------------------------------------------
class J {
 public:
  enum Kind { Null, Bool, Int, Num, Str, Arr, Obj };
  J() {}
  J(bool v) : k_(Bool), b_(v) {}
  J(int v) : k_(Int), i_(v) {}
  J(unsigned v) : k_(Int), i_(v) {}
  J(long v) : k_(Int), i_(v) {}
  J(unsigned long v) : k_(Int), i_((long long)v) {}
  J(long long v) : k_(Int), i_(v) {}
  J(unsigned long long v) : k_(Int), i_((long long)v) {}
  J(double v) : k_(Num), d_(v) {}
  J(float v) : k_(Num), d_(v) {}
  J(const char* s) : k_(Str), s_(s ? s : "") {}
  J(const std::string& s) : k_(Str), s_(s) {}
  static J arr() { J j; j.k_ = Arr; return j; }
  static J obj() { J j; j.k_ = Obj; return j; }
  J& operator[](const std::string& key) {
    if (k_ == Null) k_ = Obj;
    for (size_t i = 0; i < keys_.size(); ++i)
      if (keys_[i] == key) return vals_[i];
    keys_.push_back(key);
    vals_.emplace_back();
    return vals_.back();
  }
  void push(const J& v) {
    if (k_ == Null) k_ = Arr;
    vals_.push_back(v);
  }
  bool is_null() const { return k_ == Null; }
  bool is_true() const { return k_ == Bool && b_; }
  std::string dump() const {
    std::string o;
    dump_(o, 0);
    o += '\n';
    return o;
  }

 private:
  Kind k_ = Null;
  bool b_ = false;
  long long i_ = 0;
  double d_ = 0;
  std::string s_;
  std::vector<std::string> keys_;
  std::vector<J> vals_;
  static void esc(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char ch : s) {
      if (ch == '"') o += "\\\"";
      else if (ch == '\\') o += "\\\\";
      else if (ch == '\n') o += "\\n";
      else if (ch == '\r') o += "\\r";
      else if (ch == '\t') o += "\\t";
      else if (ch < 0x20) {
        char b[8];
        snprintf(b, sizeof b, "\\u%04x", ch);
        o += b;
      } else o += (char)ch;
    }
    o += '"';
  }
  bool scalar() const { return k_ != Arr && k_ != Obj; }
  void dump_(std::string& o, int ind) const {
    char b[64];
    switch (k_) {
      case Null: o += "null"; break;
      case Bool: o += b_ ? "true" : "false"; break;
      case Int: snprintf(b, sizeof b, "%lld", i_); o += b; break;
      case Num:
        if (!std::isfinite(d_)) o += "null";
        else { snprintf(b, sizeof b, "%.10g", d_); o += b; }
        break;
      case Str: esc(o, s_); break;
      case Arr: {
        bool flat = true;
        for (const J& v : vals_) flat = flat && v.scalar();
        if (vals_.empty()) { o += "[]"; break; }
        if (flat) {
          o += '[';
          for (size_t i = 0; i < vals_.size(); ++i) {
            if (i) o += ", ";
            vals_[i].dump_(o, ind);
          }
          o += ']';
        } else {
          o += "[\n";
          for (size_t i = 0; i < vals_.size(); ++i) {
            o.append(ind + 2, ' ');
            vals_[i].dump_(o, ind + 2);
            o += (i + 1 < vals_.size()) ? ",\n" : "\n";
          }
          o.append(ind, ' ');
          o += ']';
        }
        break;
      }
      case Obj: {
        if (keys_.empty()) { o += "{}"; break; }
        o += "{\n";
        for (size_t i = 0; i < keys_.size(); ++i) {
          o.append(ind + 2, ' ');
          esc(o, keys_[i]);
          o += ": ";
          vals_[i].dump_(o, ind + 2);
          o += (i + 1 < keys_.size()) ? ",\n" : "\n";
        }
        o.append(ind, ' ');
        o += '}';
        break;
      }
    }
  }
};

// ---- stats ---------------------------------------------------------------------------------------
struct Summ {
  size_t n = 0;
  double mean = NAN, stdev = NAN, min = NAN, p50 = NAN, p90 = NAN, p99 = NAN, p999 = NAN, max = NAN;
};
static double pct_sorted(const std::vector<double>& s, double p) {
  if (s.empty()) return NAN;
  const double idx = p / 100.0 * (double)(s.size() - 1);
  const size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - (double)lo);
}
static Summ summarize(std::vector<double> v) {
  Summ s;
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / (double)v.size();
  double ss = 0;
  for (double x : v) ss += (x - s.mean) * (x - s.mean);
  s.stdev = v.size() > 1 ? std::sqrt(ss / (double)(v.size() - 1)) : 0.0;
  s.min = v.front();
  s.max = v.back();
  s.p50 = pct_sorted(v, 50);
  s.p90 = pct_sorted(v, 90);
  s.p99 = pct_sorted(v, 99);
  s.p999 = pct_sorted(v, 99.9);
  return s;
}
static J summ_json(const Summ& s) {
  J j = J::obj();
  j["n"] = (unsigned long long)s.n;
  j["mean"] = s.mean;
  j["stdev"] = s.stdev;
  j["min"] = s.min;
  j["p50"] = s.p50;
  j["p90"] = s.p90;
  j["p99"] = s.p99;
  j["p99_9"] = s.p999;
  j["max"] = s.max;
  return j;
}
static J vec_json(const std::vector<double>& v) {
  J a = J::arr();
  for (double x : v) a.push(x);
  return a;
}

// ---- host barrier with vote --------------------------------------------------------------------
// All n threads call vote(ok). Returns 1 if every vote was ok, 0 if any was not, -1 if the barrier
// is broken (a thread failed on the host) or the wait exceeded timeout_s. Both ranks vote the same
// number of times, so they always stop at the same batch.
class HostSync {
 public:
  explicit HostSync(int n) : n_(n) {}
  int vote(bool ok, double timeout_s) {
    std::unique_lock<std::mutex> lk(m_);
    if (broken_) return -1;
    acc_ = acc_ && ok;
    const u64 my_gen = gen_;
    if (++arrived_ == n_) {
      result_ = acc_;
      acc_ = true;
      arrived_ = 0;
      ++gen_;
      cv_.notify_all();
      return result_ ? 1 : 0;
    }
    const bool woke = cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                                   [&] { return gen_ != my_gen || broken_; });
    if (gen_ != my_gen) return result_ ? 1 : 0;
    if (!woke) {
      broken_ = true;
      timed_out_ = true;  // the caller must report this as an error, never as an early "ok"
      cv_.notify_all();
    }
    return -1;
  }
  void brk() {
    std::lock_guard<std::mutex> lk(m_);
    broken_ = true;
    cv_.notify_all();
  }
  bool timed_out() {
    std::lock_guard<std::mutex> lk(m_);
    return timed_out_;
  }
  std::atomic<long long> value{0};  // small value exchange (rank 0 posts before a vote)

 private:
  std::mutex m_;
  std::condition_variable cv_;
  int n_;
  int arrived_ = 0;
  u64 gen_ = 0;
  bool acc_ = true, result_ = true, broken_ = false, timed_out_ = false;
};

// ---- watchdog waits ------------------------------------------------------------------------------
// true = completed; false = the wall-clock limit expired (the stream is reported stuck).
static bool wait_event_wd(hipEvent_t ev, double limit_s) {
  const auto t0 = std::chrono::steady_clock::now();
  for (u32 spins = 0;; ++spins) {
    const hipError_t e = hipEventQuery(ev);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) throw HipError(std::string("hipEventQuery -> ") + hipGetErrorString(e));
    if ((spins & 255u) == 0u && secs_since(t0) > limit_s) return false;
    if (spins < 200000u) std::this_thread::yield();
    else Sleep(1);
  }
}
static bool drain_stream_wd(hipStream_t st, double limit_s) {
  const auto t0 = std::chrono::steady_clock::now();
  for (u32 spins = 0;; ++spins) {
    const hipError_t e = hipStreamQuery(st);
    if (e == hipSuccess) return true;
    if (e != hipErrorNotReady) throw HipError(std::string("hipStreamQuery -> ") + hipGetErrorString(e));
    if ((spins & 255u) == 0u && secs_since(t0) > limit_s) return false;
    if (spins < 200000u) std::this_thread::yield();
    else Sleep(1);
  }
}

// =================================================================================================
// Data pattern and bf16 helpers (host + device)
// =================================================================================================
__host__ __device__ __forceinline__ u32 fbits(float f) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __float_as_uint(f);
#else
  u32 u;
  std::memcpy(&u, &f, 4);
  return u;
#endif
}
__host__ __device__ __forceinline__ float bitsf(u32 u) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}
// fp32 -> bf16 bits, round-to-nearest-even (no NaN inputs here).
__host__ __device__ __forceinline__ u32 f2bf(float f) {
  u32 u = fbits(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return u >> 16;
}
// 32-bit integer mixer (lowbias32).
__host__ __device__ __forceinline__ u32 mix32(u32 x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}
// (hash(call, elem, rank) % 97) - 48: small integers, so every fp32 sum of two is exact in bf16. The
// hash covers the full call index (calls < 2^28 by the --count cap), so data that is stale by ANY
// number of calls, or read from a wrong block/word offset, matches only by chance (~1/97 per element).
__host__ __device__ __forceinline__ int pat_val(u64 call, u32 elem, int rank) {
  const u32 h = mix32(mix32((u32)call * 0x9E3779B1u + (u32)rank * 0xC2B2AE3Du) ^ (elem * 0x85EBCA77u));
  return (int)(h % 97u) - 48;
}
__host__ __device__ __forceinline__ u32 expected_bits(u64 call, u32 elem) {
  return f2bf((float)(pat_val(call, elem, 0) + pat_val(call, elem, 1)));
}

// =================================================================================================
// Device primitives
// =================================================================================================
__device__ __forceinline__ u32 ld_acq_sys(const u32* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u32 ld_rlx_sys(const u32* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u64 ld64_rlx_sys(const u64* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rel_sys(u32* p, u32 v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rlx_sys(u32* p, u32 v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u64 now_ticks() { return (u64)wall_clock64(); }

// One 16-B store with scope:SCOPE_SYS (write-through to system memory). There is no 16-B atomic
// store; the LL protocol needs only 8-B single-copy atomicity of each half, which this provides.
typedef u32 u32x4_t __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void st128_sys(void* p, u32 a, u32 b, u32 c, u32 d) {
  u32x4_t v = {a, b, c, d};
  asm volatile("global_store_b128 %0, %1, off scope:SCOPE_SYS" : : "v"(p), "v"(v) : "memory");
}

__device__ __forceinline__ u32 add_bf16x2(u32 a, u32 b) {
  const float a0 = __uint_as_float(a << 16), a1 = __uint_as_float(a & 0xFFFF0000u);
  const float b0 = __uint_as_float(b << 16), b1 = __uint_as_float(b & 0xFFFF0000u);
  return f2bf(a0 + b0) | (f2bf(a1 + b1) << 16);
}
__device__ __forceinline__ uint4 add_bf16x8(const uint4& a, const uint4& b) {
  uint4 r;
  r.x = add_bf16x2(a.x, b.x);
  r.y = add_bf16x2(a.y, b.y);
  r.z = add_bf16x2(a.z, b.z);
  r.w = add_bf16x2(a.w, b.w);
  return r;
}

// The rank's sticky local abort (VRAM): once set, every later AR kernel of this rank skips the call.
__device__ __forceinline__ u32 ld_sticky(const Status* st) {
  return __hip_atomic_load(&st->sticky, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void mark_sticky(Status* st, u32 seq) {
  __hip_atomic_store(&st->sticky, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  atomicMin(&st->abort_seq_min, seq);
}

// Record a timeout / protocol violation / budget overrun: release-store MY abort word (host memory,
// system scope), set the rank's sticky word, then claim the Status record (first failure wins) in VRAM.
__device__ void note_failure(u32* my_abort, Status* st, u32 code, u32 block, u32 seq, u32 phase, u32 word,
                             u32 got) {
  st_rel_sys(my_abort, code);
  mark_sticky(st, seq);
  atomicAdd(&st->n_timeouts, 1u);
  if (atomicCAS(&st->claimed, 0u, 1u) == 0u) {
    st->code = code;
    st->block = block;
    st->seq = seq;
    st->phase = phase;
    st->word = word;
    st->got = got;
  }
}
// A block/thread bailed out because an abort word was set (secondary failure).
__device__ void note_abort_exit(Status* st, u32 seq) {
  atomicAdd(&st->n_abort_exits, 1u);
  mark_sticky(st, seq);
}

// Spin until (int)(*f - want) >= 0.
//   spin_acq = 0 (default): system-scope RELAXED polls, then one system-scope acquire fence after
//                           success (synchronizes-with the writer's release, same guarantee).
//   spin_acq = 1:           a system-scope ACQUIRE load on every poll (an L2 invalidate per poll).
// Every 16th poll also loads both abort words, issued right after the flag load so one wait covers all
// three (rlx mode; in acq mode the acquire load's own wait comes first).
// Returns 0 = reached, 1 = timed out, 2 = an abort word is set. *seen = last flag value read.
__device__ __forceinline__ int spin_flag_ge(const u32* f, u32 want, const u32* ab_mine, const u32* ab_peer,
                                            u64 timeout, int spin_acq, u32* seen) {
  const u64 t0 = now_ticks();
  for (u32 it = 1;; ++it) {
    const bool slow = (it & 15u) == 0u;
    const u32 v = spin_acq ? ld_acq_sys(f) : ld_rlx_sys(f);  // flag first: issued before the abort loads
    u32 ab = 0;
    if (slow) ab = ld_rlx_sys(ab_mine) | ld_rlx_sys(ab_peer);
    if ((int)(v - want) >= 0) {
      if (!spin_acq) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // "" = system scope
      *seen = v;
      return 0;
    }
    if (slow) {
      if (ab != 0u) {
        *seen = v;
        return 2;
      }
      if (now_ticks() - t0 > timeout) {
        *seen = v;
        return 1;
      }
    }
    __builtin_amdgcn_s_sleep(1);
  }
}

// =================================================================================================
// Kernels: utility
// =================================================================================================
__global__ void noop_kernel() {}

// WallClockRate validation: wall_clock64 at entry and exit of a busy wait bounded by an ITERATION count
// (s_sleep 127 ~ 8K shader clocks each), never by the clock under test.
__global__ void clock_probe_kernel(u64* out, int iters) {
  if (threadIdx.x != 0) return;
  const u64 t0 = (u64)wall_clock64();
  for (int i = 0; i < iters; ++i) __builtin_amdgcn_s_sleep(127);
  const u64 t1 = (u64)wall_clock64();
  out[0] = t0;
  out[1] = t1;
}

// Fill a VRAM buffer with a hash pattern (decode filler ring: realistic, incompressible data).
__global__ void fill_hash_kernel(uint4* __restrict__ p, u64 n16, u32 salt) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (u64)gridDim.x * blockDim.x) {
    const u32 x = (u32)i * 4u ^ salt ^ (u32)(i >> 30);
    p[i] = make_uint4(mix32(x), mix32(x + 1u), mix32(x + 2u), mix32(x + 3u));
  }
}

// ring[k][j] = pattern(base_call + k, j, rank); grid.y = calls in the ring.
__global__ void gen_kernel(u16* __restrict__ ring, u64 base_call, u32 n_elem, int rank) {
  const u32 k = blockIdx.y;
  const u64 call = base_call + k;
  u16* row = ring + (size_t)k * n_elem;
  for (u32 j = blockIdx.x * blockDim.x + threadIdx.x; j < n_elem; j += gridDim.x * blockDim.x)
    row[j] = (u16)(fbits((float)pat_val(call, j, rank)) >> 16);  // small integers: exact
}

// Compare every output against the exact bf16 sum; count mismatches and keep the smallest
// (call, elem) with its got-bits in one 64-bit key via atomicMin.
__global__ void verify_kernel(const u16* __restrict__ ring, u64 base_call, u32 n_elem, Status* st) {
  const u32 k = blockIdx.y;
  const u64 call = base_call + k;
  const u16* row = ring + (size_t)k * n_elem;
  for (u32 j = blockIdx.x * blockDim.x + threadIdx.x; j < n_elem; j += gridDim.x * blockDim.x) {
    const u32 got = row[j];
    if (got != expected_bits(call, j)) {
      atomicAdd(&st->mismatches, 1u);
      const u64 key = (call << 36) | ((u64)j << 16) | (u64)got;
      atomicMin(&st->min_key, key);
    }
  }
}

// Copy the VRAM Status to the rank's MIRROR line in host memory (read by the host after the batch).
__global__ void publish_kernel(const u32* __restrict__ src, u32* dst, int nwords) {
  const int t = threadIdx.x;
  if (t < nwords) st_rlx_sys(dst + t, src[t]);
  __threadfence_system();
}

// Memory-bound decode filler: stream rd_words 16-B words (VRAM ring window) and read-modify-write
// wr_words 16-B words of a per-rank "state" buffer (leaves dirty L2 lines, like the GDN state update).
__global__ void __launch_bounds__(256) filler_kernel(const uint4* __restrict__ rd, u64 rd_words,
                                                     uint4* __restrict__ wr, u64 wr_words, u32* sink) {
  const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const u64 nthr = (u64)gridDim.x * blockDim.x;
  u32 acc = 0;
  u64 i = tid;
  for (; i + 3 * nthr < rd_words; i += 4 * nthr) {
    const uint4 a = rd[i], b = rd[i + nthr], c = rd[i + 2 * nthr], d = rd[i + 3 * nthr];
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < rd_words; i += nthr) {
    const uint4 a = rd[i];
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  for (u64 j = tid; j < wr_words; j += nthr) {
    uint4 v = wr[j];
    v.x += 1u;
    v.y += 1u;
    v.z += 1u;
    v.w += 1u;
    wr[j] = v;
  }
  if (acc == 0x9E3779B9u) sink[blockIdx.x & 255u] = acc;  // keeps the reads alive; practically never taken
}

// =================================================================================================
// Kernels: the all-reduce (FLAG and LL) and its local stand-in
// =================================================================================================
struct ArArgs {
  const uint4* in;        // this call's input (VRAM), n16 16-B words
  uint4* out;             // this call's output (VRAM)
  char* peer_mbox;        // PEER's mailbox base (this device's mapping of host memory)
  const char* my_mbox;    // MY mailbox base
  u32* peer_flags;        // PEER's flag lines (FLAG variant)
  const u32* my_flags;    // MY flag lines
  u32* my_abort;
  const u32* peer_abort;
  u32* seq;               // VRAM per-block seq counters
  Status* st;             // VRAM
  u64* trace;             // VRAM, 4 u64 per call (block 0 only) or nullptr
  u32 trace_cap;
  int n16;                // payload 16-B words
  int w;                  // payload words per block
  u32 blk_stride;         // bytes per block region in a slot (multiple of 128)
  u32 slot_stride;        // bytes per slot
  int drain;              // FLAG: 1 = all-thread __threadfence_system, 3 = all-thread s_wait_storecnt 0
  int acq;                // FLAG: 1 = all-thread __threadfence_system after the wait
  int spin_acq;           // FLAG: 1 = acquire load per poll, 0 = relaxed polls + one acquire fence
  u32 seq_base;           // session's seq base (call index = seq - 1 - seq_base)
  u64 timeout;            // wall_clock64 ticks
};

// Entry of both AR kernels (tid0): bump this block's seq (VRAM; only block b of this rank touches
// seq[b], launches are stream-ordered) and read the rank's sticky abort word in the same load batch.
// Returns true if the call must be skipped (the rank already failed or saw an abort).
__device__ __forceinline__ bool ar_entry(const ArArgs& a, int b, u32* s_out) {
  const u32 sticky = ld_sticky(a.st);
  const u32 s = a.seq[b] + 1u;
  a.seq[b] = s;
  *s_out = s;
  if (sticky != 0u) {
    atomicAdd(&a.st->n_skipped, 1u);
    atomicMin(&a.st->abort_seq_min, s);
    return true;
  }
  return false;
}

__global__ void __launch_bounds__(1024) ar_flag_kernel(ArArgs a) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ u32 s_seq;
  __shared__ int s_fail;  // 0 ok, 1 timeout, 2 abort exit, 3 protocol violation, 4 skipped (sticky)
  __shared__ u64 s_t0;
  if (tid == 0) {
    s_t0 = now_ticks();
    u32 s0 = 0;
    s_fail = ar_entry(a, b, &s0) ? 4 : 0;
    s_seq = s0;
  }
  __syncthreads();
  if (s_fail != 0) return;  // block-uniform: an aborted rank pushes nothing more (sticky abort)
  const u32 s = s_seq;
  const u32 slot = s & 1u;
  const int start = b * a.w;
  const int end = (start + a.w < a.n16) ? start + a.w : a.n16;
  const int cnt = end - start;
  const uint4* in = a.in + start;
  uint4* out = a.out + start;
  uint4* dst = (uint4*)(a.peer_mbox + (size_t)slot * a.slot_stride + (size_t)b * a.blk_stride);
  const uint4* src = (const uint4*)(a.my_mbox + (size_t)slot * a.slot_stride + (size_t)b * a.blk_stride);

  // PUSH my chunk into the peer's slot: up to 4 words per thread. The VRAM loads are unconditional (index
  // clamped into the chunk, so they stay in flight together); only the stores are predicated.
  for (int k0 = tid; k0 < cnt; k0 += 4 * nt) {
    uint4 x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = in[min(k0 + j * nt, cnt - 1)];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (k0 + j * nt < cnt) dst[k0 + j * nt] = x[j];
  }
  // DRAIN (every thread): make the pushes visible at system scope before the flag. (drain=3 is
  // ISA-identical to no drain in WGP mode -- see the header; tid0's release writes everything back.)
  if (a.drain == 1) __threadfence_system();
  else asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();

  u64 t_push = 0, t_wait = 0;
  if (tid == 0) {
    st_rel_sys(&a.peer_flags[(size_t)b * kFlagStride], s);  // RELEASE: data before flag
    t_push = now_ticks();  // after the release: includes its system writeback (the drain=3 flush)
    u32 seen = 0;
    int rc = spin_flag_ge(&a.my_flags[(size_t)b * kFlagStride], s, a.my_abort, a.peer_abort, a.timeout,
                          a.spin_acq, &seen);
    if (rc == 0 && (int)(seen - s) >= 2) {  // legal: s or s+1 only
      if ((ld_rlx_sys(a.my_abort) | ld_rlx_sys(a.peer_abort)) != 0u) rc = 2;  // peer pushed after an abort
      else {
        note_failure(a.my_abort, a.st, ABORT_PROTOCOL, (u32)b, s, PH_FLAG_WAIT, 0xFFFFFFFFu, seen);
        rc = 3;
      }
    }
    if (rc == 1) note_failure(a.my_abort, a.st, ABORT_TIMEOUT, (u32)b, s, PH_FLAG_WAIT, 0xFFFFFFFFu, seen);
    else if (rc == 2) note_abort_exit(a.st, s);
    s_fail = rc;
    t_wait = now_ticks();
  }
  __syncthreads();

  if (s_fail == 0) {
    if (a.acq) __threadfence_system();
    // REDUCE: up to 4 words per thread, all host-memory (and VRAM) loads issued before any store, so a
    // thread has up to 4 PCIe reads in flight instead of one per word.
    for (int k0 = tid; k0 < cnt; k0 += 4 * nt) {
      uint4 p[4], x[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (k0 + j * nt < cnt) {
          p[j] = src[k0 + j * nt];
          x[j] = in[k0 + j * nt];
        }
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (k0 + j * nt < cnt) out[k0 + j * nt] = add_bf16x8(x[j], p[j]);
    }
  }
  // (a failed block does NOT reduce: out keeps stale data, and the host reports the abort)

  if (a.trace != nullptr && b == 0) {  // block-uniform condition
    __syncthreads();
    if (tid == 0) {
      u64* t = a.trace + (size_t)((s - 1u - a.seq_base) % a.trace_cap) * 4;
      t[0] = s_t0;
      t[1] = t_push;
      t[2] = t_wait;
      t[3] = now_ticks();
    }
  }
}

// LL receive of one group of NW payload words (k0, k0+nt, ...): all 4*NW 8-B half loads of a poll are
// unconditional, so they are in flight together (a word that is already complete is simply re-read; the
// peer cannot overwrite it before this call ends). Returns 0 = reduced, 1 = timeout, 2 = abort exit,
// 3 = protocol violation. `it` counts polls across groups (abort/timeout checks every 16th poll).
template <int NW>
__device__ __forceinline__ int ll_recv_group(const ArArgs& a, const u64* src, const uint4* in, uint4* out, int k0,
                                             int nt, u32 s, int b, int start, u64 t0, u32& it) {
  uint4 mine[NW];
#pragma unroll
  for (int j = 0; j < NW; ++j) mine[j] = in[k0 + j * nt];
  for (;;) {
    const bool slow = ((++it) & 15u) == 0u;
    u64 h[NW][4];
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const u64* p = src + 4 * (size_t)(k0 + j * nt);
      h[j][0] = ld64_rlx_sys(p);
      h[j][1] = ld64_rlx_sys(p + 1);
      h[j][2] = ld64_rlx_sys(p + 2);
      h[j][3] = ld64_rlx_sys(p + 3);
    }
    u32 ab = 0;
    if (slow) ab = ld_rlx_sys(a.my_abort) | ld_rlx_sys(a.peer_abort);
    int ahead_k = -1, stale_k = -1;
    u32 ahead_got = 0, stale_got = 0;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const u32 q0 = (u32)(h[j][0] >> 32), q1 = (u32)(h[j][1] >> 32), q2 = (u32)(h[j][2] >> 32),
                q3 = (u32)(h[j][3] >> 32);
      const int d0 = (int)(q0 - s), d1 = (int)(q1 - s), d2 = (int)(q2 - s), d3 = (int)(q3 - s);
      if (d0 > 0 || d1 > 0 || d2 > 0 || d3 > 0) {  // wrap-safe "ahead of s": impossible under the protocol
        ahead_k = k0 + j * nt;
        ahead_got = d0 > 0 ? q0 : d1 > 0 ? q1 : d2 > 0 ? q2 : q3;
      } else if ((d0 | d1 | d2 | d3) != 0) {  // some half still behind s (stale)
        stale_k = k0 + j * nt;
        stale_got = d0 < 0 ? q0 : d1 < 0 ? q1 : d2 < 0 ? q2 : q3;
      }
    }
    if (ahead_k >= 0) {
      if ((ld_rlx_sys(a.my_abort) | ld_rlx_sys(a.peer_abort)) != 0u) {  // peer pushed after an abort
        note_abort_exit(a.st, s);
        return 2;
      }
      note_failure(a.my_abort, a.st, ABORT_PROTOCOL, (u32)b, s, PH_LL_WAIT, (u32)(start + ahead_k), ahead_got);
      return 3;
    }
    if (stale_k < 0) {  // every half of every word carries s
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        uint4 peer;
        peer.x = (u32)h[j][0];
        peer.y = (u32)h[j][1];
        peer.z = (u32)h[j][2];
        peer.w = (u32)h[j][3];
        out[k0 + j * nt] = add_bf16x8(mine[j], peer);
      }
      return 0;
    }
    if (slow) {
      if (ab != 0u) {
        note_abort_exit(a.st, s);
        return 2;
      }
      if (now_ticks() - t0 > a.timeout) {
        note_failure(a.my_abort, a.st, ABORT_TIMEOUT, (u32)b, s, PH_LL_WAIT, (u32)(start + stale_k), stale_got);
        return 1;
      }
    }
    __builtin_amdgcn_s_sleep(1);
  }
}

__global__ void __launch_bounds__(1024) ar_ll_kernel(ArArgs a) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ u32 s_seq;
  __shared__ int s_skip;
  __shared__ u64 s_t0;
  if (tid == 0) {
    s_t0 = now_ticks();
    u32 s0 = 0;
    s_skip = ar_entry(a, b, &s0) ? 1 : 0;
    s_seq = s0;
  }
  __syncthreads();
  if (s_skip != 0) return;  // block-uniform: an aborted rank pushes nothing more (sticky abort)
  const u32 s = s_seq;
  const u32 slot = s & 1u;
  const int start = b * a.w;
  const int end = (start + a.w < a.n16) ? start + a.w : a.n16;
  const int cnt = end - start;
  const uint4* in = a.in + start;
  uint4* out = a.out + start;
  // 2 wire words (32 B) per payload word
  uint4* dst = (uint4*)(a.peer_mbox + (size_t)slot * a.slot_stride + (size_t)b * a.blk_stride);
  const u64* src = (const u64*)(a.my_mbox + (size_t)slot * a.slot_stride + (size_t)b * a.blk_stride);

  // PUSH: payload word {x,y,z,w} -> wire {x,s,y,s} {z,s,w,s}, each one 16-B scope:SYS store
  // (VRAM loads unconditional with a clamped index so they stay in flight together; stores predicated)
  for (int k0 = tid; k0 < cnt; k0 += 4 * nt) {
    uint4 x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = in[min(k0 + j * nt, cnt - 1)];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int k = k0 + j * nt;
      if (k < cnt) {
        st128_sys(dst + 2 * k, x[j].x, s, x[j].y, s);
        st128_sys(dst + 2 * k + 1, x[j].z, s, x[j].w, s);
      }
    }
  }
  const bool tracing = (a.trace != nullptr && b == 0);  // block-uniform
  u64 t_push = 0;
  if (tracing) {
    __syncthreads();
    if (tid == 0) t_push = now_ticks();
  }

  // RECEIVE + REDUCE: a thread takes its payload words in groups of NW <= 4 (NW is a compile-time
  // constant, so every poll issues all 4*NW half loads unconditionally before one wait), polls until the
  // four seq halves of every word in the group equal s, then reduces the group. Any half AHEAD of s is
  // illegal (see the header).
  const u64 t0 = now_ticks();
  u32 it = 0;
  int fail = 0;  // 1 timeout, 2 abort exit, 3 protocol violation
  for (int k0 = tid; k0 < cnt && fail == 0; k0 += 4 * nt) {
    const int nw = min(4, (cnt - k0 + nt - 1) / nt);
    if (nw == 1) fail = ll_recv_group<1>(a, src, in, out, k0, nt, s, b, start, t0, it);
    else if (nw == 2) fail = ll_recv_group<2>(a, src, in, out, k0, nt, s, b, start, t0, it);
    else if (nw == 3) fail = ll_recv_group<3>(a, src, in, out, k0, nt, s, b, start, t0, it);
    else fail = ll_recv_group<4>(a, src, in, out, k0, nt, s, b, start, t0, it);
  }

  if (tracing) {
    __syncthreads();
    if (tid == 0) {
      const u64 t_end = now_ticks();
      u64* t = a.trace + (size_t)((s - 1u - a.seq_base) % a.trace_cap) * 4;
      t[0] = s_t0;
      t[1] = t_push;
      t[2] = t_end;  // LL: the wait is fused with the reduce
      t[3] = t_end;
    }
  }
}

// Decode run (b): same grid, same chunking, out = in + local copy; no host traffic.
__global__ void __launch_bounds__(1024) local_standin_kernel(const uint4* __restrict__ in,
                                                             const uint4* __restrict__ local,
                                                             uint4* __restrict__ out, int n16, int w) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int start = b * w;
  const int end = (start + w < n16) ? start + w : n16;
  for (int k = start + tid; k < end; k += nt) out[k] = add_bf16x8(in[k], local[k]);
}

// =================================================================================================
// Kernels: ping-pong (loopback and cross-device)
// =================================================================================================
struct PingArgs {
  u32* flag_out;          // written by me (system-scope release)
  const u32* flag_in;     // read by me (system-scope acquire)
  u32* my_abort;
  const u32* peer_abort;
  u32 base;               // rounds are base+1 .. base+rounds
  int rounds;
  u32* rtt;               // VRAM, one tick count per round (initiator only)
  Status* st;
  u64 timeout;            // per-round spin timeout (ticks)
  u64 budget;             // per-launch budget (ticks since kernel entry): no new round after it
  int spin_acq;
};

// Launch runtime is bounded by the clock, not only by the host's calibration: before starting a new
// round the initiator checks the time since kernel entry and fails with ABORT_BUDGET past a.budget,
// so a launch lasts at most budget + one round (<= budget + timeout). The peer then sees the abort
// word and exits.
__device__ void ping_initiator(const PingArgs& a, u64 t_launch) {
  u64 t_last = t_launch;
  for (int i = 0; i < a.rounds; ++i) {
    const u32 r = a.base + 1u + (u32)i;
    if (t_last - t_launch > a.budget) {
      note_failure(a.my_abort, a.st, ABORT_BUDGET, blockIdx.x, r, PH_PING_BUDGET, 0xFFFFFFFFu, (u32)i);
      return;
    }
    const u64 t0 = now_ticks();
    st_rel_sys(a.flag_out, r);
    u32 seen = 0;
    const int rc = spin_flag_ge(a.flag_in, r, a.my_abort, a.peer_abort, a.timeout, a.spin_acq, &seen);
    const u64 t1 = now_ticks();
    if (rc != 0) {
      if (rc == 1) note_failure(a.my_abort, a.st, ABORT_TIMEOUT, blockIdx.x, r, PH_PING_INIT, 0xFFFFFFFFu, seen);
      else note_abort_exit(a.st, r);
      return;
    }
    a.rtt[i] = (u32)(t1 - t0);
    t_last = t1;
  }
}
// The responder checks its own (larger) budget after every echo -- off the critical path, since the
// initiator needs a full round trip before the next flag can arrive.
__device__ void ping_responder(const PingArgs& a, u64 t_launch) {
  for (int i = 0; i < a.rounds; ++i) {
    const u32 r = a.base + 1u + (u32)i;
    u32 seen = 0;
    const int rc = spin_flag_ge(a.flag_in, r, a.my_abort, a.peer_abort, a.timeout, a.spin_acq, &seen);
    if (rc != 0) {
      if (rc == 1) note_failure(a.my_abort, a.st, ABORT_TIMEOUT, blockIdx.x, r, PH_PING_RESP, 0xFFFFFFFFu, seen);
      else note_abort_exit(a.st, r);
      return;
    }
    st_rel_sys(a.flag_out, r);
    if (i + 1 < a.rounds && now_ticks() - t_launch > a.budget) {
      note_failure(a.my_abort, a.st, ABORT_BUDGET, blockIdx.x, r + 1u, PH_PING_BUDGET, 0xFFFFFFFFu, (u32)(i + 1));
      return;
    }
  }
}
// loopback: block 0 = initiator (writes A, waits B), block 1 = responder (waits A, writes B)
__global__ void loopback2_kernel(PingArgs init, PingArgs resp) {
  if (threadIdx.x != 0) return;
  const u64 t_launch = now_ticks();
  if (blockIdx.x == 0) {
    ping_initiator(init, t_launch);
  } else {
    __hip_atomic_store(&resp.st->resp_started, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    ping_responder(resp, t_launch);
  }
}
__global__ void ping_kernel(PingArgs a, int role) {
  if (threadIdx.x != 0) return;
  const u64 t_launch = now_ticks();
  if (role == 0) ping_initiator(a, t_launch);
  else ping_responder(a, t_launch);
}

// =================================================================================================
// Options
// =================================================================================================
struct Opt {
  std::string mode;
  int dev0 = 0, dev1 = 1, device = -1;
  std::string alloc = "hostmalloc";
  std::string json;
  std::string partner = "gpu";
  long long iters = -1;  // mode default
  std::string variant = "flag";
  int drain = 1, acq = 1;
  std::string spin = "rlx";
  long long bytes = 10240;
  int nb = 8, nt = 256;
  std::string pattern = "isolated";
  bool graph = false;
  int timeout_ms = 100;
  bool trace = false;
  int tokens = 50;
  int filler_mb = 60, dirty_mb = 4;
  long long count = 10000000;
  int rounds_per_launch = 0;  // 0 = calibrate
};

static void usage() {
  printf(
      "usage: ar_bench.exe <mode> [options] [--json <path>]\n"
      "modes:\n"
      "  probe       device attributes, peer access, both shared-host-memory allocation paths (no kernels)\n"
      "  loopback    --device D [--partner gpu|cpu] [--iters N=100000]: GPU<->host-memory RTT on one card\n"
      "  pingpong    [--iters N=100000]: dev0<->dev1 RTT through the shared host mailbox\n"
      "  allreduce   [--variant flag|ll] [--drain 1|3] [--acq 0|1] [--bytes N=10240] [--nb 8] [--nt 256]\n"
      "              [--iters 5000] [--pattern isolated|decode] [--graph] [--trace] [--timeout-ms 100]\n"
      "              [--tokens 50] [--filler-mb 60] [--dirty-mb 4]\n"
      "  stress      [--count 10000000] + the allreduce knobs (incl. --pattern decode: count in ARs,\n"
      "              rounded up to whole 128-AR tokens): verified, stops at the first failure\n"
      "  sweep       the standard decision matrix, one JSON\n"
      "common: --dev0 0 --dev1 1 --alloc hostmalloc|register --timeout-ms N(<=500) --spin rlx|acq\n"
      "        --rounds-per-launch N(0 = calibrate, <=20000)\n"
      "exit codes: 0 ok, 1 error, 2 data mismatch / protocol violation, 3 timeout/abort/stuck\n");
}

static bool parse_args(int argc, char** argv, Opt& o, std::string& err) {
  if (argc < 2) {
    err = "missing mode";
    return false;
  }
  o.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    auto to_ll = [&](const char* name) -> long long {
      const char* v = need(name);
      char* e = nullptr;
      const long long x = strtoll(v, &e, 0);
      if (!e || *e) throw std::runtime_error(std::string("bad integer for ") + name + ": " + v);
      return x;
    };
    try {
      if (a == "--dev0") o.dev0 = (int)to_ll("--dev0");
      else if (a == "--dev1") o.dev1 = (int)to_ll("--dev1");
      else if (a == "--device") o.device = (int)to_ll("--device");
      else if (a == "--alloc") o.alloc = need("--alloc");
      else if (a == "--json") o.json = need("--json");
      else if (a == "--partner") o.partner = need("--partner");
      else if (a == "--iters") o.iters = to_ll("--iters");
      else if (a == "--variant") o.variant = need("--variant");
      else if (a == "--drain") o.drain = (int)to_ll("--drain");
      else if (a == "--acq") o.acq = (int)to_ll("--acq");
      else if (a == "--spin") o.spin = need("--spin");
      else if (a == "--bytes") o.bytes = to_ll("--bytes");
      else if (a == "--nb") o.nb = (int)to_ll("--nb");
      else if (a == "--nt") o.nt = (int)to_ll("--nt");
      else if (a == "--pattern") o.pattern = need("--pattern");
      else if (a == "--graph") o.graph = true;
      else if (a == "--timeout-ms") o.timeout_ms = (int)to_ll("--timeout-ms");
      else if (a == "--trace") o.trace = true;
      else if (a == "--tokens") o.tokens = (int)to_ll("--tokens");
      else if (a == "--filler-mb") o.filler_mb = (int)to_ll("--filler-mb");
      else if (a == "--dirty-mb") o.dirty_mb = (int)to_ll("--dirty-mb");
      else if (a == "--count") o.count = to_ll("--count");
      else if (a == "--rounds-per-launch") o.rounds_per_launch = (int)to_ll("--rounds-per-launch");
      else if (a == "-h" || a == "--help") {
        err = "help";
        return false;
      } else {
        err = "unknown option: " + a;
        return false;
      }
    } catch (const std::exception& e) {
      err = e.what();
      return false;
    }
  }
  auto bad = [&](const std::string& m) {
    err = m;
    return false;
  };
  if (o.alloc != "hostmalloc" && o.alloc != "register") return bad("--alloc must be hostmalloc|register");
  if (o.partner != "gpu" && o.partner != "cpu") return bad("--partner must be gpu|cpu");
  if (o.variant != "flag" && o.variant != "ll") return bad("--variant must be flag|ll");
  if (o.drain != 1 && o.drain != 3) return bad("--drain must be 1 or 3 (0 and 2 are unsafe and not offered)");
  if (o.acq != 0 && o.acq != 1) return bad("--acq must be 0 or 1");
  if (o.spin != "rlx" && o.spin != "acq") return bad("--spin must be rlx|acq");
  if (o.bytes < 16 || o.bytes % 16 != 0 || (size_t)o.bytes > kMaxBytes)
    return bad("--bytes must be a multiple of 16 in [16, 1 MiB]");
  if (o.nb < 1 || o.nb > kNbMax) return bad("--nb must be in [1, 64]");
  if (o.nt < 32 || o.nt > 1024 || o.nt % 32 != 0) return bad("--nt must be a multiple of 32 in [32, 1024]");
  if (o.pattern != "isolated" && o.pattern != "decode") return bad("--pattern must be isolated|decode");
  if (o.timeout_ms < 1 || o.timeout_ms > kTimeoutCapMs) return bad("--timeout-ms must be in [1, 500]");
  if (o.tokens < 1 || o.tokens > 100000) return bad("--tokens must be in [1, 100000]");
  if (o.filler_mb < 0 || o.filler_mb > 1024) return bad("--filler-mb must be in [0, 1024]");
  if (o.dirty_mb < 0 || o.dirty_mb > 256) return bad("--dirty-mb must be in [0, 256]");
  if (o.count < 1 || o.count > 200000000LL) return bad("--count must be in [1, 200000000]");
  if (o.iters == 0 || o.iters < -1 || o.iters > 100000000LL) return bad("--iters must be in [1, 1e8]");
  if (o.rounds_per_launch < 0 || o.rounds_per_launch > kMaxChunk)
    return bad("--rounds-per-launch must be in [0 (calibrate), 20000]");
  if (o.dev0 == o.dev1) return bad("--dev0 and --dev1 must differ");
  return true;
}

// =================================================================================================
// Device info
// =================================================================================================
static double wallclock_khz(int dev) {
  int v = 0;
  HIPC(hipDeviceGetAttribute(&v, hipDeviceAttributeWallClockRate, dev));
  if (v <= 0) throw HipError("hipDeviceAttributeWallClockRate is 0");
  return (double)v;
}

// ---- WallClockRate validation (once per device, before the first spinning kernel) -----------------
// Every device timeout (timeout_ms x rate) and every in-kernel ns figure trusts the reported rate, while
// the counter actually read is __builtin_readsteadycounter (s_sendmsg_rtn_b64 GET_REALTIME). Run a busy
// kernel (bounded by an iteration count, ~10 ms at full clock) that reads the counter at entry and exit,
// time it on the host (steady_clock around launch + drain, minus the empty-launch overhead), and
// compare. The GPU is busy for the whole interval, which is the regime the timeouts run in.
static std::mutex g_clock_mu;
static J g_clock_json = J::arr();
static bool g_clock_done[64] = {};

static void validate_wallclock(int dev) {
  std::lock_guard<std::mutex> lk(g_clock_mu);
  if (dev < 0 || dev >= 64 || g_clock_done[dev]) return;
  HIPC(hipSetDevice(dev));
  const double reported = wallclock_khz(dev);
  hipStream_t st = nullptr;
  u64* d = nullptr;
  HIPC(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  HIPC(hipMalloc(&d, 2 * sizeof(u64)));
  auto run = [&](int iters, double* host_s) -> u64 {  // returns device ticks, *host_s = host seconds
    const auto a = std::chrono::steady_clock::now();
    (void)hipGetLastError();
    clock_probe_kernel<<<1, 32, 0, st>>>(d, iters);
    check_launch("clock_probe");
    if (!drain_stream_wd(st, kWatchdogS)) {
      g_stuck = true;
      throw HipError("stream stuck in the WallClockRate check");
    }
    *host_s = secs_since(a);
    u64 v[2] = {0, 0};
    HIPC(hipMemcpyAsync(v, d, sizeof v, hipMemcpyDeviceToHost, st));  // stream idle: cannot block on a kernel
    if (!drain_stream_wd(st, kWatchdogS)) {
      g_stuck = true;
      throw HipError("stream stuck in the WallClockRate check");
    }
    return v[1] - v[0];
  };
  double hs = 0, ovh = INFINITY;
  (void)run(0, &hs);  // warm-up (code-object load)
  for (int i = 0; i < 3; ++i) {
    (void)run(0, &hs);
    ovh = std::min(ovh, hs);
  }
  int iters = 1200;  // ~10 ms at ~1 GHz+; grown if the interval is too short to be precise
  u64 ticks = 0;
  double busy_s = 0;
  for (int attempt = 0; attempt < 4; ++attempt) {
    ticks = run(iters, &hs);
    busy_s = hs - ovh;
    if (busy_s >= 0.005 || iters >= 16000) break;  // <= 19200 iters: < 100 ms given the previous run's pace
    iters *= 4;
  }
  (void)hipFree(d);
  (void)hipStreamDestroy(st);
  const double measured = busy_s > 0 ? (double)ticks / (busy_s * 1000.0) : NAN;
  const double rel = std::fabs(measured - reported) / reported;
  J j = J::obj();
  j["device"] = dev;
  j["reported_khz"] = reported;
  j["measured_khz"] = measured;
  j["rel_diff"] = rel;
  j["busy_interval_ms_host"] = busy_s * 1000.0;
  j["busy_interval_ms_device"] = (double)ticks / reported;
  j["launch_overhead_us"] = ovh * 1e6;
  j["sleep_iters"] = iters;
  j["tolerance"] = kClockTolerance;
  const bool ok = std::isfinite(rel) && rel <= kClockTolerance;
  j["ok"] = ok;
  g_clock_json.push(j);
  g_clock_done[dev] = true;
  printf("[clock] dev%d WallClockRate reported %.0f kHz, measured %.0f kHz vs host clock (%.2f%% off, %s)\n", dev,
         reported, measured, rel * 100.0, ok ? "ok" : "MISMATCH");
  if (!ok)
    throw std::runtime_error("hipDeviceAttributeWallClockRate of device " + std::to_string(dev) +
                             " disagrees with the host clock by more than 5%: timeouts and ns figures would be wrong");
}

static J device_json(int dev, bool print) {
  J j = J::obj();
  j["hip_device"] = dev;
  hipDeviceProp_t p;
  hipError_t e = hipGetDeviceProperties(&p, dev);
  if (e != hipSuccess) {
    j["error"] = std::string("hipGetDeviceProperties: ") + hipGetErrorString(e);
    if (print) printf("  dev%d: hipGetDeviceProperties failed: %s\n", dev, hipGetErrorString(e));
    return j;
  }
  char pci[64];
  snprintf(pci, sizeof pci, "%04x:%02x:%02x", p.pciDomainID, p.pciBusID, p.pciDeviceID);
  j["name"] = p.name;
  j["gcnArchName"] = p.gcnArchName;
  j["pciDomainID"] = p.pciDomainID;
  j["pciBusID"] = p.pciBusID;
  j["pciDeviceID"] = p.pciDeviceID;
  j["pci"] = pci;
  j["totalGlobalMem"] = (unsigned long long)p.totalGlobalMem;
  j["multiProcessorCount"] = p.multiProcessorCount;
  j["l2CacheSize"] = p.l2CacheSize;
  struct A {
    const char* name;
    hipDeviceAttribute_t a;
  };
  const A attrs[] = {
      {"CanMapHostMemory", hipDeviceAttributeCanMapHostMemory},
      {"CanUseHostPointerForRegisteredMem", hipDeviceAttributeCanUseHostPointerForRegisteredMem},
      {"ConcurrentKernels", hipDeviceAttributeConcurrentKernels},
      {"ComputePreemptionSupported", hipDeviceAttributeComputePreemptionSupported},
      {"HostNativeAtomicSupported", hipDeviceAttributeHostNativeAtomicSupported},
      {"IsLargeBar", hipDeviceAttributeIsLargeBar},
      {"WallClockRate_kHz", hipDeviceAttributeWallClockRate},
      {"ClockRate_kHz", hipDeviceAttributeClockRate},
      {"KernelExecTimeout", hipDeviceAttributeKernelExecTimeout},
      {"UnifiedAddressing", hipDeviceAttributeUnifiedAddressing},
      {"PageableMemoryAccess", hipDeviceAttributePageableMemoryAccess},
  };
  J at = J::obj();
  std::string line;
  for (const A& x : attrs) {
    int v = 0;
    hipError_t ea = hipDeviceGetAttribute(&v, x.a, dev);
    if (ea == hipSuccess) {
      at[x.name] = v;
      line += std::string(" ") + x.name + "=" + std::to_string(v);
    } else {
      at[x.name] = std::string("error: ") + hipGetErrorString(ea);
      line += std::string(" ") + x.name + "=ERR";
    }
  }
  j["attributes"] = at;
  if (print) {
    printf("  dev%d: %s (%s) pci %s CUs=%d mem=%.1f GiB\n", dev, p.name, p.gcnArchName, pci, p.multiProcessorCount,
           (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("        %s\n", line.c_str());
  }
  return j;
}

// =================================================================================================
// Shared host region (the mailbox): allocation paths A (hipHostMalloc) and B (hipHostRegister)
// =================================================================================================
struct SharedRegion {
  std::string path;
  size_t bytes = 0;
  char* host = nullptr;
  bool registered = false;
  std::vector<int> devs;
  std::vector<char*> dptr;  // per devs[i]
  // First per-block seq of the next AR session on this region: above every seq an earlier session
  // used, so stale flags / LL seq halves are always "behind" even if a zeroing memset was not snooped.
  u32 next_seq_base = 0;

  SharedRegion() = default;
  SharedRegion(const SharedRegion&) = delete;
  SharedRegion& operator=(const SharedRegion&) = delete;
  ~SharedRegion() { release(); }

  void alloc(const std::string& path_, size_t bytes_, const std::vector<int>& devs_) {
    path = path_;
    bytes = round_up(bytes_, 65536);
    devs = devs_;
    HIPC(hipSetDevice(devs[0]));
    if (path == "hostmalloc") {
      void* h = nullptr;
      HIPC(hipHostMalloc(&h, bytes, hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
      host = (char*)h;
    } else {
      void* h = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!h) throw std::runtime_error("VirtualAlloc failed");
      host = (char*)h;
      hipError_t e = hipHostRegister(h, bytes, hipHostRegisterMapped | hipHostRegisterPortable);
      if (e != hipSuccess) {
        VirtualFree(h, 0, MEM_RELEASE);
        host = nullptr;
        throw HipError(hip_msg(e, "hipHostRegister(Mapped|Portable)", __FILE__, __LINE__));
      }
      registered = true;
    }
    std::memset(host, 0, bytes);
    for (int d : devs) {
      HIPC(hipSetDevice(d));
      void* p = nullptr;
      hipError_t e = hipHostGetDevicePointer(&p, host, 0);
      if (e != hipSuccess)
        throw HipError(hip_msg(e, "hipHostGetDevicePointer", __FILE__, __LINE__) + " for device " + std::to_string(d) +
                       " (alloc path " + path + "; run `probe`, try the other --alloc)");
      dptr.push_back((char*)p);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
  void release() {
    if (!host) return;
    if (g_stuck.load()) return;  // kernels may still touch it: leak
    if (registered) {
      (void)hipHostUnregister(host);
      VirtualFree(host, 0, MEM_RELEASE);
    } else {
      (void)hipHostFree(host);
    }
    host = nullptr;
    dptr.clear();
  }
  void zero() {  // only while no kernel on any device touches the region
    std::memset(host, 0, bytes);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }
  u32 load(size_t off) const { return std::atomic_ref<u32>(*(u32*)(host + off)).load(std::memory_order_acquire); }
  void store(size_t off, u32 v) { std::atomic_ref<u32>(*(u32*)(host + off)).store(v, std::memory_order_release); }
};

struct RegionPlan {
  size_t mbox_region = 0;
  size_t mbox_off[2] = {0, 0};
  size_t bytes = 0;
};
static RegionPlan plan_region(size_t mbox_bytes) {
  RegionPlan p;
  p.mbox_region = round_up(std::max<size_t>(mbox_bytes, 1), 65536);
  p.mbox_off[0] = kOffMbox0;
  p.mbox_off[1] = kOffMbox0 + p.mbox_region;
  p.bytes = kOffMbox0 + 2 * p.mbox_region;
  return p;
}

// All-reduce geometry
struct Geom {
  int n16 = 0;             // payload 16-B words per rank
  int nb = 0;              // effective blocks
  int w = 0;               // payload words per block
  u32 wire = 16;           // mailbox bytes per payload word (16 FLAG, 32 LL)
  u32 blk_stride = 0;      // bytes, multiple of 128
  u32 slot_stride = 0;     // nb * blk_stride
  size_t mbox_bytes = 0;   // 2 slots
};
static Geom make_geom(size_t bytes, int nb_req, bool ll) {
  Geom g;
  g.n16 = (int)(bytes / 16);
  const int nb = std::max(1, std::min(std::min(nb_req, kNbMax), g.n16));
  g.w = (g.n16 + nb - 1) / nb;
  g.nb = (g.n16 + g.w - 1) / g.w;
  g.wire = ll ? 32u : 16u;
  g.blk_stride = (u32)round_up((size_t)g.w * g.wire, kLine);
  g.slot_stride = (u32)g.nb * g.blk_stride;
  g.mbox_bytes = 2ull * g.slot_stride;
  return g;
}

// =================================================================================================
// Mode 1: probe (NO kernel launches)
// =================================================================================================
static J probe_path(const std::string& path, int dev0, int dev1, bool print) {
  const size_t bytes = 4u << 20;
  J j = J::obj();
  j["path"] = path;
  j["bytes"] = (unsigned long long)bytes;
  j["flags"] = path == "hostmalloc" ? "hipHostMalloc(Coherent|Mapped|Portable), dev0 current"
                                    : "VirtualAlloc + hipHostRegister(Mapped|Portable), dev0 current";
  void* host = nullptr;
  bool ok_alloc = false;
  std::string err;
  hipError_t e = hipSetDevice(dev0);
  if (e == hipSuccess) {
    if (path == "hostmalloc") {
      e = hipHostMalloc(&host, bytes, hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable);
      ok_alloc = (e == hipSuccess);
      if (!ok_alloc) err = std::string("hipHostMalloc: ") + hipGetErrorString(e);
    } else {
      host = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!host) err = "VirtualAlloc failed";
      else {
        e = hipHostRegister(host, bytes, hipHostRegisterMapped | hipHostRegisterPortable);
        ok_alloc = (e == hipSuccess);
        if (!ok_alloc) {
          err = std::string("hipHostRegister: ") + hipGetErrorString(e);
          VirtualFree(host, 0, MEM_RELEASE);
          host = nullptr;
        }
      }
    }
  } else {
    err = std::string("hipSetDevice: ") + hipGetErrorString(e);
  }
  j["alloc_ok"] = ok_alloc;
  if (!ok_alloc) j["alloc_error"] = err;
  j["host_ptr"] = hex64((u64)(uintptr_t)host);
  if (print)
    printf("  path %-10s: alloc %s host=%s%s\n", path.c_str(), ok_alloc ? "OK  " : "FAIL",
           hex64((u64)(uintptr_t)host).c_str(), ok_alloc ? "" : (" (" + err + ")").c_str());
  bool all_ok = ok_alloc;
  J per = J::arr();
  if (ok_alloc) {
    std::memset(host, 0, bytes);
    for (int d : {dev0, dev1}) {
      J pd = J::obj();
      pd["device"] = d;
      void* p = nullptr;
      hipError_t es = hipSetDevice(d);
      hipError_t eg = es == hipSuccess ? hipHostGetDevicePointer(&p, host, 0) : es;
      pd["ok"] = (eg == hipSuccess);
      if (eg != hipSuccess) pd["error"] = std::string(hipGetErrorString(eg));
      pd["device_ptr"] = hex64((u64)(uintptr_t)p);
      pd["equals_host_ptr"] = (eg == hipSuccess && p == host);
      hipPointerAttribute_t pa;
      std::memset(&pa, 0, sizeof pa);
      hipError_t ep = es == hipSuccess ? hipPointerGetAttributes(&pa, host) : es;
      if (ep == hipSuccess) {
        J pj = J::obj();
        pj["type"] = (int)pa.type;
        pj["device"] = pa.device;
        pj["devicePointer"] = hex64((u64)(uintptr_t)pa.devicePointer);
        pj["hostPointer"] = hex64((u64)(uintptr_t)pa.hostPointer);
        pj["allocationFlags"] = pa.allocationFlags;
        pd["pointer_attributes"] = pj;
      } else {
        pd["pointer_attributes_error"] = std::string(hipGetErrorString(ep));
      }
      (void)hipGetLastError();
      if (print)
        printf("      dev%d: hipHostGetDevicePointer %s dptr=%s %s\n", d, eg == hipSuccess ? "OK  " : "FAIL",
               hex64((u64)(uintptr_t)p).c_str(),
               eg == hipSuccess ? (p == host ? "(== host ptr)" : "(!= host ptr)") : hipGetErrorString(eg));
      all_ok = all_ok && (eg == hipSuccess);
      per.push(pd);
    }
    (void)hipSetDevice(dev0);
    if (path == "hostmalloc") {
      hipError_t ef = hipHostFree(host);
      j["free_ok"] = (ef == hipSuccess);
    } else {
      hipError_t eu = hipHostUnregister(host);
      j["free_ok"] = (eu == hipSuccess);
      VirtualFree(host, 0, MEM_RELEASE);
    }
  }
  j["per_device"] = per;
  j["usable_for_both_devices"] = all_ok;
  (void)hipGetLastError();
  return j;
}

static int run_probe(const Opt& o, J& out) {
  int n = 0;
  HIPC(hipGetDeviceCount(&n));
  int rt = 0, drv = 0;
  (void)hipRuntimeGetVersion(&rt);
  (void)hipDriverGetVersion(&drv);
  char hvd_buf[256] = {0};
  const DWORD hvd_len = GetEnvironmentVariableA("HIP_VISIBLE_DEVICES", hvd_buf, sizeof hvd_buf);
  const char* hvd = (hvd_len > 0 && hvd_len < sizeof hvd_buf) ? hvd_buf : nullptr;
  printf("[probe] HIP runtime %d, driver %d, %d device(s), HIP_VISIBLE_DEVICES=%s\n", rt, drv, n, hvd ? hvd : "<unset>");
  out["hip_runtime_version"] = rt;
  out["hip_driver_version"] = drv;
  out["device_count"] = n;
  out["HIP_VISIBLE_DEVICES"] = hvd ? J(hvd) : J();
  if (o.dev0 >= n || o.dev1 >= n || o.dev0 < 0 || o.dev1 < 0) {
    printf("[probe] ERROR: --dev0 %d / --dev1 %d not both present\n", o.dev0, o.dev1);
    out["error"] = "devices not present";
    return EXIT_ERR;
  }
  J devs = J::arr();
  for (int d = 0; d < n; ++d) devs.push(device_json(d, true));
  out["devices"] = devs;
  int p01 = -1, p10 = -1;
  hipError_t e1 = hipDeviceCanAccessPeer(&p01, o.dev0, o.dev1);
  hipError_t e2 = hipDeviceCanAccessPeer(&p10, o.dev1, o.dev0);
  J peer = J::obj();
  peer["dev0_can_access_dev1"] = e1 == hipSuccess ? J(p01) : J(std::string("error: ") + hipGetErrorString(e1));
  peer["dev1_can_access_dev0"] = e2 == hipSuccess ? J(p10) : J(std::string("error: ") + hipGetErrorString(e2));
  out["peer_access"] = peer;
  printf("  peer access: dev%d->dev%d = %d, dev%d->dev%d = %d\n", o.dev0, o.dev1, p01, o.dev1, o.dev0, p10);
  printf("  shared 4 MiB mailbox allocation paths:\n");
  J paths = J::obj();
  J pa = probe_path("hostmalloc", o.dev0, o.dev1, true);
  J pb = probe_path("register", o.dev0, o.dev1, true);
  const bool ok_a = pa["usable_for_both_devices"].is_true(), ok_b = pb["usable_for_both_devices"].is_true();
  paths["hostmalloc"] = pa;
  paths["register"] = pb;
  out["alloc_paths"] = paths;
  out["selected_alloc"] = o.alloc;
  const bool sel_ok = o.alloc == "hostmalloc" ? ok_a : ok_b;
  out["selected_alloc_usable"] = sel_ok;
  if (!sel_ok) {
    const char* other = o.alloc == "hostmalloc" ? "register" : "hostmalloc";
    const bool other_ok = o.alloc == "hostmalloc" ? ok_b : ok_a;
    printf("[probe] ERROR: the selected --alloc %s path is not usable on both devices%s\n", o.alloc.c_str(),
           other_ok ? (std::string("; rerun every later mode with --alloc ") + other).c_str() : "; neither path works");
    out["error"] = "selected alloc path unusable";
    return EXIT_ERR;
  }
  printf("[probe] selected --alloc %s is usable on both devices\n", o.alloc.c_str());
  return EXIT_OK;
}

// =================================================================================================
// Helpers shared by the ping modes
// =================================================================================================
static Status read_status_d2h(hipStream_t st, const Status* d) {
  Status s;
  HIPC(hipMemcpyAsync(&s, d, sizeof s, hipMemcpyDeviceToHost, st));
  if (!drain_stream_wd(st, kWatchdogS)) {
    g_stuck = true;
    throw HipError("stream stuck while reading Status");
  }
  return s;
}
static J status_json(const Status& s) {
  J j = J::obj();
  j["failed"] = s.claimed != 0;
  j["code"] = abort_code_name(s.code);
  if (s.claimed) {
    j["block"] = s.block;
    j["seq"] = s.seq;
    j["phase"] = phase_name(s.phase);
    j["word"] = s.word == 0xFFFFFFFFu ? J() : J(s.word);
    j["got"] = s.got;
  }
  j["n_timeouts"] = s.n_timeouts;
  j["n_abort_exits"] = s.n_abort_exits;
  j["verify_mismatches"] = s.mismatches;
  return j;
}
// Status in VRAM: zero, then the two "min" fields to all-ones.
static void init_status_async(Status* d, hipStream_t st) {
  HIPC(hipMemsetAsync(d, 0, sizeof(Status), st));
  HIPC(hipMemsetAsync(&d->min_key, 0xFF, sizeof(u64), st));
  HIPC(hipMemsetAsync(&d->abort_seq_min, 0xFF, sizeof(u32), st));
}
// Rounds per ping launch from the calibration launch's mean RTT: ~kChunkTargetMs per launch, clamped
// to [kMinChunk, kMaxChunk]. The in-kernel budget (kPingBudgetMs) bounds a launch whatever this says.
static int chunk_from_rtt(double mean_ticks, double khz, int override_rounds) {
  if (override_rounds > 0) return override_rounds;
  const double ms_per_round = mean_ticks / khz;
  int c = ms_per_round > 0 ? (int)(kChunkTargetMs / ms_per_round) : kMaxChunk;
  return std::max(kMinChunk, std::min(kMaxChunk, c));
}
static const char* ping_fail_hint(u32 code) {
  return code == ABORT_BUDGET
             ? "a ping launch exceeded its in-kernel budget (RTT far slower than calibrated, or the GPU was preempted)"
             : "";
}

// =================================================================================================
// Mode 2: loopback (one device; partner = second block of the same kernel, or a CPU thread)
// =================================================================================================
static int run_loopback(const Opt& o, int dev, const std::string& partner, J& out) {
  const long long iters = o.iters > 0 ? o.iters : 100000;
  out["device"] = dev;
  out["partner"] = partner;
  out["iters"] = iters;
  out["alloc"] = o.alloc;
  out["timeout_ms"] = o.timeout_ms;
  out["spin"] = o.spin;
  out["launch_budget_ms"] = kPingBudgetMs;
  printf("[loopback] dev%d partner=%s iters=%lld alloc=%s spin=%s\n", dev, partner.c_str(), iters, o.alloc.c_str(),
         o.spin.c_str());

  validate_wallclock(dev);  // before the first spinning kernel on this device
  SharedRegion sh;
  sh.alloc(o.alloc, plan_region(0).bytes, {dev});
  HIPC(hipSetDevice(dev));
  hipStream_t st = nullptr;
  HIPC(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  hipEvent_t ev = nullptr;
  HIPC(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  const double khz = wallclock_khz(dev);
  out["wallclock_khz"] = khz;
  u32* d_rtt = nullptr;
  Status* d_st = nullptr;
  HIPC(hipMalloc(&d_rtt, sizeof(u32) * kMaxChunk));
  HIPC(hipMalloc(&d_st, sizeof(Status)));
  init_status_async(d_st, st);
  HIPC(hipMemsetAsync(d_rtt, 0, sizeof(u32) * kMaxChunk, st));

  char* base = sh.dptr[0];
  const size_t offA = kOffFlags[1], offB = kOffFlags[0];  // A written by the initiator, B by the partner
  PingArgs pi{};
  pi.flag_out = (u32*)(base + offA);
  pi.flag_in = (const u32*)(base + offB);
  pi.my_abort = (u32*)(base + kOffAbort[0]);
  pi.peer_abort = (const u32*)(base + kOffAbort[1]);
  pi.rtt = d_rtt;
  pi.st = d_st;
  pi.timeout = (u64)o.timeout_ms * (u64)khz;
  pi.budget = (u64)kPingBudgetMs * (u64)khz;
  pi.spin_acq = o.spin == "acq" ? 1 : 0;
  PingArgs pr = pi;
  pr.flag_out = (u32*)(base + offB);
  pr.flag_in = (const u32*)(base + offA);
  pr.rtt = nullptr;
  pr.budget = (u64)(kPingBudgetMs + kPingRespExtraMs) * (u64)khz;

  // CPU partner: poll A with std::atomic, echo into B
  const long long total_rounds = kCalibRounds + iters;
  std::atomic<bool> cpu_stop{false};
  std::atomic<int> cpu_state{0};  // 0 running/done, 1 timeout, 2 abort seen
  std::thread cpu_thr;
  if (partner == "cpu") {
    cpu_thr = std::thread([&] {
      const double cpu_limit_s = 5.0;  // covers host relaunch gaps between GPU launches
      for (long long i = 1; i <= total_rounds; ++i) {
        const u32 r = (u32)i;
        const auto t0 = std::chrono::steady_clock::now();
        for (u32 spins = 0;; ++spins) {
          const u32 a = sh.load(offA);
          if ((int)(a - r) >= 0) break;
          if ((spins & 1023u) == 0u) {
            if (cpu_stop.load() || sh.load(kOffAbort[0]) != 0u) {
              cpu_state = 2;
              return;
            }
            if (secs_since(t0) > cpu_limit_s) {
              cpu_state = 1;
              return;
            }
          }
          YieldProcessor();
        }
        sh.store(offB, r);
      }
    });
  }

  int code = EXIT_OK;
  std::vector<double> rtt_ns;
  J launches = J::arr();
  u32 base_round = 0;
  long long remaining = iters;
  int chunk = 0;
  int n_launch_edges = 0;
  bool calib = true;
  std::vector<u32> h_rtt(kMaxChunk);
  try {
    noop_kernel<<<1, 64, 0, st>>>();
    check_launch("noop");
    if (!drain_stream_wd(st, kWatchdogS)) {
      g_stuck = true;
      throw HipError("stream stuck on warm-up");
    }
    while (true) {
      const int n = calib ? kCalibRounds : (int)std::min<long long>(chunk, remaining);
      pi.base = pr.base = base_round;
      pi.rounds = pr.rounds = n;
      (void)hipGetLastError();
      if (partner == "gpu") loopback2_kernel<<<2, 64, 0, st>>>(pi, pr);
      else ping_kernel<<<1, 64, 0, st>>>(pi, 0);
      check_launch("loopback");
      HIPC(hipEventRecord(ev, st));
      if (!wait_event_wd(ev, kWatchdogS)) {
        g_stuck = true;
        code = EXIT_TIMEOUT;
        out["stuck"] = true;
        printf("[loopback] ERROR: stream stuck > %.0f s (watchdog)\n", kWatchdogS);
        break;
      }
      const Status s = read_status_d2h(st, d_st);
      HIPC(hipMemcpyAsync(h_rtt.data(), d_rtt, sizeof(u32) * n, hipMemcpyDeviceToHost, st));
      if (!drain_stream_wd(st, kWatchdogS)) {
        g_stuck = true;
        throw HipError("stream stuck on rtt readback");
      }
      const u32 ab0 = sh.load(kOffAbort[0]);
      if (ab0 != 0 || s.claimed) {
        code = EXIT_TIMEOUT;
        J fj = status_json(s);
        if (partner == "gpu") fj["responder_block_started"] = s.resp_started != 0;
        if (s.code == ABORT_BUDGET) fj["hint"] = ping_fail_hint(s.code);
        out["failure"] = fj;
        printf("[loopback] ABORT (%s) at round %u phase %s; partner state %d%s%s\n", abort_code_name(s.code), s.seq,
               phase_name(s.phase), cpu_state.load(),
               partner == "gpu" ? (s.resp_started ? "; responder block started" : "; responder block NEVER started")
                                : "",
               s.code == ABORT_BUDGET ? " -- launch budget exceeded" : "");
        break;
      }
      double sum = 0;
      for (int i = 0; i < n; ++i) sum += h_rtt[i];
      J lj = J::obj();
      lj["rounds"] = n;
      lj["calibration"] = calib;
      lj["mean_ns"] = sum / n * 1e6 / khz;
      launches.push(lj);
      if (calib) {
        chunk = chunk_from_rtt(sum / n, khz, o.rounds_per_launch);
        calib = false;
      } else {
        for (int i = 1; i < n; ++i) rtt_ns.push_back((double)h_rtt[i] * 1e6 / khz);  // skip launch-edge round
        ++n_launch_edges;
        remaining -= n;
      }
      base_round += (u32)n;
      if (remaining <= 0) break;
    }
  } catch (const std::exception& e) {
    printf("[loopback] ERROR: %s\n", e.what());
    out["error"] = e.what();
    code = worse_code(code, EXIT_ERR);
  }
  cpu_stop = true;
  if (cpu_thr.joinable()) cpu_thr.join();
  out["rounds_per_launch"] = chunk;
  out["launches"] = launches;
  out["partner_cpu_state"] = partner == "cpu" ? J(cpu_state.load()) : J();
  const Summ s = summarize(rtt_ns);
  out["rtt_ns"] = summ_json(s);
  out["excluded_launch_edge_rounds"] = n_launch_edges;
  out["note"] = "calibration launch and the first round of every launch are excluded from rtt_ns";
  out["rtt_ns_samples"] = vec_json(rtt_ns);
  out["status"] = code == EXIT_OK ? "ok" : code == EXIT_TIMEOUT ? "timeout" : "error";
  if (code == EXIT_OK)
    printf("[loopback] dev%d partner=%s: RTT p50 %.0f ns  p99 %.0f ns  mean %.0f ns  (n=%zu, %d rounds/launch)\n", dev,
           partner.c_str(), s.p50, s.p99, s.mean, s.n, chunk);
  if (!g_stuck) {
    (void)hipFree(d_rtt);
    (void)hipFree(d_st);
    (void)hipEventDestroy(ev);
    (void)hipStreamDestroy(st);
  }
  return code;
}

// =================================================================================================
// Mode 3: pingpong (dev0 initiator <-> dev1 responder through the shared host mailbox)
// =================================================================================================
struct PingRank {
  std::string error;
  bool stuck = false;
  bool completed = false;  // all requested rounds measured
  Status st{};
  std::vector<double> rtt_ns;
  int launches = 0;
  int chunk = 0;
  double khz = 0;
};

static void ping_rank_thread(const Opt& o, SharedRegion& sh, HostSync& sync, int r, long long iters, PingRank& R) {
  const int dev = r == 0 ? o.dev0 : o.dev1;
  hipStream_t st = nullptr;
  hipEvent_t ev = nullptr;
  u32* d_rtt = nullptr;
  Status* d_st = nullptr;
  auto barrier_failed = [&]() {  // a vote returned -1: record a barrier timeout as an error
    if (sync.timed_out() && R.error.empty()) R.error = "host barrier timeout (> 120 s): the peer thread stalled";
  };
  auto run = [&]() {  // returns early on stuck / barrier failure; cleanup below always runs
    HIPC(hipSetDevice(dev));
    HIPC(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    HIPC(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    R.khz = wallclock_khz(dev);
    HIPC(hipMalloc(&d_rtt, sizeof(u32) * kMaxChunk));
    HIPC(hipMalloc(&d_st, sizeof(Status)));
    init_status_async(d_st, st);
    char* base = sh.dptr[r];
    PingArgs a{};
    // rank 0 writes rank 1's flag line 0; rank 1 writes rank 0's flag line 0
    a.flag_out = (u32*)(base + kOffFlags[1 - r]);
    a.flag_in = (const u32*)(base + kOffFlags[r]);
    a.my_abort = (u32*)(base + kOffAbort[r]);
    a.peer_abort = (const u32*)(base + kOffAbort[1 - r]);
    a.rtt = r == 0 ? d_rtt : nullptr;
    a.st = d_st;
    a.timeout = (u64)o.timeout_ms * (u64)R.khz;
    a.budget = (u64)(r == 0 ? kPingBudgetMs : kPingBudgetMs + kPingRespExtraMs) * (u64)R.khz;
    a.spin_acq = o.spin == "acq" ? 1 : 0;
    noop_kernel<<<1, 64, 0, st>>>();
    check_launch("noop");
    if (!drain_stream_wd(st, kWatchdogS)) {
      R.stuck = true;
      g_stuck = true;
      sync.brk();
      return;
    }
    if (sync.vote(true, kBarrierS) != 1) {
      barrier_failed();
      return;
    }
    std::vector<u32> h_rtt(kMaxChunk);
    u32 base_round = 0;
    long long remaining = iters;
    bool calib = true;
    int chunk = kCalibRounds;
    while (true) {
      const int n = calib ? kCalibRounds : (int)std::min<long long>(chunk, remaining);
      a.base = base_round;
      a.rounds = n;
      (void)hipGetLastError();
      ping_kernel<<<1, 64, 0, st>>>(a, r);
      check_launch("ping");
      HIPC(hipEventRecord(ev, st));
      if (!wait_event_wd(ev, kWatchdogS)) {
        R.stuck = true;
        g_stuck = true;
        sync.brk();
        return;
      }
      R.st = read_status_d2h(st, d_st);
      ++R.launches;
      double sum = 0;
      if (r == 0) {
        HIPC(hipMemcpyAsync(h_rtt.data(), d_rtt, sizeof(u32) * n, hipMemcpyDeviceToHost, st));
        if (!drain_stream_wd(st, kWatchdogS)) {
          R.stuck = true;
          g_stuck = true;
          sync.brk();
          return;
        }
        for (int i = 0; i < n; ++i) sum += h_rtt[i];
      }
      const bool ok = R.st.claimed == 0 && sh.load(kOffAbort[0]) == 0 && sh.load(kOffAbort[1]) == 0;
      if (calib && r == 0) sync.value = chunk_from_rtt(sum / n, R.khz, o.rounds_per_launch);
      const int v = sync.vote(ok, kBarrierS);
      if (v != 1) {
        if (v < 0) barrier_failed();
        return;
      }
      if (calib) {
        chunk = (int)sync.value.load();
        R.chunk = chunk;
        calib = false;
      } else {
        if (r == 0)
          for (int i = 1; i < n; ++i) R.rtt_ns.push_back((double)h_rtt[i] * 1e6 / R.khz);  // skip launch edge
        remaining -= n;
      }
      base_round += (u32)n;
      if (remaining <= 0) {
        R.completed = true;
        return;
      }
    }
  };
  try {
    run();
  } catch (const std::exception& e) {
    R.error = e.what();
    sh.store(kOffAbort[r], ABORT_HOST);
    sync.brk();
  }
  if (st && !g_stuck) {
    bool idle = false;
    try {
      idle = drain_stream_wd(st, kWatchdogS);  // throws on a sticky device error (e.g. after a TDR)
    } catch (...) {
      idle = false;
    }
    if (idle) {
      if (d_rtt) (void)hipFree(d_rtt);
      if (d_st) (void)hipFree(d_st);
      if (ev) (void)hipEventDestroy(ev);
      (void)hipStreamDestroy(st);
    } else {
      R.stuck = true;
      g_stuck = true;
    }
  }
}

static int run_pingpong(const Opt& o, J& out) {
  const long long iters = o.iters > 0 ? o.iters : 100000;
  out["dev0"] = o.dev0;
  out["dev1"] = o.dev1;
  out["iters"] = iters;
  out["alloc"] = o.alloc;
  out["timeout_ms"] = o.timeout_ms;
  out["spin"] = o.spin;
  out["launch_budget_ms"] = kPingBudgetMs;
  printf("[pingpong] dev%d <-> dev%d iters=%lld alloc=%s spin=%s\n", o.dev0, o.dev1, iters, o.alloc.c_str(),
         o.spin.c_str());
  validate_wallclock(o.dev0);
  validate_wallclock(o.dev1);
  SharedRegion sh;
  sh.alloc(o.alloc, plan_region(0).bytes, {o.dev0, o.dev1});
  HostSync sync(2);
  PingRank R[2];
  std::thread t0(ping_rank_thread, std::cref(o), std::ref(sh), std::ref(sync), 0, iters, std::ref(R[0]));
  std::thread t1(ping_rank_thread, std::cref(o), std::ref(sh), std::ref(sync), 1, iters, std::ref(R[1]));
  t0.join();
  t1.join();
  int code = EXIT_OK;
  J ranks = J::arr();
  for (int r = 0; r < 2; ++r) {
    J rj = J::obj();
    rj["rank"] = r;
    rj["device"] = r == 0 ? o.dev0 : o.dev1;
    rj["role"] = r == 0 ? "initiator" : "responder";
    rj["launches"] = R[r].launches;
    rj["status"] = status_json(R[r].st);
    if (!R[r].error.empty()) {
      rj["error"] = R[r].error;
      code = worse_code(code, EXIT_ERR);
      printf("[pingpong] rank %d ERROR: %s\n", r, R[r].error.c_str());
    }
    if (R[r].stuck) {
      rj["stuck"] = true;
      code = worse_code(code, EXIT_TIMEOUT);
      printf("[pingpong] rank %d: stream stuck (watchdog)\n", r);
    }
    if (R[r].st.claimed) {
      code = worse_code(code, EXIT_TIMEOUT);
      if (R[r].st.code == ABORT_BUDGET) rj["hint"] = ping_fail_hint(R[r].st.code);
      printf("[pingpong] rank %d ABORT (%s) at round %u phase %s\n", r, abort_code_name(R[r].st.code), R[r].st.seq,
             phase_name(R[r].st.phase));
    }
    rj["completed"] = R[r].completed;
    ranks.push(rj);
  }
  const u32 ab0 = sh.load(kOffAbort[0]), ab1 = sh.load(kOffAbort[1]);
  if (ab0 || ab1) code = worse_code(code, EXIT_TIMEOUT);
  if (code == EXIT_OK && (!R[0].completed || !R[1].completed)) {  // stopped early with no recorded cause
    code = EXIT_ERR;
    out["error"] = "pingpong stopped before all rounds were measured, with no recorded cause";
    printf("[pingpong] ERROR: stopped before all rounds were measured (no recorded cause)\n");
  }
  out["abort_words"] = J::arr();
  out["abort_words"].push(ab0);
  out["abort_words"].push(ab1);
  out["ranks"] = ranks;
  out["rounds_per_launch"] = R[0].chunk;
  out["wallclock_khz"] = R[0].khz;
  const Summ s = summarize(R[0].rtt_ns);
  out["rtt_ns"] = summ_json(s);
  std::vector<double> one_way;
  for (double x : R[0].rtt_ns) one_way.push_back(x / 2);
  out["one_way_ns"] = summ_json(summarize(one_way));
  out["note"] = "calibration launch and the first round of every launch are excluded (launch-edge skew)";
  out["rtt_ns_samples"] = vec_json(R[0].rtt_ns);
  out["status"] = code == EXIT_OK ? "ok" : code == EXIT_TIMEOUT ? "timeout" : "error";
  if (code == EXIT_OK)
    printf("[pingpong] RTT p50 %.0f ns  p99 %.0f ns  mean %.0f ns -> one-way p50 %.0f ns  (n=%zu, %d rounds/launch)\n",
           s.p50, s.p99, s.mean, s.p50 / 2, s.n, R[0].chunk);
  return code;
}

// =================================================================================================
// Modes 4/5: allreduce (isolated | decode) and stress -- one "session" = one configuration
// =================================================================================================
struct ArCfg {
  std::string variant = "flag";
  int drain = 1, acq = 1;
  int spin_acq = 0;
  size_t bytes = 10240;
  int nb = 8, nt = 256;
  long long iters = 5000;
  std::string pattern = "isolated";
  bool graph = false;
  int timeout_ms = 100;
  bool trace = false;
  int tokens = 50;
  int filler_mb = 60, dirty_mb = 4;
  bool stress = false;
  long long count = 10000000;
};
static std::string cfg_label(const ArCfg& c) {
  char b[192];
  if (c.variant == "ll")
    snprintf(b, sizeof b, "ll bytes=%zu nb=%d nt=%d%s", c.bytes, c.nb, c.nt, c.graph ? " graph" : "");
  else
    snprintf(b, sizeof b, "flag(d%d,a%d%s) bytes=%zu nb=%d nt=%d%s", c.drain, c.acq, c.spin_acq ? ",spin=acq" : "",
             c.bytes, c.nb, c.nt, c.graph ? " graph" : "");
  return b;
}

enum PlanStatus { PS_OK = 0, PS_STOPPED = 1, PS_STUCK = 2, PS_BROKEN = 3 };
static const char* plan_status_name(int ps) {
  switch (ps) {
    case PS_OK: return "ok";
    case PS_STOPPED: return "stopped (a rank's check failed)";
    case PS_STUCK: return "stuck stream";
    case PS_BROKEN: return "host barrier broken";
    default: return "?";
  }
}

// Decode conditions, interleaved token by token inside ONE timed plan: (a) the real all-reduce,
// (b) a local stand-in kernel on the same grid in the AR slot, (c) fillers only (no AR-slot kernel).
enum { COND_AR = 0, COND_STANDIN = 1, COND_FILLERS = 2, kConds = 3 };

struct ArSession {
  ArCfg cfg;
  Geom g;
  SharedRegion* sh = nullptr;
  size_t mbox_off[2] = {0, 0};
  int devs[2] = {0, 1};
  u32 seq_base = 0;  // first per-block seq of this session is seq_base + 1
  HostSync sync{2};
};

struct RankRes {
  int rank = 0, dev = 0;
  std::string error;
  bool stuck = false;
  int plan_status = PS_OK;
  std::string stopped_in;          // "barrier" | "warmup" | "timed"
  Status mirror{};
  u32 abort_seen[2] = {0, 0};
  double khz = 0;
  u32 seq_base = 0;
  std::vector<double> ms[kConds];  // timed batch ms per condition (isolated/stress: ms[0] only)
  long long calls_before_timed = 0, calls_total = 0;
  long long batches_ok = 0;        // checked batches of the timed plan, incl. the discarded warm-up ones
  long long batches_planned = 0;
  int discard = 0;
  int K = 0;
  int fgrid = 0;
  long long starved = 0;           // timed batches enqueued after the GPU had already drained the previous one
  double seconds = 0;
  std::vector<u64> trace;
  long long trace_call_lo = 0, trace_call_hi = 0;  // [lo, hi) call indices with a trace record
  u32 trace_cap = 0;
};

// One rank's execution environment for the batch runner.
struct Runner {
  ArSession* S = nullptr;
  int r = 0;
  hipStream_t st = nullptr;
  hipEvent_t ev_s[kEvSlots] = {}, ev_e[kEvSlots] = {}, ev_d[kEvSlots] = {};
  u16* in_ring = nullptr;
  u16* out_ring = nullptr;
  u32 n_elem = 0;
  u32 aux_x = 1;              // gen/verify grid.x
  Status* stv = nullptr;
  u32* mirror_dev = nullptr;
  const volatile u32* mirror_host = nullptr;
  long long calls = 0;        // data call index of the next all-reduce == its seq - 1 - seq_base
  Status last{};
  u32 last_abort[2] = {0, 0};
  long long batches_ok = 0;
  long long starved = 0;
  bool barrier_timeout = false;

  Status read_mirror() const {
    std::atomic_thread_fence(std::memory_order_acquire);
    Status s;
    u32* d = (u32*)&s;
    for (int i = 0; i < 16; ++i) d[i] = mirror_host[i];
    return s;
  }
  bool check() {
    last = read_mirror();
    last_abort[0] = S->sh->load(kOffAbort[0]);
    last_abort[1] = S->sh->load(kOffAbort[1]);
    return last_abort[0] == 0 && last_abort[1] == 0 && last.mismatches == 0 && last.claimed == 0;
  }
};

struct Plan {
  long long nbatches = 0;
  int K = 1;                     // all-reduces per AR batch (ring rows; gen/verify grid.y)
  std::vector<std::function<void()>> bodies;  // batch n enqueues bodies[n % bodies.size()]
  int ar_body = 0;               // the body holding the K all-reduces (verified; advances the call index)
  bool gen_all = false;          // gen before EVERY batch (identical pre-token work in every condition)
  bool graph = false;            // replay prepared graphs instead of enqueueing the bodies
  int discard = 0;               // leading batches that are not timed
  std::function<void(long long)> on_batch;  // after each checked batch -- AFTER the next enqueue
};

// Number of AR-body batches among batches [0, n) of a plan with C bodies.
static long long ar_batches_in(long long n, int C, int ar_body) {
  if (ar_body < 0 || n <= ar_body) return 0;
  return (n - ar_body - 1) / C + 1;
}

struct Prepared {
  std::vector<hipGraph_t> graphs;
  std::vector<hipGraphExec_t> execs;
};
// Graph mode: capture, instantiate and UPLOAD every body before the start barrier, so neither rank's
// capture/instantiate/upload time can land inside the other rank's first all-reduce spin.
static void prepare_plan(Runner& E, const Plan& P, Prepared& G) {
  if (!P.graph) return;
  for (const auto& body : P.bodies) {
    hipGraph_t g = nullptr;
    hipGraphExec_t x = nullptr;
    (void)hipGetLastError();
    HIPC(hipStreamBeginCapture(E.st, hipStreamCaptureModeThreadLocal));
    body();
    const hipError_t le = hipGetLastError();
    HIPC(hipStreamEndCapture(E.st, &g));
    G.graphs.push_back(g);
    if (le != hipSuccess && le != hipErrorNotReady)
      throw HipError(std::string("launch during capture -> ") + hipGetErrorString(le));
    HIPC(hipGraphInstantiate(&x, g, nullptr, nullptr, 0));
    G.execs.push_back(x);
    HIPC(hipGraphUpload(x, E.st));
  }
  if (!drain_stream_wd(E.st, kWatchdogS)) {
    g_stuck = true;
    throw HipError("stream stuck while uploading graphs");
  }
}
static void destroy_prepared(Prepared& G) {
  for (hipGraphExec_t x : G.execs) (void)hipGraphExecDestroy(x);
  for (hipGraph_t g : G.graphs) (void)hipGraphDestroy(g);
  G.execs.clear();
  G.graphs.clear();
}

// Pipelined batch runner: kInflight batches are enqueued ahead; after each batch completes (watchdog
// wait) the host reads its hipEvent time, checks both abort words + this rank's Status mirror, and
// votes with the peer thread; both stop at the same batch, and every batch enqueued by one rank was
// enqueued by the other, so in-flight batches always pair up and drain. The next batch is enqueued
// right after the vote and BEFORE any host-side work (progress printing), so a stalled host thread
// idles both GPUs instead of leaving the peer with an unpaired batch.
// ms: one vector per body (may be null).
static int run_plan(Runner& E, const Plan& P, const Prepared& G, std::vector<double>* ms) {
  const int C = (int)P.bodies.size();
  const long long call0 = E.calls;
  const dim3 aux_grid(E.aux_x, (u32)P.K);
  auto enqueue = [&](long long n) {
    const int s = (int)(n % kEvSlots);
    const int bi = (int)(n % C);
    const bool is_ar = bi == P.ar_body;
    const u64 base = (u64)(call0 + ar_batches_in(n, C, P.ar_body) * P.K);
    // host-starved: the GPU already finished the previous batch before this one was enqueued
    if (n > P.discard && n >= kInflight && hipEventQuery(E.ev_d[(n - 1) % kEvSlots]) == hipSuccess) ++E.starved;
    (void)hipGetLastError();
    if (is_ar || P.gen_all) gen_kernel<<<aux_grid, 256, 0, E.st>>>(E.in_ring, base, E.n_elem, E.r);
    HIPC(hipEventRecord(E.ev_s[s], E.st));
    if (P.graph) HIPC(hipGraphLaunch(G.execs[bi], E.st));
    else P.bodies[bi]();
    HIPC(hipEventRecord(E.ev_e[s], E.st));
    if (is_ar) verify_kernel<<<aux_grid, 256, 0, E.st>>>(E.out_ring, base, E.n_elem, E.stv);
    publish_kernel<<<1, 64, 0, E.st>>>((const u32*)E.stv, E.mirror_dev, 16);
    HIPC(hipEventRecord(E.ev_d[s], E.st));
    check_launch("batch");
  };
  int ps = PS_OK;
  long long enq = 0, n = 0;
  for (; enq < std::min<long long>(kInflight, P.nbatches); ++enq) enqueue(enq);
  for (n = 0; n < P.nbatches; ++n) {
    const int s = (int)(n % kEvSlots);
    if (!wait_event_wd(E.ev_d[s], kWatchdogS)) {
      ps = PS_STUCK;
      g_stuck = true;
      E.S->sync.brk();
      break;
    }
    if (ms && n >= P.discard) {
      float t = 0;
      HIPC(hipEventElapsedTime(&t, E.ev_s[s], E.ev_e[s]));
      ms[n % C].push_back(t);
    }
    const bool ok = E.check();
    const int v = E.S->sync.vote(ok, kBarrierS);
    if (v != 1) {
      ps = v == 0 ? PS_STOPPED : PS_BROKEN;
      if (v < 0 && E.S->sync.timed_out()) E.barrier_timeout = true;
      break;
    }
    ++E.batches_ok;
    if (enq < P.nbatches) {  // enqueue FIRST (both ranks always have the same work queued) ...
      enqueue(enq);
      ++enq;
    }
    if (P.on_batch) P.on_batch(n + 1);  // ... then any host-side work that might block
  }
  if (ps != PS_STUCK) {
    for (long long m = n + 1; m < enq; ++m) {
      if (!wait_event_wd(E.ev_d[m % kEvSlots], kWatchdogS)) {
        ps = PS_STUCK;
        g_stuck = true;
        E.S->sync.brk();
        break;
      }
    }
    if (ps != PS_STUCK) E.check();  // final state (sticky aborts from the drained batches)
  }
  E.calls = call0 + ar_batches_in(enq, C, P.ar_body) * P.K;
  return ps;
}

static void ar_rank(ArSession& S, int r, RankRes& R) {
  const ArCfg& c = S.cfg;
  const Geom& g = S.g;
  const int dev = S.devs[r];
  R.rank = r;
  R.dev = dev;
  R.seq_base = S.seq_base;
  hipStream_t st = nullptr;
  std::vector<void*> allocs;
  std::vector<hipEvent_t> events;
  Prepared G;
  auto dmalloc = [&](size_t n) -> void* {
    void* p = nullptr;
    HIPC(hipMalloc(&p, n));
    allocs.push_back(p);
    return p;
  };
  Runner E;
  try {
    HIPC(hipSetDevice(dev));  // FIRST
    HIPC(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    for (int i = 0; i < kEvSlots; ++i) {
      HIPC(hipEventCreate(&E.ev_s[i]));
      events.push_back(E.ev_s[i]);
      HIPC(hipEventCreate(&E.ev_e[i]));
      events.push_back(E.ev_e[i]);
      HIPC(hipEventCreateWithFlags(&E.ev_d[i], hipEventDisableTiming));
      events.push_back(E.ev_d[i]);
    }
    R.khz = wallclock_khz(dev);
    int cus = 0;
    HIPC(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));

    const bool decode = (c.pattern == "decode");
    const bool ll = (c.variant == "ll");
    const int K = decode ? kDecodeK : kIsoK;
    R.K = K;
    const u32 n_elem = (u32)(c.bytes / 2);
    // plan shape: C interleaved bodies (decode timing: 3 conditions), discarded leading batches
    int C = 1, discard = 0;
    long long plan_batches = 0;
    if (c.stress) {
      plan_batches = (c.count + K - 1) / K;  // every batch verified, none discarded
    } else if (decode) {
      C = kConds;
      discard = kConds;  // one warm-up token per condition
      plan_batches = (long long)kConds * (c.tokens + 1);
    } else {
      discard = kIsoDiscard;
      plan_batches = (c.iters + K - 1) / K + discard;
    }
    R.discard = discard;
    R.batches_planned = plan_batches;
    const long long ar_batches = ar_batches_in(plan_batches, C, 0);

    // ---- VRAM (allocated before any timing) ----
    u16* in_ring = (u16*)dmalloc((size_t)K * c.bytes);
    u16* out_ring = (u16*)dmalloc((size_t)K * c.bytes);
    uint4* local = (uint4*)dmalloc(c.bytes);
    u32* seq = (u32*)dmalloc(sizeof(u32) * kNbMax);
    Status* stv = (Status*)dmalloc(sizeof(Status));
    u64* trace = nullptr;
    u32 trace_cap = 0;
    if (c.trace) {
      trace_cap = (u32)(2 + ar_batches * K + 64);
      trace = (u64*)dmalloc((size_t)trace_cap * 4 * sizeof(u64));
      HIPC(hipMemsetAsync(trace, 0, (size_t)trace_cap * 4 * sizeof(u64), st));
    }
    // per-block seq starts at the session base (above every seq an earlier session used)
    HIPC(hipMemsetD32Async((hipDeviceptr_t)seq, (int)S.seq_base, kNbMax, st));
    init_status_async(stv, st);
    HIPC(hipMemsetAsync(in_ring, 0, (size_t)K * c.bytes, st));
    HIPC(hipMemsetAsync(out_ring, 0, (size_t)K * c.bytes, st));

    // decode filler: a VRAM ring larger than 256 MiB (Infinity Cache is 64 MiB) filled with a hash
    // pattern + a dirty "state" buffer. One wave of resident blocks (occupancy-sized grid), so the
    // --dirty-mb RMW happens at the end of the filler and those lines are still dirty at AR time.
    uint4* fring = nullptr;
    uint4* fdirty = nullptr;
    u32* fsink = nullptr;
    u64 rd_words = 0, dirty_words = 0;
    std::vector<u64> foff(kDecodeK, 0);
    int occ = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, filler_kernel, 256, 0) != hipSuccess || occ < 1) occ = 4;
    (void)hipGetLastError();
    const int fgrid = std::max(1, cus) * occ;
    R.fgrid = fgrid;
    if (decode) {
      const u64 rd_bytes = (u64)c.filler_mb << 20;
      const u64 ring_bytes = round_up(std::max<u64>(512ull << 20, 2 * rd_bytes), 4096);
      rd_words = rd_bytes / 16;
      dirty_words = ((u64)c.dirty_mb << 20) / 16;
      fring = (uint4*)dmalloc(ring_bytes);
      fdirty = (uint4*)dmalloc(std::max<u64>(dirty_words, 1) * 16);
      fsink = (u32*)dmalloc(256 * sizeof(u32));
      fill_hash_kernel<<<std::max(1, cus) * 4, 256, 0, st>>>(fring, ring_bytes / 16, 0x5A5A0000u + (u32)r);
      HIPC(hipMemsetAsync(fdirty, 0, std::max<u64>(dirty_words, 1) * 16, st));
      const u64 span = ring_bytes - rd_bytes;
      for (int p = 0; p < kDecodeK; ++p)
        foff[p] = span ? (((u64)p * rd_bytes) % span) / 4096 * 4096 / 16 : 0;  // in 16-B words
    }

    // ---- runner / args ----
    char* base = S.sh->dptr[r];
    E.S = &S;
    E.r = r;
    E.st = st;
    E.in_ring = in_ring;
    E.out_ring = out_ring;
    E.n_elem = n_elem;
    E.aux_x = (u32)std::max<u32>(1, std::min<u32>(64, (n_elem + 255) / 256));
    E.stv = stv;
    E.mirror_dev = (u32*)(base + kOffMirror[r]);
    E.mirror_host = (const volatile u32*)(S.sh->host + kOffMirror[r]);

    ArArgs A{};
    A.peer_mbox = base + S.mbox_off[1 - r];
    A.my_mbox = base + S.mbox_off[r];
    A.peer_flags = (u32*)(base + kOffFlags[1 - r]);
    A.my_flags = (const u32*)(base + kOffFlags[r]);
    A.my_abort = (u32*)(base + kOffAbort[r]);
    A.peer_abort = (const u32*)(base + kOffAbort[1 - r]);
    A.seq = seq;
    A.st = stv;
    A.trace = trace;
    A.trace_cap = trace_cap ? trace_cap : 1;
    A.n16 = g.n16;
    A.w = g.w;
    A.blk_stride = g.blk_stride;
    A.slot_stride = g.slot_stride;
    A.drain = c.drain;
    A.acq = c.acq;
    A.spin_acq = c.spin_acq;
    A.seq_base = S.seq_base;
    A.timeout = (u64)c.timeout_ms * (u64)R.khz;

    auto launch_ar = [&](int pos) {
      ArArgs a = A;
      a.in = (const uint4*)(in_ring + (size_t)pos * n_elem);
      a.out = (uint4*)(out_ring + (size_t)pos * n_elem);
      if (ll) ar_ll_kernel<<<g.nb, c.nt, 0, st>>>(a);
      else ar_flag_kernel<<<g.nb, c.nt, 0, st>>>(a);
    };
    auto launch_local = [&](int pos) {
      local_standin_kernel<<<g.nb, c.nt, 0, st>>>((const uint4*)(in_ring + (size_t)pos * n_elem), local,
                                                   (uint4*)(out_ring + (size_t)pos * n_elem), g.n16, g.w);
    };
    auto launch_filler = [&](int p) {
      filler_kernel<<<fgrid, 256, 0, st>>>(fring + foff[p], rd_words, fdirty, dirty_words, fsink);
    };
    auto token = [&](int cond) {  // 64 layers x [filler, slot, filler, slot]
      for (int l = 0; l < kLayers; ++l)
        for (int h = 0; h < 2; ++h) {
          const int p = 2 * l + h;
          launch_filler(p);
          if (cond == COND_AR) launch_ar(p);
          else if (cond == COND_STANDIN) launch_local(p);
        }
    };

    // ---- warm-up: no-op kernel; build (and in graph mode capture/instantiate/upload) the plans ----
    (void)hipGetLastError();
    gen_kernel<<<dim3(E.aux_x, 1), 256, 0, st>>>((u16*)local, 0xABCDEFull, n_elem, r);
    noop_kernel<<<1, 64, 0, st>>>();
    check_launch("warm-up");
    if (!drain_stream_wd(st, kWatchdogS)) {
      R.stuck = true;
      g_stuck = true;
      S.sync.brk();
      return;
    }
    Plan warm;
    warm.nbatches = 1;
    warm.K = 1;
    warm.bodies = {[&] { launch_ar(0); }};
    Plan P;
    P.nbatches = plan_batches;
    P.K = K;
    P.graph = c.graph;
    P.discard = discard;
    if (decode && !c.stress) {
      P.bodies = {[&] { token(COND_AR); }, [&] { token(COND_STANDIN); }, [&] { token(COND_FILLERS); }};
      P.gen_all = true;  // identical pre-token work (gen writes the input ring) in every condition
    } else if (decode) {
      P.bodies = {[&] { token(COND_AR); }};
    } else {
      P.bodies = {[&] {
        for (int q = 0; q < K; ++q) launch_ar(q);
      }};
    }
    prepare_plan(E, P, G);  // graph capture/instantiate/upload happen here, before the start barrier

    // ---- start barrier, then one throwaway (verified) all-reduce ----
    const int v0 = S.sync.vote(true, kBarrierS);
    if (v0 != 1) {
      R.plan_status = v0 == 0 ? PS_STOPPED : PS_BROKEN;
      R.stopped_in = "barrier";
      if (v0 < 0 && S.sync.timed_out()) E.barrier_timeout = true;
    } else {
      const int ps = run_plan(E, warm, Prepared{}, nullptr);
      if (ps != PS_OK) {
        R.plan_status = ps;
        R.stopped_in = "warmup";
      }
    }

    // ---- timed plan: isolated batches of K all-reduces | interleaved decode tokens | stress ----
    if (R.plan_status == PS_OK) {
      const int v1 = S.sync.vote(true, kBarrierS);
      if (v1 != 1) {
        R.plan_status = v1 == 0 ? PS_STOPPED : PS_BROKEN;
        R.stopped_in = "barrier";
        if (v1 < 0 && S.sync.timed_out()) E.barrier_timeout = true;
      } else {
        R.calls_before_timed = E.calls;
        const auto t_start = std::chrono::steady_clock::now();
        auto t_last = t_start;
        if (c.stress && r == 0) {
          P.on_batch = [&](long long done) {
            if (secs_since(t_last) >= 5.0) {
              t_last = std::chrono::steady_clock::now();
              const double el = secs_since(t_start);
              const long long calls = done * K;
              printf("[stress] %lld / %lld calls verified OK, %.0f calls/s, %.0f s elapsed\n", calls, plan_batches * K,
                     calls / el, el);
              fflush(stdout);
            }
          };
        }
        const long long ok_before = E.batches_ok;
        const long long starved_before = E.starved;
        const int ps = run_plan(E, P, G, R.ms);
        R.seconds = secs_since(t_start);
        R.batches_ok = E.batches_ok - ok_before;
        R.starved = E.starved - starved_before;
        if (ps != PS_OK) {
          R.plan_status = ps;
          R.stopped_in = "timed";
        }
        R.trace_call_lo = R.calls_before_timed + ar_batches_in(discard, C, 0) * K;
        R.trace_call_hi = R.calls_before_timed + ar_batches_in(R.batches_ok, C, 0) * K;  // checked batches only
        if (c.trace && R.plan_status != PS_STUCK) {
          R.trace.resize((size_t)trace_cap * 4);
          R.trace_cap = trace_cap;
          HIPC(hipMemcpyAsync(R.trace.data(), trace, R.trace.size() * sizeof(u64), hipMemcpyDeviceToHost, st));
          if (!drain_stream_wd(st, kWatchdogS)) {
            R.stuck = true;
            g_stuck = true;
          }
        }
      }
    }
    R.calls_total = E.calls;
    R.mirror = E.last;
    R.abort_seen[0] = E.last_abort[0];
    R.abort_seen[1] = E.last_abort[1];
    if (E.barrier_timeout && R.error.empty())
      R.error = "host barrier timeout (> 120 s): a rank's host thread stalled; the run is incomplete";
    (void)S.sync.vote(true, kBarrierS);  // both ranks done before anyone frees anything
  } catch (const std::exception& e) {
    R.error = e.what();
    S.sh->store(kOffAbort[r], ABORT_HOST);  // peer fails fast instead of timing out on every call
    S.sync.brk();
  }
  if (R.plan_status == PS_STUCK) R.stuck = true;
  if (st && !g_stuck) {
    bool idle = false;
    try {
      idle = drain_stream_wd(st, kWatchdogS);
    } catch (...) {
      idle = false;
    }
    if (idle) {
      destroy_prepared(G);
      for (hipEvent_t e : events) (void)hipEventDestroy(e);
      for (void* p : allocs) (void)hipFree(p);
      (void)hipStreamDestroy(st);
    } else {
      R.stuck = true;
      g_stuck = true;
    }
  }
}

static long long seq_to_call(u32 seq, u32 seq_base) { return (long long)(u32)(seq - 1u - seq_base); }

// Decode a Status/verify record into a failure report. first_fail_call: the earliest call (either rank)
// at which a block failed, bailed or skipped; mismatches at or after it are consequential.
static J failure_json(const RankRes& R, const Geom& g, bool ll, long long first_fail_call) {
  J f = J::obj();
  const Status& s = R.mirror;
  f["abort_words"] = J::arr();
  f["abort_words"].push(R.abort_seen[0]);
  f["abort_words"].push(R.abort_seen[1]);
  if (s.claimed) {
    J t = J::obj();
    t["code"] = abort_code_name(s.code);
    t["protocol_violation"] = s.code == ABORT_PROTOCOL;
    t["block"] = s.block;
    t["seq"] = s.seq;
    const long long call = seq_to_call(s.seq, R.seq_base);
    t["call_index"] = call;
    t["call_index_in_timed_run"] = call - R.calls_before_timed;
    t["phase"] = phase_name(s.phase);
    t["word"] = s.word == 0xFFFFFFFFu ? J() : J(s.word);
    t["observed"] = s.got;
    t["n_failing"] = s.n_timeouts;
    f["first_failure"] = t;
  }
  f["n_abort_exits"] = s.n_abort_exits;
  f["n_skipped_blocks"] = s.n_skipped;
  f["first_failed_or_skipped_call_this_rank"] =
      s.abort_seq_min != 0xFFFFFFFFu ? J(seq_to_call(s.abort_seq_min, R.seq_base)) : J();
  f["verify_mismatches"] = s.mismatches;
  if (s.mismatches) {
    const u64 key = s.min_key;
    const u64 call = key >> 36;
    const u32 elem = (u32)((key >> 16) & 0xFFFFFu);
    const u32 got = (u32)(key & 0xFFFFu);
    const u32 exp = expected_bits(call, elem);
    const u32 word = elem / 8;
    J m = J::obj();
    m["call_index"] = (unsigned long long)call;
    m["call_index_in_timed_run"] = (long long)call - R.calls_before_timed;
    m["element"] = elem;
    m["block"] = g.w ? word / (u32)g.w : 0;
    m["payload_word"] = word;
    if (ll) m["wire_word_in_block"] = 2 * (word - (word / (u32)g.w) * (u32)g.w) + (elem % 8) / 4;
    m["expected_bits"] = exp;
    m["got_bits"] = got;
    m["expected"] = (double)bitsf(exp << 16);
    m["got"] = (double)bitsf(got << 16);
    m["post_abort_consequential"] = first_fail_call >= 0 && (long long)call >= first_fail_call;
    f["first_mismatch"] = m;
  }
  return f;
}

// Per-rank decode statistics from the interleaved per-token samples (token i of (a), (b), (c) ran
// back to back, so paired differences cancel slow drift).
struct DecStats {
  size_t n = 0;
  double ta = NAN, tb = NAN, tc = NAN;       // mean per-token ms per condition
  double L = NAN, L_se = NAN;                // vs stand-in: mean(a_i - b_i) / 128, us
  double Lnok = NAN, Lnok_se = NAN;          // vs fillers only: mean(a_i - c_i) / 128, us
  std::vector<double> L_quintiles;           // L over 5 contiguous fifths of the tokens (drift check)
};
static DecStats decode_stats(const RankRes& X) {
  DecStats d;
  const size_t n = std::min(X.ms[COND_AR].size(), std::min(X.ms[COND_STANDIN].size(), X.ms[COND_FILLERS].size()));
  d.n = n;
  if (n == 0) return d;
  const double k = 1000.0 / kDecodeK;
  auto mean_sd = [](const std::vector<double>& v, double* sd) {
    double s = 0;
    for (double x : v) s += x;
    const double m = s / (double)v.size();
    double ss = 0;
    for (double x : v) ss += (x - m) * (x - m);
    *sd = v.size() > 1 ? std::sqrt(ss / (double)(v.size() - 1)) : 0.0;
    return m;
  };
  std::vector<double> a(X.ms[COND_AR].begin(), X.ms[COND_AR].begin() + n);
  std::vector<double> b(X.ms[COND_STANDIN].begin(), X.ms[COND_STANDIN].begin() + n);
  std::vector<double> c(X.ms[COND_FILLERS].begin(), X.ms[COND_FILLERS].begin() + n);
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
  d.L_se = sd / std::sqrt((double)n) * k;
  d.Lnok = mean_sd(dac, &sd) * k;
  d.Lnok_se = sd / std::sqrt((double)n) * k;
  if (n >= 5)
    for (int q = 0; q < 5; ++q) {
      const size_t lo = n * q / 5, hi = n * (q + 1) / 5;
      std::vector<double> part(dab.begin() + lo, dab.begin() + hi);
      d.L_quintiles.push_back(mean_sd(part, &sd) * k);
    }
  return d;
}

struct ArOutcome {
  int code = EXIT_OK;
  bool violation = false;
  double us_per_ar = NAN;  // isolated: max over ranks
  double tok_ms_ar = NAN, tok_ms_local = NAN, tok_ms_fill = NAN;  // decode, max over ranks
  double L_us = NAN, L_se = NAN, L_nok_us = NAN, L_nok_se = NAN;
};

static ArOutcome run_ar_session(const Opt& o, const ArCfg& c, SharedRegion& sh, const RegionPlan& rp, J& out) {
  ArOutcome oc;
  ArSession S;
  S.cfg = c;
  S.g = make_geom(c.bytes, c.nb, c.variant == "ll");
  S.sh = &sh;
  S.mbox_off[0] = rp.mbox_off[0];
  S.mbox_off[1] = rp.mbox_off[1];
  S.devs[0] = o.dev0;
  S.devs[1] = o.dev1;
  S.seq_base = sh.next_seq_base;
  const Geom& g = S.g;
  const bool decode = c.pattern == "decode";

  J cj = J::obj();
  cj["variant"] = c.variant;
  if (c.variant == "flag") {
    cj["drain"] = c.drain;
    cj["drain_note"] = c.drain == 3 ? "drain 3 is ISA-identical to no drain in WGP mode (see README)" : "all-thread __threadfence_system";
    cj["acq"] = c.acq;
    cj["spin"] = c.spin_acq ? "acq (acquire load per poll)" : "rlx (relaxed polls + one acquire fence)";
  }
  cj["bytes"] = (unsigned long long)c.bytes;
  cj["nb_requested"] = c.nb;
  cj["nb"] = g.nb;
  cj["nt"] = c.nt;
  cj["payload_words_per_block"] = g.w;
  cj["mailbox_block_stride"] = g.blk_stride;
  cj["mailbox_slot_stride"] = g.slot_stride;
  cj["wire_bytes_per_rank"] = (unsigned long long)((size_t)g.n16 * g.wire);
  cj["pattern"] = decode ? "decode" : "isolated";
  cj["stress"] = c.stress;
  cj["graph"] = c.graph;
  cj["timeout_ms"] = c.timeout_ms;
  cj["alloc"] = o.alloc;
  cj["dev0"] = o.dev0;
  cj["dev1"] = o.dev1;
  cj["seq_base"] = S.seq_base;
  if (c.stress) cj["count_requested"] = c.count;
  if (decode) {
    if (!c.stress) cj["tokens"] = c.tokens;
    cj["filler_mb"] = c.filler_mb;
    cj["dirty_mb"] = c.dirty_mb;
    cj["layers"] = kLayers;
    cj["ars_per_token"] = kDecodeK;
  }
  if (!c.stress && !decode) {
    cj["iters_requested"] = c.iters;
    cj["discarded_warmup_batches"] = kIsoDiscard;
  }
  cj["ars_per_verified_batch"] = decode ? kDecodeK : kIsoK;
  cj["verify_cadence"] = "every output of every AR is verified, once per batch (50 ARs isolated/stress, one 128-AR "
                         "token in decode); spec asked 'after every call' (stress) / 'every <=64 calls' -- every call "
                         "has its own ring row, so nothing goes unverified and no verify kernel sits between ARs";
  cj["verify_timing"] = "gen/verify/publish run OUTSIDE the hipEvent pair of every batch (not subtracted, excluded)";
  out["config"] = cj;

  if (g.mbox_bytes > rp.mbox_region) {
    out["status"] = "error";
    out["error"] = "mailbox region too small for this configuration";
    oc.code = EXIT_ERR;
    return oc;
  }
  sh.zero();  // both GPUs idle: no rank thread exists yet

  RankRes R[2];
  std::thread t0(ar_rank, std::ref(S), 0, std::ref(R[0]));
  std::thread t1(ar_rank, std::ref(S), 1, std::ref(R[1]));
  t0.join();
  t1.join();
  // the next session on this region starts above every seq used here (per-block seq = base + calls)
  sh.next_seq_base = S.seq_base + (u32)std::max(R[0].calls_total, R[1].calls_total) + 64u;

  // ---- outcome ----
  const bool ll = c.variant == "ll";
  const u32 ab0 = sh.load(kOffAbort[0]), ab1 = sh.load(kOffAbort[1]);
  bool any_err = false, any_timeout = false, any_mismatch = false, any_violation = false, incomplete = false;
  long long first_fail_call = -1;
  for (int r = 0; r < 2; ++r) {
    const RankRes& X = R[r];
    any_err = any_err || !X.error.empty();
    any_violation = any_violation || (X.mirror.claimed && X.mirror.code == ABORT_PROTOCOL);
    any_timeout = any_timeout || X.stuck || (X.mirror.claimed && X.mirror.code != ABORT_PROTOCOL);
    for (u32 w : X.abort_seen) {
      if (w == ABORT_PROTOCOL) any_violation = true;
      else if (w != 0) any_timeout = true;
    }
    any_mismatch = any_mismatch || X.mirror.mismatches;
    incomplete = incomplete || X.plan_status != PS_OK || X.batches_ok < X.batches_planned;
    if (X.mirror.abort_seq_min != 0xFFFFFFFFu) {
      const long long fc = seq_to_call(X.mirror.abort_seq_min, X.seq_base);
      if (first_fail_call < 0 || fc < first_fail_call) first_fail_call = fc;
    }
  }
  for (u32 w : {ab0, ab1}) {
    if (w == ABORT_PROTOCOL) any_violation = true;
    else if (w == ABORT_TIMEOUT) any_timeout = true;
  }
  oc.violation = any_violation;
  oc.code = any_err         ? EXIT_ERR
            : any_violation ? EXIT_MISMATCH
            : any_timeout   ? EXIT_TIMEOUT
            : any_mismatch  ? EXIT_MISMATCH
            : incomplete    ? EXIT_ERR
                            : EXIT_OK;
  if (oc.code == EXIT_ERR && !any_err) out["error"] = "the run stopped early with no recorded cause";
  const char* status_str = oc.code == EXIT_OK                        ? "ok"
                           : oc.code == EXIT_ERR                     ? "error"
                           : any_violation                           ? "protocol_violation"
                           : oc.code == EXIT_TIMEOUT                 ? "timeout"
                                                                     : "mismatch";
  out["status"] = status_str;
  out["exit_code"] = oc.code;
  out["protocol_violation"] = any_violation;
  out["abort_words_final"] = J::arr();
  out["abort_words_final"].push(ab0);
  out["abort_words_final"].push(ab1);
  out["first_failed_call_any_rank"] = first_fail_call >= 0 ? J(first_fail_call) : J();

  J ranks = J::arr();
  double us_max = NAN, ta_max = NAN, tb_max = NAN, tc_max = NAN, se_max = NAN, se_nok_max = NAN;
  auto upd_max = [](double& m, double v) {
    if (std::isfinite(v)) m = std::isfinite(m) ? std::max(m, v) : v;
  };
  for (int r = 0; r < 2; ++r) {
    const RankRes& X = R[r];
    J rj = J::obj();
    rj["rank"] = r;
    rj["device"] = X.dev;
    rj["wallclock_khz"] = X.khz;
    if (!X.error.empty()) rj["error"] = X.error;
    if (X.stuck) rj["stuck"] = true;
    if (!X.stopped_in.empty()) {
      rj["stopped_in"] = X.stopped_in;
      rj["plan_status"] = plan_status_name(X.plan_status);
    }
    rj["calls_total"] = X.calls_total;
    rj["timed_batches_completed"] = std::max(0LL, X.batches_ok - X.discard);
    rj["discarded_batches"] = X.discard;
    rj["host_starved_batches"] = X.starved;
    if (decode) rj["filler_grid"] = X.fgrid;
    rj["failure"] = failure_json(X, g, ll, first_fail_call);
    double sum = 0;
    for (double v : X.ms[COND_AR]) sum += v;
    if (!decode && !c.stress) {
      const double n_ar = (double)X.ms[0].size() * X.K;
      const double us = n_ar > 0 ? sum * 1000.0 / n_ar : NAN;
      rj["timed_all_reduces"] = n_ar;
      rj["mean_us_per_ar"] = us;
      std::vector<double> per;
      for (double v : X.ms[0]) per.push_back(v * 1000.0 / X.K);
      rj["batch_us_per_ar"] = summ_json(summarize(per));
      rj["batch_ms"] = vec_json(X.ms[0]);
      upd_max(us_max, us);
    }
    if (decode && !c.stress) {
      const DecStats d = decode_stats(X);
      rj["tokens_per_condition"] = (unsigned long long)d.n;
      rj["per_token_ms_allreduce"] = d.ta;
      rj["per_token_ms_standin"] = d.tb;
      rj["per_token_ms_fillers_only"] = d.tc;
      rj["L_us_this_rank"] = d.L;
      rj["L_us_se_this_rank"] = d.L_se;
      rj["L_us_vs_no_ar_kernel_this_rank"] = d.Lnok;
      rj["L_us_vs_no_ar_kernel_se_this_rank"] = d.Lnok_se;
      rj["L_us_by_quintile"] = vec_json(d.L_quintiles);
      rj["token_ms_allreduce"] = summ_json(summarize(X.ms[COND_AR]));
      rj["token_ms_standin"] = summ_json(summarize(X.ms[COND_STANDIN]));
      rj["token_ms_fillers_only"] = summ_json(summarize(X.ms[COND_FILLERS]));
      rj["batch_ms_allreduce"] = vec_json(X.ms[COND_AR]);
      rj["batch_ms_standin"] = vec_json(X.ms[COND_STANDIN]);
      rj["batch_ms_fillers_only"] = vec_json(X.ms[COND_FILLERS]);
      upd_max(ta_max, d.ta);
      upd_max(tb_max, d.tb);
      upd_max(tc_max, d.tc);
      upd_max(se_max, d.L_se);
      upd_max(se_nok_max, d.Lnok_se);
    }
    if (c.stress) {
      rj["seconds"] = X.seconds;
      rj["calls_verified"] = X.batches_ok * X.K;
    }
    // in-kernel trace (block 0): push / wait / reduce / total
    if (!X.trace.empty() && X.trace_call_hi > X.trace_call_lo) {
      std::vector<double> push, wait, red, tot;
      const double k2ns = 1e6 / X.khz;
      for (long long call = X.trace_call_lo; call < X.trace_call_hi; ++call) {
        const u64* t = &X.trace[(size_t)(call % X.trace_cap) * 4];
        if (t[0] == 0 || t[3] < t[0] || t[1] < t[0] || t[2] < t[1] || t[3] < t[2]) continue;
        push.push_back((double)(t[1] - t[0]) * k2ns);
        wait.push_back((double)(t[2] - t[1]) * k2ns);
        red.push_back((double)(t[3] - t[2]) * k2ns);
        tot.push_back((double)(t[3] - t[0]) * k2ns);
      }
      J tj = J::obj();
      tj["unit"] = "ns";
      tj["note"] = ll ? "LL: push = entry -> all pushes issued (+ a trace-only barrier); wait includes the fused reduce "
                        "(reduce_ns == 0)"
                      : "flag: push = entry -> after the flag release store (push + drain + barrier + the release's "
                        "system writeback, which is where drain=3 flushes); wait = spin until the peer's flag";
      tj["push_ns"] = summ_json(summarize(push));
      tj["wait_ns"] = summ_json(summarize(wait));
      tj["reduce_ns"] = summ_json(summarize(red));
      tj["total_ns"] = summ_json(summarize(tot));
      tj["samples"] = (unsigned long long)tot.size();
      tj["push_ns_samples"] = vec_json(push);
      tj["wait_ns_samples"] = vec_json(wait);
      tj["reduce_ns_samples"] = vec_json(red);
      tj["total_ns_samples"] = vec_json(tot);
      rj["trace"] = tj;
    }
    ranks.push(rj);
  }
  out["ranks"] = ranks;

  J res = J::obj();
  if (!decode && !c.stress) {
    res["mean_us_per_ar_max_over_ranks"] = us_max;
    oc.us_per_ar = us_max;
  }
  if (decode && !c.stress) {
    res["per_token_ms_allreduce_max_over_ranks"] = ta_max;
    res["per_token_ms_standin_max_over_ranks"] = tb_max;
    res["per_token_ms_fillers_only_max_over_ranks"] = tc_max;
    oc.tok_ms_ar = ta_max;
    oc.tok_ms_local = tb_max;
    oc.tok_ms_fill = tc_max;
    oc.L_us = (ta_max - tb_max) * 1000.0 / kDecodeK;
    oc.L_se = se_max;
    oc.L_nok_us = (ta_max - tc_max) * 1000.0 / kDecodeK;
    oc.L_nok_se = se_nok_max;
    res["L_us_in_engine"] = oc.L_us;
    res["L_us_in_engine_se"] = oc.L_se;
    res["L_us_vs_no_ar_kernel"] = oc.L_nok_us;
    res["L_us_vs_no_ar_kernel_se"] = oc.L_nok_se;
    res["L_formula"] = "L_us_in_engine = (T_a - T_b) / 128 (spec: vs a local stand-in kernel in the AR slot); "
                       "L_us_vs_no_ar_kernel = (T_a - T_c) / 128 (vs fillers only: also charges the AR slot's own "
                       "dispatch + kernel boundary). T = per-token ms, max over ranks. se = standard error of the "
                       "paired per-token differences (max over ranks).";
    res["interleaving"] = "tokens alternate (a) AR, (b) stand-in, (c) fillers only; gen runs before every token of "
                          "every condition; verify runs after (a) tokens only (outside the timed events)";
  }
  if (c.stress) {
    const double secs = std::max(R[0].seconds, R[1].seconds);
    const long long verified = std::min(R[0].batches_ok, R[1].batches_ok) * (long long)R[0].K;
    res["calls_verified"] = verified;
    res["calls_requested_rounded"] = R[0].batches_planned * (long long)R[0].K;
    res["seconds"] = secs;
    res["calls_per_s"] = secs > 0 ? verified / secs : NAN;
  }
  out["result"] = res;

  // ---- human-readable ----
  const std::string lbl = cfg_label(c);
  if (c.stress) {
    printf("[stress%s] %s: %s, %lld calls verified in %.1f s\n", decode ? " decode" : "", lbl.c_str(), status_str,
           std::min(R[0].batches_ok, R[1].batches_ok) * (long long)R[0].K, std::max(R[0].seconds, R[1].seconds));
  } else if (decode) {
    printf("[decode] %s tokens=%d: per-token ms  AR %.3f  stand-in %.3f  fillers-only %.3f  ->  L = %.2f +- %.2f us/AR "
           "(vs stand-in), %.2f +- %.2f us/AR (vs no AR kernel)  [%s]\n",
           lbl.c_str(), c.tokens, ta_max, tb_max, tc_max, oc.L_us, oc.L_se, oc.L_nok_us, oc.L_nok_se,
           oc.code == EXIT_OK ? "ok" : "FAILED");
  } else {
    double u[2] = {NAN, NAN};
    for (int r = 0; r < 2; ++r) {
      double s = 0;
      for (double v : R[r].ms[0]) s += v;
      const double n = (double)R[r].ms[0].size() * R[r].K;
      u[r] = n > 0 ? s * 1000.0 / n : NAN;
    }
    printf("[isolated] %s: %.2f us/AR (r0 %.2f, r1 %.2f)  [%s]\n", lbl.c_str(), us_max, u[0], u[1],
           oc.code == EXIT_OK ? "ok" : "FAILED");
  }
  for (int r = 0; r < 2; ++r) {
    const RankRes& X = R[r];
    if (!X.error.empty()) printf("    rank %d ERROR: %s\n", r, X.error.c_str());
    if (X.stuck) printf("    rank %d: stream STUCK (host watchdog %.0f s)\n", r, kWatchdogS);
    if (X.mirror.claimed) {
      const long long call = seq_to_call(X.mirror.seq, X.seq_base);
      printf("    rank %d %s: %s, block %u, call %lld (%lld into the timed run), phase %s, word %s, observed %u\n", r,
             X.mirror.code == ABORT_PROTOCOL ? "PROTOCOL VIOLATION" : "TIMEOUT/ABORT", abort_code_name(X.mirror.code),
             X.mirror.block, call, call - X.calls_before_timed, phase_name(X.mirror.phase),
             X.mirror.word == 0xFFFFFFFFu ? "-" : std::to_string(X.mirror.word).c_str(), X.mirror.got);
    }
    if (X.mirror.mismatches) {
      const u64 key = X.mirror.min_key;
      const u64 call = key >> 36;
      const u32 elem = (u32)((key >> 16) & 0xFFFFFu), got = (u32)(key & 0xFFFFu), exp = expected_bits(call, elem);
      const bool conseq = first_fail_call >= 0 && (long long)call >= first_fail_call;
      printf("    rank %d MISMATCH%s: %u elements; first at call %llu (%lld into the timed run), block %u, word %u, elem "
             "%u: expected %g (0x%04x) got %g (0x%04x)\n",
             r, conseq ? " (post-abort, consequential)" : "", X.mirror.mismatches, call,
             (long long)call - X.calls_before_timed, (elem / 8) / (u32)g.w, elem / 8, elem, (double)bitsf(exp << 16),
             exp, (double)bitsf(got << 16), got);
    }
    if ((X.mirror.n_abort_exits || X.mirror.n_skipped) && !X.mirror.claimed)
      printf("    rank %d: %u blocks/threads bailed on a set abort word, %u blocks skipped calls (sticky abort)\n", r,
             X.mirror.n_abort_exits, X.mirror.n_skipped);
    if (!X.stopped_in.empty() && X.error.empty() && !X.stuck && !X.mirror.claimed)
      printf("    rank %d: stopped in %s (%s)\n", r, X.stopped_in.c_str(), plan_status_name(X.plan_status));
  }
  if (ab0 || ab1) printf("    abort words: rank0=%u (%s) rank1=%u (%s)\n", ab0, abort_code_name(ab0), ab1, abort_code_name(ab1));
  if (oc.code == EXIT_ERR && !any_err) printf("    ERROR: the run stopped early with no recorded cause\n");
  fflush(stdout);
  return oc;
}

static ArCfg cfg_from_opt(const Opt& o) {
  ArCfg c;
  c.variant = o.variant;
  c.drain = o.drain;
  c.acq = o.acq;
  c.spin_acq = o.spin == "acq" ? 1 : 0;
  c.bytes = (size_t)o.bytes;
  c.nb = o.nb;
  c.nt = o.nt;
  c.iters = o.iters > 0 ? o.iters : 5000;
  c.pattern = o.pattern;
  c.graph = o.graph;
  c.timeout_ms = o.timeout_ms;
  c.trace = o.trace;
  c.tokens = o.tokens;
  c.filler_mb = o.filler_mb;
  c.dirty_mb = o.dirty_mb;
  c.count = o.count;
  return c;
}

static int run_allreduce_mode(const Opt& o, bool stress, J& out) {
  ArCfg c = cfg_from_opt(o);
  c.stress = stress;  // stress --pattern decode: count in ARs, rounded up to whole tokens, every token verified
  validate_wallclock(o.dev0);
  validate_wallclock(o.dev1);
  const Geom g = make_geom(c.bytes, c.nb, c.variant == "ll");
  const RegionPlan rp = plan_region(g.mbox_bytes);
  SharedRegion sh;
  sh.alloc(o.alloc, rp.bytes, {o.dev0, o.dev1});
  const ArOutcome oc = run_ar_session(o, c, sh, rp, out);
  return oc.code;
}

// =================================================================================================
// Mode 6: sweep -- the standard decision matrix in one JSON
// =================================================================================================
static int run_sweep(const Opt& o, J& out) {
  int code = EXIT_OK;
  auto fatal = [](int c) { return c == EXIT_ERR || c == EXIT_TIMEOUT; };
  printf("[sweep] alloc=%s dev0=%d dev1=%d timeout_ms=%d graph=%d spin=%s\n", o.alloc.c_str(), o.dev0, o.dev1,
         o.timeout_ms, (int)o.graph, o.spin.c_str());
  // 1. probe (fails if the selected --alloc path is unusable on either device)
  {
    J p = J::obj();
    const int c = run_probe(o, p);
    out["probe"] = p;
    code = worse_code(code, c);
    if (fatal(c)) return code;
  }
  // 2. loopback on dev1 then dev0 (GPU partner)
  {
    J lb = J::arr();
    Opt ol = o;
    ol.iters = 100000;
    for (int dev : {o.dev1, o.dev0}) {
      J j = J::obj();
      int c = EXIT_OK;
      try {
        c = run_loopback(ol, dev, "gpu", j);
      } catch (const std::exception& e) {
        j["error"] = e.what();
        printf("[loopback] ERROR: %s\n", e.what());
        c = EXIT_ERR;
      }
      lb.push(j);
      code = worse_code(code, c);
      if (fatal(c)) {
        out["loopback"] = lb;
        return code;
      }
    }
    out["loopback"] = lb;
  }
  // 3. pingpong
  {
    J j = J::obj();
    Opt op = o;
    op.iters = 100000;
    int c = EXIT_OK;
    try {
      c = run_pingpong(op, j);
    } catch (const std::exception& e) {
      j["error"] = e.what();
      printf("[pingpong] ERROR: %s\n", e.what());
      c = EXIT_ERR;
    }
    out["pingpong"] = j;
    code = worse_code(code, c);
    if (fatal(c)) return code;
  }
  // 4. per variant (safe one first): its isolated matrix, then its decode rows at 10 and 80 KiB with the
  //    best isolated nb -- so the safe variant's decode L is in the JSON before a relaxed variant can
  //    end the sweep with an abort.
  struct V {
    const char* name;
    const char* variant;
    int drain, acq;
    const char* stress_args;
  };
  const V variants[] = {{"flag(drain1,acq1)", "flag", 1, 1, "--variant flag --drain 1 --acq 1"},
                        {"flag(drain3,acq0)", "flag", 3, 0, "--variant flag --drain 3 --acq 0"},
                        {"ll", "ll", 1, 1, "--variant ll"}};
  const size_t bytes_list[] = {10240, 81920, 655360};
  const int nb_list[] = {1, 4, 8, 16, 32};
  size_t max_mbox = 0;
  for (size_t by : bytes_list)
    for (int nb : nb_list) max_mbox = std::max(max_mbox, make_geom(by, nb, true).mbox_bytes);
  const RegionPlan rp = plan_region(max_mbox);
  SharedRegion sh;
  try {
    sh.alloc(o.alloc, rp.bytes, {o.dev0, o.dev1});
  } catch (const std::exception& e) {
    out["error"] = e.what();
    printf("[sweep] ERROR: %s\n", e.what());
    return worse_code(code, EXIT_ERR);
  }
  J iso = J::arr();
  J dec = J::arr();
  double us[3][3][5];
  int st[3][3][5];
  double L[3][2], Lse[3][2], Lnok[3][2], Lnokse[3][2], tokA[3][2], tokB[3][2], tokC[3][2];
  int best_nb[3][2], dst[3][2];
  for (auto& a : us)
    for (auto& b : a)
      for (double& x : b) x = NAN;
  for (auto& a : st)
    for (auto& b : a)
      for (int& x : b) x = -1;
  for (int vi = 0; vi < 3; ++vi)
    for (int bi = 0; bi < 2; ++bi) {
      L[vi][bi] = Lse[vi][bi] = Lnok[vi][bi] = Lnokse[vi][bi] = tokA[vi][bi] = tokB[vi][bi] = tokC[vi][bi] = NAN;
      best_nb[vi][bi] = -1;
      dst[vi][bi] = -1;
    }
  bool stop = false;
  for (int vi = 0; vi < 3 && !stop; ++vi) {
    for (int bi = 0; bi < 3 && !stop; ++bi) {
      for (int ni = 0; ni < 5 && !stop; ++ni) {
        ArCfg c = cfg_from_opt(o);
        c.variant = variants[vi].variant;
        c.drain = variants[vi].drain;
        c.acq = variants[vi].acq;
        c.bytes = bytes_list[bi];
        c.nb = nb_list[ni];
        c.pattern = "isolated";
        c.iters = 2000;
        c.stress = false;  // c.trace = --trace (off by default: it adds a block-0 barrier)
        J j = J::obj();
        j["name"] = variants[vi].name;
        const ArOutcome oc = run_ar_session(o, c, sh, rp, j);
        iso.push(j);
        us[vi][bi][ni] = oc.code == EXIT_OK ? oc.us_per_ar : NAN;
        st[vi][bi][ni] = oc.code;
        code = worse_code(code, oc.code);
        if (fatal(oc.code)) stop = true;
      }
    }
    for (int bi = 0; bi < 2 && !stop; ++bi) {
      int bn = -1;
      double bu = INFINITY;
      for (int ni = 0; ni < 5; ++ni)
        if (std::isfinite(us[vi][bi][ni]) && us[vi][bi][ni] < bu) {
          bu = us[vi][bi][ni];
          bn = nb_list[ni];
        }
      best_nb[vi][bi] = bn;
      J j = J::obj();
      j["name"] = variants[vi].name;
      if (bn < 0) {
        j["skipped"] = "no passing isolated configuration";
        dec.push(j);
        continue;
      }
      ArCfg c = cfg_from_opt(o);
      c.variant = variants[vi].variant;
      c.drain = variants[vi].drain;
      c.acq = variants[vi].acq;
      c.bytes = bytes_list[bi];
      c.nb = bn;
      c.pattern = "decode";
      c.trace = false;
      c.stress = false;
      const ArOutcome oc = run_ar_session(o, c, sh, rp, j);
      dec.push(j);
      dst[vi][bi] = oc.code;
      if (oc.code == EXIT_OK) {
        L[vi][bi] = oc.L_us;
        Lse[vi][bi] = oc.L_se;
        Lnok[vi][bi] = oc.L_nok_us;
        Lnokse[vi][bi] = oc.L_nok_se;
        tokA[vi][bi] = oc.tok_ms_ar;
        tokB[vi][bi] = oc.tok_ms_local;
        tokC[vi][bi] = oc.tok_ms_fill;
      }
      code = worse_code(code, oc.code);
      if (fatal(oc.code)) stop = true;
    }
  }
  out["isolated"] = iso;
  out["decode"] = dec;
  if (stop) out["stopped_early"] = "abort, stuck stream or host error: the sweep did not continue";

  // 5. table + decision summary
  printf("\n==== isolated all-reduce, us per AR (hipEvent, max over ranks); FAIL = mismatch/timeout ====\n");
  printf("%-20s %8s", "variant", "bytes");
  for (int nb : nb_list) printf("   nb=%-4d", nb);
  printf("\n");
  J table = J::arr();
  for (int vi = 0; vi < 3; ++vi)
    for (int bi = 0; bi < 3; ++bi) {
      printf("%-20s %8zu", variants[vi].name, bytes_list[bi]);
      J row = J::obj();
      row["variant"] = variants[vi].name;
      row["bytes"] = (unsigned long long)bytes_list[bi];
      J cells = J::obj();
      for (int ni = 0; ni < 5; ++ni) {
        const int s = st[vi][bi][ni];
        if (s < 0) printf("   %-7s", "-");
        else if (s != EXIT_OK) printf("   %-7s", s == EXIT_MISMATCH ? "MISM" : s == EXIT_TIMEOUT ? "TOUT" : "ERR");
        else printf("   %7.2f", us[vi][bi][ni]);
        cells["nb" + std::to_string(nb_list[ni])] = s == EXIT_OK ? J(us[vi][bi][ni]) : J();
      }
      row["us_per_ar"] = cells;
      table.push(row);
      printf("\n");
    }
  printf("\n==== decode emulation (%d tokens/condition, interleaved; 64 layers x [filler, AR, filler, AR]) ====\n",
         o.tokens);
  printf("%-20s %8s %4s %10s %10s %10s %15s %15s\n", "variant", "bytes", "nb", "tok(AR)", "tok(stdin)", "tok(fill)",
         "L_us(stand-in)", "L_us(no-kernel)");
  J dtable = J::arr();
  for (int vi = 0; vi < 3; ++vi)
    for (int bi = 0; bi < 2; ++bi) {
      printf("%-20s %8zu %4d %10.3f %10.3f %10.3f %8.2f+-%-5.2f %8.2f+-%-5.2f%s\n", variants[vi].name, bytes_list[bi],
             best_nb[vi][bi], tokA[vi][bi], tokB[vi][bi], tokC[vi][bi], L[vi][bi], Lse[vi][bi], Lnok[vi][bi],
             Lnokse[vi][bi], dst[vi][bi] > 0 ? (dst[vi][bi] == EXIT_MISMATCH ? "  MISM" : dst[vi][bi] == EXIT_TIMEOUT ? "  TOUT" : "  ERR") : "");
      J row = J::obj();
      row["variant"] = variants[vi].name;
      row["bytes"] = (unsigned long long)bytes_list[bi];
      row["nb"] = best_nb[vi][bi];
      row["status_exit_code"] = dst[vi][bi] >= 0 ? J(dst[vi][bi]) : J();
      row["per_token_ms_allreduce"] = tokA[vi][bi];
      row["per_token_ms_standin"] = tokB[vi][bi];
      row["per_token_ms_fillers_only"] = tokC[vi][bi];
      row["L_us"] = L[vi][bi];
      row["L_us_se"] = Lse[vi][bi];
      row["L_us_vs_no_ar_kernel"] = Lnok[vi][bi];
      row["L_us_vs_no_ar_kernel_se"] = Lnokse[vi][bi];
      dtable.push(row);
    }
  J summary = J::obj();
  summary["isolated_table"] = table;
  summary["decode_table"] = dtable;
  J dec_j = J::obj();
  int bv[2] = {-1, -1};
  for (int bi = 0; bi < 2; ++bi)
    for (int vi = 0; vi < 3; ++vi)
      if (std::isfinite(L[vi][bi]) && (bv[bi] < 0 || L[vi][bi] < L[bv[bi]][bi])) bv[bi] = vi;
  auto jnum = [](double v) { return std::isfinite(v) ? J(v) : J(); };
  auto jle = [](double v, double t) { return std::isfinite(v) ? J(v <= t) : J(); };
  const char* size_key[2] = {"10KiB", "80KiB"};
  for (int bi = 0; bi < 2; ++bi) {
    const int vi = bv[bi];
    const double l = vi >= 0 ? L[vi][bi] : NAN, lse = vi >= 0 ? Lse[vi][bi] : NAN;
    const double ln = vi >= 0 ? Lnok[vi][bi] : NAN, lnse = vi >= 0 ? Lnokse[vi][bi] : NAN;
    const std::string k = size_key[bi];
    dec_j["best_L_us_" + k] = jnum(l);
    dec_j["best_L_us_se_" + k] = jnum(lse);
    dec_j["best_L_variant_" + k] = vi >= 0 ? J(variants[vi].name) : J();
    dec_j["best_L_nb_" + k] = vi >= 0 ? J(best_nb[vi][bi]) : J();
    dec_j["best_L_vs_no_ar_kernel_us_" + k] = jnum(ln);
    dec_j["best_L_vs_no_ar_kernel_us_se_" + k] = jnum(lnse);
  }
  const double l10 = bv[0] >= 0 ? L[bv[0]][0] : NAN, l80 = bv[1] >= 0 ? L[bv[1]][1] : NAN;
  const double n10 = bv[0] >= 0 ? Lnok[bv[0]][0] : NAN;
  dec_j["target_1p5x_L10KiB_le_7us"] = jle(l10, 7.0);
  dec_j["target_1p5x_L10KiB_le_12us"] = jle(l10, 12.0);
  dec_j["target_1p5x_L10KiB_vs_no_ar_kernel_le_7us"] = jle(n10, 7.0);
  dec_j["target_1p5x_L10KiB_vs_no_ar_kernel_le_12us"] = jle(n10, 12.0);
  dec_j["dflash2_L80KiB_le_15us"] = jle(l80, 15.0);
  dec_j["break_even_L10KiB_le_80us"] = jle(l10, 80.0);
  // projection: per-rank step at zero comm cost (17.0-17.6 ms) + 128 ARs x L, vs 27.8 ms single-GPU
  auto proj = [&](double l, const char* tag) {
    if (!std::isfinite(l)) return;
    const double lo = kTokMsZeroCommLo + kDecodeK * l / 1000.0, hi = kTokMsZeroCommHi + kDecodeK * l / 1000.0;
    dec_j[std::string("projected_tok_ms_") + tag] = J::arr();
    dec_j[std::string("projected_tok_ms_") + tag].push(lo);
    dec_j[std::string("projected_tok_ms_") + tag].push(hi);
    dec_j[std::string("projected_speedup_") + tag] = J::arr();
    dec_j[std::string("projected_speedup_") + tag].push(kTokMsSingleGpu / hi);
    dec_j[std::string("projected_speedup_") + tag].push(kTokMsSingleGpu / lo);
  };
  proj(l10, "10KiB");
  proj(n10, "10KiB_vs_no_ar_kernel");
  if (bv[0] >= 0) {
    char cmd[256];
    snprintf(cmd, sizeof cmd, "ar_bench.exe stress %s --bytes 10240 --nb %d --pattern decode --count 1280000 --alloc %s",
             variants[bv[0]].stress_args, best_nb[bv[0]][0], o.alloc.c_str());
    dec_j["requires_stress_validation"] = std::string(variants[bv[0]].name) + ": not stress-validated by the sweep; run `" +
                                          cmd + "` (and the isolated stress) before adopting it";
  }
  dec_j["baseline_note"] = "L_us (vs stand-in) is the spec's number and assumes the engine's 17.0-17.6 ms zero-comm "
                           "step already has a kernel in each AR slot (e.g. the residual add the AR is fused into). If "
                           "that baseline had no kernel there (AR as a separate extra kernel), use the conservative "
                           "L_vs_no_ar_kernel.";
  dec_j["note"] = "thresholds from the TP=2 plan: 1.5x needs L(10 KiB) <= ~7-12 us, DFlash2 verify L(80 KiB) <= ~15 us, "
                  "break-even ~80 us. Isolated 640 KiB rows are the prefill all-reduce. projected_speedup = 27.8 / "
                  "(17.0..17.6 + 0.128 x L).";
  summary["decision"] = dec_j;
  out["summary"] = summary;
  printf("\nbest in-engine L (vs stand-in): 10 KiB %.2f +- %.2f us (%s), 80 KiB %.2f us (%s)  [targets: <=7-12 us, "
         "<=15 us; break-even ~80 us]\n",
         l10, bv[0] >= 0 ? Lse[bv[0]][0] : NAN, bv[0] >= 0 ? variants[bv[0]].name : "-", l80,
         bv[1] >= 0 ? variants[bv[1]].name : "-");
  printf("                 (vs no AR kernel): 10 KiB %.2f us\n", n10);
  if (std::isfinite(l10))
    printf("projected decode step %.2f-%.2f ms -> %.2fx-%.2fx vs %.1f ms single-GPU\n",
           kTokMsZeroCommLo + kDecodeK * l10 / 1000.0, kTokMsZeroCommHi + kDecodeK * l10 / 1000.0,
           kTokMsSingleGpu / (kTokMsZeroCommHi + kDecodeK * l10 / 1000.0),
           kTokMsSingleGpu / (kTokMsZeroCommLo + kDecodeK * l10 / 1000.0), kTokMsSingleGpu);
  return code;
}

// =================================================================================================
// main
// =================================================================================================
static bool write_json(const std::string& path, const J& j) {
  FILE* f = nullptr;
  if (fopen_s(&f, path.c_str(), "wb") != 0) f = nullptr;
  if (!f) {
    printf("ERROR: cannot open %s for writing\n", path.c_str());
    return false;
  }
  const std::string s = j.dump();
  const bool ok = fwrite(s.data(), 1, s.size(), f) == s.size();
  fclose(f);
  if (ok) printf("JSON written to %s\n", path.c_str());
  return ok;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  Opt o;
  std::string err;
  if (!parse_args(argc, argv, o, err)) {
    usage();
    if (err != "help") printf("error: %s\n", err.c_str());
    return err == "help" ? EXIT_OK : EXIT_ERR;
  }
  J out = J::obj();
  out["tool"] = "ar_bench";
  out["mode"] = o.mode;
  {
    J av = J::arr();
    for (int i = 0; i < argc; ++i) av.push(argv[i]);
    out["argv"] = av;
  }
  int code = EXIT_OK;
  {
    char hvd[64] = {0};
    if (GetEnvironmentVariableA("HIP_VISIBLE_DEVICES", hvd, sizeof hvd) > 0) {
      printf("WARNING: HIP_VISIBLE_DEVICES=%s is set; device ordinals are remapped (the plan assumes it is unset)\n", hvd);
      out["HIP_VISIBLE_DEVICES"] = hvd;
    }
  }
  try {
    int n = 0;
    HIPC(hipGetDeviceCount(&n));
    const bool need_two = o.mode != "loopback";
    if (need_two && (o.dev0 < 0 || o.dev1 < 0 || o.dev0 >= n || o.dev1 >= n))
      throw std::runtime_error("need devices " + std::to_string(o.dev0) + " and " + std::to_string(o.dev1) +
                               ", runtime sees " + std::to_string(n));
    if (o.mode == "probe") {
      code = run_probe(o, out);
    } else if (o.mode == "loopback") {
      if (o.device < 0 || o.device >= n) throw std::runtime_error("loopback needs --device D (0.." + std::to_string(n - 1) + ")");
      J lj = J::obj();
      code = run_loopback(o, o.device, o.partner, lj);
      out["loopback"] = lj;
    } else if (o.mode == "pingpong") {
      J pj = J::obj();
      code = run_pingpong(o, pj);
      out["pingpong"] = pj;
    } else if (o.mode == "allreduce") {
      J aj = J::obj();
      code = run_allreduce_mode(o, false, aj);
      out["allreduce"] = aj;
    } else if (o.mode == "stress") {
      J sj = J::obj();
      code = run_allreduce_mode(o, true, sj);
      out["stress"] = sj;
    } else if (o.mode == "sweep") {
      code = run_sweep(o, out);
    } else {
      usage();
      printf("error: unknown mode '%s'\n", o.mode.c_str());
      return EXIT_ERR;
    }
  } catch (const std::exception& e) {
    printf("ERROR: %s\n", e.what());
    out["error"] = e.what();
    code = worse_code(code, EXIT_ERR);
  }
  if (g_stuck) {  // a watchdog fired: a stuck stream is a timeout (exit 3), even if it surfaced as an exception
    code = EXIT_TIMEOUT;
    out["stuck"] = true;
  }
  {
    std::lock_guard<std::mutex> lk(g_clock_mu);
    out["wallclock_check"] = g_clock_json;
  }
  out["exit_code"] = code;
  if (!o.json.empty()) write_json(o.json, out);
  printf("exit code %d (%s)\n", code,
         code == EXIT_OK ? "ok" : code == EXIT_MISMATCH ? "data mismatch" : code == EXIT_TIMEOUT ? "timeout/abort" : "error");
  fflush(stdout);
  if (g_stuck) std::_Exit(code);  // kernels may still be running: skip runtime teardown
  return code;
}
