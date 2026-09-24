// r4dx tensor-parallel device kernels (docs/tp.md 6.3.1): the two-rank all-reduce through pinned
// host memory that both GPUs map, plus its helpers. A port of tools/tp_bench/ar_bench.hip's
// flag(drain 3, acq 0) variant -- the protocol, the memory-ordering argument and the safety
// properties are that file's header comment and tools/tp_bench/README.md; only names changed, and
// the trace code is gone.
//
// Implemented in src/kernels/src/r4dx_tp_kernels.hip, the second hipcc object of r4dx_kernels,
// compiled with r4dx_kernels.hip's flags: WGP mode, NEVER -mcumode. The drain-3 argument relies on
// the WGP barrier lowering (README "drain 3 note"); src/kernels/check_tp_isa.cmake checks the
// emitted ISA at build time.
//
// C ABI like kernels.h: device pointers and streams as int64_t. Every entry point reports a bad
// argument or a failed launch by throwing (std::invalid_argument / r4dx::core::HipError), so a
// caller compiled with /EHsc must not rely on these being nothrow (tests/kernels/CMakeLists.txt's
// /EHc- comment); r4dx_tp's sources are built with /EHc- for that reason.
//
// Not counted by r4dx_kernel_launch_counter_get(): that counter serves the profiled paths, which
// tensor parallelism refuses (docs/tp.md 1.2).
#pragma once

#include <cstdint>
#include <cstring>

// ---- protocol constants (shared by the host side, src/model/tp/) -------------------------------
enum : uint32_t {
  kR4dxTpLineBytes = 128,  // every flag, abort word and per-block mailbox region starts on a line
  kR4dxTpFlagStride = 32,  // u32 per flag line
  kR4dxTpNbMax = 64,       // max blocks per all-reduce (flag lines per channel and rank)
};
// R4dxTpStatus::code and the ABORT words: docs/tp.md 6.1 TpAbortCode (1 timeout, 2 protocol).
// R4dxTpStatus::phase:
enum : uint32_t { kR4dxTpPhaseNone = 0, kR4dxTpPhaseFlagWait = 1 };

// Per-rank diagnostics in VRAM -- tp_bench's `Status`, identical layout (16 u32). The first failing
// block claims it (claimed 0 -> 1) and records code/block/seq/phase/got. For the all-reduce kernel
// `word` holds the CHANNEL of the failing call (tp_bench stored 0xffffffff there: its FLAG variant
// had no word; docs/tp.md Appendix B N30). `sticky` != 0 makes every later all-reduce of this rank,
// on either channel, skip the whole call. mismatches/min_key are only written by the test verify
// kernel below (on its own record, never an endpoint's).
struct R4dxTpStatus {
  uint32_t claimed;        // 0 until the first failing block claims the record
  uint32_t code;           // kAbortTimeout / kAbortProtocol
  uint32_t block;
  uint32_t seq;            // per-block seq of the failing call
  uint32_t phase;
  uint32_t word;           // all-reduce: the channel of the failing call
  uint32_t got;            // flag value observed at failure
  uint32_t n_timeouts;     // failing blocks
  uint32_t n_abort_exits;  // blocks that bailed because an abort word was set
  uint32_t mismatches;     // test verify kernel: mismatching elements
  uint32_t sticky;         // != 0 once any block of this rank failed or bailed
  uint32_t abort_seq_min;  // min seq at which a block failed / bailed / skipped (init 0xffffffff)
  uint64_t min_key;        // test verify kernel: min over mismatches of (call<<36 | elem<<16 | got)
  uint32_t n_skipped;      // blocks that skipped a whole call because `sticky` was set
  uint32_t resp_started;   // unused (tp_bench's loopback diagnostic; kept for the identical layout)
};
static_assert(sizeof(R4dxTpStatus) == 64, "R4dxTpStatus must be 16 u32");

// All-reduce arguments, passed BY VALUE to the kernel (docs/tp.md 6.3.1). Pointers are this
// device's view (hipHostGetDevicePointer) of the shared host region, or VRAM.
struct R4dxTpArArgs {
  int64_t buf;                 // uint16_t* bf16, n16*8 elements, in place: the push reads it, the
                               // reduce rewrites it; each 16-B word is read and written by the same
                               // thread
  int64_t peer_mbox, my_mbox;  // this channel's receive mailboxes (peer's: written by me)
  int64_t peer_flags, my_flags;  // u32*, one 128-B line per block
  int64_t my_abort, peer_abort;  // u32* in host memory
  int64_t seq;                 // u32[kR4dxTpNbMax] VRAM, this channel
  int64_t status;              // R4dxTpStatus* VRAM (one per rank, shared by both channels)
  int32_t n16, nb, w;          // payload 16-B words, blocks (the grid), words per block
  uint32_t blk_stride, slot_stride;  // bytes; blk_stride % 128 == 0
  int32_t drain, acq, spin_acq;      // production: 3, 0, 0
  uint64_t timeout_ticks;            // timeout_ms * WallClockRate (kHz)
  int32_t channel;                   // recorded in R4dxTpStatus::word on failure (N30)
  int32_t reserved;
};

extern "C" {

// One all-reduce: grid a->nb, block nt (production 256). Per block b (unchanged from tp_bench):
// s = ++seq[b] (sticky -> skip the whole call); slot = s & 1; push this block's words of buf into
// peer_mbox[slot][b]; drain; __syncthreads; tid0 release-stores peer_flags[b] = s at system scope
// (its global_wb scope:SCOPE_SYS writes back every wave's pushed lines); tid0 spins until
// (int)(my_flags[b] - s) >= 0, checking both abort words every 16 polls and wall_clock64 against
// timeout_ticks; >= s+2 with no abort set is a protocol violation; __syncthreads; reduce
// buf[k] = bf16_rne(f32(buf[k]) + f32(my_mbox[slot][b][k])). A block that times out release-stores
// *my_abort = 1 (timeout), sets `sticky`, records block/seq/phase/channel, and does not reduce.
// Argument checks (throw std::invalid_argument): non-null pointers, n16 >= 1, nb in [1, 64],
// w >= 1, (nb - 1) * w < n16 <= nb * w, blk_stride % 128 == 0 and >= w*16, slot_stride >=
// nb*blk_stride, nt a multiple of 32 in [32, 1024], drain in {1, 3}, acq/spin_acq in {0, 1},
// timeout_ticks > 0.
void r4dx_tp_ar_flag_bf16(const R4dxTpArArgs* a, int nt, int64_t stream);

// EmulatedComm's add (docs/tp.md 6.5): inout[i] = bf16_rne(f32(inout[i]) + f32(peer[i])), the SAME
// add_bf16x8 as the all-reduce, operand order (mine, peer). n bf16 elements, n % 8 == 0.
void r4dx_tp_add_bf16(int64_t inout, int64_t peer, int64_t n, int64_t stream);

// WallClockRate check (docs/tp.md 2.9 step 4): one thread reads wall_clock64 at entry and exit of a
// busy wait bounded by an ITERATION count (`iters` x s_sleep 127), never by the clock under test;
// out_u64x2 (VRAM) = {t0, t1}.
void r4dx_tp_clock_probe(int64_t out_u64x2, int iters, int64_t stream);

// Copies the 16 u32 of a VRAM R4dxTpStatus to a rank's MIRROR line in host memory (system-scope
// stores + a system fence).
void r4dx_tp_publish_status(int64_t status_dev, int64_t mirror_host, int64_t stream);

// ---- test and benchmark support (tests/kernels: test_tp_allreduce_*, tool_tp_ar_*) -------------
// tp_bench's data pattern: rank r's element j of call c is (hash(c, j, r) % 97) - 48 in bf16, so
// every sum of two is exact in bf16 and data stale by any number of calls matches only by chance.
// Host twins: r4dx_tp_test_pattern / r4dx_tp_test_expected_bits below (inline, same arithmetic).

// ring[k][j] = pattern(base_call + k, j, rank) for k < calls, j < n_elem (row stride n_elem).
void r4dx_tp_test_gen_bf16(int64_t ring, uint64_t base_call, uint32_t n_elem, int calls, int rank,
                           int64_t stream);
// Counts elements of ring[k][j] != expected(base_call + k, j) into rec->mismatches and keeps
// atomicMin of (call << 36 | j << 16 | got) in rec->min_key (rec: a test-owned R4dxTpStatus with
// min_key initialised to all ones). n_elem < 2^20, calls < 2^28.
void r4dx_tp_test_verify_bf16(int64_t ring, uint64_t base_call, uint32_t n_elem, int calls,
                              int64_t rec, int64_t stream);
// tp_bench's decode filler: stream rd_words 16-B words from `rd`, read-modify-write wr_words 16-B
// words of `wr` (leaves dirty L2 lines), grid `grid` x 256.
void r4dx_tp_test_filler(int64_t rd, uint64_t rd_words, int64_t wr, uint64_t wr_words,
                         int64_t sink_u32x256, int grid, int64_t stream);
// Fills n16 16-B words with a hash pattern (the filler's incompressible source ring).
void r4dx_tp_test_fill_hash(int64_t p, uint64_t n16, uint32_t salt, int grid, int64_t stream);
// tp_bench's decode condition (b): a local stand-in on the all-reduce's grid (nb blocks x nt, block
// b owns words [b*w, min((b+1)*w, n16))), buf = add_bf16x8(buf, local), in place; no host traffic.
void r4dx_tp_test_standin_bf16(int64_t buf, int64_t local, int n16, int nb, int w, int nt,
                               int64_t stream);
// hipOccupancyMaxActiveBlocksPerMultiprocessor of the filler at 256 threads (>= 1).
int r4dx_tp_test_filler_occupancy();
// Submission probe (tool_tp_submit_probe, docs/tp.md Appendix B N56): one thread writes the device
// wall_clock64 to host_u64x2[1], then release-stores `value` to host_u64x2[0], both at system scope.
// host_u64x2 is this device's view of mapped pinned host memory; a host thread polling [0] sees the
// kernel start with no HIP call of its own.
void r4dx_tp_test_marker(int64_t host_u64x2, uint64_t value, int64_t stream);

}  // extern "C"

// Host twins of the device pattern (tp_bench's pat_val / expected_bits).
inline uint32_t r4dx_tp_test_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}
inline int r4dx_tp_test_pattern(uint64_t call, uint32_t elem, int rank) {
  const uint32_t h = r4dx_tp_test_mix32(
      r4dx_tp_test_mix32(static_cast<uint32_t>(call) * 0x9E3779B1u +
                         static_cast<uint32_t>(rank) * 0xC2B2AE3Du) ^
      (elem * 0x85EBCA77u));
  return static_cast<int>(h % 97u) - 48;
}
// bf16 bits of a small integer (|v| < 256: exact).
inline uint16_t r4dx_tp_test_int_bf16(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  static_assert(sizeof(u) == sizeof(f), "");
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}
inline uint16_t r4dx_tp_test_expected_bits(uint64_t call, uint32_t elem) {
  return r4dx_tp_test_int_bf16(r4dx_tp_test_pattern(call, elem, 0) +
                               r4dx_tp_test_pattern(call, elem, 1));
}
