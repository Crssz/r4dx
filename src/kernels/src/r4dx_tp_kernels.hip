// r4dx_tp_kernels.hip -- the tensor-parallel all-reduce (docs/tp.md 6.3.1): a port of
// tools/tp_bench/ar_bench.hip's flag(drain 3, acq 0) kernel and its helpers, renamed, with the trace
// code removed and the buffer in place. The protocol, the memory-ordering argument and the safety
// properties are ar_bench.hip's header comment; this file keeps its code line for line where it can.
//
// Second hipcc object of r4dx_kernels (src/kernels/CMakeLists.txt), compiled with the SAME flags as
// r4dx_kernels.hip: WGP mode, never -mcumode. Under WGP mode __syncthreads' workgroup release
// already waits for every wave's stores, so drain 3 is ISA-identical to no drain; it is correct
// because tid0's system-scope release store starts with `global_wb scope:SCOPE_SYS`, which is
// L2-wide and writes back the other waves' pushed lines (tools/tp_bench/README.md "drain 3 note").
// src/kernels/check_tp_isa.cmake fails the build unless the emitted ISA still shows exactly that.
//
// Slot reuse (both ranks, per block): rank A's block b writes B's slot p = s & 1 at call s. B's
// block b last read slot p at call s-2. A reaches call s only after its call s-1 finished, which
// needed B's flag >= s-1, which B posts only after its call s-2 kernel completed (same-stream
// order). So a slot is never overwritten while it is still being read -- provided every push
// follows the protocol, hence the sticky local abort (an aborted rank pushes nothing more).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/kernels/tp_kernels.h"

namespace {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

enum : u32 { ABORT_TIMEOUT = 1, ABORT_PROTOCOL = 2 };  // docs/tp.md 6.1 TpAbortCode
enum : u32 { PH_FLAG_WAIT = kR4dxTpPhaseFlagWait };
constexpr int kFlagStride = kR4dxTpFlagStride;

typedef R4dxTpStatus Status;

// =================================================================================================
// bf16 helpers (ar_bench.hip 527-569)
// =================================================================================================
__host__ __device__ __forceinline__ u32 fbits(float f) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __float_as_uint(f);
#else
  u32 u;
  __builtin_memcpy(&u, &f, 4);
  return u;
#endif
}
// fp32 -> bf16 bits, round-to-nearest-even (no NaN inputs on the all-reduce's data path: the
// operands are finite bf16 activations).
__host__ __device__ __forceinline__ u32 f2bf(float f) {
  u32 u = fbits(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return u >> 16;
}
// 32-bit integer mixer (lowbias32) -- test pattern only.
__host__ __device__ __forceinline__ u32 mix32(u32 x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}
__host__ __device__ __forceinline__ int pat_val(u64 call, u32 elem, int rank) {
  const u32 h = mix32(mix32((u32)call * 0x9E3779B1u + (u32)rank * 0xC2B2AE3Du) ^ (elem * 0x85EBCA77u));
  return (int)(h % 97u) - 48;
}
__host__ __device__ __forceinline__ u32 expected_bits(u64 call, u32 elem) {
  return f2bf((float)(pat_val(call, elem, 0) + pat_val(call, elem, 1)));
}

// =================================================================================================
// Device primitives (ar_bench.hip 574-589)
// =================================================================================================
__device__ __forceinline__ u32 ld_acq_sys(const u32* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u32 ld_rlx_sys(const u32* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rel_sys(u32* p, u32 v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_rlx_sys(u32* p, u32 v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u64 now_ticks() { return (u64)wall_clock64(); }

// ar_bench.hip 599-611; shared by the all-reduce and r4dx_tp_add_bf16 (emulation), so an emulated
// run adds with exactly the arithmetic of a real one.
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
// (ar_bench.hip 614-620)
__device__ __forceinline__ u32 ld_sticky(const Status* st) {
  return __hip_atomic_load(&st->sticky, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void mark_sticky(Status* st, u32 seq) {
  __hip_atomic_store(&st->sticky, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  atomicMin(&st->abort_seq_min, seq);
}

// Record a timeout / protocol violation: release-store MY abort word (host memory, system scope),
// set the rank's sticky word, then claim the Status record (first failure wins) in VRAM.
// (ar_bench.hip 624-637)
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
// A block bailed out because an abort word was set (secondary failure). (ar_bench.hip 639-642)
__device__ void note_abort_exit(Status* st, u32 seq) {
  atomicAdd(&st->n_abort_exits, 1u);
  mark_sticky(st, seq);
}

// Spin until (int)(*f - want) >= 0. (ar_bench.hip 651-676)
//   spin_acq = 0 (production): system-scope RELAXED polls, then one system-scope acquire fence after
//                              success (synchronizes-with the writer's release, same guarantee).
//   spin_acq = 1:              a system-scope ACQUIRE load on every poll.
// Every 16th poll also loads both abort words, issued right after the flag load so one wait covers
// all three. Returns 0 = reached, 1 = timed out, 2 = an abort word is set. *seen = last flag value.
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

// The kernel's parameter: R4dxTpArArgs with typed pointers (tp_bench's ArArgs), built by the host
// launcher and passed by value. Typed pointers matter: the backend promotes pointers it loads from
// the kernel-argument segment to the global address space (global_load/global_store), while an
// int64 -> pointer cast inside the kernel would leave every access a generic flat_* instruction.
struct ArArgs {
  uint4* buf;
  char* peer_mbox;
  const char* my_mbox;
  u32* peer_flags;
  const u32* my_flags;
  u32* my_abort;
  const u32* peer_abort;
  u32* seq;
  Status* st;
  int n16, w;
  u32 blk_stride, slot_stride;
  int drain, acq, spin_acq;
  u64 timeout;
  u32 channel;
};
ArArgs typed_args(const R4dxTpArArgs& r) {
  ArArgs a;
  a.buf = (uint4*)r.buf;
  a.peer_mbox = (char*)r.peer_mbox;
  a.my_mbox = (const char*)r.my_mbox;
  a.peer_flags = (u32*)r.peer_flags;
  a.my_flags = (const u32*)r.my_flags;
  a.my_abort = (u32*)r.my_abort;
  a.peer_abort = (const u32*)r.peer_abort;
  a.seq = (u32*)r.seq;
  a.st = (Status*)r.status;
  a.n16 = r.n16;
  a.w = r.w;
  a.blk_stride = r.blk_stride;
  a.slot_stride = r.slot_stride;
  a.drain = r.drain;
  a.acq = r.acq;
  a.spin_acq = r.spin_acq;
  a.timeout = r.timeout_ticks;
  a.channel = (u32)r.channel;
  return a;
}

// Entry (tid0): bump this block's seq (VRAM; only block b of this rank touches seq[b], launches are
// stream-ordered) and read the rank's sticky abort word in the same load batch. Returns true if the
// call must be skipped (the rank already failed or saw an abort). (ar_bench.hip 791-802)
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

// ar_bench.hip 804-893 (ar_flag_kernel), in place, trace removed, the channel recorded on failure.
__global__ void __launch_bounds__(1024) r4dx_tp_ar_flag_bf16_kernel(ArArgs a) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ u32 s_seq;
  __shared__ int s_fail;  // 0 ok, 1 timeout, 2 abort exit, 3 protocol violation, 4 skipped (sticky)
  if (tid == 0) {
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
  const uint4* in = a.buf + start;
  uint4* out = a.buf + start;
  uint4* dst = (uint4*)(a.peer_mbox + (size_t)slot * a.slot_stride + (size_t)b * a.blk_stride);
  const uint4* src = (const uint4*)(a.my_mbox + (size_t)slot * a.slot_stride + (size_t)b * a.blk_stride);

  // PUSH my chunk into the peer's slot: up to 4 words per thread. The VRAM loads are unconditional
  // (index clamped into the chunk, so they stay in flight together); only the stores are predicated.
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

  if (tid == 0) {
    st_rel_sys(&a.peer_flags[(size_t)b * kFlagStride], s);  // RELEASE: data before flag
    u32 seen = 0;
    int rc = spin_flag_ge(&a.my_flags[(size_t)b * kFlagStride], s, a.my_abort, a.peer_abort, a.timeout,
                          a.spin_acq, &seen);
    if (rc == 0 && (int)(seen - s) >= 2) {  // legal: s or s+1 only
      if ((ld_rlx_sys(a.my_abort) | ld_rlx_sys(a.peer_abort)) != 0u) rc = 2;  // peer pushed after an abort
      else {
        note_failure(a.my_abort, a.st, ABORT_PROTOCOL, (u32)b, s, PH_FLAG_WAIT, a.channel, seen);
        rc = 3;
      }
    }
    if (rc == 1) note_failure(a.my_abort, a.st, ABORT_TIMEOUT, (u32)b, s, PH_FLAG_WAIT, a.channel, seen);
    else if (rc == 2) note_abort_exit(a.st, s);
    s_fail = rc;
  }
  __syncthreads();

  if (s_fail == 0) {
    if (a.acq) __threadfence_system();
    // REDUCE: up to 4 words per thread, all host-memory (and VRAM) loads issued before any store, so a
    // thread has up to 4 PCIe reads in flight instead of one per word. In place: word k0 + j*nt is
    // the one this same thread pushed above, and every push read finished before the barriers.
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
  // (a failed block does NOT reduce: buf keeps this rank's partial, and the host reports the abort)
}

// EmulatedComm's add (docs/tp.md 6.5): same add_bf16x8, operand order (mine, peer).
__global__ void __launch_bounds__(256) r4dx_tp_add_bf16_kernel(uint4* __restrict__ inout,
                                                               const uint4* __restrict__ peer, u64 n16) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (u64)gridDim.x * blockDim.x)
    inout[i] = add_bf16x8(inout[i], peer[i]);
}

// WallClockRate validation (ar_bench.hip 685-692).
__global__ void r4dx_tp_clock_probe_kernel(u64* out, int iters) {
  if (threadIdx.x != 0) return;
  const u64 t0 = (u64)wall_clock64();
  for (int i = 0; i < iters; ++i) __builtin_amdgcn_s_sleep(127);
  const u64 t1 = (u64)wall_clock64();
  out[0] = t0;
  out[1] = t1;
}

// Copy the VRAM Status to the rank's MIRROR line in host memory (ar_bench.hip 728-732).
__global__ void r4dx_tp_publish_status_kernel(const u32* __restrict__ src, u32* dst, int nwords) {
  const int t = threadIdx.x;
  if (t < nwords) st_rlx_sys(dst + t, src[t]);
  __threadfence_system();
}

// ---- test and benchmark support (ar_bench.hip 695-759, 1042-1049) ------------------------------
__global__ void r4dx_tp_test_fill_hash_kernel(uint4* __restrict__ p, u64 n16, u32 salt) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (u64)gridDim.x * blockDim.x) {
    const u32 x = (u32)i * 4u ^ salt ^ (u32)(i >> 30);
    p[i] = make_uint4(mix32(x), mix32(x + 1u), mix32(x + 2u), mix32(x + 3u));
  }
}

// ring[k][j] = pattern(base_call + k, j, rank); grid.y = calls in the ring.
__global__ void r4dx_tp_test_gen_kernel(u16* __restrict__ ring, u64 base_call, u32 n_elem, int rank) {
  const u32 k = blockIdx.y;
  const u64 call = base_call + k;
  u16* row = ring + (size_t)k * n_elem;
  for (u32 j = blockIdx.x * blockDim.x + threadIdx.x; j < n_elem; j += gridDim.x * blockDim.x)
    row[j] = (u16)(fbits((float)pat_val(call, j, rank)) >> 16);  // small integers: exact
}

// Compare every output against the exact bf16 sum; count mismatches and keep the smallest
// (call, elem) with its got-bits in one 64-bit key via atomicMin. Each thread accumulates locally and
// issues one atomic pair at the end (tp_bench issued one per mismatching element, which takes tens
// of seconds on one contended address when a whole batch is wrong).
__global__ void r4dx_tp_test_verify_kernel(const u16* __restrict__ ring, u64 base_call, u32 n_elem, Status* st) {
  const u32 k = blockIdx.y;
  const u64 call = base_call + k;
  const u16* row = ring + (size_t)k * n_elem;
  u32 bad = 0;
  u64 key_min = ~0ull;
  for (u32 j = blockIdx.x * blockDim.x + threadIdx.x; j < n_elem; j += gridDim.x * blockDim.x) {
    const u32 got = row[j];
    if (got != expected_bits(call, j)) {
      ++bad;
      const u64 key = (call << 36) | ((u64)j << 16) | (u64)got;
      key_min = key < key_min ? key : key_min;
    }
  }
  if (bad != 0u) {
    atomicAdd(&st->mismatches, bad);
    atomicMin(&st->min_key, key_min);
  }
}

// Memory-bound decode filler: stream rd_words 16-B words (VRAM ring window) and read-modify-write
// wr_words 16-B words of a per-rank "state" buffer (leaves dirty L2 lines, like the GDN state update).
__global__ void __launch_bounds__(256) r4dx_tp_test_filler_kernel(const uint4* __restrict__ rd, u64 rd_words,
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

// Submission probe (docs/tp.md Appendix B N56): one thread stores the device wall clock to out[1],
// then release-stores `value` to out[0], both at system scope into mapped host memory, so a host
// thread polling out[0] sees the kernel START without making any HIP call.
__global__ void r4dx_tp_test_marker_kernel(u64* out, u64 value) {
  if (threadIdx.x != 0) return;
  __hip_atomic_store(out + 1, (u64)wall_clock64(), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(out, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Decode condition (b): same grid, same chunking as the all-reduce, buf = buf + local; no host traffic.
__global__ void __launch_bounds__(1024) r4dx_tp_test_standin_kernel(uint4* __restrict__ buf,
                                                                    const uint4* __restrict__ local, int n16,
                                                                    int w) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int start = b * w;
  const int end = (start + w < n16) ? start + w : n16;
  for (int k = start + tid; k < end; k += nt) buf[k] = add_bf16x8(buf[k], local[k]);
}

[[noreturn]] void Bad(const std::string& what) { throw std::invalid_argument(what); }

void CheckNt(const char* who, int nt) {
  if (nt < 32 || nt > 1024 || nt % 32 != 0)
    Bad(std::string(who) + ": nt = " + std::to_string(nt) + " (need a multiple of 32 in [32, 1024])");
}

}  // namespace

extern "C" {

void r4dx_tp_ar_flag_bf16(const R4dxTpArArgs* a, int nt, int64_t stream) {
  static const char* kWho = "r4dx_tp_ar_flag_bf16";
  if (a == nullptr) Bad(std::string(kWho) + ": null args");
  if (a->buf == 0 || a->peer_mbox == 0 || a->my_mbox == 0 || a->peer_flags == 0 || a->my_flags == 0 ||
      a->my_abort == 0 || a->peer_abort == 0 || a->seq == 0 || a->status == 0)
    Bad(std::string(kWho) + ": null pointer argument");
  if (a->n16 < 1 || a->nb < 1 || a->nb > static_cast<int>(kR4dxTpNbMax) || a->w < 1 ||
      static_cast<int64_t>(a->nb - 1) * a->w >= a->n16 ||
      static_cast<int64_t>(a->nb) * a->w < a->n16)
    Bad(std::string(kWho) + ": bad chunking n16 " + std::to_string(a->n16) + ", nb " + std::to_string(a->nb) +
        ", w " + std::to_string(a->w) + " (need nb in [1, 64] and (nb-1)*w < n16 <= nb*w)");
  if (a->blk_stride % kR4dxTpLineBytes != 0 || static_cast<int64_t>(a->blk_stride) < static_cast<int64_t>(a->w) * 16 ||
      static_cast<uint64_t>(a->slot_stride) < static_cast<uint64_t>(a->nb) * a->blk_stride)
    Bad(std::string(kWho) + ": bad mailbox strides blk " + std::to_string(a->blk_stride) + ", slot " +
        std::to_string(a->slot_stride));
  if ((a->drain != 1 && a->drain != 3) || (a->acq != 0 && a->acq != 1) || (a->spin_acq != 0 && a->spin_acq != 1))
    Bad(std::string(kWho) + ": drain must be 1|3, acq and spin_acq 0|1");
  if (a->timeout_ticks == 0) Bad(std::string(kWho) + ": timeout_ticks is 0");
  CheckNt(kWho, nt);
  hipLaunchKernelGGL(r4dx_tp_ar_flag_bf16_kernel, dim3(static_cast<unsigned>(a->nb)), dim3(static_cast<unsigned>(nt)),
                     0, (hipStream_t)stream, typed_args(*a));
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_add_bf16(int64_t inout, int64_t peer, int64_t n, int64_t stream) {
  if (n <= 0) return;
  if (n % 8 != 0 || inout == 0 || peer == 0)
    Bad("r4dx_tp_add_bf16: n = " + std::to_string(n) + " (need non-null buffers and n % 8 == 0)");
  const u64 n16 = static_cast<u64>(n) / 8;
  const u64 blocks = std::min<u64>((n16 + 255) / 256, 4096);
  hipLaunchKernelGGL(r4dx_tp_add_bf16_kernel, dim3(static_cast<unsigned>(blocks)), dim3(256), 0, (hipStream_t)stream,
                     (uint4*)inout, (const uint4*)peer, n16);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_clock_probe(int64_t out_u64x2, int iters, int64_t stream) {
  if (out_u64x2 == 0 || iters < 0) Bad("r4dx_tp_clock_probe: null output or negative iters");
  hipLaunchKernelGGL(r4dx_tp_clock_probe_kernel, dim3(1), dim3(32), 0, (hipStream_t)stream, (u64*)out_u64x2, iters);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_publish_status(int64_t status_dev, int64_t mirror_host, int64_t stream) {
  if (status_dev == 0 || mirror_host == 0) Bad("r4dx_tp_publish_status: null pointer");
  hipLaunchKernelGGL(r4dx_tp_publish_status_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream,
                     (const u32*)status_dev, (u32*)mirror_host, 16);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_test_gen_bf16(int64_t ring, uint64_t base_call, uint32_t n_elem, int calls, int rank, int64_t stream) {
  if (calls <= 0 || n_elem == 0) return;
  if (ring == 0) Bad("r4dx_tp_test_gen_bf16: null ring");
  const u32 gx = std::max<u32>(1, std::min<u32>(64, (n_elem + 255) / 256));
  hipLaunchKernelGGL(r4dx_tp_test_gen_kernel, dim3(gx, static_cast<unsigned>(calls)), dim3(256), 0,
                     (hipStream_t)stream, (u16*)ring, (u64)base_call, n_elem, rank);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_test_verify_bf16(int64_t ring, uint64_t base_call, uint32_t n_elem, int calls, int64_t rec,
                              int64_t stream) {
  if (calls <= 0 || n_elem == 0) return;
  if (ring == 0 || rec == 0) Bad("r4dx_tp_test_verify_bf16: null pointer");
  if (n_elem >= (1u << 20)) Bad("r4dx_tp_test_verify_bf16: n_elem must be < 2^20 (mismatch key packing)");
  const u32 gx = std::max<u32>(1, std::min<u32>(64, (n_elem + 255) / 256));
  hipLaunchKernelGGL(r4dx_tp_test_verify_kernel, dim3(gx, static_cast<unsigned>(calls)), dim3(256), 0,
                     (hipStream_t)stream, (const u16*)ring, (u64)base_call, n_elem, (Status*)rec);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_test_filler(int64_t rd, uint64_t rd_words, int64_t wr, uint64_t wr_words, int64_t sink_u32x256,
                         int grid, int64_t stream) {
  if (grid < 1 || sink_u32x256 == 0) Bad("r4dx_tp_test_filler: grid < 1 or null sink");
  hipLaunchKernelGGL(r4dx_tp_test_filler_kernel, dim3(static_cast<unsigned>(grid)), dim3(256), 0, (hipStream_t)stream,
                     (const uint4*)rd, (u64)rd_words, (uint4*)wr, (u64)wr_words, (u32*)sink_u32x256);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_test_fill_hash(int64_t p, uint64_t n16, uint32_t salt, int grid, int64_t stream) {
  if (n16 == 0) return;
  if (p == 0 || grid < 1) Bad("r4dx_tp_test_fill_hash: null pointer or grid < 1");
  hipLaunchKernelGGL(r4dx_tp_test_fill_hash_kernel, dim3(static_cast<unsigned>(grid)), dim3(256), 0,
                     (hipStream_t)stream, (uint4*)p, (u64)n16, salt);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_test_standin_bf16(int64_t buf, int64_t local, int n16, int nb, int w, int nt, int64_t stream) {
  if (buf == 0 || local == 0 || n16 < 1 || nb < 1 || w < 1) Bad("r4dx_tp_test_standin_bf16: bad arguments");
  CheckNt("r4dx_tp_test_standin_bf16", nt);
  hipLaunchKernelGGL(r4dx_tp_test_standin_kernel, dim3(static_cast<unsigned>(nb)), dim3(static_cast<unsigned>(nt)), 0,
                     (hipStream_t)stream, (uint4*)buf, (const uint4*)local, n16, w);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_tp_test_marker(int64_t host_u64x2, uint64_t value, int64_t stream) {
  if (host_u64x2 == 0) Bad("r4dx_tp_test_marker: null output");
  hipLaunchKernelGGL(r4dx_tp_test_marker_kernel, dim3(1), dim3(32), 0, (hipStream_t)stream, (u64*)host_u64x2,
                     (u64)value);
  R4DX_HIP_CHECK(hipGetLastError());
}

int r4dx_tp_test_filler_occupancy() {
  int occ = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, r4dx_tp_test_filler_kernel, 256, 0) != hipSuccess || occ < 1)
    occ = 4;
  (void)hipGetLastError();
  return occ;
}

}  // extern "C"
