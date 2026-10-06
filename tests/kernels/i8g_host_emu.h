// tests/kernels/i8g_host_emu.h -- a host (plain C++) emulation of the device-side names int8_gemm_proto_kernels.h
// uses, so that test_int8_gemm_proto_emu.cpp can EXECUTE the prototype's kernel source on the CPU, with real
// threads: one OS thread per GPU thread, a workgroup barrier, per-wave collectives (the iu8 WMMA, the DPP-style
// partner fetch of the FWHT, shuffles), shared memory per workgroup, global atomics for the ticket. Several
// workgroups run concurrently, so the last-block-finishes protocol is exercised for real.
//
// What it is NOT: it knows nothing about the hardware. The WMMA is the software tile of i8p::EmuWmmaI8 in the layout
// this prototype defines (D: lane L, element e = row 8 (L >> 4) + e, column L & 15; A row r on lanes r and r + 16,
// B column c on lanes c and c + 16, the same 8 k in both halves), and the trellis decode is a deterministic STUB (a hash of
// the lane's words to eight f16 values per fragment): it checks that the kernel's plumbing (which words reach which
// lane and block, the quantizer, the LDS staging, the K slicing, the reduction, the ticket protocol, the FWHT
// epilogue) is right, not that the real decode or the hardware agree. The real decode is shipped and tested
// (r4d_trellis_dq.h); the hardware is the GPU selftest's job.
#pragma once

#ifndef I8G_EMU
#error "i8g_host_emu.h is the host emulation: compile with I8G_EMU defined (see test_int8_gemm_proto_emu.cpp)"
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace emu {

struct Barrier {
  std::mutex m;
  std::condition_variable cv;
  int n;
  int count = 0;
  unsigned gen = 0;
  explicit Barrier(int n_ = 1) : n(n_) {}
  void wait() {
    std::unique_lock<std::mutex> l(m);
    const unsigned g = gen;
    if (++count == n) {
      count = 0;
      ++gen;
      cv.notify_all();
    } else {
      cv.wait(l, [&] { return gen != g; });
    }
  }
};
struct Dim { unsigned x = 0, y = 0, z = 0; };
// a wave: its barrier and the exchange buffers of the collectives
struct Wave : Barrier {
  alignas(16) unsigned char xa[32][8];
  alignas(16) unsigned char xb[32][8];
  float xf[32];
  Wave() : Barrier(32) {}
};
struct Group {
  Barrier bar;
  std::unique_ptr<Wave[]> waves;
  alignas(16) unsigned char lds[65536];
  explicit Group(int nthreads) : bar(nthreads), waves(new Wave[nthreads / 32]) {}
};

inline thread_local Dim t_tid, t_bid;
inline thread_local Group* t_group = nullptr;
inline thread_local Wave* t_wave = nullptr;
inline thread_local int t_lane = 0;

inline void wg_barrier() { t_group->bar.wait(); }

// v_wmma_i32_16x16x16_iu8, signed x signed: a, b are the lane's 8 bytes (2 ints), c the lane's 8 accumulators
template <class V2, class V8>
inline V8 wmma_iu8(V2 a, V2 b, V8 c) {
  Wave& w = *t_wave;
  const int L = t_lane;
  std::memcpy(w.xa[L], &a, 8);
  std::memcpy(w.xb[L], &b, 8);
  w.wait();
  V8 d;
  const int col = L & 15;
  for (int e = 0; e < 8; ++e) {
    const int row = 8 * (L >> 4) + e;
    int s = c[e];
    for (int h = 0; h < 2; ++h)
      for (int p = 0; p < 8; ++p) s += (int)(signed char)w.xa[row + 16 * h][p] * (int)(signed char)w.xb[col + 16 * h][p];
    d[e] = s;
  }
  w.wait();
  return d;
}
inline float shfl_xor(float v, int m) {
  Wave& w = *t_wave;
  w.xf[t_lane] = v;
  w.wait();
  const float r = w.xf[t_lane ^ m];
  w.wait();
  return r;
}
// v_perm_b32: output byte j = byte (sel_j) of {s0 : s1}, 0..3 from s1, 4..7 from s0
inline unsigned perm(unsigned s0, unsigned s1, unsigned sel) {
  const unsigned long long cat = ((unsigned long long)s0 << 32) | s1;
  unsigned out = 0;
  for (int j = 0; j < 4; ++j) {
    const unsigned c = (sel >> (8 * j)) & 0xFFu;
    const unsigned byte = c < 8 ? (unsigned)((cat >> (8 * c)) & 0xFFull) : (c == 12 ? 0u : 0xFFu);
    out |= byte << (8 * j);
  }
  return out;
}

// run `fn(args...)` as a kernel: grid (gx, gy), `block` threads (a multiple of 32) per workgroup; workgroups run
// concurrently as far as the thread budget allows
template <class F, class... Args>
void run_block(unsigned bx, unsigned by, unsigned block, F fn, Args... args) {
  auto g = std::make_unique<Group>((int)block);
  std::vector<std::thread> th;
  th.reserve(block);
  for (unsigned t = 0; t < block; ++t)
    th.emplace_back([&, t] {
      t_tid = Dim{t, 0, 0};
      t_bid = Dim{bx, by, 0};
      t_group = g.get();
      t_wave = &g->waves[t / 32];
      t_lane = (int)(t & 31);
      fn(args...);
    });
  for (auto& x : th) x.join();
}
template <class F, class... Args>
void launch(unsigned gx, unsigned gy, unsigned block, F fn, Args... args) {
  const unsigned total = gx * gy;
  const unsigned conc = std::max(1u, std::min(total, 1536u / block));
  std::atomic<unsigned> next{0};
  std::vector<std::thread> workers;
  for (unsigned w = 0; w < conc; ++w)
    workers.emplace_back([&] {
      for (;;) {
        const unsigned id = next.fetch_add(1);
        if (id >= total) break;
        run_block(id % gx, id / gx, block, fn, args...);
      }
    });
  for (auto& t : workers) t.join();
}

}  // namespace emu

// ---- the device-side names --------------------------------------------------------------------------------------
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define __shared__
#define threadIdx (emu::t_tid)
#define blockIdx (emu::t_bid)
#define __syncthreads() emu::wg_barrier()
#define __threadfence() std::atomic_thread_fence(std::memory_order_seq_cst)
#define __shfl_xor(v, m, w) emu::shfl_xor((v), (m))
#define __builtin_amdgcn_readfirstlane(x) (x)
#define __builtin_amdgcn_sched_barrier(x) ((void)0)
#define __builtin_amdgcn_rcpf(x) (1.0f / (x))
#define __builtin_amdgcn_perm(a, b, s) emu::perm((a), (b), (s))
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, cl) emu::wmma_iu8((a), (b), (c))

inline unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }

typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 r4d_h2 __attribute__((ext_vector_type(2)));
typedef unsigned r4d_u32x4 __attribute__((ext_vector_type(4)));
struct uint2 { unsigned x, y; };
inline uint2 make_uint2(unsigned x, unsigned y) { return uint2{x, y}; }

// r4d_fwht128.h's partner fetch (the DPP row_xmask of lane l ^ MASK)
template <int MASK>
inline float r4d_fwht_partner(float v) { return emu::shfl_xor(v, MASK); }

// r4d_trellis_dq.h's lane maps, verbatim (pure integer math) ...
inline unsigned r4d_trellis_k4_off_ab(int lane) {
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  return (unsigned)(t * 32 + 4 * c + 2 * h) * 4u;
}
inline unsigned r4d_trellis_k4_off_p(int lane) {
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  return (unsigned)(t * 32 + ((4 * c + 2 * h + 31) & 31)) * 4u;
}
inline unsigned r4d_trellis_k5_word(int lane, int i) {
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  return (unsigned)(t * 40 + (5 * c - 2 + 3 * h + i + 40) % 40);
}
inline unsigned r4d_trellis_k5_shift(int lane) { return 16u * (unsigned)(lane >> 4); }
// ... and a STUB decode: a deterministic hash of the lane's words to 8 + 8 small f16 values (multiples of 1/16 in
// [-3.9, 3.9], like the codebook's range), so a lane that reads the wrong word, or a block that lands in the wrong
// place, shows in the result
inline void emu_stub_values(unsigned a, unsigned b, unsigned c, unsigned d, v8h& f0, v8h& f1) {
  for (int e = 0; e < 8; ++e) {
    unsigned h = a * 2654435761u ^ (b + 0x9E3779B9u) * 40503u ^ (c + 77u) * 9973u ^ (d + 5u) * 7919u ^ (unsigned)(e + 1) * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
    f0[e] = (_Float16)((float)((int)((h >> 9) & 0x7Fu) - 63) * 0.0625f);
    f1[e] = (_Float16)((float)((int)((h >> 19) & 0x7Fu) - 63) * 0.0625f);
  }
}
inline void r4d_trellis_k4_decode(unsigned P, unsigned A, unsigned B, v8h& f0, v8h& f1) { emu_stub_values(P, A, B, 4, f0, f1); }
inline void r4d_trellis_k5_decode(unsigned (&W)[5], unsigned sh, v8h& f0, v8h& f1) {
  emu_stub_values(W[0] ^ (W[1] << 3), W[2] + W[3] * 7u, W[4], sh + 5, f0, f1);
}
