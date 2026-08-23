// r4d_ar_oneshot_2rank_exact.hip: r4d_ar_oneshot_2rank_exact -- multi-block one-shot P2P-BAR
// all-reduce, exactly two ranks, for dual R9700 (gfx1201/RDNA4, PCIe). "One-shot" is the
// algorithm class: each rank pushes its whole input, so the scratch is sized by the full message
// and there is no ring or reduce-scatter stage. "2rank" is a hard constraint, not a tuning
// choice -- the handshake is one peer flag per block, not a tree. Pure HIP behind the C ABI in
// r4d.h (wheel-ROCm ships no rocThrust, which torch/extension.h needs); the caller passes raw
// data_ptr / numel / dtype / stream. r4d_module.hip is the pybind surface.
//
// Design:
//  * PUSH model: each rank writes ITS input into the PEER's IPC scratch, then reduces
//    (local input + peer's data now in MY scratch) -> out. Only scratch+flags are
//    IPC-shared; input/output stay local -> no cudagraph buffer registration needed.
//  * MULTI-BLOCK, per-chunk independent flags (NO grid barrier): block b owns a
//    contiguous chunk of 16B words -> pushes it to peer, sets peer_flags[b], spins on
//    my_flags[b], reduces its chunk. Spreads the push across CUs (PCIe saturation) and
//    the reduce across CUs (HBM saturation): the single-block bandwidth fix.
//  * cudagraph-safe: seq is a DEVICE-resident PER-BLOCK counter the kernel increments.
//    Both ranks execute the identical graph sequence (SPMD) so seq_ctr[b] stays in
//    lockstep across ranks regardless of which batch-size graphs ran.
//  * race-free reuse: scratch DOUBLE-BUFFERED by (seq&1); same-stream serialization +
//    the per-block flag handshake guarantee a slot is drained before reuse.
//  * FINE-GRAINED memory (hipDeviceMallocFinegrained) => uncached => posted straight to
//    the fabric => SCOPE_SYS release/acquire skip the L2 writeback. drain/acq explicit
//    fences are tunable (all-thread system fences; default drain=1/acq=0 on fine-grained).
#include "r4d.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// vectorized 128-bit add: an int4 packs LANES elements of T; fp32-accumulate reduce.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns words [start,end) of the 16B-word
// space and drives its own flag handshake; fully independent, no cross-block sync.
template <typename T, int LANES>
__global__ void r4d_ar_oneshot_2rank_exact_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;  // per-block, replay-safe
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s & 1u);
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // push chunk -> peer slot, 4x-unrolled for memory-level parallelism (PCIe saturation)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = in4[k];
  }
  // drain modes (ordering of the pushes vs the release flag, so the peer sees data first):
  //  0 = none (relies purely on PCIe posted-write ordering + the release flag)
  //  1 = all-thread __threadfence_system  (safe but emits an L2 wb-invalidate per
  //      wavefront (WASTED on fine-grained/uncached memory; floods at high nt*nb)
  //  2 = single-thread __threadfence_system (cheap but only drains wavefront 0's stores)
  //  3 = all-thread store-completion wait (s_wait_storecnt 0): every wavefront drains
  //      its posted writes into the fabric with NO L2 wb-invalidate; PCIe keeps posted
  //      writes to the same peer ordered, so the later release flag lands after the data.
  //      Correct for FINE-GRAINED memory (uncached; coarse would leave data in L2).
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (load_sys_acq(&my_flags[b]) < s) { if (++z > R4D_AR_SPIN_MAX) break; }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  // reduce chunk (vectorized, 4x-unrolled): out = local input + peer input (staged in local scratch)
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          ms[k]);
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     ms[k + nt]);
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], ms[k + 2 * nt]);
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], ms[k + 3 * nt]);
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], ms[k]);
  }
}

// The IPC handle is returned through a caller-owned buffer rather than a std::string so this
// stays a C ABI; R4D_AR_HANDLE_BYTES is the fixed size the header promises.
static_assert(sizeof(hipIpcMemHandle_t) <= R4D_AR_HANDLE_BYTES, "ipc handle larger than the ABI buffer");

long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len) {
  void* p = nullptr;
  if (finegrained) HIPOK(hipExtMallocWithFlags(&p, size, hipDeviceMallocFinegrained));
  else HIPOK(hipMalloc(&p, size));
  HIPOK(hipMemset(p, 0, size));
  hipIpcMemHandle_t h;
  HIPOK(hipIpcGetMemHandle(&h, p));
  std::memcpy(out_handle, &h, sizeof(h));
  *out_len = (int)sizeof(h);
  return (long)p;
}
long r4d_ar_ipc_open(const char* handle, int len) {
  hipIpcMemHandle_t h;
  if (len != (int)sizeof(h)) throw std::runtime_error("bad ipc handle size");
  std::memcpy(&h, handle, sizeof(h));
  void* p = nullptr;
  HIPOK(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess));
  return (long)p;
}
void r4d_ar_ipc_free(long p) { if (p) (void)hipFree((void*)p); }
void r4d_ar_ipc_memzero(long p, long size) { if (p) HIPOK(hipMemset((void*)p, 0, size)); }

// single-process 2-GPU testing only: enable direct peer access (the runtime path uses IPC handles)
void r4d_ar_ipc_enable_peer(long peer) {
  hipError_t e = hipDeviceEnablePeerAccess((int)peer, 0);
  if (e == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); return; }
  if (e != hipSuccess)
    throw std::runtime_error(std::string("enable_peer: ") + hipGetErrorString(e));
}

void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags,
                            long my_flags, long seq_ctrs, long slot_stride16,
                            long inp, long out, long n_elem, long dtype,
                            long stream_i, long nblocks, long nthreads,
                            long drain, long acq) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
#define L(T, LN) r4d_ar_oneshot_2rank_exact_kernel<T, LN><<<nb, nt, 0, st>>>(              \
    (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,        \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16,              \
    (int)slot_stride16, (int)drain, (int)acq)
  if (dtype == 0) L(__hip_bfloat16, 8);
  else if (dtype == 1) L(__half, 8);
  else if (dtype == 2) L(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef L
}

