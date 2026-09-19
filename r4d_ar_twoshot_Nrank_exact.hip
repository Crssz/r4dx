// r4d_ar_twoshot_Nrank_exact.hip: r4d_ar_twoshot_4rank_exact and r4d_ar_twoshot_8rank_exact 
// reduce-scatter + all-gather at 4 or 8 ranks, exact bf16/fp16/fp32 wire. Per-block pipelined
// handshake instead of a grid arrival election. Sends 2*(ws-1)/ws*N bytes per rank in two hops,
// so it covers the sizes where the one-shot's (ws-1)*N send is too heavy.
//
// Shard s = contiguous 1/ws of the message, owned by rank s. Phase 1: push my contribution to
// each peer-owned shard into the owner's slot[me] region A, flag 2s-1; reduce my shard fp32 in
// ascending rank order (own contribution from the input). Phase 2: reduced words go to my out
// AND every peer's slot[me] region B, flag 2s; assemble out from region B. Region A at 0, B at
// shard16; the slot stride must hold 2*shard16.
//
// SHARES the wide family's flag/seq/scratch set: every wide kernel publishes in the 2*seq flag
// space (this one adds the 2s-1 half-step) and advances the per-block counter once per AR, so
// parity and the monotonic spin survive any mixed one-shot/two-shot stream on one buffer set.
// The slot stride must hold the largest requirement of every family in the mix.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include <stdexcept>
#include <string>

template <typename T, int LANES, int WS>
__global__ void r4d_ar_twoshot_wide_exact_kernel(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, int shard16,
    int slot_stride16, int rank, int drain, int acq, int pub) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;  // per-block, replay-safe
  __syncthreads();
  const unsigned int s = s_seq;
  const int parity = (int)(s & 1u);

  const int w = (shard16 + nb - 1) / nb;   // block b owns words [w0,w1) of the SHARD space
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > shard16) w1 = shard16;

  // region A of my slot in every peer's scratch (B sits at +shard16)
  int4* pa[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  // ---- phase 1 push: my contribution to each peer-owned shard -> owner's region A
  for (int p = 0; p < NPEERS; ++p) {
    const int4* src = in4 + (size_t)peers.rank[p] * shard16;
    int4* dst = pa[p];
    for (int k = w0 + tid; k < w1; k += nt) dst[k] = src[k];
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

  // ---- reduce my shard (fp32, ascending rank, own contribution from the input), and phase 2
  // push: the reduced word goes to my out and every peer's region B in the same pass
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (in4 + (size_t)rank * shard16)
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  int4* my_out = out4 + (size_t)rank * shard16;
  for (int k = w0 + tid; k < w1; k += nt) {
    float acc[LANES];
#pragma unroll
    for (int j = 0; j < LANES; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
      const int4 v = src[r][k];
      const T* pv = reinterpret_cast<const T*>(&v);
#pragma unroll
      for (int j = 0; j < LANES; ++j) acc[j] += r4d_ar_to_f(pv[j]);
    }
    int4 o;
    T* po = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < LANES; ++j) r4d_ar_from_f(acc[j], po[j]);
    my_out[k] = o;
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) pa[p][shard16 + k] = o;  // region B
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  // ---- assemble: every peer-owned shard from my region B copies
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + shard16;
    int4* dst = out4 + (size_t)owner * shard16;
    for (int k = w0 + tid; k < w1; k += nt) dst[k] = srcB[k];
  }
}

// --- host side ---
template <int WS>
static void ar_twoshot_launch(const int64_t* peer_scratch, const int64_t* peer_flags, int64_t my_scratch,
                              int64_t my_flags, int64_t seq_ctrs, int64_t slot_stride16, int64_t inp,
                              int64_t out, int64_t n_elem, int64_t dtype, int64_t rank, int64_t stream_i,
                              int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq, int64_t pub) {
  constexpr int NPEERS = WS - 1;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_reduce: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  const int esize = (dtype == 2) ? 4 : 2;
  const int64_t nbytes = (int64_t)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  if (n_elem > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS)
    throw std::runtime_error("all_reduce: n_elem " + std::to_string(n_elem) +
                             " exceeds the twoshot band (" +
                             std::to_string(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) +
                             " elems); use the fallback path");
  const int64_t n16 = nbytes / 16;
  if (n16 % WS != 0)
    throw std::runtime_error("all_reduce: message (" + std::to_string(n16) +
                             " 16B words) does not shard evenly over " + std::to_string(WS) +
                             " ranks");
  const int64_t shard16 = n16 / WS;
  if (2 * shard16 > (int64_t)slot_stride16)
    throw std::runtime_error("all_reduce: shard (" + std::to_string(shard16) +
                             " 16B words) needs 2x inside the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > shard16) nb = (int)shard16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("all_reduce: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // scratch must back 2*WS slots of the declared stride, flags one word per (block, source
  // rank), seq one word per block -- checked against the allocations, not the arguments.
  const int64_t slot_bytes = 2L * WS * (int64_t)slot_stride16 * 16;
  const int64_t flag_bytes = (int64_t)nb * WS * (int64_t)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "all_reduce", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "all_reduce", "my_flags");
  r4d_ar_check_extent(seq_ctrs, (int64_t)nb * (int64_t)sizeof(unsigned int), "all_reduce", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "all_reduce", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "all_reduce", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
#define R4D_AR_TS_LAUNCH(T, LN)                                                       \
  r4d_ar_twoshot_wide_exact_kernel<T, LN, WS><<<nb, nt, 0, st>>>(                     \
      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \
      (int4*)out, (unsigned int*)seq_ctrs, (int)shard16, (int)slot_stride16,          \
      (int)rank, (int)drain, (int)acq, (int)pub)
  if (dtype == 0) R4D_AR_TS_LAUNCH(__hip_bfloat16, 8);
  else if (dtype == 1) R4D_AR_TS_LAUNCH(__half, 8);
  else if (dtype == 2) R4D_AR_TS_LAUNCH(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef R4D_AR_TS_LAUNCH
}

void r4d_ar_twoshot_4rank_exact(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                                int64_t peer_flags0, int64_t peer_flags1, int64_t peer_flags2,
                                int64_t my_scratch, int64_t my_flags, int64_t seq_ctrs,
                                int64_t slot_stride16, int64_t inp, int64_t out, int64_t n_elem, int64_t dtype,
                                int64_t rank, int64_t stream, int64_t nblocks, int64_t nthreads, int64_t drain,
                                int64_t acq, int64_t pub) {
  const int64_t ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const int64_t pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_twoshot_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}

void r4d_ar_twoshot_8rank_exact(int64_t peer_scratch0, int64_t peer_scratch1, int64_t peer_scratch2,
                                int64_t peer_scratch3, int64_t peer_scratch4, int64_t peer_scratch5,
                                int64_t peer_scratch6, int64_t peer_flags0, int64_t peer_flags1,
                                int64_t peer_flags2, int64_t peer_flags3, int64_t peer_flags4,
                                int64_t peer_flags5, int64_t peer_flags6, int64_t my_scratch,
                                int64_t my_flags, int64_t seq_ctrs, int64_t slot_stride16, int64_t inp,
                                int64_t out, int64_t n_elem, int64_t dtype, int64_t rank, int64_t stream,
                                int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq, int64_t pub) {
  const int64_t ps[7] = {peer_scratch0, peer_scratch1, peer_scratch2, peer_scratch3,
                      peer_scratch4, peer_scratch5, peer_scratch6};
  const int64_t pf[7] = {peer_flags0, peer_flags1, peer_flags2, peer_flags3,
                      peer_flags4, peer_flags5, peer_flags6};
  ar_twoshot_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub);
}
