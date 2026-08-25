// r4d_ar_twoshot_Nrank_ti8.hip: r4d_ar_twoshot_4rank_ti8 -- reduce-scatter + all-gather with
// the tiered-int8 wire on both hops, at 4 ranks. Codec (tiers 1/2/3) from r4d_ar_ti8.h,
// transport from r4d_ar_twoshot_Nrank_exact.hip. The second hop costs 38/64 of its exact 
// bytes. Own contribution reads exact from the input at fold time.
//
// Cross-rank identity: the phase-2 encode input is the deterministic fp32 fold, so the wire 
// words are identical on every rank by construction, and the owner stores the decode of the
// words it published -- never the exact fold -- so every rank holds the same bytes for 
// every shard.
//
// Slot: region A = ti8 wire (38*sg bytes, codes | rec | scales), region B at align16(A end)
// a second wire image of the same layout; the stride must hold both.
#include "r4d.h"
#include "r4d_ar_ti8.h"
#include "r4d_ar_wide.h"
#include <stdexcept>
#include <string>

template <typename T, int WS>
__global__ void r4d_ar_twoshot_wide_ti8_kernel(
    const T* __restrict__ in, R4DArWidePeers<char, WS - 1> peers, char* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, T* __restrict__ out,
    unsigned int* __restrict__ seq_ctrs, int shard_elems, long slot_stride_bytes,
    long b_off_bytes, int rank, int drain, int acq, int pub) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;  // per-block, replay-safe
  __syncthreads();
  const unsigned int s = s_seq;
  const int parity = (int)(s & 1u);

  const int sg = shard_elems / TI8_GROUP;  // groups per shard
  const int rec_off = TI8_REC_OFF * sg, sc_off = TI8_SC_OFF * sg;  // wire layout: codes | rec | scales
  const int gpb = (sg + nb - 1) / nb;      // block b owns shard groups [gA, gB)
  const int gA = b * gpb;
  const int gB = (gA + gpb < sg) ? gA + gpb : sg;

  char* pslot[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pslot[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride_bytes;

  // ---- phase 1: encode my contribution to each peer-owned shard into its owner's
  // region A. My own shard's contribution stays exact, read from the input at fold time.
  for (int p = 0; p < NPEERS; ++p) {
    const T* src = in + (size_t)peers.rank[p] * shard_elems;
    for (int k = 4 * gA + tid; k < 4 * gB; k += nt) {
      const int lig = k & 3;
      const int g = k >> 2;
      uint2 codes;
      unsigned int rec;
      __half scale;
      ti8_encode_group<T>(src + k * TI8_EPL, lig, &codes, &rec, &scale);
      ((uint2*)pslot[p])[k] = codes;
      if (lig == 0) {
        ((unsigned int*)(pslot[p] + rec_off))[g] = rec;
        ((__half*)(pslot[p] + sc_off))[g] = scale;
      }
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

  // ---- reduce my shard: own contribution exact from the input, peers decoded from
  // region A, fp32 fold in ascending rank order; the reduced group is re-encoded and its
  // wire words go to every peer's region B, the owner keeping their decode.
  const T* my_in = in + (size_t)rank * shard_elems;
  T* my_out = out + (size_t)rank * shard_elems;
  for (int k = 4 * gA + tid; k < 4 * gB; k += nt) {
    const int lig = k & 3;
    const int g = k >> 2;
    float acc[TI8_EPL];
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i) acc[i] = 0.f;
    for (int r = 0; r < WS; ++r) {  // ascending source order, own exact from the input
      float v[TI8_EPL];
      if (r == rank) {
        ti8_load8(my_in + k * TI8_EPL, v);
      } else {
        const char* base =
            my_base + ((size_t)parity * WS + r) * (size_t)slot_stride_bytes;
        const uint2 c = ((const uint2*)base)[k];
        const unsigned int rc = ((const unsigned int*)(base + rec_off))[g];
        const float scl = __half2float(((const __half*)(base + sc_off))[g]);
        ti8_decode_group<T>(c, rc, scl, lig, v);
      }
#pragma unroll
      for (int i = 0; i < TI8_EPL; ++i) acc[i] += v[i];
    }
    uint2 codes;
    unsigned int rc2;
    __half scl2;
    ti8_encode_group_v<T>(acc, lig, &codes, &rc2, &scl2);
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) {
      char* bB = pslot[p] + b_off_bytes;
      ((uint2*)bB)[k] = codes;
      if (lig == 0) {
        ((unsigned int*)(bB + rec_off))[g] = rc2;
        ((__half*)(bB + sc_off))[g] = scl2;
      }
    }
    float v2[TI8_EPL];
    ti8_decode_group<T>(codes, rc2, __half2float(scl2), lig, v2);
    ti8_store8(my_out + k * TI8_EPL, v2);
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);

  // ---- assemble: decode each peer-owned shard from region B, the mirror of the owner's
  // roundtrip store.
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const char* baseB = my_base +
        ((size_t)parity * WS + owner) * (size_t)slot_stride_bytes + b_off_bytes;
    T* dst = out + (size_t)owner * shard_elems;
    for (int k = 4 * gA + tid; k < 4 * gB; k += nt) {
      const int lig = k & 3;
      const int g = k >> 2;
      const uint2 c = ((const uint2*)baseB)[k];
      const unsigned int rc2 = ((const unsigned int*)(baseB + rec_off))[g];
      const float scl = __half2float(((const __half*)(baseB + sc_off))[g]);
      float v[TI8_EPL];
      ti8_decode_group<T>(c, rc2, scl, lig, v);
      ti8_store8(dst + k * TI8_EPL, v);
    }
  }
}

// ---------------------------------------------------------------------------------- host side
template <int WS>
static void ar_twoshot_ti8_launch(const long* peer_scratch, const long* peer_flags,
                                  long my_scratch, long my_flags, long seq_ctrs,
                                  long slot_stride_bytes, long inp, long out, long n_elem,
                                  long dtype, long rank, long stream_i, long nblocks,
                                  long nthreads, long drain, long acq, long pub) {
  constexpr int NPEERS = WS - 1;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("ti8 ar: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  if (dtype != 0 && dtype != 1) throw std::runtime_error("ti8 ar: only bf16/fp16 payloads");
  if (n_elem % ((long)TI8_GROUP * WS) != 0)
    throw std::runtime_error("ti8 ar: n_elem must tile " + std::to_string(TI8_GROUP) +
                             "-elem groups over " + std::to_string(WS) + " shards");
  const int shard_elems = (int)(n_elem / WS);
  const int sg = shard_elems / TI8_GROUP;
  const long b_off = ((long)TI8_GROUP_BYTES * sg + 15) / 16 * 16;  // one wire image, 16B-aligned
  if (2 * b_off > slot_stride_bytes)
    throw std::runtime_error("ti8 ar: slot stride " + std::to_string(slot_stride_bytes) +
                             " too small for two wire images (needs " +
                             std::to_string(2 * b_off) + " bytes)");

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sg) nb = sg;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<char, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("ti8 ar: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (char*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // scratch must back 2*WS slots of the declared stride, flags one word per (block, source
  // rank), seq one word per block -- checked against the allocations, not the arguments.
  const long slot_bytes = 2L * WS * slot_stride_bytes;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "ti8 ar", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "ti8 ar", "my_flags");
  r4d_ar_check_extent(seq_ctrs, (long)nb * (long)sizeof(unsigned int), "ti8 ar", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "ti8 ar", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "ti8 ar", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
#define R4D_AR_TS8_LAUNCH(T)                                                            \
  r4d_ar_twoshot_wide_ti8_kernel<T, WS><<<nb, nt, 0, st>>>(                             \
      (const T*)inp, peers, (char*)my_scratch, (unsigned int*)my_flags, (T*)out,        \
      (unsigned int*)seq_ctrs, shard_elems, (long)slot_stride_bytes, b_off, (int)rank,  \
      (int)drain, (int)acq, (int)pub)
  if (dtype == 0) R4D_AR_TS8_LAUNCH(__hip_bfloat16);
  else R4D_AR_TS8_LAUNCH(__half);
#undef R4D_AR_TS8_LAUNCH
}

void r4d_ar_twoshot_4rank_ti8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                              long peer_flags0, long peer_flags1, long peer_flags2,
                              long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_twoshot_ti8_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride_bytes, inp,
                           out, n_elem, dtype, rank, stream, nblocks, nthreads, drain,
                           acq, pub);
}

// The tiered-int8 wire format, which the caller needs to size its packed scratch.
void r4d_ar_ti8_dims(int* group, int* group_bytes) {
  if (group)       *group       = TI8_GROUP;
  if (group_bytes) *group_bytes = TI8_GROUP_BYTES;
}
