// QSA layer, stage 2: indexer scoring and exact top-k block selection with the `recent` tie rule (docs plan Q23).
// Contracts and rounding points: include/r4dx/model/qwen4exp/qsa_kernels.h.
//
// Selection algorithm (qsa_select kernel; one workgroup of 1024 threads per query row):
//   scores are fp32 and >= +0 (the accumulation starts from +0.0f, so no -0.0f), so their IEEE bit patterns order
//   like unsigned integers. Let nb = complete blocks, K = topk (< nb on this path). The result is the set of K
//   blocks that are the largest under the total order (score descending, block index descending).
//   1. Radix select the K-th largest score S: four passes over the 8-bit digits of the key, most significant
//      first. Pass j histograms the digit j of the scores whose higher digits equal the prefix found so far (LDS
//      atomics), then one thread scans the 256 bins from the top to find the digit that contains the K-th
//      largest (`krem` counts how many of the candidates still have to be taken from the matching bin). After
//      pass 3 the prefix is S, n_eq = number of blocks scoring exactly S, and `krem` = how many of those n_eq
//      are inside the top K (1 <= krem <= n_eq); every block scoring > S is inside.
//   2. Among the n_eq blocks that tie at S the most recent ones win, so the n_eq - krem OLDEST tied blocks are
//      dropped. A pass over the blocks in ascending index order (1024 per step) numbers the tied blocks
//      (ordinal = count of earlier tied blocks) and keeps a block iff score > S or (score == S and
//      ordinal >= n_eq - krem); the kept blocks are numbered by a second exclusive prefix count (wave ballot +
//      LDS wave totals), which makes the output ascending and deterministic.
//   3. Block b kept as the r-th gets tokens ratio*b .. ratio*b + ratio - 1 at output slot ratio*r; the
//      incomplete tail p + 1 - ratio*nb (<= 3) tokens follow. sel_count = ratio*K + tail.
// Cost: 4 histogram passes + 1 compaction pass over nb fp32 scores in L2 (nb <= 65536 at 262144 context).
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/qsa_device.h"
#include "r4dx/model/qwen4exp/qsa_kernels.h"

namespace qsa = r4dx::model::qwen4exp::qsa;
using qsa::Bf16ToFloat;

namespace {

constexpr int kScoreThreads = 256;
constexpr int kSelThreads = 1024;
constexpr int kSelWaves = kSelThreads / 32;
constexpr int kIndexDim = 128;
constexpr int kIndexHeads = 4;
constexpr float kSqrtIndexDim = 11.313708498984761f;  // sqrt(128) as an fp32 divisor (HF: / math.sqrt(128))

// thread = one block key (one 256-byte row); RPT queries of consecutive rows share every key load.
template <int RPT>
__global__ __launch_bounds__(kScoreThreads) void ScoreKernel(Q4QsaScoreArgs a) {
  // [dim quad][RPT * 4 query-head slots][4 dims]: a wave reads one 16-byte slot per query head, all lanes the same
  __shared__ __attribute__((aligned(16))) float qlds[(kIndexDim / 4) * RPT * kIndexHeads * 4];
  const int r0 = blockIdx.y * RPT;
  int nb[RPT];
  int nb_max = 0;
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int r = r0 + i;
    nb[i] = 0;
    if (r < a.rows) {
      const int nbi = (a.positions[a.row0 + r] + 1) / a.ratio;
      nb[i] = nbi > a.topk ? nbi : 0;  // dense rows are not scored
    }
    nb_max = max(nb_max, nb[i]);
  }
  if (static_cast<int>(blockIdx.x) * kScoreThreads >= nb_max) return;  // uniform over the workgroup

  for (int idx = threadIdx.x; idx < RPT * kIndexHeads * kIndexDim; idx += kScoreThreads) {
    const int i = idx / (kIndexHeads * kIndexDim);
    const int h = (idx / kIndexDim) % kIndexHeads;
    const int d = idx % kIndexDim;
    float v = 0.0f;
    if (r0 + i < a.rows) v = Bf16ToFloat(a.idx_q[(static_cast<int64_t>(a.row0 + r0 + i) * kIndexHeads + h) * kIndexDim + d]);
    qlds[((d >> 2) * RPT * kIndexHeads + i * kIndexHeads + h) * 4 + (d & 3)] = v;
  }
  __syncthreads();

  const int b = blockIdx.x * kScoreThreads + threadIdx.x;
  if (b >= nb_max) return;
  float acc[RPT][kIndexHeads];
#pragma unroll
  for (int i = 0; i < RPT; ++i)
#pragma unroll
    for (int h = 0; h < kIndexHeads; ++h) acc[i][h] = 0.0f;

  const uint4* key = reinterpret_cast<const uint4*>(a.block_keys + static_cast<int64_t>(b) * kIndexDim);
#pragma unroll 2
  for (int c = 0; c < kIndexDim / 8; ++c) {
    const uint4 raw = key[c];
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    float kf[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      kf[2 * j] = __uint_as_float(w[j] << 16);
      kf[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
    }
#pragma unroll
    for (int qd = 0; qd < 2; ++qd) {
      const int dq = c * 2 + qd;
#pragma unroll
      for (int i = 0; i < RPT; ++i)
#pragma unroll
        for (int h = 0; h < kIndexHeads; ++h) {
          const float4 q = *reinterpret_cast<const float4*>(&qlds[(dq * RPT * kIndexHeads + i * kIndexHeads + h) * 4]);
          acc[i][h] += kf[qd * 4 + 0] * q.x;
          acc[i][h] += kf[qd * 4 + 1] * q.y;
          acc[i][h] += kf[qd * 4 + 2] * q.z;
          acc[i][h] += kf[qd * 4 + 3] * q.w;
        }
    }
  }
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    if (b < nb[i]) {
      float s = 0.0f;  // +0 start: relu results and their sum are never -0
#pragma unroll
      for (int h = 0; h < kIndexHeads; ++h) s += fmaxf(acc[i][h], 0.0f);
      a.scores[static_cast<int64_t>(r0 + i) * a.score_stride + b] = s / kSqrtIndexDim;
    }
  }
}

__global__ __launch_bounds__(kSelThreads) void SelectKernel(Q4QsaSelectArgs a) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t wave_sel[kSelWaves];
  __shared__ uint32_t wave_tie[kSelWaves];
  __shared__ uint32_t s_digit, s_above;
  __shared__ uint32_t s_neq;

  const int r = blockIdx.x;
  const int row = a.row0 + r;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int p = a.positions[row];
  const int nb = (p + 1) / a.ratio;
  if (nb <= a.topk) {
    if (tid == 0) a.sel_count[row] = -1;  // dense row
    return;
  }
  const uint32_t* scores = reinterpret_cast<const uint32_t*>(a.scores) + static_cast<int64_t>(r) * a.score_stride;

  // ---- 1. radix select of the K-th largest key ----
  uint32_t prefix = 0, mask = 0;
  uint32_t krem = static_cast<uint32_t>(a.topk);
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int b = tid; b < nb; b += kSelThreads) {
      const uint32_t key = scores[b];
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t running = 0;
      uint32_t digit = 0, above = 0;
      for (int dgt = 255; dgt >= 0; --dgt) {
        const uint32_t c = hist[dgt];
        if (running + c >= krem) {
          digit = static_cast<uint32_t>(dgt);
          above = running;
          break;
        }
        running += c;
      }
      s_digit = digit;
      s_above = above;
      if (pass == 3) s_neq = hist[digit];
    }
    __syncthreads();
    prefix |= s_digit << shift;
    mask |= 255u << shift;
    krem -= s_above;
    __syncthreads();  // everyone has read s_digit / s_above / hist before the next pass rewrites them
  }
  const uint32_t skip_ties = s_neq - krem;  // oldest tied blocks dropped

  // ---- 2. ascending compaction ----
  int32_t* out = a.sel_tokens + static_cast<int64_t>(row) * a.sel_stride;
  uint32_t sel_base = 0, tie_base = 0;
  const unsigned long long lane_lt = (1ull << lane) - 1ull;
  for (int b0 = 0; b0 < nb; b0 += kSelThreads) {
    const int b = b0 + tid;
    const bool in = b < nb;
    const uint32_t key = in ? scores[b] : 0u;
    const bool gt = in && key > prefix;
    const bool eq = in && key == prefix;
    const unsigned long long eq_mask = __ballot(eq);
    const uint32_t tie_in_wave = __popcll(eq_mask & lane_lt);
    // Selection needs the tie ordinal, which needs the earlier waves' tie counts first.
    if (lane == 0) wave_tie[wave] = __popcll(eq_mask);
    __syncthreads();
    uint32_t tie_before = tie_base, tie_total = 0;
    for (int w = 0; w < kSelWaves; ++w) {
      if (w < wave) tie_before += wave_tie[w];
      tie_total += wave_tie[w];
    }
    const bool keep = gt || (eq && (tie_before + tie_in_wave) >= skip_ties);
    const unsigned long long keep_mask = __ballot(keep);
    const uint32_t keep_in_wave = __popcll(keep_mask & lane_lt);
    if (lane == 0) wave_sel[wave] = __popcll(keep_mask);
    __syncthreads();
    uint32_t sel_before = sel_base, sel_total = 0;
    for (int w = 0; w < kSelWaves; ++w) {
      if (w < wave) sel_before += wave_sel[w];
      sel_total += wave_sel[w];
    }
    if (keep) {
      const int slot = static_cast<int>(sel_before + keep_in_wave) * a.ratio;
      for (int j = 0; j < a.ratio; ++j) out[slot + j] = b * a.ratio + j;
    }
    tie_base += tie_total;
    sel_base += sel_total;
    __syncthreads();  // wave_tie / wave_sel are rewritten by the next step
  }

  // ---- 3. the incomplete tail ----
  const int tail = p + 1 - nb * a.ratio;
  const int body = a.topk * a.ratio;
  if (tid < tail) out[body + tid] = nb * a.ratio + tid;
  if (tid == 0) a.sel_count[row] = body + tail;
}

}  // namespace

extern "C" hipError_t r4dx_q4_qsa_scores(const Q4QsaScoreArgs* a, hipStream_t stream) {
  if (a->rows <= 0 || a->blocks_max <= 0) return hipSuccess;
  const unsigned gx = static_cast<unsigned>((a->blocks_max + kScoreThreads - 1) / kScoreThreads);
  if (a->rows >= 4) {
    hipLaunchKernelGGL(ScoreKernel<4>, dim3(gx, static_cast<unsigned>((a->rows + 3) / 4)), dim3(kScoreThreads), 0,
                       stream, *a);
  } else {
    hipLaunchKernelGGL(ScoreKernel<1>, dim3(gx, static_cast<unsigned>(a->rows)), dim3(kScoreThreads), 0, stream, *a);
  }
  return hipGetLastError();
}

extern "C" hipError_t r4dx_q4_qsa_select(const Q4QsaSelectArgs* a, hipStream_t stream) {
  if (a->rows <= 0) return hipSuccess;
  if (a->ratio <= 0 || a->ratio > kSelThreads) return hipErrorInvalidValue;  // the tail (< ratio) is written one token per thread
  hipLaunchKernelGGL(SelectKernel, dim3(static_cast<unsigned>(a->rows)), dim3(kSelThreads), 0, stream, *a);
  return hipGetLastError();
}
