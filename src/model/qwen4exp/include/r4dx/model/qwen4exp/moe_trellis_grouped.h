// src/model/qwen4exp/include/r4dx/model/qwen4exp/moe_trellis_grouped.h
//
// Grouped (all selected experts in one launch) trellis decode MoE, M = 1. The kernels are instantiated for exactly one
// geometry: hidden 2560, expert intermediate 640 (gate_up N 1280, two parts), at most 16 experts per token, at most
// 1024 experts, trellis bits 3 or 4. Every launcher returns the launch's hipError_t (hipGetLastError()); a bits value
// outside {3, 4} returns hipErrorInvalidValue. Callers validate the geometry once (Qwen4ExpStage's constructor) and
// check the returned code with Check (common.h).
//
// Tickets: the GEMV kernels index the split-group tickets at `expert * 10` (gate_up) / `expert * 20` (down), so
// all_tickets / all_tickets_down must be dedicated zeroed arrays of experts * 10 / experts * 20 u32; a launch that
// completes leaves them zero again.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

namespace r4dx {
namespace model {
namespace qwen4exp {
// Input: ids [top_k] in descending score order, weights [top_k] in descending score order.
// Output: ids_sorted [top_k], weights_sorted [top_k] in ascending expert ID order.
hipError_t GroupedMoeDecodeSort(const int32_t* ids_in, const float* weights_in,
                                int32_t* ids_sorted_out, float* weights_sorted_out,
                                int top_k, void* stream);

// Transforms x [1, 2560] bf16 into A0 [top_k, 2560] fp16 and A1 [top_k, 2560] fp16
// for the top_k experts in a single launch.
hipError_t GroupedTrellisInputGateUp(const uint16_t* x, const int32_t* ids_sorted,
                                     const float* all_suh, uint16_t* out_a0, uint16_t* out_a1,
                                     int top_k, void* stream);

// Grouped trellis GEMV for gate_up (M=1, K=2560, N=1280, parts=2, top_k<=16).
// Output: out_c [top_k, 1280] bf16.
hipError_t GroupedTrellisGemvGateUp(int kb, const uint16_t* in_a0, const uint16_t* in_a1,
                                    const int32_t* ids_sorted,
                                    const uint32_t* all_w, const float* all_svh,
                                    uint16_t* out_c, float* ws, unsigned* all_tickets,
                                    int top_k, void* stream);

// Fuses silu(gate) * up (intermediate 640) with Trellis input transform for down
// into out_a0 [top_k, 640] fp16 in a single launch.
hipError_t GroupedSiluMulTrellisInputDown(const uint16_t* gu, const int32_t* ids_sorted,
                                          const float* all_suh_down, uint16_t* out_a0,
                                          int top_k, void* stream);

// Grouped trellis GEMV for down (M=1, K=640, N=2560, parts=1, top_k<=16).
// Output: out_yp [top_k, 2560] bf16.
hipError_t GroupedTrellisGemvDown(int kb, const uint16_t* in_a0,
                                  const int32_t* ids_sorted,
                                  const uint32_t* all_w_down, const float* all_svh_down,
                                  uint16_t* out_yp, float* ws, unsigned* all_tickets_down,
                                  int top_k, void* stream);

// Combines the top_k expert outputs and shared expert in-place into out [1, hidden] bf16 (any hidden; top_k <= 16).
hipError_t GroupedMoeDecodeCombine(const uint16_t* yp, const float* weights_sorted,
                                   const uint16_t* shared_y, const uint16_t* shared_gate_logit,
                                   uint16_t* out, int top_k, int hidden, void* stream);

// ---- qwen3_5_moe geometry (Ornith-1.5-122B): hidden 3072, expert intermediate 1024 (gate_up N 2048 = two parts of 1024),
// top_k <= 16, any expert count, trellis bits 3 or 4. The same kernels as above, instantiated for this shape; the Sort and
// Combine launchers above serve it unchanged. The split-group counts below size the workspaces and tickets:
//   ws (gate_up) >= top_k * kGroupedQ35GuWsPerSlot floats, ws (down) >= top_k * kGroupedQ35DownWsPerSlot floats,
//   all_tickets >= experts * kGroupedQ35GuTickets u32, all_tickets_down >= experts * kGroupedQ35DownTickets u32
// (dedicated zeroed arrays, a launch that completes leaves them zero again).
// The split-group counts are selectable at run time (GroupedQ35SetSplits, benchmarks only); the workspaces are sized for the largest.
// 4 / 4 is what the host-routed ApplyLinear picks at M = 1 for these shapes (linear.cpp FallbackTuning: SKG = clamp(128 / (N / 128), 1,
// 4) as a power of two, halved until (K / 16) % (SK * SKG * U) == 0), so the fp32 sums run in the same order and the bf16 outputs are
// bit-identical to that path. Other counts are within 3% of the speed (docs/qwen35moe-plan.md 7.1) but round differently.
constexpr int kGroupedQ35GuSkg = 4;
constexpr int kGroupedQ35DownSkg = 4;
constexpr int kGroupedQ35GuWsPerSlot = 12 * 2048;
constexpr int kGroupedQ35DownWsPerSlot = 8 * 3072;
constexpr int kGroupedQ35GuTickets = 2048 / 128;
constexpr int kGroupedQ35DownTickets = 3072 / 128;

// Sort for the router-order combine: ids_sorted_out[rank[j]] = ids_in[j]. (Weights stay in router order.)
hipError_t GroupedMoeDecodeSortRanked(const int32_t* ids_in, int32_t* ids_sorted_out, int32_t* rank_out, int top_k, void* stream);
// out[d] = bf16( sum_j yp[rank[j]][d] * weights[j] (j in router order, fp32) + sigmoid-gated shared expert ): the host-routed
// combine's arithmetic, bit for bit.
hipError_t GroupedMoeDecodeCombineRouterOrder(const uint16_t* yp, const float* weights, const int32_t* rank, const uint16_t* shared_y,
                                              const uint16_t* shared_gate_logit, uint16_t* out, int top_k, int hidden,
                                              void* stream);

// Benchmark hook: the split-group counts of the next gate_up / down GEMV launches (gate_up 2,3,4,6,8,12; down 1,2,4,8). Process-wide,
// not thread safe; the defaults are kGroupedQ35GuSkg / kGroupedQ35DownSkg.
hipError_t GroupedQ35SetSplits(int gate_up_skg, int down_skg);

// x [1, 3072] bf16 -> A0 / A1 [top_k, 3072] fp16 (the two gate_up parts' sign vectors, all_suh [experts, 2 * 3072]).
hipError_t GroupedTrellisInputGateUpQ35(const uint16_t* x, const int32_t* ids_sorted, const float* all_suh,
                                        uint16_t* out_a0, uint16_t* out_a1, int top_k, void* stream);
// K 3072, N 2048 (parts 1024 + 1024) -> out_c [top_k, 2048] bf16. all_w holds the experts back to back (N * K * kb / 32 words).
hipError_t GroupedTrellisGemvGateUpQ35(int kb, const uint16_t* in_a0, const uint16_t* in_a1, const int32_t* ids_sorted,
                                       const uint32_t* all_w, const float* all_svh, uint16_t* out_c, float* ws,
                                       unsigned* all_tickets, int top_k, void* stream);
// silu(gate) * up (gate = gu[0, 1024), up = gu[1024, 2048), silu rounded to bf16 first like HF) fused with the down input
// transform: gu [top_k, 2048] bf16 -> out_a0 [top_k, 1024] fp16.
hipError_t GroupedSiluMulTrellisInputDownQ35(const uint16_t* gu, const int32_t* ids_sorted, const float* all_suh_down,
                                             uint16_t* out_a0, int top_k, void* stream);
// K 1024, N 3072 -> out_yp [top_k, 3072] bf16.
hipError_t GroupedTrellisGemvDownQ35(int kb, const uint16_t* in_a0, const int32_t* ids_sorted, const uint32_t* all_w_down,
                                     const float* all_svh_down, uint16_t* out_yp, float* ws, unsigned* all_tickets_down,
                                     int top_k, void* stream);

// ---- qwen3_5_moe at TP=2 (docs/qwen35moe-tp-design.md 3.4): a rank holds half of every expert (expert width 512, shared width 512), the same
// kernels instantiated for it: gate_up K 3072 / N 1024 (parts 512 + 512, n_split 512) and down K 512 / N 3072. The input transform of
// x (GroupedTrellisInputGateUpQ35: K 3072, sign vectors replicated), Sort and Combine are the launchers above, unchanged. Workspaces / tickets:
//   ws (gate_up) >= top_k * kGroupedQ35Tp2GuWsPerSlot floats, ws (down) >= top_k * kGroupedQ35Tp2DownWsPerSlot (= kGroupedQ35DownWsPerSlot),
//   all_tickets >= experts * kGroupedQ35Tp2GuTickets u32, all_tickets_down >= experts * kGroupedQ35Tp2DownTickets u32.
// all_w holds the rank's experts back to back (N * K * kb / 32 words each at the RANK's N, K); all_suh_down is [experts, 512].
constexpr int kGroupedQ35Tp2Inter = 512;
constexpr int kGroupedQ35Tp2GuSkg = 4;
constexpr int kGroupedQ35Tp2DownSkg = 4;
constexpr int kGroupedQ35Tp2GuWsPerSlot = 12 * 1024;
constexpr int kGroupedQ35Tp2DownWsPerSlot = 8 * 3072;
constexpr int kGroupedQ35Tp2GuTickets = 1024 / 128;
constexpr int kGroupedQ35Tp2DownTickets = 3072 / 128;
// Benchmark hook (the P2 sweep): the split-group counts of the next TP=2 launches (gate_up 2,3,4,6,8,12; down 1,2,4,8). Process-wide.
hipError_t GroupedQ35Tp2SetSplits(int gate_up_skg, int down_skg);
// K 3072, N 1024 (parts 512 + 512) -> out_c [top_k, 1024] bf16.
hipError_t GroupedTrellisGemvGateUpQ35Tp2(int kb, const uint16_t* in_a0, const uint16_t* in_a1, const int32_t* ids_sorted,
                                          const uint32_t* all_w, const float* all_svh, uint16_t* out_c, float* ws,
                                          unsigned* all_tickets, int top_k, void* stream);
// gu [top_k, 1024] bf16 -> out_a0 [top_k, 512] fp16.
hipError_t GroupedSiluMulTrellisInputDownQ35Tp2(const uint16_t* gu, const int32_t* ids_sorted, const float* all_suh_down,
                                                uint16_t* out_a0, int top_k, void* stream);
// K 512, N 3072 -> out_yp [top_k, 3072] bf16.
hipError_t GroupedTrellisGemvDownQ35Tp2(int kb, const uint16_t* in_a0, const int32_t* ids_sorted, const uint32_t* all_w_down,
                                        const float* all_svh_down, uint16_t* out_yp, float* ws, unsigned* all_tickets_down,
                                        int top_k, void* stream);
// ---- qwen3_5_moe geometry, M ROWS (a speculative verify window): 2..16 tokens at once, ~top_k * rows (token, expert) pairs over
// the distinct experts they pick. The M = 1 kernels above run one expert per grid slot and fill one of the 16 rows of every WMMA
// tile; these put an expert's tokens in the tile's rows (the activation row of lane l is the group's l-th pair), so each routed
// expert's weights are decoded and streamed ONCE however many of the rows chose it. Nothing leaves the device: the plan kernel
// sorts the pairs by expert and writes the groups to the device, the grids are launched for the largest possible group count
// and the blocks past the real one return at once.
//
// The plan (int32, kMoePlanInts of them):
//   [kMoePlanNGroups]                   number of distinct experts
//   [kMoePlanPairExpert + p]            expert of sorted pair p (pairs sorted by (expert, source slot))
//   [kMoePlanPairToken + p]             token row of sorted pair p
//   [kMoePlanPosOfSrc + s]              sorted position of source slot s = token * top_k + j (j = router rank)
//   [kMoePlanGrpExpert/Start/Count + g] group g: expert, first sorted pair, number of pairs (<= 16)
constexpr int kMoeRowsMax = 16;
constexpr int kMoePlanMaxPairs = 256;  // rows * top_k
constexpr int kMoePlanNGroups = 0;
constexpr int kMoePlanPairExpert = 16;
constexpr int kMoePlanPairToken = kMoePlanPairExpert + kMoePlanMaxPairs;
constexpr int kMoePlanPosOfSrc = kMoePlanPairToken + kMoePlanMaxPairs;
constexpr int kMoePlanGrpExpert = kMoePlanPosOfSrc + kMoePlanMaxPairs;
constexpr int kMoePlanGrpStart = kMoePlanGrpExpert + kMoePlanMaxPairs;
constexpr int kMoePlanGrpCount = kMoePlanGrpStart + kMoePlanMaxPairs;
constexpr int kMoePlanInts = kMoePlanGrpCount + kMoePlanMaxPairs;
// Split-group counts of the M-row GEMMs (1, 2 or 4): ws floats >= max_groups * skg * 16 * N, tickets u32 >= max_groups * N / 128.
// With skg == kGroupedQ35GuSkg / kGroupedQ35DownSkg (4 / 4, the M = 1 kernels' counts) a row's k-split sums run in the M = 1 kernel's order
// and the output equals the M = 1 grouped kernel's bit for bit, row by row.
constexpr int kMoeRowsMaxSkg = 4;

// ids: int32 [rows * top_k] (router order per token) -> plan. rows * top_k <= kMoePlanMaxPairs.
hipError_t GroupedMoePlanRows(const int32_t* ids, int32_t* plan, int rows, int top_k, void* stream);
// x [rows, 3072] bf16 -> A0 / A1 [pairs, 3072] fp16 (row p: the sorted pair p's token transformed with its expert's sign vectors).
hipError_t GroupedTrellisInputGateUpQ35Rows(const uint16_t* x, const int32_t* plan, const float* all_suh, uint16_t* out_a0,
                                            uint16_t* out_a1, int n_pairs, void* stream);
// -> out_c [pairs, 2048] bf16. `max_groups` = min(experts, pairs): the grid is launched for it.
hipError_t GroupedTrellisGemvGateUpQ35Rows(int kb, const uint16_t* in_a0, const uint16_t* in_a1, const int32_t* plan,
                                           const uint32_t* all_w, const float* all_svh, uint16_t* out_c, float* ws,
                                           unsigned* all_tickets, int max_groups, int skg, void* stream);
// gu [pairs, 2048] bf16 -> out_a0 [pairs, 1024] fp16 (silu rounded to bf16 first, like the M = 1 kernel).
hipError_t GroupedSiluMulTrellisInputDownQ35Rows(const uint16_t* gu, const int32_t* plan, const float* all_suh_down,
                                                 uint16_t* out_a0, int n_pairs, void* stream);
// -> out_yp [pairs, 3072] bf16.
hipError_t GroupedTrellisGemvDownQ35Rows(int kb, const uint16_t* in_a0, const int32_t* plan, const uint32_t* all_w_down,
                                         const float* all_svh_down, uint16_t* out_yp, float* ws, unsigned* all_tickets_down,
                                         int max_groups, int skg, void* stream);
// out[t, d] = bf16( sum_j yp[posOfSrc[t * top_k + j], d] * weights[t * top_k + j] (j in router order, fp32)
//                   + Rb(Rb(sigmoid(shared_gate_logit[t])) * shared_y[t, d]) ): the M = 1 router-order combine, per token.
hipError_t GroupedMoeCombineRowsRouterOrder(const uint16_t* yp, const float* weights, const int32_t* plan, const uint16_t* shared_y,
                                            const uint16_t* shared_gate_logit, uint16_t* out, int rows, int top_k, int hidden,
                                            void* stream);

}  // namespace qwen4exp
}  // namespace model
}  // namespace r4dx

