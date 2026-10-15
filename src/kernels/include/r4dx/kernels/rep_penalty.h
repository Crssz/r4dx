// Repetition penalty on device logits (docs/lfm2-plan.md 5.4). Same plain C ABI as kernels.h: device
// pointers and the stream as int64_t. Implemented in src/kernels/src/rep_penalty.hip (its own hipcc object,
// r4dx_kernels.hip's flags), CPU twin in rep_penalty.hpp (bit-identical rule).
#pragma once

#include <cstdint>

extern "C" {

// HF RepetitionPenaltyLogitsProcessor, in place, for ONE fp32 logits row:
//   for each id in seen_ids[0, n_seen):  logits[id] = logits[id] < 0 ? logits[id] * p : logits[id] / p
// `seen_ids` is a device int32 list of DISTINCT token ids (the host keeps the set and uploads only deltas;
// a repeated id would be penalized twice -- a caller error, not checked). An id outside [0, vocab) is
// skipped (never written). p == 1 (or n_seen <= 0) returns without a launch, so the default-off path costs
// nothing. Must run BEFORE r4dx_topk_lse_f32 / the greedy argmax, so the summary path sees the penalized row.
// Preconditions (throw): logits and seen_ids non-null when n_seen > 0, p > 0 and finite, vocab > 0.
void r4dx_apply_repetition_penalty_f32(int64_t logits, int64_t seen_ids, int64_t n_seen, int64_t vocab,
                                       float penalty, int64_t stream);

}  // extern "C"