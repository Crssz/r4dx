// Repetition penalty, host side (docs/lfm2-plan.md 5.4). Header-only and HIP-free, like sampler.hpp.
//
// HF RepetitionPenaltyLogitsProcessor: for every DISTINCT token id seen so far (prompt + generated),
//   logit = logit < 0 ? logit * p : logit / p
// applied to the raw logits BEFORE temperature / top-k / top-p (HF's order), and in greedy decoding too.
// p == 1 is the exact identity -- callers skip the whole path then, so a request without the penalty
// takes today's code byte for byte (Qwen/Gemma). The device twin is r4dx_apply_repetition_penalty_f32
// (rep_penalty.h): same rule, same fp32 operations.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace r4dx::kernels {

// One logit: the rule above, in fp32 exactly as torch does (multiply for negatives, divide otherwise).
inline float PenalizeLogit(float logit, float p) { return logit < 0.0f ? logit * p : logit / p; }

// In-place over a host logits row. Ids outside [0, vocab) are skipped; `ids` must be distinct (a repeat
// would be penalized twice). No-op for p == 1.
inline void ApplyRepetitionPenalty(float* logits, int64_t vocab, const int32_t* ids, size_t n_ids, float p) {
  if (p == 1.0f) return;
  for (size_t i = 0; i < n_ids; ++i) {
    const int32_t id = ids[i];
    if (id < 0 || id >= vocab) continue;
    logits[id] = PenalizeLogit(logits[id], p);
  }
}

// The set of distinct token ids seen so far, kept incrementally by the host: O(1) per generated token, the
// id list is what a device kernel is handed (only the part added since the last upload, see `uploaded`).
class SeenTokens {
 public:
  explicit SeenTokens(int64_t vocab) : seen_(static_cast<size_t>(vocab > 0 ? vocab : 0), false) {}

  // True iff `id` was new. Ids outside [0, vocab) are ignored (and report false).
  bool Add(int32_t id) {
    if (id < 0 || static_cast<size_t>(id) >= seen_.size() || seen_[static_cast<size_t>(id)]) return false;
    seen_[static_cast<size_t>(id)] = true;
    ids_.push_back(id);
    return true;
  }
  void AddAll(const std::vector<int32_t>& ids) {
    for (int32_t id : ids) Add(id);
  }

  const std::vector<int32_t>& ids() const { return ids_; }
  size_t size() const { return ids_.size(); }
  bool contains(int32_t id) const {
    return id >= 0 && static_cast<size_t>(id) < seen_.size() && seen_[static_cast<size_t>(id)];
  }

  // Delta bookkeeping for a device mirror: ids_[uploaded, size()) are the ones not yet on the device.
  size_t uploaded() const { return uploaded_; }
  void MarkUploaded() { uploaded_ = ids_.size(); }

  void Apply(float* logits, int64_t vocab, float p) const {
    ApplyRepetitionPenalty(logits, vocab, ids_.data(), ids_.size(), p);
  }

 private:
  std::vector<bool> seen_;
  std::vector<int32_t> ids_;
  size_t uploaded_ = 0;
};

}  // namespace r4dx::kernels