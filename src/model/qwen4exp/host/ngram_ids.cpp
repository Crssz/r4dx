// NgramIds: see ngram_ids.h.
#include "r4dx/model/qwen4exp/ngram_ids.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace r4dx::model::qwen4exp {

NgramIds::NgramIds(const NgramSidecarHeader& h) : eos_(h.eos_token_id), mult_(h.multipliers) {
  for (size_t i = 0; i < kHeads; ++i) {
    if (h.head_offsets[i] > h.padded_rows || h.head_primes[i] > h.padded_rows - h.head_offsets[i])
      throw std::runtime_error("ngram ids: head " + std::to_string(i) + " covers rows [" + std::to_string(h.head_offsets[i]) +
                               ", " + std::to_string(h.head_offsets[i] + h.head_primes[i]) + ") beyond padded_rows " +
                               std::to_string(h.padded_rows));
    primes_[i] = h.head_primes[i];
    offsets_[i] = h.head_offsets[i];
  }
  max_token_ = std::numeric_limits<int64_t>::max();
  for (int64_t m : mult_) {
    if (m <= 0) throw std::runtime_error("ngram ids: non-positive multiplier");
    max_token_ = std::min(max_token_, std::numeric_limits<int64_t>::max() / m);
  }
  if (eos_ < 0 || eos_ > max_token_)
    throw std::runtime_error("ngram ids: eos token id " + std::to_string(eos_) + " is outside [0, " + std::to_string(max_token_) +
                             "] (int64 product overflow)");
  Reset();
}

void NgramIds::CheckToken(int64_t t, const char* where) const {
  if (t < 0 || t > max_token_)
    throw std::runtime_error(std::string("ngram ids: ") + where + " token id " + std::to_string(t) + " is outside [0, " +
                             std::to_string(max_token_) + "]");
}

void NgramIds::Reset() { state_.tokens.fill(eos_); }

void NgramIds::Restore(const State& s) {
  for (int64_t t : s.tokens) CheckToken(t, "snapshot");
  state_ = s;
}

void NgramIds::Generate(const int32_t* tokens, size_t n, int64_t* ids_out) {
  for (size_t i = 0; i < n; ++i) CheckToken(tokens[i], "input");

  // Rolling window over the history [c0, c1, tokens...]: b = token 1 back, a = token 2 back, and `seg` =
  // the number of tokens since the last EOS before the current position, saturated at 2 (all the hash
  // ever asks). The EOS itself ends a segment, so after an EOS seg restarts at 0.
  int64_t a = state_.tokens[0];
  int64_t b = state_.tokens[1];
  int seg = (a == eos_) ? 0 : 1;
  seg = (b == eos_) ? 0 : std::min(seg + 1, 2);

  const int64_t m0 = mult_[0], m1 = mult_[1], m2 = mult_[2];
  for (size_t i = 0; i < n; ++i) {
    const int64_t t = tokens[i];
    const int64_t s1 = seg >= 1 ? b : eos_;
    const int64_t s2 = seg >= 2 ? a : eos_;
    const uint64_t bi = static_cast<uint64_t>(t * m0) ^ static_cast<uint64_t>(s1 * m1);  // order 2
    const uint64_t tri = bi ^ static_cast<uint64_t>(s2 * m2);                            // order 3
    int64_t* out = ids_out + i * kHeads;
    constexpr size_t kPerOrder = NgramSidecarHeader::kHeadsPerNgram;
    for (size_t h = 0; h < kPerOrder; ++h) out[h] = static_cast<int64_t>(bi % primes_[h] + offsets_[h]);
    for (size_t h = kPerOrder; h < kHeads; ++h) out[h] = static_cast<int64_t>(tri % primes_[h] + offsets_[h]);
    a = b;
    b = t;
    seg = (t == eos_) ? 0 : std::min(seg + 1, 2);
  }

  if (n >= 2) {
    state_.tokens = {tokens[n - 2], tokens[n - 1]};
  } else if (n == 1) {
    state_.tokens = {state_.tokens[1], tokens[0]};
  }
}

}  // namespace r4dx::model::qwen4exp
