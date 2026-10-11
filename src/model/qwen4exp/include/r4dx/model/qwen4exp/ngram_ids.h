// r4dx::model::qwen4exp::NgramIds -- the 16 n-gram row ids of every token (docs/qwen4exp-plan.md 2.3, 4.8).
//
// Bit-exact C++ port of tools/reference/qwen4exp/ngram_ids.py (itself a port of
// `Qwen4ExpTextNGramEmbedding` in transformers' modular_qwen4_exp.py). Everything the hash needs (the
// splitmix64 multipliers, head primes and offsets, EOS id) comes from the sidecar header, nothing is
// recomputed here.
//
// For the token at position p, with s1 / s2 the tokens 1 / 2 positions back, replaced by EOS when they lie
// before the most recent EOS of the history (EOS-delimited segments; the EOS token itself belongs to the
// segment it ends):
//     head h in [0, 8):   id = ((t*m0) ^ (s1*m1))            % prime[h] + offset[h]
//     head h in [8, 16):  id = ((t*m0) ^ (s1*m1) ^ (s2*m2))  % prime[h] + offset[h]
// All in int64: the multipliers are bounded by INT64_MAX / vocab_size, so no product overflows for token ids
// below the vocabulary; Generate throws for any token whose product would overflow (or that is negative)
// instead of wrapping.
//
// The generator carries the last 2 tokens of the history across calls, so feeding a sequence in any
// chunking gives the ids of one call over the whole sequence. Not thread-safe.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "r4dx/model/qwen4exp/ngram_sidecar.h"

namespace r4dx::model::qwen4exp {

class NgramIds {
 public:
  static constexpr size_t kHeads = NgramSidecarHeader::kHeads;
  static constexpr size_t kContext = NgramSidecarHeader::kNgramSize - 1;

  // The last 2 tokens of the history; the start-of-sequence state is {EOS, EOS}. This is exactly the
  // `previous_context` of the Python reference and what a prefix checkpoint has to keep.
  struct State {
    std::array<int64_t, kContext> tokens{};
    bool operator==(const State& o) const { return tokens == o.tokens; }
  };

  // Throws std::runtime_error when the header cannot index its own table (offset + prime > padded_rows)
  // or the EOS id / multipliers violate the int64 no-overflow precondition.
  explicit NgramIds(const NgramSidecarHeader& header);

  // Ids of `n` tokens, written as [n, 16] int64 global row ids; advances the history by `n` tokens.
  // Validates every token first and throws (state untouched) when one is negative or too large for the
  // overflow-free product.
  void Generate(const int32_t* tokens, size_t n, int64_t* ids_out);

  // Back to the start of a sequence ({EOS, EOS}).
  void Reset();

  State Snapshot() const { return state_; }
  // Restores a snapshot (prefix reuse). Throws when its tokens are out of the valid range.
  void Restore(const State& state);

  int64_t eos_token_id() const { return eos_; }
  // Largest token id Generate accepts.
  int64_t max_token_id() const { return max_token_; }

 private:
  void CheckToken(int64_t t, const char* where) const;

  int64_t eos_ = 0;
  int64_t max_token_ = 0;
  std::array<int64_t, NgramSidecarHeader::kNgramSize> mult_{};
  std::array<uint64_t, kHeads> primes_{};
  std::array<uint64_t, kHeads> offsets_{};
  State state_{};
};

}  // namespace r4dx::model::qwen4exp
