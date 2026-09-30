// r4dx::model prefill chunk size (docs/trellis-m256.md, docs/prefill.md): R4DX_PREFILL_CHUNK and the
// decision of when a prompt Prefill call runs 256-row super-chunks instead of 64-row chunks.
//
// Header-only and free of HIP so the parser and the decision table have a CPU unit test
// (tests/model/test_prefill_chunk.cpp); Model::Load and Model::Prefill are the only callers.
//
// R4DX_PREFILL_CHUNK (read once per process, like R4DX_PREFILL_SPLITKV -- attention_layer.hpp's
// ParsePrefillAttnMode):
//   - unset or empty: 256 (the default). Prompt Prefill calls run 256-row super-chunks (layer-major: the
//     trellis linears see 256 rows through libr4d's M = 256 GEMM, everything sequence-dependent runs in
//     64-row sub-slices in order), bit-identical to the 64-row path;
//   - "0" or "64": the kill switch -- today's 64-row chunks exactly, no wide buffers (the load-time
//     activation buffers and arena are the 64-row sizes, so the VRAM is what it was before the default
//     changed);
//   - "256": the same as unset, spelled out;
//   - anything else: a warning on stderr, then 64 (an unreadable request is read as "keep the old path").
// The super-chunk changes no bits, so it only applies to the shape of call it was built and validated for;
// every other case falls back to 64-row chunks and says so once, never throws (DecidePrefillChunk below).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace r4dx::model {

inline constexpr int kPrefillChunkBase = 64;   // Model::max_chunk_: today's chunk, and the sub-slice of a wide one
inline constexpr int kPrefillChunkWide = 256;  // the super-chunk, and the default request

inline int ParsePrefillChunk(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "256") == 0) return kPrefillChunkWide;
  if (std::strcmp(e, "64") == 0 || std::strcmp(e, "0") == 0) return kPrefillChunkBase;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_CHUNK='%s' not recognized (0|64|256); using 64\n", e);
  return kPrefillChunkBase;
}
inline int PrefillChunkRequest() {
  static const int v = ParsePrefillChunk(std::getenv("R4DX_PREFILL_CHUNK"));
  return v;
}

// What decides it, for one Model (at load, then at every Prefill call).
struct PrefillChunkInputs {
  int requested = kPrefillChunkWide;  // ModelOptions::prefill_chunk, else PrefillChunkRequest()
  bool buffers_wide = true;           // Load() sized the activation buffers and arena for 256 rows
  bool mrope_active = false;          // an image has been spliced into this conversation (3-axis rope rows)
  bool rotated_container = false;     // a quant2 container (residual rotation / Hadamard signs)
  bool tensor_parallel = false;       // this Model is a TP rank (comm != nullptr)
};

// kPrefillChunkWide when this Model / call may run 256-row super-chunks, else kPrefillChunkBase. `*why` (if
// non-null) names the reason a 256 request could not be honoured (a string literal); nullptr when 256 is
// used, and when 64 is what was asked for (the kill switch is a choice, not a fallback). Never throws.
//
// What the wide path serves (docs/prefill.md): TP = 1 prompt prefill, including the prompt-checkpoint and
// prefix-reuse suffix prefills (the chunk grid is anchored at the call start), --mtp (the MTP head primes
// its KV per 64-row slice of a super-chunk), --dflash and a target feature capture (the capture, the
// drafter injection and the per-chunk callback run per 64-row slice). It falls back, with a one-line
// reason, for tensor parallelism, a quant2 (rotated) container and a conversation that has had an image
// spliced in (its rope rows are 3-axis; PrefillMultimodal with images is 64-row anyway).
struct PrefillChunkReasons {
  static constexpr const char* kMrope = "an image was spliced into this conversation (3-axis rope rows)";
  static constexpr const char* kRotated = "a quant2 (rotated) container";
  static constexpr const char* kNotWide = "this Model's buffers were sized for 64 rows at load";
};
inline int DecidePrefillChunk(const PrefillChunkInputs& in, const char** why = nullptr) {
  if (why != nullptr) *why = nullptr;
  if (in.requested != kPrefillChunkWide) return kPrefillChunkBase;
  const char* reason = nullptr;
  if (in.tensor_parallel) reason = "tensor parallelism (--tp 2): the all-reduce buffer is sized for 64 rows";
  else if (in.rotated_container) reason = PrefillChunkReasons::kRotated;
  else if (in.mrope_active) reason = PrefillChunkReasons::kMrope;
  else if (!in.buffers_wide) reason = PrefillChunkReasons::kNotWide;
  if (reason != nullptr) {
    if (why != nullptr) *why = reason;
    return kPrefillChunkBase;
  }
  return kPrefillChunkWide;
}

// The next chunk of a Prefill call with `remaining` (> 0) tokens left under `wide` (kPrefillChunkWide or
// kPrefillChunkBase): a full super-chunk while at least 256 rows remain, else an ordinary chunk of up to 64
// rows -- so the grid is anchored at the call start, the tail (fewer than 256 rows) is the very chunks the
// 64-row path makes for it, and its last chunk holds the remainder (1..64 rows). Model::Prefill and
// PrefillProfiled walk exactly this.
inline long long PrefillNextChunk(long long remaining, int wide) {
  if (wide > kPrefillChunkBase && remaining >= wide) return wide;
  return remaining < kPrefillChunkBase ? remaining : kPrefillChunkBase;
}

// The chunk sizes of a whole call over `n` tokens, in order; `rows` receives at most `cap` entries and the
// count is returned. The unit test's reference for the grid.
inline int PrefillChunkGrid(long long n, int wide, int* rows, int cap) {
  int count = 0;
  for (long long off = 0; off < n; ++count) {
    const long long c = PrefillNextChunk(n - off, wide);
    if (count < cap) rows[count] = static_cast<int>(c);
    off += c;
  }
  return count;
}

}  // namespace r4dx::model
