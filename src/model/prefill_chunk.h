// r4dx::model prefill chunk size (docs/trellis-m256.md): R4DX_PREFILL_CHUNK and the decision of when a
// prompt Prefill call may run 256-row super-chunks instead of today's 64-row chunks.
//
// Header-only and free of HIP so the parser and the decision table have a CPU unit test
// (tests/model/test_prefill_chunk.cpp); Model::Load and Model::Prefill are the only callers.
//
// R4DX_PREFILL_CHUNK (read once per process, like R4DX_PREFILL_SPLITKV -- attention_layer.hpp's
// ParsePrefillAttnMode):
//   - unset, empty or "64": today's behaviour, nothing changes (the default);
//   - "256": prompt Prefill calls run 256-row super-chunks (layer-major: the trellis linears see 256
//     rows through libr4d's M = 256 GEMM, everything sequence-dependent runs in 64-row sub-slices in
//     order), bit-identical to the 64-row path, where the configuration allows it (below);
//   - anything else: a warning on stderr, then 64.
// The super-chunk is an opt-in prefill speed option that changes no bits, so it only ever applies to
// the one shape of call it was built and validated for; every other case falls back, never throws.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace r4dx::model {

inline constexpr int kPrefillChunkDefault = 64;
inline constexpr int kPrefillChunkWide = 256;

inline int ParsePrefillChunk(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "64") == 0) return kPrefillChunkDefault;
  if (std::strcmp(e, "256") == 0) return kPrefillChunkWide;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_CHUNK='%s' not recognized (64|256); using 64\n", e);
  return kPrefillChunkDefault;
}
inline int PrefillChunkRequest() {
  static const int v = ParsePrefillChunk(std::getenv("R4DX_PREFILL_CHUNK"));
  return v;
}

// What decides it, for one Model at one Prefill call.
struct PrefillChunkInputs {
  int requested = kPrefillChunkDefault;  // PrefillChunkRequest()
  bool buffers_wide = false;             // Load() sized the activation buffers and arena for 256 rows
  bool tensor_parallel = false;          // this Model is a TP rank (comm != nullptr)
  bool mtp = false;                      // an MTP head is attached (its KV priming reads <= 64 rows)
  bool dflash = false;                   // a DFlash2 drafter is attached (feature injection <= 64 rows)
  bool dflash_capture = false;           // a feature capture is attached (per-64-row chunk drain)
  bool mrope_active = false;             // an image has been spliced into this conversation (3-axis rope)
  bool rotated_container = false;        // a quant2 container (residual rotation / Hadamard signs)
  bool on_chunk_captured = false;        // the caller wants a callback after every 64-row chunk
};

// kPrefillChunkWide when this call may run 256-row super-chunks, else kPrefillChunkDefault. When a 256
// request cannot be honoured, `*why` (if non-null) names the reason (a string literal), and it is
// nullptr when 64 was what was asked for. Never throws.
inline int DecidePrefillChunk(const PrefillChunkInputs& in, const char** why = nullptr) {
  if (why != nullptr) *why = nullptr;
  if (in.requested != kPrefillChunkWide) return kPrefillChunkDefault;
  const char* reason = nullptr;
  if (in.tensor_parallel) reason = "tensor parallelism (--tp 2): the all-reduce buffer is sized for 64 rows";
  else if (in.mtp) reason = "an MTP head is attached (--mtp): its KV priming is limited to 64 rows";
  else if (in.dflash) reason = "a DFlash drafter is attached (--dflash): its feature injection is limited to 64 rows";
  else if (in.dflash_capture) reason = "a target feature capture is attached: it drains once per 64-row chunk";
  else if (in.mrope_active) reason = "an image was spliced into this conversation (3-axis rope rows)";
  else if (in.rotated_container) reason = "a quant2 (rotated) container";
  else if (in.on_chunk_captured) reason = "the caller asked for a callback after every 64-row chunk";
  else if (!in.buffers_wide) reason = "this Model's buffers were not sized for 256 rows at load";
  if (reason != nullptr) {
    if (why != nullptr) *why = reason;
    return kPrefillChunkDefault;
  }
  return kPrefillChunkWide;
}

}  // namespace r4dx::model
