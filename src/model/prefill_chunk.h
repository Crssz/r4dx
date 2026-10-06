// r4dx::model prefill chunk size (docs/trellis-m256.md, docs/prefill.md): R4DX_PREFILL_CHUNK and the
// decision of when a prompt Prefill call runs 256-row super-chunks instead of 64-row chunks.
//
// Header-only and free of HIP so the parser and the decision table have a CPU unit test
// (tests/model/test_prefill_chunk.cpp); Model::Load and Model::Prefill are the only callers.
//
// R4DX_PREFILL_CHUNK (read once per process, like R4DX_PREFILL_SPLITKV -- attention_layer.hpp's
// ParsePrefillAttnMode):
//   - unset or empty: 256 (the default). Prompt Prefill calls run 256-row super-chunks (layer-major: the
//     trellis linears see 256 rows through libr4d's M = 256 GEMM, the GDN sequence ops run once over the
//     256 rows -- or in 64-row sub-slices under R4DX_GDN_SLICE=64 -- and attention, MTP and DFlash per
//     64-row slice), bit-identical to the 64-row path;
//   - "0" or "64": the kill switch -- today's 64-row chunks exactly, no wide buffers (the load-time
//     activation buffers and arena are the 64-row sizes, so the VRAM is what it was before the default
//     changed);
//   - "256": the same as unset, spelled out;
//   - anything else: a warning on stderr, then 64 (an unreadable request is read as "keep the old path").
// The super-chunk changes no bits, so it only applies to the shape of call it was built and validated for;
// every other case falls back to 64-row chunks and says so once, never throws (DecidePrefillChunk below).
//
// On a wide Model (one that runs super-chunks), two more switches pick how the GDN layer runs its sequence
// ops (conv prep, kkt solve, chunk scan + state commit, gated norm; docs/trellis-m256.md "GDN sequence
// ops"). Both are meant to change no bits and both are read once per process. Neither does anything on a
// 64-row Model (R4DX_PREFILL_CHUNK=0, ModelOptions::prefill_chunk = 64, or a fallback): such a Model runs
// exactly the pre-gdn256 kernels, so the kill switch is the true old path whatever these say.
//   R4DX_GDN_SLICE -- unset, empty, "0" or "256": one call each over the whole 256 rows of a super-chunk
//     (the default; chunk 64 inside the kernels, the fp32 state carried in registers between chunks);
//     "64": the old path, four 64-row sub-slices in order with the state handed on through the slot;
//     anything else: a warning, then 64.
//   R4DX_GDN_CONV -- unset, empty or "2": r4d_gdn_conv_prep2 (the default; the same bytes on a grid that
//     fills the device, validated on device 1 2026-10-06), for every prefill call of a wide Model (its
//     64-row tail chunks too, DecideGdnConv); "1": the original r4d_gdn_conv_prep; anything else: a
//     warning, then 1.
// An unreadable value keeps the old path, as R4DX_PREFILL_CHUNK does.
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

// R4DX_GDN_SLICE: 0 = one call per GDN sequence op over the whole prefill chunk (GdnLayerParams::seq_slice
// 0), 64 = the 4 x 64 sub-slice path (seq_slice 64).
inline int ParseGdnSlice(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "0") == 0 || std::strcmp(e, "256") == 0) return 0;
  if (std::strcmp(e, "64") == 0) return kPrefillChunkBase;
  std::fprintf(stderr, "r4dx: R4DX_GDN_SLICE='%s' not recognized (0|64|256); using 64\n", e);
  return kPrefillChunkBase;
}
inline int GdnSliceRequest() {
  static const int v = ParseGdnSlice(std::getenv("R4DX_GDN_SLICE"));
  return v;
}

// R4DX_GDN_CONV: 2 = r4d_gdn_conv_prep2 (the default), 1 = the original r4d_gdn_conv_prep.
inline constexpr int kGdnConvV1 = 1;
inline constexpr int kGdnConvV2 = 2;
inline int ParseGdnConv(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "2") == 0) return kGdnConvV2;
  if (std::strcmp(e, "1") == 0) return kGdnConvV1;
  std::fprintf(stderr, "r4dx: R4DX_GDN_CONV='%s' not recognized (1|2); using 1\n", e);
  return kGdnConvV1;
}
inline int GdnConvRequest() {
  static const int v = ParseGdnConv(std::getenv("R4DX_GDN_CONV"));
  return v;
}

// The conv prep kernel one Model uses for all its prefill calls (GdnLayerParams::conv_prep), decided once
// at load: the request only on a wide Model (`wide`: Load sized it for 256-row super-chunks); a 64-row
// Model always runs the original r4d_gdn_conv_prep, so R4DX_PREFILL_CHUNK=0 is the pre-gdn256 path.
inline int DecideGdnConv(int requested, bool wide) {
  return (wide && requested == kGdnConvV2) ? kGdnConvV2 : kGdnConvV1;
}

// What decides it, for one Model, once at load (the buffers are sized by it).
struct PrefillChunkInputs {
  int requested = kPrefillChunkWide;  // ModelOptions::prefill_chunk, else PrefillChunkRequest()
  bool rotated_container = false;     // a quant2 container (residual rotation / Hadamard signs)
};

// kPrefillChunkWide when this Model may run 256-row super-chunks, else kPrefillChunkBase. `*why` (if
// non-null) names the reason a 256 request could not be honoured (a string literal); nullptr when 256 is
// used, and when 64 is what was asked for (the kill switch is a choice, not a fallback). Never throws.
//
// What the wide path serves (docs/prefill.md): TP = 1 and TP = 2 prompt prefill (a 256-row all-reduce is
// four 64-row ones over the same bytes), including the prompt-checkpoint and prefix-reuse suffix prefills
// (the chunk grid is anchored at the call start), --mtp (the MTP head primes its KV per 64-row slice of a
// super-chunk), --dflash and a target feature capture (the capture, the drafter injection and the
// per-chunk callback run per 64-row slice), and image prompts (PrefillMultimodal walks the same grid; the
// splice and the 3-axis rope rows are functions of the absolute position). It falls back, with a one-line
// reason, only for a quant2 (rotated) container.
inline int DecidePrefillChunk(const PrefillChunkInputs& in, const char** why = nullptr) {
  if (why != nullptr) *why = nullptr;
  if (in.requested != kPrefillChunkWide) return kPrefillChunkBase;
  if (in.rotated_container) {
    if (why != nullptr) *why = "a quant2 (rotated) container";
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
