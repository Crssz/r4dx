// r4dx::model int8 prefill (docs/int8-prefill.md "Production path"): R4DX_PREFILL_INT8 and the decision of when a
// Model runs the int8 x int8 trellis GEMM for the 256-row super-chunks of its prompt Prefill calls.
//
// Header-only and free of HIP so the parser and the decision table have a CPU unit test
// (tests/model/test_prefill_int8_cpu.cpp); Model::Load is the only caller.
//
// R4DX_PREFILL_INT8 (read once per process, like R4DX_PREFILL_CHUNK -- prefill_chunk.h). ON BY DEFAULT:
//   - unset, empty, "1" or "on": on, where this Model can use it (DecidePrefillInt8). The full 256-row super-chunks
//     of a Prefill call -- and only those -- run their trellis linears through libr4d's int8 x int8 GEMM
//     (A quantized per (row, 128 k), the decoded weight per (column, 128 k), int32 WMMA, a per-128 fp32
//     rescale; the f16 output transform after it). Everything else stays f16: tails of fewer than 256 rows
//     (64-row slices), R4DX_PREFILL_CHUNK=0 / 64, a quant2 container, TP = 2 (unless R4DX_PREFILL_INT8_TP2=1, below), MTP and DFlash 64-row slices, decode
//     and verify windows, the vision tower, and PrefillMultimodal with images (image accuracy is unmeasured; a text-only call with no image ever seen is a Prefill call);
//   - "0" or "off": the kill switch. Nothing changes: no scale table is built (-0.7 GiB on the 27B), no buffer or
//     launch differs, every byte of every path is the one of a build without the int8 kernel (the f16 kernels' ISA is
//     pinned by the build, their output by tests/model/test_prefill_int8);
//   - anything else: a warning on stderr, then the default (on): an unreadable request keeps the default.
// It is a validated accuracy trade, not a bit-identical optimization: with it on, a super-chunk row is NOT the row
// the 64-row path computes (docs/int8-prefill.md lists what that costs: the chunk-size identity of the KV bytes, and
// their independence of the prefix-cache state; measured KL(off || on) 0.0011 canon / 0.0017 at 8k / 0.0104 at 32k,
// greedy text unchanged, TTFT -18.9 % at 8k and -15.7 % at 32k). It was opt-in until the user approved making it the
// default (docs/int8-prefill.md "Now the default").
//
// A Model that cannot use it says why once at load (the reasons of DecidePrefillInt8) and runs f16. It cannot be
// combined with R4DX_FAKEQ_ACT / R4DX_FAKEQ_W (the accuracy-experiment switches that round to int8 and run the f16
// GEMM, the int8 GEMM would quantize twice): when int8 is only the default those switches win (a reason at load);
// when it was asked for explicitly (R4DX_PREFILL_INT8=1/on or ModelOptions::prefill_int8 = 1) Model::Load throws.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace r4dx::model {

inline constexpr int kPrefillInt8Off = 0;
inline constexpr int kPrefillInt8On = 1;

// 1 for unset / empty / "1" / "on" (the default is on), 0 for "0" / "off"; anything else warns and is the default.
inline int ParsePrefillInt8(const char* e) {
  if (e == nullptr || *e == '\0') return kPrefillInt8On;
  if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return kPrefillInt8Off;
  if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) return kPrefillInt8On;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_INT8='%s' not recognized (0|off|1|on); using the default (on)\n", e);
  return kPrefillInt8On;
}
// True when the environment ASKED for on ("1" / "on"), as opposed to the default being on (unset, empty, unreadable).
inline bool ParsePrefillInt8Explicit(const char* e) { return e != nullptr && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0); }
inline int PrefillInt8Request() {
  static const int v = ParsePrefillInt8(std::getenv("R4DX_PREFILL_INT8"));
  return v;
}
inline bool PrefillInt8RequestExplicit() {
  static const bool v = ParsePrefillInt8Explicit(std::getenv("R4DX_PREFILL_INT8"));
  return v;
}

// R4DX_PREFILL_INT8_FUSEDQ (docs/int8-prefill.md "The fused quantizer"): whether the producers of a trellis A (the input
// transform, the silu_mul / gate-mul fused producers) write the int8 operand themselves where the call will take the int8
// GEMM, instead of the f16 A plus a separate r4d_trellis_i8_quant_act launch. Same bytes either way. true for unset / empty /
// "1" / "on" (the default), false for "0" / "off" (the kill switch); anything else warns and is the default.
inline bool ParsePrefillInt8FusedQ(const char* e) {
  if (e == nullptr || *e == '\0') return true;
  if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return false;
  if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) return true;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_INT8_FUSEDQ='%s' not recognized (0|off|1|on); using the default (on)\n", e);
  return true;
}

// R4DX_PREFILL_INT8_SCALES (docs/int8-prefill.md "Coarse scales"; read once per process, Model::Load resolves it): the scale
// granularity of the int8 GEMM when it is on. "blk128" (unset, empty: the default) is today's: A one scale per (row, 128 k),
// the decoded weight one per (column, 128 k), the int32 partial sum of every 128 k rescaled in fp32. "coarse": A one scale
// per ROW and the weight one per COLUMN over the whole K, so the int32 accumulator needs no per-128 rescale (one fp32
// multiply at the end; the weight table is [N] floats per linear instead of [K / 128][N], 128 times smaller). Opt-in: a
// bit coarser model of the linear (its accuracy cost is its own KL gate), faster in the bench (speed bound x1.56 against
// x1.37). Anything else warns and means the default. Needs R4DX_PREFILL_INT8 on to matter at all.
inline constexpr int kPrefillInt8ScalesBlk128 = 0;
inline constexpr int kPrefillInt8ScalesCoarse = 1;
inline int ParsePrefillInt8Scales(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "blk128") == 0) return kPrefillInt8ScalesBlk128;
  if (std::strcmp(e, "coarse") == 0) return kPrefillInt8ScalesCoarse;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_INT8_SCALES='%s' not recognized (blk128|coarse); using the default (blk128)\n", e);
  return kPrefillInt8ScalesBlk128;
}
inline int PrefillInt8ScalesRequest() {
  static const int v = ParsePrefillInt8Scales(std::getenv("R4DX_PREFILL_INT8_SCALES"));
  return v;
}
// ModelOptions::prefill_int8_scales: -1 follows the environment, 0 (blk128) and 1 (coarse) force it (tests load both in one process).
inline bool ValidPrefillInt8ScalesOption(int opt) { return opt == -1 || opt == 0 || opt == 1; }
inline int ResolvePrefillInt8Scales(int opt) { return opt < 0 ? PrefillInt8ScalesRequest() : (opt != 0 ? kPrefillInt8ScalesCoarse : kPrefillInt8ScalesBlk128); }
inline const char* PrefillInt8ScalesName(int scales) { return scales == kPrefillInt8ScalesCoarse ? "coarse" : "blk128"; }

// R4DX_PREFILL_INT8_TP2 (docs/int8-prefill.md "Tensor parallel"; read once per process, Model::Load resolves it): whether a
// tensor-parallel (TP = 2) Model may run the int8 GEMM on its rank shards, when R4DX_PREFILL_INT8 is on. OFF BY DEFAULT: "1" / "on" ask
// for it; unset, empty, "0" / "off" keep the f16 kernels at TP = 2 (the bytes of main's TP = 2 exactly: no scale table, no scope);
// anything else warns and keeps the default (off). It is only a request: a shard shape without a row in
// gemm_tuning_table_trellis_i8_tp2.inc (and the coarse one) still runs f16 with a once-per-shape notice, and with no row at all the
// Model stays f16 and says so at load. No effect at TP = 1.
inline bool ParsePrefillInt8Tp2(const char* e) {
  if (e == nullptr || *e == '\0') return false;
  if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return false;
  if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) return true;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_INT8_TP2='%s' not recognized (0|off|1|on); using the default (off)\n", e);
  return false;
}
inline bool PrefillInt8Tp2Request() {
  static const bool v = ParsePrefillInt8Tp2(std::getenv("R4DX_PREFILL_INT8_TP2"));
  return v;
}
// ModelOptions::prefill_int8_tp2: -1 follows the environment (default off), 0 and 1 force it (the tests load both in one process).
inline bool ValidPrefillInt8Tp2Option(int opt) { return opt == -1 || opt == 0 || opt == 1; }
inline bool ResolvePrefillInt8Tp2(int opt) { return opt < 0 ? PrefillInt8Tp2Request() : opt != 0; }

// ModelOptions::prefill_int8: -1 follows the environment (default: on), 0 and 1 force the request whatever the
// environment says (the identity tests load Models of 0, the int8 tests of 1). Anything else is the caller's bug.
inline bool ValidPrefillInt8Option(int opt) { return opt == -1 || opt == 0 || opt == 1; }
inline int ResolvePrefillInt8Request(int opt) { return opt < 0 ? PrefillInt8Request() : (opt != 0 ? kPrefillInt8On : kPrefillInt8Off); }
// Whether the on request was ASKED for (ModelOptions::prefill_int8 = 1, or R4DX_PREFILL_INT8=1/on) rather than the default.
inline bool ResolvePrefillInt8Explicit(int opt) { return opt < 0 ? PrefillInt8RequestExplicit() : opt != 0; }

// What decides it, for one Model, once at load.
struct PrefillInt8Inputs {
  int requested = kPrefillInt8Off;  // ResolvePrefillInt8Request
  bool wide = false;                // this Model runs 256-row super-chunks (DecidePrefillChunk)
  int tp_world = 1;                 // ModelOptions::tp.world
  bool tp2_enabled = false;         // ResolvePrefillInt8Tp2: a TP > 1 Model may use int8 on its rank shards (default off)
  bool has_trellis = false;         // the container's body is trellis (Container::HasTrellis)
  bool rotated_container = false;   // a quant2 container (residual rotation / Hadamard signs)
  bool fakeq_active = false;        // R4DX_FAKEQ_ACT / R4DX_FAKEQ_W set (the default yields to them; an explicit on throws before this)
  int tables_built = -1;            // trellis linears that got a scale table; -1 = not built yet (the first decision)
};

// True when this Model runs the int8 GEMM in its super-chunks. `*why` (if non-null) names the reason an "on"
// request could not be honoured (a string literal); nullptr when it is used, and when it was not asked for (off
// is a choice, not a fallback). Never throws. The order of the reasons is the order of the checks.
//
// What it serves: TP = 1 prompt prefill of a trellis container that runs 256-row super-chunks, including the
// prefix-reuse suffix prefills (the chunk grid is anchored at each Prefill call's start: the same token can be
// int8 in one run and f16 in another, so the KV bytes of a prompt depend on cache state -- docs/int8-prefill.md),
// --mtp and --dflash (their capture and priming read int8-perturbed hidden states, which the accuracy gates cover).
// TP = 2 keeps the f16 kernel unless R4DX_PREFILL_INT8_TP2 asked for int8 on the rank shards (the kernel is shape-generic and
// bit-tested at the rank shapes; the wiring is docs/int8-prefill.md "Tensor parallel", its KL gate is not run yet, so it is off).
inline bool DecidePrefillInt8(const PrefillInt8Inputs& in, const char** why = nullptr) {
  if (why != nullptr) *why = nullptr;
  if (in.requested != kPrefillInt8On) return false;
  const auto refuse = [&](const char* reason) {
    if (why != nullptr) *why = reason;
    return false;
  };
  if (in.fakeq_active) return refuse("R4DX_FAKEQ_ACT / R4DX_FAKEQ_W are set (they round to int8 themselves)");
  if (in.rotated_container) return refuse("a quant2 (rotated) container");
  if (!in.wide) return refuse("this Model does not run 256-row prefill super-chunks (R4DX_PREFILL_CHUNK=0/64)");
  if (in.tp_world > 1 && !in.tp2_enabled) {
    return refuse("tensor parallelism (TP = 2 keeps the f16 kernel; R4DX_PREFILL_INT8_TP2=1 asks for int8 on the rank shards, off until validated)");
  }
  if (!in.has_trellis) return refuse("the container has no trellis linears");
  if (in.tables_built == 0) {
    return refuse(in.tp_world > 1 ? "no TP rank-shard linear has an int8 plan (the TP = 2 tuning tables have no row for its shapes yet: "
                                    "tool_int8_gemm_proto --tp 2 --emit-rows; see the per-shape notices)"
                                  : "no trellis linear has an int8 plan (see the per-shape notices)");
  }
  return true;
}

}  // namespace r4dx::model
