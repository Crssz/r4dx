// r4dx::model int8 prefill (docs/int8-prefill.md "Production path"): R4DX_PREFILL_INT8 and the decision of when a
// Model runs the int8 x int8 trellis GEMM for the 256-row super-chunks of its prompt Prefill calls.
//
// Header-only and free of HIP so the parser and the decision table have a CPU unit test
// (tests/model/test_prefill_int8_cpu.cpp); Model::Load is the only caller.
//
// R4DX_PREFILL_INT8 (read once per process, like R4DX_PREFILL_CHUNK -- prefill_chunk.h):
//   - unset, empty, "0" or "off": off (the default). Nothing changes: no scale table is built, no buffer or launch
//     differs, every byte of every path is the one of a build without the int8 kernel (the f16 kernels' ISA is
//     pinned by the build, their output by tests/model/test_prefill_int8);
//   - "1" or "on": on, where this Model can use it (DecidePrefillInt8). The full 256-row super-chunks of a
//     Prefill call -- and only those -- run their trellis linears through libr4d's int8 x int8 GEMM
//     (A quantized per (row, 128 k), the decoded weight per (column, 128 k), int32 WMMA, a per-128 fp32
//     rescale; the f16 output transform after it). Everything else stays f16: tails of fewer than 256 rows
//     (64-row slices), R4DX_PREFILL_CHUNK=0 / 64, a quant2 container, MTP and DFlash 64-row slices, decode
//     and verify windows, the vision tower, and PrefillMultimodal with images (image accuracy is unmeasured; a text-only call with no image ever seen is a Prefill call);
//   - anything else: a warning on stderr, then off (an unreadable request is read as "keep the old path").
// It is a research-grade accuracy trade, not a bit-identical optimization: with it on, a super-chunk row is NOT
// the row the 64-row path computes (docs/int8-prefill.md lists what that costs: the chunk-size identity of the
// KV bytes, and their independence of the prefix-cache state), so it is never on unless asked for.
//
// A Model that cannot use it says why once at load (the reasons of DecidePrefillInt8) and runs f16. It cannot be
// combined with R4DX_FAKEQ_ACT / R4DX_FAKEQ_W (the accuracy-experiment switches that round to int8 and run the f16
// GEMM): Model::Load throws, since the two would quantize twice.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace r4dx::model {

inline constexpr int kPrefillInt8Off = 0;
inline constexpr int kPrefillInt8On = 1;

// 1 for "1" / "on", 0 for unset / empty / "0" / "off"; anything else warns and is 0.
inline int ParsePrefillInt8(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return kPrefillInt8Off;
  if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0) return kPrefillInt8On;
  std::fprintf(stderr, "r4dx: R4DX_PREFILL_INT8='%s' not recognized (0|off|1|on); using off\n", e);
  return kPrefillInt8Off;
}
inline int PrefillInt8Request() {
  static const int v = ParsePrefillInt8(std::getenv("R4DX_PREFILL_INT8"));
  return v;
}

// ModelOptions::prefill_int8: -1 follows the environment, 0 and 1 force the request whatever the environment says
// (the identity tests load Models of each). Anything else is the caller's bug.
inline bool ValidPrefillInt8Option(int opt) { return opt == -1 || opt == 0 || opt == 1; }
inline int ResolvePrefillInt8Request(int opt) { return opt < 0 ? PrefillInt8Request() : (opt != 0 ? kPrefillInt8On : kPrefillInt8Off); }

// What decides it, for one Model, once at load.
struct PrefillInt8Inputs {
  int requested = kPrefillInt8Off;  // ResolvePrefillInt8Request
  bool wide = false;                // this Model runs 256-row super-chunks (DecidePrefillChunk)
  int tp_world = 1;                 // ModelOptions::tp.world
  bool has_trellis = false;         // the container's body is trellis (Container::HasTrellis)
  bool rotated_container = false;   // a quant2 container (residual rotation / Hadamard signs)
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
// TP = 2 keeps the f16 kernel in v1 (the kernel is shape-generic and bit-tested at the rank shapes; it needs the
// wiring and its own KL gate).
inline bool DecidePrefillInt8(const PrefillInt8Inputs& in, const char** why = nullptr) {
  if (why != nullptr) *why = nullptr;
  if (in.requested != kPrefillInt8On) return false;
  const auto refuse = [&](const char* reason) {
    if (why != nullptr) *why = reason;
    return false;
  };
  if (in.rotated_container) return refuse("a quant2 (rotated) container");
  if (!in.wide) return refuse("this Model does not run 256-row prefill super-chunks (R4DX_PREFILL_CHUNK=0/64)");
  if (in.tp_world > 1) return refuse("tensor parallelism (TP = 2 keeps the f16 kernel)");
  if (!in.has_trellis) return refuse("the container has no trellis linears");
  if (in.tables_built == 0) return refuse("no trellis linear has an int8 plan (see the per-shape notices)");
  return true;
}

}  // namespace r4dx::model
