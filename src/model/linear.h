// r4dx::model::ApplyLinear -- y[M,N] = x[M,K] @ W[N,K]^T for whichever layout `w` was loaded as,
// chunking M into <=64-row slices through the matching r4d skinny GEMM family
// (docs/architecture.md "Interim chunked prefill": every r4d_gemm_*_nt_m64 caps M at 64, so a
// wider prefill call is simply several of them back to back).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"

namespace r4dx::model {

// Tiling parameters for one r4d_gemm_*_nt_m64 launch. Trellis (docs/trellis-kernel.md 4.3, 5.3)
// reads MB as MT (row tiles per block) and NPW as NP (tile pairs per wave), and is the only layout
// that reads the last two: SKG, the blocks splitting K across the grid, and U, the k-tiles per
// unrolled step. Every other layout's rows keep their five values and ignore these.
struct LinearTuning {
  int WV, SK, MB, NPW, NT;
  int SKG = 1;
  int U = 2;
};

// One measured (layout, N, K, M) -> LinearTuning row (tools/profile/tune_gemm.py's sweep output,
// src/model/gemm_tuning_table.inc, included by linear.cpp). `M` is the exact chunk row count the
// row was measured at (one of the M-bands tools/profile/tune_gemm.py swept: 1,2,4,8,16,32,64) --
// PickTuning below looks up the smallest tabulated M that is >= the caller's chunk M (a tuning
// measured for a wider M is still legal, just not necessarily optimal, for a narrower call; the
// reverse -- a narrower M's tuning serving a wider call -- is also legal, but is more likely to
// underutilize the GPU on the wider call, so PickTuning rounds up, never down). One exception: a
// chunk of M <= 16 rows (one row tile) always gets the M=1 band's tuning (NT aside), whatever the
// M=2..16 bands measured. WV/MB/NPW/NT only choose how the work is laid out, but SK also sets the
// order the partial sums are added in, so two M-bands with different SK give the same row
// different last bits -- and a speculative verify row must equal the single-row decode row exactly
// (linear.cpp's kRowTile).
//
// `group` (quant2 Q3, docs/quant2.md section 5.1) is the w4a16 group the row was measured at, and
// is part of the key for w4a16: 0 -- what every row generated before per-tensor groups leaves it at
// -- means "this build's default group", so those rows serve exactly the linears they served
// before and nothing else; a row naming 32 or 64 serves only linears at that group. A w4a16
// linear at a non-default group with no row of its own falls back to FallbackTuning, like an
// untuned shape. Ignored for every other layout.
//
// `rate` (trellis only, docs/trellis-kernel.md 5.3): the trellis bits per weight (KB, 4 or 5) the
// row was measured at, part of the key like w4a16's group -- a row serves only linears at its
// rate. The trellis rows live in src/model/gemm_tuning_table_trellis.inc (tests/kernels/
// tool_trellis_gemm_bench.exe --inc-out writes it): an M = 1 row per (N, K, KB) (--modes full
// --joint) and, for prefill chunks, M = 32 and M = 64 rows (--modes ptune, milestone M5).
struct GemmTuningRow {
  Layout layout;
  int64_t N, K, M;
  LinearTuning tuning;
  int group = 0;
  int rate = 0;
};

// Picks a WV/SK/MB/NPW/NT for a (layout, N, K) GEMM chunk of M rows. Looks up
// src/model/gemm_tuning_table.inc's measured table first (tools/profile/tune_gemm.py, keyed by
// (layout,N,K,M-band)); falls back to a hand-derived, constraint-legal-for-every-shape-this-model-
// has default (see linear.cpp) when the table has no row for this exact (layout,N,K) shape --
// e.g. a shape the sweep did not cover, or the table file is missing/empty (this model's tests use
// a 4-layer container with the same shapes as the real 64-layer one, so in practice every shape
// PickTuning ever sees during normal operation IS covered by the table once tune_gemm.py has run;
// the fallback exists for robustness, not because it is expected to fire in production).
// `variant`: the w4a16 group for kW4a16 (QuantLinear::w4a16_group): 0 or this build's default group
// resolve exactly as before per-tensor groups; 32/64 otherwise look up only rows measured at
// that group (GemmTuningRow::group), then FallbackTuning. Every pick is legal for the kernel at
// that group: K % (SK * max(group, 64)) == 0. For kTrellis it is the linear's rate
// (QuantLinear::trellis_bits), matched against GemmTuningRow::rate; the trellis rows -- on a TP
// thread its per-rank rows (gemm_tuning_table_trellis_tp2.inc) first, then the TP = 1 ones -- are
// consulted after the TP table (on a TP thread) and the main table, then the M-aware fallback of
// docs/trellis-kernel.md 4.4 -- an M <= 16 chunk gets the M = 1 pick whole (NT included, 10.1), an
// M > 16 chunk the prefill row of the smallest M >= its own that fits it, else an unsplit
// 128-column block with every row tile in it. Every other layout ignores it.
LinearTuning PickTuning(Layout layout, int64_t N, int64_t K, int64_t M, int variant = 0);

// The group an r4d_gemm_w4a16_nt_m64 launch for a QuantLinear with this w4a16_group runs at: 0
// becomes r4d_gemm_w4a16_nt_m64_group(), anything else is returned as is.
int EffectiveW4a16Group(int w4a16_group);

// Tensor parallel (docs/tp.md 2.7): marks the CALLING thread as one that runs a TP=2 rank's Model
// (Model::Load calls it on every load with ModelOptions::tp.world > 1 -- true on a rank thread in
// TpModel, the main thread in tool_tp_step_bench -- and clears it when that load throws). On such a
// thread PickTuning looks up src/model/gemm_tuning_table_tp2.inc (the per-rank (N,K) shapes, same
// M-band and w4a16-group rules) FIRST, then the main table, then the fallback; its cache is
// thread_local and keyed on the flag too. A TP=1 load sets it false, so a TP=1 Model consults the
// main table alone, exactly as before, even on a thread that loaded a rank earlier. The
// TP rows live in their own table because one per-rank key -- (w4a16, 17408, 5120) -- is also the
// TP=1 DFlash drafter's gate_proj/up_proj, which the main table deliberately leaves untuned. The
// trellis rows follow the same rule: src/model/gemm_tuning_table_trellis_tp2.inc on such a thread,
// before the TP=1 trellis rows.
void SetTp2TuningForThisThread(bool enabled);

// The r4dx_epilogue (kernels.h) a fused producer must emit to feed `layout`'s GEMM directly --
// r4dx_epilogue_none for kBf16 (which never quantizes its activation input), r4dx_epilogue_f16 for
// kW4a16. Shared by
// every ApplyLinear caller that wants to pre-fuse its producer's cast epilogue (docs/r9700.md
// R2/P2) so both sides of the wiring agree on the mapping in exactly one place.
// kTrellis is r4dx_epilogue_none (docs/trellis-kernel.md 5.3): its input transform is per LINEAR
// (x * suh, then a Hadamard), which no producer epilogue value can name, so every layer runs its
// plain bf16 path and ApplyLinear applies the transform itself; a fused trellis producer passes a
// PreQuantizedActivation with transform_id instead (below), never through this function.
int EpilogueForLayout(Layout layout);

// A producer's already-cast activation (docs/r9700.md R2/P2's fused epilogue output, kernels.h's
// r4dx_epilogue), ready to feed `w`'s GEMM directly. `epilogue` must equal
// EpilogueForLayout(w.layout) exactly -- ApplyLinear throws otherwise, rather than silently
// reinterpreting bytes in the wrong format. `data` is [M,K] contiguous in the format `epilogue`
// selects (f16 uint16_t).
//
// Trellis (docs/trellis-kernel.md 5.3): an already input-transformed A (r4dx_trellis_input_bf16's
// output for THIS linear) is passed with `transform_id` = w.trellis_suh.data() (the transform is
// per linear, so ApplyLinear throws when the id is another linear's) and `epilogue` none. `data`
// then holds the linear's parts, each f16 [M][K] with row stride K, part p at data + p *
// part_stride elements (part_stride >= M * K; ignored for a one-part linear); a <= 64-row chunk m0
// reads each part at + m0 * K.
//
// R4DX_PREFILL_INT8_FUSEDQ (docs/int8-prefill.md "The fused quantizer"): a trellis producer that knows the call will
// take the int8 GEMM (TrellisI8FusedQ below: the one decision both sides make) writes the int8 operand instead of
// the f16 A: `a8` (A8 fragment layout, 256 * K bytes per part) and `sa` (fp32 [K / 128][256] per part), part p at
// a8 + p * 256 * K and sa + p * (K / 128) * 256, both 16-byte aligned, with `transform_id` set as above and `data`
// null (no f16 A exists; part_stride is unused). ApplyLinear then runs no quantizer launch. A pre with `a8` handed
// to a call that does not take the int8 GEMM (the producer and the consumer disagree) throws: there is no f16 A to
// fall back to.
struct PreQuantizedActivation {
  int epilogue = 0;  // r4dx_epilogue_none means "no pre-quantized input provided"
  const void* data = nullptr;
  const void* transform_id = nullptr;  // trellis only: the linear's trellis_suh.data()
  int64_t part_stride = 0;             // trellis only: elements between the parts in `data`
  const void* a8 = nullptr;            // trellis int8 only: the int8 operand (replaces `data`)
  const void* sa = nullptr;            // trellis int8 only: its scales
};

// x: device bf16 [M, K], row-major, CONTIGUOUS (row stride exactly K -- every r4d_gemm_*_nt_m64
// entry point reads its A/C operands at a hardcoded stride of K/N respectively; there is no
// strided-view form to call into). y: device bf16 [M, N], row-major, contiguous, disjoint from x.
// `arena` supplies this call's activation-quant scratch (w4a16's f16 cast,
// trellis's transformed parts and split-group partials -- at most 2 * 64 * K
// f16 plus SKG * 64 * N fp32) for up to a 64-row chunk; the caller is responsible for giving the
// arena enough headroom and Reset()-ing it between top-level forward-pass steps (Arena's own
// contract -- see r4dx/core/arena.hpp), not between individual ApplyLinear calls (this function
// does not reset it, so several ApplyLinear calls in the same layer share one growing allocation,
// which is the intended usage).
// `stream` is a raw hipStream_t (not core::Stream&) so this can be called from components that
// only have a raw stream handle (e.g. r4dx::model::attention::AttentionLayer, which owns its
// stream as a plain hipStream_t parameter) without pulling in core::Stream's RAII ownership --
// r4dx::core::Stream converts implicitly (operator hipStream_t()), so every existing call site
// that passes a core::Stream& is unaffected.
// `pre`, when non-null and pre->epilogue != r4dx_epilogue_none, skips this call's own internal
// quant/cast launch entirely and feeds `pre->data` (offset per <=64-row chunk exactly like `x` is)
// straight to the GEMM -- see PreQuantizedActivation's doc above. `x` is still
// required even when `pre` is given (kBf16 always reads it directly; a caller that only produced a
// quantized epilogue for a NON-bf16 layout does not need to also keep the plain bf16 buffer alive
// for THIS call, but ApplyLinear does not special-case that -- every existing caller already has
// both). A trellis `w` takes `pre` when pre->transform_id is set (its already transformed parts,
// PreQuantizedActivation's doc), and otherwise runs its input transform itself.
// `temporal_weight_loads` (trellis only; every other layout ignores it): load the weights with the
// normal cache policy (NT = 0) whatever the tuning row says. The trellis rows load non-temporally,
// which evicts almost nothing from the L2 -- so a GEMM right behind a kernel that left dirty lines
// there (the GDN recurrent update's new state) would leave them to drain, a few at a time, through
// every later weight stream; normal loads flush them in one burst instead (docs/trellis-kernel.md
// 10.6). It changes no bits: NT is a cache hint only.
void ApplyLinear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M, const PreQuantizedActivation* pre = nullptr,
                  bool temporal_weight_loads = false);

// ---- the M = 256 trellis GEMM (docs/trellis-m256.md; R4DX_PREFILL_CHUNK=256) ----------------------
// Rows of the wide launch: a 256-row call to ApplyLinear may run ONE launch of libr4d's
// r4d_gemm_trellis_nt_m256 instead of four 64-row ones, with the same bytes (the kernel reproduces,
// bit for bit, the shipped M = 64 tuning row's K-slice and SKG summation order).
inline constexpr int64_t kTrellisM256Rows = 256;

// Whether, and how, a trellis linear of this shape and rate runs through the M = 256 kernel: `SK` /
// `SKG` are the slice counts of the row its 64-row chunks run today (PickTuning at M = 64, plus the
// part-boundary fallback of ApplyLinear), `SKW` the K slices the kernel keeps resident per workgroup.
// `ok` is false -- and `why` says so -- for a combination without an exact configuration (rate other
// than 4 / 5, SK 16 at KB 5, a K that leaves a k-tile tail no instantiation covers, ...), which keeps the
// 64-row slicing. `part_n0` is trellis_part_n[0] (ignored for parts == 1). A pure function of its
// arguments (and R4DX_M256_SHAPES, a debug filter); tests/kernels/test_trellis_m256 and ApplyLinear share it.
struct TrellisM256Plan {
  bool ok = false;
  int SK = 0, SKG = 0, SKW = 0;
  std::string why;
};
TrellisM256Plan PlanTrellisM256(int64_t N, int64_t K, int kb, int parts, int64_t part_n0);

// RAII: while alive on this thread, a 256-row ApplyLinear of a trellis linear with a plan takes the
// M = 256 kernel. Off by default: Model turns it on around the layers of a 256-row prefill
// super-chunk only, so no other caller of ApplyLinear (decode, verify windows, MTP, the drafter,
// tests) can reach the wide kernel, whatever M it passes.
class ScopedTrellisM256 {
 public:
  explicit ScopedTrellisM256(bool on);
  ~ScopedTrellisM256();
  ScopedTrellisM256(const ScopedTrellisM256&) = delete;
  ScopedTrellisM256& operator=(const ScopedTrellisM256&) = delete;

 private:
  bool prev_;
};
bool TrellisM256Active();

// ---- the int8 x int8 prefill GEMM (R4DX_PREFILL_INT8; prefill_int8.h, docs/int8-prefill.md "Production path") ----
// One row of the int8 tuning table (src/model/gemm_tuning_table_trellis_i8.inc): the (skw, skg) of libr4d's
// r4d_gemm_trellis_nt_i8 for the trellis linear class of shape N x K at rate `rate` (KB).
struct TrellisI8Row {
  int64_t N, K;
  int rate;
  int skw, skg;
};
// The table, for tests and diagnostics (a pure read of the .inc).
const TrellisI8Row* TrellisI8Rows(size_t* count);

// Whether, and how, a trellis linear of this shape and rate runs through the int8 GEMM at M = 256: `skw` / `skg`
// are the K slices per workgroup and the K groups across the grid of its table row. `ok` is false -- and `why`
// says so -- for a combination without a legal configuration (no row for the class, a K or N the kernel
// rejects, a part boundary that is not a whole 128-block, a shape R4DX_M256_SHAPES excludes), which keeps the
// f16 kernel. `part_n0` is trellis_part_n[0] (ignored for parts == 1). A pure function of its arguments (and
// R4DX_M256_SHAPES, the debug filter the f16 plan honours too); the load-time table builder, ApplyLinear and
// tests/model/test_prefill_int8_cpu share it.
struct TrellisI8Plan {
  bool ok = false;
  int skw = 0, skg = 0;
  std::string why;
};
TrellisI8Plan PlanTrellisI8(int64_t N, int64_t K, int kb, int parts, int64_t part_n0);

// RAII: while alive on this thread, a 256-row ApplyLinear of a trellis linear with an int8 plan (and a scale
// table) takes the int8 x int8 GEMM instead of the f16 M = 256 kernel; it also needs ScopedTrellisM256 (the
// super-chunk scope) to be on. Off by default: Model turns it on around the layers of a 256-row prefill
// super-chunk of a Prefill call (not PrefillMultimodal, not decode, not a verify window, not a 64-row chunk)
// when R4DX_PREFILL_INT8 is on, so no other caller of ApplyLinear can reach it.
class ScopedTrellisI8 {
 public:
  explicit ScopedTrellisI8(bool on);
  ~ScopedTrellisI8();
  ScopedTrellisI8(const ScopedTrellisI8&) = delete;
  ScopedTrellisI8& operator=(const ScopedTrellisI8&) = delete;

 private:
  bool prev_;
};
bool TrellisI8Active();

// R4DX_PREFILL_INT8_FUSEDQ (read once; default on, "0" / "off" is the kill switch: the quantizer runs as its own launch
// again): whether a producer writes the int8 operand itself. Off too whenever R4DX_TRELLIS_A_STATS is set (that hook
// tallies the f16 A, which a fused producer never writes).
bool TrellisI8FusedQEnabled();
// RAII, for tests: while alive on this thread, TrellisI8FusedQEnabled() answers `mode` (0 off, 1 on, whatever the
// environment says; the A-stats hook still forces off), so one process can run the fused and the separate chain on
// one Model and compare their bytes. -1 follows the environment (the default).
class ScopedTrellisI8FusedQ {
 public:
  explicit ScopedTrellisI8FusedQ(int mode);
  ~ScopedTrellisI8FusedQ();
  ScopedTrellisI8FusedQ(const ScopedTrellisI8FusedQ&) = delete;
  ScopedTrellisI8FusedQ& operator=(const ScopedTrellisI8FusedQ&) = delete;

 private:
  int prev_;
};
// Where the int8 GEMMs' A8 + SA came from, process-wide since start (tests prove the path they mean ran): a fused producer's
// (PreQuantizedActivation::a8), this call's own fused transform (r4dx_trellis_input_i8), or the separate quantizer launch.
struct TrellisI8OperandCounts {
  int64_t from_producer = 0, from_own_transform = 0, separate = 0;
};
TrellisI8OperandCounts TrellisI8OperandCountsGet();
// Whether ApplyLinear(w, ..., M) takes the int8 GEMM right now: a trellis linear, M == 256, both scopes open
// (ScopedTrellisM256, ScopedTrellisI8), an int8 plan and its scale table. ApplyLinear's own test, factored out.
bool TrellisI8Takes(const QuantLinear& w, int64_t M);
// The decision a producer of w's A makes before it writes A8 + SA rather than f16: the switch is on and the call will
// take the int8 GEMM. A pure function of the linear, M and the thread's scopes, so a producer asked in the same scope
// as its ApplyLinear agrees with it; a producer that fused and a consumer that cannot take int8 is a bug and throws.
bool TrellisI8FusedQ(const QuantLinear& w, int64_t M);
// The int8 operand of `parts` (1..2) 256-row parts of K columns, from the arena: A8 `parts * 256 K` bytes, SA
// `parts * K / 128 * 256` floats, both 16-byte aligned, the second part following the first as
// PreQuantizedActivation documents.
struct TrellisI8Operand {
  int8_t* a8 = nullptr;
  float* sa = nullptr;
};
TrellisI8Operand AllocTrellisI8Operand(core::Arena& arena, int64_t K, int parts);

// R4DX_PREFILL_INT8's one-time pass (Model::Load, only when the switch is on): the weight scale table of the
// trellis linear `w` (libr4d's r4d_trellis_i8_wscale: s = max|w| / 127 per (column, 128 k), stored as 1 / f16(1 / s))
// in w.trellis_i8_sw, K * N / 32 bytes. Only for a linear whose PlanTrellisI8 is ok; returns whether it built one.
// Stream-ordered. A non-trellis linear is left alone.
bool BuildTrellisI8Scale(QuantLinear& w, hipStream_t stream);

// RAII: while alive on this thread, every trellis linear's ApplyLinear rounds its transformed f16 A to int8
// and back in place right before the GEMM (fake_quant_act.h's `mode`, kernels.h's r4dx_fake_quant_act_f16;
// docs/int8-prefill.md). Mode 0 (kFakeQuantOff) does nothing at all. Model turns it on around the layers of
// a prompt-prefill chunk only (R4DX_FAKEQ_ACT), so decode, verify windows, the MTP / DFlash heads and the
// vision tower never see it. End() restores the previous mode early (idempotent).
class ScopedFakeQuantAct {
 public:
  explicit ScopedFakeQuantAct(int mode);
  ~ScopedFakeQuantAct() { End(); }
  ScopedFakeQuantAct(const ScopedFakeQuantAct&) = delete;
  ScopedFakeQuantAct& operator=(const ScopedFakeQuantAct&) = delete;
  void End();

 private:
  int prev_;
  bool live_;
};

// R4DX_FAKEQ_W (fake_quant_w.h, docs/int8-prefill.md), the weight side of the above: while alive on this
// thread, every trellis linear's ApplyLinear runs the `_wq` GEMM kernels, which round each decoded weight to
// int8 and back right before the WMMA with the linear's scale table (QuantLinear::trellis_wscale, built by
// BuildTrellisWScale). Mode 0 (kFakeQuantWOff) does nothing at all: the shipped kernels, bit for bit. A
// trellis linear without a table under a non-zero mode throws (the measurement must not silently run
// unquantized). Model opens it around the layers of a prompt-prefill chunk only, next to ScopedFakeQuantAct.
class ScopedFakeQuantW {
 public:
  explicit ScopedFakeQuantW(int mode);
  ~ScopedFakeQuantW() { End(); }
  ScopedFakeQuantW(const ScopedFakeQuantW&) = delete;
  ScopedFakeQuantW& operator=(const ScopedFakeQuantW&) = delete;
  void End();

 private:
  int prev_;
  bool live_;
};

// R4DX_FAKEQ_W's one-time pass (Model::Load, only when the switch is on): decodes the trellis linear `w` with
// the GEMM's own decode and stores the per-(column, k group) scale table in w.trellis_wscale (fp32,
// [K / 16 >> gsh][N] for `mode`'s group, K * N / 32 bytes at col128). Stream-ordered (the GEMMs that
// read it run on `stream`). A non-trellis linear is left alone.
void BuildTrellisWScale(QuantLinear& w, int mode, hipStream_t stream);

// docs/trellis-kernel.md 4.8 / 5.4 (M5): whether the layers use the trellis fused producers and
// shared input transforms below -- on unless R4DX_DISABLE_EPILOGUE=1 (the same A/B switch
// tools/validate_fusion.ps1 uses for the other layouts' fused epilogues, EpilogueForLayout) or
// R4DX_TRELLIS_A_STATS is set (that hook tallies the A ApplyLinear's own transform writes, so it wants
// every linear to run it). Both paths give the same bytes; this only picks the launches.
bool TrellisFusionEnabled();

// docs/trellis-kernel.md 4.9 / 5.4 (M5): one input transform for n (2..3) trellis linears that read
// the same activation x [M, K] (attn.qg / k / v, gdn.in_proj_qkv / in_proj_z) --
// r4dx_trellis_input_bf16 with nout = n, which computes each output exactly as the linear's own
// ApplyLinear would, so the bytes are the same -- and pre[i] set up to hand linear i its A
// (PreQuantizedActivation::transform_id). Returns false, launching and allocating nothing, unless
// TrellisFusionEnabled() and every linear is a one-part trellis linear of x's K with the first one's
// prescale; the caller then passes no `pre`. The A buffers come from `arena`.
bool SharedTrellisInput(hipStream_t stream, core::Arena& arena, const uint16_t* x, int64_t M,
                        const QuantLinear* const* ws, int n, PreQuantizedActivation* pre);

}  // namespace r4dx::model
