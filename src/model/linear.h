// r4dx::model::ApplyLinear -- y[M,N] = x[M,K] @ W[N,K]^T for whichever layout `w` was loaded as,
// chunking M into <=64-row slices through the matching r4d skinny GEMM family
// (docs/architecture.md "Interim chunked prefill": every r4d_gemm_*_nt_m64 caps M at 64, so a
// wider prefill call is simply several of them back to back).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

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
// before and nothing else; a row naming 32/64/128 serves only linears at that group. A w4a16
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
// resolve exactly as before per-tensor groups; 32/64/128 otherwise look up only rows measured at
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
// kW4a16, r4dx_epilogue_int8_fraga8 for kW4a8, r4dx_epilogue_fp8_e4m3_row for kMxfp4. Shared by
// every ApplyLinear caller that wants to pre-fuse its producer's quant epilogue (docs/r9700.md
// R2/P2) so both sides of the wiring agree on the mapping in exactly one place.
// kTrellis is r4dx_epilogue_none (docs/trellis-kernel.md 5.3): its input transform is per LINEAR
// (x * suh, then a Hadamard), which no producer epilogue value can name, so every layer runs its
// plain bf16 path and ApplyLinear applies the transform itself; a fused trellis producer passes a
// PreQuantizedActivation with transform_id instead (below), never through this function.
int EpilogueForLayout(Layout layout);

// A producer's already-quantized activation (docs/r9700.md R2/P2's fused epilogue output,
// kernels.h's r4dx_epilogue), ready to feed `w`'s GEMM directly. `epilogue` must equal
// EpilogueForLayout(w.layout) exactly -- ApplyLinear throws otherwise, rather than silently
// reinterpreting bytes in the wrong format. `data` is [M,K] contiguous in the format `epilogue`
// selects (f16 uint16_t, fp8e4m3 uint8_t, or int8 fragA8-permuted int8_t); `scale` is [M] fp32,
// unused (may be nullptr) for r4dx_epilogue_f16.
//
// Trellis (docs/trellis-kernel.md 5.3): an already input-transformed A (r4dx_trellis_input_bf16's
// output for THIS linear) is passed with `transform_id` = w.trellis_suh.data() (the transform is
// per linear, so ApplyLinear throws when the id is another linear's) and `epilogue` none. `data`
// then holds the linear's parts, each f16 [M][K] with row stride K, part p at data + p *
// part_stride elements (part_stride >= M * K; ignored for a one-part linear); a <= 64-row chunk m0
// reads each part at + m0 * K.
struct PreQuantizedActivation {
  int epilogue = 0;  // r4dx_epilogue_none means "no pre-quantized input provided"
  const void* data = nullptr;
  const float* scale = nullptr;
  const void* transform_id = nullptr;  // trellis only: the linear's trellis_suh.data()
  int64_t part_stride = 0;             // trellis only: elements between the parts in `data`
};

// x: device bf16 [M, K], row-major, CONTIGUOUS (row stride exactly K -- every r4d_gemm_*_nt_m64
// entry point reads its A/C operands at a hardcoded stride of K/N respectively; there is no
// strided-view form to call into). y: device bf16 [M, N], row-major, contiguous, disjoint from x.
// `arena` supplies this call's activation-quant scratch (w4a16's f16 cast, w4a8's int8 quant,
// mxfp4's fp8 quant, trellis's transformed parts and split-group partials -- at most 2 * 64 * K
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
// quant/cast launch entirely and feeds `pre->data`/`pre->scale` (offset per <=64-row chunk exactly
// like `x` is) straight to the GEMM -- see PreQuantizedActivation's doc above. `x` is still
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
