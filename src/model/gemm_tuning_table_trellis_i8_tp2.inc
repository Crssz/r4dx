// The tuning table of the int8 x int8 prefill GEMM (R4DX_PREFILL_INT8, docs/int8-prefill.md "Tensor parallel") for the shard shapes
// of a TP = 2 rank: per (N, K, KB) the (skw, skg) of r4d_gemm_trellis_nt_i8 at M = 256. linear.cpp includes this file inside its own
// namespace; PlanTrellisI8(..., tp_shard = true) reads it (and only it) for a linear with QuantLinear::trellis_tp_shard, so a
// rank-shard shape that equals a TP = 1 class's (the rank's attn.qg is gdn.in_proj_z's 6144 x 5120) still needs its own row.
//
// EMPTY: no row here has been measured yet (the writing session had no GPU). A shard shape without a row runs the f16 kernel, with
// the once-per-shape stderr notice, so R4DX_PREFILL_INT8_TP2=1 is correct but changes nothing until this file is filled.
// The one row below, N == 0, is the placeholder of an empty table (a C++ array cannot be empty): no plan ever matches it, and the
// CPU test skips it. Fill it from the GPU (the rank shapes, both rates; the bench's --tp 2 selects them):
//   tool_int8_gemm_proto.exe --tp 2 --mode time --kb both --emit-rows src/model/gemm_tuning_table_trellis_i8_tp2.inc
// (the rule is the per-128 table's: best verified median, then the smallest skg within 1%) and paste the output over this file.
// The classes the bench times, with the K / 128 each (skw, skg) must divide (kernel legality: (K / 128) % (skw * skg) == 0):
//   mlp.gate_up 17408 x 5120 (two parts, part boundary 8704), K / 128 = 40
//   mlp.down     5120 x 8704, K / 128 = 68 (so skw * skg is 2 or 4: (2,1) (2,2) (4,1))
//   gdn.in_proj_qkv 5120 x 5120, gdn.in_proj_z 3072 x 5120, attn.qg 6144 x 5120, attn.k / attn.v 512 x 5120: K / 128 = 40
//   gdn.out_proj / attn.o 5120 x 3072, K / 128 = 24
//
// {N, K, KB, skw, skg}
const TrellisI8Row kTrellisI8Tp2Table[] = {
    {0, 0, 0, 0, 0},   // placeholder (empty table)
};
