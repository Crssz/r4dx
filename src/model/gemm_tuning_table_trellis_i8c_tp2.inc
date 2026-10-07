// The tuning table of the COARSE-scale int8 x int8 prefill GEMM (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse
// scales") for the shard shapes of a TP = 2 rank (R4DX_PREFILL_INT8_TP2, "Tensor parallel"): per (N, K, KB) the (skw, skg) of
// r4d_gemm_trellis_nt_i8c at M = 256. Same shape of file, same reader and same reasons as gemm_tuning_table_trellis_i8_tp2.inc, which
// has the class list and the K / 128 each row must divide.
//
// EMPTY, as that one: a shard shape without a row runs the f16 kernel with the once-per-shape notice. The N == 0 row is the
// placeholder of an empty table; regenerate on the GPU with
//   tool_int8_gemm_proto.exe --tp 2 --mode time --kb both --emit-rows-coarse src/model/gemm_tuning_table_trellis_i8c_tp2.inc
// and paste the output over this file.
//
// {N, K, KB, skw, skg}
const TrellisI8Row kTrellisI8cTp2Table[] = {
    {0, 0, 0, 0, 0},   // placeholder (empty table)
};
