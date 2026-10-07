// The tuning table of the COARSE-scale int8 x int8 prefill GEMM (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse
// scales"): per (N, K, KB) the (skw, skg) of r4d_gemm_trellis_nt_i8c. Same shape of file as gemm_tuning_table_trellis_i8.inc
// (linear.cpp includes both inside its own namespace, PlanTrellisI8(..., coarse = true) reads this one,
// tests/model/test_prefill_int8_cpu.cpp checks every row against the kernel's own legality rule and that every model class
// has one at both rates).
//
// SEEDED, NOT MEASURED: the rows are the per-128 table's (the kernel's legality is the same, r4d_gemm_trellis_nt_i8_check;
// any legal row is correct, the choice moves the time and, through the order of the fp32 partial sums, the last bits). The coarse kernel
// has no per-128 rescale and a lighter K loop, so its best (skw, skg) is likely not the per-128 one's; regenerate on the GPU with
//   tool_int8_gemm_proto.exe --mode time --kb both --emit-rows-coarse src/model/gemm_tuning_table_trellis_i8c.inc
// (the rule is the same: best verified median, then the smallest skg within 1%) and paste the output over this file.
//
// {N, K, KB, skw, skg}
const TrellisI8Row kTrellisI8cTable[] = {
    {34816, 5120, 4, 2, 1},   // mlp.gate_up KB4 (seed: the per-128 row)
    {34816, 5120, 5, 2, 1},   // mlp.gate_up KB5
    {5120, 17408, 4, 2, 2},   // mlp.down KB4
    {5120, 17408, 5, 4, 1},   // mlp.down KB5
    {10240, 5120, 4, 4, 1},   // gdn.in_proj_qkv KB4
    {10240, 5120, 5, 4, 1},   // gdn.in_proj_qkv KB5
    {6144, 5120, 4, 4, 1},    // gdn.in_proj_z KB4
    {6144, 5120, 5, 4, 2},    // gdn.in_proj_z KB5
    {5120, 6144, 4, 4, 2},    // gdn.out_proj / attn.o KB4
    {5120, 6144, 5, 4, 1},    // gdn.out_proj / attn.o KB5
    {12288, 5120, 4, 4, 1},   // attn.qg KB4
    {12288, 5120, 5, 4, 1},   // attn.qg KB5
    {1024, 5120, 4, 4, 1},    // attn.k / attn.v KB4
    {1024, 5120, 5, 4, 1},    // attn.k / attn.v KB5
};
