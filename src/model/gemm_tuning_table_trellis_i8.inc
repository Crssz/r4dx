// The tuning table of the int8 x int8 prefill GEMM (R4DX_PREFILL_INT8, docs/int8-prefill.md "Production path"):
// per (N, K, KB) the (skw, skg) of r4d_gemm_trellis_nt_i8 -- skw K slices resident per workgroup, skg K groups
// across the grid -- that linear class runs at M = 256. linear.cpp includes this file inside its own namespace and
// PlanTrellisI8 reads it; tests/model/test_prefill_int8_cpu.cpp checks every row against the kernel's own
// legality rule (r4d_gemm_trellis_nt_i8_check) and that every model class has one at both rates.
//
// SEED rows: the best verified configuration per class of the int8gemm prototype's timing run
// (E:\models\r4dx\int8gemm\all.log, HIP device 1, R9700 gfx1201, the kernel source of the production unit up to one
// scalar select), median of 11 rounds, cold weights, the time in us in the comment. The rule is "best median,
// then the smallest skg within 1%" (tests/kernels/tool_int8_gemm_proto.exe --emit-rows regenerates this file with
// it; the first GPU session re-measures against the retuned f16 table and replaces these). Any row is legal
// whatever its speed: the choice moves the time and (through the fp32 rescale order) the last bits, nothing else.
// A (class, KB) without a row runs the f16 kernel, with a one-line notice per shape.
//
// {N, K, KB, skw, skg}
const TrellisI8Row kTrellisI8Table[] = {
    {34816, 5120, 4, 2, 1},   // mlp.gate_up KB4 607 us
    {34816, 5120, 5, 2, 2},   // mlp.gate_up KB5 645 us
    {5120, 17408, 4, 4, 1},   // mlp.down KB4 308 us
    {5120, 17408, 5, 2, 2},   // mlp.down KB5 327 us
    {10240, 5120, 4, 2, 2},   // gdn.in_proj_qkv KB4 192 us
    {10240, 5120, 5, 2, 2},   // gdn.in_proj_qkv KB5 206 us
    {6144, 5120, 4, 4, 1},    // gdn.in_proj_z KB4 123 us
    {6144, 5120, 5, 4, 2},    // gdn.in_proj_z KB5 130 us
    {5120, 6144, 4, 4, 2},    // gdn.out_proj / attn.o KB4 120 us
    {5120, 6144, 5, 2, 4},    // gdn.out_proj / attn.o KB5 127 us
    {12288, 5120, 4, 4, 1},   // attn.qg KB4 227 us
    {12288, 5120, 5, 2, 2},   // attn.qg KB5 243 us
    {1024, 5120, 4, 4, 2},    // attn.k / attn.v KB4 29 us
    {1024, 5120, 5, 4, 1},    // attn.k / attn.v KB5 29 us
};
