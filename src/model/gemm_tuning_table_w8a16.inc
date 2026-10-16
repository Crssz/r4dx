// The w8a16 GEMM's tuning rows (docs/w8a16.md 4.5): r4dx_gemm_w8a16_nt_m64, keyed by (N, K, M-band, group).
// tests/kernels/tool_w8a16_gemm_bench.exe --inc-out regenerates this file: an M = 1 row per (N, K, G) and
// M = 32 / 64 prefill rows. Row format (src/model/linear.h):
//   {Layout::kW8a16, N, K, M, {WV, SK, MB, NPW, NT}, group = G}
// linear.cpp includes this file inside its own namespace (w8a16) and PickTuning reads it for kW8a16 only.
//
// EMPTY: a one-row placeholder with N == 0, which matches no linear (a C++ array cannot be empty; the
// gemm_tuning_table_trellis_i8_tp2.inc precedent). Until the bench has run, every w8a16 linear takes
// linear.cpp's W8a16FallbackTuning.
static const GemmTuningRow kGemmTuningTable[] = {
    {Layout::kW8a16, 0, 0, 0, {1, 1, 1, 1, 1}, 0},  // placeholder, matches nothing
};
