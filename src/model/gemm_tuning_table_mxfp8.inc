// MXFP8 tuning rows (docs/inkling2-plan.md 4.1), keyed like the w8a16 rows (layout, N, K, M band). A one-row
// placeholder (N == 0 matches nothing; a C++ array cannot be empty) until a bench has measured r4dx_gemm_mxfp8_nt on
// the Inkling2 shapes, so every mxfp8 linear takes Mxfp8FallbackTuning (src/model/linear.cpp).
static const GemmTuningRow kGemmTuningTable[] = {
    {Layout::kMxfp8, 0, 0, 0, {1, 1, 1, 1, 1}, 0},
};
