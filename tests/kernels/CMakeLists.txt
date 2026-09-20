set(R4DX_KERNEL_TESTS
  test_rmsnorm
  test_residual_add
  test_silu_mul
  test_rope
  test_quant_act_fp8
  test_mxfp4_gemm
  test_kv_write
  test_sampler
  test_attn_decode
  test_gdn_chunk_scan
  test_kernel_bandwidth
  test_fused_quant
  test_embedding_gather
  test_gather_i32
)

foreach(t ${R4DX_KERNEL_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE r4dx_core r4dx_kernels r4d_core)
  add_test(NAME ${t} COMMAND ${t})
  # The project rule is HIP device 1 ONLY (2x R9700, device 0 forbidden outright). Every test
  # calls hipSetDevice(0) assuming HIP_VISIBLE_DEVICES=1 has remapped physical device 1 to local
  # index 0; that remap previously depended entirely on tests/run_tests.ps1 exporting the
  # variable before invoking ctest, so a bare `ctest --preset win-hip` (or running a test binary
  # directly) silently ran the whole GPU suite on physical device 0. Setting it here makes the
  # test's own environment correct regardless of how ctest was invoked.
  set_tests_properties(${t} PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")
endforeach()

# test_kernel_bandwidth's golden (tests/kernels/golden/kernel_bandwidth_golden.bin) is gitignored
# (review finding, 2026-09-20 -- a regenerable ~19 MB capture, not a committed fixture); the test
# itself SKIPs (return 77) when it's absent, same convention tests/model's golden-data tests use.
set_tests_properties(test_kernel_bandwidth PROPERTIES SKIP_RETURN_CODE 77)

# test_fused_quant (docs/r9700.md R2/P2) additionally needs r4dx_model_kernels for
# r4dx_model_cast_bf16_to_f16 (the "old" w4a16 cast path it byte-diffs the new fused f16 epilogue
# against) and src/model/kernels/model_kernels.h's declaration.
target_include_directories(test_fused_quant PRIVATE "${CMAKE_SOURCE_DIR}/src/model")
target_link_libraries(test_fused_quant PRIVATE r4dx_model_kernels)
