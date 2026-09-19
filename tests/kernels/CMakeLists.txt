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
