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
  # DFlash2 drafter device pieces (docs/dflash2.md "Kernels", Milestone 5 S1). Every one of these
  # ends with a fixture-A comparison and so declares SKIP_RETURN_CODE 77 below.
  test_rope_neox
  test_topk16
  test_dflash_attn
  test_dflash_conv
  test_rmsnorm_plain
  # Milestone 6 S1 -- sampling (docs/sampling.md). test_topk_lse is the only GPU test of the three;
  # the other two are pure CPU (header-only sampler/summary-sampler) and are registered here simply
  # because that is where the sampler's own tests already live (test_sampler).
  test_topk_lse
  test_sampler_canonical
  test_summary_sampler
  # Milestone 8 stage 3 -- the vision tower's own device primitives (docs/vision.md). CPU-reference
  # only (no golden data, no container), so it is an ordinary always-on GPU test with no
  # SKIP_RETURN_CODE; the golden comparison for these same kernels is tests/vision/test_vision_tower.
  test_vision_kernels
  # Milestone 8 stage 4 -- the 3-axis multimodal mrope (docs/vision.md "Text-side splicing").
  # CPU-reference only, always on. Its last case checks a rejected section vector, hence the
  # /EHc- entry below (see that comment for why extern "C" throws are otherwise uncatchable).
  test_rope_mrope3
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

# The five DFlash2 kernel tests each run their CPU-reference checks unconditionally and then
# compare against tools/reference/golden_out/dflash2/fixture_a, which is gitignored and
# regenerable (docs/dflash2.md section 9 -- `python tools/reference/dflash2_ref.py --gen-fixtures
# all --seed 0`). When it is absent they return 77 (CTest SKIPPED) AFTER the CPU-reference part
# has passed, and 1 if that part failed -- so a skip never hides a real regression.
set_tests_properties(test_rope_neox test_topk16 test_dflash_attn test_dflash_conv
                     test_rmsnorm_plain PROPERTIES SKIP_RETURN_CODE 77)

# `/EHc-` -- required for the precondition checks in test_dflash_attn.cpp and test_dflash_conv.cpp
# to be able to CATCH anything at all.
#
# The whole project compiles with CMake's Windows default `/EHsc`, whose `c` means "assume every
# extern \"C\" function is nothrow". Every kernel entry point in src/kernels is `extern "C"` and
# several of them (13 throw sites as of Milestone 5, plus R4DX_HIP_CHECK) report a violated
# precondition or a failed HIP call by throwing std::runtime_error. Under `/EHsc` the CALLER emits
# no unwind edge for those calls, so a throw that really does happen unwinds into a frame marked
# nothrow and the process is killed immediately -- exit code 0xC0000409, no message, and neither
# `catch (const std::exception&)` nor `catch (...)` ever runs. Verified on this toolchain: the
# kernels object itself emits the correct MSVC EH ABI (_CxxThrowException / __CxxFrameHandler3),
# so the defect is purely the caller-side nothrow assumption.
#
# `/EHc-` turns that assumption off for these test targets only, which is the narrowest change
# that makes the preconditions testable. NOTE that this does NOT fix the same exposure in
# r4dx-cli / r4dx-server / src/model, which still call the identical entry points under `/EHsc`
# and would still be killed rather than unwound by a HIP error raised inside src/kernels -- see
# docs/dflash2.md's "Kernels" subsection, which records this as an open project-wide question
# rather than something a kernel stage should decide unilaterally.
set_source_files_properties(test_dflash_attn.cpp test_dflash_conv.cpp test_topk_lse.cpp
                            test_rope_mrope3.cpp PROPERTIES COMPILE_OPTIONS "/EHc-")

# tool_sampler_bench: Milestone 6 S1 item 4 -- r4dx_topk_lse_f32's kernel time (hip events) and the
# host cost of SampleFromSummary vs the full-vocab SampleCanonical/SampleLegacy per token. Prints
# numbers for docs/sampling.md and asserts no pass/fail contract, so it follows tests/model's
# "built but deliberately never add_test()'d" tool convention (tool_hseed_drift et al) and never
# affects the ctest count. Run manually:
#   $env:HIP_VISIBLE_DEVICES='1'; build\win-hip\tests\kernels\tool_sampler_bench.exe
add_executable(tool_sampler_bench tool_sampler_bench.cpp)
target_link_libraries(tool_sampler_bench PRIVATE r4dx_core r4dx_kernels r4d_core)

# test_fused_quant (docs/r9700.md R2/P2) additionally needs r4dx_model_kernels for
# r4dx_model_cast_bf16_to_f16 (the "old" w4a16 cast path it byte-diffs the new fused f16 epilogue
# against) and src/model/kernels/model_kernels.h's declaration.
target_include_directories(test_fused_quant PRIVATE "${CMAKE_SOURCE_DIR}/src/model")
target_link_libraries(test_fused_quant PRIVATE r4dx_model_kernels)
