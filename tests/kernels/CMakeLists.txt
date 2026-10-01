set(R4DX_KERNEL_TESTS
  test_rmsnorm
  test_residual_add
  test_silu_mul
  test_rope
  test_kv_write
  test_sampler
  test_attn_decode
  # prefill M1: split-KV prefill attention -- splits<=1 bit-identical to the plain launch, split
  # outputs at the unsplit kernel's own error against a CPU fp32 reference, deterministic.
  test_attn_prefill_splitkv
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
  # quant2 Q2a/Q2b rotation kernels (docs/quant2.md sections 3-4): r4dx_rotate_residual_bf16,
  # r4dx_hadamard_inplace_bf16, r4dx_silu_mul_hadamard_bf16 and src/model/attention's
  # r4dx_model_attn_gate_mul_hadamard_bf16, each against rotation_ref.hpp's fp64 reference.
  # CPU-reference only, always on; both check precondition throws, hence the /EHc- entry below.
  test_rotate_residual
  test_attn_gate_mul_hadamard
  # Trellis weights, milestone M1 (docs/trellis-kernel.md 4.2-4.5, 6): libr4d's trellis decode,
  # r4d_trellis_reconstruct_f16 and the raw r4d_gemm_trellis_nt_m64 bit-exact against the CPU
  # decode (trellis_ref.hpp) and the goldens tools/reference/trellis_golden.py writes into
  # golden/trellis/. The golden part runs last and returns 77 when that directory is absent (after
  # every CPU-reference check has passed); it checks precondition throws, hence /EHc- below.
  test_trellis_decode
  # Trellis weights, milestone M2 (docs/trellis-kernel.md 4.3-4.5, 4.8, 6): the input transform
  # r4dx_trellis_input_bf16 bit-exact against trellis_ref.hpp's fp32 emulation (and libr4d's wave
  # FWHT against FwhtLds), and the whole linear -- transform, r4d_gemm_trellis_nt_m64 and its
  # FWHT/svh/bf16 epilogue -- against fp64, with row identity for every trellis tuning row,
  # determinism, the ticket reset and the precondition throws. Same golden/77 and /EHc- conventions
  # as test_trellis_decode.
  test_trellis_input
  test_trellis_gemm
  # The M = 256 trellis GEMM (docs/trellis-m256.md, R4DX_PREFILL_CHUNK=256): one launch against four of
  # the shipped M = 64 kernel, byte for byte, on synthetic data, for every class the model wires to it at
  # both rates (plus run-to-run, ticket reset, a negative control and the refusal messages). Skips (77)
  # only without a HIP device; needs no golden data.
  test_trellis_m256
  # Gemma 4 device pieces (docs/gemma4-plan.md 3.4 / 3.5, task M1-15; the GPU step is M1-16:
  #   ctest -R "rope_proportional|rmsnorm_noscale|gelu_mul|postnorm|softcap|embed_scaled").
  # CPU references live in gemma_ref.hpp; always-on GPU tests, no golden data.
  test_rope_proportional
  test_rmsnorm_noscale
  test_gelu_mul
  test_gemma_postnorm
  test_embed_scaled
  test_widen_softcap
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
set_tests_properties(test_trellis_decode test_trellis_input test_trellis_gemm test_trellis_m256 PROPERTIES SKIP_RETURN_CODE 77)

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
# test_fused_quant.cpp checks that the rmsnorm / residual_rmsnorm / silu_mul host entries throw on
# an epilogue value they do not know (docs/trellis-kernel.md 5.3).
set_source_files_properties(test_dflash_attn.cpp test_dflash_conv.cpp test_topk_lse.cpp
                            test_rope_mrope3.cpp test_rotate_residual.cpp
                            test_attn_gate_mul_hadamard.cpp test_trellis_decode.cpp
                            test_trellis_input.cpp test_trellis_gemm.cpp test_fused_quant.cpp
                            test_trellis_m256.cpp test_rope_proportional.cpp
                            test_gemma_postnorm.cpp
                            PROPERTIES COMPILE_OPTIONS "/EHc-")

# test_attn_gate_mul_hadamard's kernel lives in r4dx_model_attention (src/model/attention), not
# r4dx_kernels; so does r4dx_model_attn_gate_mul_bf16, test_trellis_input's v1 producer for the
# fused gate-mul (M5).
target_link_libraries(test_attn_gate_mul_hadamard PRIVATE r4dx_model_attention)
target_link_libraries(test_trellis_input PRIVATE r4dx_model_attention)
# test_trellis_m256 asks src/model/linear.cpp for the shipped M = 64 row (PickTuning) and the plan
# ApplyLinear runs the M = 256 kernel with (PlanTrellisM256).
target_link_libraries(test_trellis_m256 PRIVATE r4dx_model_linear)

# Gemma 4 stage-0 reference attention (docs/gemma4-plan.md 3.3, task M1-17; GPU step M1-18:
# `ctest -R attn_ref_gemma`). The kernel lives in r4dx_model_attention (src/model/attention/src/
# attn_ref.hip); the test also uses its SlidingKvCache / SlidingRingGeometry headers.
add_executable(test_attn_ref_gemma test_attn_ref_gemma.cpp)
target_link_libraries(test_attn_ref_gemma PRIVATE r4dx_model_attention r4dx_core r4dx_kernels r4d_core)
add_test(NAME test_attn_ref_gemma COMMAND test_attn_ref_gemma)
set_tests_properties(test_attn_ref_gemma PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")

# Pure CPU (no HIP call, no kernel library): the sliding-window / ring arithmetic of
# third_party/libr4d/r4d_attn_window.h and SlidingRingGeometry against brute force and a simulated
# ring (task M1-17; M1-32's kernels call the same header). Plain add_test(): they always run.
add_executable(test_attn_window_math test_attn_window_math.cpp)
target_include_directories(test_attn_window_math PRIVATE "${CMAKE_SOURCE_DIR}/third_party/libr4d")
add_test(NAME test_attn_window_math COMMAND test_attn_window_math)
add_executable(test_ring_math test_ring_math.cpp)
target_include_directories(test_ring_math PRIVATE "${CMAKE_SOURCE_DIR}/third_party/libr4d"
                           "${CMAKE_SOURCE_DIR}/src/model/attention/include")
add_test(NAME test_ring_math COMMAND test_ring_math)

# test_widen_softcap: r4dx_model_widen_softcap_bf16_to_f32 lives in r4dx_model_kernels (model_kernels.hip).
target_include_directories(test_widen_softcap PRIVATE "${CMAKE_SOURCE_DIR}/src/model")
target_link_libraries(test_widen_softcap PRIVATE r4dx_model_kernels)

# test_rope_proportional_cpu: pure CPU (no HIP call, no kernel library) -- pins gemma_ref.hpp's HF
# rope construction against the kernel's pair loop (the geometry r4dx_rope_proportional_bf16 runs).
add_executable(test_rope_proportional_cpu test_rope_proportional_cpu.cpp)
add_test(NAME test_rope_proportional_cpu COMMAND test_rope_proportional_cpu)

# test_rotation_ref_cpu: pure CPU (no HIP call, no kernel library) -- pins rotation_ref.hpp, the
# fp64 reference the two tests above compare against, to the rotation contract's definitions
# (dense Sylvester FWHT, the closed-form rows of Q, norms/inverses, TP K-slicing).
add_executable(test_rotation_ref_cpu test_rotation_ref_cpu.cpp)
add_test(NAME test_rotation_ref_cpu COMMAND test_rotation_ref_cpu)

# tool_sampler_bench: Milestone 6 S1 item 4 -- r4dx_topk_lse_f32's kernel time (hip events) and the
# host cost of SampleFromSummary vs the full-vocab SampleCanonical/SampleLegacy per token. Prints
# numbers for docs/sampling.md and asserts no pass/fail contract, so it follows tests/model's
# "built but deliberately never add_test()'d" tool convention (tool_hseed_drift et al) and never
# affects the ctest count. Run manually:
#   $env:HIP_VISIBLE_DEVICES='1'; build\win-hip\tests\kernels\tool_sampler_bench.exe
add_executable(tool_sampler_bench tool_sampler_bench.cpp)
target_link_libraries(tool_sampler_bench PRIVATE r4dx_core r4dx_kernels r4d_core)

# tool_attn_prefill_bench: prefill M0 (tools/prefill) -- the production prefill attention call at
# context depth D, today's 64-row chunk against bigger chunks and an emulated split-KV partial pass.
# Built, never add_test()'d, same convention as tool_sampler_bench.
add_executable(tool_attn_prefill_bench tool_attn_prefill_bench.cpp)
target_link_libraries(tool_attn_prefill_bench PRIVATE r4dx_core r4d_core)

# tool_attn_prefill_precision: prefill M1 diagnosis -- plain, exact-wide and split-KV prefill
# attention against an fp64 CPU reference (docs/prefill.md "M1 results"). Built, never add_test()'d.
add_executable(tool_attn_prefill_precision tool_attn_prefill_precision.cpp)
target_link_libraries(tool_attn_prefill_precision PRIVATE r4dx_core r4dx_kernels r4d_core)

# tool_trellis_gemm_bench: trellis milestone M1's measurements (docs/trellis-kernel.md 4.6, 6
# "Benches"): the 336-GEMM decode-step replay against production w4a16, op rates and WMMA/VALU/
# memory overlap, the split-group tail and the M = 64 prefill shape; and M2's full-linear replay
# (--modes full: input transforms + GEMMs with the epilogue against q2ab's casts, Hadamards and
# rotations, KB 4 or the 4.5 bpw mix, the joint M = 1 / M = 8 tuning pick whose --inc-out wrote
# src/model/gemm_tuning_table_trellis.inc, the trellis rows linear.cpp's PickTuning reads); writes
# JSON to --out. Built,
# never add_test()'d (the real runs are scheduled GPU time):
#   $env:HIP_VISIBLE_DEVICES='1'; build\win-hip\tests\kernels\tool_trellis_gemm_bench.exe --out x.json
# Its device side (fills, the clock probe, the op-rate and pipe loops) is a hipcc object of its own,
# built exactly like src/kernels' objects and including libr4d's r4d_trellis_dq.h, so the pipe loops
# run the GEMM's own decode. It links r4dx_model_linear for PickTuning, production's w4a16 dispatch,
# and r4dx_model_attention for the attention gate-mul pair it times (q2ab's Hadamard form against the
# plain one v1 runs), and catches the raw GEMM's rejections of illegal tunings, hence /EHc-.
set(R4DX_TQ_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tool_trellis_gemm_bench_kernels.hip")
set(R4DX_TQ_BENCH_OBJ "${CMAKE_CURRENT_BINARY_DIR}/objs/tool_trellis_gemm_bench_kernels.obj")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/objs")
set(R4DX_TQ_BENCH_FLAGS
  -O3 -std=c++17 --offload-arch=${R4D_GFX_ARCH}
  -Wno-unused-result -ffp-contract=off
  --rocm-path=${R4D_ROCM_ROOT}
  -DNDEBUG
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
  -I "${CMAKE_SOURCE_DIR}/third_party/libr4d"
  # clock_probe_device.h: the clock probe and timer stamp, shared with r4dx's in-model probe.
  -I "${CMAKE_SOURCE_DIR}/src/model/kernels"
)
if(EXISTS "${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
  list(APPEND R4DX_TQ_BENCH_FLAGS --rocm-device-lib-path=${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode)
endif()
add_custom_command(
  OUTPUT "${R4DX_TQ_BENCH_OBJ}"
  COMMAND "${CMAKE_COMMAND}" -E env
          "PATH=${R4D_ROCM_ROOT}/bin;${R4D_ROCM_ROOT}/lib/llvm/bin;$ENV{PATH}"
          "${R4D_ROCM_ROOT}/bin/hipcc.exe" ${R4DX_TQ_BENCH_FLAGS} -c "${R4DX_TQ_BENCH_SRC}"
          -o "${R4DX_TQ_BENCH_OBJ}"
  DEPENDS "${R4DX_TQ_BENCH_SRC}" "${CMAKE_SOURCE_DIR}/third_party/libr4d/r4d_trellis_dq.h"
          "${CMAKE_SOURCE_DIR}/third_party/libr4d/r4d_gdn_wmma.h"
          "${CMAKE_SOURCE_DIR}/src/model/kernels/clock_probe_device.h"
  COMMENT "hipcc (${R4D_GFX_ARCH}) tool_trellis_gemm_bench_kernels.obj"
  VERBATIM
)
set_source_files_properties("${R4DX_TQ_BENCH_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_executable(tool_trellis_gemm_bench tool_trellis_gemm_bench.cpp "${R4DX_TQ_BENCH_OBJ}")
target_include_directories(tool_trellis_gemm_bench BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(tool_trellis_gemm_bench PRIVATE r4dx_model_linear r4dx_model_attention r4dx_core
                      r4dx_kernels r4d_core)
set_source_files_properties(tool_trellis_gemm_bench.cpp PROPERTIES COMPILE_OPTIONS "/EHc-")

# ---- tensor-parallel all-reduce, phase P3 (docs/tp.md 6.3, 10.1) ---------------------------------
# All four link r4dx_tp (src/model/tp: TpGroup, HostMailboxComm) and share tp_ar_harness.h. They call
# the r4dx_tp_* extern "C" entry points, which throw, hence /EHc- (see the comment above).
#
# test_tp_allreduce_cpu_peer: ONE GPU (device 1, like every test here) against a CPU thread acting
# as rank 1 through a real pinned mailbox -- always on.
add_executable(test_tp_allreduce_cpu_peer test_tp_allreduce_cpu_peer.cpp)
target_link_libraries(test_tp_allreduce_cpu_peer PRIVATE r4dx_tp r4dx_core r4dx_kernels r4d_core)
add_test(NAME test_tp_allreduce_cpu_peer COMMAND test_tp_allreduce_cpu_peer)
set_tests_properties(test_tp_allreduce_cpu_peer PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")

# test_tp_allreduce_2gpu: BOTH GPUs, LABEL tp2gpu, and opt-in (docs/tp.md 10.1 "Two-GPU tests are
# opt-in, not just labelled"). Deliberately NO HIP_VISIBLE_DEVICES property: a per-test value would
# be the only way it could ever see two devices, and under the win-hip test preset
# (HIP_VISIBLE_DEVICES=1) it must see one. The binary itself exits 77 (SKIP) unless R4DX_TP2GPU=1
# and two devices are visible; tests\run_tests.ps1 -TwoGpu is what sets that up, and the default
# run_tests.ps1 path excludes the label (-LE tp2gpu).
add_executable(test_tp_allreduce_2gpu test_tp_allreduce_2gpu.cpp)
target_link_libraries(test_tp_allreduce_2gpu PRIVATE r4dx_tp r4dx_core r4dx_kernels r4d_core)
add_test(NAME test_tp_allreduce_2gpu COMMAND test_tp_allreduce_2gpu)
set_tests_properties(test_tp_allreduce_2gpu PROPERTIES LABELS tp2gpu SKIP_RETURN_CODE 77)

# tool_tp_ar_stress (G5: `--count 10000000 --pattern decode`) and tool_tp_ar_latency (G5's decode-
# pattern L at 10 / 80 / 640 KiB): built, never add_test()'d -- two-GPU gate tools, run by hand with
# HIP_VISIBLE_DEVICES unset and the production server stopped (docs/tp.md 11, P3).
# tool_tp_submit_probe (docs/tp.md Appendix B N56): ONE device, a few ms of GPU work -- when does
# the runtime submit an enqueued kernel, and what forces it? The evidence behind the TP submission
# bounding (N57). Same built-only convention.
foreach(t tool_tp_ar_stress tool_tp_ar_latency tool_tp_submit_probe)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE r4dx_tp r4dx_core r4dx_kernels r4d_core)
endforeach()
set_source_files_properties(test_tp_allreduce_cpu_peer.cpp test_tp_allreduce_2gpu.cpp tool_tp_ar_stress.cpp
                            tool_tp_ar_latency.cpp tool_tp_submit_probe.cpp PROPERTIES COMPILE_OPTIONS "/EHc-")

# test_fused_quant (docs/r9700.md R2/P2) additionally needs r4dx_model_kernels for
# r4dx_model_cast_bf16_to_f16 (the "old" w4a16 cast path it byte-diffs the new fused f16 epilogue
# against) and src/model/kernels/model_kernels.h's declaration.
target_include_directories(test_fused_quant PRIVATE "${CMAKE_SOURCE_DIR}/src/model")
target_link_libraries(test_fused_quant PRIVATE r4dx_model_kernels)
