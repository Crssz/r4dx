# tests/model/qwen4exp: one executable + one ctest per file, registered explicitly below (like the other tests/ CMake files).
#   q4exp_add_test(<name> KERNELS)  links r4dx_model_qwen4exp_kernels only (kernels, host helpers; no Container/Model)
#   q4exp_add_test(<name> MODEL)    links r4dx_model (Container / Model / TextModel of the arch)
#   GPU                             adds the label q4exp_gpu
#   SINGLE_DEVICE                   runs on device 1 (HIP_VISIBLE_DEVICES=1; the test may still pick devices itself)
#   TWO_DEVICES                     removes HIP_VISIBLE_DEVICES from the test's environment: both cards visible
# Every test gets the label q4exp and SKIP_RETURN_CODE 77 (a missing fixture / device is a SKIP, never a PASS).
# Fixtures are found through env vars the test documents; tests never download.

function(q4exp_add_test name kind)
  cmake_parse_arguments(ARG "GPU;SINGLE_DEVICE;TWO_DEVICES" "" "" ${ARGN})
  add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
  if(kind STREQUAL "MODEL")
    target_link_libraries(${name} PRIVATE r4dx_model r4dx_model_qwen4exp_kernels r4dx_core r4d_core)
  else()
    target_link_libraries(${name} PRIVATE r4dx_model_qwen4exp_kernels r4dx_core r4d_core)
  endif()
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../kernels")
  add_test(NAME ${name} COMMAND ${name})
  set(labels q4exp)
  if(ARG_GPU)
    list(APPEND labels q4exp_gpu)
  endif()
  set_tests_properties(${name} PROPERTIES LABELS "${labels}" SKIP_RETURN_CODE 77)
  if(ARG_SINGLE_DEVICE)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")
  elseif(ARG_TWO_DEVICES)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT_MODIFICATION "HIP_VISIBLE_DEVICES=unset:")
  endif()
endfunction()

# CPU-only: the n-gram sidecar reader.
q4exp_add_test(test_ngram_ids KERNELS)
q4exp_add_test(test_ngram_reader KERNELS)

# Kernel tests on fixtures / random data (single device).
q4exp_add_test(test_gpu_gdn_sigmoid KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_fuse_decode KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_hc KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_moe_block KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_moe_route KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_ple KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_qsa KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_trellis_kb3 KERNELS GPU SINGLE_DEVICE)
q4exp_add_test(test_gpu_w4a16_k640 KERNELS GPU SINGLE_DEVICE)

# Model tests (single device).
q4exp_add_test(test_model_q4exp_gdn MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_layers MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_loader MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_moe_grouped MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_real_prefix MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_state MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_textmodel MODEL GPU SINGLE_DEVICE)
q4exp_add_test(test_model_q4exp_tiny MODEL GPU SINGLE_DEVICE)

# The two-card split: stage 0 / stage 1 on ordinals 0 / 1, so both cards must be visible (with one it exits 77).
q4exp_add_test(test_model_q4exp_split MODEL GPU TWO_DEVICES)
q4exp_add_test(test_q4exp_grouped_synth MODEL GPU SINGLE_DEVICE)
