# tests/model/qwen35moe: one executable + one ctest per file, registered explicitly below (like tests/model/qwen4exp).
#   q35moe_add_test(<name>)   links r4dx_model (Container / Model / TextModel of the arch)
#   SINGLE_DEVICE             runs on device 1 (HIP_VISIBLE_DEVICES=1)
#   TWO_DEVICES               removes HIP_VISIBLE_DEVICES from the test's environment: both cards visible
# Every test gets the labels q35moe and q35moe_gpu and SKIP_RETURN_CODE 77 (a missing fixture / device is a SKIP, never a PASS).
# Fixtures (tools/reference/qwen35moe/tiny.py + tiny_golden.py + r4dx-convert) are found through R4DX_Q35MOE_TINY_DIR.

function(q35moe_add_test name)
  cmake_parse_arguments(ARG "SINGLE_DEVICE;TWO_DEVICES" "" "" ${ARGN})
  add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
  target_link_libraries(${name} PRIVATE r4dx_model r4dx_model_qwen4exp_kernels r4dx_core r4d_core)
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "q35moe;q35moe_gpu" SKIP_RETURN_CODE 77)
  if(ARG_SINGLE_DEVICE)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")
  elseif(ARG_TWO_DEVICES)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT_MODIFICATION "HIP_VISIBLE_DEVICES=unset:")
  endif()
endfunction()

q35moe_add_test(test_model_q35moe_tiny SINGLE_DEVICE)
q35moe_add_test(test_model_q35moe_grouped SINGLE_DEVICE)
q35moe_add_test(test_q35moe_grouped_moe_kernel SINGLE_DEVICE)
q35moe_add_test(test_model_q35moe_spec SINGLE_DEVICE)
q35moe_add_test(test_q35moe_dflash2_golden SINGLE_DEVICE)
# --tp 2 (docs/qwen35moe-tp-design.md P1): two rank models on ONE device (EmulatedComm) against the single-device model, bf16 and trellis tiny containers.
q35moe_add_test(test_model_q35moe_tp_tiny SINGLE_DEVICE)
q35moe_add_test(test_q35moe_tp_container SINGLE_DEVICE)

# test_q35moe_tp_shard (docs/qwen35moe-tp-design.md P0 / gate G1): the --tp 2 sharding contract of qwen3_5_moe -- Qwen35MoeConfig::Shard,
# tp::RuleFor's qwen3_5_moe table, the trellis byte plans, and (when E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx exists, read-only)
# decode(rank slice) == slice(decode(full)) on the real container's bytes. CPU only, no HIP call, so plain add_test (a missing
# container is a printed SKIP inside the test, not a failure).
add_executable(test_q35moe_tp_shard test_q35moe_tp_shard.cpp)
target_link_libraries(test_q35moe_tp_shard PRIVATE r4dx_tp_shard r4dx_convert r4dx_core)
add_test(NAME test_q35moe_tp_shard COMMAND test_q35moe_tp_shard)
set_tests_properties(test_q35moe_tp_shard PROPERTIES LABELS "q35moe" ENVIRONMENT "HIP_VISIBLE_DEVICES=-1")

# tool_q35moe_tp_step: the P2 economic gate (layer split vs one TP rank without a peer, GPU-event spans); not a ctest.
add_executable(tool_q35moe_tp_step tool_q35moe_tp_step.cpp)
target_link_libraries(tool_q35moe_tp_step PRIVATE r4dx_model r4dx_model_qwen4exp_kernels r4dx_core r4d_core)

# tool_q35moe_dflash: the real-container DFlash2 benchmark / lossless audit (not a ctest: it needs the 52 GB Ornith container and both cards).
add_executable(tool_q35moe_dflash tool_q35moe_dflash.cpp)
target_link_libraries(tool_q35moe_dflash PRIVATE r4dx_model r4dx_model_qwen4exp_kernels r4dx_core r4d_core)
