# tests/model/attention: golden test for r4dx::model::attention::AttentionLayer
# (src/model/attention/**) against tools/reference/layer_golden.py's layer 3 golden and the real
# Qwen3.8-27B checkpoint's bf16 weights (D:/models/r4dx/qwen38-27b-l4-bf16.r4dx). GPU test (HIP
# device 1 only), same ENVIRONMENT convention as tests/kernels/CMakeLists.txt.
#
# NOTE: tests/model/CMakeLists.txt did not exist when this component started (only
# src/model/CMakeLists.txt did) -- per the task brief this file was created standalone; whoever
# creates tests/model/CMakeLists.txt (the model-core agent) should add `add_subdirectory(attention)`
# to it, the same way tests/CMakeLists.txt already does for core/kernels/tokenizer/convert. See
# src/model/attention/CMakeLists.txt's own note for the matching src/model/CMakeLists.txt gap.

add_executable(test_attn_layer test_attn_layer.cpp)
target_link_libraries(test_attn_layer PRIVATE r4dx_model_attention r4dx_core r4dx_kernels r4d_core r4dx_convert)
target_compile_definitions(test_attn_layer PRIVATE
  R4DX_GOLDEN_ATTN_PATH="${CMAKE_SOURCE_DIR}/tools/reference/golden_out/layer_003_full_attention.safetensors"
  R4DX_BF16_CONTAINER_PATH="D:/models/r4dx/qwen38-27b-l4-bf16.r4dx"
)
add_test(NAME test_attn_layer COMMAND test_attn_layer)
# SKIP_RETURN_CODE 77: the test exits 77 (CTest SKIPPED, not FAILED) when its golden safetensors or
# its bf16 test container is absent -- the same convention tests/model/CMakeLists.txt applies to
# every sibling golden test.
set_tests_properties(test_attn_layer PROPERTIES
  ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
  SKIP_RETURN_CODE 77
)
