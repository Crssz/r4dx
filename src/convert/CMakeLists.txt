# HF checkpoint -> r4dx container weight converter (docs/container-format.md is the contract).
#
# r4dx_convert (INTERFACE): the header-only quantize/pack/safetensors-IO library under include/ --
# reused as-is by tests/convert/*.cpp (CTest unit tests link this directly, no separate static lib
# needed since everything here is templates/inline functions). Depends on r4dx_core (src/core) for
# the bf16/f16/fp8 host<->float conversions, and on the vendored nlohmann/json for both the
# safetensors JSON header and the container's own header/metadata.
add_library(r4dx_convert INTERFACE)
target_include_directories(r4dx_convert INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(r4dx_convert INTERFACE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(r4dx_convert INTERFACE r4dx_core)

# r4dx_convert_gemma: the Gemma 4 branch's tensor layout (gemma_layout.hpp -- names, shape checks,
# coverage audit). Plain C++ over std::function handles: r4dx-convert drives it with its own closures,
# tests/convert/test_gemma_layout.cpp with a recording kit. No HIP, no weights.
add_library(r4dx_convert_gemma STATIC gemma_layout.cpp)
target_include_directories(r4dx_convert_gemma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_convert_gemma PUBLIC r4dx_convert)

# r4dx_convert_lfm2: the LFM2 (lfm2_moe) branch's tensor layout (lfm2_layout.hpp -- names, shape checks, per-expert
# linears, coverage audit). Same shape as r4dx_convert_gemma; tests/convert/test_lfm2_layout.cpp drives it.
add_library(r4dx_convert_lfm2 STATIC lfm2_layout.cpp)
target_include_directories(r4dx_convert_lfm2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_convert_lfm2 PUBLIC r4dx_convert)

# r4dx_convert_llama: the dense Llama branch's tensor layout (llama_layout.hpp -- names, shape checks, coverage audit;
# docs/nanbeige-plan.md 4). Same shape as r4dx_convert_lfm2; tests/convert/test_llama_layout.cpp drives it.
add_library(r4dx_convert_llama STATIC llama_layout.cpp)
target_include_directories(r4dx_convert_llama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_convert_llama PUBLIC r4dx_convert)

# r4dx_convert_qwen4exp: the qwen4_exp branch -- its tensor layout, coverage audit, per-(layer, class)
# trellis KB rule and n-gram sidecar record (qwen4exp_layout.hpp), and the virtual AWQ source that
# presents compressed-tensors pack-quantized linears as bf16 `<b>.weight` tensors (awq_source.hpp).
# Plain C++, no HIP, no weights.
add_library(r4dx_convert_qwen4exp STATIC qwen4exp_layout.cpp awq_source.cpp)
target_include_directories(r4dx_convert_qwen4exp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_convert_qwen4exp PUBLIC r4dx_convert)

# r4dx_convert_qwen35moe: the qwen3_5_moe branch (Ornith-1.5-122B, docs/qwen35moe-plan.md) -- its tensor layout and coverage
# audit (qwen35moe_layout.hpp, built on qwen4_exp's Kit) and the virtual per-expert source over HF's fused expert tensors
# (fused_experts_source.hpp). Plain C++, no HIP, no weights.
add_library(r4dx_convert_qwen35moe STATIC qwen35moe_layout.cpp fused_experts_source.cpp)
target_include_directories(r4dx_convert_qwen35moe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_convert_qwen35moe PUBLIC r4dx_convert r4dx_convert_qwen4exp)

if(R4DX_BUILD_CONVERT)
  add_executable(r4dx-convert main.cpp)
  target_link_libraries(r4dx-convert PRIVATE r4dx_convert r4dx_convert_gemma r4dx_convert_lfm2 r4dx_convert_llama r4dx_convert_qwen4exp r4dx_convert_qwen35moe)
endif()
