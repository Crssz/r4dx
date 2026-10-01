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

if(R4DX_BUILD_CONVERT)
  add_executable(r4dx-convert main.cpp)
  target_link_libraries(r4dx-convert PRIVATE r4dx_convert r4dx_convert_gemma)
endif()
