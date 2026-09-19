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

if(R4DX_BUILD_CONVERT)
  add_executable(r4dx-convert main.cpp)
  target_link_libraries(r4dx-convert PRIVATE r4dx_convert)
endif()
