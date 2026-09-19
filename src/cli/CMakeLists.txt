# CLI text generation entry point (docs/architecture.md "src/cli/"): r4dx-cli, a single-process
# prompt-in/tokens-out (or interactive --chat) loop over r4dx::model::Model.
add_library(r4dx_cli INTERFACE)
target_include_directories(r4dx_cli INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(r4dx-cli main.cpp)
# main.cpp is the first r4dx target to link both r4dx_tokenizer (chat_template.h, built against
# third_party/nlohmann's vendored v3.11.3) and anything HIP-including (r4dx_hip_runtime's
# INTERFACE include dir is C:/opt/rocm/include, which itself ships a DIFFERENT nlohmann/json.hpp
# -- v3.12.0, a distinct ABI inline namespace). Both are found via the same unqualified
# #include "nlohmann/json.hpp", so whichever -I comes first on this TU's command line wins; listed
# here explicitly, ahead of target_link_libraries below, so it wins over the transitively-added
# ROCm path and main.cpp's calls into ChatTemplate::render() (a non-inline, cross-TU symbol built
# into r4dx_tokenizer.lib against v3.11.3) link instead of hitting an ABI-mismatched undefined
# symbol (json_abi_v3_12_0 vs json_abi_v3_11_3).
target_include_directories(r4dx-cli PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(r4dx-cli PRIVATE
  r4dx_model
  r4dx_tokenizer
  r4dx_core
  r4dx_kernels
  r4d_core
  r4dx_hip_runtime
)
