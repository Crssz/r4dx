# src/model/attention: the Qwen3.5 full-attention decoder layer (docs/architecture.md "Attention
# layer"). Owns two tiny HIP kernels (fused-qg split, output-gate sigmoid-multiply) that neither
# r4d.h nor src/kernels provides, built via hipcc the same way src/kernels/CMakeLists.txt builds
# r4dx_kernels (CMake's HIP language fights clang-cl on this toolchain; docs/build-windows.md).
# R4D_ROCM_ROOT / R4D_GFX_ARCH are CACHE variables already set by third_party/CMakeLists.txt (this
# subdirectory is only ever reached after add_subdirectory(third_party) in the root CMakeLists.txt).
#
# NOTE for whoever wires this in: this component owns src/model/attention/** and
# tests/model/attention/** only. Two things outside those paths still need a one-line edit that is
# NOT made here (out of this component's assigned paths per the task brief):
#   1. src/model/CMakeLists.txt (currently an INTERFACE placeholder, owned by the model-core
#      agent) needs `add_subdirectory(attention)` added for r4dx_model_attention to be part of the
#      build.
#   2. tests/model/CMakeLists.txt does not exist yet (only src/model/CMakeLists.txt did when this
#      component started). tests/model/attention/CMakeLists.txt below is self-contained; whoever
#      creates tests/model/CMakeLists.txt (the model-core agent, per the task brief) should add
#      `add_subdirectory(attention)` to it, the same way tests/CMakeLists.txt already does for
#      core/kernels/tokenizer/convert.

set(R4DX_MODEL_ATTN_HIPCC "${R4D_ROCM_ROOT}/bin/hipcc.exe")
set(R4DX_MODEL_ATTN_ROCM_BIN "${R4D_ROCM_ROOT}/bin")
set(R4DX_MODEL_ATTN_ROCM_LLVM_BIN "${R4D_ROCM_ROOT}/lib/llvm/bin")
set(R4DX_MODEL_ATTN_DEVICE_LIB_PATH "${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")

if(NOT EXISTS "${R4DX_MODEL_ATTN_HIPCC}")
  message(FATAL_ERROR "hipcc.exe not found at ${R4DX_MODEL_ATTN_HIPCC}; set -DR4D_ROCM_ROOT=<path to ROCm SDK>")
endif()

set(R4DX_MODEL_ATTN_FLAGS
  -O3 -std=c++17 --offload-arch=${R4D_GFX_ARCH}
  -Wno-unused-result -ffp-contract=off
  --rocm-path=${R4D_ROCM_ROOT}
  -DNDEBUG
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
  -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
  -I "${CMAKE_CURRENT_SOURCE_DIR}/../../core/include"
)
if(EXISTS "${R4DX_MODEL_ATTN_DEVICE_LIB_PATH}")
  list(APPEND R4DX_MODEL_ATTN_FLAGS --rocm-device-lib-path=${R4DX_MODEL_ATTN_DEVICE_LIB_PATH})
endif()

set(R4DX_MODEL_ATTN_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/objs")
file(MAKE_DIRECTORY "${R4DX_MODEL_ATTN_OBJ_DIR}")
set(R4DX_MODEL_ATTN_HIPCC_PATH "${R4DX_MODEL_ATTN_ROCM_BIN};${R4DX_MODEL_ATTN_ROCM_LLVM_BIN};$ENV{PATH}")

set(R4DX_MODEL_ATTN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/attn_kernels.hip")
set(R4DX_MODEL_ATTN_OBJ "${R4DX_MODEL_ATTN_OBJ_DIR}/attn_kernels.obj")

add_custom_command(
  OUTPUT "${R4DX_MODEL_ATTN_OBJ}"
  COMMAND "${CMAKE_COMMAND}" -E env "PATH=${R4DX_MODEL_ATTN_HIPCC_PATH}"
          "${R4DX_MODEL_ATTN_HIPCC}" ${R4DX_MODEL_ATTN_FLAGS} -c "${R4DX_MODEL_ATTN_SRC}" -o "${R4DX_MODEL_ATTN_OBJ}"
  DEPENDS "${R4DX_MODEL_ATTN_SRC}"
          "${CMAKE_CURRENT_SOURCE_DIR}/include/r4dx/model/attention/attn_kernels.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../core/include/r4dx/core/dtype.hpp"
  COMMENT "hipcc (${R4D_GFX_ARCH}) r4dx_model_attention attn_kernels.obj"
  VERBATIM
)
set_source_files_properties("${R4DX_MODEL_ATTN_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(r4dx_model_attention STATIC "${R4DX_MODEL_ATTN_OBJ}")
set_target_properties(r4dx_model_attention PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(r4dx_model_attention PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(r4dx_model_attention PUBLIC r4dx_core r4dx_kernels r4d_core)
