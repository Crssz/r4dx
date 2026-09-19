# src/kernels: r4dx-owned HIP kernels (rmsnorm, residual add, rope, silu_mul, fp8 activation
# quant, kv cache paging writes -- docs/architecture.md "Own kernels"), plus the host-only
# embedding-gather and CPU-sampler headers (no device code, so no hipcc step for those).
#
# The kernels themselves are real __global__ device code, so -- like third_party/libr4d's
# r4d_core -- they cannot go through CMake's HIP language (it fights clang-cl on this toolchain;
# see docs/build-windows.md) and instead go through hipcc.exe via add_custom_command, the same
# recipe third_party/CMakeLists.txt uses, reusing the R4D_ROCM_ROOT / R4D_GFX_ARCH cache
# variables it defines (both CACHE, so already set by the time this subdirectory runs -- the root
# CMakeLists.txt add_subdirectory(third_party)'s before add_subdirectory(src/kernels)).

set(R4D_ROCM_ROOT "C:/opt/rocm" CACHE PATH "ROCm SDK root (hipcc.exe under bin/, headers under include/hip)")
set(R4D_GFX_ARCH "gfx1201" CACHE STRING "HIP offload arch")

set(R4DX_HIPCC "${R4D_ROCM_ROOT}/bin/hipcc.exe")
set(R4DX_ROCM_BIN "${R4D_ROCM_ROOT}/bin")
set(R4DX_ROCM_LLVM_BIN "${R4D_ROCM_ROOT}/lib/llvm/bin")
set(R4DX_DEVICE_LIB_PATH "${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")

if(NOT EXISTS "${R4DX_HIPCC}")
  message(FATAL_ERROR "hipcc.exe not found at ${R4DX_HIPCC}; set -DR4D_ROCM_ROOT=<path to ROCm SDK>")
endif()

set(R4DX_KERNELS_FLAGS
  -O3 -std=c++17 --offload-arch=${R4D_GFX_ARCH}
  -Wno-unused-result -ffp-contract=off
  --rocm-path=${R4D_ROCM_ROOT}
  -DNDEBUG
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
  -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
  -I "${CMAKE_CURRENT_SOURCE_DIR}/../core/include"
)
if(EXISTS "${R4DX_DEVICE_LIB_PATH}")
  list(APPEND R4DX_KERNELS_FLAGS --rocm-device-lib-path=${R4DX_DEVICE_LIB_PATH})
endif()

set(R4DX_KERNELS_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/objs")
file(MAKE_DIRECTORY "${R4DX_KERNELS_OBJ_DIR}")
set(R4DX_KERNELS_HIPCC_PATH "${R4DX_ROCM_BIN};${R4DX_ROCM_LLVM_BIN};$ENV{PATH}")

set(R4DX_KERNELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/r4dx_kernels.hip")
set(R4DX_KERNELS_OBJ "${R4DX_KERNELS_OBJ_DIR}/r4dx_kernels.obj")

add_custom_command(
  OUTPUT "${R4DX_KERNELS_OBJ}"
  COMMAND "${CMAKE_COMMAND}" -E env "PATH=${R4DX_KERNELS_HIPCC_PATH}"
          "${R4DX_HIPCC}" ${R4DX_KERNELS_FLAGS} -c "${R4DX_KERNELS_SRC}" -o "${R4DX_KERNELS_OBJ}"
  DEPENDS "${R4DX_KERNELS_SRC}"
          "${CMAKE_CURRENT_SOURCE_DIR}/include/r4dx/kernels/kernels.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/../core/include/r4dx/core/dtype.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/../core/include/r4dx/core/error.hpp"
  COMMENT "hipcc (${R4D_GFX_ARCH}) r4dx_kernels.obj"
  VERBATIM
)
set_source_files_properties("${R4DX_KERNELS_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(r4dx_kernels STATIC "${R4DX_KERNELS_OBJ}")
set_target_properties(r4dx_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(r4dx_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(r4dx_kernels PUBLIC r4dx_hip_runtime)
