# src/model/qwen4exp: kernels and host helpers for the qwen4_exp arch (docs/qwen4exp-plan.md).
#
# Drop-in layout, so parallel authors never edit this file:
#   kernels/*.hip       one hipcc object each (same toolchain recipe as ../attention/CMakeLists.txt)
#   host/*.cpp          plain C++ (clang-cl), may call the kernels' extern "C" entry points
#   include/r4dx/model/qwen4exp/*.h   public headers (one per kernel group / helper)
# All of it lands in the static library r4dx_model_qwen4exp_kernels. It must NOT depend on r4dx_model
# (r4dx_model links it, see ../CMakeLists.txt); the Container / Model / TextModel files of this arch are
# listed in r4dx_model itself, like gemma_*.cpp.
# CONFIGURE_DEPENDS: a new file is picked up by the next build without a manual reconfigure.

set(Q4_HIPCC "${R4D_ROCM_ROOT}/bin/hipcc.exe")
set(Q4_ROCM_BIN "${R4D_ROCM_ROOT}/bin")
set(Q4_ROCM_LLVM_BIN "${R4D_ROCM_ROOT}/lib/llvm/bin")
set(Q4_DEVICE_LIB_PATH "${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
if(NOT EXISTS "${Q4_HIPCC}")
  message(FATAL_ERROR "hipcc.exe not found at ${Q4_HIPCC}; set -DR4D_ROCM_ROOT=<path to ROCm SDK>")
endif()

set(Q4_HIP_FLAGS
  -O3 -std=c++17 --offload-arch=${R4D_GFX_ARCH}
  -Wno-unused-result -ffp-contract=off
  --rocm-path=${R4D_ROCM_ROOT}
  -DNDEBUG
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
  -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
  -I "${CMAKE_CURRENT_SOURCE_DIR}/../../core/include"
  -I "${CMAKE_CURRENT_SOURCE_DIR}/../../kernels/include"
  -I "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/libr4d"
)
if(EXISTS "${Q4_DEVICE_LIB_PATH}")
  list(APPEND Q4_HIP_FLAGS --rocm-device-lib-path=${Q4_DEVICE_LIB_PATH})
endif()

set(Q4_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/objs")
file(MAKE_DIRECTORY "${Q4_OBJ_DIR}")
set(Q4_HIPCC_PATH "${Q4_ROCM_BIN};${Q4_ROCM_LLVM_BIN};$ENV{PATH}")

file(GLOB Q4_HIP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.hip")
file(GLOB Q4_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/r4dx/model/qwen4exp/*.h"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/r4dx/model/qwen4exp/*.hpp")
set(Q4_OBJS "")
foreach(src ${Q4_HIP_SOURCES})
  get_filename_component(name "${src}" NAME_WE)
  set(obj "${Q4_OBJ_DIR}/${name}.obj")
  add_custom_command(
    OUTPUT "${obj}"
    COMMAND "${CMAKE_COMMAND}" -E env "PATH=${Q4_HIPCC_PATH}"
            "${Q4_HIPCC}" ${Q4_HIP_FLAGS} -c "${src}" -o "${obj}"
    DEPENDS "${src}" ${Q4_HEADERS}
            "${CMAKE_CURRENT_SOURCE_DIR}/../../core/include/r4dx/core/dtype.hpp"
    COMMENT "hipcc (${R4D_GFX_ARCH}) r4dx_model_qwen4exp ${name}.obj"
    VERBATIM)
  set_source_files_properties("${obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND Q4_OBJS "${obj}")
endforeach()

file(GLOB Q4_HOST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp")
if(NOT Q4_OBJS AND NOT Q4_HOST_SOURCES)
  # A STATIC library needs one source: until the first kernel / host file lands, an empty TU.
  set(Q4_EMPTY_TU "${CMAKE_CURRENT_BINARY_DIR}/q4exp_empty.cpp")
  file(WRITE "${Q4_EMPTY_TU}" "// intentionally empty: src/model/qwen4exp has no sources yet\n")
  set(Q4_HOST_SOURCES "${Q4_EMPTY_TU}")
endif()

add_library(r4dx_model_qwen4exp_kernels STATIC ${Q4_OBJS} ${Q4_HOST_SOURCES})
set_target_properties(r4dx_model_qwen4exp_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(r4dx_model_qwen4exp_kernels PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../convert/include"
)
# Links r4dx_model_arch (Qwen4ExpConfig) rather than r4dx_model.
target_link_libraries(r4dx_model_qwen4exp_kernels PUBLIC
  r4dx_core r4dx_kernels r4d_core r4dx_model_linear r4dx_model_arch r4dx_model_attention)
