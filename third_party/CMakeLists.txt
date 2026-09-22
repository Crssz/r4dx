# third_party/CMakeLists.txt
#
# r4d_core: a static library built from 15 of libr4d's HIP translation units (attention, GDN,
# skinny/quantized GEMMs, the kernel registry -- NOT r4d_module.hip, the pybind11 glue, and NOT any
# r4d_ar_* unit, the multi-GPU all-reduce kernels this single-GPU engine never calls).
#
# CMake's HIP language support fights clang-cl on this toolchain, so each unit is compiled by
# driving hipcc.exe directly through add_custom_command (the same recipe libr4d's own
# build_windows.ps1 uses), and the resulting .obj files are handed to add_library() as pre-built
# "external objects" -- a standard CMake pattern for objects that did not come from CMake's own
# compile rules.
#
# r4dx_hip_runtime: an INTERFACE target carrying the HIP include dir and the amdhip64 import lib,
# for any consumer (kernels, tests) that calls into the HIP runtime directly (hipMalloc,
# hipStreamCreate, ...) without going through r4d_core.

set(R4D_ROCM_ROOT "C:/opt/rocm" CACHE PATH "ROCm SDK root (hipcc.exe under bin/, headers under include/hip)")
set(R4D_GFX_ARCH "gfx1201" CACHE STRING "HIP offload arch")

set(R4D_HIPCC "${R4D_ROCM_ROOT}/bin/hipcc.exe")
set(R4D_ROCM_BIN "${R4D_ROCM_ROOT}/bin")
set(R4D_ROCM_LLVM_BIN "${R4D_ROCM_ROOT}/lib/llvm/bin")
set(R4D_DEVICE_LIB_PATH "${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
set(R4D_HIP_LIB "${R4D_ROCM_ROOT}/lib/amdhip64.lib")

if(NOT EXISTS "${R4D_HIPCC}")
  message(FATAL_ERROR "hipcc.exe not found at ${R4D_HIPCC}; set -DR4D_ROCM_ROOT=<path to ROCm SDK>")
endif()

set(LIBR4D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libr4d")
if(NOT EXISTS "${LIBR4D_DIR}/r4d.h")
  message(FATAL_ERROR "third_party/libr4d submodule is empty; run "
                       "'git submodule update --init third_party/libr4d' (branch windows-llp64)")
endif()

set(R4D_BASE_FLAGS
  -O3 -std=c++17 --offload-arch=${R4D_GFX_ARCH}
  -Wno-unused-result -ffp-contract=off
  --rocm-path=${R4D_ROCM_ROOT}
  -DNDEBUG
  # Match the dynamic (DLL) MSVC CRT the rest of r4dx is built with (CMAKE_MSVC_RUNTIME_LIBRARY
  # = MultiThreadedDLL, set in the root CMakeLists.txt) so the final link does not hit an LNK2038
  # CRT mismatch between these hipcc-compiled objects and the clang-cl-compiled r4dx sources.
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
)
if(EXISTS "${R4D_DEVICE_LIB_PATH}")
  list(APPEND R4D_BASE_FLAGS --rocm-device-lib-path=${R4D_DEVICE_LIB_PATH})
endif()

# unit -> extra per-unit flags, mirroring libr4d/build_windows.ps1's $UNITS table, restricted to
# the 15 units r4dx links (r4d_module.hip and every r4d_ar_* unit are excluded on purpose). One
# deliberate deviation from that table: r4d_gemm_w4a16_nt_m64's group is an r4dx build option (see
# below) rather than the kernel's own 128 default.
set(R4D_UNITS
  r4d_attn_paged_h256_gqa6
  r4d_attn_vit_h72_bf16
  r4d_gdn_chunk_scan_k128_v128_c64_bf16
  r4d_gdn_conv_w4_h128_bf16
  r4d_gdn_kkt_solve_k128_c64_bf16
  r4d_gdn_recurrent_update_k128_v128_bf16_fp32state
  r4d_gdn_gated_rmsnorm_h128_bf16
  r4d_gemm_bf16_nt_m16
  r4d_gemm_bf16_nt_m64
  r4d_gemm_w4a16_nt_m64
  r4d_gemm_w4a8_nt_m64
  r4d_gemm_mxfp4a8_nt_m64
  r4d_quant_act_i8
  r4d_dflash_conv_t2_g16_bf16
  r4d_registry
)
set(R4D_EXTRA_r4d_gdn_chunk_scan_k128_v128_c64_bf16 -mcumode)
set(R4D_EXTRA_r4d_gemm_w4a8_nt_m64 -DR4D_GEMM_W4A8_GROUP=128)
# w4a16's group is the R4DX_W4A16_GROUP build option (root CMakeLists.txt, which also validates
# it). The SAME number reaches r4dx_convert's kW4A16Group through src/convert/CMakeLists.txt; the
# converter asserts this kernel's _group() export against it at startup, and Container::Load
# asserts it against every container's own __metadata__.quant.w4a16.group.
set(R4D_EXTRA_r4d_gemm_w4a16_nt_m64 -DR4D_GEMM_W4_GROUP=${R4DX_W4A16_GROUP})

set(R4D_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/r4d_objs")
file(MAKE_DIRECTORY "${R4D_OBJ_DIR}")

# hipcc needs its own clang/lld-link/device-lib tools on PATH ahead of anything else; run it
# through `cmake -E env` rather than mutating the generator's own PATH.
set(R4D_HIPCC_PATH "${R4D_ROCM_BIN};${R4D_ROCM_LLVM_BIN};$ENV{PATH}")

set(R4D_OBJS "")
foreach(unit ${R4D_UNITS})
  set(src "${LIBR4D_DIR}/${unit}.hip")
  set(obj "${R4D_OBJ_DIR}/${unit}.obj")
  set(extra "${R4D_EXTRA_${unit}}")
  add_custom_command(
    OUTPUT "${obj}"
    COMMAND "${CMAKE_COMMAND}" -E env "PATH=${R4D_HIPCC_PATH}"
            "${R4D_HIPCC}" ${R4D_BASE_FLAGS} ${extra} -c "${src}" -o "${obj}"
    DEPENDS "${src}" "${LIBR4D_DIR}/r4d.h"
    COMMENT "hipcc (${R4D_GFX_ARCH}) ${unit}.obj"
    VERBATIM
  )
  list(APPEND R4D_OBJS "${obj}")
endforeach()

set_source_files_properties(${R4D_OBJS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(r4d_core STATIC ${R4D_OBJS})
set_target_properties(r4d_core PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(r4d_core PUBLIC "${LIBR4D_DIR}")
target_link_libraries(r4d_core PUBLIC "${R4D_HIP_LIB}")

add_library(r4dx_hip_runtime INTERFACE)
target_include_directories(r4dx_hip_runtime INTERFACE "${R4D_ROCM_ROOT}/include")
target_link_libraries(r4dx_hip_runtime INTERFACE "${R4D_HIP_LIB}")
target_compile_definitions(r4dx_hip_runtime INTERFACE __HIP_PLATFORM_AMD__)
