cmake_minimum_required(VERSION 3.28)

# NEW CMP0091 lets CMAKE_MSVC_RUNTIME_LIBRARY below actually select the CRT (must be set before
# project()). r4dx targets the dynamic CRT (/MD, msvcrt.dll family) to match the -D_DLL -D_MT
# --dependent-lib=msvcrt flags third_party/CMakeLists.txt passes to hipcc for r4d_core.
cmake_policy(SET CMP0091 NEW)

project(r4dx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(R4DX_BUILD_TESTS "Build tests" ON)
option(R4DX_BUILD_SERVER "Build the OpenAI-compatible chat server" ON)
option(R4DX_BUILD_CONVERT "Build the HF -> r4dx weight converter" ON)

# ---- w4a16 group size ----------------------------------------------------------------------------
#
# K per (scale, zero) pair in the w4a16 layout is 64 by default, a compile-time constant that has to be
# the same in two places:
#   * the kernel  -- R4D_GEMM_W4_GROUP in third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip, whose _group()
#                    export the converter asserts against at startup;
#   * the packer  -- kW4A16Group in src/convert/include/r4dx_convert/quant_int4.hpp, which decides the
#                    .w4a16.wsz stride.
# A container also records the value it was packed with in __metadata__.quant.w4a16.group, and
# Container::Load (src/model/container.cpp) refuses a container whose group differs from the kernel
# this binary was built with -- a silent mismatch is garbage GEMMs, not an error. The lm_head and any
# per-tensor group rule use group 32 or 64 through r4d_gemm_w4a16_nt_m64_g (docs/quant2.md section 5).
# Bits per weight is 4 + 32/group: 4.5 at 64, 5 at 32.

# CXX-only project: the r4d_core static library is produced by driving hipcc.exe directly through
# add_custom_command (see third_party/CMakeLists.txt) rather than enabling CMake's HIP language,
# which fights clang-cl on this toolchain. Everything above r4d_core is plain MSVC/clang-cl C++
# that links the resulting .obj/.lib the same way it would link any other static library.
add_subdirectory(third_party)   # -> r4d_core, r4dx_hip_runtime
add_subdirectory(src/core)
add_subdirectory(src/kernels)
add_subdirectory(src/tokenizer)
# src/convert and src/vision precede src/model: r4dx_convert is the header-only SafetensorsReader
# both the container loader and the vision weight loader read through, and r4dx_model links
# r4dx_vision_tower (Container owns the loaded vision.* weights -- docs/vision.md "Load policy").
add_subdirectory(src/convert)
add_subdirectory(src/vision)
add_subdirectory(src/model)
add_subdirectory(src/server)
add_subdirectory(src/cli)

if(R4DX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
