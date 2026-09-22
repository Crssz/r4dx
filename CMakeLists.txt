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

# ---- w4a16 group size (docs/build-windows.md "w4a16 group size") -------------------------------
#
# K per (scale, zero) pair in the w4a16 layout. It is ONE knob because it has to reach two places
# that must never disagree:
#   * the kernel  -- third_party/CMakeLists.txt passes it as -DR4D_GEMM_W4_GROUP to
#                    r4d_gemm_w4a16_nt_m64, whose _group() export the converter asserts against;
#   * the packer  -- src/convert/CMakeLists.txt passes it as -DR4DX_W4A16_GROUP to r4dx_convert's
#                    kW4A16Group (quant_int4.hpp), which decides the .w4a16.wsz stride.
# A container also records the value it was packed with in __metadata__.quant.w4a16.group, and
# Container::Load (src/model/container.cpp) refuses a container whose group differs from the
# kernel this binary was built with -- a silent mismatch is garbage GEMMs, not an error.
#
# The kernel packs R4D_GEMM_W4_KPB=64 contiguous K per weight block, so the group must be a
# multiple of 64: 64 and 128 are the only values libr4d accepts unmodified. Bits per weight is
# 4 + 32/group (4.25 at 128, 4.5 at 64) -- docs/validation.md "Milestone 11 / group size" has what
# the extra quarter-bit buys.
#
# The DEFAULT is 64 as of Milestone 11 (docs/validation.md "Milestone 11 / recipe"): group 64 plus
# a bf16 attn.k/attn.v cuts mean KL 27% for +0.9 GiB, and those two are the only levers in the
# per-class sensitivity table worth more than 0.015 nats/GiB. The consequence is that every
# container packed before v6 -- qwen38-27b-v5.r4dx, v4, v3, the old dflash2 drafters -- is group
# 128 and this build REFUSES to load it, by name and with both numbers, rather than reading the
# scales at the wrong stride. Configure with -DR4DX_W4A16_GROUP=128 to get the old behaviour back.
# Build a non-default group in its OWN build dir (preset win-hip-g128), never on top of
# build/win-hip: the container and the binaries must ship as a matched pair.
set(R4DX_W4A16_GROUP "64" CACHE STRING "w4a16 group size: K per (scale, zero) pair (64 or 128)")
set_property(CACHE R4DX_W4A16_GROUP PROPERTY STRINGS 64 128)
if(NOT R4DX_W4A16_GROUP MATCHES "^[0-9]+$")
  message(FATAL_ERROR "R4DX_W4A16_GROUP must be a positive integer (got '${R4DX_W4A16_GROUP}')")
endif()
math(EXPR R4DX_W4A16_GROUP_REM "${R4DX_W4A16_GROUP} % 64")
if(R4DX_W4A16_GROUP LESS 64 OR NOT R4DX_W4A16_GROUP_REM EQUAL 0)
  message(FATAL_ERROR
          "R4DX_W4A16_GROUP must be a positive multiple of R4D_GEMM_W4_KPB (64) -- got "
          "'${R4DX_W4A16_GROUP}'; r4d_gemm_w4a16_nt_m64 derives bpg = group / 64 and would read "
          "the packed weights at the wrong stride")
endif()
message(STATUS "r4dx: w4a16 group = ${R4DX_W4A16_GROUP} "
               "(bits/weight = 4 + 32/${R4DX_W4A16_GROUP})")

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
