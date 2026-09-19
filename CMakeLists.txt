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

# CXX-only project: the r4d_core static library is produced by driving hipcc.exe directly through
# add_custom_command (see third_party/CMakeLists.txt) rather than enabling CMake's HIP language,
# which fights clang-cl on this toolchain. Everything above r4d_core is plain MSVC/clang-cl C++
# that links the resulting .obj/.lib the same way it would link any other static library.
add_subdirectory(third_party)   # -> r4d_core, r4dx_hip_runtime
add_subdirectory(src/core)
add_subdirectory(src/kernels)
add_subdirectory(src/tokenizer)
add_subdirectory(src/model)
add_subdirectory(src/convert)
add_subdirectory(src/server)
add_subdirectory(src/cli)

if(R4DX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
