# src/core: device/stream/buffer plumbing (r4dx::core::DeviceBuffer/PinnedBuffer/Stream/Event/
# Arena/TensorView/dtype conversions) and r4dx::core::r4d, the typed C++ wrapper over r4d.h.
#
# Header-only: every r4d call here is a host-side API call (hipMalloc, hipMemcpy, the r4d.h entry
# points themselves) that clang-cl compiles directly, the same way tests/smoke_r4d.cpp already
# does -- there is no device (__global__) code in src/core, so it needs no hipcc step of its own.
add_library(r4dx_core INTERFACE)
target_include_directories(r4dx_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(r4dx_core INTERFACE r4dx_hip_runtime r4d_core)
