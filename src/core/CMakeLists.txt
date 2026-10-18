# src/core: device/stream/buffer plumbing (r4dx::core::DeviceBuffer/PinnedBuffer/Stream/Event/
# Arena/TensorView/dtype conversions) and r4dx::core::r4d, the typed C++ wrapper over r4d.h.
#
# Header-only: every r4d call here is a host-side API call (hipMalloc, hipMemcpy, the r4d.h entry
# points themselves) that clang-cl compiles directly, the same way tests/smoke_r4d.cpp already
# does -- there is no device (__global__) code in src/core, so it needs no hipcc step of its own.
add_library(r4dx_core INTERFACE)
target_include_directories(r4dx_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(r4dx_core INTERFACE r4dx_hip_runtime r4d_core)

# ---- r4dx_direct_storage: DirectStorage file -> VRAM loading (direct_storage.hpp, docs/direct-storage.md).
# Not header-only: it pulls in D3D12 / DXGI / dstorage.h, which no other core header may see.
# dstorage.dll is loaded at run time (GetProcAddress), so only the vendored headers are needed to
# build; the vendored x64 DLLs are the fallback location it searches after the process directory.
set(R4DX_DSTORAGE_VENDOR "${CMAKE_SOURCE_DIR}/third_party/directstorage")
file(TO_NATIVE_PATH "${R4DX_DSTORAGE_VENDOR}/bin/x64" R4DX_DSTORAGE_DLL_DIR)
string(REPLACE "\\" "\\\\" R4DX_DSTORAGE_DLL_DIR_ESCAPED "${R4DX_DSTORAGE_DLL_DIR}")
add_library(r4dx_direct_storage STATIC direct_storage.cpp)
target_include_directories(r4dx_direct_storage
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${R4DX_DSTORAGE_VENDOR}/include")
target_compile_definitions(r4dx_direct_storage PRIVATE R4DX_DSTORAGE_DIR="${R4DX_DSTORAGE_DLL_DIR_ESCAPED}")
target_link_libraries(r4dx_direct_storage PUBLIC r4dx_hip_runtime d3d12 dxgi)
target_link_libraries(r4dx_core INTERFACE r4dx_direct_storage)
