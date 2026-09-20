# OpenAI-compatible chat API (streaming, minja templates) -- docs/architecture.md "src/server/".
#
# r4dx_server_core: the CPU-only half (request parsing/validation, OpenAI response JSON building,
# SSE chunk formatting, the bounded request queue, ResponseSink) -- no HIP, no r4dx::model
# dependency, so tests/server/** links this alone without a GPU. Mirrors src/cli's own
# header-only-argument-parsing split (cli_args.h / r4dx_cli) for the same reason; server_args.h
# and request_queue.h are header-only so they need no .cpp here.
add_library(r4dx_server_core STATIC
  openai_types.cpp
  sse.cpp
  response_sink.cpp
  tool_call_parser.cpp
)
target_include_directories(r4dx_server_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# nlohmann/json.hpp is vendored under third_party/nlohmann/, so third_party/ itself is the include
# root that makes `#include "nlohmann/json.hpp"` resolve (same reasoning as
# src/tokenizer/CMakeLists.txt). PUBLIC: openai_types.h (a public header of this target) includes
# it, so consumers linking r4dx_server_core need this path too.
target_include_directories(r4dx_server_core PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(r4dx_server_core PUBLIC cxx_std_17)

if(R4DX_BUILD_SERVER)
  add_executable(r4dx-server main.cpp http_server.cpp engine.cpp)
  # Same nlohmann/json ABI-ordering fix src/cli/CMakeLists.txt documents in detail: this target is
  # the server's own "first target to link both r4dx_tokenizer (built against third_party/
  # nlohmann's vendored v3.11.3) and anything HIP-including" (r4dx_hip_runtime's INTERFACE include
  # dir, C:/opt/rocm/include, ships a DIFFERENT nlohmann/json.hpp -- v3.12.0, a distinct ABI inline
  # namespace). Both are found via the same unqualified `#include "nlohmann/json.hpp"`, so
  # whichever -I comes first on a given TU's command line wins; listed here explicitly, ahead of
  # target_link_libraries below, so engine.cpp/http_server.cpp/main.cpp (which all pull in both
  # r4dx::ChatTemplate -- a non-inline, cross-TU symbol built into r4dx_tokenizer.lib against
  # v3.11.3 -- and r4dx::model::Model, HIP-including) link against the matching ABI instead of an
  # ABI-mismatched undefined symbol (json_abi_v3_12_0 vs json_abi_v3_11_3).
  target_include_directories(r4dx-server PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/httplib
  )
  target_link_libraries(r4dx-server PRIVATE
    r4dx_server_core
    r4dx_model
    r4dx_tokenizer
    r4dx_core
    r4dx_kernels
    r4d_core
    r4dx_hip_runtime
  )
endif()
