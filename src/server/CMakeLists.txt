# OpenAI-compatible chat API (streaming, minja templates). Placeholder until filled in.
add_library(r4dx_server INTERFACE)
target_include_directories(r4dx_server INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(R4DX_BUILD_SERVER)
  # The `r4dx-server` executable target is added here once src/server/main.cpp exists.
endif()
