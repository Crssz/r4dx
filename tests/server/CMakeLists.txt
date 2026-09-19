# tests/server/: unit tests for src/server's CPU-only half (request parsing/validation, response
# JSON building, SSE chunk formatting, the bounded request queue, ResponseSink) -- no HIP device
# needed, plain add_test() COMMANDs like tests/cli's test_cli_args, not the ENVIRONMENT/
# SKIP_RETURN_CODE-gated pattern tests/model uses. src/server/engine.cpp/http_server.cpp (the
# HIP+httplib-dependent half) are exercised instead by tools/server/smoke.ps1 against a real
# container on HIP device 1 -- see docs/server.md.
set(R4DX_SERVER_TESTS
  test_server_args
  test_openai_types
  test_sse
  test_response_sink
  test_request_queue
  test_prefix_state
)

foreach(t ${R4DX_SERVER_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE r4dx_server_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
