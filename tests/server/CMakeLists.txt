# tests/server/: unit tests for src/server's CPU-only half (request parsing/validation, response
# JSON building, SSE chunk formatting, the bounded request queue, ResponseSink) -- no HIP device
# needed, plain add_test() COMMANDs like tests/cli's test_cli_args, not the ENVIRONMENT/
# SKIP_RETURN_CODE-gated pattern tests/model uses. src/server/engine.cpp/http_server.cpp (the
# HIP+httplib-dependent half) are exercised against a real container on HIP device 1 by
# tools/server/smoke.ps1 -- see docs/server.md -- and, against CPU fakes of the model, by
# test_engine_recovery and test_http_server below.
set(R4DX_SERVER_TESTS
  test_server_args
  test_openai_types
  test_sse
  test_response_sink
  test_request_queue
  test_prefix_state
  test_tool_call_parser
  test_tool_stream_gate
  test_reasoning_splitter
)

foreach(t ${R4DX_SERVER_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE r4dx_server_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# test_engine_recovery: Engine::RunRequest's error path (docs/tp.md 2.4, 8.4), through engine.cpp
# itself, against CPU fakes of TpModel's state machine and of the TP=1 model
# (EngineOptions::model_loader). It makes no GPU call, but engine.cpp is HIP-dependent, so it links
# exactly what src/server/CMakeLists.txt's r4dx-server links -- including that target's explicit
# third_party include ahead of the HIP include dir (the nlohmann/json ABI-ordering fix documented
# there) -- and, like test_pick_tuning, runs with HIP_VISIBLE_DEVICES=1: the HIP runtime it loads
# must never see device 0. Needs the real tokenizer directory (tests/tokenizer's
# R4DX_TOKENIZER_MODEL_DIR) and exits 77 (SKIPPED) without it.
if(R4DX_BUILD_SERVER)
  add_executable(test_engine_recovery test_engine_recovery.cpp ${CMAKE_SOURCE_DIR}/src/server/engine.cpp)
  target_include_directories(test_engine_recovery PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/src/server
  )
  target_compile_definitions(test_engine_recovery PRIVATE
    R4DX_TOKENIZER_MODEL_DIR="${R4DX_TOKENIZER_MODEL_DIR}"
  )
  target_link_libraries(test_engine_recovery PRIVATE
    r4dx_server_core
    r4dx_model
    r4dx_tokenizer
    r4dx_core
    r4dx_kernels
    r4d_core
    r4dx_hip_runtime
  )
  add_test(NAME test_engine_recovery COMMAND test_engine_recovery)
  set_tests_properties(test_engine_recovery PROPERTIES
    ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
    SKIP_RETURN_CODE 77
  )

  # test_http_server: http_server.cpp's routes end to end -- a real Engine and HttpServer on an
  # OS-chosen loopback port (HttpServer::BindToAnyPort), a scripted CPU fake of the TP=1 model,
  # httplib's own client (the response Content-Type charset, and streamed vs non-streamed content as
  # a client reads them). Same GPU-free setup, link line, include order and tokenizer-directory skip
  # as test_engine_recovery above (http_server.h's content-type constants are checked before that
  # skip), plus third_party/httplib for the server and the client.
  add_executable(test_http_server test_http_server.cpp
    ${CMAKE_SOURCE_DIR}/src/server/engine.cpp
    ${CMAKE_SOURCE_DIR}/src/server/http_server.cpp
  )
  target_include_directories(test_http_server PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/httplib
    ${CMAKE_SOURCE_DIR}/src/server
  )
  target_compile_definitions(test_http_server PRIVATE
    R4DX_TOKENIZER_MODEL_DIR="${R4DX_TOKENIZER_MODEL_DIR}"
  )
  target_link_libraries(test_http_server PRIVATE
    r4dx_server_core
    r4dx_model
    r4dx_tokenizer
    r4dx_core
    r4dx_kernels
    r4d_core
    r4dx_hip_runtime
  )
  add_test(NAME test_http_server COMMAND test_http_server)
  set_tests_properties(test_http_server PROPERTIES
    ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
    SKIP_RETURN_CODE 77
  )
endif()
