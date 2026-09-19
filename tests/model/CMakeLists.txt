# tests/model/: golden tests for the model-core component (container loader, linear apply, GDN
# layer, MLP, final norm + lm_head -- src/model/** except src/model/attention/**). Both tests need
# real data not vendored into the repo (a bf16-layout r4dx test container, and
# tools/reference/golden_out/*.safetensors); each checks for its own files at runtime and exits 77
# (CTest SKIPPED, not FAILED) if either is missing, the same convention
# tests/tokenizer/CMakeLists.txt's tokenizer_golden test uses.
set(R4DX_MODEL_TESTS
  test_gdn_layer
  test_final_lm_head
  test_forward_smoke
  test_mtp
)

foreach(t ${R4DX_MODEL_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
  add_test(NAME ${t} COMMAND ${t})
  set_tests_properties(${t} PROPERTIES
    ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
    SKIP_RETURN_CODE 77
  )
endforeach()

# test_mtp_round: pure CPU unit test for src/model/mtp_round.hpp (ProcessMtpRound) -- no HIP
# device, model container, or GDN/KV state needed (header-only std::vector<int32_t> arithmetic),
# same "plain add_test(), no ENVIRONMENT/SKIP_RETURN_CODE" pattern tests/server's HIP-free tests
# use. Kept out of R4DX_MODEL_TESTS above so it never gets SKIPPED on a machine without a real
# container -- it should always run.
add_executable(test_mtp_round test_mtp_round.cpp)
target_include_directories(test_mtp_round PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_mtp_round COMMAND test_mtp_round)

# tests/model/attention/: golden test for r4dx::model::attention::AttentionLayer, created
# standalone by that component before this file existed -- see its own CMakeLists.txt header note.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/attention/CMakeLists.txt")
  add_subdirectory(attention)
endif()
