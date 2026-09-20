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

# test_mtp's CheckChatMultiTurnMidRoundStop (docs/status.md Known-gaps: "no automated
# multi-turn-with-a-forced-mid-round-stop regression test") drives the exact production
# r4dx::server::PrefixState (src/server/prefix_state.h) -- header-only, no HIP/r4dx::model
# dependency of its own, same reasoning tests/server/test_prefix_state.cpp relies on -- so it is
# safe to include directly from this HIP-linked test binary without linking r4dx_server itself.
target_include_directories(test_mtp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/server)

# tool_hseed_drift: one-off diagnostic for docs/mtp.md's "Acceptance gap investigation" (h_seed
# drift vs a bf16 reference, per layout) -- built here (same libs, same SKIP-if-missing-container
# convention as the tests above) but deliberately NOT registered via add_test: it prints numbers for
# a human/doc to read, it does not assert a pass/fail contract, so it must never affect the 35/35
# ctest count. Run manually: build\win-hip\tests\model\tool_hseed_drift.exe (HIP_VISIBLE_DEVICES=1).
add_executable(tool_hseed_drift tool_hseed_drift.cpp)
target_link_libraries(tool_hseed_drift PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

# tool_vocab_calib: one-off diagnostic for docs/r9700.md R9's "reduced-vocab draft head" -- chooses
# and measures coverage for the MTP draft head's vocabulary subset against a real calibration
# corpus (D:/models/wikitext-2-raw) run through the real 64-layer container. Same "built but never
# add_test()'d" convention as tool_hseed_drift above (prints numbers + writes a JSON file for a
# human/converter run to consume, no pass/fail contract). Needs r4dx_tokenizer for the real BPE
# tokenizer, unlike tool_hseed_drift.
add_executable(tool_vocab_calib tool_vocab_calib.cpp)
target_link_libraries(tool_vocab_calib PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core r4dx_tokenizer)

# test_mtp_round: pure CPU unit test for src/model/mtp_round.hpp (ProcessMtpRound) -- no HIP
# device, model container, or GDN/KV state needed (header-only std::vector<int32_t> arithmetic),
# same "plain add_test(), no ENVIRONMENT/SKIP_RETURN_CODE" pattern tests/server's HIP-free tests
# use. Kept out of R4DX_MODEL_TESTS above so it never gets SKIPPED on a machine without a real
# container -- it should always run.
add_executable(test_mtp_round test_mtp_round.cpp)
target_include_directories(test_mtp_round PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_mtp_round COMMAND test_mtp_round)

# test_dflash_draft_weights: task A1 item 3's CPU-only DflashDraftWeights loader stub
# (src/model/dflash_draft_weights.h -- metadata + tensor directory, no device upload). Links only
# r4dx_convert (header-only: SafetensorsReader, GgufReader, dflash2_container writer helpers used
# to build the test's own tiny fixture container) + r4dx_core (bf16 conversions) -- deliberately
# NOT r4dx_model/r4d_core/r4dx_hip_runtime, so this binary has no HIP dependency at all, same
# "always runs, no ENVIRONMENT/SKIP_RETURN_CODE" pattern as test_mtp_round above.
add_executable(test_dflash_draft_weights test_dflash_draft_weights.cpp)
target_include_directories(test_dflash_draft_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
target_link_libraries(test_dflash_draft_weights PRIVATE r4dx_convert r4dx_core)
target_compile_definitions(test_dflash_draft_weights PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../convert/fixtures")
add_test(NAME test_dflash_draft_weights COMMAND test_dflash_draft_weights)

# tests/model/attention/: golden test for r4dx::model::attention::AttentionLayer, created
# standalone by that component before this file existed -- see its own CMakeLists.txt header note.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/attention/CMakeLists.txt")
  add_subdirectory(attention)
endif()
