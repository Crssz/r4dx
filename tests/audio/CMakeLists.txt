# tests/audio/: CPU-only tests for src/audio (docs/gemma4-audio.md). No HIP device, no container.
#   test_audio         always-on unit test: framing, placeholder expansion, WAV decode, the CPU embedder.
#   test_audio_golden  vs transformers' Gemma4Unified processor/embedder, from
#                      tools/reference/gemma/audio_golden_gemma.py's output (gitignored); exit 77 = SKIPPED
#                      when that output is absent (the tests/vision convention).
add_executable(test_audio test_audio.cpp)
target_link_libraries(test_audio PRIVATE r4dx_audio)
add_test(NAME test_audio COMMAND test_audio)

add_executable(test_audio_golden test_audio_golden.cpp)
target_link_libraries(test_audio_golden PRIVATE r4dx_audio)
target_compile_definitions(test_audio_golden PRIVATE
  R4DX_AUDIO_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tools/reference/golden_out/gemma")
add_test(NAME test_audio_golden COMMAND test_audio_golden)
set_tests_properties(test_audio_golden PROPERTIES SKIP_RETURN_CODE 77)

# Engine audio part JSON -> real Gemma chat template -> one <|audio|> per clip (real-template part needs the model dir; skipped otherwise).
add_executable(test_audio_template test_audio_template.cpp)
target_link_libraries(test_audio_template PRIVATE r4dx_audio r4dx_tokenizer)
add_test(NAME test_audio_template COMMAND test_audio_template)
