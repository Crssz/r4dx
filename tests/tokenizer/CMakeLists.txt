# tokenizer_golden: replays tests/tokenizer/golden.json (produced offline by
# tools/tok_ref/gen_golden.py against the real HF tokenizer, CPU-only, no GPU) through r4dx's own
# Tokenizer + ChatTemplate and diffs every case. Needs the real model directory on disk
# (C:\AI\models\Qwen3.8-27B: tokenizer.json, tokenizer_config.json, chat_template.jinja,
# generation_config.json) -- it is not vendored into the repo (12+ MB tokenizer.json), so this
# test is skipped rather than failed if that directory is absent.
set(R4DX_TOKENIZER_MODEL_DIR "C:/AI/models/Qwen3.8-27B" CACHE PATH
    "Directory containing tokenizer.json/tokenizer_config.json/chat_template.jinja/generation_config.json")

add_executable(tokenizer_golden golden_test.cpp)
target_link_libraries(tokenizer_golden PRIVATE r4dx_tokenizer)
target_compile_definitions(tokenizer_golden PRIVATE
  TOKENIZER_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  R4DX_TOKENIZER_MODEL_DIR="${R4DX_TOKENIZER_MODEL_DIR}"
)

# Always register the test (so a machine without the model directory still shows an explicit
# SKIPPED result in `ctest` output, instead of silently reporting 100% pass with zero tokenizer
# coverage). golden_test.cpp itself checks for R4DX_TOKENIZER_MODEL_DIR/tokenizer.json at runtime
# and exits 77 if it is absent; CTest's SKIP_RETURN_CODE maps that to a SKIPPED (not PASSED/FAILED)
# result.
add_test(NAME tokenizer_golden COMMAND tokenizer_golden)
set_tests_properties(tokenizer_golden PROPERTIES SKIP_RETURN_CODE 77)

if(NOT EXISTS "${R4DX_TOKENIZER_MODEL_DIR}/tokenizer.json")
  message(WARNING "tests/tokenizer: ${R4DX_TOKENIZER_MODEL_DIR}/tokenizer.json not found; "
                   "'tokenizer_golden' is registered but will report SKIPPED at run time. Set "
                   "-DR4DX_TOKENIZER_MODEL_DIR=... to point at the Qwen3.8-27B checkout.")
endif()
