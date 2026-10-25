# tokenizer_golden: replays tests/tokenizer/golden.json (produced offline by
# tools/tok_ref/gen_golden.py against the real HF tokenizer, CPU-only, no GPU) through r4dx's own
# Tokenizer + ChatTemplate and diffs every case. Needs the real model directory on disk
# (<R4DX_MODELS_ROOT>\Huihui-Qwen3.8-27B-abliterated: tokenizer.json, tokenizer_config.json,
# chat_template.jinja, generation_config.json) -- it is not vendored into the repo (12+ MB
# tokenizer.json), so this test is skipped rather than failed if that directory is absent. The
# fine-tune's tokenizer files are byte-identical (SHA-256) to the base Qwen3.8-27B's that
# golden.json was cut from (docs/huihui.md), so any Qwen3.8-27B-family checkpoint dir works.
set(R4DX_TOKENIZER_MODEL_DIR "${R4DX_MODELS_ROOT}/Huihui-Qwen3.8-27B-abliterated" CACHE PATH
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

# Microbench + large-corpus HF agreement check; not a CTest test (needs a corpus written by
# tools/tok_ref/bench_hf.py or check_hf_equiv.py --emit-jsonl). First measured numbers: header of bench_tokenizer.cpp.
add_executable(bench_tokenizer bench_tokenizer.cpp)
target_link_libraries(bench_tokenizer PRIVATE r4dx_tokenizer)

# Model-free checks of the local minja patches + ChatTemplateOptions (Gemma 4 chat template needs them).
add_executable(tokenizer_minja_patches test_minja_patches.cpp)
target_link_libraries(tokenizer_minja_patches PRIVATE r4dx_tokenizer)
add_test(NAME tokenizer_minja_patches COMMAND tokenizer_minja_patches)

# Gemma 4 (SPM-style byte_fallback backend + Gemma chat template): the same executable replays
# golden_gemma.json (tools/tok_ref/gen_golden_gemma.py) against the assembled tokenizer dir --
# google/gemma-4-12B-it's tokenizer.json (identical in the Huihui repo), tokenizer_config.json,
# chat_template.jinja, generation_config.json. Skipped (77) when the directory is absent.
set(R4DX_GEMMA_TOKENIZER_DIR "${R4DX_MODELS_ROOT}/Huihui-gemma-4-12B-it-abliterated-tok" CACHE PATH
    "Directory containing the Gemma 4 tokenizer.json/tokenizer_config.json/chat_template.jinja/generation_config.json")
add_test(NAME tokenizer_golden_gemma
         COMMAND tokenizer_golden --golden ${CMAKE_CURRENT_SOURCE_DIR}/golden_gemma.json
                                  --model-dir ${R4DX_GEMMA_TOKENIZER_DIR})
set_tests_properties(tokenizer_golden_gemma PROPERTIES SKIP_RETURN_CODE 77)

# LFM2.5-24B-A2B (byte-level BPE with the Llama-3 split regex, no normalizer, no automatic BOS, chat
# template with Pythonic tool calls): golden_lfm2.json from tools/tok_ref/gen_golden_lfm2.py. The model dir
# is the checkpoint dir itself (tokenizer.json, tokenizer_config.json, chat_template.jinja,
# generation_config.json). Override with the R4DX_LFM2_TOKENIZER_DIR environment variable or cache entry.
if(DEFINED ENV{R4DX_LFM2_TOKENIZER_DIR})
  set(_r4dx_lfm2_tok_default "$ENV{R4DX_LFM2_TOKENIZER_DIR}")
else()
  set(_r4dx_lfm2_tok_default "${R4DX_MODELS_ROOT}/LiquidAI/LFM2.5-24B-A2B")
endif()
set(R4DX_LFM2_TOKENIZER_DIR "${_r4dx_lfm2_tok_default}" CACHE PATH
    "Directory containing the LFM2.5-24B-A2B tokenizer.json/tokenizer_config.json/chat_template.jinja/generation_config.json")
add_test(NAME tokenizer_golden_lfm2
         COMMAND tokenizer_golden --golden ${CMAKE_CURRENT_SOURCE_DIR}/golden_lfm2.json
                                  --model-dir ${R4DX_LFM2_TOKENIZER_DIR})
set_tests_properties(tokenizer_golden_lfm2 PROPERTIES SKIP_RETURN_CODE 77)

# Nanbeige4.3-33B (the Llama `legacy: false` SentencePiece shape: Metaspace prepend "first" + byte_fallback BPE + a
# Strip decoder; chat template with <think> and XML / JSON tool calls): golden_nanbeige.json from
# tools/tok_ref/gen_golden_nanbeige.py. The model dir is the -tok directory (chat_template.jinja extracted from
# tokenizer_config.json, docs/nanbeige-plan.md 5.2). Override with R4DX_NANBEIGE_TOKENIZER_DIR.
if(DEFINED ENV{R4DX_NANBEIGE_TOKENIZER_DIR})
  set(_r4dx_nanbeige_tok_default "$ENV{R4DX_NANBEIGE_TOKENIZER_DIR}")
else()
  set(_r4dx_nanbeige_tok_default "${R4DX_MODELS_ROOT}/Nanbeige/Nanbeige4.3-33B-tok")
endif()
set(R4DX_NANBEIGE_TOKENIZER_DIR "${_r4dx_nanbeige_tok_default}" CACHE PATH
    "Directory containing the Nanbeige4.3-33B tokenizer.json/tokenizer_config.json/chat_template.jinja/generation_config.json")
add_test(NAME tokenizer_golden_nanbeige
         COMMAND tokenizer_golden --golden ${CMAKE_CURRENT_SOURCE_DIR}/golden_nanbeige.json
                                  --model-dir ${R4DX_NANBEIGE_TOKENIZER_DIR})
set_tests_properties(tokenizer_golden_nanbeige PROPERTIES SKIP_RETURN_CODE 77)

if(NOT EXISTS "${R4DX_TOKENIZER_MODEL_DIR}/tokenizer.json")
  message(WARNING "tests/tokenizer: ${R4DX_TOKENIZER_MODEL_DIR}/tokenizer.json not found; "
                   "'tokenizer_golden' is registered but will report SKIPPED at run time. Set "
                   "-DR4DX_TOKENIZER_MODEL_DIR=... to point at a Qwen3.8-27B-family checkout.")
endif()
