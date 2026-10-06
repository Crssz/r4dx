# Machine-specific locations, derived from the environment so that no local account name is
# hardcoded into the repository. Override either at configure time (-DR4DX_REFERENCE_VENV=...) or
# via the matching environment variable.
if(NOT R4DX_REFERENCE_VENV AND DEFINED ENV{R4DX_REFERENCE_VENV})
  file(TO_CMAKE_PATH "$ENV{R4DX_REFERENCE_VENV}" R4DX_REFERENCE_VENV)
endif()

# Repo root, for the tests that read tools/reference/golden_out/** by absolute path. Must precede
# the add_subdirectory() calls below so those targets inherit the definition.
add_compile_definitions(R4DX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(smoke_r4d smoke_r4d.cpp)
target_link_libraries(smoke_r4d PRIVATE r4d_core r4dx_hip_runtime)
add_test(NAME smoke_r4d COMMAND smoke_r4d)

# src/convert's own unit tests (CPU-only, no HIP device needed).
add_subdirectory(convert)

# src/core and src/kernels own everything under these two subdirectories (tests/core/**,
# tests/kernels/**); wired in here since this file is the one place add_subdirectory() for the
# whole tests/ tree already lives.
add_subdirectory(core)
add_subdirectory(kernels)

# tests/tokenizer/: owned by the tokenizer component (src/tokenizer/**, tests/tokenizer/**,
# tools/tok_ref/**); this add_subdirectory() line is the only touch it makes outside those paths.
add_subdirectory(tokenizer)

# tests/model/: owned by the model-core component (src/model/** except src/model/attention/**,
# tests/model/** except tests/model/attention/**); this add_subdirectory() line is the only touch
# it makes outside those paths.
add_subdirectory(model)

# tests/vision/: CPU-only golden tests for the host-side vision path (src/vision/**,
# tests/vision/**, docs/vision.md); this add_subdirectory() line is the only touch that component
# makes outside those paths (plus its own src/vision entry in the root CMakeLists.txt).
add_subdirectory(vision)
add_subdirectory(audio)

# tests/cli/: r4dx-cli argument-parsing unit test (src/cli/**).
add_subdirectory(cli)

# tests/server/: r4dx-server's CPU-only unit tests (src/server/**, tests/server/**,
# docs/server.md); this add_subdirectory() line is the only touch this component makes outside
# those paths.
add_subdirectory(server)

# The Python reference tests below need a reference interpreter with torch + transformers + numpy
# importable. Candidates, the first one that exists AND imports all three at CONFIGURE time wins:
#   1. -DR4DX_REFERENCE_PYTHON=<python.exe> (explicit; e.g. an interpreter that is not in a venv);
#   2. R4DX_REFERENCE_VENV's Scripts/python.exe (the -D value or the environment variable);
#   3. the R4DX_REFERENCE_PYTHON environment variable;
#   4. python on PATH -- looked up afresh on every configure (NO_CACHE), so a PATH change is seen.
# The interpreter is machine-local and can be absent or lack those packages, so with no usable
# candidate the tests are not registered; a STATUS line says so, and ctest never lists them (an
# unregistered test cannot count as a failure). build.ps1 re-runs the configure step on every build,
# so setting either environment variable brings them back on the next build.
#
# Two tiers. reference_hessian_rms, reference_gguf_dequant and reference_trellis_quant run on any
# such interpreter. reference_manifest, reference_dflash2 and reference_hessian_corpus compare
# against goldens and tokenizer ids made by the reference venv (transformers 5.17.0, plus its
# optional GDN kernel packages), so they are registered only when the chosen interpreter's
# transformers is exactly that version; on any other they would fail for the environment, not the
# code.
set(_r4dx_ref_transformers "5.17.0")
set(_r4dx_ref_candidates "")
if(R4DX_REFERENCE_PYTHON)
  list(APPEND _r4dx_ref_candidates "${R4DX_REFERENCE_PYTHON}")
endif()
if(R4DX_REFERENCE_VENV)
  list(APPEND _r4dx_ref_candidates "${R4DX_REFERENCE_VENV}/Scripts/python.exe")
endif()
if(NOT "$ENV{R4DX_REFERENCE_PYTHON}" STREQUAL "")
  file(TO_CMAKE_PATH "$ENV{R4DX_REFERENCE_PYTHON}" _r4dx_env_python)
  list(APPEND _r4dx_ref_candidates "${_r4dx_env_python}")
endif()
find_program(_r4dx_path_python NAMES python NO_CACHE)
if(_r4dx_path_python)
  list(APPEND _r4dx_ref_candidates "${_r4dx_path_python}")
endif()
set(_r4dx_ref_ok FALSE)
set(_r4dx_ref_python "")
set(_r4dx_ref_version "")
foreach(_r4dx_cand IN LISTS _r4dx_ref_candidates)
  if(NOT EXISTS "${_r4dx_cand}")
    continue()
  endif()
  execute_process(COMMAND "${_r4dx_cand}" -c
                          "import torch, transformers, numpy; print(transformers.__version__)"
                  RESULT_VARIABLE _r4dx_ref_rc OUTPUT_VARIABLE _r4dx_ref_out
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_r4dx_ref_rc EQUAL 0)
    set(_r4dx_ref_ok TRUE)
    set(_r4dx_ref_python "${_r4dx_cand}")
    set(_r4dx_ref_version "${_r4dx_ref_out}")
    break()
  endif()
  message(STATUS "r4dx: reference interpreter candidate '${_r4dx_cand}' cannot import "
                 "torch/transformers/numpy -- skipped")
endforeach()
set(_r4dx_ref_canonical FALSE)
if(_r4dx_ref_ok AND _r4dx_ref_version STREQUAL _r4dx_ref_transformers)
  set(_r4dx_ref_canonical TRUE)
endif()

if(_r4dx_ref_ok)
  message(STATUS "r4dx: Python reference tests use '${_r4dx_ref_python}' "
                 "(transformers ${_r4dx_ref_version})")

  # tools/reference/hessian_capture.py's weightless rms taps (--rms-taps / --rms-only, docs/quant2.md
  # 3.2): a tiny random 2-layer Qwen3_5 stack driven through the tool's own capture, gates, manifest
  # merge and --regate on the CPU. No checkpoint, no GPU (it checks no CUDA/HIP context was made).
  add_test(
      NAME reference_hessian_rms
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_hessian_rms.py"
  )
  set_tests_properties(reference_hessian_rms PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=6")

  # tools/reference/gguf_dequant.py + gguf_validate.py (the --weights-gguf path of
  # full_logits_golden.py): hand-built blocks per ggml type, a synthetic GGUF, the qwen35 map, and
  # -- when the Unsloth GGUF and the bf16 checkpoint are on disk -- real tensors. numpy only, no GPU.
  add_test(
      NAME reference_gguf_dequant
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_gguf_dequant.py"
  )
  set_tests_properties(reference_gguf_dequant PROPERTIES ENVIRONMENT "PYTHONIOENCODING=utf-8")

  # tools/reference/fp8_dequant.py (block-FP8 -> bf16 checkpoint, header differ): synthetic tiny
  # checkpoint, ragged shapes, both scale names, bit-exact, refusals. torch + safetensors, CPU only.
  add_test(
      NAME reference_fp8_dequant
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_fp8_dequant.py"
  )
  set_tests_properties(reference_fp8_dequant PROPERTIES ENVIRONMENT "PYTHONIOENCODING=utf-8")

  # tools/reference/trellis_quant.py (the EXL3 trellis oracle, docs/trellis.md) and its native
  # encoders (trellis_viterbi.hip, built on first use with the venv's ROCm clang++): codebooks,
  # brute-force Viterbi optimality, native == torch bit for bit, the bitstream, LDLQ / regularize /
  # refit, the int4 baseline, allocation and the --weights-override plumbing. CPU; the HIP encoder
  # is checked too when the environment has HIP_VISIBLE_DEVICES=1, SKIPped otherwise.
  add_test(
      NAME reference_trellis_quant
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_trellis_quant.py"
  )
  set_tests_properties(reference_trellis_quant PROPERTIES ENVIRONMENT "PYTHONIOENCODING=utf-8")

  # tools/reference/arch_table.py and the Gemma 4 port of hessian_capture / imatrix_capture / trellis_quant
  # (docs/gemma4-plan.md M1-25 / M1-28): the tables against the real 677-tensor header, the token-id corpus,
  # the --dry-run CLIs, a real CPU capture checked against hooks on the HF layers, rotation_oracle's folds and,
  # with r4dx-convert built, the whole rotated chain (oracle -> r4dx-convert --rotate --trellis-from). CPU only;
  # prints SKIP and exits 0 when the interpreter's transformers has no gemma4_unified.
  add_test(
      NAME reference_arch_gemma
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_arch_gemma.py"
  )
  set_tests_properties(reference_arch_gemma PROPERTIES ENVIRONMENT "PYTHONIOENCODING=utf-8;CUDA_VISIBLE_DEVICES=")
  if(TARGET r4dx-convert)
    set_property(TEST reference_arch_gemma APPEND PROPERTY ENVIRONMENT "R4DX_CONVERT_EXE=$<TARGET_FILE:r4dx-convert>")
  endif()

  # tools/reference/gemma_mix.py: the Gemma mix4.5m ranking and its budget math on the real 328 linears
  # (docs/gemma4-plan.md 11). CPU only; the trellis_quant wiring part skips without torch.
  add_test(
      NAME reference_gemma_mix
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_gemma_mix.py"
  )
  set_tests_properties(reference_gemma_mix PROPERTIES ENVIRONMENT "PYTHONIOENCODING=utf-8;CUDA_VISIBLE_DEVICES=")
endif()

if(_r4dx_ref_canonical)
  # tools/reference's pure-Python golden-generator smoke test (CPU-only, no HIP device, no C++
  # build -- see tools/reference/README.md). test_manifest.py is a bare-assert script with no
  # pytest dependency, so this is a plain COMMAND, not a build target. This add_test() line is the
  # only touch this component makes outside tools/reference/**, tests/reference/**, and
  # docs/validation.md.
  add_test(
      NAME reference_manifest
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_manifest.py"
  )

  # tools/reference/dflash2_selftest.py -- the DFlash2 reference determinism gate (see
  # docs/dflash2.md section 10). CPU-only, numpy-only for the default fast path (no
  # --with-real-sanity here, since that flag needs the full 27B checkpoint on disk and is meant for
  # manual invocation, not the default ctest run). Fixtures under tools/reference/golden_out/dflash2/
  # are regenerated on demand by the script itself when missing, so this runs unmodified on a clean
  # checkout. Same venv as reference_manifest above; this add_test() line is the only touch this
  # component makes outside tools/reference/**, docs/dflash2.md, and docs/validation.md.
  add_test(
      NAME reference_dflash2
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_SOURCE_DIR}/tools/reference/dflash2_selftest.py"
  )

  # hessian_capture.py's corpus source 4, the self-generated samples.jsonl (--gen-file, docs/quant2.md
  # "Corpus v2"): contract checks, chat-template rendering, per-category packing, provenance and the
  # kl_corpus disjointness gate, on tests/reference/fixtures/hessian_gen_samples.jsonl. No GPU; the
  # rendering/packing part reads the checkpoint's tokenizer and SKIPs when it is not on disk.
  # PYTHONIOENCODING: ctest pipes stdout, and a pipe on Windows gets the ANSI code page (cp1252),
  # which cannot hold the Thai the gate prints. The test checks that path itself on its own strict
  # cp1252 stream; this only keeps the log readable.
  add_test(
      NAME reference_hessian_corpus
      COMMAND "${_r4dx_ref_python}"
              "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_hessian_corpus.py"
  )
  set_tests_properties(reference_hessian_corpus PROPERTIES ENVIRONMENT
                       "OMP_NUM_THREADS=1;PYTHONIOENCODING=utf-8")
elseif(_r4dx_ref_ok)
  message(STATUS "r4dx: reference_manifest, reference_dflash2 and reference_hessian_corpus NOT "
                 "registered -- '${_r4dx_ref_python}' has transformers ${_r4dx_ref_version}, not "
                 "the reference venv's ${_r4dx_ref_transformers} their goldens were made with")
else()
  message(STATUS "r4dx: reference_manifest, reference_dflash2, reference_hessian_rms, "
                 "reference_hessian_corpus, reference_gguf_dequant and reference_trellis_quant NOT "
                 "registered -- no reference interpreter imports torch/transformers/numpy (set "
                 "R4DX_REFERENCE_PYTHON or R4DX_REFERENCE_VENV to one that does and reconfigure)")
endif()
