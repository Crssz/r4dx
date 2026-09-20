# Machine-specific locations, derived from the environment so that no local account name is
# hardcoded into the repository. Override either at configure time (-DR4DX_REFERENCE_VENV=...) or
# via the matching environment variable.
if(NOT R4DX_REFERENCE_VENV)
  if(DEFINED ENV{R4DX_REFERENCE_VENV})
    file(TO_CMAKE_PATH "$ENV{R4DX_REFERENCE_VENV}" R4DX_REFERENCE_VENV)
  else()
    file(TO_CMAKE_PATH "$ENV{USERPROFILE}" _r4dx_home)
    set(R4DX_REFERENCE_VENV "${_r4dx_home}/dev/vLLM_for_AMD/.venv-rocm10")
  endif()
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

# tests/cli/: r4dx-cli argument-parsing unit test (src/cli/**).
add_subdirectory(cli)

# tests/server/: r4dx-server's CPU-only unit tests (src/server/**, tests/server/**,
# docs/server.md); this add_subdirectory() line is the only touch this component makes outside
# those paths.
add_subdirectory(server)

# tools/reference's pure-Python golden-generator smoke test (CPU-only, no HIP device, no C++
# build -- see tools/reference/README.md). Needs the reference venv's python.exe (torch +
# transformers importable); test_manifest.py is a bare-assert script with no pytest dependency, so
# this is a plain COMMAND, not a build target. This add_test() line is the only touch this
# component makes outside tools/reference/**, tests/reference/**, and docs/validation.md.
add_test(
    NAME reference_manifest
    COMMAND "${R4DX_REFERENCE_VENV}/Scripts/python.exe"
            "${CMAKE_CURRENT_SOURCE_DIR}/reference/test_manifest.py"
)

# tools/reference/dflash2_selftest.py -- the DFlash2 reference determinism gate (see
# docs/dflash2.md section 10). CPU-only, numpy-only for the default fast path (no --with-real-
# sanity here, since that flag needs the full 27B checkpoint on disk and is meant for manual
# invocation, not the default ctest run). Fixtures under tools/reference/golden_out/dflash2/ are
# regenerated on demand by the script itself when missing, so this runs unmodified on a clean
# checkout. Same venv as reference_manifest above; this add_test() line is the only touch this
# component makes outside tools/reference/**, docs/dflash2.md, and docs/validation.md.
add_test(
    NAME reference_dflash2
    COMMAND "${R4DX_REFERENCE_VENV}/Scripts/python.exe"
            "${CMAKE_SOURCE_DIR}/tools/reference/dflash2_selftest.py"
)
