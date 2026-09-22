# r4dx-convert unit tests (task item 5): quantizer round-trip error bounds on random data, and
# packer byte-exactness against fixtures generated from the Python references
# (tools/convert_ref/gen_fixtures.py -> tests/convert/fixtures/). Both link r4dx_convert directly
# -- everything under src/convert/include is header-only, so no separate static library exists to
# link instead.

add_executable(convert_quantize_roundtrip test_quantize_roundtrip.cpp)
target_link_libraries(convert_quantize_roundtrip PRIVATE r4dx_convert)
add_test(NAME convert_quantize_roundtrip COMMAND convert_quantize_roundtrip)

add_executable(convert_pack_bytes test_pack_bytes.cpp)
target_link_libraries(convert_pack_bytes PRIVATE r4dx_convert)
target_compile_definitions(convert_pack_bytes PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME convert_pack_bytes COMMAND convert_pack_bytes)

# Review finding (major): test_pack_bytes.cpp only gates the packer against fixtures produced by
# the same author's Python references (a shared permutation bug would pass silently), and the only
# check against the real kernels' actual indexing was a manual, non-CTest GPU script. This test
# decodes the packed bytes by the kernels' own literal indexing (re-derived from the .hip sources,
# not by inverting the packer) -- CPU-only, always runs.
# --quant search (src/convert/include/r4dx_convert/quant_search.hpp): never worse than RTN on any
# group, a large win on an outlier-dominated group once an imatrix de-weights the outlier channel,
# and bit-exact agreement with tools/convert_ref/{w4_ref,mxfp4_ref}.py's search on the shared
# random fixture. Needs the fixtures directory for the last part.
add_executable(convert_quant_search test_quant_search.cpp)
target_link_libraries(convert_quant_search PRIVATE r4dx_convert)
target_compile_definitions(convert_quant_search PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME convert_quant_search COMMAND convert_quant_search)

add_executable(convert_kernel_decode test_kernel_decode.cpp)
target_link_libraries(convert_kernel_decode PRIVATE r4dx_convert)
add_test(NAME convert_kernel_decode COMMAND convert_kernel_decode)

# --kv-calib (task item 2): r4dx_convert::ResolveKvDescale's amax/448.0 convention, missing-layer/
# missing-key/wrong-length fallback-to-1.0 warning paths, and the calib_applicable=false silent path.
add_executable(convert_kv_calib test_kv_calib.cpp)
target_link_libraries(convert_kv_calib PRIVATE r4dx_convert)
add_test(NAME convert_kv_calib COMMAND convert_kv_calib)

# bf16 linear-weight layout (task item 2: --layouts bf16 for all linear weights): plain row-major
# on-disk bytes, no WMMA-fragment permutation, plan/emit byte-size agreement.
add_executable(convert_bf16_layout test_bf16_layout.cpp)
target_link_libraries(convert_bf16_layout PRIVATE r4dx_convert)
add_test(NAME convert_bf16_layout COMMAND convert_bf16_layout)

# DFlash2 task A1 item 1: minimal GGUF v3 reader (metadata KV of every type incl. arrays/strings,
# tensor directory, F32/F16/BF16/Q8_0 dequant) against tests/convert/fixtures/tiny.gguf (generated
# by tools/convert_ref/make_tiny_gguf.py, no gguf-py dependency either side).
add_executable(convert_gguf_reader test_gguf_reader.cpp)
target_link_libraries(convert_gguf_reader PRIVATE r4dx_convert)
target_compile_definitions(convert_gguf_reader PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME convert_gguf_reader COMMAND convert_gguf_reader)

# DFlash2 task A1 item 6: DFlash2 draft container metadata + tensor directory round-trip on the
# 2-layer/64-dim/512-vocab miniature GGUF (tests/convert/fixtures/dflash_mini.gguf), plus the
# linear-weight orientation test.
add_executable(convert_dflash_container test_dflash_container.cpp)
target_link_libraries(convert_dflash_container PRIVATE r4dx_convert)
target_compile_definitions(convert_dflash_container PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME convert_dflash_container COMMAND convert_dflash_container)
