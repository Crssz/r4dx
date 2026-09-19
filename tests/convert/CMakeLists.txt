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
add_executable(convert_kernel_decode test_kernel_decode.cpp)
target_link_libraries(convert_kernel_decode PRIVATE r4dx_convert)
add_test(NAME convert_kernel_decode COMMAND convert_kernel_decode)
