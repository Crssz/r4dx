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

# --ldlq (docs/quant2.md section 2.2, gate G1): dense_linalg.hpp's GEMM / Cholesky / triangular
# inverse against double references (bit-identical across thread counts and the AVX-512 / scalar
# paths), FactorHessian's damping and retry, H = I reproducing the no-refit group search byte for
# byte, a proxy-loss win of at least 20% over search+diag(H) on correlated inputs for all three
# layouts, thread-count-independent output bytes, HessianStore on fixtures/hess_small (written by
# tools/reference/hessian_capture.py --write-fixture), and the packers accepting LDLQ output. CPU-only.
add_executable(convert_quant_ldlq test_quant_ldlq.cpp)
target_link_libraries(convert_quant_ldlq PRIVATE r4dx_convert)
target_compile_definitions(convert_quant_ldlq PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME convert_quant_ldlq COMMAND convert_quant_ldlq)

# --rotate (docs/quant2.md sections 3-4, rotation.hpp): splitmix64 generation of rotation.signs /
# mix5 / had_*_signs, Q and Hb against their closed forms (every entry of the production 5120 Q),
# orthogonality and round trips, fold exactness on a random miniature layer (in-projection with the
# norm folded, out-projection, and the q2ab down / o / gdn_out K side), the Hessian change of basis
# against a direct capture on the folded input plus the tr(W H W^T) invariance, HessianStore's
# transform hook and cache key on fixtures/hess_small, and the imatrix diagonal model. CPU-only.
add_executable(convert_rotation test_rotation.cpp)
target_link_libraries(convert_rotation PRIVATE r4dx_convert)
target_compile_definitions(convert_rotation PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME convert_rotation COMMAND convert_rotation)

# The weightless rms Hessian path for rotated in-projections (docs/quant2.md 3.2; rotation.hpp's
# TransformHessianRmsQ, hessian.json "rms_keys"): Q^T H Q against the dense closed-form Q at the
# miniature and production sizes, a dead norm channel ((1 + w) == 0: the division path refuses, the
# rms path matches a direct capture on rms(x) Q), HessianStore's rms_keys parsing and refusals, and
# -- when r4dx-convert is built -- the exe on a synthetic 2-layer checkpoint: unrotated output
# byte-identical to the pre-rms build (pinned digest) whether or not rms_keys exist, q2ab refused
# during planning without them, and with them the recorded ldlq_rms_linears and w4a16 bytes equal to
# fold -> Q^T H_rms Q -> LDLQ. CPU-only; writes ~250 MB under %TEMP% and removes it.
add_executable(convert_rms_hessian test_rms_hessian.cpp)
target_link_libraries(convert_rms_hessian PRIVATE r4dx_convert)
if(TARGET r4dx-convert)
  target_compile_definitions(convert_rms_hessian PRIVATE
    R4DX_CONVERT_EXE="$<TARGET_FILE:r4dx-convert>")
  add_dependencies(convert_rms_hessian r4dx-convert)
endif()
add_test(NAME convert_rms_hessian COMMAND convert_rms_hessian)

# --record-reuse-guard / --reuse-tensors-from (docs/quant2.md 5.2; reuse_guard.hpp): the library
# pieces (file hashing, the guard diff, ContainerWriter's chunked write and header patch) and -- when
# r4dx-convert is built -- the exe on a synthetic 2-shard checkpoint: a reuse run with one group rule
# from a no-rule baseline is byte-identical to the full run with that rule (header modulo
# reused_from), the reverse too, with the Q3 sweep's recipe and with --rotate q2ab --ldlq; so is a
# reuse whose --keep-bf16 differs (a keep added, removed, together with a rule, and on the rotated
# LDLQ fixture); every guard field (real flags, input files, checkpoint files, the binary, and each
# recorded leaf mutated in a copy of the baseline) and every inconsistent baseline is refused before
# an output exists.
# CPU-only; writes ~450 MB under %TEMP% and removes it.
add_executable(convert_reuse test_reuse.cpp)
target_link_libraries(convert_reuse PRIVATE r4dx_convert)
if(TARGET r4dx-convert)
  target_compile_definitions(convert_reuse PRIVATE
    R4DX_CONVERT_EXE="$<TARGET_FILE:r4dx-convert>")
  add_dependencies(convert_reuse r4dx-convert)
endif()
add_test(NAME convert_reuse COMMAND convert_reuse)

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

# --keep-bf16 (Milestone 11, docs/validation.md "Milestone 11 / sensitivity"): regex selection over
# container base names, "matched nothing is a warning not an error", the byte accounting against
# what ContainerWriter really plans, and that a kept linear emits ONLY <base>.bf16.w. CPU-only.
add_executable(convert_keep_bf16 test_keep_bf16.cpp)
target_link_libraries(convert_keep_bf16 PRIVATE r4dx_convert)
add_test(NAME convert_keep_bf16 COMMAND convert_keep_bf16)

# --w4a16-group-rule (docs/quant2.md section 5, Q3; w4a16_groups.hpp): rule parsing and first-match
# selection over real base names, planning checks (K % g, K % 64, N % 16, the bf16-companion guard),
# byte accounting at mixed groups (LinearLayoutBytes, --keep-bf16's extra bytes), the renamed
# `.w4a16.wsz.g<g>` tensor and its bytes, LDLQ at g32/g128 through EmitLinearLayouts, and -- when
# r4dx-convert is built -- the exe's --selftest: byte-identical to pre-Q3 without rules, the
# __metadata__ map and run record with them, and the refusals. CPU-only (the exe only asks r4d_core
# which groups it instantiates).
add_executable(convert_w4a16_groups test_w4a16_groups.cpp)
target_link_libraries(convert_w4a16_groups PRIVATE r4dx_convert)
target_compile_definitions(convert_w4a16_groups PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
if(TARGET r4dx-convert)
  target_compile_definitions(convert_w4a16_groups PRIVATE
    R4DX_CONVERT_EXE="$<TARGET_FILE:r4dx-convert>")
  add_dependencies(convert_w4a16_groups r4dx-convert)
endif()
add_test(NAME convert_w4a16_groups COMMAND convert_w4a16_groups)

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
