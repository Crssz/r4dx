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
  test_dflash_feature_capture
  test_dflash_draft
  test_dflash_e2e
  test_teacher_forced_logprobs
  # Tensor-parallel P2a (docs/tp.md 10.1): the sharded Container::Load of both ranks, every layout,
  # read back against the file's bytes. Device 1, SKIP 77 without the 4-layer container.
  test_tp_loader
  # Tensor-parallel P2b (docs/tp.md 10.1): TpModel in --tp-mode emulate (both ranks on device 1)
  # against the TP=1 Model -- numerics, greedy/sampled merges, VerifyWindow, fault recovery.
  test_tp_emulation
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

# test_dflash_draft (Milestone 5 stage S2, docs/dflash2.md "Implementation") reads the same
# tools/reference/golden_out/dflash2/fixture_{a,b,c} .npy arrays the five S1 kernel tests do, so it
# reuses their reader (tests/kernels/npy_fixture.hpp) rather than duplicating it -- an include dir
# is the whole coupling; no tests/kernels target is linked, and nothing in that directory changes.
# It also SKIPs (77) when the fixtures or the DFlash2 draft container are absent, exactly like those
# tests; the foreach() above already sets SKIP_RETURN_CODE for every R4DX_MODEL_TESTS entry.
target_include_directories(test_dflash_draft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../kernels)

# test_mtp's CheckChatMultiTurnMidRoundStop (docs/status.md Known-gaps: "no automated
# multi-turn-with-a-forced-mid-round-stop regression test") drives the exact production
# r4dx::server::PrefixState (src/server/prefix_state.h) -- header-only, no HIP/r4dx::model
# dependency of its own, same reasoning tests/server/test_prefix_state.cpp relies on -- so it is
# safe to include directly from this HIP-linked test binary without linking r4dx_server itself.
target_include_directories(test_mtp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/server)

# test_dflash_e2e (Milestone 5 stage S3 item 3): the --dflash variant of test_mtp's own
# CheckChatMultiTurnMidRoundStop above -- same PrefixState dependency, same reasoning.
target_include_directories(test_dflash_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/server)

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

# tool_dflash_probe: Milestone 5 stage S2 items 4 and 5 (docs/dflash2.md "Implementation") -- the
# REAL-input anchor check (dump the target's captured features for a tokenized prompt, draft, and
# hand the Python reference the same inputs) plus the isolated DraftRound/InjectFeatures cost.
# Needs the full 27B container, a DFlash2 draft container AND the tokenizer, and prints numbers for
# a human/doc rather than asserting a contract -- so the same "built, never add_test()'d"
# convention as tool_hseed_drift/tool_vocab_calib above.
add_executable(tool_dflash_probe tool_dflash_probe.cpp)
target_link_libraries(tool_dflash_probe PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core r4dx_tokenizer)

# tool_sampled_bench: Milestone 6 stage S2 item 4 (docs/sampling.md section 10) -- ms per PLAIN
# SAMPLED decode token through the pre-S2 full-vocab path vs the device-row-summary path, on the
# real container. Same "built, never add_test()'d" convention as the three tools above (it prints
# numbers for a human and for docs/sampling.md; it asserts no contract), and the same
# SKIP-if-container-missing exit 77.
add_executable(tool_sampled_bench tool_sampled_bench.cpp)
target_link_libraries(tool_sampled_bench PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

# tool_teacher_forced_logprobs: the r4dx half of Rung 4 (docs/validation.md "Rung 4 tooling") --
# a teacher-forced per-position next-token log-prob dump for a fixed token sequence, in the shared
# on-disk format the HF-transformers reference half writes too. Needs the real container and a
# tokens.json that are not vendored into the repo, and produces artifacts rather than asserting a
# contract, so it follows the same "built, never add_test()'d" convention as tool_hseed_drift /
# tool_vocab_calib / tool_dflash_probe / tool_sampled_bench above. The contract IS asserted, on the
# 4-layer container, by test_teacher_forced_logprobs in R4DX_MODEL_TESTS above -- both drive the
# same pass (tests/model/teacher_forced.h), so the tool's numbers are the ones ctest checks.
add_executable(tool_teacher_forced_logprobs tool_teacher_forced_logprobs.cpp)
target_link_libraries(tool_teacher_forced_logprobs PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

# tool_tp_step_bench: tensor-parallel P2a gate G3 (docs/tp.md 11) -- ms/token of ONE rank's shard
# with a no-op all-reduce (tp::MakeNoopComm) vs the TP=1 Model, same tool, same protocol. Prints
# numbers for the gate script, asserts nothing: built, never add_test()'d, like the tools above.
# Renders the standard prompt through the real chat template, so it links r4dx_tokenizer -- and,
# like r4dx-cli (src/cli/CMakeLists.txt), puts third_party ahead of the transitive ROCm include
# path so chat_template.h's nlohmann/json.hpp is the vendored v3.11.3 r4dx_tokenizer was built with.
add_executable(tool_tp_step_bench tool_tp_step_bench.cpp)
target_include_directories(tool_tp_step_bench PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(tool_tp_step_bench PRIVATE r4dx_model r4dx_tokenizer r4dx_convert r4dx_core
                                                 r4d_core)

# tool_tp_soak: tensor-parallel P4 gate G8 (docs/tp.md 10.5) -- one real TpModel for --minutes of
# random prefill/decode iterations with a fixed canary; wrapped by tools/tp/soak.ps1 (pre-flight,
# TDR check). Takes both GPUs for as long as it runs: built, never add_test()'d.
add_executable(tool_tp_soak tool_tp_soak.cpp)
target_link_libraries(tool_tp_soak PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

# test_tp_real_vs_emulation: tensor-parallel P4 (docs/tp.md 10.1) -- emulate vs real, byte for byte,
# plus real-GPU fault recovery. BOTH GPUs, LABEL tp2gpu, opt-in exactly like
# tests/kernels/test_tp_allreduce_2gpu: deliberately NO HIP_VISIBLE_DEVICES property (under the
# win-hip test preset it sees one device and skips), SKIP 77 unless R4DX_TP2GPU=1 and two devices are
# visible. tests\run_tests.ps1 -TwoGpu runs it; the default run excludes the label.
add_executable(test_tp_real_vs_emulation test_tp_real_vs_emulation.cpp)
target_link_libraries(test_tp_real_vs_emulation PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
add_test(NAME test_tp_real_vs_emulation COMMAND test_tp_real_vs_emulation)
set_tests_properties(test_tp_real_vs_emulation PROPERTIES LABELS tp2gpu SKIP_RETURN_CODE 77)

# test_mtp_round: pure CPU unit test for src/model/mtp_round.hpp (ProcessMtpRound) -- no HIP
# device, model container, or GDN/KV state needed (header-only std::vector<int32_t> arithmetic),
# same "plain add_test(), no ENVIRONMENT/SKIP_RETURN_CODE" pattern tests/server's HIP-free tests
# use. Kept out of R4DX_MODEL_TESTS above so it never gets SKIPPED on a machine without a real
# container -- it should always run.
add_executable(test_mtp_round test_mtp_round.cpp)
target_include_directories(test_mtp_round PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_mtp_round COMMAND test_mtp_round)

# test_pick_tuning: every tuning PickTuning hands out for a shape src/model/gemm_tuning_table.inc
# covers must be launchable at THIS build's kernel groups -- the table is swept at one
# R4DX_W4A16_GROUP but compiled into both builds (linear.cpp's ResolveTuning skips the rows that
# are not). Host-only (no HIP call, no container), so it always runs; HIP_VISIBLE_DEVICES=1 only
# because linking r4dx_model loads the HIP runtime, which must never see device 0.
add_executable(test_pick_tuning test_pick_tuning.cpp)
target_include_directories(test_pick_tuning PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
target_link_libraries(test_pick_tuning PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
add_test(NAME test_pick_tuning COMMAND test_pick_tuning)
set_tests_properties(test_pick_tuning PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")

# Tensor-parallel P1 (docs/tp.md 10.1): four CPU-only tests of the TP building blocks under
# src/model/tp/ -- no HIP call, no device, no container -- so, like test_mtp_round above, plain
# add_test() with no ENVIRONMENT / SKIP_RETURN_CODE: they always run.
#   test_tp_config      ModelConfig::Shard (model_config.h)
#   test_tp_shard       RuleFor/RankRows/PlanRows/PlanCols/Gather against the converter's packers
#   test_tp_vocab_merge the vocab-split merges (tp_vocab.h) against full-row CPU references
#   test_tp_rank_worker the facade/rank hand-off and progress watchdog (tp_rank_worker.h)
# (the fifth P1 test, test_tp_host_exchange, lives in tests/core with the header it tests).
add_executable(test_tp_config test_tp_config.cpp)
target_link_libraries(test_tp_config PRIVATE r4dx_tp_shard)
add_test(NAME test_tp_config COMMAND test_tp_config)

add_executable(test_tp_shard test_tp_shard.cpp)
target_link_libraries(test_tp_shard PRIVATE r4dx_tp_shard r4dx_convert)
add_test(NAME test_tp_shard COMMAND test_tp_shard)

# tp_vocab.h includes the HIP-free r4dx/kernels/summary_sampler.hpp; an include dir is the whole
# coupling (linking r4dx_kernels would drag in its hipcc object for nothing).
add_executable(test_tp_vocab_merge test_tp_vocab_merge.cpp)
target_include_directories(test_tp_vocab_merge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model
                                                       ${CMAKE_SOURCE_DIR}/src/kernels/include)
add_test(NAME test_tp_vocab_merge COMMAND test_tp_vocab_merge)

add_executable(test_tp_rank_worker test_tp_rank_worker.cpp)
target_include_directories(test_tp_rank_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_tp_rank_worker COMMAND test_tp_rank_worker)

# test_dflash_draft_weights: task A1 item 3's CPU-only DflashDraftWeights loader stub
# (src/model/dflash_draft_weights.h -- metadata + tensor directory, no device upload). Links only
# r4dx_convert (header-only: SafetensorsReader, GgufReader, dflash2_container writer helpers used
# to build the test's own tiny fixture container) + r4dx_core (bf16 conversions) -- deliberately
# NOT r4dx_model, and it makes no HIP *call*, so it keeps the "always runs, no
# ENVIRONMENT/SKIP_RETURN_CODE" pattern of test_mtp_round above. (r4dx_core does INTERFACE-link
# r4d_core/r4dx_hip_runtime, and since Milestone 11 this test needs that: dflash_draft_weights.h
# calls r4d_gemm_w4a16_nt_m64_group() through quant_linear.h's CheckW4a16Group, which is a host
# function -- no device required.)
add_executable(test_dflash_draft_weights test_dflash_draft_weights.cpp)
target_include_directories(test_dflash_draft_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
target_link_libraries(test_dflash_draft_weights PRIVATE r4dx_convert r4dx_core)
target_compile_definitions(test_dflash_draft_weights PRIVATE
  R4DX_CONVERT_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../convert/fixtures")
add_test(NAME test_dflash_draft_weights COMMAND test_dflash_draft_weights)

# test_keep_bf16 (Milestone 11, docs/validation.md "Milestone 11 / sensitivity"): converts a real
# 4-layer container with `r4dx-convert --keep-bf16 <regex>` and checks that exactly the matched
# linears load as bf16 while the rest load as w4a16, and that the mixed-layout model generates. Not
# in R4DX_MODEL_TESTS above because it needs the r4dx-convert TARGET (it runs the binary), which
# R4DX_BUILD_CONVERT can switch off; otherwise it follows the same conventions (device 1, SKIP 77
# when the HF checkpoint this machine may not have is absent).
if(TARGET r4dx-convert)
  add_executable(test_keep_bf16 test_keep_bf16.cpp)
  target_link_libraries(test_keep_bf16 PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
  target_compile_definitions(test_keep_bf16 PRIVATE
    R4DX_CONVERT_EXE="$<TARGET_FILE:r4dx-convert>")
  add_dependencies(test_keep_bf16 r4dx-convert)
  add_test(NAME test_keep_bf16 COMMAND test_keep_bf16)
  set_tests_properties(test_keep_bf16 PROPERTIES
    ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
    SKIP_RETURN_CODE 77
  )
endif()

# tests/model/attention/: golden test for r4dx::model::attention::AttentionLayer, created
# standalone by that component before this file existed -- see its own CMakeLists.txt header note.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/attention/CMakeLists.txt")
  add_subdirectory(attention)
endif()
