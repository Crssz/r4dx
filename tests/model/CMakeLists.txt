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
  # Model::SaveCheckpoint/RestoreCheckpoint (docs/server.md "Prefix cache"): a restore then a prefill
  # is bit-identical to never having generated in between -- plain decode, multi-token speculative
  # rounds, TpModel in emulate mode. Device 1, SKIP 77 without the 4-layer MTP container.
  test_prompt_checkpoint
  # Tensor-parallel P2a (docs/tp.md 10.1): the sharded Container::Load of both ranks, every layout,
  # read back against the file's bytes. Device 1, SKIP 77 without the 4-layer container.
  test_tp_loader
  # Tensor-parallel P2b (docs/tp.md 10.1): TpModel in --tp-mode emulate (both ranks on device 1)
  # against the TP=1 Model -- numerics, greedy/sampled merges, VerifyWindow, fault recovery; P5: the
  # MTP and DFlash2 rounds (the H6/H7 merges through the R4DX_TP_TESTING hooks) and the state they
  # commit, tiny-temperature speculative rounds, and vision on the real container (rank 0's encode,
  # host rows spliced on both ranks) against TP=1.
  test_tp_emulation
  # Trellis milestone M4 (docs/trellis-kernel.md 5.1-5.3, 6): the loader and ApplyLinear on
  # tests/convert's tiny trellis containers against the CPU references of tests/kernels/
  # trellis_ref.hpp, the pre-transformed path, row identity, the ticket reset and every loader
  # refusal of 2.5 at TP=1 and TP=2. Device 1, SKIP 77 without the tiny containers.
  test_trellis_linear
  # The 256-row prefill chunk (docs/prefill.md) against the 64-row one, bit for bit: last-row logits,
  # greedy / MTP / DFlash tokens, the digest of every piece of per-sequence state, the tail lengths of the
  # chunk grid and prefix-reuse suffixes; on the 4-layer container (bf16, w4a16) and the real trellis
  # container. Device 1, SKIP 77 without the containers. Links the R4DX_TP_TESTING library variant for
  # Model::DebugStateDigest. It pins the f16 / exact-attention paths (prefill_int8 = 0 in every Load, and
  # R4DX_PREFILL_SPLITKV=exact unless the environment sets it: split-KV and int8 prefill are the lossy defaults since
  # 2026-10-07); test_prefill_chunk_identity_defaults below runs the 4-layer part with the split-KV default.
  test_prefill_chunk_identity
  # R4DX_PREFILL_INT8 (docs/int8-prefill.md "Production path") against the f16 path on the real trellis container loaded
  # four times (off, on with per-128 scales pinned, on with coarse scales pinned, the defaults = int8 on since 2026-10-07 and coarse scales
  # since 2026-10-08; the defaults must equal the pinned coarse load): T < 256 byte-identical on and off, T >= 256 deterministic with the
  # int8 chunk counts the chunk grid predicts, the state digest differing from f16 (negative control), the last row's
  # logits close, a text-only PrefillMultimodal == Prefill in its live state and int8 chunk count (KV digests excluded there:
  # Reset() leaves stale pages that the digest covers); the 4-layer containers refuse the switch. Device 1, SKIP 77 without the containers. Links the
  # R4DX_TP_TESTING library variant for Model::DebugStateDigest. Written, NOT run by its author.
  test_prefill_int8
  # R4DX_PREFILL_INT8_TP2 (docs/int8-prefill.md "Tensor parallel"): TpModel (emulate, both ranks on device 1) on the real trellis
  # container with the TP switch off (R4DX_PREFILL_INT8_TP2=0, the kill switch) and on (per-128 scales pinned), then the defaults (TP = 2 int8
  # on, coarse scales). Empty TP = 2 tuning tables: the switch refuses int8, both ranks stay f16 and every observable
  # equals the off run's. Rows present: int8 chunk counts per rank, no-super-chunk scenarios identical, determinism, KL bounds, and
  # TP = 2 int8 against TP = 1 int8 (two more loads). Device 1, SKIP 77 without the container. Written, NOT run by its author.
  test_prefill_int8_tp2
  # The GDN sequence ops of a 256-row super-chunk run once (the default) against the old 64-row sub-slices
  # (R4DX_GDN_SLICE=64), and r4d_gdn_conv_prep2 (the default; R4DX_GDN_CONV=1 opts out) against r4d_gdn_conv_prep, byte for
  # byte (docs/trellis-m256.md "GDN sequence ops"): part A synthetic at the real shape (fresh and has_init
  # calls, 256 / 320 / 200 / 17 / 2 / 64 rows, a random initial state), part B GdnLayer::Forward on the
  # 4-layer container's layer 0. Device 1; 77 without a device, or without the container after part A.
  test_gdn_seq256_identity
  # Pipeline-parallel prefill, Phase 1 (docs/pp-prefill.md, gate G1b): the PP-EMULATE mode -- every prompt-prefill
  # chunk as stage A = layers [0, k) and stage B = [k, N) on ONE device, the carry (residual stream, fused-norm
  # pair, DFlash columns, KV rows) through pinned host memory with the device-side destinations poisoned --
  # against the monolithic prefill, bit for bit: logits, tokens, the digest of all state, the DFlash capture;
  # test_prefill_chunk_identity's shape list, plain / --mtp 3 / --dflash / image rows, the arena poison
  # controls and the four negative controls (each must fail the comparison). 4-layer container (bf16, w4a16) and
  # the real trellis container. Device 1, SKIP 77 without the containers. Links the R4DX_TP_TESTING library
  # variant for Model::DebugStateDigest. Written, NOT run by its author (GPU).
  test_pp_emulate_identity
  # Pipeline-parallel prefill, Phase 2 (docs/pp-prefill.md, gate G2a): the REAL two-GPU pipeline (PpModel: stage A on the desktop
  # card, stage B the decode Model) against the monolithic prefill on the same Model, bit for bit -- test_pp_emulate_identity's
  # shape list plus warm turns (the sync-back), speculative rounds, a checkpoint restore, the pipelining threshold, image rows;
  # the live state of both stages compared after every hand-off; two negative controls. NEEDS BOTH GPUs and
  # both cards visible (HIP_VISIBLE_DEVICES=0,1, set below); SKIP 77 otherwise or without the containers. Written, NOT run by its author (GPU).
  test_pp_real_identity
)

foreach(t ${R4DX_MODEL_TESTS})
  add_executable(${t} ${t}.cpp)
  # The tensor-parallel tests link the R4DX_TP_TESTING variant of r4dx_model (its test hooks,
  # src/model/CMakeLists.txt); every other test links the production library.
  if(t STREQUAL "test_tp_emulation" OR t STREQUAL "test_prefill_chunk_identity" OR t STREQUAL "test_prefill_int8" OR
     t STREQUAL "test_pp_emulate_identity" OR t STREQUAL "test_pp_real_identity")
    target_link_libraries(${t} PRIVATE r4dx_model_tptest r4dx_convert r4dx_core r4d_core)
  else()
    target_link_libraries(${t} PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
  endif()
  add_test(NAME ${t} COMMAND ${t})
  set_tests_properties(${t} PROPERTIES
    ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
    SKIP_RETURN_CODE 77
  )
endforeach()
# The real pipeline needs both cards visible: "0,1" is the natural order (the same as unset; this ROCm does not reorder), which
# overrides the loop's HIP_VISIBLE_DEVICES=1. PpModel then puts stage B (decode) on ordinal 1 = physical device 1 (the
# headless card, pci 07) and stage A on ordinal 0 (the desktop card) -- docs/pp-prefill.md 1.2.
set_tests_properties(test_pp_real_identity PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=0,1")

# The 64-row == 256-row chunk identity again with the DEFAULT prompt-prefill attention (split-KV from 2048 tokens of
# context, exact-wide below): the 4-layer part, whose 8145-token scenario crosses the engage depth. The split law is a
# function of (context depth, 64-row slice), so the two chunk sizes should still leave the same bits; written, not run
# by its author (CPU-only session). test_prefill_chunk_identity itself pins R4DX_PREFILL_SPLITKV=exact (the f16 paths).
add_test(NAME test_prefill_chunk_identity_defaults COMMAND test_prefill_chunk_identity)
set_tests_properties(test_prefill_chunk_identity_defaults PROPERTIES
  ENVIRONMENT "HIP_VISIBLE_DEVICES=1;R4DX_PREFILL_SPLITKV=split;R4DX_TEST_ONLY=l4/"
  SKIP_RETURN_CODE 77)

# Trellis (docs/trellis-kernel.md 6): test_trellis_linear and test_tp_loader's trellis cases load
# the tiny containers convert_trellis_import (tests/convert) keeps in its build directory -- the
# fixture trellis_tiny, so ctest runs that test first -- and read tests/kernels/trellis_ref.hpp's
# CPU references (an include dir, as test_dflash_draft below; no tests/kernels target is linked).
foreach(t test_trellis_linear test_tp_loader)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../kernels)
  target_compile_definitions(${t} PRIVATE
    R4DX_TRELLIS_TINY_DIR="${CMAKE_BINARY_DIR}/tests/convert/trellis_import")
  set_tests_properties(${t} PROPERTIES FIXTURES_REQUIRED trellis_tiny)
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
# corpus (<R4DX_MODELS_ROOT>/wikitext-2-raw) run through the real 64-layer container. Same "built but never
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
# test_tf_truncate: CPU-only (no GPU/container) check that --max-tokens' truncation + sha256 agree with
# kl_report.py's raw truncation.
add_executable(test_tf_truncate test_tf_truncate.cpp)
target_link_libraries(test_tf_truncate PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
add_test(NAME test_tf_truncate COMMAND test_tf_truncate)
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
target_link_libraries(test_tp_real_vs_emulation PRIVATE r4dx_model_tptest r4dx_convert r4dx_core r4d_core)
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
# test_prefill_chunk: pure CPU unit test for src/model/prefill_chunk.h (docs/trellis-m256.md): the
# R4DX_PREFILL_CHUNK parser and the decision table for 256-row prefill super-chunks, including the
# fallback for --tp 2, --mtp, --dflash and the other cases the wide path does not serve. No HIP.
add_executable(test_prefill_chunk test_prefill_chunk.cpp)
target_include_directories(test_prefill_chunk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_prefill_chunk COMMAND test_prefill_chunk)
# test_decode_legacy: pure CPU unit test for the decode-t1 host contracts (docs/perf.md): the
# R4DX_DECODE_LEGACY parser (src/core/include/r4dx/core/decode_legacy.hpp) and the one-copy step
# metadata layout (src/model/step_meta.h). Header-only, no HIP; always runs.
add_executable(test_decode_legacy test_decode_legacy.cpp)
target_include_directories(test_decode_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/include)
add_test(NAME test_decode_legacy COMMAND test_decode_legacy)

# test_prefill_int8_cpu: pure CPU unit test for src/model/prefill_int8.h (R4DX_PREFILL_INT8's parser and decision table) and
# the planning side of src/model/linear.cpp (PlanTrellisI8, its tuning table gemm_tuning_table_trellis_i8.inc checked
# against the kernel's own legality rule, ScopedTrellisI8); docs/int8-prefill.md "Production path". Also the TP = 2 rank
# shapes of "Tensor parallel", derived from r4dx_tp_shard's real sharding rules. No HIP call;
# HIP_VISIBLE_DEVICES=-1 hides every device from the runtime that linking r4dx_model_linear loads.
add_executable(test_prefill_int8_cpu test_prefill_int8_cpu.cpp)
target_link_libraries(test_prefill_int8_cpu PRIVATE r4dx_model_linear r4dx_tp_shard)
add_test(NAME test_prefill_int8_cpu COMMAND test_prefill_int8_cpu)
set_tests_properties(test_prefill_int8_cpu PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=-1")
# /EHc-: see tests/kernels/CMakeLists.txt (an extern "C" throw is otherwise uncatchable).
set_source_files_properties(test_prefill_int8_cpu.cpp PROPERTIES COMPILE_OPTIONS "/EHc-")

# test_fake_quant_act_cpu: pure CPU unit test for src/model/fake_quant_act.h (docs/int8-prefill.md): the
# R4DX_FAKEQ_ACT parser and the fp32 reference of the int8 round trip, by hand values. No HIP.
add_executable(test_fake_quant_act_cpu test_fake_quant_act_cpu.cpp)
target_include_directories(test_fake_quant_act_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_fake_quant_act_cpu COMMAND test_fake_quant_act_cpu)

# test_pp_plan_cpu: pure CPU unit test for src/model/pp_plan.h (docs/pp-prefill.md): the KV blocks a chunk touches,
# the R4DX_PP_EMULATE parser, the two-stage pipeline simulation, the stage / hop bench CSVs and the projection with
# its go rule. No HIP call, no container; always runs.
add_executable(test_pp_plan_cpu test_pp_plan_cpu.cpp)
target_include_directories(test_pp_plan_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_pp_plan_cpu COMMAND test_pp_plan_cpu)

# test_pp_channel_cpu / test_pp_sync_cpu: pure CPU unit tests of the real pipeline's runtime (docs/pp-prefill.md
# Phase 2): the stage hand-over ring (src/model/pp_channel.h -- FIFO, back-pressure, poison, bounded waits, a
# million-item two-thread stress run, the slot payload arithmetic) and the stale-state rules of the stage-A mirror
# (src/model/pp_sync.h -- a random walk against an abstract model of both devices, the switches, the device order).
# No HIP call, no container; always run.
add_executable(test_pp_channel_cpu test_pp_channel_cpu.cpp)
target_include_directories(test_pp_channel_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_pp_channel_cpu COMMAND test_pp_channel_cpu)
add_executable(test_pp_sync_cpu test_pp_sync_cpu.cpp)
target_include_directories(test_pp_sync_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
add_test(NAME test_pp_sync_cpu COMMAND test_pp_sync_cpu)

# tool_pp_project: the CPU half of pipeline-parallel prefill Phase 0 (docs/pp-prefill.md section 8) -- the projected
# TTFT at 8k / 32k / 64k and the go rule from the two cards' stage benches and the hop bench. Built, never
# add_test()'d (it consumes measurements; tools/.../pp_phase0.ps1 runs it).
add_executable(tool_pp_project tool_pp_project.cpp)
target_include_directories(tool_pp_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)

# tool_pp_stage_bench / tool_pp_hop_bench: the GPU halves of Phase 0 -- per-stage cost of a cold prefill on one card
# (the PP-emulate mode, Model::PpChunkTimes: layer ranges, the host hop, DFlash injection / MTP priming), and the
# D2H / H2D bandwidth of the stage-boundary hop between the two cards. Built, never add_test()'d (they print / write
# numbers); the Phase 0 script runs them.
add_executable(tool_pp_stage_bench tool_pp_stage_bench.cpp)
target_link_libraries(tool_pp_stage_bench PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
add_executable(tool_pp_hop_bench tool_pp_hop_bench.cpp)
target_link_libraries(tool_pp_hop_bench PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

# tool_pp_soak: the pipeline-parallel-prefill soak (docs/pp-prefill.md Phase 2, gate G2c) -- one PpModel, random prompts of
# 2k-32k tokens, each generation run monolithically on the decode Model and pipelined, tokens compared; JSON log;
# tools/pp/soak.ps1 wraps it in the TDR watch. Built, never add_test()'d (both GPUs, minutes to an hour).
add_executable(tool_pp_soak tool_pp_soak.cpp)
target_link_libraries(tool_pp_soak PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

# test_pick_tuning: every tuning PickTuning hands out for a shape src/model/gemm_tuning_table.inc
# covers must be launchable at THIS build's kernel groups -- the table is swept at one w4a16 group
# (linear.cpp's ResolveTuning skips the rows that are not launchable at another). Host-only (no HIP call, no container), so it always runs; HIP_VISIBLE_DEVICES=1 only
# because linking r4dx_model loads the HIP runtime, which must never see device 0.
add_executable(test_pick_tuning test_pick_tuning.cpp)
target_include_directories(test_pick_tuning PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
target_link_libraries(test_pick_tuning PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
add_test(NAME test_pick_tuning COMMAND test_pick_tuning)
set_tests_properties(test_pick_tuning PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")

# test_trellis_gemma_plan (docs/gemma4-plan.md 3.7, M1-29): host-only legality of every Gemma 4 12B trellis
# linear shape (TP = 1 and a TP = 2 rank, KB 4 and 5): the 64-row tuning is legal, PlanTrellisM256 says ok
# and agrees with libr4d's own M = 256 check. No HIP call; HIP_VISIBLE_DEVICES=-1 hides every device from
# the runtime that linking r4dx_model_linear loads.
add_executable(test_trellis_gemma_plan test_trellis_gemma_plan.cpp)
target_link_libraries(test_trellis_gemma_plan PRIVATE r4dx_model_linear)
add_test(NAME test_trellis_gemma_plan COMMAND test_trellis_gemma_plan)
set_tests_properties(test_trellis_gemma_plan PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=-1")
# /EHc-: see tests/kernels/CMakeLists.txt (an extern "C" throw is otherwise uncatchable).
set_source_files_properties(test_trellis_gemma_plan.cpp PROPERTIES COMPILE_OPTIONS "/EHc-")

# test_trellis_bench_shapes: host-only check of the Gemma shape table tool_trellis_gemm_bench's
# `--group gemma` tunes (tests/kernels/trellis_bench_shapes.hpp) against the plan / tune_gemm.py shapes.
add_executable(test_trellis_bench_shapes test_trellis_bench_shapes.cpp)
target_link_libraries(test_trellis_bench_shapes PRIVATE r4dx_model_linear)
add_test(NAME test_trellis_bench_shapes COMMAND test_trellis_bench_shapes)
set_tests_properties(test_trellis_bench_shapes PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=-1")
set_source_files_properties(test_trellis_bench_shapes.cpp PROPERTIES COMPILE_OPTIONS "/EHc-")

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

# test_gemma_config (docs/gemma4-plan.md M1-1): GemmaConfig::FromJson on the real Huihui text_config
# (5:1 pattern asserted, every refusal), GemmaConfig::Shard rules, ModelConfig::arch, DetectArch on
# synthetic safetensors-shell headers. CPU-only, header + arch.cpp only.
add_executable(test_gemma_config test_gemma_config.cpp)
target_link_libraries(test_gemma_config PRIVATE r4dx_model_arch)
add_test(NAME test_gemma_config COMMAND test_gemma_config)

# test_gemma_container (docs/gemma4-plan.md M1-19): GemmaContainer::Inspect (the CPU half of the Gemma loader --
# `__metadata__` + the tensor directory, no HIP call) on a tiny 6-layer fixture converted by the real r4dx-convert
# (when built), the real Huihui bf16.r4dx header (SKIP when absent), and option-A rotation metadata. CPU only:
# HIP_VISIBLE_DEVICES is set EMPTY, so the process cannot see a GPU at all (linking r4dx_model loads the runtime).
add_executable(test_gemma_container test_gemma_container.cpp)
target_link_libraries(test_gemma_container PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
if(TARGET r4dx-convert)
  target_compile_definitions(test_gemma_container PRIVATE R4DX_CONVERT_EXE="$<TARGET_FILE:r4dx-convert>")
  add_dependencies(test_gemma_container r4dx-convert)
endif()
add_test(NAME test_gemma_container COMMAND test_gemma_container)
if(TARGET convert_gemma_trellis)
  # part (d): the rotated trellis containers convert_gemma_trellis leaves behind (SKIP when absent).
  set_tests_properties(test_gemma_container PROPERTIES ENVIRONMENT
    "HIP_VISIBLE_DEVICES=;R4DX_GEMMA_TRELLIS_KEEP=${CMAKE_BINARY_DIR}/gemma_trellis_keep" FIXTURES_REQUIRED gemma_trellis_containers)
else()
  set_tests_properties(test_gemma_container PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=")
endif()

# test_gemma_forward_smoke (docs/gemma4-plan.md M1-20 / M1-22 smoke): the whole Gemma 4 stack on the real bf16.r4dx
# against the M0-8 reference log-probs (full_logits_gemma.py's .logprobs.f16) over the first tokens of the first KL
# segment -- KL / top-1 per KV mode, DecodeStepGreedy vs argmax(DecodeStep), a sampled step, Reset. GPU test (device 1);
# SKIP 77 while the M0-8 outputs (R4DX_GEMMA_KL_REF_DIR) or the container are missing. ctest -R gemma_forward.
# tool_gemma_rot_probe: per-layer rel L2 of a rotated Gemma container vs the unrotated one on the same layer input
# (debugging tool, not a test; GPU, one device).
add_executable(tool_gemma_rot_probe tool_gemma_rot_probe.cpp)
target_link_libraries(tool_gemma_rot_probe PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)

add_executable(test_gemma_forward_smoke test_gemma_forward_smoke.cpp)
target_link_libraries(test_gemma_forward_smoke PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
target_compile_definitions(test_gemma_forward_smoke PRIVATE R4DX_SOURCE_DIR_STR="${CMAKE_SOURCE_DIR}")
add_test(NAME gemma_forward_smoke COMMAND test_gemma_forward_smoke)
set_tests_properties(gemma_forward_smoke PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1" SKIP_RETURN_CODE 77)

# test_gemma_audio (docs/gemma4-audio.md M3-2): the audio splice (GemmaModel::PrefillAudio) on a container converted with
# --audio on -- finite non-flat logits for a synthetic clip, the audio changes the output, a 30 s / 750-token clip, a
# prefill split inside the span agrees with the one-call one. GPU test (device 1); SKIP 77 while the container is missing
# (R4DX_GEMMA_AUDIO_CONTAINER). The CPU half is tests/audio. ctest -R gemma_audio.
add_executable(test_gemma_audio test_gemma_audio.cpp)
target_link_libraries(test_gemma_audio PRIVATE r4dx_model r4dx_audio r4dx_convert r4dx_core r4d_core)
add_test(NAME gemma_audio COMMAND test_gemma_audio)
set_tests_properties(gemma_audio PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1" SKIP_RETURN_CODE 77)
# test_gemma_vision_gpu (docs/gemma4-plan.md M2-3): the Gemma 4 vision path on an UNROTATED bf16 container converted with
# `r4dx-convert --vision on` (R4DX_GEMMA_VISION_CONTAINER) against tools/reference/gemma/vision_golden_gemma.py's golden
# (R4DX_GEMMA_VISION_GOLDEN): the embedder, layers 0 / 5 with an image block, and the multimodal prefill. NOT RUN by its
# author (GPU). SKIP 77 while the container or the golden is missing. ctest -R gemma_vision_gpu.
add_executable(test_gemma_vision_gpu test_gemma_vision_gpu.cpp)
target_link_libraries(test_gemma_vision_gpu PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
target_compile_definitions(test_gemma_vision_gpu PRIVATE R4DX_SOURCE_DIR_STR="${CMAKE_SOURCE_DIR}")
add_test(NAME gemma_vision_gpu COMMAND test_gemma_vision_gpu)
set_tests_properties(gemma_vision_gpu PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1" SKIP_RETURN_CODE 77)

# test_gemma_rot_trellis (docs/gemma4-plan.md M1-31): GemmaModel on the rotated trellis containers convert_gemma_trellis
# keeps (R4DX_GEMMA_TRELLIS_KEEP), against the same checkpoint unrotated bf16 (converted here by r4dx-convert): the
# loader accepts the pair, default KV is bf16, logits track the reference, fp8 KV is finite. GPU test (device 1); NOT RUN
# by its author. SKIP 77 while the kept containers are missing. ctest -R gemma_rot_trellis.
add_executable(test_gemma_rot_trellis test_gemma_rot_trellis.cpp)
target_link_libraries(test_gemma_rot_trellis PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
if(TARGET r4dx-convert)
  target_compile_definitions(test_gemma_rot_trellis PRIVATE R4DX_CONVERT_EXE="$<TARGET_FILE:r4dx-convert>")
  add_dependencies(test_gemma_rot_trellis r4dx-convert)
endif()
add_test(NAME gemma_rot_trellis COMMAND test_gemma_rot_trellis)
set_tests_properties(gemma_rot_trellis PROPERTIES SKIP_RETURN_CODE 77
  ENVIRONMENT "HIP_VISIBLE_DEVICES=1;R4DX_GEMMA_TRELLIS_KEEP=${CMAKE_BINARY_DIR}/gemma_trellis_keep")
if(TARGET convert_gemma_trellis)
  set_tests_properties(gemma_rot_trellis PROPERTIES FIXTURES_REQUIRED gemma_trellis_containers)
endif()
add_executable(test_tp_shard test_tp_shard.cpp)
target_link_libraries(test_tp_shard PRIVATE r4dx_tp_shard r4dx_convert)
add_test(NAME test_tp_shard COMMAND test_tp_shard)

# test_gemma_tp_shard (docs/gemma4-plan.md M1b-1): the Gemma 4 TP=2 sharding contract on the real 12B config -- tp::RuleFor's
# Gemma table vs GemmaConfig::Shard, the byte plans (bf16 / w4a16 / trellis) partition every linear, and a tiny-model
# algebra check that the sharded forward equals the unsharded one. CPU only, no HIP, no container.
add_executable(test_gemma_tp_shard test_gemma_tp_shard.cpp)
target_link_libraries(test_gemma_tp_shard PRIVATE r4dx_tp_shard)
add_test(NAME test_gemma_tp_shard COMMAND test_gemma_tp_shard)

# test_gemma_dflash_tp (docs/gemma4-plan.md, docs/tp.md Gemma section): Gemma DFlash under --tp 2. BOTH GPUs, LABEL tp2gpu, opt-in
# like test_tp_real_vs_emulation (SKIP 77 unless R4DX_TP2GPU=1 and two devices are visible). Never run while another job owns a GPU.
add_executable(test_gemma_dflash_tp test_gemma_dflash_tp.cpp)
target_link_libraries(test_gemma_dflash_tp PRIVATE r4dx_model r4dx_convert r4dx_core r4d_core)
target_compile_definitions(test_gemma_dflash_tp PRIVATE R4DX_SOURCE_DIR_STR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_gemma_dflash_tp COMMAND test_gemma_dflash_tp)
set_tests_properties(test_gemma_dflash_tp PROPERTIES LABELS tp2gpu SKIP_RETURN_CODE 77)

# test_gemma_vision_tp (docs/gemma4-plan.md, docs/tp.md Gemma section): Gemma vision + audio under --tp 2 vs TP=1. BOTH GPUs, LABEL
# tp2gpu, opt-in (SKIP 77 unless R4DX_TP2GPU=1 and two devices are visible). Written, not run by its author.
add_executable(test_gemma_vision_tp test_gemma_vision_tp.cpp)
target_link_libraries(test_gemma_vision_tp PRIVATE r4dx_model r4dx_audio r4dx_convert r4dx_core r4d_core)
add_test(NAME test_gemma_vision_tp COMMAND test_gemma_vision_tp)
set_tests_properties(test_gemma_vision_tp PROPERTIES LABELS tp2gpu SKIP_RETURN_CODE 77)

# test_rotation_meta: quant2's `__metadata__.rotation` parse (src/model/rotation_meta.h, header-only
# and HIP-free) and the TP slicing of the rotation.* tensors (docs/quant2.md section 3.1). CPU-only
# like the TP tests above; r4dx_tp_shard supplies src/model + third_party on the include path.
add_executable(test_rotation_meta test_rotation_meta.cpp)
target_link_libraries(test_rotation_meta PRIVATE r4dx_tp_shard)
add_test(NAME test_rotation_meta COMMAND test_rotation_meta)

# test_w4a16_group_meta: quant2 Q3's `__metadata__.quant.w4a16.groups` parse and scale-tensor names
# (src/model/w4a16_group_meta.h, header-only and HIP-free), read back from r4dx-convert's own
# W4a16GroupRules, and each group's TP=2 wsz slices (docs/quant2.md section 5.1). CPU-only, like
# test_rotation_meta; r4dx_convert supplies the converter headers and its kW4A16Group.
add_executable(test_w4a16_group_meta test_w4a16_group_meta.cpp)
target_link_libraries(test_w4a16_group_meta PRIVATE r4dx_tp_shard r4dx_convert)
add_test(NAME test_w4a16_group_meta COMMAND test_w4a16_group_meta)

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

# test_dflash_hf_weights (docs/gemma4-plan.md D-3 / D-5): the loader reads a `r4dx-convert --dflash-hf` container --
# block 16, 6 target layers, softcap / embed_scale / variant, per-layer pattern -- and defaults when they are absent. CPU-only.
add_executable(test_dflash_hf_weights test_dflash_hf_weights.cpp)
target_include_directories(test_dflash_hf_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
target_link_libraries(test_dflash_hf_weights PRIVATE r4dx_convert r4dx_core)
add_test(NAME test_dflash_hf_weights COMMAND test_dflash_hf_weights)

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
