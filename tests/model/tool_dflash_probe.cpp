// tests/model/tool_dflash_probe.cpp -- Milestone 5 stage S2 items 4 and 5 (docs/dflash2.md
// "Implementation"): the REAL-input anchor check and the isolated draft-round cost.
//
// This is the one check that catches a systematic port error. Everything in
// tests/model/test_dflash_draft.cpp runs against the Python reference's SYNTHETIC target (vocab
// 4096, seeded-random embedding/lm_head): it proves the drafter's arithmetic matches the reference
// given the same inputs, but both sides are then driven by the same made-up numbers. This tool
// closes that loop against reality -- it drives the REAL 64-layer Qwen3.8-27B target, captures the
// residual stream entering layers {6,20,34,48,62} for a real tokenized prompt, injects it, drafts,
// and dumps exactly the inputs `tools/reference/dflash2_ref.py --real` needs to redo the same round
// in fp32 numpy from the original Q8_0 GGUF. If the two drafted chains agree, every convention in
// docs/dflash2.md is right end to end on real activations.
//
// USAGE (HIP device 1, one GPU process at a time):
//   tool_dflash_probe.exe [--target <container>] [--layout w4a16]
//                          [--draft <draft container>] [--draft-layout bf16]
//                          [--tokenizer <dir>] [--out <dir>] [--rounds 50]
//                          [--bench-tokens 512] [--no-bench] [--no-dump] [--wired]
//
// --wired (review finding, 2026-09-21, item 5): re-runs ONLY the item-4 anchor check, this time
// through Model's own internal dflash_ (ModelOptions::dflash_container/dflash_draft_k,
// Model::Prefill's auto-inject, Model::DecodeStepDflashGreedy) instead of a separately-owned
// DflashDraft fed via SetDflashCaptureObserver -- i.e. against the WIRED stage-S3 driver a real
// r4dx-cli/r4dx-server generation loop actually uses, not just the stage-S2 drafter module in
// isolation. Dumps to build/logs/dflash_probe/wired_promptN; ignores --rounds/--bench-tokens/
// --no-bench/--draft-layout (the wired path reads the draft container's own layout from its
// metadata, same as Model::Load always does).
// Defaults are this project's standard real-model locations. Prints a report to stdout and writes
// one subdirectory per prompt under --out (default build/logs/dflash_probe, gitignored scratch),
// each holding `features.bin` (raw fp32 [n, 25600], C order) plus `manifest.json`.
//
// Then, per prompt:
//   <reference venv>/python.exe tools/reference/dflash2_ref.py --real <that subdirectory> \
//       --target-dir C:/AI/models/Qwen3.8-27B
//
// Built but deliberately NOT registered with add_test(): it prints numbers for a human/doc to read
// and needs the full 27B checkpoint, the DFlash2 draft container AND the tokenizer -- the same
// "built, never add_test()'d" convention tool_hseed_drift.cpp and tool_vocab_calib.cpp use.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "container.h"
#include "dflash_draft.h"
#include "model.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/stream.hpp"
#include "test_common.h"
#include "tokenizer.h"

using r4dx::core::Arena;
using r4dx::core::Bf16ToFloat;
using r4dx::core::Stream;
using r4dx::model::DflashDraft;
using r4dx::model::DflashDraftOptions;
using r4dx::model::DflashDraftResult;
using r4dx::model::DflashRoundTrace;
using r4dx::model::Layout;
using r4dx::model::LayoutFromName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx_test::FileExists;
using r4dx_test::SkipMissing;

namespace {

// Default target: the production container matching this build's w4a16 group (v6 at 64, v3 at 128
// -- tests/model/test_container_path.h). The bf16 drafter below is group-independent.
const char* kDefaultTarget = r4dx_test::ProductionTargetPath();
const char* kDefaultDraft = "D:/models/r4dx/qwen38-27b-dflash2-bf16.r4dx";
const char* kDefaultTokenizerDir = "C:/AI/models/Qwen3.8-27B";
const char* kDefaultOutDir = "build/logs/dflash_probe";

// The two short real prompts item 4 asks for: one natural-language, one code, so the captured
// residual streams are not two samples of the same distribution.
const char* kPrompts[2] = {
    "The capital of France is Paris, and the capital of Germany is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
};

std::string Arg(int argc, char** argv, const std::string& flag, const std::string& dflt) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) return argv[i + 1];
  }
  return dflt;
}
bool HasFlag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (flag == argv[i]) return true;
  }
  return false;
}

int32_t Argmax(const std::vector<float>& v) {
  int32_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[static_cast<size_t>(best)]) best = static_cast<int32_t>(i);
  }
  return best;
}

template <typename T>
void WriteBin(const std::string& path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()),
          static_cast<std::streamsize>(v.size() * sizeof(T)));
}

void WriteDump(const std::string& dir, const std::vector<float>& features, int64_t rows,
               int64_t cols, int32_t anchor_id, const std::string& prompt_text,
               const std::vector<int32_t>& token_ids, const DflashRoundTrace& tr,
               const std::vector<int32_t>& drafted) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  WriteBin(dir + "/features.bin", features);
  // The port's OWN round, beside the inputs that produced it: dflash2_ref.py's `--real` mode picks
  // these up and prints a per-tensor comparison plus the hybrid-walk attribution (see
  // _compare_with_port there). x_final_normed and gate involve no lm_head at all, so they are what
  // actually isolates the drafter; cand/unary additionally carry the target lm_head's precision.
  {
    std::vector<float> xf(tr.x_final_normed.size());
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = Bf16ToFloat(tr.x_final_normed[i]);
    WriteBin(dir + "/x_final_normed.bin", xf);
  }
  WriteBin(dir + "/gate.bin", tr.gate);
  WriteBin(dir + "/cand.bin", tr.cand);
  WriteBin(dir + "/unary.bin", tr.unary);

  std::ofstream man(dir + "/manifest.json");
  man << "{\n";
  man << "  \"schema\": \"dflash2_real_dump_v1\",\n";
  man << "  \"features_file\": \"features.bin\",\n";
  man << "  \"features_dtype\": \"float32\",\n";
  man << "  \"features_shape\": [" << rows << ", " << cols << "],\n";
  man << "  \"anchor_id\": " << anchor_id << ",\n";
  man << "  \"n_injected\": " << rows << ",\n";
  man << "  \"prompt_tokens\": [";
  for (size_t i = 0; i < token_ids.size(); ++i) man << (i ? "," : "") << token_ids[i];
  man << "],\n";
  man << "  \"drafted_tokens\": [";
  for (size_t i = 0; i < drafted.size(); ++i) man << (i ? "," : "") << drafted[i];
  man << "],\n";
  man << "  \"port_tensors\": {\"x_final_normed.bin\": [" << tr.x_final_normed.size() / 5120
      << ", 5120], \"gate.bin\": [" << tr.gate.size() / 256
      << ", 256], \"cand.bin\": [" << tr.cand.size() / 16
      << ", 16], \"unary.bin\": [" << tr.unary.size() / 16 << ", 16]},\n";
  man << "  \"prompt\": \"";
  for (char c : prompt_text) {
    if (c == '"') man << "\\\"";
    else if (c == '\\') man << "\\\\";
    else if (c == '\n') man << "\\n";
    else man << c;
  }
  man << "\"\n}\n";
  man.close();
}

// Review finding (2026-09-21, item 5): the WIRED check -- drives the anchor comparison through
// Model's own internal dflash_ (ModelOptions::dflash_container/dflash_draft_k, Model::Prefill's
// auto-inject, Model::DecodeStepDflashGreedy), exactly the path a real generation loop
// (r4dx-cli/r4dx-server) takes, rather than a separately-owned DflashDraft fed via
// SetDflashCaptureObserver. Loads exactly ONE Model (the target's own dflash_container attaches
// the drafter internally, so there is no second ~GB-scale object to keep alive alongside it) --
// deliberately does not also run the unwired item-4/item-5 sections in the same process, to avoid
// ever holding two full 27B-class Models' weights in VRAM at once.
int RunWired(const std::string& target_path, const std::string& layout_name,
             const std::string& draft_path, const std::string& tok_dir,
             const std::string& out_dir) {
  r4dx::Tokenizer::Options tok_opts;
  tok_opts.allow_unimplemented_normalizer = true;
  r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tok_dir, tok_opts);

  ModelOptions opts;
  opts.container_path = target_path;
  opts.layout = LayoutFromName(layout_name);
  opts.max_ctx = 8192;
  opts.mtp_draft_k = 0;
  opts.dflash_container = draft_path;
  opts.dflash_draft_k = 7;  // real container's block_size(8)-1 -- Model::Load validates this itself
  std::printf("[probe --wired] loading target %s (layout=%s) with dflash_container=%s\n",
              target_path.c_str(), layout_name.c_str(), draft_path.c_str());
  Model model = Model::Load(opts);
  const int64_t cols = model.DflashFeatureCols();
  std::printf("[probe --wired] target_layers = [");
  for (size_t i = 0; i < model.DflashTargetLayers().size(); ++i) {
    std::printf("%s%lld", i ? "," : "", (long long)model.DflashTargetLayers()[i]);
  }
  std::printf("], feature cols = %lld\n", (long long)cols);

  std::vector<float> captured;  // host accumulation for the dump, drained via on_chunk_captured

  int failures = 0;
  for (int p = 0; p < 2; ++p) {
    std::printf("\n=== [wired] prompt %d ===\n%s\n", p, kPrompts[p]);
    model.Reset();  // also resets the internal dflash_ ring (model.h's Reset() doc comment)
    captured.clear();

    const std::vector<int32_t> ids = tok.encode(kPrompts[p], /*parse_special=*/false);
    std::printf("[probe --wired] %zu tokens\n", ids.size());
    // Drain the model's OWN public capture buffer after every prefill chunk (model.h's
    // DflashFeatureBuffer()/DflashFeatureRows()) -- this is purely for the dump; the internal
    // dflash_ member is already injected automatically by RunChunk regardless of this callback.
    // DflashFeatureBuffer() is a DEVICE pointer (dflash_features_dev_ lives on the GPU) -- must
    // hipMemcpy it to host staging before touching it from host code, exactly the same D2H copy
    // the original SetDflashCaptureObserver-based path above does with `chunk_staging`.
    std::vector<uint16_t> chunk_staging;
    const auto on_chunk = [&] {
      const int64_t rows = model.DflashFeatureRows();
      const uint16_t* buf_dev = model.DflashFeatureBuffer();
      chunk_staging.resize(static_cast<size_t>(rows * cols));
      R4DX_HIP_CHECK(hipMemcpy(chunk_staging.data(), buf_dev,
                               chunk_staging.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
      const size_t base = captured.size();
      captured.resize(base + chunk_staging.size());
      for (size_t i = 0; i < chunk_staging.size(); ++i) {
        captured[base + i] = Bf16ToFloat(chunk_staging[i]);
      }
    };
    const std::vector<float> logits = model.Prefill(ids, on_chunk);
    const int32_t anchor = Argmax(logits);

    if (static_cast<int64_t>(captured.size()) != static_cast<int64_t>(ids.size()) * cols) {
      std::printf("FAIL: [wired] captured %zu floats, expected %lld\n", captured.size(),
                  (long long)(static_cast<int64_t>(ids.size()) * cols));
      ++failures;
      continue;
    }
    std::printf("[probe --wired] anchor_id = %d (%s), n_injected = %zu\n", anchor,
                tok.decode({anchor}).c_str(), ids.size());

    DflashRoundTrace tr;
    std::vector<int32_t> drafted_tokens;
    const std::vector<int32_t> round =
        model.DecodeStepDflashGreedy(anchor, /*k=*/7, /*p_min=*/0.0f, /*n_min=*/0,
                                      /*walk_len_out=*/nullptr, &tr, &drafted_tokens);
    std::printf("[probe --wired] DraftRound's raw chain (pre-verify): [");
    for (size_t i = 0; i < drafted_tokens.size(); ++i) {
      std::printf("%s%d", i ? "," : "", drafted_tokens[i]);
    }
    std::printf("]\n[probe --wired] post-verify committed round: [");
    for (size_t i = 0; i < round.size(); ++i) std::printf("%s%d", i ? "," : "", round[i]);
    std::printf("]\n[probe --wired] drafted as text: \"%s\"\n", tok.decode(drafted_tokens).c_str());

    const std::string dir = out_dir + "/wired_prompt" + std::to_string(p);
    WriteDump(dir, captured, static_cast<int64_t>(ids.size()), cols, anchor, kPrompts[p], ids, tr,
              drafted_tokens);
    std::printf("[probe --wired] wrote %s/{features.bin,manifest.json} (%.1f MB)\n", dir.c_str(),
                static_cast<double>(captured.size() * sizeof(float)) / (1024.0 * 1024.0));
    std::printf("[probe --wired] reference: python tools/reference/dflash2_ref.py --real %s "
                "--target-dir %s\n", dir.c_str(), tok_dir.c_str());
  }
  return failures == 0 ? 0 : 1;
}

struct Timing {
  double mean_ms = 0, min_ms = 0, max_ms = 0;
};

Timing Summarize(std::vector<double> v) {
  Timing t;
  if (v.empty()) return t;
  std::sort(v.begin(), v.end());
  t.min_ms = v.front();
  t.max_ms = v.back();
  double s = 0;
  for (double x : v) s += x;
  t.mean_ms = s / static_cast<double>(v.size());
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  const std::string target_path = Arg(argc, argv, "--target", kDefaultTarget);
  const std::string layout_name = Arg(argc, argv, "--layout", "w4a16");
  const std::string draft_path = Arg(argc, argv, "--draft", kDefaultDraft);
  const std::string draft_layout_name = Arg(argc, argv, "--draft-layout", "bf16");
  const std::string tok_dir = Arg(argc, argv, "--tokenizer", kDefaultTokenizerDir);
  const std::string out_dir = Arg(argc, argv, "--out", kDefaultOutDir);
  const int rounds = std::stoi(Arg(argc, argv, "--rounds", "50"));
  const int bench_tokens = std::stoi(Arg(argc, argv, "--bench-tokens", "512"));
  const bool do_bench = !HasFlag(argc, argv, "--no-bench");
  const bool do_dump = !HasFlag(argc, argv, "--no-dump");
  // Review finding (2026-09-21, item 5): everything above drives a SEPARATELY-owned DflashDraft via
  // SetDflashCaptureObserver (the external-observer path a caller that manages its own drafter
  // object uses). `--wired` instead drives the item-4 anchor check through the SAME path a real
  // generation loop takes -- ModelOptions::dflash_container/dflash_draft_k, Model's own internal
  // dflash_, Model::Prefill's auto-inject and Model::DecodeStepDflashGreedy -- so the numbers this
  // prints are a check of the WIRED stage-S3 driver, not just the stage-S2 drafter module in
  // isolation. Dumps to build/logs/dflash_probe/wired_promptN (a different directory from the
  // unwired mode's promptN, so both can be compared side by side against the Python reference).
  const bool wired = HasFlag(argc, argv, "--wired");

  if (!FileExists(target_path)) return SkipMissing(target_path);
  if (!FileExists(draft_path)) return SkipMissing(draft_path);
  if (!FileExists(tok_dir + "/tokenizer.json")) return SkipMissing(tok_dir + "/tokenizer.json");

  if (wired) return RunWired(target_path, layout_name, draft_path, tok_dir, out_dir);

  r4dx::Tokenizer::Options tok_opts;
  tok_opts.allow_unimplemented_normalizer = true;  // same escape hatch src/cli/main.cpp uses
  r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(tok_dir, tok_opts);

  ModelOptions opts;
  opts.container_path = target_path;
  opts.layout = LayoutFromName(layout_name);
  opts.max_ctx = 8192;   // this tool never generates past a few hundred positions; keep KV small so
                          // the target, the drafter and its ring all fit alongside each other
  opts.mtp_draft_k = 0;  // DFlash2 needs no MTP head at all -- see Model::VerifyWindow's own comment
  opts.dflash_draft_k = 7;
  std::printf("[probe] loading target %s (layout=%s)\n", target_path.c_str(), layout_name.c_str());
  Model model = Model::Load(opts);

  DflashDraftOptions dopts;
  dopts.container_path = draft_path;
  dopts.layout = LayoutFromName(draft_layout_name);
  std::printf("[probe] loading drafter %s (layout=%s)\n", draft_path.c_str(),
              draft_layout_name.c_str());
  DflashDraft draft = DflashDraft::Load(dopts);

  const std::vector<int64_t> target_layers = draft.Config().target_layers;
  std::printf("[probe] target_layers = [");
  for (size_t i = 0; i < target_layers.size(); ++i) std::printf("%s%lld", i ? "," : "",
                                                                (long long)target_layers[i]);
  std::printf("], feature cols = %lld, block_size = %lld, mask_id = %lld\n",
              (long long)draft.FeatureCols(), (long long)draft.BlockSize(),
              (long long)draft.MaskTokenId());
  model.AttachDflashFeatureCapture(target_layers);

  Stream dstream;
  Arena darena;
  darena.Reserve(64ull * 1024 * 1024);
  const auto embed = r4dx::model::MakeTargetEmbeddingProvider(model.GetContainer());
  const auto lm_head = r4dx::model::MakeTargetLmHeadProvider(model.GetContainer());

  const int64_t cols = draft.FeatureCols();
  std::vector<float> captured;          // host accumulation for the dump, [rows, cols] fp32
  std::vector<uint16_t> chunk_staging;  // one chunk's bf16 rows
  bool collect = true;

  // The per-RunChunk capture drain (model.h's SetDflashCaptureObserver). Two jobs: keep the
  // drafter's KV ring in lockstep with the target's own committed positions, and (when collecting)
  // accumulate the same rows on the host so they can be handed to the Python reference verbatim.
  model.SetDflashCaptureObserver([&](const uint16_t* feat, int64_t rows, int64_t start_pos) {
    draft.InjectFeatures(dstream, darena, feat, rows, start_pos);
    dstream.Synchronize();  // the NEXT RunChunk overwrites `feat` at row 0, so the injection must
                            // have actually consumed it before this callback returns
    darena.Reset();
    if (!collect) return;
    chunk_staging.resize(static_cast<size_t>(rows * cols));
    R4DX_HIP_CHECK(hipMemcpy(chunk_staging.data(), feat,
                             chunk_staging.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const size_t base = captured.size();
    captured.resize(base + chunk_staging.size());
    for (size_t i = 0; i < chunk_staging.size(); ++i) {
      captured[base + i] = Bf16ToFloat(chunk_staging[i]);
    }
  });

  // ---- item 4: the real-input anchor check -------------------------------------------------------
  for (int p = 0; p < 2; ++p) {
    std::printf("\n=== prompt %d ===\n%s\n", p, kPrompts[p]);
    model.Reset();
    draft.Reset();
    captured.clear();

    const std::vector<int32_t> ids = tok.encode(kPrompts[p], /*parse_special=*/false);
    std::printf("[probe] %zu tokens\n", ids.size());
    const std::vector<float> logits = model.Prefill(ids);
    const int32_t anchor = Argmax(logits);

    if (static_cast<int64_t>(captured.size()) != static_cast<int64_t>(ids.size()) * cols) {
      std::printf("FAIL: captured %zu floats, expected %lld -- the per-chunk observer dropped rows\n",
                  captured.size(), (long long)(static_cast<int64_t>(ids.size()) * cols));
      return 1;
    }
    if (draft.InjectedCount() != static_cast<int64_t>(ids.size())) {
      std::printf("FAIL: drafter injected %lld positions for a %zu-token prompt\n",
                  (long long)draft.InjectedCount(), ids.size());
      return 1;
    }
    std::printf("[probe] anchor_id = %d (%s), n_injected = %lld\n", anchor,
                tok.decode({anchor}).c_str(), (long long)draft.InjectedCount());

    DflashRoundTrace tr;
    const DflashDraftResult res =
        draft.DraftRound(dstream, darena, anchor, draft.BlockSize() - 1, /*p_min=*/0.0f,
                         /*n_min=*/0, embed, lm_head, &tr);
    darena.Reset();
    std::printf("[probe] drafted %zu tokens: [", res.tokens.size());
    for (size_t i = 0; i < res.tokens.size(); ++i) std::printf("%s%d", i ? "," : "", res.tokens[i]);
    std::printf("]\n[probe] as text: \"%s\"\n", tok.decode(res.tokens).c_str());
    for (int64_t t = 0; t < draft.BlockSize(); ++t) {
      std::printf("  block pos %lld: top-16 = [", (long long)t);
      for (int64_t j = 0; j < 16; ++j) {
        std::printf("%s%d", j ? ", " : "", tr.cand[static_cast<size_t>(t * 16 + j)]);
      }
      std::printf("]\n");
    }

    if (do_dump) {
      const std::string dir = out_dir + "/prompt" + std::to_string(p);
      WriteDump(dir, captured, static_cast<int64_t>(ids.size()), cols, anchor, kPrompts[p], ids, tr,
                res.tokens);
      std::printf("[probe] wrote %s/{features.bin,manifest.json} (%.1f MB)\n", dir.c_str(),
                  static_cast<double>(captured.size() * sizeof(float)) / (1024.0 * 1024.0));
      std::printf("[probe] reference: python tools/reference/dflash2_ref.py --real %s "
                  "--target-dir %s\n", dir.c_str(), tok_dir.c_str());
    }
  }

  // ---- item 5: isolated cost ----------------------------------------------------------------------
  if (do_bench) {
    std::printf("\n=== isolated cost (draft container %s, layout %s) ===\n", draft_path.c_str(),
                draft_layout_name.c_str());
    collect = false;
    captured.clear();
    model.Reset();
    draft.Reset();
    // A realistic frontier: bench a round whose attention reads a few hundred injected positions,
    // not the ~15 a one-line prompt leaves behind.
    std::string long_text;
    while (static_cast<int>(tok.encode(long_text, false).size()) < bench_tokens) {
      long_text += "The quick brown fox jumps over the lazy dog near the river bank. ";
    }
    std::vector<int32_t> long_ids = tok.encode(long_text, /*parse_special=*/false);
    if (static_cast<int>(long_ids.size()) > bench_tokens) long_ids.resize(bench_tokens);
    const std::vector<float> lg = model.Prefill(long_ids);
    const int32_t anchor = Argmax(lg);
    std::printf("[bench] n_injected = %lld, anchor = %d\n", (long long)draft.InjectedCount(),
                anchor);

    hipEvent_t ev0, ev1;
    R4DX_HIP_CHECK(hipEventCreate(&ev0));
    R4DX_HIP_CHECK(hipEventCreate(&ev1));

    // Empty-stream synchronize cost, so the wall-vs-device gap below can be attributed rather than
    // guessed at (a Windows scheduler-wait sync with 10 ms timer granularity would show up here).
    {
      std::vector<double> sync_ms;
      for (int i = 0; i < 50; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        dstream.Synchronize();
        sync_ms.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      const Timing s = Summarize(sync_ms);
      std::printf("[bench] empty-stream Synchronize(): mean %.4f ms (max %.4f)\n", s.mean_ms,
                  s.max_ms);
    }

    for (int i = 0; i < 5; ++i) {  // warm up: GEMM tuning lookup, first-touch of every weight page
      (void)draft.DraftRound(dstream, darena, anchor, draft.BlockSize() - 1, 0.0f, 0, embed,
                             lm_head, nullptr);
      darena.Reset();
    }
    std::vector<double> gpu_ms, wall_ms;
    for (int i = 0; i < rounds; ++i) {
      double dev = 0.0;
      const auto t0 = std::chrono::steady_clock::now();
      (void)draft.DraftRound(dstream, darena, anchor, draft.BlockSize() - 1, 0.0f, 0, embed,
                             lm_head, nullptr, &dev);
      const auto t1 = std::chrono::steady_clock::now();
      gpu_ms.push_back(dev);
      wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      darena.Reset();
    }
    const Timing g = Summarize(gpu_ms), w = Summarize(wall_ms);
    std::printf("[bench] DraftRound over %d rounds: device   mean %.3f ms (min %.3f, max %.3f)\n",
                rounds, g.mean_ms, g.min_ms, g.max_ms);
    std::printf("[bench] DraftRound over %d rounds: wall     mean %.3f ms (min %.3f, max %.3f)"
                "   [wall - device = host selector walk + launch/sync overhead: %.3f ms]\n",
                rounds, w.mean_ms, w.min_ms, w.max_ms, w.mean_ms - g.mean_ms);

    // A 64-row InjectFeatures -- the widest a prefill-chunk drain ever hands over. Timed LAST
    // because each call advances the ring's frontier (injection is append-only by contract).
    {
      const int64_t rows = 64;
      r4dx::core::DeviceBuffer<uint16_t> feat(static_cast<size_t>(rows * cols));
      feat.Zero();
      for (int i = 0; i < 3; ++i) {
        draft.InjectFeatures(dstream, darena, feat.data(), rows, draft.InjectedCount());
        dstream.Synchronize();
        darena.Reset();
      }
      std::vector<double> inj_ms, inj_wall_ms;
      for (int i = 0; i < 20; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        R4DX_HIP_CHECK(hipEventRecord(ev0, dstream.get()));
        draft.InjectFeatures(dstream, darena, feat.data(), rows, draft.InjectedCount());
        R4DX_HIP_CHECK(hipEventRecord(ev1, dstream.get()));
        R4DX_HIP_CHECK(hipEventSynchronize(ev1));
        const auto t1 = std::chrono::steady_clock::now();
        float ms = 0;
        R4DX_HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        inj_ms.push_back(ms);
        inj_wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        darena.Reset();
      }
      const Timing t = Summarize(inj_ms), tw = Summarize(inj_wall_ms);
      // InjectFeatures does NOT synchronize internally, so here the outer event pair is valid --
      // and device vs wall agreeing is the cross-check that the instrumentation itself is sound.
      std::printf("[bench] InjectFeatures(64 rows): device mean %.3f ms (min %.3f, max %.3f), "
                  "wall mean %.3f ms\n", t.mean_ms, t.min_ms, t.max_ms, tw.mean_ms);
    }
    R4DX_HIP_CHECK(hipEventDestroy(ev0));
    R4DX_HIP_CHECK(hipEventDestroy(ev1));

    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) == hipSuccess) {
      std::printf("[bench] VRAM in use: %.2f GiB of %.2f GiB\n",
                  static_cast<double>(total_b - free_b) / (1024.0 * 1024.0 * 1024.0),
                  static_cast<double>(total_b) / (1024.0 * 1024.0 * 1024.0));
    }
  }

  model.ClearDflashCaptureObserver();
  std::printf("\n[probe] done\n");
  return 0;
}
