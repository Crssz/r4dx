// tests/model/tool_teacher_forced_logprobs.cpp -- the r4dx half of Rung 4 (docs/validation.md
// "Rung 4 tooling"): a teacher-forced dump of the engine's per-position next-token log-probabilities
// for a FIXED token sequence, in the shared on-disk format the reference (HF transformers) half
// writes too, so a KL report can pair them row for row.
//
// Built by tests/model/CMakeLists.txt but deliberately NOT registered with add_test() -- the same
// convention tool_hseed_drift / tool_vocab_calib / tool_dflash_probe / tests/vision's
// tool_vision_chat follow: it needs a real container and a tokens file that are not vendored into
// the repo, and it produces artifacts for a human/another script rather than asserting a contract.
// The contract that IS asserted automatically lives in tests/model/test_teacher_forced_logprobs.cpp
// (ctest `test_teacher_forced_logprobs`), which drives the same pass (tests/model/teacher_forced.h)
// against the 4-layer container.
//
//   $env:HIP_VISIBLE_DEVICES='1'
//   build\win-hip\tests\model\tool_teacher_forced_logprobs.exe `
//       --model D:/models/r4dx/qwen38-27b-v6.r4dx --layout w4a16 `
//       --tokens tokens.json --out-dir logprobs_out
//
// Options:
//   --model <path>        r4dx container (required; must match this build's w4a16 group --
//                         qwen38-27b-v6.r4dx on the default group-64 build, qwen38-27b-v3.r4dx on
//                         win-hip-g128 -- or --layout w4a16 is refused at load)
//   --layout <name>       body layout: bf16 | w4a16 | w4a8 | mxfp4   (default w4a16)
//   --tokens <path>       tokens.json in the shared format (required):
//                           {"tokenizer": "...", "segments": [{"name": "...", "token_ids": [...]}]}
//                         Produce it with tools/reference/make_tokens_json.py.
//   --out-dir <dir>       where the .logprobs.f16 / .meta.json pair per segment goes (required
//                         unless --no-write). The directory must already exist.
//   --segment <name>      evaluate only this segment (default: every segment in the file)
//   --max-ctx N           KV cache capacity in tokens (default 8192 -- a Rung 4 sequence is short
//                         and the full 262144 default would reserve 8+ GiB for nothing)
//   --layers N            load only the first N layers; for the 4-layer test containers, whose
//                         config.json still declares the full 64 (default: every layer)
//   --vision {auto|on|off}  default OFF: the tower plays no part in a text-only log-prob dump and
//                         skipping it reclaims ~0.89 GiB (docs/vision.md "Load policy")
//   --check-greedy N      the LAST N tokens of each segment were produced by greedy decoding on
//                         this same container+layout, so assert argmax(row i) == token_ids[i+1] for
//                         every row that predicts one of them. Nonzero mismatches make the tool
//                         exit 1.
//   --no-write            run the whole pass and print the checks, write nothing
//   --quiet               no per-row progress lines
//   --embed-device-resident {on|off}   text.embed_tokens VRAM mirror (default on, like the CLI)
//   --tp {1|2}            tensor parallel (docs/tp.md 9.1, 10.4): 1 (default) is the single-device
//                         Model; 2 runs the pass through r4dx::model::TpModel
//   --tp-mode {real|emulate|noop}   with --tp 2 (default real; real arrives in docs/tp.md P4)
//   --tp-devices a[,b]    with --tp 2: process-visible HIP ordinals (default auto, docs/tp.md 9.2)
//   --tp-rank r           with --tp-mode noop: which shard to load
//
// MTP and DFlash2 are unconditionally off (ModelOptions::mtp_draft_k stays 0, dflash_container
// stays empty): both are speculation strategies for GENERATING, and this tool never generates -- it
// scores a fixed sequence. Leaving them off also keeps Model's draft window at 1, i.e. exactly the
// pre-speculation GDN state layout and decode path.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <map>
#include <memory>
#include <sstream>

#include "model.h"
#include "teacher_forced.h"
#include "test_common.h"
#include "text_model.h"

using r4dx_test::FileExists;
using r4dx_test::SkipMissing;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

double VramUsedGiB() {
  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return -1.0;
  return static_cast<double>(total_b - free_b) / (1024.0 * 1024.0 * 1024.0);
}

// The model's VRAM use summed over its ranks' DEVICES (docs/tp.md 2.8: `Vram()` for the [vram]
// line). At TP=1 that is one hipMemGetInfo on this thread, i.e. exactly VramUsedGiB(); under
// --tp-mode emulate both ranks report the same device, which is counted once.
double ModelVramGiB(const r4dx::model::TextModel& m) {
  std::map<int, double> per_device;
  for (const r4dx::model::VramReport& r : m.Vram()) per_device[r.device] = r.used_gib;
  double sum = 0.0;
  for (const auto& kv : per_device) sum += kv.second;
  return sum;
}

std::vector<int> ParseDevices(const std::string& s) {
  std::vector<int> d;
  if (s.empty() || s == "auto") return d;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) d.push_back(std::stoi(item));
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_path, out_dir, only_segment;
  std::string layout = "w4a16";
  std::string vision = "off";
  std::string embed_resident = "on";
  std::string tp_mode = "real", tp_devices = "auto";
  int tp = 1, tp_rank = 0;
  bool tp_options_given = false;
  int64_t max_ctx = 8192, layers = -1, check_greedy = 0;
  bool no_write = false, quiet = false;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return std::string(argv[++i]);
      };
      if (a == "--model") model_path = next();
      else if (a == "--layout") layout = next();
      else if (a == "--tokens") tokens_path = next();
      else if (a == "--out-dir") out_dir = next();
      else if (a == "--segment") only_segment = next();
      else if (a == "--max-ctx") max_ctx = std::stoll(next());
      else if (a == "--layers") layers = std::stoll(next());
      else if (a == "--vision") vision = next();
      else if (a == "--check-greedy") check_greedy = std::stoll(next());
      else if (a == "--no-write") no_write = true;
      else if (a == "--quiet") quiet = true;
      else if (a == "--embed-device-resident") embed_resident = next();
      else if (a == "--tp") tp = std::stoi(next());
      else if (a == "--tp-mode") { tp_mode = next(); tp_options_given = true; }
      else if (a == "--tp-devices") { tp_devices = next(); tp_options_given = true; }
      else if (a == "--tp-rank") { tp_rank = std::stoi(next()); tp_options_given = true; }
      else {
        std::fprintf(stderr, "unrecognized argument: %s\n", a.c_str());
        return 2;
      }
    }
    if (model_path.empty() || tokens_path.empty()) {
      std::fprintf(stderr, "usage: tool_teacher_forced_logprobs --model <container.r4dx> "
                            "--layout w4a16 --tokens <tokens.json> --out-dir <dir> "
                            "[--segment <name>] [--max-ctx N] [--layers N] [--vision off] "
                            "[--check-greedy N] [--no-write] [--quiet] "
                            "[--embed-device-resident {on|off}] [--tp {1|2}] "
                            "[--tp-mode {real|emulate|noop}] [--tp-devices a[,b]] [--tp-rank r]\n");
      return 2;
    }
    if (embed_resident != "on" && embed_resident != "off") {
      std::fprintf(stderr, "--embed-device-resident must be 'on' or 'off'\n");
      return 2;
    }
    if (tp != 1 && tp != 2) {
      std::fprintf(stderr, "--tp must be 1 or 2\n");
      return 2;
    }
    if (tp == 1 && tp_options_given) {
      std::fprintf(stderr, "--tp-mode/--tp-devices/--tp-rank need --tp 2\n");
      return 2;
    }
    if (tp_mode != "real" && tp_mode != "emulate" && tp_mode != "noop") {
      std::fprintf(stderr, "--tp-mode must be 'real', 'emulate' or 'noop'\n");
      return 2;
    }
    if (out_dir.empty() && !no_write) {
      std::fprintf(stderr, "--out-dir is required (or pass --no-write)\n");
      return 2;
    }
    if (!FileExists(model_path)) return SkipMissing(model_path);
    if (!FileExists(tokens_path)) return SkipMissing(tokens_path);

    const r4dx_tf::TokensFile tf = r4dx_tf::ReadTokensJson(tokens_path);
    std::printf("[tokens] %s: tokenizer=%s, %zu segment(s)\n", tokens_path.c_str(),
                tf.tokenizer.c_str(), tf.segments.size());

    ModelOptions opts;
    opts.container_path = model_path;
    opts.layout = r4dx::model::LayoutFromName(layout);
    opts.max_ctx = max_ctx;
    opts.layer_limit = layers;
    opts.mtp_draft_k = 0;      // see this file's header comment
    opts.dflash_draft_k = 0;
    opts.vision = vision == "on"     ? ModelOptions::VisionMode::kOn
                  : vision == "auto" ? ModelOptions::VisionMode::kAuto
                                     : ModelOptions::VisionMode::kOff;
    opts.embed_device_resident = embed_resident != "off";

    r4dx::model::TpOptions tpo;
    tpo.world = tp;
    if (tp == 2) {
      tpo.mode = tp_mode == "emulate" ? r4dx::model::TpOptions::Mode::kEmulate
                 : tp_mode == "noop"  ? r4dx::model::TpOptions::Mode::kNoop
                                      : r4dx::model::TpOptions::Mode::kReal;
      tpo.devices = ParseDevices(tp_devices);
      tpo.noop_rank = tp_rank;
    }

    // Under --tp 2 this thread is the TpModel facade and makes no HIP call (docs/tp.md 2.1).
    const double vram_before = tp == 1 ? VramUsedGiB() : 0.0;
    std::unique_ptr<r4dx::model::TextModel> model_ptr = r4dx::model::LoadTextModel(opts, tpo);
    r4dx::model::TextModel& model = *model_ptr;
    const double vram_after_load = ModelVramGiB(model);
    const int64_t vocab = model.Config().vocab_size;
    std::printf("[model] %s layout=%s layers=%lld/%lld vocab=%lld max_ctx=%lld%s\n",
                model_path.c_str(), layout.c_str(),
                static_cast<long long>(model.NumLoadedLayers()),
                static_cast<long long>(model.Config().num_hidden_layers),
                static_cast<long long>(vocab), static_cast<long long>(max_ctx),
                tp == 2 ? (" tp=2 mode=" + tp_mode).c_str() : "");
    if (tp == 1) {
      std::printf("[vram]  %.2f GiB after load (delta %.2f GiB)\n", vram_after_load,
                  vram_after_load - vram_before);
    } else {
      std::printf("[vram]  %.2f GiB after load (summed over the ranks' devices)\n", vram_after_load);
    }

    int64_t total_rows = 0, total_mismatches = 0, segments_run = 0;
    double total_wall = 0.0, worst_lse = 0.0, peak_vram = vram_after_load;

    for (const r4dx_tf::Segment& seg : tf.segments) {
      if (!only_segment.empty() && seg.name != only_segment) continue;
      ++segments_run;
      r4dx_tf::SegmentOptions so;
      so.out_dir = no_write ? std::string() : out_dir;
      so.layout_name = layout;
      so.check_greedy_last_n = check_greedy;
      so.progress_every = quiet ? 0 : 64;
      so.container_path = model_path;
      std::printf("[segment] %s: T=%zu tokens -> %zu rows x %lld vocab\n", seg.name.c_str(),
                  seg.token_ids.size(), seg.token_ids.size() - 1, static_cast<long long>(vocab));
      const r4dx_tf::SegmentResult r = r4dx_tf::RunSegment(model, seg, so);
      peak_vram = std::max(peak_vram, ModelVramGiB(model));
      total_rows += r.rows;
      total_wall += r.wall_s;
      total_mismatches += r.greedy_mismatches;
      worst_lse = std::max(worst_lse, r.max_abs_lse);
      std::printf("[segment] %s: %lld rows in %.2fs (%.1f ms/row), max |logsumexp(row)| = %.3e, "
                  "sha256(token_ids)=%s\n",
                  seg.name.c_str(), static_cast<long long>(r.rows), r.wall_s,
                  1000.0 * r.wall_s / static_cast<double>(std::max<int64_t>(1, r.rows)),
                  r.max_abs_lse, r.sha256.c_str());
      if (check_greedy > 0) {
        std::printf("[greedy]  %s: %lld/%lld rows argmax == next token%s\n", seg.name.c_str(),
                    static_cast<long long>(r.greedy_checked - r.greedy_mismatches),
                    static_cast<long long>(r.greedy_checked),
                    r.greedy_mismatches == 0 ? "" : "  <-- MISMATCH");
        if (r.greedy_mismatches > 0) {
          std::printf("[greedy]  %s: first mismatch at row %lld: argmax=%d, token_ids[%lld]=%d\n",
                      seg.name.c_str(), static_cast<long long>(r.first_mismatch_row),
                      r.first_mismatch_got, static_cast<long long>(r.first_mismatch_row + 1),
                      r.first_mismatch_want);
        }
      }
      if (!no_write) {
        std::printf("[write]   %s/%s.logprobs.f16 (%lld x %lld f16 = %.1f MiB) + %s.meta.json\n",
                    out_dir.c_str(), seg.name.c_str(), static_cast<long long>(r.rows),
                    static_cast<long long>(r.V),
                    static_cast<double>(r.rows) * static_cast<double>(r.V) * 2.0 / (1024.0 * 1024.0),
                    seg.name.c_str());
      }
    }

    if (segments_run == 0) {
      std::fprintf(stderr, "no segment matched --segment %s\n", only_segment.c_str());
      return 2;
    }
    std::printf("[total]  %lld segment(s), %lld rows, %.2fs wall (%.1f ms/row), "
                "max |logsumexp| %.3e, peak VRAM %.2f GiB\n",
                static_cast<long long>(segments_run), static_cast<long long>(total_rows),
                total_wall, 1000.0 * total_wall / static_cast<double>(std::max<int64_t>(1, total_rows)),
                worst_lse, peak_vram);

    // The pass's own arithmetic invariant, always checked (it costs one extra reduction per row and
    // is the cheapest possible evidence that the dump is a log-probability distribution at all).
    if (!(worst_lse < 1e-2)) {
      std::fprintf(stderr, "FAIL: max |logsumexp(row)| = %.3e is not < 1e-2\n", worst_lse);
      return 1;
    }
    if (check_greedy > 0 && total_mismatches > 0) {
      std::fprintf(stderr, "FAIL: %lld greedy-consistency mismatch(es)\n",
                   static_cast<long long>(total_mismatches));
      return 1;
    }
    std::printf("[PASS] logsumexp invariant holds%s\n",
                check_greedy > 0 ? " and every checked row's argmax is the next token" : "");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_teacher_forced_logprobs: %s\n", e.what());
    return 1;
  }
  return 0;
}
