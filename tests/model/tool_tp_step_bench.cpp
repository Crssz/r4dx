// tests/model/tool_tp_step_bench.cpp -- the per-rank decode step of tensor parallelism, measured
// before any threading exists (docs/tp.md P2a, gate G3).
//
// `--rank r` loads ONE rank's shard of the container (ModelOptions::tp = {world 2, rank r} with a
// tp::MakeNoopComm endpoint -- no peer, and NO kernel at the three all-reduce sites, the baseline
// tp_bench's L_vs_no_ar_kernel is defined against, docs/tp.md 1.4) onto this process's device, and
// times DecodeStepGreedy. `--tp1` times the ordinary TP=1 Model the same way, so G3's ratio compares
// two numbers from the same tool, prompt and protocol. The noop rank's tokens are meaningless (half
// of every row-parallel sum is missing), so every run -- TP=1 included -- feeds the same fixed token
// cycle taken from the prompt instead of its own output: the work per step is identical and garbage
// output can neither change it nor hit EOS.
//
// Protocol: the standard prompt (docs/perf.md; --prompt overrides) rendered through the real chat
// template with thinking off, prefilled; one untimed warm-up pass (lazy module loads, PickTuning's
// cache, first-touch page faults); then --repeats timed passes, each Reset() + Prefill + --tokens
// DecodeStepGreedy calls, reported per pass and as the median ms/token.
//
// Pre-flight (docs/tp.md 9.2): refuses to start (exit 1) unless the device has --need-gib free --
// the production server holds ~28 GiB of device 1.
//
// Built, never add_test()'d (prints numbers for G3, asserts nothing), like tool_sampled_bench.
// Usage (HIP_VISIBLE_DEVICES=1):
//   tool_tp_step_bench --model D:\models\r4dx\qwen38-27b-v6.r4dx --layout w4a16 --max-ctx 2048
//                      (--tp1 | --rank 0|1) [--tokens 128] [--repeats 3] [--json out.json]
//                      [--tokenizer-dir C:\AI\models\Qwen3.8-27B] [--prompt "..."] [--need-gib X]
//                      [--layers N]   (layer_limit, for the 4-layer test containers; G3 omits it)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat_template.h"
#include "model.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/error.hpp"
#include "test_common.h"
#include "tokenizer.h"
#include "tp/tp_comm_noop.h"

using r4dx::model::LayoutFromName;
using r4dx::model::LayoutName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

constexpr const char* kStandardPrompt =
    "Write a haiku about GPUs, then explain what a GPU is in two sentences.";

struct Args {
  std::string model = r4dx_test::ProductionTargetPath();
  std::string layout = "w4a16";
  std::string tokenizer_dir = "C:/AI/models/Qwen3.8-27B";
  std::string prompt = kStandardPrompt;
  std::string json;
  int64_t max_ctx = 2048;
  int64_t layers = -1;  // ModelOptions::layer_limit: -1 = every layer (G3); 4 for the l4 containers
  bool tp1 = false;
  int rank = -1;
  int tokens = 128;
  int repeats = 3;
  int warmup = 16;
  double need_gib = -1.0;  // default: 17 GiB for --tp1, 11 GiB for one rank's shard
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "tool_tp_step_bench: %s\nusage: tool_tp_step_bench --model <c.r4dx> [--layout "
               "w4a16] [--max-ctx 2048] (--tp1 | --rank 0|1) [--tokens 128] [--repeats 3] "
               "[--warmup 16] [--json out.json] [--tokenizer-dir <dir>] [--prompt <text>] "
               "[--need-gib X] [--layers N]\n",
               why.c_str());
  std::exit(2);
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(s + " needs a value");
      return argv[++i];
    };
    if (s == "--model") a.model = next();
    else if (s == "--layout") a.layout = next();
    else if (s == "--tokenizer-dir") a.tokenizer_dir = next();
    else if (s == "--prompt") a.prompt = next();
    else if (s == "--json") a.json = next();
    else if (s == "--max-ctx") a.max_ctx = std::stoll(next());
    else if (s == "--layers") a.layers = std::stoll(next());
    else if (s == "--tp1") a.tp1 = true;
    else if (s == "--rank") a.rank = std::stoi(next());
    else if (s == "--tokens") a.tokens = std::stoi(next());
    else if (s == "--repeats") a.repeats = std::stoi(next());
    else if (s == "--warmup") a.warmup = std::stoi(next());
    else if (s == "--need-gib") a.need_gib = std::stod(next());
    else Usage("unknown argument " + s);
  }
  if (a.tp1 == (a.rank >= 0)) Usage("pass exactly one of --tp1 and --rank r");
  if (a.rank > 1) Usage("--rank must be 0 or 1 (TP=2)");
  if (a.tokens < 1 || a.repeats < 1 || a.warmup < 0) Usage("--tokens/--repeats must be >= 1");
  if (a.need_gib < 0) a.need_gib = a.tp1 ? 17.0 : 11.0;
  return a;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// One pass: fresh sequence, prefill, then `tokens` greedy steps fed from the fixed cycle.
double RunPass(Model& m, const std::vector<int32_t>& prompt, int tokens) {
  m.Reset();
  m.Prefill(prompt);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < tokens; ++i) {
    (void)m.DecodeStepGreedy(prompt[static_cast<size_t>(i) % prompt.size()]);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / tokens;
}

int Run(const Args& a) {
  if (!r4dx_test::FileExists(a.model)) {
    std::fprintf(stderr, "tool_tp_step_bench: container not found: %s\n", a.model.c_str());
    return 77;
  }
  if (!r4dx_test::FileExists(a.tokenizer_dir + "/tokenizer.json")) {
    std::fprintf(stderr, "tool_tp_step_bench: tokenizer not found in %s\n",
                 a.tokenizer_dir.c_str());
    return 77;
  }

  // Pre-flight (docs/tp.md 9.2): this process's device must have room for the run.
  int device = 0;
  R4DX_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop{};
  R4DX_HIP_CHECK(hipGetDeviceProperties(&prop, device));
  size_t free_b = 0, total_b = 0;
  R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
  if (free_gib < a.need_gib) {
    std::fprintf(stderr,
                 "need %.1f GiB free on HIP device %d, have %.2f GiB -- is the production server "
                 "running?\n",
                 a.need_gib, device, free_gib);
    return 1;
  }

  // The standard prompt through the real chat template, thinking off (r4dx-cli --think off).
  r4dx::Tokenizer::Options tok_opts;
  tok_opts.allow_unimplemented_normalizer = true;  // this checkpoint's NFC normalizer (r4dx-cli)
  const r4dx::Tokenizer tok = r4dx::Tokenizer::from_directory(a.tokenizer_dir, tok_opts);
  const r4dx::ChatTemplate tmpl = r4dx::ChatTemplate::from_directory(a.tokenizer_dir);
  r4dx::ChatJson messages = r4dx::ChatJson::array();
  messages.push_back({{"role", "user"}, {"content", a.prompt}});
  r4dx::ChatJson extra = r4dx::ChatJson::object();
  extra["enable_thinking"] = false;
  const std::string rendered = tmpl.render(messages, /*add_generation_prompt=*/true,
                                           r4dx::ChatJson::array(), extra);
  const std::vector<r4dx::TokenId> ids = tok.encode(rendered, /*parse_special=*/true);
  const std::vector<int32_t> prompt(ids.begin(), ids.end());
  if (prompt.empty()) throw std::runtime_error("the rendered prompt tokenized to nothing");

  ModelOptions opts;
  opts.container_path = a.model;
  opts.layout = LayoutFromName(a.layout);
  opts.max_ctx = a.max_ctx;
  opts.layer_limit = a.layers;
  opts.vision = ModelOptions::VisionMode::kOff;
  std::unique_ptr<r4dx::core::TpComm> comm;
  if (!a.tp1) {
    comm = r4dx::model::tp::MakeNoopComm(/*world=*/2, a.rank);
    opts.tp.world = 2;
    opts.tp.rank = a.rank;
    opts.tp.comm = comm.get();
  }
  const auto load_t0 = std::chrono::steady_clock::now();
  Model m = Model::Load(opts);
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count();

  const std::string mode = a.tp1 ? "tp1" : "noop rank " + std::to_string(a.rank) + "/2";
  // Each process takes the embedding-mirror decision from its own free VRAM (TP=1's heuristic, and
  // decided = -1 on the rank), so it is recorded: G3 compares like with like only when both agree.
  const bool embed_dev = m.GetContainer().EmbedTokensDeviceResident();
  std::printf("[tool_tp_step_bench] %s: %s layout=%s max_ctx=%lld prompt=%zu tokens, HIP device %d "
              "(%s, pci bus %d), load %.1f s, embed_tokens %s\n",
              mode.c_str(), a.model.c_str(), LayoutName(opts.layout),
              static_cast<long long>(a.max_ctx), prompt.size(), device, prop.name, prop.pciBusID,
              load_s, embed_dev ? "device-resident" : "host gather");
  std::fflush(stdout);

  if (a.warmup > 0) (void)RunPass(m, prompt, a.warmup);  // untimed: first touch of everything

  const uint64_t ar_before = comm ? comm->Stats().ar_calls[0] + comm->Stats().ar_calls[1] : 0;
  std::vector<double> runs;
  for (int r = 0; r < a.repeats; ++r) {
    runs.push_back(RunPass(m, prompt, a.tokens));
    std::printf("[tool_tp_step_bench] pass %d: %.3f ms/token (%.2f tok/s)\n", r + 1, runs.back(),
                1000.0 / runs.back());
    std::fflush(stdout);
  }
  const double median = Median(runs);
  std::printf("[tool_tp_step_bench] %s median: %.3f ms/token (%.2f tok/s) over %d x %d tokens\n",
              mode.c_str(), median, 1000.0 / median, a.repeats, a.tokens);
  if (comm) {
    // Every decode step must have hit the three all-reduce sites of every layer: 128 per token on
    // the 64-layer model (docs/tp.md 6.2) -- plus the prefill chunks' own.
    const uint64_t ar = comm->Stats().ar_calls[0] + comm->Stats().ar_calls[1] - ar_before;
    std::printf("[tool_tp_step_bench] no-op all-reduce sites hit: %llu over the timed passes "
                "(%.1f per decode token incl. prefill)\n",
                static_cast<unsigned long long>(ar),
                static_cast<double>(ar) / (static_cast<double>(a.repeats) * a.tokens));
  }

  if (!a.json.empty()) {
    nlohmann::json j;
    j["median_ms"] = median;
    j["runs_ms"] = runs;
    j["rank"] = a.tp1 ? -1 : a.rank;
    j["mode"] = a.tp1 ? "tp1" : "noop";
    j["device_name"] = std::string(prop.name);
    j["pci_bus"] = prop.pciBusID;
    j["hip_device"] = device;
    j["model"] = a.model;
    j["layout"] = LayoutName(opts.layout);
    j["max_ctx"] = a.max_ctx;
    j["layers"] = m.GetContainer().NumLoadedLayers();
    j["tokens"] = a.tokens;
    j["repeats"] = a.repeats;
    j["prompt_tokens"] = prompt.size();
    j["embed_device_resident"] = embed_dev;
    std::ofstream out(a.json);
    if (!out) throw std::runtime_error("cannot write " + a.json);
    out << j.dump(2) << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  return r4dx_test::RunGuardedMain("tool_tp_step_bench", [&] { return Run(a); });
}
