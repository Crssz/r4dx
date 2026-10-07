// tests/model/tool_pp_stage_bench.cpp -- pipeline-parallel prefill, Phase 0 (docs/pp-prefill.md section 8):
// what one device spends on each stage of a cold prefill, measured with the PP-emulate mode so the stage ranges
// are the real ones (layers [0, k) = stage A, [k, N) = stage B) and the stage-boundary hop is the real
// export / import of the carry through pinned host memory (Model::SetPpEmulate, PpChunkTimes).
//
// One process loads the full model on ONE device (HIP_VISIBLE_DEVICES picks it: 1 = the headless card, 0 = the
// desktop card), warms up, and runs the requested prefills in order. A run is `<kind>:<size>[x<repeats>]`:
//   mono:8k      the monolithic prefill (RunChunk as it always was) -- its TTFT is the baseline;
//   emu33:8k     the same prefill through the two-stage path at split 33, per-chunk stage times recorded;
// with sizes 8k = 8145, 32k = 32623, 64k = 65529 tokens (the design's three prompts) or n<N> for a raw count.
// A repeated run keeps the median (TTFT; per chunk and per column). The chunk grid, the int8 / split-KV defaults
// and the synthetic token ids are the production prefill's, so what is measured is the production per-chunk cost.
// `--dflash` / `--mtp K` load the drafter / the MTP head: stage B's epilogue then includes the DFlash injection
// and the MTP priming, which the CSV carries per chunk (the design's Phase 0 "< 12 ms per chunk" gate).
//
// Output (--out): the CSV tests/model's pp_plan.h parses (mono,<label>,<ms> and chunk,<label>,... lines), read by
// tool_pp_project (the projection and the go rule). `--barrier <prefix>` makes two processes start their runs
// together: it writes <prefix>.ready after the warm-up and waits for <prefix>.go (the Phase 0 script creates it
// once both are ready), so the "both GPUs busy" measurement really overlaps.
//
// R4DX_CLOCK_PROBE=1 in the environment adds the in-model shader-clock probe (debug_probe.h): its summary on
// stderr at exit is the dual-GPU clock note (the probe adds launches, so run it separately from the timing).
//
// Built, never add_test()'d (GPU; prints numbers). Usage, one device at a time:
//   $env:HIP_VISIBLE_DEVICES = '1'
//   tool_pp_stage_bench --runs mono:8k,emu33:8k,mono:32k,emu33:32k --out stage_dev1.csv
//   (--model / --layout default to the production container; --dflash [drafter.r4dx], --mtp 3, --max-ctx N,
//    --split-default 33, --need-gib 20, --barrier <prefix>, --device-note <text> for the CSV header)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "pp_plan.h"
#include "r4dx/core/error.hpp"
#include "test_common.h"

using r4dx::model::LayoutFromName;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
namespace pp = r4dx::model::pp;

namespace {

struct Run {
  bool emulate = false;
  int64_t split = 0;
  int64_t tokens = 0;
  int repeats = 1;
  std::string size_tag;  // "8k" / "32k" / "64k" / "n123"
  std::string label;     // "mono-8k" / "emu33-8k"
};

struct Args {
  std::string model = r4dx_test::ProductionTargetPath();
  std::string layout = r4dx_test::ProductionLayoutName();
  std::string out;
  std::string runs = "mono:8k,emu33:8k";
  std::string barrier;
  std::string device_note;
  std::string dflash;   // drafter container, "" = none
  int64_t dflash_k = 7;
  int64_t mtp = 0;
  int64_t max_ctx = 0;  // 0 = the largest run + 512
  double need_gib = 20.0;
};

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "tool_pp_stage_bench: %s\nusage: tool_pp_stage_bench --runs mono:8k,emu33:8k[x3],... --out file.csv "
               "[--model c.r4dx] [--layout trellis] [--dflash [drafter.r4dx]] [--dflash-k 7] [--mtp K] "
               "[--max-ctx N] [--need-gib 20] [--barrier prefix] [--device-note text]\n",
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
    else if (s == "--out") a.out = next();
    else if (s == "--runs") a.runs = next();
    else if (s == "--barrier") a.barrier = next();
    else if (s == "--device-note") a.device_note = next();
    else if (s == "--dflash") {
      // an optional path: the next argument unless it is another option
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.dflash = argv[++i];
      else a.dflash = r4dx_test::ProductionDrafterPath();
    }
    else if (s == "--dflash-k") a.dflash_k = std::stoll(next());
    else if (s == "--mtp") a.mtp = std::stoll(next());
    else if (s == "--max-ctx") a.max_ctx = std::stoll(next());
    else if (s == "--need-gib") a.need_gib = std::stod(next());
    else Usage("unknown argument " + s);
  }
  if (a.out.empty()) Usage("--out is required");
  if (!a.dflash.empty() && a.mtp > 0) Usage("--dflash and --mtp are mutually exclusive (docs/dflash2.md)");
  return a;
}

int64_t SizeTokens(const std::string& tag) {
  if (tag == "8k") return 8145;
  if (tag == "32k") return 32623;
  if (tag == "64k") return 65529;
  if (tag.size() > 1 && tag[0] == 'n') return std::stoll(tag.substr(1));
  Usage("unknown size '" + tag + "' (8k, 32k, 64k or n<tokens>)");
}

std::vector<Run> ParseRuns(const std::string& spec) {
  std::vector<Run> runs;
  size_t pos = 0;
  while (pos <= spec.size()) {
    const size_t comma = spec.find(',', pos);
    const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    pos = comma == std::string::npos ? spec.size() + 1 : comma + 1;
    if (item.empty()) continue;
    const size_t colon = item.find(':');
    if (colon == std::string::npos) Usage("run '" + item + "' needs <kind>:<size>");
    Run r;
    const std::string kind = item.substr(0, colon);
    std::string size = item.substr(colon + 1);
    const size_t x = size.find('x');
    if (x != std::string::npos) {
      r.repeats = std::stoi(size.substr(x + 1));
      size = size.substr(0, x);
    }
    if (kind == "mono") {
      r.emulate = false;
    } else if (kind.rfind("emu", 0) == 0 && kind.size() > 3) {
      r.emulate = true;
      r.split = std::stoll(kind.substr(3));
    } else {
      Usage("unknown run kind '" + kind + "' (mono or emu<split>)");
    }
    r.size_tag = size;
    r.tokens = SizeTokens(size);
    r.label = kind + "-" + size;
    runs.push_back(r);
  }
  if (runs.empty()) Usage("no runs");
  return runs;
}

std::vector<int32_t> Tokens(int64_t n) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = static_cast<int32_t>(200 + (i * 53 + 17 + (i / 7) * 3) % 5000);
  return ids;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

bool Exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// Two processes start together: <prefix>.ready now, then wait for <prefix>.go (the script's job).
void Barrier(const std::string& prefix) {
  if (prefix.empty()) return;
  { std::ofstream(prefix + ".ready") << "ready\n"; }
  std::fprintf(stderr, "[pp-stage] barrier: waiting for %s.go\n", prefix.c_str());
  for (int i = 0; i < 12000; ++i) {  // 10 minutes
    if (Exists(prefix + ".go")) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  throw std::runtime_error("barrier timeout waiting for " + prefix + ".go");
}

double NowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int RunTool(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  const std::vector<Run> runs = ParseRuns(args.runs);
  int64_t max_tokens = 512;
  for (const Run& r : runs) max_tokens = std::max(max_tokens, r.tokens);
  if (!r4dx_test::FileExists(args.model)) return r4dx_test::SkipMissing(args.model);

  size_t free_b = 0, total_b = 0;
  R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
  if (free_gib < args.need_gib) {
    std::fprintf(stderr, "tool_pp_stage_bench: only %.1f GiB free on this device (need %.1f): is a server running?\n",
                 free_gib, args.need_gib);
    return 1;
  }
  int dev = -1;
  (void)hipGetDevice(&dev);
  hipDeviceProp_t prop{};
  (void)hipGetDeviceProperties(&prop, dev);
  char bus[64] = "?";
  (void)hipDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), dev);
  std::fprintf(stderr, "[pp-stage] device ordinal %d: %s (PCI %s), %.1f GiB free\n", dev, prop.name, bus, free_gib);

  ModelOptions o;
  o.container_path = args.model;
  o.layout = LayoutFromName(args.layout);
  o.max_ctx = args.max_ctx > 0 ? args.max_ctx : max_tokens + 512;
  o.vision = ModelOptions::VisionMode::kOff;  // the prefill under test is text; its 0.9 GiB is not ours to pay
  o.pp_emulate_split = 0;                     // off at load; every emu run turns it on itself
  if (!args.dflash.empty()) {
    o.dflash_container = args.dflash;
    o.dflash_draft_k = args.dflash_k;
  }
  o.mtp_draft_k = args.mtp;
  std::fprintf(stderr, "[pp-stage] loading %s (layout %s, max_ctx %lld%s%s)\n", args.model.c_str(), args.layout.c_str(),
               static_cast<long long>(o.max_ctx), args.dflash.empty() ? "" : ", --dflash", args.mtp > 0 ? ", --mtp" : "");
  Model m = Model::Load(o);
  const int64_t layers = m.GetContainer().NumLoadedLayers();
  for (const Run& r : runs) {
    if (r.emulate && !pp::ValidSplit(r.split, layers)) {
      std::fprintf(stderr, "tool_pp_stage_bench: split %lld is not inside [1, %lld)\n", static_cast<long long>(r.split),
                   static_cast<long long>(layers));
      return 2;
    }
  }

  // Warm-up: the first call of every kernel module, PickTuning's cache, first-touch page faults -- monolithic
  // and through the two-stage path (its pinned staging is allocated on first use).
  {
    const std::vector<int32_t> w = Tokens(768);
    (void)m.Prefill(w);
    m.Reset();
    Model::PpEmulateConfig cfg;
    cfg.split = layers / 2;
    cfg.timing = true;
    m.SetPpEmulate(cfg);
    (void)m.Prefill(w);
    m.SetPpEmulate(Model::PpEmulateConfig{});
    m.ClearPpChunkTimeLog();
    m.Reset();
  }
  Barrier(args.barrier);

  std::string csv;
  csv += "# tool_pp_stage_bench: device ordinal " + std::to_string(dev) + " (" + prop.name + ", PCI " + bus + "), model " +
         args.model + "\n";
  csv += "# runs " + args.runs + (args.dflash.empty() ? "" : "; --dflash") + (args.mtp > 0 ? "; --mtp " + std::to_string(args.mtp) : "") +
         (args.device_note.empty() ? "" : "; " + args.device_note) + "\n";
  for (const Run& r : runs) {
    const std::vector<int32_t> ids = Tokens(r.tokens);
    std::vector<double> ttft;
    std::vector<std::vector<Model::PpChunkTimes>> logs;
    for (int rep = 0; rep < r.repeats; ++rep) {
      m.Reset();
      if (r.emulate) {
        Model::PpEmulateConfig cfg;
        cfg.split = r.split;
        cfg.timing = true;
        m.SetPpEmulate(cfg);
        m.ClearPpChunkTimeLog();
      }
      const double t0 = NowMs();
      (void)m.Prefill(ids);
      const double t1 = NowMs();
      ttft.push_back(t1 - t0);
      if (r.emulate) {
        logs.push_back(m.PpChunkTimeLog());
        m.SetPpEmulate(Model::PpEmulateConfig{});
      }
      std::fprintf(stderr, "[pp-stage] %s rep %d: %.1f ms%s\n", r.label.c_str(), rep, t1 - t0,
                   r.emulate ? " (two-stage path, stages timed apart)" : "");
    }
    const double med = Median(ttft);
    if (!r.emulate) {
      csv += pp::FormatMonoLine(r.label, med) + "\n";
      std::fprintf(stderr, "[pp-stage] %s: TTFT %.1f ms (median of %d)\n", r.label.c_str(), med, r.repeats);
      continue;
    }
    csv += "# " + r.label + ": two-stage wall time " + std::to_string(med) + " ms (median of " + std::to_string(r.repeats) +
           "), stages timed apart so this exceeds the monolithic TTFT by the hop and the extra synchronizes\n";
    const size_t n = logs.front().size();
    for (const auto& l : logs) {
      if (l.size() != n) throw std::runtime_error("repeats made different chunk counts");
    }
    double sum_a = 0, sum_b = 0, sum_hop = 0, sum_epi = 0, sum_mtp = 0, sum_inj = 0;
    for (size_t c = 0; c < n; ++c) {
      const auto col = [&](double Model::PpChunkTimes::*f) {
        std::vector<double> v;
        for (const auto& l : logs) v.push_back(l[c].*f);
        return Median(v);
      };
      pp::Sample s;
      s.pos = logs.front()[c].pos;
      s.rows = logs.front()[c].rows;
      s.a_ms = col(&Model::PpChunkTimes::a_ms);
      s.d2h_ms = col(&Model::PpChunkTimes::d2h_ms);
      s.h2d_ms = col(&Model::PpChunkTimes::h2d_ms);
      s.b_ms = col(&Model::PpChunkTimes::b_ms);
      s.epilogue_ms = col(&Model::PpChunkTimes::epilogue_ms);
      s.mtp_ms = col(&Model::PpChunkTimes::mtp_ms);
      s.inject_ms = col(&Model::PpChunkTimes::inject_ms);
      s.hop_bytes = logs.front()[c].hop_bytes;
      csv += pp::FormatChunkLine(r.label, c, s) + "\n";
      sum_a += s.a_ms;
      sum_b += s.b_ms;
      sum_hop += s.d2h_ms + s.h2d_ms;
      sum_epi += s.epilogue_ms;
      sum_mtp += s.mtp_ms;
      sum_inj += s.inject_ms;
    }
    std::fprintf(stderr,
                 "[pp-stage] %s: %zu chunks; stage A %.1f ms, stage B %.1f ms, hop %.1f ms, epilogue %.1f ms (MTP %.1f, "
                 "DFlash injection %.1f)\n",
                 r.label.c_str(), n, sum_a, sum_b, sum_hop, sum_epi, sum_mtp, sum_inj);
  }
  std::ofstream f(args.out, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + args.out);
  f << csv;
  std::fprintf(stderr, "[pp-stage] wrote %s\n", args.out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return r4dx_test::RunGuardedMain("tool_pp_stage_bench", [&] { return RunTool(argc, argv); });
}
