// tool_q35moe_tp_step -- the P2 economic gate of docs/qwen35moe-tp-design.md (G3'): the GPU time of one Ornith decode step on the LAYER SPLIT
// against one TP rank's step with NO peer, measured by the same tool, prompt and protocol.
//
//   --mode split   the ordinary two-stage Qwen35MoeModel (both visible devices): stage 0 + stage 1 layers + the head, summed GPU spans
//   --mode noop    rank 0 (--rank r) of a TP=2 load on ONE device with tp::MakeNoopComm: its shard's layers + its vocab-slice head, no all-reduce
//                  kernel and no merge (the noop tokens are meaningless -- half of every row-parallel sum is missing -- so every run feeds the same fixed
//                  token cycle instead of its own output: identical work per step, nothing can hit EOS)
//
// The spans are hipEvent pairs on each stage's stream (Qwen35MoeModel::EnableStepProfile): a decode step only ENQUEUES its layers, so host
// wall time is not the GPU time. Per step the tool also prints the host wall of DecodeStepGreedy. G3' passes when median(noop GPU step) <=
// 0.62 x median(split GPU step); 0.62-0.75 continue and report; above 0.75 stop (design 6.3, P2).
//
// Protocol: a fixed pseudo-random prompt of --prompt-len tokens (Prefill, in 256-row chunks: the GPU time of the last chunk is reported too),
// --warmup untimed decode steps, then --tokens timed ones, repeated --repeats times (Reset + Prefill each). Built, never add_test()'d.
// Usage (HIP_VISIBLE_DEVICES=1 for noop on device 1; both devices visible for split):
//   tool_q35moe_tp_step --mode split|noop [--container <path>] [--prompt-len 512] [--tokens 64] [--warmup 16] [--repeats 3] [--max-ctx 2048]
//                       [--rank 0] [--device 0] [--sweep gu,down]   (Tp2 split-group counts, noop only)  [--json out.json]
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

#include "nlohmann/json.hpp"
#include "qwen35moe_model.h"
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"
#include "tp/tp_comm_noop.h"

using r4dx::model::Qwen35MoeModel;
using r4dx::model::Qwen35MoeModelOptions;

namespace {

struct Args {
  std::string mode = "split";
  std::string container = "E:\\models\\r4dx\\ornith-1.5-122b-trellis-mix-kv.r4dx";
  int prompt_len = 512, tokens = 64, warmup = 16, repeats = 3, rank = 0, device = 0;
  int64_t max_ctx = 2048;
  std::string sweep, json;
};

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(k + " needs a value");
      return argv[++i];
    };
    if (k == "--mode") a.mode = val();
    else if (k == "--container") a.container = val();
    else if (k == "--prompt-len") a.prompt_len = std::stoi(val());
    else if (k == "--tokens") a.tokens = std::stoi(val());
    else if (k == "--warmup") a.warmup = std::stoi(val());
    else if (k == "--repeats") a.repeats = std::stoi(val());
    else if (k == "--rank") a.rank = std::stoi(val());
    else if (k == "--device") a.device = std::stoi(val());
    else if (k == "--max-ctx") a.max_ctx = std::stoll(val());
    else if (k == "--sweep") a.sweep = val();
    else if (k == "--json") a.json = val();
    else throw std::invalid_argument("unknown argument " + k);
  }
  if (a.mode != "split" && a.mode != "noop") throw std::invalid_argument("--mode must be split or noop");
  return a;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}
double Pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5)];
}

struct Series {
  std::vector<double> gpu, wall, l0, l1, head;
};

Series RunDecode(Qwen35MoeModel& m, const std::vector<int32_t>& prompt, const Args& a) {
  Series s;
  for (int rep = 0; rep < a.repeats; ++rep) {
    m.Reset();
    (void)m.Prefill(prompt);
    for (int i = 0; i < a.warmup + a.tokens; ++i) {
      const int32_t tok = prompt[static_cast<size_t>(i) % prompt.size()];
      const auto t0 = std::chrono::steady_clock::now();
      (void)m.DecodeStepGreedy(tok);
      const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (i < a.warmup) continue;
      const Qwen35MoeModel::StepGpu g = m.LastStepGpu();
      s.gpu.push_back(g.Total());
      s.wall.push_back(wall);
      s.l0.push_back(g.layers_ms[0]);
      s.l1.push_back(g.layers_ms[1]);
      s.head.push_back(g.head_ms);
    }
  }
  return s;
}

void Report(const char* tag, const Series& s) {
  std::printf("%s: GPU step median %.3f ms (p10 %.3f, p90 %.3f) = layers %.3f + %.3f, head %.3f | host wall median %.3f ms | %zu steps\n", tag, Median(s.gpu),
              Pct(s.gpu, 0.1), Pct(s.gpu, 0.9), Median(s.l0), Median(s.l1), Median(s.head), Median(s.wall), s.gpu.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = Parse(argc, argv);
    int visible = 0;
    if (hipGetDeviceCount(&visible) != hipSuccess || visible < 1) throw std::runtime_error("no HIP device");
    std::printf("mode %s, %d device(s) visible, container %s\n", a.mode.c_str(), visible, a.container.c_str());

    Qwen35MoeModelOptions o;
    o.container_path = a.container;
    o.max_ctx = a.max_ctx;
    std::unique_ptr<r4dx::core::TpComm> noop;
    if (a.mode == "noop") {
      noop = r4dx::model::tp::MakeNoopComm(2, a.rank);
      o.tp_world = 2;
      o.tp_rank = a.rank;
      o.tp_device = a.device;
      o.tp_comm = noop.get();
    } else if (visible >= 2) {
      o.devices = {0, 1};
    }
    const auto t_load = std::chrono::steady_clock::now();
    auto m = Qwen35MoeModel::Load(o);
    std::printf("loaded in %.1f s; %d stage(s), split at layer %lld, grouped decode MoE in %lld layers\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), m->NumStages(),
                static_cast<long long>(m->SplitLayer()), static_cast<long long>(m->GroupedDecodeLayers()));
    for (const auto& v : m->Vram()) std::printf("  stage device %d: %.2f GiB used of %.2f (%.2f GiB of buffers)\n", v.device, v.used_gib, v.total_gib, v.buffers_gib);
    m->EnableStepProfile(true);

    std::vector<int32_t> prompt(static_cast<size_t>(a.prompt_len));  // fixed pseudo-random ids inside the vocabulary (the same for every run)
    {
      const int64_t vocab = m->GlobalConfig().vocab_size;
      const int64_t lo = std::min<int64_t>(1000, vocab / 4), span = std::min<int64_t>(99000, vocab - lo);
      uint32_t s = 20261023u;
      for (auto& t : prompt) {
        s = s * 1664525u + 1013904223u;
        t = static_cast<int32_t>(lo + static_cast<int64_t>(s >> 8) % span);
      }
    }

    nlohmann::json out;
    out["mode"] = a.mode;
    // the last 256-row prefill chunk
    {
      std::vector<double> chunk;
      for (int rep = 0; rep < std::max(a.repeats, 2); ++rep) {
        m->Reset();
        (void)m->Prefill(prompt);
        const Qwen35MoeModel::StepGpu g = m->LastStepGpu();
        if (rep > 0) chunk.push_back(g.Total());  // (the first pass warms the kernels up)
      }
      std::printf("prefill, last chunk of %d tokens (%lld rows): GPU median %.2f ms (layers + head)\n", a.prompt_len,
                  static_cast<long long>(m->LastTimings().rows), Median(chunk));
      out["prefill_last_chunk_gpu_ms"] = Median(chunk);
      out["prefill_last_chunk_rows"] = m->LastTimings().rows;
    }

    if (a.mode == "noop" && !a.sweep.empty()) {
      // "gu,down" pairs separated by ';' -> the Tp2 grouped GEMV split-group counts, one decode series each
      std::vector<std::pair<int, int>> pairs;
      size_t pos = 0;
      while (pos < a.sweep.size()) {
        const size_t semi = a.sweep.find(';', pos);
        const std::string item = a.sweep.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        const size_t comma = item.find(',');
        pairs.emplace_back(std::stoi(item.substr(0, comma)), std::stoi(item.substr(comma + 1)));
        if (semi == std::string::npos) break;
        pos = semi + 1;
      }
      for (const auto& [gu, dn] : pairs) {
        if (r4dx::model::qwen4exp::GroupedQ35Tp2SetSplits(gu, dn) != hipSuccess) throw std::runtime_error("bad split pair");
        char tag[64];
        std::snprintf(tag, sizeof tag, "noop splits gate_up %d down %d", gu, dn);
        const Series r = RunDecode(*m, prompt, a);
        Report(tag, r);
        out["sweep"].push_back({{"gu", gu}, {"down", dn}, {"gpu_ms", Median(r.gpu)}});
      }
      r4dx::model::qwen4exp::GroupedQ35Tp2SetSplits(r4dx::model::qwen4exp::kGroupedQ35Tp2GuSkg, r4dx::model::qwen4exp::kGroupedQ35Tp2DownSkg);
    }

    const Series r = RunDecode(*m, prompt, a);
    Report(a.mode.c_str(), r);
    out["gpu_step_median_ms"] = Median(r.gpu);
    out["gpu_step_p10_ms"] = Pct(r.gpu, 0.1);
    out["gpu_step_p90_ms"] = Pct(r.gpu, 0.9);
    out["layers_ms"] = {Median(r.l0), Median(r.l1)};
    out["head_ms"] = Median(r.head);
    out["wall_step_median_ms"] = Median(r.wall);
    out["steps"] = r.gpu.size();
    if (!a.json.empty()) std::ofstream(a.json) << out.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_q35moe_tp_step: %s\n", e.what());
    return 1;
  }
}
