// tool_q35moe_tp_e2e -- the layer split against --tp 2 through the TextModel interface, in ONE process and one session (docs/qwen35moe-tp-design.md
// G7'): the same container, the same fixed pseudo-random prompt, the same fixed token cycle fed to every decode step, host wall clock around the
// calls (what a client sees). Both modes need BOTH cards (26 GiB each), so they run one after the other: load, measure, destroy, load the other.
//
//   per mode: load time; prefill wall at --prompt-len (TTFT, median of --repeats); decode ms per token (median of --tokens steps after --warmup) and the
//   tokens per second that is; a free-running greedy chain of --chain tokens from the prompt (the model's own outputs).
//   Then the ratio TP / split of the decode rate, and how many leading tokens of the two greedy chains agree.
//
// Usage (both cards visible; R4DX_ALLOW_GPU0=1; take the gpu and gpu-dev0 locks): tool_q35moe_tp_e2e [--container <path>] [--modes split,tp]
//   [--prompt-len 512] [--tokens 128] [--warmup 16] [--repeats 3] [--chain 48] [--max-ctx 4096] [--json out.json]
// Built, never add_test()'d.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "nlohmann/json.hpp"
#include "qwen35moe_tp_model.h"
#include "text_model.h"

using r4dx::model::ModelOptions;
using r4dx::model::TextModel;
using r4dx::model::TpOptions;

namespace {

struct Args {
  std::string container = "E:\\models\\r4dx\\ornith-1.5-122b-trellis-mix-kv.r4dx";
  std::string modes = "split,tp", json;
  int prompt_len = 512, tokens = 128, warmup = 16, repeats = 3, chain = 48;
  int64_t max_ctx = 4096;
  bool rank_profile = false;  // --rank-profile: tp mode only, a per-rank GPU-span pass
};

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(k + " needs a value");
      return argv[++i];
    };
    if (k == "--container") a.container = val();
    else if (k == "--modes") a.modes = val();
    else if (k == "--prompt-len") a.prompt_len = std::stoi(val());
    else if (k == "--tokens") a.tokens = std::stoi(val());
    else if (k == "--warmup") a.warmup = std::stoi(val());
    else if (k == "--repeats") a.repeats = std::stoi(val());
    else if (k == "--chain") a.chain = std::stoi(val());
    else if (k == "--max-ctx") a.max_ctx = std::stoll(val());
    else if (k == "--json") a.json = val();
    else if (k == "--rank-profile") a.rank_profile = true;
    else throw std::invalid_argument("unknown argument " + k);
  }
  return a;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}
double Ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct ModeResult {
  double load_s = 0, prefill_ms = 0, decode_ms = 0, decode_p10 = 0, decode_p90 = 0;
  std::vector<int32_t> chain;
};

ModeResult Measure(const std::string& mode, const Args& a, const std::vector<int32_t>& prompt) {
  ModeResult r;
  ModelOptions o;
  o.container_path = a.container;
  o.max_ctx = a.max_ctx;
  o.layout = r4dx::model::Layout::kTrellis;
  TpOptions tp;
  if (mode == "tp") {
    tp.world = 2;
    tp.mode = TpOptions::Mode::kReal;
  }
  const auto t_load = std::chrono::steady_clock::now();
  std::unique_ptr<TextModel> m = r4dx::model::LoadTextModel(o, tp);
  r.load_s = Ms(t_load) / 1000.0;
  std::printf("[%s] loaded in %.1f s (TpWorld %d)\n", mode.c_str(), r.load_s, m->TpWorld());
  for (const auto& v : m->Vram()) std::printf("[%s]   rank %d device %d: %.2f GiB used\n", mode.c_str(), v.rank, v.device, v.used_gib);

  std::vector<double> prefill, decode;
  for (int rep = 0; rep < a.repeats + 1; ++rep) {  // the first pass warms the kernels up
    m->Reset();
    const auto t0 = std::chrono::steady_clock::now();
    (void)m->Prefill(prompt);
    const double pf = Ms(t0);
    if (rep > 0) prefill.push_back(pf);
    for (int i = 0; i < a.warmup + a.tokens; ++i) {
      const int32_t tok = prompt[static_cast<size_t>(i) % prompt.size()];
      const auto t1 = std::chrono::steady_clock::now();
      (void)m->DecodeStepGreedy(tok);
      const double ms = Ms(t1);
      if (rep > 0 && i >= a.warmup) decode.push_back(ms);
    }
  }
  r.prefill_ms = Median(prefill);
  r.decode_ms = Median(decode);
  if (mode == "tp" && a.rank_profile) {
    // one more pass with the per-rank GPU spans on: where the ms of a TP step go (the layers' span includes the time spent in the all-reduces)
    auto* tpm = dynamic_cast<r4dx::model::Qwen35MoeTpModel*>(m.get());
    tpm->RunCollectiveForTest([](r4dx::model::Qwen35MoeModel& mm, int) { mm.EnableStepProfile(true); });
    m->Reset();
    (void)m->Prefill(prompt);
    std::vector<double> lay[2], hd[2], wall;
    auto lay_sink = std::make_shared<std::vector<std::vector<double>>>(2);
    auto hd_sink = std::make_shared<std::vector<std::vector<double>>>(2);
    for (int i = 0; i < a.warmup + a.tokens; ++i) {
      const int32_t tok = prompt[static_cast<size_t>(i) % prompt.size()];
      const auto t1 = std::chrono::steady_clock::now();
      (void)m->DecodeStepGreedy(tok);
      const double ms = Ms(t1);
      tpm->RunCollectiveForTest([&](r4dx::model::Qwen35MoeModel& mm, int rank) {
        if (i < a.warmup) return;
        const auto g = mm.LastStepGpu();
        (*lay_sink)[static_cast<size_t>(rank)].push_back(g.layers_ms[0]);
        (*hd_sink)[static_cast<size_t>(rank)].push_back(g.head_ms);
      });
      if (i >= a.warmup) wall.push_back(ms);
    }
    for (int rk = 0; rk < 2; ++rk) {
      std::printf("[tp] rank %d: layers GPU span median %.3f ms, head %.3f ms\n", rk, Median((*lay_sink)[static_cast<size_t>(rk)]), Median((*hd_sink)[static_cast<size_t>(rk)]));
    }
    std::printf("[tp] profiled pass: host wall median %.3f ms\n", Median(wall));
    tpm->RunCollectiveForTest([](r4dx::model::Qwen35MoeModel& mm, int) { mm.EnableStepProfile(false); });
    std::printf("[tp] %s\n", tpm->StatsLine().c_str());
  }
  std::sort(decode.begin(), decode.end());
  r.decode_p10 = decode[static_cast<size_t>(0.1 * static_cast<double>(decode.size() - 1))];
  r.decode_p90 = decode[static_cast<size_t>(0.9 * static_cast<double>(decode.size() - 1))];
  // the free-running greedy chain
  m->Reset();
  std::vector<float> logits = m->Prefill(prompt);
  int32_t tok = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  r.chain.push_back(tok);
  for (int i = 1; i < a.chain; ++i) {
    tok = m->DecodeStepGreedy(tok);
    r.chain.push_back(tok);
  }
  std::printf("[%s] prefill %d tokens: %.1f ms | decode: %.3f ms/token median (p10 %.3f, p90 %.3f) = %.1f tok/s\n", mode.c_str(), a.prompt_len, r.prefill_ms,
              r.decode_ms, r.decode_p10, r.decode_p90, 1000.0 / r.decode_ms);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = Parse(argc, argv);
    int visible = 0;
    if (hipGetDeviceCount(&visible) != hipSuccess || visible < 2) throw std::runtime_error("both cards must be visible");
    std::vector<int32_t> prompt(static_cast<size_t>(a.prompt_len));
    uint32_t s = 20261023u;
    for (auto& t : prompt) {
      s = s * 1664525u + 1013904223u;
      t = static_cast<int32_t>(1000 + (s >> 8) % 99000);
    }
    nlohmann::json out;
    std::vector<std::pair<std::string, ModeResult>> results;
    size_t pos = 0;
    while (pos < a.modes.size()) {
      const size_t comma = a.modes.find(',', pos);
      const std::string mode = a.modes.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      results.emplace_back(mode, Measure(mode, a, prompt));
      const ModeResult& r = results.back().second;
      out[mode] = {{"load_s", r.load_s}, {"prefill_ms", r.prefill_ms}, {"decode_ms", r.decode_ms}, {"decode_tok_s", 1000.0 / r.decode_ms}, {"chain", r.chain}};
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    if (results.size() == 2) {
      const ModeResult &x = results[0].second, &y = results[1].second;
      int agree = 0;
      while (agree < static_cast<int>(x.chain.size()) && agree < static_cast<int>(y.chain.size()) && x.chain[static_cast<size_t>(agree)] == y.chain[static_cast<size_t>(agree)]) ++agree;
      std::printf("\n%s -> %s: decode %.3f -> %.3f ms/token = %.2fx; prefill %d tokens %.1f -> %.1f ms = %.2fx; greedy chains agree on %d of %zu leading tokens\n",
                  results[0].first.c_str(), results[1].first.c_str(), x.decode_ms, y.decode_ms, x.decode_ms / y.decode_ms, a.prompt_len, x.prefill_ms, y.prefill_ms,
                  x.prefill_ms / y.prefill_ms, agree, x.chain.size());
      out["decode_speedup"] = x.decode_ms / y.decode_ms;
      out["prefill_speedup"] = x.prefill_ms / y.prefill_ms;
      out["chain_agree"] = agree;
    }
    if (!a.json.empty()) std::ofstream(a.json) << out.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_q35moe_tp_e2e: %s\n", e.what());
    return 1;
  }
}
