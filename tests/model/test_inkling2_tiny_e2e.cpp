// tests/model/test_inkling2_tiny_e2e.cpp -- docs/inkling2-plan.md 5 / 7: the whole inkling2_loop stack end to end on the
// TINY synthetic checkpoint (tools/convert_ref/make_tiny_inkling2.py: hidden 64, 16 layers, block 4, loop [4, 12) x 3 =
// 32 visits, full attention at 3 / 7 / 11 / 15, window 8, rel extent 8, 8 experts top-2 (NVFP4 in 0-11, MXFP8 in 12-15),
// vocab 112 / 90 logits) against tests/convert/data/inkling2_tiny/logits.npy -- HF Inkling2LoopForCausalLM in float32 on
// tokens.json's 40 tokens (every row).
//
// GPU test (one device; a second with R4DX_INKLING2_TINY_TP_REAL=1); WRITTEN, NOT YET RUN by its author. SKIP 77 when the
// fixture cannot be made. The fixture (R4DX_INKLING2_TINY_DIR, default %TEMP%\r4dx_inkling2_tiny_e2e) is built on first use,
// on the CPU:
//   1. the checkpoint: tests/convert/data/inkling2_tiny/model.safetensors when present, else
//      `<python> -I tools/convert_ref/make_tiny_inkling2.py --out <dir>/ckpt` (deterministic; needs numpy + torch CPU:
//      R4DX_INKLING2_PYTHON, default D:\venvs\r4dx-inkling-ref\Scripts\python.exe);
//   2. `r4dx-convert --input <ckpt> --output <dir>/tiny-bf16.r4dx --layouts bf16` and
//      `... --output <dir>/tiny-mxfp8.r4dx --layouts mxfp8 --no-bf16` (this build's converter).
//
// The tiny random model is CHAOTIC under bf16 numerics: HF's own bf16 forward is at mean KL(fp32 || bf16) 0.265 (worst row
// 0.55, top-1 13/40; measured 2026-11-22, tools: the reference venv on this checkpoint), because rounding flips its
// top-2-of-8 routing. So the bounds against HF are the gate rule over that noise, and the sharp check is against the CPU
// TWIN (tests/model/inkling2_cpu_graph.h: the same graph, the same container weights, the same rounding points; its
// exact-fp32 mode reproduces HF to KL 6e-12, test_inkling2_cpu_graph), whose only difference from the device is the
// summation order. Bounds are UNVERIFIED (set before the first GPU run).
//
// Per (container, mode): TP=1 for both; TP=2 (--tp-mode emulate, both ranks on one device) for bf16 -- the mxfp8 tiny
// fixture cannot be sharded (its expert down K = 32 splits into 16-wide halves, not whole 32-K groups) and must be refused
// with that reason:
//   1. rows: Prefill(token 0), then DecodeStep for tokens 1..39 -> 40 rows. vs HF fp32: mean KL <= 1.5 x 0.265 + 0.005
//      (bf16) / 0.55 (mxfp8: e4m3 linears, NVFP4 experts re-encoded; its twin measures 0.32), every row < 1.5. vs the twin:
//      row 0 KL < 1e-2 (one token: no chaos yet), mean KL <= 0.3.
//   2. chunked prefill: prefill_chunk 7, Prefill [0, 20) and [20, 33) (row 19, row 32: chunk boundaries inside both calls)
//      then DecodeStep 33..39 -> those rows vs the golden under the same bound, and vs the decode feed (mean KL <= 0.1).
//   3. determinism: Reset and the decode feed again -> identical bytes.
//   4. checkpoint: Prefill [0, 20), SaveCheckpoint, DecodeStep 20..24, RestoreCheckpoint, DecodeStep 20..24 again ->
//      identical; PositionCount follows.
//   5. DecodeStepGreedy == argmax(DecodeStep) in the same state; DecodeStepSampled returns an id in [0, 90).
// Across modes: TP=2 rows vs TP=1 rows of the bf16 container, mean KL <= 0.3 (the all-reduce sums two bf16 partials).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "inkling2_cpu_graph.h"
#include "inkling2_local_text_model.h"
#include "model.h"  // ModelOptions
#include "nlohmann/json.hpp"
#include "test_common.h"
#include "text_model.h"

using namespace r4dx;
using namespace r4dx::model;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
std::string EnvOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? std::string(v) : fallback;
}

std::vector<double> LogSoftmax(const float* x, int64_t V) {
  double mx = -1e300;
  for (int64_t i = 0; i < V; ++i) mx = std::max(mx, static_cast<double>(x[i]));
  double s = 0;
  for (int64_t i = 0; i < V; ++i) s += std::exp(static_cast<double>(x[i]) - mx);
  const double lse = mx + std::log(s);
  std::vector<double> out(static_cast<size_t>(V));
  for (int64_t i = 0; i < V; ++i) out[static_cast<size_t>(i)] = static_cast<double>(x[i]) - lse;
  return out;
}
double Kl(const float* p, const float* q, int64_t V) {
  const std::vector<double> lp = LogSoftmax(p, V), lq = LogSoftmax(q, V);
  double kl = 0;
  for (int64_t i = 0; i < V; ++i) kl += std::exp(lp[static_cast<size_t>(i)]) * (lp[static_cast<size_t>(i)] - lq[static_cast<size_t>(i)]);
  return kl;
}
int64_t Argmax(const float* x, int64_t V) { return std::max_element(x, x + V) - x; }

// A float32 C-order .npy of 2 dims.
std::vector<float> ReadNpy(const fs::path& p, int64_t* rows, int64_t* cols) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (all.size() < 10 || all.compare(1, 5, "NUMPY") != 0) throw std::runtime_error(p.string() + " is not an .npy");
  const int major = static_cast<unsigned char>(all[6]);
  size_t hl = 0, off = 0;
  if (major == 1) {
    hl = static_cast<unsigned char>(all[8]) | (static_cast<size_t>(static_cast<unsigned char>(all[9])) << 8);
    off = 10;
  } else {
    hl = static_cast<unsigned char>(all[8]) | (static_cast<size_t>(static_cast<unsigned char>(all[9])) << 8) |
         (static_cast<size_t>(static_cast<unsigned char>(all[10])) << 16) | (static_cast<size_t>(static_cast<unsigned char>(all[11])) << 24);
    off = 12;
  }
  const std::string hdr = all.substr(off, hl);
  if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error(p.string() + ": want little-endian float32, C order");
  const size_t s = hdr.find('('), e = hdr.find(')');
  const std::string dims = hdr.substr(s + 1, e - s - 1);
  *rows = std::stoll(dims.substr(0, dims.find(',')));
  *cols = std::stoll(dims.substr(dims.find(',') + 1));
  std::vector<float> v(static_cast<size_t>(*rows * *cols));
  if (all.size() < off + hl + v.size() * 4) throw std::runtime_error(p.string() + " is truncated");
  std::memcpy(v.data(), all.data() + off + hl, v.size() * 4);
  return v;
}

int RunCmd(const std::string& exe, const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + exe + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  std::printf("[fixture] %s %s\n", exe.c_str(), args.c_str());
  return std::system(cmd.c_str());
}

struct Golden {
  std::vector<int32_t> ids;
  std::vector<float> logits;  // [N, V]
  int64_t N = 0, V = 0;
  const float* Row(int64_t r) const { return logits.data() + r * V; }
};

struct RowStats {
  double mean_kl = 0, max_kl = 0;
  int top1 = 0;
};
RowStats Compare(const Golden& g, const std::vector<int64_t>& rows, const std::vector<std::vector<float>>& got) {
  RowStats s;
  for (size_t i = 0; i < rows.size(); ++i) {
    const double kl = Kl(g.Row(rows[i]), got[i].data(), g.V);
    s.mean_kl += kl;
    s.max_kl = std::max(s.max_kl, kl);
    s.top1 += Argmax(got[i].data(), g.V) == Argmax(g.Row(rows[i]), g.V) ? 1 : 0;
  }
  s.mean_kl /= static_cast<double>(rows.size());
  return s;
}

struct Bounds {
  double mean, max;  // vs HF fp32
};

// The CPU twin's decode feed on `container` (inkling2_cpu_graph.h, the GPU rounding points).
std::vector<std::vector<float>> Twin(const fs::path& container, const Golden& g) {
  std::ifstream f(container, std::ios::binary);
  uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 8);
  std::string h(static_cast<size_t>(n), '\0');
  f.read(h.data(), static_cast<std::streamsize>(n));
  const nlohmann::json meta = nlohmann::json::parse(h).at("__metadata__");
  const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(container.u8string()));
  inkling2_cpu::Engine e;
  e.cfg = Inkling2Config::FromModelConfig(meta.at("model_config"));
  e.w.r = &r;
  e.visits = e.cfg.Visits();
  e.Reset();
  std::vector<std::vector<float>> rows;
  for (int64_t p = 0; p < g.N; ++p) rows.push_back(e.Chunk({g.ids[static_cast<size_t>(p)]}));
  return rows;
}

std::vector<std::vector<float>> DecodeFeed(TextModel& m, const Golden& g) {
  std::vector<std::vector<float>> out;
  m.Reset();
  out.push_back(m.Prefill({g.ids[0]}));
  for (int64_t p = 1; p < g.N; ++p) out.push_back(m.DecodeStep(g.ids[static_cast<size_t>(p)]));
  return out;
}

std::vector<std::vector<float>> RunMode(TextModel& m, const Golden& g, const Bounds& b, const std::string& tag,
                                        TextModel* chunked, const std::vector<std::vector<float>>* twin) {
  std::printf("---- %s ----\n", tag.c_str());
  const int64_t V = g.V;
  // 1. every row through decode
  const std::vector<std::vector<float>> got = DecodeFeed(m, g);
  bool shape = true;
  for (const auto& r : got) shape = shape && static_cast<int64_t>(r.size()) == V;
  Check(shape && m.PositionCount() == g.N, tag + ": 40 rows of 90 logits, PositionCount 40");
  std::vector<int64_t> all(static_cast<size_t>(g.N));
  for (int64_t i = 0; i < g.N; ++i) all[static_cast<size_t>(i)] = i;
  if (shape) {
    const RowStats s = Compare(g, all, got);
    std::printf("  %s: mean KL(HF||r4dx) %.3e, worst row %.3e, top-1 %d/%lld\n", tag.c_str(), s.mean_kl, s.max_kl, s.top1,
                static_cast<long long>(g.N));
    Check(s.mean_kl <= b.mean && s.max_kl < b.max, tag + ": decode rows vs HF fp32 within the noise bound");
    if (twin != nullptr) {
      double tk = 0;
      for (int64_t i = 0; i < g.N; ++i) tk += Kl((*twin)[static_cast<size_t>(i)].data(), got[static_cast<size_t>(i)].data(), V);
      tk /= static_cast<double>(g.N);
      const double row0 = Kl((*twin)[0].data(), got[0].data(), V);
      std::printf("  %s vs the CPU twin: row 0 KL %.3e, mean KL %.3e\n", tag.c_str(), row0, tk);
      Check(row0 < 1e-2 && tk <= 0.3, tag + ": decode rows vs the CPU twin (row 0 < 1e-2, mean <= 0.3)");
    }
  }
  // 2. chunked prefill (prefill_chunk 7) on the second model when given
  if (chunked != nullptr && shape) {
    chunked->Reset();
    std::vector<int64_t> rows = {19, 32};
    std::vector<std::vector<float>> c;
    c.push_back(chunked->Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 20)));
    c.push_back(chunked->Prefill(std::vector<int32_t>(g.ids.begin() + 20, g.ids.begin() + 33)));
    for (int64_t p = 33; p < g.N; ++p) {
      rows.push_back(p);
      c.push_back(chunked->DecodeStep(g.ids[static_cast<size_t>(p)]));
    }
    const RowStats s = Compare(g, rows, c);
    double vs_decode = 0;
    for (size_t i = 0; i < rows.size(); ++i) vs_decode += Kl(got[static_cast<size_t>(rows[i])].data(), c[i].data(), V);
    vs_decode /= static_cast<double>(rows.size());
    std::printf("  %s chunked (7): mean KL %.3e, worst %.3e, vs the decode feed mean %.3e\n", tag.c_str(), s.mean_kl, s.max_kl,
                vs_decode);
    Check(s.mean_kl <= b.mean && s.max_kl < b.max, tag + ": chunked prefill (chunk 7, rows 19 / 32 / 33..39) within the bound");
    Check(vs_decode <= 0.1, tag + ": chunked prefill ~ decode feed (mean KL <= 0.1)");
  }
  // 3. determinism
  Check(DecodeFeed(m, g) == got, tag + ": the same feed after Reset() gives identical bytes");
  // 4. checkpoint
  m.Reset();
  (void)m.Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 20));
  m.SaveCheckpoint();
  std::vector<std::vector<float>> a, bb;
  for (int64_t p = 20; p < 25; ++p) a.push_back(m.DecodeStep(g.ids[static_cast<size_t>(p)]));
  m.RestoreCheckpoint();
  Check(m.PositionCount() == 20, tag + ": RestoreCheckpoint rewinds PositionCount to 20");
  for (int64_t p = 20; p < 25; ++p) bb.push_back(m.DecodeStep(g.ids[static_cast<size_t>(p)]));
  Check(a == bb, tag + ": RestoreCheckpoint + the same tokens reproduce the logits bit for bit");
  // 5. greedy / sampled
  m.Reset();
  (void)m.Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 30));
  const int32_t greedy = m.DecodeStepGreedy(g.ids[30]);
  m.Reset();
  (void)m.Prefill(std::vector<int32_t>(g.ids.begin(), g.ids.begin() + 30));
  const std::vector<float> full = m.DecodeStep(g.ids[30]);
  Check(greedy == static_cast<int32_t>(Argmax(full.data(), V)), tag + ": DecodeStepGreedy == argmax(DecodeStep)");
  kernels::SampleParams sp;
  sp.temperature = 1.0f;
  std::mt19937_64 rng(1234);
  const int32_t sampled = m.DecodeStepSampled(g.ids[31], sp, rng);
  Check(sampled >= 0 && sampled < V, tag + ": DecodeStepSampled returns an id in [0, 90)");
  m.Reset();
  return got;
}

}  // namespace

int main(int argc, char** argv) {
  // --tp2-real (the tp2gpu registration): the bf16 TP=1 baseline and TP=2 over both GPUs only; SKIP unless R4DX_TP2GPU=1
  // (tests\run_tests.ps1 -TwoGpu sets it).
  const bool tp2_real_only = argc > 1 && std::string(argv[1]) == "--tp2-real";
  return r4dx_test::RunGuardedMain("test_inkling2_tiny_e2e", [tp2_real_only]() -> int {
    if (tp2_real_only && EnvOr("R4DX_TP2GPU", "0") != "1") return r4dx_test::SkipMissing("R4DX_TP2GPU=1 (tests\\run_tests.ps1 -TwoGpu)");
    // With both GPUs visible (-TwoGpu unsets HIP_VISIBLE_DEVICES) the TP=1 baseline runs on device 1, never device 0.
    if (tp2_real_only) {
      int n = 0;
      if (hipGetDeviceCount(&n) != hipSuccess || n < 2) return r4dx_test::SkipMissing("two visible HIP devices");
      if (hipSetDevice(1) != hipSuccess) throw std::runtime_error("hipSetDevice(1) failed");
    }
    const fs::path src_data = fs::path(R4DX_SOURCE_DIR) / "tests" / "convert" / "data" / "inkling2_tiny";
    const fs::path dir = EnvOr("R4DX_INKLING2_TINY_DIR", (fs::temp_directory_path() / "r4dx_inkling2_tiny_e2e").u8string());
    const fs::path bf16 = dir / "tiny-bf16.r4dx", mx = dir / "tiny-mxfp8.r4dx";
    fs::create_directories(dir);
    if (!fs::exists(bf16) || !fs::exists(mx)) {
#ifndef R4DX_CONVERT_EXE
      return r4dx_test::SkipMissing(bf16.u8string() + " (and this build has no r4dx-convert to make it)");
#else
      fs::path ckpt = src_data;
      if (!fs::exists(src_data / "model.safetensors")) {
        ckpt = dir / "ckpt";
        if (!fs::exists(ckpt / "model.safetensors")) {
          const std::string py = EnvOr("R4DX_INKLING2_PYTHON", "D:/venvs/r4dx-inkling-ref/Scripts/python.exe");
          if (!fs::exists(py)) return r4dx_test::SkipMissing(py + " (numpy + torch, for make_tiny_inkling2.py)");
          const std::string gen = (fs::path(R4DX_SOURCE_DIR) / "tools" / "convert_ref" / "make_tiny_inkling2.py").u8string();
          if (RunCmd(py, "-I \"" + gen + "\" --out \"" + ckpt.u8string() + "\"", dir / "make_tiny.log") != 0)
            throw std::runtime_error("make_tiny_inkling2.py failed (" + (dir / "make_tiny.log").u8string() + ")");
        }
      }
      const std::string in = "--input \"" + ckpt.u8string() + "\" ";
      if (RunCmd(R4DX_CONVERT_EXE, in + "--output \"" + bf16.u8string() + "\" --layouts bf16", dir / "convert_bf16.log") != 0)
        throw std::runtime_error("r4dx-convert --layouts bf16 failed (" + (dir / "convert_bf16.log").u8string() + ")");
      if (RunCmd(R4DX_CONVERT_EXE, in + "--output \"" + mx.u8string() + "\" --layouts mxfp8 --no-bf16", dir / "convert_mxfp8.log") != 0)
        throw std::runtime_error("r4dx-convert --layouts mxfp8 failed (" + (dir / "convert_mxfp8.log").u8string() + ")");
#endif
    }
    Golden g;
    g.logits = ReadNpy(src_data / "logits.npy", &g.N, &g.V);
    {
      std::ifstream f(src_data / "tokens.json");
      const nlohmann::json j = nlohmann::json::parse(f);
      for (const auto& t : j.at("tokens")) g.ids.push_back(t.get<int32_t>());
    }
    if (static_cast<int64_t>(g.ids.size()) != g.N || g.V != 90) throw std::runtime_error("logits.npy / tokens.json disagree");

    const auto opts = [&](const fs::path& c, Layout lay, int chunk) {
      ModelOptions o;
      o.container_path = c.u8string();
      o.layout = lay;
      o.max_ctx = 512;
      o.prefill_chunk = chunk;
      o.prompt_checkpoint = true;
      return o;
    };
    const Bounds bf16_bound{1.5 * 0.265 + 0.005, 1.5}, mx_bound{0.55, 1.5};
    std::vector<std::vector<float>> bf16_tp1;
    {
      TpOptions tp;
      std::unique_ptr<TextModel> m = LoadTextModel(opts(bf16, Layout::kBf16, 256), tp);
      std::unique_ptr<TextModel> c = LoadTextModel(opts(bf16, Layout::kBf16, 7), tp);
      Check(m->Config().arch == Arch::kInkling2Loop && m->Config().vocab_size == 90 && m->TpWorld() == 1,
            "bf16: LoadTextModel dispatches the tiny container to the Inkling2 engine (vocab 90)");
      Check(!m->HasVision() && !m->MtpEnabled() && !m->DflashEnabled(), "bf16: no vision / MTP / DFlash");
      const std::vector<std::vector<float>> twin = Twin(bf16, g);
      bf16_tp1 = RunMode(*m, g, bf16_bound, "bf16 TP=1", c.get(), &twin);
    }
    std::vector<std::vector<float>> mx_tp1;
    if (!tp2_real_only) {
      TpOptions tp;
      std::unique_ptr<TextModel> m = LoadTextModel(opts(mx, Layout::kMxfp8, 256), tp);
      std::unique_ptr<TextModel> c = LoadTextModel(opts(mx, Layout::kMxfp8, 7), tp);
      const std::vector<std::vector<float>> twin = Twin(mx, g);
      mx_tp1 = RunMode(*m, g, mx_bound, "mxfp8 TP=1", c.get(), &twin);
      {
        auto* lm = dynamic_cast<Inkling2LocalTextModel*>(m.get());
        Check(lm != nullptr && lm->model().GetContainer().BodyFallbacks() == 0, "mxfp8: every linear loaded mxfp8 (no bf16 fallback)");
      }
    }
    if (!tp2_real_only) {
      bool refused = false;
      try {
        TpOptions tp;
        tp.world = 2;
        tp.mode = TpOptions::Mode::kEmulate;
        (void)LoadTextModel(opts(mx, Layout::kMxfp8, 256), tp);
      } catch (const std::exception& e) {
        refused = true;
        std::printf("  mxfp8 TP=2 refused: %s\n", e.what());
      }
      Check(refused, "mxfp8 TP=2 on the tiny fixture is refused (expert down K 32 does not split into whole 32-K groups)");
    }
    const auto cross = [&](TextModel& m, const std::string& tag) {
      const std::vector<std::vector<float>> got = RunMode(m, g, bf16_bound, tag, nullptr, nullptr);
      double kl = 0;
      for (size_t i = 0; i < got.size() && i < bf16_tp1.size(); ++i) kl += Kl(bf16_tp1[i].data(), got[i].data(), g.V);
      kl /= static_cast<double>(std::max<size_t>(1, got.size()));
      std::printf("  %s vs bf16 TP=1: mean KL %.3e\n", tag.c_str(), kl);
      Check(kl <= 0.3, tag + " rows agree with the TP=1 rows (mean KL <= 0.3)");
    };
    if (!tp2_real_only) {
      TpOptions tp;
      tp.world = 2;
      tp.mode = TpOptions::Mode::kEmulate;
      std::unique_ptr<TextModel> m = LoadTextModel(opts(bf16, Layout::kBf16, 256), tp);
      Check(m->TpWorld() == 2 && m->Config().arch == Arch::kInkling2Loop, "bf16 TP=2 emulate: an inkling2 tensor-parallel model");
      cross(*m, "bf16 TP=2 emulate");
    }
    if (tp2_real_only || EnvOr("R4DX_INKLING2_TINY_TP_REAL", "0") == "1") {
      TpOptions tp;
      tp.world = 2;
      tp.mode = TpOptions::Mode::kReal;
      std::unique_ptr<TextModel> m = LoadTextModel(opts(bf16, Layout::kBf16, 256), tp);
      cross(*m, "bf16 TP=2 real");
    }
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
