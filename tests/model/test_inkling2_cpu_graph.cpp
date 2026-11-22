// tests/model/test_inkling2_cpu_graph.cpp -- CPU-only (no HIP). docs/inkling2-plan.md 3.3 / 5: the Inkling2Model graph
// (tests/model/inkling2_cpu_graph.h, Inkling2Model::RunChunk's state machine line for line over inkling2_ref.hpp) on the
// TINY fixture against HF's fp32 logits (tests/convert/data/inkling2_tiny/logits.npy, every row of tokens.json's 40).
//
// The fixture: R4DX_INKLING2_TINY_DIR (default %TEMP%\r4dx_inkling2_tiny_e2e) holding tiny-bf16.r4dx, tiny-mxfp8.r4dx and
// the checkpoint ckpt/ (test_inkling2_tiny_e2e's header says how they are made; SKIP 77 without tiny-bf16.r4dx).
//   1. EXACT mode (fp32 throughout, experts decoded from the checkpoint; needs ckpt/model.safetensors): the semantics check
//      -- decode feed vs HF fp32: mean KL <= 1e-6, every row < 1e-5, top-1 40/40 (measured 2026-11-22: mean 6.4e-12).
//   2. the GPU rounding points on tiny-bf16.r4dx and on tiny-mxfp8.r4dx: reported (the tiny random model is chaotic:
//      bf16 rounding flips its top-2-of-8 routing, so these are not bounded against HF), and a sanity bound (mean KL < 1).
//   In every mode: chunked prefill (chunks 7 and 16 up to row 19, then decode) == the decode feed, bit for bit.
// R4DX_INKLING2_CPU_DUMP=<file>: also writes every visit's output h for the first 3 tokens (f32 [visits, 3, H] + the
// last logits row), in the order HF's decoder-layer forward hooks fire (a debugging aid).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "inkling2_cpu_graph.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using inkling2_cpu::Vec;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

nlohmann::json ContainerMetadata(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 8);
  std::string h(static_cast<size_t>(n), '\0');
  f.read(h.data(), static_cast<std::streamsize>(n));
  return nlohmann::json::parse(h).at("__metadata__");
}

struct Stats {
  double mean = 0, worst = 0;
  int top1 = 0;
};

// Runs one mode; returns its decode rows.
std::vector<Vec> RunMode(const std::string& tag, const fs::path& container, const fs::path* ckpt_dir, const Vec& golden,
                         int64_t N, int64_t V, const std::vector<int32_t>& ids, Stats* out) {
  const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(container.u8string()));
  std::unique_ptr<r4dx_convert::SafetensorsReader> ck;
  inkling2_cpu::Engine e;
  e.cfg = inkling2_cpu::Inkling2Config::FromModelConfig(ContainerMetadata(container).at("model_config"));
  e.w.r = &r;
  e.visits = e.cfg.Visits();
  inkling2_cpu::g_exact = ckpt_dir != nullptr;
  if (ckpt_dir != nullptr) {
    ck = std::make_unique<r4dx_convert::SafetensorsReader>(r4dx_convert::Utf8ToWide((*ckpt_dir / "model.safetensors").u8string()));
    e.w.ckpt = ck.get();
  }
  Check(e.cfg.unpadded_vocab_size == V && static_cast<int64_t>(ids.size()) == N, tag + ": the golden matches the container");
  if (const char* dp = std::getenv("R4DX_INKLING2_CPU_DUMP"); dp != nullptr && *dp != '\0' && ckpt_dir != nullptr) {
    Vec d;
    e.dump = &d;
    e.Reset();
    const Vec lg = e.Chunk(std::vector<int32_t>(ids.begin(), ids.begin() + 3));
    e.dump = nullptr;
    d.insert(d.end(), lg.begin(), lg.end());
    std::ofstream o(dp, std::ios::binary);
    o.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size() * 4));
    std::printf("  wrote %zu floats to %s\n", d.size(), dp);
  }
  const auto feed = [&](int chunk) {
    e.Reset();
    std::vector<Vec> rows(static_cast<size_t>(N));
    int64_t p = 0;
    if (chunk > 1) {
      for (; p < 20; p += std::min<int64_t>(chunk, 20 - p)) {
        const int64_t n = std::min<int64_t>(chunk, 20 - p);
        rows[static_cast<size_t>(p + n - 1)] = e.Chunk(std::vector<int32_t>(ids.begin() + p, ids.begin() + p + n));
      }
    }
    for (; p < N; ++p) rows[static_cast<size_t>(p)] = e.Chunk({ids[static_cast<size_t>(p)]});
    return rows;
  };
  const std::vector<Vec> dec = feed(1);
  Stats s;
  for (int64_t p = 0; p < N; ++p) {
    const float* g = golden.data() + p * V;
    const double kl = inkling2_cpu::Kl(g, dec[static_cast<size_t>(p)].data(), V);
    s.mean += kl;
    s.worst = std::max(s.worst, kl);
    s.top1 += (std::max_element(g, g + V) - g) ==
              (std::max_element(dec[static_cast<size_t>(p)].begin(), dec[static_cast<size_t>(p)].end()) - dec[static_cast<size_t>(p)].begin());
  }
  s.mean /= static_cast<double>(N);
  std::printf("  %s: decode feed vs HF fp32: mean KL %.3e, worst %.3e, top-1 %d/%lld\n", tag.c_str(), s.mean, s.worst, s.top1,
              static_cast<long long>(N));
  for (int chunk : {7, 16}) {
    const std::vector<Vec> c = feed(chunk);
    bool same = true;
    for (int64_t p = 0; p < N; ++p)
      if (!c[static_cast<size_t>(p)].empty()) same = same && c[static_cast<size_t>(p)] == dec[static_cast<size_t>(p)];
    Check(same, tag + ": chunked prefill (chunk " + std::to_string(chunk) + ") == the decode feed, bit for bit");
  }
  *out = s;
  inkling2_cpu::g_exact = false;
  return dec;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    const char* env = std::getenv("R4DX_INKLING2_TINY_DIR");
    const fs::path dir = env != nullptr && *env != '\0' ? fs::path(env) : fs::temp_directory_path() / "r4dx_inkling2_tiny_e2e";
    const fs::path bf16 = dir / "tiny-bf16.r4dx", mx = dir / "tiny-mxfp8.r4dx";
    if (!fs::exists(bf16)) {
      std::printf("SKIP: %s (see test_inkling2_tiny_e2e.cpp's header)\n", bf16.u8string().c_str());
      return 77;
    }
    const fs::path data = fs::path(R4DX_SOURCE_DIR) / "tests" / "convert" / "data" / "inkling2_tiny";
    int64_t N = 0, V = 0;
    const Vec golden = inkling2_cpu::ReadNpy(data / "logits.npy", &N, &V);
    std::vector<int32_t> ids;
    {
      std::ifstream f(data / "tokens.json");
      const nlohmann::json tj = nlohmann::json::parse(f);
      for (const auto& t : tj.at("tokens")) ids.push_back(t.get<int32_t>());
    }
    fs::path ckpt = data;
    if (!fs::exists(ckpt / "model.safetensors")) ckpt = dir / "ckpt";
    Stats s;
    if (fs::exists(ckpt / "model.safetensors")) {
      RunMode("exact (fp32, checkpoint experts)", bf16, &ckpt, golden, N, V, ids, &s);
      Check(s.mean <= 1e-6 && s.worst < 1e-5 && s.top1 == N, "exact mode reproduces HF fp32 (mean KL <= 1e-6, rows < 1e-5, top-1 40/40)");
    } else {
      std::printf("SKIP: the exact mode (no checkpoint at %s)\n", (ckpt / "model.safetensors").u8string().c_str());
    }
    RunMode("bf16 container, GPU rounding points", bf16, nullptr, golden, N, V, ids, &s);
    Check(s.mean < 1.0, "bf16 container: sanity bound (mean KL < 1; the tiny model's routing is not bf16-stable)");
    if (fs::exists(mx)) {
      RunMode("mxfp8 container, GPU rounding points", mx, nullptr, golden, N, V, ids, &s);
      Check(s.mean < 1.0, "mxfp8 container: sanity bound (mean KL < 1)");
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "FAIL: unexpected exception: %s\n", ex.what());
    return 1;
  }
  if (g_failures != 0) {
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
