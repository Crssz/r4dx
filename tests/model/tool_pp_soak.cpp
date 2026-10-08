// tests/model/tool_pp_soak.cpp -- the pipeline-parallel-prefill soak (docs/pp-prefill.md Phase 2, gate G2c).
//
// ONE PpModel (stage A on the desktop card, stage B the decode Model) for the whole run. Each iteration draws a random
// prompt slice (--min-prompt..--max-prompt tokens, default 2048..32768) of the kl_corpus token pool, a decode length and
// a mode, and runs the SAME generation twice from a fresh state: first monolithically on the decode Model alone (the
// reference: today's Model::Prefill and decode), then through the PpModel (pipelined prefill, decode unchanged). Every token
// of the two must be equal -- one divergence ends the soak (exit 2). Modes: greedy, seeded sampled (DecodeStepSampled;
// T 0.7, top_k 20, top_p 0.8), with --dflash a DFlash2 greedy round loop, and "turns": a prompt, some decode, then the
// conversation continues with a second prompt tail and more decode (the sync-back of the warm turn). --ref-every N runs the
// reference only every N-th iteration (the PpModel's own verify, --verify, covers the others).
//
// One JSON object per line is appended to --json (a "start" line, a "loaded" line, one "iter" line per iteration with the
// measured prefill times of both paths, an "error" line on failure, a "summary" line and a "teardown" line once the PpModel
// is destroyed), each flushed to disk before the next iteration starts, so a crash -- a TDR can take the display driver
// down with it -- still leaves every finished iteration on disk. On ANY error the soak stops at once without Reset()
// (a recovery would hide the failure it exists to find). Pass (exit 0): every comparison equal, this process's device
// buffers drifted <= 64 MiB on each device from right after load to the end. Exit 2: a token divergence; exit 3: another
// pass criterion failed. The TDR watch is tools/pp/soak.ps1's, which wraps this tool.
//
// Pre-flight: before loading, a helper thread requires --need-gib free on both devices.
//
// Built, never add_test()'d. Usage (HIP_VISIBLE_DEVICES unset, production server stopped; R4DX_PP_DEVICES=B,A picks the
// placement, default decode on the last visible ordinal = physical device 1):
//   tool_pp_soak --model <R4DX_MODELS_ROOT>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx --layout trellis --minutes 60
//                --max-ctx 36864 --json build\logs\pp_soak.jsonl [--seed 1] [--iterations N]
//                [--tokens tools\reference\kl_corpus\tokens.json] [--min-prompt 2048] [--max-prompt 32768]
//                [--min-decode 32] [--max-decode 256] [--ref-every 1] [--verify] [--pp-split N] [--pp-min-rows N]
//                [--need-gib 14] [--layers N] [--dflash <drafter.r4dx> [--dflash-k 7]]
#include <hip/hip_runtime.h>
#include <io.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "model.h"  // also nlohmann/json.hpp, via model_config.h
#include "pp_model.h"
#include "r4dx/kernels/sampler.hpp"
#include "test_common.h"

namespace {

using Clock = std::chrono::steady_clock;
using r4dx::model::ModelOptions;
using r4dx::model::PpModel;
using r4dx::model::PpOptions;
namespace kernels = r4dx::kernels;

constexpr double kMaxVramDriftMiB = 64.0;

struct Args {
  std::string model = r4dx_test::ProductionTargetPath();
  std::string layout = r4dx_test::ProductionLayoutName();
  std::string tokens = "tools/reference/kl_corpus/tokens.json";
  std::string json;
  double minutes = 60.0;
  int64_t iterations = 0;  // 0 = until --minutes
  int64_t max_ctx = 36864;
  int64_t layers = -1;
  uint64_t seed = 1;
  int min_prompt = 2048, max_prompt = 32768, min_decode = 32, max_decode = 256;
  int ref_every = 1;
  bool verify = false;
  int pp_split = 0, pp_min_rows = -1;
  double need_gib = 14.0;
  std::string dflash;
  int64_t dflash_k = 7;
};

// Rows beyond the prompt (and a "turns" tail) one iteration can feed: two decode runs, each overshooting by up to one
// DFlash round, and a margin.
int64_t DecodeSlack(int64_t max_decode, int64_t dflash_k) { return 2 * max_decode + 2 * (dflash_k + 1) + 64; }

[[noreturn]] void Usage(const std::string& why) {
  std::fprintf(stderr,
               "tool_pp_soak: %s\nusage: tool_pp_soak [--model <c.r4dx>] [--layout trellis] [--minutes 60] [--iterations N] "
               "[--max-ctx 36864] [--seed 1] --json <log.jsonl> [--tokens <tokens.json>] [--min-prompt 2048] "
               "[--max-prompt 32768] [--min-decode 32] [--max-decode 256] [--ref-every 1] [--verify] [--pp-split N] "
               "[--pp-min-rows N] [--need-gib X] [--layers N] [--dflash <drafter.r4dx> [--dflash-k 7]]\n",
               why.c_str());
  std::exit(1);
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Usage(s + " needs a value");
      return argv[++i];
    };
    try {
      if (s == "--model") a.model = next();
      else if (s == "--layout") a.layout = next();
      else if (s == "--tokens") a.tokens = next();
      else if (s == "--json") a.json = next();
      else if (s == "--minutes") a.minutes = std::stod(next());
      else if (s == "--iterations") a.iterations = std::stoll(next());
      else if (s == "--max-ctx") a.max_ctx = std::stoll(next());
      else if (s == "--layers") a.layers = std::stoll(next());
      else if (s == "--seed") a.seed = std::stoull(next());
      else if (s == "--min-prompt") a.min_prompt = std::stoi(next());
      else if (s == "--max-prompt") a.max_prompt = std::stoi(next());
      else if (s == "--min-decode") a.min_decode = std::stoi(next());
      else if (s == "--max-decode") a.max_decode = std::stoi(next());
      else if (s == "--ref-every") a.ref_every = std::stoi(next());
      else if (s == "--verify") a.verify = true;
      else if (s == "--pp-split") a.pp_split = std::stoi(next());
      else if (s == "--pp-min-rows") a.pp_min_rows = std::stoi(next());
      else if (s == "--need-gib") a.need_gib = std::stod(next());
      else if (s == "--dflash") a.dflash = next();
      else if (s == "--dflash-k") a.dflash_k = std::stoll(next());
      else Usage("unknown argument " + s);
    } catch (const std::exception&) {
      Usage("bad value for " + s);
    }
  }
  if (a.json.empty()) Usage("--json <log.jsonl> is required (the per-iteration log)");
  if (!(a.minutes > 0) && a.iterations <= 0) Usage("--minutes must be > 0 (or give --iterations)");
  if (a.ref_every < 1) Usage("--ref-every must be >= 1");
  if (a.min_prompt < 1 || a.max_prompt < a.min_prompt || a.min_decode < 1 || a.max_decode < a.min_decode) {
    Usage("need 1 <= --min-prompt <= --max-prompt and 1 <= --min-decode <= --max-decode");
  }
  // A "turns" iteration feeds a second tail (at most 4096 rows, clamped per iteration to what is left of --max-ctx) and
  // twice the decode length (plus DFlash rounds' slack): the prompt alone must leave room for the decode and one tail row.
  // (This used to demand room for the whole 4096-row tail too, which refused the documented default
  // --max-ctx 36864 with the default --max-prompt 32768.)
  if (a.max_prompt + DecodeSlack(a.max_decode, a.dflash_k) + 1 > a.max_ctx) {
    Usage("--max-prompt + 2 * --max-decode (+ slack) + 1 must fit in --max-ctx");
  }
  if (!a.dflash.empty() && (a.dflash_k < 1 || a.dflash_k > 7)) Usage("--dflash-k must be in [1, 7]");
  return a;
}

// ---- the log ------------------------------------------------------------------------------------------

class JsonLog {
 public:
  explicit JsonLog(const std::string& path) {
    if (fopen_s(&f_, path.c_str(), "ab") != 0) f_ = nullptr;
    if (f_ == nullptr) throw std::runtime_error("cannot open " + path + " for appending");
  }
  ~JsonLog() {
    if (f_ != nullptr) std::fclose(f_);
  }
  void Line(const std::string& obj) {
    std::fwrite(obj.data(), 1, obj.size(), f_);
    std::fputc('\n', f_);
    std::fflush(f_);
    (void)_commit(_fileno(f_));
  }

 private:
  FILE* f_ = nullptr;
};

std::string Str(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"') o += "\\\"";
    else if (c == '\\') o += "\\\\";
    else if (c == '\n') o += "\\n";
    else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else o += static_cast<char>(c);
  }
  return o + "\"";
}
std::string Num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, "%.6g", v);
  return b;
}
template <class T>
std::string Arr(const std::vector<T>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + Num(static_cast<double>(v[i]));
  return o + "]";
}

std::string Preflight(double need_gib) {
  std::string msg;
  std::thread t([&] {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) n = 0;
    if (n < 2) msg = "the pipeline needs two visible HIP devices (HIP_VISIBLE_DEVICES unset)";
    for (int d = 0; d < n && d < 2 && msg.empty(); ++d) {
      size_t free_b = 0, total_b = 0;
      if (hipSetDevice(d) != hipSuccess || hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        msg = "cannot query HIP device " + std::to_string(d);
        break;
      }
      const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
      if (free_gib < need_gib) {
        char b[256];
        std::snprintf(b, sizeof b, "need %.1f GiB free on HIP device %d, have %.2f GiB -- is the production server running?",
                      need_gib, d, free_gib);
        msg = b;
      }
    }
    (void)hipGetLastError();
  });
  t.join();
  return msg;
}

// ---- one generation, on either path ---------------------------------------------------------------------

enum class Mode { kGreedy, kSampled, kTurns, kDflash };
const char* ModeName(Mode m) {
  switch (m) {
    case Mode::kGreedy: return "greedy";
    case Mode::kSampled: return "sampled";
    case Mode::kTurns: return "turns";
    case Mode::kDflash: return "dflash";
  }
  return "?";
}

struct Gen {
  std::vector<int32_t> tokens;
  double prefill_s = 0, decode_s = 0;  // the FIRST prefill (the cold prompt) and all the decode
  int64_t rows = 0;
};

// `M` is the Model (the monolithic reference) or the PpModel; both have these calls. `parts` are the Prefill calls (more than
// one only in "turns"), with `decode` greedy / sampled / speculative steps after each.
template <class M>
Gen Generate(M& m, const std::vector<std::vector<int32_t>>& parts, int decode, Mode mode, uint64_t sample_seed, int64_t dflash_k) {
  Gen g;
  const int64_t V = m.Config().vocab_size;
  kernels::SampleParams sp;
  sp.temperature = 0.7f;
  sp.top_k = 20;
  sp.top_p = 0.8f;
  sp.seed = sample_seed;
  std::mt19937_64 rng = kernels::MakeRng(sample_seed);
  m.Reset();
  for (size_t p = 0; p < parts.size(); ++p) {
    const auto t0 = Clock::now();
    const std::vector<float> logits = m.Prefill(parts[p]);
    const double pf = std::chrono::duration<double>(Clock::now() - t0).count();
    if (p == 0) g.prefill_s = pf;
    g.rows += static_cast<int64_t>(parts[p].size());
    const bool sampled = mode == Mode::kSampled;
    int32_t next = sampled ? kernels::Sample(logits.data(), V, sp, rng) : kernels::Argmax(logits.data(), V);
    const auto t1 = Clock::now();
    if (mode == Mode::kDflash) {
      g.tokens.push_back(next);
      const size_t want = g.tokens.size() + static_cast<size_t>(decode);
      while (g.tokens.size() < want) {
        const std::vector<int32_t> r = m.DecodeStepDflashGreedy(next, dflash_k, 0.0f, 0);
        g.tokens.insert(g.tokens.end(), r.begin(), r.end());
        next = r.back();
      }
    } else {
      for (int i = 0; i < decode; ++i) {
        g.tokens.push_back(next);
        next = sampled ? m.DecodeStepSampled(next, sp, rng) : m.DecodeStepGreedy(next);
      }
      g.tokens.push_back(next);
    }
    g.decode_s += std::chrono::duration<double>(Clock::now() - t1).count();
    g.rows += decode;
  }
  return g;
}

struct Drift {
  std::vector<double> buffers_gib, used_gib;
};
Drift Snap(PpModel& m) {
  Drift d;
  for (const r4dx::model::VramReport& v : m.Vram()) {
    d.buffers_gib.push_back(v.buffers_gib);
    d.used_gib.push_back(v.used_gib);
  }
  return d;
}
std::vector<double> DriftMiB(const std::vector<double>& now, const std::vector<double>& first) {
  std::vector<double> d;
  for (size_t i = 0; i < now.size() && i < first.size(); ++i) d.push_back((now[i] - first[i]) * 1024.0);
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = Parse(argc, argv);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (!r4dx_test::FileExists(a.model)) {
    std::fprintf(stderr, "tool_pp_soak: container not found: %s\n", a.model.c_str());
    return 1;
  }
  const std::string pre = Preflight(a.need_gib);
  if (!pre.empty()) {
    std::fprintf(stderr, "tool_pp_soak: %s\n", pre.c_str());
    return 1;
  }
  std::unique_ptr<JsonLog> log;
  try {
    log = std::make_unique<JsonLog>(a.json);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_pp_soak: %s\n", e.what());
    return 1;
  }
  const auto wall0 = Clock::now();
  const auto since = [&] { return std::chrono::duration<double>(Clock::now() - wall0).count(); };
  const auto fail = [&](const char* what, const std::string& why, int64_t iter) {
    std::fprintf(stderr, "tool_pp_soak: %s at iteration %lld: %s\n", what, static_cast<long long>(iter), why.c_str());
    log->Line("{\"type\":\"error\",\"t_s\":" + Num(since()) + ",\"iter\":" + std::to_string(iter) + ",\"what\":" + Str(what) +
              ",\"error\":" + Str(why) + "}");
    return 1;
  };

  std::vector<int32_t> pool;
  try {
    std::ifstream f(a.tokens, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + a.tokens);
    nlohmann::json j;
    f >> j;
    for (const auto& s : j.at("segments")) {
      for (const auto& t : s.at("token_ids")) pool.push_back(t.get<int32_t>());
    }
  } catch (const std::exception& e) {
    return fail("tokens", e.what(), -1);
  }
  if (pool.size() < static_cast<size_t>(a.max_prompt)) {
    return fail("tokens", "the token pool has fewer tokens than --max-prompt", -1);
  }

  ModelOptions opts;
  opts.container_path = a.model;
  opts.layout = r4dx::model::LayoutFromName(a.layout);
  opts.max_ctx = a.max_ctx;
  opts.layer_limit = a.layers;
  opts.vision = ModelOptions::VisionMode::kOff;
  const bool dflash = !a.dflash.empty();
  if (dflash) {
    opts.dflash_container = a.dflash;
    opts.dflash_draft_k = a.dflash_k;
  }
  PpOptions po;
  po.split = a.pp_split;
  if (a.pp_min_rows > 0) po.min_rows = a.pp_min_rows;
  po.verify = a.verify;

  log->Line("{\"type\":\"start\",\"model\":" + Str(a.model) + ",\"layout\":" + Str(a.layout) + ",\"minutes\":" + Num(a.minutes) +
            ",\"iterations\":" + std::to_string(a.iterations) + ",\"max_ctx\":" + std::to_string(a.max_ctx) + ",\"seed\":" +
            std::to_string(a.seed) + ",\"pp_split\":" + std::to_string(a.pp_split) + ",\"pp_min_rows\":" +
            std::to_string(po.min_rows) + ",\"verify\":" + (a.verify ? "true" : "false") + ",\"ref_every\":" +
            std::to_string(a.ref_every) + ",\"pool_tokens\":" + std::to_string(pool.size()) + ",\"dflash\":" + Str(a.dflash) +
            ",\"dflash_k\":" + std::to_string(dflash ? a.dflash_k : 0) + "}");

  std::unique_ptr<PpModel> m;
  const auto load0 = Clock::now();
  try {
    m = PpModel::Load(opts, po);
  } catch (const std::exception& e) {
    return fail("load", e.what(), -1);
  }
  const double load_s = std::chrono::duration<double>(Clock::now() - load0).count();
  std::printf("[soak] loaded in %.1f s (split %lld)\n", load_s, static_cast<long long>(m->Split()));
  Drift first;
  try {
    first = Snap(*m);
  } catch (const std::exception& e) {
    return fail("first snapshot", e.what(), -1);
  }
  log->Line("{\"type\":\"loaded\",\"t_s\":" + Num(since()) + ",\"load_s\":" + Num(load_s) + ",\"split\":" +
            std::to_string(m->Split()) + ",\"buffers_gib\":" + Arr(first.buffers_gib) + ",\"vram_used_gib\":" +
            Arr(first.used_gib) + "}");

  std::mt19937_64 rng(a.seed);
  int64_t iter = 0, compared = 0, tokens_total = 0;
  bool diverged = false;
  double best_speedup = 0, sum_pp_s = 0, sum_ref_s = 0;
  try {
    for (;; ++iter) {
      if (a.iterations > 0 ? iter >= a.iterations : since() >= a.minutes * 60.0) break;
      std::uniform_int_distribution<int> plen(a.min_prompt, a.max_prompt), dlen(a.min_decode, a.max_decode),
          mode_d(0, dflash ? 3 : 2), tail_d(1, 4096);
      std::uniform_int_distribution<size_t> start_d(0, pool.size() - 1);
      const int P = plen(rng), D = dlen(rng);
      const Mode mode = static_cast<Mode>(mode_d(rng));  // 0..2 (greedy, sampled, turns), 3 = DFlash with --dflash
      const size_t start = start_d(rng);
      // The tail leaves room in --max-ctx for the prompt and both decode runs (Parse guarantees at least one row).
      const int tail = static_cast<int>(
          std::max<int64_t>(1, std::min<int64_t>(tail_d(rng), a.max_ctx - P - DecodeSlack(D, a.dflash_k))));
      const uint64_t sample_seed = rng();
      std::vector<std::vector<int32_t>> parts(mode == Mode::kTurns ? 2 : 1);
      parts[0].resize(static_cast<size_t>(P));
      for (int i = 0; i < P; ++i) parts[0][static_cast<size_t>(i)] = pool[(start + static_cast<size_t>(i)) % pool.size()];
      if (mode == Mode::kTurns) {
        parts[1].resize(static_cast<size_t>(tail));
        for (int i = 0; i < tail; ++i) parts[1][static_cast<size_t>(i)] = pool[(start + static_cast<size_t>(P + i)) % pool.size()];
      }

      const bool ref = iter % a.ref_every == 0;
      Gen reference;
      if (ref) reference = Generate(m->DecodeModel(), parts, D, mode, sample_seed, a.dflash_k);
      const Gen g = Generate(*m, parts, D, mode, sample_seed, a.dflash_k);
      tokens_total += g.rows;
      std::string cmp = "\"skipped\"";
      double speedup = 0;
      if (ref) {
        ++compared;
        if (g.tokens == reference.tokens) {
          cmp = "\"equal\"";
        } else {
          size_t k = 0;
          while (k < g.tokens.size() && k < reference.tokens.size() && g.tokens[k] == reference.tokens[k]) ++k;
          cmp = "\"DIFF at token " + std::to_string(k) + "\"";
          diverged = true;
        }
        speedup = reference.prefill_s / g.prefill_s;
        best_speedup = std::max(best_speedup, speedup);
        sum_pp_s += g.prefill_s;
        sum_ref_s += reference.prefill_s;
      }
      const Drift s = Snap(*m);
      log->Line("{\"type\":\"iter\",\"iter\":" + std::to_string(iter) + ",\"t_s\":" + Num(since()) + ",\"mode\":\"" +
                ModeName(mode) + "\",\"prompt_tokens\":" + std::to_string(P) + ",\"tail_tokens\":" +
                std::to_string(mode == Mode::kTurns ? tail : 0) + ",\"decode_tokens\":" + std::to_string(D) + ",\"start\":" +
                std::to_string(start) + ",\"pp_prefill_s\":" + Num(g.prefill_s) + ",\"mono_prefill_s\":" +
                Num(ref ? reference.prefill_s : 0.0) + ",\"prefill_speedup\":" + Num(speedup) + ",\"pp_decode_tok_s\":" +
                Num(static_cast<double>(D) / g.decode_s) + ",\"mono_decode_tok_s\":" +
                Num(ref ? static_cast<double>(D) / reference.decode_s : 0.0) + ",\"compare\":" + cmp +
                ",\"buffer_drift_mib\":" + Arr(DriftMiB(s.buffers_gib, first.buffers_gib)) + ",\"vram_used_drift_mib\":" +
                Arr(DriftMiB(s.used_gib, first.used_gib)) + "}");
      std::printf("[soak] iter %lld  %.0f s  %-7s prompt %5d  decode %3d  prefill pp %.2f s%s  %s\n",
                  static_cast<long long>(iter), since(), ModeName(mode), P, D, g.prefill_s,
                  ref ? (std::string("  mono ") + Num(reference.prefill_s) + " s  x" + Num(speedup)).c_str() : "",
                  ref ? cmp.c_str() : "");
      if (diverged) break;
    }
  } catch (const std::exception& e) {
    // No Reset(), no retry: the first error ends the soak.
    return fail("PpModel error", e.what(), iter);
  }

  Drift last;
  try {
    last = Snap(*m);
  } catch (const std::exception& e) {
    return fail("final snapshot", e.what(), iter);
  }
  const std::vector<double> drift = DriftMiB(last.buffers_gib, first.buffers_gib),
                            used_drift = DriftMiB(last.used_gib, first.used_gib);
  double worst_drift = 0;
  for (double d : drift) worst_drift = std::max(worst_drift, std::fabs(d));
  std::vector<std::string> failures;
  if (diverged) failures.push_back("token divergence between the monolithic and the pipelined run");
  if (worst_drift > kMaxVramDriftMiB) failures.push_back("buffer drift " + Num(worst_drift) + " MiB > 64 MiB");
  if (iter == 0) failures.push_back("no iteration ran");
  if (compared == 0) failures.push_back("no iteration was compared against the reference");
  const int code = failures.empty() ? 0 : diverged ? 2 : 3;
  std::string fl = "[";
  for (size_t i = 0; i < failures.size(); ++i) fl += (i ? "," : "") + Str(failures[i]);
  fl += "]";
  log->Line("{\"type\":\"summary\",\"t_s\":" + Num(since()) + ",\"load_s\":" + Num(load_s) + ",\"iterations\":" +
            std::to_string(iter) + ",\"compared\":" + std::to_string(compared) + ",\"tokens\":" + std::to_string(tokens_total) +
            ",\"buffer_drift_mib\":" + Arr(drift) + ",\"vram_used_drift_mib\":" + Arr(used_drift) +
            ",\"mean_prefill_speedup\":" + Num(sum_pp_s > 0 ? sum_ref_s / sum_pp_s : 0.0) + ",\"best_prefill_speedup\":" +
            Num(best_speedup) + ",\"pp\":" + Str(m->StatsLine()) + ",\"exit_code\":" + std::to_string(code) +
            ",\"failures\":" + fl + "}");
  std::printf("[soak] %s: %lld iterations (%lld compared), %lld tokens in %.0f s; buffer drift %s MiB; %s\n",
              code == 0 ? "PASS" : "FAIL", static_cast<long long>(iter), static_cast<long long>(compared),
              static_cast<long long>(tokens_total), since(), Arr(drift).c_str(), m->StatsLine().c_str());
  for (const std::string& f : failures) std::printf("[soak]   %s\n", f.c_str());
  m.reset();
  log->Line("{\"type\":\"teardown\",\"t_s\":" + Num(since()) + ",\"exit_code\":" + std::to_string(code) + "}");
  std::printf("[soak] PpModel torn down; exiting with code %d\n", code);
  return code;
}
