// DFlash2 on the real Ornith-1.5-122B container, in ONE process (the model loads once): for every prompt, a plain greedy run and
// DFlash2 greedy runs at several draft lengths k (and p_min), timed the same way, with the lossless audit and the acceptance statistics.
//
//   tool_q35moe_dflash --model <container.r4dx> --dflash <drafter.r4dx> --set <tokens.json>:<prompt_len> [--set ...]
//                      [--gen N] [--k 3,5,7,15] [--pmin 0,0.3] [--reps 2] [--max-ctx 4096] [--name a,b,c]
//
// <tokens.json> is r4dx-cli's --dump-token-ids file (the chat-templated prompt ids first; <prompt_len> of them are the prompt, as the
// CLI's "[stats] prefill: N tok" line says).
//
// THE LOSSLESS AUDIT. The plain run is the reference stream S. A DFlash2 run must produce S token for token, except that a verify
// window runs batched GEMMs (M = k + 1 rows) where plain decode runs M = 1 kernels, so the two round differently and a token whose
// top two logits are within that noise can come out the other way (docs/dflash2.md section 7a item 4). The tool therefore measures
// the noise itself and judges every divergence against it:
//   1. the reference run records, per position, the decode logit of the token it picked (= the row maximum);
//   2. a teacher-forced pass feeds S through verify windows of 16 rows and compares each row with the decode run: the difference of the
//      logit of S's token between the two paths (the NOISE: mean / p99 / max), and the rows whose argmax is not S's token (FLIPS), with
//      the gap they flipped by;
//   3. at a divergence of a DFlash2 stream the decode logits of that position are replayed (plain DecodeStep, the exact numerics
//      that produced S) and the gap between S's token and DFlash2's is reported; it is judged a near tie when it is within the
//      measured max noise (x 1.5).
// Output ends with a table (one row per prompt and k), tab-friendly for pasting into docs/qwen35moe-plan.md.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "qwen35moe_model.h"

using namespace r4dx::model;
using Clock = std::chrono::steady_clock;

namespace {

double MsSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct PromptSet {
  std::string name;
  std::vector<int32_t> prompt;
};

std::vector<std::string> Split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, c)) out.push_back(tok);
  return out;
}

PromptSet LoadSet(const std::string& spec, const std::string& name) {
  const size_t colon = spec.rfind(':');
  if (colon == std::string::npos) throw std::runtime_error("--set wants <tokens.json>:<prompt_len>");
  const std::string path = spec.substr(0, colon);
  const size_t plen = static_cast<size_t>(std::stoll(spec.substr(colon + 1)));
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  const nlohmann::json j = nlohmann::json::parse(f);
  const std::vector<int32_t> ids = j.at("segments").at(0).at("token_ids").get<std::vector<int32_t>>();
  if (plen == 0 || plen > ids.size()) throw std::runtime_error("prompt_len out of range for " + path);
  PromptSet p;
  p.name = name;
  p.prompt.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(plen));
  return p;
}

struct Run {
  std::vector<int32_t> tokens;  // N generated tokens (the first is the prefill's)
  double prefill_ms = 0, decode_ms = 0;
  int64_t rounds = 0, drafted = 0, accepted = 0;
};

int32_t ArgMax(const std::vector<float>& v) { return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

Run PlainRun(Qwen35MoeModel& m, const PromptSet& p, int gen) {
  m.Reset();
  Run r;
  const auto t0 = Clock::now();
  const std::vector<float> lg = m.Prefill(p.prompt);
  r.prefill_ms = MsSince(t0);
  r.tokens.push_back(ArgMax(lg));
  const auto t1 = Clock::now();
  while (static_cast<int>(r.tokens.size()) < gen) r.tokens.push_back(m.DecodeStepGreedy(r.tokens.back()));
  r.decode_ms = MsSince(t1);
  return r;
}

// The reference stream with its decode logits: top1[i] = the decode logit of tokens[i] (the row maximum), margin[i] = top1 - top2.
struct Reference {
  std::vector<int32_t> tokens;
  std::vector<float> top1, margin;
};

Reference ReferenceRun(Qwen35MoeModel& m, const PromptSet& p, int gen) {
  m.Reset();
  Reference r;
  auto note = [&](const std::vector<float>& lg) {
    const int32_t t = ArgMax(lg);
    float second = -1e30f;
    for (size_t i = 0; i < lg.size(); ++i) {
      if (static_cast<int32_t>(i) != t) second = std::max(second, lg[i]);
    }
    r.tokens.push_back(t);
    r.top1.push_back(lg[static_cast<size_t>(t)]);
    r.margin.push_back(lg[static_cast<size_t>(t)] - second);
  };
  note(m.Prefill(p.prompt));
  while (static_cast<int>(r.tokens.size()) < gen) note(m.DecodeStep(r.tokens.back()));
  return r;
}

struct Audit {
  int64_t rows = 0, flips = 0;
  double noise_mean = 0, noise_p99 = 0, noise_max = 0;  // |verify logit - decode logit| of S's token
  double worst_flip_gap = 0;                            // verify row max - verify logit of S's token, over the flips
};

// Teacher-forced: S through verify windows of 16 rows (all committed), each row against the decode run.
Audit AuditRun(Qwen35MoeModel& m, const PromptSet& p, const Reference& ref, int window, bool verbose) {
  m.Reset();
  Audit a;
  std::vector<double> noise;
  std::vector<double> by_row_sum(16, 0.0), by_row_max(16, 0.0);
  std::vector<int64_t> by_row_n(16, 0);
  struct Bad { double d; int pos, row; double margin; };
  std::vector<Bad> worst;
  const std::vector<float> lg = m.Prefill(p.prompt);
  (void)lg;
  const int n = static_cast<int>(ref.tokens.size());
  const int64_t vocab = m.Config().vocab_size;
  int i = 0;  // ref.tokens[i] is the next token to feed
  while (i + 1 < n) {
    const int w = std::min<int>(window, n - 1 - i);
    std::vector<int32_t> cands(ref.tokens.begin() + i, ref.tokens.begin() + i + w);
    const std::vector<int32_t> am = m.VerifyWindow(cands);
    const std::vector<float> vl = m.LastVerifyLogits();
    for (int r = 0; r < w; ++r) {
      const int32_t want = ref.tokens[static_cast<size_t>(i + r + 1)];
      const float* row = vl.data() + static_cast<size_t>(r) * vocab;
      const double at = row[want];
      const double dn = std::fabs(at - ref.top1[static_cast<size_t>(i + r + 1)]);
      noise.push_back(dn);
      by_row_sum[static_cast<size_t>(r)] += dn;
      by_row_max[static_cast<size_t>(r)] = std::max(by_row_max[static_cast<size_t>(r)], dn);
      ++by_row_n[static_cast<size_t>(r)];
      if (dn > 2.0) worst.push_back({dn, i + r + 1, r, static_cast<double>(ref.margin[static_cast<size_t>(i + r + 1)])});
      ++a.rows;
      if (am[static_cast<size_t>(r)] != want) {
        ++a.flips;
        a.worst_flip_gap = std::max(a.worst_flip_gap, static_cast<double>(row[am[static_cast<size_t>(r)]]) - at);
      }
    }
    m.CommitVerifiedWindow(w);
    i += w;
  }
  if (verbose) {
    std::printf("   audit window %d, noise by row index (mean/max):", window);
    for (int r = 0; r < window && r < 16; ++r) {
      if (by_row_n[static_cast<size_t>(r)] > 0) std::printf(" r%d %.2f/%.1f", r, by_row_sum[static_cast<size_t>(r)] / static_cast<double>(by_row_n[static_cast<size_t>(r)]), by_row_max[static_cast<size_t>(r)]);
    }
    std::printf("\n");
    std::sort(worst.begin(), worst.end(), [](const Bad& x, const Bad& y) { return x.d > y.d; });
    std::printf("   rows with |diff| > 2: %zu of %zu; worst:", worst.size(), noise.size());
    for (size_t j = 0; j < worst.size() && j < 8; ++j) std::printf(" [pos %d row %d diff %.1f dec-margin %.2f]", worst[j].pos, worst[j].row, worst[j].d, worst[j].margin);
    std::printf("\n");
  }
  std::sort(noise.begin(), noise.end());
  double sum = 0;
  for (double v : noise) sum += v;
  if (!noise.empty()) {
    a.noise_mean = sum / static_cast<double>(noise.size());
    a.noise_p99 = noise[static_cast<size_t>(0.99 * static_cast<double>(noise.size() - 1))];
    a.noise_max = noise.back();
  }
  return a;
}

// Plain decode of the reference stream S with the logits of the NEXT token S[i+1] kept (teacher forced): compares one decode
// implementation with another (e.g. the grouped M = 1 MoE against the host-routed one) on the very same history.
Audit ForcedDecodeAudit(Qwen35MoeModel& m, const PromptSet& p, const Reference& ref) {
  m.Reset();
  Audit a;
  std::vector<double> noise;
  (void)m.Prefill(p.prompt);
  for (size_t i = 0; i + 1 < ref.tokens.size(); ++i) {
    const std::vector<float> lg = m.DecodeStep(ref.tokens[i]);
    const int32_t want = ref.tokens[i + 1];
    noise.push_back(std::fabs(static_cast<double>(lg[static_cast<size_t>(want)]) - ref.top1[i + 1]));
    ++a.rows;
    if (ArgMax(lg) != want) ++a.flips;
  }
  std::sort(noise.begin(), noise.end());
  double sum = 0;
  for (double v : noise) sum += v;
  a.noise_mean = sum / static_cast<double>(noise.size());
  a.noise_p99 = noise[static_cast<size_t>(0.99 * static_cast<double>(noise.size() - 1))];
  a.noise_max = noise.back();
  return a;
}

Run DflashRun(Qwen35MoeModel& m, const PromptSet& p, int gen, int k, float p_min) {
  m.Reset();
  Run r;
  const auto t0 = Clock::now();
  const std::vector<float> lg = m.Prefill(p.prompt);
  r.prefill_ms = MsSince(t0);
  r.tokens.push_back(ArgMax(lg));
  const auto t1 = Clock::now();
  while (static_cast<int>(r.tokens.size()) < gen) {
    int64_t walk = 0;
    const std::vector<int32_t> round = m.DecodeStepDflashGreedy(r.tokens.back(), k, p_min, 0, &walk);
    ++r.rounds;
    r.drafted += walk;
    r.accepted += static_cast<int64_t>(round.size()) - 1;
    for (int32_t t : round) r.tokens.push_back(t);
  }
  r.decode_ms = MsSince(t1);
  r.tokens.resize(static_cast<size_t>(gen));
  return r;
}

}  // namespace

int main(int argc, char** argv) try {
  std::string model, dflash, names;
  std::vector<std::string> sets;
  int gen = 400, reps = 2, max_ctx = 4096;
  std::vector<int> ks = {3, 5, 7, 15};
  std::vector<float> pmins = {0.0f};
  std::vector<int> audit_windows;
  bool audit_only = false, host_moe_compare = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--dflash") dflash = next();
    else if (a == "--set") sets.push_back(next());
    else if (a == "--name") names = next();
    else if (a == "--gen") gen = std::stoi(next());
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--max-ctx") max_ctx = std::stoi(next());
    else if (a == "--k") {
      ks.clear();
      for (const std::string& s : Split(next(), ',')) ks.push_back(std::stoi(s));
    } else if (a == "--audit-windows") {
      for (const std::string& s : Split(next(), ',')) audit_windows.push_back(std::stoi(s));
    } else if (a == "--audit-only") {
      audit_only = true;
    } else if (a == "--host-moe-compare") {
      host_moe_compare = true;
    } else if (a == "--pmin") {
      pmins.clear();
      for (const std::string& s : Split(next(), ',')) pmins.push_back(std::stof(s));
    } else {
      throw std::runtime_error("unknown argument " + a);
    }
  }
  if (model.empty() || dflash.empty() || sets.empty()) {
    std::fprintf(stderr, "usage: tool_q35moe_dflash --model <c.r4dx> --dflash <d.r4dx> --set <tokens.json>:<prompt_len> ... [--gen N] [--k 3,5,7,15] [--pmin 0,0.3] [--reps 2] [--name a,b]\n");
    return 2;
  }
  const std::vector<std::string> nm = Split(names, ',');
  std::vector<PromptSet> prompts;
  for (size_t i = 0; i < sets.size(); ++i) prompts.push_back(LoadSet(sets[i], i < nm.size() ? nm[i] : "p" + std::to_string(i)));

  Qwen35MoeModelOptions o;
  o.container_path = model;
  o.max_ctx = max_ctx;
  o.dflash_container = dflash;
  o.dflash_draft_k = *std::max_element(ks.begin(), ks.end());
  auto m = Qwen35MoeModel::Load(o);

  std::vector<std::string> rows;
  char buf[640];
  int ties = 0, fails = 0;
  for (const PromptSet& p : prompts) {
    std::printf("\n== %s: %zu prompt tokens, %d generated\n", p.name.c_str(), p.prompt.size(), gen);
    m->SetDflashInjectionEnabled(false);  // the drafter is not what the plain run and the audit measure
    const Reference ref = ReferenceRun(*m, p, gen);
    const Audit au = AuditRun(*m, p, ref, static_cast<int>(m->SpecWindow()), true);
    for (int w : audit_windows) {
      const Audit aw = AuditRun(*m, p, ref, w, true);
      std::printf("   window %d: noise mean %.3f p99 %.3f max %.3f, flips %lld of %lld\n", w, aw.noise_mean, aw.noise_p99, aw.noise_max, static_cast<long long>(aw.flips), static_cast<long long>(aw.rows));
    }
    if (host_moe_compare) {
      m->SetGroupedDecode(false);
      const Audit ah = ForcedDecodeAudit(*m, p, ref);
      m->SetGroupedDecode(true);
      std::printf("   decode with the HOST-ROUTED M = 1 MoE vs the grouped one (teacher forced): noise mean %.3f p99 %.3f max %.3f, flips %lld of %lld\n",
                  ah.noise_mean, ah.noise_p99, ah.noise_max, static_cast<long long>(ah.flips), static_cast<long long>(ah.rows));
    }
    if (audit_only) continue;
    PlainRun(*m, p, gen);  // warm-up
    double plain_ms = 1e30;
    for (int r = 0; r < reps; ++r) plain_ms = std::min(plain_ms, PlainRun(*m, p, gen).decode_ms);
    m->SetDflashInjectionEnabled(true);
    const double plain_tps = (gen - 1) * 1000.0 / plain_ms;
    std::printf("   plain: %.1f tok/s (%.1f ms/token)\n", plain_tps, plain_ms / (gen - 1));
    std::printf("   verify-vs-decode audit over %lld teacher-forced rows: logit noise mean %.3f, p99 %.3f, max %.3f; %lld argmax flips (%.2f%%), worst flip gap %.3f\n",
                static_cast<long long>(au.rows), au.noise_mean, au.noise_p99, au.noise_max, static_cast<long long>(au.flips),
                100.0 * static_cast<double>(au.flips) / static_cast<double>(std::max<int64_t>(1, au.rows)), au.worst_flip_gap);
    std::snprintf(buf, sizeof(buf), "%s\tplain\t-\t%.1f\t1.00x\t1.00\t-\t-\tnoise p99 %.2f max %.2f, flips %.2f%%", p.name.c_str(), plain_tps,
                  au.noise_p99, au.noise_max, 100.0 * static_cast<double>(au.flips) / static_cast<double>(std::max<int64_t>(1, au.rows)));
    rows.push_back(buf);
    const double tie_bound = 1.5 * std::max(au.noise_max, 0.25);

    for (int k : ks) {
      for (float pm : pmins) {
        Run best;
        double best_ms = 1e30;
        for (int r = 0; r < reps; ++r) {
          Run run = DflashRun(*m, p, gen, k, pm);
          if (run.decode_ms < best_ms) {
            best_ms = run.decode_ms;
            best = std::move(run);
          }
        }
        size_t agree = 0;
        while (agree < best.tokens.size() && best.tokens[agree] == ref.tokens[agree]) ++agree;
        std::string note = "identical";
        if (agree < best.tokens.size()) {
          // replay the reference's decode numerics up to the divergence and look at the logits there
          m->SetDflashInjectionEnabled(false);
          m->Reset();
          std::vector<float> lg = m->Prefill(p.prompt);
          for (size_t i = 0; i < agree; ++i) lg = m->DecodeStep(ref.tokens[i]);
          m->SetDflashInjectionEnabled(true);
          const float top = *std::max_element(lg.begin(), lg.end());
          const float want = lg[static_cast<size_t>(ref.tokens[agree])], got = lg[static_cast<size_t>(best.tokens[agree])];
          const double gap = static_cast<double>(want) - static_cast<double>(got);
          const bool tie = gap <= tie_bound;
          std::snprintf(buf, sizeof(buf), "diverges at token %zu: plain %d (decode logit %.3f) vs dflash %d (decode logit %.3f), gap %.3f vs noise bound %.3f -> %s",
                        agree, ref.tokens[agree], want, best.tokens[agree], got, gap, tie_bound, tie ? "within the verify-vs-decode noise" : "NOT EXPLAINED");
          (void)top;
          note = buf;
          (tie ? ties : fails)++;
        }
        const double tps = (gen - 1) * 1000.0 / best_ms;
        const double per_round = static_cast<double>(best.tokens.size() - 1) / static_cast<double>(std::max<int64_t>(1, best.rounds));
        const double acc_rate = best.drafted > 0 ? 100.0 * static_cast<double>(best.accepted) / static_cast<double>(best.drafted) : 0.0;
        std::printf("   k=%-2d p_min=%.2f: %.1f tok/s (%.2fx), %.2f tokens/round over %lld rounds (%.1f drafted/round), %.1f%% of drafts accepted; %s\n", k, pm,
                    tps, tps / plain_tps, per_round, static_cast<long long>(best.rounds),
                    static_cast<double>(best.drafted) / static_cast<double>(std::max<int64_t>(1, best.rounds)), acc_rate, note.c_str());
        std::snprintf(buf, sizeof(buf), "%s\tdflash\t%d/%.2f\t%.1f\t%.2fx\t%.2f\t%.1f%%\t%s", p.name.c_str(), k, pm, tps, tps / plain_tps, per_round,
                      acc_rate, agree == best.tokens.size() ? "identical" : (std::string("diverges@") + std::to_string(agree)).c_str());
        rows.push_back(buf);
      }
    }
  }
  std::printf("\nprompt\tmode\tk/p_min\ttok/s\tspeedup\ttok/round\taccepted\tstream vs plain\n");
  for (const std::string& r : rows) std::printf("%s\n", r.c_str());
  std::printf("\n%d divergence(s) within the measured verify-vs-decode noise, %d not explained by it\n", ties, fails);
  const std::string line = m->RoundStatsLine();
  if (!line.empty()) std::printf("%s\n", line.c_str());
  return fails == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
  return 2;
}


