// tests/model/test_gemma_forward_smoke.cpp -- docs/gemma4-plan.md M1-20/M1-22 smoke: the whole Gemma 4 text stack
// (GemmaModel on the real bf16.r4dx) against the M0-8 reference log-probs (tools/reference/gemma/full_logits_gemma.py).
// GPU test, HIP device 1 only. NOT RUN BY THE AUTHOR (no GPU work was allowed); exits 77 (SKIPPED) while the M0-8
// outputs or the container are missing.
//
// Reference format (full_logits_gemma.py, the shared kl_report.py format):
//     <ref-dir>/<segment>.logprobs.f16   little-endian float16, row-major [T-1, V]; row i = log p(next | tokens[0..i]),
//                                        fp32 log_softmax floored at -1e4
//     <ref-dir>/<segment>.meta.json      {"T", "V", "rows", "sha256_of_token_ids_json", "segment", ...}
// Tokens: tools/reference/kl_corpus/tokens_gemma.json (BOS = 2 first), `segments[i] = {name, token_ids}`.
//
// The pass is the one tool_teacher_forced_logprobs makes, shortened to the first kTokens tokens of the first
// segment: Prefill(tokens[0..P)) gives row P-1, DecodeStep(tokens[i]) gives row i. Per KV mode (bf16 KV first, then
// fp8 with the container's static descales -- placeholder 1.0 until M1-23):
//   * mean KL(ref || r4dx) over the rows and top-1 agreement (gates: bf16 KV KL <= 0.02, top-1 >= 90%; fp8 KV
//     KL <= 0.25 -- a smoke, the real gate is M1-22/M1-30 on the full corpus with the noise floor);
//   * DecodeStepGreedy == argmax(DecodeStep) for the same history (the device argmax path);
//   * a sampled step at temperature 1 returns a token inside the top-64 summary or the full-row fallback (no throw).
// R4DX_GEMMA_KL_REF_DIR / R4DX_GEMMA_BF16_CONTAINER / R4DX_GEMMA_SMOKE_TOKENS (default 96) override the inputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gemma_model.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "test_common.h"

#ifndef R4DX_SOURCE_DIR_STR
#define R4DX_SOURCE_DIR_STR "."
#endif

using namespace r4dx;
using namespace r4dx::model;
using nlohmann::json;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::vector<float> LogSoftmax(const std::vector<float>& logits) {
  float m = -INFINITY;
  for (float v : logits) m = std::max(m, v);
  double s = 0;
  for (float v : logits) s += std::exp(static_cast<double>(v) - m);
  const float lse = m + static_cast<float>(std::log(s));
  std::vector<float> lp(logits.size());
  for (size_t i = 0; i < lp.size(); ++i) lp[i] = std::max(logits[i] - lse, -1e4f);
  return lp;
}

struct RowStats {
  double kl = 0;
  bool top1 = false;
};

RowStats Compare(const std::vector<float>& logits, const uint16_t* ref_f16, int64_t V) {
  const std::vector<float> lp = LogSoftmax(logits);
  RowStats s;
  int64_t best_ref = 0, best_got = 0;
  float vr = -INFINITY, vg = -INFINITY;
  for (int64_t j = 0; j < V; ++j) {
    const float r = core::F16ToFloat(ref_f16[j]);
    if (r > vr) { vr = r; best_ref = j; }
    if (lp[static_cast<size_t>(j)] > vg) { vg = lp[static_cast<size_t>(j)]; best_got = j; }
    const double p = std::exp(static_cast<double>(r));
    if (p > 1e-12) s.kl += p * (static_cast<double>(r) - lp[static_cast<size_t>(j)]);
  }
  s.top1 = best_ref == best_got;
  return s;
}

int64_t Argmax(const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); }

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_gemma_forward_smoke", []() -> int {
    const auto env = [](const char* k, const std::string& d) { const char* v = std::getenv(k); return v && *v ? std::string(v) : d; };
    const std::string container = env("R4DX_GEMMA_BF16_CONTAINER", "D:\\models\\r4dx\\huihui-gemma\\bf16.r4dx");
    const std::string ref_dir = env("R4DX_GEMMA_KL_REF_DIR", "D:\\models\\r4dx\\huihui-gemma\\kl\\ref");
    const std::string tokens_path = env("R4DX_GEMMA_TOKENS", std::string(R4DX_SOURCE_DIR_STR) + "/tools/reference/kl_corpus/tokens_gemma.json");
    const int64_t kTokens = std::atoll(env("R4DX_GEMMA_SMOKE_TOKENS", "96").c_str());
    if (!r4dx_test::FileExists(container)) return r4dx_test::SkipMissing(container);
    if (!r4dx_test::FileExists(tokens_path)) return r4dx_test::SkipMissing(tokens_path);

    json doc;
    {
      std::ifstream f(tokens_path);
      doc = json::parse(f);
    }
    const json& seg = doc.at("segments").at(0);
    const std::string name = seg.at("name").get<std::string>();
    const std::string lp_path = ref_dir + "/" + name + ".logprobs.f16", meta_path = ref_dir + "/" + name + ".meta.json";
    if (!r4dx_test::FileExists(lp_path)) return r4dx_test::SkipMissing(lp_path + " (M0-8 full_logits_gemma.py)");
    if (!r4dx_test::FileExists(meta_path)) return r4dx_test::SkipMissing(meta_path);
    json meta;
    {
      std::ifstream f(meta_path);
      meta = json::parse(f);
    }
    std::vector<int32_t> ids = seg.at("token_ids").get<std::vector<int32_t>>();
    const int64_t V = meta.at("V").get<int64_t>();
    const int64_t T = std::min<int64_t>(static_cast<int64_t>(ids.size()), kTokens);
    if (static_cast<int64_t>(ids.size()) != meta.at("T").get<int64_t>()) {
      std::fprintf(stderr, "[FAIL] segment %s has %zu tokens, the reference was made over %lld\n", name.c_str(), ids.size(),
                   static_cast<long long>(meta.at("T").get<int64_t>()));
      return 1;
    }
    if (ids.empty() || ids[0] != 2) {
      std::fprintf(stderr, "[FAIL] the tokens must start with BOS = 2 (teacher-forced on both sides)\n");
      return 1;
    }
    // Only the first T-1 rows are needed.
    std::vector<uint16_t> ref(static_cast<size_t>((T - 1) * V));
    {
      std::ifstream f(lp_path, std::ios::binary);
      f.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * 2));
      if (!f) {
        std::fprintf(stderr, "[FAIL] %s is shorter than %lld rows of %lld float16\n", lp_path.c_str(), static_cast<long long>(T - 1),
                     static_cast<long long>(V));
        return 1;
      }
    }
    std::printf("segment %s: %lld tokens, V = %lld, reference %s\n", name.c_str(), static_cast<long long>(T), static_cast<long long>(V),
                lp_path.c_str());

    const int64_t P0 = std::min<int64_t>(48, T - 1);
    for (const bool bf16_kv : {true, false}) {
      GemmaModelOptions o;
      o.container_path = container;
      o.kv = bf16_kv ? GemmaKvMode::kBf16 : GemmaKvMode::kFp8;
      o.max_ctx = 4096;
      GemmaModel m = GemmaModel::Load(o);
      if (m.Config().vocab_size != V) {
        std::fprintf(stderr, "[FAIL] model vocab %lld != reference V %lld\n", static_cast<long long>(m.Config().vocab_size), static_cast<long long>(V));
        return 1;
      }
      std::printf("---- KV %s ----\n", bf16_kv ? "bf16" : "fp8");
      double kl_sum = 0;
      int64_t top1 = 0, rows = 0;
      std::vector<float> last_logits;
      auto account = [&](int64_t row, const std::vector<float>& logits) {
        const RowStats s = Compare(logits, ref.data() + row * V, V);
        kl_sum += s.kl;
        top1 += s.top1 ? 1 : 0;
        ++rows;
      };
      // Prefill the first P0 tokens (row P0-1), then teacher-force the rest through the decode path.
      account(P0 - 1, m.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + P0)));
      std::vector<std::vector<float>> decode_logits;
      for (int64_t i = P0; i <= T - 2; ++i) {
        decode_logits.push_back(m.DecodeStep(ids[static_cast<size_t>(i)]));
        account(i, decode_logits.back());
      }
      const double mean_kl = kl_sum / static_cast<double>(rows), top1_frac = static_cast<double>(top1) / static_cast<double>(rows);
      std::printf("  %lld rows: mean KL %.5f, top-1 agreement %.3f\n", static_cast<long long>(rows), mean_kl, top1_frac);
      Check(mean_kl <= (bf16_kv ? 0.02 : 0.25), std::string(bf16_kv ? "bf16" : "fp8") + " KV: mean KL vs the bf16 HF reference");
      if (bf16_kv) Check(top1_frac >= 0.90, "bf16 KV: top-1 agreement >= 90%");

      if (bf16_kv && !decode_logits.empty()) {
        // The device-argmax decode path must name the argmax of the full-logits path for the same history.
        m.Reset();
        m.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + P0));
        int agree = 0, n = 0;
        for (int64_t i = P0; i <= T - 2 && n < 12; ++i, ++n) {
          const int32_t g = m.DecodeStepGreedy(ids[static_cast<size_t>(i)]);
          agree += g == Argmax(decode_logits[static_cast<size_t>(i - P0)]) ? 1 : 0;
        }
        Check(agree == n, "DecodeStepGreedy == argmax(DecodeStep) over the same history (" + std::to_string(n) + " steps)");
        // A sampled step must not throw and must return a valid id (summary path or the full-row fallback).
        m.Reset();
        m.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + P0));
        kernels::SampleParams sp;
        sp.temperature = 1.0f;
        std::mt19937_64 rng(123);
        const int32_t tok = m.DecodeStepSampled(ids[static_cast<size_t>(P0)], sp, rng);
        Check(tok >= 0 && tok < V, "DecodeStepSampled returns a token id in [0, vocab)");
        // Checkpoint-free restore sanity: Reset + the same prefill reproduces the same first row.
        m.Reset();
        const std::vector<float> again = m.Prefill(std::vector<int32_t>(ids.begin(), ids.begin() + P0));
        const RowStats s = Compare(again, ref.data() + (P0 - 1) * V, V);
        Check(s.kl < 0.05, "Reset() then the same prefill is still on the reference");
      }
    }
    if (g_failures != 0) {
      std::printf("%d check(s) FAILED\n", g_failures);
      return 1;
    }
    std::printf("all checks passed\n");
    return 0;
  });
}
