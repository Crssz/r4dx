// --tp 2 on the tiny qwen3_5_moe container, EMULATE mode (two rank models on ONE device, EmulatedComm) against the single-device model
// (docs/qwen35moe-tp-design.md P1 gate). No facade: two threads each load a rank-sharded Qwen35MoeModel (tp_world 2) and run the same script,
// meeting only in the all-reduces and the vocab merges, exactly as the real ranks will.
//
// Checked, for the bf16 tiny container and the trellis tiny-mix container (the grouped Tp2 decode kernels run on the latter):
//   * rank 0 and rank 1 agree bit for bit on everything they compute (the residual after every layer, the merged logits, tokens);
//   * every layer's residual after a T = 70 prefill, and the isolated GDN layer 0 and attention layer 3 (attn_out, moe_out, residual_out) over
//     the golden's own residual input, against the single-device model: mean|err| / rms(ref) within the bf16 noise of one extra partial
//     rounding per sublayer (the bounds are printed);
//   * the last-row logits of T = 1 / 7 / 70 / 300 prefills and of decode steps (full-row gather), argmax equal;
//   * 24 greedy decode steps (greedy pair merge) and 8 sampled steps (row-summary merge) from T = 70: the TP tokens against the single-device ones.
// Fixtures as q35moe_model_util.h; the trellis container is R4DX_Q35MOE_TRELLIS (default E:\models\r4dx\q35moe-tiny\tiny-mix.r4dx).
// Exit code 77 = a fixture is missing.
#include <hip/hip_runtime.h>

#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <random>
#include <thread>

#include "q35moe_model_util.h"
#include "tp/tp_group.h"

using namespace q35m;
using r4dx::model::Layout;
namespace tp = r4dx::model::tp;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

constexpr int kTs[] = {1, 7, 70, 300};
constexpr int kDecode = 24;
constexpr int kSampled = 8;

struct Result {
  std::map<int, std::vector<float>> prefill_logits;  // T -> last-row logits
  std::map<std::string, Captured> caps;               // "run:name"
  std::vector<int32_t> greedy, sampled;
  std::vector<std::vector<float>> decode_logits, hostrouted_logits;  // the second: the same steps with the grouped decode MoE switched off
  int64_t grouped_layers = 0;
};

std::vector<int32_t> Ids(int T) { return Golden(T, GoldenPath(T)).ids(); }

void Merge(Result& r, const std::string& prefix, const CaptureStore& s) {
  for (const auto& [name, c] : s.all()) r.caps[prefix + name] = c;
}

// The script every model runs (the single-device one once, each rank once).
Result Scenario(Qwen35MoeModel& m) {
  Result r;
  r.grouped_layers = m.GroupedDecodeLayers();
  for (int T : kTs) {
    m.Reset();
    r.prefill_logits[T] = m.Prefill(Ids(T));
  }
  {  // every layer after a T = 70 prefill
    CaptureStore s;
    m.Reset();
    m.SetCapture(s.Fn());
    (void)m.Prefill(Ids(70));
    m.SetCapture({});
    Merge(r, "prefill70:", s);
  }
  {  // layer-isolated: the golden's own residual into ONE layer
    const Golden g(70, IsoGoldenPath(70));
    for (int layer : {0, 3}) {
      CaptureStore s;
      m.SetCapture(s.Fn());
      const std::string in = "layer" + std::to_string(layer) + ".residual_in";
      m.RunLayerIsolated(Ids(70), layer, g.Bf16(in));
      m.SetCapture({});
      Merge(r, "iso" + std::to_string(layer) + ":", s);
    }
  }
  m.Reset();
  (void)m.Prefill(Ids(70));
  int32_t tok = static_cast<int32_t>(ArgMax(r.prefill_logits[70].data(), static_cast<int64_t>(r.prefill_logits[70].size())));
  for (int i = 0; i < kDecode; ++i) {
    tok = m.DecodeStepGreedy(tok);
    r.greedy.push_back(tok);
  }
  m.Reset();
  (void)m.Prefill(Ids(70));
  {
    std::mt19937_64 rng(12345);
    r4dx::kernels::SampleParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 40;
    int32_t t2 = r.greedy[0];
    for (int i = 0; i < kSampled; ++i) {
      t2 = m.DecodeStepSampled(t2, sp, rng);
      r.sampled.push_back(t2);
    }
  }
  m.Reset();
  (void)m.Prefill(Ids(70));
  {
    int32_t t3 = r.greedy[0];
    for (int i = 0; i < 3; ++i) {
      r.decode_logits.push_back(m.DecodeStep(t3));
      t3 = static_cast<int32_t>(ArgMax(r.decode_logits.back().data(), static_cast<int64_t>(r.decode_logits.back().size())));
    }
  }
  if (r.grouped_layers > 0) {
    m.SetGroupedDecode(false);
    m.Reset();
    (void)m.Prefill(Ids(70));
    int32_t t4 = r.greedy[0];
    for (int i = 0; i < 3; ++i) {
      r.hostrouted_logits.push_back(m.DecodeStep(t4));
      t4 = static_cast<int32_t>(ArgMax(r.hostrouted_logits.back().data(), static_cast<int64_t>(r.hostrouted_logits.back().size())));
    }
    m.SetGroupedDecode(true);
  }
  m.Reset();
  return r;
}

// A reusable barrier (C++17).
class Barrier {
 public:
  explicit Barrier(int n) : n_(n) {}
  void Wait() {
    std::unique_lock<std::mutex> lk(m_);
    const int gen = gen_;
    if (++count_ == n_) {
      count_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen != gen_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  int n_, count_ = 0, gen_ = 0;
};

Qwen35MoeModelOptions BaseOptions(const std::string& path, const std::string& layout) {
  Qwen35MoeModelOptions o;
  o.container_path = path;
  o.layout = r4dx::model::LayoutFromName(layout);
  o.max_ctx = 4096;
  return o;
}

Result RunSingle(const std::string& path, const std::string& layout) {
  Qwen35MoeModelOptions o = BaseOptions(path, layout);
  o.devices = {TestDevice(), TestDevice()};
  auto m = Qwen35MoeModel::Load(o);
  return Scenario(*m);
}

std::vector<Result> RunTwoRanks(const std::string& path, const std::string& layout) {
  auto group = tp::TpGroup::Create(tp::TpGroup::Mode::kEmulate, 2, tp::TpGroup::Geometry{}, 500);
  std::vector<Result> out(2);
  std::vector<std::exception_ptr> err(2);
  Barrier loaded(2);
  std::promise<std::shared_ptr<const r4dx::core::PinnedBuffer<uint16_t>>> embed_promise;
  auto embed_future = embed_promise.get_future().share();
  std::vector<std::thread> threads;
  for (int rank = 0; rank < 2; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        R4DX_HIP_CHECK(hipSetDevice(TestDevice()));
        auto ep = group->CreateEndpoint(rank, nullptr);
        Qwen35MoeModelOptions o = BaseOptions(path, layout);
        o.tp_world = 2;
        o.tp_rank = rank;
        o.tp_device = TestDevice();
        o.tp_comm = ep.get();
        if (rank == 1) o.shared_embed_host = embed_future.get();  // the process's ONE pinned copy of text.embed_tokens
        std::unique_ptr<Qwen35MoeModel> m;
        try {
          m = Qwen35MoeModel::Load(o);
        } catch (...) {
          if (rank == 0) embed_promise.set_exception(std::current_exception());
          throw;
        }
        if (rank == 0) embed_promise.set_value(m->EmbedHostShared());
        loaded.Wait();
        out[rank] = Scenario(*m);
        m.reset();
        R4DX_HIP_CHECK(hipDeviceSynchronize());
      } catch (...) {
        err[rank] = std::current_exception();
      }
    });
  }
  for (auto& t : threads) t.join();
  for (int r = 0; r < 2; ++r) {
    if (err[r]) std::rethrow_exception(err[r]);
  }
  return out;
}

struct Bound {
  double mean_norm, atol_rel, rtol, max_bad;
};

void CompareCaps(const Result& ref, const Result& got, const std::string& tag, const std::string& prefix, const Bound& b, bool bitwise) {
  PrintStatsHeader();
  bool all_ok = true;
  for (const auto& [name, rc] : ref.caps) {
    if (name.rfind(prefix, 0) != 0) continue;
    const auto it = got.caps.find(name);
    if (it == got.caps.end() && name.size() > 7 && name.compare(name.size() - 7, 7, ":logits") == 0) continue;  // (a rank holds a vocab slice: no all-rows logits capture)
    if (it == got.caps.end()) {
      std::fprintf(stderr, "FAIL: %s: capture %s missing\n", tag.c_str(), name.c_str());
      ++g_failures;
      continue;
    }
    const Captured& gc = it->second;
    if (rc.dtype == Qwen35MoeCapture::Dtype::kI32) {
      int64_t diff_rows = 0;
      const size_t k = static_cast<size_t>(rc.cols), rows = rc.i32.size() / k;
      for (size_t r = 0; r < rows; ++r) {
        std::vector<int32_t> a(rc.i32.begin() + r * k, rc.i32.begin() + (r + 1) * k), c(gc.i32.begin() + r * k, gc.i32.begin() + (r + 1) * k);
        std::sort(a.begin(), a.end());
        std::sort(c.begin(), c.end());
        diff_rows += a != c ? 1 : 0;
      }
      std::printf("  %-34s expert SET differs in %lld of %zu rows\n", name.c_str(), static_cast<long long>(diff_rows), rows);
      if (bitwise) all_ok = all_ok && diff_rows == 0;
      continue;
    }
    if (rc.bf16.size() != gc.bf16.size()) {
      all_ok = false;
      std::fprintf(stderr, "FAIL: %s %s: sizes differ\n", tag.c_str(), name.c_str());
      continue;
    }
    const int64_t n = static_cast<int64_t>(rc.bf16.size());
    if (bitwise) {
      const bool same = std::memcmp(rc.bf16.data(), gc.bf16.data(), rc.bf16.size() * 2) == 0;
      if (!same) std::fprintf(stderr, "  %s %s differs between the ranks\n", tag.c_str(), name.c_str());
      all_ok = all_ok && same;
      continue;
    }
    const Stats s = CompareBf16(gc.bf16.data(), rc.bf16.data(), n);
    const double bad = BadFraction(gc.bf16.data(), rc.bf16.data(), n, b.atol_rel, b.rtol);
    PrintStats(name, s);
    const bool ok = !s.nan && s.mean_norm <= b.mean_norm && bad <= b.max_bad;
    if (!ok) std::fprintf(stderr, "  %s %s: mean/rms %.5f (<= %.3f), outside %.4f%% (<= %.2f%%)\n", tag.c_str(), name.c_str(), s.mean_norm, b.mean_norm, 100.0 * bad, 100.0 * b.max_bad);
    all_ok = all_ok && ok;
  }
  Check(all_ok, tag + " [" + prefix + "*]" + (bitwise ? " bit-identical" : " within bounds"));
}

void CheckLogits(const Result& ref, const Result& got, const std::string& tag) {
  for (int T : kTs) {
    const auto& a = ref.prefill_logits.at(T);
    const auto& b = got.prefill_logits.at(T);
    const FStats s = CompareF32(b.data(), a.data(), static_cast<int64_t>(a.size()));
    const bool am = ArgMax(a.data(), static_cast<int64_t>(a.size())) == ArgMax(b.data(), static_cast<int64_t>(b.size()));
    std::printf("  %s T=%-3d logits mean/rms %.5f max/rms %.4f argmax %s\n", tag.c_str(), T, s.mean_norm, s.max_norm, am ? "equal" : "DIFFERS");
    Check(!s.nan && s.mean_norm <= 0.05 && am, tag + " prefill logits T=" + std::to_string(T));
  }
  for (size_t i = 0; i < ref.decode_logits.size(); ++i) {
    const FStats s = CompareF32(got.decode_logits[i].data(), ref.decode_logits[i].data(), static_cast<int64_t>(ref.decode_logits[i].size()));
    std::printf("  %s decode logits step %zu mean/rms %.5f\n", tag.c_str(), i, s.mean_norm);
    Check(!s.nan && s.mean_norm <= 0.05, tag + " decode logits step " + std::to_string(i));
  }
}

int Agree(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  int n = 0;
  while (n < static_cast<int>(a.size()) && n < static_cast<int>(b.size()) && a[static_cast<size_t>(n)] == b[static_cast<size_t>(n)]) ++n;
  return n;
}

bool RunLayout(const std::string& path, const std::string& layout) {
  std::printf("\n===== %s (%s) =====\n", layout.c_str(), path.c_str());
  if (!Exists(path)) {
    std::printf("SKIP: %s not found\n", path.c_str());
    return false;
  }
  const Result one = RunSingle(path, layout);
  const std::vector<Result> two = RunTwoRanks(path, layout);
  std::printf("grouped decode layers: single %lld, rank0 %lld, rank1 %lld\n", static_cast<long long>(one.grouped_layers),
              static_cast<long long>(two[0].grouped_layers), static_cast<long long>(two[1].grouped_layers));
  if (layout == "trellis") Check(two[0].grouped_layers == 4 && two[1].grouped_layers == 4, "trellis ranks take the grouped Tp2 decode MoE in all 4 layers");
  if (layout == "bf16") Check(two[0].grouped_layers == 0, "a bf16 container never takes the grouped path");

  // 1. the ranks agree exactly
  CompareCaps(two[0], two[1], layout + " rank0 vs rank1", "", Bound{}, /*bitwise=*/true);
  for (int T : kTs) {
    Check(two[0].prefill_logits.at(T) == two[1].prefill_logits.at(T), layout + " merged prefill logits identical on both ranks, T=" + std::to_string(T));
  }
  Check(two[0].greedy == two[1].greedy, layout + " greedy tokens identical on both ranks");
  Check(two[0].sampled == two[1].sampled, layout + " sampled tokens identical on both ranks");
  Check(two[0].decode_logits == two[1].decode_logits, layout + " decode logits identical on both ranks");
  if (two[0].grouped_layers > 0) {
    // the grouped Tp2 kernels against the host-routed per-expert GEMMs at the rank's shapes (same 4 / 4 split-group order as FallbackTuning picks at TP=1)
    bool exact = two[0].decode_logits.size() == two[0].hostrouted_logits.size();
    double worst = 0;
    for (size_t i = 0; i < two[0].decode_logits.size() && i < two[0].hostrouted_logits.size(); ++i) {
      exact = exact && two[0].decode_logits[i] == two[0].hostrouted_logits[i];
      worst = std::max(worst, CompareF32(two[0].decode_logits[i].data(), two[0].hostrouted_logits[i].data(), static_cast<int64_t>(two[0].decode_logits[i].size())).mean_norm);
    }
    std::printf("TP2 grouped decode MoE vs host-routed MoE: decode logits %s, worst mean/rms %.6f\n", exact ? "bit-identical" : "differ", worst);
    Check(worst <= 0.01, layout + " TP2 grouped decode MoE equals the host-routed MoE within 0.01 mean/rms (identical expected)");
  }

  // 2. TP=2 against the single-device model
  const Bound layer_bound{0.02, 0.08, 0.05, 0.12};  // measured: moe_out 0.006-0.007, residual_out 0.002-0.004 (mean|err| / rms)
  CompareCaps(one, two[0], layout + " TP2 vs TP1", "prefill70:", Bound{0.12, 0.2, 0.1, 0.10}, false);
  CompareCaps(one, two[0], layout + " TP2 vs TP1 isolated GDN layer 0", "iso0:", layer_bound, false);
  CompareCaps(one, two[0], layout + " TP2 vs TP1 isolated attention layer 3", "iso3:", layer_bound, false);
  CheckLogits(one, two[0], layout + " TP2 vs TP1");
  const int ag = Agree(one.greedy, two[0].greedy);
  std::printf("greedy: %d of %d tokens agree with the single-device model\n", ag, kDecode);
  Check(ag >= 20, layout + " greedy: at least 20 of 24 leading tokens equal the single-device run (bf16 near-ties may flip a later one)");
  const int as = Agree(one.sampled, two[0].sampled);
  std::printf("sampled: %d of %d leading tokens agree\n", as, kSampled);
  return true;
}

}  // namespace

int main() {
  Unbuffer();
  if (!Exists(GoldenPath(70)) || !Exists(IsoGoldenPath(70))) {
    std::printf("SKIP: golden fixtures missing under %s\n", TinyDir().c_str());
    return kSkip;
  }
  int ran = 0;
  try {
    const char* e = std::getenv("R4DX_Q35MOE_TRELLIS");
    const std::string trellis = e != nullptr && e[0] != 0 ? e : std::string("E:\\models\\r4dx\\q35moe-tiny\\tiny-mix.r4dx");
    ran += RunLayout(ContainerPath("bf16"), "bf16") ? 1 : 0;
    ran += RunLayout(trellis, "trellis") ? 1 : 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", ex.what());
    return 1;
  }
  if (ran == 0) return kSkip;
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
