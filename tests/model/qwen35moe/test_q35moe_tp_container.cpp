// Qwen35MoeContainer::LoadShard against the TP=1 load, byte for byte, on the tiny containers (docs/qwen35moe-tp-design.md P1: "Gather-built stacks
// equal the TP=1 expert views sliced"). For the bf16 and the trellis tiny container, both ranks of a TP=2 load are compared with the single load:
//   * every routed expert of every layer, both classes: the rank's view into the stacked buffer (at the expert's stride) equals the plan-gathered
//     slice of the TP=1 view -- trellis words, widened sign vectors, bf16 elements -- and its QuantLinear shape / parts are the rank's;
//   * the stacks themselves have exactly E * per-expert-rank-size elements (the stride the grouped kernels index with);
//   * the mixers (GDN qkv / out_proj, attention qg / o), the bf16 shared expert and the vocab slice of lm_head;
//   * a rank's DeviceBytes() is about half the single load's (the replicated tensors are the difference).
// Exit code 77 = a fixture is missing. One device (the test device).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#include "q35moe_model_util.h"
#include "qwen35moe_container.h"
#include "qwen35moe_tp_shard.h"
#include "tp/tp_shard.h"

using namespace q35m;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;
using r4dx::model::Qwen35MoeContainer;
using r4dx::model::Qwen35MoeLoadOptions;
namespace tp = r4dx::model::tp;
namespace q35tp = r4dx::model::qwen35moe_tp;

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

template <class T>
std::vector<T> Host(const r4dx::core::DeviceBuffer<T>& b) {
  std::vector<T> v(b.size());
  b.CopyToHost(v.data(), v.size());
  return v;
}

std::vector<uint8_t> Bytes(const std::vector<float>& f) {  // sign vectors: compared as floats through their fp16 runs
  std::vector<uint8_t> b(f.size() * sizeof(float));
  std::memcpy(b.data(), f.data(), b.size());
  return b;
}

// the floats of `full` selected by runs of FP16 bytes (src_off / bytes in fp16 elements * 2)
std::vector<float> FloatRuns(const std::vector<float>& full, const std::vector<tp::ByteRun>& runs) {
  std::vector<float> out;
  for (const tp::ByteRun& r : runs) out.insert(out.end(), full.begin() + static_cast<ptrdiff_t>(r.src_off / 2), full.begin() + static_cast<ptrdiff_t>((r.src_off + r.bytes) / 2));
  return out;
}

struct Ctx {
  const r4dx::model::ModelConfig* gm;
  int rank;
  bool ok = true;
  int64_t compared = 0;
  void Bad(const std::string& what) {
    if (ok) std::fprintf(stderr, "  mismatch: %s\n", what.c_str());
    ok = false;
  }
};

// One trellis / bf16 linear: `full` is the TP=1 QuantLinear, `part` the rank's.
void CompareLinear(Ctx& c, const std::string& base, const QuantLinear& full, const QuantLinear& part, const std::vector<int64_t>& parts) {
  const tp::ShardRule rule = tp::RuleFor(base, *c.gm);
  if (part.layout != full.layout) return c.Bad(base + ": layout differs");
  if (full.layout == Layout::kBf16) {
    tp::PartShape sh;
    sh.part = tp::Part::kBf16;
    sh.N = full.N;
    sh.K = full.K;
    const auto runs = rule.split == tp::Split::kRows ? tp::PlanRows(sh, tp::RankRows(rule, 2, c.rank)) : tp::PlanCols(sh, tp::RankCols(rule, 2, c.rank));
    const auto fh = Host(full.bf16_w);
    const std::vector<uint8_t> want = tp::Gather(reinterpret_cast<const uint8_t*>(fh.data()), fh.size() * 2, runs);
    const auto ph = Host(part.bf16_w);
    if (ph.size() * 2 != want.size() || std::memcmp(ph.data(), want.data(), want.size()) != 0) c.Bad(base + ": bf16 slice differs");
    ++c.compared;
    return;
  }
  if (full.layout != Layout::kTrellis) return c.Bad(base + ": unexpected layout");
  const q35tp::TrellisSlicePlan p = q35tp::PlanTrellisLinear(rule, full.N, full.K, full.trellis_bits, parts, 2, c.rank);
  if (part.N != p.N || part.K != p.K || part.trellis_part_n[0] != p.part_n[0] || part.trellis_part_n[1] != p.part_n[1] || !part.trellis_tp_shard) {
    return c.Bad(base + ": rank shape / parts / tp flag differ from the plan");
  }
  const auto fw = Host(full.trellis_w);
  const std::vector<uint8_t> want_w = tp::Gather(reinterpret_cast<const uint8_t*>(fw.data()), fw.size() * 4, p.w);
  const auto pw = Host(part.trellis_w);
  if (pw.size() * 4 != want_w.size() || std::memcmp(pw.data(), want_w.data(), want_w.size()) != 0) c.Bad(base + ": trellis words differ");
  if (Host(part.trellis_suh) != FloatRuns(Host(full.trellis_suh), p.suh)) c.Bad(base + ": suh differs");
  if (Host(part.trellis_svh) != FloatRuns(Host(full.trellis_svh), p.svh)) c.Bad(base + ": svh differs");
  ++c.compared;
}

void RunContainer(const std::string& path, const std::string& layout) {
  std::printf("\n===== %s (%s) =====\n", layout.c_str(), path.c_str());
  if (!Exists(path)) {
    std::printf("SKIP: %s not found\n", path.c_str());
    return;
  }
  Qwen35MoeLoadOptions lo;
  lo.layout = r4dx::model::LayoutFromName(layout);
  lo.device = TestDevice();
  lo.load_head = true;
  const Qwen35MoeContainer one = Qwen35MoeContainer::Load(path, lo);
  const auto& cfg = one.Config();
  const r4dx::model::ModelConfig gm = cfg.ToModelConfig();
  Check(one.LocalConfig().tp_world == 1, layout + ": the TP=1 container's local config is the global one");
  for (int rank = 0; rank < 2; ++rank) {
    Qwen35MoeLoadOptions so = lo;
    so.tp_world = 2;
    so.tp_rank = rank;
    const Qwen35MoeContainer part = Qwen35MoeContainer::Load(path, so);
    Ctx c{&gm, rank};
    Check(part.LocalConfig().tp_world == 2 && part.LocalConfig().tp_rank == rank && part.LocalConfig().moe_intermediate_size == cfg.moe_intermediate_size / 2,
          layout + " rank " + std::to_string(rank) + ": local config is Shard()");
    const int64_t E = cfg.num_experts;
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
      const std::string b = "text.layers." + std::to_string(i) + ".";
      const auto& fl = one.Layer(i);
      const auto& pl = part.Layer(i);
      if (fl.gdn) {
        const std::vector<int64_t> none;
        CompareLinear(c, b + "gdn.in_proj_qkv", fl.gdn->in_proj_qkv, pl.gdn->in_proj_qkv, none);
        CompareLinear(c, b + "gdn.out_proj", fl.gdn->out_proj, pl.gdn->out_proj, none);
      } else {
        const std::vector<int64_t> none;
        CompareLinear(c, b + "attn.qg", fl.attn->qg, pl.attn->qg, none);
        CompareLinear(c, b + "attn.o", fl.attn->o, pl.attn->o, none);
      }
      const std::vector<int64_t> none;
      CompareLinear(c, b + "moe.shared.gate", fl.moe.shared.gate, pl.moe.shared.gate, none);
      CompareLinear(c, b + "moe.shared.down", fl.moe.shared.down, pl.moe.shared.down, none);
      for (int64_t e = 0; e < E; ++e) {
        const std::string eb = b + "moe.experts." + std::to_string(e) + ".";
        const std::vector<int64_t> gu_parts =
            fl.moe.experts[static_cast<size_t>(e)].gate_up.layout == Layout::kTrellis ? std::vector<int64_t>{cfg.moe_intermediate_size, cfg.moe_intermediate_size} : none;
        CompareLinear(c, eb + "gate_up", fl.moe.experts[static_cast<size_t>(e)].gate_up, pl.moe.experts[static_cast<size_t>(e)].gate_up, gu_parts);
        CompareLinear(c, eb + "down", fl.moe.experts[static_cast<size_t>(e)].down, pl.moe.experts[static_cast<size_t>(e)].down, none);
      }
      // the stacks are exactly E strides of the rank's per-expert size
      const auto& gs = pl.moe.gate_up_store;
      const auto& ds = pl.moe.down_store;
      if (gs.layout == Layout::kTrellis) {
        const auto g = q35tp::PlanExpertClass(gm, true, gs.bits, 2, rank);
        const auto d = q35tp::PlanExpertClass(gm, false, ds.bits, 2, rank);
        if (gs.trellis_w.size() != g.StackWords() || ds.trellis_w.size() != d.StackWords() || gs.trellis_suh.size() != static_cast<size_t>(E) * g.suh_per_rank ||
            gs.trellis_svh.size() != static_cast<size_t>(E) * g.svh_per_rank || ds.trellis_suh.size() != static_cast<size_t>(E) * d.suh_per_rank ||
            ds.trellis_svh.size() != static_cast<size_t>(E) * d.svh_per_rank) {
          c.Bad(b + "expert stacks are not E x the rank's per-expert size");
        }
      } else if (gs.bf16_w.size() * 2 != static_cast<size_t>(E) * (2 * cfg.moe_intermediate_size / 2) * cfg.hidden_size * 2) {
        c.Bad(b + "bf16 gate_up stack is not E x the rank's per-expert size");
      }
    }
    // lm_head: this rank's vocab rows
    if (one.LmHead().layout == Layout::kBf16) {
      const auto fh = Host(one.LmHead().bf16_w);
      const auto ph = Host(part.LmHead().bf16_w);
      const size_t half = fh.size() / 2;
      if (ph.size() != half || std::memcmp(ph.data(), fh.data() + static_cast<size_t>(rank) * half, half * 2) != 0) c.Bad("lm_head vocab slice differs");
    } else {
      const QuantLinear& l = part.LmHead();
      if (l.N != cfg.vocab_size / 2) c.Bad("lm_head rank rows");
    }
    Check(c.ok, layout + " rank " + std::to_string(rank) + ": " + std::to_string(c.compared) + " linears / expert views equal the plan-gathered TP=1 bytes");
    const double ratio = static_cast<double>(part.DeviceBytes()) / static_cast<double>(one.DeviceBytes());
    std::printf("  rank %d device bytes %.3f GiB vs %.3f GiB single (ratio %.3f)\n", rank, static_cast<double>(part.DeviceBytes()) / 1073741824.0,
                static_cast<double>(one.DeviceBytes()) / 1073741824.0, ratio);
    Check(ratio > 0.45 && ratio < 0.56, layout + " rank " + std::to_string(rank) + ": about half the single load's bytes");
  }
}

}  // namespace

int main() {
  Unbuffer();
  try {
    const char* e = std::getenv("R4DX_Q35MOE_TRELLIS");
    const std::string trellis = e != nullptr && e[0] != 0 ? e : std::string("E:\\models\\r4dx\\q35moe-tiny\\tiny-mix.r4dx");
    if (!Exists(ContainerPath("bf16")) && !Exists(trellis)) return kSkip;
    RunContainer(ContainerPath("bf16"), "bf16");
    RunContainer(trellis, "trellis");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", ex.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
