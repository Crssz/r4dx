// tests/kernels/tool_trellis_gemm_bench.cpp -- trellis weights, milestone M1's and M2's measurements
// (docs/trellis-kernel.md 4.6, 6 "Benches", 7 M1/M2). HIP device 1 only; prints a summary and writes
// every number to the JSON file named by --out. M1's four modes (the fifth, the in-model clock
// probe, is r4dx's R4DX_CLOCK_PROBE debug hook, src/model/debug_probe.h -- the same probe kernel,
// src/model/kernels/clock_probe_device.h; its median SCLK comes back in as --in-model-sclk), M2's
// full-linear replay and M5's prefill tuning:
//
//   replay   One decode step's 336 body GEMMs (the 400 HF linears, gate/up fused) in layer order,
//            at full size with random weights: trellis KB = 4 through r4d_gemm_trellis_nt_m64_raw
//            (gate_up as production's two parts: two A buffers, n_split = 17408), against
//            q2ab_hv2_q3's production w4a16 -- its per-shape groups (docs/quant2.md 7: g128
//            mlp.down L0-31 and mlp.gate_up L32-63; g32 attn.o, gdn.in_proj_z and gdn.out_proj L0-31
//            and mlp.down L32-63; bf16 attn.k/v L32-63; g64 elsewhere) and production's dispatch
//            (src/model/linear.cpp's PickTuning: the M = 1 tuning for every M <= 16, NT = 0 at
//            M > 1; the default-group entry at g64). The JSON says which (shape, group) classes
//            production serves from its tuning table and which from the fallback (the table has no
//            g32 / g128 rows). With --sweep there is also a TUNED w4a16 baseline: every class swept
//            by the same method as trellis, so the comparison is not swept trellis against untuned
//            w4a16. Both formats' weights are resident at once (~24 GiB); the chains alternate
//            rep by rep in one process, at M = 1 and M = 8.
//            THE M1 GATE: S = w4a16 - trellis per token, taken per alternating pair of chains; GO
//            when its lower quartile is >= 0.16 ms, STOP when its upper quartile is < -0.14 ms. The
//            design's clock rule applies it at the in-model clock: if the replay's probe SCLK is
//            more than 3% above --in-model-sclk (the same probe kernel, run after every MLP of a
//            real decode), trellis time is rescaled by (1 - f) + f * replay / in-model, f =
//            --alu-fraction (default 1: all of it scales with the clock, the conservative bound).
//            Without --in-model-sclk the verdict is reported as provisional.
//            Also: per-(shape, group) us and effective GB/s (each chain after a cache flush), and
//            the shader clock: in-kernel for trellis (block 0 records clock64 / wall_clock64 at
//            entry and exit, the _raw entry's `clk`), and for both formats from a one-wave probe
//            kernel launched after every linear in a separate, untimed pass (probe against probe
//            is the like-for-like comparison; the w4a16 kernel has no clock hook).
//            Trellis tuning: docs/trellis-kernel.md 4.4's fallback rule, or with --sweep the best of
//            the legal knob space per shape (timed over all of that shape's instances after a
//            256 MiB cache flush, so the working set is never cache-resident), M = 8 re-sweeping
//            only NT (row identity).
//   ops      Issue rates (ops per clock per SIMD, independent chains) of v_mad_u32_u16,
//            v_pk_mad_u16, v_sad_u8, v_sad_hi_u8, v_mul_lo_u32 (and v_alignbit, v_pk_fma_f16,
//            v_add_nc_u32, v_fma_f32 for reference, and the exact forms the decode compiles to) at
//            1, 2 and 4 waves per SIMD; the cost of a near dependency (distance 1-4, with and
//            without s_delay_alu); and the GEMM's K loop taken apart (memory only, decode only, WMMA
//            only, decode -> WMMA, decode || WMMA, everything; the pipes share the GEMM's decode
//            block and its A-fragment loads) over the whole GPU at 1-16 waves per SIMD, giving the
//            decode's VALU per clock, WMMA/VALU overlap and the exposed fraction beta of 4.6. The
//            op rates are issue rates, not the decode's cost: the pipes measure that.
//   split    The split 128-group tail: us per linear on mlp.down (N 5120, K 17408, all 64
//            instances) at SKG {1,2,4,8} x block width Wc {32,64,128}, M = 1 and 8, against unsplit
//            controls of equal parallelism (Wc 128, SKG 1, SK 2 SKG: the same waves, each with the
//            same K range), so the tail is split - control (the raw ticket and workspace round
//            trip; the FWHT epilogue is M2's).
//   prefill  M = 64 on mlp.gate_up and mlp.down: trellis at (MT, NP) in {(4,1), (2,2), (4,2)}, U = 1,
//            best of WV / SK / SKG, against w4a16 at production's M = 64 tuning (and, with --sweep, a
//            tuned one), per w4a16 group.
//   full     Milestone M2 (docs/trellis-kernel.md 7 M2, 4.6 point 3, 10.1): the FULL linear chain of
//            one decode step, at M = 1 and M = 8, in one process and rotating chain by chain:
//              T_raw   the 336 raw GEMMs (M1's basis for S),
//              T_full  per linear r4dx_trellis_input_bf16 (x * suh -> FWHT-128 -> f16; gate_up's two
//                      parts in one call) and r4d_gemm_trellis_nt_m64 with its FWHT/svh/bf16 epilogue,
//              B_gemm  production w4a16 GEMMs (M1's baseline),
//              B_full  what q2ab_hv2_q3 runs around them: the bf16 -> f16 cast of every w4a16 linear
//                      (not the bf16 k/v), the 48 GDN in-place Hadamards (block 128 on out_core) and
//                      the 2 residual rotations, plus the GEMMs,
//            and the components alone (transforms, full GEMMs, q2ab's extras). Per rep:
//              S = B_gemm - T_raw,  X = (T_full - T_raw) - (B_full - B_gemm),  X - S = T_full - B_full;
//            the M2 gate is X <= 0.30 ms (the design target; 10.1 relaxes the hard cap to ~1.3 ms) and
//            X - S <= 0.14 ms. Also the per-key cost of the transform and the epilogue (flushed), the
//            probe clock of both full chains, and (info, outside X: v1 runs the plain kernel where
//            q2ab runs its Hadamard form) q2ab's silu_mul Hadamard and attention gate-mul Hadamard
//            over the plain ones.
//            Trellis tuning, per (N, K, KB) -- one row serves k and v, and gdn.out_proj and attn.o --:
//            --tuning-file (a JSON of tunings, or `table` for
//            src/model/gemm_tuning_table_trellis.inc as built), else --joint's pick, else 4.4's
//            fallback.
//            --joint picks each key's tuning for M = 1 and M = 8 TOGETHER (row identity gives both one
//            tuning): the legal space under 10.1's rules (NT 1, MT 1, SKG <= 4, >= 2 waves per SIMD)
//            is screened per key on flushed chains of all its instances by t(M=1) + t(M=8), and the
//            best four per key are then compared on the WHOLE full-linear step chain (paired, both M),
//            one key at a time. --tunings-out writes the result as JSON, --inc-out as the table
//            include (every key of this run, plus any --tuning-file key this run did not have).
//   ptune    Milestone M5: the prefill rows (M > 16) of the trellis table, per M of --ptune-m (default
//            32, 64) -- PrefillTune's comment has the method: every legal tuning screened per key on
//            flushed full-GEMM chains at that M, the best four (and each with NT flipped) compared on
//            the whole prefill chunk, and the chunk timed at the fallback, at the picks and for q2ab's
//            production path. A key that --tuning-file already gives a row at that M keeps it (unless
//            --joint-all), so a --kb mix run after a --kb 4 run adds the KB = 5 rows only.
//            --tunings-out / --inc-out write the M = 1 rows of --tuning-file with the prefill rows.
//   disp     M5 part 3 (docs/trellis-kernel.md 10.6): gate_up and down (the table's M = 1 rows) timed
//            after a pad of k one-wave workgroups and a one-workgroup locator that records the slot
//            the dispatcher hands out next -- does a GEMM's speed depend on where its workgroups
//            start? (It does not: +-0.1 us over every slot and k.) Per synchronize and back to back.
//   --kb 4 | mix   the trellis rate: KB = 4 everywhere, or EXL3's 4.5 bpw allocation (KB = 5 on
//            layers 0-15 and 48-63, KB = 4 on 16-47; --kb-manifest <mix4.5m weights_override.json>
//            reads the exact per-tensor K instead and checks gate K == up K).
//   --tp 2   one TP = 2 rank's shapes (ShardShapesTp2: the column-parallel linears split N, the
//            row-parallel ones K), for the tuning modes (full --joint, ptune) only: their rows are
//            the per-rank table (--inc-out: src/model/gemm_tuning_table_trellis_tp2.inc, which
//            `--tuning-file table` then reads). q2ab's extras in --modes full stay at TP = 1 sizes.
//
//   $env:HIP_VISIBLE_DEVICES='1'; build\win-hip\tests\kernels\tool_trellis_gemm_bench.exe `
//       --out <json> [--modes replay,ops,split,prefill,full,ptune,disp] [--reps 10] [--layers 64] [--sweep]
//       [--in-model-sclk <MHz>] [--alu-fraction <0..1>] [--kb 4|mix] [--kb-manifest <json>]
//       [--tuning-file <json>|table]... [--joint] [--joint-all] [--tunings-out <json>]
//       [--inc-out <path>] [--ptune-m 32,64] [--tp 1|2]
//
// TIMING is GPU-side: one-thread kernels write wall_clock64() before and after the launches (Timer
// below; hip events proved unreliable for short intervals on this stack). The REALTIME counter's
// rate is calibrated against host time at start-up and used for every time and every SCLK (the
// attribute says 100 MHz; this part measured 98.9-99.4). Every mode warms the GPU up first (a cold
// GPU measured 2-3x fewer ops per counted clock).
//
// Built, never add_test()'d (tests/kernels/CMakeLists.txt): it asserts no contract, and its real
// runs are GPU-time the user schedules.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kernels/model_kernels.h"   // r4dx_model_cast_bf16_to_f16: production's w4a16 cast
#include "linear.h"   // r4dx::model::PickTuning: production's w4a16 / bf16 dispatch
#include "nlohmann/json.hpp"
#include "r4d.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"           // r4dx_trellis_input_bf16, the silu_mul pair
#include "r4dx/kernels/rotate_residual.h"   // q2ab's residual rotation and GDN Hadamard
#include "r4dx/model/attention/attn_kernels.h"   // the attention gate-mul pair (info)
#include "trellis_tuning_rows.hpp"          // src/model/gemm_tuning_table_trellis.inc, as built

extern "C" {
void r4dx_tq_bench_fill_u32(int64_t p, int64_t n, uint32_t seed, uint32_t and_mask, uint32_t or_mask,
                            int64_t stream);
void r4dx_tq_bench_read(int64_t p, int64_t bytes, int64_t sink, int64_t stream);
void r4dx_tq_bench_stamp(int64_t out, int64_t stream);
void r4dx_tq_bench_where(int64_t out, int wgs, int threads, int64_t stream);
void r4dx_tq_bench_clock_probe(int64_t out, int iters, int64_t stream);
void r4dx_tq_bench_op_rate(int op, int dist, int delay, int wgs, int waves_per_wg, int iters, int64_t cyc,
                           int64_t hwid, int64_t sink, int64_t stream);
void r4dx_tq_bench_pipe(int mode, int wgs, int waves_per_wg, int iters, int64_t w, int64_t a,
                        int64_t bytes_per_wave, int64_t cyc, int64_t hwid, int64_t sink, int64_t clk,
                        int64_t stream);
}

// Production's measured tuning rows (the same file src/model/linear.cpp includes), only to report
// which linears PickTuning serves from a row and which from its fallback.
namespace tq_table {
using r4dx::model::GemmTuningRow;
using r4dx::model::Layout;
#include "gemm_tuning_table.inc"
}  // namespace tq_table

using json = nlohmann::json;
using r4dx::model::Layout;
using r4dx::model::LinearTuning;
using r4dx::model::PickTuning;

namespace {

template <typename T>
int64_t P(T* p) {
  return reinterpret_cast<int64_t>(p);
}

// ---- small helpers ------------------------------------------------------------------------------

struct Args {
  std::string out;
  std::set<std::string> modes{"replay", "ops", "split", "prefill"};
  int reps = 10;
  int layers = 64;
  bool sweep = false;
  int sweep_reps = 2;
  int probe_iters = 400;
  double in_model_sclk = 0.0;   // MHz; 0 = not given, the gate is provisional
  double alu_fraction = 1.0;    // the share of trellis GEMM time that scales with 1 / SCLK
  double warmup_s = 0.3;
  // M2 (mode full)
  std::string kb = "4";                     // "4" or "mix"
  std::string kb_manifest;                  // mix: the exact per-tensor K
  std::vector<std::string> tuning_files;    // JSON tuning sets, or "table"
  bool joint = false;                       // pick the keys without a --tuning-file tuning
  bool joint_all = false;                   // ... and re-pick those too
  int joint_top = 4;                        // candidates per key taken to the whole-chain stage
  int joint_reps = 6;                       // paired whole-chain reps per comparison
  std::string tunings_out, inc_out;
  // M5 (mode ptune): the prefill chunk sizes to tune, and the shapes of one TP = 2 rank (--tp 2).
  std::vector<int> ptune_m{32, 64};
  int tp = 1;
};

double Quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double x = q * static_cast<double>(v.size() - 1);
  const size_t i = static_cast<size_t>(x);
  const double f = x - static_cast<double>(i);
  return i + 1 < v.size() ? v[i] * (1.0 - f) + v[i + 1] * f : v[i];
}
double Median(const std::vector<double>& v) { return Quantile(v, 0.5); }
double Min(const std::vector<double>& v) { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); }
double Max(const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
double Mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}
json Spread(const std::vector<double>& v) {
  return json{{"median", Median(v)}, {"p25", Quantile(v, 0.25)}, {"p75", Quantile(v, 0.75)},
              {"min", Min(v)}, {"max", Max(v)}, {"n", v.size()}};
}

// A device arena: one hipMalloc, 256-byte aligned sub-allocations.
struct Blob {
  uint8_t* base = nullptr;
  size_t size = 0, used = 0;
  explicit Blob(size_t bytes) : size(bytes) {
    if (bytes) R4DX_HIP_CHECK(hipMalloc(&base, bytes));
  }
  ~Blob() {
    if (base) (void)hipFree(base);
  }
  Blob(const Blob&) = delete;
  Blob& operator=(const Blob&) = delete;
  template <typename T>
  T* Take(size_t bytes) {
    const size_t off = (used + 255) & ~static_cast<size_t>(255);
    if (off + bytes > size) throw std::runtime_error("Blob: out of space");
    used = off + bytes;
    return reinterpret_cast<T*>(base + off);
  }
};
size_t Round256(size_t b) { return (b + 255) & ~static_cast<size_t>(255); }

// GPU-side timer: a one-thread kernel writes wall_clock64() (the ~100 MHz REALTIME counter) before
// and after the launches, and the difference, less the cost of an empty bracket, is their time.
// hip events are not used: on this stack (HIP on Windows) an event pair around one 10-20 us launch
// measured from -30 us to 220 us (scratch probe, 300 samples), so a sweep's minimum picked
// near-zero garbage. Kernels of one stream run in order, so the stamps bracket the work exactly.
class Timer {
 public:
  explicit Timer(hipStream_t s) : s_(s) { R4DX_HIP_CHECK(hipMalloc(&stamps_, 2 * sizeof(unsigned long long))); }
  ~Timer() { (void)hipFree(stamps_); }
  // Sets the counter rate (MHz, calibrated) and measures the empty bracket (the stamp kernel's own
  // launch).
  void Calibrate(double wall_mhz) {
    mhz_ = wall_mhz;
    overhead_ms_ = 0.0;
    std::vector<double> v;
    for (int i = 0; i < 64; ++i) v.push_back(Ms([] {}));
    overhead_ms_ = Median(v);
  }
  double OverheadMs() const { return overhead_ms_; }
  // Milliseconds of fn's launches on the stream.
  double Ms(const std::function<void()>& fn) {
    r4dx_tq_bench_stamp(reinterpret_cast<int64_t>(stamps_), reinterpret_cast<int64_t>(s_));
    fn();
    r4dx_tq_bench_stamp(reinterpret_cast<int64_t>(stamps_ + 1), reinterpret_cast<int64_t>(s_));
    R4DX_HIP_CHECK(hipStreamSynchronize(s_));
    unsigned long long t[2];
    R4DX_HIP_CHECK(hipMemcpy(t, stamps_, sizeof t, hipMemcpyDeviceToHost));
    return static_cast<double>(t[1] - t[0]) / mhz_ / 1000.0 - overhead_ms_;
  }

 private:
  hipStream_t s_;
  unsigned long long* stamps_ = nullptr;
  double mhz_ = 100.0;
  double overhead_ms_ = 0.0;
};

// ---- the model ----------------------------------------------------------------------------------

struct Shape {
  const char* cls;
  int N, K;
};
// The body's shapes at TP = 1; --tp 2 makes them one rank's (ShardShapesTp2) before anything reads
// them.
Shape kQkv{"gdn.in_proj_qkv", 10240, 5120}, kZ{"gdn.in_proj_z", 6144, 5120},
    kOutProj{"gdn.out_proj", 5120, 6144}, kQg{"attn.qg", 12288, 5120}, kAttnK{"attn.k", 1024, 5120},
    kAttnV{"attn.v", 1024, 5120}, kAttnO{"attn.o", 5120, 6144}, kGateUp{"mlp.gate_up", 34816, 5120},
    kDown{"mlp.down", 5120, 17408};
const Shape* const kShapes[] = {&kQkv, &kZ, &kOutProj, &kQg, &kAttnK, &kAttnV, &kAttnO, &kGateUp, &kDown};

// One TP = 2 rank's shapes (docs/trellis-kernel.md 2.4, docs/tp.md 5.2): the column-parallel linears
// keep half the rows, the row-parallel ones (gdn.out_proj, attn.o, mlp.down) half of K. gate_up
// stays two parts, each half as wide (n_split = N / 2 = 8704).
void ShardShapesTp2() {
  for (Shape* s : {&kQkv, &kZ, &kQg, &kAttnK, &kAttnV, &kGateUp}) s->N /= 2;
  for (Shape* s : {&kOutProj, &kAttnO, &kDown}) s->K /= 2;
}

// q2ab_hv2_q3's w4a16 group of each body linear (docs/quant2.md 7); 0 = kept bf16.
int ProductionGroup(const Shape& s, int layer) {
  const bool lo = layer < 32;
  if (&s == &kAttnO) return 32;
  if (&s == &kZ || &s == &kOutProj) return lo ? 32 : 64;
  if (&s == &kDown) return lo ? 128 : 32;
  if (&s == &kGateUp) return lo ? 64 : 128;
  if (&s == &kAttnK || &s == &kAttnV) return lo ? 64 : 0;
  return 64;
}

// The per-tensor K of a trellis_quant.py weights_override.json (the mix4.5m manifest), keyed by
// (layer, shape); gate and up must agree (they are one fused linear). nlohmann reads the manifest's
// 'K' and 'k' keys as the distinct keys they are.
std::map<std::pair<int, const Shape*>, int> ReadKbManifest(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read --kb-manifest " + path);
  const json d = json::parse(f);
  const std::pair<const char*, const Shape*> mods[] = {
      {"linear_attn.in_proj_qkv", &kQkv}, {"linear_attn.in_proj_z", &kZ}, {"linear_attn.out_proj", &kOutProj},
      {"self_attn.q_proj", &kQg},         {"self_attn.k_proj", &kAttnK},  {"self_attn.v_proj", &kAttnV},
      {"self_attn.o_proj", &kAttnO},      {"mlp.gate_proj", &kGateUp},    {"mlp.up_proj", &kGateUp},
      {"mlp.down_proj", &kDown}};
  std::map<std::pair<int, const Shape*>, int> out;
  for (auto& kv : d.at("tensors").items()) {
    const std::string& name = kv.key();
    const size_t lp = name.find("layers.");
    if (lp == std::string::npos) continue;
    const int layer = std::stoi(name.substr(lp + 7));
    const size_t dot = name.find('.', lp + 7);
    const std::string mod = name.substr(dot + 1, name.rfind(".weight") - dot - 1);
    const Shape* s = nullptr;
    for (auto& m : mods)
      if (mod == m.first) s = m.second;
    if (!s) throw std::runtime_error("--kb-manifest: unknown module " + name);
    const double K = kv.value().at("K").get<double>();
    if (K != 4.0 && K != 5.0) throw std::runtime_error("--kb-manifest: " + name + " has K " + std::to_string(K));
    const int kb = static_cast<int>(K);
    auto it = out.find({layer, s});
    if (it != out.end() && it->second != kb)
      throw std::runtime_error("--kb-manifest: gate K != up K in layer " + std::to_string(layer));
    out[{layer, s}] = kb;
  }
  return out;
}

// Whether production's PickTuning (TP = 1) serves this linear from a measured row of its table,
// mirroring linear.cpp's BestRow (a chunk of M <= 16 rows resolves through the M = 1 band; a w4a16
// row serves only its own group, 0 meaning the build default, and only if K splits into whole
// groups). false = FallbackTuning.
bool HasTableRow(Layout layout, int N, int K, int M, int group) {
  const int64_t m = M <= 16 ? 1 : M;
  const int g = layout == Layout::kW4a16 ? r4dx::model::EffectiveW4a16Group(group) : 0;
  for (const tq_table::GemmTuningRow& row : tq_table::kGemmTuningTable) {
    if (row.layout != layout || row.N != N || row.K != K || row.M < m) continue;
    if (layout == Layout::kW4a16) {
      const int rg = row.group == 0 ? r4d_gemm_w4a16_nt_m64_group() : row.group;
      if (rg != g || K % (row.tuning.SK * std::max(g, 64)) != 0) continue;
    }
    return true;
  }
  return false;
}

json TuningJson(const LinearTuning& t) {
  return json{{"WV", t.WV}, {"SK", t.SK}, {"MB", t.MB}, {"NPW", t.NPW}, {"NT", t.NT}};
}

struct Linear {
  int layer;
  const Shape* s;
  int group;                  // w4a16 group, 0 = bf16
  int kb = 4;                 // trellis bits per weight
  uint32_t* tw = nullptr;     // trellis pair-grid words, N*K*kb/8 bytes
  uint32_t* tickets = nullptr;
  float* suh = nullptr;       // trellis input scales, fp32 [parts][K]
  float* svh = nullptr;       // trellis output scales, fp32 [N]
  uint8_t* wq = nullptr;      // w4a16 packed weight
  uint32_t* wsz = nullptr;    // w4a16 (scale, zero) dwords
  uint16_t* wbf = nullptr;    // bf16 weight
  int Parts() const { return s == &kGateUp ? 2 : 1; }   // mlp.gate_up: gate and up
  size_t TrellisBytes() const { return static_cast<size_t>(s->N) * s->K / 8 * kb; }
  // What a container stores besides the words: fp16 suh [parts][K] and svh [N].
  size_t TrellisScaleBytes() const { return static_cast<size_t>(Parts()) * s->K * 2 + static_cast<size_t>(s->N) * 2; }
  size_t BaselineBytes() const {
    if (group == 0) return static_cast<size_t>(s->N) * s->K * 2;
    return static_cast<size_t>(s->N) * s->K / 2 + static_cast<size_t>(s->N) * (s->K / group) * 4;
  }
};

// A (shape, w4a16 group) class: one baseline kernel and tuning, the unit the baseline is swept and
// reported by.
using ClassKey = std::pair<const Shape*, int>;
std::string ClassName(const ClassKey& k) {
  return std::string(k.first->cls) + (k.second == 0 ? " bf16" : " g" + std::to_string(k.second));
}

struct TTune {
  int WV, SK, MT, NP, SKG, U, NT;
  int Wc() const { return WV * NP * 32; }
  json Json() const {
    return json{{"WV", WV}, {"SK", SK}, {"MT", MT}, {"NP", NP}, {"SKG", SKG}, {"U", U}, {"NT", NT},
                {"Wc", Wc()}};
  }
  std::string Str() const {
    char b[96];
    std::snprintf(b, sizeof b, "WV%d NP%d SK%d SKG%d U%d MT%d NT%d", WV, NP, SK, SKG, U, MT, NT);
    return b;
  }
  bool operator==(const TTune& o) const {
    return WV == o.WV && SK == o.SK && MT == o.MT && NP == o.NP && SKG == o.SKG && U == o.U && NT == o.NT;
  }
};

// A trellis tuning key: the table has one row per (N, K, KB), so attn.k and attn.v share a tuning,
// and so do gdn.out_proj and attn.o.
using TKey = std::tuple<int, int, int>;
TKey KeyOf(const Linear& l) { return TKey{l.s->N, l.s->K, l.kb}; }
std::string KeyName(const TKey& k) {
  return std::to_string(std::get<0>(k)) + "x" + std::to_string(std::get<1>(k)) + " KB" +
         std::to_string(std::get<2>(k));
}

// docs/trellis-kernel.md 4.4's fallback: M <= 16 -> WV 4, NP 1, SK 2, MT 1, U 2, NT 1, SKG =
// clamp(128 / (N/128), 1, 4) rounded down to a power of two and reduced until it divides the K
// tiles; M > 16 -> SKG 1, MT min(4, ceil(M/16)), NT 0. (M1's replay ran the first revision's cap
// of 8, which differs only on attn.k/v: SKG 8 there; 10.1 dropped it.)
TTune FallbackTuning(int N, int K, int M) {
  TTune t{4, 2, 1, 1, 1, 2, 1};
  if (M > 16) {
    t.MT = std::min(4, (M + 15) / 16);
    t.NT = 0;
    return t;
  }
  int skg = std::max(1, std::min(4, 128 / (N / 128)));
  int p = 1;
  while (p * 2 <= skg) p *= 2;
  skg = p;
  while (skg > 1 && (K / 16) % (t.SK * skg * t.U) != 0) skg /= 2;
  t.SKG = skg;
  return t;
}

// The M1 gate on per-pair S (w4a16 - trellis, ms per token): GO when the lower quartile is >= 0.16,
// STOP when the upper quartile is < -0.14, between otherwise (docs/trellis-kernel.md 4.6 point 3).
std::string Verdict(const std::vector<double>& S) {
  if (Quantile(S, 0.25) >= 0.16) return "GO";
  if (Quantile(S, 0.75) < -0.14) return "STOP";
  return "between";
}

class Bench {
 public:
  Bench(const Args& a, hipStream_t st) : args_(a), st_(st), timer_(st) {}

  // Device info, the clock calibration and the small buffers every mode shares.
  void Setup() {
    hipDeviceProp_t prop;
    R4DX_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int wall_khz = 0;
    R4DX_HIP_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    wgps_ = prop.multiProcessorCount;
    small_ = std::make_unique<Blob>(160ull << 20);
    probe_ = small_->Take<unsigned long long>(4096 * 8 * 8);
    sink_ = small_->Take<uint32_t>(256);
    wall_mhz_ = wall_khz / 1000.0;
    const json cal = CalibrateClocks();
    timer_.Calibrate(wall_mhz_);
    out_["device"] = {{"name", prop.name},
                      {"gcn_arch", prop.gcnArchName},
                      {"multiprocessors_wgps", prop.multiProcessorCount},
                      {"clock_rate_khz", prop.clockRate},
                      {"memory_clock_khz", prop.memoryClockRate},
                      {"wall_clock_mhz_attribute", wall_khz / 1000.0},
                      {"wall_clock_mhz", wall_mhz_},
                      {"w4a16_default_group", r4d_gemm_w4a16_nt_m64_group()}};
    out_["clock_calibration"] = cal;
    // A (f16 for trellis / w4a16, a second f16 part for trellis gate_up, bf16 for the bf16 k/v),
    // C, workspace: sized for M = 64, per K of the shapes (a TP = 2 rank's with --tp 2).
    std::set<int> ks;
    for (const Shape* s : kShapes) ks.insert(s->K);
    for (int K : ks) {
      a_f16_[K] = small_->Take<uint16_t>(static_cast<size_t>(64) * K * 2);
      a_f16b_[K] = small_->Take<uint16_t>(static_cast<size_t>(64) * K * 2);
      a_bf16_[K] = small_->Take<uint16_t>(static_cast<size_t>(64) * K * 2);
      Fill(a_f16_[K], static_cast<size_t>(64) * K / 2, 11 + K, 0xB7FFB7FFu, 0x20002000u);
      Fill(a_f16b_[K], static_cast<size_t>(64) * K / 2, 17 + K, 0xB7FFB7FFu, 0x20002000u);
      Fill(a_bf16_[K], static_cast<size_t>(64) * K / 2, 13 + K, 0x807F807Fu, 0x3C003C00u);
    }
    c_f32_ = small_->Take<float>(static_cast<size_t>(64) * 34816 * 4);
    c_bf16_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 34816 * 2);
    ws_bytes_ = r4d_gemm_trellis_nt_m64_ws_bytes(64, 34816, 8);
    ws_ = small_->Take<float>(ws_bytes_);
    clk_ = small_->Take<unsigned long long>(4096 * 4 * 8);
    // q2ab's extras (mode full): the residual rotation's signs and mix, the GDN Hadamard's signs,
    // their activations, and the silu_mul and attention gate-mul pairs' (the gate-mul reads out_core_
    // as attn_out and had_signs_ as attn.o's signs).
    rot_signs_ = small_->Take<float>(5120 * 4);
    rot_mix5_ = small_->Take<float>(25 * 4);
    had_signs_ = small_->Take<float>(6144 * 4);
    resid_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 5120 * 2);
    out_core_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 6144 * 2);
    gate_in_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 6144 * 2);
    gated_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 6144 * 2);
    Fill(gate_in_, static_cast<size_t>(64) * 6144 / 2, 38, 0x807F807Fu, 0x3F003F00u);
    silu_in_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 34816 * 2);
    silu_out_ = small_->Take<uint16_t>(static_cast<size_t>(64) * 17408 * 2);
    silu_signs_ = small_->Take<float>(17408 * 4);
    Fill(silu_signs_, 17408, 37, 0x80000000u, 0x3F800000u);
    Fill(rot_signs_, 5120, 31, 0x80000000u, 0x3F800000u);    // +-1
    Fill(had_signs_, 6144, 32, 0x80000000u, 0x3F800000u);
    Fill(rot_mix5_, 25, 33, 0x807FFFFFu, 0x3E000000u);       // +-[0.125, 0.25)
    Fill(resid_, static_cast<size_t>(64) * 5120 / 2, 34, 0x807F807Fu, 0x3C003C00u);
    Fill(out_core_, static_cast<size_t>(64) * 6144 / 2, 35, 0x807F807Fu, 0x3C003C00u);
    Fill(silu_in_, static_cast<size_t>(64) * 34816 / 2, 36, 0x807F807Fu, 0x3F003F00u);
    flush_ = std::make_unique<Blob>(kFlushBytes);
    Fill(flush_->base, kFlushBytes / 4, 7, ~0u, 0u);
    R4DX_HIP_CHECK(hipStreamSynchronize(st_));
    out_["timer"] = {{"method", "wall_clock64 stamp kernels bracketing the launches"},
                     {"wall_clock_mhz", wall_mhz_},
                     {"empty_bracket_us", timer_.OverheadMs() * 1000.0},
                     {"flush", "read 256 MiB (clean lines) before each flushed chain, outside the bracket"},
                     {"flush_bytes", kFlushBytes}};
    std::printf("clocks: REALTIME %.3f MHz against host time (attribute %.1f); clock64 %.0f MHz; probe "
                "%.3f cycles per dependent add; stamp timer overhead %.2f us\n",
                wall_mhz_, wall_khz / 1000.0, cal["clock64_mhz_vs_host"].get<double>(),
                cal["cycles_per_add"]["median"].get<double>(), timer_.OverheadMs() * 1000.0);
  }

  // REALTIME (wall_clock64, the timer's counter) and SHADER_CYCLES (clock64) against host time:
  // the one-wave probe at 100k and 1M iterations of 8 dependent adds, three times; the differences
  // of the pairs cancel the launch and synchronize overhead (about 13 ms apart, so a few us of host
  // jitter is ~0.05%). The median REALTIME rate becomes wall_mhz_. Each probe's clock64 cycles per
  // dependent add is recorded: constant if clock64 counts real shader cycles (the question a cold
  // GPU's 2-3x fewer ops per counted clock raised), whatever the clock was.
  json CalibrateClocks() {
    struct Probe {
      double host_us, dc, dw;
    };
    auto probe = [&](int iters) {
      const auto h0 = std::chrono::steady_clock::now();
      r4dx_tq_bench_clock_probe(P(probe_), iters, P(st_));
      R4DX_HIP_CHECK(hipStreamSynchronize(st_));
      const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count();
      unsigned long long k[4];
      R4DX_HIP_CHECK(hipMemcpy(k, probe_, sizeof k, hipMemcpyDeviceToHost));
      return Probe{us, static_cast<double>(k[2] - k[0]), static_cast<double>(k[3] - k[1])};
    };
    const int kShort = 100000, kLong = 1000000;
    probe(1000);
    std::vector<double> wall, cyc, per_add, sclk;
    json raw = json::array();
    for (int i = 0; i < 3; ++i) {
      const Probe a = probe(kShort), b = probe(kLong);
      const double dus = b.host_us - a.host_us;
      wall.push_back((b.dw - a.dw) / dus);
      cyc.push_back((b.dc - a.dc) / dus);
      for (const Probe& p : {a, b}) sclk.push_back(p.dc / p.dw);   // x wall rate below
      per_add.push_back(a.dc / (8.0 * kShort));
      per_add.push_back(b.dc / (8.0 * kLong));
      raw.push_back({{"short", {{"host_us", a.host_us}, {"clock64", a.dc}, {"wall_clock64", a.dw}}},
                     {"long", {{"host_us", b.host_us}, {"clock64", b.dc}, {"wall_clock64", b.dw}}}});
    }
    wall_mhz_ = Median(wall);
    for (double& s : sclk) s *= wall_mhz_;
    return json{{"method", "one-wave probe, 1e5 and 1e6 x 8 dependent v_add, host steady_clock, 3 pairs"},
                {"wall_clock64_mhz_vs_host", Spread(wall)},
                {"clock64_mhz_vs_host", Median(cyc)},
                {"sclk_mhz_from_ratio", Spread(sclk)},
                {"cycles_per_add", Spread(per_add)},
                {"probes", raw}};
  }

  // Evicts the last-level cache (64 MiB MALL) by reading 256 MiB, outside any timed bracket: the
  // lines it leaves are clean, so the next chain pays no write-back.
  void Flush() { r4dx_tq_bench_read(P(flush_->base), static_cast<int64_t>(kFlushBytes), P(sink_), P(st_)); }

  // Both formats' weights for the body, layer by layer (after the ops mode, which wants the memory
  // for its own stream buffer).
  void SetupModel() {
    for (int L = 0; L < args_.layers; ++L) {
      const bool attn = L % 4 == 3;
      std::vector<const Shape*> order;
      if (attn) order = {&kQg, &kAttnK, &kAttnV, &kAttnO};
      else order = {&kQkv, &kZ, &kOutProj};
      order.push_back(&kGateUp);
      order.push_back(&kDown);
      for (const Shape* s : order) linears_.push_back(Linear{L, s, ProductionGroup(*s, L)});
    }
    // The trellis rate per linear: 4 everywhere, or the 4.5 bpw mix.
    std::map<std::pair<int, const Shape*>, int> manifest_kb;
    if (args_.kb == "mix" && !args_.kb_manifest.empty()) manifest_kb = ReadKbManifest(args_.kb_manifest);
    for (Linear& l : linears_) {
      if (args_.kb != "mix") continue;
      if (manifest_kb.empty()) {
        l.kb = (l.layer < 16 || l.layer >= 48) ? 5 : 4;   // EXL3's 4.5 allocation (mix4.5m)
      } else {
        auto it = manifest_kb.find({l.layer, l.s});
        if (it == manifest_kb.end())
          throw std::runtime_error("--kb-manifest has no K for layer " + std::to_string(l.layer) + " " + l.s->cls);
        l.kb = it->second;
      }
    }
    size_t tb = 0, wb = 0, tk = 0, sb = 0;
    for (const Linear& l : linears_) {
      tb += Round256(l.TrellisBytes());
      sb += Round256(static_cast<size_t>(l.Parts()) * l.s->K * 4) + Round256(static_cast<size_t>(l.s->N) * 4);
      tk += Round256(static_cast<size_t>(l.s->N / 128) * 4);
      if (l.group == 0) wb += Round256(l.BaselineBytes());
      else wb += Round256(static_cast<size_t>(l.s->N) * l.s->K / 2) +
                 Round256(static_cast<size_t>(l.s->N) * (l.s->K / l.group) * 4);
    }
    size_t free_b = 0, total_b = 0;
    R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    std::printf("allocating %.2f GiB trellis + %.2f GiB baseline weights (%zu linears, %d layers; "
                "%.2f of %.2f GiB free)\n",
                tb / 1073741824.0, wb / 1073741824.0, linears_.size(), args_.layers, free_b / 1073741824.0,
                total_b / 1073741824.0);
    if (tb + wb + tk + sb + (256ull << 20) > free_b)
      throw std::runtime_error("not enough free device memory for both formats; use --layers N");
    trellis_ = std::make_unique<Blob>(tb);
    base_ = std::make_unique<Blob>(wb);
    tick_ = std::make_unique<Blob>(tk);
    scales_ = std::make_unique<Blob>(sb);
    uint32_t seed = 1;
    for (Linear& l : linears_) {
      l.tw = trellis_->Take<uint32_t>(l.TrellisBytes());
      Fill(l.tw, l.TrellisBytes() / 4, seed++, ~0u, 0u);
      l.tickets = tick_->Take<uint32_t>(static_cast<size_t>(l.s->N / 128) * 4);
      // suh ~ +-[2^-7, 2^-6), svh ~ +-[1, 2): the oracle's orders of magnitude.
      l.suh = scales_->Take<float>(static_cast<size_t>(l.Parts()) * l.s->K * 4);
      l.svh = scales_->Take<float>(static_cast<size_t>(l.s->N) * 4);
      Fill(l.suh, static_cast<size_t>(l.Parts()) * l.s->K, seed++, 0x807FFFFFu, 0x3C000000u);
      Fill(l.svh, static_cast<size_t>(l.s->N), seed++, 0x807FFFFFu, 0x3F800000u);
      if (l.group == 0) {
        l.wbf = base_->Take<uint16_t>(l.BaselineBytes());
        Fill(l.wbf, l.BaselineBytes() / 4, seed++, 0x807F807Fu, 0x3C003C00u);
      } else {
        const size_t q = static_cast<size_t>(l.s->N) * l.s->K / 2;
        const size_t z = static_cast<size_t>(l.s->N) * (l.s->K / l.group) * 4;
        l.wq = base_->Take<uint8_t>(q);
        l.wsz = base_->Take<uint32_t>(z);
        Fill(l.wq, q / 4, seed++, ~0u, 0u);
        // f16 scale in [2^-5, 2^-4) in the low half, -(1024 + 8) in the high half.
        Fill(l.wsz, z / 4, seed++, 0x000003FFu, 0xE4082800u);
      }
    }
    for (const Linear& l : linears_) {
      const ClassKey k{l.s, l.group};
      if (classes_.find(k) == classes_.end()) class_order_.push_back(k);
      classes_[k].push_back(&l);
      const TKey tk2 = KeyOf(l);
      if (keys_.find(tk2) == keys_.end()) key_order_.push_back(tk2);
      keys_[tk2].push_back(&l);
      std::string& names = key_classes_[tk2];
      if (names.find(l.s->cls) == std::string::npos) names += (names.empty() ? "" : ", ") + std::string(l.s->cls);
    }
    R4DX_HIP_CHECK(hipMemsetAsync(tick_->base, 0, tick_->size, st_));
    R4DX_HIP_CHECK(hipStreamSynchronize(st_));
  }

  // ---- launches --------------------------------------------------------------------------------
  // gate_up runs as production's two parts: output columns >= 17408 (up) read their own A.
  void LaunchTrellis(const Linear& l, int M, const TTune& t, unsigned long long* clk) {
    const bool two = l.s == &kGateUp;
    r4d_gemm_trellis_nt_m64_raw(P(a_f16_.at(l.s->K)), two ? P(a_f16b_.at(l.s->K)) : 0,
                                two ? l.s->N / 2 : l.s->N, P(l.tw), P(c_f32_), P(ws_), P(l.tickets), M,
                                l.s->K, l.s->N, l.kb, t.WV, t.SK, t.MT, t.NP, t.SKG, t.U, t.NT, P(clk),
                                P(st_));
  }
  // M2: the linear's GEMM with its output transform (bf16 C), prescale 0.
  void LaunchTrellisFull(const Linear& l, int M, const TTune& t) {
    const bool two = l.s == &kGateUp;
    r4d_gemm_trellis_nt_m64(P(a_f16_.at(l.s->K)), two ? P(a_f16b_.at(l.s->K)) : 0,
                            two ? l.s->N / 2 : l.s->N, P(l.tw), P(l.svh), P(c_bf16_), P(ws_), P(l.tickets),
                            M, l.s->K, l.s->N, l.kb, t.WV, t.SK, t.MT, t.NP, t.SKG, t.U, t.NT, kOutScale,
                            P(st_));
  }
  // M2: the linear's input transform (v1: every linear transforms its own input; gate_up's two
  // parts in one call), into the A buffers the GEMM reads.
  void LaunchTransform(const Linear& l, int M) {
    const int K = l.s->K;
    const int64_t suh[2] = {P(l.suh), P(l.suh + K)};
    const int64_t out[2] = {P(a_f16_.at(K)), P(a_f16b_.at(K))};
    r4dx_trellis_input_bf16(P(a_bf16_.at(K)), M, K, l.Parts(), suh, out, 0, P(st_));
  }
  // q2ab's activation cast in front of a w4a16 linear (ApplyLinear's r4dx_model_cast_bf16_to_f16;
  // the bf16 k/v take none).
  void LaunchCast(const Linear& l, int M) {
    if (l.group == 0) return;
    r4dx_model_cast_bf16_to_f16(P(a_bf16_.at(l.s->K)), P(a_f16_.at(l.s->K)), static_cast<int64_t>(M) * l.s->K,
                                P(st_));
  }
  // q2ab's other online rotations: the GDN in-place Hadamard on out_proj's input (block 128,
  // gdn_layer.cpp) and the residual rotations at the stack's entry and exit (model.cpp).
  void LaunchGdnHadamard(int M) {
    r4dx_hadamard_inplace_bf16(P(out_core_), M, 6144, P(had_signs_), 128, P(st_));
  }
  void LaunchRotation(int M, bool inverse) {
    r4dx_rotate_residual_bf16(P(resid_), M, 5120, P(rot_signs_), P(rot_mix5_), inverse ? 1 : 0, P(st_));
  }
  // Production's tuning for a linear (linear.cpp ApplyLinear's PickTuning).
  static LinearTuning ProductionTuning(const Linear& l, int M) {
    return l.group == 0 ? PickTuning(Layout::kBf16, l.s->N, l.s->K, M)
                        : PickTuning(Layout::kW4a16, l.s->N, l.s->K, M, l.group);
  }
  // Production's path at tuning t: the default-group entry at the build's default group and the
  // per-group entry otherwise; bf16 through r4d_gemm_bf16_nt_m64.
  void LaunchBaseline(const Linear& l, int M, const LinearTuning& t) {
    if (l.group == 0) {
      r4d_gemm_bf16_nt_m64(P(a_bf16_.at(l.s->K)), P(l.wbf), P(c_bf16_), M, l.s->K, l.s->N, t.WV, t.SK,
                           t.MB, P(st_));
    } else if (l.group == r4d_gemm_w4a16_nt_m64_group()) {
      r4d_gemm_w4a16_nt_m64(P(a_f16_.at(l.s->K)), P(l.wq), P(l.wsz), P(c_bf16_), M, l.s->K, l.s->N,
                            t.WV, t.SK, t.MB, t.NPW, t.NT, P(st_));
    } else {
      r4d_gemm_w4a16_nt_m64_g(l.group, P(a_f16_.at(l.s->K)), P(l.wq), P(l.wsz), P(c_bf16_), M, l.s->K,
                              l.s->N, t.WV, t.SK, t.MB, t.NPW, t.NT, P(st_));
    }
  }
  void LaunchBaseline(const Linear& l, int M) { LaunchBaseline(l, M, ProductionTuning(l, M)); }

  std::vector<const Linear*> Instances(const Shape* s) const {
    std::vector<const Linear*> v;
    for (const Linear& l : linears_)
      if (l.s == s) v.push_back(&l);
    return v;
  }

  // Milliseconds of one chain, or a negative value when the host rejects the tuning or a launch
  // fails (hipLaunchKernelGGL reports that only through hipGetLastError; a failed launch would
  // otherwise time as a near-empty chain and win every sweep).
  double Chain(const std::function<void()>& launches, const std::string& what) {
    double ms;
    try {
      (void)hipGetLastError();
      ms = timer_.Ms(launches);
    } catch (const std::exception&) {
      R4DX_HIP_CHECK(hipStreamSynchronize(st_));
      return -1.0;
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      std::printf("  launch failed (%s): %s\n", hipGetErrorString(e), what.c_str());
      return -1.0;
    }
    return ms;
  }
  double ChainTrellis(const std::vector<const Linear*>& ls, int M, const TTune& t, bool clk = false) {
    return Chain([&] {
      for (size_t i = 0; i < ls.size(); ++i) LaunchTrellis(*ls[i], M, t, clk ? clk_ + 4 * i : nullptr);
    }, "trellis " + t.Json().dump() + " on " + ls[0]->s->cls + " M=" + std::to_string(M));
  }
  // A baseline chain at one tuning (a class), or at production's tunings (t == nullptr).
  double ChainBaseline(const std::vector<const Linear*>& ls, int M, const LinearTuning* t) {
    return Chain([&] {
      for (const Linear* l : ls) {
        if (t) LaunchBaseline(*l, M, *t);
        else LaunchBaseline(*l, M);
      }
    }, "baseline on " + std::string(ls[0]->s->cls) + " M=" + std::to_string(M));
  }
  // The same, throwing on a failure: the replay's chains are all legal by construction.
  static double Must(double ms, const char* what) {
    if (ms < 0) throw std::runtime_error(std::string("replay chain failed: ") + what);
    return ms;
  }

  // ---- replay ----------------------------------------------------------------------------------
  using TrellisTunes = std::map<const Shape*, TTune>;
  using BaseTunes = std::map<ClassKey, LinearTuning>;

  void Replay() {
    json r;
    TrellisTunes tune1;
    for (const Shape* s : kShapes) tune1[s] = FallbackTuning(s->N, s->K, 1);
    BaseTunes base1;                                  // tuned w4a16 / bf16 at M <= 16 (--sweep)
    Warm();
    if (args_.sweep) {
      json sw, bw;
      for (const Shape* s : kShapes) {
        if (Instances(s).empty()) continue;
        tune1[s] = Sweep(s, 1, sw[s->cls]);
      }
      for (const ClassKey& k : class_order_) base1[k] = SweepBaseline(k, 1, bw[ClassName(k)]);
      r["sweep_m1"] = sw;
      r["baseline_sweep_m1"] = bw;
    }
    // Which (shape, group) classes production serves from its table and which from the fallback.
    json src;
    for (const ClassKey& k : class_order_) {
      const Linear& l = *classes_.at(k)[0];
      const Layout lay = k.second == 0 ? Layout::kBf16 : Layout::kW4a16;
      src[ClassName(k)] = {{"instances", classes_.at(k).size()},
                           {"source", HasTableRow(lay, k.first->N, k.first->K, 1, k.second) ? "table" : "fallback"},
                           {"tuning_m1", TuningJson(ProductionTuning(l, 1))},
                           {"tuning_m8", TuningJson(ProductionTuning(l, 8))},
                           {"tuning_m64", TuningJson(ProductionTuning(l, 64))},
                           {"source_m64", HasTableRow(lay, k.first->N, k.first->K, 64, k.second) ? "table" : "fallback"}};
    }
    r["w4a16_tuning_source"] = src;
    for (int M : {1, 8}) {
      TrellisTunes tune = tune1;                     // M <= 16 runs the M = 1 tuning (row identity)
      BaseTunes base = base1;
      if (args_.sweep && M > 1) {
        json sw, bw;
        for (const Shape* s : kShapes) {
          if (Instances(s).empty()) continue;
          double best = 1e30;
          for (int nt : {0, 1}) {
            TTune t = tune1[s];
            t.NT = nt;
            const double ms = SweepTime(s, M, t);
            sw[s->cls]["NT" + std::to_string(nt)] = ms;
            if (ms > 0 && ms < best) best = ms, tune[s] = t;
          }
        }
        for (const ClassKey& k : class_order_) {
          if (k.second == 0) continue;               // bf16 has no NT
          double best = 1e30;
          for (int nt : {0, 1}) {
            LinearTuning t = base1[k];
            t.NT = nt;
            const double ms = SweepTimeBaseline(k, M, t);
            bw[ClassName(k)]["NT" + std::to_string(nt)] = ms;
            if (ms > 0 && ms < best) best = ms, base[k] = t;
          }
        }
        r["sweep_m" + std::to_string(M) + "_nt"] = sw;
        r["baseline_sweep_m" + std::to_string(M) + "_nt"] = bw;
      }
      r["M" + std::to_string(M)] = ReplayAt(M, tune, args_.sweep ? &base : nullptr);
    }
    out_["replay"] = r;
  }

  // At least warmup_s of alternating trellis / w4a16 chains (the first loops of a cold GPU ran 2-3x
  // slower per counted clock).
  void Warm() {
    double ms = 0.0;
    std::vector<const Linear*> all;
    for (const Linear& l : linears_) all.push_back(&l);
    for (int i = 0; (i < 2 || ms < 1000.0 * args_.warmup_s) && i < 1000; ++i) {
      ms += Must(Chain([&] {
        for (const Linear* l : all) LaunchTrellis(*l, 1, FallbackTuning(l->s->N, l->s->K, 1), nullptr);
      }, "warm-up"), "warm-up trellis");
      ms += Must(ChainBaseline(all, 1, nullptr), "warm-up w4a16");
    }
  }

  json ReplayAt(int M, const TrellisTunes& tune, const BaseTunes* base) {
    json j;
    json tj;
    for (auto& kv : tune) tj[kv.first->cls] = kv.second.Json();
    j["trellis_tuning"] = tj;
    if (base) {
      json bj;
      for (auto& kv : *base) bj[ClassName(kv.first)] = TuningJson(kv.second);
      j["w4a16_tuned_tuning"] = bj;
    }
    std::vector<const Linear*> all;
    for (const Linear& l : linears_) all.push_back(&l);
    size_t tbytes = 0, bbytes = 0;
    for (const Linear* l : all) tbytes += l->TrellisBytes(), bbytes += l->BaselineBytes();

    auto chain_t = [&](bool clk) {
      return Must(Chain([&] {
        for (size_t i = 0; i < all.size(); ++i)
          LaunchTrellis(*all[i], M, tune.at(all[i]->s), clk ? clk_ + 4 * i : nullptr);
      }, "trellis replay"), "trellis");
    };
    auto chain_b = [&] { return Must(ChainBaseline(all, M, nullptr), "w4a16"); };
    auto chain_bt = [&] {
      return Must(Chain([&] {
        for (const Linear* l : all) LaunchBaseline(*l, M, base->at(ClassKey{l->s, l->group}));
      }, "tuned w4a16 replay"), "tuned w4a16");
    };
    Warm();
    // The chains of one rep run back to back in a rotating order; S is taken per rep, so each
    // S sample compares chains a few ms apart.
    std::vector<double> tt, tb, tbt;
    for (int r = 0; r < args_.reps; ++r) {
      const int n = base ? 3 : 2;
      for (int k = 0; k < n; ++k) {
        const int which = (r + k) % n;
        if (which == 0) tt.push_back(chain_t(true));
        else if (which == 1) tb.push_back(chain_b());
        else tbt.push_back(chain_bt());
      }
    }
    const double mt = Median(tt), mb = Median(tb);
    std::vector<double> S(tt.size()), St;
    for (size_t r = 0; r < tt.size(); ++r) S[r] = tb[r] - tt[r];
    for (size_t r = 0; r < tbt.size(); ++r) St.push_back(tbt[r] - tt[r]);
    j["linears"] = all.size();
    j["trellis_ms"] = Spread(tt);
    j["trellis_ms"]["all"] = tt;
    j["w4a16_ms"] = Spread(tb);
    j["w4a16_ms"]["all"] = tb;
    j["trellis_bytes"] = tbytes;
    j["w4a16_bytes"] = bbytes;
    j["byte_ratio"] = static_cast<double>(tbytes) / bbytes;
    j["time_ratio"] = mt / mb;
    j["eff_bw_ratio"] = (tbytes / mt) / (bbytes / mb);   // trellis's effective bandwidth / w4a16's
    j["trellis_gbps"] = tbytes / mt / 1e6;
    j["w4a16_gbps"] = bbytes / mb / 1e6;
    j["S_ms"] = Median(S);
    j["S_ms_per_pair"] = Spread(S);
    if (base) {
      j["w4a16_tuned_ms"] = Spread(tbt);
      j["w4a16_tuned_ms"]["all"] = tbt;
      j["S_vs_tuned_ms_per_pair"] = Spread(St);
      j["eff_bw_ratio_vs_tuned"] = (tbytes / mt) / (bbytes / Median(tbt));
    }

    // The in-kernel clock of the last timed trellis chain: SCLK per launch from block 0's deltas.
    std::vector<unsigned long long> clk(4 * all.size());
    R4DX_HIP_CHECK(hipMemcpy(clk.data(), clk_, clk.size() * 8, hipMemcpyDeviceToHost));
    std::vector<double> sclk;
    std::map<std::string, std::vector<double>> sclk_cls;
    for (size_t i = 0; i < all.size(); ++i) {
      const double dc = static_cast<double>(clk[4 * i + 2] - clk[4 * i]);
      const double dw = static_cast<double>(clk[4 * i + 3] - clk[4 * i + 1]);
      if (dw < 20) continue;                       // block 0 too short to resolve
      const double mhz = dc / dw * wall_mhz_;
      sclk.push_back(mhz);
      sclk_cls[all[i]->s->cls].push_back(mhz);
    }
    json sj = {{"median_mhz", Median(sclk)}, {"min_mhz", Min(sclk)}, {"launches", sclk.size()}};
    for (auto& kv : sclk_cls) sj["by_shape"][kv.first] = Median(kv.second);
    j["trellis_sclk_in_kernel"] = sj;

    // Probe pass (untimed): a one-wave clock probe after every linear, both formats -- the same
    // kernel and method as the in-model probe the clock rule compares with.
    const json pt = ProbePass(all, M, tune, true), pb = ProbePass(all, M, tune, false);
    j["probe_sclk"] = {{"trellis", pt}, {"w4a16", pb},
                       {"trellis_over_w4a16", pt["median_mhz"].get<double>() / pb["median_mhz"].get<double>()}};

    // THE GATE, at the in-model clock when it is known (docs/trellis-kernel.md 7 M1).
    const double replay_sclk = pt["median_mhz"].get<double>();
    json rule = {{"replay_probe_sclk_mhz", replay_sclk}, {"alu_fraction", args_.alu_fraction}};
    double scale = 1.0;
    const bool have_model = args_.in_model_sclk > 0.0;
    if (have_model) {
      const double ratio = replay_sclk / args_.in_model_sclk;
      rule["in_model_sclk_mhz"] = args_.in_model_sclk;
      rule["replay_over_in_model"] = ratio;
      if (ratio > 1.03) scale = (1.0 - args_.alu_fraction) + args_.alu_fraction * ratio;
      rule["applied"] = ratio > 1.03;
    } else {
      rule["in_model_sclk_mhz"] = nullptr;
      rule["applied"] = false;
    }
    rule["trellis_time_scale"] = scale;
    std::vector<double> Sg(tt.size());
    for (size_t r = 0; r < tt.size(); ++r) Sg[r] = tb[r] - tt[r] * scale;
    const bool full = args_.layers == 64;
    const std::string verdict = !full ? "n/a (fewer than 64 layers)" : Verdict(Sg);
    json gate = {{"verdict", verdict},
                 {"provisional", !have_model},
                 {"S_ms", Spread(Sg)},
                 {"rule", "GO: S lower quartile >= 0.16 ms; STOP: S upper quartile < -0.14 ms (per alternating pair)"},
                 {"clock_rule", rule}};
    if (!have_model)
      gate["note"] = "no --in-model-sclk: the clock rule was not applied (run r4dx's in-model clock probe, "
                     "docs/trellis-kernel.md 6 mode 5, and pass its median)";
    if (base) {
      std::vector<double> Sgt;
      for (size_t r = 0; r < tbt.size(); ++r) Sgt.push_back(tbt[r] - tt[r] * scale);
      gate["verdict_vs_tuned_w4a16"] = !full ? "n/a" : Verdict(Sgt);
      gate["S_vs_tuned_ms"] = Spread(Sgt);
    }
    j["gate"] = gate;

    // Per (shape, group) class: its instances back to back, per format, each timed chain after a
    // cache flush (16 attn.k instances are 42 MB: a repeat would otherwise be served by the 64 MiB
    // MALL, which the full step never does).
    json ps;
    const int reps = std::max(3, args_.reps / 2);
    for (const ClassKey& k : class_order_) {
      const std::vector<const Linear*>& ls = classes_.at(k);
      size_t tb2 = 0, bb2 = 0;
      for (const Linear* l : ls) tb2 += l->TrellisBytes(), bb2 += l->BaselineBytes();
      std::vector<double> a, b, bt;
      Must(ChainTrellis(ls, M, tune.at(k.first)), "per-shape trellis");
      Must(ChainBaseline(ls, M, nullptr), "per-shape w4a16");
      for (int r = 0; r < reps; ++r) {
        Flush();
        a.push_back(Must(ChainTrellis(ls, M, tune.at(k.first)), "per-shape trellis"));
        Flush();
        b.push_back(Must(ChainBaseline(ls, M, nullptr), "per-shape w4a16"));
        if (base) {
          Flush();
          bt.push_back(Must(ChainBaseline(ls, M, &base->at(k)), "per-shape tuned w4a16"));
        }
      }
      const double ma = Median(a), mb2 = Median(b);
      json e = {{"N", k.first->N},
                {"K", k.first->K},
                {"group", k.second},
                {"instances", ls.size()},
                {"trellis_us", 1000.0 * ma / ls.size()},
                {"w4a16_us", 1000.0 * mb2 / ls.size()},
                {"trellis_gbps", tb2 / ma / 1e6},
                {"w4a16_gbps", bb2 / mb2 / 1e6},
                {"trellis_ms_total", ma},
                {"w4a16_ms_total", mb2}};
      if (base) {
        e["w4a16_tuned_us"] = 1000.0 * Median(bt) / ls.size();
        e["w4a16_tuned_gbps"] = bb2 / Median(bt) / 1e6;
      }
      ps[ClassName(k)] = e;
    }
    j["per_class"] = ps;
    std::printf("replay M=%d: trellis %.3f ms, w4a16 %.3f ms per token (GEMMs only, median of %d), S = "
                "%+.3f ms [IQR %+.3f..%+.3f], eff-BW ratio %.3f (byte ratio %.3f) -> %s%s; SCLK in-kernel "
                "%.0f, probe trellis %.0f / w4a16 %.0f MHz\n",
                M, mt, mb, args_.reps, Median(Sg), Quantile(Sg, 0.25), Quantile(Sg, 0.75),
                j["eff_bw_ratio"].get<double>(), j["byte_ratio"].get<double>(), verdict.c_str(),
                have_model ? "" : " (PROVISIONAL: no --in-model-sclk)", Median(sclk), replay_sclk,
                pb["median_mhz"].get<double>());
    if (base)
      std::printf("  against the tuned w4a16 baseline: %.3f ms, S = %+.3f ms -> %s\n", Median(tbt),
                  Median(St), gate["verdict_vs_tuned_w4a16"].get<std::string>().c_str());
    for (auto& kv : ps.items()) {
      std::printf("  %-22s trellis %8.2f us %6.1f GB/s | w4a16 %8.2f us %6.1f GB/s", kv.key().c_str(),
                  kv.value()["trellis_us"].get<double>(), kv.value()["trellis_gbps"].get<double>(),
                  kv.value()["w4a16_us"].get<double>(), kv.value()["w4a16_gbps"].get<double>());
      if (base)
        std::printf(" | tuned %8.2f us %6.1f GB/s", kv.value()["w4a16_tuned_us"].get<double>(),
                    kv.value()["w4a16_tuned_gbps"].get<double>());
      std::printf("\n");
    }
    return j;
  }

  // cycles_per_add here includes the probe's fixed cost (its clock reads) spread over only
  // 8 * probe_iters adds, so it sits above the calibration's 1e5-1e6-iteration value; what must hold
  // is that it does not move with the clock across the probes of a run.
  json ProbePass(const std::vector<const Linear*>& all, int M, const TrellisTunes& tune, bool trellis) {
    const size_t n = std::min<size_t>(all.size(), 4096);
    for (size_t i = 0; i < n; ++i) {
      if (trellis) LaunchTrellis(*all[i], M, tune.at(all[i]->s), nullptr);
      else LaunchBaseline(*all[i], M);
      r4dx_tq_bench_clock_probe(P(probe_ + 8 * i), args_.probe_iters, P(st_));
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(st_));
    R4DX_HIP_CHECK(hipGetLastError());
    std::vector<unsigned long long> pr(8 * n);
    R4DX_HIP_CHECK(hipMemcpy(pr.data(), probe_, pr.size() * 8, hipMemcpyDeviceToHost));
    std::vector<double> mhz, us, per_add;
    for (size_t i = 0; i < n; ++i) {
      const double dc = static_cast<double>(pr[8 * i + 2] - pr[8 * i]);
      const double dw = static_cast<double>(pr[8 * i + 3] - pr[8 * i + 1]);
      if (dw <= 0) continue;
      mhz.push_back(dc / dw * wall_mhz_);
      us.push_back(dw / wall_mhz_);
      per_add.push_back(dc / (8.0 * args_.probe_iters));
    }
    return {{"median_mhz", Median(mhz)}, {"min_mhz", Min(mhz)}, {"p25_mhz", Quantile(mhz, 0.25)},
            {"p75_mhz", Quantile(mhz, 0.75)}, {"probe_us", Median(us)}, {"probes", mhz.size()},
            {"cycles_per_add", Spread(per_add)}};
  }

  // ---- sweeps ----------------------------------------------------------------------------------
  // Median of sweep_reps chains over every instance of the shape, each after a cache flush (after
  // one warm-up that doubles as the legality probe); negative when the host rejects the tuning.
  double SweepTime(const Shape* s, int M, const TTune& t) {
    std::vector<const Linear*> ls = Instances(s);
    if (ChainTrellis(ls, M, t) < 0) return -1.0;
    std::vector<double> v;
    for (int r = 0; r < args_.sweep_reps; ++r) {
      Flush();
      const double ms = ChainTrellis(ls, M, t);
      if (ms < 0) return -1.0;
      v.push_back(ms);
    }
    return Median(v) / ls.size();
  }
  TTune Sweep(const Shape* s, int M, json& log) {
    TTune best = FallbackTuning(s->N, s->K, M);
    const double fb = SweepTime(s, M, best);
    double best_ms = fb > 0 ? fb : 1e30;           // a rejected fallback must not beat every tuning
    log["fallback"] = {{"tuning", best.Json()}, {"ms_per_linear", fb}};
    int legal = 0;
    for (int WV : {1, 2, 4})
      for (int NP : {1, 2, 4})
        for (int SK : {1, 2, 4, 8, 16})
          for (int SKG : {1, 2, 4, 8})
            for (int U : {1, 2, 4})
              for (int NT : {0, 1}) {
                const TTune t{WV, SK, 1, NP, SKG, U, NT};
                const double ms = SweepTime(s, M, t);
                if (ms < 0) continue;
                ++legal;
                if (ms < best_ms) best_ms = ms, best = t;
              }
    log["legal"] = legal;
    log["best"] = {{"tuning", best.Json()}, {"ms_per_linear", best_ms}};
    std::printf("  sweep %-16s M=%d: %d legal, best WV%d NP%d SK%d SKG%d U%d NT%d %.2f us (fallback %.2f)\n",
                s->cls, M, legal, best.WV, best.NP, best.SK, best.SKG, best.U, best.NT, 1000 * best_ms,
                1000 * fb);
    return best;
  }

  // The baseline's counterpart, per (shape, group) class, by the same method: w4a16 over WV x SK x
  // NPW x NT (x MB at M > 16), bf16 over WV x SK (x MB); the kernels' hosts reject what is illegal.
  double SweepTimeBaseline(const ClassKey& k, int M, const LinearTuning& t) {
    const std::vector<const Linear*>& ls = classes_.at(k);
    if (ChainBaseline(ls, M, &t) < 0) return -1.0;
    std::vector<double> v;
    for (int r = 0; r < args_.sweep_reps; ++r) {
      Flush();
      const double ms = ChainBaseline(ls, M, &t);
      if (ms < 0) return -1.0;
      v.push_back(ms);
    }
    return Median(v) / ls.size();
  }
  LinearTuning SweepBaseline(const ClassKey& k, int M, json& log) {
    const LinearTuning prod = ProductionTuning(*classes_.at(k)[0], M);
    LinearTuning best = prod;
    const double pm = SweepTimeBaseline(k, M, prod);
    double best_ms = pm > 0 ? pm : 1e30;
    log["production"] = {{"tuning", TuningJson(prod)}, {"ms_per_linear", pm}};
    const bool bf16 = k.second == 0;
    std::vector<int> mbs = M <= 16 ? std::vector<int>{1} : std::vector<int>{1, 2, 4};
    int legal = 0;
    for (int WV : {1, 2, 4, 8, 16, 32})
      for (int SK : {1, 2, 4, 8, 16, 32})
        for (int MB : mbs)
          for (int NPW : bf16 ? std::vector<int>{1} : std::vector<int>{1, 4})
            for (int NT : bf16 ? std::vector<int>{1} : std::vector<int>{0, 1}) {
              if (WV * SK * 32 > 1024) continue;
              const LinearTuning t{WV, SK, MB, NPW, NT};
              const double ms = SweepTimeBaseline(k, M, t);
              if (ms < 0) continue;
              ++legal;
              if (ms < best_ms) best_ms = ms, best = t;
            }
    log["legal"] = legal;
    log["best"] = {{"tuning", TuningJson(best)}, {"ms_per_linear", best_ms}};
    std::printf("  baseline sweep %-22s M=%d: %d legal, best WV%d SK%d MB%d NPW%d NT%d %.2f us (production %.2f)\n",
                ClassName(k).c_str(), M, legal, best.WV, best.SK, best.MB, best.NPW, best.NT, 1000 * best_ms,
                1000 * pm);
    return best;
  }

  // ---- full linear (M2) ------------------------------------------------------------------------
  using KeyTunes = std::map<TKey, TTune>;
  enum class TChain { kRaw, kFull, kGemm, kXform };
  enum class BChain { kGemm, kFull, kExtras };

  std::vector<const Linear*> All() const {
    std::vector<const Linear*> v;
    for (const Linear& l : linears_) v.push_back(&l);
    return v;
  }

  static TTune TuneFromJson(const json& j) {
    return TTune{j.at("WV").get<int>(), j.at("SK").get<int>(), j.value("MT", 1), j.at("NP").get<int>(),
                 j.at("SKG").get<int>(), j.at("U").get<int>(), j.value("NT", 1)};
  }

  // A prefill row (M5): its (N, K, KB) and the chunk M it was tuned at (> 16). The table serves a
  // chunk of M rows from the smallest row M >= it (src/model/linear.cpp's BestRow).
  using PKey = std::pair<TKey, int>;
  using PrefillTunes = std::map<PKey, TTune>;

  // --tuning-file: JSON as --tunings-out writes it ({"tunings": [{N, K, KB, [M,] WV, ...}]}, no M
  // meaning 1), or "table" for the rows of src/model/gemm_tuning_table_trellis.inc as this binary was
  // built (with --tp 2, gemm_tuning_table_trellis_tp2.inc's). Each tuning goes to `f` with its M.
  void ForEachFileTuning(const std::function<void(const TKey&, int, const TTune&)>& f) const {
    const auto rows = [&](const auto& table) {
      for (const trellis_rows::GemmTuningRow& row : table) {
        if (row.layout != trellis_rows::Layout::kTrellis) continue;
        const trellis_rows::LinearTuning& t = row.tuning;
        f(TKey{static_cast<int>(row.N), static_cast<int>(row.K), row.rate}, static_cast<int>(row.M),
          TTune{t.WV, t.SK, t.MB, t.NPW, t.SKG, t.U, t.NT});
      }
    };
    for (const std::string& file : args_.tuning_files) {
      if (file == "table") {
        if (args_.tp == 2) rows(trellis_rows::tp2::kGemmTuningTable);
        else rows(trellis_rows::kGemmTuningTable);
        continue;
      }
      std::ifstream in(file, std::ios::binary);
      if (!in) throw std::runtime_error("cannot read --tuning-file " + file);
      const json d = json::parse(in);
      const json& arr = d.contains("tunings") ? d.at("tunings") : d;
      for (const json& e : arr)
        f(TKey{e.at("N").get<int>(), e.at("K").get<int>(), e.at("KB").get<int>()}, e.value("M", 1),
          TuneFromJson(e));
    }
  }
  // The decode rows (M <= 16: one M = 1 row per key, row identity).
  KeyTunes LoadTunings() const {
    KeyTunes out;
    ForEachFileTuning([&](const TKey& k, int M, const TTune& t) {
      if (M <= 16) out[k] = t;
    });
    return out;
  }
  // The prefill rows (M > 16).
  PrefillTunes LoadPrefillTunings() const {
    PrefillTunes out;
    ForEachFileTuning([&](const TKey& k, int M, const TTune& t) {
      if (M > 16) out[{k, M}] = t;
    });
    return out;
  }

  // One trellis step over `ls` in layer order: per linear the transform and the full GEMM (what v1's
  // ApplyLinear runs), or one of them, or M1's raw GEMM.
  void LaunchTChain(const std::vector<const Linear*>& ls, int M, const KeyTunes& tune, TChain what) {
    for (const Linear* l : ls) {
      const TTune& t = tune.at(KeyOf(*l));
      switch (what) {
        case TChain::kRaw: LaunchTrellis(*l, M, t, nullptr); break;
        case TChain::kFull: LaunchTransform(*l, M); LaunchTrellisFull(*l, M, t); break;
        case TChain::kGemm: LaunchTrellisFull(*l, M, t); break;
        case TChain::kXform: LaunchTransform(*l, M); break;
      }
    }
  }
  // One q2ab_hv2_q3 step over the whole body: the residual rotation at the stack's entry, per
  // linear the cast and the GEMM, the GDN Hadamard in front of every out_proj, the inverse rotation
  // at the exit; or the GEMMs alone, or everything but the GEMMs.
  void LaunchBChain(const std::vector<const Linear*>& ls, int M, BChain what) {
    const bool extras = what != BChain::kGemm, gemm = what != BChain::kExtras;
    if (extras) LaunchRotation(M, false);
    for (const Linear* l : ls) {
      if (extras) {
        if (l->s == &kOutProj) LaunchGdnHadamard(M);
        LaunchCast(*l, M);
      }
      if (gemm) LaunchBaseline(*l, M);
    }
    if (extras) LaunchRotation(M, true);
  }

  // A key's instances as flushed full-GEMM chains (median of sweep_reps, ms per linear), after one
  // unflushed chain that doubles as the legality probe; negative when the host rejects the tuning.
  double SweepTimeKey(const std::vector<const Linear*>& ls, int M, const TTune& t) {
    auto chain = [&] {
      return Chain([&] {
        for (const Linear* l : ls) LaunchTrellisFull(*l, M, t);
      }, "trellis full " + t.Str() + " on " + ls[0]->s->cls);
    };
    if (chain() < 0) return -1.0;
    std::vector<double> v;
    for (int r = 0; r < args_.sweep_reps; ++r) {
      Flush();
      const double ms = chain();
      if (ms < 0) return -1.0;
      v.push_back(ms);
    }
    return Median(v) / ls.size();
  }

  // docs/trellis-kernel.md 10.1's joint pick. Stage 1 screens every tuning 10.1 allows (NT 1, MT 1,
  // SKG <= 4, >= 2 waves per SIMD) per key by t(M=1) + t(M=8) on flushed chains of the key's
  // instances; stage 2 takes each key's best joint_top into the whole full-linear step, one key at a
  // time (largest first), and keeps a candidate only when paired whole-step chains say it saves
  // time at M = 1 + M = 8 (median < -5 us and upper quartile < 0).
  KeyTunes JointSelect(KeyTunes tune, const std::set<TKey>& fixed, json& log) {
    const int simds = 4 * std::max(1, wgps_);
    std::map<TKey, std::vector<std::pair<double, TTune>>> top;
    std::map<TKey, double> weight;   // the key's best screened ms per token (both M)
    for (const TKey& k : key_order_) {
      if (fixed.count(k) && !args_.joint_all) continue;
      const std::vector<const Linear*>& ls = keys_.at(k);
      const int N = std::get<0>(k);
      std::vector<std::pair<double, TTune>> scored;
      json all = json::array();
      for (int WV : {1, 2, 4})
        for (int NP : {1, 2, 4})
          for (int SK : {1, 2, 4, 8, 16})
            for (int SKG : {1, 2, 4})
              for (int U : {1, 2, 4}) {
                const TTune t{WV, SK, 1, NP, SKG, U, 1};
                const int Wc = t.Wc();
                if (Wc > 256 || N % Wc != 0) continue;
                if ((N / Wc) * SKG * WV * SK < 2 * simds) continue;   // >= 2 waves per SIMD
                const double t1 = SweepTimeKey(ls, 1, t);
                if (t1 < 0) continue;
                const double t8 = SweepTimeKey(ls, 8, t);
                if (t8 < 0) continue;
                scored.push_back({t1 + t8, t});
                all.push_back({{"tuning", t.Json()}, {"m1_us", 1000 * t1}, {"m8_us", 1000 * t8}});
              }
      std::sort(scored.begin(), scored.end(),
                [](const std::pair<double, TTune>& a, const std::pair<double, TTune>& b) { return a.first < b.first; });
      if (scored.empty()) throw std::runtime_error("joint: no legal tuning for " + KeyName(k));
      if (static_cast<int>(scored.size()) > args_.joint_top) scored.resize(args_.joint_top);
      json best = json::array();
      for (auto& s : scored) best.push_back({{"tuning", s.second.Json()}, {"m1_plus_m8_us", 1000 * s.first}});
      log["screen"][KeyName(k)] = {{"classes", key_classes_.at(k)}, {"instances", ls.size()},
                                   {"legal", all.size()}, {"best", best}, {"all", all}};
      std::printf("  joint screen %-18s (%s): %zu legal, best %s %.2f us (M=1 + M=8 per linear)\n",
                  KeyName(k).c_str(), key_classes_.at(k).c_str(), all.size(), scored[0].second.Str().c_str(),
                  1000 * scored[0].first);
      tune[k] = scored[0].second;
      top[k] = scored;
      weight[k] = scored[0].first * ls.size();
    }
    // Stage 2: the whole step.
    const std::vector<const Linear*> all = All();
    auto step = [&](const KeyTunes& tn, int M) {
      return Must(Chain([&] { LaunchTChain(all, M, tn, TChain::kFull); }, "joint step"), "joint step");
    };
    std::vector<TKey> order;
    for (auto& kv : top) order.push_back(kv.first);
    std::sort(order.begin(), order.end(), [&](const TKey& a, const TKey& b) { return weight[a] > weight[b]; });
    Warm();
    json stage2 = json::array();
    for (const TKey& k : order) {
      for (size_t i = 1; i < top[k].size(); ++i) {
        KeyTunes alt = tune;
        alt[k] = top[k][i].second;
        if (alt[k] == tune[k]) continue;
        std::vector<double> d, d1, d8;
        for (int r = 0; r < args_.joint_reps; ++r) {
          double c1, c8, a1, a8;
          if (r % 2 == 0) {
            c1 = step(tune, 1), a1 = step(alt, 1), c8 = step(tune, 8), a8 = step(alt, 8);
          } else {
            a1 = step(alt, 1), c1 = step(tune, 1), a8 = step(alt, 8), c8 = step(tune, 8);
          }
          d1.push_back(a1 - c1);
          d8.push_back(a8 - c8);
          d.push_back((a1 + a8) - (c1 + c8));
        }
        const bool take = Median(d) < -0.005 && Quantile(d, 0.75) < 0.0;
        stage2.push_back({{"key", KeyName(k)}, {"current", tune[k].Json()}, {"candidate", alt[k].Json()},
                          {"delta_ms_m1_plus_m8", Spread(d)}, {"delta_ms_m1", Median(d1)}, {"delta_ms_m8", Median(d8)},
                          {"taken", take}});
        std::printf("  joint step %-18s %s -> %s: %+.3f ms (M=1 %+.3f, M=8 %+.3f) per step pair%s\n",
                    KeyName(k).c_str(), tune[k].Str().c_str(), alt[k].Str().c_str(), Median(d), Median(d1),
                    Median(d8), take ? "  TAKEN" : "");
        if (take) tune[k] = alt[k];
      }
    }
    log["whole_step"] = stage2;
    return tune;
  }

  // --tunings-out (JSON) and --inc-out (the table include): the M = 1 rows in `tune` and the prefill
  // rows in `prefill` (a key of another run keeps its rows when it came in through --tuning-file).
  void WriteTunings(const KeyTunes& tune, const std::map<TKey, std::string>& source,
                    const PrefillTunes& prefill = {}) const {
    // Every row, sorted by KB, then the shapes' layer order, then M.
    struct Row {
      TKey k;
      int M;
      TTune t;
      std::string src;
    };
    std::vector<Row> rows;
    for (auto& kv : tune)
      rows.push_back({kv.first, 1, kv.second, source.count(kv.first) ? source.at(kv.first) : "file"});
    for (auto& kv : prefill) {
      const auto s = prefill_source_.find(kv.first);
      rows.push_back({kv.first.first, kv.first.second, kv.second, s == prefill_source_.end() ? "file" : s->second});
    }
    std::vector<std::pair<int, int>> shape_order;
    for (const Shape* s : kShapes)
      if (std::find(shape_order.begin(), shape_order.end(), std::make_pair(s->N, s->K)) == shape_order.end())
        shape_order.push_back({s->N, s->K});
    auto rank = [&](const Row& r) {
      const auto it = std::find(shape_order.begin(), shape_order.end(),
                                std::make_pair(std::get<0>(r.k), std::get<1>(r.k)));
      return std::make_tuple(std::get<2>(r.k), static_cast<int>(it - shape_order.begin()), std::get<0>(r.k),
                             std::get<1>(r.k), r.M);
    };
    std::sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) { return rank(a) < rank(b); });
    // The linears every (N, K) serves: this run's, else the default names of the table's shapes.
    std::map<std::pair<int, int>, std::string> names =
        args_.tp == 1 ? std::map<std::pair<int, int>, std::string>{
                            {{10240, 5120}, "gdn.in_proj_qkv"}, {{6144, 5120}, "gdn.in_proj_z"},
                            {{5120, 6144}, "gdn.out_proj, attn.o"}, {{12288, 5120}, "attn.qg"},
                            {{1024, 5120}, "attn.k, attn.v"}, {{34816, 5120}, "mlp.gate_up"},
                            {{5120, 17408}, "mlp.down"}}
                      : std::map<std::pair<int, int>, std::string>{
                            {{5120, 5120}, "gdn.in_proj_qkv"}, {{3072, 5120}, "gdn.in_proj_z"},
                            {{5120, 3072}, "gdn.out_proj, attn.o"}, {{6144, 5120}, "attn.qg"},
                            {{512, 5120}, "attn.k, attn.v"}, {{17408, 5120}, "mlp.gate_up"},
                            {{5120, 8704}, "mlp.down"}};
    for (auto& kv : key_classes_) names[{std::get<0>(kv.first), std::get<1>(kv.first)}] = kv.second;

    if (!args_.tunings_out.empty()) {
      json arr = json::array();
      for (const Row& r : rows) {
        json e = r.t.Json();
        e["N"] = std::get<0>(r.k);
        e["K"] = std::get<1>(r.k);
        e["KB"] = std::get<2>(r.k);
        e["M"] = r.M;
        const auto nm = names.find({std::get<0>(r.k), std::get<1>(r.k)});
        e["classes"] = nm == names.end() ? "" : nm->second;
        e["source"] = r.src;
        arr.push_back(e);
      }
      std::ofstream f(args_.tunings_out, std::ios::binary);
      if (!f) throw std::runtime_error("cannot write " + args_.tunings_out);
      f << json{{"format", "r4dx-trellis-tunings"}, {"version", 1}, {"tp", args_.tp}, {"tunings", arr}}.dump(1)
        << "\n";
      std::printf("wrote %s\n", args_.tunings_out.c_str());
    }
    if (!args_.inc_out.empty()) {
      std::ostringstream o;
      o << "// GENERATED by tests/kernels/tool_trellis_gemm_bench.exe (docs/trellis-kernel.md 7 M2, M5, 10.1);\n"
           "// --inc-out wrote this file. Measured on HIP device 1, R9700 (gfx1201), at full size with random\n"
           "// weights. KB = 4 rows from --kb 4 runs, KB = 5 rows from --kb mix runs (EXL3's 4.5 bpw\n"
           "// allocation); the JSON beside each run has every candidate's time.\n"
           "//\n";
      if (args_.tp == 2) {
        o << "// TP = 2: one rank's (N, K) (docs/trellis-kernel.md 2.4, 5.5; the tool's --tp 2). linear.cpp\n"
             "// includes this file inside namespace trellis_tp2, and a TP rank thread\n"
             "// (SetTp2TuningForThisThread) reads it before the TP = 1 trellis rows, so a rank shape that equals\n"
             "// a TP = 1 shape (the rank's attn.qg is gdn.in_proj_z's 6144 x 5120) still gets its own row.\n"
             "//\n";
      }
      o << "// M = 1 rows (--modes full --joint): every legal tuning under 10.1's rules (NT 1, MT 1, SKG <= 4,\n"
           "// >= 2 waves per SIMD) screened per (N, K, KB) on flushed chains by t(M = 1) + t(M = 8), the best\n"
           "// four then compared on the whole full-linear decode step (input transforms + GEMMs with the\n"
           "// epilogue), paired, at M = 1 and M = 8 together. A trellis chunk of M <= 16 rows always runs its\n"
           "// M = 1 row (row identity, 4.3), so these serve decode, MTP verify and DFlash verify alike; NT = 1\n"
           "// at every M <= 16 (10.1: trellis must not take w4a16's NT = 0 rule for M > 1).\n"
           "//\n"
           "// M = 32 and M = 64 rows (--modes ptune, M5): prefill chunks. Every legal tuning (MT up to the\n"
           "// chunk's row tiles, a split 128-group only with all of them in its block, >= 1 wave per SIMD)\n"
           "// screened per (N, K, KB) on flushed chains of the full GEMM at that M, the best four and each of\n"
           "// them with NT flipped then compared on the whole prefill chunk (input transforms + GEMMs), paired.\n"
           "// linear.cpp's BestRow serves a chunk of M rows from the smallest row M >= it that fits it, so\n"
           "// M = 17..32 run the M = 32 row and M = 33..64 the M = 64 row; a key without them takes 4.4's\n"
           "// fallback.\n"
           "//\n"
           "// Row format, for the structs as docs/trellis-kernel.md 5.3 extends them (src/model/linear.h;\n"
           "// linear.cpp includes this file inside its own namespace and PickTuning reads it for every trellis\n"
           "// linear; tests/kernels/trellis_tuning_rows.hpp reads the same file for test_trellis_gemm's\n"
           "// row-identity check and the bench's `--tuning-file table`):\n"
           "//   {Layout::kTrellis, N, K, M, {WV, SK, MB = MT, NPW = NP, NT, SKG, U}, group = 0, rate = KB}\n"
           "// i.e. LinearTuning's `int SKG = 1, U = 2` after NT and GemmTuningRow's `int rate = 0` after group.\n"
           "// The array has the main table's name, like gemm_tuning_table_tp2.inc: include it inside its own\n"
           "// namespace.\n"
           "static const GemmTuningRow kGemmTuningTable[] = {\n";
      for (const Row& r : rows) {
        const auto nm = names.find({std::get<0>(r.k), std::get<1>(r.k)});
        char b[256];
        std::snprintf(b, sizeof b, "    {Layout::kTrellis, %d, %d, %d, {%d, %d, %d, %d, %d, %d, %d}, 0, %d},  // %s Wc %d\n",
                      std::get<0>(r.k), std::get<1>(r.k), r.M, r.t.WV, r.t.SK, r.t.MT, r.t.NP, r.t.NT, r.t.SKG,
                      r.t.U, std::get<2>(r.k), nm == names.end() ? "?" : nm->second.c_str(), r.t.Wc());
        o << b;
      }
      o << "};\n";
      std::ofstream f(args_.inc_out, std::ios::binary);
      if (!f) throw std::runtime_error("cannot write " + args_.inc_out);
      f << o.str();
      std::printf("wrote %s (%zu rows)\n", args_.inc_out.c_str(), rows.size());
    }
  }

  // One-wave clock probes after every linear of a full chain (untimed): trellis (transform + GEMM)
  // or q2ab (cast + GEMM).
  json ProbePassFull(const std::vector<const Linear*>& all, int M, const KeyTunes& tune, bool trellis) {
    const size_t n = std::min<size_t>(all.size(), 4096);
    for (size_t i = 0; i < n; ++i) {
      if (trellis) {
        LaunchTransform(*all[i], M);
        LaunchTrellisFull(*all[i], M, tune.at(KeyOf(*all[i])));
      } else {
        LaunchCast(*all[i], M);
        LaunchBaseline(*all[i], M);
      }
      r4dx_tq_bench_clock_probe(P(probe_ + 8 * i), args_.probe_iters, P(st_));
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(st_));
    R4DX_HIP_CHECK(hipGetLastError());
    std::vector<unsigned long long> pr(8 * n);
    R4DX_HIP_CHECK(hipMemcpy(pr.data(), probe_, pr.size() * 8, hipMemcpyDeviceToHost));
    std::vector<double> mhz;
    for (size_t i = 0; i < n; ++i) {
      const double dc = static_cast<double>(pr[8 * i + 2] - pr[8 * i]);
      const double dw = static_cast<double>(pr[8 * i + 3] - pr[8 * i + 1]);
      if (dw > 0) mhz.push_back(dc / dw * wall_mhz_);
    }
    return {{"median_mhz", Median(mhz)}, {"p25_mhz", Quantile(mhz, 0.25)}, {"p75_mhz", Quantile(mhz, 0.75)},
            {"min_mhz", Min(mhz)}, {"probes", mhz.size()}};
  }

  json ReplayFullAt(int M, const KeyTunes& tune) {
    const std::vector<const Linear*> all = All();
    auto T = [&](TChain w) {
      return Must(Chain([&] { LaunchTChain(all, M, tune, w); }, "trellis step"), "trellis step");
    };
    auto B = [&](BChain w) { return Must(Chain([&] { LaunchBChain(all, M, w); }, "w4a16 step"), "w4a16 step"); };
    for (int i = 0; i < 2; ++i) T(TChain::kFull), B(BChain::kFull);
    // Seven chains per rep in a rotating order; every derived quantity is taken per rep.
    constexpr int kChains = 7;
    std::vector<double> t_raw, t_full, t_gemm, t_xform, b_gemm, b_full, b_extra;
    for (int r = 0; r < args_.reps; ++r)
      for (int k = 0; k < kChains; ++k) switch ((r + k) % kChains) {
          case 0: t_raw.push_back(T(TChain::kRaw)); break;
          case 1: t_full.push_back(T(TChain::kFull)); break;
          case 2: b_gemm.push_back(B(BChain::kGemm)); break;
          case 3: b_full.push_back(B(BChain::kFull)); break;
          case 4: t_gemm.push_back(T(TChain::kGemm)); break;
          case 5: t_xform.push_back(T(TChain::kXform)); break;
          case 6: b_extra.push_back(B(BChain::kExtras)); break;
        }
    const size_t n = t_raw.size();
    std::vector<double> S(n), X(n), D(n), Xc(n), epi(n);
    for (size_t r = 0; r < n; ++r) {
      S[r] = b_gemm[r] - t_raw[r];
      X[r] = (t_full[r] - t_raw[r]) - (b_full[r] - b_gemm[r]);
      D[r] = t_full[r] - b_full[r];
      epi[r] = t_gemm[r] - t_raw[r];
      Xc[r] = t_xform[r] + epi[r] - b_extra[r];
    }
    size_t tbytes = 0, tsbytes = 0, bbytes = 0;
    for (const Linear* l : all) tbytes += l->TrellisBytes(), tsbytes += l->TrellisScaleBytes(), bbytes += l->BaselineBytes();
    json j;
    json tj;
    for (auto& kv : tune) tj[KeyName(kv.first)] = kv.second.Json();
    j["trellis_tuning"] = tj;
    j["linears"] = all.size();
    auto put = [&](const char* name, const std::vector<double>& v) {
      j[name] = Spread(v);
      j[name]["all"] = v;
    };
    put("trellis_raw_gemm_ms", t_raw);
    put("trellis_full_ms", t_full);
    put("trellis_full_gemm_ms", t_gemm);
    put("trellis_transform_ms", t_xform);
    put("w4a16_gemm_ms", b_gemm);
    put("w4a16_full_ms", b_full);
    put("w4a16_extras_ms", b_extra);
    j["S_ms"] = Spread(S);
    j["X_ms"] = Spread(X);
    j["X_minus_S_ms"] = Spread(D);
    j["X_from_components_ms"] = Spread(Xc);
    j["epilogue_and_tails_ms"] = Spread(epi);
    j["trellis_bytes"] = tbytes;
    j["trellis_scale_bytes"] = tsbytes;
    j["w4a16_bytes"] = bbytes;
    const double x = Median(X), d = Median(D);
    j["gate"] = {{"X_le_0.30_ms", x <= 0.30},
                 {"X_le_1.3_ms_hard_cap", x <= 1.3},
                 {"X_minus_S_le_0.14_ms", d <= 0.14},
                 {"X_minus_S_upper_quartile_le_0.14_ms", Quantile(D, 0.75) <= 0.14},
                 {"rule", "X <= 0.30 ms per token (design; 10.1: hard cap ~1.3) and X - S <= 0.14 ms, medians of per-rep values"}};
    // q2ab's extras one kind at a time, and its silu_mul Hadamard over the plain silu_mul v1 runs
    // (information: A3 sees it, the X of 4.6 point 3 does not count it).
    {
      std::vector<const Linear*> cast_ls;
      int n_out = 0;
      for (const Linear* l : all) {
        if (l->group != 0) cast_ls.push_back(l);
        n_out += l->s == &kOutProj;
      }
      auto med = [&](const std::function<void()>& f) {
        Must(Chain(f, "extras"), "extras");
        std::vector<double> v;
        for (int r = 0; r < std::max(3, args_.reps / 2); ++r) v.push_back(Must(Chain(f, "extras"), "extras"));
        return Median(v);
      };
      const int n_mlp = static_cast<int>(Instances(&kDown).size());
      j["w4a16_extras_by_kind_ms"] = {
          {"casts", med([&] { for (const Linear* l : cast_ls) LaunchCast(*l, M); })},
          {"cast_count", cast_ls.size()},
          {"gdn_hadamards", med([&] { for (int i = 0; i < n_out; ++i) LaunchGdnHadamard(M); })},
          {"gdn_hadamard_count", n_out},
          {"residual_rotations", med([&] { LaunchRotation(M, false); LaunchRotation(M, true); })},
          {"trellis_transforms", med([&] { for (const Linear* l : all) LaunchTransform(*l, M); })},
          {"transform_count", all.size()}};
      const double sh = med([&] {
        for (int i = 0; i < n_mlp; ++i)
          r4dx_silu_mul_hadamard_bf16(P(silu_in_), P(silu_out_), M, 17408, 34816, P(st_), r4dx_epilogue_none, 0, 0,
                                      P(silu_signs_), 512);
      });
      const double sp = med([&] {
        for (int i = 0; i < n_mlp; ++i) r4dx_silu_mul_bf16(P(silu_in_), P(silu_out_), M, 17408, 34816, P(st_));
      });
      j["info_silu_mul"] = {{"q2ab_hadamard_ms", sh}, {"plain_ms", sp}, {"q2ab_minus_plain_ms", sh - sp},
                            {"count", n_mlp},
                            {"note", "v1 trellis runs the plain silu_mul where q2ab runs the Hadamard one; not in X"}};
      // attention.o's input: q2ab's gate-mul rotates it in the same launch (block = head_dim 256),
      // v1 runs the plain gate-mul and rotates in the transform. [M, 24 heads x 256] bf16.
      const int n_attn = static_cast<int>(Instances(&kAttnO).size());
      const int64_t n_gate = static_cast<int64_t>(M) * 6144;
      const double gh = med([&] {
        for (int i = 0; i < n_attn; ++i)
          r4dx_model_attn_gate_mul_hadamard_bf16(P(out_core_), P(gate_in_), P(gated_), n_gate, P(st_),
                                                 P(had_signs_), 256, 6144);
      });
      const double gp = med([&] {
        for (int i = 0; i < n_attn; ++i)
          r4dx_model_attn_gate_mul_bf16(P(out_core_), P(gate_in_), P(gated_), n_gate, P(st_));
      });
      j["info_attn_gate_mul"] = {
          {"q2ab_hadamard_ms", gh}, {"plain_ms", gp}, {"q2ab_minus_plain_ms", gh - gp}, {"count", n_attn},
          {"note", "v1 trellis runs the plain gate-mul where q2ab runs the Hadamard one; not in X"}};
    }
    // Per key, flushed, us per linear: M1's raw GEMM, the full GEMM, and the transform.
    json pk;
    const int kr = std::max(3, args_.reps / 3);
    for (const TKey& k : key_order_) {
      const std::vector<const Linear*>& ls = keys_.at(k);
      const TTune& t = tune.at(k);
      auto flushed = [&](const std::function<void()>& f) {
        Must(Chain(f, "per key"), "per key");
        std::vector<double> v;
        for (int r = 0; r < kr; ++r) {
          Flush();
          v.push_back(Must(Chain(f, "per key"), "per key"));
        }
        return 1000.0 * Median(v) / ls.size();
      };
      const double raw = flushed([&] { for (const Linear* l : ls) LaunchTrellis(*l, M, t, nullptr); });
      const double full = flushed([&] { for (const Linear* l : ls) LaunchTrellisFull(*l, M, t); });
      const double xf = flushed([&] { for (const Linear* l : ls) LaunchTransform(*l, M); });
      size_t b = 0;
      for (const Linear* l : ls) b += l->TrellisBytes();
      pk[KeyName(k)] = {{"classes", key_classes_.at(k)}, {"instances", ls.size()}, {"tuning", t.Json()},
                        {"raw_gemm_us", raw}, {"full_gemm_us", full}, {"epilogue_us", full - raw},
                        {"transform_us", xf}, {"full_gemm_gbps", b / ls.size() / full / 1e3}};
    }
    j["per_key"] = pk;
    // Clocks of the two full chains.
    const json pt = ProbePassFull(all, M, tune, true), pb = ProbePassFull(all, M, tune, false);
    j["probe_sclk"] = {{"trellis_full", pt}, {"w4a16_full", pb},
                       {"trellis_over_w4a16", pt["median_mhz"].get<double>() / pb["median_mhz"].get<double>()}};
    std::printf("full M=%d (%s): trellis %.3f ms [raw GEMMs %.3f, full GEMMs %.3f, transforms %.3f] | w4a16 %.3f ms "
                "[GEMMs %.3f, extras %.3f] per token\n",
                M, args_.kb == "mix" ? "4.5 mix" : "KB 4", Median(t_full), Median(t_raw), Median(t_gemm),
                Median(t_xform), Median(b_full), Median(b_gemm), Median(b_extra));
    std::printf("  S = %+.3f ms, X = %+.3f ms [IQR %+.3f..%+.3f] (components %+.3f), X - S = %+.3f ms [IQR %+.3f..%+.3f]"
                " -> X <= 0.30: %s, X - S <= 0.14: %s; probe SCLK trellis %.0f / w4a16 %.0f MHz\n",
                Median(S), x, Quantile(X, 0.25), Quantile(X, 0.75), Median(Xc), d, Quantile(D, 0.25),
                Quantile(D, 0.75), x <= 0.30 ? "yes" : "NO", d <= 0.14 ? "yes" : "NO",
                pt["median_mhz"].get<double>(), pb["median_mhz"].get<double>());
    for (auto& kv : pk.items())
      std::printf("  %-18s %-24s raw %8.2f us, full %8.2f (epilogue %+6.2f), transform %5.2f us per linear\n",
                  kv.key().c_str(), kv.value()["classes"].get<std::string>().c_str(),
                  kv.value()["raw_gemm_us"].get<double>(), kv.value()["full_gemm_us"].get<double>(),
                  kv.value()["epilogue_us"].get<double>(), kv.value()["transform_us"].get<double>());
    {
      const json &si = j["info_silu_mul"], &ga = j["info_attn_gate_mul"];
      std::printf("  info, outside X (v1 runs the plain kernel where q2ab runs its Hadamard form): silu_mul x%d "
                  "q2ab %.3f / plain %.3f ms, attn gate-mul x%d q2ab %.3f / plain %.3f ms -> v1 %+.3f ms per token\n",
                  si["count"].get<int>(), si["q2ab_hadamard_ms"].get<double>(), si["plain_ms"].get<double>(),
                  ga["count"].get<int>(), ga["q2ab_hadamard_ms"].get<double>(), ga["plain_ms"].get<double>(),
                  -(si["q2ab_minus_plain_ms"].get<double>() + ga["q2ab_minus_plain_ms"].get<double>()));
    }
    return j;
  }

  void ReplayFull() {
    json r;
    r["kb"] = args_.kb;
    r["kb_manifest"] = args_.kb_manifest;
    std::map<std::string, int> rates;
    size_t tbytes = 0;
    for (const Linear& l : linears_) ++rates["KB" + std::to_string(l.kb)], tbytes += l.TrellisBytes() + l.TrellisScaleBytes();
    r["linears_per_kb"] = rates;
    r["trellis_body_bytes_with_scales"] = tbytes;
    std::printf("full: trellis %s, %zu linears (", args_.kb == "mix" ? "4.5 bpw mix" : "KB 4", linears_.size());
    for (auto& kv : rates) std::printf(" %s: %d", kv.first.c_str(), kv.second);
    std::printf(" ), %.3f GiB of words and scales\n", tbytes / 1073741824.0);
    KeyTunes tune = LoadTunings();
    std::set<TKey> fixed;
    std::map<TKey, std::string> source;
    for (const TKey& k : key_order_) {
      if (tune.count(k)) {
        fixed.insert(k);
        source[k] = "file";
      } else {
        const TTune f = FallbackTuning(std::get<0>(k), std::get<1>(k), 1);
        tune[k] = f;
        source[k] = "fallback (4.4)";
      }
    }
    Warm();
    if (args_.joint) {
      json log;
      tune = JointSelect(tune, fixed, log);
      for (const TKey& k : key_order_)
        if (!fixed.count(k) || args_.joint_all) source[k] = "joint";
      r["joint"] = log;
    }
    json tj;
    for (const TKey& k : key_order_)
      tj[KeyName(k)] = {{"classes", key_classes_.at(k)}, {"instances", keys_.at(k).size()},
                        {"source", source.at(k)}, {"tuning", tune.at(k).Json()}};
    r["tunings"] = tj;
    WriteTunings(tune, source, LoadPrefillTunings());
    for (int M : {1, 8}) r["M" + std::to_string(M)] = ReplayFullAt(M, tune);
    out_["full"] = r;
  }

  // ---- prefill tuning (M5) -----------------------------------------------------------------------
  // Rows for prefill chunks (M > 16; docs/trellis-kernel.md 4.7, 5.6), per M of --ptune-m and per
  // (N, K, KB) that --tuning-file does not already give a row at that M (all of them with
  // --joint-all). Stage 1 screens every legal tuning on flushed chains of the key's instances, full
  // GEMM with its epilogue (the input transform does not depend on the tuning): MT a divisor of the
  // chunk's row tiles (a larger one only idles rows), a split 128-group (SKG > 1 or Wc < 128) only
  // with every row tile in its block (the kernel's rule), at least one wave per SIMD, NT 0; then the
  // best joint_top again with NT 1. Stage 2 takes each key's best joint_top into the whole prefill
  // chunk (every linear's transform and full GEMM at M), paired against the current pick, one key at
  // a time (largest first), and keeps a candidate when the chunk gets faster (median < -5 us, upper
  // quartile < 0), as the M = 1 joint pick does. Reports each key at the fallback (4.4) and at its
  // pick, and the whole chunk at the fallback, at the picks and for q2ab_hv2_q3's production path at
  // the same M (casts, w4a16 GEMMs at PickTuning's rows, GDN Hadamards and rotations: the linear-side
  // prefill ratio of A3p). --tunings-out / --inc-out write these rows with the M = 1 rows and any other
  // prefill rows that came in through --tuning-file.
  void PrefillTune() {
    json r;
    const KeyTunes decode = LoadTunings();
    PrefillTunes rows = LoadPrefillTunings();
    std::map<TKey, std::string> source;
    for (auto& kv : decode) source[kv.first] = "file";
    for (const TKey& k : key_order_)
      if (!decode.count(k))
        std::printf("  note: --tuning-file has no M = 1 row for %s; none is written for it\n", KeyName(k).c_str());
    const int simds = 4 * std::max(1, wgps_);
    const std::vector<const Linear*> all = All();
    Warm();
    for (int M : args_.ptune_m) {
      json jm;
      const int mtiles = (M + 15) / 16;
      KeyTunes fb, tune;
      std::map<TKey, std::vector<std::pair<double, TTune>>> top;
      std::map<TKey, double> weight;
      for (const TKey& k : key_order_) {
        fb[k] = FallbackTuning(std::get<0>(k), std::get<1>(k), M);
        const auto have = rows.find({k, M});
        if (have != rows.end() && !args_.joint_all) {
          tune[k] = have->second;
          jm["keys"][KeyName(k)] = {{"source", "file"}, {"tuning", have->second.Json()}};
          continue;
        }
        const std::vector<const Linear*>& ls = keys_.at(k);
        const int N = std::get<0>(k);
        std::vector<std::pair<double, TTune>> scored;
        json cand = json::array();
        for (int WV : {1, 2, 4})
          for (int NP : {1, 2, 4})
            for (int U : {1, 2, 4})
              for (int MT = 1; MT <= std::min(4, mtiles); ++MT)
                for (int SK : {1, 2, 4, 8, 16})
                  for (int SKG : {1, 2, 4}) {
                    const TTune t{WV, SK, MT, NP, SKG, U, 0};
                    const int Wc = t.Wc();
                    if (mtiles % MT != 0 || Wc > 256 || N % Wc != 0) continue;
                    if ((SKG > 1 || Wc < 128) && MT < mtiles) continue;
                    if ((N / Wc) * SKG * (mtiles / MT) * WV * SK < simds) continue;
                    const double ms = SweepTimeKey(ls, M, t);
                    if (ms < 0) continue;
                    scored.push_back({ms, t});
                    cand.push_back({{"tuning", t.Json()}, {"us", 1000 * ms}});
                  }
        auto by_time = [](const std::pair<double, TTune>& a, const std::pair<double, TTune>& b) {
          return a.first < b.first;
        };
        std::sort(scored.begin(), scored.end(), by_time);
        if (static_cast<int>(scored.size()) > args_.joint_top) scored.resize(args_.joint_top);
        for (size_t i = 0, n = scored.size(); i < n; ++i) {
          TTune t = scored[i].second;
          t.NT = 1;
          const double ms = SweepTimeKey(ls, M, t);
          if (ms < 0) continue;
          scored.push_back({ms, t});
          cand.push_back({{"tuning", t.Json()}, {"us", 1000 * ms}});
        }
        std::sort(scored.begin(), scored.end(), by_time);
        if (static_cast<int>(scored.size()) > args_.joint_top) scored.resize(args_.joint_top);
        const double t_fb = SweepTimeKey(ls, M, fb[k]);
        if (scored.empty()) {
          std::printf("  ptune %-18s M=%d: no legal tuning screened, keeping the fallback\n", KeyName(k).c_str(), M);
          tune[k] = fb[k];
          continue;
        }
        tune[k] = scored[0].second;
        top[k] = scored;
        weight[k] = scored[0].first * ls.size();
        json best = json::array();
        for (auto& s : scored) best.push_back({{"tuning", s.second.Json()}, {"us", 1000 * s.first}});
        jm["keys"][KeyName(k)] = {{"source", "ptune"}, {"classes", key_classes_.at(k)}, {"instances", ls.size()},
                                  {"fallback", {{"tuning", fb[k].Json()}, {"us", 1000 * t_fb}}},
                                  {"screened", cand.size()}, {"best", best}, {"all", cand}};
        std::printf("  ptune screen %-18s (%s) M=%d: %zu legal, best %s %.2f us per linear (fallback %s %.2f)\n",
                    KeyName(k).c_str(), key_classes_.at(k).c_str(), M, cand.size(), scored[0].second.Str().c_str(),
                    1000 * scored[0].first, fb[k].Str().c_str(), 1000 * t_fb);
      }
      // Stage 2: the whole prefill chunk.
      auto chunk = [&](const KeyTunes& tn) {
        return Must(Chain([&] { LaunchTChain(all, M, tn, TChain::kFull); }, "ptune chunk"), "ptune chunk");
      };
      std::vector<TKey> order;
      for (auto& kv : top) order.push_back(kv.first);
      std::sort(order.begin(), order.end(), [&](const TKey& a, const TKey& b) { return weight[a] > weight[b]; });
      json stage2 = json::array();
      for (const TKey& k : order) {
        for (size_t i = 1; i < top[k].size(); ++i) {
          KeyTunes alt = tune;
          alt[k] = top[k][i].second;
          if (alt[k] == tune[k]) continue;
          std::vector<double> d;
          for (int rep = 0; rep < args_.joint_reps; ++rep) {
            double c, a;
            if (rep % 2 == 0) c = chunk(tune), a = chunk(alt);
            else a = chunk(alt), c = chunk(tune);
            d.push_back(a - c);
          }
          const bool take = Median(d) < -0.005 && Quantile(d, 0.75) < 0.0;
          stage2.push_back({{"key", KeyName(k)}, {"current", tune[k].Json()}, {"candidate", alt[k].Json()},
                            {"delta_ms", Spread(d)}, {"taken", take}});
          std::printf("  ptune chunk %-18s M=%d %s -> %s: %+.3f ms per chunk%s\n", KeyName(k).c_str(), M,
                      tune[k].Str().c_str(), alt[k].Str().c_str(), Median(d), take ? "  TAKEN" : "");
          if (take) tune[k] = alt[k];
        }
      }
      jm["whole_chunk_stage"] = stage2;
      for (const TKey& k : key_order_) {
        if (top.count(k)) prefill_source_[{k, M}] = "ptune";
        rows[{k, M}] = tune[k];
        jm["picks"][KeyName(k)] = tune[k].Json();
      }
      // The whole chunk, paired: fallback, picks, q2ab.
      std::vector<double> c_fb, c_tn, c_q2;
      for (int rep = 0; rep < args_.reps; ++rep) {
        for (int j = 0; j < 3; ++j) switch ((rep + j) % 3) {
            case 0: c_fb.push_back(chunk(fb)); break;
            case 1: c_tn.push_back(chunk(tune)); break;
            case 2: c_q2.push_back(Must(Chain([&] { LaunchBChain(all, M, BChain::kFull); }, "q2ab chunk"), "q2ab chunk"));
                    break;
          }
      }
      jm["chunk_ms"] = {{"trellis_fallback", Spread(c_fb)}, {"trellis_tuned", Spread(c_tn)},
                        {"q2ab_production", Spread(c_q2)},
                        {"tuned_over_fallback", Median(c_tn) / Median(c_fb)},
                        {"q2ab_over_tuned", Median(c_q2) / Median(c_tn)}};
      std::printf("ptune M=%d: whole chunk (transforms + GEMMs, %zu linears) trellis fallback %.3f ms, tuned %.3f ms "
                  "(%.3fx); q2ab production %.3f ms -> linear-side prefill speed %.3fx q2ab\n",
                  M, all.size(), Median(c_fb), Median(c_tn), Median(c_tn) / Median(c_fb), Median(c_q2),
                  Median(c_q2) / Median(c_tn));
      r["M" + std::to_string(M)] = jm;
    }
    WriteTunings(decode, source, rows);
    out_["ptune"] = r;
  }

  // ---- split tail ------------------------------------------------------------------------------
  // Each split row (Wc, SKG) against the unsplit control of equal parallelism: Wc 128 (WV 4, NP 1),
  // SKG 1, SK 2 SKG -- the same number of waves, each over the same K range, so split - control is
  // the tail (partial store, fence, ticket, the last block's reduction) and not a change in grid
  // occupancy. SKG 8 has no control (SK 16 x WV 4 is 2048 threads).
  void Split() {
    json j;
    std::vector<const Linear*> ls = Instances(&kDown);
    if (ls.empty()) return;
    auto time = [&](int M, const TTune& t) {
      if (ChainTrellis(ls, M, t) < 0) return -1.0;
      std::vector<double> v;
      for (int r = 0; r < args_.reps; ++r) {
        const double ms = ChainTrellis(ls, M, t);
        if (ms < 0) return -1.0;
        v.push_back(ms);
      }
      return 1000.0 * Median(v) / ls.size();
    };
    Warm();
    for (int M : {1, 8}) {
      json jm;
      std::map<int, double> control;
      for (int SKG : {1, 2, 4, 8}) {
        const TTune c{4, 2 * SKG, 1, 1, 1, 2, 1};
        const double us = time(M, c);
        control[SKG] = us;
        jm["control"][std::to_string(SKG)] = us < 0 ? json("illegal") : json{{"tuning", c.Json()}, {"us_per_linear", us}};
      }
      for (int Wc : {32, 64, 128}) {
        for (int SKG : {1, 2, 4, 8}) {
          const TTune t{Wc / 32, 2, 1, 1, SKG, 2, 1};
          const double us = time(M, t);
          if (us < 0) {
            jm[std::to_string(Wc)][std::to_string(SKG)] = "rejected";
            continue;
          }
          const bool split = SKG > 1 || Wc < 128;
          json e = {{"tuning", t.Json()}, {"us_per_linear", us}, {"split", split},
                    {"contributors", SKG * std::max(1, 128 / Wc)}};
          if (control[SKG] > 0) {
            e["control_us"] = control[SKG];
            e["tail_us"] = us - control[SKG];
          }
          jm[std::to_string(Wc)][std::to_string(SKG)] = e;
          std::printf("split tail mlp.down M=%d Wc=%3d SKG=%d: %7.2f us per linear%s", M, Wc, SKG, us,
                      split ? " (split)" : "");
          if (control[SKG] > 0) std::printf(", control %7.2f, tail %+6.2f us", control[SKG], us - control[SKG]);
          std::printf("\n");
        }
      }
      j["M" + std::to_string(M)] = jm;
    }
    out_["split"] = j;
  }

  // ---- prefill ---------------------------------------------------------------------------------
  // Trellis: the best (WV, SK, SKG) per (MT, NP) over all of the shape's instances; then that best,
  // production's w4a16 and (with --sweep) a tuned w4a16, per w4a16 group (production runs each group
  // with its own tuning, and the g32 / g128 ones fall back).
  void Prefill() {
    json j;
    const int M = 64;
    Warm();
    for (const Shape* s : {&kGateUp, &kDown}) {
      std::vector<const Linear*> ls = Instances(s);
      if (ls.empty()) continue;
      json sj;
      const std::pair<int, int> kCombos[] = {{4, 1}, {2, 2}, {4, 2}};
      std::map<std::string, TTune> bests;
      for (auto mn : kCombos) {
        const int MT = mn.first, NP = mn.second;
        double best = 1e30;
        TTune bt{};
        json all;
        for (int WV : {1, 2, 4})
          for (int SK : {1, 2, 4, 8})
            for (int SKG : {1, 2, 4}) {
              const TTune t{WV, SK, MT, NP, SKG, 1, 0};
              if (ChainTrellis(ls, M, t) < 0) continue;
              std::vector<double> v;
              for (int r = 0; r < std::max(2, args_.reps / 3); ++r) v.push_back(ChainTrellis(ls, M, t));
              const double us = 1000.0 * Median(v) / ls.size();
              all.push_back({{"tuning", t.Json()}, {"us_per_linear", us}});
              if (us < best) best = us, bt = t;
            }
        const std::string key = "MT" + std::to_string(MT) + "_NP" + std::to_string(NP);
        bests[key] = bt;
        sj["trellis"][key] = {{"best", {{"tuning", bt.Json()}, {"us_per_linear", best}}}, {"all", all}};
        std::printf("prefill %-12s M=64 trellis (MT %d, NP %d): best %.2f us per linear (WV%d SK%d SKG%d)\n",
                    s->cls, MT, NP, best, bt.WV, bt.SK, bt.SKG);
      }
      for (const ClassKey& k : class_order_) {
        if (k.first != s) continue;
        const std::vector<const Linear*>& cl = classes_.at(k);
        auto reps_of = [&](const std::function<double()>& f) {
          f();
          std::vector<double> v;
          for (int r = 0; r < args_.reps; ++r) v.push_back(f());
          return 1000.0 * Median(v) / cl.size();
        };
        json g;
        for (auto& kv : bests) {
          const TTune t = kv.second;
          g["trellis_us"][kv.first] = reps_of([&] { return Must(ChainTrellis(cl, M, t), "prefill trellis"); });
        }
        const LinearTuning prod = ProductionTuning(*cl[0], M);
        const double wp = reps_of([&] { return Must(ChainBaseline(cl, M, nullptr), "prefill w4a16"); });
        g["w4a16_us"] = wp;
        g["w4a16_tuning"] = TuningJson(prod);
        g["w4a16_tuning_source"] = HasTableRow(k.second == 0 ? Layout::kBf16 : Layout::kW4a16, s->N, s->K, M,
                                               k.second) ? "table" : "fallback";
        g["instances"] = cl.size();
        double wt = -1.0;
        if (args_.sweep) {
          json log;
          const LinearTuning bt = SweepBaseline(k, M, log);
          wt = reps_of([&] { return Must(ChainBaseline(cl, M, &bt), "prefill tuned w4a16"); });
          g["w4a16_tuned_us"] = wt;
          g["w4a16_tuned_sweep"] = log;
        }
        std::printf("prefill %-22s M=64 w4a16 (production, %s): %.2f us per linear", ClassName(k).c_str(),
                    g["w4a16_tuning_source"].get<std::string>().c_str(), wp);
        if (wt > 0) std::printf(", tuned %.2f us", wt);
        std::printf("\n");
        sj["by_group"][ClassName(k)] = g;
      }
      size_t tb = 0, bb = 0;
      for (const Linear* l : ls) tb += l->TrellisBytes(), bb += l->BaselineBytes();
      sj["trellis_bytes_per_linear"] = tb / ls.size();
      sj["w4a16_bytes_per_linear"] = bb / ls.size();
      j[s->cls] = sj;
    }
    out_["prefill"] = j;
  }

  // ---- op rates and pipes ----------------------------------------------------------------------
  struct WaveStats {
    double mean_cyc = 0, max_cyc = 0;
    int waves = 0, simds = 0, max_per_simd = 0, min_per_simd = 0;
  };
  static WaveStats Stats(const std::vector<unsigned long long>& cyc, const std::vector<uint32_t>& hw, int n) {
    WaveStats s;
    std::map<uint32_t, int> per_simd;
    for (int i = 0; i < n; ++i) {
      s.mean_cyc += static_cast<double>(cyc[i]) / n;
      s.max_cyc = std::max(s.max_cyc, static_cast<double>(cyc[i]));
      // SIMD_ID [9:8], WGP_ID [13:10], SA_ID [16], SE_ID [20:18]: everything but WAVE_ID.
      per_simd[hw[i] & 0x1D3F00u]++;
    }
    s.waves = n;
    s.simds = static_cast<int>(per_simd.size());
    s.min_per_simd = 1 << 30;
    for (auto& kv : per_simd) {
      s.max_per_simd = std::max(s.max_per_simd, kv.second);
      s.min_per_simd = std::min(s.min_per_simd, kv.second);
    }
    return s;
  }

  void Ops() {
    json j;
    const char* names[] = {"v_mad_u32_u16", "v_pk_mad_u16", "v_sad_u8", "v_sad_hi_u8", "v_mul_lo_u32",
                           "v_alignbit_b32", "v_pk_fma_f16", "v_add_nc_u32", "v_fma_f32",
                           "v_mad_u32_u16 op_sel:[1,0,0,0] (decode)", "v_pk_mad_u16 op_sel_hi:[1,1,1] (decode)",
                           "v_pk_mad_u16 op_sel_hi:[1,0,1] (decode)", "v_sad_u8 0x64006400 (decode)",
                           "v_pk_fma_f16 0xc931 (decode)"};
    constexpr int kOps = 14;
    Blob buf(64ull << 20);
    auto* cyc = buf.Take<unsigned long long>(65536 * 8);
    auto* hw = buf.Take<uint32_t>(65536 * 4);
    auto* sink = buf.Take<uint32_t>(65536 * 4);
    auto* clk = buf.Take<unsigned long long>(64);
    auto* act = buf.Take<uint8_t>(1 << 20);
    Fill(act, (1 << 20) / 4, 5, 0xB7FFB7FFu, 0x20002000u);

    // HIP counts WGPs as its multiprocessors on gfx12 (32 on the R9700: 64 CUs in WGP mode).
    const int wgps = std::max(1, wgps_);
    const size_t total = 512ull << 20;               // >> the 64 MiB last-level cache
    Blob words(total + (1 << 20));
    Fill(words.base, total / 4, 99, ~0u, 0u);
    // w waves per SIMD: one workgroup of 4w waves per WGP up to 32 waves (1024 threads), then
    // 4w / 32 such workgroups per WGP. Steps per wave even (the pipe loop is a ping-pong pair).
    auto geometry = [&](int w, int& wgs, int& wpw, int& iters) {
      wpw = std::min(4 * w, 32);
      wgs = wgps * std::max(1, 4 * w / 32);
      iters = static_cast<int>(total / (static_cast<size_t>(wgs) * wpw * 512)) & ~1;
    };
    auto pipe = [&](int mode, int w) {
      int wgs, wpw, iters;
      geometry(w, wgs, wpw, iters);
      r4dx_tq_bench_pipe(mode, wgs, wpw, iters, P(words.base), P(act), static_cast<int64_t>(iters) * 512,
                         P(cyc), P(hw), P(sink), P(clk), P(st_));
    };
    // Warm-up: ~0.3 s of the decode loop on every SIMD. Without it the first op-rate loops of a cold
    // GPU measured 2-3x fewer ops per counted clock than the same loops a second later.
    timer_.Ms([&] {
      for (int i = 0; i < 300; ++i) pipe(3, 4);
    });
    {
      // The same probe as the calibration, now warm: its cycles per add must match the cold ones.
      r4dx_tq_bench_clock_probe(P(clk), 100000, P(st_));
      R4DX_HIP_CHECK(hipStreamSynchronize(st_));
      unsigned long long k[4];
      R4DX_HIP_CHECK(hipMemcpy(k, clk, sizeof k, hipMemcpyDeviceToHost));
      const double dc = static_cast<double>(k[2] - k[0]), dw = static_cast<double>(k[3] - k[1]);
      j["warm_probe"] = {{"cycles_per_add", dc / 8e5}, {"sclk_mhz", dc / dw * wall_mhz_},
                         {"cold_cycles_per_add", out_["clock_calibration"]["cycles_per_add"]["median"]}};
      std::printf("warm probe: %.3f cycles per dependent add (cold %.3f), SCLK %.0f MHz\n", dc / 8e5,
                  out_["clock_calibration"]["cycles_per_add"]["median"].get<double>(), dc / dw * wall_mhz_);
    }

    // One op-rate measurement: one workgroup of 4 SIMDs x w waves (WGP mode), best of three; ops
    // per clock per SIMD = the busiest SIMD's waves over the longest of them.
    auto op_rate = [&](int op, int dist, bool delay, int w) {
      const int iters = 4096, wpw = 4 * w;          // 24 ops per iteration per wave
      double rate = 0.0;
      WaveStats s;
      for (int rep = 0; rep < 3; ++rep) {
        r4dx_tq_bench_op_rate(op, dist, delay ? 1 : 0, 1, wpw, iters, P(cyc), P(hw), P(sink), P(st_));
        R4DX_HIP_CHECK(hipStreamSynchronize(st_));
        std::vector<unsigned long long> c(wpw);
        std::vector<uint32_t> h(wpw);
        R4DX_HIP_CHECK(hipMemcpy(c.data(), cyc, wpw * 8, hipMemcpyDeviceToHost));
        R4DX_HIP_CHECK(hipMemcpy(h.data(), hw, wpw * 4, hipMemcpyDeviceToHost));
        const WaveStats sr = Stats(c, h, wpw);
        const double r = sr.max_per_simd * (24.0 * iters) / sr.max_cyc;
        if (r > rate) rate = r, s = sr;
      }
      return std::make_pair(rate, s);
    };
    for (int op = 0; op < kOps; ++op) {
      for (int w : {1, 2, 4}) {
        const auto rs = op_rate(op, 8, false, w);
        j["op_rates"][names[op]][std::to_string(w) + "w"] = {
            {"ops_per_clk_per_simd", rs.first}, {"waves", rs.second.waves}, {"simds", rs.second.simds},
            {"max_waves_per_simd", rs.second.max_per_simd}, {"max_cycles", rs.second.max_cyc}};
        std::printf("op %-40s %d wave(s)/SIMD: %.3f per clock per SIMD (issue rate, 8 independent chains)\n",
                    names[op], w, rs.first);
      }
    }
    // Near dependencies: every op reads the one `dist` before it, without and with the
    // s_delay_alu the compiler emits for its own instructions (never for inline asm). At 4 waves per
    // SIMD a stall that holds only its own wave is hidden by the others; one that blocks the SIMD
    // is not.
    for (int op : {7, 9, 10, 12, 13}) {
      for (int dist : {1, 2, 3, 4}) {
        for (bool delay : {false, true}) {
          for (int w : {1, 4}) {
            const auto rs = op_rate(op, dist, delay, w);
            j["dependency"][names[op]]["d" + std::to_string(dist) + (delay ? "_delay" : "")]
             [std::to_string(w) + "w"] = rs.first;
            std::printf("dep %-40s distance %d%-14s %d wave(s)/SIMD: %.3f per clock per SIMD\n", names[op],
                        dist, delay ? " + s_delay_alu" : "", w, rs.first);
          }
        }
      }
    }

    // Pipes over the whole GPU at 1-16 waves per SIMD.
    const char* modes[] = {"mem", "valu", "wmma", "alu", "alu_indep", "all"};
    for (int w : {1, 2, 4, 8, 16}) {
      int wgs, wpw, iters_w;
      geometry(w, wgs, wpw, iters_w);
      const int waves = wgs * wpw;
      std::map<std::string, double> us;
      const std::string wk = std::to_string(w) + "w";
      for (int mode = 0; mode < 6; ++mode) {
        auto run = [&] { return timer_.Ms([&] { pipe(mode, w); }); };
        run();
        std::vector<double> v;
        for (int r = 0; r < 5; ++r) v.push_back(run());
        std::vector<unsigned long long> c(waves), k(4);
        std::vector<uint32_t> h(waves);
        R4DX_HIP_CHECK(hipMemcpy(c.data(), cyc, waves * 8, hipMemcpyDeviceToHost));
        R4DX_HIP_CHECK(hipMemcpy(h.data(), hw, waves * 4, hipMemcpyDeviceToHost));
        R4DX_HIP_CHECK(hipMemcpy(k.data(), clk, 32, hipMemcpyDeviceToHost));
        const WaveStats s = Stats(c, h, waves);
        const double mhz = static_cast<double>(k[2] - k[0]) / std::max(1.0, static_cast<double>(k[3] - k[1])) * wall_mhz_;
        us[modes[mode]] = 1000.0 * Median(v);
        const double bytes = static_cast<double>(waves) * iters_w * 512;
        json e = {{"us", us[modes[mode]]}, {"mean_wave_cycles", s.mean_cyc}, {"max_wave_cycles", s.max_cyc},
                  {"cycles_per_step_max_wave", s.max_cyc / iters_w}, {"cycles_per_step_mean_wave", s.mean_cyc / iters_w},
                  {"sclk_mhz_wg0", mhz}, {"waves", waves}, {"steps_per_wave", iters_w}, {"simds", s.simds},
                  {"max_waves_per_simd", s.max_per_simd}, {"min_waves_per_simd", s.min_per_simd},
                  {"gbps_if_memory", bytes / (us[modes[mode]] * 1e-6) / 1e9}};
        // The decode's useful VALU (2 x 62 per step) per clock per SIMD, from the kernel time: the
        // SIMD-level issue rate the per-wave cycle counts cannot give once waves share a SIMD.
        if (mode == 1 || mode >= 3)
          e["decode_valu_per_clk_per_simd"] =
              s.max_per_simd * static_cast<double>(iters_w) * 124.0 / (us[modes[mode]] * mhz);
        j["pipes"][wk][modes[mode]] = e;
      }
      // WMMA/VALU overlap from the kernel times (the same work per SIMD in every mode, at about the
      // same clock -- sclk_mhz_wg0): 1 = the shorter pipe fully hidden under the longer, 0 = they
      // add. Per-wave cycles are not used: at 2+ waves per SIMD the waves of one SIMD are staggered,
      // so a wave's own lifetime undercounts the SIMD's busy time.
      auto ov = [&](const char* mixed) {
        const double a = us["wmma"], b = us["valu"], m = us[mixed];
        return (a + b - m) / std::max(1e-9, std::min(a, b));
      };
      // beta of docs/trellis-kernel.md 4.6: time = max(alu, mem) + beta * min(alu, mem). Every
      // mode runs the same decode block and the loads the GEMM issues, so the modes differ only in
      // what they leave out.
      const double tm = us["mem"], ta = us["alu"], tall = us["all"];
      const double beta = (tall - std::max(tm, ta)) / std::max(1e-9, std::min(tm, ta));
      j["pipes"][wk]["derived"] = {
          {"wmma_valu_overlap_dependent", ov("alu")}, {"wmma_valu_overlap_independent", ov("alu_indep")},
          {"beta_alu_vs_mem", beta}, {"exposed_alu_fraction", (tall - tm) / std::max(1e-9, ta)},
          {"alu_over_mem", ta / std::max(1e-9, tm)}};
      std::printf("pipes %2d wave(s)/SIMD: mem %.0f us, valu %.0f (%.2f VALU/clk/SIMD), wmma %.0f, alu %.0f, "
                  "alu_indep %.0f, all %.0f (%.2f); overlap dep %.2f indep %.2f; beta %.2f; alu/mem %.2f\n",
                  w, us["mem"], us["valu"], j["pipes"][wk]["valu"]["decode_valu_per_clk_per_simd"].get<double>(),
                  us["wmma"], us["alu"], us["alu_indep"], us["all"],
                  j["pipes"][wk]["all"]["decode_valu_per_clk_per_simd"].get<double>(), ov("alu"), ov("alu_indep"),
                  beta, ta / std::max(1e-9, tm));
    }
    out_["ops"] = j;
  }

  // M5 part 3 (mode disp): does a trellis GEMM's time depend on where the dispatcher starts its
  // workgroups? The in-model probe found the same GEMM 5-7% apart from layer to layer, in a pattern
  // fixed by the launch sequence (not by weight addresses or time). Before every timed GEMM a pad
  // of k one-wave workgroups (k drawn per rep) and a one-workgroup locator run; the locator records
  // the HW_ID1 of the slot it got. `a`: one rep per synchronize (the Timer); `b`: 64 reps enqueued
  // back to back, stamped, one synchronize (the model's shape). Rows: [k, hw_id1, us].
  void Disp() {
    json j;
    uint32_t* hw = reinterpret_cast<uint32_t*>(clk_);   // [0, 256): the pads' writes; then locators
    uint32_t rng = 12345;
    const auto next_k = [&] {
      rng = rng * 1664525u + 1013904223u;
      return static_cast<int>((rng >> 8) % 97u);
    };
    for (const Shape* s : {&kGateUp, &kDown}) {
      const std::vector<const Linear*> inst = Instances(s);
      const Linear& l0 = *inst[0];
      TTune t{};
      bool found = false;
      for (const trellis_rows::GemmTuningRow& row : trellis_rows::kGemmTuningTable) {
        if (row.layout == trellis_rows::Layout::kTrellis && row.N == s->N && row.K == s->K && row.M == 1 &&
            row.rate == l0.kb) {
          const trellis_rows::LinearTuning& r = row.tuning;
          t = TTune{r.WV, r.SK, r.MB, r.NPW, r.SKG, r.U, r.NT};
          found = true;
        }
      }
      if (!found) throw std::runtime_error(std::string("disp: no M = 1 table row for ") + s->cls);
      json c;
      c["tuning"] = t.Json();
      c["instances"] = inst.size();
      // Warm up.
      for (int i = 0; i < 64; ++i) LaunchTrellisFull(*inst[i % inst.size()], 1, t);
      R4DX_HIP_CHECK(hipStreamSynchronize(st_));
      json a = json::array();
      for (int rep = 0; rep < 1536; ++rep) {
        const int k = next_k();
        if (k > 0) r4dx_tq_bench_where(P(hw), k, 32, P(st_));
        r4dx_tq_bench_where(P(hw + 256), 1, 32, P(st_));
        const double ms = timer_.Ms([&] { LaunchTrellisFull(*inst[rep % inst.size()], 1, t); });
        uint32_t id = 0;
        R4DX_HIP_CHECK(hipMemcpy(&id, hw + 256, 4, hipMemcpyDeviceToHost));
        a.push_back({k, id, ms * 1000.0});
      }
      c["a"] = a;
      json b = json::array();
      unsigned long long* st = probe_;   // 2 stamps per rep
      for (int round = 0; round < 24; ++round) {
        std::vector<int> ks;
        for (int rep = 0; rep < 64; ++rep) {
          const int k = next_k();
          ks.push_back(k);
          if (k > 0) r4dx_tq_bench_where(P(hw), k, 32, P(st_));
          r4dx_tq_bench_where(P(hw + 256 + rep), 1, 32, P(st_));
          r4dx_tq_bench_stamp(P(st + 2 * rep), P(st_));
          LaunchTrellisFull(*inst[rep % inst.size()], 1, t);
          r4dx_tq_bench_stamp(P(st + 2 * rep + 1), P(st_));
        }
        R4DX_HIP_CHECK(hipStreamSynchronize(st_));
        std::vector<unsigned long long> sv(128);
        std::vector<uint32_t> ids(64);
        R4DX_HIP_CHECK(hipMemcpy(sv.data(), st, 128 * 8, hipMemcpyDeviceToHost));
        R4DX_HIP_CHECK(hipMemcpy(ids.data(), hw + 256, 64 * 4, hipMemcpyDeviceToHost));
        for (int rep = 0; rep < 64; ++rep) {
          const double us = static_cast<double>(sv[2 * rep + 1] - sv[2 * rep]) / wall_mhz_ -
                            timer_.OverheadMs() * 1000.0;
          b.push_back({ks[rep], ids[rep], us});
        }
      }
      c["b"] = b;
      j[s->cls] = c;
      std::printf("disp %s: %zu + %zu reps\n", s->cls, a.size(), b.size());
    }
    out_["disp"] = j;
  }

  void Write() {
    std::ofstream f(args_.out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + args_.out);
    f << out_.dump(1) << "\n";
    std::printf("wrote %s\n", args_.out.c_str());
  }

  json& Out() { return out_; }

 private:
  template <typename T>
  void Fill(T* p, size_t words, uint32_t seed, uint32_t and_mask, uint32_t or_mask) {
    r4dx_tq_bench_fill_u32(P(p), static_cast<int64_t>(words), seed, and_mask, or_mask, P(st_));
  }

  const Args& args_;
  hipStream_t st_;
  Timer timer_;
  json out_;
  double wall_mhz_ = 100.0;
  int wgps_ = 32;
  static constexpr size_t kFlushBytes = 256ull << 20;
  const float kOutScale = static_cast<float>(1.0 / std::sqrt(128.0));   // 2^-s / sqrt(128), s = 0
  std::unique_ptr<Blob> small_, trellis_, base_, tick_, scales_, flush_;
  std::map<int, uint16_t*> a_f16_, a_f16b_, a_bf16_;
  float *rot_signs_ = nullptr, *rot_mix5_ = nullptr, *had_signs_ = nullptr, *silu_signs_ = nullptr;
  uint16_t *resid_ = nullptr, *out_core_ = nullptr, *silu_in_ = nullptr, *silu_out_ = nullptr;
  uint16_t *gate_in_ = nullptr, *gated_ = nullptr;
  std::map<TKey, std::vector<const Linear*>> keys_;
  std::vector<TKey> key_order_;
  std::map<TKey, std::string> key_classes_;
  std::map<PKey, std::string> prefill_source_;   // the prefill rows this run picked (ptune)
  float* c_f32_ = nullptr;
  uint16_t* c_bf16_ = nullptr;
  float* ws_ = nullptr;
  size_t ws_bytes_ = 0;
  unsigned long long* clk_ = nullptr;
  unsigned long long* probe_ = nullptr;
  uint32_t* sink_ = nullptr;
  std::vector<Linear> linears_;
  std::map<ClassKey, std::vector<const Linear*>> classes_;
  std::vector<ClassKey> class_order_;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
      return argv[++i];
    };
    if (k == "--out") a.out = next();
    else if (k == "--reps") a.reps = std::stoi(next());
    else if (k == "--layers") a.layers = std::stoi(next());
    else if (k == "--sweep") a.sweep = true;
    else if (k == "--sweep-reps") a.sweep_reps = std::stoi(next());
    else if (k == "--probe-iters") a.probe_iters = std::stoi(next());
    else if (k == "--in-model-sclk") a.in_model_sclk = std::stod(next());
    else if (k == "--alu-fraction") a.alu_fraction = std::stod(next());
    else if (k == "--warmup-s") a.warmup_s = std::stod(next());
    else if (k == "--kb") a.kb = next();
    else if (k == "--kb-manifest") a.kb_manifest = next();
    else if (k == "--tuning-file") a.tuning_files.push_back(next());
    else if (k == "--joint") a.joint = true;
    else if (k == "--joint-all") a.joint = a.joint_all = true;
    else if (k == "--joint-top") a.joint_top = std::stoi(next());
    else if (k == "--joint-reps") a.joint_reps = std::stoi(next());
    else if (k == "--tunings-out") a.tunings_out = next();
    else if (k == "--inc-out") a.inc_out = next();
    else if (k == "--tp") a.tp = std::stoi(next());
    else if (k == "--ptune-m") {
      a.ptune_m.clear();
      std::string v = next();
      size_t p = 0;
      while (p <= v.size()) {
        const size_t q = v.find(',', p);
        const int m = std::stoi(v.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (m <= 16 || m > 64) throw std::runtime_error("--ptune-m: every M must be 17..64 (M <= 16 is the M = 1 row)");
        a.ptune_m.push_back(m);
        if (q == std::string::npos) break;
        p = q + 1;
      }
    } else if (k == "--modes") {
      a.modes.clear();
      std::string v = next();
      size_t p = 0;
      while (p <= v.size()) {
        const size_t q = v.find(',', p);
        const std::string m = v.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (m != "replay" && m != "ops" && m != "split" && m != "prefill" && m != "full" && m != "ptune" &&
            m != "disp")
          throw std::runtime_error("unknown mode '" + m + "' (replay, ops, split, prefill, full, ptune, disp)");
        a.modes.insert(m);
        if (q == std::string::npos) break;
        p = q + 1;
      }
    } else {
      throw std::runtime_error("unknown argument " + k +
                               " (--out <json> [--modes replay,ops,split,prefill,full,ptune,disp] [--reps N] "
                               "[--layers N] [--sweep] [--sweep-reps N] [--probe-iters N] "
                               "[--in-model-sclk MHz] [--alu-fraction F] [--warmup-s S] [--kb 4|mix] "
                               "[--kb-manifest json] [--tuning-file json|table]... [--joint] [--joint-all] "
                               "[--joint-top N] [--joint-reps N] [--tunings-out json] [--inc-out path] "
                               "[--ptune-m 32,64] [--tp 1|2])");
    }
  }
  if (a.tp != 1 && a.tp != 2) throw std::runtime_error("--tp must be 1 or 2");
  if (a.tp == 2) {
    for (const std::string& m : a.modes)
      if (m != "full" && m != "ptune")
        throw std::runtime_error("--tp 2 serves the tuning modes only (full, ptune), not " + m);
  }
  if (a.out.empty()) throw std::runtime_error("--out <json> is required");
  if (a.kb != "4" && a.kb != "mix") throw std::runtime_error("--kb must be 4 or mix");
  if (!a.kb_manifest.empty() && a.kb != "mix") throw std::runtime_error("--kb-manifest needs --kb mix");
  if (a.joint_top < 1 || a.joint_reps < 1) throw std::runtime_error("--joint-top / --joint-reps must be >= 1");
  if (a.layers < 1 || a.layers > 64) throw std::runtime_error("--layers must be 1..64");
  if (a.reps < 1) throw std::runtime_error("--reps must be >= 1");
  if (a.in_model_sclk < 0) throw std::runtime_error("--in-model-sclk must be a positive MHz value");
  if (a.alu_fraction < 0 || a.alu_fraction > 1) throw std::runtime_error("--alu-fraction must be 0..1");
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    R4DX_HIP_CHECK(hipSetDevice(0));
    hipStream_t st;
    R4DX_HIP_CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    if (args.tp == 2) ShardShapesTp2();
    {
      Bench b(args, st);
      const bool model = args.modes.count("replay") || args.modes.count("split") || args.modes.count("prefill") ||
                         args.modes.count("full") || args.modes.count("ptune") || args.modes.count("disp");
      b.Setup();
      b.Out()["args"] = {{"reps", args.reps}, {"layers", args.layers}, {"sweep", args.sweep},
                         {"sweep_reps", args.sweep_reps}, {"in_model_sclk", args.in_model_sclk},
                         {"alu_fraction", args.alu_fraction}, {"warmup_s", args.warmup_s},
                         {"modes", std::vector<std::string>(args.modes.begin(), args.modes.end())},
                         {"kb", args.kb}, {"kb_manifest", args.kb_manifest}, {"tuning_files", args.tuning_files},
                         {"joint", args.joint}, {"joint_all", args.joint_all}, {"joint_top", args.joint_top},
                         {"joint_reps", args.joint_reps}, {"ptune_m", args.ptune_m}, {"tp", args.tp}};
      // The JSON is rewritten after every mode, so a later failure keeps the earlier numbers.
      if (args.modes.count("ops")) b.Ops(), b.Write();
      if (model) b.SetupModel();
      if (args.modes.count("full")) b.ReplayFull(), b.Write();
      if (args.modes.count("ptune")) b.PrefillTune(), b.Write();
      if (args.modes.count("replay")) b.Replay(), b.Write();
      if (args.modes.count("split")) b.Split(), b.Write();
      if (args.modes.count("prefill")) b.Prefill(), b.Write();
      if (args.modes.count("disp")) b.Disp(), b.Write();
      if (!model && !args.modes.count("ops")) b.Write();
    }
    R4DX_HIP_CHECK(hipStreamDestroy(st));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tool_trellis_gemm_bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
