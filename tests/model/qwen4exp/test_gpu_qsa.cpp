// QsaLayerCore (src/model/qwen4exp/include/r4dx/model/qwen4exp/qsa.h) against HF `Qwen4ExpTextAttention` with the
// QSA indexer (`tie_rule="recent"`, plan Q23). Fixtures: tools/reference/qwen4exp/qsa_fixtures.py, default dir
// E:\models\r4dx\q4exp-fixtures\QsaCore (override: R4DX_QSA_FIXTURES). Exit 77 when the fixtures are missing.
//
//   1. attention cases (budget 64 and 2048): chunked prefill + decode; per chunk the gated attention output vs HF's
//      bf16 graph (`out`) and vs an fp32 attention on the same selected sets (`out32`), the selected token sets
//      (must be EXACTLY equal), the fp8 K/V bytes in the cache, and at the end the raw index keys and the block keys
//      (bit equal). Then a rewind: poison the state beyond a position, TruncateTo(pos), replay the tail chunks.
//   2. selection cases (budget 2048, 2051 / 2052 / 3000 / 8192 tokens): the selected sets of every sparse row of a
//      chunked prefill must equal the Python reference exactly, ties included (the fixtures report the tied rows).
//   3. timing + finite output: a 262144-token layer (budget 2048) prefilled in 256-token chunks with random data;
//      at 2048 / 32768 / 131072 / 262128 tokens, 16 decode steps are timed. Skip with R4DX_QSA_PERF=0;
//      R4DX_QSA_PERF_CTX overrides the context length (a multiple of 256, >= 4096).
// Run on the headless card: HIP_VISIBLE_DEVICES=1 (the ctest label does it).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/qwen4exp/qsa.h"

using r4dx::core::Arena;
using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::FloatToBf16;
using r4dx::model::attention::PagedKvCache;
using r4dx::model::qwen4exp::QsaDims;
using r4dx::model::qwen4exp::QsaInputs;
using r4dx::model::qwen4exp::QsaLayerCore;
using r4dx::model::qwen4exp::QsaWeights;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

// ---- fixture reader (index.txt / params.txt / <key>.bin, as tools/reference/qwen4exp/qsa_fixtures.py) ----------
struct Fixture {
  struct Entry {
    std::string kind;
    std::vector<int64_t> shape;
  };
  std::string dir;
  std::map<std::string, Entry> index;
  std::map<std::string, std::string> params;

  bool Load(const std::string& d) {
    dir = d;
    std::ifstream fi(dir + "/index.txt");
    std::ifstream fp(dir + "/params.txt");
    if (!fi || !fp) return false;
    std::string line;
    while (std::getline(fi, line)) {
      std::istringstream ss(line);
      std::string key;
      Entry e;
      int rank = 0;
      if (!(ss >> key >> e.kind >> rank)) continue;
      for (int i = 0; i < rank; ++i) {
        int64_t v = 0;
        ss >> v;
        e.shape.push_back(v);
      }
      index[key] = e;
    }
    while (std::getline(fp, line)) {
      const size_t sp = line.find(' ');
      if (sp != std::string::npos) params[line.substr(0, sp)] = line.substr(sp + 1);
    }
    return true;
  }
  bool Has(const std::string& key) const { return index.count(key) != 0; }
  int Int(const std::string& key) const {
    const auto it = params.find(key);
    if (it == params.end()) throw std::runtime_error("fixture param missing: " + key);
    return std::stoi(it->second);
  }
  const Entry& Meta(const std::string& key) const {
    const auto it = index.find(key);
    if (it == index.end()) throw std::runtime_error("fixture tensor missing: " + key);
    return it->second;
  }
  template <class T>
  std::vector<T> Read(const std::string& key, const char* kind) const {
    const Entry& e = Meta(key);
    if (e.kind != kind) throw std::runtime_error("fixture tensor " + key + " is " + e.kind + ", wanted " + kind);
    int64_t n = 1;
    for (int64_t d : e.shape) n *= d;
    std::vector<T> out(static_cast<size_t>(n));
    std::ifstream f(dir + "/" + key + ".bin", std::ios::binary);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(T)));
    if (!f) throw std::runtime_error("short read of fixture tensor " + key);
    return out;
  }
  std::vector<uint16_t> Bf16(const std::string& k) const { return Read<uint16_t>(k, "bf16"); }
  std::vector<float> F32(const std::string& k) const { return Read<float>(k, "f32"); }
  std::vector<int32_t> I32(const std::string& k) const { return Read<int32_t>(k, "i32"); }
};

template <class T>
DeviceBuffer<T> Upload(const std::vector<T>& h) {
  DeviceBuffer<T> d(h.size());
  d.CopyFromHost(h);
  return d;
}

std::vector<int32_t> Positions(int first, int t) {
  std::vector<int32_t> p(static_cast<size_t>(t));
  for (int i = 0; i < t; ++i) p[static_cast<size_t>(i)] = first + i;
  return p;
}

struct ErrStats {
  double max_abs = 0.0, sum_abs = 0.0, max_ref = 0.0;
  int64_t n = 0, bad_nonfinite = 0;
  void Add(const uint16_t* got, const uint16_t* ref, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
      const double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
      if (!std::isfinite(g)) ++bad_nonfinite;
      const double e = std::fabs(g - r);
      max_abs = std::max(max_abs, e);
      sum_abs += e;
      max_ref = std::max(max_ref, std::fabs(r));
    }
    n += count;
  }
  double Mean() const { return n ? sum_abs / static_cast<double>(n) : 0.0; }
};

// Compares the core's selection of the last Forward with the fixture's lists. Returns the number of mismatching
// sparse rows; `sparse_rows` counts the rows compared.
int CompareSelection(const QsaLayerCore& core, const std::vector<int32_t>& fx_sel, const std::vector<int32_t>& fx_cnt,
                     int T, int row_base, int* sparse_rows) {
  const int stride = core.SelectStride();
  std::vector<int32_t> tok(static_cast<size_t>(T) * stride), cnt(static_cast<size_t>(T));
  R4DX_HIP_CHECK(hipMemcpy(tok.data(), core.SelectedTokens(), tok.size() * 4, hipMemcpyDeviceToHost));
  R4DX_HIP_CHECK(hipMemcpy(cnt.data(), core.SelectedCounts(), cnt.size() * 4, hipMemcpyDeviceToHost));
  int bad = 0;
  for (int t = 0; t < T; ++t) {
    const int want = fx_cnt[static_cast<size_t>(t)];
    if (want < 0) continue;  // dense row: nothing selected
    ++*sparse_rows;
    bool ok = cnt[static_cast<size_t>(t)] == want;
    for (int j = 0; ok && j < want; ++j) {
      ok = tok[static_cast<size_t>(t) * stride + j] == fx_sel[static_cast<size_t>(t) * stride + j];
    }
    if (!ok) {
      if (bad < 6) {
        std::string only_mine, only_ref;
        const int32_t* m = &tok[static_cast<size_t>(t) * stride];
        const int32_t* r = &fx_sel[static_cast<size_t>(t) * stride];
        const int nm = cnt[static_cast<size_t>(t)];
        for (int j = 0; j < nm; j += 4) {
          if (!std::binary_search(r, r + want, m[j])) only_mine += " block " + std::to_string(m[j] / 4);
        }
        for (int j = 0; j < want; j += 4) {
          if (!std::binary_search(m, m + nm, r[j])) only_ref += " block " + std::to_string(r[j] / 4);
        }
        std::printf("    selection mismatch at position %d: count %d vs %d; only mine:%s; only reference:%s\n",
                    row_base + t, nm, want, only_mine.c_str(), only_ref.c_str());
      }
      ++bad;
    }
  }
  return bad;
}

// Exact reference of the selection, from the device's own indexer queries and block keys: scores in fp32 with the
// documented operation order (dot over d ascending, relu, sum over the heads, / sqrt(128)), then the K best by
// (score descending, block index descending) via a full sort, written as ascending tokens + tail. Compared with the
// kernel's lists token for token. Returns the mismatching rows; `rows` counts the sparse rows checked.
int CpuSelectionCheck(const QsaLayerCore& core, int first_pos, int T, int* rows) {
  const int ratio = core.Dims().compress_ratio, topk = core.TopkBlocks(), stride = core.SelectStride();
  const int nb_end = (first_pos + T) / ratio;
  std::vector<uint16_t> q(static_cast<size_t>(T) * 512), bk(static_cast<size_t>(nb_end) * 128);
  R4DX_HIP_CHECK(hipMemcpy(q.data(), core.LastIndexQueries(), q.size() * 2, hipMemcpyDeviceToHost));
  if (!bk.empty()) R4DX_HIP_CHECK(hipMemcpy(bk.data(), core.BlockKeys(), bk.size() * 2, hipMemcpyDeviceToHost));
  std::vector<int32_t> tok(static_cast<size_t>(T) * stride), cnt(static_cast<size_t>(T));
  R4DX_HIP_CHECK(hipMemcpy(tok.data(), core.SelectedTokens(), tok.size() * 4, hipMemcpyDeviceToHost));
  R4DX_HIP_CHECK(hipMemcpy(cnt.data(), core.SelectedCounts(), cnt.size() * 4, hipMemcpyDeviceToHost));
  std::vector<float> bkf(bk.size()), qf(q.size());
  for (size_t i = 0; i < bk.size(); ++i) bkf[i] = Bf16ToFloat(bk[i]);
  for (size_t i = 0; i < q.size(); ++i) qf[i] = Bf16ToFloat(q[i]);
  int bad = 0;
  std::vector<float> sc;
  std::vector<int> order;
  for (int t = 0; t < T; ++t) {
    const int p = first_pos + t;
    const int nb = (p + 1) / ratio;
    if (nb <= topk) continue;
    ++*rows;
    sc.assign(static_cast<size_t>(nb), 0.0f);
    for (int b = 0; b < nb; ++b) {
      float s = 0.0f;
      for (int h = 0; h < 4; ++h) {
        float acc = 0.0f;
        const float* k = &bkf[static_cast<size_t>(b) * 128];
        const float* qh = &qf[(static_cast<size_t>(t) * 4 + h) * 128];
        for (int d = 0; d < 128; ++d) acc += k[d] * qh[d];
        s += std::max(acc, 0.0f);
      }
      sc[static_cast<size_t>(b)] = s / 11.313708498984761f;
    }
    order.resize(static_cast<size_t>(nb));
    for (int b = 0; b < nb; ++b) order[static_cast<size_t>(b)] = b;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return sc[static_cast<size_t>(a)] != sc[static_cast<size_t>(b)] ? sc[static_cast<size_t>(a)] > sc[static_cast<size_t>(b)] : a > b;
    });
    std::vector<int> keep(order.begin(), order.begin() + topk);
    std::sort(keep.begin(), keep.end());
    std::vector<int32_t> want;
    for (int b : keep) {
      for (int j = 0; j < ratio; ++j) want.push_back(b * ratio + j);
    }
    for (int tk = nb * ratio; tk <= p; ++tk) want.push_back(tk);
    bool ok = cnt[static_cast<size_t>(t)] == static_cast<int32_t>(want.size());
    for (size_t j = 0; ok && j < want.size(); ++j) ok = tok[static_cast<size_t>(t) * stride + j] == want[j];
    if (!ok) {
      if (bad < 3) std::printf("    CPU-reference selection mismatch at position %d\n", p);
      ++bad;
    }
  }
  return bad;
}

// ---- 1. attention cases ------------------------------------------------------------------------------------
struct CaseTotals {
  ErrStats hf, f32, hf32;  // mine vs HF, mine vs fp32 reference, HF vs fp32 reference
  int sel_rows = 0, sel_bad = 0, cpu_bad = 0;
};

void RunAttentionCase(const Fixture& fx, const std::string& name) {
  std::printf("[case %s]\n", name.c_str());
  const int budget = fx.Int(name + ".budget");
  const int nch = fx.Int(name + ".nchunks");
  int total = 0;
  std::vector<int> starts(static_cast<size_t>(nch)), lens(static_cast<size_t>(nch));
  for (int c = 0; c < nch; ++c) {
    starts[static_cast<size_t>(c)] = fx.Int(name + ".c" + std::to_string(c) + ".start");
    lens[static_cast<size_t>(c)] = fx.Int(name + ".c" + std::to_string(c) + ".len");
    total += lens[static_cast<size_t>(c)];
  }
  QsaDims dims;
  dims.budget = budget;
  dims.max_ctx = (total + 63) / 64 * 64 + 64;
  QsaLayerCore core(dims);
  PagedKvCache kv(dims.heads_kv, dims.head_dim, 16, dims.max_ctx);
  Arena arena(core.ArenaBytes(QsaLayerCore::kMaxChunk) + 4096);

  const auto q_norm = Upload(fx.Bf16(name + ".q_norm")), k_norm = Upload(fx.Bf16(name + ".k_norm"));
  const auto iq_norm = Upload(fx.Bf16(name + ".idx_q_norm")), ik_norm = Upload(fx.Bf16(name + ".idx_k_norm"));
  const auto kd = Upload(fx.F32(name + ".k_descale")), vd = Upload(fx.F32(name + ".v_descale"));
  QsaWeights w;
  w.q_norm = q_norm.data();
  w.k_norm = k_norm.data();
  w.idx_q_norm = iq_norm.data();
  w.idx_k_norm = ik_norm.data();
  w.k_descale = kd.data();
  w.v_descale = vd.data();

  const std::vector<float> hf_inv = fx.F32(name + ".inv_freq");
  int inv_diff = 0;
  for (int i = 0; i < 32; ++i) inv_diff += std::memcmp(&hf_inv[static_cast<size_t>(i)], &core.InvFreqHost()[i], 4) != 0;
  Check(inv_diff == 0, "rope inv_freq table bit-equal to HF's (" + std::to_string(inv_diff) + " of 32 differ)");

  hipStream_t stream = nullptr;
  CaseTotals tot;
  std::vector<uint8_t> exp_kq(static_cast<size_t>(total) * 512), exp_vq(static_cast<size_t>(total) * 512);
  std::vector<uint16_t> exp_raw(static_cast<size_t>(total) * 128);
  bool any_chunk_failed = false;

  auto run_chunk = [&](int c) {
    const std::string pre = name + ".c" + std::to_string(c);
    const int start = starts[static_cast<size_t>(c)], T = lens[static_cast<size_t>(c)];
    const auto qg = Upload(fx.Bf16(pre + ".qg")), k = Upload(fx.Bf16(pre + ".k")), v = Upload(fx.Bf16(pre + ".v"));
    const std::vector<uint16_t> idxqk_h = fx.Bf16(pre + ".idxqk");
    const auto idxqk = Upload(idxqk_h);
    const auto pos = Upload(Positions(start, T));
    QsaInputs in{qg.data(), k.data(), v.data(), idxqk.data()};
    arena.Reset();
    const uint16_t* out = core.Forward(stream, arena, in, w, kv, start, T, pos.data());
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> got(static_cast<size_t>(T) * 6144);
    R4DX_HIP_CHECK(hipMemcpy(got.data(), out, got.size() * 2, hipMemcpyDeviceToHost));
    const auto ref_hf = fx.Bf16(pre + ".out"), ref32 = fx.Bf16(pre + ".out32");
    CaseTotals chunk;
    chunk.hf.Add(got.data(), ref_hf.data(), static_cast<int64_t>(got.size()));
    chunk.f32.Add(got.data(), ref32.data(), static_cast<int64_t>(got.size()));
    chunk.hf32.Add(ref_hf.data(), ref32.data(), static_cast<int64_t>(got.size()));
    chunk.sel_bad = CompareSelection(core, fx.I32(pre + ".sel"), fx.I32(pre + ".selcount"), T, start, &chunk.sel_rows);
    int cpu_rows = 0;
    chunk.cpu_bad = CpuSelectionCheck(core, start, T, &cpu_rows);
    std::printf("    chunk %d [%d, %d): vs HF max %.4g mean %.3g | vs fp32 ref max %.4g mean %.3g (|ref| max %.3g) | sparse rows %d, bad %d\n",
                c, start, start + T, chunk.hf.max_abs, chunk.hf.Mean(), chunk.f32.max_abs, chunk.f32.Mean(),
                chunk.f32.max_ref, chunk.sel_rows, chunk.sel_bad);
    auto merge = [](ErrStats& a, const ErrStats& b) {
      a.max_abs = std::max(a.max_abs, b.max_abs);
      a.sum_abs += b.sum_abs;
      a.max_ref = std::max(a.max_ref, b.max_ref);
      a.n += b.n;
      a.bad_nonfinite += b.bad_nonfinite;
    };
    merge(tot.hf, chunk.hf);
    merge(tot.f32, chunk.f32);
    merge(tot.hf32, chunk.hf32);
    tot.sel_rows += chunk.sel_rows;
    tot.sel_bad += chunk.sel_bad;
    tot.cpu_bad += chunk.cpu_bad;
    if (chunk.hf.bad_nonfinite) any_chunk_failed = true;
    const auto kq = fx.I32(pre + ".kq"), vq = fx.I32(pre + ".vq");
    for (size_t i = 0; i < kq.size(); ++i) {
      exp_kq[static_cast<size_t>(start) * 512 + i] = static_cast<uint8_t>(kq[i]);
      exp_vq[static_cast<size_t>(start) * 512 + i] = static_cast<uint8_t>(vq[i]);
    }
    for (int t = 0; t < T; ++t) {
      std::memcpy(&exp_raw[static_cast<size_t>(start + t) * 128], &idxqk_h[static_cast<size_t>(t) * 640 + 512], 256);
    }
  };

  // final-state checks: fp8 cache bytes, raw keys, block keys
  auto check_state = [&](const char* when) {
    std::vector<uint8_t> cache(static_cast<size_t>(kv.MaxBlocks()) * kv.KvBlockStride());
    R4DX_HIP_CHECK(hipMemcpy(cache.data(), kv.Data(), cache.size(), hipMemcpyDeviceToHost));
    int64_t kv_bad = 0, kv_total = 0;
    for (int p = 0; p < total; ++p) {
      for (int h = 0; h < 2; ++h) {
        const size_t base = static_cast<size_t>(p / 16) * kv.KvBlockStride() + static_cast<size_t>(h) * kv.KvHeadStride() +
                            static_cast<size_t>(p % 16) * 512;
        for (int d = 0; d < 256; ++d) {
          kv_bad += cache[base + d] != exp_kq[static_cast<size_t>(p) * 512 + h * 256 + d];
          kv_bad += cache[base + 256 + d] != exp_vq[static_cast<size_t>(p) * 512 + h * 256 + d];
        }
      }
      kv_total += 1024;
    }
    // HF rounds q/k/v-side values in bf16 with torch's reduction order: an fp8 rounding boundary can flip for a
    // vanishing fraction of elements; anything above 0.2% is a real mismatch.
    Check(kv_bad * 500 <= kv_total, std::string(when) + ": fp8 K/V cache bytes " + std::to_string(kv_bad) + " of " +
                                       std::to_string(kv_total) + " differ from HF's quantization");
    std::vector<uint16_t> raw(static_cast<size_t>(total) * 128);
    R4DX_HIP_CHECK(hipMemcpy(raw.data(), core.RawKeys(), raw.size() * 2, hipMemcpyDeviceToHost));
    Check(raw == exp_raw, std::string(when) + ": raw index keys equal the projection's key head");
    const int nblocks = fx.Int(name + ".nblocks");
    if (nblocks > 0) {
      const auto bk_ref = fx.Bf16(name + ".blockkeys");
      std::vector<uint16_t> bk(static_cast<size_t>(nblocks) * 128);
      R4DX_HIP_CHECK(hipMemcpy(bk.data(), core.BlockKeys(), bk.size() * 2, hipMemcpyDeviceToHost));
      int64_t diff = 0, off_by_one_ulp = 0;
      for (size_t i = 0; i < bk.size(); ++i) {
        if (bk[i] != bk_ref[i]) {
          ++diff;
          off_by_one_ulp += std::abs(static_cast<int>(bk[i]) - static_cast<int>(bk_ref[i])) <= 1;
        }
      }
      std::printf("    %s: block keys %d x 128: %lld elements differ from HF (%lld by one bf16 ulp)\n", when, nblocks,
                  static_cast<long long>(diff), static_cast<long long>(off_by_one_ulp));
      Check(diff * 1000 <= static_cast<int64_t>(bk.size()) && diff == off_by_one_ulp,
            std::string(when) + ": block keys (pool + k_layernorm + rope) match HF (<= 0.1% off by one bf16 ulp)");
    }
  };

  for (int c = 0; c < nch; ++c) run_chunk(c);
  Check(core.ValidPositions() == total, "ValidPositions() == tokens after the case");
  check_state("end");

  // Tolerances. HF's graph rounds the scores and the probabilities to bf16 and sums in torch's order, the fp32
  // reference does not. ulp_max = bf16 ulp of the largest |out| (2^(floor(log2 x) - 7)).
  const double ulp_max = std::ldexp(1.0, static_cast<int>(std::floor(std::log2(tot.f32.max_ref))) - 7);
  std::printf("  totals: vs HF max %.4g mean %.3g; vs fp32 ref max %.4g mean %.3g; HF vs fp32 ref max %.4g mean %.3g; "
              "|out| max %.3g (bf16 ulp there %.3g); sparse rows %d bad %d\n",
              tot.hf.max_abs, tot.hf.Mean(), tot.f32.max_abs, tot.f32.Mean(), tot.hf32.max_abs, tot.hf32.Mean(),
              tot.f32.max_ref, ulp_max, tot.sel_rows, tot.sel_bad);
  Check(tot.hf.bad_nonfinite == 0 && !any_chunk_failed, "output finite");
  Check(tot.cpu_bad == 0, "selected token sets exactly equal the CPU reference computed from the device's own indexer queries and block keys");
  Check(tot.sel_bad == 0, "selected token sets exactly equal HF's `recent` reference (" + std::to_string(tot.sel_rows) + " sparse rows)");
  Check(tot.f32.max_abs <= 2.0 * ulp_max && tot.f32.Mean() <= 1e-4 * tot.f32.max_ref,
        "output within 2 bf16 ulps (of |out|max) of the fp32-math reference, mean error <= 1e-4 |out|max");
  Check(tot.hf.max_abs <= 1.1 * tot.hf32.max_abs + 2.0 * ulp_max && tot.hf.Mean() <= 1.1 * tot.hf32.Mean() + 1e-4 * tot.f32.max_ref,
        "output as close to HF's bf16 graph as HF's own graph is to the fp32 reference (+2 ulp)");

  // ---- rewind: poison everything beyond the rewind position, TruncateTo, replay the tail ----
  if (nch >= 2) {
    const int rc = nch / 2;
    const int pos = starts[static_cast<size_t>(rc)];
    const CaseTotals before = tot;
    const ErrStats z{};
    tot.hf = z;
    tot.f32 = z;
    tot.sel_rows = tot.sel_bad = tot.cpu_bad = 0;
    core.TruncateTo(pos);
    Check(core.ValidPositions() == pos, "TruncateTo(" + std::to_string(pos) + ") rewinds ValidPositions()");
    R4DX_HIP_CHECK(hipMemset(const_cast<uint16_t*>(core.RawKeys()) + static_cast<size_t>(pos) * 128, 0x7f,
                             static_cast<size_t>(dims.max_ctx - pos) * 128 * 2));
    const int first_dead_block = pos / 4;
    R4DX_HIP_CHECK(hipMemset(const_cast<uint16_t*>(core.BlockKeys()) + static_cast<size_t>(first_dead_block) * 128, 0x7f,
                             static_cast<size_t>(dims.max_ctx / 4 - first_dead_block) * 128 * 2));
    const int first_dead_kv_block = (pos + 15) / 16;
    R4DX_HIP_CHECK(hipMemset(kv.Data() + static_cast<size_t>(first_dead_kv_block) * kv.KvBlockStride(), 0x7f,
                             static_cast<size_t>(kv.MaxBlocks() - first_dead_kv_block) * kv.KvBlockStride()));
    for (int c = rc; c < nch; ++c) run_chunk(c);
    std::printf("  replay from position %d: vs HF max %.4g; vs fp32 ref max %.4g; sparse rows %d bad %d\n", pos,
                tot.hf.max_abs, tot.f32.max_abs, tot.sel_rows, tot.sel_bad);
    Check(tot.hf.bad_nonfinite == 0 && tot.sel_bad == 0 && tot.cpu_bad == 0 && tot.f32.max_abs <= 2.0 * ulp_max,
          "replay after TruncateTo + poisoned state reproduces the outputs and selections");
    check_state("after rewind");
    tot = before;
    bool threw = false;
    try {
      core.TruncateTo(pos);
      arena.Reset();
      const auto qg = Upload(fx.Bf16(name + ".c" + std::to_string(rc) + ".qg"));
      const auto pos_d = Upload(Positions(pos + 1, 1));
      QsaInputs in{qg.data(), qg.data(), qg.data(), qg.data()};
      core.Forward(stream, arena, in, w, kv, pos + 1, 1, pos_d.data());
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "Forward beyond ValidPositions() throws (no state hole)");
  }
}

// ---- 2. selection cases --------------------------------------------------------------------------------------
std::vector<uint16_t> RandomBf16(size_t n, uint32_t seed, float std_dev) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, std_dev);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(nd(rng));
  return v;
}

void RunSelectionCase(const Fixture& fx, const std::string& name) {
  std::printf("[case %s]\n", name.c_str());
  const int n = fx.Int(name + ".n");
  QsaDims dims;
  dims.budget = fx.Int(name + ".budget");
  dims.max_ctx = (n + 63) / 64 * 64;
  QsaLayerCore core(dims);
  PagedKvCache kv(dims.heads_kv, dims.head_dim, 16, dims.max_ctx);
  Arena arena(core.ArenaBytes(QsaLayerCore::kMaxChunk) + 4096);

  const std::vector<uint16_t> idxqk_h = fx.Bf16(name + ".idxqk");
  const std::vector<int32_t> sel_h = fx.I32(name + ".sel"), cnt_h = fx.I32(name + ".selcount");
  const int stride = core.SelectStride();
  const auto iq_norm = Upload(fx.Bf16(name + ".q_norm")), ik_norm = Upload(fx.Bf16(name + ".k_norm"));
  const auto zero_norm = Upload(std::vector<uint16_t>(256, 0));
  const auto desc = Upload(std::vector<float>(2, 0.05f));
  QsaWeights w;
  w.q_norm = zero_norm.data();
  w.k_norm = zero_norm.data();
  w.idx_q_norm = iq_norm.data();
  w.idx_k_norm = ik_norm.data();
  w.k_descale = desc.data();
  w.v_descale = desc.data();
  // attention inputs do not matter for the selection; fixed random data
  const auto qg = Upload(RandomBf16(size_t{256} * 12288, 1, 1.0f));
  const auto kk = Upload(RandomBf16(size_t{256} * 512, 2, 1.0f));
  const auto vv = Upload(RandomBf16(size_t{256} * 512, 3, 1.0f));
  const auto idxqk = Upload(idxqk_h);  // the whole sequence; chunk c reads rows [start, start + T)

  int rows = 0, bad = 0, cpu_rows = 0, cpu_bad = 0;
  int64_t q_diff = 0, q_total = 0, q_big = 0;
  const std::vector<uint16_t> hf_q = fx.Bf16(name + ".idxq");
  ErrStats fin;
  for (int start = 0; start < n; start += 256) {
    const int T = std::min(256, n - start);
    const auto pos = Upload(Positions(start, T));
    QsaInputs in{qg.data(), kk.data(), vv.data(), idxqk.data() + static_cast<size_t>(start) * 640};
    arena.Reset();
    const uint16_t* out = core.Forward(nullptr, arena, in, w, kv, start, T, pos.data());
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<int32_t> fx_sel(sel_h.begin() + static_cast<size_t>(start) * stride,
                                sel_h.begin() + static_cast<size_t>(start + T) * stride);
    std::vector<int32_t> fx_cnt(cnt_h.begin() + start, cnt_h.begin() + start + T);
    bad += CompareSelection(core, fx_sel, fx_cnt, T, start, &rows);
    cpu_bad += CpuSelectionCheck(core, start, T, &cpu_rows);
    {
      std::vector<uint16_t> q(static_cast<size_t>(T) * 512);
      R4DX_HIP_CHECK(hipMemcpy(q.data(), core.LastIndexQueries(), q.size() * 2, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < q.size(); ++i) {
        const uint16_t ref = hf_q[static_cast<size_t>(start) * 512 + i];
        if (q[i] != ref) {
          ++q_diff;
          // one bf16 ulp of a q element (|q| < 4) is 2^-6; rope sums two such terms, so allow 2 ulps = 0.04
          q_big += std::fabs(Bf16ToFloat(q[i]) - Bf16ToFloat(ref)) > 0.04f;
        }
      }
      q_total += static_cast<int64_t>(q.size());
    }
    std::vector<uint16_t> got(static_cast<size_t>(T) * 6144);
    R4DX_HIP_CHECK(hipMemcpy(got.data(), out, got.size() * 2, hipMemcpyDeviceToHost));
    for (uint16_t g : got) fin.bad_nonfinite += !std::isfinite(Bf16ToFloat(g));
  }
  std::printf("  %d tokens, %d sparse rows: %d differ from HF's list, %d differ from the CPU reference; %d rows tied at the cut (fixture)\n",
              n, rows, bad, cpu_bad, fx.Int(name + ".ties"));
  std::printf("  indexer queries: %lld of %lld elements differ from HF (%lld by more than 0.04)\n",
              static_cast<long long>(q_diff), static_cast<long long>(q_total), static_cast<long long>(q_big));
  Check(q_big == 0 && q_diff * 10000 <= q_total, "indexer queries (q_layernorm + rope) match HF (<= 0.01% elements differ, none by more than 2 bf16 ulps)");
  Check(cpu_bad == 0 && cpu_rows == rows,
        "selected token sets exactly equal the CPU reference computed from the device's own queries and block keys, ties included (" +
            std::to_string(cpu_rows) + " rows)");
  // HF's norms sum in torch's (machine-dependent) reduction order, so a bf16 element of q or of a block key can round
  // one ulp differently; a row whose K-th and (K+1)-th score are closer than that effect then selects the other block.
  // Such rows are rare (<= 0.1%); every other row must equal HF's list exactly.
  Check(bad * 1000 <= rows, "selected token sets equal HF's `recent` reference on >= 99.9% of the rows, rest explained by q / block-key ulp noise (" +
                                std::to_string(bad) + " of " + std::to_string(rows) + " rows differ)");
  Check(fin.bad_nonfinite == 0, "output finite");
  const int nblocks = fx.Int(name + ".nblocks");
  const auto bk_ref = fx.Bf16(name + ".blockkeys");
  std::vector<uint16_t> bk(static_cast<size_t>(nblocks) * 128);
  R4DX_HIP_CHECK(hipMemcpy(bk.data(), core.BlockKeys(), bk.size() * 2, hipMemcpyDeviceToHost));
  size_t diff = 0;
  for (size_t i = 0; i < bk.size(); ++i) diff += bk[i] != bk_ref[i];
  std::printf("  block keys: %zu of %zu elements differ from HF\n", diff, bk.size());
  Check(diff * 1000 <= bk.size(), "block keys match HF (<= 0.1% elements off)");
}

// ---- 3. timing + finite output at long context -------------------------------------------------------------------
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void RunPerf(int max_ctx) {
  std::printf("[perf] budget 2048, context %d, random data, 256-token prefill chunks, 16 decode steps at each checkpoint\n",
              max_ctx);
  QsaDims dims;
  dims.budget = 2048;
  dims.max_ctx = max_ctx;
  QsaLayerCore core(dims);
  PagedKvCache kv(dims.heads_kv, dims.head_dim, 16, dims.max_ctx);
  Arena arena(core.ArenaBytes(QsaLayerCore::kMaxChunk) + 4096);
  const auto nw = Upload(RandomBf16(256, 9, 0.1f));
  const auto desc = Upload(std::vector<float>(2, 0.05f));
  QsaWeights w{nw.data(), nw.data(), nw.data(), nw.data(), desc.data(), desc.data()};
  const auto qg = Upload(RandomBf16(size_t{256} * 12288, 1, 1.0f));
  const auto kk = Upload(RandomBf16(size_t{256} * 512, 2, 1.0f));
  const auto vv = Upload(RandomBf16(size_t{256} * 512, 3, 1.0f));
  const auto ix = Upload(RandomBf16(size_t{256} * 640, 4, 1.0f));
  DeviceBuffer<int32_t> pos(256);
  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));

  std::vector<int> checkpoints;
  for (int c : {2048, 32768, 131072}) {
    if (c + 16 < max_ctx) checkpoints.push_back(c);
  }
  checkpoints.push_back(max_ctx - 16);
  size_t next_cp = 0;
  int cur = 0;
  bool finite = true;
  double last_prefill_ms = 0.0;
  auto step = [&](int first, int T) {
    const std::vector<int32_t> p = Positions(first, T);
    pos.CopyFromHost(p);
    arena.Reset();
    R4DX_HIP_CHECK(hipEventRecord(e0, nullptr));
    const uint16_t* out = core.Forward(nullptr, arena, QsaInputs{qg.data(), kk.data(), vv.data(), ix.data()}, w, kv, first, T, pos.data());
    R4DX_HIP_CHECK(hipEventRecord(e1, nullptr));
    R4DX_HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    std::vector<uint16_t> got(static_cast<size_t>(T) * 6144);
    R4DX_HIP_CHECK(hipMemcpy(got.data(), out, got.size() * 2, hipMemcpyDeviceToHost));
    for (uint16_t g : got) finite = finite && std::isfinite(Bf16ToFloat(g));
    return static_cast<double>(ms);
  };
  const auto t_start = std::chrono::steady_clock::now();
  while (cur < max_ctx) {
    if (next_cp < checkpoints.size() && cur == checkpoints[next_cp]) {
      std::vector<double> ms;
      for (int i = 0; i < 16; ++i) ms.push_back(step(cur + i, 1));
      cur += 16;
      std::printf("  context %6d: decode (T=1) median %.3f ms, min %.3f ms, max %.3f ms | last 256-token prefill chunk %.2f ms\n",
                  cur - 16, Median(ms), *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end()),
                  last_prefill_ms);
      ++next_cp;
      continue;
    }
    const int limit = next_cp < checkpoints.size() ? checkpoints[next_cp] : max_ctx;
    const int T = std::min(256, limit - cur);
    last_prefill_ms = step(cur, T);
    cur += T;
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::printf("  prefilled + decoded %d tokens in %.1f s\n", max_ctx, secs);
  Check(finite, "all outputs finite up to " + std::to_string(max_ctx) + " tokens");
  R4DX_HIP_CHECK(hipEventDestroy(e0));
  R4DX_HIP_CHECK(hipEventDestroy(e1));
}

}  // namespace

int main() {
  const char* env = std::getenv("R4DX_QSA_FIXTURES");
  const std::string dir = env ? env : "E:\\models\\r4dx\\q4exp-fixtures\\QsaCore";
  Fixture fx;
  if (!fx.Load(dir)) {
    std::fprintf(stderr, "[SKIP] QSA fixtures not found in %s (run tools/reference/qwen4exp/qsa_fixtures.py)\n", dir.c_str());
    return 77;
  }
  try {
    hipDeviceProp_t prop{};
    R4DX_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device 0: %s\n", prop.name);
    for (const char* name : {"b64_p1", "b64_p5", "b64_p64", "b64_p70", "b64_p300", "b64_c3x100", "b64_mid", "b2048_p2052",
                             "b2048_p2400"}) {
      RunAttentionCase(fx, name);
    }
    for (const char* name : {"sel2051", "sel2052", "sel3000", "sel8192"}) RunSelectionCase(fx, name);
    const char* perf = std::getenv("R4DX_QSA_PERF");
    if (!perf || std::strcmp(perf, "0") != 0) {
      const char* ctx = std::getenv("R4DX_QSA_PERF_CTX");
      RunPerf(ctx ? std::atoi(ctx) : 262144);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s (%d failed checks)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures);
  return g_failures ? 1 : 0;
}
