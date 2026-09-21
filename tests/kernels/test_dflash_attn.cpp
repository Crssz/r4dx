// tests/kernels/test_dflash_attn.cpp -- r4dx_dflash_attn_bf16 (docs/dflash2.md "Kernels"), the
// DFlash2 draft block's own attention: NON-causal within the block, sliding-window over the
// injected-feature ring, GQA.
//
// Three independent checks:
//  (1) A CPU fp64 reference over an (n_injected x T) sweep chosen to hit every edge of the ring
//      and of the window: n = 0 (nothing injected at all, so only the block's own 8 keys are
//      visible), n < window (no clipping), n exactly at the ring size (the first wrap), n just
//      past it, and n far past it (every visible key read through a modulo). T = 1/4/8.
//      Both sides consume the SAME bf16 inputs, so this isolates the kernel's indexing, masking
//      and softmax from bf16 input quantisation and the tolerance can be tight.
//  (1b) The same reference over `store_begin > 0` -- the COLD-RING GAP (docs/dflash2.md section 5):
//      the visible store is the contiguous run [store_begin, n_injected), not [0, n_injected).
//      Every ring slot the kernel would have read at store_begin=0 but must NOT read at this
//      store_begin is deliberately filled with 1e4 junk, so reading even one of them makes that
//      key's score dominate the softmax and moves the output by orders of magnitude rather than by
//      a tolerance. store_begin == n_injected (the whole store invalid) is included.
//  (2) Fixture A's own layer-0 attention: (attn_q_l0, attn_k_l0, attn_v_l0, injected_k_l0,
//      injected_v_l0, n_injected=40) -> attn_out_l0, all produced by tools/reference/dflash2_ref.py
//      with no C++ in the loop. This is what pins the CONVENTION (that the block's keys are
//      appended AFTER the store, that the block is non-causal, that q-head h reads kv-head h/4,
//      that the scale is 1/sqrt(head_dim)); check (1) cannot, since its reference is a second
//      implementation of the same convention written in this file.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "npy_fixture.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;
using namespace r4dx_test;

namespace {

constexpr int kHeadDim = 128;
constexpr int kHeadsQ = 32;
constexpr int kHeadsKv = 8;
constexpr int kWindow = 2048;
constexpr int kSlots = 2048;

// The visible key list for query row t, in the EXACT order the kernel builds it: the injected
// store's window-clipped range first, then all T of the block's own keys. `src` is the store slot
// for a store key and -(1+j) for block key j, so the caller needs no second lookup.
// `store_begin` is the first VALID injected position (0 == all of them), i.e. the low end of the
// range is clamped to it and not to 0 -- the cold-ring gap the kernel's own contract describes.
std::vector<int> VisibleKeys(int t, int n_injected, int T, int window, int slots, int store_begin) {
  const int q_pos = n_injected + t;
  int lo = q_pos - window + 1;
  if (lo < store_begin) lo = store_begin;
  std::vector<int> keys;
  for (int p = lo; p < n_injected; ++p) keys.push_back(p % slots);
  for (int j = 0; j < T; ++j) keys.push_back(-(1 + j));
  return keys;
}

// fp64 reference. Inputs are the same bf16 bits the kernel reads.
std::vector<float> AttnRef(const std::vector<uint16_t>& q, const std::vector<uint16_t>& k_block,
                           const std::vector<uint16_t>& v_block,
                           const std::vector<uint16_t>& k_store,
                           const std::vector<uint16_t>& v_store, int T, int n_injected,
                           int store_begin, int window, int slots, double scale) {
  const int ratio = kHeadsQ / kHeadsKv;
  std::vector<float> out(static_cast<size_t>(T) * kHeadsQ * kHeadDim);
  std::vector<double> scores;
  for (int t = 0; t < T; ++t) {
    const std::vector<int> keys = VisibleKeys(t, n_injected, T, window, slots, store_begin);
    for (int hq = 0; hq < kHeadsQ; ++hq) {
      const int kvh = hq / ratio;
      const uint16_t* qp = q.data() + (static_cast<size_t>(t) * kHeadsQ + hq) * kHeadDim;
      scores.assign(keys.size(), 0.0);
      for (size_t j = 0; j < keys.size(); ++j) {
        const uint16_t* kp =
            keys[j] >= 0
                ? k_store.data() + (static_cast<size_t>(keys[j]) * kHeadsKv + kvh) * kHeadDim
                : k_block.data() +
                      (static_cast<size_t>(-keys[j] - 1) * kHeadsKv + kvh) * kHeadDim;
        double dot = 0.0;
        for (int d = 0; d < kHeadDim; ++d) {
          dot += static_cast<double>(Bf16ToFloat(qp[d])) * static_cast<double>(Bf16ToFloat(kp[d]));
        }
        scores[j] = dot * scale;
      }
      double m = -INFINITY;
      for (double s : scores) m = std::max(m, s);
      double l = 0.0;
      for (double& s : scores) {
        s = std::exp(s - m);
        l += s;
      }
      double acc[kHeadDim] = {0.0};
      for (size_t j = 0; j < keys.size(); ++j) {
        const uint16_t* vp =
            keys[j] >= 0
                ? v_store.data() + (static_cast<size_t>(keys[j]) * kHeadsKv + kvh) * kHeadDim
                : v_block.data() +
                      (static_cast<size_t>(-keys[j] - 1) * kHeadsKv + kvh) * kHeadDim;
        for (int d = 0; d < kHeadDim; ++d) acc[d] += scores[j] * Bf16ToFloat(vp[d]);
      }
      float* op = out.data() + (static_cast<size_t>(t) * kHeadsQ + hq) * kHeadDim;
      for (int d = 0; d < kHeadDim; ++d) op[d] = static_cast<float>(acc[d] / l);
    }
  }
  return out;
}

std::vector<uint16_t> RandomBf16(size_t n, std::mt19937* rng, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> out(n);
  for (auto& v : out) v = FloatToBf16(d(*rng));
  return out;
}

}  // namespace

int main() {
  // Unbuffered: a GPU test that dies (a kernel fault, or an abort out of a failed HIP check)
  // takes the CRT's stdout buffer with it, and a crash report with zero output says nothing about
  // which check was running. Costs nothing at this output volume.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  bool ok = true;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  // ---- 1. CPU fp64 reference over the ring/window sweep ----
  {
    std::mt19937 rng(23);
    // One store, reused by every case: the kernel's own reads are what decide which slots matter,
    // so filling all of them with distinct random values means a wrong slot index reads garbage
    // rather than accidentally-right data.
    const std::vector<uint16_t> k_store =
        RandomBf16(static_cast<size_t>(kSlots) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
    const std::vector<uint16_t> v_store =
        RandomBf16(static_cast<size_t>(kSlots) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
    DeviceBuffer<uint16_t> ks_d(k_store.size()), vs_d(v_store.size());
    ks_d.CopyFromHost(k_store);
    vs_d.CopyFromHost(v_store);

    std::printf("[1] cpu fp64 reference, n_injected x T sweep (window=%d, slots=%d)\n", kWindow,
                kSlots);
    for (int n : {0, 3, 40, 2047, 2048, 2049, 2100, 5000}) {
      for (int T : {1, 4, 8}) {
        const std::vector<uint16_t> q =
            RandomBf16(static_cast<size_t>(T) * kHeadsQ * kHeadDim, &rng, -1.5f, 1.5f);
        const std::vector<uint16_t> kb =
            RandomBf16(static_cast<size_t>(T) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
        const std::vector<uint16_t> vb =
            RandomBf16(static_cast<size_t>(T) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);

        const std::vector<float> ref =
            AttnRef(q, kb, vb, k_store, v_store, T, n, /*store_begin=*/0, kWindow, kSlots, scale);

        DeviceBuffer<uint16_t> q_d(q.size()), kb_d(kb.size()), vb_d(vb.size());
        DeviceBuffer<uint16_t> out_d(static_cast<size_t>(T) * kHeadsQ * kHeadDim);
        q_d.CopyFromHost(q);
        kb_d.CopyFromHost(kb);
        vb_d.CopyFromHost(vb);
        r4dx_dflash_attn_bf16(reinterpret_cast<int64_t>(q_d.data()),
                              reinterpret_cast<int64_t>(kb_d.data()),
                              reinterpret_cast<int64_t>(vb_d.data()),
                              reinterpret_cast<int64_t>(ks_d.data()),
                              reinterpret_cast<int64_t>(vs_d.data()),
                              reinterpret_cast<int64_t>(out_d.data()), T, kHeadsQ, kHeadsKv,
                              kHeadDim, n, /*store_begin=*/0, kWindow, kSlots, scale, 0);
        R4DX_HIP_CHECK(hipDeviceSynchronize());

        const ErrStats s = CompareToRef(out_d.CopyToHost(), ref);
        // Both sides read identical bf16 inputs; the ONLY divergences are (a) the kernel's fp32
        // dot/softmax/weighted-sum vs the reference's fp64, and (b) the final RTNE to bf16, which
        // alone costs up to 2^-9 relative on an output whose magnitude here is O(0.1..1).
        const bool pass = s.norm_rel < 4e-3 && s.max_abs < 2e-2;
        std::printf("  n=%-5d T=%d  max_abs=%.3e max_rel=%.3e norm_rel=%.3e  %s\n", n, T,
                    s.max_abs, s.max_rel, s.norm_rel, pass ? "PASS" : "FAIL");
        ok = ok && pass;
      }
    }
  }

  // ---- 1b. Cold-ring gap: store_begin > 0, with the invisible slots filled with junk ----
  // Junk = 1e4, not NaN: a NaN would poison the softmax and be caught by literally any comparison,
  // including one that reads the slot and then correctly masks it. A large FINITE value is the
  // sharper probe -- it only shows up if its key actually enters the score row, where it takes over
  // the softmax completely (scores are O(1) here, so exp(1e4*q.k*scale) saturates). If the kernel
  // clamps `lo` correctly, the output is bit-for-bit what the same store WITHOUT the junk would
  // give, and the fp64 reference (which applies the same clamp) matches to check [1]'s own floor.
  {
    std::mt19937 rng(101);
    const uint16_t kJunk = FloatToBf16(1e4f);
    std::printf("[1b] cold-ring gap sweep (store_begin > 0; slots below it filled with 1e4 junk)\n");
    for (int n : {40, 2100, 5000}) {
      for (int sb : {n - 1, n - 17, n}) {
        for (int T : {1, 8}) {
          std::vector<uint16_t> k_store =
              RandomBf16(static_cast<size_t>(kSlots) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
          std::vector<uint16_t> v_store =
              RandomBf16(static_cast<size_t>(kSlots) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
          // Every position the kernel WOULD read at store_begin=0 but must not read at this
          // store_begin: p in [max(0, n - window + 1), sb). That range is under `window` wide, so
          // no two of its positions share a slot and junking them cannot clobber a valid one.
          const int junk_lo = std::max(0, n - kWindow + 1);
          int junked = 0;
          for (int p = junk_lo; p < sb; ++p) {
            const size_t base = static_cast<size_t>(p % kSlots) * kHeadsKv * kHeadDim;
            for (size_t i = 0; i < static_cast<size_t>(kHeadsKv) * kHeadDim; ++i) {
              k_store[base + i] = kJunk;
              v_store[base + i] = kJunk;
            }
            ++junked;
          }
          DeviceBuffer<uint16_t> ks_d(k_store.size()), vs_d(v_store.size());
          ks_d.CopyFromHost(k_store);
          vs_d.CopyFromHost(v_store);

          const std::vector<uint16_t> q =
              RandomBf16(static_cast<size_t>(T) * kHeadsQ * kHeadDim, &rng, -1.5f, 1.5f);
          const std::vector<uint16_t> kb =
              RandomBf16(static_cast<size_t>(T) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
          const std::vector<uint16_t> vb =
              RandomBf16(static_cast<size_t>(T) * kHeadsKv * kHeadDim, &rng, -1.5f, 1.5f);
          const std::vector<float> ref =
              AttnRef(q, kb, vb, k_store, v_store, T, n, sb, kWindow, kSlots, scale);

          DeviceBuffer<uint16_t> q_d(q.size()), kb_d(kb.size()), vb_d(vb.size());
          DeviceBuffer<uint16_t> out_d(static_cast<size_t>(T) * kHeadsQ * kHeadDim);
          q_d.CopyFromHost(q);
          kb_d.CopyFromHost(kb);
          vb_d.CopyFromHost(vb);
          r4dx_dflash_attn_bf16(reinterpret_cast<int64_t>(q_d.data()),
                                reinterpret_cast<int64_t>(kb_d.data()),
                                reinterpret_cast<int64_t>(vb_d.data()),
                                reinterpret_cast<int64_t>(ks_d.data()),
                                reinterpret_cast<int64_t>(vs_d.data()),
                                reinterpret_cast<int64_t>(out_d.data()), T, kHeadsQ, kHeadsKv,
                                kHeadDim, n, sb, kWindow, kSlots, scale, 0);
          R4DX_HIP_CHECK(hipDeviceSynchronize());

          const ErrStats s = CompareToRef(out_d.CopyToHost(), ref);
          const bool pass = s.norm_rel < 4e-3 && s.max_abs < 2e-2;
          std::printf("  n=%-5d store_begin=%-5d T=%d junked=%-5d visible_store=%-5d  "
                      "max_abs=%.3e norm_rel=%.3e  %s\n",
                      n, sb, T, junked, n - sb, s.max_abs, s.norm_rel, pass ? "PASS" : "FAIL");
          ok = ok && pass;
        }
      }
    }
  }

  // ---- 2. Preconditions must throw, not silently read out of the kernel's fixed LDS geometry ----
  {
    DeviceBuffer<uint16_t> dummy(static_cast<size_t>(8) * kHeadsQ * kHeadDim);
    DeviceBuffer<uint16_t> store(static_cast<size_t>(kSlots) * kHeadsKv * kHeadDim);
    const int64_t p = reinterpret_cast<int64_t>(dummy.data());
    const int64_t sp = reinterpret_cast<int64_t>(store.data());
    struct Case {
      const char* what;
      int T, hq, hkv, hd, n_injected, store_begin, window, slots;
    };
    const Case cases[] = {
        {"T=9 (> block size 8)", 9, 32, 8, 128, 0, 0, 2048, 2048},
        {"head_dim=160 (> 128)", 8, 32, 8, 160, 0, 0, 2048, 2048},
        {"head_dim=100 (not a multiple of 32)", 8, 32, 8, 100, 0, 0, 2048, 2048},
        {"heads_q/heads_kv = 8 (> 4)", 8, 64, 8, 128, 0, 0, 2048, 2048},
        {"window=2049 (> 2048)", 8, 32, 8, 128, 0, 0, 2049, 4096},
        {"slots < window", 8, 32, 8, 128, 0, 0, 2048, 1024},
        {"store_begin=-1 (< 0)", 8, 32, 8, 128, 40, -1, 2048, 2048},
        {"store_begin=41 (> n_injected)", 8, 32, 8, 128, 40, 41, 2048, 2048},
    };
    int threw = 0;
    for (const Case& c : cases) {
      bool did = false;
      try {
        r4dx_dflash_attn_bf16(p, p, p, sp, sp, p, c.T, c.hq, c.hkv, c.hd, c.n_injected,
                              c.store_begin, c.window, c.slots, scale, 0);
      } catch (const std::exception&) {
        did = true;
      }
      if (did) ++threw;
      std::printf("  %-38s threw=%s\n", c.what, did ? "yes" : "NO");
    }
    const int n_cases = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    std::printf("[2] preconditions: %d/%d threw  %s\n", threw, n_cases,
                threw == n_cases ? "PASS" : "FAIL");
    ok = ok && threw == n_cases;
    R4DX_HIP_CHECK(hipDeviceSynchronize());
  }

  // ---- 3. Fixture A: layer-0 attention, the whole convention end to end ----
  if (!FixtureAvailable()) {
    std::fprintf(stderr,
                 "[SKIP] fixture A not found at %s -- regenerate with\n"
                 "       <reference venv>/python.exe tools/reference/dflash2_ref.py "
                 "--gen-fixtures all --seed 0\n",
                 FixtureDir().c_str());
    return ok ? kSkipReturnCode : 1;
  }
  {
    const std::string d = FixtureDir();
    const int64_t n_injected = NpyShape(d + "/injected_k_l0.npy")[0];
    const std::vector<float> q_f = LoadNpyF32(d + "/attn_q_l0.npy", {8, 32, 128});
    const std::vector<float> kb_f = LoadNpyF32(d + "/attn_k_l0.npy", {8, 8, 128});
    const std::vector<float> vb_f = LoadNpyF32(d + "/attn_v_l0.npy", {8, 8, 128});
    const std::vector<float> ks_f = LoadNpyF32(d + "/injected_k_l0.npy", {n_injected, 8, 128});
    const std::vector<float> vs_f = LoadNpyF32(d + "/injected_v_l0.npy", {n_injected, 8, 128});
    // attn_out_l0 is saved as [T, heads_q*head_dim] (attention_gqa's own return shape); the flat
    // element order is identical to the kernel's [T, heads_q, head_dim].
    const std::vector<float> ref = LoadNpyF32(d + "/attn_out_l0.npy", {8, 4096});

    const int T = 8;
    const std::vector<uint16_t> q = ToBf16(q_f);
    const std::vector<uint16_t> kb = ToBf16(kb_f);
    const std::vector<uint16_t> vb = ToBf16(vb_f);
    // The store is a ring of kSlots positions; the fixture's n_injected positions occupy
    // slot = position % kSlots, which for n_injected=40 < kSlots is slot == position. The rest of
    // the ring is left zeroed and must never be read (n_injected < window, so nothing is clipped).
    const std::vector<uint16_t> ks = ToBf16(ks_f);
    const std::vector<uint16_t> vs = ToBf16(vs_f);

    DeviceBuffer<uint16_t> q_d(q.size()), kb_d(kb.size()), vb_d(vb.size());
    DeviceBuffer<uint16_t> ks_d(static_cast<size_t>(kSlots) * 8 * 128);
    DeviceBuffer<uint16_t> vs_d(static_cast<size_t>(kSlots) * 8 * 128);
    DeviceBuffer<uint16_t> out_d(static_cast<size_t>(T) * 32 * 128);
    q_d.CopyFromHost(q);
    kb_d.CopyFromHost(kb);
    vb_d.CopyFromHost(vb);
    ks_d.Zero();
    vs_d.Zero();
    ks_d.CopyFromHost(ks.data(), ks.size());
    vs_d.CopyFromHost(vs.data(), vs.size());

    r4dx_dflash_attn_bf16(reinterpret_cast<int64_t>(q_d.data()),
                          reinterpret_cast<int64_t>(kb_d.data()),
                          reinterpret_cast<int64_t>(vb_d.data()),
                          reinterpret_cast<int64_t>(ks_d.data()),
                          reinterpret_cast<int64_t>(vs_d.data()),
                          reinterpret_cast<int64_t>(out_d.data()), T, 32, 8, 128,
                          static_cast<int>(n_injected), /*store_begin=*/0, kWindow, kSlots, scale,
                          0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());

    const ErrStats s = CompareToRef(out_d.CopyToHost(), ref);
    std::printf("[3] fixture A layer-0 attention (n_injected=%lld, T=8): max_abs=%.3e "
                "max_rel=%.3e norm_rel=%.3e\n",
                (long long)n_injected, s.max_abs, s.max_rel, s.norm_rel);
    // Tolerance here is set by the INPUTS, not by the kernel: the fixture is fp32 and every
    // q/k/v element is rounded to bf16 (~2^-9 relative) before the kernel sees it, which perturbs
    // the pre-softmax scores and therefore the attention weights themselves -- so this cannot go
    // below check [1]'s bf16-output floor of ~1.7e-3, and measures 3.4e-3. 1e-2 is a 3x band
    // around that. max_abs/max_rel are printed but NOT gated, for the reason test_rope_neox.cpp's
    // METRICS note gives: some outputs cancel to near zero, where any disagreement reads as a
    // large relative error. A wrong mask, head mapping, key order or scale moves norm_rel to
    // O(0.1..1), i.e. two orders of magnitude over this gate.
    const bool pass = s.norm_rel < 1e-2;
    std::printf("[3] %s\n", pass ? "PASS" : "FAIL");
    ok = ok && pass;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
