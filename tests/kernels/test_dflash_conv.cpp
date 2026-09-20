// tests/kernels/test_dflash_conv.cpp -- r4dx_dflash_conv_bf16 (docs/dflash2.md "Kernels"), the
// thin r4dx-side wrapper that turns the DFlash2 container's own tensor layouts into
// r4d_dflash_conv_t2_g16_bf16's (x, delta, base, dpitch, NG) contract.
//
// The kernel math itself is libr4d's and already covered by libr4d's own harness; what this test
// exists to pin is the ADDRESS ARITHMETIC, which is the part a port gets wrong silently:
//   * `delta = dyn + side*taps*NG` while `dpitch` stays the FULL 2*taps*NG row pitch (r4d.h:118 --
//     delta is a SLICE of the projection, not a compacted copy). Using side*taps*NG as the pitch
//     too would read side 0's taps for both sides on row 0 and walk off the end afterwards.
//   * `base = base + side*taps*H` over a [2(side)][taps][H] channel-fastest tensor.
//   * blockmask: one block starting at row 0, so `(t & blockmask) >= tap` must degenerate to
//     `t >= tap` -- i.e. block_size must be the power of two >= T, never a smaller one.
// Both are checked twice: against a CPU reference (which re-derives the formula from
// docs/dflash2.md, independently of the offsets above) and against fixture A's REAL layer-0
// arrays for BOTH sides, where the inputs, the dyn projection and the base weight all come from
// tools/reference/dflash2_ref.py.
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

constexpr int kTaps = 2;
constexpr int kGroup = 16;

// docs/dflash2.md, verbatim:
//   out[t,c] = (base[side,0,c] + dyn[t, side*taps*NG + 0*NG + g]) * x[t,c]
//            + (base[side,1,c] + dyn[t, side*taps*NG + 1*NG + g]) * x[t-1,c] * (t >= 1)
// with g = c / group. Accumulated in fp32 from the same bf16 bits the kernel reads (libr4d's body
// accumulates in fp32 too), so a mismatch here is an indexing bug, not a rounding difference.
std::vector<float> ConvRef(const std::vector<uint16_t>& x, const std::vector<uint16_t>& dyn,
                           const std::vector<uint16_t>& base, int T, int H, int side) {
  const int NG = H / kGroup;
  const int dpitch = 2 * kTaps * NG;
  std::vector<float> out(static_cast<size_t>(T) * H, 0.0f);
  for (int t = 0; t < T; ++t) {
    for (int tap = 0; tap < kTaps; ++tap) {
      if (t < tap) continue;
      for (int c = 0; c < H; ++c) {
        const int g = c / kGroup;
        const float b =
            Bf16ToFloat(base[(static_cast<size_t>(side) * kTaps + tap) * H + c]);
        const float d = Bf16ToFloat(
            dyn[static_cast<size_t>(t) * dpitch + side * kTaps * NG + tap * NG + g]);
        const float xv = Bf16ToFloat(x[static_cast<size_t>(t - tap) * H + c]);
        out[static_cast<size_t>(t) * H + c] += (b + d) * xv;
      }
    }
  }
  return out;
}

bool RunSide(const char* label, const std::vector<uint16_t>& x, const std::vector<uint16_t>& dyn,
             const std::vector<uint16_t>& base, const std::vector<float>& ref, int T, int H,
             int side, int block_size, double norm_tol) {
  DeviceBuffer<uint16_t> x_d(x.size()), dyn_d(dyn.size()), base_d(base.size());
  DeviceBuffer<uint16_t> out_d(static_cast<size_t>(T) * H);
  x_d.CopyFromHost(x);
  dyn_d.CopyFromHost(dyn);
  base_d.CopyFromHost(base);
  out_d.Zero();
  r4dx_dflash_conv_bf16(reinterpret_cast<int64_t>(x_d.data()),
                        reinterpret_cast<int64_t>(dyn_d.data()),
                        reinterpret_cast<int64_t>(base_d.data()),
                        reinterpret_cast<int64_t>(out_d.data()), T, H, side, block_size, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const ErrStats s = CompareToRef(out_d.CopyToHost(), ref);
  const bool pass = s.norm_rel < norm_tol;
  std::printf("  %-42s side=%d  max_abs=%.3e max_rel=%.3e norm_rel=%.3e  %s\n", label, side,
              s.max_abs, s.max_rel, s.norm_rel, pass ? "PASS" : "FAIL");
  return pass;
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

  // ---- 1. CPU reference, the real draft geometry (H=5120, T=8, block_size=8) plus a shorter
  //         block (T=5 inside a block_size-8 block, the "partial last block" shape). ----
  {
    std::mt19937 rng(31);
    const int H = 5120;
    const int NG = H / kGroup;
    std::printf("[1] cpu reference, H=%d\n", H);
    for (int T : {1, 5, 8}) {
      const std::vector<uint16_t> x = RandomBf16(static_cast<size_t>(T) * H, &rng, -2.0f, 2.0f);
      const std::vector<uint16_t> dyn =
          RandomBf16(static_cast<size_t>(T) * 2 * kTaps * NG, &rng, -0.5f, 0.5f);
      const std::vector<uint16_t> base = RandomBf16(2 * kTaps * H, &rng, -1.0f, 1.0f);
      for (int side = 0; side < 2; ++side) {
        const std::vector<float> ref = ConvRef(x, dyn, base, T, H, side);
        char label[64];
        std::snprintf(label, sizeof(label), "T=%d block=8", T);
        // Identical bf16 inputs and the same two-tap accumulation, so the divergence is exactly
        // one bf16 output step: this file's reference is compiled with -ffp-contract=off while
        // libr4d's body is free to contract its second tap into an FMA, which moves the fp32
        // result by up to an fp32 ulp and can therefore tip the final RTNE to the neighbouring
        // bf16 value. bf16's relative step is 2^-9 = 1.95e-3, so 3e-3 sits just above that floor
        // (measured: 1.66e-3) -- and 100x below where a wrong tap, side, group or pitch lands.
        ok = RunSide(label, x, dyn, base, ref, T, H, side, 8, 3e-3) && ok;
      }
    }
  }

  // ---- 2. Preconditions ----
  {
    DeviceBuffer<uint16_t> d(1024);
    const int64_t p = reinterpret_cast<int64_t>(d.data());
    struct Case {
      const char* what;
      int T, H, side, block_size;
    };
    const Case cases[] = {
        {"side=2", 8, 5120, 2, 8},
        {"block_size=6 (not a power of two)", 8, 5120, 0, 6},
        {"block_size=4 (< T)", 8, 5120, 0, 4},
        {"H=5121 (not a multiple of 16)", 8, 5121, 0, 8},
    };
    int threw = 0;
    for (const Case& c : cases) {
      bool did = false;
      try {
        r4dx_dflash_conv_bf16(p, p, p, p, c.T, c.H, c.side, c.block_size, 0);
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
  }

  // ---- 3. Fixture A: layer-0 attention conv, both sides ----
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
    const int T = 8, H = 5120;
    const std::vector<uint16_t> dyn = ToBf16(LoadNpyF32(d + "/attn_dyn_l0.npy", {8, 1280}));
    const std::vector<uint16_t> base =
        ToBf16(LoadNpyF32(d + "/attn_conv_base_l0.npy", {2, 2, 5120}));
    // Side 0: the pre-attention conv. Input is the layer's post-attn_norm hidden state
    // (attn_conv_x_l0), output is the tensor that goes INTO attention (attn_conv_in_l0).
    const std::vector<uint16_t> x0 = ToBf16(LoadNpyF32(d + "/attn_conv_x_l0.npy", {8, 5120}));
    const std::vector<float> ref0 = LoadNpyF32(d + "/attn_conv_in_l0.npy", {8, 5120});
    // Side 1: the post-Wo conv. Input is Wo's output (attn_o_preconv_l0), output is what gets
    // added back into the residual stream (attn_conv_out_l0).
    const std::vector<uint16_t> x1 = ToBf16(LoadNpyF32(d + "/attn_o_preconv_l0.npy", {8, 5120}));
    const std::vector<float> ref1 = LoadNpyF32(d + "/attn_conv_out_l0.npy", {8, 5120});

    std::printf("[3] fixture A layer-0 attention conv\n");
    // Tolerance: the fixture is fp32 and x/dyn/base are ALL rounded to bf16 before the kernel sees
    // them. Each output is a sum of two products of two bf16-rounded factors, so the relative
    // error floor is ~3 * 2^-9 ~ 6e-3; 1.5e-2 leaves headroom for the cancellation between the
    // two taps without being loose enough to hide a wrong tap, side or group index (each of which
    // moves norm_rel to O(1)).
    ok = RunSide("attn_conv_x_l0 -> attn_conv_in_l0", x0, dyn, base, ref0, T, H, 0, 8, 1.5e-2) &&
         ok;
    ok = RunSide("attn_o_preconv_l0 -> attn_conv_out_l0", x1, dyn, base, ref1, T, H, 1, 8,
                 1.5e-2) &&
         ok;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
