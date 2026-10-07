// tests/kernels/test_argmax_multi.cpp -- decode-t1 item 1: the multi-workgroup argmax (r4dx_argmax_f32 /
// _val_f32 / _bf16 / _val_bf16 / _rows_f32) against the one-workgroup kernel it replaced
// (r4dx_argmax_val_f32_single), BIT FOR BIT: the winning index and the winning value's bits, over rows built
// to hurt -- ties across every block / thread / vocab-tail boundary, NaN, -inf, +inf, signed zeros, rows with
// nothing greater than -inf (index 0), a one-element vocab, odd sizes -- and against argmax_ref.hpp's host
// emulation (the order algebra itself is tests/kernels/test_argmax_order_cpu.cpp).
// Each case runs the new kernel several times (a run-to-run difference would be a race on the scratch).
// Needs a HIP device; run with R4DX_DECODE_LEGACY unset (with `argmax` the entry points ARE the old kernel,
// and the test says so and skips, 77).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "r4dx/core/decode_legacy.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/argmax_ref.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;
using r4dx::kernels::ArgmaxDirectRef;
using r4dx::kernels::ArgmaxPairRef;

namespace {

int g_fail = 0;
int g_cases = 0;

uint32_t Bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}
float FromBits(uint32_t b) {
  float v;
  std::memcpy(&v, &b, 4);
  return v;
}
float Nan() { return std::numeric_limits<float>::quiet_NaN(); }
float Inf() { return std::numeric_limits<float>::infinity(); }

// A bf16 pattern's exact fp32 widening.
float Widen(uint16_t b) { return FromBits(static_cast<uint32_t>(b) << 16); }
uint16_t TruncBf16(float f) { return static_cast<uint16_t>(Bits(f) >> 16); }

struct Out {
  int32_t idx;
  float val;
};

// The one-workgroup kernel's answer for a fp32 row.
Out RunSingle(const DeviceBuffer<float>& row, int64_t vocab) {
  DeviceBuffer<int32_t> oi(1);
  DeviceBuffer<float> ov(1);
  r4dx_argmax_val_f32_single(reinterpret_cast<int64_t>(row.data()), reinterpret_cast<int64_t>(oi.data()),
                              reinterpret_cast<int64_t>(ov.data()), vocab, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Out o{};
  oi.CopyToHost(&o.idx, 1);
  ov.CopyToHost(&o.val, 1);
  return o;
}

void Fail(const std::string& what, const Out& want, const Out& got) {
  ++g_fail;
  std::fprintf(stderr, "FAIL %s: want (%d, bits %08x) got (%d, bits %08x)\n", what.c_str(), want.idx,
               Bits(want.val), got.idx, Bits(got.val));
}

// One fp32 row (`x`) through every entry point; `bf16_exact` rows are also run through the bf16 ones.
void CheckRow(const std::vector<float>& x, const char* what, bool bf16_exact) {
  const int64_t n = static_cast<int64_t>(x.size());
  DeviceBuffer<float> d(x.size());
  d.CopyFromHost(x.data(), x.size());
  const Out want = RunSingle(d, n);
  const ArgmaxPairRef host = ArgmaxDirectRef(x.data(), n);
  ++g_cases;
  if (host.i != want.idx || Bits(host.v) != Bits(want.val)) {
    Fail(std::string(what) + " [single vs host reference]", {host.i, host.v}, want);
  }
  for (int rep = 0; rep < 4; ++rep) {
    DeviceBuffer<int32_t> oi(1), oi2(1);
    DeviceBuffer<float> ov(1);
    r4dx_argmax_val_f32(reinterpret_cast<int64_t>(d.data()), reinterpret_cast<int64_t>(oi.data()),
                         reinterpret_cast<int64_t>(ov.data()), n, 0);
    r4dx_argmax_f32(reinterpret_cast<int64_t>(d.data()), reinterpret_cast<int64_t>(oi2.data()), n, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Out got{};
    oi.CopyToHost(&got.idx, 1);
    ov.CopyToHost(&got.val, 1);
    int32_t idx2 = -1;
    oi2.CopyToHost(&idx2, 1);
    ++g_cases;
    if (got.idx != want.idx || Bits(got.val) != Bits(want.val)) Fail(std::string(what) + " [f32 val]", want, got);
    if (idx2 != want.idx) Fail(std::string(what) + " [f32 idx-only]", want, {idx2, 0.0f});
  }
  if (bf16_exact) {
    std::vector<uint16_t> b(x.size());
    for (size_t i = 0; i < x.size(); ++i) b[i] = TruncBf16(x[i]);  // exact: x is a widened bf16
    DeviceBuffer<uint16_t> db(b.size());
    db.CopyFromHost(b.data(), b.size());
    DeviceBuffer<int32_t> oi(1), oi2(1);
    DeviceBuffer<float> ov(1);
    r4dx_argmax_val_bf16(reinterpret_cast<int64_t>(db.data()), reinterpret_cast<int64_t>(oi.data()),
                          reinterpret_cast<int64_t>(ov.data()), n, 0);
    r4dx_argmax_bf16(reinterpret_cast<int64_t>(db.data()), reinterpret_cast<int64_t>(oi2.data()), n, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Out got{};
    oi.CopyToHost(&got.idx, 1);
    ov.CopyToHost(&got.val, 1);
    int32_t idx2 = -1;
    oi2.CopyToHost(&idx2, 1);
    ++g_cases;
    if (got.idx != want.idx || Bits(got.val) != Bits(want.val)) Fail(std::string(what) + " [bf16 val]", want, got);
    if (idx2 != want.idx) Fail(std::string(what) + " [bf16 idx-only]", want, {idx2, 0.0f});
  }
}

std::vector<float> RandomBf16Row(std::mt19937& rng, int64_t n, float scale) {
  std::uniform_real_distribution<float> uni(-scale, scale);
  std::vector<float> x(static_cast<size_t>(n));
  for (float& v : x) v = Widen(TruncBf16(uni(rng)));
  return x;
}

}  // namespace

int main() {
  if (DecodeLegacy(DecodeItem::kArgmax)) {
    std::printf("test_argmax_multi: R4DX_DECODE_LEGACY has `argmax` set -- the entry points are the old kernel; skipping\n");
    return 77;
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(8102026);

  // ---- random rows: decode's vocab, the MTP reduced-vocab head's size class, tiny and ragged ones ----
  for (int64_t n : {1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4096, 12345, 32768, 65537, 248320, 248321}) {
    CheckRow(RandomBf16Row(rng, n, 4.0f), "random", true);
  }

  // ---- ties: plateau of the maximum at positions straddling every boundary ----
  for (int64_t n : {257, 4096, 248320}) {
    for (int rep = 0; rep < 6; ++rep) {
      std::vector<float> x = RandomBf16Row(rng, n, 1.0f);
      for (float& v : x) v = std::round(v * 4) / 4;  // few distinct values
      const float top = 7.0f;
      const int copies = 2 + static_cast<int>(rng() % 5);
      for (int c = 0; c < copies; ++c) x[rng() % static_cast<uint64_t>(n)] = top;
      CheckRow(x, "tied maxima", true);
    }
    // the maximum exactly at, and one before / after, the thread / block strides
    for (int64_t pos : std::vector<int64_t>{0, 1, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, n / 2, n - 2, n - 1}) {
      if (pos >= n) continue;
      std::vector<float> x(static_cast<size_t>(n), 0.5f);
      x[static_cast<size_t>(pos)] = 3.0f;
      CheckRow(x, "single max at boundary", true);
      x[static_cast<size_t>(n - 1)] = 3.0f;  // and a tie at the very end: the lower index must still win
      CheckRow(x, "max plus tie at end", true);
    }
    std::vector<float> flat(static_cast<size_t>(n), 1.5f);
    CheckRow(flat, "all equal", true);
  }

  // ---- NaN, infinities, signed zeros, rows with nothing greater than -inf ----
  {
    const int64_t n = 248320;
    CheckRow(std::vector<float>(static_cast<size_t>(n), Nan()), "all NaN", true);
    CheckRow(std::vector<float>(static_cast<size_t>(n), -Inf()), "all -inf", true);
    std::vector<float> m(static_cast<size_t>(n), Nan());
    m[7] = -Inf();
    m[200000] = -Inf();
    CheckRow(m, "NaN with -inf only (answer 0, not the first -inf)", true);
    std::vector<float> one(static_cast<size_t>(n), Nan());
    one[247999] = -1e30f;
    CheckRow(one, "NaN with one finite", true);
    std::vector<float> nan0(static_cast<size_t>(n), -2.0f);
    nan0[0] = Nan();
    nan0[300] = -0.5f;
    nan0[170000] = -0.5f;
    CheckRow(nan0, "NaN at 0, tie later", true);
    std::vector<float> pinf(static_cast<size_t>(n), 1.0f);
    pinf[64000] = Inf();
    pinf[64001] = Inf();
    pinf[10] = Nan();
    CheckRow(pinf, "+inf plateau", true);
    std::vector<float> zeros(static_cast<size_t>(n), -0.0f);
    zeros[400] = 0.0f;
    CheckRow(zeros, "signed zeros", true);
    std::vector<float> zeros2(static_cast<size_t>(n), 0.0f);
    zeros2[5] = -0.0f;
    zeros2[600] = 0.0f;
    CheckRow(zeros2, "signed zeros 2", true);
  }
  for (int rep = 0; rep < 30; ++rep) {
    const int64_t n = 1 + static_cast<int64_t>(rng() % 60000);
    std::vector<float> x(static_cast<size_t>(n));
    for (float& v : x) {
      switch (rng() % 9) {
        case 0: v = Nan(); break;
        case 1: v = -Inf(); break;
        case 2: v = Inf(); break;
        case 3: v = 0.0f; break;
        case 4: v = -0.0f; break;
        case 5: v = 2.0f; break;
        default: v = Widen(TruncBf16(std::round(static_cast<float>(static_cast<int>(rng() % 17) - 8) * 0.5f))); break;
      }
    }
    CheckRow(x, "garbage mix", true);
  }

  // ---- fp32 rows that are not bf16-representable (the fp32 entry points only) ----
  {
    std::uniform_real_distribution<float> uni(-10.0f, 10.0f);
    std::vector<float> x(248320);
    for (float& v : x) v = uni(rng);
    CheckRow(x, "fp32 random", false);
    for (float& v : x) v = std::round(v);  // heavy ties in fp32
    CheckRow(x, "fp32 ties", false);
  }

  // ---- the rows entry point: strides, the TP pair layout, more rows than one scratch group ----
  for (int64_t rows : {1, 4, 8, 16, 17, 20}) {
    const int64_t vocab = 5000, stride = vocab + 17;
    std::vector<float> all(static_cast<size_t>(rows * stride), -1.0f);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t i = 0; i < vocab; ++i) all[static_cast<size_t>(r * stride + i)] = Widen(TruncBf16(std::round(static_cast<float>(rng() % 9) - 4.0f)));
      all[static_cast<size_t>(r * stride + static_cast<int64_t>(rng() % vocab))] = 5.0f;
      all[static_cast<size_t>(r * stride + vocab - 1)] = 5.0f;  // a tie at the end
      if (r == 3) std::fill(all.begin() + r * stride, all.begin() + r * stride + vocab, Nan());
    }
    DeviceBuffer<float> d(all.size());
    d.CopyFromHost(all.data(), all.size());
    DeviceBuffer<int32_t> oi(static_cast<size_t>(rows)), pairs(static_cast<size_t>(2 * rows));
    r4dx_argmax_rows_f32(reinterpret_cast<int64_t>(d.data()), stride, reinterpret_cast<int64_t>(oi.data()), 0,
                          1, rows, vocab, 0);
    r4dx_argmax_rows_f32(reinterpret_cast<int64_t>(d.data()), stride, reinterpret_cast<int64_t>(pairs.data()),
                          reinterpret_cast<int64_t>(pairs.data() + 1), 2, rows, vocab, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<int32_t> idx = oi.CopyToHost(), pr = pairs.CopyToHost();
    for (int64_t r = 0; r < rows; ++r) {
      DeviceBuffer<float> row(static_cast<size_t>(vocab));
      row.CopyFromHost(all.data() + r * stride, static_cast<size_t>(vocab));
      const Out want = RunSingle(row, vocab);
      ++g_cases;
      if (idx[static_cast<size_t>(r)] != want.idx) Fail("rows idx r=" + std::to_string(r), want, {idx[static_cast<size_t>(r)], 0.0f});
      if (pr[static_cast<size_t>(2 * r)] != want.idx ||
          Bits(FromBits(static_cast<uint32_t>(pr[static_cast<size_t>(2 * r + 1)]))) != Bits(want.val)) {
        Fail("rows pair r=" + std::to_string(r), want,
             {pr[static_cast<size_t>(2 * r)], FromBits(static_cast<uint32_t>(pr[static_cast<size_t>(2 * r + 1)]))});
      }
    }
  }

  if (g_fail != 0) {
    std::fprintf(stderr, "test_argmax_multi: %d of %d comparisons FAILED\n", g_fail, g_cases);
    return 1;
  }
  std::printf("test_argmax_multi: OK (%d comparisons, multi-workgroup == one-workgroup bit for bit)\n", g_cases);
  return 0;
}
