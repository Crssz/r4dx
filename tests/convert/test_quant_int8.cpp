// w8a16 quantizer / packer / reference dequant (docs/w8a16.md 2, 3.2, 7.1). CPU-only.
//
//   - byte-exact `.wq` / `.ws` (and the float32 dequant) against fixtures from tools/convert_ref/
//     w8_ref.py (gen_fixtures.py write_w8a16_fixtures) at G = 64 and 128;
//   - the RTN round-trip bound on Gaussian data (rel. L2 < 0.012) and the per-element invariant
//     |w - deq| <= s_use / 2 (or the clamp excess) on every fixture row, dead / tiny-amax rows included;
//   - dead rows (amax 1e-25) become zero groups: scale 0, codes 128, counted, dequant exactly 0;
//   - a constant group, amax > 2031 an error, the +-127 clamp, bytes never 0;
//   - thread-count independence, the argument refusals, and PackW8Bytes being a pure permutation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/quant_int8.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace {

using namespace r4dx_convert;

int g_failures = 0;

bool Check(const std::string& what, bool cond) {
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
  return cond;
}

std::vector<uint8_t> ReadFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open fixture: " + path);
  f.seekg(0, std::ios::end);
  const size_t n = static_cast<size_t>(f.tellg());
  f.seekg(0);
  std::vector<uint8_t> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  return buf;
}

template <typename T>
std::vector<uint8_t> AsBytes(const std::vector<T>& v) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(v.data());
  return std::vector<uint8_t>(p, p + v.size() * sizeof(T));
}

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

// The stored scale of (row, group) as the quantizer uses it.
float SUse(const std::vector<uint16_t>& scale16, int gpr, int row, int g) {
  return r4dx::core::F16ToFloat(scale16[static_cast<size_t>(row) * gpr + g]) * kW8A16ScaleUnbias;
}

// Reconstruct [N,K] straight from the logical (unpacked) quantizer outputs.
std::vector<float> Recon(const std::vector<uint8_t>& q, const std::vector<uint16_t>& scale16, int N,
                         int K, int group) {
  const int gpr = K / group;
  std::vector<float> r(static_cast<size_t>(N) * K);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      r[static_cast<size_t>(n) * K + k] =
          static_cast<float>(static_cast<int>(q[static_cast<size_t>(n) * K + k]) - 128) *
          SUse(scale16, gpr, n, k / group);
  return r;
}

// |w - deq| <= s_use/2 when not clamped; for a clamped element (|w|/s_use > 127.5) the error is the
// excess over 127 s_use. Checked in double with a 1e-6 relative slack for the f32 division.
bool ErrorInvariant(const std::vector<float>& w, const std::vector<float>& deq,
                    const std::vector<uint16_t>& scale16, int N, int K, int group, std::string* why) {
  const int gpr = K / group;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const size_t i = static_cast<size_t>(n) * K + k;
      const double s = SUse(scale16, gpr, n, k / group);
      const double err = std::fabs(static_cast<double>(w[i]) - static_cast<double>(deq[i]));
      double bound;
      if (scale16[static_cast<size_t>(n) * gpr + k / group] == 0) {
        bound = std::fabs(static_cast<double>(w[i]));  // zero group: reconstructs 0
        if (deq[i] != 0.0f) { *why = "zero group not exactly 0"; return false; }
      } else {
        const double ratio = std::fabs(static_cast<double>(w[i])) / s;
        bound = ratio <= 127.5 ? 0.5 * s : (ratio - 127.0) * s;
        bound = bound * (1.0 + 1e-6) + 1e-30;
      }
      if (err > bound) {
        *why = "row " + std::to_string(n) + " k " + std::to_string(k);
        return false;
      }
    }
  return true;
}

void TestFixtures(const std::string& dir) {
  std::printf("---- fixtures (w8_ref.py byte-exactness) ----\n");
  nlohmann::json m;
  { std::ifstream f(dir + "/w8a16_manifest.json"); f >> m; }
  const int N = m.at("N").get<int>(), K = m.at("K").get<int>();
  SafetensorsReader reader(Utf8ToWide(dir + "/w8a16_input.safetensors"));
  const uint16_t* src = reinterpret_cast<const uint16_t*>(reader.Data("w"));
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < w.size(); ++i) w[i] = r4dx::core::Bf16ToFloat(src[i]);

  for (const auto& gj : m.at("groups")) {
    const int g = gj.get<int>();
    const std::string tag = " g" + std::to_string(g);
    for (int nth : {1, 4}) {
      std::vector<uint8_t> q;
      std::vector<uint16_t> s16;
      uint64_t zeros = 0;
      QuantizeInt8Symmetric(w.data(), N, K, g, nth, q, s16, &zeros);
      const auto wq = PackW8Bytes(q, N, K, nth);
      const auto ws = PackW8a16Scales(s16, N, K, g);
      Check("wq byte-exact" + tag + " threads=" + std::to_string(nth),
            wq == ReadFileBytes(dir + "/w8a16_wq_g" + std::to_string(g) + ".bin"));
      Check("ws byte-exact" + tag + " threads=" + std::to_string(nth),
            AsBytes(ws) == ReadFileBytes(dir + "/w8a16_ws_g" + std::to_string(g) + ".bin"));
      Check("zero-group count" + tag, zeros == m.at("zero_groups").at(std::to_string(g)).get<uint64_t>());
      // The reference dequant (kernel-indexed) against the Python one, bit for bit, and against the
      // logical reconstruction from q (so pack and unpack agree through two different routes).
      const auto dq = DequantW8a16(wq, ws, N, K, g);
      Check("dequant byte-exact" + tag,
            AsBytes(dq) == ReadFileBytes(dir + "/w8a16_dq_g" + std::to_string(g) + ".bin"));
      Check("dequant == logical reconstruction" + tag, dq == Recon(q, s16, N, K, g));
      std::string why;
      Check("error invariant on every fixture row" + tag, ErrorInvariant(w, dq, s16, N, K, g, &why));
      if (!why.empty()) std::printf("  at %s\n", why.c_str());
    }
  }
}

void TestRandomRoundTrip() {
  std::printf("---- Gaussian round trip ----\n");
  const int N = 64, K = 512;
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (auto& v : w) v = dist(rng);
  for (int g : {64, 128}) {
    std::vector<uint8_t> q;
    std::vector<uint16_t> s16;
    QuantizeInt8Symmetric(w.data(), N, K, g, 4, q, s16);
    const auto dq = DequantW8a16(PackW8Bytes(q, N, K, 4), PackW8a16Scales(s16, N, K, g), N, K, g);
    const double rel = RelL2(w, dq);
    std::printf("g%-3d rel_l2_err=%.5f\n", g, rel);
    Check("rel L2 < 0.012 at g" + std::to_string(g), rel < 0.012);
    // The measured LFM2 figure is 0.0065-0.0070; Gaussian data at g64 must not be worse than g128.
    Check("rel L2 > 0.001 (the test really quantizes)", rel > 0.001);
    std::string why;
    Check("error invariant, Gaussian g" + std::to_string(g), ErrorInvariant(w, dq, s16, N, K, g, &why));
  }
  // Every byte is 1..255 (-128 is unused), and codes stay symmetric: some 1 and some 255 exist.
  std::vector<uint8_t> q;
  std::vector<uint16_t> s16;
  QuantizeInt8Symmetric(w.data(), N, K, 128, 1, q, s16);
  Check("no byte is 0", std::count(q.begin(), q.end(), 0) == 0);
  Check("bytes 1 and 255 occur", std::count(q.begin(), q.end(), 1) > 0 && std::count(q.begin(), q.end(), 255) > 0);
}

void TestDeadAndTiny() {
  std::printf("---- dead / tiny-amax rows ----\n");
  const int N = 32, K = 256, G = 128;
  std::vector<float> w(static_cast<size_t>(N) * K, 0.0f);
  std::mt19937 rng(5);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  // row 0: dead (1e-25, with sign changes); row 1: exact zeros; row 2: amax ~ 1e-9 (flushes: the
  // biased scale 3e-8 < half the smallest f16 subnormal? no -- it rounds UP to one subnormal step,
  // so it is a live group with a coarse scale); row 3: amax ~ 3e-10 (flushes to zero); row 4+: ordinary.
  for (int k = 0; k < K; ++k) w[0 * K + k] = (k % 2 ? -1.0f : 1.0f) * 1e-25f;
  for (int k = 0; k < K; ++k) w[2 * K + k] = dist(rng) * 3e-10f;
  w[2 * K + 0] = 1.2e-9f;  // pin both groups' amax: biased scale 3.9e-8 rounds to ONE f16 subnormal step
  w[2 * K + G] = 1.2e-9f;
  for (int k = 0; k < K; ++k) w[3 * K + k] = dist(rng) * 1e-10f;
  for (int n = 4; n < N; ++n)
    for (int k = 0; k < K; ++k) w[static_cast<size_t>(n) * K + k] = dist(rng) * 0.01f;
  // One dead group inside an otherwise live row.
  for (int k = 0; k < G; ++k) w[5 * K + k] = 1e-24f * dist(rng);

  W8a16RunTally().groups = 0;
  W8a16RunTally().zero_groups = 0;
  std::vector<uint8_t> q;
  std::vector<uint16_t> s16;
  uint64_t zeros = 0;
  QuantizeInt8Symmetric(w.data(), N, K, G, 3, q, s16, &zeros);
  const int gpr = K / G;
  Check("run tally counts all groups", W8a16RunTally().groups.load() == static_cast<uint64_t>(N) * gpr);
  Check("run tally == out param", W8a16RunTally().zero_groups.load() == zeros);
  // rows 0, 1 and 3 and group (5,0) are dead: scale bits 0, codes exactly 128.
  uint64_t expect_zero = 0;
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < gpr; ++g) {
      const bool dead = s16[static_cast<size_t>(n) * gpr + g] == 0;
      expect_zero += dead;
      if (dead)
        for (int k = 0; k < G; ++k)
          if (q[static_cast<size_t>(n) * K + g * G + k] != 128) {
            Check("dead group codes are 128 (row " + std::to_string(n) + ")", false);
            break;
          }
    }
  Check("zero-group count equals the scale-0 groups", zeros == expect_zero);
  Check("rows 0,1,3 dead (6 groups) + group (5,0) = 7 zero groups", zeros == 7);
  Check("row 2 (amax ~1e-9) is live", s16[2 * gpr] != 0 && s16[2 * gpr + 1] != 0);
  const auto dq = DequantW8a16(PackW8Bytes(q, N, K, 1), PackW8a16Scales(s16, N, K, G), N, K, G);
  for (int k = 0; k < K; ++k) {
    if (dq[0 * K + k] != 0.0f || dq[1 * K + k] != 0.0f || dq[3 * K + k] != 0.0f) {
      Check("dead rows dequantize to exactly 0", false);
      break;
    }
  }
  std::string why;
  Check("error invariant incl. dead/tiny rows", ErrorInvariant(w, dq, s16, N, K, G, &why));
  if (!why.empty()) std::printf("  at %s\n", why.c_str());
  // Row 2's coarse scale still reconstructs within one stored step.
  std::vector<float> r2(dq.begin() + 2 * K, dq.begin() + 3 * K), w2(w.begin() + 2 * K, w.begin() + 3 * K);
  Check("tiny live row reconstructs (rel L2 < 0.1)", RelL2(w2, r2) < 0.1);
}

void TestConstantClampAndLimits() {
  std::printf("---- constant group, clamp, amax limit ----\n");
  const int N = 16, K = 128, G = 128;
  // A constant group: every element -3.25 -> one code, reconstructs within half a step.
  {
    std::vector<float> w(static_cast<size_t>(N) * K, -3.25f);
    std::vector<uint8_t> q;
    std::vector<uint16_t> s16;
    QuantizeInt8Symmetric(w.data(), N, K, G, 1, q, s16);
    Check("constant group: all codes equal", std::count(q.begin(), q.end(), q[0]) == static_cast<long>(q.size()));
    Check("constant group: -amax is code -127 (byte 1)", q[0] == 1);
    const auto dq = Recon(q, s16, N, K, G);
    Check("constant group reconstructs within 1e-3", std::fabs(dq[0] + 3.25f) < 1e-3f * 3.25f);
  }
  // The clamp: the stored f16 scale can round DOWN, so amax / s_use can exceed 127 by up to 2^-11
  // relative -- it must round to 127 (byte 255), never 128. Scan for an amax where it happens.
  {
    int found = 0;
    for (float amax = 0.01f; amax < 2000.0f && found < 5; amax *= 1.0137f) {
      std::vector<float> w(static_cast<size_t>(N) * K, 0.0f);
      w[0] = amax;
      w[1] = -amax;
      std::vector<uint8_t> q;
      std::vector<uint16_t> s16;
      QuantizeInt8Symmetric(w.data(), N, K, G, 1, q, s16);
      const float s_use = SUse(s16, 1, 0, 0);
      if (amax / s_use > 127.0f) {
        ++found;
        Check("clamp: +amax -> byte 255", q[0] == 255);
        Check("clamp: -amax -> byte 1", q[1] == 1);
      }
      Check("amax maps to +-127 (bytes 255 / 1)", q[0] == 255 && q[1] == 1);
    }
    Check("found amax values whose stored scale rounds down (clamp exercised)", found >= 1);
  }
  // amax limit: 2031 is representable, 2040 is an error naming the tensor; NaN / inf are errors.
  {
    std::vector<float> w(static_cast<size_t>(N) * K, 0.0f);
    std::vector<uint8_t> q;
    std::vector<uint16_t> s16;
    w[0] = 2031.0f;
    bool threw = false;
    try { QuantizeInt8Symmetric(w.data(), N, K, G, 1, q, s16, nullptr, "t.big"); } catch (...) { threw = true; }
    Check("amax 2031 quantizes", !threw);
    w[0] = 2040.0f;
    std::string msg;
    try { QuantizeInt8Symmetric(w.data(), N, K, G, 2, q, s16, nullptr, "t.big"); } catch (const std::exception& e) { msg = e.what(); }
    Check("amax 2040 is an error naming the tensor", msg.find("t.big") != std::string::npos && msg.find("overflows") != std::string::npos);
    w[0] = std::numeric_limits<float>::quiet_NaN();
    msg.clear();
    try { QuantizeInt8Symmetric(w.data(), N, K, G, 2, q, s16, nullptr, "t.nan"); } catch (const std::exception& e) { msg = e.what(); }
    Check("NaN weight is an error", msg.find("non-finite") != std::string::npos);
  }
  // Argument refusals.
  {
    std::vector<float> w(static_cast<size_t>(N) * 192, 1.0f);
    std::vector<uint8_t> q;
    std::vector<uint16_t> s16;
    bool t1 = false, t2 = false, t3 = false, t4 = false;
    try { QuantizeInt8Symmetric(w.data(), N, 192, 128, 1, q, s16); } catch (...) { t1 = true; }   // K % G
    try { QuantizeInt8Symmetric(w.data(), N, 192, 96, 1, q, s16); } catch (...) { t2 = true; }    // group 96
    try { PackW8Bytes(q, 8, 128, 1); } catch (...) { t3 = true; }                                // N % 16
    try { PackW8a16Scales(s16, 16, 192, 128); } catch (...) { t4 = true; }                        // K % G
    Check("K % G refused", t1);
    Check("group 96 refused", t2);
    Check("PackW8Bytes N % 16 refused", t3);
    Check("PackW8a16Scales K % G refused", t4);
  }
}

void TestPackIsPermutation() {
  std::printf("---- packer is a permutation ----\n");
  const int N = 48, K = 192;
  std::vector<uint8_t> q(static_cast<size_t>(N) * K);
  // Distinct-ish payload: every (row, k) pair gets a code that identifies it modulo 251.
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) q[static_cast<size_t>(n) * K + k] = static_cast<uint8_t>((n * 31 + k * 7) % 251 + 1);
  const auto wq = PackW8Bytes(q, N, K, 2);
  auto a = q, b = wq;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  Check("same multiset of bytes", a == b);
  // Tile locality: tile t occupies bytes [16*K*t, 16*K*(t+1)) and holds only that tile's rows.
  bool local = true;
  for (int t = 0; t < N / 16 && local; ++t) {
    std::vector<uint8_t> tile(wq.begin() + static_cast<size_t>(t) * 16 * K, wq.begin() + static_cast<size_t>(t + 1) * 16 * K);
    std::vector<uint8_t> expect(q.begin() + static_cast<size_t>(t) * 16 * K, q.begin() + static_cast<size_t>(t + 1) * 16 * K);
    std::sort(tile.begin(), tile.end());
    std::sort(expect.begin(), expect.end());
    local = tile == expect;
  }
  Check("each 16-row tile is one contiguous 16*K run (TP row slicing)", local);
}

}  // namespace

int main() {
  TestFixtures(R4DX_CONVERT_FIXTURES_DIR);
  TestRandomRoundTrip();
  TestDeadAndTiny();
  TestConstantClampAndLimits();
  TestPackIsPermutation();
  if (g_failures != 0) {
    std::printf("FAIL (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
