// inkling2_expert_codec.hpp (docs/inkling2-plan.md 3.2, 4): NVFP4 / MXFP8 expert decoding to f32 and the w13 gate/up
// de-interleave, against goldens produced by transformers' own `Inkling2LoopDequantizeExperts` (float32) and
// `Interleave(dim=1)` (tools/convert_ref/make_inkling2_expert_golden.py -> tests/convert/data/inkling2_expert_golden.json,
// where a numpy port of the op was asserted bit-identical to the HF op). Plus hand-built unit cases that pin the nibble
// order, the E2M1 table, the factor, the NaN refusals and the row order. CPU only.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/inkling2_expert_codec.hpp"

namespace ik = r4dx_convert::inkling2;
using nlohmann::json;

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

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
}

std::vector<uint8_t> FromHex(const std::string& h) {
  std::vector<uint8_t> out(h.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(std::stoi(h.substr(2 * i, 2), nullptr, 16));
  return out;
}

std::vector<uint32_t> Bits(const std::vector<float>& v) {
  std::vector<uint32_t> b(v.size());
  std::memcpy(b.data(), v.data(), v.size() * 4);
  return b;
}

std::vector<float> Floats(const std::vector<uint8_t>& v) {
  std::vector<float> f(v.size() / 4);
  std::memcpy(f.data(), v.data(), v.size());
  return f;
}

float U32AsFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void TestGolden() {
  std::printf("---- goldens from transformers' Inkling2LoopDequantizeExperts ----\n");
  std::ifstream f(std::string(R4DX_CONVERT_DATA_DIR) + "/inkling2_expert_golden.json");
  if (!f) {
    Check(false, "cannot open inkling2_expert_golden.json");
    return;
  }
  const json g = json::parse(f);
  for (const json& c : g.at("cases")) {
    const std::string name = c.at("name").get<std::string>();
    const bool nvfp4 = c.at("format") == "nvfp4";
    const int64_t E = c.at("experts"), rows = c.at("rows"), cols = c.at("cols");
    const std::vector<uint8_t> weight = FromHex(c.at("weight_hex")), scale = FromHex(c.at("scale_hex"));
    std::vector<uint8_t> amax;
    ik::ExpertTensors t;
    t.format = nvfp4 ? ik::ExpertFormat::kNvfp4 : ik::ExpertFormat::kMxfp8;
    t.experts = E;
    t.rows = rows;
    t.cols = cols;
    t.weight = weight.data();
    t.scale = scale.data();
    if (nvfp4) {
      for (uint32_t b : c.at("amax_bits").get<std::vector<uint32_t>>())
        for (int k = 0; k < 4; ++k) amax.push_back(static_cast<uint8_t>(b >> (8 * k)));
      t.amax = amax.data();
      t.denominator = static_cast<float>(c.at("denominator").get<double>());
    }
    Check(static_cast<int64_t>(weight.size()) == E * t.WeightBytesPerExpert() && static_cast<int64_t>(scale.size()) == E * t.ScaleBytesPerExpert(),
          name + ": byte counts match the format's per-expert sizes");
    const std::vector<uint32_t> want = c.at("decoded_bits").get<std::vector<uint32_t>>();
    std::vector<float> all(static_cast<size_t>(E * rows * cols)), all_de;
    for (int64_t e = 0; e < E; ++e) ik::DecodeExpert(t, e, all.data() + e * rows * cols);
    Check(Bits(all) == want, name + ": DecodeExpert == HF float32 dequantization, bit for bit, every expert");
    if (c.at("interleaved_w13").get<bool>()) {
      const std::vector<uint32_t> want_de = c.at("deinterleaved_bits").get<std::vector<uint32_t>>();
      all_de.assign(all.size(), -1.0f);
      for (int64_t e = 0; e < E; ++e) ik::DeinterleaveGateUp(all.data() + e * rows * cols, all_de.data() + e * rows * cols, rows, cols);
      Check(Bits(all_de) == want_de, name + ": DeinterleaveGateUp == HF Interleave(dim=1), bit for bit");
      // Rows [0, rows/2) are the even source rows (gate), [rows/2, rows) the odd ones (up).
      bool order = true;
      for (int64_t e = 0; e < E; ++e)
        for (int64_t r = 0; r < rows / 2; ++r)
          order = order && std::memcmp(all_de.data() + (e * rows + r) * cols, all.data() + (e * rows + 2 * r) * cols, cols * 4) == 0 &&
                  std::memcmp(all_de.data() + (e * rows + rows / 2 + r) * cols, all.data() + (e * rows + 2 * r + 1) * cols, cols * 4) == 0;
      Check(order, name + ": gate = even source rows (first half), up = odd source rows (second half)");
    }
    if (!nvfp4 && c.at("interleaved_w13").get<bool>()) {
      const std::vector<uint8_t> want_codes = FromHex(c.at("raw_codes_deinterleaved_hex")), want_scales = FromHex(c.at("raw_scales_deinterleaved_hex"));
      std::vector<uint8_t> codes(want_codes.size()), scales(want_scales.size());
      const int64_t gpr = cols / 32;
      for (int64_t e = 0; e < E; ++e) {
        std::vector<uint8_t> rc(static_cast<size_t>(rows * cols)), rs(static_cast<size_t>(rows * gpr));
        ik::RawMxfp8Expert(t, e, rc.data(), rs.data());
        ik::DeinterleaveGateUp(rc.data(), codes.data() + e * rows * cols, rows, cols);
        ik::DeinterleaveGateUp(rs.data(), scales.data() + e * rows * gpr, rows, gpr);
      }
      Check(codes == want_codes && scales == want_scales, name + ": raw MXFP8 codes + scale bytes de-interleaved unchanged (the bit-exact copy path)");
      // The raw bytes decode to the same values as the f32 path (decode of the raw copy == HF dequantization).
      bool same = true;
      for (int64_t e = 0; e < E && same; ++e)
        for (int64_t r = 0; r < rows; ++r)
          for (int64_t k = 0; k < cols; ++k) {
            const float v = r4dx_convert::mxfp8::DecodeE4m3(codes[static_cast<size_t>((e * rows + r) * cols + k)]) *
                            r4dx_convert::mxfp8::ScaleValue(scales[static_cast<size_t>((e * rows + r) * gpr + k / 32)]);
            same = same && std::memcmp(&v, &all_de[static_cast<size_t>((e * rows + r) * cols + k)], 4) == 0;
          }
      Check(same, name + ": raw copy decodes to the HF values");
    }
  }
}

void TestNibbleOrderAndTable() {
  std::printf("---- NVFP4 nibble order, E2M1 table, factor (hand-built) ----\n");
  // 1 expert, 1 row, 16 columns: byte j holds element 2j (LOW nibble) and 2j + 1 (HIGH nibble).
  std::vector<uint8_t> w(8), s = {0x38};  // block scale 1.0
  for (int j = 0; j < 8; ++j) w[static_cast<size_t>(j)] = static_cast<uint8_t>((2 * j) | ((2 * j + 1) << 4));  // elements 0..15 hold codes 0..15
  float amax = 1.0f;
  uint8_t amax_b[4];
  std::memcpy(amax_b, &amax, 4);
  ik::ExpertTensors t;
  t.format = ik::ExpertFormat::kNvfp4;
  t.experts = 1;
  t.rows = 1;
  t.cols = 16;
  t.weight = w.data();
  t.scale = s.data();
  t.amax = amax_b;
  t.denominator = 1.0f;
  std::vector<float> out(16);
  ik::DecodeExpert(t, 0, out.data());
  bool table = true;
  for (int i = 0; i < 16; ++i) table = table && out[static_cast<size_t>(i)] == ik::kE2m1[i] && std::signbit(out[static_cast<size_t>(i)]) == std::signbit(ik::kE2m1[i]);
  Check(table, "element i = E2M1[code i]: even element in the LOW nibble, odd in the HIGH nibble, -0 keeps its sign");
  const float want[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  bool vals = true;
  for (int i = 0; i < 16; ++i) vals = vals && out[static_cast<size_t>(i)] == want[i];
  Check(vals, "E2M1 values {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives");
  // 0x21: low nibble 1 (0.5), high nibble 2 (1.0): the pair order [0.5, 1.0], not [1.0, 0.5].
  w[0] = 0x21;
  ik::DecodeExpert(t, 0, out.data());
  Check(out[0] == 0.5f && out[1] == 1.0f, "byte 0x21 decodes to [0.5, 1.0]");
  // Scale and factor: scale code 0x40 = 2.0, amax 3.0 / denominator 6.0 = 0.5  ->  e2m1 * 2 * 0.5.
  s[0] = 0x40;
  amax = 3.0f;
  std::memcpy(amax_b, &amax, 4);
  t.denominator = 6.0f;
  ik::DecodeExpert(t, 0, out.data());
  Check(out[0] == 0.5f * 2.0f * 0.5f && out[3] == 1.5f * 2.0f * 0.5f && ik::ExpertFactor(t, 0) == 0.5f, "value = e2m1 * scale * (amax / denominator)");
  // Smallest subnormal scale (0x01 = 2^-9) times the smallest e2m1.
  s[0] = 0x01;
  t.denominator = 3.0f;
  ik::DecodeExpert(t, 0, out.data());
  Check(out[0] == 0.5f * std::ldexp(1.0f, -9) && out[1] == 1.0f * std::ldexp(1.0f, -9), "an e4m3 subnormal block scale is honoured");
  // Two experts / rows / blocks index independently: expert 1 uses amax[1] and its own bytes.
  std::vector<uint8_t> w2(8 * 2 * 2 * 1, 0x11), s2(2 * 2 * 1 * 2, 0x38);  // E=2, rows=2, cols=16 -> weight 2*2*8, scale 2*2*1
  s2.resize(4);
  float am[2] = {1.0f, 4.0f};
  uint8_t am_b[8];
  std::memcpy(am_b, am, 8);
  ik::ExpertTensors t2 = t;
  t2.experts = 2;
  t2.rows = 2;
  t2.weight = w2.data();
  t2.scale = s2.data();
  t2.amax = am_b;
  t2.denominator = 2.0f;
  std::vector<float> o2(2 * 16);
  ik::DecodeExpert(t2, 1, o2.data());
  Check(o2[0] == 0.5f * 1.0f * 2.0f && o2[31] == 0.5f * 1.0f * 2.0f, "expert index selects its own amax (4 / 2 = 2)");
  ik::DecodeExpert(t2, 0, o2.data());
  Check(o2[0] == 0.5f * 1.0f * 0.5f, "expert 0 uses amax[0] / denominator");
}

void TestMxfp8Hand() {
  std::printf("---- MXFP8 (hand-built) ----\n");
  std::vector<uint8_t> w(32), s = {127};
  for (int i = 0; i < 32; ++i) w[static_cast<size_t>(i)] = static_cast<uint8_t>(0x38 + i % 3);  // 1.0, 1.125, 1.25
  ik::ExpertTensors t;
  t.format = ik::ExpertFormat::kMxfp8;
  t.experts = 1;
  t.rows = 1;
  t.cols = 32;
  t.weight = w.data();
  t.scale = s.data();
  std::vector<float> out(32);
  ik::DecodeExpert(t, 0, out.data());
  Check(out[0] == 1.0f && out[1] == 1.125f && out[2] == 1.25f, "scale byte 127 = 2^0");
  s[0] = 120;
  ik::DecodeExpert(t, 0, out.data());
  Check(out[0] == std::ldexp(1.0f, -7) && out[2] == 1.25f * std::ldexp(1.0f, -7), "value = e4m3 * 2^(scale - 127)");
  s[0] = 0;
  ik::DecodeExpert(t, 0, out.data());
  Check(out[0] == std::ldexp(1.0f, -127), "scale byte 0 = 2^-127 (a float subnormal, exact)");
  w[3] = 0x7F;
  Check(Throws([&] { ik::DecodeExpert(t, 0, out.data()); }, "NaN e4m3"), "a NaN e4m3 code is refused");
  w[3] = 0x38;
  s[0] = 255;
  Check(Throws([&] { ik::DecodeExpert(t, 0, out.data()); }, "255"), "E8M0 scale byte 255 is refused");
  std::vector<uint8_t> codes(32), scales(1);
  Check(Throws([&] { ik::RawMxfp8Expert(t, 0, codes.data(), scales.data()); }, "255"), "the raw copy refuses scale byte 255 too");
  s[0] = 100;
  w[5] = 0xFF;
  Check(Throws([&] { ik::RawMxfp8Expert(t, 0, codes.data(), scales.data()); }, "NaN"), "the raw copy refuses a NaN code");
  Check(Throws([&] { ik::DecodeExpert(t, 1, out.data()); }, "out of range"), "expert index out of range throws");
  // NVFP4: a NaN block scale.
  std::vector<uint8_t> w4(8, 0x11), s4 = {0x7F};
  uint8_t amax_b[4];
  const float one = 1.0f;
  std::memcpy(amax_b, &one, 4);
  ik::ExpertTensors t4;
  t4.format = ik::ExpertFormat::kNvfp4;
  t4.experts = 1;
  t4.rows = 1;
  t4.cols = 16;
  t4.weight = w4.data();
  t4.scale = s4.data();
  t4.amax = amax_b;
  std::vector<float> o4(16);
  Check(Throws([&] { ik::DecodeExpert(t4, 0, o4.data()); }, "NaN F8_E4M3"), "a NaN NVFP4 block scale is refused");
}

void TestDeinterleave() {
  std::printf("---- DeinterleaveGateUp ----\n");
  const int rows = 6, len = 3;
  std::vector<int> src(rows * len), dst(rows * len, -1);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < len; ++k) src[static_cast<size_t>(r * len + k)] = r * 10 + k;
  ik::DeinterleaveGateUp(src.data(), dst.data(), rows, len);
  const int order[rows] = {0, 2, 4, 1, 3, 5};
  bool ok = true;
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < len; ++k) ok = ok && dst[static_cast<size_t>(r * len + k)] == order[r] * 10 + k;
  Check(ok, "rows [0 1 2 3 4 5] -> [0 2 4 | 1 3 5]: gate (even) first, up (odd) second");
  Check(Throws([&] { ik::DeinterleaveGateUp(src.data(), dst.data(), 5, len); }, "odd"), "an odd row count is refused");
}

}  // namespace

int main() {
  TestGolden();
  TestNibbleOrderAndTable();
  TestMxfp8Hand();
  TestDeinterleave();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all inkling2 expert codec checks passed\n");
  return 0;
}
