// w8a16 layout plumbing (docs/w8a16.md 5.2, 7.1; ctest `convert_w8a16_layout`). CPU-only.
//
//   - PlanLinearLayouts: tensor names, shapes and byte sizes (alone, next to bf16, next to w4a16);
//     LinearLayoutTensorNames and LinearLayoutBytes agree with it;
//   - the LayoutSetId <-> ParseLayoutSetId round trip (`w8a16.g128`, `bf16+w8a16.g64`) and its refusals
//     (`w8a16.g96`, `trellis.k5+w8a16.g128`, a non-canonical order);
//   - the shape refusals (N % 16, K % G, group 96) and trellis next to w8a16;
//   - EmitLinearLayouts writes exactly Quantize+Pack's bytes; a kept-bf16 linear writes no w8a16 tensor;
//     LDLQ with w8a16 is refused instead of silently RTN-ing;
//   - --keep-bf16's extra-byte accounting prices a w8a16 linear correctly;
//   - when r4dx-convert is built: `--selftest --layouts w8a16` on the fixture (bytes == w8_ref.py's, the
//     metadata block, g64, bf16 companion, keep-bf16, mixed w4a16+w8a16, the refusals).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/keep_bf16.hpp"
#include "r4dx_convert/linear_layouts.hpp"
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

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
}

std::string TempPath(const std::string& leaf) {
  const char* t = std::getenv("TEMP");
  return std::string(t ? t : ".") + "\\" + leaf;
}

struct Container {
  bool ok = false;
  nlohmann::json header;
  std::map<std::string, std::string> tensors;
};

Container ReadContainer(const std::string& path) {
  Container c;
  std::ifstream f(path, std::ios::binary);
  if (!f) return c;
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string file = ss.str();
  if (file.size() < 8) return c;
  uint64_t n = 0;
  std::memcpy(&n, file.data(), 8);
  if (8 + n > file.size()) return c;
  c.header = nlohmann::json::parse(file.substr(8, static_cast<size_t>(n)));
  const size_t data0 = 8 + static_cast<size_t>(n);
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto off = it.value().at("data_offsets");
    const size_t a = off[0].get<size_t>(), b = off[1].get<size_t>();
    c.tensors[it.key()] = file.substr(data0 + a, b - a);
  }
  c.ok = true;
  return c;
}

std::vector<float> Fixture(int N, int K) {
  std::vector<float> w(static_cast<size_t>(N) * K);
  uint32_t s = 12345;
  for (auto& v : w) {
    s = s * 1664525u + 1013904223u;
    v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
  }
  return w;
}

LayoutSet W8(int group, bool bf16 = false, bool w4 = false) {
  LayoutSet ls;
  ls.bf16 = bf16;
  ls.w4a16 = w4;
  ls.w8a16 = true;
  ls.w8a16_group = group;
  return ls;
}

void TestPlanning() {
  std::printf("---- planning ----\n");
  const int N = 48, K = 256;
  for (int g : {64, 128}) {
    ContainerWriter w;
    PlanLinearLayouts(w, "x", N, K, W8(g));
    Check("g" + std::to_string(g) + ": two tensors", w.PlannedTensorCount() == 2);
    Check("names", w.PlannedName(0) == "x.w8a16.wq" && w.PlannedName(1) == "x.w8a16.ws");
    Check("wq shape {N*K} bytes N*K", w.PlannedShape(0) == std::vector<int64_t>({N * K}) && w.PlannedBytes(0) == static_cast<uint64_t>(N) * K);
    Check("ws shape {N*K/G, 2} bytes N*K/G*2",
          w.PlannedShape(1) == std::vector<int64_t>({N * K / g, 2}) && w.PlannedBytes(1) == static_cast<uint64_t>(N) * K / g * 2);
    Check("LinearLayoutBytes == planned", LinearLayoutBytes(N, K, W8(g)) == w.PlannedDataBytes());
    Check("LinearLayoutTensorNames == planned names",
          LinearLayoutTensorNames("x", W8(g)) == std::vector<std::string>({w.PlannedName(0), w.PlannedName(1)}));
  }
  {  // next to bf16 and next to w4a16: every layout's bytes add up, in the documented order
    ContainerWriter w;
    PlanLinearLayouts(w, "x", N, K, W8(128, /*bf16=*/true, /*w4=*/true));
    Check("bf16+w4a16+w8a16: 5 tensors", w.PlannedTensorCount() == 5);
    Check("order bf16, w4a16, w8a16", w.PlannedName(0) == "x.bf16.w" && w.PlannedName(1) == "x.w4a16.wq" &&
                                          w.PlannedName(3) == "x.w8a16.wq" && w.PlannedName(4) == "x.w8a16.ws");
    Check("LinearLayoutBytes == planned (all three)", LinearLayoutBytes(N, K, W8(128, true, true)) == w.PlannedDataBytes());
    Check("names agree", LinearLayoutTensorNames("x", W8(128, true, true)).size() == 5);
  }
  {  // exact byte math on a real LFM2 shape: expert gate_up 3072x2048 at g128
    const uint64_t NK = 3072ull * 2048;
    Check("byte math 3072x2048 g128 = N*K + N*K/64", LinearLayoutBytes(3072, 2048, W8(128)) == NK + NK / 64);
    Check("byte math 2048x1536 g64 = N*K + N*K/32", LinearLayoutBytes(2048, 1536, W8(64)) == 2048ull * 1536 * 33 / 32);
  }
  // Refusals, naming the linear.
  {
    ContainerWriter w;
    Check("N % 16 refused, names the linear", Throws([&] { PlanLinearLayouts(w, "bad.n", 24, 256, W8(128)); }, "bad.n"));
    Check("K % G refused", Throws([&] { PlanLinearLayouts(w, "bad.k", 48, 192, W8(128)); }, "bad.k"));
    Check("K % 64 refused at g64", Throws([&] { PlanLinearLayouts(w, "bad.k64", 48, 160, W8(64)); }, "bad.k64"));
    Check("group 96 refused", Throws([&] { PlanLinearLayouts(w, "bad.g", 48, 192, W8(96)); }, "96"));
    Check("K=192 is fine at g64", !Throws([&] { PlanLinearLayouts(w, "ok", 48, 192, W8(64)); }));
  }
  {  // trellis next to w8a16 is refused (both directions)
    ContainerWriter w;
    LayoutSet ls = TrellisLayoutSet(5);
    ls.w8a16 = true;
    Check("trellis + w8a16 refused", Throws([&] { PlanLinearLayouts(w, "t", 256, 256, ls); }, "w8a16"));
  }
}

void TestLayoutSetId() {
  std::printf("---- LayoutSetId ----\n");
  auto round_trip = [](const LayoutSet& ls, const std::string& id) {
    LayoutSet back;
    return LayoutSetId(ls) == id && ParseLayoutSetId(id, &back) && LayoutSetId(back) == id && back.w8a16 == ls.w8a16 &&
           back.w8a16_group == ls.w8a16_group && back.bf16 == ls.bf16 && back.w4a16 == ls.w4a16;
  };
  Check("w8a16.g128", round_trip(W8(128), "w8a16.g128"));
  Check("bf16+w8a16.g64", round_trip(W8(64, true), "bf16+w8a16.g64"));
  Check("bf16+w4a16.g64+w8a16.g128", round_trip(W8(128, true, true), "bf16+w4a16.g64+w8a16.g128"));
  LayoutSet x;
  for (const char* bad : {"w8a16.g96", "w8a16.g", "w8a16.g0128", "w8a16", "w8a16.g128+bf16", "w8a16.g128+w8a16.g128",
                          "trellis.k5+w8a16.g128", "w8a16.g12x", "bf16+bf16", "w8a16.g1280"})
    Check(std::string("refused: ") + bad, !ParseLayoutSetId(bad, &x));
  Check("an old id still parses (w4a16.g64)", ParseLayoutSetId("w4a16.g64", &x) && !x.w8a16);
}

void TestEmission() {
  std::printf("---- emission ----\n");
  const int N = 32, K = 256, G = 128;
  const std::vector<float> w = Fixture(N, K);
  std::vector<uint8_t> q;
  std::vector<uint16_t> s16;
  QuantizeInt8Symmetric(w.data(), N, K, G, 1, q, s16);
  const auto wq = PackW8Bytes(q, N, K, 1);
  const auto ws = PackW8a16Scales(s16, N, K, G);

  const std::string path = TempPath("r4dx_test_w8a16_layout.r4dx");
  {
    ContainerWriter writer;
    PlanLinearLayouts(writer, "t", N, K, W8(G, /*bf16=*/true));
    nlohmann::json meta;
    meta["r4dx_format_version"] = "1";
    writer.FinalizeHeader(path, meta);
    EmitLinearLayouts(writer, "t", w, N, K, W8(G, true), 3);
    writer.Finish();
  }
  const Container c = ReadContainer(path);
  Check("container readable", c.ok);
  if (c.ok) {
    Check("wq bytes == Quantize+Pack", c.tensors.at("t.w8a16.wq") == std::string(wq.begin(), wq.end()));
    Check("ws bytes == Quantize+Pack", c.tensors.at("t.w8a16.ws") ==
                                          std::string(reinterpret_cast<const char*>(ws.data()), ws.size() * 2));
    Check("bf16 companion present", c.tensors.count("t.bf16.w") == 1);
    Check("header dtype/shape of ws", c.header.at("t.w8a16.ws").at("shape") == nlohmann::json::array({N * K / G, 2}));
  }
  std::remove(path.c_str());

  // A kept-bf16 linear emits only the bf16 tensor (the loader's fallback shape).
  {
    ContainerWriter writer;
    PlanLinearLayouts(writer, "k", N, K, KeptBf16LayoutSet());
    Check("kept linear plans one tensor", writer.PlannedTensorCount() == 1 && writer.PlannedName(0) == "k.bf16.w");
  }
  // LDLQ with w8a16 is refused, not silently RTN.
  {
    ContainerWriter writer;
    PlanLinearLayouts(writer, "l", N, K, W8(G));
    nlohmann::json meta;
    meta["r4dx_format_version"] = "1";
    const std::string p2 = TempPath("r4dx_test_w8a16_layout_ldlq.r4dx");
    writer.FinalizeHeader(p2, meta);
    LdlqFactor f;
    f.K = K;
    QuantOptions opts;
    opts.ldlq = &f;
    Check("LDLQ + w8a16 refused", Throws([&] { EmitLinearLayouts(writer, "l", w, N, K, W8(G), 1, opts); }, "LDLQ"));
    std::remove(p2.c_str());
  }
  // --keep-bf16 prices the linear it replaced at the w8a16 bytes.
  {
    KeepBf16Selector sel("^t$");
    std::ostringstream log;
    const LayoutSet would = W8(G);
    sel.Record("t", N, K, would, log);
    const int64_t expect = static_cast<int64_t>(N) * K * 2 - (static_cast<int64_t>(N) * K + static_cast<int64_t>(N) * K / G * 2);
    Check("keep-bf16 extra bytes == bf16 bytes - w8a16 bytes", sel.ExtraBytes() == expect);
  }
}

#if defined(R4DX_CONVERT_EXE)

int Run(const std::string& args, const std::string& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

std::string ReadText(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool Has(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

void TestExe(const std::string& fixtures) {
  std::printf("---- r4dx-convert --selftest --layouts w8a16 ----\n");
  const std::string input = fixtures + "/w8a16_input.safetensors";
  const std::string out = TempPath("r4dx_test_w8a16_layout_exe.r4dx");
  const std::string log = TempPath("r4dx_test_w8a16_layout_exe.log");
  const std::string base = "--selftest --selftest-input \"" + input + "\" --selftest-output \"" + out + "\" --threads 3 ";
  auto fx = [&](const std::string& name) { return ReadText(fixtures + "/" + name); };
  nlohmann::json manifest = nlohmann::json::parse(fx("w8a16_manifest.json"));

  {  // g128 (the default), w8a16 only
    const int rc = Run(base + "--layouts w8a16 --no-bf16", log);
    const Container c = ReadContainer(out);
    Check("selftest w8a16: exit 0", rc == 0 && c.ok);
    if (rc != 0) std::printf("%s", ReadText(log).c_str());
    if (c.ok) {
      Check("exactly the two w8a16 tensors", c.tensors.size() == 2 && c.tensors.count("selftest.w8a16.wq") && c.tensors.count("selftest.w8a16.ws"));
      Check("wq byte-exact against w8_ref.py (fixture)", c.tensors.at("selftest.w8a16.wq") == fx("w8a16_wq_g128.bin"));
      Check("ws byte-exact against w8_ref.py (fixture)", c.tensors.at("selftest.w8a16.ws") == fx("w8a16_ws_g128.bin"));
      const nlohmann::json md = c.header.at("__metadata__");
      const nlohmann::json want = {{"version", 1}, {"group", 128}, {"scale", "f16"}, {"scale_log2_bias", 12},
                                   {"code", "int8_sym127_offset_binary_x80"},
                                   {"fragment_permutation", "wmma16x16x16_f16_b_lane16_k32block"}};
      Check("metadata quant.w8a16 block", md.at("quant").contains("w8a16") && md.at("quant").at("w8a16") == want);
      Check("zero-group count in the log",
            Has(ReadText(log), std::to_string(manifest.at("zero_groups").at("128").get<int>()) + " zero groups"));
    }
  }
  {  // g64
    const int rc = Run(base + "--layouts w8a16 --no-bf16 --w8a16-group 64", log);
    const Container c = ReadContainer(out);
    Check("selftest g64: exit 0", rc == 0 && c.ok);
    if (c.ok) {
      Check("g64 wq/ws byte-exact", c.tensors.at("selftest.w8a16.wq") == fx("w8a16_wq_g64.bin") && c.tensors.at("selftest.w8a16.ws") == fx("w8a16_ws_g64.bin"));
      Check("g64 metadata", c.header.at("__metadata__").at("quant").at("w8a16").at("group") == 64);
    }
  }
  {  // bf16 companion
    const int rc = Run(base + "--layouts w8a16", log);
    const Container c = ReadContainer(out);
    Check("w8a16 + bf16 companion (bf16 default on)", rc == 0 && c.ok && c.tensors.size() == 3 && c.tensors.count("selftest.bf16.w"));
  }
  {  // keep-bf16 wins: bf16 only, no metadata block
    const int rc = Run(base + "--layouts w8a16 --no-bf16 --keep-bf16 selftest", log);
    const Container c = ReadContainer(out);
    Check("--keep-bf16 wins over w8a16", rc == 0 && c.ok && c.tensors.size() == 1 && c.tensors.count("selftest.bf16.w") &&
                                              !c.header.at("__metadata__").at("quant").contains("w8a16"));
  }
  // The w8a16 fixture holds dead rows and amax 2000, which the 4-bit grid refuses (f16 scale flush); the
  // w4a16 cases run on the ordinary fixture (input.safetensors) instead.
  const std::string base4 = "--selftest --selftest-input \"" + fixtures + "/input.safetensors\" --selftest-output \"" + out +
                            "\" --threads 3 ";
  {  // mixed 4-bit + 8-bit in one set: --quant search then applies to the 4-bit grid only
    Run(base4 + "--layouts w8a16 --no-bf16", log);
    const Container only8 = ReadContainer(out);
    const int rc = Run(base4 + "--layouts w4a16+w8a16 --no-bf16 --quant search", log);
    const Container c = ReadContainer(out);
    Check("w4a16+w8a16 together (search for w4a16): exit 0", rc == 0 && c.ok && c.tensors.size() == 4);
    if (rc != 0) std::printf("%s", ReadText(log).c_str());
    if (c.ok && only8.ok && c.tensors.count("selftest.w8a16.wq"))
      Check("the w8a16 bytes are RTN regardless of --quant search",
            c.tensors.at("selftest.w8a16.wq") == only8.tensors.at("selftest.w8a16.wq") &&
                c.tensors.at("selftest.w8a16.ws") == only8.tensors.at("selftest.w8a16.ws"));
  }
  {  // a plain w4a16 selftest has no w8a16 key at all
    const int rc = Run(base4 + "--layouts w4a16", log);
    const Container c = ReadContainer(out);
    Check("w4a16 only: no quant.w8a16", rc == 0 && c.ok && !c.header.at("__metadata__").at("quant").contains("w8a16"));
    if (rc != 0) std::printf("%s", ReadText(log).c_str());
  }
  {  // refusals
    int rc = Run(base + "--layouts w8a16 --no-bf16 --quant search", log);
    Check("--quant search with w8a16 only: refused", rc != 0 && Has(ReadText(log), "--quant search"));
    rc = Run(base + "--layouts w8a16 --no-bf16 --hessian-dir nowhere --ldlq .*", log);
    Check("--ldlq with w8a16: refused as deferred", rc != 0 && Has(ReadText(log), "not implemented"));
    rc = Run(base + "--layouts w8a16 --no-bf16 --w8a16-group 96", log);
    Check("--w8a16-group 96: refused", rc != 0 && Has(ReadText(log), "64 or 128"));
    rc = Run(base + "--layouts w4a16 --w8a16-group 64", log);
    Check("--w8a16-group without w8a16: refused", rc != 0 && Has(ReadText(log), "needs w8a16"));
    rc = Run(base + "--layouts w8a16 --no-bf16 --w8a16-group x", log);
    Check("--w8a16-group x: refused", rc != 0);
  }
  std::remove(out.c_str());
  std::remove(log.c_str());
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  TestPlanning();
  TestLayoutSetId();
  TestEmission();
#if defined(R4DX_CONVERT_EXE)
  TestExe(R4DX_CONVERT_FIXTURES_DIR);
#else
  std::printf("SKIP: r4dx-convert is not part of this build\n");
#endif
  if (g_failures != 0) {
    std::printf("FAIL (%d check(s))\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
