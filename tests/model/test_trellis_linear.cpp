// tests/model/test_trellis_linear.cpp -- the trellis runtime v1 (docs/trellis-kernel.md 5.1-5.3,
// 6 `test_trellis_linear`, milestone M4) on the tiny containers tests/convert's
// convert_trellis_import keeps in the build tree (tiny_k4.r4dx: every body linear KB = 4;
// tiny_mix.r4dx: layer 0 at KB = 5, layer 1 at KB = 4; hidden 256, intermediate 1024, the real
// model's head shapes in miniature; FIXTURES_REQUIRED trellis_tiny):
//
//   loader     Container::Load with --layout trellis: every body linear kTrellis with its rate,
//              parts (mlp.gate_up: two, gate's rows then up's), prescale, the words as stored and
//              suh / svh widened from the file's fp16; lm_head w4a16 whether the head is requested
//              as trellis (the body layout, as Model::Load passes it) or w4a16; one ticket slice
//              of N/128 per trellis linear, disjoint, zero after the load and after
//              ZeroTrellisTickets; EpilogueForLayout(kTrellis) == r4dx_epilogue_none.
//   linear     ApplyLinear on qkv, z, out_proj, qg, k, v, o, gate_up (two parts) and down, at
//              M in {1, 16, 17, 64, 65, 130} -- one and several 64-row chunks, the M <= 16 split
//              tuning and the M > 16 unsplit one: every element within 4 bf16 ulp (plus 1e-4 rms
//              of its row) of the fp64 linear of the f16 A that trellis_ref's bit-exact emulation
//              of the input transform gives, from the file's own words, suh and svh.
//   pre        the same call fed an A the test transformed itself (PreQuantizedActivation with
//              transform_id and a part stride that is neither M * K nor 64 * K) is byte-identical,
//              and the A is trellis_ref's bit for bit; a transform_id of another linear, a trellis
//              A handed to a w4a16 linear, a too-short part stride, and a part not 16-byte aligned
//              (data or part stride) throw.
//   row id.    at M = 2, 4, 8 and 16 (MTP and DFlash verify windows, one row tile), every row equals
//              the M = 1 call on that row, byte for byte.
//   M5         ApplyLinear's temporal_weight_loads (NT = 0, gdn.out_proj's) is byte-identical; and
//              SharedTrellisInput's one transform for gdn.in_proj_qkv / in_proj_z and attn.qg / k / v
//              then each linear on its A equals the linear's own call, at M 1, 4, 17 and 64, while a
//              two-part linear, another K or a lone linear is declined.
//   prescale   docs/trellis-kernel.md 4.8's s, which both tiny containers leave at 0: header-patched
//              copies of tiny_k4 with s = 3 container-wide and s = -3 as one linear's own override
//              load it per linear, match the fp64 linear at their s, and -- a power of two being
//              exact -- equal the s = 0 load byte for byte on every row whose A scales exactly.
//   tickets    tickets deliberately left non-zero (a launch that never completed) leave the output
//              unwritten, and after Container::ZeroTrellisTickets the next call is exact again.
//   TP = 2     each rank's ApplyLinear on its LoadShard slice: a column-parallel rank gives the
//              full linear's columns at its rows, a row-parallel rank the linear of its K range
//              (transforms applied to its partial), and the partials add up to the linear.
//   refusals   docs/trellis-kernel.md 2.5, through Container::Load at TP = 1 and TP = 2 (LoadShard)
//              on header-patched copies of tiny_k4 under %TEMP%: a trellis body without --layout
//              trellis, --layout trellis without the block, every unknown format and Hadamard
//              field, a rate the format does not have, unknown and malformed linears fields, a
//              prescale out of range, a reconstruction check that did not pass (pending, FAILED,
//              failed != 0, checked != the HF tensors, no record), a .trellis tensor without an
//              entry, an entry without its tensors, byte sizes / parts that do not match the shape,
//              a rotated trellis container, and a w4a16 head whose unrecorded group hides scales
//              sized for another group than the kernel reads; and, LoadShard only, a part
//              boundary a rank's rows cross. (A rank range that is not whole 128-blocks cannot be reached by a header
//              patch: every tiny shape splits into whole blocks.)
//
// GPU test on HIP device 1 (ctest sets HIP_VISIBLE_DEVICES=1); SKIPs (77) when the tiny containers
// are absent (convert_trellis_import skipped: no golden fixture).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "container.h"
#include "linear.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"
#include "tp/tp_shard.h"
#include "trellis_ref.hpp"  // tests/kernels: the CPU decode, transform and linear references

using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::F16ToFloat;
using r4dx::core::FloatToBf16;
using r4dx::model::ApplyLinear;
using r4dx::model::Container;
using r4dx::model::ContainerLoadOptions;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::PreQuantizedActivation;
using r4dx::model::QuantLinear;
namespace fs = std::filesystem;

namespace {

const std::string kTinyDir = R4DX_TRELLIS_TINY_DIR;

struct Checker {
  int checks = 0;
  int failures = 0;
  void Expect(bool ok, const std::string& what) {
    ++checks;
    if (ok) return;
    ++failures;
    if (failures <= 40) std::fprintf(stderr, "[FAIL] %s\n", what.c_str());
  }
};

template <typename T>
int64_t P(const T* p) {
  return reinterpret_cast<int64_t>(p);
}

// The header of a container as JSON (the file's own bytes).
nlohmann::json ReadHeader(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t len = 0;
  f.read(reinterpret_cast<char*>(&len), 8);
  std::string h(static_cast<size_t>(len), '\0');
  f.read(h.data(), static_cast<std::streamsize>(len));
  if (!f) throw std::runtime_error("short header in " + path);
  return nlohmann::json::parse(h);
}

// Activations as trellis_golden.py's random_x (bf16; row scales 2^-8..2^3, two outliers per row) --
// the same generator as tests/kernels/test_trellis_gemm.cpp's.
std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-8.f, 3.f), od(20.f, 60.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  std::vector<float> row(static_cast<size_t>(K));
  for (int64_t m = 0; m < M; ++m) {
    const float s = std::exp2(ud(rng));
    for (auto& v : row) v = nd(rng) * s;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    for (int64_t k = 0; k < K; ++k) {
      x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
    }
  }
  return x;
}

// ---- the file side of one trellis linear: the reference -----------------------------------------

struct RefLinear {
  std::string base;
  int64_t N = 0, K = 0;
  int KB = 0, parts = 1, prescale = 0;
  int64_t n_split = 0;
  std::vector<uint16_t> q;        // decoded Q [K][N] f16 bits
  std::vector<float> suh, svh;    // [parts][K], [N]
  std::vector<uint16_t> suh_f16, svh_f16;
};

std::vector<uint16_t> F16s(const r4dx_convert::SafetensorsReader& r, const std::string& name) {
  const auto& m = r.Meta(name);
  std::vector<uint16_t> v(static_cast<size_t>((m.end - m.begin) / 2));
  std::memcpy(v.data(), r.Data(name), v.size() * 2);
  return v;
}

RefLinear ReadRef(const r4dx_convert::SafetensorsReader& r, const nlohmann::json& trellis,
                  const std::string& base, int64_t N, int64_t K) {
  RefLinear L;
  L.base = base;
  L.N = N;
  L.K = K;
  const nlohmann::json& e = trellis.at("linears").at(base);
  L.KB = e.at("bits").get<int>();
  L.parts = e.contains("parts") ? static_cast<int>(e.at("parts").size()) : 1;
  L.n_split = L.parts > 1 ? e.at("parts")[0].get<int64_t>() : N;
  L.prescale = e.value("prescale_log2", trellis.at("prescale_log2").get<int>());
  const auto& wm = r.Meta(base + ".trellis.w");
  std::vector<uint32_t> grid(static_cast<size_t>((wm.end - wm.begin) / 4));
  std::memcpy(grid.data(), r.Data(base + ".trellis.w"), grid.size() * 4);
  L.q = trellis_ref::DecodePairGrid(grid, K, N, L.KB);
  L.suh_f16 = F16s(r, base + ".trellis.suh");
  L.svh_f16 = F16s(r, base + ".trellis.svh");
  for (uint16_t h : L.suh_f16) L.suh.push_back(F16ToFloat(h));
  for (uint16_t h : L.svh_f16) L.svh.push_back(F16ToFloat(h));
  return L;
}

// The body linears of the tiny config (2 layers: 0 GDN, 1 full attention) with their [N, K] -- the
// shapes Container::Load asks for.
struct LinearRef {
  const QuantLinear* q;
  std::string base;
  int64_t N, K;
};
std::vector<LinearRef> BodyLinears(const Container& c) {
  const auto& g = c.GlobalConfig();
  const int64_t hidden = g.hidden_size, inter = g.intermediate_size;
  const int64_t attn_out = g.num_attention_heads * g.head_dim;
  const int64_t kv = g.num_key_value_heads * g.head_dim;
  std::vector<LinearRef> out;
  for (int64_t i = 0; i < c.NumLoadedLayers(); ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    const auto& lw = c.Layer(i);
    if (lw.gdn) {
      out.push_back(
          {&lw.gdn->in_proj_qkv, b + "gdn.in_proj_qkv", 2 * g.KeyDim() + g.ValueDim(), hidden});
      out.push_back({&lw.gdn->in_proj_z, b + "gdn.in_proj_z", g.ValueDim(), hidden});
      out.push_back({&lw.gdn->out_proj, b + "gdn.out_proj", hidden, g.ValueDim()});
    } else {
      out.push_back({&lw.attn->qg, b + "attn.qg", 2 * attn_out, hidden});
      out.push_back({&lw.attn->k, b + "attn.k", kv, hidden});
      out.push_back({&lw.attn->v, b + "attn.v", kv, hidden});
      out.push_back({&lw.attn->o, b + "attn.o", hidden, attn_out});
    }
    out.push_back({&lw.mlp.gate_up, b + "mlp.gate_up", 2 * inter, hidden});
    out.push_back({&lw.mlp.down, b + "mlp.down", hidden, inter});
  }
  return out;
}

// ---- loader ------------------------------------------------------------------------------------

void CheckLoaded(const Container& c, const r4dx_convert::SafetensorsReader& r,
                 const nlohmann::json& trellis, const std::string& tag, Checker& ck) {
  ck.Expect(c.HasTrellis() && c.Trellis().linears.size() == trellis.at("linears").size(),
            tag + ": Trellis() holds every linears entry");
  ck.Expect(r4dx::model::EpilogueForLayout(Layout::kTrellis) == r4dx_epilogue_none,
            "EpilogueForLayout(kTrellis) == r4dx_epilogue_none");
  std::vector<std::pair<const uint32_t*, int64_t>> slices;
  for (const LinearRef& l : BodyLinears(c)) {
    const QuantLinear& q = *l.q;
    const std::string w = tag + " " + l.base;
    ck.Expect(q.layout == Layout::kTrellis, w + ": loaded as " + LayoutName(q.layout));
    if (q.layout != Layout::kTrellis) continue;
    const RefLinear ref = ReadRef(r, trellis, l.base, l.N, l.K);
    ck.Expect(q.N == l.N && q.K == l.K, w + ": shape");
    ck.Expect(q.trellis_bits == ref.KB && q.trellis_parts == ref.parts &&
                  q.trellis_prescale_log2 == ref.prescale,
              w + ": bits / parts / prescale");
    ck.Expect(q.trellis_part_n[0] == ref.n_split &&
                  q.trellis_part_n[1] == (ref.parts > 1 ? l.N - ref.n_split : 0),
              w + ": part widths");
    const std::string wname = l.base + ".trellis.w";
    const uint8_t* wfile = r.Data(wname);
    const std::vector<uint8_t> file_w(wfile, wfile + (r.Meta(wname).end - r.Meta(wname).begin));
    std::vector<uint8_t> dev_w(q.trellis_w.bytes());
    R4DX_HIP_CHECK(
        hipMemcpy(dev_w.data(), q.trellis_w.data(), dev_w.size(), hipMemcpyDeviceToHost));
    ck.Expect(dev_w == file_w, w + ": words as stored");
    ck.Expect(q.trellis_suh.CopyToHost() == ref.suh, w + ": suh widened from the file's fp16");
    ck.Expect(q.trellis_svh.CopyToHost() == ref.svh, w + ": svh widened from the file's fp16");
    ck.Expect(q.trellis_tickets != nullptr, w + ": has tickets");
    if (q.trellis_tickets != nullptr) slices.push_back({q.trellis_tickets, l.N / 128});
  }
  // Disjoint slices of one buffer, all zero.
  bool disjoint = true, zero = true;
  for (size_t i = 0; i < slices.size(); ++i) {
    for (size_t j = 0; j < slices.size(); ++j) {
      if (i == j) continue;
      const auto a = slices[i], b = slices[j];
      if (a.first < b.first + b.second && b.first < a.first + a.second) disjoint = false;
    }
    std::vector<uint32_t> t(static_cast<size_t>(slices[i].second));
    R4DX_HIP_CHECK(hipMemcpy(t.data(), slices[i].first, t.size() * 4, hipMemcpyDeviceToHost));
    for (uint32_t x : t) zero = zero && x == 0;
  }
  ck.Expect(disjoint, tag + ": ticket slices are disjoint");
  ck.Expect(zero, tag + ": tickets are zero after the load");
  ck.Expect(c.LmHead().layout == Layout::kW4a16, tag + ": lm_head loads w4a16");
}

// ---- linear ------------------------------------------------------------------------------------

struct Tol {
  size_t bad = 0;
  double worst = 0.0;
};

Tol Compare(const std::vector<uint16_t>& y, const std::vector<double>& ya, int64_t M, int64_t N) {
  Tol t;
  for (int64_t m = 0; m < M; ++m) {
    double ss = 0.0;
    for (int64_t n = 0; n < N; ++n) ss += ya[m * N + n] * ya[m * N + n];
    const double floor = 1e-4 * std::sqrt(ss / static_cast<double>(N));
    for (int64_t n = 0; n < N; ++n) {
      const size_t i = static_cast<size_t>(m * N + n);
      const double lim = 4.0 * trellis_ref::Bf16Ulp(ya[i]) + floor;
      const double err = std::fabs(static_cast<double>(Bf16ToFloat(y[i])) - ya[i]);
      if (!(err <= lim)) ++t.bad;
      t.worst = std::max(t.worst, err / lim);
    }
  }
  return t;
}

constexpr uint16_t kSentinel = 0x7FC1u;  // a bf16 NaN no linear of finite values produces

std::vector<uint16_t> Run(hipStream_t s, r4dx::core::Arena& arena, const QuantLinear& q,
                          const DeviceBuffer<uint16_t>& x, int64_t M,
                          const PreQuantizedActivation* pre = nullptr,
                          bool temporal_weight_loads = false) {
  DeviceBuffer<uint16_t> y(static_cast<size_t>(M * q.N));
  std::vector<uint16_t> sent(y.size(), kSentinel);
  y.CopyFromHost(sent);
  arena.Reset();
  ApplyLinear(s, arena, q, x.data(), y.data(), M, pre, temporal_weight_loads);
  R4DX_HIP_CHECK(hipStreamSynchronize(s));
  return y.CopyToHost();
}

double g_worst = 0.0;

void CheckLinear(hipStream_t s, r4dx::core::Arena& arena, const QuantLinear& q,
                 const RefLinear& ref, const std::string& tag, std::mt19937_64& rng, Checker& ck) {
  const int64_t N = ref.N, K = ref.K;
  for (int64_t M : {int64_t{1}, int64_t{16}, int64_t{17}, int64_t{64}, int64_t{65}, int64_t{130}}) {
    const std::string w = tag + " " + ref.base + " M=" + std::to_string(M);
    const std::vector<uint16_t> x = RandomX(rng, M, K);
    DeviceBuffer<uint16_t> x_d(x.size());
    x_d.CopyFromHost(x);
    const std::vector<uint16_t> y = Run(s, arena, q, x_d, M);

    std::vector<std::vector<uint16_t>> a;
    std::vector<const uint16_t*> a_ptrs;
    for (int p = 0; p < ref.parts; ++p) {
      a.push_back(trellis_ref::TransformInput(x, M, K, ref.suh.data() + p * K, ref.prescale));
    }
    for (const auto& v : a) a_ptrs.push_back(v.data());
    const std::vector<double> ya =
        trellis_ref::LinearFromA(a_ptrs, M, K, N, ref.n_split, ref.q, ref.svh.data(), ref.prescale);
    const Tol t = Compare(y, ya, M, N);
    g_worst = std::max(g_worst, t.worst);
    ck.Expect(t.bad == 0, w + ": " + std::to_string(t.bad) + " element(s) beyond 4 bf16 ulp " +
                              "(worst " + std::to_string(t.worst) + " of the tolerance)");

    // pre: the test's own transform, parts at a stride that is neither M * K nor 64 * K.
    const int64_t stride = M * K + 384;
    DeviceBuffer<uint16_t> a_d(static_cast<size_t>(stride * ref.parts));
    DeviceBuffer<float> suh_d(ref.suh.size());
    suh_d.CopyFromHost(ref.suh);
    const int64_t sp[2] = {P(suh_d.data()), P(suh_d.data() + K)};
    const int64_t op[2] = {P(a_d.data()), P(a_d.data() + stride)};
    r4dx_trellis_input_bf16(P(x_d.data()), M, K, ref.parts, sp, op, ref.prescale, P(s));
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    const std::vector<uint16_t> a_h = a_d.CopyToHost();
    bool a_ok = true;
    for (int p = 0; p < ref.parts; ++p) {
      a_ok = a_ok && std::memcmp(a_h.data() + p * stride, a[p].data(), a[p].size() * 2) == 0;
    }
    ck.Expect(a_ok, w + ": the transform's A is trellis_ref's bit for bit");
    PreQuantizedActivation pre;
    pre.data = a_d.data();
    pre.transform_id = q.trellis_suh.data();
    pre.part_stride = stride;
    ck.Expect(Run(s, arena, q, x_d, M, &pre) == y,
              w + ": pre (part stride " + std::to_string(stride) + ") byte-identical");
    // temporal_weight_loads (M5, gdn.out_proj's): NT = 0 is a cache hint, the same bytes.
    ck.Expect(Run(s, arena, q, x_d, M, nullptr, /*temporal_weight_loads=*/true) == y,
              w + ": temporal weight loads byte-identical");
  }

  // Row identity: every row of an M <= 16 call is the M = 1 call on that row -- at the MTP verify
  // window (--mtp 3: M = 4), the DFlash one (k = 7: M = 8), M = 2 and the whole row tile.
  for (int64_t M : {int64_t{2}, int64_t{4}, int64_t{8}, int64_t{16}}) {
    const std::vector<uint16_t> x = RandomX(rng, M, K);
    DeviceBuffer<uint16_t> x_d(x.size());
    x_d.CopyFromHost(x);
    const std::vector<uint16_t> y = Run(s, arena, q, x_d, M);
    bool same = true;
    for (int64_t m = 0; m < M; ++m) {
      DeviceBuffer<uint16_t> row(static_cast<size_t>(K));
      R4DX_HIP_CHECK(hipMemcpy(row.data(), x_d.data() + m * K, K * 2, hipMemcpyDeviceToDevice));
      const std::vector<uint16_t> y1 = Run(s, arena, q, row, 1);
      same = same && std::memcmp(y1.data(), y.data() + m * N, N * 2) == 0;
    }
    ck.Expect(same, tag + " " + ref.base + " M=" + std::to_string(M) +
                        ": every row equals the M = 1 call (row identity)");
  }
}

// M5 (docs/trellis-kernel.md 4.9 / 5.4): SharedTrellisInput's one transform for the linears that read
// one activation -- gdn.in_proj_qkv / in_proj_z, attn.qg / k / v -- then each linear's ApplyLinear on
// its A gives the bytes of the linear's own call; and it declines (false) a group it cannot serve.
void CheckSharedInput(hipStream_t s, r4dx::core::Arena& arena, const Container& c,
                      const std::string& tag, std::mt19937_64& rng, Checker& ck) {
  for (int64_t i = 0; i < c.NumLoadedLayers(); ++i) {
    const auto& lw = c.Layer(i);
    std::vector<const QuantLinear*> ws;
    if (lw.gdn) ws = {&lw.gdn->in_proj_qkv, &lw.gdn->in_proj_z};
    else ws = {&lw.attn->qg, &lw.attn->k, &lw.attn->v};
    const int n = static_cast<int>(ws.size());
    const int64_t K = ws[0]->K;
    for (int64_t M : {int64_t{1}, int64_t{4}, int64_t{17}, int64_t{64}}) {
      const std::string w = tag + " layer " + std::to_string(i) + " shared input M=" + std::to_string(M);
      const std::vector<uint16_t> x = RandomX(rng, M, K);
      DeviceBuffer<uint16_t> x_d(x.size());
      x_d.CopyFromHost(x);
      std::vector<std::vector<uint16_t>> own;
      for (const QuantLinear* q : ws) own.push_back(Run(s, arena, *q, x_d, M));
      arena.Reset();
      PreQuantizedActivation pre[3];
      const bool ok = r4dx::model::SharedTrellisInput(s, arena, x_d.data(), M, ws.data(), n, pre);
      ck.Expect(ok, w + ": served");
      if (!ok) continue;
      std::vector<DeviceBuffer<uint16_t>> ys;
      for (int j = 0; j < n; ++j) {
        ys.emplace_back(static_cast<size_t>(M * ws[j]->N));
        ApplyLinear(s, arena, *ws[j], x_d.data(), ys.back().data(), M, &pre[j]);
      }
      R4DX_HIP_CHECK(hipStreamSynchronize(s));
      for (int j = 0; j < n; ++j) {
        ck.Expect(ys[j].CopyToHost() == own[static_cast<size_t>(j)],
                  w + ": linear " + std::to_string(j) + " byte-identical to its own call");
      }
    }
    // Declined: a two-part linear (gate_up), a different K (out_proj / o, mlp.down), one linear.
    PreQuantizedActivation pre[3];
    const QuantLinear* two_part[2] = {ws[0], &lw.mlp.gate_up};
    const QuantLinear* other_k[2] = {ws[0], lw.gdn ? &lw.gdn->out_proj : &lw.attn->o};
    arena.Reset();
    DeviceBuffer<uint16_t> x_d(static_cast<size_t>(K));
    ck.Expect(!r4dx::model::SharedTrellisInput(s, arena, x_d.data(), 1, two_part, 2, pre),
              tag + " layer " + std::to_string(i) + ": a two-part linear is declined");
    ck.Expect(!r4dx::model::SharedTrellisInput(s, arena, x_d.data(), 1, other_k, 2, pre),
              tag + " layer " + std::to_string(i) + ": a linear of another K is declined");
    ck.Expect(!r4dx::model::SharedTrellisInput(s, arena, x_d.data(), 1, ws.data(), 1, pre),
              tag + " layer " + std::to_string(i) + ": a single linear is declined");
  }
}

template <class Fn>
bool Throws(Fn&& fn, const std::string& needle = "") {
  try {
    fn();
  } catch (const std::exception& e) {
    return needle.empty() || std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

void CheckPreRefusals(hipStream_t s, r4dx::core::Arena& arena, const Container& c, Checker& ck) {
  const auto& lw0 = c.Layer(0);
  const QuantLinear& qkv = lw0.gdn->in_proj_qkv;
  const QuantLinear& gate_up = lw0.mlp.gate_up;
  const int64_t K = qkv.K;
  DeviceBuffer<uint16_t> x(static_cast<size_t>(4 * K)), a(static_cast<size_t>(8 * K)),
      y(static_cast<size_t>(4 * gate_up.N));
  x.Zero();
  a.Zero();
  PreQuantizedActivation pre;
  pre.data = a.data();
  pre.transform_id = gate_up.trellis_suh.data();  // not qkv's
  pre.part_stride = 4 * K;
  ck.Expect(Throws([&] { ApplyLinear(s, arena, qkv, x.data(), y.data(), 4, &pre); },
                   "not this linear's"),
            "ApplyLinear refuses another linear's transform_id");
  const QuantLinear& head = c.LmHead();  // w4a16
  pre.transform_id = gate_up.trellis_suh.data();
  DeviceBuffer<uint16_t> yh(static_cast<size_t>(4 * head.N));
  ck.Expect(Throws([&] { ApplyLinear(s, arena, head, x.data(), yh.data(), 4, &pre); },
                   "not this linear's"),
            "ApplyLinear refuses a trellis A for a w4a16 linear");
  pre.transform_id = gate_up.trellis_suh.data();
  pre.part_stride = 3 * K;  // < M * K for M = 4 with two parts
  ck.Expect(Throws([&] { ApplyLinear(s, arena, gate_up, x.data(), y.data(), 4, &pre); },
                   "part_stride"),
            "ApplyLinear refuses a part stride shorter than M * K");
  // The GEMM's 8-byte A loads read a misaligned part wrong, silently: 16-byte aligned parts only.
  pre.transform_id = qkv.trellis_suh.data();
  pre.data = a.data() + 1;  // 2 bytes past a 16-byte boundary
  ck.Expect(Throws([&] { ApplyLinear(s, arena, qkv, x.data(), y.data(), 4, &pre); }, "aligned"),
            "ApplyLinear refuses a trellis A that is not 16-byte aligned");
  pre.transform_id = gate_up.trellis_suh.data();
  pre.data = a.data();
  pre.part_stride = 4 * K + 4;  // long enough, but part 1 starts 8 bytes off
  ck.Expect(Throws([&] { ApplyLinear(s, arena, gate_up, x.data(), y.data(), 4, &pre); },
                   "multiple of 8"),
            "ApplyLinear refuses a part stride that misaligns part 1");
  R4DX_HIP_CHECK(hipStreamSynchronize(s));
}

// Tickets left non-zero by a launch that never completed: no block of a split group recognizes
// itself as the last, so the output stays unwritten; Container::ZeroTrellisTickets restores it.
void CheckTicketReset(hipStream_t s, r4dx::core::Arena& arena, Container& c, std::mt19937_64& rng,
                      Checker& ck) {
  const QuantLinear& k = c.Layer(1).attn->k;  // M = 1: the fallback's SKG 4, a split group
  const r4dx::model::LinearTuning t =
      r4dx::model::PickTuning(Layout::kTrellis, k.N, k.K, 1, k.trellis_bits);
  ck.Expect(t.SKG > 1 || t.WV * t.NPW * 32 < 128, "attn.k's M = 1 tuning splits its 128-groups");
  const std::vector<uint16_t> xh = RandomX(rng, 1, k.K);
  DeviceBuffer<uint16_t> x(xh.size());
  x.CopyFromHost(xh);
  const std::vector<uint16_t> good = Run(s, arena, k, x, 1);
  R4DX_HIP_CHECK(hipMemsetAsync(k.trellis_tickets, 0x01, static_cast<size_t>(k.N / 128) * 4, s));
  const std::vector<uint16_t> bad = Run(s, arena, k, x, 1);
  bool untouched = true;
  for (uint16_t v : bad) untouched = untouched && v == kSentinel;
  ck.Expect(untouched, "non-zero tickets: the split groups are never finished (output unwritten)");
  c.ZeroTrellisTickets(s);
  R4DX_HIP_CHECK(hipStreamSynchronize(s));
  bool zero = true;
  for (const LinearRef& l : BodyLinears(c)) {
    std::vector<uint32_t> tk(static_cast<size_t>(l.q->N / 128));
    R4DX_HIP_CHECK(
        hipMemcpy(tk.data(), l.q->trellis_tickets, tk.size() * 4, hipMemcpyDeviceToHost));
    for (uint32_t v : tk) zero = zero && v == 0;
  }
  ck.Expect(zero, "ZeroTrellisTickets zeroes every linear's tickets");
  ck.Expect(Run(s, arena, k, x, 1) == good, "after ZeroTrellisTickets the call is exact again");
}

// ---- TP = 2 ------------------------------------------------------------------------------------
// docs/trellis-kernel.md 4.5 ("row-parallel linears under TP") and 5.5: each rank's ApplyLinear on
// its own shard (LoadShard), M = 5. A column-parallel rank's output column j is the full linear's
// column at the rank's row j, within the 4-ulp tolerance of the full fp64 reference. A row-parallel
// rank computes the whole linear of its K range -- the input transform per 128-block, the output
// transform (FWHT, svh) on its partial -- within the tolerance of its own fp64 reference, and the
// two ranks' references add up to the full one: the transform is linear, so the all-reduce of the
// transformed partials is the linear.
void CheckTpRanks(hipStream_t s, r4dx::core::Arena& arena, const std::string& path,
                  const r4dx_convert::SafetensorsReader& r, const nlohmann::json& trellis,
                  const Container& full, std::mt19937_64& rng, Checker& ck) {
  constexpr int kWorld = 2;
  constexpr int64_t M = 5;
  std::unique_ptr<Container> rank[kWorld];
  for (int w = 0; w < kWorld; ++w) {
    ContainerLoadOptions o;
    o.layout = o.lm_head_layout = o.mtp_head_layout = Layout::kTrellis;
    o.tp_world = kWorld;
    o.tp_rank = w;
    o.embed_device_resident_decided = 0;
    rank[w] = std::make_unique<Container>(Container::Load(path, o));
  }
  const auto& g = full.GlobalConfig();
  const std::vector<LinearRef> lins = BodyLinears(full);
  std::vector<LinearRef> rank_lins[kWorld] = {BodyLinears(*rank[0]), BodyLinears(*rank[1])};
  for (size_t li = 0; li < lins.size(); ++li) {
    const LinearRef& l = lins[li];
    const RefLinear ref = ReadRef(r, trellis, l.base, l.N, l.K);
    const std::string tag = "TP=2 " + l.base;
    const std::vector<uint16_t> x = RandomX(rng, M, l.K);
    std::vector<std::vector<uint16_t>> a;
    std::vector<const uint16_t*> a_ptrs;
    for (int p = 0; p < ref.parts; ++p) {
      a.push_back(trellis_ref::TransformInput(x, M, l.K, ref.suh.data() + p * l.K, ref.prescale));
    }
    for (const auto& v : a) a_ptrs.push_back(v.data());
    const std::vector<double> ya = trellis_ref::LinearFromA(a_ptrs, M, l.K, l.N, ref.n_split, ref.q,
                                                            ref.svh.data(), ref.prescale);
    const r4dx::model::tp::ShardRule rule = r4dx::model::tp::RuleFor(l.base, g);
    std::vector<double> sum(ya.size(), 0.0);
    for (int w = 0; w < kWorld; ++w) {
      const QuantLinear& q = *rank_lins[w][li].q;
      const std::string wt = tag + " rank " + std::to_string(w);
      if (rule.split == r4dx::model::tp::Split::kRows) {
        std::vector<int64_t> row_of;
        for (const auto& seg : rule.segments) {
          for (int64_t i = 0; i < seg.rows / kWorld; ++i) {
            row_of.push_back(seg.begin + w * (seg.rows / kWorld) + i);
          }
        }
        DeviceBuffer<uint16_t> x_d(x.size());
        x_d.CopyFromHost(x);
        const std::vector<uint16_t> y = Run(s, arena, q, x_d, M);
        std::vector<double> want(static_cast<size_t>(M * q.N));
        for (int64_t m = 0; m < M; ++m)
          for (int64_t j = 0; j < q.N; ++j)
            want[static_cast<size_t>(m * q.N + j)] = ya[static_cast<size_t>(m * l.N + row_of[j])];
        const Tol t = Compare(y, want, M, q.N);
        ck.Expect(q.N == static_cast<int64_t>(row_of.size()) && t.bad == 0,
                  wt + ": its columns are the full linear's rows (" + std::to_string(t.bad) +
                      " beyond tolerance)");
      } else {
        const int64_t kc = l.K / kWorld, k0 = w * kc;
        std::vector<uint16_t> xr(static_cast<size_t>(M * kc));
        for (int64_t m = 0; m < M; ++m)
          std::memcpy(&xr[static_cast<size_t>(m * kc)], &x[static_cast<size_t>(m * l.K + k0)],
                      static_cast<size_t>(kc) * 2);
        const std::vector<uint16_t> ar =
            trellis_ref::TransformInput(xr, M, kc, ref.suh.data() + k0, ref.prescale);
        const std::vector<uint16_t> qr(ref.q.begin() + k0 * l.N, ref.q.begin() + (k0 + kc) * l.N);
        const std::vector<double> yr = trellis_ref::LinearFromA({ar.data()}, M, kc, l.N, l.N, qr,
                                                                ref.svh.data(), ref.prescale);
        for (size_t i = 0; i < sum.size(); ++i) sum[i] += yr[i];
        DeviceBuffer<uint16_t> x_d(xr.size());
        x_d.CopyFromHost(xr);
        const std::vector<uint16_t> y = Run(s, arena, q, x_d, M);
        const Tol t = Compare(y, yr, M, l.N);
        ck.Expect(q.K == kc && t.bad == 0, wt + ": the linear of its K range (" +
                                               std::to_string(t.bad) + " beyond tolerance)");
      }
    }
    if (rule.split == r4dx::model::tp::Split::kCols) {
      double worst = 0.0, scale = 0.0;
      for (size_t i = 0; i < ya.size(); ++i) {
        worst = std::max(worst, std::fabs(sum[i] - ya[i]));
        scale = std::max(scale, std::fabs(ya[i]));
      }
      ck.Expect(worst <= 1e-9 * scale, tag + ": the ranks' partials add up to the linear");
    }
  }
}

// ---- refusals -----------------------------------------------------------------------------------

// A copy of `src` whose header is `header` (the data section as is; offsets are relative to it).
void WritePatched(const std::string& src, const fs::path& dst, const nlohmann::json& header) {
  std::ifstream in(src, std::ios::binary);
  uint64_t len = 0;
  in.read(reinterpret_cast<char*>(&len), 8);
  in.seekg(static_cast<std::streamoff>(8 + len));
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::string h = header.dump();
  while (h.size() % 8 != 0) h.push_back(' ');
  const uint64_t hl = h.size();
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(&hl), 8);
  out.write(h.data(), static_cast<std::streamsize>(h.size()));
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!out) throw std::runtime_error("cannot write " + dst.u8string());
}

bool LoadThrows(const std::string& path, Layout layout, int world, const std::string& needle,
                std::string* got) {
  try {
    ContainerLoadOptions o;
    o.layout = o.lm_head_layout = o.mtp_head_layout = layout;
    o.embed_device_resident = false;
    if (world > 1) {
      o.tp_world = world;
      o.tp_rank = 0;
      o.embed_device_resident_decided = 0;
    }
    (void)Container::Load(path, o);
  } catch (const std::exception& e) {
    *got = e.what();
    return needle.empty() || got->find(needle) != std::string::npos;
  }
  *got = "(did not throw)";
  return false;
}

void CheckRefusals(const std::string& tiny, Checker& ck) {
  const nlohmann::json base = ReadHeader(tiny);
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir =
      fs::temp_directory_path() / ("r4dx_test_trellis_linear_" + std::to_string(stamp));
  fs::create_directories(dir);
  struct Case {
    const char* what;
    std::function<void(nlohmann::json&)> patch;
    Layout layout;
    const char* needle;
    bool shard_only = false;  // refused by LoadShard only; the TP = 1 load of it must succeed
  };
  const std::string lin = "text.layers.0.mlp.gate_up";
  const auto tq = [](nlohmann::json& h) -> nlohmann::json& {
    return h["__metadata__"]["quant"]["trellis"];
  };
  const auto verify = [](nlohmann::json& h) -> nlohmann::json& {
    return h["__metadata__"]["r4dx_convert_run"]["trellis"]["verify"];
  };
  const std::vector<Case> cases = {
      {"a trellis body loaded as w4a16", [](nlohmann::json&) {}, Layout::kW4a16,
       "run with --layout trellis"},
      {"a trellis body loaded as bf16", [](nlohmann::json&) {}, Layout::kBf16,
       "run with --layout trellis"},
      {"--layout trellis without the block",
       [&](nlohmann::json& h) { h["__metadata__"]["quant"].erase("trellis"); }, Layout::kTrellis,
       "needs a trellis container"},
      {"trellis tensors without the block (w4a16)",
       [&](nlohmann::json& h) { h["__metadata__"]["quant"].erase("trellis"); }, Layout::kW4a16,
       "no __metadata__.quant.trellis"},
      {"format", [&](nlohmann::json& h) { tq(h)["format"] = "r4dx-trellis2"; }, Layout::kTrellis,
       "\"format\""},
      {"version", [&](nlohmann::json& h) { tq(h)["version"] = 2; }, Layout::kTrellis,
       "\"version\""},
      {"codebook", [&](nlohmann::json& h) { tq(h)["codebook"] = "3inst"; }, Layout::kTrellis,
       "\"codebook\""},
      {"codebook constant",
       [&](nlohmann::json& h) { tq(h)["codebook_consts"]["mult"] = "0x83dcd12e"; },
       Layout::kTrellis, "codebook_consts.mult"},
      {"state bits", [&](nlohmann::json& h) { tq(h)["state_bits"] = 12; }, Layout::kTrellis,
       "state_bits"},
      {"tail biting", [&](nlohmann::json& h) { tq(h)["tail_biting"] = false; }, Layout::kTrellis,
       "tail_biting"},
      {"position order", [&](nlohmann::json& h) { tq(h)["position_order"] = "row-major"; },
       Layout::kTrellis, "position_order"},
      {"bitstream", [&](nlohmann::json& h) { tq(h)["bitstream"] = "ring-u32-lsb-first"; },
       Layout::kTrellis, "bitstream"},
      {"tile grid", [&](nlohmann::json& h) { tq(h)["tile_grid"] = "k-major"; }, Layout::kTrellis,
       "tile_grid"},
      {"hadamard block", [&](nlohmann::json& h) { tq(h)["hadamard"]["block"] = 64; },
       Layout::kTrellis, "hadamard.block"},
      {"hadamard order", [&](nlohmann::json& h) { tq(h)["hadamard"]["order"] = "walsh"; },
       Layout::kTrellis, "hadamard.order"},
      {"hadamard scale", [&](nlohmann::json& h) { tq(h)["hadamard"]["scale"] = "1/128"; },
       Layout::kTrellis, "hadamard.scale"},
      {"hadamard input", [&](nlohmann::json& h) { tq(h)["hadamard"]["input"] = "H then x*suh"; },
       Layout::kTrellis, "hadamard.input"},
      {"hadamard output",
       [&](nlohmann::json& h) { tq(h)["hadamard"]["output"] = "*svh then H"; }, Layout::kTrellis,
       "hadamard.output"},
      {"prescale out of range", [&](nlohmann::json& h) { tq(h)["prescale_log2"] = 17; },
       Layout::kTrellis, "prescale_log2"},
      {"a linear's prescale out of range",
       [&](nlohmann::json& h) { tq(h)["linears"][lin]["prescale_log2"] = -17; }, Layout::kTrellis,
       "prescale_log2"},
      {"rate 3", [&](nlohmann::json& h) { tq(h)["linears"][lin]["bits"] = 3; }, Layout::kTrellis,
       "not a rate of this format"},
      {"rate 3.5", [&](nlohmann::json& h) { tq(h)["linears"][lin]["bits"] = 3.5; },
       Layout::kTrellis, "bits"},
      {"unknown linears field", [&](nlohmann::json& h) { tq(h)["linears"][lin]["group"] = 64; },
       Layout::kTrellis, "not a field this runtime knows"},
      {"parts not 128-blocks",
       [&](nlohmann::json& h) { tq(h)["linears"][lin]["parts"] = {1000, 1048}; }, Layout::kTrellis,
       "multiples of 128"},
      {"parts not summing to N",
       [&](nlohmann::json& h) { tq(h)["linears"][lin]["parts"] = {1024, 896}; }, Layout::kTrellis,
       "parts sum to"},
      {"byte size (rate 5 on KB 4 words)",
       [&](nlohmann::json& h) { tq(h)["linears"]["text.layers.0.mlp.down"]["bits"] = 5; },
       Layout::kTrellis, "'.trellis.w' is"},
      {"a .trellis tensor without an entry",
       [&](nlohmann::json& h) {
         tq(h)["linears"].erase("text.layers.1.attn.k");
         verify(h)["checked"] = 12;  // so the verify record still matches the linears
       },
       Layout::kTrellis, "has no 'text.layers.1.attn.k'"},
      {"an entry without its tensors",
       [&](nlohmann::json& h) {
         h["text.layers.1.attn.v.trellis_svh_renamed"] = h["text.layers.1.attn.v.trellis.svh"];
         h.erase("text.layers.1.attn.v.trellis.svh");
       },
       Layout::kTrellis, "has no 'text.layers.1.attn.v.trellis.svh'"},
      {"verify pending",
       [&](nlohmann::json& h) { verify(h)["result"] = "pending: the reconstruction check"; },
       Layout::kTrellis, "did not pass"},
      {"verify FAILED", [&](nlohmann::json& h) { verify(h)["result"] = "FAILED 1/13 HF tensors"; },
       Layout::kTrellis, "did not pass"},
      {"verify not run",
       [&](nlohmann::json& h) { verify(h)["result"] = "not run (--trellis-verify none)"; },
       Layout::kTrellis, "did not pass"},
      {"verify failed != 0", [&](nlohmann::json& h) { verify(h)["failed"] = 1; }, Layout::kTrellis,
       "verify.failed"},
      {"verify checked != HF tensors", [&](nlohmann::json& h) { verify(h)["checked"] = 12; },
       Layout::kTrellis, "verify.checked"},
      {"no verify record",
       [&](nlohmann::json& h) { h["__metadata__"]["r4dx_convert_run"].erase("trellis"); },
       Layout::kTrellis, "no r4dx_convert_run.trellis.verify"},
      // docs/trellis-kernel.md 3.4: a trellis body is never rotated. The rotation parse runs first
      // and implements hidden 5120 only, so the patch also says the model is that wide (nothing is
      // read against it before the refusal).
      {"a rotated trellis container",
       [&](nlohmann::json& h) {
         nlohmann::json& mc = h["__metadata__"]["model_config"];
         (mc.contains("text_config") ? mc["text_config"] : mc)["hidden_size"] = 5120;
         h["__metadata__"]["rotation"] = {
             {"kind", "q2a"}, {"seed", 1}, {"hidden", 5120}, {"block", 1024}};
       },
       Layout::kTrellis, "mutually exclusive"},
      // The w4a16 lm_head of a container that does not record quant.w4a16.group (so it parses to
      // the historical default 128 and CheckW4a16Group is skipped) with scales half the size the
      // build's group needs -- on a group-64 build, exactly a group-128 wsz. The kernel reads the
      // build's group, so the size check must be made at THAT group (W4a16LoadGroups::KernelGroup),
      // not at the unrecorded 128, or N*K/64 dwords are read from an N*K/128 buffer.
      {"unrecorded w4a16 group, half-size scales",
       [&](nlohmann::json& h) {
         h["__metadata__"]["quant"]["w4a16"].erase("group");
         nlohmann::json& t = h["lm_head.w4a16.wsz"];
         const uint64_t begin = t["data_offsets"][0].get<uint64_t>();
         const uint64_t end = t["data_offsets"][1].get<uint64_t>();
         const uint64_t half = (end - begin) / 2;
         t["data_offsets"][1] = begin + half;
         t["shape"] = {half / 4, 4};
       },
       Layout::kTrellis, "lm_head.w4a16.wsz' is "},
      // docs/trellis-kernel.md 2.4: gate_up's TP = 2 rows are each part's halves, so parts of
      // 384 + 1664 put rank 0's gate rows [0, 512) across the boundary. Legal at TP = 1.
      {"a part boundary a rank's rows cross",
       [&](nlohmann::json& h) { tq(h)["linears"][lin]["parts"] = {384, 1664}; }, Layout::kTrellis,
       "cross a part boundary", /*shard_only=*/true},
  };
  int n = 0;
  for (const Case& c : cases) {
    nlohmann::json h = base;
    c.patch(h);
    const fs::path p = dir / ("case" + std::to_string(n++) + ".r4dx");
    WritePatched(tiny, p, h);
    for (int world : {1, 2}) {
      std::string got;
      if (c.shard_only && world == 1) {
        ck.Expect(!LoadThrows(p.u8string(), c.layout, world, "", &got) &&
                      got == "(did not throw)",
                  std::string("loads at TP = 1 (Load): ") + c.what + " -- got: " +
                      got.substr(0, 240));
        continue;
      }
      const bool ok = LoadThrows(p.u8string(), c.layout, world, c.needle, &got);
      ck.Expect(ok, std::string("refusal (") + (world == 1 ? "Load" : "LoadShard") + "): " +
                        c.what + " -- got: " + got.substr(0, 240));
    }
  }
  // And the unpatched copy loads (the patching itself is not what refuses).
  {
    const fs::path p = dir / "unpatched.r4dx";
    WritePatched(tiny, p, base);
    for (int world : {1, 2}) {
      std::string got;
      ck.Expect(!LoadThrows(p.u8string(), Layout::kTrellis, world, "", &got) &&
                    got == "(did not throw)",
                std::string("the unpatched copy loads (") + (world == 1 ? "Load" : "LoadShard") +
                    ")");
    }
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- prescale -----------------------------------------------------------------------------------
// docs/trellis-kernel.md 4.8: the transform writes A = f16(H(x * suh) * 2^s / sqrt(128)) and the
// GEMM's out_scale = 2^-s / sqrt(128) undoes the 2^s. Both tiny containers carry s = 0, which every
// other check runs at, so a mismatch between the two sides would pass them all: a header-patched copy
// of tiny_k4 sets s = 3 container-wide and gives layer 0's mlp.down its own s = -3 (the entry's
// override wins). Every linear loads its s and matches the fp64 linear of trellis_ref's A at that s
// (CheckLinear, the whole suite). And a power of two is exact in the transform's fp32 product, in
// every fp32 sum and in the epilogue, so a row whose A is exactly 2^s times the s = 0 row's A (no
// element entering or leaving the f16 subnormal range, none overflowing) equals the s = 0 load's row
// byte for byte (CheckPrescaleIdentity, M = 16).

// The rows of an M = 16 call compared with the s = 0 linear `q0` where the A scales exactly; returns
// how many rows that was.
int CheckPrescaleIdentity(hipStream_t s, r4dx::core::Arena& arena, const QuantLinear& qs,
                          const QuantLinear& q0, const RefLinear& ref, const std::string& w,
                          std::mt19937_64& rng, Checker& ck) {
  constexpr int64_t M = 16;
  const int64_t N = ref.N, K = ref.K;
  const std::vector<uint16_t> x = RandomX(rng, M, K);
  DeviceBuffer<uint16_t> x_d(x.size());
  x_d.CopyFromHost(x);
  const std::vector<uint16_t> ys = Run(s, arena, qs, x_d, M);
  const std::vector<uint16_t> y0 = Run(s, arena, q0, x_d, M);
  std::vector<bool> exact(static_cast<size_t>(M), true);
  for (int p = 0; p < ref.parts; ++p) {
    const std::vector<uint16_t> as =
        trellis_ref::TransformInput(x, M, K, ref.suh.data() + p * K, ref.prescale);
    const std::vector<uint16_t> a0 = trellis_ref::TransformInput(x, M, K, ref.suh.data() + p * K, 0);
    for (int64_t i = 0; i < M * K; ++i) {
      const float vs = F16ToFloat(as[static_cast<size_t>(i)]);
      const float v0 = F16ToFloat(a0[static_cast<size_t>(i)]);
      if (!(std::isfinite(vs) && vs == std::ldexp(v0, ref.prescale))) {
        exact[static_cast<size_t>(i / K)] = false;
      }
    }
  }
  int compared = 0;
  bool same = true;
  for (int64_t m = 0; m < M; ++m) {
    if (!exact[static_cast<size_t>(m)]) continue;
    ++compared;
    same = same && std::memcmp(ys.data() + m * N, y0.data() + m * N, N * 2) == 0;
  }
  ck.Expect(same, w + ": the rows whose A scales exactly (" + std::to_string(compared) +
                      " of 16) equal the s = 0 load's");
  return compared;
}

void CheckPrescale(hipStream_t s, r4dx::core::Arena& arena, const std::string& tiny,
                   const r4dx_convert::SafetensorsReader& r, const Container& c0,
                   std::mt19937_64& rng, Checker& ck) {
  constexpr int kContainerS = 3, kOverrideS = -3;
  const std::string override_base = "text.layers.0.mlp.down";
  nlohmann::json h = ReadHeader(tiny);
  nlohmann::json& tq = h["__metadata__"]["quant"]["trellis"];
  tq["prescale_log2"] = kContainerS;
  tq["linears"][override_base]["prescale_log2"] = kOverrideS;
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir =
      fs::temp_directory_path() / ("r4dx_test_trellis_prescale_" + std::to_string(stamp));
  fs::create_directories(dir);
  const fs::path p = dir / "prescale.r4dx";
  WritePatched(tiny, p, h);
  int compared = 0;
  {
    const Container c = Container::Load(p.u8string(), Layout::kTrellis, Layout::kTrellis);
    const std::vector<LinearRef> lins = BodyLinears(c), lins0 = BodyLinears(c0);
    for (size_t i = 0; i < lins.size(); ++i) {
      const LinearRef& l = lins[i];
      const int want = l.base == override_base ? kOverrideS : kContainerS;
      const std::string w = "prescale " + l.base;
      ck.Expect(l.q->trellis_prescale_log2 == want, w + ": loads s = " + std::to_string(want));
      const RefLinear ref = ReadRef(r, tq, l.base, l.N, l.K);
      CheckLinear(s, arena, *l.q, ref, "prescale s=" + std::to_string(want), rng, ck);
      compared += CheckPrescaleIdentity(s, arena, *l.q, *lins0[i].q, ref, w, rng, ck);
    }
  }
  ck.Expect(compared > 0, "prescale: some rows' A scales exactly (" + std::to_string(compared) +
                              " rows compared with s = 0)");
  std::printf("test_trellis_linear: prescale s = %d / %d checked, %d rows byte-compared with s = 0\n",
              kContainerS, kOverrideS, compared);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

int RunTest() {
  const std::string k4 = kTinyDir + "/tiny_k4.r4dx", mix = kTinyDir + "/tiny_mix.r4dx";
  if (!r4dx_test::FileExists(k4)) return r4dx_test::SkipMissing(k4);
  if (!r4dx_test::FileExists(mix)) return r4dx_test::SkipMissing(mix);
  R4DX_HIP_CHECK(hipSetDevice(0));
  Checker ck;
  std::mt19937_64 rng(20260927);
  r4dx::core::Stream stream;
  r4dx::core::Arena arena;
  arena.Reserve(64ull << 20);

  for (const std::string& path : {k4, mix}) {
    const std::string tag = fs::path(path).filename().u8string();
    const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(path));
    const nlohmann::json trellis = ReadHeader(path).at("__metadata__").at("quant").at("trellis");
    // The heads requested as the body layout, as Model::Load does.
    Container c = Container::Load(path, Layout::kTrellis, Layout::kTrellis);
    CheckLoaded(c, r, trellis, tag, ck);
    for (const LinearRef& l : BodyLinears(c)) {
      if (l.q->layout != Layout::kTrellis) continue;
      CheckLinear(stream.get(), arena, *l.q, ReadRef(r, trellis, l.base, l.N, l.K), tag, rng, ck);
    }
    CheckTpRanks(stream.get(), arena, path, r, trellis, c, rng, ck);
    CheckSharedInput(stream.get(), arena, c, tag, rng, ck);
    if (path == k4) {
      CheckPreRefusals(stream.get(), arena, c, ck);
      CheckTicketReset(stream.get(), arena, c, rng, ck);
      CheckPrescale(stream.get(), arena, path, r, c, rng, ck);
      // The heads requested as w4a16 explicitly: the same load.
      ContainerLoadOptions o;
      o.layout = Layout::kTrellis;
      o.lm_head_layout = o.mtp_head_layout = Layout::kW4a16;
      const Container c2 = Container::Load(path, o);
      ck.Expect(c2.LmHead().layout == Layout::kW4a16, tag + ": lm_head w4a16 when asked for w4a16");
    }
    std::printf("test_trellis_linear: %s checked (%d failure(s) so far)\n", tag.c_str(),
                ck.failures);
    std::fflush(stdout);
  }
  CheckRefusals(k4, ck);
  std::printf("test_trellis_linear: worst element %.3f of the 4-ulp tolerance\n", g_worst);
  std::printf("test_trellis_linear: %d/%d checks passed\n", ck.checks - ck.failures, ck.checks);
  return ck.failures == 0 ? 0 : 1;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_trellis_linear", [] { return RunTest(); }); }
