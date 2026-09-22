// r4dx-convert --quant search (src/convert/include/r4dx_convert/quant_search.hpp): the three
// properties the search has to have.
//
//   (i)   never worse than RTN, per (row, group), for every layout and both weightings. This is a
//         STRUCTURAL claim (the RTN grid is candidate 0 and later candidates must win strictly),
//         so the test checks it group by group rather than in aggregate.
//   (ii)  an outlier-dominated group -- one big value among the rest, all small, which drags the
//         min/max grid so wide that the whole bulk collapses onto two or three codes -- is
//         measurably recovered, especially once an imatrix says the outlier channel barely matters.
//   (iii) bit-exact agreement with the Python reference (tools/convert_ref/w4_ref.py's
//         quantize_*_search, mxfp4_ref.py's quantize_search) on a random matrix, via the fixtures
//         tools/convert_ref/gen_fixtures.py writes. Same gate shape as test_pack_bytes.cpp.
//
// All three are checked at BOTH int4 group sizes r4d_gemm_w4a16_nt_m64 can be built with
// (R4DX_W4A16_GROUP = 64 or 128, root CMakeLists.txt), so the default build gates the group-64
// path as well as its own.
//
// R4DX_CONVERT_FIXTURES_DIR is injected by tests/convert/CMakeLists.txt as an absolute path.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace {

using namespace r4dx_convert;

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

bool CompareBytes(const std::string& label, const std::vector<uint8_t>& got,
                  const std::vector<uint8_t>& want) {
  if (got.size() != want.size()) {
    std::fprintf(stderr, "FAIL %s: size mismatch got=%zu want=%zu\n", label.c_str(), got.size(),
                 want.size());
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "FAIL %s: first mismatch at byte %zu: got=0x%02x want=0x%02x\n",
                   label.c_str(), i, got[i], want[i]);
      return false;
    }
  }
  std::printf("OK   %s (%zu bytes byte-exact)\n", label.c_str(), got.size());
  return true;
}

// Weighted squared reconstruction error of one (row, group), in double so the comparison itself
// never hides a difference the float32 objective saw.
double GroupError(const float* x, const float* wt, int group, float scale, const uint8_t* q,
                  int zero) {
  double e = 0.0;
  for (int k = 0; k < group; ++k) {
    const double recon =
        static_cast<double>(scale) * (static_cast<double>(q[k]) - static_cast<double>(zero));
    const double d = static_cast<double>(x[k]) - recon;
    e += static_cast<double>(wt ? wt[k] : 1.0f) * d * d;
  }
  return e;
}

double Mxfp4GroupError(const float* x, const float* wt, int group, uint8_t raw,
                       const uint8_t* packed_row, int kbase) {
  const double scale = std::ldexp(1.0, static_cast<int>(raw) - 127);
  double e = 0.0;
  for (int k = 0; k < group; ++k) {
    const int kk = kbase + k;
    const uint8_t byte = packed_row[kk / 2];
    const uint8_t code = (kk % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
    const double mag = static_cast<double>(kE2M1Magnitude[code & 0x7]);
    const double recon = ((code & 0x8) ? -mag : mag) * scale;
    const double d = static_cast<double>(x[k]) - recon;
    e += static_cast<double>(wt ? wt[k] : 1.0f) * d * d;
  }
  return e;
}

// Fixture filename suffix for an int4 group -- see test_pack_bytes.cpp's GroupSuffix.
std::string GroupSuffix(int group) {
  return group == 128 ? std::string() : "_g" + std::to_string(group);
}

// ---- (i) search <= RTN on every group --------------------------------------------------------
// `int4_group` is the w4a16/w4a8 group under test: property (i) is a claim about the SEARCH, not
// about one group, so it is checked at every group the kernel can be built with.
bool NeverWorseThanRtn(const std::string& label, const std::vector<float>& w, int N, int K,
                       const std::vector<float>& imatrix, bool weighted, int int4_group,
                       bool include_mxfp4) {
  const float* wt_all = weighted ? imatrix.data() : nullptr;
  ImportanceVector imp;
  if (weighted) {
    imp.data = imatrix.data();
    imp.size = K;
  }
  const int gpr4 = K / int4_group;
  bool ok = true;
  int64_t improved = 0, groups = 0;
  double rtn_total = 0.0, search_total = 0.0;

  {  // w4a16
    std::vector<uint8_t> rq, rz, sq, sz;
    std::vector<float> rs, ss;
    QuantizeInt4Asymmetric(w.data(), N, K, int4_group, 1, rq, rs, rz);
    QuantizeInt4AsymmetricSearch(w.data(), N, K, int4_group, imp, 1, sq, ss, sz);
    for (int r = 0; r < N; ++r) {
      for (int g = 0; g < gpr4; ++g) {
        const size_t gi = static_cast<size_t>(r) * gpr4 + g;
        const size_t off = static_cast<size_t>(r) * K + static_cast<size_t>(g) * int4_group;
        const float* wtg = wt_all ? wt_all + static_cast<size_t>(g) * int4_group : nullptr;
        const double er = GroupError(w.data() + off, wtg, int4_group, rs[gi], rq.data() + off, rz[gi]);
        const double es = GroupError(w.data() + off, wtg, int4_group, ss[gi], sq.data() + off, sz[gi]);
        ++groups;
        rtn_total += er;
        search_total += es;
        if (es < er) ++improved;
        if (es > er) {
          std::fprintf(stderr, "FAIL %s w4a16 row %d group %d: search err %.9g > rtn err %.9g\n",
                       label.c_str(), r, g, es, er);
          ok = false;
        }
      }
    }
  }
  {  // w4a8 (zero pinned to 8)
    std::vector<uint8_t> rq, sq;
    std::vector<float> rs, ss;
    QuantizeInt4SymmetricPinned8(w.data(), N, K, int4_group, 1, rq, rs);
    QuantizeInt4Pinned8Search(w.data(), N, K, int4_group, imp, 1, sq, ss);
    for (int r = 0; r < N; ++r) {
      for (int g = 0; g < gpr4; ++g) {
        const size_t gi = static_cast<size_t>(r) * gpr4 + g;
        const size_t off = static_cast<size_t>(r) * K + static_cast<size_t>(g) * int4_group;
        const float* wtg = wt_all ? wt_all + static_cast<size_t>(g) * int4_group : nullptr;
        const double er = GroupError(w.data() + off, wtg, int4_group, rs[gi], rq.data() + off, 8);
        const double es = GroupError(w.data() + off, wtg, int4_group, ss[gi], sq.data() + off, 8);
        ++groups;
        rtn_total += er;
        search_total += es;
        if (es < er) ++improved;
        if (es > er) {
          std::fprintf(stderr, "FAIL %s w4a8 row %d group %d: search err %.9g > rtn err %.9g\n",
                       label.c_str(), r, g, es, er);
          ok = false;
        }
      }
    }
  }
  if (include_mxfp4) {  // mxfp4 -- its group is its own constant, so it is checked once, not once
                        // per int4 group
    const int gprm = K / kMxfp4Group;
    Mxfp4Quantized rm = QuantizeMxfp4(w.data(), N, K, kMxfp4Group, 1);
    Mxfp4Quantized sm = QuantizeMxfp4Search(w.data(), N, K, kMxfp4Group, imp, 1);
    for (int r = 0; r < N; ++r) {
      const uint8_t* rrow = rm.packed.data() + static_cast<size_t>(r) * (K / 2);
      const uint8_t* srow = sm.packed.data() + static_cast<size_t>(r) * (K / 2);
      for (int g = 0; g < gprm; ++g) {
        const size_t gi = static_cast<size_t>(r) * gprm + g;
        const int kbase = g * kMxfp4Group;
        const float* xg = w.data() + static_cast<size_t>(r) * K + kbase;
        const float* wtg = wt_all ? wt_all + kbase : nullptr;
        const double er = Mxfp4GroupError(xg, wtg, kMxfp4Group, rm.escale[gi], rrow, kbase);
        const double es = Mxfp4GroupError(xg, wtg, kMxfp4Group, sm.escale[gi], srow, kbase);
        ++groups;
        rtn_total += er;
        search_total += es;
        if (es < er) ++improved;
        if (es > er) {
          std::fprintf(stderr, "FAIL %s mxfp4 row %d group %d: search err %.9g > rtn err %.9g\n",
                       label.c_str(), r, g, es, er);
          ok = false;
        }
      }
    }
  }

  std::printf("%-34s %6lld groups, %6lld strictly improved, total err %.6g -> %.6g (%.2f%% of RTN)\n",
              label.c_str(), static_cast<long long>(groups), static_cast<long long>(improved),
              rtn_total, search_total,
              rtn_total > 0.0 ? 100.0 * search_total / rtn_total : 0.0);
  return ok;
}

}  // namespace

int main() {
  const std::string dir = R4DX_CONVERT_FIXTURES_DIR;
  bool ok = true;

  // Every int4 group r4d_gemm_w4a16_nt_m64 can be built with (R4DX_W4A16_GROUP must be a multiple
  // of the kernel's 64-wide packed block, so 64 and 128 are the whole set). All three properties
  // below are claims about the SEARCH, not about one group, so each is checked at both -- and the
  // default build therefore gates the group-64 path too.
  const int kInt4Groups[] = {128, 64};

  // ---- (i) on a random Gaussian matrix, unweighted and imatrix-weighted ----------------------
  {
    const int N = 32, K = 256;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::lognormal_distribution<float> idist(0.0f, 2.0f);
    std::vector<float> w(static_cast<size_t>(N) * K);
    for (auto& v : w) v = dist(rng);
    std::vector<float> imatrix(K);
    for (auto& v : imatrix) v = idist(rng);

    for (int group : kInt4Groups) {
      const std::string tag = " g" + std::to_string(group);
      const bool first = (group == kInt4Groups[0]);
      ok &= NeverWorseThanRtn("(i) random unweighted" + tag, w, N, K, imatrix, /*weighted=*/false,
                              group, /*include_mxfp4=*/first);
      ok &= NeverWorseThanRtn("(i) random imatrix" + tag, w, N, K, imatrix, /*weighted=*/true,
                              group, /*include_mxfp4=*/first);
    }
  }

  // ---- (ii) the outlier-dominated group -------------------------------------------------------
  // 16 rows x `group` K = one group per row. Every row is N(0, 0.02) except channel 0, which
  // carries a 1.0 outlier -- so the min/max grid's step is ~50x the bulk's own spread and RTN
  // throws the entire bulk onto a handful of codes. The imatrix says channel 0 is worth ~1e-6 of
  // the others, which is exactly the situation the weighted refit is for.
  for (int int4_group : kInt4Groups) {
    const int N = 16, K = int4_group;
    std::mt19937 rng(99);
    std::normal_distribution<float> small(0.0f, 0.02f);
    std::vector<float> w(static_cast<size_t>(N) * K);
    for (auto& v : w) v = small(rng);
    for (int r = 0; r < N; ++r) w[static_cast<size_t>(r) * K] = (r % 2 == 0) ? 1.0f : -1.0f;

    std::vector<float> imatrix(K, 1.0f);
    imatrix[0] = 1e-6f;
    ImportanceVector imp;
    imp.data = imatrix.data();
    imp.size = K;

    std::vector<uint8_t> rq, rz, sq, sz, wq, wz;
    std::vector<float> rs, ss, ws;
    QuantizeInt4Asymmetric(w.data(), N, K, int4_group, 1, rq, rs, rz);
    QuantizeInt4AsymmetricSearch(w.data(), N, K, int4_group, ImportanceVector{}, 1, sq, ss, sz);
    QuantizeInt4AsymmetricSearch(w.data(), N, K, int4_group, imp, 1, wq, ws, wz);

    double rtn = 0.0, unw = 0.0, wei = 0.0;      // under the IMATRIX-weighted metric
    double rtn_u = 0.0, unw_u = 0.0;             // under the UNWEIGHTED (plain MSE) metric
    for (int r = 0; r < N; ++r) {
      const size_t off = static_cast<size_t>(r) * K;
      rtn += GroupError(w.data() + off, imatrix.data(), K, rs[r], rq.data() + off, rz[r]);
      unw += GroupError(w.data() + off, imatrix.data(), K, ss[r], sq.data() + off, sz[r]);
      wei += GroupError(w.data() + off, imatrix.data(), K, ws[r], wq.data() + off, wz[r]);
      rtn_u += GroupError(w.data() + off, nullptr, K, rs[r], rq.data() + off, rz[r]);
      unw_u += GroupError(w.data() + off, nullptr, K, ss[r], sq.data() + off, sz[r]);
    }
    std::printf("(ii) outlier group w4a16 g%-3d imatrix-weighted err: rtn=%.6g  search=%.6g  "
                "search+imatrix=%.6g   (unweighted metric: rtn=%.6g search=%.6g)\n",
                int4_group, rtn, unw, wei, rtn_u, unw_u);
    // The unweighted search is compared against RTN under the metric it actually MINIMIZES. That
    // matters: `unw <= rtn` under the imatrix-weighted metric is not a claim the algorithm makes
    // (nothing optimized the weighted objective there) and it held at group 128 only by luck --
    // a 0.27% margin, which flips to 0.11% the wrong way at group 64, where an all-but-one-channel
    // group is half as wide and the outlier is twice as dominant. Under the unweighted metric the
    // guarantee IS structural (the RTN grid is candidate 0, quant_search.hpp), so that is what is
    // asserted; the weighted numbers above are printed for comparison only.
    if (!(unw_u <= rtn_u)) {
      std::fprintf(stderr, "FAIL (ii): unweighted search is worse than RTN on its own objective\n");
      ok = false;
    }
    if (!(wei < rtn)) {
      std::fprintf(stderr, "FAIL (ii): imatrix search did not beat RTN on the outlier group\n");
      ok = false;
    }
    // The point of the case: down-weighting the outlier channel has to be a substantial win on the
    // bulk, not a rounding-level one. The ceiling is structural -- the candidate scales bottom out
    // at 0.85x the min/max grid, and on a group whose error is pure quantization noise that caps
    // the improvement at 1 - 0.85^2 = 28%, which this case sits right at (observed ~32%, since the
    // refit adds a little on top). A bound of 25% therefore fails on a real regression without
    // demanding something the algorithm cannot deliver.
    if (!(wei < 0.75 * rtn)) {
      std::fprintf(stderr,
                   "FAIL (ii): imatrix search recovered only %.1f%% of the outlier group's error\n",
                   100.0 * (1.0 - wei / rtn));
      ok = false;
    }
    // ...and the imatrix has to be what does it: the unweighted search barely moves, because to it
    // the single outlier channel is worth as much as any of the group-1 bulk channels.
    if (!(wei < unw)) {
      std::fprintf(stderr, "FAIL (ii): the imatrix bought nothing over the unweighted search\n");
      ok = false;
    }
  }

  // ---- (iii) bit-exact vs the Python reference on the shared random fixture -------------------
  {
    nlohmann::json manifest;
    {
      std::ifstream f(dir + "/manifest.json");
      f >> manifest;
    }
    const int N = manifest.at("N").get<int>();
    const int K = manifest.at("K").get<int>();

    SafetensorsReader reader(Utf8ToWide(dir + "/input.safetensors"));
    const uint16_t* src = reinterpret_cast<const uint16_t*>(reader.Data("w"));
    std::vector<float> w(static_cast<size_t>(N) * K);
    for (size_t i = 0; i < w.size(); ++i) w[i] = r4dx::core::Bf16ToFloat(src[i]);

    const auto imat_bytes = ReadFileBytes(dir + "/search_imatrix.bin");
    if (imat_bytes.size() != static_cast<size_t>(K) * 4)
      throw std::runtime_error("search_imatrix.bin has the wrong length -- regenerate the fixtures");
    std::vector<float> imatrix(static_cast<size_t>(K));
    std::memcpy(imatrix.data(), imat_bytes.data(), imat_bytes.size());

    for (int pass = 0; pass < 2; ++pass) {
      const bool weighted = (pass == 1);
      const std::string sfx = weighted ? "search_imat" : "search";
      ImportanceVector imp;
      if (weighted) {
        imp.data = imatrix.data();
        imp.size = K;
      }
      for (int int4_group : kInt4Groups) {
        const std::string g = GroupSuffix(int4_group), tag = " g" + std::to_string(int4_group);
        {
          std::vector<uint8_t> q, zero;
          std::vector<float> scale;
          QuantizeInt4AsymmetricSearch(w.data(), N, K, int4_group, imp, 1, q, scale, zero);
          ok &= CompareBytes(sfx + " w4a16.wq" + tag, AsBytes(PackW4Nibbles(q, N, K, 1)),
                             ReadFileBytes(dir + "/" + sfx + "_w4a16_wq" + g + ".bin"));
          ok &= CompareBytes(sfx + " w4a16.wsz" + tag,
                             AsBytes(PackW4A16Scales(scale, zero, N, K, int4_group)),
                             ReadFileBytes(dir + "/" + sfx + "_w4a16_wsz" + g + ".bin"));
        }
        {
          std::vector<uint8_t> q;
          std::vector<float> scale;
          QuantizeInt4Pinned8Search(w.data(), N, K, int4_group, imp, 1, q, scale);
          ok &= CompareBytes(sfx + " w4a8.wq" + tag, AsBytes(PackW4Nibbles(q, N, K, 1)),
                             ReadFileBytes(dir + "/" + sfx + "_w4a8_wq" + g + ".bin"));
          ok &= CompareBytes(sfx + " w4a8.ws" + tag,
                             AsBytes(PackW4A8Scales(scale, N, K, int4_group)),
                             ReadFileBytes(dir + "/" + sfx + "_w4a8_ws" + g + ".bin"));
        }
      }
      {
        Mxfp4Quantized mq = QuantizeMxfp4Search(w.data(), N, K, kMxfp4Group, imp, 1);
        ok &= CompareBytes(sfx + " mxfp4.wq", PackMxfp4Wq(mq.packed, N, K, 1),
                           ReadFileBytes(dir + "/" + sfx + "_mxfp4_wq.bin"));
        ok &= CompareBytes(sfx + " mxfp4.ws", PackMxfp4Ws(mq.escale, N, K, kMxfp4Group),
                           ReadFileBytes(dir + "/" + sfx + "_mxfp4_ws.bin"));
        ok &= CompareBytes(sfx + " mxfp4.wref", mq.wref,
                           ReadFileBytes(dir + "/" + sfx + "_mxfp4_wref.bin"));
      }
    }
  }

  if (!ok) return 1;
  std::printf("PASS\n");
  return 0;
}
