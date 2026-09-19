// r4dx_convert::ResolveKvDescale (task item 2: --kv-calib) unit tests: the exact descale
// convention (amax / 448.0), the missing-layer/missing-key/wrong-length fallback-to-1.0 + warning
// paths, and the "calibration not applicable to this layer" (have_calib=false, e.g. MTP) silent
// path that must NOT warn.
#include <cmath>
#include <cstdio>

#include "nlohmann/json.hpp"
#include "r4dx_convert/kv_calib.hpp"

namespace {

bool Check(const char* label, bool cond) {
  std::printf("%-40s %s\n", label, cond ? "OK" : "FAIL");
  return cond;
}

bool AllClose(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-6f) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::fabs(a[i] - b[i]) > tol) return false;
  return true;
}

}  // namespace

int main() {
  using namespace r4dx_convert;
  bool ok = true;

  // ---- no calibration requested at all: 1.0, no warning ---------------------------------------
  {
    nlohmann::json empty;
    auto r = ResolveKvDescale(empty, /*have_calib=*/false, /*layer_idx=*/3, /*kv_heads=*/4, "k");
    ok &= Check("no-calib: values all 1.0", AllClose(r.values, {1.0f, 1.0f, 1.0f, 1.0f}));
    ok &= Check("no-calib: not from_calibration", !r.from_calibration);
    ok &= Check("no-calib: no warning", r.warning.empty());
  }

  // ---- real calibration entry: descale = amax / 448.0 -----------------------------------------
  {
    nlohmann::json calib = {
        {"3", {{"k_amax", {5.5625, 6.09375, 5.5, 5.5625}}, {"v_amax", {11.4375, 7.5, 16.375, 15.0}}}},
    };
    auto rk = ResolveKvDescale(calib, /*have_calib=*/true, 3, 4, "k");
    auto rv = ResolveKvDescale(calib, /*have_calib=*/true, 3, 4, "v");
    ok &= Check("k: from_calibration", rk.from_calibration);
    ok &= Check("k: no warning", rk.warning.empty());
    ok &= Check("k: descale == amax/448",
                AllClose(rk.values, {5.5625f / 448.0f, 6.09375f / 448.0f, 5.5f / 448.0f, 5.5625f / 448.0f}));
    ok &= Check("v: from_calibration", rv.from_calibration);
    ok &= Check("v: descale == amax/448",
                AllClose(rv.values, {11.4375f / 448.0f, 7.5f / 448.0f, 16.375f / 448.0f, 15.0f / 448.0f}));
  }

  // ---- layer missing from the file: fallback to 1.0, with a warning ---------------------------
  {
    nlohmann::json calib = {{"3", {{"k_amax", {1.0, 1.0, 1.0, 1.0}}, {"v_amax", {1.0, 1.0, 1.0, 1.0}}}}};
    auto r = ResolveKvDescale(calib, /*have_calib=*/true, /*layer_idx=*/7, 4, "k");
    ok &= Check("missing layer: values 1.0", AllClose(r.values, {1.0f, 1.0f, 1.0f, 1.0f}));
    ok &= Check("missing layer: not from_calibration", !r.from_calibration);
    ok &= Check("missing layer: warning present", !r.warning.empty());
  }

  // ---- layer present but missing the requested amax key ---------------------------------------
  {
    nlohmann::json calib = {{"3", {{"k_amax", {1.0, 1.0, 1.0, 1.0}}}}};  // no v_amax
    auto r = ResolveKvDescale(calib, /*have_calib=*/true, 3, 4, "v");
    ok &= Check("missing key: fallback 1.0", AllClose(r.values, {1.0f, 1.0f, 1.0f, 1.0f}));
    ok &= Check("missing key: warning present", !r.warning.empty());
  }

  // ---- wrong-length amax array (kv_heads mismatch) ---------------------------------------------
  {
    nlohmann::json calib = {{"3", {{"k_amax", {1.0, 2.0, 3.0}}}}};  // 3 entries, kv_heads=4
    auto r = ResolveKvDescale(calib, /*have_calib=*/true, 3, 4, "k");
    ok &= Check("wrong length: fallback 1.0", AllClose(r.values, {1.0f, 1.0f, 1.0f, 1.0f}));
    ok &= Check("wrong length: warning present", !r.warning.empty());
  }

  // ---- calibration exists in the file but doesn't apply to this layer (e.g. MTP) --------------
  {
    nlohmann::json calib = {{"3", {{"k_amax", {1.0, 2.0, 3.0, 4.0}}}}};
    auto r = ResolveKvDescale(calib, /*have_calib=*/false, 3, 4, "k");  // calib_applicable=false
    ok &= Check("not-applicable: fallback 1.0", AllClose(r.values, {1.0f, 1.0f, 1.0f, 1.0f}));
    ok &= Check("not-applicable: no warning", r.warning.empty());
  }

  if (!ok) return 1;
  std::printf("PASS\n");
  return 0;
}
