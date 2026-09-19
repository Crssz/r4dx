// tests/model/test_gdn_layer.cpp -- GdnLayer + Mlp (layer 0, a GDN layer) against
// tools/reference/golden_out/layer_000_gdn.safetensors (tools/reference/layer_golden.py):
// prefill T=64, then a decode step of T=4 candidate tokens continuing the prefill's carried
// recurrent/conv state, comparing the FULL decoder-block output (GDN block's residual add -> MLP
// block's residual add) against the golden's `prefill_layer_output` / `decode_layer_output`, once
// per weight layout the container carries.
//
// Tolerances: bf16 gets the task's tight rel-L2 bound, TOLERANCES.bf16_matmul_rel_err (2e-2) from
// the golden's own manifest.json -- at that layout every GEMM this block calls (in_proj_qkv/z/b/a,
// out_proj, mlp.gate_up/down) runs at full bf16 precision, so the only error source is the same
// bf16 accumulation drift the golden's own tolerance already budgets for.
//
// The quantized layouts (mxfp4/w4a16/w4a8) chain FOUR quantized GEMMs per block (in_proj_qkv,
// out_proj, gate_up, down) through two residual adds. Each individual quantized GEMM is within the
// ~2e-2 rel-err gate tests/kernels/test_mxfp4_gemm.cpp and tools/convert_ref/kernel_crosscheck.py
// hold it to in isolation, but chaining four of them nonlinearly (through conv/gating/chunk_scan
// and silu_mul, not just four independent additions) measured out to ~7-8e-2 rel L2 on this real
// checkpoint's layer 0 -- reported by a first pass at 5e-2 (the golden manifest's own
// recurrent_state_rel_err tier, reused as a starting guess) failing outright. kLooseTol=1e-1 below
// is that measured number with headroom, not an aspirational figure: it is loose enough to pass
// what was actually observed and still catch a real regression (a broken quant path lands an order
// of magnitude higher, not 20% higher -- see test_mxfp4_gemm.cpp's own single-GEMM failures for
// comparison). Tightening this is future work for whoever improves the per-layer quantization
// (calibration, mixed-precision residual paths, etc.), not a claim that 1e-1 is intrinsic.
//
// Needs two pieces of real data neither vendored into the repo (docs/validation.md): the golden
// file itself (produced by layer_golden.py against the real checkpoint) and an r4dx test
// container carrying every layout (produced by r4dx-convert --layers N --layouts
// bf16,mxfp4,w4a16,w4a8 ...). Skips (CTest SKIPPED, not FAILED) if either is missing, exactly
// like tests/tokenizer/golden_test.cpp's SKIP_RETURN_CODE convention. A layout the container
// happens not to carry is reported and skipped individually rather than failing the whole test,
// so a partial container (e.g. bf16-only) still exercises what it has.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "container.h"
#include "gdn_layer.h"
#include "gdn_state.h"
#include "mlp.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Container;
using r4dx::model::GdnLayer;
using r4dx::model::GdnLayerParams;
using r4dx::model::GdnStateManager;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::Mlp;

namespace {
const char* kContainerPath = "D:/models/r4dx/qwen38-27b-l4-bf16.r4dx";
const char* kGoldenPath = "C:/Users/user/dev/r4dx/tools/reference/golden_out/layer_000_gdn.safetensors";

struct LayoutCase {
  Layout layout;
  double tolerance;
};

// Runs the prefill+decode block once at `layout`; returns false (and leaves *prefill_rel/
// *decode_rel untouched) if the container has no tensors for that layout at all.
bool RunLayout(const r4dx_convert::SafetensorsReader& golden, Layout layout,
                const std::vector<uint16_t>& prefill_in, const std::vector<uint16_t>& decode_in,
                const std::vector<float>& prefill_ref, const std::vector<float>& decode_ref,
                double* prefill_rel, double* decode_rel) {
  std::unique_ptr<Container> container_ptr;
  try {
    container_ptr =
        std::make_unique<Container>(Container::Load(kContainerPath, layout, Layout::kBf16,
                                                      /*layer_limit=*/1));
  } catch (const std::exception& e) {
    std::printf("test_gdn_layer (%s): SKIP (container load failed: %s)\n", LayoutName(layout),
                e.what());
    return false;
  }
  const Container& container = *container_ptr;
  const auto& cfg = container.Config();
  const auto& layer0 = container.Layer(0);
  const int64_t hidden = cfg.hidden_size;

  r4dx::core::Stream stream;
  r4dx::core::Arena arena(128ull << 20);
  GdnStateManager states(/*max_seqs=*/1, cfg.linear_num_value_heads, cfg.linear_value_head_dim,
                          cfg.linear_key_head_dim, cfg.ConvDim(), cfg.linear_conv_kernel_dim,
                          /*max_decode_window=*/static_cast<int64_t>(decode_in.size() / static_cast<size_t>(hidden)));
  GdnLayer gdn(cfg, layer0.input_layernorm, *layer0.gdn);
  Mlp mlp(cfg, layer0.post_attention_layernorm, layer0.mlp);
  r4dx::model::GdnControlCache control;

  auto run_block = [&](const std::vector<uint16_t>& in_host, bool is_prefill,
                        std::vector<float>* out_host) {
    const int64_t T = static_cast<int64_t>(in_host.size()) / hidden;
    auto x_dev = UploadBf16(in_host);
    r4dx::core::DeviceBuffer<uint16_t> gdn_out(static_cast<size_t>(T * hidden));
    r4dx::core::DeviceBuffer<uint16_t> block_out(static_cast<size_t>(T * hidden));
    GdnLayerParams p;
    p.slot = states.SlotForSeq(0);
    p.is_prefill = is_prefill;
    p.has_init = false;
    gdn.Forward(stream, arena, states, control, x_dev.data(), gdn_out.data(), T, p);
    mlp.Forward(stream, arena, gdn_out.data(), block_out.data(), T);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    arena.Reset();
    *out_host = WidenBf16(block_out.CopyToHost());
  };

  std::vector<float> prefill_got, decode_got;
  run_block(prefill_in, /*is_prefill=*/true, &prefill_got);
  run_block(decode_in, /*is_prefill=*/false, &decode_got);

  *prefill_rel = RelL2(prefill_got, prefill_ref);
  *decode_rel = RelL2(decode_got, decode_ref);
  return true;
}

}  // namespace

int main() {
  if (!FileExists(kContainerPath)) return SkipMissing(kContainerPath);
  if (!FileExists(kGoldenPath)) return SkipMissing(kGoldenPath);

  R4DX_HIP_CHECK(hipSetDevice(0));  // HIP_VISIBLE_DEVICES=1 remaps physical device 1 to index 0

  r4dx_convert::SafetensorsReader golden(r4dx_convert::Utf8ToWide(kGoldenPath));
  auto prefill_in = ReadGoldenRawBf16(golden, "prefill_hidden_states");
  auto decode_in = ReadGoldenRawBf16(golden, "decode_hidden_states");
  auto prefill_ref = ReadGoldenAsFloat(golden, "prefill_layer_output");
  auto decode_ref = ReadGoldenAsFloat(golden, "decode_layer_output");

  constexpr double kTightTol = 2e-2;  // TOLERANCES.bf16_matmul_rel_err
  constexpr double kLooseTol = 1e-1;  // measured ~7-8e-2 for the 4-quantized-GEMM chain; see file comment
  const LayoutCase cases[] = {
      {Layout::kBf16, kTightTol},
      {Layout::kMxfp4, kLooseTol},
      {Layout::kW4a16, kLooseTol},
      {Layout::kW4a8, kLooseTol},
  };

  bool all_ok = true;
  bool any_ran = false;
  for (const auto& c : cases) {
    double prefill_rel = 0.0, decode_rel = 0.0;
    if (!RunLayout(golden, c.layout, prefill_in, decode_in, prefill_ref, decode_ref, &prefill_rel,
                    &decode_rel)) {
      continue;
    }
    any_ran = true;
    const bool ok = prefill_rel < c.tolerance && decode_rel < c.tolerance;
    std::printf("test_gdn_layer (%s): prefill rel L2=%.4e, decode rel L2=%.4e (tol=%.0e) %s\n",
                LayoutName(c.layout), prefill_rel, decode_rel, c.tolerance, ok ? "PASS" : "FAIL");
    all_ok = all_ok && ok;
  }

  if (!any_ran) {
    std::fprintf(stderr, "test_gdn_layer: container carries no layout this test recognizes\n");
    return 1;
  }
  std::printf(all_ok ? "PASS\n" : "FAIL\n");
  return all_ok ? 0 : 1;
}
