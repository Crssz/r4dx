// Dedicated coverage for the `bf16` linear-weight layout (task item 2: "--layouts to accept
// 'bf16' for ALL linear weights", needed for tight engine-vs-transformers-golden validation via
// r4d_gemm_bf16_nt_m64). docs/container-format.md: "bf16: <name>.bf16.w -- [N, K] uint8 pairs (no
// permutation, row-major, K-contiguous)" -- this test asserts PlanLinearLayouts/EmitLinearLayouts
// produce exactly that: correct planned shape/byte-size, and on-disk bytes that are plain
// row-major bf16 (element [n,k] at byte offset (n*K+k)*2, decodable back to ~the original float
// with ordinary bf16 rounding error, no WMMA-fragment permutation like the quantized layouts get).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/linear_layouts.hpp"

namespace {

bool Check(const char* label, bool cond) {
  std::printf("%-28s %s\n", label, cond ? "OK" : "FAIL");
  return cond;
}

}  // namespace

int main() {
  using namespace r4dx_convert;

  const int N = 32, K = 64;  // small, only bf16 requested so no group-size divisibility needed
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0f, 3.0f);
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (auto& v : w) v = dist(rng);

  LayoutSet layouts;
  layouts.bf16 = true;
  layouts.mxfp4 = layouts.w4a16 = layouts.w4a8 = false;

  bool ok = true;
  const std::string tmp_path = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") +
                                "\\r4dx_test_bf16_layout.r4dx";
  {
    // Scoped so ~ContainerWriter() closes its FILE* before we reopen the same path below --
    // otherwise the reopen can hit a sharing violation while the writer's handle is still open.
    ContainerWriter writer;
    PlanLinearLayouts(writer, "t", N, K, layouts);

    ok &= Check("planned tensor count == 1", writer.PlannedTensorCount() == 1);
    ok &= Check("planned bytes == N*K*2", writer.PlannedDataBytes() == static_cast<uint64_t>(N) * K * 2);

    nlohmann::json meta;
    meta["r4dx_format_version"] = "1";
    writer.FinalizeHeader(tmp_path, meta);
    EmitLinearLayouts(writer, "t", w, N, K, layouts, /*nthreads=*/1);
    writer.Finish();
  }

  // Re-derive the same bytes EmitLinearLayouts wrote and check they are plain row-major bf16 --
  // element [n,k] at (n*K+k)*2, no permutation -- by reading the file back directly.
  std::FILE* f = nullptr;
  fopen_s(&f, tmp_path.c_str(), "rb");
  ok &= Check("container reopened", f != nullptr);
  if (f) {
    std::fseek(f, 0, SEEK_END);
    long total = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> file_bytes(static_cast<size_t>(total));
    std::fread(file_bytes.data(), 1, file_bytes.size(), f);
    std::fclose(f);

    // Tensor data starts right after the 8-byte length + JSON header; PlanLinearLayouts planned
    // exactly one tensor ("t.bf16.w") at offset 0, so it's the file's very last N*K*2 bytes.
    const size_t data_len = static_cast<size_t>(N) * K * 2;
    ok &= Check("file large enough", file_bytes.size() >= data_len);
    const uint8_t* tensor_bytes = file_bytes.data() + (file_bytes.size() - data_len);

    double max_abs_err = 0.0;
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        const size_t byte_off = (static_cast<size_t>(n) * K + k) * 2;
        uint16_t raw;
        std::memcpy(&raw, tensor_bytes + byte_off, 2);
        const float decoded = r4dx::core::Bf16ToFloat(raw);
        const float original = w[static_cast<size_t>(n) * K + k];
        max_abs_err = std::max(max_abs_err, static_cast<double>(std::fabs(decoded - original)));
      }
    }
    // bf16 has ~8 bits of mantissa; for |w|~O(3) that's an absolute error well under 0.05.
    ok &= Check("row-major decode matches original (bf16 rounding)", max_abs_err < 0.05);
    std::printf("max_abs_err=%.6f\n", max_abs_err);
  }
  std::remove(tmp_path.c_str());

  if (!ok) return 1;
  std::printf("PASS\n");
  return 0;
}
