// tests/vision/test_gemma_embedder_golden.cpp -- the encoder-free embedder math on RANDOM weights:
// r4dx::vision::GemmaEmbedHost (LayerNorm(patch_dim) -> Linear+bias -> LayerNorm -> + pos[x,0] + pos[y,1] ->
// LayerNorm -> RMSNorm(no weight) -> Linear) against (a) the numpy reference and (b) the transformers
// Gemma4UnifiedVisionEmbedder (bf16, CPU) outputs tools/reference/gemma/vision_embedder_golden.py dumped.
// This is the oracle the device path (src/model/gemma_vision.cpp) is compared with on the GPU
// (tests/model/test_gemma_vision_gpu.cpp), so it pins the rounding points: bf16 after every op.
//
// Exits 77 (SKIPPED) when the golden file is absent.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gemma_test_util.h"
#include "gemma_vision.h"

using namespace r4dx::vision;

namespace {

struct Reader {
  const std::vector<uint8_t>& b;
  size_t off = 0;
  std::vector<float> F(size_t n) {
    std::vector<float> v(n);
    std::memcpy(v.data(), b.data() + off, n * 4);
    off += n * 4;
    return v;
  }
  std::vector<int32_t> I(size_t n) {
    std::vector<int32_t> v(n);
    std::memcpy(v.data(), b.data() + off, n * 4);
    off += n * 4;
    return v;
  }
};

// Counts of elements within `ulps` bf16 ulps (relative 2^-8 each, with a floor), and the worst relative error.
void Compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, double* worst,
             size_t* flips) {
  *worst = 0.0;
  *flips = 0;
  GCHECK(got.size() == want.size());
  for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - want[i]);
    const double scale = std::max(std::fabs(static_cast<double>(want[i])), std::ldexp(1.0, -6));
    const double rel = d / scale;
    if (d > 0) ++*flips;
    if (rel > *worst) *worst = rel;
  }
  std::printf("%s: worst rel err %.4g, %zu of %zu elements differ\n", what, *worst, *flips, got.size());
}

}  // namespace

int main() {
  const std::string path = gemma_test::GoldenDir() + "/vision_embedder.bin";
  std::vector<uint8_t> buf;
  if (!gemma_test::ReadFile(path, &buf)) {
    return gemma_test::Skip(path, "python tools/reference/gemma/vision_embedder_golden.py");
  }
  int32_t hdr[5];
  std::memcpy(hdr, buf.data(), sizeof(hdr));
  Reader r{buf, sizeof(hdr)};
  GemmaEmbedderWeights w;
  w.patch_dim = hdr[0];
  w.mm_dim = hdr[1];
  w.out_dim = hdr[2];
  w.posemb_size = hdr[3];
  const int64_t n = hdr[4];
  const size_t P = static_cast<size_t>(w.patch_dim), D = static_cast<size_t>(w.mm_dim), O = static_cast<size_t>(w.out_dim);
  w.ln1_w = r.F(P);
  w.ln1_b = r.F(P);
  w.dense_w = r.F(D * P);
  w.dense_b = r.F(D);
  w.ln2_w = r.F(D);
  w.ln2_b = r.F(D);
  w.pos_table = r.F(static_cast<size_t>(w.posemb_size) * 2 * D);
  w.pos_norm_w = r.F(D);
  w.pos_norm_b = r.F(D);
  w.proj_w = r.F(O * D);
  const std::vector<float> pixels = r.F(static_cast<size_t>(n) * P);
  const std::vector<int32_t> pos = r.I(static_cast<size_t>(n) * 2);
  const std::vector<float> out_torch = r.F(static_cast<size_t>(n) * O);
  const std::vector<float> out_numpy = r.F(static_cast<size_t>(n) * O);
  GCHECK(r.off == buf.size());

  const std::vector<float> got = GemmaEmbedHost(w, pixels.data(), pos.data(), n);
  double worst = 0;
  size_t flips = 0;
  Compare("host vs numpy reference", got, out_numpy, &worst, &flips);
  GCHECK(worst < 0.02);                                  // one bf16 ulp is 2^-8 relative (0.4%) at most
  GCHECK(flips * 20 <= got.size());                      // <= 5% of elements may flip a rounding
  Compare("host vs transformers (bf16, CPU)", got, out_torch, &worst, &flips);
  GCHECK(worst < 0.02);
  GCHECK(flips * 20 <= got.size());

  // Unscaled: the output must not carry the sqrt(hidden) text-row scale (rms ~ 1/sqrt(mm) * weight scale only).
  double ss = 0;
  for (float v : got) ss += static_cast<double>(v) * v;
  const double rms = std::sqrt(ss / static_cast<double>(got.size()));
  GCHECK(rms < 5.0);

  // Position handling: swapping x and y must change the result (x indexes axis 0, y axis 1).
  std::vector<int32_t> swapped = pos;
  for (int64_t i = 0; i < n; ++i) std::swap(swapped[static_cast<size_t>(i) * 2], swapped[static_cast<size_t>(i) * 2 + 1]);
  const std::vector<float> got_sw = GemmaEmbedHost(w, pixels.data(), swapped.data(), n);
  GCHECK(got_sw != got);

  if (gemma_test::g_failures != 0) {
    std::fprintf(stderr, "test_gemma_embedder_golden: %d failure(s)\n", gemma_test::g_failures);
    return 1;
  }
  std::printf("test_gemma_embedder_golden: OK\n");
  return 0;
}
