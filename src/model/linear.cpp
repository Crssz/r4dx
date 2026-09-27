#include "linear.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/model_kernels.h"
#include "r4d.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {

constexpr int64_t kMaxChunkM = 64;

// Rows per M tile of every r4d_gemm_*_nt_m64 kernel. Each output element is summed in an order set
// by SK alone -- the groups of one K slice in order, then the SK slices in a fixed order -- so a row's
// result does not depend on WV/MB/NPW/NT or on the other rows of the launch, but it DOES depend on
// SK: a different split adds the same partial products in a different order and can round
// differently. The measured table picks its SK per M-band, and its M=1 and M=4/M=8 picks differ for
// most shapes, which made every row of a speculative verify window (M = k+1) differ in its last bits
// from the single-row decode (M=1) row for the same context -- and a sampled round then emitted a
// different token whenever a draw landed between two near-tied candidates (docs/mtp.md, "Sampled
// rounds are bit-exact"). A chunk of at most kRowTile rows is a single row tile, so ResolveTuning
// gives all of them the M=1 band's WV/SK/MB/NPW. Only NT (a cache hint on the weight loads, which
// changes no arithmetic) differs for M=2..16 at w4a16: NT=0 there, because the M=1 band's NT=1 was
// the slower choice once the verify window's rows are distinct -- measured on qwen38-27b-v6.r4dx,
// one VerifyWindow at position 60, mean of two runs: 8 rows 32.12 ms with the old per-band table,
// 32.72 ms with the M=1 tuning as is, 32.27 ms with it and NT=0; 4 rows 30.00 / 30.12 / 29.83 ms.
constexpr int64_t kRowTile = 16;

// Hand-derived fallback, legal for every (layout,N,K) shape this model has (used only when
// gemm_tuning_table.inc has no row for the requested shape -- see PickTuning below and linear.h's
// comment). Every quantized GEMM family this model calls needs K divisible by SK*group() (bf16:
// group 16, w4a16: 64 or 128 (R4DX_W4A16_GROUP), w4a8: 128, mxfp4: group 32) and N divisible by
// 16. This model's only K values are hidden_size=5120, intermediate_size=17408, and
// value_dim=6144 (attn.o's K = num_heads*head_dim = 6144 too) -- all three are multiples of 512
// (5120/512=10, 17408/512=34, 6144/512=12), which is the tightest of the group requirements
// (SK=4 * group=128), so SK=4 clears every layout at once. WV=4/SK=4 keeps the block at 512
// threads (WV*SK*32, under the 1024 cap every kernel enforces) and the LDS reduction buffer at 16
// KiB (under the 64 KiB cap); MB=1
// and NPW=1 are the simplest legal choice for every kernel (MB in 1..4, NPW in {1,4} for w4a16 /
// {1,2,4,8} for w4a8/mxfp4) and NT=1 takes the non-temporal weight-load path
// r4d_gemm_w4a16_nt_m64.hip's own comment recommends for a weight that is read once per step and
// never reused.
//
// quant2 Q3 (docs/quant2.md section 5.1): this is also the tuning of every w4a16 linear at a
// per-tensor group the table has no row for. The kernel's rule at group g is K % (SK * max(g, 64))
// -- a split must start on a 64-K packed block, which a group of 32 does not guarantee -- so SK=4
// needs K % 256 at g32/g64 and K % 512 at g128: the K % 512 check below already covers all three.
// `w4a16_group` is the EFFECTIVE group (EffectiveW4a16Group); the explicit check only keeps a future
// edit of SK from quietly breaking the per-tensor groups.
//
// Trellis (docs/trellis-kernel.md 4.4) has its own, M-aware rule, since a split 128-column group
// needs every row tile in one block (a prefill chunk under the M <= 16 rule would throw):
//   M <= 16 (only ever asked at M = 1: ResolveTuning maps the whole row tile there): 128-wide
//     blocks (WV 4, NP 1), SK 2, MT 1, U 2, NT 1, and SKG = clamp(128 / (N/128), 1, 4) rounded down
//     to a power of two, halved until (K/16) % (SK*SKG*U) == 0 -- a cross-block split for the
//     narrow shapes, so the grid is not 8 blocks wide (4.3);
//   M > 16: SKG 1, MT = min(4, ceil(M/16)), the same block, NT 0 -- never split.
// Legal for every K and N that are multiples of 128 (the loader's rule) and every part boundary
// that is (Wc = 128 divides it).
LinearTuning FallbackTuning(Layout layout, int64_t N, int64_t K, int64_t M, int variant) {
  if (layout == Layout::kTrellis) {
    if (K <= 0 || N <= 0 || K % 128 != 0 || N % 128 != 0) {
      throw std::runtime_error("r4dx::model::PickTuning: trellis [" + std::to_string(N) + ", " +
                               std::to_string(K) + "] is not whole 128-blocks");
    }
    LinearTuning t{/*WV=*/4, /*SK=*/2, /*MB=*/1, /*NPW=*/1, /*NT=*/1};
    t.U = 2;
    if (M > kRowTile) {
      t.SKG = 1;
      t.MB = static_cast<int>(std::min<int64_t>(4, (M + 15) / 16));
      t.NT = 0;
      return t;
    }
    int skg = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(4, 128 / (N / 128))));
    int pow2 = 1;
    while (pow2 * 2 <= skg) pow2 *= 2;
    skg = pow2;
    while (skg > 1 && (K / 16) % (t.SK * skg * t.U) != 0) skg /= 2;
    t.SKG = skg;
    return t;
  }
  const int w4a16_group = variant;
  if (K % 512 != 0) {
    throw std::runtime_error("r4dx::model::PickTuning: K=" + std::to_string(K) +
                              " is not a multiple of 512 (SK=4 * w4a16/w4a8 group 128) -- this "
                              "model shape was not anticipated, pick a smaller SK");
  }
  if (N % 16 != 0) {
    throw std::runtime_error("r4dx::model::PickTuning: N=" + std::to_string(N) +
                              " is not a multiple of 16");
  }
  constexpr int kSk = 4;
  if (layout == Layout::kW4a16 && K % (kSk * std::max(w4a16_group, 64)) != 0) {
    throw std::runtime_error("r4dx::model::PickTuning: K=" + std::to_string(K) +
                              " does not split into SK=4 whole groups at w4a16 group " +
                              std::to_string(w4a16_group));
  }
  return LinearTuning{/*WV=*/4, /*SK=*/kSk, /*MB=*/1, /*NPW=*/1, /*NT=*/1};
}

// r4d_gemm_w4a16_nt_m64_group(): the group the historical entry serves and every row with
// GemmTuningRow::group == 0 was meant for.
int DefaultW4a16Group() {
  static const int kGroup = r4d_gemm_w4a16_nt_m64_group();
  return kGroup;
}

// tools/profile/tune_gemm.py's measured sweep, if it has been generated (src/model/CMakeLists.txt
// treats a missing file as a build error -- see that file's comment -- so an empty table checked
// into the tree, `{}`, is what a fresh checkout without having run the sweep gets; PickTuning below
// falls back to FallbackTuning() row-by-row in that case, not a hard failure).
#include "gemm_tuning_table.inc"

// Tensor parallel (docs/tp.md 2.7): the per-rank TP=2 (N,K) rows, generated by the same
// tools/profile/tune_gemm.py into the same array name, and consulted only on a thread that called
// SetTp2TuningForThisThread(true).
namespace tp2 {
#include "gemm_tuning_table_tp2.inc"
}  // namespace tp2

// Trellis (docs/trellis-kernel.md 5.3, 10.2): tests/kernels/tool_trellis_gemm_bench.exe's joint
// M = 1 / M = 8 pick, one M = 1 row per (N, K, KB) at TP = 1, into the same array name. Consulted
// for kTrellis only, after the TP table and the main table (which carry no trellis rows today).
namespace trellis {
#include "gemm_tuning_table_trellis.inc"
}  // namespace trellis

thread_local bool t_tp2_tuning = false;

// The trellis kernel's legality rules that depend on the tuning and the chunk
// (docs/trellis-kernel.md 4.3; libr4d's r4d_trellis_check is the authority and throws on everything
// else): whole blocks of Wc = WV*NP*32 columns, whole U-steps of every K slice, and a split
// 128-group (SKG > 1 or Wc < 128) only with every row tile in its block. A table row that fails
// them for the requested chunk is skipped like a w4a16 row at the wrong group. The one rule that
// needs the linear -- no block across a two-part linear's part boundary -- is not known to the
// pick; ApplyLinear's TrellisChunkTuning checks it.
bool TrellisRowFits(int64_t N, int64_t K, int64_t M, const LinearTuning& t) {
  const int64_t wc = static_cast<int64_t>(t.WV) * t.NPW * 32;
  if (wc <= 0 || N % wc != 0) return false;
  if ((K / 16) % (static_cast<int64_t>(t.SK) * t.SKG * t.U) != 0) return false;
  const bool split = t.SKG > 1 || wc < 128;
  return !split || (M + 15) / 16 <= t.MB;
}

// The best row of `table` for (layout, N, K, M), or nullptr. The table's w4a16 rows are swept at
// one R4DX_W4A16_GROUP (tools/profile/tune_gemm.py), but one table serves every build: a row that
// splits K into SK slices of whole groups at 64 need not at 128 (SK=16 at K=5120), and the kernel
// throws on it. Such a row is skipped here, so this build falls through to the next wider M-band's
// row, then the next table / FallbackTuning, never an illegal launch.
//
// quant2 Q3: for w4a16 the group is part of the key. `w4a16_group` is the EFFECTIVE group of the
// launch (EffectiveW4a16Group); a row's is its GemmTuningRow::group, 0 meaning the build default.
// At the default group every existing row matches and the legality test is the one above
// (max(g, 64) == g for every default a build accepts), so the default path picks what it always
// picked. At another group only rows measured there match, and the test is the kernel's own
// K % (SK * max(g, 64)).
//
// Trellis (docs/trellis-kernel.md 5.3): `variant` is the linear's rate, matched against
// GemmTuningRow::rate, and a row must fit the chunk (TrellisRowFits). For w4a16 `variant` is the
// effective group, as above; every other layout ignores it.
template <size_t kRows>
const GemmTuningRow* BestRow(const GemmTuningRow (&table)[kRows], Layout layout, int64_t N,
                             int64_t K, int64_t M, int variant) {
  const GemmTuningRow* best = nullptr;
  for (const GemmTuningRow& row : table) {
    if (row.layout != layout || row.N != N || row.K != K) continue;
    if (row.M < M) continue;  // only ever round UP to a wider-or-equal measured M-band
    if (layout == Layout::kW4a16) {
      const int w4a16_group = variant;
      const int row_group = row.group == 0 ? DefaultW4a16Group() : row.group;
      if (row_group != w4a16_group) continue;
      if (K % (row.tuning.SK * std::max(w4a16_group, 64)) != 0) continue;
    }
    if (layout == Layout::kTrellis) {
      if (row.rate != variant) continue;
      if (!TrellisRowFits(N, K, M, row.tuning)) continue;
    }
    if (best == nullptr || row.M < best->M) best = &row;
  }
  return best;
}

}  // namespace

void SetTp2TuningForThisThread(bool enabled) { t_tp2_tuning = enabled; }

int EffectiveW4a16Group(int w4a16_group) {
  return w4a16_group == 0 ? DefaultW4a16Group() : w4a16_group;
}

// Internal linkage (not declared in linear.h) -- the actual table scan, now called only on a
// PickTuning cache miss (see below). `tp2`: the TP table first (docs/tp.md 2.7), then the main one,
// then (kTrellis only) the trellis rows. `variant`: the effective group for kW4a16, the rate for
// kTrellis, 0 for every other layout.
static LinearTuning ResolveTuning(Layout layout, int64_t N, int64_t K, int64_t M, bool tp2,
                                  int variant) {
  // Any chunk that fits in one row tile resolves through the M=1 band (kRowTile's comment): a
  // verify window's rows then sum in exactly the order a single-row decode's do. Trellis keeps the
  // M=1 band's NT as well (docs/trellis-kernel.md 10.1: its measured M <= 16 pick is NT 1).
  if (M > 1 && M <= kRowTile) {
    LinearTuning t = ResolveTuning(layout, N, K, 1, tp2, variant);
    if (layout == Layout::kW4a16) t.NT = 0;
    return t;
  }
  if (tp2) {
    if (const GemmTuningRow* row = BestRow(tp2::kGemmTuningTable, layout, N, K, M, variant)) {
      return row->tuning;
    }
  }
  if (const GemmTuningRow* row = BestRow(kGemmTuningTable, layout, N, K, M, variant)) {
    return row->tuning;
  }
  if (layout == Layout::kTrellis) {
    if (const GemmTuningRow* row = BestRow(trellis::kGemmTuningTable, layout, N, K, M, variant)) {
      return row->tuning;
    }
  }
  return FallbackTuning(layout, N, K, M, variant);
}

LinearTuning PickTuning(Layout layout, int64_t N, int64_t K, int64_t M, int variant) {
  // Cache the resolved LinearTuning per (layout,N,K,M) (review finding, 2026-09-19): PickTuning is
  // called once per <=64-row sub-chunk of every GEMM -- roughly 6 GEMMs x 64 layers per decode
  // token -- and a linear scan of kGemmTuningTable's ~196 rows on every one of those calls is
  // ~75k row comparisons of pure host work per token on the exact hot path the host-overhead pass
  // (2026-09-19) was trying to minimize. Model is single-sequence / single-worker-thread
  // (model.h's own SCOPE comment; src/server also serializes all requests through one worker
  // thread onto one Model). Under tensor parallelism (docs/tp.md 2.7) each rank's Model runs on its
  // own thread, so the cache is thread_local: every thread builds its own (identical) copy and no
  // locking is needed. N/K fit in 20 bits (this model's widest is intermediate_size=17408 < 2^20),
  // M in 8 bits (<=64), layout in 4 bits (48..51) -- packed key never collides for any shape this
  // model has. Bit 52 is SetTp2TuningForThisThread's flag, so a thread that flips it cannot be
  // served a row resolved under the other setting. Bits 53 and up hold the effective w4a16 group
  // (quant2 Q3; 0 for every other layout, so their keys do not split by it), which is at most 256
  // for any build r4dx configures -- 9 bits, 53..61, clear of the sign bit. A trellis linear's rate
  // (4 or 5, docs/trellis-kernel.md 5.3) takes the same bits: the layout field already keeps the
  // two apart.
  static thread_local std::unordered_map<int64_t, LinearTuning> cache;
  const bool tp2 = t_tp2_tuning;
  const int v = layout == Layout::kW4a16 ? EffectiveW4a16Group(variant)
                : layout == Layout::kTrellis ? variant
                                             : 0;
  if (v < 0 || v >= 512) {
    throw std::runtime_error("r4dx::model::PickTuning: variant " + std::to_string(v) +
                             " does not fit the tuning cache key");
  }
  const int64_t key = (static_cast<int64_t>(v) << 53) | (static_cast<int64_t>(tp2) << 52) |
                      (static_cast<int64_t>(layout) << 48) | (N << 28) | (K << 8) | M;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const LinearTuning t = ResolveTuning(layout, N, K, M, tp2, v);
  cache.emplace(key, t);
  return t;
}

int EpilogueForLayout(Layout layout) {
  // Milestone 4 follow-up to docs/r9700.md R2/P2 (2026-09-20): root-caused and re-enabled for
  // w4a8/mxfp4. History: the fused epilogues (kernels.h's r4dx_epilogue: f16, fp8 row-major, int8
  // fragA8) were built and verified BYTE-EXACT in isolation (tests/kernels/test_fused_quant.cpp,
  // 135/135, M in {1,2,4,16,64} x K in {5120,6144,17408}), but wiring them into the model changed
  // w4a8/mxfp4's real generated text even though an in-model diagnostic showed zero differing
  // elements on the fused GEMMs' own outputs -- shipped disabled pending root-cause (see git log
  // on this file / docs/status.md's original R2/P2 section for the full incident writeup this pass
  // inherited).
  //
  // ROOT CAUSE (this pass): not the epilogue kernels' own math (already independently verified),
  // and not which GEMM consumed a fused buffer. It is `r4dx::core::Arena::Alloc`
  // (r4dx/core/arena.hpp): it only aligned each allocation's OWN start to the caller-requested
  // alignment, never its END. A fused epilogue's per-row fp32 scale scratch is exactly `T` floats
  // (T = chunk row count, 1..64) -- 4*T bytes, a multiple of 16 only when T%4==0 -- so at decode
  // (T=1) and MTP verify (T=2..4) the NEXT default-aligned allocation in the same layer (e.g.
  // GdnLayer::Forward's `mixed_qkv`) starts a few bytes short of 16-byte alignment, and every
  // allocation after it inherits the same drift for the rest of the layer. Every third_party/libr4d
  // kernel this arena feeds (GDN conv/kkt/chunk-scan, every quantized GEMM family, attention's
  // decode scratch) reads its operands with unchecked wide (16-byte) vector loads and silently
  // reads the wrong bytes when handed a misaligned pointer -- unlike this project's OWN
  // P6-rewritten rmsnorm/residual_rmsnorm/silu_mul kernels, which fall back to a scalar loop when
  // misaligned. Before this fusion pass every arena allocation's SIZE happened to already be a
  // multiple of 16 bytes (hidden/conv_dim/intermediate are all multiples of 8 bf16 elements), so
  // `offset_` was always incidentally 16-aligned and this was never triggered -- several call
  // sites' own comments already flagged that invariant as "incidental, not enforced" (gdn_layer.cpp,
  // attention_layer.hpp, this file's i8_scratch/fp8_scratch allocations). FIX (arena.hpp): every
  // Alloc call now rounds its own END up to 16 bytes too, so every FUTURE allocation starts
  // 16-aligned again regardless of what alignment it individually requests. Covered by a new
  // host-side ArenaAlignmentInvariant check in tests/kernels/test_fused_quant.cpp (reproduces the
  // exact odd-T scale-then-buffer allocation sequence GdnLayer/AttentionLayer/Mlp use, T=1..64, and
  // asserts every subsequent pointer is 16-aligned) and re-verified end-to-end BYTE-IDENTICAL
  // generated text (SHA-256), fusion-on vs fusion-off, for w4a8 and mxfp4 across three prompt
  // lengths x `--mtp {0,3}` (tools/validate_fusion.ps1 -- also usable as a standing regression gate;
  // docs/status.md's R2/P2 section has the full run log).
  //
  // w4a16's r4dx_epilogue_f16 is a SEPARATE, unrelated issue: it is equally CORRECT now (verified
  // the same way) but still REGRESSES decode wall-clock (-4.3%, reproducible, unchanged from the
  // original incident): r4dx_model_cast_bf16_to_f16 launches a FLAT elementwise grid
  // (blocks=ceil(M*K/256), ~20 independent workgroups at decode T=1/K=5120), while fusing it into
  // rmsnorm/residual_rmsnorm/silu_mul's own one-workgroup-per-row epilogue collapses that work onto
  // a SINGLE workgroup -- a real parallelism loss the saved launch does not cover. Per this task's
  // own instruction ("do not re-enable in that form"), w4a16 stays r4dx_epilogue_none until a
  // wide-grid f16 cast or a prefill-only fusion (dim3(rows)=dim3(T), no parallelism loss at T>1) is
  // built -- Milestone 5+ work, not part of this pass.
  //
  // R4DX_DISABLE_EPILOGUE=1 forces every layout back to r4dx_epilogue_none regardless of the mapping
  // below -- the A/B toggle tools/validate_fusion.ps1 uses to regenerate the "fusion off" baseline
  // every run from the SAME binary rather than requiring a second build. Unset in every other
  // caller (every ctest binary, every plain CLI/server invocation), which see the mapping directly.
  static const bool kDisabled = [] {
    const char* e = std::getenv("R4DX_DISABLE_EPILOGUE");
    return e != nullptr && e[0] == '1';
  }();
  if (kDisabled) return r4dx_epilogue_none;
  switch (layout) {
    case Layout::kBf16:
      return r4dx_epilogue_none;  // never quantizes its activation input -- nothing to fuse.
    case Layout::kW4a16:
      return r4dx_epilogue_none;  // Problem B: parallelism loss at decode, see comment above.
    case Layout::kW4a8:
      return r4dx_epilogue_int8_fraga8;
    case Layout::kMxfp4:
      return r4dx_epilogue_fp8_e4m3_row;
    case Layout::kTrellis:
      // docs/trellis-kernel.md 5.3: the input transform is per linear (x * suh, then H), which no
      // producer epilogue can name -- ApplyLinear runs it (v1), or a fused producer hands it over
      // through PreQuantizedActivation::transform_id (M5). Every layer's own epilogue plumbing then
      // takes its plain bf16 path.
      return r4dx_epilogue_none;
  }
  return r4dx_epilogue_none;
}

namespace {

// The tuning of one m-row chunk of trellis linear `w`. PickTuning knows (N, K, M, rate) but not the
// part boundary, and the kernel needs every block on one side of it (n_split % Wc == 0) while the
// loader guarantees only whole 128-blocks: a Wc = 256 row on a two-part linear whose boundary is an
// odd multiple of 128 would pass TrellisRowFits and throw at its first launch. Such a chunk takes
// 4.4's fallback instead (Wc = 128, legal at every boundary the loader accepts; one tuning for every
// m <= 16, so row identity holds). Today's table has Wc 32 and 128 rows only, and both boundaries
// (17408, and 8704 on a TP = 2 rank) are multiples of 256, so the shipped rows never take it.
LinearTuning TrellisChunkTuning(const QuantLinear& w, int64_t m) {
  const LinearTuning t = PickTuning(Layout::kTrellis, w.N, w.K, m, w.trellis_bits);
  const int64_t wc = static_cast<int64_t>(t.WV) * t.NPW * 32;
  if (w.trellis_parts > 1 && w.trellis_part_n[0] % wc != 0) {
    return FallbackTuning(Layout::kTrellis, w.N, w.K, m, w.trellis_bits);
  }
  return t;
}

// docs/trellis-kernel.md 4.8's A-range study (milestone M4). With R4DX_TRELLIS_A_STATS=<file>, every
// A the trellis input transform writes is copied back and tallied per linear -- elements, exact
// zeros, f16 subnormals, non-finite values, max |A|, the sum of squares, and a histogram of
// floor(log2 |A|) over [-24, 15] (the f16 range, subnormals included) -- and the tallies are written
// to <file> as JSON when the process exits, one record per linear in the order the linears first ran
// (a forward pass's order: layer by layer, each layer's linears as its code calls them). A
// measurement hook, not a mode: it synchronizes the stream on every trellis linear it sees. Unset --
// every normal run -- the trellis path pays one test of a static pointer.
class TrellisAStats {
 public:
  static constexpr int kLog2Lo = -24, kBins = 40;  // floor(log2 |A|) in [-24, 15]

  // nullptr unless R4DX_TRELLIS_A_STATS names a file.
  static TrellisAStats* Get() {
    static TrellisAStats* const stats = [] {
      const char* p = std::getenv("R4DX_TRELLIS_A_STATS");
      static TrellisAStats instance(p != nullptr ? p : "");
      return (p != nullptr && p[0] != '\0') ? &instance : nullptr;
    }();
    return stats;
  }

  // One chunk of `w`'s A: `parts` buffers of m x K f16, `part_stride` elements apart.
  void Record(const QuantLinear& w, const uint16_t* a, int parts, int64_t m, int64_t K,
              int64_t part_stride, hipStream_t s) {
    std::vector<uint16_t> host(static_cast<size_t>(m * K));
    std::lock_guard<std::mutex> lock(mu_);
    auto it = recs_.find(w.trellis_suh.data());
    if (it == recs_.end()) {
      Rec r;
      r.order = static_cast<int>(recs_.size());
      r.N = w.N;
      r.K = w.K;
      r.parts = w.trellis_parts;
      r.prescale_log2 = w.trellis_prescale_log2;
      it = recs_.emplace(w.trellis_suh.data(), r).first;
    }
    Rec& r = it->second;
    ++r.calls;
    for (int p = 0; p < parts; ++p) {
      R4DX_HIP_CHECK(hipMemcpyAsync(host.data(), a + p * part_stride, host.size() * 2,
                                    hipMemcpyDeviceToHost, s));
      R4DX_HIP_CHECK(hipStreamSynchronize(s));
      for (uint16_t h : host) Tally(r, h);
    }
  }

  ~TrellisAStats() {
    if (path_.empty() || recs_.empty()) return;
    std::vector<const Rec*> order(recs_.size());
    for (const auto& kv : recs_) order[static_cast<size_t>(kv.second.order)] = &kv.second;
    std::FILE* f = std::fopen(path_.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "r4dx: R4DX_TRELLIS_A_STATS: cannot write %s\n", path_.c_str());
      return;
    }
    std::fprintf(f, "{\"hist_log2_lo\": %d, \"linears\": [\n", kLog2Lo);
    for (size_t i = 0; i < order.size(); ++i) {
      const Rec& r = *order[i];
      std::fprintf(f,
                   "  {\"order\": %d, \"N\": %lld, \"K\": %lld, \"parts\": %d, \"prescale_log2\": %d, "
                   "\"calls\": %lld, \"elems\": %lld, \"zeros\": %lld, \"subnormals\": %lld, "
                   "\"nonfinite\": %lld, \"max_abs\": %.9g, \"sum_sq\": %.17g, \"hist\": [",
                   r.order, static_cast<long long>(r.N), static_cast<long long>(r.K), r.parts,
                   r.prescale_log2, static_cast<long long>(r.calls),
                   static_cast<long long>(r.elems), static_cast<long long>(r.zeros),
                   static_cast<long long>(r.subnormals), static_cast<long long>(r.nonfinite),
                   static_cast<double>(core::F16ToFloat(r.max_bits)), r.sum_sq);
      for (int b = 0; b < kBins; ++b) {
        std::fprintf(f, "%s%lld", b == 0 ? "" : ", ", static_cast<long long>(r.hist[b]));
      }
      std::fprintf(f, "]}%s\n", i + 1 < order.size() ? "," : "");
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
    std::fprintf(stderr, "r4dx: R4DX_TRELLIS_A_STATS: %zu trellis linears written to %s\n",
                 order.size(), path_.c_str());
  }

 private:
  struct Rec {
    int order = 0, parts = 0, prescale_log2 = 0;
    int64_t N = 0, K = 0;
    int64_t calls = 0, elems = 0, zeros = 0, subnormals = 0, nonfinite = 0;
    uint16_t max_bits = 0;  // the largest |A| seen, as f16 bits with the sign cleared
    double sum_sq = 0.0;
    int64_t hist[kBins] = {};
  };

  explicit TrellisAStats(std::string path) : path_(std::move(path)) {}

  static void Tally(Rec& r, uint16_t h) {
    ++r.elems;
    h &= 0x7FFFu;
    if (h == 0) {
      ++r.zeros;
      return;
    }
    const int e = h >> 10;
    if (e == 31) {
      ++r.nonfinite;
      return;
    }
    int lg = e - 15;
    if (e == 0) {  // subnormal: (h & 0x3FF) * 2^-24
      ++r.subnormals;
      lg = -24;
      for (unsigned mant = h >> 1; mant != 0; mant >>= 1) ++lg;
    }
    ++r.hist[lg - kLog2Lo];
    r.max_bits = std::max(r.max_bits, h);
    const double v = core::F16ToFloat(h);
    r.sum_sq += v * v;
  }

  std::string path_;
  std::mutex mu_;
  std::map<const void*, Rec> recs_;
};

}  // namespace

void ApplyLinear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M, const PreQuantizedActivation* pre) {
  if (w.N <= 0 || w.K <= 0) throw std::runtime_error("r4dx::model::ApplyLinear: empty weight");
  const int64_t N = w.N, K = w.K;
  const hipStream_t s = stream;

  // R2/P2 (docs/r9700.md): a caller may have already produced this call's quantized activation as
  // a fused producer epilogue (rmsnorm/residual_rmsnorm/silu_mul, kernels.h's r4dx_epilogue) --
  // when the format matches this weight's layout, skip the per-chunk quant/cast launch below
  // entirely and read straight from the caller's buffer instead of arena scratch.
  const bool have_pre = (pre != nullptr) && (pre->epilogue != r4dx_epilogue_none);
  if (have_pre && pre->epilogue != EpilogueForLayout(w.layout)) {
    throw std::runtime_error(
        "r4dx::model::ApplyLinear: PreQuantizedActivation.epilogue does not match w.layout "
        "(caller bug -- see EpilogueForLayout)");
  }
  // Trellis (docs/trellis-kernel.md 5.3): an A already input-transformed for THIS linear -- the
  // transform is per linear (its own suh), so the id must be this weight's, not merely a trellis
  // one's.
  const bool trellis_pre = (pre != nullptr) && (pre->transform_id != nullptr);
  if (trellis_pre && (w.layout != Layout::kTrellis || pre->transform_id != w.trellis_suh.data())) {
    throw std::runtime_error(
        "r4dx::model::ApplyLinear: PreQuantizedActivation.transform_id is not this linear's "
        "trellis input transform (caller bug -- a trellis A is only valid for the linear whose suh "
        "made it)");
  }
  if (w.layout == Layout::kTrellis) {
    if (w.trellis_parts < 1 || w.trellis_parts > 2 || w.trellis_tickets == nullptr ||
        w.trellis_suh.size() != static_cast<size_t>(w.trellis_parts * K) ||
        w.trellis_svh.size() != static_cast<size_t>(N) ||
        (w.trellis_parts == 2 ? w.trellis_part_n[0] + w.trellis_part_n[1] : w.trellis_part_n[0]) !=
            N) {
      throw std::runtime_error("r4dx::model::ApplyLinear: trellis weight [" + std::to_string(N) +
                               ", " + std::to_string(K) + "] is not a loaded one (parts, suh, svh "
                               "or tickets inconsistent -- see QuantLinear's trellis fields)");
    }
    if (trellis_pre && (pre->data == nullptr ||
                        (w.trellis_parts > 1 && pre->part_stride < M * K))) {
      throw std::runtime_error("r4dx::model::ApplyLinear: trellis PreQuantizedActivation needs "
                               "data and, for two parts, part_stride >= M * K");
    }
    // The GEMM reads A with 8-byte loads and checks no alignment itself (a misaligned operand is
    // read wrong, silently -- EpilogueForLayout's ROOT CAUSE note), so every part must start where an
    // arena allocation would: data 16-byte aligned, and part_stride whole 16 bytes (8 elements).
    // The chunk offset m0 * K keeps it (K is whole 128-blocks).
    if (trellis_pre && (reinterpret_cast<uintptr_t>(pre->data) % 16 != 0 ||
                        (w.trellis_parts > 1 && pre->part_stride % 8 != 0))) {
      throw std::runtime_error("r4dx::model::ApplyLinear: trellis PreQuantizedActivation needs "
                               "16-byte aligned data and, for two parts, a part_stride that is a "
                               "multiple of 8 elements");
    }
  }

  // Per-chunk activation-quant scratch, sized for the largest chunk (<=64 rows) and reused across
  // every chunk this call makes -- one arena bump, not one per chunk. Skipped entirely when a
  // pre-quantized buffer for this layout was provided (nothing to compute).
  uint16_t* f16_scratch = nullptr;   // w4a16; trellis: its parts, kMaxChunkM * K apart
  int8_t* i8_scratch = nullptr;      // w4a8
  float* i8_scale_scratch = nullptr;
  uint8_t* fp8_scratch = nullptr;    // mxfp4
  float* fp8_scale_scratch = nullptr;
  float* trellis_ws = nullptr;       // trellis: split-group partials
  if (!have_pre) {
    switch (w.layout) {
      case Layout::kBf16:
        break;
      case Layout::kW4a16:
        f16_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(kMaxChunkM * K));
        break;
      case Layout::kTrellis: {
        if (!trellis_pre) {
          f16_scratch = arena.Alloc<uint16_t>(
              static_cast<size_t>(w.trellis_parts * kMaxChunkM * K), /*align_bytes=*/16);
        }
        // docs/trellis-kernel.md 4.5: a 128-column group summed by more than one block (SKG > 1,
        // or a block narrower than 128 columns -- the Wc = 32 rows of out_proj/o, k/v and down,
        // even at SKG 1) meets in `ws`; every slot is written before it is read, so it is never
        // cleared.
        size_t ws_bytes = 0;
        for (int64_t m0 = 0; m0 < M; m0 += kMaxChunkM) {
          const int m = static_cast<int>(std::min(kMaxChunkM, M - m0));
          const LinearTuning t = TrellisChunkTuning(w, m);
          if (t.SKG > 1 || t.WV * t.NPW * 32 < 128) {
            ws_bytes = std::max(ws_bytes, core::r4d::GemmTrellisWsBytes(m, static_cast<int>(N),
                                                                         t.SKG));
          }
        }
        if (ws_bytes > 0) {
          trellis_ws = arena.Alloc<float>(ws_bytes / sizeof(float), /*align_bytes=*/16);
        }
        break;
      }
      case Layout::kW4a8:
        // align_bytes=16 (review finding, 2026-09-20): r4d_gemm_w4a8's fragment read
        // (global_load_b64) reads this buffer with a wide load -- same alignment class as
        // attention_layer.hpp's own decode scratch fix. Every preceding arena allocation in a
        // layer happens to be 16-aligned today (hidden/conv_dim/intermediate are all multiples of
        // 16 at 2 bytes/element), which is incidental, not enforced -- pass it explicitly instead.
        i8_scratch = arena.Alloc<int8_t>(static_cast<size_t>(kMaxChunkM * K), /*align_bytes=*/16);
        i8_scale_scratch = arena.Alloc<float>(static_cast<size_t>(kMaxChunkM));
        break;
      case Layout::kMxfp4:
        fp8_scratch = arena.Alloc<uint8_t>(static_cast<size_t>(kMaxChunkM * K), /*align_bytes=*/16);
        fp8_scale_scratch = arena.Alloc<float>(static_cast<size_t>(kMaxChunkM));
        break;
    }
  }

  for (int64_t m0 = 0; m0 < M; m0 += kMaxChunkM) {
    const int m = static_cast<int>(std::min(kMaxChunkM, M - m0));
    const uint16_t* xc = x + m0 * K;
    uint16_t* yc = y + m0 * N;
    // Picked per sub-chunk (not once for the whole call): a decode call (M=1) and a prefill call
    // (M up to 64, chunked here into <=64-row slices) want different WV/SK/MB/NPW even for the
    // same (layout,N,K) -- PickTuning's table is keyed by the exact per-launch row count `m`, not
    // the caller's total `M` (see linear.h's PickTuning comment on M-band rounding). A trellis
    // chunk's pick is also checked against the linear's part boundary (TrellisChunkTuning).
    const LinearTuning t = w.layout == Layout::kTrellis
                               ? TrellisChunkTuning(w, m)
                               : PickTuning(w.layout, N, K, m, w.w4a16_group);

    switch (w.layout) {
      case Layout::kBf16:
        core::r4d::GemmBf16NtM64(xc, w.bf16_w.data(), yc, m, static_cast<int>(K),
                                  static_cast<int>(N), t.WV, t.SK, t.MB, s);
        break;
      case Layout::kW4a16: {
        const uint16_t* a;
        if (have_pre) {
          a = reinterpret_cast<const uint16_t*>(pre->data) + m0 * K;
        } else {
          r4dx_model_cast_bf16_to_f16(reinterpret_cast<int64_t>(xc),
                                       reinterpret_cast<int64_t>(f16_scratch),
                                       static_cast<int64_t>(m) * K, reinterpret_cast<int64_t>(s));
          a = f16_scratch;
        }
        // quant2 Q3 (docs/quant2.md section 5.1): a linear at the build's default group -- every
        // linear of a container without __metadata__.quant.w4a16.groups -- keeps the historical
        // entry, so the default path launches exactly the kernel it always did. Only a linear the
        // container packed at another group goes through the per-group entry.
        const int g = EffectiveW4a16Group(w.w4a16_group);
        if (g == DefaultW4a16Group()) {
          core::r4d::GemmW4a16NtM64(a, w.wq.data(), w.w4a16_wsz.data(), yc, m,
                                     static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                     t.NPW, t.NT, s);
        } else {
          core::r4d::GemmW4a16NtM64G(g, a, w.wq.data(), w.w4a16_wsz.data(), yc, m,
                                      static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                      t.NPW, t.NT, s);
        }
        break;
      }
      case Layout::kW4a8: {
        const int8_t* a;
        const float* a_scale;
        if (have_pre) {
          a = reinterpret_cast<const int8_t*>(pre->data) + m0 * K;
          a_scale = pre->scale + m0;
        } else {
          core::r4d::QuantActI8(xc, i8_scratch, i8_scale_scratch, m, static_cast<int>(K), s);
          a = i8_scratch;
          a_scale = i8_scale_scratch;
        }
        core::r4d::GemmW4a8NtM64(a, a_scale, w.wq.data(), w.w4a8_ws.data(), yc, m,
                                  static_cast<int>(K), static_cast<int>(N), t.WV, t.SK, t.MB,
                                  t.NPW, t.NT, s);
        break;
      }
      case Layout::kMxfp4: {
        const uint8_t* a;
        const float* a_scale;
        if (have_pre) {
          a = reinterpret_cast<const uint8_t*>(pre->data) + m0 * K;
          a_scale = pre->scale + m0;
        } else {
          r4dx_quant_act_fp8e4m3_row(reinterpret_cast<int64_t>(xc),
                                      reinterpret_cast<int64_t>(fp8_scratch),
                                      reinterpret_cast<int64_t>(fp8_scale_scratch), m,
                                      static_cast<int>(K), reinterpret_cast<int64_t>(s));
          a = fp8_scratch;
          a_scale = fp8_scale_scratch;
        }
        core::r4d::GemmMxfp4a8NtM64(a, a_scale, w.mxfp4_wq.data(), w.mxfp4_ws.data(),
                                     w.mxfp4_wref.data(), yc, m, static_cast<int>(K),
                                     static_cast<int>(N), t.WV, t.SK, t.MB, t.NPW, s);
        break;
      }
      case Layout::kTrellis: {
        // docs/trellis-kernel.md 5.3 (v1): the input transform -- one launch for every part, A =
        // f16(H(x * suh_p) * 2^s / sqrt(128)) -- then one GEMM whose epilogue applies the output
        // side (FWHT, svh, 2^-s / sqrt(128), one bf16 rounding). Part 1 (mlp.gate_up's up) reads
        // its own A from output column part_n[0] on.
        const int parts = w.trellis_parts;
        const uint16_t* a0;
        const uint16_t* a1 = nullptr;
        if (trellis_pre) {
          const uint16_t* base = static_cast<const uint16_t*>(pre->data);
          a0 = base + m0 * K;
          if (parts > 1) a1 = base + pre->part_stride + m0 * K;
        } else {
          const int64_t part_stride = kMaxChunkM * K;
          const int64_t suh[2] = {reinterpret_cast<int64_t>(w.trellis_suh.data()),
                                  reinterpret_cast<int64_t>(w.trellis_suh.data() + K)};
          const int64_t out[2] = {reinterpret_cast<int64_t>(f16_scratch),
                                  reinterpret_cast<int64_t>(f16_scratch + part_stride)};
          r4dx_trellis_input_bf16(reinterpret_cast<int64_t>(xc), m, K, parts, suh, out,
                                  w.trellis_prescale_log2, reinterpret_cast<int64_t>(s));
          if (TrellisAStats* st = TrellisAStats::Get()) {
            st->Record(w, f16_scratch, parts, m, K, part_stride, s);
          }
          a0 = f16_scratch;
          if (parts > 1) a1 = f16_scratch + part_stride;
        }
        const float out_scale = static_cast<float>(std::ldexp(1.0, -w.trellis_prescale_log2) /
                                                   std::sqrt(128.0));
        const int n_split = static_cast<int>(parts > 1 ? w.trellis_part_n[0] : N);
        core::r4d::GemmTrellisNtM64(a0, a1, n_split, w.trellis_w.data(), w.trellis_svh.data(), yc,
                                    trellis_ws, w.trellis_tickets, m, static_cast<int>(K),
                                    static_cast<int>(N), w.trellis_bits, t.WV, t.SK, t.MB, t.NPW,
                                    t.SKG, t.U, t.NT, out_scale, s);
        break;
      }
    }
  }
}

}  // namespace r4dx::model
