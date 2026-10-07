// The one-copy per-call metadata of a decode-sized Model call (decode-t1 item 5, docs/perf.md): the
// call's embed ids, its full-attention layers' KV positions (== slot mapping == rope positions) and the
// attention seqused_k, laid out in one int32 array so a single async H2D from pinned memory replaces an
// id copy plus two blocking hipMemcpy uploads. Header-only and HIP-free so tests/model/test_decode_legacy
// can pin the layout on the host.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace r4dx::model {

// int32 element offsets into the array (and its device mirror, Model::step_meta_dev_). Up to 64 rows:
// ids at [kStepMetaIds, +64), positions at [kStepMetaPos, +64), seqused_k at kStepMetaSeq. Each section
// starts on a 256-byte boundary of the (hipMalloc'd, so 256-byte aligned) device array.
inline constexpr int64_t kStepMetaIds = 0;
inline constexpr int64_t kStepMetaPos = 64;
inline constexpr int64_t kStepMetaSeq = 128;
inline constexpr int64_t kStepMetaRows = 64;
// The length of the one copy: through seqused_k, padded to a multiple of 4 elements (16 bytes).
inline constexpr int64_t kStepMetaInts = kStepMetaSeq + 4;

// Fills `h` (kStepMetaInts elements) for a call over `ids` starting at sequence position `pos`:
// positions[t] = pos + t, seqused_k = pos + T. The unused rows and the padding are left as they are.
inline void FillStepMeta(int32_t* h, const std::vector<int32_t>& ids, int64_t pos) {
  const int64_t T = static_cast<int64_t>(ids.size());
  if (T < 1 || T > kStepMetaRows) throw std::runtime_error("FillStepMeta: 1..64 rows");
  for (int64_t t = 0; t < T; ++t) {
    h[kStepMetaIds + t] = ids[static_cast<size_t>(t)];
    h[kStepMetaPos + t] = static_cast<int32_t>(pos + t);
  }
  h[kStepMetaSeq] = static_cast<int32_t>(pos + T);
}

}  // namespace r4dx::model
