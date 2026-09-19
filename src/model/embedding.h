// r4dx::model::EmbedTokens -- host embedding gather (r4dx::kernels::EmbeddingGatherHost,
// docs/architecture.md "text.embed_tokens (host table; device-side gather for batches)") plus the
// async upload to device the task's "embedding gather+upload" item asks for. Header-only: both
// steps are a handful of lines around an existing kernels/ helper, not worth a .cpp.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/embedding.hpp"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

// table: [vocab, hidden] bf16 host pointer (Container::EmbedTokensHost()). staging: caller-owned
// pinned buffer, capacity >= token_ids.size()*hidden (kept across calls so a decode loop does not
// re-allocate pinned memory every step). out: caller-owned device buffer, same capacity; `out`'s
// first token_ids.size()*hidden elements are valid once `stream` reaches this point.
inline void EmbedTokens(core::Stream& stream, const uint16_t* table, int64_t vocab, int64_t hidden,
                         const std::vector<int32_t>& token_ids,
                         core::PinnedBuffer<uint16_t>& staging, core::DeviceBuffer<uint16_t>& out) {
  const size_t n = token_ids.size() * static_cast<size_t>(hidden);
  if (staging.size() < n || out.size() < n) {
    throw std::runtime_error("r4dx::model::EmbedTokens: staging/out buffer smaller than token_ids.size()*hidden");
  }
  kernels::EmbeddingGatherHost(table, vocab, hidden, token_ids, staging.data());
  out.CopyFromHostAsync(staging.data(), n, stream);
}

// Device-resident counterpart (MTP device-residency pass, docs/mtp.md "device-resident draft
// loop"): table_dev is a DEVICE [vocab, hidden] bf16 pointer (Container::EmbedTokensDevice(),
// non-null only when the container was loaded with embed_device_resident and it fit in VRAM --
// see Container::Load's own comment). ids_dev is a DEVICE int32[n] pointer -- may be
// r4dx_argmax_f32's own out_idx output directly, with no host round trip at all. out_dev: caller-
// owned device buffer, capacity >= n*hidden. Thin wrapper around r4dx_embedding_gather_bf16
// (src/kernels) purely for call-site symmetry with EmbedTokens above.
inline void EmbedTokensDeviceGather(core::Stream& stream, const uint16_t* table_dev, int64_t hidden,
                                     const int32_t* ids_dev, int64_t n, uint16_t* out_dev) {
  r4dx_embedding_gather_bf16(reinterpret_cast<int64_t>(table_dev), reinterpret_cast<int64_t>(ids_dev),
                              reinterpret_cast<int64_t>(out_dev), n, hidden,
                              reinterpret_cast<int64_t>(stream.get()));
}

}  // namespace r4dx::model
