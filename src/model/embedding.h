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

}  // namespace r4dx::model
