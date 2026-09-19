// r4dx::kernels::EmbeddingGatherHost -- embedding lookup is host-side (docs/architecture.md:
// "text.embed_tokens (host table; device-side gather for batches)"). `text.embed_tokens` is
// small enough (vocab 248320 * hidden 5120 * 2 bytes bf16 ~= 2.5 GiB... actually kept host-
// resident specifically so it never occupies device memory) that a per-step gather of a handful
// of rows into a pinned staging buffer, then one async H2D copy, is cheaper than keeping the
// whole table on the GPU. A device-side gather kernel for batched prefill is a straightforward
// follow-up (TODO) once src/model needs to gather more than a few rows per step.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace r4dx::kernels {

// table: [vocab, hidden] bf16 host pointer (text.embed_tokens). token_ids: token ids to gather,
// each required to be in [0, vocab). out_host: caller-owned [token_ids.size() * hidden] bf16
// buffer (ordinarily an r4dx::core::PinnedBuffer<uint16_t>) that receives the gathered rows,
// ready for DeviceBuffer::CopyFromHostAsync.
inline void EmbeddingGatherHost(const uint16_t* table, int64_t vocab, int64_t hidden,
                                 const std::vector<int32_t>& token_ids, uint16_t* out_host) {
  for (size_t t = 0; t < token_ids.size(); ++t) {
    int32_t id = token_ids[t];
    if (id < 0 || id >= vocab) {
      throw std::out_of_range("EmbeddingGatherHost: token id out of [0, vocab) range");
    }
    const uint16_t* row = table + static_cast<int64_t>(id) * hidden;
    std::memcpy(out_host + t * static_cast<size_t>(hidden), row, hidden * sizeof(uint16_t));
  }
}

}  // namespace r4dx::kernels
