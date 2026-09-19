// r4dx::model::SpanAccumulator / SpanEntry / ProfiledCall -- shared hipEvent-pair GPU timing
// helper, factored out of Model::DecodeStepProfiled's original anonymous-namespace class (tools/
// profile pass, 2026-09-19) so GdnLayer::Forward, AttentionLayer::Forward and Mlp::Forward can also
// record per-kernel spans into the SAME accumulator Model::DecodeStepProfiled/PrefillProfiled owns
// (Milestone 3 profiling-truth pass, docs/r9700.md R5/Q2/Q3/Q7, 2026-09-20).
//
// Spans are recorded async (no per-span sync -- that would serialize the pipeline and skew exactly
// the numbers this is trying to measure); Finish() does ONE hipEventSynchronize at the very end,
// then reads back every pair's elapsedTime and accumulates it into the named bucket (several calls
// under the same name -- e.g. one per GDN layer, or one per prefill chunk -- sum into one entry).
//
// Naming convention (Q7): any span name prefixed "gemm:" is one of this model's weight-streaming
// r4d_gemm_*_nt_m64 GEMMs (in/out projections, gate_up/down, lm_head) dispatched through
// r4dx::model::ApplyLinear or a plain bf16 GemmBf16NtM64 call. Callers that want a "GEMM share vs
// non-GEMM share" split (docs/r9700.md Q7's prefill ask) sum entries by that prefix rather than by
// an explicit allowlist -- see src/cli/main.cpp's profile-report code for the one place this is
// done.
//
// Zero-overhead when not profiling: ProfiledCall(nullptr, ...) below calls `fn()` directly with no
// hipEvent, no std::function materialized, no extra branch cost beyond one null check -- every real
// (non---profile) DecodeStep/DecodeStepGreedy/Prefill call site passes prof=nullptr through
// GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward's new trailing parameter, so this
// instrumentation existing in the code does not touch the hot path's generated code beyond that one
// branch.
#pragma once

#include <hip/hip_runtime.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "r4dx/core/error.hpp"

namespace r4dx::model {

struct SpanEntry {
  std::string name;
  double ms = 0.0;
  int count = 0;
};

class SpanAccumulator {
 public:
  void Add(hipStream_t stream, const std::string& name, const std::function<void()>& body) {
    hipEvent_t start, end;
    R4DX_HIP_CHECK(hipEventCreate(&start));
    R4DX_HIP_CHECK(hipEventCreate(&end));
    R4DX_HIP_CHECK(hipEventRecord(start, stream));
    body();
    R4DX_HIP_CHECK(hipEventRecord(end, stream));
    pending_.push_back({name, start, end});
  }

  std::vector<SpanEntry> Finish() {
    if (!pending_.empty()) {
      R4DX_HIP_CHECK(hipEventSynchronize(pending_.back().end));
    }
    std::vector<SpanEntry> out;
    std::unordered_map<std::string, size_t> index;
    for (auto& p : pending_) {
      float ms = 0.0f;
      R4DX_HIP_CHECK(hipEventElapsedTime(&ms, p.start, p.end));
      static_cast<void>(hipEventDestroy(p.start));
      static_cast<void>(hipEventDestroy(p.end));
      auto it = index.find(p.name);
      if (it == index.end()) {
        index[p.name] = out.size();
        out.push_back({p.name, static_cast<double>(ms), 1});
      } else {
        out[it->second].ms += ms;
        out[it->second].count += 1;
      }
    }
    pending_.clear();
    return out;
  }

 private:
  struct Pending {
    std::string name;
    hipEvent_t start, end;
  };
  std::vector<Pending> pending_;
};

// See file comment: nullptr `prof` calls `fn()` directly, no instrumentation overhead.
template <typename Fn>
inline void ProfiledCall(SpanAccumulator* prof, hipStream_t stream, const char* name, Fn&& fn) {
  if (prof != nullptr) {
    prof->Add(stream, name, std::function<void()>(std::forward<Fn>(fn)));
  } else {
    fn();
  }
}

}  // namespace r4dx::model
