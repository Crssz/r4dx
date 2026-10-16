// Decode-path fusion switches (docs/qwen4exp.md "Decode performance"). Every fusion is on by default and bit-identical to the
// unfused chain it replaces; R4DX_Q4_FUSE_<NAME>=0 turns one off. The environment is re-read on every call so a test (or the
// in-process A/B bench) can flip a switch between runs; Qwen4ExpStage re-captures its decode graphs when the mask changes.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace r4dx::model::qwen4exp {

enum FuseBit : uint32_t {
  kFuseHc = 1u << 0,   // R4DX_Q4_FUSE_HC: hyper-connection Prep as 3 launches (norm, down + inject GEMM with silu / gate, up GEMM with mix)
  kFuseQsa = 1u << 1,  // R4DX_Q4_FUSE_QSA: decode attention spreads each kv head's twelve query heads over four workgroups
};

inline bool FuseEnvOn(const char* name) {
  const char* e = std::getenv(name);
  return !(e != nullptr && e[0] == '0' && e[1] == '\0');
}

inline uint32_t FuseMask() {
  uint32_t m = 0;
  if (FuseEnvOn("R4DX_Q4_FUSE_HC")) m |= kFuseHc;
  if (FuseEnvOn("R4DX_Q4_FUSE_QSA")) m |= kFuseQsa;
  return m;
}

}  // namespace r4dx::model::qwen4exp
