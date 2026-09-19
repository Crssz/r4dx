"""Shared float32-exact helpers for the r4dx-convert Python reference implementations.

Every function here is written to match r4dx-convert's C++ (src/convert/include/r4dx_convert/
quant_int4.hpp, quant_mxfp4.hpp) bit-for-bit: same dtype (float32 throughout, never float64,
because a stray Python-float promotion changes the last bit of a division or a round) and the same
round-half-away-from-zero tie-break (NOT numpy's default round-half-to-even), because that is what
C++'s `std::floor(x+0.5f)` / `std::ceil(x-0.5f)` computes.

Run with the read-only reference venv:
  C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe
"""
import numpy as np

F32 = np.float32


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """float32 array -> int32 array, matching r4dx_convert::RoundHalfAwayFromZero exactly."""
    x = x.astype(F32)
    out = np.empty(x.shape, dtype=np.int32)
    pos = x >= F32(0.0)
    out[pos] = np.floor(x[pos] + F32(0.5)).astype(np.int32)
    out[~pos] = np.ceil(x[~pos] - F32(0.5)).astype(np.int32)
    return out


def f16_bits(x) -> int:
    """float -> IEEE-754 binary16 bit pattern, round-to-nearest-even (matches
    r4dx::core::FloatToF16 -- both are correctly-rounded fp32->fp16 conversions)."""
    return int(np.array(F32(x)).astype(np.float16).view(np.uint16))
