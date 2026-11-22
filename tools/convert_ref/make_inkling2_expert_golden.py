#!/usr/bin/env python
"""Goldens for tests/convert/test_inkling2_expert_codec.cpp: small NVFP4 / MXFP8 expert groups (hand-built bytes that hit
every nibble, e4m3 subnormals, scale edges), decoded by transformers' OWN `Inkling2LoopDequantizeExperts` (float32) and
by an independent numpy port, which must agree bit for bit; the w13 cases are also passed through the real
`Interleave(dim=1)`. Writes tests/convert/data/inkling2_expert_golden.json (f32 results as uint32 bit patterns).

  D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools/convert_ref/make_inkling2_expert_golden.py --out <json>
"""
import argparse
import json

import numpy as np
import torch
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


class _Model:  # the op asks the model for the parameter's dtype; float32 here (HF would use the param dtype)
    @staticmethod
    def get_parameter(_name):
        return torch.zeros(1, dtype=torch.float32)


def e4m3_values(codes):
    return torch.from_numpy(np.ascontiguousarray(codes, dtype=np.uint8)).view(torch.float8_e4m3fn).float().numpy()


def hf_decode(weight, scale, amax, denom, weight_dtype, scale_dtype):
    op = Inkling2LoopDequantizeExperts()
    w = torch.from_numpy(weight).view(weight_dtype)
    s = torch.from_numpy(scale).view(scale_dtype)
    srcs = [r"w_weight", r"w_scale", r"w_amax", r"w_den"]
    inp = {srcs[0]: w, srcs[1]: s}
    if amax is not None:
        inp[srcs[2]] = torch.from_numpy(amax)
        inp[srcs[3]] = torch.from_numpy(denom)
    else:
        inp = {srcs[0]: w, srcs[1]: s}
    # the op unpacks (weight, scale, amax, denominator) from the source patterns in order; MXFP8 has no amax
    tensors = dict(inp)
    for k in srcs:
        tensors.setdefault(k, None)
    out = op.convert(tensors, srcs, ["out"], model=_Model(), full_layer_name="x")
    return out["out"].numpy()


def np_decode(weight, scale, amax, denom):
    """Independent port: same f32 operation order."""
    if amax is not None:
        e, rows, half = weight.shape
        cols = half * 2
        lo, hi = E2M1[weight & 0xF], E2M1[weight >> 4]
        vals = np.stack([lo, hi], axis=-1).reshape(e, rows, cols // 16, 16)
        vals = vals * e4m3_values(scale)[..., None]
        factor = (amax / denom).astype(np.float32)
        vals = vals * factor[:, None, None, None]
        return vals.reshape(e, rows, cols).astype(np.float32)
    e, rows, cols = weight.shape
    vals = e4m3_values(weight).reshape(e, rows, cols // 32, 32)
    vals = vals * np.exp2(scale.astype(np.float32) - 127.0)[..., None]
    return vals.reshape(e, rows, cols).astype(np.float32)


def deinterleave(x):
    return np.concatenate([x[:, 0::2], x[:, 1::2]], axis=1)


def hf_interleave(x):
    return Interleave(dim=1).convert({"x": torch.from_numpy(x)}, ["x"], ["y"])["y"].numpy()


def valid_e4m3(rng, shape):
    c = rng.integers(0, 256, size=shape).astype(np.uint8)
    return np.where((c & 0x7F) == 0x7F, c & 0x80, c).astype(np.uint8)  # NaN codes -> +-0


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0xE41)
    cases = []

    def nvfp4_case(name, rows, cols, interleave):
        e = 3
        weight = rng.integers(0, 256, size=(e, rows, cols // 2)).astype(np.uint8)
        weight[0, 0, :] = np.arange(cols // 2) * 17 % 256          # every nibble pattern appears
        scale = valid_e4m3(rng, (e, rows, cols // 16))
        scale[0, 0, 0], scale[0, 0, 1] = 0x01, 0x7E                # smallest subnormal, largest finite
        amax = np.array([0.5, 3.0, 1.0e-3], dtype=np.float32)
        denom = np.array([2688.0], dtype=np.float32)
        hf = hf_decode(weight, scale, amax, denom, torch.uint8, torch.float8_e4m3fn)
        ref = np_decode(weight, scale, amax, denom)
        assert hf.dtype == np.float32 and np.array_equal(hf.view(np.uint32), ref.view(np.uint32)), name
        c = {"name": name, "format": "nvfp4", "experts": e, "rows": rows, "cols": cols, "interleaved_w13": interleave,
             "weight_hex": weight.tobytes().hex(), "scale_hex": scale.tobytes().hex(), "amax": amax.tolist(),
             "amax_bits": bits(amax), "denominator": 2688.0, "decoded_bits": bits(hf)}
        if interleave:
            de = hf_interleave(hf)
            assert np.array_equal(de.view(np.uint32), deinterleave(hf).view(np.uint32))
            c["deinterleaved_bits"] = bits(de)
        cases.append(c)

    def mxfp8_case(name, rows, cols, interleave):
        e = 3
        weight = valid_e4m3(rng, (e, rows, cols))
        scale = rng.integers(100, 141, size=(e, rows, cols // 32)).astype(np.uint8)
        scale[0, 0, 0] = 0                                          # 2^-127, subnormal in f32
        scale[0, 1, 0] = 254                                        # 2^127 ...
        weight[0, 1, 0:32] = rng.integers(0, 0x30, size=32).astype(np.uint8)  # ... with small codes (finite)
        weight[0, 0, 0:4] = [0x01, 0x80, 0x07, 0xFE]                # subnormal, -0, largest subnormal, -448
        hf = hf_decode(weight, scale, None, None, torch.float8_e4m3fn, torch.uint8)
        ref = np_decode(weight, scale, None, None)
        assert hf.dtype == np.float32 and np.array_equal(hf.view(np.uint32), ref.view(np.uint32)), name
        c = {"name": name, "format": "mxfp8", "experts": e, "rows": rows, "cols": cols, "interleaved_w13": interleave,
             "weight_hex": weight.tobytes().hex(), "scale_hex": scale.tobytes().hex(), "decoded_bits": bits(hf)}
        if interleave:
            de = hf_interleave(hf)
            assert np.array_equal(de.view(np.uint32), deinterleave(hf).view(np.uint32))
            c["deinterleaved_bits"] = bits(de)
            c["raw_codes_deinterleaved_hex"] = deinterleave(weight).tobytes().hex()
            c["raw_scales_deinterleaved_hex"] = deinterleave(scale).tobytes().hex()
        cases.append(c)

    nvfp4_case("nvfp4_w13", 8, 32, True)
    nvfp4_case("nvfp4_w2", 4, 48, False)
    mxfp8_case("mxfp8_w13", 8, 64, True)
    mxfp8_case("mxfp8_w2", 4, 32, False)
    with open(a.out, "w") as f:
        json.dump({"note": "transformers Inkling2LoopDequantizeExperts (float32) == numpy port; Interleave(dim=1) for w13",
                   "cases": cases}, f)
    print("wrote", a.out, len(cases), "cases")


if __name__ == "__main__":
    main()
