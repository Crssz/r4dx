"""CPU tests of the pack-quantized int4 dequant (plan M0-4, Python half). No GPU, no real weights.

    $env:HIP_VISIBLE_DEVICES=''; D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe -m pytest tools\\reference\\qwen4exp\\test_awq_dequant.py -q

Needs compressed-tensors (only installed in r4dx-qwen4-ref).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402
from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor  # noqa: E402
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme  # noqa: E402
from compressed_tensors.quantization.utils import calculate_qparams  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import awq_dequant as ad  # noqa: E402

# config.json quantization_config.config_groups.group_0.weights of the checkpoint (rev 598afc96).
CHECKPOINT_WEIGHT_ARGS = {
    "actorder": None, "block_structure": None, "dynamic": False, "group_size": 128, "num_bits": 4,
    "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
    "symmetric": False, "type": "int", "zp_dtype": "torch.int8",
}


def _scheme() -> QuantizationScheme:
    return QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**CHECKPOINT_WEIGHT_ARGS))


def _ct_compress(weight: torch.Tensor, scheme: QuantizationScheme) -> dict[str, torch.Tensor]:
    """Min/max qparams per group + PackedQuantizationCompressor.compress, as compressed-tensors does."""
    args = scheme.weights
    n, k = weight.shape
    grouped = weight.view(n, k // args.group_size, args.group_size)
    scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    assert scale.dtype == torch.bfloat16 and zp.dtype == torch.int8
    return PackedQuantizationCompressor.compress(
        {"weight": weight, "weight_scale": scale, "weight_zero_point": zp}, scheme)


def _ct_decompress(state: dict[str, torch.Tensor], scheme: QuantizationScheme) -> torch.Tensor:
    return PackedQuantizationCompressor.decompress(dict(state), scheme)["weight"]


def _bits(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.int16)


def _random_weight(n: int, k: int, seed: int, offset: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.02 + offset
    w[0, : ad.GROUP_SIZE] = 0.0  # one all-zero group: eps scale path
    w[1, ad.GROUP_SIZE: 2 * ad.GROUP_SIZE] = w[1, ad.GROUP_SIZE: 2 * ad.GROUP_SIZE].abs()  # min clamped to 0
    return w.to(torch.bfloat16)


@pytest.mark.parametrize("n,k,offset", [(640, 2560, 0.0), (2560, 640, 0.0), (12288, 2560, 0.0),
                                        (640, 2560, 0.01)])
def test_matches_compressed_tensors_bit_for_bit(n, k, offset):
    scheme = _scheme()
    state = _ct_compress(_random_weight(n, k, seed=n * 7 + k, offset=offset), scheme)
    assert state["weight_packed"].dtype == torch.int32 and tuple(state["weight_packed"].shape) == (n, k // 8)
    assert state["weight_scale"].dtype == torch.bfloat16 and tuple(state["weight_scale"].shape) == (n, k // 128)
    assert state["weight_zero_point"].dtype == torch.int32
    assert tuple(state["weight_zero_point"].shape) == (n // 8, k // 128)
    assert state["weight_shape"].tolist() == [n, k]

    ref = _ct_decompress(state, scheme)
    assert ref.dtype == torch.bfloat16 and tuple(ref.shape) == (n, k)
    out = ad.dequant_pack_quantized(state["weight_packed"], state["weight_scale"],
                                    state["weight_zero_point"], state["weight_shape"])
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (n, k)
    assert torch.equal(_bits(out), _bits(ref))
    assert ref.abs().max() > 0  # non-vacuous


def _pack_words(nibbles: list[int]) -> int:
    """8 unsigned nibbles -> signed int32, nibble j at bits [4j, 4j+4)."""
    assert len(nibbles) == 8
    word = sum((u & 0xF) << (4 * j) for j, u in enumerate(nibbles))
    return word - (1 << 32) if word >= (1 << 31) else word


def _hand_case():
    """N=9 (zero-point packing pads 9 rows to 16), K=256 (2 groups); nibbles by formula."""
    n, k, gs = 9, 256, 128
    groups = k // gs
    q_u = [[(3 * r + c) % 16 for c in range(k)] for r in range(n)]
    zp_u = [[(r + 5 * g) % 16 for g in range(groups)] for r in range(n)]
    scale = [[2.0 ** -(r % 4 + 3 * g + 1) for g in range(groups)] for r in range(n)]
    packed = torch.tensor([[_pack_words(q_u[r][8 * w: 8 * w + 8]) for w in range(k // 8)] for r in range(n)],
                          dtype=torch.int32)
    zp_rows = zp_u + [[0] * groups] * (16 - n)  # padding nibbles
    zp_packed = torch.tensor([[_pack_words([zp_rows[8 * w + j][g] for j in range(8)]) for g in range(groups)]
                              for w in range(2)], dtype=torch.int32)
    expected = torch.tensor([[(q_u[r][c] - zp_u[r][c // gs]) * scale[r][c // gs] for c in range(k)]
                             for r in range(n)], dtype=torch.float64).to(torch.bfloat16)
    state = {"weight_packed": packed, "weight_scale": torch.tensor(scale, dtype=torch.bfloat16),
             "weight_zero_point": zp_packed, "weight_shape": torch.tensor([n, k])}
    return state, expected, q_u, zp_u


def test_hand_built_packing_order():
    state, expected, q_u, zp_u = _hand_case()
    # Pin the literal layout: row 0 word 0 holds q_u[0][0..7] = 0..7 low nibble first.
    assert int(state["weight_packed"][0, 0]) == 0x76543210
    # zero-point word (0, g=0) holds rows 0..7 of group 0: zp_u[r][0] = r.
    assert int(state["weight_zero_point"][0, 0]) == 0x76543210
    assert int(state["weight_zero_point"][1, 0]) == 8  # row 8 -> nibble 0 of word 1, rest padding 0
    assert ad.unpack_nibbles_dim1(state["weight_packed"], 256).tolist() == q_u

    out = ad.dequant_pack_quantized(**state)
    assert torch.equal(_bits(out), _bits(expected))
    # Spot values: W[r, c] = (q_u - zp_u) * scale.
    assert float(out[0, 7]) == (7 - 0) * 2.0 ** -1
    assert float(out[8, 200]) == (((24 + 200) % 16) - ((8 + 5) % 16)) * 2.0 ** -4
    # compressed-tensors agrees with the hand layout.
    assert torch.equal(_bits(_ct_decompress(state, _scheme())), _bits(expected))


def test_rejects_malformed():
    state, _, _, _ = _hand_case()
    with pytest.raises(ValueError):
        ad.dequant_pack_quantized(state["weight_packed"], state["weight_scale"].float(),
                                  state["weight_zero_point"], state["weight_shape"])
    with pytest.raises(ValueError):
        ad.dequant_pack_quantized(state["weight_packed"][:, :-1], state["weight_scale"],
                                  state["weight_zero_point"], state["weight_shape"])
    with pytest.raises(ValueError):
        ad.dequant_pack_quantized(state["weight_packed"], state["weight_scale"],
                                  state["weight_zero_point"][:1], state["weight_shape"])


def test_iter_dequant_shard(tmp_path):
    scheme = _scheme()
    tensors: dict[str, torch.Tensor] = {}
    refs = {}
    for i, (n, k) in enumerate([(640, 2560), (2560, 640)]):
        prefix = f"model.language_model.layers.{i}.mlp.experts.{i}.up_proj"
        state = _ct_compress(_random_weight(n, k, seed=100 + i), scheme)
        refs[f"{prefix}.weight"] = _ct_decompress(state, scheme)
        tensors.update({f"{prefix}.{name}": t.contiguous() for name, t in state.items()})
    tensors["model.language_model.layers.0.input_layernorm.weight"] = torch.ones(2560, dtype=torch.bfloat16)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard))

    got = dict(ad.iter_dequant_shard(shard))
    assert set(got) == set(refs)
    for name, ref in refs.items():
        assert torch.equal(_bits(got[name]), _bits(ref)), name
