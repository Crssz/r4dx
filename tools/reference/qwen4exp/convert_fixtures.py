"""Fixtures for r4dx-convert's qwen4_exp branch (plan M1-2): tests/convert/test_convert_qwen4exp*.cpp.

CPU only (torch's CPU device; torch.cuda is never touched). Run with the reference venv:

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools/reference/qwen4exp/convert_fixtures.py awq-mini
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools/reference/qwen4exp/convert_fixtures.py tiny-trellis

Output (override with --out-dir): E:\\models\\r4dx\\q4exp-fixtures\\ConvertQ4\\

awq-mini-trellis -> <out>/awq-mini-trellis-<form>/   the tiny-trellis oracle (below) for the awq-mini checkpoint, its weights being
    the dequantized quads (so the converter's reconstruction check reads them through the virtual AWQ source)

awq-mini -> <out>/awq-mini/
    hf/                        a qwen4_exp checkpoint in miniature (hidden 256, 2 layers: GDN + QSA, 2 experts) in the
                               AWQ checkpoint's format: every body linear is a compressed-tensors `pack-quantized`
                               int4 / group 128 asymmetric quad written by compressed-tensors' own
                               PackedQuantizationCompressor, two shards plus mtp / vision / n-gram tensors the converter
                               must skip; config.json carries the real quantization_config.
    expected_bf16.safetensors  `<b>.weight` bf16 of every quad: awq_dequant.dequant_pack_quantized (the reference the C++
                               dequant must equal bit for bit)
    sidecar.r4dx-ngram         a small n-gram sidecar (ngram_sidecar.SidecarHeader + random rows) matching the config
    expected.json              the linears (`<b>` -> [N, K]) and the sha256 of the sidecar body

real-spot -> <out>/real-spot/expected_bf16.safetensors
    awq_dequant.dequant_pack_quantized of a handful of tensors of the REAL checkpoint (E:\\models\\r4dx\\Swift-1.5-Qwen3.8-Flash-Next-
    W4A16-AWQ: GDN in_proj_qkv, QSA q_proj and o_proj, experts' gate/up/down at assorted layers), `<b>.weight` bf16.

tiny-trellis -> <out>/tiny-trellis-<form>/   (needs E:\\models\\r4dx\\tiny-q4exp\\hf, tools/reference/qwen4exp/tiny.py's output)
    weights_override.json + L<ii>.safetensors  an oracle in trellis_quant.py's MIX manifest form for every body linear of the
                               tiny checkpoint (attn q/k/v/o, gdn in_proj_qkv/z/out_proj, every expert gate/up/down). The
                               trellis words are RANDOM ring words, not an encoder's output (no Hessians exist for the tiny
                               model; the converter's import and reconstruction check do not care how good the bits are): each
                               record's rel_weight_err is the real trellis_quant.rel_cos of W_hat (decoded from those words,
                               trellis_quant.reconstruct) against the tiny checkpoint's actual weight, which is exactly what
                               the converter's check recomputes. Forms: uniform4 (all KB 4), mix34 (experts of layers 0-1 at
                               KB 3, everything else KB 4: one KB per layer and class), bad (mix34 but expert 3 of layer 2 (mini: expert 1 of layer 1) at
                               KB 5: the converter must refuse, naming it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import awq_dequant as ad  # noqa: E402
import ngram_sidecar as ns  # noqa: E402

DEFAULT_OUT = Path(r"E:\models\r4dx\q4exp-fixtures\ConvertQ4")
TINY_HF = Path(r"E:\models\r4dx\tiny-q4exp\hf")
BF = torch.bfloat16
PREFIX = "model.language_model.layers."

# ---- the miniature -------------------------------------------------------------------------------------------
H = 256
VOCAB = 128
LAYERS = 2
LAYER_TYPES = ["linear_attention", "full_attention"]
HEADS, KV, HEAD_DIM = 2, 1, 128
EXPERTS, MOE_INTER, SHARED_INTER = 2, 128, 128
HC, HC_LOWRANK = 4, 32
W = HC * H
LIN_K_HEADS, LIN_V_HEADS, LIN_DIM, CONV = 2, 4, 128, 4
IDX_HEADS, IDX_KV, IDX_DIM = 4, 1, 128
PLE_DIM, PLE_CONV = 256, 4
PLE_LAYER = 0  # ple_layer_ids [1]
QKV = 2 * LIN_K_HEADS * LIN_DIM + LIN_V_HEADS * LIN_DIM
Z = LIN_V_HEADS * LIN_DIM
EOS = 127
NGRAM_ROWS = 1000

CHECKPOINT_WEIGHT_ARGS = {
    "actorder": None, "block_structure": None, "dynamic": False, "group_size": 128, "num_bits": 4,
    "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
    "symmetric": False, "type": "int", "zp_dtype": "torch.int8",
}


def quantization_config(ignore: list[str]) -> dict:
    return {
        "config_groups": {"group_0": {"format": "pack-quantized", "input_activations": None,
                                      "output_activations": None, "targets": ["Linear"],
                                      "weights": CHECKPOINT_WEIGHT_ARGS}},
        "format": "pack-quantized", "global_compression_ratio": None, "ignore": ignore, "kv_cache_scheme": None,
        "quant_method": "compressed-tensors", "quantization_status": "compressed", "sparsity_config": {},
        "transform_config": {}, "version": "0.18.1a20260919",
    }


def mini_config(ignore: list[str]) -> dict:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"], "language_model_only": False,
        "model_type": "qwen4_exp", "tie_word_embeddings": False, "quantization_config": quantization_config(ignore),
        "text_config": {
            "model_type": "qwen4_exp_text", "hidden_size": H, "num_hidden_layers": LAYERS, "layer_types": LAYER_TYPES,
            "vocab_size": VOCAB, "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HEAD_DIM,
            "num_experts": EXPERTS, "num_experts_per_tok": 2, "moe_intermediate_size": MOE_INTER,
            "shared_expert_intermediate_size": SHARED_INTER, "hc_count": HC, "hc_lowrank": HC_LOWRANK,
            "linear_num_key_heads": LIN_K_HEADS, "linear_num_value_heads": LIN_V_HEADS,
            "linear_key_head_dim": LIN_DIM, "linear_value_head_dim": LIN_DIM, "linear_conv_kernel_dim": CONV,
            "indexer_n_heads": IDX_HEADS, "indexer_kv_heads": IDX_KV, "indexer_head_dim": IDX_DIM,
            "ple_embed_dim": PLE_DIM, "ple_conv_kernel_size": PLE_CONV, "ple_layer_ids": [PLE_LAYER + 1],
            "eos_token_id": EOS, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
        },
    }


def scheme():
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    return QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**CHECKPOINT_WEIGHT_ARGS))


def ct_compress(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """compressed-tensors' own min/max qparams + PackedQuantizationCompressor.compress."""
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization.utils import calculate_qparams

    sch = scheme()
    args = sch.weights
    n, k = weight.shape
    grouped = weight.view(n, k // args.group_size, args.group_size)
    scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    return PackedQuantizationCompressor.compress({"weight": weight, "weight_scale": scale, "weight_zero_point": zp}, sch)


def rand_weight(gen: torch.Generator, n: int, k: int, kind: int) -> torch.Tensor:
    w = torch.randn(n, k, generator=gen) * 0.02
    if kind % 3 == 1:
        w += 0.01  # a positive offset: asymmetric zero points away from the middle
    if kind % 3 == 2:
        w[0, :128] = 0.0  # an all-zero group: the eps scale path
        w[1, 128:256] = w[1, 128:256].abs()  # min clamped to 0
    return w.to(BF)


def save_st(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata={"format": "pt"})


def build_awq_mini(out: Path) -> None:
    import shutil

    if out.exists():
        shutil.rmtree(out)
    (out / "hf").mkdir(parents=True)
    gen = torch.Generator().manual_seed(20261011)
    shards: dict[str, dict[str, torch.Tensor]] = {"model-00001-of-00002.safetensors": {},
                                                  "model-00002-of-00002.safetensors": {}, "model_mtp.safetensors": {}}
    expected: dict[str, torch.Tensor] = {}
    linears: dict[str, list[int]] = {}
    ignore: list[str] = []
    counter = [0]

    def put(shard: str, name: str, t: torch.Tensor) -> None:
        shards[shard][name] = t

    def bf(shard: str, name: str, *shape: int, sigma: float = 0.05) -> None:
        put(shard, name, (torch.randn(*shape, generator=gen) * sigma).to(BF))

    def quad(shard: str, base: str, n: int, k: int) -> None:
        w = rand_weight(gen, n, k, counter[0])
        counter[0] += 1
        st = ct_compress(w)
        for suffix in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
            put(shard, f"{base}.{suffix}", st[suffix].contiguous())
        expected[f"{base}.weight"] = ad.dequant_pack_quantized(st["weight_packed"], st["weight_scale"],
                                                               st["weight_zero_point"], st["weight_shape"].tolist())
        linears[base] = [n, k]

    for i in range(LAYERS):
        shard = "model-00001-of-00002.safetensors" if i == 0 else "model-00002-of-00002.safetensors"
        p = f"{PREFIX}{i}."
        for blk in ("attn", "mlp"):
            h = f"{p}{blk}_hyper_connection."
            bf(shard, h + "hc_norm.weight", W, sigma=0.3)
            bf(shard, h + "input_mix_weight_down.weight", HC_LOWRANK, W)
            bf(shard, h + "input_mix_weight_up.weight", W, HC_LOWRANK)
            bf(shard, h + "block_inject_weight.weight", HC, W)
            ignore += [h[:-1] + s for s in (".input_mix_weight_down", ".input_mix_weight_up", ".block_inject_weight")]
        if LAYER_TYPES[i] == "full_attention":
            q_rows = HEADS * HEAD_DIM
            quad(shard, p + "self_attn.q_proj", 2 * q_rows, H)
            quad(shard, p + "self_attn.k_proj", KV * HEAD_DIM, H)
            quad(shard, p + "self_attn.v_proj", KV * HEAD_DIM, H)
            quad(shard, p + "self_attn.o_proj", H, q_rows)
            bf(shard, p + "self_attn.q_norm.weight", HEAD_DIM, sigma=0.3)
            bf(shard, p + "self_attn.k_norm.weight", HEAD_DIM, sigma=0.3)
            bf(shard, p + "self_attn.indexer.index_qk_proj.weight", (IDX_HEADS + IDX_KV) * IDX_DIM, H)
            bf(shard, p + "self_attn.indexer.q_layernorm.weight", IDX_DIM, sigma=0.3)
            bf(shard, p + "self_attn.indexer.k_layernorm.weight", IDX_DIM, sigma=0.3)
            ignore.append(p + "self_attn.indexer.index_qk_proj")
        else:
            quad(shard, p + "linear_attn.in_proj_qkv", QKV, H)
            quad(shard, p + "linear_attn.in_proj_z", Z, H)
            quad(shard, p + "linear_attn.out_proj", H, Z)
            bf(shard, p + "linear_attn.in_proj_a.weight", LIN_V_HEADS, H)
            bf(shard, p + "linear_attn.in_proj_b.weight", LIN_V_HEADS, H)
            bf(shard, p + "linear_attn.conv1d.weight", QKV, 1, CONV)
            bf(shard, p + "linear_attn.A_log", LIN_V_HEADS, sigma=1.0)
            bf(shard, p + "linear_attn.dt_bias", LIN_V_HEADS, sigma=1.0)
            bf(shard, p + "linear_attn.norm.weight", LIN_DIM, sigma=0.3)
            ignore += [p + "linear_attn.in_proj_a", p + "linear_attn.in_proj_b"]
        if i == PLE_LAYER:
            bf(shard, p + "ple.key_proj.weight", W, PLE_DIM)
            bf(shard, p + "ple.value_proj.weight", H, PLE_DIM)
            for nm in ("norm_key", "norm_query", "norm_conv"):
                bf(shard, f"{p}ple.{nm}.weight", W, sigma=0.3)
            bf(shard, p + "ple.conv1d.weight", W, 1, PLE_CONV)
            ignore += [p + "ple.key_proj", p + "ple.value_proj"]
            # the n-gram table's tensors: skipped by the converter (they live in the sidecar)
            put(shard, p + "ple.ple_embedding.layer_multipliers", torch.tensor([3, 5, 7], dtype=torch.int64))
            put(shard, p + "ple.ple_embedding.ngram_heads_offsets", torch.arange(16, dtype=torch.int64))
            put(shard, p + "ple.ple_embedding.ngram_heads_vocab_sizes", torch.full((16,), 61, dtype=torch.int64))
            bf(shard, p + "ple.ple_embedding.ngram_embedding.shard_0.weight", 8, 16)
        bf(shard, p + "mlp.gate.weight", EXPERTS, H)
        bf(shard, p + "mlp.shared_expert.gate_proj.weight", SHARED_INTER, H)
        bf(shard, p + "mlp.shared_expert.up_proj.weight", SHARED_INTER, H)
        bf(shard, p + "mlp.shared_expert.down_proj.weight", H, SHARED_INTER)
        bf(shard, p + "mlp.shared_expert_gate.weight", 1, H)
        ignore += [p + "mlp.gate", p + "mlp.shared_expert.gate_proj", p + "mlp.shared_expert.up_proj",
                   p + "mlp.shared_expert.down_proj", p + "mlp.shared_expert_gate"]
        for e in range(EXPERTS):
            q = f"{p}mlp.experts.{e}."
            quad(shard, q + "gate_proj", MOE_INTER, H)
            quad(shard, q + "up_proj", MOE_INTER, H)
            quad(shard, q + "down_proj", H, MOE_INTER)
    g = "model-00001-of-00002.safetensors"
    bf(g, "model.language_model.embed_tokens.weight", VOCAB, H, sigma=0.02)
    bf(g, "lm_head.weight", VOCAB, H, sigma=0.02)
    last = "model-00002-of-00002.safetensors"
    h = "model.language_model.hyper_connection_mixer."
    bf(last, h + "hc_norm.weight", W, sigma=0.3)
    bf(last, h + "input_mix_weight_down.weight", HC_LOWRANK, W)
    bf(last, h + "input_mix_weight_up.weight", W, HC_LOWRANK)
    # skipped on purpose: vision and the MTP head
    bf(last, "model.visual.patch_embed.proj.weight", 4, 4)
    bf(last, "model.visual.blocks.0.attn.qkv.weight", 8, 4)
    mtp = "model_mtp.safetensors"
    bf(mtp, "mtp.fc_hidden.weight", H, H)
    bf(mtp, "mtp.layers.0.mlp.experts.down_proj", EXPERTS, H, MOE_INTER)

    weight_map = {}
    for shard, tensors in shards.items():
        save_st(out / "hf" / shard, tensors)
        weight_map.update({name: shard for name in tensors})
    (out / "hf" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1), encoding="utf-8")
    (out / "hf" / "config.json").write_text(json.dumps(mini_config(ignore), indent=2), encoding="utf-8")
    save_st(out / "expected_bf16.safetensors", expected)

    # the sidecar: rows of random bf16, behind the real header packer
    rng = np.random.default_rng(7)
    body = rng.integers(0, 256, size=NGRAM_ROWS * 2 * PLE_DIM // 16, dtype=np.uint8).tobytes()
    sha = hashlib.sha256(body).digest()
    hdr = ns.SidecarHeader(version=1, rows=NGRAM_ROWS, row_bytes=2 * PLE_DIM // 16, heads=16, head_dim=PLE_DIM // 16,
                           ngram_size=3, heads_per_ngram=8, eos_token_id=EOS, seed=11, multipliers=(3, 5, 7),
                           head_offsets=tuple(range(16)), head_primes=tuple(61 + j for j in range(16)),
                           padded_rows=NGRAM_ROWS, body_sha256=sha)
    (out / "sidecar.r4dx-ngram").write_bytes(hdr.pack() + body)
    (out / "expected.json").write_text(json.dumps({
        "linears": linears, "sidecar_body_sha256": sha.hex(), "sidecar_rows": NGRAM_ROWS,
        "sidecar_row_bytes": 2 * PLE_DIM // 16, "layers": LAYERS, "experts": EXPERTS,
        "hidden": H}, indent=1), encoding="utf-8")
    print(f"[convert_fixtures] awq-mini: {len(linears)} pack-quantized linears -> {out}")


# ---- real-spot -------------------------------------------------------------------------------------------------
REAL = Path(r"E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ")
REAL_SPOT = ["layers.0.linear_attn.in_proj_qkv", "layers.4.linear_attn.out_proj", "layers.3.self_attn.q_proj",
             "layers.47.self_attn.o_proj", "layers.5.mlp.experts.100.down_proj", "layers.20.mlp.experts.7.gate_proj",
             "layers.33.mlp.experts.511.up_proj", "layers.46.linear_attn.in_proj_z"]


def build_real_spot(out: Path) -> None:
    from safetensors import safe_open

    out.mkdir(parents=True, exist_ok=True)
    wm = json.loads((REAL / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    expected: dict[str, torch.Tensor] = {}
    for rel in REAL_SPOT:
        base = f"model.language_model.{rel}"
        parts = {}
        for suffix in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
            with safe_open(str(REAL / wm[f"{base}.{suffix}"]), framework="pt", device="cpu") as f:
                parts[suffix] = f.get_tensor(f"{base}.{suffix}")
        expected[f"{base}.weight"] = ad.dequant_pack_quantized(parts["weight_packed"], parts["weight_scale"],
                                                               parts["weight_zero_point"], parts["weight_shape"].tolist())
    save_st(out / "expected_bf16.safetensors", expected)
    print(f"[convert_fixtures] real-spot: {len(expected)} tensors -> {out}")


# ---- tiny-trellis ---------------------------------------------------------------------------------------------
_BODY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                   r"|mlp\.experts\.(\d+)\.(gate|up|down)_proj)\.weight$")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_st_raw(path: Path, tensors: list[tuple[str, str, list[int], bytes]], metadata: dict) -> None:
    header, off, data = {}, 0, []
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data.append(raw)
    header["__metadata__"] = metadata
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(data))


def build_tiny_trellis(hf: Path, out: Path, form: str, weights_file: Path | None = None) -> None:
    """`weights_file`: a safetensors of `<hf name>` bf16 weights to use instead of the checkpoint's own tensors (the AWQ mini
    checkpoint, whose linears are quads: expected_bf16.safetensors holds their dequantized form)."""
    import shutil
    from safetensors import safe_open

    import trellis_quant as tq

    bad_at = [0, 0]

    def rate(layer: int, expert: int | None, name: str) -> int:
        if form == "uniform4":
            return 4
        if form == "bad" and layer == bad_at[0] and expert == bad_at[1]:
            return 5
        return 3 if (expert is not None and layer in (0, 1)) else 4

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    wm = json.loads((hf / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    if weights_file is not None:
        wm = {n: str(weights_file) for n in safe_open(str(weights_file), framework="pt", device="cpu").keys()}
    names = sorted(n for n in wm if _BODY.match(n))
    config_sha = hashlib.sha256((hf / "config.json").read_bytes()).hexdigest()
    cb = torch.from_numpy(tq.codebook_np("mul1").copy())
    by_layer: dict[int, list[str]] = {}
    for n in names:
        by_layer.setdefault(int(_BODY.match(n).group(1)), []).append(n)
    n_experts = 1 + max(int(_BODY.match(n).group(5)) for n in names if _BODY.match(n).group(5) is not None)
    bad_at[:] = [min(2, max(by_layer)), 3 if n_experts > 3 else 1]  # the "bad" form: this expert at KB 5
    tensors_rec: dict[str, dict] = {}
    open_files: dict[str, object] = {}
    for layer, layer_names in sorted(by_layer.items()):
        recs, st_tensors = {}, []
        for idx, name in enumerate(layer_names):
            m = _BODY.match(name)
            expert = int(m.group(5)) if m.group(5) is not None else None
            kb = rate(layer, expert, name)
            shard = wm[name]
            if shard not in open_files:
                open_files[shard] = safe_open(str(hf / shard) if weights_file is None else shard, framework="pt", device="cpu")
            wt = open_files[shard].get_tensor(name).to(torch.float32)  # [n, k]
            n_, k_ = wt.shape
            g = np.random.default_rng([0x7E1F, layer, idx, kb])
            words = g.integers(0, 1 << 32, size=(k_ // 16, n_ // 16, 8 * kb), dtype=np.uint64).astype(np.uint32)
            suh = (g.uniform(0.5, 1.5, size=k_) * g.choice([-1.0, 1.0], size=k_)).astype(np.float16)
            svh = (g.uniform(0.005, 0.02, size=n_) * g.choice([-1.0, 1.0], size=n_)).astype(np.float16)
            w_hat = tq.reconstruct(torch.from_numpy(words.view(np.int32).copy()), torch.from_numpy(suh.copy()),
                                   torch.from_numpy(svh.copy()), float(kb), cb)  # [k][n]
            rel, cos = tq.rel_cos(wt.t().contiguous(), w_hat)
            st_tensors += [(name + ".trellis", "I32", list(words.shape), words.view(np.int32).tobytes()),
                           (name + ".suh", "F16", [k_], suh.tobytes()), (name + ".svh", "F16", [n_], svh.tobytes())]
            bits_trellis = words.size * 32
            bits_scales = (suh.size + svh.size) * 16
            recs[name] = {"encoding": tq.ENCODING_TRELLIS, "K": float(kb), "codebook": "mul1", "k": k_, "n": n_,
                          "shape_hf": [n_, k_], "words_shape": list(words.shape),
                          "bits": {"trellis": bits_trellis, "scales": bits_scales, "marker": 32,
                                   "total": bits_trellis + bits_scales + 32,
                                   "bpw": (bits_trellis + bits_scales + 32) / (k_ * n_),
                                   "bpw_trellis": bits_trellis / (k_ * n_)},
                          "rel_weight_err": rel, "cos": cos, "hessian_basis": "matched"}
        st = out / f"L{layer:02d}.safetensors"
        write_st_raw(st, st_tensors, {"format": tq.OVERRIDE_FORMAT, "layer": str(layer)})
        fsha = sha256_file(st)
        for name, rec in recs.items():
            rec["file"] = str(st.resolve())  # the mix form's absolute file
            rec["file_sha256"] = fsha
            tensors_rec[name] = rec
        print(f"[convert_fixtures] tiny-trellis {form}: layer {layer}: {len(recs)} tensors")
    counts: dict[str, int] = {}
    for rec in tensors_rec.values():
        counts[str(rec["K"])] = counts.get(str(rec["K"]), 0) + 1
    man = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
           "complete": True, "missing": [], "missing_count": 0, "bpw_target": None, "K_uniform": 4.0 if form == "uniform4" else None,
           "model_dir": str(hf), "config_sha256": config_sha, "hessian_dir": "(synthetic)",
           "hessian_manifest_sha256": "2" * 64, "hessian_basis": "matched", "codebook": "mul1",
           "recipe": {"note": "synthetic: random ring words, not an encoder's output (convert_fixtures.py)"},
           "allocation": {"rule": f"synthetic form {form}", "tensors_per_K": counts, "sources": {},
                          "source_code_sha256": {}},
           "summary": tq.summarize_tensors(tensors_rec), "tensors": tensors_rec,
           "generated_at": "2026-10-10T00:00:00+00:00", "provenance": {"tool": "tools/reference/qwen4exp/convert_fixtures.py"}}
    (out / tq.MANIFEST_NAME).write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[convert_fixtures] tiny-trellis {form}: {len(tensors_rec)} tensors, K counts {counts} -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["awq-mini", "awq-mini-trellis", "real-spot", "tiny-trellis"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--hf-dir", type=Path, default=TINY_HF)
    ap.add_argument("--form", choices=["uniform4", "mix34", "bad"], default="mix34")
    a = ap.parse_args()
    if a.what == "awq-mini":
        build_awq_mini(a.out_dir / "awq-mini")
    elif a.what == "awq-mini-trellis":
        mini = a.out_dir / "awq-mini"
        build_tiny_trellis(mini / "hf", a.out_dir / f"awq-mini-trellis-{a.form}", a.form, mini / "expected_bf16.safetensors")
    elif a.what == "real-spot":
        build_real_spot(a.out_dir / "real-spot")
    else:
        build_tiny_trellis(a.hf_dir, a.out_dir / f"tiny-trellis-{a.form}", a.form)
    if torch.cuda.is_initialized():
        raise SystemExit("torch.cuda was initialized -- this script must stay on the CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
