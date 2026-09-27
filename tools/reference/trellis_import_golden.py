"""tools/reference/trellis_import_golden.py -- CPU-only fixture for r4dx-convert --trellis-from.

docs/trellis-kernel.md section 6 (`convert_trellis_import`): a synthetic 2-layer checkpoint and
synthetic oracle override directories in BOTH manifest forms trellis_quant.py writes, plus the bytes
the converter must write for them. tests/convert/test_trellis_import.cpp reads it; it exits 77
(CTest SKIPPED) when this directory is absent.

Output, tools/reference/golden_out/trellis_import/ (gitignored, like every golden_out tree):

  ckpt_k4/     config.json, model.safetensors.index.json, one bf16 shard. Layer 0 is GDN, layer 1
               full attention, hidden 256; every weight the oracle covers is W_hat of oracle_k4's
               words times (1 + a g), g standard normal and the amplitude a drawn per 16 x 16 tile
               from [0.002, 0.02], in bf16 -- so rel_weight_err is about 0.012 and the error is
               spread UNEVENLY: a reconstruction check that skipped rows, blocks or tiles would miss
               the recorded rel by far more than the converter's strict 1e-4 bound (with an even
               error, a partial check reproduces rel exactly). The rest is random.
               The config is the real model's in miniature and complete enough for the runtime:
               ModelConfig::FromJson parses it, and ModelConfig::Shard accepts it at TP = 2 with
               every trellis rank range a multiple of 128 (docs/trellis-kernel.md 2.4-2.5, 6:
               test_trellis_linear and test_tp_loader load the converted tiny containers).
  ckpt_mix/    the same config.json byte for byte (same config_sha256); layer 0's covered weights are
               derived from oracle_k5's words, layer 1's from oracle_k4's -- mix/'s allocation.
  oracle_k4/   a quantize-model directory at K = 4 (trellis_quant.py write_manifest): L00/L01
               .safetensors with `<hf>.trellis` I32 [k/16, n/16, 32], `.suh` F16 [k], `.svh` F16 [n]
               of random ring words and scales, and weights_override.json with RELATIVE `file`s,
               stale_layers / layers_done / top-level code_sha256. rel_weight_err is
               trellis_quant.rel_cos against ckpt_k4.
  oracle_k5/   the same at K = 5 (40 words per tile); rel_weight_err against ckpt_mix.
  mix/         weights_override.json in cmd_mix's form: ABSOLUTE `file`s into oracle_k5 (layer 0)
               and oracle_k4 (layer 1), float K, `allocation` (sources, source_code_sha256), no
               stale_layers / layers_done / top-level code_sha256.
  expected.json
               per form ("k4", "mix") and container base: bits, parts, and the sha256 / size of the
               three tensors the converter must write -- `.trellis.w` = trellis_golden.to_pair_grid of
               the oracle words (the reference regrid), `.trellis.suh` = the parts' suh concatenated,
               `.trellis.svh` likewise; the sha256 of trellis_quant.decode_words' Q [K][N] fp16 of
               every oracle tensor at its rate ("decode"); the sha256 of codebook_np("mul1") as fp16;
               config_sha256; the manifests' sha256; the HF -> base map.

The weights are W_hat = trellis_quant.reconstruct(words, suh, svh) on torch's CPU device, so the
fixture's decode and reconstruction are the oracle's own code; torch.cuda is never touched.

Usage (the reference venv's python; CPU only):

  & $py tools\\reference\\trellis_import_golden.py            # (re)generate
  & $py tools\\reference\\trellis_import_golden.py --check    # the files against expected.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import trellis_quant as tq  # noqa: E402  (imports torch; CPU only)
import trellis_golden as tg  # noqa: E402  (to_pair_grid: the reference regrid)
import torch  # noqa: E402

DEFAULT_OUT = HERE / "golden_out" / "trellis_import"
FORMAT = "r4dx-trellis-import-golden"
VERSION = 1
SEED = 0x7E1F

# The real model's shapes in miniature (head_dim 256, GDN head dims 128), sized for TP = 2: every
# row-parallel rank K (attn.o 4*256/2, gdn.out_proj 1024/2, mlp.down 1024/2) a multiple of 512
# (ModelConfig::Shard), every column-parallel rank segment a multiple of 128 (q/k 128, v 512, z 512,
# qg 1024, attn.k/v 256, gate/up 512 each; the trellis loader's rule, docs/trellis-kernel.md 2.5).
HIDDEN = 256
INTER = 1024
VOCAB = 128
HEAD_DIM = 256
N_HEADS = 4
N_KV = 2
GDN_K_HEADS = 2
GDN_K_DIM = 128
GDN_V_HEADS = 8
GDN_V_DIM = 128
GDN_KEY = GDN_K_HEADS * GDN_K_DIM  # 256
GDN_V = GDN_V_HEADS * GDN_V_DIM  # 1024
GDN_QKV = 2 * GDN_KEY + GDN_V  # 1536
LAYER_TYPES = ["linear_attention", "full_attention"]
PREFIX = "model.language_model.layers."

# (layer, HF module, [n, k], container base suffix). Every n and k a multiple of 128.
TRELLIS = [
    (0, "linear_attn.in_proj_qkv", [GDN_QKV, HIDDEN], "gdn.in_proj_qkv"),
    (0, "linear_attn.in_proj_z", [GDN_V, HIDDEN], "gdn.in_proj_z"),
    (0, "linear_attn.out_proj", [HIDDEN, GDN_V], "gdn.out_proj"),
    (0, "mlp.gate_proj", [INTER, HIDDEN], "mlp.gate_up"),
    (0, "mlp.up_proj", [INTER, HIDDEN], "mlp.gate_up"),
    (0, "mlp.down_proj", [HIDDEN, INTER], "mlp.down"),
    (1, "self_attn.q_proj", [2 * N_HEADS * HEAD_DIM, HIDDEN], "attn.qg"),
    (1, "self_attn.k_proj", [N_KV * HEAD_DIM, HIDDEN], "attn.k"),
    (1, "self_attn.v_proj", [N_KV * HEAD_DIM, HIDDEN], "attn.v"),
    (1, "self_attn.o_proj", [HIDDEN, N_HEADS * HEAD_DIM], "attn.o"),
    (1, "mlp.gate_proj", [INTER, HIDDEN], "mlp.gate_up"),
    (1, "mlp.up_proj", [INTER, HIDDEN], "mlp.gate_up"),
    (1, "mlp.down_proj", [HIDDEN, INTER], "mlp.down"),
]
# Everything else the converter reads (random bf16), [shape].
OTHERS = {
    "model.language_model.embed_tokens.weight": [VOCAB, HIDDEN],
    "model.language_model.norm.weight": [HIDDEN],
    "lm_head.weight": [VOCAB, HIDDEN],
}
for _l in (0, 1):
    OTHERS[f"{PREFIX}{_l}.input_layernorm.weight"] = [HIDDEN]
    OTHERS[f"{PREFIX}{_l}.post_attention_layernorm.weight"] = [HIDDEN]
OTHERS.update({
    f"{PREFIX}0.linear_attn.in_proj_a.weight": [GDN_V_HEADS, HIDDEN],
    f"{PREFIX}0.linear_attn.in_proj_b.weight": [GDN_V_HEADS, HIDDEN],
    f"{PREFIX}0.linear_attn.conv1d.weight": [GDN_QKV, 1, 4],
    f"{PREFIX}0.linear_attn.A_log": [GDN_V_HEADS],
    f"{PREFIX}0.linear_attn.dt_bias": [GDN_V_HEADS],
    f"{PREFIX}0.linear_attn.norm.weight": [GDN_V_DIM],
    f"{PREFIX}1.self_attn.q_norm.weight": [HEAD_DIM],
    f"{PREFIX}1.self_attn.k_norm.weight": [HEAD_DIM],
})
MIX_K = {0: 5, 1: 4}  # mix/: the rate of every tensor of a layer
FIXED_TIME = "2026-09-27T00:00:00+00:00"
CODE_SHA = {"trellis_quant": "0" * 64, "trellis_viterbi": "1" * 64}
HESS_SHA = "2" * 64


def hf_name(layer: int, module: str) -> str:
    return f"{PREFIX}{layer}.{module}.weight"


def base_of(layer: int, suffix: str) -> str:
    return f"text.layers.{layer}.{suffix}"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]], metadata: dict | None = None) -> None:
    """name, dtype, shape, raw bytes -> a safetensors file (header padded to 8 bytes with spaces)."""
    header, off, data = {}, 0, []
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data.append(raw)
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(data))


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits (round to nearest even), as uint16."""
    return (tg.bf16_rn(np.ascontiguousarray(x, dtype=np.float32)).view(np.uint32) >> 16).astype(np.uint16)


def bf16_value(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def rng_for(*tag) -> np.random.Generator:
    return np.random.default_rng([SEED, *tag])


def oracle_tensors(idx: int, n: int, k: int, KB: int):
    """Random ring words [k/16][n/16][8KB] (uint32), suh [k] and svh [n] fp16 with random signs."""
    r = rng_for(idx, KB)
    words = r.integers(0, 1 << 32, size=(k // 16, n // 16, 8 * KB), dtype=np.uint64).astype(np.uint32)
    suh = (r.uniform(0.5, 1.5, size=k) * r.choice([-1.0, 1.0], size=k)).astype(np.float16)
    svh = (r.uniform(0.005, 0.02, size=n) * r.choice([-1.0, 1.0], size=n)).astype(np.float16)
    return words, suh, svh


def reconstruct(words: np.ndarray, suh: np.ndarray, svh: np.ndarray, KB: int) -> np.ndarray:
    """trellis_quant.reconstruct on the CPU: W_hat [k][n] float32 (in, out)."""
    cb = torch.from_numpy(tq.codebook_np("mul1").copy())
    w = torch.from_numpy(words.view(np.int32).copy())
    return tq.reconstruct(w, torch.from_numpy(suh.copy()), torch.from_numpy(svh.copy()), float(KB), cb).numpy()


def perturb(w_hat: np.ndarray, idx: int) -> np.ndarray:
    """W_hat [k][n] -> the checkpoint's float32 weight [n][k]: W_hat * (1 + a g), g standard normal
    per weight, the amplitude a per 16 x 16 tile uniform in [0.002, 0.02] (an uneven error, see the
    module docstring). The same pattern for both checkpoints (seeded by the tensor, not the rate)."""
    k, n = w_hat.shape
    r = rng_for(1000 + idx)
    amp = np.repeat(np.repeat(r.uniform(0.002, 0.02, size=(k // 16, n // 16)), 16, axis=0), 16, axis=1)
    return (w_hat * (1.0 + amp * r.standard_normal(size=(k, n)))).astype(np.float32).T


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    for d in ("ckpt_k4", "ckpt_mix", "oracle_k4", "oracle_k5", "mix"):
        (out / d).mkdir(parents=True)

    config = {
        "architectures": ["SyntheticTrellisImportTest"],
        "text_config": {
            "hidden_size": HIDDEN, "num_hidden_layers": 2, "layer_types": LAYER_TYPES,
            "num_attention_heads": N_HEADS, "num_key_value_heads": N_KV, "head_dim": HEAD_DIM,
            "linear_num_key_heads": GDN_K_HEADS, "linear_key_head_dim": GDN_K_DIM,
            "linear_num_value_heads": GDN_V_HEADS, "linear_value_head_dim": GDN_V_DIM,
            "linear_conv_kernel_dim": 4, "intermediate_size": INTER, "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
        },
    }
    config_bytes = json.dumps(config, indent=2).encode("utf-8")
    config_sha = sha256_bytes(config_bytes)

    # ---- the oracle tensors at both rates, and the two checkpoints' covered weights ----
    oracle = {}  # (hf, KB) -> (words, suh, svh, W_hat [k][n])
    for idx, (layer, mod, (n, k), _) in enumerate(TRELLIS):
        name = hf_name(layer, mod)
        for KB in (4, 5):
            words, suh, svh = oracle_tensors(idx, n, k, KB)
            oracle[(name, KB)] = (words, suh, svh, reconstruct(words, suh, svh, KB))
    ckpt_bits = {"ckpt_k4": {}, "ckpt_mix": {}}
    for idx, (layer, mod, (n, k), _) in enumerate(TRELLIS):
        name = hf_name(layer, mod)
        ckpt_bits["ckpt_k4"][name] = bf16_bits(perturb(oracle[(name, 4)][3], idx))
        ckpt_bits["ckpt_mix"][name] = bf16_bits(perturb(oracle[(name, MIX_K[layer])][3], idx))
    r = rng_for(999)
    others = {}
    for name, shape in OTHERS.items():
        sigma = 0.3 if len(shape) == 1 else 0.02
        others[name] = bf16_bits((r.standard_normal(shape) * sigma).astype(np.float32))

    for ck in ("ckpt_k4", "ckpt_mix"):
        d = out / ck
        (d / "config.json").write_bytes(config_bytes)
        tens = []
        for name in sorted(list(OTHERS) + list(ckpt_bits[ck])):
            bits = ckpt_bits[ck].get(name)
            if bits is None:
                bits = others[name]
            tens.append((name, "BF16", list(bits.shape), bits.tobytes()))
        shard = "model-00001-of-00001.safetensors"
        write_safetensors(d / shard, tens, {"format": "pt"})
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {t[0]: shard for t in tens}}, indent=2), encoding="utf-8")

    # ---- quantize-model directories ----
    recipe = {"note": "synthetic: random ring words, not an encoder's output (tools/reference/trellis_import_golden.py)"}
    job_common = {"model_dir": str(out / "ckpt_k4"), "config_sha256": config_sha, "hessian_dir": "(synthetic)",
                  "hessian_manifest_sha256": HESS_SHA, "hessian_basis": "matched", "codebook": "mul1",
                  "recipe": recipe, "code_sha256": CODE_SHA}

    def record(name: str, KB: int, file: str, file_sha: str, ref_ckpt: str) -> dict:
        words, suh, svh, W_hat = oracle[(name, KB)]
        k, n = W_hat.shape
        W = bf16_value(ckpt_bits_all[ref_ckpt][name]).T  # [k][n]: what the oracle's rel_cos compares
        rel, cos = tq.rel_cos(torch.from_numpy(np.ascontiguousarray(W)), torch.from_numpy(W_hat))
        bits_trellis = words.size * 32
        bits_scales = (suh.size + svh.size) * 16
        return {"encoding": tq.ENCODING_TRELLIS, "K": float(KB), "codebook": "mul1", "k": k, "n": n,
                "shape_hf": [n, k], "words_shape": list(words.shape),
                "bits": {"trellis": bits_trellis, "scales": bits_scales, "marker": 32,
                         "total": bits_trellis + bits_scales + 32,
                         "bpw": (bits_trellis + bits_scales + 32) / (k * n), "bpw_trellis": bits_trellis / (k * n)},
                "rel_weight_err": rel, "cos": cos, "hessian_basis": "matched", "file": file, "file_sha256": file_sha}

    ckpt_bits_all = {ck: {**others, **ckpt_bits[ck]} for ck in ckpt_bits}
    dirs = {4: ("oracle_k4", "ckpt_k4"), 5: ("oracle_k5", "ckpt_mix")}
    manifests = {}
    for KB, (dname, ref_ckpt) in dirs.items():
        d = out / dname
        tensors = {}
        for layer in (0, 1):
            st = d / f"L{layer:02d}.safetensors"
            tl = []
            for lay, mod, _, _ in TRELLIS:
                if lay != layer:
                    continue
                name = hf_name(lay, mod)
                words, suh, svh, _ = oracle[(name, KB)]
                tl.append((name + ".trellis", "I32", list(words.shape), words.view(np.int32).tobytes()))
                tl.append((name + ".suh", "F16", [suh.size], suh.tobytes()))
                tl.append((name + ".svh", "F16", [svh.size], svh.tobytes()))
            write_safetensors(st, tl, {"format": tq.OVERRIDE_FORMAT, "layer": str(layer)})
            fsha = sha256_file(st)
            for lay, mod, _, _ in TRELLIS:
                if lay == layer:
                    name = hf_name(lay, mod)
                    tensors[name] = record(name, KB, st.name, fsha, ref_ckpt)
        man = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
               "complete": True, "missing": [], "missing_count": 0, "bpw_target": None, "K_uniform": float(KB),
               "layers_done": [0, 1], "stale_layers": [], **job_common,
               "summary": tq.summarize_tensors(tensors), "tensors": tensors,
               "generated_at": FIXED_TIME, "provenance": {"tool": "tools/reference/trellis_import_golden.py"}}
        (d / tq.MANIFEST_NAME).write_text(json.dumps(man, indent=2), encoding="utf-8")
        manifests[KB] = man

    # ---- the mix (cmd_mix's form) ----
    mix_tensors, counts = {}, {}
    for layer, mod, _, _ in TRELLIS:
        name = hf_name(layer, mod)
        KB = MIX_K[layer]
        rec = dict(manifests[KB]["tensors"][name])
        rec["file"] = str((out / dirs[KB][0] / rec["file"]).resolve())
        mix_tensors[name] = rec
        counts[str(float(KB))] = counts.get(str(float(KB)), 0) + 1
    job_keys = ("model_dir", "config_sha256", "hessian_manifest_sha256", "hessian_basis", "codebook")
    base = manifests[4]
    mix = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
           "complete": True, "missing": [], "missing_count": 0, "bpw_target": 4.5, "K_uniform": None,
           **{k: base.get(k) for k in job_keys}, "recipe": base.get("recipe"), "hessian_dir": base.get("hessian_dir"),
           "allocation": {"rule": "synthetic: layer 0 at K = 5, layer 1 at K = 4", "tensors_per_K": counts,
                          "sources": {str(float(KB)): str((out / dirs[KB][0] / tq.MANIFEST_NAME).resolve()) for KB in (4, 5)},
                          "source_code_sha256": {str(float(KB)): CODE_SHA for KB in (4, 5)}},
           "summary": tq.summarize_tensors(mix_tensors), "tensors": mix_tensors,
           "generated_at": FIXED_TIME, "provenance": {"tool": "tools/reference/trellis_import_golden.py"}}
    (out / "mix" / tq.MANIFEST_NAME).write_text(json.dumps(mix, indent=2), encoding="utf-8")

    # ---- what the converter must write ----
    def expect(rate_of) -> dict:
        per_base: dict[str, dict] = {}
        for layer, mod, _, suffix in TRELLIS:
            per_base.setdefault(base_of(layer, suffix), {"layer": layer, "hf": []})["hf"].append(hf_name(layer, mod))
        out_j = {}
        for b, e in per_base.items():
            KB = rate_of(e["layer"])
            parts = [oracle[(h, KB)] for h in e["hf"]]
            w = tg.to_pair_grid([p[0] for p in parts])
            suh = b"".join(p[1].tobytes() for p in parts)
            svh = b"".join(p[2].tobytes() for p in parts)
            ns = [p[2].size for p in parts]
            k = parts[0][1].size
            out_j[b] = {"bits": KB, "parts": ns if len(ns) > 1 else None, "hf": e["hf"], "N": sum(ns), "K": k,
                        "w_bytes": w.nbytes, "w_sha256": sha256_bytes(w.tobytes()),
                        "suh_bytes": len(suh), "suh_sha256": sha256_bytes(suh),
                        "svh_bytes": len(svh), "svh_sha256": sha256_bytes(svh)}
        return out_j

    decode = {}
    for layer, mod, _, _ in TRELLIS:
        name = hf_name(layer, mod)
        for KB in (4, 5):
            q = tg.decode_q(oracle[(name, KB)][0], KB)  # [K][N] fp16, trellis_quant.decode_words
            decode[f"{name}@{KB}"] = sha256_bytes(np.ascontiguousarray(q).tobytes())
    cb16 = tq.codebook_np("mul1").astype(np.float16)
    assert np.array_equal(cb16.astype(np.float32), tq.codebook_np("mul1"))
    expected = {
        "format": FORMAT, "version": VERSION, "config_sha256": config_sha,
        "codebook_f16_sha256": sha256_bytes(cb16.tobytes()),
        "manifest_sha256": {"k4": sha256_file(out / "oracle_k4" / tq.MANIFEST_NAME),
                            "k5": sha256_file(out / "oracle_k5" / tq.MANIFEST_NAME),
                            "mix": sha256_file(out / "mix" / tq.MANIFEST_NAME)},
        "forms": {"k4": expect(lambda layer: 4), "mix": expect(lambda layer: MIX_K[layer])},
        "decode": decode,
        "hf_tensors": len(TRELLIS),
        "trellis_hf": [hf_name(la, m) for la, m, _, _ in TRELLIS],
    }
    (out / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return expected


def cmd_check(out: Path) -> int:
    exp = json.loads((out / "expected.json").read_text(encoding="utf-8"))
    bad = []
    for form, d in (("k4", "oracle_k4"), ("k5", "oracle_k5"), ("mix", "mix")):
        if sha256_file(out / d / tq.MANIFEST_NAME) != exp["manifest_sha256"][form]:
            bad.append(f"{d}/{tq.MANIFEST_NAME}")
    for d in ("oracle_k4", "oracle_k5"):
        man = json.loads((out / d / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
        for rec in man["tensors"].values():
            if sha256_file(out / d / rec["file"]) != rec["file_sha256"]:
                bad.append(f"{d}/{rec['file']}")
    for ck in ("ckpt_k4", "ckpt_mix"):
        if sha256_file(out / ck / "config.json") != exp["config_sha256"]:
            bad.append(f"{ck}/config.json")
    for b in sorted(set(bad)):
        print(f"[trellis-import-golden] MISMATCH {b}")
    print(f"[trellis-import-golden] {'OK' if not bad else 'FAILED'}: {out}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="verify the files against expected.json")
    args = ap.parse_args()
    if args.check:
        return cmd_check(args.out_dir)
    exp = build(args.out_dir)
    rels = []
    for d in ("oracle_k4", "oracle_k5"):
        man = json.loads((args.out_dir / d / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
        rels += [(d, n, r["rel_weight_err"]) for n, r in man["tensors"].items()]
    print(f"[trellis-import-golden] wrote {args.out_dir}: {exp['hf_tensors']} oracle tensors per rate, "
          f"forms {sorted(exp['forms'])}, rel_weight_err "
          f"{min(r for _, _, r in rels):.5f}..{max(r for _, _, r in rels):.5f}")
    if torch.cuda.is_initialized():
        raise SystemExit("torch.cuda was initialized -- this script must stay on the CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
