"""CPU-only tests of the post-corpus GPU pipeline pieces: kv_calibrate_full.py (Gemma KV descales) and
gen_samples.py --merge. A tiny random-config checkpoint, no GPU, no real weights.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\gemma\\test_pipeline_tiny.py -q
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

import gen_samples as gs  # noqa: E402
import importlib.util  # noqa: E402

# tools/reference/kv_calibrate_full.py (Qwen) has the same module name: load the Gemma one by path.
_spec = importlib.util.spec_from_file_location("gemma_kv_calibrate_full", HERE / "kv_calibrate_full.py")
kc = importlib.util.module_from_spec(_spec)
sys.modules["gemma_kv_calibrate_full"] = kc
_spec.loader.exec_module(kc)
from gemma.ref import GemmaReference, build_tiny_checkpoint  # noqa: E402


def test_kv_calibrate_tiny_matches_independent_recompute(tmp_path):
    """amax of K (post k_norm + rope) and V (v_norm of v_proj, or of the RAW k_proj on full layers) equals a
    from-scratch recomputation with the HF building blocks, and the JSON has what the converter reads."""
    out = tmp_path / "kv.json"
    assert kc.main(["--tiny", "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    summ = json.loads((tmp_path / "kv.json.summary.json").read_text(encoding="utf-8"))
    assert summ["complete"] and set(doc) == {"0", "1", "2", "3"}

    with tempfile.TemporaryDirectory() as td:
        d = build_tiny_checkpoint(Path(td) / "m", 0)          # same seed as the script's --tiny
        ref = GemmaReference(d, torch.device("cpu"))
        gen = torch.Generator().manual_seed(0 + 5)
        seqs = [[ref.arch.bos_id] + torch.randint(4, ref.arch.vocab, (n - 1,), generator=gen).tolist()
                for n in (37, 21, 50)]
        want = {i: {"k": None, "v": None} for i in range(4)}
        with torch.no_grad():
            for ids in seqs:
                rec = {}
                ref.forward_hidden(ids, record=lambda i, h: rec.__setitem__(i, h.clone()))
                T = len(ids)
                pos = torch.arange(T).unsqueeze(0)
                for i in range(4):
                    x = ref.embed(ids).unsqueeze(0) if i == 0 else rec[i - 1]
                    layer = ref.build_layer(i)
                    at = layer.self_attn
                    hd = at.head_dim
                    h = layer.input_layernorm(x)
                    kraw = at.k_proj(h).view(1, T, -1, hd)
                    vraw = at.v_proj(h).view(1, T, -1, hd) if at.v_proj is not None else kraw
                    cos, sin = ref.rotary(x, pos, ref.layer_types[i])
                    k = ref.m.apply_rotary_pos_emb(at.k_norm(kraw), cos, sin, unsqueeze_dim=2)
                    v = kc.rms_noscale(vraw, ref.arch.eps)
                    for nm, t in (("k", k), ("v", v)):
                        a = t[0].transpose(0, 1).float().abs().amax(dim=(1, 2))
                        want[i][nm] = a if want[i][nm] is None else torch.maximum(want[i][nm], a)
    for i in range(4):
        e = doc[str(i)]
        assert e["layer_type"] == ("full_attention" if i % 2 else "sliding_attention")
        assert e["head_dim"] == (32 if i % 2 else 16)
        assert len(e["k_amax"]) == len(e["v_amax"]) == e["kv_heads"] == 1
        assert torch.allclose(torch.tensor(e["k_amax"]), want[i]["k"], rtol=0, atol=1e-6), i
        assert torch.allclose(torch.tensor(e["v_amax"]), want[i]["v"], rtol=0, atol=1e-6), i
        assert e["k_descale"][0] == pytest.approx(e["k_amax"][0] / 448.0)
        assert e["selfcheck"]["v_equals_vnorm_of_source_max_abs_diff"] == 0.0
        assert e["selfcheck"]["k_differs_from_k_norm_output"]
        assert e["fp16_range"]["fits_f16"]
        assert e["tail_exact"]
    # k_eq_v: on a full layer V is NOT K (K is normed with a weight and rotated), and its source is raw k_proj
    assert "RAW k_proj" in doc["1"]["v_tap"] and "v_proj" in doc["0"]["v_tap"]
    assert doc["1"]["selfcheck"]["k_vs_v_max_abs_diff"] > 0


def test_kv_calibrate_partial_refused_without_flag(tmp_path):
    with pytest.raises(SystemExit):
        kc.main(["--tiny", "--max-layers", "2", "--out", str(tmp_path / "kv.json")])
    assert kc.main(["--tiny", "--max-layers", "2", "--allow-partial", "--out", str(tmp_path / "kv.json")]) == 0
    assert set(json.loads((tmp_path / "kv.json").read_text(encoding="utf-8"))) == {"0", "1"}


def _rec(e, ids=(2, 5, 6)):
    return {"id": e["id"], "category": e["category"], "prompt_sha256": gs.gc.entry_sha256(e), "rejected": False,
            "token_ids": list(ids), "enable_thinking": e["enable_thinking"], "completion_tokens": 2,
            "kl_token_overlap": 0, "reject_reason": None, "finish_reason": "eos"}


def test_merge_shards(tmp_path):
    ents = gs.TINY_PROMPTS
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(_rec(e)) + "\n" for e in ents[:2]), encoding="utf-8")
    # shard b still being written: one whole record and a torn last line
    b.write_bytes((json.dumps(_rec(ents[2])) + "\n" + '{"id": "chat/002", "pro').encode())
    before = b.read_bytes()
    out = tmp_path / "samples.jsonl"
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(ents), encoding="utf-8")
    # the tiny prompts have no prompt file in the real schema, so validate the structure with --no-prompt-check
    assert gs.main(["--merge", str(a), str(b), "--out", str(out), "--no-prompt-check"]) == 0
    assert b.read_bytes() == before                      # read-only on a shard that is still being written
    ids = [json.loads(ln)["id"] for ln in out.read_text(encoding="utf-8").splitlines()]
    assert ids == sorted(ids) and len(ids) == 3
    info = json.loads((tmp_path / "samples.merge.json").read_text(encoding="utf-8"))
    assert info["shards"][1]["torn_lines_skipped"] == 1 and info["summary"]["samples"] == 3
    with pytest.raises(SystemExit):                      # an id in two shards
        gs.merge([a, a], tmp_path / "dup.jsonl")

def test_mix_gemma_proxy_rank(tmp_path):
    """`trellis_quant.py mix` on Gemma ranks promotion groups by proxy gain per extra bit, not EXL3's edge-first
    order; Qwen's default (exl3) is untouched. Fake uniform-rate manifests of the real 328 Gemma linears (reads the
    checkpoint's config and safetensors header only; a tiny checkpoint when the real one is absent)."""
    import argparse

    import trellis_quant as tq

    model_dir = Path(os.environ.get("R4DX_MODEL_DIR", r"D:\models\Huihui-gemma-4-12B-it-abliterated"))
    td_keep = None
    if not (model_dir / "model.safetensors").exists():
        td_keep = tempfile.TemporaryDirectory()
        model_dir = build_tiny_checkpoint(Path(td_keep.name) / "m", 0, layers=8)
    lin = tq.model_linears(model_dir)
    arch = tq.get_arch("gemma4_unified")
    assert arch.is_gemma
    layers = sorted({t["layer"] for t in lin})
    mid = layers[len(layers) // 2]
    job = {"model_dir": str(model_dir), "config_sha256": "x", "hessian_manifest_sha256": "y",
           "hessian_basis": "matched", "codebook": "mul1", "arch": "gemma4_unified"}

    def proxy(t, K):
        if t["layer"] == mid and t["module"] in ("self_attn.k_proj", "self_attn.q_proj"):
            return 0.05 if K == 4 else 0.01     # a mid-stack attention group the extra bit helps a lot
        if t["layer"] == 0:
            return 0.02 if K == 4 else 0.02     # an edge layer the extra bit does not help at all
        return 0.01 if K == 4 else 0.0025

    for K in (4, 5):
        d = tmp_path / f"K{K}"
        d.mkdir()
        tens = {}
        for t in lin:
            bits = t["numel"] * K
            sc = 16 * (t["k"] + t["n"])
            tens[t["name"]] = {"encoding": tq.ENCODING_TRELLIS, "K": float(K), "codebook": "mul1", "k": t["k"],
                               "n": t["n"], "file": f"L{t['layer']:02d}.safetensors", "proxy": proxy(t, K),
                               "bits": {"trellis": bits, "scales": sc, "total": bits + sc + 32,
                                        "bpw": (bits + sc + 32) / t["numel"]}}
        tq.write_json_atomic(d / tq.MANIFEST_NAME, {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION,
                                                    "complete": True, "tensors": tens, **job})
    out = {}
    for rank in ("auto", "exl3"):
        tq.cmd_mix(argparse.Namespace(src=[str(tmp_path / "K4"), str(tmp_path / "K5")], bpw=4.5,
                                      out_dir=tmp_path / f"mix_{rank}", rank=rank))
        m = tq.load_manifest(tmp_path / f"mix_{rank}")
        out[rank] = ({n: r["K"] for n, r in m["tensors"].items()}, m)
    ks, man = out["auto"]
    assert man["allocation"]["rank"] == "proxy-gain-per-bit" and "group_scores" in man["allocation"]
    assert out["exl3"][1]["allocation"]["rank"] == "exl3-edges-first"
    budget = int(4.5 * sum(t["numel"] for t in lin))
    assert sum(t["numel"] * ks[t["name"]] for t in lin) <= budget
    for t in lin:   # every group atomic
        if t["qgroup"] == "self_attn.qkv":
            assert len({ks[u["name"]] for u in lin if u["layer"] == t["layer"] and u["qgroup"] == "self_attn.qkv"}) == 1
    assert ks[tq.hf_name(mid, "self_attn.q_proj")] == 5.0 and ks[tq.hf_name(mid, "self_attn.k_proj")] == 5.0
    assert ks[tq.hf_name(0, "mlp.down_proj")] == 4.0            # no gain at layer 0: not promoted by proxy rank
    assert out["exl3"][0][tq.hf_name(0, "mlp.down_proj")] == 5.0   # EXL3's edge-first order promotes it
    if td_keep is not None:
        td_keep.cleanup()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
