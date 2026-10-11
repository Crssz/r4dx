"""CPU tests of the n-gram id reference and the sidecar tooling (plan M0-3). No GPU, no real weights.

    $env:HIP_VISIBLE_DEVICES=''; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\qwen4exp\\test_ngram.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

import ngram_sidecar as ns  # noqa: E402
from ngram_ids import NGramHash  # noqa: E402

PLE_LAYER_IDX = 1  # 0-based decoder layer of ple_layer_ids [2]
REAL = dict(vocab_size=248320, eos_token_id=248044, ngram_vocab_size_base=20_000_000, hidden_size=2560,
            ple_embed_dim=2560, seed=1234)


def _text_config(**kw) -> Qwen4ExpTextConfig:
    """A tiny but valid Qwen4ExpTextConfig with one PLE layer at layer index 1."""
    base = dict(vocab_size=97, hidden_size=32, ple_embed_dim=32, num_hidden_layers=4, ple_layer_ids=[2],
                ngram_vocab_size_base=50, eos_token_id=7, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8, shared_expert_intermediate_size=8,
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                hc_lowrank=8, indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8,
                indexer_compress_ratio=4)
    base.update(kw)
    return Qwen4ExpTextConfig(**base)


def _ours(cfg: Qwen4ExpTextConfig) -> NGramHash:
    return NGramHash.from_config(cfg.to_dict(), ple_layer_index=0)


def _hf_ids(emb: hf.Qwen4ExpTextNGramEmbedding, chunks, cache) -> list[np.ndarray]:
    """Run HF's forward chunk by chunk and capture the ids handed to the embedding lookup."""
    seen = []
    handle = emb.ngram_embedding.register_forward_hook(lambda m, inp, out: seen.append(inp[0].clone()))
    try:
        for chunk in chunks:
            emb(torch.tensor([chunk], dtype=torch.long), cache)
    finally:
        handle.remove()
    return [s[0].numpy() for s in seen]


def _sequences(eos: int, vocab: int, rng) -> dict[str, list[int]]:
    body = rng.integers(0, vocab, size=40).tolist()
    body = [t if t != eos else (eos + 1) % vocab for t in body]
    return {
        "no_eos": body,
        "eos_start": [eos] + body,
        "eos_middle": body[:13] + [eos] + body[13:27] + [eos, eos] + body[27:],
        "eos_end": body + [eos],
        "eos_pairs": [eos, body[0], eos, body[1], body[2], eos, eos, body[3]],
    }


# --- (a) ids equal HF's forward ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["no_eos", "eos_start", "eos_middle", "eos_end", "eos_pairs"])
def test_ids_match_hf_tiny_single_call(name):
    cfg = _text_config()
    torch.manual_seed(0)
    emb = hf.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, PLE_LAYER_IDX, 0)
    ours = _ours(cfg)
    seq = _sequences(cfg.eos_token_id, cfg.vocab_size, np.random.default_rng(1))[name]
    (want,) = _hf_ids(emb, [seq], None)
    got, ctx = ours.ngram_ids(seq)
    assert got.dtype == np.int64 and got.shape == (len(seq), 16)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(ctx, seq[-2:])
    # the looked-up rows are table[ids] head after head (the sidecar's row order)
    with torch.no_grad():
        out = emb(torch.tensor([seq]), None)[0].numpy()
    table = emb.ngram_embedding.weight.detach().numpy()
    np.testing.assert_array_equal(out, table[got].reshape(len(seq), -1))


@pytest.mark.parametrize("splits", [(1, 2), (5, 17), (13, 14), (14, 15), (28, 29), (40, 41)])
def test_ids_match_hf_tiny_three_chunk_cache(splits):
    cfg = _text_config()
    emb = hf.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, PLE_LAYER_IDX, 0)
    ours = _ours(cfg)
    seq = _sequences(cfg.eos_token_id, cfg.vocab_size, np.random.default_rng(2))["eos_middle"]
    a, b = splits  # EOS sits at 13, 28 and 29
    chunks = [seq[:a], seq[a:b], seq[b:]]
    want = _hf_ids(emb, chunks, DynamicCache(config=cfg))
    ctx = None
    for chunk, w in zip(chunks, want):
        got, ctx = ours.ngram_ids(chunk, ctx)
        np.testing.assert_array_equal(got, w)
    whole, _ = ours.ngram_ids(seq)
    np.testing.assert_array_equal(np.concatenate(want), whole)


class _IdCapture(torch.nn.Module):
    """Stand-in for the 320M-row nn.Embedding: records ids, returns zeros."""

    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.seen = []

    def forward(self, ids):
        self.seen.append(ids.clone())
        return torch.zeros(*ids.shape, self.embedding_dim)


@pytest.fixture(scope="module")
def real_hf_embedding():
    mp = pytest.MonkeyPatch()
    mp.setattr(torch.nn, "Embedding", _IdCapture)
    try:
        cfg = _text_config(**REAL)
        emb = hf.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, PLE_LAYER_IDX, 0)
    finally:
        mp.undo()
    assert isinstance(emb.ngram_embedding, _IdCapture)
    return cfg, emb


def test_ids_match_hf_real_hashing_random_tokens(real_hf_embedding):
    cfg, emb = real_hf_embedding
    ours = _ours(cfg)
    rng = np.random.default_rng(3)
    seqs = _sequences(cfg.eos_token_id, cfg.vocab_size, rng)
    seqs["long_random"] = rng.integers(0, cfg.vocab_size, size=3000).tolist()
    seqs["extremes"] = [0, cfg.vocab_size - 1, cfg.vocab_size - 1, 0, cfg.eos_token_id, cfg.vocab_size - 1]
    for name, seq in seqs.items():
        emb.ngram_embedding.seen.clear()
        emb(torch.tensor([seq]), None)
        got, _ = ours.ngram_ids(seq)
        np.testing.assert_array_equal(got, emb.ngram_embedding.seen[0][0].numpy(), err_msg=name)
        assert got.min() >= 0 and got.max() < ours.padded_rows
    seq = seqs["eos_middle"]
    emb.ngram_embedding.seen.clear()
    chunks = [seq[:9], seq[9:10], seq[10:]]
    cache = DynamicCache(config=cfg)
    for c in chunks:
        emb(torch.tensor([c]), cache)
    ctx = None
    for c, w in zip(chunks, emb.ngram_embedding.seen):
        got, ctx = ours.ngram_ids(c, ctx)
        np.testing.assert_array_equal(got, w[0].numpy())


# --- (b) real-config constants -----------------------------------------------------------------------------------------

def test_real_config_constants_match_hf(real_hf_embedding):
    cfg, emb = real_hf_embedding
    ours = _ours(cfg)
    np.testing.assert_array_equal(ours.multipliers, emb.layer_multipliers.numpy())
    np.testing.assert_array_equal(ours.head_primes, emb.ngram_heads_vocab_sizes.numpy())
    np.testing.assert_array_equal(ours.head_offsets, emb.ngram_heads_offsets.numpy())
    assert ours.padded_rows == emb.ngram_embedding.num_embeddings == 320_001_536
    assert ours.head_dim == emb.ngram_embedding.embedding_dim == 160
    assert ours.eos_token_id == 248044
    print("\nreal-config multipliers", ours.multipliers.tolist())
    print("real-config head primes", ours.head_primes.tolist())
    print("real-config head offsets", ours.head_offsets.tolist())
    print("real-config total rows", ours.total_rows, "padded rows", ours.padded_rows)


def test_real_config_json_if_present():
    path = Path(r"E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ\config.json")
    if not path.exists():
        pytest.skip("checkpoint config.json not downloaded")
    from_file = NGramHash.from_config_file(path)
    ref = _ours(_text_config(**REAL))
    for attr in ("multipliers", "head_primes", "head_offsets"):
        np.testing.assert_array_equal(getattr(from_file, attr), getattr(ref, attr))
    assert (from_file.padded_rows, from_file.head_dim, from_file.eos_token_id) == (320_001_536, 160, 248044)


# --- (c) repack + verify on a synthetic 3-shard checkpoint ------------------------------------------------------------

def _build_checkpoint(root: Path, *, bad_buffer: bool = False) -> tuple[Path, np.ndarray]:
    """config.json + index + four safetensors files holding a 3-shard bf16 table and the hash buffers.

    shard_0 shares embedding-model-00002 with another tensor (kept), shard_1 sits alone in model-00002 (kept: not an
    embedding-model file), shard_2 sits alone in embedding-model-00001 (deletable)."""
    from safetensors.torch import save_file

    cfg = _text_config(ple_embed_dim=2560, hidden_size=2560, split_ngram_parts=3)
    h = _ours(cfg)
    root.mkdir(parents=True, exist_ok=True)
    full = cfg.to_dict()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": full}), encoding="utf-8")
    rng = np.random.default_rng(4)
    table = torch.from_numpy(rng.standard_normal((h.padded_rows, h.head_dim), dtype=np.float32)).to(torch.bfloat16)
    parts = torch.tensor_split(table, [470, 940])
    pre = f"model.language_model.layers.{PLE_LAYER_IDX}.ple.ple_embedding."
    mult = torch.from_numpy(h.multipliers.copy())
    if bad_buffer:
        mult[1] += 2
    files = {
        "model-00001-of-00002.safetensors": {
            pre + "layer_multipliers": mult,
            pre + "ngram_heads_vocab_sizes": torch.from_numpy(h.head_primes.copy()),
            pre + "ngram_heads_offsets": torch.from_numpy(h.head_offsets.copy()),
            "model.language_model.embed_tokens.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        },
        "model-00002-of-00002.safetensors": {pre + "ngram_embedding.shard_1.weight": parts[1].clone()},
        "embedding-model-00001-of-00002.safetensors": {pre + "ngram_embedding.shard_2.weight": parts[2].clone()},
        "embedding-model-00002-of-00002.safetensors": {
            "model.language_model.layers.0.dummy.weight": torch.ones(3, dtype=torch.bfloat16),
            pre + "ngram_embedding.shard_0.weight": parts[0].clone(),
        },
    }
    weight_map = {}
    for file, tensors in files.items():
        save_file(tensors, str(root / file), metadata={"format": "pt"})
        weight_map.update({k: file for k in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}),
                                                      encoding="utf-8")
    return root, table.view(torch.int16).numpy().view(np.uint8).reshape(h.padded_rows, -1)


def test_header_layout():
    assert ns._HEADER.size == 376
    hdr = ns.SidecarHeader(version=1, rows=5, row_bytes=320, heads=16, head_dim=160, ngram_size=3, heads_per_ngram=8,
                           eos_token_id=248044, seed=1234, multipliers=(1, 2, 3), head_offsets=tuple(range(16)),
                           head_primes=tuple(range(100, 116)), padded_rows=5, body_sha256=bytes(range(32)))
    raw = hdr.pack()
    assert len(raw) == 4096 and raw[:8] == b"R4DXNGRM"
    assert int.from_bytes(raw[12:20], "little") == 5
    assert int.from_bytes(raw[40:48], "little", signed=True) == 248044
    assert int.from_bytes(raw[56:64], "little") == 1
    assert int.from_bytes(raw[80 + 8 * 15:88 + 8 * 15], "little") == 15
    assert int.from_bytes(raw[208:216], "little") == 100
    assert int.from_bytes(raw[336:344], "little") == 5
    assert raw[344:376] == bytes(range(32)) and not any(raw[376:])
    assert ns.SidecarHeader.unpack(raw) == hdr


def test_repack_verify_roundtrip(tmp_path):
    import hashlib

    model, table = _build_checkpoint(tmp_path / "model")
    out = tmp_path / "t.r4dx-ngram"
    assert ns.main(["repack", "--model-dir", str(model), "--out", str(out)]) == 0
    assert not (tmp_path / "t.r4dx-ngram.partial").exists()
    raw = out.read_bytes()
    hdr = ns.SidecarHeader.unpack(raw[:4096])
    h = NGramHash.from_config_file(model / "config.json")
    assert (hdr.rows, hdr.padded_rows, hdr.row_bytes, hdr.heads, hdr.head_dim) == (h.padded_rows, h.padded_rows, 320,
                                                                                    16, 160)
    assert hdr.head_primes == tuple(h.head_primes.tolist()) and hdr.head_offsets == tuple(h.head_offsets.tolist())
    assert hdr.multipliers == tuple(h.multipliers.tolist())
    assert (hdr.eos_token_id, hdr.seed, hdr.ngram_size, hdr.heads_per_ngram) == (7, 1234, 3, 8)
    body = raw[4096:]
    assert body == table.tobytes()  # global id order == HF's shard concatenation
    assert hdr.body_sha256 == hashlib.sha256(body).digest()
    assert ns.main(["verify", "--model-dir", str(model), "--sidecar", str(out), "--samples", "200"]) == 0
    assert ns.verify(model, out, samples=0) == []

    # one flipped body byte: the sha256 rejects it, and so does the row check (row 470 = shard_1's first row, which
    # verify always compares)
    bad = bytearray(raw)
    bad[4096 + 470 * 320 + 5] ^= 0x01
    corrupt = tmp_path / "corrupt.r4dx-ngram"
    corrupt.write_bytes(bad)
    errors = ns.verify(model, corrupt, samples=0)
    assert errors[0].startswith("body sha256")
    assert ns.main(["verify", "--model-dir", str(model), "--sidecar", str(corrupt)]) == 1
    full = ns.verify(model, corrupt, samples=10)
    assert any(e.startswith("row 470 (") and "shard_1.weight row 0)" in e for e in full)
    bad = bytearray(raw)
    bad[4096 + 777 * 320 + 319] ^= 0x80  # a row no sample visits: the sha256 alone must reject it
    corrupt.write_bytes(bad)
    errors = ns.verify(model, corrupt, samples=0)
    assert len(errors) == 1 and errors[0].startswith("body sha256")

    # a header that disagrees with the checkpoint, a truncated body, a wrong magic
    bad = bytearray(raw)
    bad[208:216] = (h.head_primes[0] + 2).item().to_bytes(8, "little")
    (tmp_path / "hdr.r4dx-ngram").write_bytes(bad)
    assert any(e.startswith("header head_primes") for e in ns.verify(model, tmp_path / "hdr.r4dx-ngram"))
    (tmp_path / "short.r4dx-ngram").write_bytes(raw[:-320])
    assert any(e.startswith("file size") for e in ns.verify(model, tmp_path / "short.r4dx-ngram"))
    (tmp_path / "magic.r4dx-ngram").write_bytes(b"X" + raw[1:])
    assert ns.verify(model, tmp_path / "magic.r4dx-ngram")[0].startswith("header: bad magic")


def test_ids_index_the_sidecar(tmp_path):
    """Sidecar rows fetched by our ids equal HF's embedding output when HF's table is the concatenated shards."""
    model, table = _build_checkpoint(tmp_path / "model")
    out = tmp_path / "t.r4dx-ngram"
    ns.repack(model, out, log=lambda *_: None)
    cfg = _text_config(ple_embed_dim=2560, hidden_size=2560, split_ngram_parts=3)
    emb = hf.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, PLE_LAYER_IDX, 0)
    body = np.frombuffer(out.read_bytes()[4096:], dtype=np.uint8).reshape(-1, 320)
    with torch.no_grad():
        emb.ngram_embedding.weight.copy_(torch.from_numpy(body.copy()).view(torch.bfloat16).float())
    seq = _sequences(cfg.eos_token_id, cfg.vocab_size, np.random.default_rng(5))["eos_middle"]
    with torch.no_grad():
        want = emb(torch.tensor([seq]), None)[0].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint8)
    ids, _ = NGramHash.from_config_file(model / "config.json").ngram_ids(seq)
    np.testing.assert_array_equal(body[ids].reshape(len(seq), -1), want.reshape(len(seq), -1))


def test_repack_rejects_bad_buffer(tmp_path):
    model, _ = _build_checkpoint(tmp_path / "model", bad_buffer=True)
    with pytest.raises(ValueError, match="layer_multipliers"):
        ns.repack(model, tmp_path / "t.r4dx-ngram", log=lambda *_: None)
    assert not list(tmp_path.glob("*.r4dx-ngram*"))


def test_delete_shards_after_verify(tmp_path):
    model, _ = _build_checkpoint(tmp_path / "model")
    out = tmp_path / "t.r4dx-ngram"
    assert ns.main(["repack", "--model-dir", str(model), "--out", str(out), "--delete-shards-after-verify",
                    "--samples", "50"]) == 0
    left = sorted(p.name for p in model.glob("*.safetensors"))
    assert left == ["embedding-model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors"]


def test_delete_skipped_when_verify_fails(tmp_path, monkeypatch):
    model, _ = _build_checkpoint(tmp_path / "model")
    monkeypatch.setattr(ns, "verify", lambda *a, **k: ["forced failure"])
    assert ns.main(["repack", "--model-dir", str(model), "--out", str(tmp_path / "t.r4dx-ngram"),
                    "--delete-shards-after-verify"]) == 1
    assert len(list(model.glob("*.safetensors"))) == 4
