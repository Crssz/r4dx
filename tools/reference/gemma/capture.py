"""tools/reference/gemma/capture.py -- the Gemma 4 side of hessian_capture.py (docs/gemma4-plan.md M1-25).

    GemmaCaptureRef(model_dir, device)   the surface hessian_capture.run_capture drives, over GemmaReference:
                                         `embed`, `build_layer`, `capture_layer_forward` (per-layer-type rope,
                                         mask and the shared-kv dict of the HF decoder layer), `final_norm_hidden`
    build_tokens_corpus(path, seq_len)   the calibration corpus from tools/reference/gemma/gen_samples.py's
                                         token-id JSONL (the "tokens" gen format): the model's own ids, no
                                         tokenizer and no chat template involved
    TokenIdsCorpus                       the stand-in for common.RefTokenizer in the manifest's corpus record

torch / transformers are imported lazily (build_tokens_corpus and TokenIdsCorpus need neither), so the
dry-run of hessian_capture.py --arch gemma4_unified runs on any python with numpy.
"""

from __future__ import annotations

import hashlib
import json
from collections import UserDict
from pathlib import Path

GEN_CATEGORIES = ("thai_prose", "english_prose", "chat", "code", "multilingual")


class TokenIdsCorpus:
    """What `corpus_record` / `print` need of a tokenizer when the corpus is already token ids."""

    mode = "token-ids"

    def describe(self) -> str:
        return "none: the corpus is token ids (gen_samples.py's token_ids), no tokenizer is involved"

    def provenance(self) -> dict:
        return {"mode": self.mode,
                "note": "calibration windows are the model's own token ids (gen_samples.py: raw = BOS + generated "
                        "ids, chat = the rendered final turn + its generated ids); nothing is re-tokenized"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_token_samples(path: Path) -> tuple[list[dict], list[dict]]:
    """(usable samples, rejected samples) of a gen_samples.py JSONL, usable ones in id order. A line that
    breaks the contract (not JSON, no id / category / token_ids, token ids not ints, a usable sample whose
    kl_token_overlap is > 0) stops the run: a corpus must never include text of the held-out KL set."""
    usable, rejected, seen = [], [], set()
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"[hessian] {path}:{ln}: not JSON ({e})") from None
            for key in ("id", "category", "rejected"):
                if key not in s:
                    raise SystemExit(f"[hessian] {path}:{ln}: sample has no '{key}' (not a gen_samples.py line)")
            if s["id"] in seen:
                raise SystemExit(f"[hessian] {path}:{ln}: duplicate sample id {s['id']!r}")
            seen.add(s["id"])
            if s["category"] not in GEN_CATEGORIES:
                raise SystemExit(f"[hessian] {path}:{ln}: category {s['category']!r} is not one of {GEN_CATEGORIES}")
            if s["rejected"]:
                rejected.append(s)
                continue
            ids = s.get("token_ids")
            if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(t, int) for t in ids):
                raise SystemExit(f"[hessian] {path}:{ln}: sample {s['id']!r} has no usable 'token_ids' "
                                 f"(a gen_samples.py file written before the token format?)")
            if int(s.get("kl_token_overlap") or 0) > 0:
                raise SystemExit(f"[hessian] {path}:{ln}: sample {s['id']!r} shares a token run with the held-out "
                                 f"KL corpus (kl_token_overlap {s['kl_token_overlap']}) but is not marked rejected")
            usable.append(s)
    usable.sort(key=lambda s: s["id"])
    return usable, rejected


def build_tokens_corpus(path: Path, seq_len: int, max_seqs: int | None = None):
    """Sequences from a token-id JSONL: one per usable sample, its first `seq_len` ids (a longer sample's
    tail is dropped and counted: the first window keeps BOS / the chat prefix, a later one would not).
    `max_seqs`: at most that many per category, spread evenly over the category's samples (id order).
    Returns (seqs, sources) in hessian_capture.build_corpus' shapes: `Seq` is imported lazily from
    hessian_capture (this module stays importable without it)."""
    from hessian_capture import Seq

    path = Path(path)
    usable, rejected = load_token_samples(path)
    sha = _sha256_file(path)
    seqs, sources = [], []
    for cat in GEN_CATEGORIES:
        mine = [s for s in usable if s["category"] == cat]
        if max_seqs is not None and len(mine) > max_seqs:
            step = len(mine) / max_seqs
            mine = [mine[int(i * step)] for i in range(max_seqs)]
        dropped = 0
        for s in mine:
            ids = [int(t) for t in s["token_ids"]]
            dropped += max(0, len(ids) - seq_len)
            seqs.append(Seq(s["id"], "tokens", ids[:seq_len]))
        n_tok = sum(len(q.token_ids) for q in seqs if q.name.startswith(cat + "/"))
        sources.append({"source": "tokens", "name": cat, "path": str(path), "sha256": sha,
                        "samples_used": [s["id"] for s in mine], "samples_available": len(
                            [s for s in usable if s["category"] == cat]),
                        "samples_rejected": len([s for s in rejected if s["category"] == cat]),
                        "tokens": n_tok, "sequences": len(mine), "dropped_tail_tokens": dropped,
                        "window": f"the first {seq_len} ids of each sample"})
    sources = [s for s in sources if s["sequences"] > 0]
    if not seqs:
        raise SystemExit(f"[hessian] {path}: no usable (non-rejected) sample")
    return seqs, sources


class GemmaCaptureRef:
    """GemmaReference, streaming (one layer on the device at a time), behind the surface
    hessian_capture.run_capture drives. Numerics are GemmaReference's (= HF's: embed scale in bf16, the
    sandwich norms, layer_scalar, attention scaling 1.0)."""

    def __init__(self, model_dir, device, dtype=None, verbose: bool = True):
        from gemma.ref import GemmaReference

        self.ref = GemmaReference(model_dir, device, dtype=dtype, resident=False, verbose=verbose)
        self.device = device
        self.layer_types = list(self.ref.layer_types)
        self.n_layers = self.ref.n_layers
        self.text_config = self.ref.text_config
        self.vocab_size = self.ref.vocab_size
        self._cache: dict[int, tuple] = {}

    def embed(self, token_ids):
        """[1, T, hidden] bf16: the embedding rows times bf16(sqrt(hidden)) (the layer-major capture
        keeps one such tensor per sequence)."""
        return self.ref.embed(token_ids).unsqueeze(0)

    def build_layer(self, i: int):
        return self.ref.build_layer(i)

    def _tables(self, x, T: int):
        import torch
        from gemma.arch import FULL, SLIDING
        from gemma.ref import additive_mask

        if T not in self._cache:
            if len(self._cache) >= 4:  # sequences have many different lengths: keep a few tables, not all
                self._cache.clear()
            pos = torch.arange(T, device=self.device).unsqueeze(0)
            pe = {lt: self.ref.rotary(x, pos, lt) for lt in sorted(set(self.layer_types))}
            masks = {FULL: additive_mask(T, T, 0, None, self.device, self.ref.dtype),
                     SLIDING: additive_mask(T, T, 0, self.ref.arch.window, self.device, self.ref.dtype)}
            self._cache[T] = (pos, pe, masks)
        return self._cache[T]

    def capture_layer_forward(self, layer, i: int, x):
        """One decoder layer, exactly GemmaReference.forward_hidden's call: per-layer-type rope tables
        and mask (sliding keeps `kv > q - window`), a fresh shared-kv dict, no cache."""
        T = x.shape[1]
        pos, pe, masks = self._tables(x, T)
        lt = self.layer_types[i]
        return layer(x, shared_kv_states=UserDict(), position_embeddings=pe[lt], attention_mask=masks[lt],
                     position_ids=pos, past_key_values=None)

    def final_norm_hidden(self, h):
        """The post-final-norm hidden state of `h` [1, T, hidden]: the input of the (tied) lm_head,
        before the softcap."""
        return self.ref.final_norm(h)
