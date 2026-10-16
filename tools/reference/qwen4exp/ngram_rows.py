"""N-gram embedding rows for the qwen4_exp reference, without HF's in-RAM `nn.Embedding` (95 GiB on the real model).

`open_ngram_rows(model_dir, ngram_hash, sidecar=None)` returns an `NGramRows` whose `rows(ids)` gathers bf16 rows
[T, heads, head_dim] for global row ids [T, heads] (from `ngram_ids.NGramHash.ngram_ids`):

    SidecarNGramRows   np.memmap of a `ngram_sidecar.py repack` file (the real model: the checkpoint's embedding shards
                       were deleted after the repack; header checked against the config's hash).
    TableNGramRows     every checkpoint shard concatenated in RAM, exactly HF's `ngram_embedding.weight` (tiny model).
    ShardNGramRows     rows read on demand from the checkpoint's shard files (all shards present, table too big for RAM).

The three return the same bytes for the same rows (test_ref.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import raw_safetensors_header, raw_safetensors_read  # noqa: E402
from qwen4exp.ngram_ids import NGramHash  # noqa: E402
from qwen4exp.ngram_sidecar import HEADER_BYTES, NGramCheckpoint, SidecarHeader  # noqa: E402

TABLE_MAX_BYTES = 2 << 30


class NGramRows:
    head_dim: int

    def rows(self, ids: np.ndarray) -> torch.Tensor:
        """bf16 [T, heads, head_dim] (CPU) for int64 global row ids [T, heads]."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class TableNGramRows(NGramRows):
    def __init__(self, table: torch.Tensor, label: str):
        if table.dtype != torch.bfloat16 or table.dim() != 2:
            raise ValueError(f"n-gram table must be bf16 [rows, head_dim], got {table.dtype} {tuple(table.shape)}")
        self.table = table
        self.head_dim = table.shape[1]
        self.label = label

    def rows(self, ids: np.ndarray) -> torch.Tensor:
        return self.table[torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64))]

    def describe(self) -> str:
        return f"in-RAM table {self.label} ({self.table.shape[0]} rows)"


class SidecarNGramRows(NGramRows):
    def __init__(self, path: str | Path, ngram_hash: NGramHash):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            header = SidecarHeader.unpack(f.read(HEADER_BYTES))
        h = ngram_hash
        want = {
            "rows": h.padded_rows, "padded_rows": h.padded_rows, "row_bytes": h.head_dim * 2, "heads": h.heads,
            "head_dim": h.head_dim, "ngram_size": h.ngram_size, "heads_per_ngram": h.heads_per_ngram,
            "eos_token_id": h.eos_token_id, "seed": h.seed, "multipliers": tuple(int(x) for x in h.multipliers),
            "head_offsets": tuple(int(x) for x in h.head_offsets), "head_primes": tuple(int(x) for x in h.head_primes),
        }
        bad = [f"{k}: sidecar {getattr(header, k)} != config {v}" for k, v in want.items() if getattr(header, k) != v]
        if bad:
            raise ValueError(f"{self.path} does not match the model config: " + "; ".join(bad))
        size = self.path.stat().st_size
        if size != HEADER_BYTES + header.rows * header.row_bytes:
            raise ValueError(f"{self.path}: size {size} != {HEADER_BYTES} + {header.rows} * {header.row_bytes}")
        self.header = header
        self.head_dim = header.head_dim
        self.mm = np.memmap(self.path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(header.rows, header.head_dim))

    def rows(self, ids: np.ndarray) -> torch.Tensor:
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        flat = ids.reshape(-1)
        uniq, inverse = np.unique(flat, return_inverse=True)  # sorted reads: sequential on NVMe
        data = np.ascontiguousarray(self.mm[uniq])[inverse]
        return torch.from_numpy(data).view(torch.bfloat16).reshape(*ids.shape, self.head_dim)

    def describe(self) -> str:
        return f"sidecar {self.path} ({self.header.rows} rows, sha256 {self.header.body_sha256.hex()[:12]})"


class ShardNGramRows(NGramRows):
    def __init__(self, ckpt: NGramCheckpoint):
        self.ckpt = ckpt
        self.head_dim = ckpt.hash.head_dim
        self._headers: dict[Path, tuple[dict, int]] = {}
        missing = sorted({str(s.path) for s in ckpt.shards if not s.path.exists()})
        if missing:
            raise FileNotFoundError(f"{len(missing)} n-gram shard file(s) missing, e.g. {missing[0]}; "
                                    "pass the sidecar (--ngram-sidecar)")

    def rows(self, ids: np.ndarray) -> torch.Tensor:
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        out = np.empty((ids.size, self.head_dim), dtype=np.uint16)
        for k, row in enumerate(ids.reshape(-1).tolist()):
            shard, local = self.ckpt.locate(row)
            hdr = self._headers.get(shard.path)
            if hdr is None:
                hdr = self._headers[shard.path] = raw_safetensors_header(shard.path)
            out[k] = raw_safetensors_read(shard.path, shard.name, hdr, rows=(local, local + 1)).view(torch.uint16).numpy()[0]
        return torch.from_numpy(out).view(torch.bfloat16).reshape(*ids.shape, self.head_dim)

    def describe(self) -> str:
        return f"{len(self.ckpt.shards)} checkpoint shards ({self.ckpt.rows} rows), read on demand"


def open_ngram_rows(model_dir: str | Path, ngram_hash: NGramHash, sidecar: str | Path | None = None,
                    table_max_bytes: int = TABLE_MAX_BYTES) -> NGramRows:
    """Sidecar when given; else the checkpoint's own shards (in RAM when the table is at most `table_max_bytes`)."""
    if sidecar is not None:
        return SidecarNGramRows(sidecar, ngram_hash)
    ckpt = NGramCheckpoint(model_dir)
    shards = ShardNGramRows(ckpt)  # raises when a shard file is missing
    if ckpt.rows * ckpt.row_bytes > table_max_bytes:
        return shards
    parts = [raw_safetensors_read(s.path, s.name, raw_safetensors_header(s.path)) for s in ckpt.shards]
    return TableNGramRows(torch.cat(parts, dim=0), f"{len(parts)} shards of {Path(model_dir).name}")
