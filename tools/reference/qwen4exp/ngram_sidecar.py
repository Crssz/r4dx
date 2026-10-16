"""Repack the qwen4_exp n-gram embedding table into one r4dx sidecar file, and verify it (plan 2.3, 4.2, M0-3).

The HF checkpoint stores the PLE n-gram table as `split_ngram_parts` bf16 shards
`model.language_model.layers.{L}.ple.ple_embedding.ngram_embedding.shard_{i}.weight` ([rows_i, head_dim]),
spread over `model-*.safetensors` and `embedding-model-*.safetensors`; HF concatenates them in shard index
order. The sidecar holds the same rows in global row-id order behind a fixed 4096-byte header, so the C++
reader (M1-9) fetches row `r` at byte `4096 + r * row_bytes`. CPU only, no torch.

Header (little-endian, packed, zero padded to 4096 bytes):

    off  type        field
      0  char[8]     magic "R4DXNGRM"
      8  u32         version (1)
     12  u64         rows (rows in the body)
     20  u32         row_bytes (head_dim * 2, 320)
     24  u32         heads (16)
     28  u32         head_dim (160)
     32  u32         ngram_size (3)
     36  u32         heads_per_ngram (8)
     40  i64         eos token id
     48  u64         seed
     56  i64[3]      layer multipliers
     80  u64[16]     head offsets
    208  u64[16]     head primes (per-head vocab sizes)
    336  u64         padded_rows (HF's padded n-gram vocab; equals rows)
    344  u8[32]      sha256 of the body
    376  zero padding

    python ngram_sidecar.py repack --model-dir E:\\models\\r4dx\\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ --out E:\\models\\r4dx\\swift15-flash-next.r4dx-ngram
    python ngram_sidecar.py verify --model-dir E:\\models\\r4dx\\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ --sidecar E:\\models\\r4dx\\swift15-flash-next.r4dx-ngram --samples 1000

`repack --delete-shards-after-verify` runs `verify` on the new file and, only if it passes, deletes the
`embedding-model-*` files whose tensors are all n-gram shards.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngram_ids import NGramHash, text_config_of  # noqa: E402

MAGIC = b"R4DXNGRM"
VERSION = 1
HEADER_BYTES = 4096
NGRAM_SIZE = 3
HEADS = 16
_HEADER = struct.Struct(f"<8sIQIIIIIqQ{NGRAM_SIZE}q{HEADS}Q{HEADS}QQ32s")
_CHUNK_BYTES = 64 << 20
_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")


@dataclass(frozen=True)
class SidecarHeader:
    version: int
    rows: int
    row_bytes: int
    heads: int
    head_dim: int
    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    seed: int
    multipliers: tuple[int, ...]
    head_offsets: tuple[int, ...]
    head_primes: tuple[int, ...]
    padded_rows: int
    body_sha256: bytes

    def pack(self) -> bytes:
        raw = _HEADER.pack(MAGIC, self.version, self.rows, self.row_bytes, self.heads, self.head_dim, self.ngram_size,
                           self.heads_per_ngram, self.eos_token_id, self.seed, *self.multipliers, *self.head_offsets,
                           *self.head_primes, self.padded_rows, self.body_sha256)
        return raw + bytes(HEADER_BYTES - len(raw))

    @classmethod
    def unpack(cls, buf: bytes) -> "SidecarHeader":
        if len(buf) < HEADER_BYTES:
            raise ValueError(f"header is {len(buf)} bytes, expected {HEADER_BYTES}")
        v = _HEADER.unpack_from(buf)
        if v[0] != MAGIC:
            raise ValueError(f"bad magic {v[0]!r}")
        if any(buf[_HEADER.size:HEADER_BYTES]):
            raise ValueError("header padding is not zero")
        m0 = 10
        o0 = m0 + NGRAM_SIZE
        p0 = o0 + HEADS
        return cls(version=v[1], rows=v[2], row_bytes=v[3], heads=v[4], head_dim=v[5], ngram_size=v[6],
                   heads_per_ngram=v[7], eos_token_id=v[8], seed=v[9], multipliers=tuple(v[m0:o0]),
                   head_offsets=tuple(v[o0:p0]), head_primes=tuple(v[p0:p0 + HEADS]), padded_rows=v[p0 + HEADS],
                   body_sha256=v[p0 + HEADS + 1])

    def diff(self, other: "SidecarHeader", skip=("body_sha256",)) -> list[str]:
        return [f"{f.name}: {getattr(self, f.name)} != {getattr(other, f.name)}" for f in fields(self)
                if f.name not in skip and getattr(self, f.name) != getattr(other, f.name)]


def read_safetensors_header(path: Path) -> tuple[int, dict]:
    """(byte offset of the data section, header dict without `__metadata__`)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


@dataclass(frozen=True)
class _Shard:
    name: str
    path: Path
    offset: int  # absolute byte offset of the tensor in `path`
    rows: int


class NGramCheckpoint:
    """The n-gram shards and hash buffers of one HF qwen4_exp checkpoint directory (headers only in RAM)."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        tc = text_config_of(config)
        ple_layer_ids = tc.get("ple_layer_ids") or []
        if len(ple_layer_ids) != 1:
            raise ValueError(f"exactly one PLE layer is supported, config has ple_layer_ids={ple_layer_ids}")
        self.hash = NGramHash.from_config(config, ple_layer_index=0)
        if self.hash.ngram_size != NGRAM_SIZE or self.hash.heads != HEADS:
            raise ValueError(f"sidecar v{VERSION} needs ngram_size {NGRAM_SIZE} and {HEADS} heads, config has "
                             f"{self.hash.ngram_size} and {self.hash.heads}")
        self.parts = int(tc.get("split_ngram_parts", 512))
        self.prefix = f"model.language_model.layers.{ple_layer_ids[0] - 1}.ple.ple_embedding."
        self.row_bytes = self.hash.head_dim * 2

        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        shard_names = {k for k in self.weight_map if k.startswith(self.prefix + "ngram_embedding.shard_")}
        want = [f"{self.prefix}ngram_embedding.shard_{i}.weight" for i in range(self.parts)]
        if shard_names != set(want):
            raise ValueError(f"index has {len(shard_names)} n-gram shard tensors, expected shard_0..shard_{self.parts - 1}")
        self._headers: dict[str, tuple[int, dict]] = {}

        self.shards: list[_Shard] = []
        for name in want:
            path, offset, info = self._tensor(name)
            if info["dtype"] != "BF16" or len(info["shape"]) != 2 or info["shape"][1] != self.hash.head_dim:
                raise ValueError(f"{name}: {info['dtype']} {info['shape']}, expected BF16 [rows, {self.hash.head_dim}]")
            rows = info["shape"][0]
            if info["data_offsets"][1] - info["data_offsets"][0] != rows * self.row_bytes:
                raise ValueError(f"{name}: data_offsets do not match its shape")
            self.shards.append(_Shard(name, path, offset, rows))
        self.shard_starts = np.cumsum([0] + [s.rows for s in self.shards]).tolist()
        self.rows = self.shard_starts[-1]
        if self.rows != self.hash.padded_rows:
            raise ValueError(f"shards hold {self.rows} rows, the config derives {self.hash.padded_rows} padded rows")

        expect = {"layer_multipliers": self.hash.multipliers, "ngram_heads_vocab_sizes": self.hash.head_primes,
                  "ngram_heads_offsets": self.hash.head_offsets}
        for buf_name, derived in expect.items():
            stored = self._read_i64(self.prefix + buf_name)
            if not np.array_equal(stored, derived):
                raise ValueError(f"checkpoint {buf_name} {stored.tolist()} != derived {derived.tolist()}")

    def _header(self, file: str) -> tuple[int, dict]:
        if file not in self._headers:
            self._headers[file] = read_safetensors_header(self.model_dir / file)
        return self._headers[file]

    def _tensor(self, name: str) -> tuple[Path, int, dict]:
        if name not in self.weight_map:
            raise ValueError(f"{name} is not in model.safetensors.index.json")
        file = self.weight_map[name]
        data_start, header = self._header(file)
        if name not in header:
            raise ValueError(f"{name} is not in {file}")
        info = header[name]
        return self.model_dir / file, data_start + info["data_offsets"][0], info

    def _read_i64(self, name: str) -> np.ndarray:
        path, offset, info = self._tensor(name)
        if info["dtype"] != "I64":
            raise ValueError(f"{name}: dtype {info['dtype']}, expected I64")
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        return np.frombuffer(raw, dtype="<i8").astype(np.int64)

    def header(self, body_sha256: bytes) -> SidecarHeader:
        h = self.hash
        return SidecarHeader(version=VERSION, rows=self.rows, row_bytes=self.row_bytes, heads=h.heads,
                             head_dim=h.head_dim, ngram_size=h.ngram_size, heads_per_ngram=h.heads_per_ngram,
                             eos_token_id=h.eos_token_id, seed=h.seed, multipliers=tuple(int(x) for x in h.multipliers),
                             head_offsets=tuple(int(x) for x in h.head_offsets),
                             head_primes=tuple(int(x) for x in h.head_primes), padded_rows=h.padded_rows,
                             body_sha256=body_sha256)

    def iter_shard_chunks(self, shard: _Shard):
        """The shard's bytes in chunks of at most 64 MiB (never a whole shard in RAM)."""
        remaining = shard.rows * self.row_bytes
        with open(shard.path, "rb") as f:
            f.seek(shard.offset)
            while remaining:
                chunk = f.read(min(_CHUNK_BYTES, remaining))
                if not chunk:
                    raise IOError(f"{shard.path}: truncated in {shard.name}")
                remaining -= len(chunk)
                yield chunk

    def locate(self, row: int) -> tuple[_Shard, int]:
        i = bisect.bisect_right(self.shard_starts, row) - 1
        return self.shards[i], row - self.shard_starts[i]

    def deletable_files(self) -> list[Path]:
        """`embedding-model-*` files whose tensors (by the index and by their own header) are all n-gram shards."""
        files = sorted({s.path.name for s in self.shards if s.path.name.startswith("embedding-model-")})
        out = []
        for file in files:
            in_index = {k for k, v in self.weight_map.items() if v == file}
            _, header = read_safetensors_header(self.model_dir / file)
            names = in_index | set(header)
            if all(n.startswith(self.prefix) and _SHARD_RE.search(n) for n in names):
                out.append(self.model_dir / file)
        return out


def repack(model_dir: str | Path, out: str | Path, log=print) -> SidecarHeader:
    ckpt = NGramCheckpoint(model_dir)
    out = Path(out)
    partial = out.with_name(out.name + ".partial")
    sha = hashlib.sha256()
    try:
        with open(partial, "wb") as f:
            f.write(bytes(HEADER_BYTES))
            for i, shard in enumerate(ckpt.shards):
                for chunk in ckpt.iter_shard_chunks(shard):
                    sha.update(chunk)
                    f.write(chunk)
                log(f"shard {i + 1}/{len(ckpt.shards)} ({shard.path.name}) done")
            header = ckpt.header(sha.digest())
            f.seek(0)
            f.write(header.pack())
            f.flush()
            os.fsync(f.fileno())
        if partial.stat().st_size != HEADER_BYTES + ckpt.rows * ckpt.row_bytes:
            raise IOError(f"{partial}: size {partial.stat().st_size} is wrong")
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    log(f"wrote {out}: {header.rows} rows x {header.row_bytes} B, sha256 {header.body_sha256.hex()}")
    return header


def verify(model_dir: str | Path, sidecar: str | Path, samples: int = 1000, sample_seed: int = 0,
           log=print) -> list[str]:
    """Problems found (empty: the sidecar is a faithful copy of the checkpoint's n-gram table)."""
    ckpt = NGramCheckpoint(model_dir)
    sidecar = Path(sidecar)
    with open(sidecar, "rb") as f:
        try:
            header = SidecarHeader.unpack(f.read(HEADER_BYTES))
        except (ValueError, struct.error) as e:
            return [f"header: {e}"]
        errors = [f"header {d}" for d in header.diff(ckpt.header(b""))]
        size = sidecar.stat().st_size
        if size != HEADER_BYTES + header.rows * header.row_bytes:
            errors.append(f"file size {size} != {HEADER_BYTES} + {header.rows} * {header.row_bytes}")
        if errors:
            return errors

        sha = hashlib.sha256()
        f.seek(HEADER_BYTES)
        while chunk := f.read(_CHUNK_BYTES):
            sha.update(chunk)
        if sha.digest() != header.body_sha256:
            errors.append(f"body sha256 {sha.hexdigest()} != header {header.body_sha256.hex()}")

        rng = np.random.default_rng(sample_seed)
        rows = set(rng.integers(0, ckpt.rows, size=samples).tolist()) if samples > 0 else set()
        for start, end in zip(ckpt.shard_starts[:-1], ckpt.shard_starts[1:]):
            rows.update((start, end - 1))  # every shard's first and last row
        handles: dict[Path, object] = {}
        try:
            for row in sorted(rows):
                shard, local = ckpt.locate(row)
                src = handles.get(shard.path)
                if src is None:
                    src = handles[shard.path] = open(shard.path, "rb")
                src.seek(shard.offset + local * ckpt.row_bytes)
                want = src.read(ckpt.row_bytes)
                f.seek(HEADER_BYTES + row * header.row_bytes)
                got = f.read(header.row_bytes)
                if got != want:
                    errors.append(f"row {row} ({shard.name} row {local}) differs from the checkpoint")
                    if len(errors) >= 20:
                        break
        finally:
            for h in handles.values():
                h.close()
    log(f"verify {sidecar}: {len(rows)} rows compared, header sha256 {header.body_sha256.hex()}, "
        f"{'OK' if not errors else f'{len(errors)} problem(s)'}")
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("repack", help="write the sidecar from the checkpoint shards")
    rp.add_argument("--model-dir", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--samples", type=int, default=1000, help="random rows compared by the post-repack verify")
    rp.add_argument("--delete-shards-after-verify", action="store_true",
                    help="verify the new sidecar, then delete the embedding-model-* files holding only n-gram shards")
    vp = sub.add_parser("verify", help="check a sidecar against the checkpoint")
    vp.add_argument("--model-dir", required=True)
    vp.add_argument("--sidecar", required=True)
    vp.add_argument("--samples", type=int, default=1000)
    vp.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "repack":
        header = repack(args.model_dir, args.out)
        print(json.dumps({"rows": header.rows, "multipliers": header.multipliers, "head_primes": header.head_primes,
                          "head_offsets": header.head_offsets, "sha256": header.body_sha256.hex()}))
        if args.delete_shards_after_verify:
            errors = verify(args.model_dir, args.out, args.samples)
            if errors:
                print("verify FAILED, nothing deleted:\n  " + "\n  ".join(errors))
                return 1
            for path in NGramCheckpoint(args.model_dir).deletable_files():
                path.unlink()
                print(f"deleted {path}")
        return 0

    errors = verify(args.model_dir, args.sidecar, args.samples, args.sample_seed)
    if errors:
        print("verify FAILED:\n  " + "\n  ".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
