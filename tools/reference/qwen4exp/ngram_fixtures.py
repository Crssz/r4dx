"""Fixtures for the C++ n-gram id generator and NVMe row reader tests (plan 4.8; tests/model/qwen4exp/test_ngram_*.cpp).

Writes into `E:\\models\\r4dx\\q4exp-fixtures\\NgramReader\\` (override with --out):

    header.bin            the first 4096 bytes of the real sidecar
    ids.bin               token sequences + the ids `ngram_ids.py` computes for them (chunked == one call)
    rows.bin              ids and the 320 row bytes read from the REAL sidecar (ids of ids.bin case 0's first
                          24 tokens, 200 random rows, edge rows, rows that straddle a 4 KiB sector)
    small.r4dx-ngram      a complete 8-row sidecar (real header fields, rows = padded_rows = 8, body = real
                          rows 0..7, hashlib sha256 of that body in the header)

Binary layouts (little-endian):

    ids.bin   "NGID0001", u32 n_cases, then per case
                u32 T, u32 n_chunks, u32 chunk_lens[n_chunks] (sum == T, zero allowed),
                u32 has_ctx, i64 ctx[2] (the previous_context when has_ctx, else zeros = start of sequence),
                i32 tokens[T], i64 ids[T * 16], i64 final_ctx[2]
    rows.bin  "NGRW0001", u32 n, then n x (i64 row id, u8 row[320])

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe ngram_fixtures.py [--sidecar PATH] [--config PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngram_ids import NGramHash  # noqa: E402
from ngram_sidecar import HEADER_BYTES, SidecarHeader  # noqa: E402

DEFAULT_SIDECAR = r"E:\models\r4dx\swift15-flash-next.r4dx-ngram"
DEFAULT_CONFIG = r"E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ\config.json"
DEFAULT_OUT = r"E:\models\r4dx\q4exp-fixtures\NgramReader"
ROW_BYTES = 320
HEADS = 16
SECTOR = 4096


def make_cases(h: NGramHash, rng: np.random.Generator):
    """(tokens, chunk_lens, previous_context | None) tuples."""
    vocab = h.vocab_size
    eos = h.eos_token_id
    cases = []

    # 0: plain random tokens over the whole vocabulary, three calls (first 24 tokens feed rows.bin)
    t = rng.integers(0, vocab, size=300)
    t[0] = 0
    t[1] = vocab - 1
    cases.append((t, [100, 150, 50], None))

    # 1: EOS-dense, calls of 1 and 0 tokens included, an EOS first and last
    t = rng.integers(0, vocab, size=400)
    t[rng.random(400) < 0.15] = eos
    t[0] = eos
    t[-1] = eos
    cases.append((t, [1, 0, 5, 0, 2, 392], None))

    # 2: EOS placed around the chunk boundaries (100 | 101 ...), double EOS, EOS directly after the context
    t = rng.integers(0, vocab, size=256)
    for pos in (0, 1, 98, 99, 100, 101, 102, 199, 200, 255):
        t[pos] = eos
    cases.append((t, [100, 100, 56], None))

    # 3: no EOS at all, one 256-token chunk (a prefill chunk) after a non-EOS context
    t = rng.integers(0, vocab, size=256)
    t[t == eos] = 1
    cases.append((t, [256], np.array([int(rng.integers(0, vocab)), int(rng.integers(0, vocab))])))

    # 4: contexts with an EOS in either slot, short sequences
    for ctx in ((eos, 17), (17, eos), (eos, eos)):
        t = rng.integers(0, vocab, size=9)
        cases.append((t, [3, 6], np.array(ctx)))

    # 5: a single token, and a repeated token
    cases.append((np.array([42]), [1], None))
    cases.append((np.full(20, 5), [7, 13], None))
    return cases


def write_ids(h: NGramHash, path: Path, rng: np.random.Generator):
    cases = make_cases(h, rng)
    out = [b"NGID0001", struct.pack("<I", len(cases))]
    first_ids = None
    for index, (tokens, chunks, ctx) in enumerate(cases):
        tokens = np.asarray(tokens, dtype=np.int64)
        assert sum(chunks) == len(tokens), (index, chunks, len(tokens))
        ids, final = h.ngram_ids(tokens, ctx)
        # chunked == single call, ids and final context (the property the C++ carries across calls)
        pos, cur, parts = 0, ctx, []
        for c in chunks:
            part, cur = h.ngram_ids(tokens[pos:pos + c], cur)
            parts.append(part)
            pos += c
        chunked = np.concatenate(parts, axis=0)
        assert np.array_equal(chunked, ids), f"case {index}: chunked ids differ from the single call"
        assert np.array_equal(np.asarray(cur), final), f"case {index}: chunked context differs"
        assert ids.shape == (len(tokens), HEADS) and ids.min() >= 0 and ids.max() < h.padded_rows
        out.append(struct.pack("<II", len(tokens), len(chunks)))
        out.append(np.asarray(chunks, dtype="<u4").tobytes())
        out.append(struct.pack("<I", 0 if ctx is None else 1))
        out.append(np.asarray([0, 0] if ctx is None else ctx, dtype="<i8").tobytes())
        out.append(tokens.astype("<i4").tobytes())
        out.append(ids.astype("<i8").tobytes())
        out.append(np.asarray(final, dtype="<i8").tobytes())
        if index == 0:
            first_ids = ids
    path.write_bytes(b"".join(out))
    return len(cases), first_ids


def write_rows(sidecar: Path, header: SidecarHeader, first_ids: np.ndarray, path: Path, rng: np.random.Generator):
    rows = header.rows
    ids = [int(i) for i in first_ids[:24].reshape(-1)]
    ids += [int(i) for i in rng.integers(0, rows, size=200)]
    ids += [0, 1, rows - 1, rows - 2] + [int(o) for o in header.head_offsets] + [int(o + p - 1) for o, p in zip(header.head_offsets, header.head_primes)]
    # rows straddling a sector boundary: (4096 + 320 * id) % 4096 > 4096 - 320
    straddle = [i for i in range(0, 100) if (HEADER_BYTES + ROW_BYTES * i) % SECTOR > SECTOR - ROW_BYTES][:6]
    straddle += [int(i) for i in rng.integers(0, rows - 20, size=40) if (HEADER_BYTES + ROW_BYTES * int(i)) % SECTOR > SECTOR - ROW_BYTES][:10]
    ids += straddle
    assert straddle, "no straddling row picked"
    out = [b"NGRW0001", struct.pack("<I", len(ids))]
    with open(sidecar, "rb") as f:
        for i in ids:
            assert 0 <= i < rows
            f.seek(HEADER_BYTES + ROW_BYTES * i)
            row = f.read(ROW_BYTES)
            assert len(row) == ROW_BYTES
            out.append(struct.pack("<q", i) + row)
    path.write_bytes(b"".join(out))
    return len(ids), len(straddle)


def write_small(sidecar: Path, header: SidecarHeader, path: Path):
    n = 8
    with open(sidecar, "rb") as f:
        f.seek(HEADER_BYTES)
        body = f.read(n * ROW_BYTES)
    small = dataclasses.replace(header, rows=n, padded_rows=n, body_sha256=hashlib.sha256(body).digest())
    path.write_bytes(small.pack() + body)
    return small.body_sha256.hex()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sidecar = Path(a.sidecar)
    with open(sidecar, "rb") as f:
        raw = f.read(HEADER_BYTES)
    header = SidecarHeader.unpack(raw)
    h = NGramHash.from_config_file(a.config)
    # the C++ takes its multipliers / primes / offsets from the header, so the header must be the HF-derived one
    assert tuple(int(x) for x in h.multipliers) == header.multipliers, "header multipliers != config-derived"
    assert tuple(int(x) for x in h.head_primes) == header.head_primes, "header primes != config-derived"
    assert tuple(int(x) for x in h.head_offsets) == header.head_offsets, "header offsets != config-derived"
    assert h.eos_token_id == header.eos_token_id and h.padded_rows == header.padded_rows

    (out / "header.bin").write_bytes(raw)
    rng = np.random.default_rng(a.seed)
    n_cases, first_ids = write_ids(h, out / "ids.bin", rng)
    n_rows, n_straddle = write_rows(sidecar, header, first_ids, out / "rows.bin", rng)
    sha = write_small(sidecar, header, out / "small.r4dx-ngram")
    print(f"wrote {out}: header.bin, ids.bin ({n_cases} cases), rows.bin ({n_rows} rows, {n_straddle} sector-straddling), "
          f"small.r4dx-ngram (sha256 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
