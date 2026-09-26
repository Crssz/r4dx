"""tools/reference/gguf_validate.py

Checks every tensor of a llama.cpp `qwen35` GGUF against the bf16 HF checkpoint it was made from,
through `gguf_dequant.py`'s dequantizers and inverse layout transforms, which are exactly the code
path `full_logits_golden.py --weights-gguf` uses. There is no `gguf` package to compare against, so
this is the main evidence that the dequantization and the name/layout map are right.

For each GGUF tensor:

  quantized (every type but F32/BF16): dequantize, undo the layout transform, and measure against
      the checkpoint tensor W:
          rel = ||deq - W||_F / ||W||_F        cos = <deq, W> / (||deq|| ||W||)
      A wrong mapping or a missed reorder puts rel near 1 and cos far below 1. A quantizer's
      own error puts rel at 0.5-16% (by type). Each tensor must be inside its type's
      `WEIGHT_ERROR_BOUNDS` and within 1.4x its type's median rel. Both were fitted to the Unsloth
      file (see WEIGHT_ERROR_BOUNDS), so they are a tripwire, not independent evidence. For the
      head-reordered tensors, `rel_no_inverse` (the error if the reorder were NOT undone) is
      reported as well, as proof that the reorder is real.
      Per row as well (shape[0] of the HF tensor): the pooled figure hides a single bad row, so
      each row's own rel is measured, and rows above ROW_ERROR_FACTOR x the type bound are listed
      with their GGUF row, their cosine, and their blocks' f16 scales (how many are coarse, below
      gguf_dequant.F16_COARSE_SCALE: a row of ~1e-5 values, like token_embd row 107517 of the
      Unsloth file, is below what the type's scale resolves). Listed rows are reported, not failed.
      Against ggml's own C (`--ggml-c`, default: a local llama.cpp build's ggml-base library, skipped
      if there is none): the first, last, worst, every listed and a few random rows of each
      quantized tensor must decode bit-identically in ggml's `dequantize_row_*`. A difference fails
      the tensor. This is the check that does not depend on bounds fitted to the file.
  lossless (F32, BF16): the checkpoint value put through the converter's forward transform
      (`Qwen35Map.from_hf`) must equal the stored value bit for bit (-exp(A_log): within 1 ulp).
      These are the norms (+1 folded in), A_log, dt_bias, conv1d and the GDN norm.

Also fails on GGUF tensors with no HF counterpart and on shape mismatches. HF tensors the GGUF lacks
(the vision tower) are listed but are not failures: the GGUF-weights forward keeps those from the
checkpoint.

Writes `<out-dir>/gguf_validation.json` (every tensor) and `gguf_validation.md` (per type, per
class, per type x class). Exits 1 and prints every failure if anything is off.

numpy only, CPU only, memory-mapped on both sides:

    <venv>\\Scripts\\python.exe tools\\reference\\gguf_validate.py `
        --gguf D:\\...\\Qwen3.8-27B-UD-Q4_K_XL.gguf --model-dir C:\\AI\\models\\Qwen3.8-27B `
        --out-dir D:\\models\\r4dx\\kl-gguf\\validation
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gguf_dequant import (  # noqa: E402
    F16_COARSE_SCALE,
    LOSSLESS_TYPES,
    ROW_ERROR_FACTOR,
    WEIGHT_ERROR_BOUNDS,
    GGUFFile,
    GgmlC,
    Qwen35Map,
    load_ggml_c,
    lossless_matches,
    row_error_bound,
    row_outliers,
    row_rel_errors,
    tensor_class,
)

DEFAULT_MODEL_DIR = Path(r"C:\AI\models\Qwen3.8-27B")
#: A quantized tensor whose rel error exceeds this multiple of its type's median is an outlier.
PEER_OUTLIER_FACTOR = 1.4
#: Transform tags whose inverse is a permutation (reported with rel_no_inverse).
PERMUTED_TAGS = ("qkv_vrows", "vrows", "vheads", "conv_vch", "vcols")
_CHUNK_ELEMS = 1 << 22


class SafetensorsCheckpoint:
    """numpy-only, memory-mapped reader for a sharded safetensors checkpoint (no torch)."""

    _NP = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            with open(index, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {k: "model.safetensors" for k in self._header("model.safetensors")[0]}
        self._headers: dict[str, tuple[dict, int]] = {}
        self._maps: dict[str, np.memmap] = {}

    def _header(self, shard: str) -> tuple[dict, int]:
        with open(self.model_dir / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        return hdr, 8 + n

    def info(self, name: str) -> tuple[str, tuple[int, ...], str, int]:
        shard = self.weight_map[name]
        if shard not in self._headers:
            self._headers[shard] = self._header(shard)
        hdr, base = self._headers[shard]
        e = hdr[name]
        return e["dtype"], tuple(e["shape"]), shard, base + e["data_offsets"][0]

    def shape(self, name: str) -> tuple[int, ...]:
        return self.info(name)[1]

    def rows(self, name: str, r0: int = 0, r1: int | None = None) -> np.ndarray:
        """float32 rows [r0, r1) of the tensor seen as [shape[0], rest]."""
        dtype, shape, shard, off = self.info(name)
        if dtype not in self._NP:
            raise ValueError(f"{name}: unsupported safetensors dtype {dtype}")
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.model_dir / shard, dtype=np.uint8, mode="r")
        if len(shape) >= 2:
            n_rows, row = int(shape[0]), int(np.prod(shape[1:], dtype=np.int64))
        else:  # a vector is one row
            n_rows, row = 1, int(np.prod(shape, dtype=np.int64))
        r1 = n_rows if r1 is None else r1
        isz = np.dtype(self._NP[dtype]).itemsize
        raw = np.asarray(self._maps[shard][off + r0 * row * isz: off + r1 * row * isz]).view(self._NP[dtype])
        if dtype == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32, copy=False).reshape(r1 - r0, row)

    def tensor(self, name: str) -> np.ndarray:
        return self.rows(name).reshape(self.shape(name))


# ------------------------------------------------------------------------------------------------


class _Acc:
    """Running float64 sums for rel / cos over row chunks, tensor-wide and per row (a row is
    shape[0] of the HF tensor: an output row of a 2D weight; a 1D tensor is one row)."""

    def __init__(self, n_rows: int):
        self.sse = self.ssw = self.ssd = self.dot = 0.0
        self.max_abs = 0.0
        self.row_sse, self.row_ssw = np.zeros(n_rows), np.zeros(n_rows)
        self.row_ssd, self.row_dot = np.zeros(n_rows), np.zeros(n_rows)

    def add(self, deq: np.ndarray, w: np.ndarray, r0: int) -> None:
        """Rows [r0, r0 + len(deq)) as 2D arrays of equal shape."""
        d = deq.astype(np.float64)
        x = w.astype(np.float64)
        diff = d - x
        r1 = r0 + d.shape[0]
        sse, ssw, ssd, dot = (np.einsum("ij,ij->i", a, b) for a, b in ((diff, diff), (x, x), (d, d), (d, x)))
        self.row_sse[r0:r1] += sse
        self.row_ssw[r0:r1] += ssw
        self.row_ssd[r0:r1] += ssd
        self.row_dot[r0:r1] += dot
        self.sse += float(sse.sum())
        self.ssw += float(ssw.sum())
        self.ssd += float(ssd.sum())
        self.dot += float(dot.sum())
        if diff.size:
            self.max_abs = max(self.max_abs, float(np.abs(diff).max()))

    def result(self) -> dict:
        rel = (self.sse / self.ssw) ** 0.5 if self.ssw > 0 else (0.0 if self.sse == 0 else float("inf"))
        den = (self.ssw * self.ssd) ** 0.5
        cos = self.dot / den if den > 0 else (1.0 if self.sse == 0 else 0.0)
        return {"rel": rel, "cos": cos, "max_abs_err": self.max_abs}

    def row_rel(self) -> np.ndarray:
        return row_rel_errors(self.row_sse, self.row_ssw)

    def row_cos(self) -> np.ndarray:
        den = np.sqrt(self.row_ssw * self.row_ssd)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, self.row_dot / den, np.where(self.row_sse == 0, 1.0, 0.0))


def _rows2d(a: np.ndarray) -> np.ndarray:
    return a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)


def _chunked(a: np.ndarray, b: np.ndarray, acc: _Acc) -> None:
    a2 = _rows2d(a)
    b2 = b.reshape(a2.shape)
    step = max(1, _CHUNK_ELEMS // max(1, a2.shape[1]))
    for r in range(0, a2.shape[0], step):
        acc.add(a2[r:r + step], b2[r:r + step], r)


#: Row outliers listed per tensor in the report (all are counted).
_MAX_LISTED_ROWS = 20


def _row_report(row: dict, acc: _Acc, g: GGUFFile, qmap: Qwen35Map, gname: str, tag: str) -> list[dict]:
    """Per-row statistics into `row`; returns the row outliers (see gguf_dequant.ROW_ERROR_FACTOR)."""
    rr = acc.row_rel()
    norms = np.sqrt(acc.row_ssw)
    worst = int(np.argmax(rr))
    row.update(n_rows=int(rr.size), row_rel_median=float(np.median(rr)),
               row_rel_p999=float(np.quantile(rr, 0.999)), row_rel_max=float(rr[worst]),
               row_rel_max_row=worst, row_rel_max_row_norm=float(norms[worst]),
               row_norm_median=float(np.median(norms)), row_bound=row_error_bound(row["type"]))
    outl = row_outliers(g, qmap, gname, tag, rr, norms, row_cos=acc.row_cos())
    row["row_outliers_n"] = len(outl)
    row["row_outliers_coarse_scale_n"] = sum(o["coarse_scale"] for o in outl)
    row["row_outliers"] = outl[:_MAX_LISTED_ROWS]
    return outl


#: Rows of every quantized tensor re-decoded by ggml's C (besides the first, the last, the worst row
#: and every outlier row).
_C_RANDOM_ROWS = 6


def _ggml_c_check(row: dict, ggml_c: GgmlC, g: GGUFFile, gname: str, gguf_rows: list[int]) -> None:
    """Re-decode `gguf_rows` (plus the first, the last and a few seeded random rows) with ggml's own
    C dequantizer and require bit-identical float32 output."""
    t = g.tensor(gname)
    if t.type_name not in ggml_c.types:
        row["ggml_c"] = f"no C dequantizer for {t.type_name}"
        return
    rng = np.random.default_rng(zlib.crc32(gname.encode()))
    ids = sorted({0, t.n_rows - 1, *gguf_rows,
                  *rng.choice(t.n_rows, size=min(_C_RANDOM_ROWS, t.n_rows), replace=False).tolist()})
    mine = g.dequantize_row_ids(gname, ids)
    ref = ggml_c.dequantize(t.type_name, g.raw_rows(gname)[ids]).reshape(mine.shape)
    ndiff = int((mine.view(np.uint32) != ref.view(np.uint32)).sum())
    row.update(ggml_c_rows=len(ids), ggml_c_identical=ndiff == 0, ggml_c_differing_values=ndiff)


_G: GGUFFile | None = None
_M: Qwen35Map | None = None
_CK: SafetensorsCheckpoint | None = None
_C: GgmlC | None = None
_THREADS = 1


def _open_ggml_c(spec: str | None) -> GgmlC | None:
    """--ggml-c: 'auto' (the default search, None if absent), 'none', or a library path."""
    if spec in (None, "none"):
        return None
    if spec == "auto":
        return load_ggml_c()
    return GgmlC(spec)


def _init(gguf_path: str, model_dir: str, threads: int, ggml_c: str | None = None) -> None:
    global _G, _M, _CK, _C, _THREADS
    _G = GGUFFile(gguf_path)
    _M = Qwen35Map.from_gguf(_G)
    _CK = SafetensorsCheckpoint(Path(model_dir))
    _C = _open_ggml_c(ggml_c)
    _THREADS = threads


def check_tensor(gname: str, g: GGUFFile | None = None, qmap: Qwen35Map | None = None,
                 ck: SafetensorsCheckpoint | None = None, threads: int | None = None,
                 ggml_c: GgmlC | None = None) -> dict:
    """One GGUF tensor against its checkpoint tensor. Returns a JSON-able row; never raises (an
    exception becomes a failed row). With `ggml_c` (or the worker's), sampled rows of a quantized
    tensor, its worst row and its outlier rows must also decode bit-identically in ggml's C."""
    g, qmap, ck, ggml_c = g or _G, qmap or _M, ck or _CK, ggml_c or _C
    threads = _THREADS if threads is None else threads
    t0 = time.perf_counter()
    t = g.tensor(gname)
    row: dict = {"gguf": gname, "type": t.type_name, "gguf_shape": list(t.shape)}
    try:
        hit = qmap.hf_of.get(gname)
        if hit is None:
            row.update(kind="unmapped", ok=False, why="no HF counterpart")
            return row
        hf, tag = hit
        row.update(hf=hf, tag=tag, cls=tensor_class(hf))
        if hf not in ck.weight_map:
            row.update(kind="missing_in_checkpoint", ok=False, why=f"{hf} not in the checkpoint")
            return row
        hf_shape = ck.shape(hf)
        row["hf_shape"] = list(hf_shape)
        if int(np.prod(hf_shape, dtype=np.int64)) != t.n_elements:
            row.update(kind="shape", ok=False, why=f"{t.n_elements} elements vs HF {hf_shape}")
            return row

        n_hf_rows = int(hf_shape[0]) if len(hf_shape) > 1 else 1
        if t.type_name in LOSSLESS_TYPES:
            gv = g.dequantize(gname)
            w = ck.tensor(hf)
            ok, ulp = lossless_matches(qmap, tag, gv, w)
            acc = _Acc(n_hf_rows)
            _chunked(qmap.to_hf(tag, gv, hf_shape), w, acc)
            row.update(kind="lossless", ok=ok, max_ulp=ulp, **acc.result())
            if not ok:
                row["why"] = f"stored value differs from the converted checkpoint value by {ulp} ulp"
            return row

        acc = _Acc(n_hf_rows)
        if tag == "id" and len(hf_shape) == 2 and tuple(t.shape) == tuple(hf_shape):
            step = max(1, _CHUNK_ELEMS // t.row_len)
            for r0 in range(0, t.n_rows, step):
                r1 = min(t.n_rows, r0 + step)
                acc.add(g.dequantize_rows(gname, r0, r1, threads=threads), ck.rows(hf, r0, r1), r0)
        else:
            gv = g.dequantize(gname, threads=threads)
            w = ck.tensor(hf)
            _chunked(qmap.to_hf(tag, gv, hf_shape), w, acc)
            if tag in PERMUTED_TAGS:
                acc0 = _Acc(n_hf_rows)
                _chunked(gv.reshape(hf_shape), w, acc0)
                row["rel_no_inverse"] = acc0.result()["rel"]
        row.update(kind="quantized", **acc.result())
        outl = _row_report(row, acc, g, qmap, gname, tag)
        if ggml_c is not None:
            _ggml_c_check(row, ggml_c, g, gname,
                          [qmap.gguf_row(tag, row["row_rel_max_row"])] + [o["gguf_row"] for o in outl])
        max_rel, min_cos = WEIGHT_ERROR_BOUNDS.get(t.type_name, (None, None))
        if max_rel is None:
            row.update(ok=False, why=f"no error bounds for type {t.type_name}")
        elif row["rel"] > max_rel or row["cos"] < min_cos:
            row.update(ok=False, why=f"rel {row['rel']:.4f} (max {max_rel}) / cos {row['cos']:.6f} "
                                     f"(min {min_cos}) outside the {t.type_name} bounds")
        elif row.get("ggml_c_identical") is False:
            row.update(ok=False, why=f"{row['ggml_c_differing_values']} value(s) of {row['ggml_c_rows']} rows "
                                     "differ from ggml's C dequantizer")
        else:
            row["ok"] = True
        return row
    except Exception as exc:  # a crash is a failed row, reported with the rest
        row.update(kind=row.get("kind", "error"), ok=False, why=f"{type(exc).__name__}: {exc}")
        return row
    finally:
        row["seconds"] = time.perf_counter() - t0


# ------------------------------------------------------------------------------------------------


def _summ(rows: list[dict]) -> dict:
    rels = [r["rel"] for r in rows]
    coss = [r["cos"] for r in rows]
    worst_row = max(rows, key=lambda r: r.get("row_rel_max", 0.0))
    return {"n": len(rows), "rel_min": min(rels), "rel_median": statistics.median(rels),
            "rel_max": max(rels), "cos_min": min(coss),
            "row_rel_max": worst_row.get("row_rel_max"), "row_rel_max_tensor": worst_row["gguf"],
            "row_rel_max_row": worst_row.get("row_rel_max_row"),
            "row_outliers": sum(r.get("row_outliers_n", 0) for r in rows)}


def summarize(rows: list[dict]) -> dict:
    quant = [r for r in rows if r.get("kind") == "quantized"]
    by_type: dict[str, list[dict]] = {}
    by_class: dict[str, list[dict]] = {}
    by_tc: dict[str, list[dict]] = {}
    for r in quant:
        by_type.setdefault(r["type"], []).append(r)
        by_class.setdefault(r["cls"], []).append(r)
        by_tc.setdefault(f"{r['cls']} {r['type']}", []).append(r)
    # Peer rule: within a type, nobody far above the type's median.
    for tname, rs in by_type.items():
        if len(rs) < 3:
            continue
        med = statistics.median(r["rel"] for r in rs)
        for r in rs:
            r["type_median_rel"] = med
            if r["ok"] and r["rel"] > PEER_OUTLIER_FACTOR * med:
                r.update(ok=False, why=f"rel {r['rel']:.4f} > {PEER_OUTLIER_FACTOR} x the {tname} "
                                       f"median {med:.4f}")
    lossless = [r for r in rows if r.get("kind") == "lossless"]
    ll_by_tag: dict[str, dict] = {}
    for r in lossless:
        e = ll_by_tag.setdefault(r["tag"], {"n": 0, "exact": 0, "within_1ulp": 0, "failed": 0})
        e["n"] += 1
        e["exact"] += r["max_ulp"] == 0
        e["within_1ulp"] += r["max_ulp"] == 1
        e["failed"] += not r["ok"]
    # Worst rows first: every tensor with a row above its per-row bound, and the embedding and
    # lm_head always (their rows are gathered one token at a time, so one bad row is one bad token).
    listed = [r for r in quant if r.get("row_outliers_n") or r.get("cls") in ("embed_tokens", "lm_head")]
    return {
        "by_type": {k: _summ(v) for k, v in sorted(by_type.items())},
        "by_class": {k: _summ(v) for k, v in sorted(by_class.items())},
        "by_class_type": {k: _summ(v) for k, v in sorted(by_tc.items())},
        "rows": {
            "row_error_factor": ROW_ERROR_FACTOR,
            "tensors_with_row_outliers": sum(1 for r in quant if r.get("row_outliers_n")),
            "row_outliers": sum(r.get("row_outliers_n", 0) for r in quant),
            "row_outliers_coarse_scale": sum(r.get("row_outliers_coarse_scale_n", 0) for r in quant),
            "listed": [{k: r.get(k) for k in ("gguf", "type", "cls", "n_rows", "row_rel_median",
                                               "row_rel_p999", "row_rel_max", "row_rel_max_row",
                                               "row_rel_max_row_norm", "row_norm_median", "row_bound",
                                               "row_outliers_n", "row_outliers", "ggml_c_identical")}
                       for r in sorted(listed, key=lambda r: -r.get("row_rel_max", 0.0))],
        },
        "ggml_c": {
            "tensors_checked": sum(1 for r in quant if "ggml_c_identical" in r),
            "rows_checked": sum(r.get("ggml_c_rows", 0) for r in quant),
            "tensors_differing": sum(1 for r in quant if r.get("ggml_c_identical") is False),
            "types": sorted({r["type"] for r in quant if "ggml_c_identical" in r}),
        },
        "lossless_by_tag": ll_by_tag,
        "permutation_proof": {
            "tensors": sum(1 for r in quant if "rel_no_inverse" in r),
            "min_rel_no_inverse": min((r["rel_no_inverse"] for r in quant if "rel_no_inverse" in r),
                                      default=None),
        },
    }


def markdown(report: dict) -> str:
    s = report["summary"]
    L = [f"# GGUF weight validation: `{Path(report['gguf']).name}`", "",
         f"- GGUF: `{report['gguf']}` ({report['gguf_size_bytes']} bytes, header sha256 "
         f"`{report['gguf_header_sha256']}`)",
         f"- checkpoint: `{report['model_dir']}`",
         f"- {report['n_tensors']} GGUF tensors, {report['n_quantized']} quantized, "
         f"{report['n_lossless']} lossless; {len(report['failures'])} failure(s); "
         f"{report['seconds']:.0f} s",
         f"- HF tensors absent from the GGUF (kept from the checkpoint by --weights-gguf): "
         f"{report['hf_absent_count']} ({report['hf_absent_summary']})", "",
         "rel = ||deq - W||_F / ||W||_F against the bf16 checkpoint after undoing the layout "
         "transform; cos = cosine similarity.", "",
         "## Quantized tensors by type", "",
         "| type | n | rel min | rel median | rel max | cos min | bound (rel max / cos min) | "
         "worst row rel (tensor, row) |",
         "|---|--:|--:|--:|--:|--:|---|---|"]
    for k, v in s["by_type"].items():
        b = WEIGHT_ERROR_BOUNDS.get(k)
        L.append(f"| {k} | {v['n']} | {v['rel_min']:.4f} | {v['rel_median']:.4f} | {v['rel_max']:.4f} | "
                 f"{v['cos_min']:.6f} | {b[0]} / {b[1]} | {v['row_rel_max']:.4f} "
                 f"(`{v['row_rel_max_tensor']}`, {v['row_rel_max_row']}) |"
                 if b else f"| {k} | {v['n']} | - |")
    gc = s["ggml_c"]
    L += ["", "## Against ggml's own C dequantizers", "",
          (f"{gc['tensors_checked']} quantized tensors ({', '.join(gc['types'])}): {gc['rows_checked']} rows "
           f"(the first, the last, {_C_RANDOM_ROWS} random, the worst and every outlier row of each) "
           f"re-decoded by `{report['ggml_c']}`; {gc['tensors_differing']} tensor(s) differ in any bit."
           if gc["tensors_checked"] else
           "Not checked: no ggml-base library found (pass --ggml-c PATH; see gguf_dequant.GgmlC). "
           "The bit-exactness evidence is then tests/reference/test_gguf_dequant.py alone.")]
    rs = s["rows"]
    L += ["", "## Rows", "",
          f"A tensor-level rel pools every row, so it cannot show one bad row. Each quantized tensor's "
          f"rows (shape[0] of the HF tensor) are also measured one by one, and a row above "
          f"{rs['row_error_factor']} x its type's rel bound is listed. A listed row is not a failure by "
          f"itself: where ggml's C was run, it decodes the row identically, so its error is the file's. "
          f"`coarse scale` marks a row whose every block has an f16 scale |d| < {F16_COARSE_SCALE:.3g} "
          f"(2^-20: at most 4 significant bits, or 0): its values (~1e-5 or less) are below what the "
          f"type's scale resolves. `cos` is the row's cosine with the checkpoint row (a wrong row would "
          f"be near 0).", "",
          f"{rs['tensors_with_row_outliers']} tensor(s) have {rs['row_outliers']} row(s) above the "
          f"per-row bound, {rs['row_outliers_coarse_scale']} of them with a coarse scale. Listed: the "
          "embedding, the lm_head and every tensor with such a row.", "",
          "| tensor | type | rows | row rel median | p99.9 | max (row, its norm; median norm) | "
          "per-row bound | rows above it | C identical |",
          "|---|---|--:|--:|--:|---|--:|---|---|"]
    for r in rs["listed"]:
        outl = "; ".join(
            f"row {o['row']}" + (f" (GGUF {o['gguf_row']})" if o["gguf_row"] != o["row"] else "")
            + f": rel {o['rel']:.3f}, cos {o.get('cos', float('nan')):.3f}, norm {o['norm']:.3g}"
            + (f", d = 0 / coarse in {o['zero_scale_blocks']} / {o['coarse_scale_blocks']} of "
               f"{o['blocks']} blocks, max abs d {o['max_abs_scale']:.3g}" if o["blocks"] else "")
            + (" (coarse scale)" if o["coarse_scale"] else "")
            for o in r["row_outliers"]) or "none"
        if r["row_outliers_n"] > len(r["row_outliers"]):
            outl += f"; ... {r['row_outliers_n'] - len(r['row_outliers'])} more"
        cid = {True: "yes", False: "**NO**", None: "not run"}[r.get("ggml_c_identical")]
        L.append(f"| `{r['gguf']}` | {r['type']} | {r['n_rows']} | {r['row_rel_median']:.4f} | "
                 f"{r['row_rel_p999']:.4f} | {r['row_rel_max']:.4f} ({r['row_rel_max_row']}, "
                 f"{r['row_rel_max_row_norm']:.3g}; {r['row_norm_median']:.3g}) | {r['row_bound']:.3f} | "
                 f"{outl} | {cid} |")
    for title, key in (("by class", "by_class"), ("by class and type", "by_class_type")):
        L += ["", f"## Quantized tensors {title}", "",
              "| class | n | rel min | rel median | rel max | cos min |", "|---|--:|--:|--:|--:|--:|"]
        for k, v in s[key].items():
            L.append(f"| {k} | {v['n']} | {v['rel_min']:.4f} | {v['rel_median']:.4f} | "
                     f"{v['rel_max']:.4f} | {v['cos_min']:.6f} |")
    L += ["", "## Lossless (F32/BF16) tensors: stored value vs the converted checkpoint value", "",
          "| transform | n | bit-exact | 1 ulp | failed |", "|---|--:|--:|--:|--:|"]
    for k, v in s["lossless_by_tag"].items():
        L.append(f"| {k} | {v['n']} | {v['exact']} | {v['within_1ulp']} | {v['failed']} |")
    pp = s["permutation_proof"]
    if pp["tensors"]:
        L += ["", f"V-head reorder: {pp['tensors']} quantized tensors carry it. Without the inverse "
                  f"reorder their smallest rel error would be {pp['min_rel_no_inverse']:.3f}."]
    L += ["", "## Failures", ""]
    L += [f"- `{f['gguf']}` ({f.get('type')}, {f.get('hf')}): {f.get('why')}" for f in report["failures"]] or ["none"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                    help="worker processes (each holds ~1.5 GiB at peak)")
    ap.add_argument("--threads", type=int, default=1, help="dequant threads per worker")
    ap.add_argument("--only", default=None, help="regex: check only GGUF tensors whose name matches")
    ap.add_argument("--ggml-c", default="auto",
                    help="ggml-base library whose C dequantizers re-decode sampled rows of every quantized "
                         "tensor bit for bit: 'auto' (search, skip if absent), 'none', or a path")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    g = GGUFFile(args.gguf)
    qmap = Qwen35Map.from_gguf(g)
    ck = SafetensorsCheckpoint(args.model_dir)
    g.check_decodable()
    ggml_c = _open_ggml_c(args.ggml_c)
    ggml_c_spec = str(ggml_c.path) if ggml_c else "none"  # the workers load the same library
    names = [n for n in g.tensors if args.only is None or re.search(args.only, n)]
    names.sort(key=lambda n: -g.tensor(n).n_elements)  # biggest first: better load balance
    print(f"[gguf_validate] {len(names)} tensors of {args.gguf.name} vs {args.model_dir} "
          f"({args.workers} workers; ggml C: {ggml_c_spec})", flush=True)

    rows: list[dict] = []
    if args.workers <= 1:
        _init(str(args.gguf), str(args.model_dir), args.threads, ggml_c_spec)
        it = map(check_tensor, names)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(str(args.gguf), str(args.model_dir), args.threads, ggml_c_spec))
        it = ex.map(check_tensor, names, chunksize=1)
    for i, r in enumerate(it):
        rows.append(r)
        if not r.get("ok", False):
            print(f"  FAIL {r['gguf']}: {r.get('why')}", flush=True)
        if i % 50 == 0 or i == len(names) - 1:
            print(f"  [{i + 1}/{len(names)}] {time.perf_counter() - t_start:.0f}s {r['gguf']} "
                  f"{r['type']} rel={r.get('rel', float('nan')):.4f}", flush=True)
    if args.workers > 1:
        ex.shutdown()

    order = {n: i for i, n in enumerate(g.tensors)}
    rows.sort(key=lambda r: order[r["gguf"]])
    summary = summarize(rows)
    mapped_hf = set(qmap.gguf_of)
    absent = sorted(n for n in ck.weight_map if n not in mapped_hf)
    absent_groups: dict[str, int] = {}
    for n in absent:
        key = ".".join(n.split(".")[:2])
        absent_groups[key] = absent_groups.get(key, 0) + 1
    failures = [r for r in rows if not r.get("ok", False)]
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gguf": str(args.gguf),
        "gguf_size_bytes": g.file_size,
        "gguf_header_sha256": g.header_sha256,
        "model_dir": str(args.model_dir),
        "only": args.only,
        "n_tensors": len(rows),
        "n_quantized": sum(r.get("kind") == "quantized" for r in rows),
        "n_lossless": sum(r.get("kind") == "lossless" for r in rows),
        "type_counts": g.type_counts(),
        "bounds": WEIGHT_ERROR_BOUNDS,
        "bounds_note": "fitted to the Unsloth Qwen3.8-27B-UD-Q4_K_XL file: a tripwire, not independent "
                       "evidence (see gguf_dequant.WEIGHT_ERROR_BOUNDS)",
        "peer_outlier_factor": PEER_OUTLIER_FACTOR,
        "row_error_factor": ROW_ERROR_FACTOR,
        "ggml_c": ggml_c_spec,
        "hf_absent_count": len(absent),
        "hf_absent_summary": ", ".join(f"{k}.*: {v}" for k, v in sorted(absent_groups.items())) or "none",
        "hf_absent": absent,
        "summary": summary,
        "failures": failures,
        "seconds": time.perf_counter() - t_start,
        "tensors": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "gguf_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    md = markdown(report)
    with open(args.out_dir / "gguf_validation.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(md)
    print(f"[gguf_validate] wrote {args.out_dir / 'gguf_validation.json'} and gguf_validation.md")
    if failures:
        print(f"[gguf_validate] FAILED: {len(failures)} tensor(s) out of bounds or mismatched")
        return 1
    print("[gguf_validate] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
