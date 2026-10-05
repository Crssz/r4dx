"""tools/reference/fp8_dequant.py

Block-FP8 (compressed-tensors / DeepSeek style) safetensors checkpoint -> bf16 checkpoint with the
bf16 base convention, plus a header-level checkpoint differ. First user: d0xin/Swift-1.5-Qwen3.8-27B-
Uncensored-FP8 (a fine-tune of Qwen3.8-27B; 407 float8_e4m3fn Linear weights with 128x128 block scales,
the other 1199 tensors bf16), so that it can go through the existing bf16 -> trellis pipeline.

    python tools/reference/fp8_dequant.py dequant --src <fp8 dir> --out <bf16 dir> [--threads N] [--sha256]
    python tools/reference/fp8_dequant.py diff --a <dir> --b <dir> --out <json> [--sample-rel-diff N]

dequant
  For every F8_E4M3 / F8_E5M2 tensor `N` (normally `...weight`) the scale is the sibling `N_scale`
  (compressed-tensors `weight_scale`) or `N_scale_inv` (DeepSeek `weight_scale_inv`); both MULTIPLY:

      W_bf16[i, j] = bf16( float32(fp8[i, j]) * float32(scale[i // 128, j // 128]) )

  with the ragged edge cropped (scale shape must be exactly ceil(rows/128) x ceil(cols/128)). A
  missing scale, both scale names, a non-2-D fp8 tensor, a wrong scale shape, or a scale tensor with
  no fp8 weight (orphan) is a refusal, never a guess. Scale tensors and `*.input_scale` activation
  scales are dropped. Every other tensor is copied byte for byte. Output shards keep the source shard
  names; the index is rewritten (scale names removed, total_size recomputed); config.json loses
  `quantization_config`; every other non-weight file is copied. Shards are streamed (a few block rows
  at a time, never a whole shard in memory) and written to `<name>.tmp` then renamed, so a killed run
  leaves only complete shards; a rerun re-validates the existing output shards by header and size and
  skips them. `fp8_dequant_manifest.json` lists per tensor the source dtype, scale name and (with
  --sha256) the sha256 of the written bf16 bytes.

diff
  Header-only comparison of two checkpoints: tensor names (from the index and from every shard header
  present, so a partial download is compared as far as it goes), shapes, dtypes, sha256 of every
  non-weight file, config.json raw and with `quantization_config` removed. Mismatches that are just
  the fp8 + scale difference are classified as `fp8_expected`. --sample-rel-diff N additionally reads N
  common tensors (fp8 ones dequantized with their scale) and reports ||a-b||_F / ||a||_F.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import raw_safetensors_header, raw_safetensors_read, sha256_file  # noqa: E402

BLOCK = 128
FP8_TAGS = ("F8_E4M3", "F8_E5M2")
SCALE_SUFFIXES = ("_scale", "_scale_inv")        # appended to the fp8 tensor's own name
EXTRA_DROP_SUFFIXES = (".input_scale", ".input_global_scale", ".weight_global_scale")
ELT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "BOOL": 1, "I16": 2, "I32": 4,
            "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
MANIFEST = "fp8_dequant_manifest.json"
SIDECAR_DIR = ".fp8_dequant"
CHUNK_ROWS = 1024                                 # multiple of BLOCK; fp8 chunk <= ~35 MB at 34k cols
COPY_CHUNK = 64 << 20


class Refusal(Exception):
    pass


# ----------------------------------------------------------------------------- checkpoint listing
def weight_shards(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix == ".safetensors")


def read_index(d: Path) -> dict | None:
    p = d / "model.safetensors.index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def load_headers(d: Path, shards: list[Path] | None = None):
    """({tensor: (shard_path, header_entry, data_offset)}, {shard_path: header_total_bytes})"""
    tensors, sizes = {}, {}
    for p in shards if shards is not None else weight_shards(d):
        hdr, off = raw_safetensors_header(p)
        for n, e in hdr.items():
            if n in tensors:
                raise Refusal(f"tensor {n} appears in two shards: {tensors[n][0].name}, {p.name}")
            tensors[n] = (p, e, off)
        sizes[p] = off
    return tensors, sizes


def classify(tensors: dict) -> tuple[dict, set]:
    """-> ({fp8 tensor: scale tensor name}, {all dropped names}). Raises Refusal on any inconsistency."""
    scale_of: dict[str, str] = {}
    dropped: set[str] = set()
    for n, (_, e, _) in tensors.items():
        if e["dtype"] not in FP8_TAGS:
            continue
        cands = [n + s for s in SCALE_SUFFIXES if n + s in tensors]
        if not cands:
            raise Refusal(f"fp8 tensor {n} has no scale ({n}{SCALE_SUFFIXES[0]} / {n}{SCALE_SUFFIXES[1]})")
        if len(cands) > 1:
            raise Refusal(f"fp8 tensor {n} has both scale names {cands}")
        if len(e["shape"]) != 2:
            raise Refusal(f"fp8 tensor {n} is {len(e['shape'])}-D {e['shape']}; only 2-D block FP8 is handled")
        sn = cands[0]
        want = [math.ceil(e["shape"][0] / BLOCK), math.ceil(e["shape"][1] / BLOCK)]
        got = tensors[sn][1]["shape"]
        if got != want:
            raise Refusal(f"scale {sn} shape {got} != ceil(rows/{BLOCK}) x ceil(cols/{BLOCK}) = {want} "
                          f"for {n} {e['shape']}")
        if tensors[sn][1]["dtype"] not in ("BF16", "F16", "F32"):
            raise Refusal(f"scale {sn} has dtype {tensors[sn][1]['dtype']}")
        scale_of[n] = sn
        dropped.add(sn)
    for n in tensors:
        if n.endswith(EXTRA_DROP_SUFFIXES):
            dropped.add(n)
    for n in tensors:                                   # orphan scales
        for s in SCALE_SUFFIXES:
            if n.endswith(s):
                base = n[: -len(s)]
                if n not in dropped and base in tensors and tensors[base][1]["dtype"] not in FP8_TAGS:
                    continue                            # a bf16 tensor that merely has a _scale sibling: keep
                if n not in dropped:
                    raise Refusal(f"scale tensor {n} has no fp8 tensor {base}")
    return scale_of, dropped


# ----------------------------------------------------------------------------- dequant
def dequant_rows(w_u8_or_fp8, scale_rows, rows, cols):
    import torch
    s = scale_rows.float().repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]
    return (w_u8_or_fp8.float() * s).to(torch.bfloat16)


def out_entries(src_hdr: dict, scale_of: dict, dropped: set, shard: Path, tensors: dict):
    """Ordered (name, entry_in, out_dtype, out_shape, out_nbytes) of the tensors of one source shard."""
    items = sorted(((n, t[1]) for n, t in tensors.items() if t[0] == shard), key=lambda x: x[1]["data_offsets"][0])
    res = []
    for n, e in items:
        if n in dropped:
            continue
        if n in scale_of:
            nb = 2 * e["shape"][0] * e["shape"][1]
            res.append((n, e, "BF16", e["shape"], nb))
        else:
            res.append((n, e, e["dtype"], e["shape"], e["data_offsets"][1] - e["data_offsets"][0]))
    return res


def build_header(entries) -> tuple[bytes, int]:
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    for n, _, dt, shp, nb in entries:
        hdr[n] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + nb]}
        off += nb
    raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, off


def shard_valid(path: Path, entries) -> bool:
    if not path.is_file():
        return False
    try:
        hdr_bytes, total = build_header(entries)
        if path.stat().st_size != len(hdr_bytes) + total:
            return False
        with open(path, "rb") as f:
            return f.read(len(hdr_bytes)) == hdr_bytes
    except Exception:
        return False


def copy_bytes(src: Path, start: int, nbytes: int, fout, h=None):
    with open(src, "rb") as f:
        f.seek(start)
        left = nbytes
        while left:
            b = f.read(min(COPY_CHUNK, left))
            if not b:
                raise Refusal(f"{src}: truncated (short read at offset {start})")
            fout.write(b)
            if h is not None:
                h.update(b)
            left -= len(b)


def convert_shard(shard: Path, out_dir: Path, tensors, scale_of, dropped, want_sha: bool, force: bool):
    entries = out_entries(None, scale_of, dropped, shard, tensors)
    dst = out_dir / shard.name
    side = out_dir / SIDECAR_DIR / (shard.name + ".json")
    if not force and shard_valid(dst, entries) and side.is_file():
        rec = json.loads(side.read_text(encoding="utf-8"))
        if rec.get("src_size") == shard.stat().st_size and (not want_sha or rec.get("sha256_complete")):
            return shard.name, rec["tensors"], True
    hdr_bytes, total = build_header(entries)
    tmp = dst.with_name(dst.name + ".tmp")
    recs = {}
    with open(tmp, "wb") as fo:
        fo.write(hdr_bytes)
        for n, e, odt, shp, nb in entries:
            h = hashlib.sha256() if want_sha else None
            if n in scale_of:
                sp, se, so = tensors[scale_of[n]]
                sc = raw_safetensors_read(sp, scale_of[n], ({scale_of[n]: se}, so))
                rows, cols = shp
                for a in range(0, rows, CHUNK_ROWS):
                    b = min(rows, a + CHUNK_ROWS)
                    w = raw_safetensors_read(shard, n, ({n: e}, tensors[n][2]), rows=(a, b))
                    o = dequant_rows(w, sc[a // BLOCK: math.ceil(b / BLOCK)], b - a, cols).contiguous()
                    buf = o.view(torch_int16()).numpy().tobytes()
                    fo.write(buf)
                    if h:
                        h.update(buf)
                recs[n] = {"src_dtype": e["dtype"], "scale": scale_of[n], "shape": list(shp),
                           "out_dtype": "BF16", "sha256": h.hexdigest() if h else None}
            else:
                copy_bytes(shard, tensors[n][2] + e["data_offsets"][0], nb, fo, h)
                recs[n] = {"src_dtype": e["dtype"], "scale": None, "shape": list(shp),
                           "out_dtype": odt, "sha256": h.hexdigest() if h else None}
        fo.flush()
        os.fsync(fo.fileno())
    if tmp.stat().st_size != len(hdr_bytes) + total:
        raise Refusal(f"{tmp}: size {tmp.stat().st_size} != expected {len(hdr_bytes) + total}")
    os.replace(tmp, dst)
    side.parent.mkdir(parents=True, exist_ok=True)
    side_tmp = side.with_name(side.name + ".tmp")
    side_tmp.write_text(json.dumps({"src_size": shard.stat().st_size, "sha256_complete": want_sha,
                                    "tensors": recs}), encoding="utf-8")
    os.replace(side_tmp, side)
    return shard.name, recs, False


def torch_int16():
    import torch
    return torch.int16


def strip_quant(cfg):
    if isinstance(cfg, dict):
        return {k: strip_quant(v) for k, v in cfg.items() if k != "quantization_config"}
    if isinstance(cfg, list):
        return [strip_quant(v) for v in cfg]
    return cfg


def atomic_write(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def copy_nonweight(src: Path, out: Path) -> list[str]:
    copied = []
    skip_names = {"model.safetensors.index.json", "config.json", ".gitattributes", ".gitignore"}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in (".cache", ".git", SIDECAR_DIR)]
        for fn in files:
            p = Path(root) / fn
            rel = p.relative_to(src)
            if p.suffix == ".safetensors" or fn.endswith((".tmp", ".lock")) or (len(rel.parts) == 1 and fn in skip_names):
                continue
            q = out / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            if q.is_file() and q.stat().st_size == p.stat().st_size and sha256_file(q) == sha256_file(p):
                copied.append(str(rel))
                continue
            tmp = q.with_name(q.name + ".tmp")
            shutil.copyfile(p, tmp)
            os.replace(tmp, q)
            copied.append(str(rel))
    return sorted(copied)


def cmd_dequant(a) -> int:
    src, out = Path(a.src), Path(a.out)
    if src.resolve() == out.resolve():
        raise Refusal("--src and --out must differ")
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.tmp")) + list((out / SIDECAR_DIR).glob("*.tmp")):   # killed earlier run
        stale.unlink()
    idx = read_index(src)
    shards = weight_shards(src)
    if not shards:
        raise Refusal(f"{src}: no .safetensors shards")
    if idx:
        listed = set(idx["weight_map"].values())
        missing = sorted(listed - {p.name for p in shards})
        if missing:
            raise Refusal(f"index lists shards that are not present (download incomplete?): {missing}")
    tensors, _ = load_headers(src, shards)
    if idx:
        absent = sorted(set(idx["weight_map"]) - set(tensors))
        extra = sorted(set(tensors) - set(idx["weight_map"]))
        if absent or extra:
            raise Refusal(f"index/header mismatch: {len(absent)} in index only {absent[:3]}, "
                          f"{len(extra)} in headers only {extra[:3]}")
    scale_of, dropped = classify(tensors)
    print(f"{len(tensors)} tensors, {len(scale_of)} fp8 (scales dropped: {len(dropped)}), {len(shards)} shards",
          flush=True)
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max(1, a.threads)) as ex:
        futs = [ex.submit(convert_shard, p, out, tensors, scale_of, dropped, a.sha256, a.force) for p in shards]
        for f in futs:
            name, recs, skipped = f.result()
            results[name] = recs
            print(f"  {name}: {'skipped (valid)' if skipped else 'written'}  [{time.time() - t0:.0f}s]", flush=True)
    wm, total = {}, 0
    for p in shards:
        for n, r in results[p.name].items():
            wm[n] = p.name
            total += math.prod(r["shape"]) * ELT_SIZE[r["out_dtype"]]
    meta = dict(idx.get("metadata", {})) if idx else {}
    meta["total_size"] = total
    atomic_write(out / "model.safetensors.index.json",
                 json.dumps({"metadata": meta, "weight_map": dict(sorted(wm.items()))}, indent=2).encode() + b"\n")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    atomic_write(out / "config.json", (json.dumps(strip_quant(cfg), indent=2, ensure_ascii=False) + "\n").encode())
    copied = copy_nonweight(src, out)
    man = {"src": str(src), "block": BLOCK, "n_fp8": len(scale_of), "n_out_tensors": len(wm),
           "dropped": sorted(dropped), "copied_files": copied, "sha256": bool(a.sha256),
           "tensors": {n: r for p in shards for n, r in results[p.name].items()}}
    atomic_write(out / MANIFEST, json.dumps(man, indent=1).encode())
    print(f"OK {len(wm)} tensors out, {len(scale_of)} dequantized, {len(copied)} files copied", flush=True)
    return 0


# ----------------------------------------------------------------------------- diff
def file_hashes(d: Path) -> dict:
    res = {}
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x not in (".cache", ".git", SIDECAR_DIR)]
        for fn in files:
            p = Path(root) / fn
            if p.suffix in (".safetensors", ".tmp", ".lock") or fn == "model.safetensors.index.json":
                continue
            res[str(p.relative_to(d)).replace("\\", "/")] = sha256_file(p)
    return res


def json_diff(x, y, path="") -> list[str]:
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y)):
            if k not in x:
                out.append(f"{path}/{k}: only in b")
            elif k not in y:
                out.append(f"{path}/{k}: only in a")
            else:
                out += json_diff(x[k], y[k], f"{path}/{k}")
        return out
    return [] if x == y else [f"{path}: a={json.dumps(x)[:120]} b={json.dumps(y)[:120]}"]


def describe(d: Path):
    idx = read_index(d)
    tensors, _ = load_headers(d)
    names = set(tensors) | (set(idx["weight_map"]) if idx else set())
    info = {n: (t[1]["dtype"], t[1]["shape"]) for n, t in tensors.items()}
    return names, info, tensors, {p.name for p in weight_shards(d)}, idx


def scale_aware_read(tensors, name, rows):
    import torch
    p, e, off = tensors[name]
    w = raw_safetensors_read(p, name, ({name: e}, off), rows=rows)
    if e["dtype"] in FP8_TAGS:
        for s in SCALE_SUFFIXES:
            if name + s in tensors:
                sp, se, so = tensors[name + s]
                sc = raw_safetensors_read(sp, name + s, ({name + s: se}, so))
                a, b = rows
                return dequant_rows(w, sc[a // BLOCK: math.ceil(b / BLOCK)], b - a, e["shape"][1]).float()
        raise Refusal(f"{name}: fp8 without scale")
    return w.float()


def rel_diff(ta, tb, name, shape):
    num = den = 0.0
    rows = shape[0] if shape else 1
    step = CHUNK_ROWS
    for a in range(0, rows, step):
        b = min(rows, a + step)
        x = scale_aware_read(ta, name, (a, b)).double()
        y = scale_aware_read(tb, name, (a, b)).double()
        num += float(((x - y) ** 2).sum())
        den += float((x ** 2).sum())
    return math.sqrt(num) / math.sqrt(den) if den > 0 else (0.0 if num == 0 else float("inf"))


def cmd_diff(a) -> int:
    da, db = Path(a.a), Path(a.b)
    na, ia, ta, sa, xa = describe(da)
    nb, ib, tb, sb, xb = describe(db)

    def is_scale_of(n, names, info_all):
        for s in SCALE_SUFFIXES:
            if n.endswith(s) and n[: -len(s)] in names:
                return True
        return n.endswith(EXTRA_DROP_SUFFIXES)

    missing = sorted(n for n in na - nb)                    # in a, not in b
    extra = sorted(n for n in nb - na)                      # in b, not in a
    extra_scale = [n for n in extra if is_scale_of(n, nb, ib)]
    missing_scale = [n for n in missing if is_scale_of(n, na, ia)]
    shape_mm, dtype_mm, fp8_exp = [], [], []
    for n in sorted(set(ia) & set(ib)):
        (da_, sha_), (db_, shb_) = ia[n], ib[n]
        if sha_ != shb_:
            shape_mm.append({"name": n, "a": sha_, "b": shb_})
        if da_ != db_:
            rec = {"name": n, "a": da_, "b": shb_ and db_}
            (fp8_exp if (da_ in FP8_TAGS) != (db_ in FP8_TAGS) else dtype_mm).append(rec)
    unresolved = sorted((na & nb) - (set(ia) & set(ib)))   # named on both sides, header not available yet
    cfg_a = json.loads((da / "config.json").read_text(encoding="utf-8")) if (da / "config.json").is_file() else None
    cfg_b = json.loads((db / "config.json").read_text(encoding="utf-8")) if (db / "config.json").is_file() else None
    ha, hb = file_hashes(da), file_hashes(db)
    files = {"only_in_a": sorted(set(ha) - set(hb)), "only_in_b": sorted(set(hb) - set(ha)),
             "differ": sorted(k for k in set(ha) & set(hb) if ha[k] != hb[k]),
             "same": sorted(k for k in set(ha) & set(hb) if ha[k] == hb[k]),
             "sha256": {"a": ha, "b": hb}}

    def canon_sha(c):
        return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if c is not None else None

    rep = {
        "a": str(da), "b": str(db),
        "shards_present": {"a": sorted(sa), "b": sorted(sb)},
        "index_shards": {"a": sorted(set(xa["weight_map"].values())) if xa else None,
                         "b": sorted(set(xb["weight_map"].values())) if xb else None},
        "complete": {"a": bool(xa) and set(xa["weight_map"].values()) <= sa,
                     "b": bool(xb) and set(xb["weight_map"].values()) <= sb},
        "n_tensors": {"a": len(na), "b": len(nb), "a_with_header": len(ia), "b_with_header": len(ib)},
        "missing_in_b": [n for n in missing if n not in missing_scale],
        "extra_in_b": [n for n in extra if n not in extra_scale],
        "extra_in_b_scale_expected": extra_scale,
        "missing_in_b_scale_expected": missing_scale,
        "shape_mismatch": shape_mm,
        "dtype_mismatch": dtype_mm,
        "dtype_mismatch_fp8_expected": fp8_exp,
        "unresolved_no_header_yet": unresolved,
        "config": {
            "raw_sha256": {"a": ha.get("config.json"), "b": hb.get("config.json")},
            "raw_equal": ha.get("config.json") == hb.get("config.json"),
            "stripped_sha256": {"a": canon_sha(strip_quant(cfg_a)) if cfg_a else None,
                                "b": canon_sha(strip_quant(cfg_b)) if cfg_b else None},
            "stripped_equal": cfg_a is not None and cfg_b is not None and strip_quant(cfg_a) == strip_quant(cfg_b),
            "stripped_differences": json_diff(strip_quant(cfg_a), strip_quant(cfg_b)) if cfg_a and cfg_b else None,
        },
        "files": files,
    }
    if a.sample_rel_diff:
        common = [n for n in sorted(set(ia) & set(ib)) if ia[n][1] == ib[n][1] and len(ia[n][1]) >= 1
                  and n not in extra_scale and not is_scale_of(n, na, ia) and not is_scale_of(n, nb, ib)]
        if a.sample_names:
            pick = [n for n in a.sample_names.split(",") if n in common]
        else:
            k = min(a.sample_rel_diff, len(common))
            pick = [common[int(i * len(common) / k)] for i in range(k)] if k else []
        rep["rel_diff"] = {n: rel_diff(ta, tb, n, ia[n][1]) for n in pick}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write(Path(a.out), json.dumps(rep, indent=1).encode())
    print(f"wrote {a.out}: missing_in_b={len(rep['missing_in_b'])} extra_in_b={len(rep['extra_in_b'])} "
          f"shape_mm={len(shape_mm)} dtype_mm={len(dtype_mm)} fp8_expected={len(fp8_exp)} "
          f"config_stripped_equal={rep['config']['stripped_equal']} files_differ={files['differ']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dequant")
    d.add_argument("--src", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--threads", type=int, default=2)
    d.add_argument("--sha256", action="store_true", help="record sha256 of every written tensor in the manifest")
    d.add_argument("--force", action="store_true", help="rewrite shards even if a valid output exists")
    f = sub.add_parser("diff")
    f.add_argument("--a", required=True)
    f.add_argument("--b", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--sample-rel-diff", type=int, default=0)
    f.add_argument("--sample-names", default="", help="comma-separated tensor names to use instead of an even sample")
    a = ap.parse_args(argv)
    try:
        return cmd_dequant(a) if a.cmd == "dequant" else cmd_diff(a)
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
