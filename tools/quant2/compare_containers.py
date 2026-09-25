#!/usr/bin/env python3
"""Compare two .r4dx containers: tensor directory, tensor bytes, and __metadata__.

An .r4dx file is a safetensors file (docs/container-format.md; src/convert's ContainerWriter,
src/model's SafetensorsReader): an 8-byte little-endian header length, the JSON header, then the data
section. Every header key except "__metadata__" is a tensor: {"dtype", "shape", "data_offsets":
[begin, end)} with offsets relative to the data section.

This checks, for A against B:
  * the tensor directory: the same names, and per tensor the same dtype, shape and data_offsets
    (the tensor-level metadata);
  * the data: sha256 of every tensor's bytes (read in parallel, straight from the files), and of
    every byte of the data section no tensor covers (gaps, trailing bytes) -- so "identical" means
    the whole data section is identical, not just the named ranges;
  * __metadata__: every differing leaf path ("r4dx_convert_run.reused_from", ...). Paths passed as
    --allow (a path or a prefix of one) are expected to differ and are reported as such; anything
    else is a difference;
  * each file's own r4dx_convert_run.reuse_guard.data_sha256, when present, recomputed from its
    tensors the way r4dx-convert records it (reuse_guard.hpp ContainerDataSha256: sha256 over
    "<name>\\n<sha256 of the tensor's bytes>\\n", names in byte order).

Exit status 0 when the containers are identical apart from the allowed metadata paths, 1 otherwise.
Python stdlib only.

  python tools/quant2/compare_containers.py a_reuse.r4dx a_full.r4dx --allow r4dx_convert_run.reused_from
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import sys
import time

CHUNK = 16 << 20


def read_header(path):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise SystemExit(f"{path}: shorter than 8 bytes")
        (n,) = struct.unpack("<Q", raw_len)
        if n == 0 or n > (1 << 30):
            raise SystemExit(f"{path}: implausible header length {n}")
        raw = f.read(n)
    if len(raw) != n:
        raise SystemExit(f"{path}: truncated header")
    header = json.loads(raw)
    meta = header.pop("__metadata__", None)
    return {"path": path, "header_len": n, "header_raw": raw, "data_start": 8 + n,
            "file_size": os.path.getsize(path), "tensors": header, "metadata": meta}


def hash_range(path, begin, end):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        f.seek(begin)
        left = end - begin
        while left > 0:
            b = f.read(min(CHUNK, left))
            if not b:
                raise IOError(f"{path}: short read at {begin + (end - begin) - left}")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def data_ranges(c):
    """(label, begin, end) in file offsets: every tensor, then every uncovered stretch of the data
    section (gaps between tensors and anything after the last one)."""
    ds = c["data_start"]
    spans = []
    for name, t in c["tensors"].items():
        b, e = t["data_offsets"]
        if not (0 <= b <= e) or ds + e > c["file_size"]:
            raise SystemExit(f"{c['path']}: tensor {name} range [{b},{e}) is outside the file")
        spans.append((b, e, name))
    spans.sort()
    out = [("tensor:" + name, ds + b, ds + e) for b, e, name in spans]
    cur = 0
    for b, e, name in spans:
        if b > cur:
            out.append((f"gap:[{cur},{b})", ds + cur, ds + b))
        cur = max(cur, e)
    if ds + cur < c["file_size"]:
        out.append((f"gap:[{cur},end)", ds + cur, c["file_size"]))
    return out


def digest_all(c, threads):
    ranges = data_ranges(c)
    ranges_by_size = sorted(ranges, key=lambda r: r[2] - r[1], reverse=True)
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(hash_range, c["path"], b, e): label for label, b, e in ranges_by_size}
        for fut in concurrent.futures.as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def container_data_sha256(tensor_digests):
    lines = sorted((name.encode("utf-8"), d) for name, d in tensor_digests.items())
    h = hashlib.sha256()
    for name, d in lines:
        h.update(name + b"\n" + d.encode("ascii") + b"\n")
    return h.hexdigest()


def flatten(j, prefix, out):
    if isinstance(j, dict) and j:
        for k, v in j.items():
            flatten(v, f"{prefix}.{k}" if prefix else k, out)
    else:
        out[prefix] = j


def allowed(path, allow):
    return any(path == a or path.startswith(a + ".") for a in allow)


def short(v, n=160):
    s = json.dumps(v, sort_keys=True)
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--allow", action="append", default=[],
                    help="__metadata__ leaf path (or prefix) expected to differ; repeatable")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json-out", help="write the full result as JSON here")
    args = ap.parse_args()

    t0 = time.time()
    A, B = read_header(args.a), read_header(args.b)
    problems = []
    result = {"a": args.a, "b": args.b, "allow": args.allow}

    # ---- tensor directory ------------------------------------------------------------------------
    na, nb = set(A["tensors"]), set(B["tensors"])
    only_a, only_b = sorted(na - nb), sorted(nb - na)
    if only_a:
        problems.append(f"{len(only_a)} tensor(s) only in A: {only_a[:10]}")
    if only_b:
        problems.append(f"{len(only_b)} tensor(s) only in B: {only_b[:10]}")
    dir_diff = []
    for name in sorted(na & nb):
        ta, tb = A["tensors"][name], B["tensors"][name]
        if ta != tb:
            dir_diff.append(name)
            if len(dir_diff) <= 10:
                problems.append(f"tensor {name}: directory entry differs: A {short(ta)} B {short(tb)}")
    if len(dir_diff) > 10:
        problems.append(f"... {len(dir_diff)} directory entries differ in total")
    data_a = A["file_size"] - A["data_start"]
    data_b = B["file_size"] - B["data_start"]
    if data_a != data_b:
        problems.append(f"data section sizes differ: A {data_a} B {data_b}")
    result["tensors"] = {"a": len(na), "b": len(nb), "directory_differs": dir_diff}
    result["data_bytes"] = {"a": data_a, "b": data_b}

    # ---- data ------------------------------------------------------------------------------------
    print(f"[cmp] hashing {args.a} ({data_a / 2**30:.2f} GiB of data) ...", flush=True)
    da = digest_all(A, args.threads)
    print(f"[cmp] hashing {args.b} ({data_b / 2**30:.2f} GiB of data) ...", flush=True)
    db = digest_all(B, args.threads)
    labels = sorted(set(da) | set(db))
    data_diff = [lb for lb in labels if da.get(lb) != db.get(lb)]
    for lb in data_diff[:20]:
        problems.append(f"data differs: {lb} (A {da.get(lb, '(absent)')[:16]}, B {db.get(lb, '(absent)')[:16]})")
    if len(data_diff) > 20:
        problems.append(f"... {len(data_diff)} data ranges differ in total")
    gaps_a = [lb for lb in da if lb.startswith("gap:")]
    result["data"] = {"ranges_compared": len(labels), "differs": data_diff, "gaps_in_a": gaps_a}

    # ---- each file's recorded data digest ---------------------------------------------------------
    result["data_sha256"] = {}
    for tag, c, d in (("a", A, da), ("b", B, db)):
        recorded = (((c["metadata"] or {}).get("r4dx_convert_run") or {}).get("reuse_guard") or {}).get("data_sha256")
        computed = container_data_sha256({lb[len("tensor:"):]: h for lb, h in d.items() if lb.startswith("tensor:")})
        result["data_sha256"][tag] = {"recorded": recorded, "computed": computed}
        if recorded is not None and recorded != computed:
            problems.append(f"{tag.upper()}: recorded reuse_guard.data_sha256 {recorded} != recomputed {computed}")

    # ---- __metadata__ ----------------------------------------------------------------------------
    fa, fb = {}, {}
    flatten(A["metadata"] or {}, "", fa)
    flatten(B["metadata"] or {}, "", fb)
    meta_diff, meta_allowed = [], []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k, "(absent)"), fb.get(k, "(absent)")
        if k in fa and k in fb and va == vb:
            continue
        (meta_allowed if allowed(k, args.allow) else meta_diff).append(k)
        if not allowed(k, args.allow):
            problems.append(f"__metadata__.{k}: A {short(va)} B {short(vb)}")
    result["metadata"] = {"differs": meta_diff, "differs_allowed": meta_allowed}
    # The header texts with the allowed paths removed from both: the same JSON (key order included)?
    def strip(meta):
        m = json.loads(json.dumps(meta))
        for p in args.allow:
            parts, cur = p.split("."), m
            for q in parts[:-1]:
                cur = cur.get(q) if isinstance(cur, dict) else None
            if isinstance(cur, dict):
                cur.pop(parts[-1], None)
        return m
    same_order = json.dumps(strip(A["metadata"])) == json.dumps(strip(B["metadata"]))
    result["metadata"]["same_key_order_after_allowed_removed"] = same_order
    if not same_order and not meta_diff:
        problems.append("__metadata__ has the same leaves but in another key order")

    result["identical_modulo_allowed"] = not problems
    result["problems"] = problems
    result["seconds"] = round(time.time() - t0, 1)

    print(f"[cmp] A: {args.a}: {len(na)} tensors, {data_a} data bytes, header {A['header_len']} B")
    print(f"[cmp] B: {args.b}: {len(nb)} tensors, {data_b} data bytes, header {B['header_len']} B")
    print(f"[cmp] tensor directory (name, dtype, shape, data_offsets): "
          f"{'identical' if not (only_a or only_b or dir_diff) else 'DIFFERS'}")
    print(f"[cmp] data: {len(labels)} range(s) compared ({len(gaps_a)} uncovered gap(s) in A), "
          f"{len(data_diff)} differ")
    for tag in ("a", "b"):
        d = result["data_sha256"][tag]
        if d["recorded"] is not None:
            print(f"[cmp] {tag.upper()} data_sha256 recorded {d['recorded']} "
                  f"{'== recomputed' if d['recorded'] == d['computed'] else '!= recomputed ' + d['computed']}")
    print(f"[cmp] __metadata__: {len(meta_allowed)} allowed differing leaf path(s): {meta_allowed}")
    print(f"[cmp] __metadata__: {len(meta_diff)} other differing leaf path(s)")
    for p in problems:
        print(f"  DIFF {p}")
    print(f"[cmp] {'IDENTICAL' if not problems else 'DIFFERENT'} (modulo --allow {args.allow}) "
          f"in {result['seconds']} s")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
