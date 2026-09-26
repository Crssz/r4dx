"""tools/reference/trellis_isa_report.py -- per-instantiation ISA report for the trellis GEMM (no GPU).

docs/trellis-kernel.md 4.2 asks M1 to report the WHOLE-LOOP VALU count of every instantiation of
r4d_gemm_trellis_nt_m64 (not only the decode subset), and 4.3 caps every instantiation at 190 VGPRs
(8 waves per SIMD) with no scratch. This reads a device-only assembly listing of
third_party/libr4d/r4d_gemm_trellis_nt_m64.hip -- the build writes one next to its objects,
build/<preset>/third_party/r4d_objs/r4d_gemm_trellis_nt_m64-gfx1201.s, and
third_party/check_trellis_isa.cmake enforces the VGPR/scratch cap from the same file -- and prints,
per kernel:

  vgpr, sgpr, scratch bytes, spills (from the code-object metadata);
  the main K loop (the loop with the most v_wmma): its instruction mix by class, and VALU per
  (tile pair, k-tile) and per weight. One loop iteration is the kernel's ping-pong pair of steps
  (2 U k-tiles for each of NP pairs; one step at MT >= 3), counted from its WMMA: 2 MT per
  (pair, k-tile). A lane decodes 16 weights per (pair, k-tile). The decode
  subset is 62 VALU per (pair, k-tile) (6 alignbit, 16 mad_u32_u16, 16 pk_mad_u16, 8 sad_u8,
  8 sad_hi_u8, 8 pk_fma_f16); everything else in the loop is overhead;
  the loop's schedule: its s_delay_alu count, and its NEAR dependencies -- VALU (WMMA included)
  reading a VGPR written by one of the 3 VALU before it with no s_delay_alu in between -- in all
  (`near`) and those whose producer is inline asm (`near_asm`). gfx12 stalls the whole SIMD's VALU
  on a near dependency the compiler did not mark with s_delay_alu, and it marks only the ones whose
  producer it generated, so near_asm must be 0 (third_party/check_trellis_isa.cmake enforces it for
  the whole kernel); the rest are the compiler's own WMMA -> WMMA accumulator chains.

  & $py tools\\reference\\trellis_isa_report.py <listing.s> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

KERNEL_RE = re.compile(r"r4d_gemm_trellis_nt_m64_raw_kernelILi(\d+)ELi(\d+)ELi(\d+)ELb([01])E")
DECODE_OPS = ("v_alignbit_b32", "v_mad_u32_u16", "v_pk_mad_u16", "v_sad_u8", "v_sad_hi_u8",
              "v_pk_fma_f16")


def split_functions(lines: list[str]) -> dict[str, list[str]]:
    """mangled kernel name -> its body lines (label to .Lfunc_end)."""
    out: dict[str, list[str]] = {}
    cur = None
    for ln in lines:
        if cur is None:
            m = re.match(r"^([_A-Za-z0-9.$]+):\s*(;.*)?$", ln)
            if m and not m.group(1).startswith("."):
                cur = m.group(1)
                out[cur] = []
            continue
        if re.match(r"^\.Lfunc_end\d+:", ln):
            cur = None
            continue
        out[cur].append(ln)
    return out


def metadata(text: str) -> dict[str, dict[str, int]]:
    """.name -> {vgpr, sgpr, scratch, vgpr_spill, sgpr_spill} from the amdhsa.kernels block."""
    meta: dict[str, dict[str, int]] = {}
    keys = {".vgpr_count": "vgpr", ".sgpr_count": "sgpr", ".private_segment_fixed_size": "scratch",
            ".vgpr_spill_count": "vgpr_spill", ".sgpr_spill_count": "sgpr_spill"}
    block: dict[str, int] = {}
    name = None
    for ln in text.splitlines():
        s = ln.strip().lstrip("- ").strip()
        for k, v in keys.items():
            if s.startswith(k + ":"):
                block[v] = int(s.split(":", 1)[1])
        if s.startswith(".name:"):
            name = s.split(":", 1)[1].strip()
        if name and all(v in block for v in keys.values()):
            meta[name] = dict(block)
            block, name = {}, None
    return meta


def instr(ln: str) -> str | None:
    s = ln.split(";", 1)[0].strip()
    if not s or s.endswith(":") or s.startswith("."):
        return None
    return s.split()[0]


def main_loop(body: list[str]) -> list[str]:
    """The backward-branch loop with the most v_wmma (label line to branch line)."""
    labels = {}
    best: list[str] = []
    best_w = -1
    for i, ln in enumerate(body):
        m = re.match(r"^(\.LBB\d+_\d+):", ln)
        if m:
            labels[m.group(1)] = i
        m = re.match(r"^\s+s_cbranch_\w+\s+(\.LBB\d+_\d+)", ln) or re.match(r"^\s+s_branch\s+(\.LBB\d+_\d+)", ln)
        if m and m.group(1) in labels:
            seg = body[labels[m.group(1)]:i + 1]
            w = sum(1 for x in seg if (instr(x) or "").startswith("v_wmma"))
            if w > best_w:
                best, best_w = seg, w
    return best


def classify(seg: list[str]) -> Counter:
    c: Counter = Counter()
    for ln in seg:
        op = instr(ln)
        if op is None:
            continue
        if op.startswith("v_wmma"):
            c["wmma"] += 1
        elif op.startswith("v_"):
            c["valu"] += 1
            c[op if op in DECODE_OPS else "valu_other"] += 1
            if op not in DECODE_OPS:
                c["other:" + op] += 1
        elif op.startswith(("global_load", "buffer_load", "flat_load")):
            c["vmem_load"] += 1
        elif op.startswith("s_wait"):
            c["wait"] += 1
        elif op.startswith("s_"):
            c["salu"] += 1
        elif op.startswith("ds_"):
            c["lds"] += 1
        else:
            c["other"] += 1
    return c


def vregs(text: str) -> set[int]:
    out: set[int] = set()
    for a, b, c in re.findall(r"v\[(\d+):(\d+)\]|v(\d+)\b", text):
        if c:
            out.add(int(c))
        else:
            out.update(range(int(a), int(b) + 1))
    return out


def near_deps(seg: list[str]) -> Counter:
    """s_delay_alu count and near dependencies (see the module docstring) of a loop body."""
    c: Counter = Counter()
    window: list[tuple[set[int], bool]] = []    # the last 3 VALU: (destination VGPRs, from asm)
    in_asm = False
    for ln in seg:
        if ";;#ASMSTART" in ln:
            in_asm = True
        elif ";;#ASMEND" in ln:
            in_asm = False
        op = instr(ln)
        if op is None:
            continue
        if op == "s_delay_alu":
            c["delay_alu"] += 1
            window = []
            continue
        if not op.startswith("v_"):
            continue
        dst: set[int] = set()
        src: set[int] = set()
        for half in ln.split(";", 1)[0].strip().split("::"):
            parts = half.strip().split(None, 1)
            if len(parts) < 2:
                continue
            args = parts[1].split(",", 1)
            dst |= vregs(args[0])
            if len(args) > 1:
                src |= vregs(args[1])
        if op.startswith("v_wmma"):
            src |= dst
        for d, from_asm in reversed(window):
            if d & src:
                c["near"] += 1
                c["near_asm"] += from_asm
                break
        window = (window + [(dst, in_asm)])[-3:]
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("listing")
    ap.add_argument("--json", help="also write the rows as JSON")
    args = ap.parse_args()
    text = open(args.listing, encoding="utf-8", errors="replace").read()
    funcs = split_functions(text.splitlines())
    meta = metadata(text)
    rows = []
    for name, body in funcs.items():
        m = KERNEL_RE.search(name)
        if not m:
            continue
        NP, U, MT, NT = (int(g) for g in m.groups())
        md = meta.get(name, {})
        loop = main_loop(body)
        c = classify(loop)
        nd = near_deps(loop)
        tiles = c["wmma"] // (2 * MT)                        # (pair, k-tile) per loop iteration
        rows.append({
            "NP": NP, "U": U, "MT": MT, "NT": NT, **md,
            "loop_valu": c["valu"], "loop_wmma": c["wmma"], "loop_vmem": c["vmem_load"],
            "loop_salu": c["salu"], "loop_wait": c["wait"],
            "loop_delay_alu": nd["delay_alu"], "loop_near": nd["near"], "loop_near_asm": nd["near_asm"],
            "valu_per_pair_ktile": c["valu"] / tiles if tiles else 0.0,
            "valu_per_weight": c["valu"] / (16 * tiles) if tiles else 0.0,
            "decode": {op: c[op] for op in DECODE_OPS},
            "overhead_ops": {k[6:]: v for k, v in sorted(c.items()) if k.startswith("other:")},
        })
    rows.sort(key=lambda r: (r["NP"], r["U"], r["MT"], r["NT"]))
    print(f"{'NP':>2} {'U':>2} {'MT':>2} {'NT':>2} {'vgpr':>4} {'sgpr':>4} {'scr':>3} | "
          f"{'VALU':>5} {'WMMA':>4} {'VMEM':>4} {'SALU':>4} | {'VALU/(pair,kt)':>14} {'VALU/w':>6} | "
          f"{'dly':>3} {'near':>4} {'asm':>3} | overhead")
    for r in rows:
        ov = ", ".join(f"{k} {v}" for k, v in r["overhead_ops"].items())
        print(f"{r['NP']:>2} {r['U']:>2} {r['MT']:>2} {r['NT']:>2} {r.get('vgpr', -1):>4} {r.get('sgpr', -1):>4} "
              f"{r.get('scratch', -1):>3} | {r['loop_valu']:>5} {r['loop_wmma']:>4} {r['loop_vmem']:>4} "
              f"{r['loop_salu']:>4} | {r['valu_per_pair_ktile']:>14.2f} {r['valu_per_weight']:>6.3f} | "
              f"{r['loop_delay_alu']:>3} {r['loop_near']:>4} {r['loop_near_asm']:>3} | {ov}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
