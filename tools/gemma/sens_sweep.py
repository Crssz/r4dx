"""Per-layer K4 -> K5 sensitivity sweep for the Gemma trellis mix (docs/gemma4-plan.md item 12, re-rank).

For each requested layer L: a manifest that is the all-K4 oracle except layer L's tensors at K5, converted (CPU),
dumped on the chat gate corpus (GPU, --device), scored against the fp32 truth (kl_report --gate gemma-fp32), the
container and the log-prob dump deleted. Layer -1 is the all-K4 baseline. One JSON line per run in
<out>/results.jsonl: {"layer", "groups": {name: mean_kl}, "top1": {...}, "bits_extra"}.

  python tools/gemma/sens_sweep.py --device 1 --layers -1,0,2,4,... --out <models root>\\r4dx\\huihui-gemma\\sens
(<models root> = $env:R4DX_MODELS_ROOT)
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(os.environ.get("R4DX_MODELS_ROOT", r"E:\models"))
HG = MODELS_ROOT / "r4dx" / "huihui-gemma"
Q = HG / "trellis" / "q"
MODEL = str(MODELS_ROOT / "Huihui-gemma-4-12B-it-abliterated")
CONVERT = REPO / r"build\win-hip-merge\src\convert\r4dx-convert.exe"
TOOL = REPO / r"build\win-hip-merge\tests\model\tool_teacher_forced_logprobs.exe"
KL = HG / "kl"
PY = r"D:\venvs\r4dx-gemma-ref\Scripts\python.exe"


def manifest_for(layer: int, out: Path) -> tuple[Path, int]:
    # Start from a real MIX manifest (the converter's mix form: `allocation`, no `layers_done`) and set every tensor
    # entry from the K4 / K5 oracle manifests.
    base = json.loads((Q / "mix4.5m" / "weights_override.json").read_text(encoding="utf-8"))
    k4 = json.loads((Q / "K4m" / "weights_override.json").read_text(encoding="utf-8"))
    k5 = json.loads((Q / "K5m" / "weights_override.json").read_text(encoding="utf-8"))
    extra = 0
    n5 = 0
    tag = f"model.language_model.layers.{layer}."
    def entry(src: dict, k_dir: str, name: str) -> dict:
        e = dict(src["tensors"][name])
        if not Path(e["file"]).is_absolute():  # an oracle manifest names its files relative to its own directory
            e["file"] = str(Q / k_dir / e["file"])
        return e

    for name in base["tensors"]:
        if layer >= 0 and name.startswith(tag):
            extra += k5["tensors"][name]["bits"]["total"] - k4["tensors"][name]["bits"]["total"]
            base["tensors"][name] = entry(k5, "K5m", name)
            n5 += 1
        else:
            base["tensors"][name] = entry(k4, "K4m", name)
    base["bpw_target"] = None
    base["allocation"]["rule"] = f"sens_sweep: all K4, layer {layer} at K5"
    base["allocation"]["ranking"] = "sens-sweep"
    base["allocation"]["tensors_per_K"] = {"4.0": len(base["tensors"]) - n5, "5.0": n5}
    out.mkdir(parents=True, exist_ok=True)
    p = out / "weights_override.json"
    p.write_text(json.dumps(base, indent=1), encoding="utf-8")
    return p, extra


def run(cmd: list[str], env_dev: str, log: Path) -> None:
    env = dict(os.environ, HIP_VISIBLE_DEVICES=env_dev, R4DX_GEMMA_KV="bf16")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if env_dev == "-1":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    with open(log, "ab") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode} (see {log})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    done = set()
    res_path = a.out / f"results_dev{a.device}.jsonl"
    if res_path.exists():
        done = {json.loads(l)["layer"] for l in res_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    for layer in [int(x) for x in a.layers.split(",")]:
        if layer in done:
            continue
        d = a.out / (f"L{layer:02d}" if layer >= 0 else "base_k4")
        log = d / "run.log"
        man, extra = manifest_for(layer, d)
        sha = hashlib.sha256(man.read_bytes()).hexdigest()
        cont = d / "c.r4dx"
        run([str(CONVERT), "--input", MODEL, "--output", str(cont), "--trellis-from", str(d), "--trellis-manifest-sha256",
             sha, "--trellis-verify", "full", "--kv-calib", str(HG / "kvcalib.json"), "--no-bf16",
             "--lm-head", "w4a16", "--w4a16-group-rule", "^lm_head$=32", "--hessian-dir",
             str(HG / "hessian-v1"), "--ldlq", "^lm_head$", "--threads", "12"], "-1", log)
        dump = d / "dump"
        dump.mkdir(exist_ok=True)
        run([str(TOOL), "--model", str(cont), "--layout", "trellis", "--tokens", str(REPO / r"tools\reference\kl_corpus\chat_gemma.json"),
             "--out-dir", str(dump), "--max-ctx", "4096", "--quiet"], a.device, log)
        gate = d / "gate.json"
        env = dict(os.environ, HIP_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1")
        subprocess.run([PY, str(REPO / r"tools\reference\kl_report.py"), "--gate", "gemma-fp32", "--truth-dir", str(KL / "fp32" / "truth"),
                        "--noise-dir", str(KL / "fp32" / "bf16sdpa"), "--test-dir", str(dump), "--tokens",
                        str(REPO / r"tools\reference\kl_corpus\chat_gemma.json"), "--out", str(gate)],
                       stdout=open(log, "ab"), stderr=subprocess.STDOUT, env=env)
        g = json.loads(gate.read_text(encoding="utf-8"))
        groups = {x.get("name", x.get("group", str(i))): x["test"]["mean_kl"] for i, x in enumerate(g["groups"])}
        top1 = {x.get("name", x.get("group", str(i))): x["test"]["top1_pct"] for i, x in enumerate(g["groups"])}
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"layer": layer, "groups": groups, "top1": top1, "bits_extra": extra}) + "\n")
        cont.unlink(missing_ok=True)
        shutil.rmtree(dump, ignore_errors=True)
        print(f"[sens] layer {layer}: {groups}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

