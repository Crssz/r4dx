"""tests/reference/test_manifest.py

CPU-only smoke test: does NOT need the C:\\AI\\models\\Qwen3.8-27B checkpoint,
$env:HIP_VISIBLE_DEVICES, or a GPU. Runs tools/reference/layer_golden.py in
--force-random-init mode against tiny sizes and checks the manifest.json it produces has the
shapes/dtypes/status fields a CI runner without weights or a GPU can still verify -- this is the
"packer byte-exact" rung's little sibling: it doesn't check numbers, just that the tool runs
end-to-end and produces a well-formed manifest.

Plain script, no pytest dependency (the reference venv this needs -- torch + transformers
importable -- doesn't have pytest installed, and is read-only; see tools/reference/README.md).
Run directly with the reference venv's python, or via ctest (registered as `reference_manifest`
in tests/CMakeLists.txt, which invokes this exact command):

    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tests\\reference\\test_manifest.py

Exits 0 and prints "OK (<n> checks)" on success; exits 1 and prints every failed assertion
otherwise (does not stop at the first failure, so a run reports everything wrong in one shot).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_REFERENCE_DIR = REPO_ROOT / "tools" / "reference"

try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP: torch not importable -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)
try:
    import transformers  # noqa: F401
except ImportError:
    print("SKIP: transformers not importable -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)

sys.path.insert(0, str(TOOLS_REFERENCE_DIR))
import layer_golden  # noqa: E402

EXPECTED_COMPONENTS = ["layer_000_gdn", "layer_003_full_attention", "final_norm_lm_head", "mtp"]

# The tensor names layer_golden.py's GdnCapture/hooks must produce for every GDN stage, per the
# Opus review that added gdn_core_attn_out/gdn_q_l2/gdn_k_l2/gdn_g_chunk_cumsum/gdn_z/gdn_a_raw/
# gdn_b_raw/gdn_gated_norm_out/gdn_out_proj_out and fixed the conv-output naming/length bug.
EXPECTED_GDN_STAGE_TENSORS = {
    "gdn_q", "gdn_k", "gdn_v", "gdn_g", "gdn_beta", "gdn_recurrent_state",
    "gdn_core_attn_out", "gdn_q_l2", "gdn_k_l2", "gdn_g_chunk_cumsum",
    "gdn_conv_out_fn_full", "gdn_conv_out_fn_trimmed",
    "post_input_norm", "post_attn_norm", "mlp_output", "layer_output",
}

# The tensor names every full-attention-type stage (layer_003_full_attention AND mtp, since MTP's
# inner block is also a full_attention-type Qwen3_5DecoderLayer) must produce, including the
# post-rope q/k and gate tensors the Opus review added.
EXPECTED_ATTN_STAGE_TENSORS = {
    "attn_qg_raw", "attn_k_raw", "attn_v_raw", "attn_q_normed", "attn_k_normed",
    "attention_output", "attn_gate_sigmoid", "attn_post_gate",
    "attn_q_post_rope", "attn_k_post_rope",
    "post_input_norm", "post_attn_norm", "mlp_output", "layer_output",
}


class Checker:
    def __init__(self):
        self.checked = 0
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.checked += 1
        if not condition:
            self.failures.append(message)

    def report(self) -> int:
        if self.failures:
            print(f"FAILED ({len(self.failures)}/{self.checked} checks failed):")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"OK ({self.checked} checks)")
        return 0


def run_golden(out_dir: Path, prefill_len: int, decode_len: int, tiny_vocab: int) -> dict:
    """Runs layer_golden.py once (random init, CPU, tiny sizes) and returns its manifest."""
    argv = [
        "layer_golden.py",
        "--device", "cpu",
        "--force-random-init",
        "--prefill-len", str(prefill_len),
        "--decode-len", str(decode_len),
        "--tiny-vocab", str(tiny_vocab),
        "--out-dir", str(out_dir),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = layer_golden.main()
    finally:
        sys.argv = old_argv
    assert rc == 0, "layer_golden.py main() returned non-zero in --force-random-init CPU mode"
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def check_manifest_has_all_required_components(c: Checker, manifest: dict) -> None:
    for name in EXPECTED_COMPONENTS:
        c.check(name in manifest["components"], f"missing required component {name!r}")


def check_component_ran_ok_on_random_init(c: Checker, manifest: dict) -> None:
    for name in EXPECTED_COMPONENTS:
        comp = manifest["components"].get(name)
        if comp is None:
            continue  # already reported by check_manifest_has_all_required_components
        c.check(comp["status"] == "ok", f"{name}: status != 'ok' ({comp.get('run_error')})")
        weights_key = "weights_source" if "weights_source" in comp else "final_norm_weights_source"
        c.check(
            comp[weights_key].startswith("random_init"),
            f"{name}: this test passes --force-random-init, so {weights_key} should start with "
            f"'random_init', got {comp[weights_key]!r} -- the CPU-only, no-weights smoke test isn't "
            "actually weights-free",
        )


def check_component_safetensors_file_written(c: Checker, out_dir: Path, manifest: dict) -> None:
    for name in EXPECTED_COMPONENTS:
        comp = manifest["components"].get(name)
        if comp is None:
            continue
        path = out_dir / comp["file"]
        c.check(path.suffix == ".safetensors", f"{name}: output file {path} is not .safetensors")
        c.check(path.exists() and path.stat().st_size > 0, f"{name}: output file {path} missing or empty")


def check_expected_tensor_shape_and_dtype(c: Checker, manifest: dict) -> None:
    cases = [
        ("layer_000_gdn", "prefill_layer_output"),
        ("layer_000_gdn", "decode_layer_output"),
        ("layer_000_gdn", "prefill_gdn_recurrent_state"),
        ("layer_000_gdn", "decode_gdn_recurrent_state"),
        ("layer_000_gdn", "prefill_gdn_beta"),
        ("layer_000_gdn", "prefill_gdn_core_attn_out"),
        ("layer_000_gdn", "prefill_gdn_q_l2"),
        ("layer_000_gdn", "prefill_gdn_k_l2"),
        ("layer_000_gdn", "prefill_gdn_g_chunk_cumsum"),
        ("layer_003_full_attention", "prefill_layer_output"),
        ("layer_003_full_attention", "prefill_attention_output"),
        ("layer_003_full_attention", "decode_attn_qg_raw"),
        ("layer_003_full_attention", "prefill_attn_q_post_rope"),
        ("layer_003_full_attention", "prefill_attn_k_post_rope"),
        ("layer_003_full_attention", "prefill_attn_gate_sigmoid"),
        ("mtp", "prefill_layer_output"),
    ]
    for name, tensor in cases:
        comp = manifest["components"].get(name)
        if comp is None:
            continue
        entry = comp["tensors"].get(tensor)
        if entry is None:
            c.failures.append(f"{name}: missing expected tensor {tensor!r}")
            c.checked += 1
            continue
        c.check("shape" in entry and "dtype" in entry, f"{name}.{tensor}: manifest entry missing shape/dtype")
        c.check(len(entry["shape"]) >= 1, f"{name}.{tensor}: shape has no dims")
        c.check(
            all(isinstance(d, int) and d > 0 for d in entry["shape"]),
            f"{name}.{tensor}: shape has a non-positive/non-int dim: {entry['shape']}",
        )
        c.check(
            entry["dtype"] in ("float32", "bfloat16", "float16"),
            f"{name}.{tensor}: unexpected dtype {entry['dtype']!r}",
        )


def check_full_gdn_tensor_set(c: Checker, manifest: dict) -> None:
    """Pins the full set of GDN intermediates per stage -- a regression that silently drops one of
    these (e.g. a future refactor that forgets a hook) fails here instead of only being noticed by
    whoever tries to diff against the missing tensor."""
    comp = manifest["components"].get("layer_000_gdn")
    if comp is None:
        return
    for stage in ("prefill", "decode"):
        present = {
            k[len(stage) + 1 :] for k in comp["tensors"] if k.startswith(f"{stage}_")
        }
        missing = EXPECTED_GDN_STAGE_TENSORS - present
        c.check(not missing, f"layer_000_gdn[{stage}]: missing tensors {sorted(missing)}")


def check_full_attn_tensor_set(c: Checker, manifest: dict) -> None:
    for component in ("layer_003_full_attention", "mtp"):
        comp = manifest["components"].get(component)
        if comp is None:
            continue
        for stage in ("prefill", "decode"):
            present = {
                k[len(stage) + 1 :] for k in comp["tensors"] if k.startswith(f"{stage}_")
            }
            missing = EXPECTED_ATTN_STAGE_TENSORS - present
            c.check(not missing, f"{component}[{stage}]: missing tensors {sorted(missing)}")


def check_decode_conv_out_length_matches_decode_len(c: Checker, manifest: dict, decode_len: int) -> None:
    """Regression pin for the Opus review's conv-output-length finding: decode's TRIMMED conv
    output must have exactly `decode_len` columns (the untrimmed `_full` tensor legitimately
    includes the cache-prepended warm-up columns and is intentionally longer)."""
    comp = manifest["components"].get("layer_000_gdn")
    if comp is None:
        return
    entry = comp["tensors"].get("decode_gdn_conv_out_fn_trimmed")
    if entry is None:
        c.failures.append("layer_000_gdn: missing decode_gdn_conv_out_fn_trimmed")
        c.checked += 1
        return
    # shape is [channels, tokens] (conv1d layout) -- the token axis is last.
    c.check(
        entry["shape"][-1] == decode_len,
        f"layer_000_gdn: decode_gdn_conv_out_fn_trimmed's last dim is {entry['shape'][-1]}, "
        f"expected decode_len={decode_len}",
    )


def check_final_norm_lm_head_logits_shape_matches_cli_args(c: Checker, manifest: dict, prefill_len: int, tiny_vocab: int) -> None:
    comp = manifest["components"].get("final_norm_lm_head")
    if comp is None:
        return
    c.check(
        comp["tensors"]["logits"]["shape"] == [prefill_len, tiny_vocab],
        f"final_norm_lm_head: logits shape {comp['tensors']['logits']['shape']} != "
        f"[{prefill_len}, {tiny_vocab}]",
    )


def check_mtp_reports_checkpoint_absence_honestly(c: Checker, manifest: dict) -> None:
    """--force-random-init never touches the checkpoint, so the mtp component must say so rather
    than silently claiming real weights were used."""
    comp = manifest["components"].get("mtp")
    if comp is None:
        return
    c.check(comp["mtp_weights_present_in_checkpoint"] is False, "mtp: mtp_weights_present_in_checkpoint should be False")
    c.check("architecture_assumption" in comp, "mtp: missing architecture_assumption field")


def check_manifest_declares_tolerances(c: Checker, manifest: dict) -> None:
    for key in ("bf16_matmul_rel_err", "fp32_elementwise_rel_err", "recurrent_state_rel_err"):
        c.check(key in manifest["tolerances"], f"tolerances missing {key!r}")
        c.check(manifest["tolerances"].get(key, 0) > 0, f"tolerances[{key!r}] should be > 0")


def main() -> int:
    prefill_len, decode_len, tiny_vocab = 8, 2, 16
    c = Checker()
    with tempfile.TemporaryDirectory(prefix="r4dx_reference_manifest_test_") as tmp:
        out_dir = Path(tmp)
        try:
            manifest = run_golden(out_dir, prefill_len, decode_len, tiny_vocab)
        except Exception:
            print("FAILED: layer_golden.py raised while generating the golden manifest:")
            traceback.print_exc()
            return 1

        check_manifest_has_all_required_components(c, manifest)
        check_component_ran_ok_on_random_init(c, manifest)
        check_component_safetensors_file_written(c, out_dir, manifest)
        check_expected_tensor_shape_and_dtype(c, manifest)
        check_full_gdn_tensor_set(c, manifest)
        check_full_attn_tensor_set(c, manifest)
        check_decode_conv_out_length_matches_decode_len(c, manifest, decode_len)
        check_final_norm_lm_head_logits_shape_matches_cli_args(c, manifest, prefill_len, tiny_vocab)
        check_mtp_reports_checkpoint_absence_honestly(c, manifest)
        check_manifest_declares_tolerances(c, manifest)

    return c.report()


if __name__ == "__main__":
    raise SystemExit(main())
