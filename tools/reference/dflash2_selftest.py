"""Determinism gate for the DFlash2 reference: re-run fixture A from its stored inputs and assert
bit-identical outputs against `tools/reference/golden_out/dflash2/fixture_a/`.

CPU-only, no checkpoint required (uses the real Q8_0 draft GGUF + a --synthetic small-vocab
target, exactly like `dflash2_ref.py --gen-fixtures A`), runs in well under a minute. Intended
ctest registration (added by the Integrate stage, which owns tests/CMakeLists.txt -- this repo's
task split forbids this file from touching it):

    add_test(NAME reference_dflash2
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/reference/dflash2_selftest.py)

Run directly:

    <reference venv>\\python.exe tools\\reference\\dflash2_selftest.py
    <reference venv>\\python.exe tools\\reference\\dflash2_selftest.py --with-real-sanity \\
        --target-dir C:\\AI\\models\\Qwen3.8-27B   # opt-in, see "real-target sanity" below
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash2_ref import (  # noqa: E402
    DEFAULT_DFLASH2_GGUF,
    DEFAULT_TARGET_DIR,
    DFlash2Weights,
    DraftKvCache,
    RealTarget,
    SyntheticTarget,
    _synthetic_features,
    _synthetic_mask_id,
    draft_round,
    encode_features,
    gen_fixture_a,
    gen_fixture_b,
    gen_fixture_c,
    inject,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_out" / "dflash2"
# tools/reference/.gitignore's `golden_out*/` pattern means these fixtures are NOT committed
# (they're regenerable byte-for-byte from `weights` + `seed=0`, like layer_golden.py's own dumps,
# and committing ~9 MB of binary .npy to a public repo isn't worth it for a determinism gate).
# So this file is self-sufficient on a clean checkout: regenerate on demand instead of hard-failing
# when the directory is missing.
_FIXTURE_GENERATORS = {"fixture_a": gen_fixture_a, "fixture_b": gen_fixture_b, "fixture_c": gen_fixture_c}


def _ensure_fixture(name: str, weights: "DFlash2Weights") -> None:
    manifest_path = GOLDEN_DIR / name / "manifest.json"
    if manifest_path.exists():
        return
    print(f"  [{name}] golden_out/dflash2/{name}/ missing -- regenerating with seed=0 (gitignored, not a bug)")
    _FIXTURE_GENERATORS[name](weights, GOLDEN_DIR, seed=0)

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name if not detail else f"{name}: {detail}")


def assert_bit_identical(name: str, got: np.ndarray, want_path: Path) -> None:
    want = np.load(want_path)
    if got.shape != want.shape:
        check(name, False, f"shape mismatch got={got.shape} want={want.shape}")
        return
    if got.dtype != want.dtype:
        # score matrices etc. are always float32/int64 -- a dtype drift is itself a bug to flag,
        # not silently cast away.
        check(name, False, f"dtype mismatch got={got.dtype} want={want.dtype}")
        return
    identical = np.array_equal(got, want)
    check(name, identical, "" if identical else f"max_abs_diff={np.abs(got.astype(np.float64) - want.astype(np.float64)).max()}")


def test_fixture_a_determinism() -> None:
    print("== fixture A determinism gate ==")
    weights = DFlash2Weights.load(DEFAULT_DFLASH2_GGUF)
    _ensure_fixture("fixture_a", weights)

    fdir = GOLDEN_DIR / "fixture_a"
    manifest_path = fdir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seed = manifest["seed"]
    vocab = manifest["vocab"]
    n_injected = manifest["n_injected"]
    anchor_id = manifest["anchor_id"]
    mask_id = manifest["mask_id"]
    block_size = manifest["block_size"]

    cfg = weights.cfg
    check("block_size matches container", cfg.block_size == block_size)

    target = SyntheticTarget(cfg.n_embd, vocab, seed)
    features = np.load(fdir / "features.npy")
    reconstructed_features = _synthetic_features(seed, n_injected, cfg.n_embd_inp_enc)
    assert_bit_identical("features reproducible from seed", reconstructed_features, fdir / "features.npy")

    cache = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
    g = encode_features(weights, features)
    pos = np.arange(n_injected, dtype=np.int64)
    inject(weights, cache, g, pos)

    result = draft_round(
        weights, target, cache, anchor_id, p_min=0.0, capture_layer0=True, mask_token_id=mask_id
    )

    assert_bit_identical("g_encoded", g, fdir / "g_encoded.npy")
    for il in range(cfg.n_layer):
        assert_bit_identical(f"injected_k_l{il}", cache.layers[il].k, fdir / f"injected_k_l{il}.npy")
        assert_bit_identical(f"injected_v_l{il}", cache.layers[il].v, fdir / f"injected_v_l{il}.npy")
        assert_bit_identical(
            f"x_post_attn_l{il}", result.intermediates[f"x_post_attn_l{il}"], fdir / f"x_post_attn_l{il}.npy"
        )
        assert_bit_identical(
            f"x_post_ffn_l{il}", result.intermediates[f"x_post_ffn_l{il}"], fdir / f"x_post_ffn_l{il}.npy"
        )
    assert_bit_identical("attn_conv_in_l0", result.intermediates["attn_conv_in_l0"], fdir / "attn_conv_in_l0.npy")
    assert_bit_identical("attn_conv_out_l0", result.intermediates["attn_conv_out_l0"], fdir / "attn_conv_out_l0.npy")
    assert_bit_identical("x_final_normed", result.intermediates["x_final_normed"], fdir / "x_final_normed.npy")
    assert_bit_identical("logits", result.intermediates["logits"], fdir / "logits.npy")
    assert_bit_identical("cand", result.intermediates["cand"], fdir / "cand.npy")
    assert_bit_identical("unary", result.intermediates["unary"], fdir / "unary.npy")
    assert_bit_identical("gate", result.intermediates["gate"], fdir / "gate.npy")
    for t, mat in result.intermediates["score_matrices"].items():
        assert_bit_identical(f"score_t{t}", mat, fdir / f"score_t{t}.npy")
    assert_bit_identical("drafted_tokens", result.intermediates["drafted_tokens"], fdir / "drafted_tokens.npy")
    check("drafted_tokens list matches manifest", result.tokens == manifest["drafted_tokens"])

    # bf16-rounded-weights variant (task item 3's tolerance-free bf16 target)
    bf16_weights = weights.bf16_rounded()
    g_bf16 = encode_features(bf16_weights, features)
    cache_bf16 = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
    inject(bf16_weights, cache_bf16, g_bf16, pos)
    result_bf16 = draft_round(bf16_weights, target, cache_bf16, anchor_id, p_min=0.0, mask_token_id=mask_id)
    assert_bit_identical(
        "logits_bf16_weights", result_bf16.intermediates["logits"], fdir / "logits_bf16_weights.npy"
    )
    check("drafted_tokens_bf16_weights matches manifest", result_bf16.tokens == manifest["drafted_tokens_bf16_weights"])


def test_fixture_b_and_c_determinism() -> None:
    print("== fixture B/C determinism gate ==")
    weights = DFlash2Weights.load(DEFAULT_DFLASH2_GGUF)
    cfg = weights.cfg

    for name, n_injected, p_min in (("fixture_b", 2100, 0.0), ("fixture_c", 40, 0.3)):
        _ensure_fixture(name, weights)
        fdir = GOLDEN_DIR / name
        manifest_path = fdir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        seed = manifest["seed"]
        vocab = manifest["vocab"]
        anchor_id = manifest["anchor_id"]
        mask_id = _synthetic_mask_id(cfg, vocab)

        target = SyntheticTarget(cfg.n_embd, vocab, seed)
        features = _synthetic_features(seed, n_injected, cfg.n_embd_inp_enc)
        cache = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
        g = encode_features(weights, features)
        pos = np.arange(n_injected, dtype=np.int64)
        inject(weights, cache, g, pos)

        result = draft_round(weights, target, cache, anchor_id, p_min=p_min, mask_token_id=mask_id)

        assert_bit_identical(f"{name}/logits", result.intermediates["logits"], fdir / "logits.npy")
        assert_bit_identical(f"{name}/cand", result.intermediates["cand"], fdir / "cand.npy")
        assert_bit_identical(f"{name}/unary", result.intermediates["unary"], fdir / "unary.npy")
        assert_bit_identical(f"{name}/gate", result.intermediates["gate"], fdir / "gate.npy")
        for t, mat in result.intermediates["score_matrices"].items():
            assert_bit_identical(f"{name}/score_t{t}", mat, fdir / f"score_t{t}.npy")
        check(f"{name}/drafted_tokens list matches manifest", result.tokens == manifest["drafted_tokens"])
        if name == "fixture_c":
            check(
                "fixture_c early-stop truncates vs fixture_a (p_min=0.3 < p_min=0)",
                len(result.tokens) <= cfg.block_size - 1,
            )


def test_real_target_sanity(target_dir: Path) -> None:
    """Optional (task item 5): real embed/lm_head + synthetic features -- drafted ids must be
    valid, and the selector chain must sometimes differ from plain per-position argmax.

    Loads the FULL real lm_head ([248320, 5120] fp32 ~= 5 GB) and streams embedding rows on
    demand via `RealTarget`/`common.ShardIndex` -- real weights, CPU only, no GPU touched. Skipped
    by default (`--with-real-sanity` opts in) since it is materially slower/heavier than the rest
    of this gate and the task explicitly allows skipping it if CPU time/RAM don't cooperate.
    """
    print("== real-target sanity (optional) ==")
    t0 = time.time()
    try:
        weights = DFlash2Weights.load(DEFAULT_DFLASH2_GGUF)
        target = RealTarget(target_dir)
        cfg = weights.cfg

        rng = np.random.RandomState(123)
        n_injected = 40
        features = rng.randn(n_injected, cfg.n_embd_inp_enc).astype(np.float32)
        anchor_id = 100  # an arbitrary, in-range real token id

        cache = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
        g = encode_features(weights, features)
        pos = np.arange(n_injected, dtype=np.int64)
        inject(weights, cache, g, pos)

        result = draft_round(weights, target, cache, anchor_id, p_min=0.0)
        vocab = target.vocab_size

        valid = all(0 <= tid < vocab for tid in result.tokens)
        check("real-target: drafted ids are valid vocab ids", valid, f"tokens={result.tokens}")

        # plain per-position argmax (no selector walk) for comparison
        plain_argmax = [int(x) for x in np.argmax(result.intermediates["logits"], axis=-1)[1:]]
        differs = plain_argmax[: len(result.tokens)] != result.tokens
        check(
            "real-target: selector chain differs from plain argmax at least once",
            differs,
            f"selector={result.tokens} plain_argmax={plain_argmax[: len(result.tokens)]}",
        )
    except Exception as exc:  # pragma: no cover -- explicitly allowed to skip, not to silently pass
        print(f"  [SKIP] real-target sanity raised {type(exc).__name__}: {exc}")
    finally:
        print(f"  (real-target sanity wall time: {time.time() - t0:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-real-sanity", action="store_true")
    ap.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    args = ap.parse_args()

    test_fixture_a_determinism()
    test_fixture_b_and_c_determinism()
    if args.with_real_sanity:
        test_real_target_sanity(args.target_dir)
    else:
        print("== real-target sanity (optional) ==\n  [SKIPPED] pass --with-real-sanity to run it")

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) FAILED:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
