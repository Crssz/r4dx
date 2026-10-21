"""tests/reference/test_refit_fallback.py -- the oracle's scale-refit solver fallback (tools/quant2/qwen35moe/refit_fallback.py) and
the line-ending-tolerant job match of oracle_q35.py. CPU only, ~1 min:

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tests\\reference\\test_refit_fallback.py

  * numerics: with the fallback installed, a refit whose solves succeed returns exactly (bit for bit) what the original function
    returns, and its record has no `fallback` key (so layers quantized before the fallback existed stay reproducible);
  * the ladder: a solver that raises on fp32 -> "lu_fp64_cpu"; on an exactly singular A (zero row and column, no ridge) ->
    "ridge_x0.0001"; on everything -> "lstsq_min_norm"; every stage returns a finite solution that solves the system it
    could (lu / ridge: close to the exact solution);
  * end to end: quantize_group on a dead-channel-heavy Hessian with the FIRST attempt of every solve forced to raise completes,
    carries `refit[*].fallback` and `refit_fallbacks`-countable rounds, and reaches (nearly) the same proxy as the unforced run;
  * oracle_q35.jobs_match: a stored job whose code hashes are those of the CRLF (or LF) form of the current source matches, a changed
    source does not.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "reference"))
sys.path.insert(0, str(REPO / "tools" / "quant2" / "qwen35moe"))

import torch  # noqa: E402

import refit_fallback as RF  # noqa: E402
import trellis_quant as tq  # noqa: E402

CHECKS, FAILS = 0, []


def check(cond, what):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(what)
        print("FAIL:", what)


def dead_hessian(k: int, dead: int, seed: int) -> torch.Tensor:
    """X^T X / rows of a tall X whose last `dead` columns are exactly zero: the second moments of an expert with dead channels."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * k, k, generator=g) * (1.0 + 3.0 * torch.rand(k, generator=g))
    X[:, k - dead:] = 0.0
    return (X.T @ X) / X.shape[0]


def refit_inputs(k=256, n=128, dead=200, seed=5):
    g = torch.Generator().manual_seed(seed)
    H = tq.damp_hessian(dead_hessian(k, dead, seed))
    W = torch.randn(k, n, generator=g) * 0.02
    Q = W + 0.004 * torch.randn(k, n, generator=g)
    return W, Q * 1.2, torch.ones(k), torch.ones(n), H


def test_bit_identical_when_nothing_raises():
    W, Q, su, sv, H = refit_inputs()
    orig = tq.refit_scales
    a = orig(W, Q.clone(), su.clone(), sv.clone(), H)
    RF.install(tq)
    wrapped = tq.refit_scales
    check(wrapped is not orig and wrapped.__wrapped__ is orig, "install() wraps tq.refit_scales")
    b = wrapped(W, Q.clone(), su.clone(), sv.clone(), H)
    check(all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])), "no raise: Q, su, sv bit-identical to the original refit_scales")
    check(a[3] == b[3] and not any("fallback" in r for r in b[3]), "no raise: record identical, no fallback key")
    check(torch.linalg.solve is RF._REAL_SOLVE, "torch.linalg.solve is restored after the call")
    RF.install(tq)
    check(tq.refit_scales is wrapped, "install() is idempotent")


def raising_when(pred):
    real = torch.linalg.solve

    def solve(A, b):
        if pred(A):
            raise torch._C._LinAlgError("torch.linalg.solve: The solver failed because the input matrix is singular.")
        return real(A, b)

    return solve


def test_ladder():
    real = RF._REAL_SOLVE
    g = torch.Generator().manual_seed(9)
    M = torch.randn(64, 64, generator=g)
    A = (M @ M.T + 0.5 * torch.eye(64))
    b = torch.randn(64, generator=g)
    exact = torch.linalg.solve(A.double(), b.double())
    try:
        RF._REAL_SOLVE = raising_when(lambda A_: A_.dtype == torch.float32)
        RF._log().clear()
        r = RF.solve_with_fallback(A, b)
        info = RF._log().pop()
        check(info is not None and info["stage"] == "lu_fp64_cpu" and r.dtype == torch.float32, "fp32 raise -> lu_fp64_cpu, dtype kept")
        check(float((r.double() - exact).abs().max()) < 1e-4, "lu_fp64_cpu solves the same system")

        # an exactly singular A (a zero row and column): only a ridge makes it solvable
        As = A.clone()
        As[7, :] = 0.0
        As[:, 7] = 0.0
        bs = b.clone()
        bs[7] = 0.0
        RF._REAL_SOLVE = real
        RF._log().clear()
        r = RF.solve_with_fallback(As, bs)
        info = RF._log().pop()
        check(info is not None and info["stage"] == "ridge_x0.0001" and bool(torch.isfinite(r).all()),
              f"exactly singular A -> {info and info['stage']}")
        check(float(r[7].abs()) < 1e-6 and float((As @ r - bs).norm()) < 1e-2 * float(bs.norm()),
              "ridge stage: zero on the dead row, residual small")

        RF._REAL_SOLVE = raising_when(lambda A_: True)
        RF._log().clear()
        r = RF.solve_with_fallback(As, bs)
        info = RF._log().pop()
        check(info["stage"] == "lstsq_min_norm" and bool(torch.isfinite(r).all()) and float(r[7].abs()) < 1e-6,
              f"everything raising -> {info['stage']} (minimum-norm: 0 on the null direction)")
        check("error" in info and info["k"] == 64, "the record keeps the first error text and k")
    finally:
        RF._REAL_SOLVE = real


def test_end_to_end():
    real = RF._REAL_SOLVE
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    k, n, dead = 256, 256, 220
    H = dead_hessian(k, dead, 11)
    g = torch.Generator().manual_seed(12)
    Whf = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
    RF.install(tq)

    def run():
        return tq.quantize_group({"w": Whf}, H, {"w": 4}, enc, 1, {"w": 2}, hessian_basis="matched", verbose=False)["w"]

    base = run()
    check(RF.count_fallbacks(base["record"]) == 0, "unforced run: no fallback")
    try:
        RF._REAL_SOLVE = raising_when(lambda A_: A_.dtype == torch.float32)
        forced = run()
    finally:
        RF._REAL_SOLVE = real
    rec = forced["record"]
    check(RF.count_fallbacks(rec) == tq.REFIT_ROUNDS and all(r["fallback"]["stage"] == "lu_fp64_cpu" for r in rec["refit"]),
          f"forced raise on every solve: {RF.count_fallbacks(rec)} fallback round(s), all lu_fp64_cpu")
    check(torch.equal(forced["words"], base["words"]), "the trellis words (LDLQ happens before refit) are identical")
    pb, pf = base["record"]["proxy"], rec["proxy"]
    check(abs(pf - pb) <= 0.02 * pb + 1e-9, f"proxy with the fallback {pf:.6g} ~ without {pb:.6g}")
    check(bool(torch.isfinite(forced["suh"].float()).all() and torch.isfinite(forced["svh"].float()).all()), "stored scales finite")


def test_job_match_eol():
    import oracle_q35 as O

    job = {"model_dir": str(REPO), "K": 4.0, "code_sha256": tq.code_sha256()}
    stored = copy.deepcopy(job)
    check(O.jobs_match(stored, job), "identical job matches")
    for variant in ("crlf", "lf"):
        s = copy.deepcopy(job)
        for key, path in (("trellis_quant", tq.__file__), ("trellis_viterbi", tq.NATIVE_SRC)):
            data = Path(path).read_bytes().replace(b"\r\n", b"\n")
            s["code_sha256"][key] = hashlib.sha256(data.replace(b"\n", b"\r\n") if variant == "crlf" else data).hexdigest()
        check(O.jobs_match(s, job), f"stored hashes of the {variant.upper()} form of the current source match")
    s = copy.deepcopy(job)
    s["code_sha256"]["trellis_quant"] = hashlib.sha256(b"a different trellis_quant.py").hexdigest()
    check(not O.jobs_match(s, job), "a changed source does not match")
    s = copy.deepcopy(job)
    s["K"] = 3.0
    check(not O.jobs_match(s, job), "another K does not match")


def main() -> int:
    test_bit_identical_when_nothing_raises()
    test_ladder()
    test_end_to_end()
    test_job_match_eol()
    if FAILS:
        print(f"{len(FAILS)} of {CHECKS} checks FAILED")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
