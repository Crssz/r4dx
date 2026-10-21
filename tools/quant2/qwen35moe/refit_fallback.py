"""tools/quant2/qwen35moe/refit_fallback.py -- a principled fallback for a failing scale-refit solve of the trellis oracle.

`trellis_quant.refit_scales` ends every refit round with `torch.linalg.solve(A, b)`, A = (Q Q^T) o H_damped + 1e-6 mean(diag) I,
b = rowsum(Q o (H W)). On the qwen3_5_moe run (2026-10-21) the K4 oracle of layer 2 died there once, with "The solver failed because
the input matrix is singular", after ~5,000 solves had passed (layers 0-1 at K3 and K4, layer 2 at K3).

Cause (README "Singular refit"): the model's DEAD channels (an expert's down_proj input can have 1014 of 1024 channels whose second
moment is exactly 0) make H nearly block diagonal with a tiny live block, and the refit ridge is 1e-6 * mean(diag A), a mean taken
over mostly dead rows. For layer 2's 256 down_proj systems the resulting condition number is 5e6 .. 8e8 (median 4e7, 219 of 256
above 1/eps of fp32); the smallest eigenvalue of A itself is ~1e-25 relative, numerically 0. Such a system is not singular in exact
arithmetic (the ridge keeps it positive) but an fp32 pivoted LU can meet an exactly zero pivot, which is what the solver reports.
It is a knife edge: re-running layer 2 / K4 (all 8 expert chunks + projections, identical inputs, bit-identical outputs for the
chunks both runs finished) did not raise again, and 400 fp32 solves of the worst systems on the GPU did not either.

`install()` replaces `trellis_quant.refit_scales` by a wrapper that runs the ORIGINAL function with `torch.linalg.solve` swapped for
`solve_with_fallback`. Not a fork of the function: the first attempt of every solve is the original call with the original arguments,
so a group whose solves all succeed gets bit-identical results (layers 0-1 and 2-K3 stay valid and reproducible, and
`trellis_quant.py` itself, whose sha256 is part of the oracle's job key, is not touched). Only a solve that RAISES goes down the ladder:

  1. "lu_fp64_cpu":    the same A, b in float64 on the CPU (LAPACK LU);
  2. "ridge_x0.0001" / "ridge_x0.01": A + lam * mean(diag A) * I, CPU float64 (refit's own 1e-6 ridge made 100x / 10,000x larger);
  3. "lstsq_min_norm": the minimum-norm least-squares solution (gelsd): what a rank-deficient A's null space gets.

Every fallback is recorded: in the group record json (`refit[round].fallback = {stage, error, device, k}` and, only when it happened,
`refit_fallbacks` = number of rounds), so `alloc_q35.py` can report how many groups used it. Groups without a fallback have records
identical to what the oracle wrote before this module existed.
"""

from __future__ import annotations

import threading

import torch

_TL = threading.local()
_INSTALLED = False
_REAL_SOLVE = torch.linalg.solve      # the production solver; tests replace it to force failures


def solve_with_fallback(A: torch.Tensor, b: torch.Tensor):
    """`torch.linalg.solve(A, b)`; if that raises, the ladder of the module docstring. Appends one info dict per solve to the
    calling thread's log (None when the first attempt worked)."""
    try:
        r = _REAL_SOLVE(A, b)
        _log().append(None)
        return r
    except Exception as e:  # noqa: BLE001 -- torch._C._LinAlgError, or a ROCm solver RuntimeError
        info = {"error": str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__,
                "device": str(A.device), "k": int(A.shape[0])}
    A64, b64 = A.detach().double().cpu(), b.detach().double().cpu()
    dm = float(torch.diagonal(A64).mean())
    stages = [("lu_fp64_cpu", A64)]
    for lam in (1e-4, 1e-2):
        Ar = A64.clone()
        Ar.diagonal().add_(lam * dm)
        stages.append((f"ridge_x{lam:g}", Ar))
    for name, M in stages:
        try:
            r = _REAL_SOLVE(M, b64)
        except Exception:  # noqa: BLE001
            continue
        if bool(torch.isfinite(r).all()):
            info["stage"] = name
            _log().append(info)
            return r.to(device=A.device, dtype=A.dtype)
    r = torch.linalg.lstsq(A64, b64.unsqueeze(1), driver="gelsd").solution.squeeze(1)
    info["stage"] = "lstsq_min_norm"
    _log().append(info)
    return r.to(device=A.device, dtype=A.dtype)


def _log() -> list:
    if not hasattr(_TL, "log"):
        _TL.log = []
    return _TL.log


def install(tq) -> None:
    """Replace `tq.refit_scales` (idempotent). Call BEFORE oracle_q35.guard_linear_algebra() wraps it in the solver lock: the
    swap of torch.linalg.solve below is process-global and only safe while the caller holds that lock (cohort mode) or is
    single-threaded."""
    global _INSTALLED
    if _INSTALLED:
        return
    orig = tq.refit_scales

    def refit_scales(*a, **k):
        _log().clear()
        saved = torch.linalg.solve
        torch.linalg.solve = solve_with_fallback
        try:
            Q, su, sv, rec = orig(*a, **k)
        finally:
            torch.linalg.solve = saved
        infos = list(_log())
        _log().clear()
        for i, info in enumerate(infos):
            if info is not None and i < len(rec):
                rec[i]["fallback"] = info
        return Q, su, sv, rec

    refit_scales.__wrapped__ = orig
    tq.refit_scales = refit_scales
    _INSTALLED = True


def count_fallbacks(record: dict) -> int:
    """Number of refit rounds of a group record that went down the ladder."""
    return sum(1 for r in record.get("refit", []) if "fallback" in r)
