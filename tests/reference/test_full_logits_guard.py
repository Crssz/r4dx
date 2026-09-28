"""tests/reference/test_full_logits_guard.py

CPU-only test of full_logits_golden.py's GemmGuard wiring (StreamingReference.gemm_guard_state,
which reuses hessian_capture.install_gemm_guard / guard_linear). Needs NO checkpoint and NO GPU.

    (a) lm_head: a garbage first GEMM output (a 1e30 tile) is replaced by the clean recompute, the
        logits equal a clean run bit for bit, and the state counts one recompute
    (b) without the guard (gemm_guard_state None) the same garbage reaches the logits
    (c) a clean run with the guard is bit-identical to one without and records no event
    (d) install_gemm_guard on a small layer replaces a garbage nn.Linear output the same way

Plain script, no pytest dependency, like test_manifest.py:

    python tests\\reference\\test_full_logits_guard.py

Exits 0 and prints "OK (<n> checks)" on success; 1 and every failed check otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "reference"))
import full_logits_golden as flg  # noqa: E402
import hessian_capture as hc  # noqa: E402

failures: list[str] = []
n_checks = 0


def check(cond: bool, what: str) -> None:
    global n_checks
    n_checks += 1
    if not cond:
        failures.append(what)


class Head(flg.StreamingReference):
    """Just enough of a StreamingReference for logits_fp32: two 3-row lm_head blocks on the CPU."""

    def __init__(self, w: torch.Tensor):
        self.device = torch.device("cpu")
        self.vocab_size = w.shape[0]
        self._w = w

    def lm_head_blocks(self, chunk: int):
        for start in range(0, self.vocab_size, chunk):
            yield start, min(start + chunk, self.vocab_size), self._w[start:start + chunk]


class GarbageOnce:
    """Stands in for F.linear: the first call returns its result with a 1e30 tile, the rest are real."""

    def __init__(self):
        self.real = F.linear
        self.calls = 0

    def __call__(self, x, w, b=None):
        y = self.real(x, w, b)
        self.calls += 1
        if self.calls == 1:
            y = y.clone()
            y[:, :2] = 1e30
        return y


def with_garbage(fn):
    g = GarbageOnce()
    F.linear = g
    try:
        return fn()
    finally:
        F.linear = g.real


torch.manual_seed(0)
hidden = torch.randn(4, 16)
w = torch.randn(6, 16)
clean = Head(w).logits_fp32(hidden, chunk=3)

# (a)
h = Head(w)
h.gemm_guard_state = flg.new_gemm_guard_state()
got = with_garbage(lambda: h.logits_fp32(hidden, chunk=3))
check(torch.equal(got, clean), "(a) guarded logits differ from the clean run")
check(h.gemm_guard_state.gemm_guard["recomputed"] == 1, "(a) expected exactly one recompute")
check(h.gemm_guard_state.gemm_guard["failed"] == 0, "(a) a failure was recorded")
check(h.gemm_guard_state.gemm_guard["events"][0].startswith("lm_head[0:3]"), "(a) event names the block")

# (b)
bad = with_garbage(lambda: Head(w).logits_fp32(hidden, chunk=3))
check(bool((bad.abs() > 1e29).any()), "(b) the unguarded run should carry the garbage tile")

# (c)
h = Head(w)
h.gemm_guard_state = flg.new_gemm_guard_state()
check(torch.equal(h.logits_fp32(hidden, chunk=3), clean), "(c) clean guarded run differs")
check(h.gemm_guard_state.gemm_guard["recomputed"] == 0 and not h.gemm_guard_state.gemm_guard["events"],
      "(c) a clean run recorded an event")

# (d)
lin = torch.nn.Linear(16, 5, bias=False)
layer = torch.nn.Sequential(lin)
want = layer(hidden)
st = flg.new_gemm_guard_state()
hc.install_gemm_guard(layer, st, "Lxx")
got = with_garbage(lambda: layer(hidden))
check(torch.equal(got, want), "(d) the layer's garbage output was not replaced")
check(st.gemm_guard["recomputed"] == 1, "(d) expected one recompute")

if failures:
    print("FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print(f"OK ({n_checks} checks)")
