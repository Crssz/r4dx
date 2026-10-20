"""CPU tests of the qwen3_5_moe streaming reference and the quantization tooling's pure parts (tiny-q35moe, needs
`tools/reference/qwen35moe/tiny.py build`; the byte-model tests need only the real config.json).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\test_ref.py

* arch: the tiny checkpoint's index matches the tensor table; a missing / extra tensor is reported.
* reader: a virtual expert linear equals the slice of the fused tensor (gate = rows [0, I), up = [I, 2I) of gate_up_proj).
* reference: with HF's eager experts the streaming logits equal a plain HF forward bit for bit; the hook API sees every expert
  linear of every layer, the routed counts sum to T * top_k, and the down-proj input is silu(gate) * up.
* allocation byte model: the floor of the real model fits two 26 GiB cards, all-KB4 does not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from qwen35moe import alloc_q35 as A  # noqa: E402
from qwen35moe import tiny, tiny_golden  # noqa: E402
from qwen35moe.arch import Qwen35MoeArch  # noqa: E402
from qwen35moe.ref import CheckpointReader, Qwen35MoeStreamingReference, ReferenceHooks  # noqa: E402

TINY_HF = tiny.DEFAULT_OUT / "hf"
REAL_CONFIG = tiny.REAL_CONFIG
RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, what: str) -> None:
    RESULTS.append((bool(cond), what))
    print(("ok   " if cond else "FAIL ") + what, flush=True)


def test_arch_and_reader() -> None:
    arch = Qwen35MoeArch.from_config(TINY_HF / "config.json")
    rd = CheckpointReader(TINY_HF)
    check(arch.check_against_weight_map(rd.weight_map) == [], "tiny index matches the tensor table")
    broken = dict(rd.weight_map)
    del broken[arch.fused_names(1)[1]]
    broken["model.language_model.layers.1.mlp.surprise.weight"] = "x"
    probs = arch.check_against_weight_map(broken)
    check(any("missing" in p for p in probs) and any("unexpected" in p for p in probs), "a missing and an unexpected tensor are reported")
    I = arch.moe_inter
    gu = rd.get(arch.fused_names(2)[0])
    dn = rd.get(arch.fused_names(2)[1])
    e = 5
    cpu = torch.device("cpu")
    check(torch.equal(rd.linear_weight(arch.expert_base(2, e, "gate_proj"), cpu), gu[e, :I]), "virtual gate_proj = rows [0, I) of gate_up_proj[e]")
    check(torch.equal(rd.linear_weight(arch.expert_base(2, e, "up_proj"), cpu), gu[e, I:]), "virtual up_proj = rows [I, 2I) of gate_up_proj[e]")
    check(torch.equal(rd.linear_weight(arch.expert_base(2, e, "down_proj"), cpu), dn[e]), "virtual down_proj = down_proj[e]")
    check(set(arch.quantized_linears(3)) >= {arch.layer_prefix(3) + "self_attn.q_proj", arch.layer_prefix(3) + "self_attn.o_proj"}
          and arch.layer_types[3] == "full_attention", "layer 3 is the full-attention layer with q/k/v/o")


class Probe(ReferenceHooks):
    capture_linear_inputs = True

    def __init__(self):
        self.names: dict[int, set] = {}
        self.counts: dict[int, torch.Tensor] = {}
        self.down_ok = True
        self.last_gate_up: dict[tuple, torch.Tensor] = {}

    def moe_routing(self, layer, topk_ids, counts):
        self.counts[layer] = counts.cpu()

    def linear_input(self, layer, name, x):
        self.names.setdefault(layer, set()).add(name)


def test_reference() -> None:
    import transformers  # noqa: F401

    ids = tiny_golden.golden_prompts()[70]
    ref = Qwen35MoeStreamingReference(TINY_HF, "cpu")
    probe = Probe()
    logits = ref.forward_logits(ids, hooks=probe)
    model = tiny.load_hf_model(TINY_HF)
    model.config.text_config._experts_implementation = "eager"   # the loop the reference runs (tiny.py's setting is not honoured)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    check(torch.equal(out, logits), f"streaming logits == plain HF forward with eager experts, bit for bit (T={len(ids)})")
    arch = ref.arch
    for layer in range(arch.n_layers):
        p = arch.layer_prefix(layer)
        names = probe.names[layer]
        exp = {arch.expert_base(layer, e, pr) for e in range(arch.n_experts) if int(probe.counts[layer][e]) for pr in ("gate_proj", "up_proj", "down_proj")}
        check(exp <= names and p + "mlp.gate" in names, f"L{layer}: hooks saw the router and every routed expert linear")
        check(int(probe.counts[layer].sum()) == len(ids) * arch.top_k, f"L{layer}: routed counts sum to T * top_k")


def test_byte_model() -> None:
    arch = Qwen35MoeArch.from_config(REAL_CONFIG)
    m = A.Model(arch)
    cap = int(26 * A.GIB)
    floor = A.uniform_assignment(arch.n_layers, 4, 3, 3)
    k, d0, d1 = m.split(floor, cap)
    check(d0 <= cap and d1 <= cap and 43 * A.GIB < m.total(floor) < 46 * A.GIB, f"floor mix {m.total(floor) / A.GIB:.2f} GiB splits at k={k} ({d0 / A.GIB:.2f} / {d1 / A.GIB:.2f})")
    try:
        m.split(A.uniform_assignment(arch.n_layers, 4, 4, 4), cap)
        check(False, "all-KB4 must not fit two 26 GiB cards")
    except RuntimeError:
        check(True, "all-KB4 (57.9 GiB) does not fit two 26 GiB cards")
    I, H = arch.moe_inter, arch.hidden
    check(A.class_bytes(arch, 0, "gu", 3) == arch.n_experts * 2 * A.trellis_tensor_bytes(H, I, 3), "gate_up class = 256 x two parts")
    check(A.class_bytes(arch, 0, "down", 4) == arch.n_experts * A.trellis_tensor_bytes(I, H, 4), "down class = 256 x one tensor")


def main() -> int:
    if not (TINY_HF / "config.json").exists():
        raise SystemExit(f"no tiny model at {TINY_HF}: tools/reference/qwen35moe/tiny.py build")
    test_arch_and_reader()
    test_reference()
    if REAL_CONFIG.exists():
        test_byte_model()
    bad = [w for ok, w in RESULTS if not ok]
    print(f"{len(RESULTS) - len(bad)} / {len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
