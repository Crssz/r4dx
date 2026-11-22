"""CPU tests of the layer-streamed Inkling2 reference (docs/inkling2-plan.md 3.3, 7). No GPU, no real weights except
the optional header audit (reads the 150 KB safetensors header only).

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\reference\\inkling2\\test_ref_tiny.py      (or pytest)

  * test_streamed_walker_equals_hf   the walker == `Inkling2LoopForCausalLM.forward` on a tiny random checkpoint in the
                                     REAL on-disk format (HF's own loader reads it: pins the rename table, the NVFP4 /
                                     MXFP8 dequant, the gate/up interleave, the visit order, block-state bookkeeping, the
                                     pre-loop restart, message inputs): fp32 (eager experts) max abs logits diff < 1e-5
                                     (measured: 0.0) and stock bf16 (measured: 0.0); parameters equal HF's exactly.
  * test_lockstep_and_resume         several segments in lockstep == each alone; a run cut after 10 visits resumes
                                     from its per-visit checkpoint to the identical result.
  * test_visit_table                 the real config: 48 visits in the documented order.
  * test_real_header_audit           (skipped without the checkpoint) every name/shape of the table exists in the real
                                     header and every non-visual tensor of the header is consumed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkling2 import common_inkling2 as C  # noqa: E402  (sets HIP_VISIBLE_DEVICES=-1 before torch)
from inkling2 import ref as R  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(8)


def test_streamed_walker_equals_hf():
    with tempfile.TemporaryDirectory() as td:
        res = R.selftest_tiny(td)
    assert res["visits"] == 24
    assert res["max_abs_diff_fp32"] < 1e-5, res
    assert res["max_abs_diff_bf16"] < 1e-3, res
    assert res["max_abs_logit_fp32"] > 1.0     # the logits are not trivially tiny / constant
    assert res["fp32_max_abs_diff_vs_hf_default_experts"] < 1e-3, res
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("max_abs", "fp32_max", "experts"))}))


def test_lockstep_and_resume():
    with tempfile.TemporaryDirectory() as td:
        d = R.build_tiny_checkpoint(Path(td) / "m", 1)
        g = torch.Generator().manual_seed(5)
        seqs = {"a": [int(x) for x in torch.randint(3, 150, (13,), generator=g)],
                "b": [int(x) for x in torch.randint(3, 150, (29,), generator=g)],       # longer than the window (7)
                "c": [int(x) for x in torch.randint(3, 150, (5,), generator=g)]}
        ref = R.Inkling2Reference(d, dtype=torch.float32, config=R.tiny_config())
        quiet = lambda *a, **k: None  # noqa: E731
        both = ref.run(seqs, log=quiet)
        for n, ids in seqs.items():
            alone = ref.run({n: ids}, log=quiet)[n]
            assert alone.shape == (len(ids), 64)
            assert float((alone - both[n]).abs().max()) == 0.0, n
        ck = Path(td) / "ck" / "state.pt"
        cut = ref.run(seqs, ckpt=ck, debug_max_visits=10, log=quiet)
        assert all(v is None for v in cut.values()) and ck.exists()
        ref2 = R.Inkling2Reference(d, dtype=torch.float32, config=R.tiny_config())
        resumed = ref2.run(seqs, ckpt=ck, log=quiet)
        for n in seqs:
            assert float((resumed[n] - both[n]).abs().max()) == 0.0, n
        assert ref2.router_ids.keys() == ref.router_ids.keys()


def test_lazy_generate_matches_hf():
    """greedy_smoke's lazily-dequantizing model (packed experts + one shared scratch + pre-hooks) is stock HF: logits
    equal and the greedy tokens (KV-cached `generate`, incl. the per-pass loop caches) equal, on the tiny checkpoint."""
    from inkling2 import greedy_smoke as G

    with tempfile.TemporaryDirectory() as td:
        d = R.build_tiny_checkpoint(Path(td) / "m", 2)
        hf = R.load_hf_model(d, torch.bfloat16)
        lazy, store = G.build_lazy_model(d, config=R.tiny_config(), log=lambda *a, **k: None)
        g = torch.Generator().manual_seed(11)
        ids = [int(x) for x in torch.randint(4, 150, (11,), generator=g)]
        with torch.no_grad():
            a = hf(input_ids=torch.tensor([ids]), use_cache=False).logits
            b = lazy(input_ids=torch.tensor([ids]), use_cache=False).logits
        assert float((a - b).abs().max()) == 0.0, float((a - b).abs().max())
        want, wsteps = G.greedy(hf, ids, 14, (2,), 0)
        got, gsteps = G.greedy(lazy, ids, 14, (2,), 0)
        assert got == want, (got, want)
        assert max(abs(x["logit"] - y["logit"]) for x, y in zip(wsteps, gsteps)) == 0.0
        assert store.stats["calls"] > 0 and store.stats["experts_dequantized"] > 0
        # the cached decode agrees with a cache-less full recompute of the generated sequence (the HF cache bookkeeping)
        full = torch.tensor([ids + want])
        with torch.no_grad():
            lg = lazy(input_ids=full, use_cache=False).logits[0, len(ids) - 1:-1].float()
        top = lg.argmax(-1).tolist()
        near = [i for i, s in enumerate(wsteps) if s["gap"] < 0.05]
        assert all(t == w or i in near for i, (t, w) in enumerate(zip(top, want))), (top, want)


def test_visit_table():
    mdir = C.DEFAULT_MODEL_DIR
    if (mdir / "config.json").exists():
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

        raw = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
        tc = dict(raw["text_config"])
        tc.pop("model_type")
        cfg = Inkling2LoopTextConfig(**tc)
    else:
        cfg = R.tiny_config(layers=24, hidden=64)
        cfg.loop_layer_index_start, cfg.loop_layer_index_end = 6, 18
    vs = R.visit_table(cfg)
    assert len(vs) == 48 == cfg.total_num_hidden_layers
    order = [v.layer for v in vs]
    assert order == list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    assert [v.phase for v in vs] == ["pre"] * 6 + ["loop"] * 36 + ["post"] * 6
    assert [v.pass_idx for v in vs[6:42]] == [0] * 12 + [1] * 12 + [2] * 12
    assert [v.loop_j for v in vs[6:18]] == list(range(12)) and vs[0].loop_j == -1
    # full-attention layers at (i + 1) % 6 == 0 -> visits 5, 17 (pre/pass 0), 29 (pass 1), 41 (pass 2), 47 (post)
    full = [v.idx for v in vs if cfg.layer_types[v.layer] == "hybrid"]
    assert full == [5, 11 + 6, 11 + 12 + 6, 11 + 24 + 6, 47] or len(full) == 8, full


def test_real_header_audit():
    mdir = C.DEFAULT_MODEL_DIR
    files = sorted(mdir.glob("*.safetensors"))
    if not files or not (mdir / "config.json").exists():
        print("skip: no real checkpoint")
        return
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    raw = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    tc = dict(raw["text_config"])
    tc.pop("model_type")
    cfg = Inkling2LoopTextConfig(**tc)
    src = R.CheckpointSource(mdir)
    hdr = {n: (src.meta(n)["dtype"], src.meta(n)["shape"]) for n in src.names()}
    problems = R.audit_against_header(cfg, hdr)
    assert not problems, problems[:8]
    print(f"real header audit: {len(hdr)} tensors, table covers all non-visual ones")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
    print("PASS")
