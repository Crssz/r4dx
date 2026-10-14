"""Lfm2Reference: the text stack of an LFM2.5-24B-A2B (`lfm2_moe`) checkpoint composed from the HF
`modeling_lfm2_moe` decoder-layer submodules, for the golden / KL / greedy scripts (docs/lfm2-plan.md 6.2).

    ref = Lfm2Reference(model_dir)                       # weights are read layer by layer (streaming)
    h = ref.forward_hidden(token_ids)                    # [T, hidden] after the final norm
    ids = ref.last_router_ids                            # {layer: [T, top_k]} of the last forward

Why a manual layer loop instead of `model.forward`: the same computation (`selftest_tiny` proves it
bit-equal to `Lfm2MoeForCausalLM.from_pretrained` on a tiny random checkpoint written in the REAL on-disk
format), but it streams one layer at a time (the bf16 checkpoint is 48 GB, the fp32 truth would be 96 GB),
batches the FFN of several sequences into one GEMM, records the routed expert ids, and runs the fp32 twin.

Numerics: `dtype=bf16` is stock HF (bf16 weights/activations, bf16 router matmul + sigmoid, the F32
`expert_bias` buffer stays F32, cos/sin rounded to the activation dtype, sdpa or eager attention);
`dtype=float32` is the TRUTH: bf16 weights widened exactly, fp32 activations, eager attention, fp32 rope
tables, fp32 router / sigmoid / top-k, fp32 expert math, fp32 final norm and head.

On disk the experts are per-expert 2-D tensors (`experts.{e}.w1/w2/w3.weight`); the HF module holds
`gate_up_proj [E, 2I, H]` = cat(w1, w3) along dim 1 and `down_proj [E, H, I]`: `layer_state` stacks them.
torch/transformers are imported lazily so `import lfm2.ref` stays cheap for table-only users.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm2.arch import EMBED_NAME, FINAL_NORM_NAME, Lfm2Arch  # noqa: E402
from lfm2.common_lfm2 import LOGPROB_CLAMP, REAL_VOCAB, ShardIndex  # noqa: E402

ATTN_IMPLS = ("eager", "sdpa")


def causal_mask(T: int, dtype, device=None):
    """[1, 1, T, T] additive causal mask (eager attention)."""
    import torch

    allowed = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    m = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(~allowed, torch.finfo(dtype).min)
    return m.view(1, 1, T, T)


def moe_forward(block, x2d):
    """Lfm2MoeSparseMoeBlock.forward on `[N, H]` tokens, also returning the routing (HF body, unrolled)."""
    bias = block.expert_bias if block.use_expert_bias else None
    logits, weights, ids = block.gate(x2d, bias)
    return block.experts(x2d, ids, weights), logits, weights, ids


class Lfm2Reference:
    source_tag = "reference"

    def __init__(self, model_dir, device=None, dtype=None, attn: str = "eager", max_layers: int | None = None,
                 load_table: bool = True, config=None):
        import torch
        import transformers.models.lfm2_moe.modeling_lfm2_moe as modeling
        from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig

        if attn not in ATTN_IMPLS:
            raise ValueError(f"attn {attn!r}: expected one of {ATTN_IMPLS}")
        self.m = modeling
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.bfloat16 if dtype is None else dtype
        self.attn = attn
        self.arch = Lfm2Arch.from_model_dir(self.model_dir)
        self.index = ShardIndex.load(self.model_dir)
        if not self.index.weight_map:
            raise RuntimeError(f"no safetensors found under {self.model_dir}")
        self.config = config or Lfm2MoeConfig.from_pretrained(str(self.model_dir))
        self.config._attn_implementation = attn
        self.config._experts_implementation = "eager"
        self.n_layers = self.arch.n_layers if max_layers is None else max_layers
        self.max_layers = max_layers
        self.vocab_size = self.arch.vocab
        self.rotary = modeling.Lfm2MoeRotaryEmbedding(self.config)
        self.final_norm_w = self.index.get_tensor(FINAL_NORM_NAME).to(self.device, self.dtype)
        self.table = None
        if load_table:
            table = self.index.get_tensor(EMBED_NAME).to(torch.bfloat16)
            if tuple(table.shape) != (self.arch.vocab, self.arch.hidden):
                raise RuntimeError(f"{EMBED_NAME} shape {tuple(table.shape)} != config")
            self.table = table.to(self.device)
        self.last_router_ids: dict[int, "torch.Tensor"] = {}

    # -- weights ----------------------------------------------------------------------------------

    def read_tensors(self, names: list[str], dtype) -> dict:
        """Tensors `names` converted to `dtype` (a private copy each), opening every shard once."""
        import torch
        from safetensors import safe_open

        out = {}
        if self.index.single_file is not None:
            return {n: self.index.get_tensor(n).to(dtype, copy=True) for n in names}
        by_shard: dict[str, list[str]] = {}
        for n in names:
            by_shard.setdefault(self.index.weight_map[n], []).append(n)
        for shard, ns in by_shard.items():
            with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as f:
                for n in ns:
                    out[n] = f.get_tensor(n).to(dtype, copy=True)   # copy inside the `with` (mmap lifetime)
        return out

    def layer_state(self, i: int, dtype=None) -> dict:
        """HF state_dict of decoder layer i (experts stacked), params in `dtype`, expert_bias always F32."""
        import torch

        dtype = self.dtype if dtype is None else dtype
        a = self.arch
        p = a.layer_prefix(i)
        shapes = a.layer_shapes(i, with_experts=False)
        raw = self.read_tensors([p + k for k in shapes], dtype)
        state = {}
        for k in shapes:
            t = raw.pop(p + k)
            if k.endswith("expert_bias"):
                t = t.to(torch.float32)
            elif k == "conv.conv.weight":
                pass  # [H, 1, K] as nn.Conv1d expects
            state[k] = t
        if not a.is_dense(i):
            E, I, H = a.n_experts, a.moe_intermediate, a.hidden
            gu = torch.empty(E, 2 * I, H, dtype=dtype)
            dn = torch.empty(E, H, I, dtype=dtype)
            names = [f"{p}feed_forward.experts.{e}.{w}.weight" for e in range(E) for w in ("w1", "w3", "w2")]
            from safetensors import safe_open

            if self.index.single_file is not None:
                it = ((n, self.index.get_tensor(n)) for n in names)
                for n, t in it:
                    self._place(gu, dn, n, t, I)
            else:
                by_shard: dict[str, list[str]] = {}
                for n in names:
                    by_shard.setdefault(self.index.weight_map[n], []).append(n)
                for shard, ns in by_shard.items():
                    with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as f:
                        for n in ns:
                            self._place(gu, dn, n, f.get_tensor(n), I)
            state["feed_forward.experts.gate_up_proj"] = gu
            state["feed_forward.experts.down_proj"] = dn
        return state

    @staticmethod
    def _place(gu, dn, name: str, t, I: int) -> None:
        parts = name.split(".")
        e, w = int(parts[-3]), parts[-2]
        if w == "w1":
            gu[e, :I] = t.to(gu.dtype)
        elif w == "w3":
            gu[e, I:] = t.to(gu.dtype)
        else:
            dn[e] = t.to(dn.dtype)

    def build_layer(self, i: int, dtype=None):
        import torch

        with torch.device("meta"):
            layer = self.m.Lfm2MoeDecoderLayer(self.config, i)
        state = layer_state_to_device(self.layer_state(i, dtype), self.device)
        missing = set(layer.state_dict()) - set(state)
        extra = set(state) - set(layer.state_dict())
        if missing or extra:
            raise RuntimeError(f"layer {i} state mismatch: missing {sorted(missing)[:4]}, extra {sorted(extra)[:4]}")
        layer.load_state_dict(state, assign=True)
        layer.eval()
        for prm in layer.parameters():
            prm.requires_grad_(False)
        return layer

    # -- forward ----------------------------------------------------------------------------------

    def embed(self, token_ids):
        import torch

        ids = torch.as_tensor([int(t) for t in token_ids], dtype=torch.long)
        return self.table[ids.to(self.table.device)].to(self.device)

    def positions(self, T: int):
        import torch

        return torch.arange(T, device=self.device).unsqueeze(0)

    def rope(self, x, pos):
        return self.rotary(x, pos)

    def mixer(self, layer, i: int, h, pe, mask, pos):
        """h + mixer(operator_norm(h)) for ONE sequence `[1, T, H]`."""
        x = layer.operator_norm(h)
        if layer.is_attention_layer:
            y, _ = layer.self_attn(hidden_states=x, position_embeddings=pe, attention_mask=mask, position_ids=pos,
                                   past_key_values=None)
        else:
            y = layer.conv(hidden_states=x, past_key_values=None, attention_mask=None, seq_idx=None)
        return y + h

    def ffn(self, layer, i: int, h2d):
        """h + feed_forward(ffn_norm(h)) for `[N, H]` tokens (any number of sequences concatenated).
        Returns (out, ids or None)."""
        x = layer.ffn_norm(h2d)
        if self.arch.is_dense(i):
            return h2d + layer.feed_forward(x), None
        y, _, _, ids = moe_forward(layer.feed_forward, x)
        return h2d + y, ids

    def layer_step(self, layer, i: int, hs: list, pes: list, masks: list, poss: list) -> list:
        """One decoder layer over several sequences `[1, T_s, H]` (mixers separately, FFN batched)."""
        import torch

        mids = [self.mixer(layer, i, h, pe, m, p) for h, pe, m, p in zip(hs, pes, masks, poss)]
        sizes = [m.shape[1] for m in mids]
        flat = torch.cat([m[0] for m in mids], dim=0)
        out, ids = self.ffn(layer, i, flat)
        if ids is not None:
            self.last_router_ids[i] = ids.detach().cpu().split(sizes, dim=0)
        return [o.unsqueeze(0) for o in out.split(sizes, dim=0)]

    def forward_hidden(self, token_ids, record=None):
        """Hidden `[T, hidden]` after the final norm, from position 0, no cache. `record(i, h)` sees every
        layer output."""
        import torch

        with torch.no_grad():
            h = self.embed(token_ids).to(self.dtype).unsqueeze(0)
            T = h.shape[1]
            pos = self.positions(T)
            pe = self.rope(h, pos)
            mask = causal_mask(T, self.dtype, self.device) if self.attn == "eager" else None
            for i in range(self.n_layers):
                layer = self.build_layer(i)
                h = self.layer_step(layer, i, [h], [pe], [mask], [pos])[0]
                if record is not None:
                    record(i, h)
                del layer
            return self.final_norm(h[0])

    def final_norm(self, h):
        import torch

        x = h.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.arch.eps)
        return self.final_norm_w * x.to(h.dtype)         # HF: weight * normed.to(input_dtype)

    # -- logits -----------------------------------------------------------------------------------

    def logits_rows(self, hidden, mode: str = "fp32", chunk: int = 32768, real_vocab: int | None = None):
        """fp32 logits `[rows, V]` (no softcap). mode 'fp32': head accumulated in fp32 from the exact bf16
        table; 'hf-bf16': bf16 matmul output (what HF computes), widened. Columns >= real_vocab are -inf."""
        import torch

        out = torch.empty(hidden.shape[0], self.vocab_size, dtype=torch.float32)
        hf = hidden.float() if mode == "fp32" else hidden.to(torch.bfloat16)
        for s in range(0, self.vocab_size, chunk):
            e = min(s + chunk, self.vocab_size)
            w = self.table[s:e]
            out[:, s:e] = (hf @ w.float().T) if mode == "fp32" else torch.nn.functional.linear(hf, w).float()
        rv = self.vocab_size if real_vocab is None else min(real_vocab, self.vocab_size)
        if rv < self.vocab_size:
            out[:, rv:] = float("-inf")
        return out


def layer_state_to_device(state: dict, device):
    return {k: v.to(device) for k, v in state.items()}


def masked_logprobs(logits, real_vocab: int | None):
    """log_softmax over the REAL vocabulary columns only; padding columns get LOGPROB_CLAMP (fp16 dumps of
    the same width V on every side, kl_report.py compares them column by column)."""
    import torch

    V = logits.shape[-1]
    rv = V if real_vocab is None else min(real_vocab, V)
    lp = torch.full_like(logits, LOGPROB_CLAMP)
    lp[..., :rv] = torch.log_softmax(logits[..., :rv], dim=-1).clamp_min(LOGPROB_CLAMP)
    return lp


# --------------------------------------------------------------------------------------------
# Tiny random-config checkpoint in the REAL on-disk format (the CPU smoke path of every script here)
# --------------------------------------------------------------------------------------------


def tiny_config(layers: int = 4, vocab: int = 160, hidden: int = 64, experts: int = 8, top_k: int = 2):
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig

    types = ["conv", "conv", "full_attention", "conv"][:layers] if layers <= 4 else \
        ["full_attention" if i % 4 == 2 else "conv" for i in range(layers)]
    return Lfm2MoeConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, num_dense_layers=2,
        num_experts=experts, num_experts_per_tok=top_k, norm_topk_prob=True, use_expert_bias=True,
        routed_scaling_factor=1.0, norm_eps=1e-5, conv_L_cache=3, conv_bias=False, layer_types=types,
        max_position_embeddings=4096, rope_parameters={"rope_theta": 1e7, "rope_type": "default"},
        bos_token_id=1, pad_token_id=0, eos_token_id=2, tie_word_embeddings=True)


def build_tiny_checkpoint(out_dir: Path | str, seed: int = 0, layers: int = 4, vocab: int = 160) -> Path:
    """Writes a tiny random bf16 lfm2_moe checkpoint in the real on-disk format into `out_dir`: config.json,
    generation_config.json, TWO safetensors shards + index, per-expert `experts.{e}.w1/w2/w3.weight`, F32
    `expert_bias`, no lm_head. Projection weights N(0, 0.1), router N(0, 0.4), norm weights 1 + 0.05 N(0,1),
    expert_bias U(-0.3, 0.3): a dropped weight or the bias changes every output and the routing."""
    import torch
    from safetensors.torch import save_file
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = tiny_config(layers, vocab)
    torch.manual_seed(seed)
    model = Lfm2MoeForCausalLM(cfg)
    gen = torch.Generator().manual_seed(seed + 1)
    sd = {}
    with torch.no_grad():
        for name, p in model.state_dict().items():
            if "lm_head" in name:
                continue
            if name.endswith("expert_bias"):
                t = (torch.rand(p.shape, generator=gen) - 0.5) * 0.6
            elif name.endswith("feed_forward.gate.weight"):
                t = torch.randn(p.shape, generator=gen) * 0.4
            elif name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                t = 1.0 + 0.05 * torch.randn(p.shape, generator=gen)
            else:
                t = torch.randn(p.shape, generator=gen) * 0.1
            sd[name] = t.to(torch.float32 if name.endswith("expert_bias") else torch.bfloat16)
    disk = {}
    I = cfg.moe_intermediate_size
    for name, t in sd.items():
        if name.endswith("experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            for e in range(t.shape[0]):
                disk[f"{base}{e}.w1.weight"] = t[e, :I].clone()
                disk[f"{base}{e}.w3.weight"] = t[e, I:].clone()
        elif name.endswith("experts.down_proj"):
            base = name[: -len("down_proj")]
            for e in range(t.shape[0]):
                disk[f"{base}{e}.w2.weight"] = t[e].clone()
        else:
            disk[name] = t.clone()
    names = sorted(disk)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wm = {}
    for fn, ns in shards.items():
        save_file({n: disk[n] for n in ns}, str(out / fn), metadata={"format": "pt"})
        wm.update({n: fn for n in ns})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    cfg.save_pretrained(str(out))
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 1, "eos_token_id": [2], "pad_token_id": 0, "do_sample": True, "temperature": 0.1,
         "top_k": 50, "repetition_penalty": 1.05}), encoding="utf-8")
    return out


def add_tiny_args(ap) -> None:
    ap.add_argument("--tiny", action="store_true",
                    help="CPU smoke: run on a tiny random-config checkpoint (real on-disk format) built in a temp "
                         "dir; never touches the real weights or a GPU")
    ap.add_argument("--tiny-seed", type=int, default=0)


def selftest_tiny(tmp_dir: Path | str, seed: int = 0) -> dict:
    """CPU check: Lfm2Reference's streaming layer loop == `Lfm2MoeForCausalLM.from_pretrained` (HF loads the
    per-expert on-disk keys itself) on a tiny checkpoint, bit-exact in bf16 (the hf-bf16 head), and the fp32
    twin is close to it. Returns the measured differences; AssertionError on any mismatch."""
    import torch
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM

    d = build_tiny_checkpoint(tmp_dir, seed)
    arch = Lfm2Arch.from_model_dir(d)
    from lfm2.arch import check_against_header

    hdr = {}
    from common import raw_safetensors_header

    for f in sorted(Path(d).glob("*.safetensors")):
        h, _ = raw_safetensors_header(f)
        for k, v in h.items():
            hdr[k] = {"shape": v["shape"], "dtype": v["dtype"]}
    problems = check_against_header(arch, hdr)
    assert not problems, f"arch table vs tiny checkpoint: {problems[:5]}"
    ids = [1] + [int(x) for x in torch.randint(4, arch.vocab, (13,), generator=torch.Generator().manual_seed(7))]
    res = {"layers": arch.n_layers, "layer_types": arch.summary()["layer_types"]}
    for attn in ("eager", "sdpa"):
        hf = Lfm2MoeForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, attn_implementation=attn)
        hf.eval()
        assert hf.model.layers[3].feed_forward.expert_bias.dtype == torch.float32, "HF keeps expert_bias F32"
        with torch.no_grad():
            want = hf(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
        ref = Lfm2Reference(d, torch.device("cpu"), attn=attn)
        h = ref.forward_hidden(ids)
        got = ref.logits_rows(h, "hf-bf16")
        diff = float((got - want).abs().max())
        res[f"max_abs_diff_{attn}"] = diff
        assert diff == 0.0, f"Lfm2Reference ({attn}) differs from HF forward by {diff}"
        if attn == "eager":
            truth = Lfm2Reference(d, torch.device("cpu"), dtype=torch.float32, attn="eager")
            ht = truth.forward_hidden(ids)
            lt = truth.logits_rows(ht, "fp32")
            res["fp32_vs_hfbf16_logits_max_abs"] = float((lt - got).abs().max())
            res["fp32_router_ids_equal_bf16_frac"] = float(sum(
                (truth.last_router_ids[i][0] .sort(-1).values == ref.last_router_ids[i][0].sort(-1).values).all(-1).float().mean()
                for i in truth.last_router_ids) / max(1, len(truth.last_router_ids)))
        del hf
    return res


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(selftest_tiny(td), indent=2))
