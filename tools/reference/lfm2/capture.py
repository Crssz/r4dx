"""tools/reference/lfm2/capture.py -- the LFM2 (lfm2_moe) side of hessian_capture.py (docs/lfm2-plan.md 7.3, task L3-1).

    Lfm2CaptureRef(model_dir, device)   the surface hessian_capture.run_capture drives: `embed`, `build_layer`
                                        (ONE decoder layer on the device, streamed from the checkpoint's own keys: the
                                        real checkpoint is PER-EXPERT `experts.{e}.{w1,w3,w2}.weight`; the fused
                                        `experts.{gate_up_proj,down_proj}` style is read too), `capture_layer_forward`
                                        (rope tables + causal mask, one layer), `final_norm_hidden` (embedding_norm)
    ExpertTaps(layer, ...)              replaces a MoE layer's `feed_forward.experts.forward` (same math as HF's eager
                                        loop, bit for bit: tests/test_quant_tiny `layer_equal`) with one that feeds the
                                        per-expert Hessian accumulators with the ROUTED tokens' inputs (mlp_in = the
                                        gate/up input, mlp_mid = silu(gate) * up, the down input) and the layer's POOLED
                                        accumulators (every token entering the experts once; every routed pair's mid)
    expert_shrink(...)                  H_e' = (n_e H_e + n_0 H_pool) / (n_e + n_0) for experts under the rows gate

The calibration corpus is gemma.capture.build_tokens_corpus over gen_samples.py's token-id JSONL (the model's own ids).
torch / transformers are imported lazily: the dry-run of hessian_capture.py --arch lfm2_moe needs only numpy.
"""

from __future__ import annotations

import types
from pathlib import Path

# Rare-expert rule (plan 7.3): an expert linear needs ROWS_PER_K_MIN x K routed rows (the Gemma rule, rows >= 4 K);
# `n_0` of the shrinkage is that gate (so a zero-row expert becomes the pooled Hessian, weight n_0 / n_0).
ROWS_PER_K_MIN = 4


def expert_row_gate(k: int) -> int:
    return ROWS_PER_K_MIN * k


def expert_shrink(acc_e, rows_e: int, h_pool, k: int):
    """(acc_e + n_0 * h_pool) / (rows_e + n_0) for an expert with fewer than n_0 = 4 K routed rows: `acc_e` is the
    expert's fp64 SUM of x x^T (not yet divided), `h_pool` the pooled MEAN Hessian (fp64 or fp32, same device).
    Returns (the fp32 mean Hessian, the effective rows rows_e + n_0). H_e' is H_pool when rows_e == 0."""
    import torch

    n0 = expert_row_gate(k)
    h = (acc_e.to(torch.float64) + float(n0) * h_pool.to(torch.float64)) / float(rows_e + n0)
    return h.to(torch.float32), rows_e + n0


class ExpertTaps:
    """Per-expert Hessian taps of ONE MoE layer.

    accs_in[e]  / accs_mid[e]   HessianAccum (K = hidden / moe_intermediate) or None when that expert has no plan
                                (--keys filter); pool_in / pool_mid may be None. `guard(x, w, out, where)` is the
                                GemmGuard (hessian_capture.guard_linear) or None. `rows[e]` counts the routed rows.
    """

    def __init__(self, experts_module, accs_in: dict, accs_mid: dict, pool_in, pool_mid, guard=None,
                 where: str = "experts"):
        self.mod = experts_module
        self.accs_in, self.accs_mid = accs_in, accs_mid
        self.pool_in, self.pool_mid = pool_in, pool_mid
        self.guard = guard
        self.where = where
        n = experts_module.num_experts
        self.rows = [0] * n
        self._orig = None

    def install(self):
        self._orig = self.mod.__dict__.get("forward")
        self.mod.forward = types.MethodType(self._make_forward(), self.mod)
        return self

    def remove(self):
        if self._orig is None:
            self.mod.__dict__.pop("forward", None)
        else:
            self.mod.forward = self._orig

    def _make_forward(self):
        taps = self

        def forward(module, hidden_states, top_k_index, top_k_weights):
            import torch
            import torch.nn.functional as F

            if taps.pool_in is not None:
                taps.pool_in.update(hidden_states)
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=module.num_experts + 1).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in hit:
                e = int(expert_idx[0])
                if e == module.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[e])
                cur = hidden_states[token_idx]
                out = F.linear(cur, module.gate_up_proj[e])
                if taps.guard is not None:
                    out = taps.guard(cur, module.gate_up_proj[e], out, f"{taps.where}.{e}.gate_up")
                gate, up = out.chunk(2, dim=-1)
                act = module.act_fn(gate) * up
                taps.rows[e] += int(token_idx.numel())
                a = taps.accs_in.get(e)
                if a is not None:
                    a.update(cur)
                a = taps.accs_mid.get(e)
                if a is not None:
                    a.update(act)
                if taps.pool_mid is not None:
                    taps.pool_mid.update(act)
                cur_out = F.linear(act, module.down_proj[e])
                if taps.guard is not None:
                    cur_out = taps.guard(act, module.down_proj[e], cur_out, f"{taps.where}.{e}.down")
                cur_out = cur_out * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, cur_out.to(final.dtype))
            return final

        return forward


class Lfm2CaptureRef:
    """Streaming LFM2 reference behind hessian_capture.run_capture's surface (one decoder layer resident).
    Numerics are HF's eager path in the activation dtype (bf16): bf16 router, bf16 rope tables, eager attention."""

    def __init__(self, model_dir, device, dtype=None, verbose: bool = True):
        import torch
        import transformers.models.lfm2_moe.modeling_lfm2_moe as modeling
        from common import ShardIndex
        from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig

        self.m = modeling
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = torch.bfloat16 if dtype is None else dtype
        self.verbose = verbose
        self.index = ShardIndex.load(self.model_dir)
        if not self.index.weight_map:
            raise RuntimeError(f"no safetensors found under {self.model_dir}")
        self.config = Lfm2MoeConfig.from_pretrained(str(self.model_dir))
        self.config._attn_implementation = "eager"
        try:
            self.config._experts_implementation = "eager"
        except Exception:  # noqa: BLE001 -- an older transformers without the experts dispatch
            pass
        self.text_config = self.config
        self.n_layers = int(self.config.num_hidden_layers)
        nd = int(self.config.num_dense_layers)
        self.layer_types = [f"{t}_{'dense' if i < nd else 'moe'}" for i, t in enumerate(self.config.layer_types)]
        self.vocab_size = int(self.config.vocab_size)
        self.rotary = modeling.Lfm2MoeRotaryEmbedding(self.config)
        self.final_norm = modeling.Lfm2MoeRMSNorm(self.config.hidden_size, eps=self.config.norm_eps)
        self.final_norm.weight.data = self.index.get_tensor("model.embedding_norm.weight").to(self.dtype)
        self.final_norm.to(device)
        self.table = self.index.get_tensor("model.embed_tokens.weight").to(self.dtype)  # host (0.5 GB real)
        self._cache: dict[int, tuple] = {}

    # -- weights ------------------------------------------------------------------------------

    def _layer_tensor(self, i: int, key: str, kind: str):
        """The state-dict entry `key` of layer i from the checkpoint (assembling a fused expert tensor from the
        per-expert keys when the checkpoint has no fused one)."""
        import torch

        idx = self.index
        p = f"model.layers.{i}."
        if key in ("feed_forward.experts.gate_up_proj", "feed_forward.experts.down_proj") and (p + key) not in idx.weight_map:
            n = int(self.config.num_experts)
            if key.endswith("gate_up_proj"):
                return torch.stack([torch.cat([idx.get_tensor(f"{p}feed_forward.experts.{e}.w1.weight"),
                                               idx.get_tensor(f"{p}feed_forward.experts.{e}.w3.weight")], 0)
                                    for e in range(n)])
            return torch.stack([idx.get_tensor(f"{p}feed_forward.experts.{e}.w2.weight") for e in range(n)])
        name = p + key
        return idx.get_tensor(name)

    def build_layer(self, i: int):
        import torch

        with torch.device("meta"):
            layer = self.m.Lfm2MoeDecoderLayer(self.config, i)
        state = {}
        for k, v in layer.state_dict().items():
            t = self._layer_tensor(i, k, self.layer_types[i])
            dt = torch.float32 if k.endswith("expert_bias") else self.dtype
            state[k] = t.to(device=self.device, dtype=dt)
        layer.load_state_dict(state, assign=True)
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)
        return layer

    # -- forward ------------------------------------------------------------------------------

    def embed(self, token_ids):
        """[1, T, hidden] bf16 (no embedding scale in LFM2)."""
        import torch

        ids = torch.as_tensor([int(t) for t in token_ids], dtype=torch.long)
        return self.table[ids].to(self.device).unsqueeze(0)

    def _tables(self, x, T: int):
        import torch

        if T not in self._cache:
            if len(self._cache) >= 4:  # sequences have many different lengths: keep a few tables, not all
                self._cache.clear()
            pos = torch.arange(T, device=self.device).unsqueeze(0)
            pe = self.rotary(x, pos)
            mask = torch.full((T, T), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device).triu(1)
            self._cache[T] = (pos, pe, mask[None, None])
        return self._cache[T]

    def capture_layer_forward(self, layer, i: int, x):
        """One decoder layer exactly as Lfm2MoeModel.forward calls it (no cache): rope tables, the additive causal
        mask for attention layers, no mask for ShortConv layers."""
        T = x.shape[1]
        pos, pe, mask = self._tables(x, T)
        is_attn = self.layer_types[i].startswith("full_attention")
        return layer(x, position_embeddings=pe, attention_mask=mask if is_attn else None, position_ids=pos,
                     past_key_values=None)

    def final_norm_hidden(self, h):
        """The post-embedding_norm hidden state of `h` [1, T, hidden]: the input of the (tied) lm_head."""
        return self.final_norm(h)
