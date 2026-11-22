"""Goldens for src/kernels/include/r4dx/kernels/inkling2_ref.hpp (tests/kernels/test_inkling2_ref_cpu.cpp).

Every expected output is produced by the HF transformers Inkling2Loop modules themselves (fp32, CPU), so the
C++ reference is checked against the semantics it claims to implement, not against a second hand port:
  Inkling2LoopBlockAttentionResidual / MessageAttentionResidual  -> attnres (1, 3 and 4 queries)
  Inkling2LoopShortConvolution / DynamicShortConvolution         -> sconv_static / sconv_dynamic
  Inkling2LoopScaledL2Clip / LogScaledL2Clip / UnweightedRMSNorm  -> head ops
  Inkling2LoopRelativePositionBias (linear, norm, proj, gather)  -> rel_logits + the attention bias rule
  eager_attention_forward (position_bias, sinks)                 -> attention (full + sliding window)
  Inkling2LoopMessageGate                                         -> msg gate
  Inkling2LoopTopKRouter                                          -> router
  Inkling2LoopSoftcappedSwiGLU, Inkling2LoopSoftcap, Inkling2LoopRMSNorm

Run with the reference venv (transformers with inkling2_loop, torch CPU):
  D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\inkling2_kern\\gen_ref_goldens.py \\
      --out tests\\kernels\\fixtures\\inkling2\\inkling2_ref_golden.bin

File format (little endian): b"INK2GLD1", u32 record count, then per record: u32 name length, the name (utf-8),
u64 element count, that many float32 values (integer outputs such as router ids are stored as exact floats).
"""
import argparse
import os
import struct
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from transformers.models.inkling2_loop import modeling_inkling2_loop as M  # noqa: E402
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig  # noqa: E402

RECORDS = []


def put(name, t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    RECORDS.append((name, np.ascontiguousarray(a, dtype=np.float32).ravel()))


def cfg(**kw):
    c = Inkling2LoopTextConfig(**kw)
    c._attn_implementation = "eager"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(20261122)
    torch.set_default_dtype(torch.float32)
    g = torch.Generator().manual_seed(1234)

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=g) * s

    with torch.no_grad():
        # ---- attnres: 3 queries over 4 states (3 blocks + partial), H = 64 ------------------------------------
        H, T = 64, 3
        c = cfg(hidden_size=H)
        mod = M.Inkling2LoopBlockAttentionResidual(c, num_queries=3)
        mod.query.copy_(rn(3, H, s=0.7))
        blocks = torch.stack([rn(T, H, s=0.5), rn(T, H, s=3.0), rn(T, H, s=0.05)], dim=1)  # [T, 3, H]
        partial = rn(T, H, s=1.5)
        outs = mod(blocks.unsqueeze(0), partial.unsqueeze(0))
        for i in range(3):
            put(f"attnres3.s{i}", blocks[:, i])
        put("attnres3.s3", partial)
        put("attnres3.q", mod.query)
        put("attnres3.out", torch.stack([o[0] for o in outs], 0))  # [3, T, H]
        # one query, 2 states (the layer 1 shape: [embed block] + partial)
        mod1 = M.Inkling2LoopBlockAttentionResidual(c, num_queries=1)
        mod1.query.copy_(rn(H, s=1.0))
        b1 = rn(T, 1, H, s=2.0)
        p1 = rn(T, H, s=0.3)
        o1 = mod1(b1.unsqueeze(0), p1.unsqueeze(0))[0][0]
        put("attnres1.s0", b1[:, 0])
        put("attnres1.s1", p1)
        put("attnres1.q", mod1.query)
        put("attnres1.out", o1)
        # message attnres: 4 queries of loop layer 1, over 2 blocks + partial
        cm = cfg(hidden_size=H, loop_layer_index_start=6, loop_layer_index_end=12)
        mm = M.Inkling2LoopMessageAttentionResidual(cm)
        mm.query.copy_(rn(6, 4, H, s=0.8))
        bm = rn(T, 2, H, s=1.0)
        pm = rn(T, H, s=4.0)
        om = mm(bm.unsqueeze(0), pm.unsqueeze(0), 1)[0]  # [T, 4, H]
        put("attnres4.s0", bm[:, 0])
        put("attnres4.s1", bm[:, 1])
        put("attnres4.s2", pm)
        put("attnres4.q", mm.query[1])
        put("attnres4.out", om.permute(1, 0, 2))  # [4, T, H]

        # ---- sconv static: C = 8, T = 7, fresh state (cache None) ---------------------------------------------
        C, T = 8, 7
        sc = M.Inkling2LoopShortConvolution(C, 4, layer_idx=0, conv_idx=0).float()
        sc.conv1d.weight.copy_(rn(C, 1, 4, s=0.6))
        x = rn(1, T, C, s=1.3)
        put("sconv.x", x[0])
        put("sconv.w", sc.conv1d.weight[:, 0, :])  # [C, 4], tap 3 = current token
        put("sconv.out", sc(x)[0])

        # ---- sconv dynamic: C = 8 (2 heads x 4), input 6, T = 7 -----------------------------------------------
        C, Z, NH, T = 8, 6, 2, 7
        dc = M.Inkling2LoopDynamicShortConvolution(C, Z, NH, 4, layer_idx=0, conv_idx=0)
        dc.static_weight.copy_(rn(4, C, s=0.5))
        dc.weight_proj.weight.copy_(rn(NH * 4, Z, s=0.4))
        x = rn(1, T, C, s=1.1)
        z = rn(1, T, Z, s=1.0)
        put("dconv.x", x[0])
        put("dconv.static_w", dc.static_weight)  # [4, C], row = lag (0 = current token)
        put("dconv.dyn", torch.nn.functional.linear(z, dc.weight_proj.weight)[0])  # [T, NH * 4]
        put("dconv.out", dc(x, z)[0])

        # ---- head ops ------------------------------------------------------------------------------------------
        NHd, D, T = 3, 8, 6
        clip = M.Inkling2LoopScaledL2Clip(NHd, D, 2.0)
        clip.scale.copy_(torch.rand(NHd, D, generator=g) + 0.5)
        hx = rn(1, T, NHd, D, s=1.0)
        hx[0, :, 1] *= 0.1  # head 1 stays under the threshold (unchanged)
        put("clip.x", hx[0])
        put("clip.scale", clip.scale)
        put("clip.out", clip(hx)[0])
        cl = cfg(hidden_size=16, log_scaling_n_floor=4, log_scaling_alpha=0.1)
        lclip = M.Inkling2LoopLogScaledL2Clip(cl, NHd, D, 2.0)
        lclip.scale.copy_(clip.scale)
        pos = torch.arange(T).unsqueeze(0) + 1  # positions 1..6: (pos+1)/4 = 0.5 .. 1.75 (clamp and log branch)
        put("lclip.out", lclip(hx, pos)[0])
        put("hrms.out", M.Inkling2LoopUnweightedRMSNorm()(hx)[0])

        # ---- relative position bias module: hidden 10, 3 heads, d_rel 4, extent 6, log scaling ---------------
        Hd, T, DR, E = 10, 5, 4, 6
        cr = cfg(hidden_size=Hd, d_rel=DR, rel_l2_clip_threshold=1.5, log_scaling_n_floor=4, log_scaling_alpha=0.1)
        rb = M.Inkling2LoopRelativePositionBias(cr, 3, E, log_scaling=True)
        rb.linear.weight.copy_(rn(3 * DR, Hd, s=0.6))
        rb.norm.scale.copy_(torch.rand(3, DR, generator=g) + 0.5)
        rb.proj.copy_(rn(DR, E, s=1.0))
        hs = rn(1, T, Hd)
        posr = torch.arange(T).unsqueeze(0) + 2  # query positions 2..6
        lin = rb.linear(hs).view(1, T, 3, DR)
        rs = rb.norm(lin, posr)
        put("rel.lin", lin[0])
        put("rel.scale", rb.norm.scale)
        put("rel.rs", rs[0])
        put("rel.proj", rb.proj)
        put("rel.logits", (rs @ rb.proj)[0])  # [T, 3, E]

        # ---- attention: eager_attention_forward with the module's bias rule and per-(query, head) sinks -------
        class Stub(torch.nn.Module):
            def __init__(self, groups):
                super().__init__()
                self.num_key_value_groups = groups

        for case, (window, extent) in {"full": (0, 6), "swa": (4, 4)}.items():
            QH, KVH, D, CTX, QL = 4, 2, 16, 9, 5
            q = rn(1, QH, QL, D, s=1.0)
            k = rn(1, KVH, CTX, D, s=1.0)
            v = rn(1, KVH, CTX, D, s=1.0)
            rel = rn(1, QH, QL, extent, s=1.5)  # [B, heads, q, extent], what (relative_states @ proj).transpose(1, 2) is
            sinks = rn(1, QH, QL, 1, s=1.0)
            qpos = torch.arange(QL) + (CTX - QL)
            kpos = torch.arange(CTX)
            dist = qpos[:, None] - kpos[None, :]
            # Inkling2LoopRelativePositionBias.forward's last three lines, verbatim
            gather_index = dist.clamp(0, extent - 1).expand(*rel.shape[:2], -1, -1)
            pb = rel.gather(-1, gather_index)
            pb = pb.masked_fill((dist < 0) | (dist >= extent), 0.0)
            visible = dist >= 0
            if window > 0:
                visible = visible & (dist < window)
            mask = torch.zeros(QL, CTX).masked_fill(~visible, float("-inf"))[None, None]
            out, _ = M.eager_attention_forward(Stub(QH // KVH), q, k, v, mask, scaling=D ** -0.5,
                                               position_bias=pb, sinks=sinks)
            put(f"attn_{case}.q", q[0].permute(1, 0, 2))  # [QL, QH, D]
            put(f"attn_{case}.k", k[0].permute(1, 0, 2))  # [CTX, KVH, D]
            put(f"attn_{case}.v", v[0].permute(1, 0, 2))
            put(f"attn_{case}.rel", rel[0].permute(1, 0, 2))  # [QL, QH, extent]
            put(f"attn_{case}.sink", sinks[0, :, :, 0].permute(1, 0))  # [QL, QH]
            put(f"attn_{case}.out", out[0])  # [QL, QH, D] (eager returns [B, q, heads, d])
            put(f"attn_{case}.shape", [QL, QH, KVH, D, CTX, window, extent])

        # ---- message gate: hidden 8, rank 4 (all fp32: a module built in fp32 keeps down_proj fp32) ----------
        Hm, R, T = 8, 4, 3
        mg = M.Inkling2LoopMessageGate(cfg(hidden_size=Hm, loop_gate_rank=R))
        mg.down_proj.weight.copy_(rn(R, 2 * Hm, s=0.5))
        mg.forget_gate_proj.weight.copy_(rn(Hm, R, s=0.7))
        mg.message_gate_proj.weight.copy_(rn(Hm, R, s=0.7))
        xm = rn(1, T, Hm, s=2.0)
        mm_ = rn(1, T, Hm, s=0.5)
        nx = M.Inkling2LoopUnweightedRMSNorm()(xm)
        nm = M.Inkling2LoopUnweightedRMSNorm()(mm_)
        put("msg.x", xm[0])
        put("msg.m", mm_[0])
        put("msg.down", mg.down_proj.weight)
        put("msg.forget", mg.forget_gate_proj.weight)
        put("msg.gain", mg.message_gate_proj.weight)
        put("msg.g", mg.down_proj(torch.cat([nx, nm], -1))[0])
        put("msg.out", mg(xm, mm_)[0])

        # ---- router: 8 experts, top 3, hidden 16, route_scale 8 -----------------------------------------------
        Hr, E, K, T = 16, 8, 3, 5
        rt = M.Inkling2LoopTopKRouter(cfg(hidden_size=Hr, n_routed_experts=E, num_experts_per_tok=K, route_scale=8.0))
        rt.weight.copy_(rn(E, Hr, s=0.4))
        rt.e_score_correction_bias.copy_(rn(E, s=0.3))
        xr = rn(T, Hr, s=1.0)
        logits, wts, idx = rt(xr)
        scores = torch.sigmoid(logits) + rt.e_score_correction_bias
        ids_sorted, w_sorted = [], []
        for t in range(T):
            # the spec's slot order: descending score, ties to the lower index (no ties in this draw)
            order = sorted(idx[t].tolist(), key=lambda e: (-float(scores[t, e]), e))
            ids_sorted.append(order)
            w_sorted.append([float(wts[t, idx[t].tolist().index(e)]) for e in order])
        put("router.x", xr)
        put("router.w", rt.weight)
        put("router.bias", rt.e_score_correction_bias)
        put("router.logits", logits)
        put("router.ids", ids_sorted)
        put("router.weights", w_sorted)

        # ---- softcapped SwiGLU, logit softcap, weighted RMSNorm -----------------------------------------------
        sg = M.Inkling2LoopSoftcappedSwiGLU(cfg(hidden_size=16))
        gt = rn(4, 6, s=8.0)
        up = rn(4, 6, s=40.0)
        put("swiglu.g", gt)
        put("swiglu.u", up)
        put("swiglu.out", sg(gt, up))
        xs = rn(20, s=60.0)
        put("softcap.x", xs)
        put("softcap.out", M.Inkling2LoopSoftcap(30.0)(xs))
        rms = M.Inkling2LoopRMSNorm(16, eps=1e-6)
        rms.weight.copy_(rn(16, s=1.0))
        xw = rn(3, 16, s=5.0)
        put("rmsw.x", xw)
        put("rmsw.w", rms.weight)
        put("rmsw.out", rms(xw))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"INK2GLD1")
        f.write(struct.pack("<I", len(RECORDS)))
        for name, a in RECORDS:
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<Q", a.size))
            f.write(a.astype("<f4").tobytes())
    print(f"wrote {len(RECORDS)} records to {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    sys.exit(main())
