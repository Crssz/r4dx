"""tools/reference/rotation_oracle.py -- rotated trellis: the oracle quantizes the FOLDED weights against the
ROTATED Hessians (docs/gemma4-plan.md 9.7 / 4.4; the converter side is `r4dx-convert --trellis-from` with
`--rotate q2a|q2ab`, tests/convert/test_gemma_trellis.cpp).

How the two sides agree on Q (the decision, documented in docs/gemma4-plan.md section 10):

  1. `r4dx-convert --input <ckpt> --rotate q2ab [--rotation-seed S] --rotation-out rot.safetensors` writes the
     exact fp32 tensors every fold of the conversion uses (rotation.signs, rotation.mix, rotation.had_*_signs)
     plus __metadata__ {rotation, fingerprint, config_sha256}. The oracle never re-derives Q from the seed.
  2. `trellis_quant.py quantize-model --arch gemma4_unified --rotation rot.safetensors ...` loads it
     (RotationSpec), and per tap group
        weights   W' = fold(W): an in-projection (q/k/v, gate/up): (W diag(w_norm)) Q, norm weight PLAIN
                  (offset 0) from input_layernorm / pre_feedforward_layernorm; an out-projection (o, down)
                  under q2ab: W Hb (Hadamard only: Gemma's option A keeps the output in the original basis,
                  the post-norm sits between the projection and the residual add); under q2a: unchanged
        Hessian   H' = Q^T H_rms Q for in-projections, from the weightless rms tap
                  (hessian_capture.py --rms-taps, hessian.json rms_keys) -- or, without it, from the post-norm
                  H as Q^T D^-1 H D^-1 Q (refused where |w| < 1e-3); Hb^T H Hb for o / down under q2ab
     and quantizes (W', H') exactly as it does an unrotated pair (the trellis' own RHT is applied on top).
  3. The manifest records "rotation": {kind, seed, tensors_sha256, hidden, block} (the file's fingerprint), and
     r4dx-convert refuses a --trellis-from manifest whose fingerprint is not its own run's, then checks every
     reconstruction against fold(W) (not W). lm_head, embeddings and the norms are never folded.

The fold of a row is the converter's: x -> x*signs -> per-block normalized Sylvester Hadamard -> the nblk x nblk
mix across blocks (rotation.hpp ResidualRotation::Apply); a Hessian transform is applied row-wise then
column-wise with the same operator (H' = apply(apply(H)^T)^T), so no dense Q is ever built (hidden 3840).
All math in float64; results rounded to fp32 once.

torch is imported lazily by the functions that need it; the file reader needs only numpy.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

BLOB_MAGIC = "r4dx-rotation-v1\n"
MIN_NORM_ABS = 1e-3     # the converter's CheckNormInvertible bound (rotation.hpp)


def read_safetensors_f32(path: Path) -> tuple[dict, dict]:
    """(tensors name -> np.float32 array with its stored shape, __metadata__) of an F32-only safetensors
    file; plain struct/json/numpy, no safetensors package needed."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    meta = header.pop("__metadata__", {})
    data = memoryview(raw)[8 + n:]
    out = {}
    for name, e in header.items():
        if e["dtype"] != "F32":
            raise ValueError(f"{path}: {name} is {e['dtype']}, expected F32")
        b, d = e["data_offsets"]
        out[name] = np.frombuffer(data[b:d], dtype="<f4").reshape(e["shape"]).copy()
    return out, meta


def fingerprint_of(tensors: dict, order: list[str]) -> str:
    """sha256 of "r4dx-rotation-v1\\n" + per tensor (name, "\\n", u64le element count, fp32 LE bytes): the
    converter's RotationSource::Fingerprint()."""
    h = hashlib.sha256()
    h.update(BLOB_MAGIC.encode("ascii"))
    for name in order:
        a = np.ascontiguousarray(tensors[name], dtype="<f4").reshape(-1)
        h.update(name.encode("ascii") + b"\n")
        h.update(struct.pack("<Q", a.size))
        h.update(a.tobytes())
    return h.hexdigest()


def hadamard_matrix(n: int) -> np.ndarray:
    """Sylvester (natural order) Hadamard matrix, entries +-1: H[i][j] = (-1)^popcount(i & j)."""
    if n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


class RotationSpec:
    """The converter's rotation (a --rotation-out file): Q (signs, mix) and, under q2ab, the three Hadamard
    sign vectors. `apply`/`apply_had` work on torch tensors (any device), in float64."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.tensors, self.meta = read_safetensors_f32(self.path)
        for key in ("rotation", "fingerprint", "config_sha256"):
            if key not in self.meta:
                raise ValueError(f"{path}: not an r4dx-convert --rotation-out file (no '{key}' metadata)")
        self.rotation = json.loads(self.meta["rotation"])
        self.recorded = json.loads(self.meta["fingerprint"])
        self.kind = self.rotation["kind"]
        self.seed = int(self.rotation["seed"])
        self.hidden = int(self.rotation["hidden"])
        self.block = int(self.rotation["block"])
        self.nblk = int(self.rotation.get("nblk", self.hidden // self.block))
        if self.hidden != self.block * self.nblk:
            raise ValueError(f"{path}: hidden {self.hidden} != block {self.block} x nblk {self.nblk}")
        if self.rotation.get("out_fold") != "had_only":
            raise ValueError(f"{path}: rotation.out_fold is {self.rotation.get('out_fold')!r}: only Gemma's option A "
                             f"(had_only) has a trellis oracle path")
        mix_name = "rotation.mix5" if self.nblk == 5 else "rotation.mix"
        self.order = ["rotation.signs", mix_name]
        if self.kind == "q2ab":
            self.order += ["rotation.had_down_signs", "rotation.had_o_signs"]
            if "rotation.had_o_full_signs" in self.tensors:
                self.order.append("rotation.had_o_full_signs")
        missing = [n for n in self.order if n not in self.tensors]
        if missing:
            raise ValueError(f"{path}: missing tensors {missing}")
        self.signs = self.tensors["rotation.signs"].astype(np.float64)
        self.mix = self.tensors[mix_name].astype(np.float64).reshape(self.nblk, self.nblk)
        if self.signs.size != self.hidden:
            raise ValueError(f"{path}: rotation.signs has {self.signs.size} entries, hidden is {self.hidden}")
        # the file must be what its own metadata says (a truncated / edited file is refused here, and the
        # sha256 written into the oracle's manifest is the one r4dx-convert will recompute)
        self.tensors_sha256 = fingerprint_of(self.tensors, self.order)
        if self.recorded.get("tensors_sha256") != self.tensors_sha256:
            raise ValueError(f"{path}: the tensors hash to {self.tensors_sha256} but the file's fingerprint says "
                             f"{self.recorded.get('tensors_sha256')} (edited or truncated)")
        orth = np.abs(self.mix @ self.mix.T - np.eye(self.nblk)).max()
        if orth > 1e-5 or not np.all(np.abs(np.abs(self.signs) - 1.0) == 0.0):
            raise ValueError(f"{path}: the signs are not +-1 or the mix is not orthogonal (max err {orth:.2e})")
        self._cache: dict = {}

    # -- identity ---------------------------------------------------------------------------------------

    def fingerprint(self) -> dict:
        """What the manifest's "rotation" block must be (r4dx-convert's TrellisOptions::rotation)."""
        return {"kind": self.kind, "seed": self.seed, "tensors_sha256": self.tensors_sha256,
                "hidden": self.hidden, "block": self.block}

    def had(self, site: str):
        """(signs float64, block) of a Hadamard site: 'down', 'o' (sliding) or 'o_full'; None unless q2ab."""
        if self.kind != "q2ab":
            return None
        name = f"rotation.had_{site}_signs"
        if name not in self.tensors:
            raise KeyError(f"{self.path}: no {name} (site {site!r})")
        return self.tensors[name].astype(np.float64), int(self.rotation.get("had", {}).get(site, 0)) or (
            512 if site == "down" else 256)

    # -- the operators (torch float64, any device) -------------------------------------------------------

    def _t(self, key: str, make, device):
        import torch

        k = (key, str(device))
        if k not in self._cache:
            self._cache[k] = make(torch, device)
        return self._cache[k]

    def apply_q(self, x):
        """x [..., hidden] -> x Q (float64)."""
        import torch

        dev = x.device
        signs = self._t("signs", lambda t, d: t.from_numpy(self.signs).to(d), dev)
        mix = self._t("mix", lambda t, d: t.from_numpy(self.mix).to(d), dev)
        hn = self._t(f"h{self.block}", lambda t, d: t.from_numpy(hadamard_matrix(self.block) / np.sqrt(self.block)).to(d), dev)
        lead = x.shape[:-1]
        y = (x.to(torch.float64) * signs).reshape(*lead, self.nblk, self.block) @ hn
        z = torch.einsum("...ci,cb->...bi", y, mix)
        return z.reshape(*lead, self.hidden)

    def apply_had(self, x, site: str):
        """x [..., K] -> x Hb of the site (float64)."""
        import torch

        sg, block = self.had(site)
        dev = x.device
        signs = self._t(f"hs_{site}", lambda t, d: t.from_numpy(sg).to(d), dev)
        hn = self._t(f"h{block}", lambda t, d: t.from_numpy(hadamard_matrix(block) / np.sqrt(block)).to(d), dev)
        lead = x.shape[:-1]
        k = x.shape[-1]
        if k != sg.size:
            raise ValueError(f"site {site}: K {k} != {sg.size}")
        y = (x.to(torch.float64) * signs).reshape(*lead, k // block, block) @ hn
        return y.reshape(*lead, k)

    # -- what the converter folds, per linear ------------------------------------------------------------

    def had_site(self, layer_type: str, module: str):
        """The Hadamard site of an out-projection under q2ab (None otherwise)."""
        if self.kind != "q2ab":
            return None
        if module == "mlp.down_proj":
            return "down"
        if module == "self_attn.o_proj":
            return "o_full" if layer_type == "full_attention" else "o"
        return None

    @staticmethod
    def is_in_proj(module: str) -> bool:
        return module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")

    def fold_weight(self, w, layer_type: str, module: str, norm_w=None, norm_offset: float = 0.0):
        """fold(W) as r4dx-convert applies it, W [n, k] (any dtype/device) -> float32 on the same device."""
        import torch

        wd = w.to(torch.float64)
        if self.is_in_proj(module):
            if norm_w is None:
                raise ValueError(f"{module}: an in-projection fold needs its norm weight")
            wd = self.apply_q(wd * (norm_offset + norm_w.to(device=wd.device, dtype=torch.float64)))
        else:
            site = self.had_site(layer_type, module)
            if site is None:
                return w.to(torch.float32)
            wd = self.apply_had(wd, site)
        return wd.to(torch.float32)

    def transform_hessian(self, h, layer_type: str, module: str, norm_w=None, norm_offset: float = 0.0,
                          is_rms: bool = False):
        """The Hessian of fold(W)'s input: H' = Q^T H_rms Q (`is_rms`: h is the weightless rms tap), or from
        the post-norm tap H = D H_rms D as Q^T D^-1 H D^-1 Q; Hb^T H Hb for an out-projection under q2ab;
        `h` unchanged for a linear that is not folded. float32 in -> float32 out (float64 inside)."""
        import torch

        hd = h.to(torch.float64)
        if self.is_in_proj(module):
            if not is_rms:
                if norm_w is None:
                    raise ValueError(f"{module}: the post-norm Hessian needs the norm weight to divide it out")
                d = norm_offset + norm_w.to(device=hd.device, dtype=torch.float64)
                bad = (d.abs() < MIN_NORM_ABS).nonzero().flatten()
                if bad.numel():
                    raise ValueError(f"{module}: |norm weight[{int(bad[0])}]| = {float(d[bad[0]].abs()):.3g} < "
                                     f"{MIN_NORM_ABS}: the post-norm Hessian cannot be divided by it; capture "
                                     f"the weightless rms taps (hessian_capture.py --rms-taps / --rms-only)")
                hd = hd / d[:, None] / d[None, :]
            out = self.apply_q(self.apply_q(hd).T).T
        else:
            site = self.had_site(layer_type, module)
            if site is None:
                return h
            out = self.apply_had(self.apply_had(hd, site).T, site).T
        out = 0.5 * (out + out.T)  # symmetric to the last bit
        return out.to(torch.float32)
