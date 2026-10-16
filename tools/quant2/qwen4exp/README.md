# qwen4_exp trellis quantization tooling (plan M1-10 .. M1-14)

Takes `Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ` (AWQ int4, 125B, 512 experts) to a trellis container that fits two 26 GiB cards
(`docs/qwen4exp-plan.md` 3, 4.4, 4.11). Everything is per layer, resumable, and every GPU step is one bounded process under
`tools\with_lock.ps1 -Name gpu`.

```
corpus_q4.py          tokens_calib.json       hessian-v2 (410 seq, 834,820 canonical tokens) + SFT slice, ~1M tokens, segments <= 4096
hessian_q4.py         hessian\L{ii}\          per layer: .hess files, hessian.json, routing_counts.json, capture.json, DONE.json
                      (tools\reference\qwen4exp\)   + the next residual boundary work\b{ii+1}.bin
oracle_q4.py          oracle-k3\ oracle-k4\   per layer: L{ii}.{p,e00..}.safetensors + .json chunks, weights_override.json
alloc_q4.py           mix\                    mix manifest (one KB per (layer, class)), allocation.json, allocation_report.md
(tools\reference\qwen4exp\)
pipeline.ps1          the driver              -Stage corpus|hessian|oracle|verify|alloc|convert|all -Layers a..b [-Run]
tiny_e2e.py           the chain on tiny-q4exp
```

Paths are under `E:\models\r4dx\q4exp\` (`-Root`). Interpreters: `D:\venvs\r4dx-qwen4-ref` (corpus, Hessians, allocation) and
`E:\models\r4dx\huihui\venv-rocm10` (oracle: `torch.linalg.cholesky_ex` on the GPU needs MAGMA, which torch 2.9.1 lacks).

## Run it

```powershell
.\tools\quant2\qwen4exp\pipeline.ps1 -Stage all -Layers 0..47            # prints every command, runs nothing
.\tools\quant2\qwen4exp\pipeline.ps1 -Stage all -Layers 0..47 -Run       # layer-major: hessian -> oracle K3 -> oracle K4 -> verify
.\tools\quant2\qwen4exp\pipeline.ps1 -Stage alloc -Run                   # needs both oracle dirs complete
.\tools\quant2\qwen4exp\pipeline.ps1 -Stage convert -Run                 # needs the mix, the sidecar and kvcalib.json (M1-15)
```

`-Stage all` is layer-major on purpose: a layer's Hessians are ~8.4 GiB (gate/up 13 MB + down 3.3 MB per expert, packed fp32), so
`verify` deletes them as soon as both oracles for the layer are verified (sha256 of the chunk files, shapes, and two tensors per
chunk reconstructed from the stored bits against the recorded weight error). `hessian` alone for many layers would keep them all.

A GPU step is `-BudgetSeconds` (default 840) of work, then the process saves its state and exits 75; the driver releases the lock
and takes it again for the next slice, so other agents' short GPU tests interleave. State: `work\L{ii}.state` (14 GB of
accumulators), `work\b{ii}.bin|json` (the residual [T, 10240] bf16 at a layer boundary, crc32 per segment, verified when the next
layer reads it), the oracle's finished chunks (job + code + Hessian-sha pinned). Re-run the same command after any interruption.

Disk per layer step at 1M tokens: two residual boundaries 2 x 20 GB, accumulator state 14 GB, Hessians 8.4 GB, oracle outputs
~1 GB per K; the driver refuses to start below 80 GiB free. The oracle directories end up at ~50 GB (K3) and ~65 GB (K4).

## The pieces

**Corpus** (`corpus_q4.py build`). The hessian-v2 corpus re-tokenized by `hessian_capture.build_corpus` with the canonical Qwen3.5
pre-tokenizer from the Huihui 27B's tokenizer dir (the Swift tokenizer.json has the old regex), unchanged (2048-token windows:
calib 7,428, WikiText 262,144, repo code 98,304, generated 466,944 tokens), then the agent SFT slice: one conversation per
`trial_name` (seeded), rendered with the checkpoint's chat template, NFC, capped at 4096 tokens, packed first-fit with EOS
(248044) between documents into segments <= 4096, until 1,000,000 tokens. Meta (per segment source, sha256s, the draw) next to it.

**Hessians** (`hessian_q4.py capture`). Layer L alone runs over the corpus: `Qwen4ExpStreamingReference._build_layer(L)` on the AWQ
weights dequantized to bf16 (QSA ties `recent`), `ReferenceHooks.linear_input` on every linear. Taps, one Hessian per distinct
input: GDN `in` (qkv, z; the equality of z's input is asserted on every segment), `out`; QSA `in` (q, k, v), `o`; per expert
`gu` (gate and up: ONE Hessian) and `down` (the silu*up activation, 640). Router, shared experts, in_proj_a/b, indexer, hyper-connection
and PLE linears are not quantized and get none. File format = hessian_capture.py's `.hess` (R4DXHES1, packed fp32 upper triangle,
H = X^T X / rows) + `hessian.json` (`keys`: checkpoint base name -> file) that trellis_quant reads.

Accumulation: dense taps fp32 GEMM per <= 2048 rows into fp64 (as hessian_capture); experts fp32 accumulators on the device (fp64
would be 27 GB) fed by one fp32 GEMM per (segment, expert); the stored values agree with an fp64 reference to ~1e-6 (tested).
Expert weights of the layer are cached on the device (5 GiB bf16) across segments. Per expert a seeded reservoir sample of 256
routed rows (`L{ii}.probe.safetensors`, [512, 256, 2560] bf16, 0.67 GiB) is kept for the oracle's output-referred error; it is
deleted with the .hess files by `verify`.

**Fallback rule** (Q16). An expert with routed tokens n_e below the linear's input dimension (2560 gate_up, 640 down) gets a
diagonal Hessian from layer-wide second moments: gate_up `m_j = mean over all T tokens of x_j^2` (the router's input rows), down
`m_j = mean over all P = T * top_k (token, expert) pairs of h_j^2`, `H_fb = diag(m) + 0.01 mean(m) I` (header rows T resp. P).
The oracle then adds trellis' own 0.025 mean(diag) damping. Every expert's n_e, and the lists of fallback experts per class, are in
`routing_counts.json`; the flag and routed count travel into every tensor record (`hessian.fallback`, `hessian.routed`).

**Oracle** (`oracle_q4.py quantize`). `trellis_quant.quantize_group` (the function `trellis_quant.py quantize-model` calls) in
the `matched` Hessian basis (the only one the converter imports), the quantize-model seeds, one shared-Hessian group at a time;
every tensor record carries `proxy` (tr(E^T H E) / tr(W^T H W)), `proxy_abs` (tr(E^T H E)), `energy`, `rel_weight_err`, K, bits, the
Hessian provenance and, for experts, `probe` = the mean squared error of the expert's OUTPUT on its probe rows with only this group
quantized (gate+up from the stored bits with exact down; down with exact gate/up) and the mean squared output. Manifest
`weights_override.json` (format r4dx-weights-override v1, encoding trellis-exl3, keys = HF tensor names, `complete` only when every
linear is present, `config_sha256` of the checkpoint's config.json).

The Viterbi encoder (23.7k tiles/s at KB 3, 26-28k at KB 4 on the R9700, whatever the batch) is ~85% of the time: a layer is 16.7M
tiles per K (32.7k per expert: LDLQ 19.2k + the global-scale search 13.5k). `--cohort 32` (default) runs the 32 experts of a chunk as
threads of one `CohortEncoder` that merges their encoder calls (640-column tensors send 40 tiles per call, the merge 1,280): the
states, scales and records are bit-identical to `--cohort 0` (tested on tiny and on a real chunk) and the chunk is ~18% faster
(62 -> 51 s), at 8 GiB of VRAM instead of 2.

**Allocation** (`alloc_q4.py allocate`). Classes per layer: `p` (attention / GDN projections), `gu`, `down`; ONE KB per class
(plan 4.11). Bytes = what the loader allocates (trellis words + fp32 sign vectors + tickets, `trellis_tensor_bytes`), fixed bytes from
the config, the card split `choose_split` = `Qwen4ExpContainer ChooseSplit` (pinned by tests to uniform KB3 gate_up + KB4 down =
51.24 GiB, k = 24, 25.46 / 25.78 GiB). Start at the floor (p = KB 4, experts KB 3), promote by output-referred loss reduction per
GiB (module docstring of `alloc_q4.py`: expert classes from the probe errors weighted by routing frequency, projections from the
relative proxies times the mixer output energy; `--loss local` is the raw tr(E^T H E) which ranks gate_up ~30-40x above down and
is wrong for that) while both cards stay <= `-CapGib` (26). `allocation_report.md`: bytes per class, per
card, expected loss, baselines, fallback share, the promotion order.

**Convert** (`pipeline.ps1 -Stage convert`): `r4dx-convert --input <AWQ dir> --output ...swift15-flash-next-trellis-mix.r4dx
--trellis-from <mix> --trellis-manifest-sha256 <sha of the mix manifest> --trellis-verify full --ngram-sidecar <sidecar>
--layouts w4a16 --no-bf16 --lm-head w4a16 --w4a16-group-rule "^lm_head$=32" --kv-calib <kvcalib.json> --threads 32`.

## Tests

`tests\reference\test_q4exp_quant.py` (CPU, ~1 min; `--e2e [--e2e-device cuda]` adds `tiny_e2e.py`).
