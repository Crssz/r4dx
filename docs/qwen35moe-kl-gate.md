# Ornith-1.5-122B trellis container: fp8 KV calibration and the Rung 4 KL gate (2026-10-22)

Container under test: `E:\models\r4dx\ornith-1.5-122b-trellis-mix.r4dx` (placeholder KV descale 1.0) and
`E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx` (calibrated; same bytes otherwise). Method and commands:
`tools/quant2/qwen35moe/README.md` ("fp8 KV calibration and the Rung 4 KL gate"). Raw reports:
`E:\models\r4dx\q35moe\kl\gate_report.{md,json}`, `attrib*.{md,json}`; dumps in the same directory.

## Verdict

**The gate is not met, by a wide margin, and the failure is the weights, not the engine and not the KV cache.**

| pooled over 18,454 held-out rows | KL(bf16 \|\| x) | top-1 vs bf16 | KL(truth \|\| x) | top-1 vs truth |
|---|---:|---:|---:|---:|
| noise: stock HF bf16 (N) | - | - | 0.04603 | 93.47% |
| weight-only: HF bf16 stack, trellis weights reconstructed | 0.12158 | 89.09% | 0.11937 | 89.15% |
| r4dx, placeholder KV (descale 1.0) | 0.12685 | 88.32% | 0.12830 | 88.25% |
| r4dx, calibrated KV | 0.12464 | 88.41% | 0.12361 | 88.20% |
| gate | A: <= 0.02, >= 93% | | B: <= 1.5 N + 0.005 = 0.07404, >= N - 1pt = 92.47% | |

Gate A fails 6x on KL and by 4.6 points on top-1; gate B fails in every group (english 0.0520 vs limit 0.0230, thai 0.0740 vs 0.0303,
code 0.0394 vs 0.0199, ALL 0.1236 vs 0.0740). Restricting to the short material (chat on model-turn tokens, 4,102 rows: KL 0.0428, top-1
92.8% vs bf16; noise 0.0109) it is still 2x over the A bar. The bar was not changed.

* **KV calibration changes nothing measurable**: calibrated minus placeholder KL(truth || x) = -0.0047, 95% block-bootstrap
  (-0.0112 .. +0.0003), top-1 -0.04 pt (`paired_kl.py`). e4m3 is floating point, so a descale of 1.0 vs amax/448 only matters for
  subnormals/saturation; the real amax (K 7.6-21.1, V 4.7-27.4) is far below 448 either way. Ship the calibrated container anyway (it
  is the right thing, costs nothing), but it is not where the loss is.
* **The engine adds ~0.003-0.004 KL on top of the weights** (calibrated r4dx 0.12464 vs weight-only 0.12158 against bf16; 0.12361 vs 0.11937
  against truth): fp8 KV, w4a16 lm_head, f16 activations, kernels. r4dx reproduces what the reconstructed weights say.
* **The trellis mix itself costs ~0.073 KL over the noise floor** (0.11937 - 0.04603), growing with context: the excess over N is 0.047
  (positions 0-128), 0.035 (128-512), 0.044 (512-1024), 0.078 (1024-2048), 0.097 (2048-4096), 0.127 (4096-8192), 0.128 (8192+) for the
  calibrated r4dx; the weight-only run has the same shape (0.053 -> 0.190 vs truth), so it is the weights' error compounding through the
  GDN state / attention over a long context, not an fp8-KV effect. N itself rises with position too (0.014 -> 0.113), i.e. the stock bf16
  stack is already noisy on this model, which is why gate A (vs bf16, 0.02) is the harsher bar.

Groups, calibrated r4dx (rows; KL vs bf16 / vs truth / noise N; top-1 vs bf16): english 2440: 0.0529 / 0.0520 / 0.0120, 90.3%;
thai 1989: 0.0705 / 0.0740 / 0.0168, 88.1%; code 3286: 0.0398 / 0.0394 / 0.0100, 91.5%; tool 479 (reported): 0.0343 / 0.0343 / 0.0125, 94.4%;
long 10,260 (reported): 0.1836 / 0.1814 / 0.0729, 86.8%.

## Where the loss is (weight-only runs, KL against the fp32 truth; N = 0.04603)

Each row replaces only the named tensors by their trellis reconstructions in the HF bf16 stack (`ref_run.py logprobs --mix ...`), all other
weights stock; "excess" = KL - N.

| quantized part | size in the container | KL(truth \|\| x) | excess over N | share of the parts' sum |
|---|---:|---:|---:|---:|
| everything (full mix) | 50.1 GiB trellis | 0.11937 | 0.0733 | - |
| class p, K4: GDN `in_proj_qkv`/`in_proj_z`/`out_proj` (36 layers) | ~1.6 GiB | 0.07864 | 0.0326 | 36% |
| class p, K4: full-attention `q/k/v/o_proj` (12 layers) | ~0.5 GiB | 0.06359 | 0.0176 | 20% |
| (class p together) | 1.92 GiB | 0.08915 | 0.0431 | 48% |
| experts gate/up (K3 in layers 0-17.., K4 later) | 31.7 GiB | 0.07251 | 0.0265 | 29% |
| experts down | 16.5 GiB | 0.06636 | 0.0203 | 23% |
| layers 43-47 only (all classes) | ~6 GiB | 0.05134 | 0.0053 | - |
| layers 0-42 only (all classes) | ~44 GiB | 0.11527 | 0.0692 | - |

(The single-class rows add up to 0.0899, 1.23x the full excess: the errors partly overlap, so read the shares as shares of that sum.)
The GDN and attention rows are subsets of class p (their sum, 0.0502, also exceeds 0.0431).

Suspects, in the order the measurement supports:

1. **Class p (attention + GDN projections, 1.9 GiB, 4% of the bytes) causes about half of the damage** (excess 0.043 of 0.073 for the full
   mix; 0.0899 summed). Per GiB it is ~20x more harmful than the experts (0.022 KL/GiB vs 0.001). The allocation's proxy rated all of class p at
   0.68 of a total expected loss of 13.5 (5%) and already put it at K4; the proxy under-weights it. The 12 full-attention layers alone cost
   0.0176 for ~0.5 GiB.
2. **Experts at K3 vs K4** (gate/up 0.0265, down 0.0203 excess): 25 gate/up layers (0-23, 25) and 18 down layers (0-17) are K3. All-K4
   does not fit the cap (57.9 GiB vs 52 GiB available), so this is the byte budget, not a defect.
3. **Layers 43-47 are not the problem in measurement**: the allocation report gives them 62% of the expected proxy loss (L47 alone 3.2 of
   13.5), but quantizing only those five layers costs 0.0053 excess (7% of the full 0.0733); layers 0-42 carry 0.0692. The proxy ranks the late
   layers far above what the forward pass shows.
4. Not suspects: the KV cache (above), the lm_head (w4a16 g32; engine - weight-only = 0.003-0.004 in total), the Hessian/oracle chain
   (the 14/12288 + 15/12288 diagonal-Hessian fallback experts, 13 no-evidence down tensors and the 26+13 null groups are 0.1-0.2% of the
   tensors; nothing flagged POOR).

What would be tried first (a decision for the owner, none of it started): spend bytes on class p (K5/K6 or a w8a16 / bf16 variant of the
attention and GDN projections; bf16 for all of p is +5.7 GiB over the K4 trellis, to be found by moving experts down a step), and give the
allocation a measured per-class loss instead of the Hessian proxy for p. A re-quantization was not started.

## Reproduction

Paths under `E:\models\r4dx\q35moe`: `kvcalib.json`, `tokens_kvcalib.json`, `kl\tokens_gate.json`, dumps `kl\{hf_bf16,hf_fp32,hf_mixrecon,
hf_mix_*,r4dx_placeholder,r4dx_kv}`, logs `logs\kl-*.log`. Costs on this machine: kvcalib 151k tokens 21 min (2 slices); HF bf16 dump of
19.6k tokens 8 min, fp32 12 min, each weight-only run 9 min; r4dx dump 6.5 min + 90 s load per container; CPU comparison ~4 min per dump.
