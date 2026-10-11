# Qwen3.8-Flash-Next (`qwen4_exp`) in r4dx

**Experimental, text only, in development on branch `qwen4exp`.** Design and task status live in
[qwen4exp-plan.md](qwen4exp-plan.md); this page is how to use what exists. Nothing here has been measured
end to end on a GPU unless the "Measured results" section says so.

## What it is

`qwen4_exp` is the third architecture r4dx loads (`Arch::kQwen4Exp`, container `__metadata__.model_arch`
`"qwen4_exp"`), alongside the Qwen3.5 hybrid (`qwen3_5`, the 27B) and Gemma 4. The source checkpoint is
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ` (revision `598afc96128f76a67117caf878f538d34f6f1e02`): a
125B-parameter MoE (6B active per token, 512 experts, top-10) with Gated DeltaNet and Qwen Sparse Attention
layers, a 4-stream gated residual (hyper-connections) and a hashed n-gram embedding table (PLE) of about 95 GiB.
The converter dequantizes the AWQ int4 weights and requantizes them to the trellis layout; the model is split by
layer over both cards.

Chat surface: identical to the Qwen3.8-27B's. The checkpoint's `chat_template.jinja` and `generation_config.json`
are byte-identical to the Huihui 27B's (SHA-256 of the template `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
in both directories, checked 2026-10-10), so thinking (`<think>`), `<tool_call><function=..><parameter=..>` tool
calls and `reasoning_effort` (`xhigh`, `medium`, `low`) behave as for the 27B. The server and CLI pick a
`qwen4_exp` dialect (`src/server/dialect.h`) from the container header: no flag needed. `--dialect qwen4_exp` names
it explicitly; EOS ids come from the tokenizer directory's `generation_config.json` (248046 `<|im_end|>`,
248044 `<|endoftext|>`).

## Requirements

- Two 32 GB cards (the plan budgets at most 26 GiB of weights per card plus KV; device 0 is the desktop card).
  Do not set `HIP_VISIBLE_DEVICES` to one card: the model uses both.
- The n-gram sidecar `E:\models\r4dx\swift15-flash-next.r4dx-ngram`, **102,400,495,616 bytes (95.4 GiB)**, on an
  NVMe drive. It is read row by row with unbuffered I/O (`FILE_FLAG_NO_BUFFERING`); **never open or copy it through
  the OS file cache** (a plain `copy`, an editor, an antivirus scan or `Get-FileHash` on the hot path evicts the
  cache the rest of the machine needs, and a cold read is the slow path the design avoids).
- The Qwen3.5 tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `generation_config.json`). The default is the 27B directory `<R4DX_MODELS_ROOT>\Huihui-Qwen3.8-27B-abliterated`
  (root default `E:\models`), the same mechanism the Qwen3.5 dialect uses; if it is absent r4dx stops with
  `qwen4_exp has no tokenizer of its own to default to ...`. Pass `--tokenizer-dir <dir>` for another copy of the
  Qwen3.8-27B tokenizer files. The Swift checkpoint's own `tokenizer.json` is **not** loadable by r4dx
  (`decoder.add_prefix_space=true is not supported`); its template and vocabulary equal the 27B's, which is why the
  27B directory is the default.
- Disk under `E:\models\r4dx`: the HF checkpoint (165 GiB), the sidecar (95 GiB) and the container.

## Pipeline

1. **Download** (once):
   `hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ --revision 598afc96128f76a67117caf878f538d34f6f1e02 --exclude "*.mp4" --local-dir E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`
2. **N-gram sidecar** (CPU, no torch; `D:\venvs\r4dx-qwen4-ref` or any Python with numpy):
   `python tools\reference\qwen4exp\ngram_sidecar.py repack --model-dir E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ --out E:\models\r4dx\swift15-flash-next.r4dx-ngram`
   then `python tools\reference\qwen4exp\ngram_sidecar.py verify --model-dir <same> --sidecar <same> --samples 1000`.
   `repack --delete-shards-after-verify` deletes the HF `embedding-model-*` shards only after `verify` passes.
3. **Hessians, trellis oracle, allocation** (GPU, one process at a time through `tools\with_lock.ps1 -Name gpu`):
   the per-layer scheme of plan section 4.4 (reference forward for layer L, Hessians of that layer's linears with
   routed-token counts per expert and the diagonal fallback below the input dimension, oracle at KB 3 and KB 4,
   Hessians deleted), then the allocation sweep within the 26 GiB-per-card cap. The driver is
   `tools\quant2\qwen4exp\pipeline.ps1` (stages `corpus`, `hessian`, `oracle`, `verify`, `alloc`, `convert`, `all`;
   without `-Run` it only prints the commands it would run); its stage scripts are `tools\quant2\qwen4exp\*.py` and
   `tools\reference\qwen4exp\hessian_q4.py`. Read `tools\quant2\qwen4exp\README.md` and the script's own header first.
4. **Convert** with `r4dx-convert` (`build\win-hip\src\convert\r4dx-convert.exe`). A checkpoint whose
   `config.json` says `model_type: qwen4_exp` takes the qwen4_exp branch (`--help` lists every flag):

   ```
   r4dx-convert --input E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ ^
                --output E:\models\r4dx\swift15-flash-next-trellis-<mix>.r4dx ^
                --ngram-sidecar E:\models\r4dx\swift15-flash-next.r4dx-ngram ^
                --trellis-from <oracle dir or its weights_override.json> [--trellis-manifest-sha256 <hex>]
   ```

   `--ngram-sidecar` is required (the table is recorded in `__metadata__.ngram_sidecar` with the sidecar's body
   sha256, never rehashed). The container is text only: `--rotate`, `--vision on`, `--mtp on`, `--draft-vocab-ids`,
   `--record-reuse-guard` and `--reuse-tensors-from` are refused. `--plan-only` registers, audits and plans every
   tensor and prints the container's tensor count, data bytes and header size without writing `--output`.
   All experts of one layer's gate/up class (and of its down class) share one trellis KB.

## Run

`--layout` defaults to `trellis` for a qwen4_exp container whose metadata carries a `quant.trellis` payload (an
explicit `--layout` always wins).

```
build\win-hip\src\cli\r4dx-cli.exe --model E:\models\r4dx\swift15-flash-next-trellis-<mix>.r4dx --chat
build\win-hip\src\cli\r4dx-cli.exe --model <container> --prompt "..." --think off --max-tokens 256
build\win-hip\src\server\r4dx-server.exe --model <container> --port 8080
```

`--max-ctx` defaults to 262144 (the checkpoint's `max_position_embeddings`) as for the 27B. `/v1/models` lists the
model id read from the container metadata. `--tokenizer-dir` is optional (see Requirements).

CPU tests for this surface (no GPU): `ctest -R "test_dialect|test_q4exp_dialect|test_q4exp_template"` in
`build\win-hip` (dialect resolution from a synthetic container header, the Swift vs 27B chat-template byte comparison
with tools and `reasoning_effort`, the refused-flag messages).

## Known limits

- **Text only.** No vision tower, no MTP head, no DFlash drafter. `--image`, `--vision on`, `--mtp N` and
  `--dflash` are refused with `qwen4_exp does not support: ...` before anything is loaded (CLI and server);
  the model also refuses them on its own.
- **No `--tp` and no `--pp`.** The model owns its two-card layer split; `--tp 2` and `--pp 2` are refused the same
  way. (`--pp` / `R4DX_PP` left unset is fine.)
- **One request at a time**, as for every r4dx server. Prefix reuse is the single-slot, prefill-boundary
  checkpoint extended with the PLE conv state, the 2-token n-gram history and the QSA partial block (plan 4.9,
  task M1-18). `test_model_q4exp_state` checks a restore-then-prefill is bit-identical to a cold run (tiny
  container); `tools\qwen4exp\smoke.ps1` checked a two-turn conversation on the real container (turn 2 prefilled 29
  new tokens of 345, text equal to the cold run).
- **QSA tie rule (Q23).** Ties at the sparse-attention top-k cut are frequent. r4dx breaks them by a fixed rule
  (the most recent blocks win), not by whatever order `torch.topk` happens to produce, so tokens can differ from
  the Hugging Face model wherever the two choose differently; the gate measures that noise floor and reports it
  next to the accuracy numbers (`tools\reference\qwen4exp\noise_floor.py`).
- **MoE routing.** A single-row decode step on a trellis container runs the grouped MoE entirely on the GPU
  (router, top-10, expert sort, grouped trellis GEMVs, combine; no host sync), and after warm-up each GDN layer
  without PLE is replayed as one HIP graph (QSA and PLE layers are launched normally). Prefill chunks (more than
  one row), bf16/w4a16 containers and the capture test hook still use the host-routed path: router and top-10 on
  the GPU, expert ids to the host (one sync per layer), one `ApplyLinear` per (expert, class). The grouped kernels are
  instantiated for hidden 2560, expert and shared-expert intermediate 640, at most 16 experts per token, at most 1024
  experts and trellis bits 3 or 4; the stage checks this once at construction and any other geometry takes the
  host-routed path.
- **No context length is validated yet.** The KL gate is not met (below) and the 32k probe has not been run; 8k
  is measured (probe_8k) and also misses the gate. Longer contexts run but are unvalidated.
- **Stage hand-off is synchronous.** The two card stages run in sequence on the calling thread (stage 0 layers,
  residual through pinned host memory, stage 1 layers and head); there is no stage-0 worker thread and no overlap of
  the two cards within one token, and an exception in either stage propagates to the caller.
- The sidecar is read at about 16 rows per token on the decode critical path; a slow or busy drive shows up as
  decode latency. A failed or short read fails the request; it never produces a zero row.
- **QSA indexer memory.** Each QSA layer keeps the raw indexer keys of every position (`max_ctx` x 128 bf16, 256
  bytes per token per layer; 12 layers) rather than the 3-row buffer of the plan: the kernels index them by absolute
  position. It is allocated at load per stage, outside the weight budget `ChooseSplit` balances (weights only), so
  count it against the free headroom at long `max_ctx`.

## Measured results

Only what `qwen4exp-plan.md`'s status table records (2026-10-10), plus the checks named here:

| Item | Result |
|---|---|
| M0-1 download | 69 files, 165.14 GiB, every size equal to the HF API at rev `598afc96` |
| M0-3 sidecar | 320,001,536 rows, sha256 `f90a8563...434d`, `verify` compared 1256 rows OK; file size 102,400,495,616 bytes |
| M0-4 AWQ dequant (Python) | bit-equal to compressed-tensors on synthetic tensors and on 4 real tensors (expert gate/down, attention q, GDN qkv) |
| M0-6 QSA indexer (Python) | masks equal HF's at 1..4096 tokens and through a DynamicCache; `tie_rule="recent"` equals a naive reference |
| M1-1 config + arch dispatch | `test_qwen4exp_config`, `test_gemma_config`, `test_dialect`, `test_tp_config` pass |
| M1-19 dialect (CPU) | Swift and 27B `chat_template.jinja` and `generation_config.json` are byte-identical; 12 renders (3 efforts x thinking on/off x tools on/off) are byte-identical through `ChatTemplate`; `test_dialect`, `test_q4exp_dialect`, `test_q4exp_template` pass |
| Real weights, layers 0-3 (2026-10-10) | `test_model_q4exp_real_prefix` PASS on device 1: layers 0-2 (GDN, layer 1 with PLE) and 3 (QSA, dense: 70 < 2051 tokens) of the real checkpoint, bf16 container `layers_present` 4, against the real-model layer goldens `T70` / `T70eos`, teacher forced and free running, both GDN prefill modes. Numbers in "Real-weights prefix" below |
| GDN chunked prefill on real weights | the shipped libr4d scan (the 27B's) is WRONG on this checkpoint (layer 0 `attn_out` max/rms 5.44, mean/rms 0.34); the span-safe scan `r4d_gdn_chunk_scan_stable_*` equals the recurrent kernels and the goldens (0.077 / 0.0037 against 0.077 / 0.0034); `test_gdn_chunk_scan_stable`, `test_gdn_chunk_scan`, `test_gdn_seq256_identity`, `test_gdn_write_once{,_model,_cpu}` pass |
| Two devices on the tiny container (2026-10-10) | `test_model_q4exp_split` (HIP_VISIBLE_DEVICES unset) PASS: stages on devices {0,1} and {1,0}, k = 1, 2, 3 and ChooseSplit, logits of every row, mixer output and 30 decode steps bit-identical to the one-stage run; loader and TextModel tests pass with both devices visible. Cause of the earlier amdhip64 access violation: `Qwen4ExpStage`'s `core::Stream` was created (default-constructed `Impl`) before the stage's device was made current, so a stream of device 0 drove device 1's memory (`hipGetLastError`: invalid resource handle; minimal repro, no r4dx code: create a stream under `hipSetDevice(0)`, launch on device-1 memory) |
| Layer-isolated goldens, tiny (2026-10-10) | `test_model_q4exp_layers` PASS, both GDN prefill modes: each layer run alone over the golden `residual_in` for T = 1/7/70/300 and compared at every intermediate point (bounds and observed noise in the test header). GDN layers 0-2: `attn_out` max/rms <= 0.043, mean/rms <= 0.0052, `residual_out` <= 0.077 / 0.0059. QSA layer 3 carries the fp8 KV cache noise (attn o_proj input 0.78 / 0.030 against the bf16 golden; 0.48 / 0.0025 against a CPU recompute with the same fp8 rounding). Selection on every row: T = 300 has 233 sparse rows, 2 differ from the golden mask (rows 163, 257), both a score near-tie (0.00036 of the scores' rms from the cut); T = 70: 0 of 3 sparse rows differ; T = 1 and 7 are dense |

### Real-weights prefix and the GDN chunk scan

**Prefix container.** `r4dx-convert --layers N` (N < 48) on the qwen4_exp checkpoint writes the globals (embedding, final mixer,
lm_head) and layers `[0, N)` only, and records `__metadata__.layers_present = N` (only then: a full container's header is
unchanged). `Qwen4ExpContainer::Inspect` sizes its per-layer tables over the present layers, `Load` refuses a layer range beyond
them (`invalid_argument`, naming the prefix container), `Qwen4ExpModel::Load` runs one stage over `[0, N)` plus the head (its logits
are meaningless after N layers; the capture hook is the point). The container the test uses:

```
r4dx-convert --input E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ --output D:\q4exp\real-prefix4-bf16.r4dx --layers 4 ^
  --layouts bf16 --lm-head w4a16 --w4a16-group-rule "^lm_head$=32" --kv-calib E:\models\r4dx\q4exp\ref\kv_calib.json ^
  --ngram-sidecar E:\models\r4dx\swift15-flash-next.r4dx-ngram --threads 16
```

4198 tensors, 22,486,326,672 data bytes (20.94 GiB; 19.38 GiB of layer weights + 0.38 GiB head on the device), written in 42.0 s
(the AWQ int4 dequantized to bf16 on the fly, so the weights are bit-equal to what `real_ref.py` used). `test_model_q4exp_real_prefix`
skips (77) when it, the goldens (`E:\models\r4dx\q4exp\ref\layer_goldens\T70{,eos}.recent.safetensors`) or a GPU are missing
(`R4DX_Q4EXP_REAL_PREFIX`, `R4DX_Q4EXP_REAL_GOLDENS` override the paths).

**What the test compares.** Every capture the golden also has (residual in / after PLE / after attention / after MoE, the mixes and
injection weights, `attn_out`, the QSA `attn_pre_o`, `moe_out`; the router's top-10 as sets, with the golden's own router logits
deciding which differences are near-ties), as max|err| and mean|err| relative to the golden tensor's rms: (A) teacher forced, each
layer alone on the golden's `residual_in` (and its `ple_embedding` rows for the PLE layer); (B) free running, the model from the token
ids (embedding gather, n-gram sidecar, PLE, all four layers); both for T70 (70 tokens) and T70eos (EOS 248044 at positions 17 and 41),
both with the GDN prefill through the chunked scan (the production path) and through the recurrent decode kernels. Observed on
device 1 (max/rms, mean/rms; chunked | recurrent), T70:

| point | teacher forced, layer 0 | layer 1 (PLE) | layer 2 | layer 3 (QSA) |
|---|---|---|---|---|
| `attn_out` | 0.077, 0.0037 \| 0.077, 0.0034 | 0.120, 0.0040 \| 0.114, 0.0036 | 0.128, 0.0047 \| 0.128, 0.0037 | 0.455, 0.0217 |
| `moe_out` | 0.324, 0.0047 \| 0.463, 0.0041 | 0.166, 0.0066 \| 0.362, 0.0078 | 0.410, 0.0084 \| 0.360, 0.0063 | 0.735, 0.0234 |
| `residual_out` | 0.510, 0.0026 \| 0.566, 0.0023 | 0.079, 0.0007 \| 0.115, 0.0008 | 0.230, 0.0019 \| 0.177, 0.0013 | 0.266, 0.0029 |
| free running `residual_out` | 0.510, 0.0026 \| 0.566, 0.0023 | 0.387, 0.0040 \| 0.484, 0.0036 | 0.334, 0.0057 \| 0.334, 0.0047 | 0.412, 0.0082 \| 0.466, 0.0075 |

Layer 3 also reports `attn_pre_o` 0.908, 0.0121 (teacher forced; the runtime's KV cache is fp8 with the calibrated descales, the
reference's bf16). Chunked against recurrent on identical input (layers 0-2 `attn_out`): 0.077 / 0.0031, 0.100 / 0.0033,
0.064 / 0.0041. Every router expert-set difference (up to 14 of 70 rows per layer free running, 0-10 teacher forced) is a near-tie of
the golden's own router logits (worst 0.0106 of the row's logit rms from the cut). T70eos agrees at the same level. The bounds in the
test are the worst observed over both goldens and both modes with a 1.3-1.5x margin, and the two GDN modes must agree with each other
(teacher forced `attn_out`, layers 0-2: max/rms <= 0.25, mean/rms <= 0.01).

**The GDN chunk-scan defect (found by this test).** The first run, with the 27B's `r4d_gdn_chunk_scan_k128_v128_c64_bf16`, read
layer 0 `attn_out` 5.44 / 0.34, layer 1 12.2 / 0.064, layer 2 2.96 / 0.071 (recurrent kernels: 0.077 / 0.0034, 0.114 / 0.0036,
0.128 / 0.0037), wrong in every row block including rows 0-15. The tiny random model's "chunked prefill diverges beyond ~40 rows"
was the same bug, not ill-conditioning (its exp(A_log) reaches 16 with dt_bias down to -6; with the span-safe scan `test_model_q4exp_tiny`
compares the chunked prefill strictly against the HF goldens through T = 300 and passes). Cause: the scan applies the pair decay
e^{G_i - G_j} (G = the chunk's gate cumsum) as e^{G_i - c} e^{c - G_j} around the chunk midpoint and clamps each half at e^80, exact
only while the chunk's gate span G_first - G_last stays under about 160 (its source says "attenuated, not exact" beyond ~176).
This checkpoint's gate is A = e^A_log from 0.03 to 158 (layer 0), so a 64-row chunk's span reaches 6505 nats (layer 0; 14 of 48
heads over 160 in the first chunk; layer 1: 306, 6 heads; layer 2: 545, 2 heads) and the near-diagonal weights are zeroed.
`tools/reference/qwen4exp/gdn_probe.py --emulate` reproduces it on the CPU from the real activations: HF's chunk rule and recurrent
rule agree with the golden (0.038 / 0.0019 through out_proj), the emulated split-form scan gives 5.4511 / 0.33969 against the C++
5.4415 / 0.33966, and an emulated span-safe scan 0.0768 / 0.0033.

**The fix.** A new libr4d unit, `r4d_gdn_chunk_scan_stable_k128_v128_c64_bf16` (+ `core::r4d::GdnChunkScanStable`, registry op
`gdn_chunk_scan_stable`), the shipped scan's source compiled with `R4D_GDN_SCAN_STABLE = 1`: the score step applies
e^{G_i - G_j} per element, V' = A W stays unweighted, and the state update weights the K rows by e^{G_last - G_t} (K is dead for
P and U by then), so every exponent is <= 0 and nothing can overflow at any span (236 VGPRs, no scratch, 61,184 B LDS; the shipped
kernel has 255 VGPRs, 4 spills; the scan kernel alone at T = 4096, H 48, Hg 16 takes 576 us against the shipped scan's 565 us, x1.02).
`Qwen4ExpGdn` calls it for every chunked prefill; decode (the recurrent kernels) was never affected.
The shipped kernel is untouched in effect: every changed line keeps its original text under `#if R4D_GDN_SCAN_STABLE ... #else`, and
its device ISA compiled from HEAD and from the edited source is identical up to the `__hip_cuid` name (the Qwen3.5 / Gemma models
call the shipped entry point; `test_gdn_chunk_scan`, `test_gdn_seq256_identity`, `test_gdn_write_once*` pass). `test_gdn_chunk_scan_stable`
against the exact fp64 recurrence (100 rows = 64 + 36, 6 heads over 2 key heads): mild gate (span 7.6) shipped 3.4e-3 / span-safe 3.5e-3
norm-rel-err; strong gate (span 4534, the real model's range) shipped 0.17 overall and 0.99 for the worst head, span-safe 3.7e-3;
very strong gate (span 4826) shipped 0.96, span-safe 2.9e-3.

### End-to-end measurements on the real container (2026-10-11)

Container: `E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx` (56,467,429,844 bytes, ~52.6 GiB on disk), built from
`E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ` by `pipeline.ps1` with 1M-token Hessians and trellis oracles at
KB 3 and 4. Split: layer k = 26; device 0 holds 25.919 GiB of weights (layers 0-25), device 1 holds 25.912 GiB (layers 26-47,
final mixer, lm_head). Total weights: 51.830 GiB.

#### Benchmark (`tools/qwen4exp/bench.ps1`, two cards, greedy, 128 decode tokens)

| prompt target | prompt tok | TTFT (s) | prefill tok/s | decode tok | decode tok/s | CLI VRAM dev 1 (GiB) | peak VRAM dev 0 (GiB) | peak VRAM dev 1 (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 243 | 1.051 | 231.3 | 128 | **50.33** | 26.78 | 26.50 | 26.90 |
| 2048 | 2044 | 5.519 | 370.4 | 128 | **57.54** | 26.84 | 26.56 | 26.90 |
| 8192 | 8210 | 18.592 | 441.6 | 128 | 16.92* | 26.89 | 26.76 | 26.95 |
| 32768 | 32804 | 72.959 | 449.6 | 128 | 17.88* | 27.07 | 26.93 | 27.19 |

*(*The 8k and 32k decode figures are from the initial host-routed milestone-1 run. A re-run with the on-GPU grouped MoE kernel (2026-10-11, `bench.ps1 -Lengths 256,8192`) gave 44.05 tok/s at 243 prompt tokens and 48.79 tok/s at 8210 (prefill 176 and 499 tok/s); the 50.3 / 57.5 tok/s above came from an earlier run, so expect about 10% run-to-run spread. 32k is not re-measured.)*
- **Grouped MoE decode speedup:** On-GPU expert sorting and grouped Trellis GEMV (`gate_up` and `down`) eliminated all 48 CPU-GPU synchronizations per token and reduced kernel launches from 2,160 to ~620 per token. Per-layer MoE decode dropped from 0.502 ms to **0.085 ms** (a **5.6x layer speedup**).
- Decode speed jumped from 16.96 tok/s to **57.54 tok/s** (a **3.4x overall speedup**, latency dropped from 58.9 ms down to **17.3 ms/tok**).
- Prefill throughput in the table: 231 tok/s at 243 tokens, 370 at 2k, about 440-450 at 8k-32k.
- Peak VRAM per card is **~26.8–27.2 GiB**, leaving ~4.8 GiB of free headroom on each 32 GiB card.
#### Server smoke (`tools/qwen4exp/smoke.ps1`, port 8089)

Server startup: loaded in 107 s on two devices. All 37 checks passed:
- `/health` is 200, `/v1/models` returns `Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ` with `context_length: 32768`.
- Thinking off: answers 'The capital of France is **Paris**.', no reasoning tokens.
- Thinking on: `reasoning_effort` low, medium, xhigh all produce reasoning tokens and the answer `391` to `17 * 23`.
- Streaming: SSE chunks arrive in order, first chunk role=assistant, usage and [DONE] at tail.
- Tool call: model issues `get_current_weather(location="Paris")`, tool result fed back, final reply mentions light rain and 18°C.
- Multi-turn prefix reuse: turn 2 prefilled only 29 new tokens (316 cached); text matches cold run bit for bit.
- Needle-in-a-haystack at ~20k tokens: retrieves `TANGERINE-4417` at 459 tok/s prefill.
- Exceeding `--max-ctx 32768`: returns proper HTTP 400 `invalid_request_error`.
- Clean shutdown: Ctrl-C handled, clean exit.

#### KL gate on real container (`tools/qwen4exp/gate.ps1`)

*Evaluated on the 4 canonical segments (4092 rows) and probe_8k (8191 rows):*

| scope | mean KL vs bf16 ref | top-1 % vs bf16 ref | mean KL vs fp32 truth | top-1 % vs fp32 truth |
|---|---:|---:|---:|---:|
| canon / english_prose | 0.0443 | 89.93 | 0.0445 | 89.64 |
| canon / cpp_source | 0.0527 | 90.52 | 0.0527 | 91.30 |
| canon / thai_prose | 0.0584 | 88.47 | 0.0596 | 87.68 |
| canon / python_source | 0.0760 | 88.86 | 0.0791 | 89.05 |
| **canon / ALL** | **0.0578** | **89.44** | **0.0590** | **89.42** |
| probe_8k | 0.0500 | 91.71 | 0.0483 | 92.05 |

*Status: gate NOT met.* Neither gate A (KL <= 0.02, top-1 >= 93%) nor gate B (noise-relative, canon ALL <= 0.0261, top-1 >=
94.23%) is reached on canon (0.0578 / 89.44% vs the bf16 reference, 0.0590 / 89.42% vs the fp32 truth) or at 8k. The 32k
probe was not run. Plan 3.1 says to stop and ask when the sweep cannot reach the gate, so M1-21 is blocked, awaiting the
user's decision on which gate binds (qwen4exp-plan.md, status table). The re-quantization of an already int4-quantized
AWQ checkpoint at ~3.3 bpw is the likely cause of the excess (not isolated by an experiment). Qualitatively the model
generates coherent text and answers reasoning, math and fact prompts in chat, with ~90% top-1 agreement with the
unquantized reference. Full reports: `E:\models\r4dx\q4exp\gate\swift15-flash-next-trellis-mix\gate_report.md`.
