# tools/qwen4exp: gate, server smoke and bench of the qwen4_exp container

Run these on the finished container (`E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx`). Every GPU step goes through
`tools\with_lock.ps1 -Name gpu` (the quantization pipeline takes the same lock per slice). Both cards are visible
(`HIP_VISIBLE_DEVICES` removed); `-Devices a,b` / `-Split k` set `R4DX_Q4_DEVICES` / `R4DX_Q4_SPLIT` (physical 1 = headless, 0 = desktop).

| script | what | GPU lock |
|---|---|---|
| `gate.ps1` | teacher-forced log-probs per segment, then gate A (original) and gate B (noise-relative) | one acquisition **per segment** |
| `smoke.ps1` | `r4dx-server` end to end: health, chat, thinking, SSE, tools, prefix reuse, 20k needle, max-ctx 400, shutdown | one acquisition for the **whole** smoke |
| `bench.ps1` | `r4dx-cli --stats` prefill/TTFT/decode at 256 / 2048 / 8192 / 32768 tokens | one acquisition **per length** |

Build first: `tools\with_lock.ps1 -Name build -Run "cmake --build --preset win-hip --target tool_teacher_forced_logprobs r4dx-server r4dx-cli"`.

## gate.ps1

```
.\tools\qwen4exp\gate.ps1 -Container <c.r4dx> [-Layout trellis] [-Devices 0,1] [-Split k] [-Segments canon|probe8k|probe32k[,...]] [-Out <dir>] [-DryRun]
.\tools\qwen4exp\gate.ps1 -SelfTest          # CPU only
```

* Runs `build\win-hip\tests\model\tool_teacher_forced_logprobs.exe` once per segment (`--segment`, `--max-ctx` 2048 canon / 8192 probe8k / 32768 probe32k;
  tokens: `tools\reference\kl_corpus\tokens_canon.json`, `E:\models\r4dx\q4exp\ref\probes\tokens_probes.json`) into `-Out`
  (default `E:\models\r4dx\q4exp\gate\<container name>`): `<segment>.logprobs.f16`, `.meta.json`, `.tool.log`, `.gate_done.json`.
  **Resumable:** a segment whose marker matches the container (path, size, mtime), layout, devices and split is skipped.
  canon = 4 x 1023 rows (a few minutes to an hour per segment at the tool's per-token Prefill(1)/DecodeStep rate; each segment is its own lock slice);
  probe_32k is 32,767 rows and a 16 GB dump: expect more than one lock slice of time, run it alone.
* Then `tools\reference\qwen4exp\gate_eval.py` (CPU, no lock; venv `D:\venvs\r4dx-qwen4-ref`) writes `gate_report.json` / `gate_report.md` into `-Out` and prints the tables:
  * **Gate A, the original** (plan Q10): KL(bf16 reference `ref\canon` || r4dx), pooled over canon: mean KL <= 0.02 **and** top-1 >= 93 %.
    probe8k (`ref\probes\recent`) and probe32k are reported against the same thresholds, never counted.
  * **Gate B, noise-relative** (REAL_REF.md section 5, `truth_gate.py`'s arithmetic): KL(`truth-fp32` || r4dx) per group english / thai / code / ALL, mean KL <= 1.5 x N + 0.005 and
    top-1 >= N - 1 point, N = stock-bf16 noise (cached in `ref\noise_floor\truth_gate_exact.json`; `gate_eval.py --recompute-noise` recomputes it from the dumps).
    probe8k reported only; probe32k has no truth dump.
  * Per-segment values (both references) are in the report.
* Exit code: 0 = every counted gate PASS, 1 = a gate FAIL, 2 = tool/dump error. Which gate is binding is the user's call (plan Q10 note), both are printed.
* `-SelfTest`: no GPU. `truth-fp32` and the bf16 reference stand in for an r4dx dump and must give truth-vs-truth KL 0, and bf16-vs-truth canon KL 0.01386 / top-1 95.19 % (REAL_REF.md section 1 (a)) with the thresholds 0.02610 / 94.23 %.
* Plumbing mode for the tiny container (vocab 4096, cannot be served through the tokenizer): `-Tokens <tokens.json> -RefDir <reference dump dir>` runs that tokens file instead of `-Segments` and evaluates gate A against that dump only.

Unit test (CPU, synthetic + the real self-test when `E:\models\r4dx\q4exp\ref` exists):
`D:\venvs\r4dx-qwen4-ref\Scripts\python.exe -m pytest tests\reference\test_q4exp_gate.py -q`

## smoke.ps1

```
.\tools\qwen4exp\smoke.ps1 -Container <c.r4dx> [-Port 8080] [-Devices 0,1] [-Split k] [-MaxCtx 32768] [-DryRun]
```

Starts `r4dx-server --model <c> --layout trellis --tokenizer-dir E:\models\Huihui-Qwen3.8-27B-abliterated --host 127.0.0.1 --port <p> --max-ctx 32768` under one gpu-lock acquisition
(the script re-launches itself with `-Inner` inside the lock; model load happens inside it) and checks, printing a PASS/FAIL table (exit code 1 on any FAIL):
/health and /v1/models; chat (France, thinking off); `reasoning_effort` low / medium / xhigh (reasoning_content present, reasoning_tokens > 0, effort changes the rendered prompt) and `none` (absent);
SSE ordering (role chunk, content deltas, one finish chunk, `include_usage` chunk, `[DONE]`; reasoning deltas before content; streamed text == non-streamed text);
tool call round trip (`get_current_weather`, JSON arguments, tool result fed back, final answer); multi-turn prefix reuse read from `timings.prompt_n` vs `usage.prompt_tokens`
(docs/server.md "timings"), then the same conversation cold (after an unrelated request forces a reset) at temperature 0 must give the same text;
a ~20k-token needle at 25 % depth (QSA is active beyond 2051 tokens); a prompt of about twice `--max-ctx` -> HTTP 400 naming `--max-ctx`; back-to-back requests; shutdown by Ctrl-C into the server's console (a forced kill is a FAIL; exit code must be 0).
Server stderr: `%TEMP%\r4dx-q4exp-smoke.err.log`. Load time is not in the 15-minute budget of the checks; `-LoadTimeoutSec` (1500) bounds it.
`-DryRun` prints the lock command, the server command and every request body (response-dependent fields as placeholders) and the check list; nothing is started.

## bench.ps1

```
.\tools\qwen4exp\bench.ps1 -Container <c.r4dx> [-Lengths 256,2048,8192,32768] [-DecodeTokens 128] [-Devices 0,1] [-Out <dir>] [-DryRun]
```

One cold `r4dx-cli` process per length (greedy, `--think off`, `--stats`); the prompt is deterministic filler (~34.6 tokens per line) plus a "write a long essay" instruction.
Prints and writes (`bench.md`, `bench.json`, per-run logs under `-Out`, default `E:\models\r4dx\q4exp\bench\<timestamp>`) a markdown table: prompt tokens as measured, TTFT = prefill seconds,
prefill tok/s, decode tokens and tok/s, the CLI's `VRAM:` figure (one device), per-stage weight GiB from the load log and the per-adapter peak VRAM delta (Windows GPU Adapter Memory counter, adapters by LUID).
No thresholds. The CLI does not print qwen4_exp's per-stage `Vram()` at `--tp 1`; the counters above stand in for it.
