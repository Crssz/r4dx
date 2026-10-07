# tools/prefill: long-context evaluation kit (prefill M0)

This kit measures prefill speed and long-context accuracy at 8k, 32k, 64k and 128k tokens. It is
the measurement gate for the prefill work: M1 is lossless attention parallelism (the exact-wide
launch by default, split-KV opt-in), and M2 adds opt-in lossy modes. Everything here is local. Prompts are built from this
repository's own docs and sources with the checkpoint's own `tokenizer.json` and
`chat_template.jinja`, and they are frozen to disk, so every variant is measured on byte-identical
inputs. Nothing is downloaded.

Python: `C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe` (tokenizers, transformers,
numpy). The scripts below call it `$py`. Outputs go to `E:\models\r4dx\prefill-m0\` and are never
committed.

Model: every `-Model` default is the production container, the Huihui abliterated trellis mix4.5m
(`E:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx`, `-Layout trellis`; before 2026-09-29 the
base `qwen38-27b-trellis-mix45m.r4dx`), and the tokenizer is the Huihui checkpoint's
(`common.DEFAULT_MODEL_DIR`; its tokenizer files are byte-identical to the base model's, so the
frozen prompts and token files are unchanged). The M0 dense baselines under `prefill-m0\` (`kl\dense`,
`ttft`, `profile`, the task-set results) were taken on the BASE container: an M2 comparison against
them would compare two models, so re-take the dense KL dump and TTFT on the Huihui container first
(`run_kl.ps1 -OutDir ...\kl\dense-huihui`, `ttft_cli.ps1`). M1's finding -- exact-wide attention is
bit-identical to the dense kernel -- is a kernel property, not a per-model one.

## Files

| File | What it does |
|---|---|
| `pf_common.py` | Loads the text sources (docs prose and repo source files) and the tokenizer and chat template (`tools/reference/common.py`'s `RefTokenizer`, canonical ids). Also holds the length tags and the fit helper. |
| `build_tasks.py` | Builds the RULER-style task set with ground truth. Writes `tasks_<len>.jsonl`, `prompts\ttft_<len>.txt` and `manifest.json` (with sha256s). |
| `run_tasks.py` / `run_tasks.ps1` | Starts `r4dx-server` on one device (or with `-Tp 2`), sends every item and saves the answers and the server's `timings`. Runs resume from `results.jsonl`. |
| `score_tasks.py` | Scores results (exact or contains matching, see its docstring), prints tables of score and prefill speed per task and length, and diffs two runs with `--compare`. |
| `make_kl_tokens.py` | Builds the long-prefix KL token file: 3 kinds (prose, code, recall) × 4 lengths. Each segment is exactly T tokens: a prefix of T-256 tokens plus a 256-token continuation. |
| `run_kl.ps1` | Runs `tool_teacher_forced_logprobs --tail-rows 256`. The prefix goes through one chunked `Prefill`, and only the 256 continuation rows are dumped (fp16, about 121 MiB per segment). |
| `kl_compare.py` | Compares a reference dump with a variant dump: mean, median, p99 and max KL, top-1 and top-5 agreement, and perplexity. Reported per segment, per length and overall, with each side's prefill seconds. |
| `gdn256_check.ps1` | GPU validation of the GDN super-chunk change (docs/trellis-m256.md "GDN sequence ops"), device 1. It runs `test_gdn_seq256_identity`, `test_gdn_layer`, `test_gdn_chunk_scan` and `test_prefill_chunk_identity` (three times: defaults, `R4DX_GDN_CONV=1`, `R4DX_GDN_SLICE=64`; the 64-row side is always the pre-change path). It compares greedy 8k / 32k text hashes across the new path, the new path with the original conv kernel, the old 256 path and the 64-row path, and runs `--profile-prefill` at 8k for new, conv1 and old. Logs go to `build\logs\gdn256\<timestamp>\`. Since 2026-10-07 it pins `R4DX_PREFILL_INT8=0` and `R4DX_PREFILL_SPLITKV=exact` (the f16 identity); `-Defaults` (alias `-Int8`) runs the shipped defaults instead and requires only new / conv1 / old to share a hash. |
| `ttft_cli.ps1` | Measures cold TTFT through `r4dx-cli --prompt-file ... --stats`, one fresh process per run. `-ProfilePrefill` prints the `--profile-prefill` per-op table (the attention-share profile). |
| `warm_delta.py` / `warm_delta.ps1` | Warm-turn cost at depth: a cold 32k/64k prompt, then a ~4k-token user turn appended through the server's prefix reuse (plus the same 4k cold at offset 0). Writes `warm_delta.jsonl` with each request's `timings`. |
| `probe_depth.py` | Per-chunk prefill time vs depth (linears by class, chunked GDN, attention core, other) from the in-model probe timeline: run the CLI with `R4DX_PROFILE_LINEARS=all` and `R4DX_PROBE_TIMELINE=<csv>` (src/model/debug_probe.h; about 1% overhead at 128k), then `probe_depth.py --timeline <csv> --out <json>`. |
| `tests/kernels/tool_attn_prefill_bench` | The prefill attention kernel alone at depth D: today's 64-row call vs bigger q_len and an emulated split-KV partial pass (`--depths`, `--qlens`, `--splits`, `--out`). M1 adds `--splitkv` (the real split-KV entry) and `--exact g,g` (exact-wide geometries, bit-compared against the plain call). |
| `tests/kernels/tool_attn_prefill_precision` | (M1) Plain, exact-wide and split-KV prefill attention against an fp64 CPU reference over the same fp8 cache, with flat and peaky data: rms error, the share correctly rounded, and closer or farther than plain. |

Engine changes made for this kit:
- `r4dx-cli --prompt-file <path>`: the Windows command line caps `--prompt` at 32767 characters, which is about 8k tokens.
- `tool_teacher_forced_logprobs --tail-rows R [--tail-path decode|prefill]`: long-prefix mode. The sidecar records `first_row`, `prefix_tokens` and `prefill_seconds`.
- (M1) `tool_teacher_forced_logprobs --prefix-split-at K`: prefills the prefix as two calls, which shifts every chunk boundary the way a prefix-cache restore does. This gives a rounding-level calibration for a variant's KL (`docs/prefill.md`). Pass it through `run_kl.ps1 -ExtraArgs`.
- (M1) `R4DX_PREFILL_SPLITKV` picks the prompt-prefill attention path:
  - unset, empty, `split` (also `splitkv`, `auto`) (the **default since 2026-10-07**, branch `fast`): split-KV by the split law from 2048 tokens of context (`R4DX_PREFILL_SPLITKV_MIN` moves it), the exact-wide launch below. About 2x at 128k, with rounding-class drift against the dense bits (`docs/prefill.md` "Split threshold").
  - `exact` (the previous default, the kill switch): the exact-wide launch on every prompt-prefill call. Same bits as the pre-M1 prefill, about 1.5x faster at 128k.
  - `0`, `1`, `off` or `dense`: the old single-workgroup dense launch (the pre-M1 prefill: same bits, slower).
  - `N` > 1: N split-KV segments on every prompt-prefill call, capped at 32.
  - Anything else prints a warning and uses the default.
  - Before the default changed (branch `prefill` up to `dd92f6d`), unset meant split-KV with the threshold 8192 and the plain launch below it; from `dd92f6d` to 2026-10-07 it meant exact-wide. Runs recorded as "split-KV (default law)" in `docs/prefill.md`'s M1 results are `=split` today.
- `R4DX_PREFILL_INT8` (the int8 x int8 prefill GEMM, `docs/int8-prefill.md`): unset or `1` is the **default since 2026-10-07**; `0` is the kill switch. **The two defaults make prefill lossy against the old f16 / exact-wide bytes:** an OFF run of a TTFT, KL or task comparison is `$env:R4DX_PREFILL_INT8 = '0'; $env:R4DX_PREFILL_SPLITKV = 'exact'` (the environment reaches these scripts, as for `R4DX_PREFILL_CHUNK`). `gdn256_check.ps1` pins both unless run with `-Defaults`.
- `R4DX_PREFILL_CHUNK` picks the prompt-prefill chunk size. Unset (or `256`) is the **default since 2026-09-30**: 256-row super-chunks (libr4d's M = 256 trellis GEMM for the linears, 64-row sub-slices for the GDN scan and the attention core), bit-identical to the 64-row path, about 1.3x cold TTFT at 8k and 32k. `0` or `64` is the kill switch: today's 64-row chunks and buffers exactly. Anything else warns and uses 64. Every model load prints `prefill chunk: N rows (...)` with the reason when a 256 request could not be honoured. Details, the fallbacks and the measurements: `docs/prefill.md` "The 256-row prefill chunk" and `docs/trellis-m256.md`. It reaches these scripts through the environment (`$env:R4DX_PREFILL_CHUNK = '0'` for an OFF run of a TTFT or KL comparison); the KL dumps and TTFT runs under `E:\models\r4dx\linear\spike` and `E:\models\r4dx\chunk` were made that way.

## Tasks (`build_tasks.py`)

Each prompt is one user turn with thinking off. That is exactly what r4dx-server builds for
`enable_thinking:false` and what r4dx-cli builds by default. The whole templated prompt is fitted to
at most the target length:

| Length | Target | Achieved prompt tokens |
|---|---|---|
| 8k | 8192 | 7930–8191 |
| 32k | 32768 | 32450–32768 |
| 64k | 65536 | 65302–65535 |
| 128k | 131072 | 130884–131072 |

Every item starts with a unique `[item <id>]` line, so the server's prefix cache never reuses
anything and each request is a cold prefill: `timings.prompt_n` equals the prompt length.

| Task | Haystack | Question | Match |
|---|---|---|---|
| `niah_single` | docs prose | the 7-digit number for one key; needle depth swept 5–95% | number present |
| `niah_multikey` | prose + 7 distractor needles | the number for one of 8 keys | number present |
| `niah_multivalue` | prose | all 4 numbers of one key | recall of 4 |
| `niah_multiquery` | prose | the numbers of 4 different keys | recall of 4 |
| `vt` | prose + 5-hop chain + distractor chain | every variable holding a value | recall of 5 names |
| `cwe` | numbered word list; 10 words ≥30× each, the rest ≤ a third as often | the 10 most common words | recall of 10 |
| `code_qa` | real repository files (`===== FILE: path =====` blocks), target file at a swept depth | rotating subtypes: `py_caller` ("which function in F calls X", Python AST), `const` (value of a C++ `constexpr kName` or Python `UPPER = literal`, compared numerically), `def_file` ("which file defines function F") | name / number / path |

Items per task: 8 at 8k and 32k, 4 at 64k, 2 at 128k. That is 56 + 56 + 28 + 14 = 154 items.
`--items N` overrides this. Seeds are fixed (`--seed 20260928`, and each item's seed is a crc32 of
seed, task, length and index).

## Commands

Build the frozen inputs once. This needs no GPU and takes about 3 minutes in total:

```powershell
$py = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe'
& $py tools\prefill\build_tasks.py --lengths 8k,32k,64k,128k      # -> E:\models\r4dx\prefill-m0\tasks
& $py tools\prefill\make_kl_tokens.py --lengths 8k,32k,64k,128k   # -> E:\models\r4dx\prefill-m0\kl\tokens_long.json
```

**TTFT (cold prefill), CLI.** Each run is a fresh process. The first line is the timing sweep; the
second is the attention-share profile:

```powershell
.\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 8k,32k,64k,128k -Runs 2 -OutDir E:\models\r4dx\prefill-m0\ttft\dense-tp1
.\tools\prefill\ttft_cli.ps1 -Device 1 -Lengths 8k,32k,64k,128k -ProfilePrefill -OutDir E:\models\r4dx\prefill-m0\ttft\profile-tp1
.\tools\prefill\ttft_cli.ps1 -Tp 2 -Lengths 8k,32k,64k,128k -OutDir E:\models\r4dx\prefill-m0\ttft\dense-tp2
```

**Accuracy task set, server.** The server is started and stopped by the script:

```powershell
.\tools\prefill\run_tasks.ps1 -Device 1 -Lengths 8k,32k -OutDir E:\models\r4dx\prefill-m0\runs\dense-tp1
.\tools\prefill\run_tasks.ps1 -Device 1 -Lengths 64k,128k -OutDir E:\models\r4dx\prefill-m0\runs\dense-tp1   # same dir: appends
.\tools\prefill\run_tasks.ps1 -Device 1 -Lengths 8k -Limit 1 -OutDir E:\models\r4dx\prefill-m0\runs\smoke      # 1 item per task
& $py tools\prefill\score_tasks.py --results E:\models\r4dx\prefill-m0\runs\dense-tp1\results.jsonl `
    --compare E:\models\r4dx\prefill-m0\runs\variant\results.jsonl
```

- A variant binary is passed with `-Server <path>`.
- Variant flags are passed with `-ServerArgs @('--flag','value')`.
- `-MaxTokens 1 -Task niah_single` turns a run into a pure TTFT sweep through the server.

**KL, dense reference vs variant.** Both runs must use the same tokens file:

```powershell
.\tools\prefill\run_kl.ps1 -Device 1 -OutDir E:\models\r4dx\prefill-m0\kl\dense
.\tools\prefill\run_kl.ps1 -Device 0 -Tool <variant build>\tests\model\tool_teacher_forced_logprobs.exe -OutDir E:\models\r4dx\prefill-m0\kl\variant
& $py tools\prefill\kl_compare.py --ref E:\models\r4dx\prefill-m0\kl\dense --test E:\models\r4dx\prefill-m0\kl\variant `
    --tokens E:\models\r4dx\prefill-m0\kl\tokens_long.json --json E:\models\r4dx\prefill-m0\kl\dense_vs_variant.json
```

Use `-Segment prose_8k,recall_8k` to run a subset (one process per segment). `-Tp 2` runs at TP=2.

## Runtime estimates (TP=1, trellis mix4.5m)

Only 8k is measured: 7.2–7.4 s of prefill (about 1120 tok/s). The larger lengths use the
`pflash-report.md` fit (w4a16 long-context slope, trellis constant): about 42 s at 32k, 110 s at
64k and 300 s at 128k. No trellis 128k run exists yet.

| Measurement | 8k | 32k | 64k | 128k | All |
|---|---|---|---|---|---|
| Task set (items × (prefill + 0.3–2.5 s decode)) | 56 → ~8 min | 56 → ~42 min | 28 → ~55 min | 14 → ~72 min | ~3 h, plus ~10 s server load per invocation |
| KL, 3 kinds per length (prefix + 256 tail rows at ~57 ms per row) | ~1 min | ~2.5 min | ~6 min | ~16 min | ~26 min in one process; 1.45 GB of dumps |
| TTFT, CLI, 1 run | ~16 s wall | ~55 s | ~2 min | ~5.2 min | ~8.5 min per run |

- **TP=2:** short prompts run about 1.3× faster (about 1530 tok/s on the older w4a16 container). On the Huihui trellis container, cold TTFT at TP=2 is 3.9 s at 8k, 19.1 s at 32k and 49.0 s at 64k tokens, 1.33× to 1.44× faster than one card (docs/perf.md, 2026-09-30); 128k at TP=2 is not measured.
- **Memory:** `--max-ctx` is sized automatically (for example 132096 at 128k), which adds about 4.3 GiB of KV and state to 17 GiB of weights and fits on one card.
- **Keeping runs affordable:** for iteration, run 8k and 32k in full, and 64k and 128k with `-Limit 1` (7 items each).

## Smoke results (2026-09-28, HIP device 1, TP=1, main @ d7d57a2 plus this kit)

- **`ttft_cli.ps1`, 8k:** 8145 tokens in 7.20 and 7.23 s (1131 and 1127 tok/s). The answer was correct, and the CLI token count equals the builder's count.
- **`ttft_cli.ps1 -ProfilePrefill`, 8k:** 128 chunks. `attn.core_prefill` is 7.0% of gpu_sum, and GEMMs are 68.9%.
- **`run_tasks.ps1 -Lengths 8k -Limit 1`:** 7 of 7 correct. Median prefill was 1107 tok/s, and server `usage.prompt_tokens` matched the builder for every item.
- **`run_kl.ps1`, `prose_8k` and `recall_8k`:**
  - A dense rerun gives KL exactly 0: the engine is run-to-run deterministic, so any nonzero KL is caused by the variant.
  - Decode tail vs `--tail-path prefill` gives mean KL 0.00048 and p99 0.0034, with 99.8% top-1 agreement. That is the size of decode-path vs prefill-path rounding.
  - `recall_8k` continuation perplexity is 1.16, so the long-range copy works.

## Caveats and gaps

- **The KL reference is the engine's own dense run, not bf16 HF.** An HF reference at 32k–128k is not feasible on this box. The harness measures a variant's drift from today's dense prefill, which is the lossless gate for M1 and the KL gate for M2.
- **Tail feed path.** Tail rows are fed through DecodeStep by default, so a prefill-kernel variant reaches them only through the prefix's KV and GDN state.
  - Use `--tail-path prefill` to push the tail through one-token prefill calls.
  - Always compare runs with the same tail path: the two paths differ by about 5e-4 mean KL on their own.
- **The prose haystack is r4dx's own docs.** It is technical text dense with numbers, not essays. The needles use random word-pair keys and 7-digit values, which do not occur in the docs.
- **cwe vocabulary.** The docs yield 4863 distinct 4–10-letter words. At 64k and 128k the uncommon words repeat more (`ucw_freq_max` in each item's meta), and the common-word frequency is raised to stay at least 3× above them.
- **code_qa at 128k has only 2 items** (`py_caller` and `const`; no `def_file`). Use `--items` to get more.
- **Not built yet** (from the report's M0 list):
  - the prompt-length and cold-vs-warm distribution from real server logs (no request log was found locally);
  - 10–20 real agent transcripts with code-retrieval questions;
  - the 4k-delta-at-32k/64k-offset TTFT measurement. For that, the server's prefix reuse works: send a prompt, then the same prompt plus about 4k tokens, and read `timings.prompt_ms` of the second request.
