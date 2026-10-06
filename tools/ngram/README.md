# tools/ngram: would an n-gram drafter pay?

The question (2026-10-03 research): standalone prompt-lookup gave 1.3-1.8 tok/round against DFlash k=7's
3.4-4.2 on benchmark prompts, and real agent traffic gets DFlash about 2.96 tok/round. A hybrid
(n-gram when a long match exists, DFlash otherwise) might still win on agent traffic, where the model
quotes files, diffs and tool output it has just been shown. This directory measures that on the real
requests instead of guessing. The decision is one question: **does the hybrid make decode faster than
DFlash alone on real traffic?** Rounds are not equally expensive (a DFlash round = an 8-row verify plus
the drafter forward; an n-gram round = a verify of however many rows it proposed, no drafter), so the
verdict is made in estimated tok/s, not tok/round:

- **BUILD** only if the best hybrid is **>= 5% faster** than DFlash alone as logged (`--min-gain`) on
  **>= 20 requests** (`--min-requests`) of **>= 32 generated tokens** (`--gate-min-gen`);
- fewer such requests: **NOT ENOUGH DATA** (the numbers are still printed);
- otherwise **DO NOT BUILD**.

The old 2026-10-03 gates (standalone n-gram at k=7 >= 2.5 tok/round, > 50% of tokens in copy runs of
10+) are still printed as `A` / `B`, as information only: n-gram alone can pass them on a log where DFlash
already accepts more per round, and the hybrid is then slower than DFlash (a real 2-request smoke log:
n-gram k=7 3.88 tok/round, DFlash 6.20, hybrid -16.7% tok/round, -7.1% in time).

## 1. Capture: run oh-my-pi against r4dx-server with logging on

`--request-log-tokens` records every request's whole prompt, the generated tokens and DFlash's accepted
draft tokens per round (docs/server.md "Request log" > "Token capture"). Off by default; it changes no
output. **The file holds your prompts and the model's answers as token ids, which decode back to text:
keep it on the local disk and delete it when the study is done.**

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
New-Item -ItemType Directory -Force E:\models\r4dx\ngram | Out-Null
.\build\win-hip\src\server\r4dx-server.exe `
    --model E:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx --layout trellis `
    --dflash E:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx --dflash-k 7 `
    --host 127.0.0.1 --port 8080 `
    --request-log E:\models\r4dx\ngram\requests.jsonl --request-log-tokens
```

That is the production launch line of docs/usage.md ("Run the OpenAI-compatible server") plus the two
log flags; add whatever else the usual omp setup passes (`--think on`, `--max-ctx`, ...) and use DFlash
k=7 as in the study: the logged acceptance is the baseline the hybrid estimate is compared with. Point
omp's OpenAI-compatible provider at `http://127.0.0.1:8080/v1`. The file is appended to across restarts
and can be followed while the server runs (`Get-Content E:\models\r4dx\ngram\requests.jsonl -Wait -Tail 1`
prints enormous lines; `(Get-Item ...).Length` and `Measure-Object -Line` are kinder). A few days of
agent work is a few hundred MB at most, since a request after the first stores only its new tokens.

## 2. Replay

```powershell
python tools\ngram\sim_ngram.py E:\models\r4dx\ngram\requests.jsonl
python tools\ngram\sim_ngram.py E:\models\r4dx\ngram\requests.jsonl --greedy-only --rows 0 --json E:\models\r4dx\ngram\sim.json
```

Plain `python` (stdlib only; numpy is not used). Options: `--min-n/--max-n` (suffix lengths, default
2..5), `--k` (draft lengths, default `7,9`; 9 is a what-if beyond the 8-row window), `--hybrid-t`
(match-length thresholds for the hybrid, default `3,4,5`), `--dflash-tpr` (tok/round for requests
whose log has no DFlash rounds, default 2.96), `--window N` (only match sources within the last N
tokens), `--min-gen` (skip tiny generations), `--greedy-only`, `--rows` (per-request rows printed:
30 by default, 0 none, -1 all). The timing model and the gate: `--verify-ms-base 26.4`,
`--verify-ms-per-row 0.8`, `--drafter-ms 7.0`, `--inject-ms-per-round 0.875`, `--min-gain 0.05`,
`--min-requests 20`, `--gate-min-gen 32`.

Timing model (TP1, Huihui 27B trellis + w4a16 DFlash2 drafter; the report prints the values it used).
`verify(rows) = base + per_row * rows`; plain decode is 1 row = 27.2 ms (docs/perf.md top table: decode
step 27.28 ms on w4a16, 26.93 on trellis mix4.5m; plain 36.69 tok/s = 27.3 ms), a k=7 DFlash round is
`verify(8 rows) + drafter` = 26.4 + 6.4 + 7.0 = **39.8 ms** (docs/dflash2.md k sweep: 39.8 ms/round at
k=7, 0.8 ms per extra draft row; docs/perf.md 8-row verify 32.19 / 31.59 ms), an n-gram round is
`verify(proposal + 1 rows)` plus, in the hybrid, one drafter injection (InjectFeatures, 0.875 ms wall at
64 rows; it is weight-read bound, so 1-8 rows cost about the same, hence per round and not per row).
DFlash alone is priced at its logged round count and the log's own `draft_k` (an `--mtp` log has no
DFlash rounds); requests without logged DFlash rounds use `--dflash-tpr` at the k=7 round cost.
Measure your own setup (TP2, a different drafter, long contexts all move these) and pass the numbers:
the verdict depends on the ratio of n-gram round cost to DFlash round cost.

What it does, per request: rebuilds the prompt from `prompt_shared` + `prompt_ids`, then walks the
generation round by round. A round proposes the tokens that followed the **most recent** earlier
occurrence of the **longest** suffix of the context (prompt + tokens generated so far, 2..5 tokens),
up to k of them; the verify accepts the longest prefix equal to what was actually generated next, plus
one bonus token. No match = a plain 1-token step.

Reported:

| number | meaning |
|---|---|
| tok/round | generated tokens / rounds (the engine's own `tok/round` convention) for the log's DFlash, for n-gram alone at each k, and for the hybrid |
| round hit | rounds whose proposal had at least one accepted token |
| token hit, run >= 10 | share of tokens after the first inside a copy run (the match's continuation keeps equalling the output), and inside runs of 10+ tokens; plus the run-length histogram |
| hybrid | per round: n-gram if its match is >= T tokens, else a DFlash round. A DFlash round emits the logged round that starts at this position (`positional`, exact while the hybrid is lined up with the log; after an n-gram stretch it emits the rest of the logged round that covers the position, so the hybrid keeps the log's local difficulty) or the request's mean logged tok/round (`mean`: optimistic when the n-gram takes the stretches where DFlash was great, and charges DFlash rounds better than any it was logged with there; the gate uses the lower of the two) |
| estimated tok/s | decode tokens / summed round time under the timing model, for DFlash alone, n-gram alone (info) and each hybrid T (positional, mean, and the lower of the two = the conservative gain the gate uses); for the gate set, and split greedy (exact) / sampled (estimate) |
| gate | requests counted, best hybrid gain vs `--min-gain`, the old gates A/B as info, and the verdict line |

Caveats. Greedy requests (temperature 0) are replayed exactly. A sampled request is one sampled
trajectory: the replay is an unbiased but noisy estimate (a deterministic proposal is accepted with
probability p(token)); use `--greedy-only` to see the exact subset. The tok/round columns count an n-gram
round as one round like a DFlash round although it is cheaper; the estimated tok/s are what the verdict
uses. Requests that failed (non-200) are
skipped.

The hybrid still ignores that a hybrid has to keep the drafter's feature ring fed (modelled by the
per-round injection cost only; consecutive n-gram rounds could share one injection, which would make the hybrid a little faster than modelled), that an n-gram verify of fewer rows has a different kernel mix than the
8-row one, and sampled-request acceptance (see above). The 5% margin is there to cover that.

## 3. Without real data

```powershell
python tools\ngram\sim_ngram.py --selftest                 # proposer vs a brute-force definition, and the verdict on synthetic traffic: copy-heavy -> BUILD, DFlash-already-great -> DO NOT BUILD, too few/short requests -> NOT ENOUGH DATA
python tools\ngram\sim_ngram.py tools\ngram\example_requests.jsonl --rows -1   # the 3 synthetic requests, in the server's file format
python tools\ngram\sim_ngram.py --write-example tools\ngram\example_requests.jsonl   # regenerate it
```

`example_requests.jsonl` is made of random token ids (a quoted block, a fresh reply, a sampled reply
with a repeated phrase) and carries only the fields the replay reads, not the server's full line: it
shows the token fields and the `prompt_shared` encoding and runs the report, and says nothing about
real traffic.
