#requires -Version 5.1
<#
.SYNOPSIS
  The hybrid serving mode's soak (docs/pp-tp2-hybrid.md 9 P4, gate G-H4): tool_tp_soak --pp 2 under the TDR watch.

.DESCRIPTION
  A thin wrapper over tools\tp\soak.ps1 (the TDR-watched tool_tp_soak run, the G8 verdict) that supplies the hybrid's
  arguments: `--pp 2` is added unless given. Everything else you pass goes through unchanged to tool_tp_soak (see its header,
  tests\model\tool_tp_soak.cpp, "HYBRID SOAK"; --json is required). Under --pp 2 the tool defaults to --max-ctx 16384,
  --max-prompt 8192 (about 90 % of the prompts engage the pipeline), --need-gib 20, and adds the "turns" mode (warm
  conversations; with --dflash also DFlash2 rounds), two hybrid canaries (a 2-turn conversation of 1536 + 1100 rows that
  must be pipelined twice and equal its first run every --canary-every iterations, plain and with --dflash) and, with
  --tp2-ref-every N, a gross-corruption comparison against the same conversation on the plain TP=2 prefill.

  Verdict (G-H4), exit 0 only when ALL hold (tools\tp\soak.ps1 prints every failed reason):
    - no TDR (WER LiveKernelEvent 141 / System 4101) and no HIP error 719 in the log;
    - tool_tp_soak exited 0: canaries equal (TP canary and hybrid canaries), 0 all-reduce aborts, per-rank buffer drift
      <= 64 MiB, 0 device allocations inside collective commands, the hybrid engaged for the whole run with >= 1 pipelined
      call, the TP=2 reference comparison above the gross-corruption bound;
    - the log's "summary" and "teardown" lines have exit_code 0 (the TpModel with its stage Models was destroyed cleanly).
  Stage it: 5 minutes first, then 60 (each with its own --json). Stage A runs the DESKTOP card at ~100 % duty for a whole
  prefill (2-8 s at 8k-32k): run the first long runs with the user present. Refuses while an r4dx-server runs (the soak
  pre-flight, docs/tp.md 9.2).

  --pp-verify has no meaning in the hybrid (no peer mirror to digest) and is not accepted by tool_tp_soak; the canaries and
  the TP=2 reference are its comparison.

.EXAMPLE
  .\tools\hybrid\soak.ps1 --layout trellis --minutes 5 --json build\logs\hybrid_soak_5min.jsonl
.EXAMPLE
  .\tools\hybrid\soak.ps1 --layout trellis --minutes 5 --dflash E:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx --tp2-ref-every 4 --json build\logs\hybrid_soak_5min_dflash.jsonl
.EXAMPLE
  .\tools\hybrid\soak.ps1 --layout trellis --minutes 60 --dflash E:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx --tp2-ref-every 8 --json build\logs\hybrid_soak_60min.jsonl
#>
$ErrorActionPreference = "Stop"
$ToolArgs = @($args | ForEach-Object { [string]$_ })
$hasPp = $false
for ($i = 0; $i -lt $ToolArgs.Count; $i++) { if ($ToolArgs[$i] -eq '--pp') { $hasPp = $true } }
if (-not $hasPp) { $ToolArgs = @('--pp', '2') + $ToolArgs }
$tpSoak = Join-Path (Split-Path $PSScriptRoot) "tp\soak.ps1"
& $tpSoak @ToolArgs
exit $LASTEXITCODE
