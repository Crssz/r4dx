#requires -Version 5.1
<#
.SYNOPSIS
  Gate G-H3 of the hybrid serving mode (docs/pp-tp2-hybrid.md 4, 9 P4): KL(TP=1 || hybrid) against the TP=2 floor.

.DESCRIPTION
  Runs the existing long-prefix teacher-forced KL runner (tools\prefill\run_kl.ps1 = tool_teacher_forced_logprobs
  --tail-rows R; the same tool as tools\quant2\kl_rung4.ps1, whose decode path is the one docs/tp.md 10.4 measured the
  TP=2 floor 0.00088-0.00089 with) three times on the SAME tokens, SAME --max-ctx and SAME tail rows:

    1. tp1     TP=1 (HIP device 1, the headless card)               -- the reference
    2. tp2     --tp 2                                               -- the floor: KL(TP=1 || TP=2)
    3. hybrid  --tp 2 --pp 2 [--pp-min-rows 1]                      -- the prefix of every segment goes through the PP-2
                                                                       pipeline + the reshard, the tail rows are decoded at TP=2

  and compares the dumps with tools\prefill\kl_compare.py (row for row, fp64, full vocabulary). It prints KL(TP=1||hybrid),
  KL(TP=1||TP=2), the top-1 agreement of both against TP=1, and the gate

      KL_hybrid <= GateRatio (1.25) x KL_tp2          and          top1_hybrid >= top1_tp2 - Top1Margin (0.3 points)

  (not required: equality with --tp 2's own text -- the hybrid's prefill numerics are TP=1's, so it is expected CLOSER to
  TP=1 than --tp 2 is). The floor is measured here with the same protocol (tail-rows mode, prefix by ONE Prefill call);
  the documented decode-path floor (-DocFloorKl, 0.00088, docs/tp.md 10.4) is printed next to it for scale.

  PREFIX LENGTH. The hybrid pipelines a Prefill call only with >= --pp-min-rows (default 1024) new rows. The kl_corpus
  segments (tokens_canon.json: cpp_source, english_prose, python_source, thai_prose) are 1024 tokens, so at --tail-rows
  256 their prefix is 768 rows: this script then adds `--pp-min-rows 1` to the hybrid run and SAYS SO (the dispatch
  threshold is not what is measured; the numerics of the pipelined prefill + reshard are). With -Long (the 8k..128k
  segments of tools\prefill\make_kl_tokens.py) every prefix is >= 1024 rows and the default threshold stays. The tool
  itself fails the hybrid run if any segment's prefix was not pipelined or the hybrid is not engaged.

  --max-ctx is max(16384, longest segment + 64 rounded up to 1024) for all three runs (the hybrid's stage-KV capacity S must be
  >= 16384). A segment longer than S would take the TP prefill and fail the hybrid run's pipelined check (pass -HybridCtx or
  fewer/shorter segments).

  GPU: tp2 and hybrid use BOTH cards (the desktop card is device 0): run with the user present and no r4dx-server running.
  Not run by its author. Each run is one process (one model load) over all selected segments; -Only selects phases
  (tp1, tp2, hybrid, compare) so the three loads can be done separately; -Reuse skips a phase whose directory already holds
  every segment's dump.

.PARAMETER OutDir
  Root of the dumps: <OutDir>\tp1, <OutDir>\tp2, <OutDir>\hybrid, and kl_tp1_vs_tp2.json / kl_tp1_vs_hybrid.json /
  kl_tp2_vs_hybrid.json / kl_hybrid_gate.json.

.PARAMETER Tokens
  The tokens.json (shared format). Default: tools\reference\kl_corpus\tokens_canon.json (the Rung-4 canon). -Long selects
  <models-root>\r4dx\prefill-m0\kl\tokens_long.json instead.

.EXAMPLE
  .\tools\hybrid\kl_hybrid.ps1 -OutDir E:\models\r4dx\hybrid\kl-gh3
.EXAMPLE
  .\tools\hybrid\kl_hybrid.ps1 -OutDir E:\models\r4dx\hybrid\kl-gh3-long -Long -Segment prose_8k,code_8k,recall_8k
.EXAMPLE
  .\tools\hybrid\kl_hybrid.ps1 -OutDir E:\models\r4dx\hybrid\kl-gh3 -Only hybrid,compare -Reuse    # the other two dumps exist
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx",
    [string]$Layout = 'trellis',
    [string]$Tokens = '',
    [switch]$Long,
    [int]$TailRows = 0,           # 0 = the tokens file's tail_rows (run_kl.ps1 reads it from there; a different value is an error)
    [string[]]$Segment = @(),
    [int]$PpSplit = 0,
    [int]$PpMinRows = 0,          # 0 = automatic: 1 when the shortest prefix is below 1024 rows, else the hybrid's default
    [int64]$HybridCtx = 0,
    [double]$GateRatio = 1.25,
    [double]$Top1Margin = 0.3,
    [double]$DocFloorKl = 0.00088,
    [ValidateSet('tp1', 'tp2', 'hybrid', 'compare')][string[]]$Only = @('tp1', 'tp2', 'hybrid', 'compare'),
    [switch]$Reuse,
    [string]$Tool = '',
    [string]$Python = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$runKl = Join-Path $repo 'tools\prefill\run_kl.ps1'
$klCompare = Join-Path $repo 'tools\prefill\kl_compare.py'
$modelsRoot = if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' }
if (-not $Tokens) {
    $Tokens = if ($Long) { Join-Path $modelsRoot 'r4dx\prefill-m0\kl\tokens_long.json' } else { Join-Path $repo 'tools\reference\kl_corpus\tokens_canon.json' }
}
foreach ($p in $runKl, $klCompare, $Tokens, $Model, $Python) { if (-not (Test-Path $p)) { throw "[kl_hybrid] missing $p" } }
if ($TailRows -lt 0) { throw "[kl_hybrid] -TailRows must be >= 1, or 0 for the tokens file's tail_rows (the uniform pass has no prefill to pipeline)" }

# Segment lengths: the shortest prefix decides whether the hybrid's 1024-row threshold is met; the longest decides --max-ctx.
[string]$segList = ($Segment -join ',')
$info = & $Python -c @"
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
want = set(s for s in sys.argv[2][4:].split(',') if s)
segs = [s for s in d['segments'] if not want or s['name'] in want]
if not segs: sys.exit('no matching segment')
print(d.get('tail_rows', 256), min(len(s['token_ids']) for s in segs), max(len(s['token_ids']) for s in segs), len(segs))
"@ $Tokens "sel=$segList"
if ($LASTEXITCODE -ne 0) { throw "[kl_hybrid] cannot read $Tokens" }
$fileTail, $minT, $maxT, $nSeg = $info.Trim().Split(' ') | ForEach-Object { [int]$_ }
# run_kl.ps1 takes the tail from the tokens file (tail_rows, default 256) and has no way to override it: -TailRows can only
# restate it, so a different value would only mislabel the prefix sizes, the --pp-min-rows decision and the verdict.
if ($TailRows -eq 0) { $TailRows = $fileTail }
elseif ($TailRows -ne $fileTail) { throw "[kl_hybrid] -TailRows $TailRows but $Tokens says tail_rows $fileTail (run_kl.ps1 uses the file's value); omit -TailRows or edit the tokens file" }
if ($minT -le $TailRows) { throw "[kl_hybrid] the shortest segment has $minT tokens, not more than -TailRows $TailRows" }
$minPrefix = $minT - $TailRows
$maxCtx = [int][Math]::Max(16384, [Math]::Ceiling(($maxT + 64) / 1024.0) * 1024)
$hybridMinRowsArgs = @()
if ($PpMinRows -gt 0) {
    $hybridMinRowsArgs = @('--pp-min-rows', "$PpMinRows")
} elseif ($minPrefix -lt 1024) {
    $hybridMinRowsArgs = @('--pp-min-rows', '1')
}
Write-Host ("[kl_hybrid] {0} segment(s) of {1}..{2} tokens, tail {3} rows, prefix {4}..{5} rows, --max-ctx {6} for all three runs" -f `
    $nSeg, $minT, $maxT, $TailRows, $minPrefix, ($maxT - $TailRows), $maxCtx)
if ($hybridMinRowsArgs.Count -gt 0) {
    Write-Host (("[kl_hybrid] NOTE: the shortest prefix is {0} rows (< the hybrid's 1024-row threshold): the hybrid run uses {1} so that every prefix " +
                 "IS pipelined; what this measures is the numerics of the pipelined prefill + reshard, not the dispatch threshold. Use -Long for prefixes above 1024.") -f `
        $minPrefix, ($hybridMinRowsArgs -join ' '))
}

$phases = @(
    @{ Name = 'tp1';    Args = @{ Device = 1 };  Extra = @('--max-ctx', "$maxCtx") },
    @{ Name = 'tp2';    Args = @{ Tp = 2 };      Extra = @('--max-ctx', "$maxCtx") },
    @{ Name = 'hybrid'; Args = @{ Tp = 2 };      Extra = @('--max-ctx', "$maxCtx", '--pp', '2') + $(if ($PpSplit -gt 0) { @('--pp-split', "$PpSplit") } else { @() }) +
                                                         $(if ($HybridCtx -gt 0) { @('--hybrid-ctx', "$HybridCtx") } else { @() }) + $hybridMinRowsArgs }
)
# What a dump directory was produced with: run_kl.ps1 records the tool's arguments in <dir>\run_info.json. Returns '' when the
# directory's run is what phase $Name needs (tp1: no --tp / --pp; tp2: --tp 2, no --pp; hybrid: --tp 2 --pp 2; all with this
# run's tail rows and --max-ctx), else the reason it is not -- so a stale plain --tp 2 dump cannot stand in for the hybrid's.
function Get-DumpMismatch([string]$Name, [string]$Dir) {
    $info = Join-Path $Dir 'run_info.json'
    if (-not (Test-Path $info)) { return "no run_info.json in $Dir" }
    $ra = [string](Get-Content $info -Raw | ConvertFrom-Json).args
    $wantTp = $Name -ne 'tp1'
    $wantPp = $Name -eq 'hybrid'
    if (($ra -match '--tp 2') -ne $wantTp) { return "run args '$ra' do not match phase $Name (--tp 2 expected: $wantTp)" }
    if (($ra -match '--pp 2') -ne $wantPp) { return "run args '$ra' do not match phase $Name (--pp 2 expected: $wantPp)" }
    if ($ra -notmatch "--tail-rows $TailRows(\s|$)") { return "run args '$ra' do not use --tail-rows $TailRows" }
    $mc = [regex]::Matches($ra, '--max-ctx (\d+)')
    if ($mc.Count -eq 0 -or [int]$mc[$mc.Count - 1].Groups[1].Value -ne $maxCtx) { return "run args '$ra' do not end on --max-ctx $maxCtx" }
    return ''
}
$savedHip = $env:HIP_VISIBLE_DEVICES
try {
    foreach ($ph in $phases) {
        if ($Only -notcontains $ph.Name) { continue }
        $dir = Join-Path $OutDir $ph.Name
        $have = if (Test-Path $dir) { @(Get-ChildItem (Join-Path $dir '*.logprobs.f16') -ErrorAction SilentlyContinue).Count } else { 0 }
        if ($Reuse -and $have -ge $nSeg) {
            $why = Get-DumpMismatch $ph.Name $dir
            if (-not $why) {
                Write-Host "[kl_hybrid] $($ph.Name): $have dump(s) in $dir, -Reuse: not re-run"
                continue
            }
            Write-Host "[kl_hybrid] $($ph.Name): $have dump(s) in $dir are not reusable ($why): re-running"
        }
        Write-Host ("[kl_hybrid] {0:HH:mm:ss} phase {1}: run_kl.ps1 {2} --> {3}" -f (Get-Date), $ph.Name, ($ph.Extra -join ' '), $dir)
        $a = @{ Tokens = $Tokens; OutDir = $dir; Model = $Model; Layout = $Layout; ExtraArgs = [string[]]$ph.Extra }
        if ($segList) { $a.Segment = $Segment }
        if ($Tool) { $a.Tool = $Tool }
        foreach ($k in $ph.Args.Keys) { $a[$k] = $ph.Args[$k] }
        & $runKl @a
        if ($LASTEXITCODE -ne 0) { throw "[kl_hybrid] phase $($ph.Name) failed (run_kl.ps1 exit $LASTEXITCODE); see $dir\*.log and *.err.log" }
        if ($ph.Name -eq 'hybrid') {
            # The tool fails by itself when the hybrid is not engaged or a prefix was not pipelined; show its evidence.
            $logs = @(Get-ChildItem (Join-Path $dir '*.log') | Where-Object { $_.Name -notmatch '\.err\.log$' })
            foreach ($l in $logs) {
                Get-Content $l.FullName | Where-Object { $_ -match '^\[(hybrid|stats)\]' } | ForEach-Object { Write-Host "  $_" }
            }
        }
    }
} finally {
    if ($null -ne $savedHip) { $env:HIP_VISIBLE_DEVICES = $savedHip } else { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
}

if ($Only -notcontains 'compare') { Write-Host '[kl_hybrid] done (no comparison requested)'; exit 0 }

function Invoke-Compare([string]$ref, [string]$test, [string]$json) {
    $a = @($klCompare, '--ref', (Join-Path $OutDir $ref), '--test', (Join-Path $OutDir $test), '--tokens', $Tokens, '--json', (Join-Path $OutDir $json))
    if ($segList) { $a += @('--segment', $segList) }
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $Python @a *> (Join-Path $OutDir ($json -replace '\.json$', '.log')); $rc = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if ($rc -ne 0) { throw "[kl_hybrid] kl_compare.py $ref vs $test exited $rc (see $OutDir\$($json -replace '\.json$', '.log'))" }
    (Get-Content (Join-Path $OutDir $json) -Raw | ConvertFrom-Json)
}
foreach ($d in 'tp1', 'tp2', 'hybrid') {
    if (-not (Test-Path (Join-Path $OutDir $d))) { throw "[kl_hybrid] no $d dump in $OutDir (run that phase first)" }
    $why = Get-DumpMismatch $d (Join-Path $OutDir $d)
    if ($why) { throw "[kl_hybrid] the $d dump is not the one this gate needs: $why (re-run that phase)" }
}
$env:PYTHONIOENCODING = 'utf-8'
$floor = Invoke-Compare 'tp1' 'tp2' 'kl_tp1_vs_tp2.json'
$hyb = Invoke-Compare 'tp1' 'hybrid' 'kl_tp1_vs_hybrid.json'
$mid = Invoke-Compare 'tp2' 'hybrid' 'kl_tp2_vs_hybrid.json'

Write-Host ''
Write-Host ('{0,-34} {1,12} {2,12} {3,10} {4,10}' -f 'comparison (ref || test)', 'mean KL', 'p99 KL', 'top-1 %', 'rows')
foreach ($r in @(@('KL(TP=1 || TP=2)   [the floor]', $floor), @('KL(TP=1 || hybrid)      [gate]', $hyb), @('KL(TP=2 || hybrid)      [info]', $mid))) {
    $o = $r[1].overall
    Write-Host ('{0,-34} {1,12:F6} {2,12:F6} {3,10:F2} {4,10}' -f $r[0], $o.kl_mean, $o.kl_p99, $o.top1_pct, $o.rows)
}
Write-Host ''
Write-Host 'per segment (mean KL: TP=2 floor / hybrid):'
foreach ($p in $floor.segments.PSObject.Properties) {
    $h = $hyb.segments.($p.Name)
    Write-Host ('  {0,-16} {1:F6} / {2:F6}   top-1 {3:F2} / {4:F2}' -f $p.Name, $p.Value.kl_mean, $h.kl_mean, $p.Value.top1_pct, $h.top1_pct)
}

$klTp2 = [double]$floor.overall.kl_mean
$klHyb = [double]$hyb.overall.kl_mean
$t1Tp2 = [double]$floor.overall.top1_pct
$t1Hyb = [double]$hyb.overall.top1_pct
$limit = $GateRatio * $klTp2
$fail = @()
if (-not ($klHyb -le $limit)) { $fail += ('KL(TP=1||hybrid) {0:F6} > {1} x KL(TP=1||TP=2) {2:F6} = {3:F6}' -f $klHyb, $GateRatio, $klTp2, $limit) }
if (-not ($t1Hyb -ge $t1Tp2 - $Top1Margin)) { $fail += ('top-1 agreement with TP=1: hybrid {0:F2}% < TP=2 {1:F2}% - {2} points' -f $t1Hyb, $t1Tp2, $Top1Margin) }
$verdict = [ordered]@{
    kl_tp2 = $klTp2; kl_hybrid = $klHyb; kl_limit = $limit; gate_ratio = $GateRatio
    top1_tp2 = $t1Tp2; top1_hybrid = $t1Hyb; top1_margin = $Top1Margin
    documented_floor_kl = $DocFloorKl; documented_limit = $GateRatio * $DocFloorKl
    prefix_rows_min = $minPrefix; pp_min_rows_args = ($hybridMinRowsArgs -join ' '); max_ctx = $maxCtx; tail_rows = $TailRows
    tokens = $Tokens; passed = ($fail.Count -eq 0)
}
$verdict | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $OutDir 'kl_hybrid_gate.json')
Write-Host ''
Write-Host ('[kl_hybrid] KL(TP=1||hybrid) = {0:F6}   KL(TP=1||TP=2) = {1:F6}   limit {2} x floor = {3:F6}   (documented decode-path floor {4} x {2} = {5:F6})' -f `
    $klHyb, $klTp2, $GateRatio, $limit, $DocFloorKl, ($GateRatio * $DocFloorKl))
Write-Host ('[kl_hybrid] top-1 agreement with TP=1: hybrid {0:F2}%   TP=2 {1:F2}%   (margin {2} points)' -f $t1Hyb, $t1Tp2, $Top1Margin)
if ($fail.Count -gt 0) {
    Write-Host ('[kl_hybrid] G-H3 FAILED: ' + ($fail -join '; '))
    exit 1
}
Write-Host '[kl_hybrid] G-H3 PASSED: KL(TP=1||hybrid) <= 1.25 x the TP=2 floor and top-1 within 0.3 points of TP=2'
exit 0
