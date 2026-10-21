# qwen3_5_moe (Ornith-1.5-122B) trellis pipeline: calibration corpus -> per-layer Hessians -> oracles at KB 3 and KB 4 ->
# allocation under the per-card budget -> container (docs/qwen35moe-plan.md M4; the counterpart of
# tools\quant2\qwen4exp\pipeline.ps1). Without -Run NOTHING executes: every stage prints the exact commands it would run, and
# `-Stage budget` (also printed by every plan) gives the disk and GPU-time budget.
#
#   .\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47            # print the whole chain (plan + budget)
#   .\tools\quant2\qwen35moe\pipeline.ps1 -Stage corpus -Run                  # CPU, minutes -> tokens_calib.json
#   .\tools\quant2\qwen35moe\pipeline.ps1 -Stage hessian -Layers 0..3 -Run    # GPU, one bounded slice per lock
#   .\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47 -Run       # the real thing (many hours; resumable)
#
# Stages
#   corpus   CPU  corpus_q35.py build              tokens_calib.json (hessian-v2 pieces + the SFT slice, canonical tokenizer)
#   hessian  GPU  hessian_q35.py capture --layer L Hessians + routing counts + fallback of layer L (and the layer L+1 residual)
#   oracle   GPU  oracle_q35.py quantize --layer L --K 3 | 4       one oracle per K, chunks of 32 experts, manifest refreshed
#   verify   CPU  oracle_q35.py verify --layer L --delete-hessians  both K dirs verified (sha256, shapes, reconstruction), then the
#                                                                  layer's .hess files deleted
#   alloc    CPU  alloc_q35.py allocate            mix manifest + allocation report under the 26 GiB per card cap
#   convert  CPU  r4dx-convert --trellis-from <mix> ...            E:\models\r4dx\ornith-1.5-122b-trellis-mix.r4dx
#   budget   CPU  disk_budget.py                   the disk and time budget of the chain for -Layers
#   all      corpus, then per layer hessian -> oracle K3 -> oracle K4 -> verify (layer-major: only one layer's Hessians exist
#            at a time), then alloc, convert.  `-Stage hessian|oracle|verify -Layers a..b` run just that step for those layers.
#
# GPU discipline: every GPU step is ONE process under
#   tools\with_lock.ps1 -Name gpu -Run "<cmd>"
# with HIP_VISIBLE_DEVICES=<-Device> (default 0: physical device 1 has been in a faulted state since 2026-10-21; the user allowed
# both GPUs on 2026-10-21; R4DX_ALLOW_GPU0=1 is set for device 0). A step is bounded by -BudgetSeconds (default 840 = 14 min of
# work) and exits 75 when it has more to do: this driver then releases the lock (other agents get the GPU) and takes it again for
# the next slice. State between slices lives under <Root>\work (hessian accumulators, residual boundaries) and in the oracle
# chunk files, with content hashes. Everything is resumable: re-run the same command.
#
# TWO LANES (-Lanes 2, -Stage all): the K4 oracle runs on a second GPU in parallel with the Hessians + K3 oracle (README "Two lanes").
#   lane A  HIP device -DeviceA (1), lock -LockA (gpu):      per layer  hessian -> check -> oracle K3
#   lane B  HIP device -DeviceB (0), lock -LockB (gpu-dev0): per layer  (waits for the layer's hessian marker) -> oracle K4
#   verify (CPU, deletes the Hessians) runs once per layer, by whichever lane finishes second. State: <Root>\lanes\ marker files.
#   A failure in either lane writes <Root>\FAILED.txt (lane=A|B) and a STOP marker; the other lane stops at its next slice boundary.
#   Default (-Lanes 1) is the single-lane behaviour below, unchanged.
#
# WORK QUEUE (-Workers 2, -Stage all; preferred over -Lanes 2 when both GPUs are available): the Hessian chain (sequential, layer L needs the
# residual of L-1) runs on the main GPU with priority, and BOTH GPUs claim whole-layer oracle jobs (K3 then K4 of one layer) from a queue
# of layers whose Hessians are ready, by atomic claim file (README "Work queue").
#   worker G1  HIP device -DeviceA (1), lock -LockA (gpu):      Hessian chain first (at the next slice boundary), else an oracle job
#   worker G0  HIP device -DeviceB (0), lock -LockB (gpu-dev0): oracle jobs only (after the GPU probe)
#   a worker that finishes a layer's oracle refreshes the K manifests (under a machine-wide mutex) and runs verify for it.
#
# Python: -Python (the reference venv: transformers with qwen3_5_moe, torch 2.9.1+rocm) for everything: trellis_quant.block_ldl
# factors on the CPU for GPU tensors, so no MAGMA venv is needed (-OraclePython overrides).
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('corpus', 'hessian', 'oracle', 'verify', 'alloc', 'convert', 'budget', 'all')]
  [string]$Stage,
  [string]$Layers = '0..47',
  [string]$Propagate = '',            # layers inside -Layers that only pass the residual on (no Hessians / oracle): e.g. '1,2' to measure layers 0 and 3 alone
  [switch]$Run,
  [switch]$Force,
  [string]$Ks = '3,4',
  [double]$CapGib = 26,
  [string]$Floor = '',
  [int]$BudgetSeconds = 840,
  [int]$ChunkExperts = 32,
  [int]$Cohort = 32,                    # oracle worker threads per class cohort (batched encoder calls; 0 = off)
  [string]$Root = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\q35moe",
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\Ornith-1.5-122B",
  [string]$Tokens = '',
  [string]$KvCalib = '',
  [string]$Output = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\ornith-1.5-122b-trellis-mix.r4dx",
  [string]$Python = 'D:\venvs\r4dx-qwen4-ref\Scripts\python.exe',
  [string]$OraclePython = '',
  [string]$Exe = '',
  [string]$Device = '0',
  [int]$Threads = 32,
  [double]$MinFreeGiB = 40,              # free space a layer step needs under -Root (disk_budget.py: ~32 GiB of scratch + margin)
  [int]$Lanes = 1,                       # 2 = two GPU lanes in parallel (-Stage all only): lane A = hessian + K3 on -DeviceA, lane B = K4 on -DeviceB
  [string]$Lane = '',                    # internal: 'A' or 'B', set by the -Lanes 2 supervisor for the lane processes it starts
  [string]$DeviceA = '1',
  [string]$DeviceB = '0',                # the display GPU: a probe runs first, a display-driver reset event stops both lanes
  [string]$LockA = 'gpu',
  [string]$LockB = 'gpu-dev0',           # tools\with_lock.ps1 -Name gpu-dev0: tests on device 0 should take it (or both locks)
  [int]$MaxInFlight = 0,                 # layers whose Hessians may exist at once (the Hessian chain waits for verify of layer i-MaxInFlight); 0 = 2 for -Lanes 2, 3 for -Workers 2
  [int]$Workers = 1,                     # 2 = work-queue mode (-Stage all only): workers G1 (device -DeviceA: Hessian chain + oracle jobs) and G0 (device -DeviceB: oracle jobs)
  [string]$Worker = '',                  # internal: 'G0' or 'G1', set by the -Workers 2 supervisor
  [int]$OracleSliceG1 = 420,             # --budget-seconds of the oracle slices of worker G1, so the Hessian chain waits at most about this long
  [int]$CohortB = 0,                     # oracle cohort threads of lane B (0 = -Cohort)
  [int]$PollSeconds = 20,
  [double]$ProbeMinFreeGiB = 14          # lane B start: free VRAM the probe requires on the display GPU (the oracle cohort needs ~8 GiB)
)
$ErrorActionPreference = 'Stop'
$script:boundParams = $PSBoundParameters   # the lane supervisor forwards them to its lane processes
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$env:PYTHONIOENCODING = 'utf-8'

# ---- paths ------------------------------------------------------------------------------------------
$qdir = $PSScriptRoot
if (-not $Tokens) { $Tokens = Join-Path $Root 'tokens_calib.json' }
if (-not $KvCalib) { $KvCalib = Join-Path $Root 'kvcalib.json' }
if (-not $OraclePython) { $OraclePython = $Python }
if (-not $Exe) {
  # a worktree has no build tree of its own: look in this checkout, then in the main one (the parent of git's common dir)
  $roots = @($repo)
  $common = (& git -C $repo rev-parse --path-format=absolute --git-common-dir) 2>$null
  if ($common) { $roots += (Split-Path -Parent $common) }
  foreach ($r in $roots) {
    foreach ($c in 'build\win-hip\src\convert\r4dx-convert.exe', 'build\win-hip-dry\src\convert\r4dx-convert.exe') {
      if (-not $Exe -and (Test-Path (Join-Path $r $c))) { $Exe = Join-Path $r $c }
    }
  }
  if (-not $Exe) { $Exe = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe' }
}
$logdir = Join-Path $Root 'logs'
$hessScript = 'tools\reference\qwen35moe\hessian_q35.py'
$allocScript = 'tools\reference\qwen35moe\alloc_q35.py'
$corpusScript = Join-Path $qdir 'corpus_q35.py'
$oracleScript = Join-Path $qdir 'oracle_q35.py'
$budgetScript = Join-Path $qdir 'disk_budget.py'
$mixdir = Join-Path $Root 'mix'
$kList = @($Ks -split ',' | Where-Object { $_ } | ForEach-Object { [int]$_ })
function OracleDir([int]$k) { Join-Path $Root "oracle-k$k" }

# layers: "0..47", "0,3,7", "0..3,10"
function ParseLayers([string]$spec) {
  $out = @()
  foreach ($part in ($spec -split ',')) {
    if ($part -match '^(\d+)\.\.(\d+)$') { $out += [int]$Matches[1]..[int]$Matches[2] }
    elseif ($part -match '^\d+$') { $out += [int]$part }
    else { throw "[pipeline] bad -Layers part '$part'" }
  }
  return ($out | Sort-Object -Unique)
}
$layerList = ParseLayers $Layers
$propList = if ($Propagate) { ParseLayers $Propagate } else { @() }
if ($layerList | Where-Object { $_ -lt 0 -or $_ -gt 47 }) { throw '[pipeline] -Layers must be within 0..47' }

$script:LockName = 'gpu'      # lane B / worker G0 use -LockB
$script:Who = if ($Lane) { $Lane } elseif ($Worker) { $Worker } else { '' }   # 'A' 'B' (lanes), 'G0' 'G1' (workers), '' single lane
$script:maxInFlight = if ($MaxInFlight -gt 0) { $MaxInFlight } elseif ($Workers -ge 2 -or $Worker) { 3 } else { 2 }
$script:stepRc = 0
$script:laneStart = Get-Date
$script:refused = $false
$script:problems = @()
$script:failedMarker = Join-Path $Root 'FAILED.txt'   # one line: time, step, exit code, the last error line, the log; a monitor polls for it
$script:markerWritten = $false

# The line of a step's log that says why it failed: the last Python exception line, else a verify summary with problems, else the last
# "refused / NOT READY / FAILED" line, else the last line; PowerShell's own error-record decoration and torch's device-count noise are skipped.
function LastErrorLine([string]$log) {
  if (-not $log -or -not (Test-Path $log)) { return '(no log)' }
  $noise = '^\s*\+|^\s*At line|CategoryInfo|FullyQualifiedErrorId|NativeCommandError|Failed to get device count|^\s*$'
  $lines = @(Get-Content $log -Tail 200 -ErrorAction SilentlyContinue | Where-Object { $_ -and $_ -notmatch $noise })
  if ($lines.Count -eq 0) { return '(empty log)' }
  foreach ($pat in '^\s*[\w\.]*(Error|Exception|SystemExit)\b', '\b[1-9]\d* problem\(s\)', 'refused|NOT READY|FAILED|\bfailed\b|Traceback') {
    $hit = $lines | Where-Object { $_ -match $pat } | Select-Object -Last 1
    if ($hit) { return $hit.Trim() }
  }
  return $lines[-1].Trim()
}

function Write-Failed([string]$step, [string]$msg, [string]$log = '') {
  try {
    New-Item -ItemType Directory -Force (Split-Path -Parent $script:failedMarker) | Out-Null
    $tag = if ($script:Who) { $(if ($script:Who -like 'G*') { 'worker=' } else { 'lane=' }) + "$($script:Who) " } else { '' }
    $line = ("{0} FAILED {1}step={2}: {3}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $tag, $step, ($msg -replace '[\r\n]+', ' '))
    if ($line.Length -gt 600) { $line = $line.Substring(0, 600) + '...' }
    if ($log) { $line += "  [log $log]" }
    if (-not (Test-Path $script:failedMarker)) {   # the FIRST failure of a run is the one that stays (-Run deletes the marker at start)
      Set-Content -Path $script:failedMarker -Value $line -Encoding ascii
    }
    $script:markerWritten = $true
    if ($script:Who) { LMark 'STOP' "$($script:Who) failed at $step" }
  } catch { }
}

# ---- helpers ----------------------------------------------------------------------------------------
function Q([string]$a) { if ($a -match '^[A-Za-z0-9_\-\.\\:/=,]+$') { $a } else { '"' + $a.Replace('"', '\"') + '"' } }
function Fmt([string]$exe, [string[]]$a) { (@((Q $exe)) + ($a | ForEach-Object { Q $_ })) -join ' ' }
function Need([bool]$ok, [string]$msg) { if (-not $ok) { $script:problems += $msg } }
function FreeGiB([string]$path) {
  $r = [IO.Path]::GetPathRoot($path)
  return [math]::Round((Get-PSDrive $r.Substring(0, 1)).Free / 1GB, 1)
}
function HessianDone([int]$L) { Test-Path (Join-Path $Root ("hessian\L{0:D2}\DONE.json" -f $L)) }
# `verify --delete-hessians` removes a layer's .hess files and keeps DONE.json, so DONE alone does not mean the Hessians exist
function HessianFiles([int]$L) {
  $d = Join-Path $Root ("hessian\L{0:D2}" -f $L)
  (Test-Path $d) -and ($null -ne (Get-ChildItem $d -Filter '*.hess' -File -ErrorAction SilentlyContinue | Select-Object -First 1))
}
# True when every oracle chunk of layer $L is usable in every -Ks directory for the CURRENT code / job key (oracle_q35.py status)
function OracleComplete([int]$L) { OracleCompleteFor $L $kList }
function OracleCompleteFor([int]$L, [int[]]$ks) {
  $a = @($oracleScript, 'status', '--layer', "$L", '--model-dir', $Model, '--hessian-root', $Root, '--chunk-experts', "$ChunkExperts")
  foreach ($k in $ks) { $a += @('--out-dir', ((OracleDir $k) + "@$k")) }
  $saved = $env:HIP_VISIBLE_DEVICES; $env:HIP_VISIBLE_DEVICES = '-1'
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # torch writes "Failed to get device count" to stderr
  try { & $OraclePython @a *> $null; return ($LASTEXITCODE -eq 0) } finally { $env:HIP_VISIBLE_DEVICES = $saved; $ErrorActionPreference = $prev }
}

function Open-Gpu {
  if ($Device -notmatch '^[01]$') { $script:problems += "-Device must be 0 or 1, got '$Device'" }
}
function Flush-Problems([string]$name) {
  if ($script:problems.Count -gt 0) {
    foreach ($p in $script:problems) { Write-Host "[pipeline] $name NOT READY: $p" }
    if ($Run) { throw "[pipeline] $name refused: $($script:problems.Count) prerequisite problem(s) above" }
  }
  $script:problems = @()
}

# One command. CPU: run directly. GPU: each slice under tools\with_lock.ps1 -Name gpu (HIP_VISIBLE_DEVICES set inside the lock
# so nothing else is affected); exit 75 = "more to do": the lock is released and taken again for the next slice.
function Step([string]$name, [string]$exe, [string[]]$a, [bool]$gpu, [bool]$readOnly = $false, [bool]$oneSlice = $false) {
  $inner = Fmt $exe $a
  if ($gpu) {
    $pre = "set HIP_VISIBLE_DEVICES=$Device&& "
    if ($Device -eq '0') { $pre += 'set R4DX_ALLOW_GPU0=1&& ' }
    $wl = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $repo 'tools\with_lock.ps1'), '-Name', $script:LockName, '-Run', ($pre + $inner))
    $shown = "powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name $($script:LockName) -Run `"$pre$inner`""
  } else {
    $shown = "`$env:HIP_VISIBLE_DEVICES='-1'; $inner"
  }
  Write-Host "[pipeline] $name CMD: $shown"
  if (-not $Run -and -not $readOnly) { return }
  if ($readOnly) { $saved = $env:HIP_VISIBLE_DEVICES; $env:HIP_VISIBLE_DEVICES = '-1'; try { & $exe @a } finally { $env:HIP_VISIBLE_DEVICES = $saved }; return }
  New-Item -ItemType Directory -Force $logdir | Out-Null
  $log = Join-Path $logdir ("pipeline_{0}.log" -f ($name -replace '[^A-Za-z0-9_\-]', '_'))
  $t0 = Get-Date
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # PS 5.1: native stderr lines are error records
  try {
    while ($true) {
      if ($gpu) { & powershell @wl *>> $log } else {
        $saved = $env:HIP_VISIBLE_DEVICES; $env:HIP_VISIBLE_DEVICES = '-1'
        try { & $exe @a *>> $log } finally { $env:HIP_VISIBLE_DEVICES = $saved }
      }
      $rc = $LASTEXITCODE
      if ($script:Who -in 'B', 'G0' -and $gpu) {   # the display GPU: a driver reset in this run's lifetime stops everything, no blind retry
        $ev = Display-Reset $script:laneStart
        if ($ev) { Write-Failed $name "display driver reset event on the display GPU ($($script:Who)): $ev" $log; throw "[pipeline] $name stopped: $ev" }
      }
      if ($rc -eq 75 -and $gpu) {
        if ($script:Who -and (LHas 'STOP')) { Write-Host "[$($script:Who)] stop requested between slices of ${name}: state saved, resumable"; throw 'LANE_STOP' }
        if ($oneSlice) { $script:stepRc = 75; Write-Host ("[pipeline] {0}: slice done, more to do ({1:N1} min)" -f $name, ((Get-Date) - $t0).TotalMinutes); return }
        LaneStatus "$name slice done, continuing"; Write-Host ("[pipeline] {0}: slice done, more to do ({1:N1} min so far), lock released" -f $name, ((Get-Date) - $t0).TotalMinutes); continue }
      break
    }
  } finally { $ErrorActionPreference = $prev }
  $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
  if ($rc -ne 0) {
    Write-Failed $name ("exit $rc after $min min: " + (LastErrorLine $log)) $log
    throw "[pipeline] $name failed (exit $rc) after $min min, see $log"
  }
  $script:stepRc = 0
  Write-Host "[pipeline] $name done in $min min (log $log)"
}

# ---- stages -----------------------------------------------------------------------------------------
function Stage-Corpus {
  Write-Host "`n[pipeline] ===== corpus (CPU, minutes)"
  if ((Test-Path $Tokens) -and -not $Force) { Write-Host "[pipeline] corpus: $Tokens exists, skipping (-Force redoes it)"; return }
  # the optional sources (WikiText-2, corpus-v2 generated samples, the agent SFT parquet) are skipped by corpus_q35.py when absent,
  # with a warning and a line in tokens_calib.meta.json: say so up front, a real run wants them (corpus_q35.py docstring)
  foreach ($opt in @((Join-Path $Root 'sft\qwen38-agent-sft-combined-100pct.parquet'), 'E:\models\r4dx\corpus-v2\samples.jsonl', 'E:\models\wikitext-2-raw\wiki.train.raw')) {
    if (-not (Test-Path $opt)) { Write-Host "[pipeline] corpus: optional source missing, will be skipped: $opt" }
  }
  Flush-Problems 'corpus'
  Step 'corpus' $Python @($corpusScript, 'build', '--out', $Tokens, '--sft', (Join-Path $Root 'sft\qwen38-agent-sft-combined-100pct.parquet')) $false
}

function Stage-Hessian([int]$L) {
  if ($propList -contains $L) {
    Open-Gpu
    Need (Test-Path $Tokens) "no $Tokens (run -Stage corpus)"
    Flush-Problems "propagate-L$L"
    Step "propagate-L$L" $Python @($hessScript, 'capture', '--layer', "$L", '--root', $Root, '--tokens', $Tokens, '--model-dir', $Model,
         '--budget-seconds', "$BudgetSeconds", '--propagate-only') $true
    return
  }
  Open-Gpu
  Need (Test-Path $Tokens) "no $Tokens (run -Stage corpus)"
  Need ((FreeGiB $Root) -ge $MinFreeGiB) "less than $MinFreeGiB GiB free on $Root (a layer step needs two 6 GB residual boundaries + 11 GB of accumulators + 6 GB of Hessians; -Stage budget has the whole chain)"
  $recapture = $false
  if ((HessianDone $L) -and -not $Force) {
    if (HessianFiles $L) { Write-Host "[pipeline] hessian L${L}: DONE already, skipping (-Force redoes it)"; Flush-Problems "hessian-L$L"; return }
    if (OracleComplete $L) { Write-Host "[pipeline] hessian L${L}: DONE, Hessians already deleted after a verified oracle: not needed"; Flush-Problems "hessian-L$L"; return }
    # DONE, the .hess files are gone (verify --delete-hessians) and the oracle chunks are NOT usable for this code (the job key changed,
    # a chunk is missing): the Hessians are needed again. Needs the layer's input boundary: redo the layers from the first stale one.
    Write-Host "[pipeline] hessian L${L}: DONE but its .hess files were deleted and its oracle chunks are stale or missing: capturing again"
    $recapture = $true
  }
  Flush-Problems "hessian-L$L"
  $a = @($hessScript, 'capture', '--layer', "$L", '--root', $Root, '--tokens', $Tokens, '--model-dir', $Model,
         '--budget-seconds', "$BudgetSeconds")
  if ($Force -or $recapture) { $a += '--force' }
  Step "hessian-L$L" $Python $a $true
  Step "hessian-check-L$L" $Python @($hessScript, 'check', '--layer', "$L", '--root', $Root, '--eig-sample', '2') $false
}

function Stage-Oracle([int]$L, [int[]]$ks = @(), [bool]$oneSlice = $false) {
  if ($propList -contains $L) { return }
  if ($ks.Count -eq 0) { $ks = $kList }
  Open-Gpu
  Need (Test-Path $OraclePython) "no oracle python at $OraclePython"
  if (-not (HessianDone $L) -and $Run) { $script:problems += "layer $L has no Hessians (run -Stage hessian)" }
  Flush-Problems "oracle-L$L"
  foreach ($k in $ks) {
    if ($script:Who -and $Run -and (OracleCompleteFor $L @($k))) { Write-Host "[$($script:Who)] oracle-k$k-L${L}: every chunk usable already, skipping"; continue }
    $coh = if ($script:Who -in 'B', 'G0' -and $CohortB -gt 0) { $CohortB } else { $Cohort }
    $bud = if ($script:Who -eq 'G1' -and $OracleSliceG1 -gt 0) { $OracleSliceG1 } else { $BudgetSeconds }
    $a = @($oracleScript, 'quantize', '--layer', "$L", '--K', "$k", '--out-dir', (OracleDir $k), '--hessian-root', $Root,
           '--model-dir', $Model, '--chunk-experts', "$ChunkExperts", '--cohort', "$coh", '--budget-seconds', "$bud")
    if ($script:Who -like 'G*') { $a += '--no-manifest' }   # workers on different layers share one K directory: manifests are refreshed under a lock
    if ($Force) { $a += '--force' }
    Step "oracle-k$k-L$L" $OraclePython $a $true $false $oneSlice
  }
}

function Stage-Verify([int]$L) {
  if ($propList -contains $L) { return }
  $a = @($oracleScript, 'verify', '--layer', "$L", '--model-dir', $Model, '--hessian-root', $Root, '--chunk-experts', "$ChunkExperts", '--delete-hessians')
  foreach ($k in $kList) { $a += @('--out-dir', ((OracleDir $k) + "@$k")) }
  Step "verify-L$L" $OraclePython $a $false
}

function Stage-Budget {
  Write-Host "`n[pipeline] ===== budget (CPU, seconds)"
  $a = @($budgetScript, '--root', $Root, '--model-dir', $Model, '--layers', $Layers, '--ks', $Ks, '--cap-gib', "$CapGib", '--tokens-file', $Tokens)
  Step 'budget' $Python $a $false $true
}

function Stage-Alloc {
  Write-Host "`n[pipeline] ===== alloc (CPU, seconds)"
  foreach ($k in $kList) {
    $m = Join-Path (OracleDir $k) 'weights_override.json'
    if ($Run) { Need (Test-Path $m) "no $m (run -Stage oracle for every layer)" }
    if ($Run -and (Test-Path $m)) {
      $c = & $Python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['complete'])" $m
      Need ("$c".Trim() -eq 'True') "$m is not complete (not every layer has been quantized at K$k)"
    }
  }
  Flush-Problems 'alloc'
  $a = @($allocScript, 'allocate', '--out-dir', $mixdir, '--cap-gib', "$CapGib")
  foreach ($k in $kList) { $a += @('--src', (OracleDir $k)) }
  if ($Floor) { $a += @('--floor', $Floor) }
  Step 'alloc' $Python $a $false
}

function Stage-Convert {
  Write-Host "`n[pipeline] ===== convert (CPU, minutes; full reconstruction check)"
  $manifest = Join-Path $mixdir 'weights_override.json'
  $sha = '<sha256 of weights_override.json>'
  if (Test-Path $manifest) { $sha = (Get-FileHash $manifest -Algorithm SHA256).Hash.ToLower() }
  $a = @('--input', $Model, '--output', $Output,
         '--trellis-from', $mixdir, '--trellis-manifest-sha256', $sha, '--trellis-verify', 'full',
         '--layouts', 'w4a16', '--no-bf16', '--lm-head', 'w4a16', '--w4a16-group-rule', '^lm_head$=32', '--threads', "$Threads")
  if (Test-Path $KvCalib) { $a += @('--kv-calib', $KvCalib) }
  else { Write-Host "[pipeline] convert: no fp8 KV calibration at ${KvCalib}: every k/v descale is the placeholder 1.0 (the tiny container is built that way); calibrate before shipping" }
  Need (Test-Path $Exe) "converter not found at $Exe (build target r4dx-convert through tools\with_lock.ps1 -Name build)"
  Need (Test-Path $manifest) "no mix manifest $manifest (run -Stage alloc)"
  Need ((FreeGiB (Split-Path -Parent $Output)) -ge 60) "less than 60 GiB free next to $Output"
  if ((Test-Path $Output) -and -not $Force) { Write-Host "[pipeline] convert: $Output exists, skipping (-Force redoes it)"; return }
  Flush-Problems 'convert'
  Step 'convert' $Exe $a $false
  if ($Run -and (Test-Path $Output)) { Write-Host "[pipeline] container ${Output}: $([math]::Round((Get-Item $Output).Length / 1GB, 2)) GB" }
}


# ---- two lanes ----------------------------------------------------------------------------------------------------
# Marker files in <Root>\lanes\ (wiped by the supervisor at start; every fact is re-derived from the oracle dirs by the lanes):
#   hess-LNN.ready     lane A: the layer's Hessians are on disk and checked (lane B may start its oracle)
#   kK-LNN.done        the lane that owns K: every chunk of layer NN at K is usable
#   verify-LNN.claim   created atomically (CreateNew) by the lane that runs verify; the other one finds it and moves on
#   verify-LNN.done    verify ran (or was not needed): the layer's Hessians are deleted; lane A needs it for layer NN+MaxInFlight
#   STOP               a lane failed: the other stops at its next slice boundary (it is never killed)
#   lane-A.status / lane-B.status   one line: time, what the lane is doing
$script:lanesDir = Join-Path $Root 'lanes'
function Lk([string]$n) { Join-Path $script:lanesDir $n }
function LTag([int]$L) { 'L{0:D2}' -f $L }
function LMark([string]$n, [string]$text = '') {
  try { New-Item -ItemType Directory -Force $script:lanesDir | Out-Null; Set-Content -Path (Lk $n) -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $text) -Encoding ascii } catch { }
}
function LHas([string]$n) { Test-Path (Lk $n) }
function LaneStatus([string]$m) { if ($script:Who -and $Run) { LMark "lane-$($script:Who).status" $m } }
function Check-Stop {
  if ($script:Who -and $Run -and (LHas 'STOP')) { Write-Host "[$($script:Who)] stop requested (another process failed): stopping at this boundary, state is resumable"; throw 'LANE_STOP' }
}
function Wait-For([scriptblock]$cond, [string]$what) {
  if (-not $Run) { return }
  $t0 = Get-Date; $say = Get-Date
  while (-not (& $cond)) {
    Check-Stop
    LaneStatus "waiting for $what"
    if (((Get-Date) - $say).TotalMinutes -ge 10) { Write-Host ("[lane {0}] still waiting for {1} ({2:N0} min)" -f $script:Who, $what, ((Get-Date) - $t0).TotalMinutes); $say = Get-Date }
    Start-Sleep -Seconds $PollSeconds
  }
}
# Display-driver reset events (TDR) since $since; $null when none. Lane B runs on the display GPU.
function Display-Reset([datetime]$since) {
  try {
    $e = Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'Display', 'amdkmdag', 'amdwddmg'; Id = 4101, 117, 141; StartTime = $since } -MaxEvents 1 -ErrorAction Stop
    return ("event {0} {1} at {2}" -f $e.Id, $e.ProviderName, $e.TimeCreated)
  } catch { return $null }
}
# Whichever lane finishes second runs verify; the atomic CreateNew claim means no race and no double delete.
function Try-Verify([int]$L) {
  if (-not $Run) { if ($script:Who -eq 'B') { Write-Host "[lane B] (verify of layer ${L}: run by whichever lane finishes the layer second)"; Stage-Verify $L }; return }
  $t = LTag $L
  if (LHas "verify-$t.done") { return }
  foreach ($k in $kList) { if (-not (LHas "k$k-$t.done")) { return } }
  try { $fs = [IO.File]::Open((Lk "verify-$t.claim"), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write); $fs.Close() } catch { return }
  if (HessianFiles $L) { LaneStatus "verify $t"; Stage-Verify $L } else { Write-Host "[$($script:Who)] verify ${t}: Hessians already deleted, nothing to do" }
  LMark "verify-$t.done" "lane $($script:Who)"
}

function Run-Lane([string]$which) {
  $script:Lane = $which
  $script:Who = $which
  $script:LockName = if ($which -eq 'B') { $LockB } else { $LockA }
  $script:Device = if ($which -eq 'B') { $DeviceB } else { $DeviceA }
  $script:laneStart = Get-Date
  $ks = @($kList)
  if ($ks.Count -lt 2) { throw '[pipeline] -Lanes 2 needs two rates in -Ks (3,4)' }
  $myKs = if ($which -eq 'A') { @($ks[0]) } else { @($ks[1..($ks.Count - 1)]) }
  Write-Host "`n[lane $which] device $Device, lock $script:LockName, K $($myKs -join ','), layers $($layerList -join ',')"
  if ($which -eq 'B') {
    # a probe before any oracle work on the display GPU: responsive, enough free VRAM
    Step 'gpu-probe-B' $Python @((Join-Path $qdir 'gpu_probe.py'), '--min-free-gib', "$ProbeMinFreeGiB") $true
  }
  for ($i = 0; $i -lt $layerList.Count; $i++) {
    $L = $layerList[$i]; $t = LTag $L
    Check-Stop
    Write-Host "`n[lane $which] ===== layer $L"
    if ($which -eq 'A') {
      if ($i -ge $script:maxInFlight) { $g = LTag $layerList[$i - $script:maxInFlight]; Wait-For { LHas "verify-$g.done" } "verify of $g (at most $($script:maxInFlight) layers of Hessians in flight)" }
      LaneStatus "hessian $t"
      Stage-Hessian $L
      LMark "hess-$t.ready" 'hessians on disk and checked, or not needed'
    } else {
      Wait-For { LHas "hess-$t.ready" } "the Hessians of $t (lane A)"
    }
    Check-Stop
    foreach ($k in $myKs) {
      LaneStatus "oracle K$k $t"
      Stage-Oracle $L @($k)
      LMark "k$k-$t.done" "lane $which"
    }
    Try-Verify $L
  }
  LaneStatus 'finished all layers'
  Write-Host "[lane $which] finished all layers"
}

function Start-Lane([string]$which, [string]$dev, [string]$flag = 'Lane') {
  $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath, '-Stage', 'all', '-Run', "-$flag", $which, '-Device', $dev)
  foreach ($name in $script:forward.Keys) {
    $v = $script:forward[$name]
    if ($v -is [switch]) { if ($v.IsPresent) { $a += "-$name" } } else { $a += @("-$name", $(if ("$v" -match '\s') { '"' + "$v" + '"' } else { "$v" })) }
  }
  $kind = $flag.ToLower()
  $out = Join-Path $logdir "$kind-$which.out.log"; $err = Join-Path $logdir "$kind-$which.err.log"
  $pr = Start-Process -FilePath 'powershell.exe' -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $pr.Handle   # PS 5.1: without the handle cached, ExitCode of a finished process reads as $null
  Write-Host "[supervisor] $kind $which started: pid $($pr.Id), device $dev, log $out"
  return $pr
}

function Run-Supervisor {
  if ($Stage -ne 'all') { throw '[pipeline] -Lanes 2 applies to -Stage all' }
  if ($propList.Count -gt 0) { throw '[pipeline] -Lanes 2 does not support -Propagate' }
  Stage-Corpus
  if (-not $Run) {
    Write-Host "`n[supervisor] PLAN for -Lanes 2: two processes, started detached, markers in $script:lanesDir, a failure in either stops both at a slice boundary."
    Run-Lane 'A'
    Run-Lane 'B'
    $script:Lane = ''; $script:Who = ''
    Stage-Alloc
    Stage-Convert
    return
  }
  New-Item -ItemType Directory -Force $logdir | Out-Null
  if (Test-Path $script:lanesDir) { Remove-Item -Recurse -Force $script:lanesDir }
  New-Item -ItemType Directory -Force $script:lanesDir | Out-Null
  $script:forward = [ordered]@{}
  foreach ($kv in $script:boundParams.GetEnumerator()) {
    if ($kv.Key -in 'Stage', 'Run', 'Lanes', 'Lane', 'Workers', 'Worker', 'Device') { continue }
    $script:forward[$kv.Key] = $kv.Value
  }
  $pa = Start-Lane 'A' $DeviceA
  $pb = Start-Lane 'B' $DeviceB
  $say = Get-Date
  while (-not ($pa.HasExited -and $pb.HasExited)) {
    Start-Sleep -Seconds 30
    foreach ($pr in @(@('A', $pa), @('B', $pb))) {
      $w, $proc = $pr
      if ($proc.HasExited -and $proc.ExitCode -notin 0, 3 -and -not (Test-Path $script:failedMarker)) {
        Write-Failed "lane $w process" "exited with code $($proc.ExitCode) without a FAILED marker (see $logdir\lane-$w.err.log)"
        LMark 'STOP' "lane $w process died"
      }
    }
    if (((Get-Date) - $say).TotalMinutes -ge 10) {
      $v = @(Get-ChildItem $script:lanesDir -Filter 'verify-*.done' -ErrorAction SilentlyContinue).Count
      Write-Host ("[supervisor] {0} layers verified; A: {1}; B: {2}" -f $v, (Get-Content (Lk 'lane-A.status') -ErrorAction SilentlyContinue), (Get-Content (Lk 'lane-B.status') -ErrorAction SilentlyContinue))
      $say = Get-Date
    }
  }
  $pa.WaitForExit(); $pb.WaitForExit()
  Write-Host "[supervisor] lane A exit $($pa.ExitCode), lane B exit $($pb.ExitCode)"
  if (Test-Path $script:failedMarker) { throw "[pipeline] a lane failed: $(Get-Content $script:failedMarker -TotalCount 1)" }
  if ($pa.ExitCode -ne 0 -or $pb.ExitCode -ne 0) { throw "[pipeline] a lane stopped without finishing (exit A $($pa.ExitCode), B $($pb.ExitCode))" }
  Stage-Alloc
  Stage-Convert
}


# ---- work queue (-Workers 2) ---------------------------------------------------------------------------------------
# Markers (in <Root>\lanes\, wiped by the supervisor): hess-LNN.ready (the chain), oracle-LNN.claim (CreateNew: one worker owns a layer's
# oracle job, K3 then K4), kK-LNN.done, verify-LNN.done, STOP, lane-G0.status / lane-G1.status. Everything else is re-derived from disk.
function Claim-Layer {
  foreach ($L in $layerList) {
    $t = LTag $L
    if (-not (LHas "hess-$t.ready") -or (LHas "verify-$t.done")) { continue }
    try {
      $fs = [IO.File]::Open((Lk "oracle-$t.claim"), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
      $b = [Text.Encoding]::ASCII.GetBytes("$($script:Who) $(Get-Date -Format 's')"); $fs.Write($b, 0, $b.Length); $fs.Close()
      return $L
    } catch { continue }   # claimed by the other worker
  }
  return $null
}
# One slice of layer $L's oracle job (K3 then K4); $true when every K is complete.
function Job-Slice([int]$L) {
  $t = LTag $L
  foreach ($k in $kList) {
    if (LHas "k$k-$t.done") { continue }
    $script:stepRc = 0
    LaneStatus "oracle K$k $t"
    Stage-Oracle $L @($k) $true
    if ($script:stepRc -eq 75) { return $false }
    LMark "k$k-$t.done" "worker $($script:Who)"
  }
  return $true
}
# The K manifests (weights_override.json) are merged from every chunk file of the directory; two workers finishing different layers
# must not write them at the same time, so the refresh is serialized by a machine-wide mutex (and the oracle runs with --no-manifest).
function Refresh-Manifests {
  $m = New-Object System.Threading.Mutex($false, ('Global\r4dx-q35moe-manifest-' + ($Root.ToLower() -replace '[^a-z0-9]', '_')))
  $got = $false
  try {
    try { $got = $m.WaitOne([TimeSpan]::FromMinutes(30)) } catch [System.Threading.AbandonedMutexException] { $got = $true }
    if (-not $got) { throw '[pipeline] timed out waiting for the manifest lock' }
    foreach ($k in $kList) {
      Step ("manifest-k{0}-{1}" -f $k, $script:Who) $OraclePython @($oracleScript, 'manifest', '--out-dir', (OracleDir $k), '--K', "$k", '--model-dir', $Model, '--chunk-experts', "$ChunkExperts") $false
    }
  } finally { if ($got) { $m.ReleaseMutex() } }
}
function Finish-Layer([int]$L) {
  $t = LTag $L
  Refresh-Manifests
  if (HessianFiles $L) { LaneStatus "verify $t"; Stage-Verify $L } else { Write-Host "[$($script:Who)] verify ${t}: Hessians already deleted, nothing to do" }
  LMark "verify-$t.done" "worker $($script:Who)"
}

function Run-Worker([string]$which) {
  $script:Worker = $which
  $script:Who = $which
  $script:LockName = if ($which -eq 'G0') { $LockB } else { $LockA }
  $script:Device = if ($which -eq 'G0') { $DeviceB } else { $DeviceA }
  $script:laneStart = Get-Date
  Write-Host "`n[worker $which] device $Device, lock $script:LockName, $(if ($which -eq 'G1') { 'Hessian chain first, then oracle jobs' } else { 'oracle jobs' }), layers $($layerList -join ',')"
  if ($which -eq 'G0') { Step 'gpu-probe-G0' $Python @((Join-Path $qdir 'gpu_probe.py'), '--min-free-gib', "$ProbeMinFreeGiB") $true }
  $chain = 0; $cur = $null; $idle = Get-Date
  while ($true) {
    Check-Stop
    if (@($layerList | Where-Object { -not (LHas ("verify-{0}.done" -f (LTag $_))) }).Count -eq 0) { break }
    if ($which -eq 'G1' -and $chain -lt $layerList.Count) {      # 1. the chain, with priority
      $gate = $true
      if ($chain -ge $script:maxInFlight) { $gate = LHas ("verify-{0}.done" -f (LTag $layerList[$chain - $script:maxInFlight])) }
      if ($gate) {
        $L = $layerList[$chain]; $t = LTag $L
        Write-Host "`n[worker G1] ===== Hessian chain: layer $L"
        LaneStatus "hessian $t"
        Stage-Hessian $L
        LMark "hess-$t.ready" 'hessians on disk and checked, or not needed'
        $chain++; $idle = Get-Date
        continue
      }
    }
    if ($null -eq $cur) {                                           # 2. an oracle job
      $cur = Claim-Layer
      if ($null -ne $cur) { Write-Host "`n[worker $which] ===== oracle job: layer $cur (K $($kList -join ', '))" }
    }
    if ($null -ne $cur) {
      if (Job-Slice $cur) { Finish-Layer $cur; $cur = $null }
      $idle = Get-Date
      continue
    }
    LaneStatus 'waiting for a layer to become ready'                # 3. nothing runnable
    if (((Get-Date) - $idle).TotalMinutes -ge 10) { Write-Host "[worker $which] still waiting for work"; $idle = Get-Date }
    Start-Sleep -Seconds $PollSeconds
  }
  LaneStatus 'finished'
  Write-Host "[worker $which] every layer verified"
}

function Run-WorkerSupervisor {
  if ($Stage -ne 'all') { throw '[pipeline] -Workers 2 applies to -Stage all' }
  if ($propList.Count -gt 0) { throw '[pipeline] -Workers 2 does not support -Propagate' }
  Stage-Corpus
  if (-not $Run) {
    Write-Host "`n[supervisor] PLAN for -Workers 2: two processes (G1 = device $DeviceA, lock $LockA; G0 = device $DeviceB, lock $LockB), markers in $script:lanesDir."
    Write-Host '[supervisor] Which worker runs which oracle job is decided at run time (the first free worker claims the lowest ready layer); shown here for G0.'
    $script:Who = 'G1'; $script:LockName = $LockA; $script:Device = $DeviceA
    Write-Host "`n[worker G1] Hessian chain (priority, at slice boundaries of its oracle jobs; at most $($script:maxInFlight) layers of Hessians in flight):"
    foreach ($L in $layerList) { Stage-Hessian $L }
    $script:Who = 'G0'; $script:LockName = $LockB; $script:Device = $DeviceB
    Write-Host "`n[worker G0] gpu probe, then whole-layer oracle jobs; after each layer: manifest refresh (mutex) + verify:"
    Step 'gpu-probe-G0' $Python @((Join-Path $qdir 'gpu_probe.py'), '--min-free-gib', "$ProbeMinFreeGiB") $true
    foreach ($L in $layerList) { Stage-Oracle $L $kList; Stage-Verify $L }
    $script:Who = ''
    Stage-Alloc
    Stage-Convert
    return
  }
  New-Item -ItemType Directory -Force $logdir | Out-Null
  if (Test-Path $script:lanesDir) { Remove-Item -Recurse -Force $script:lanesDir }
  New-Item -ItemType Directory -Force $script:lanesDir | Out-Null
  $script:forward = [ordered]@{}
  foreach ($kv in $script:boundParams.GetEnumerator()) {
    if ($kv.Key -in 'Stage', 'Run', 'Lanes', 'Lane', 'Workers', 'Worker', 'Device') { continue }
    $script:forward[$kv.Key] = $kv.Value
  }
  $p1 = Start-Lane 'G1' $DeviceA 'Worker'
  $p0 = Start-Lane 'G0' $DeviceB 'Worker'
  $say = Get-Date
  while (-not ($p1.HasExited -and $p0.HasExited)) {
    Start-Sleep -Seconds 30
    foreach ($pr in @(@('G1', $p1), @('G0', $p0))) {
      $w, $proc = $pr
      if ($proc.HasExited -and $proc.ExitCode -notin 0, 3 -and -not (Test-Path $script:failedMarker)) {
        Write-Failed "worker $w process" "exited with code $($proc.ExitCode) without a FAILED marker (see $logdir\worker-$w.err.log)"
        LMark 'STOP' "worker $w process died"
      }
    }
    if (((Get-Date) - $say).TotalMinutes -ge 10) {
      $v = @(Get-ChildItem $script:lanesDir -Filter 'verify-*.done' -ErrorAction SilentlyContinue).Count
      Write-Host ("[supervisor] {0} layers verified; G1: {1}; G0: {2}" -f $v, (Get-Content (Lk 'lane-G1.status') -ErrorAction SilentlyContinue), (Get-Content (Lk 'lane-G0.status') -ErrorAction SilentlyContinue))
      $say = Get-Date
    }
  }
  $p1.WaitForExit(); $p0.WaitForExit()
  Write-Host "[supervisor] worker G1 exit $($p1.ExitCode), worker G0 exit $($p0.ExitCode)"
  if (Test-Path $script:failedMarker) { throw "[pipeline] a worker failed: $(Get-Content $script:failedMarker -TotalCount 1)" }
  if ($p1.ExitCode -ne 0 -or $p0.ExitCode -ne 0) { throw "[pipeline] a worker stopped without finishing (exit G1 $($p1.ExitCode), G0 $($p0.ExitCode))" }
  $script:Who = 'supervisor'; Refresh-Manifests; $script:Who = ''
  Stage-Alloc
  Stage-Convert
}

# ---- dispatch ---------------------------------------------------------------------------------------
if (-not $Run) { Write-Host '[pipeline] PLAN ONLY: no -Run, nothing is executed' }
elseif (-not $Lane -and -not $Worker -and (Test-Path $script:failedMarker)) { Remove-Item $script:failedMarker -Force }   # a lane process must not delete the other lane's marker
Write-Host "[pipeline] root $Root, layers $($layerList -join ','), K in $($kList -join ','), cap $CapGib GiB per card, device $Device"
try {
if ($Lane) { Run-Lane $Lane } elseif ($Worker) { Run-Worker $Worker } elseif ($Workers -ge 2) { Run-WorkerSupervisor } elseif ($Lanes -ge 2) { Run-Supervisor } else {
switch ($Stage) {
  'corpus'  { Stage-Corpus }
  'hessian' { foreach ($L in $layerList) { Stage-Hessian $L } }
  'oracle'  { foreach ($L in $layerList) { Stage-Oracle $L } }
  'verify'  { foreach ($L in $layerList) { Stage-Verify $L } }
  'alloc'   { Stage-Alloc }
  'convert' { Stage-Convert }
  'budget'  { Stage-Budget }
  'all' {
    if (-not $Run) { Stage-Budget }
    Stage-Corpus
    foreach ($L in $layerList) {
      Write-Host "`n[pipeline] ===== layer $L"
      Stage-Hessian $L
      Stage-Oracle $L
      Stage-Verify $L
    }
    Stage-Alloc
    Stage-Convert
  }
}
}
} catch {
  if ($_.Exception.Message -eq 'LANE_STOP') { exit 3 }   # stopped on request, not a failure of this lane
  if ($Run -and -not $script:markerWritten) { Write-Failed "stage $Stage" ($_.Exception.Message) }
  throw
}
