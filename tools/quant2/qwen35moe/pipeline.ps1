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
  [double]$MinFreeGiB = 40               # free space a layer step needs under -Root (disk_budget.py: ~32 GiB of scratch + margin)
)
$ErrorActionPreference = 'Stop'
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
    $line = ("{0} FAILED step={1}: {2}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $step, ($msg -replace '[\r\n]+', ' '))
    if ($line.Length -gt 600) { $line = $line.Substring(0, 600) + '...' }
    if ($log) { $line += "  [log $log]" }
    Set-Content -Path $script:failedMarker -Value $line -Encoding ascii
    $script:markerWritten = $true
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
function OracleComplete([int]$L) {
  $a = @($oracleScript, 'status', '--layer', "$L", '--model-dir', $Model, '--hessian-root', $Root)
  foreach ($k in $kList) { $a += @('--out-dir', ((OracleDir $k) + "@$k")) }
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
function Step([string]$name, [string]$exe, [string[]]$a, [bool]$gpu, [bool]$readOnly = $false) {
  $inner = Fmt $exe $a
  if ($gpu) {
    $pre = "set HIP_VISIBLE_DEVICES=$Device&& "
    if ($Device -eq '0') { $pre += 'set R4DX_ALLOW_GPU0=1&& ' }
    $wl = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $repo 'tools\with_lock.ps1'), '-Name', 'gpu', '-Run', ($pre + $inner))
    $shown = "powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name gpu -Run `"$pre$inner`""
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
      if ($rc -eq 75 -and $gpu) { Write-Host ("[pipeline] {0}: slice done, more to do ({1:N1} min so far), lock released" -f $name, ((Get-Date) - $t0).TotalMinutes); continue }
      break
    }
  } finally { $ErrorActionPreference = $prev }
  $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
  if ($rc -ne 0) {
    Write-Failed $name ("exit $rc after $min min: " + (LastErrorLine $log)) $log
    throw "[pipeline] $name failed (exit $rc) after $min min, see $log"
  }
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

function Stage-Oracle([int]$L) {
  if ($propList -contains $L) { return }
  Open-Gpu
  Need (Test-Path $OraclePython) "no oracle python at $OraclePython"
  if (-not (HessianDone $L) -and $Run) { $script:problems += "layer $L has no Hessians (run -Stage hessian)" }
  Flush-Problems "oracle-L$L"
  foreach ($k in $kList) {
    $a = @($oracleScript, 'quantize', '--layer', "$L", '--K', "$k", '--out-dir', (OracleDir $k), '--hessian-root', $Root,
           '--model-dir', $Model, '--chunk-experts', "$ChunkExperts", '--cohort', "$Cohort", '--budget-seconds', "$BudgetSeconds")
    if ($Force) { $a += '--force' }
    Step "oracle-k$k-L$L" $OraclePython $a $true
  }
}

function Stage-Verify([int]$L) {
  if ($propList -contains $L) { return }
  $a = @($oracleScript, 'verify', '--layer', "$L", '--model-dir', $Model, '--hessian-root', $Root, '--delete-hessians')
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

# ---- dispatch ---------------------------------------------------------------------------------------
if (-not $Run) { Write-Host '[pipeline] PLAN ONLY: no -Run, nothing is executed' }
elseif (Test-Path $script:failedMarker) { Remove-Item $script:failedMarker -Force }
Write-Host "[pipeline] root $Root, layers $($layerList -join ','), K in $($kList -join ','), cap $CapGib GiB per card, device $Device"
try {
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
} catch {
  if ($Run -and -not $script:markerWritten) { Write-Failed "stage $Stage" ($_.Exception.Message) }
  throw
}
