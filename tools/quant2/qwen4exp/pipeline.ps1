# qwen4_exp (Qwen3.8-Flash-Next) trellis pipeline: calibration corpus -> per-layer Hessians -> oracles at KB 3 and KB 4 ->
# allocation under the per-card budget -> container (docs/qwen4exp-plan.md 4.4, 4.11, M1-10 .. M1-14). Mirrors
# tools\gemma\trellis_pipeline.ps1: without -Run NOTHING executes, every stage prints the exact commands it would run.
#
#   .\tools\quant2\qwen4exp\pipeline.ps1 -Stage all -Layers 0..47            # print the whole chain
#   .\tools\quant2\qwen4exp\pipeline.ps1 -Stage corpus -Run                  # CPU, ~1 min -> tokens_calib.json
#   .\tools\quant2\qwen4exp\pipeline.ps1 -Stage hessian -Layers 0..3 -Run    # GPU, one bounded slice per lock
#   .\tools\quant2\qwen4exp\pipeline.ps1 -Stage all -Layers 0..47 -Run       # the real thing (hours; resumable, see below)
#
# Stages
#   corpus   CPU  corpus_q4.py build              tokens_calib.json (hessian-v2 + the SFT slice, canonical tokenizer, ~1M tokens)
#   hessian  GPU  hessian_q4.py capture --layer L Hessians + routing counts + fallback of layer L (and the layer L+1 residual)
#   oracle   GPU  oracle_q4.py quantize --layer L --K 3 | 4       one oracle per K, chunks of 32 experts, manifest refreshed
#   verify   CPU  oracle_q4.py verify --layer L --delete-hessians  both K dirs verified (sha256, shapes, reconstruction), then the
#                                                                  layer's .hess files deleted (8.4 GB each)
#   alloc    CPU  alloc_q4.py allocate            mix manifest + allocation report under the 26 GiB per card cap
#   convert  CPU  r4dx-convert --trellis-from <mix> ...            E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx
#   all      corpus, then per layer hessian -> oracle K3 -> oracle K4 -> verify (layer-major: only one layer's Hessians exist
#            at a time), then alloc, convert.  `-Stage hessian|oracle|verify -Layers a..b` run just that step for those layers.
#
# GPU discipline (docs/qwen4exp-plan.md 4.11 item 1): every GPU step is ONE process under
#   tools\with_lock.ps1 -Name gpu -Run "<cmd>"
# with HIP_VISIBLE_DEVICES=1; a step is bounded by -BudgetSeconds (default 840 = 14 min of work) and exits 75 when it has more
# to do: this driver then releases the lock (other agents get the GPU) and takes it again for the next slice. State between
# slices lives under <Root>\work (hessian accumulators, residual boundaries) and in the oracle chunk files, with content hashes.
# Everything is resumable: re-run the same command.
#
# Python: -Python (the reference venv: transformers with qwen4_exp) for corpus / hessian / alloc; -OraclePython (torch with MAGMA
# cholesky on the GPU) for the oracle and verify, <models-root>\r4dx\huihui\venv-rocm10 as for the 27B oracle.
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('corpus', 'hessian', 'oracle', 'verify', 'alloc', 'convert', 'all')]
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
  [string]$Root = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\q4exp",
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ",
  [string]$Sidecar = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\swift15-flash-next.r4dx-ngram",
  [string]$Tokens = '',
  [string]$KvCalib = '',
  [string]$Output = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\swift15-flash-next-trellis-mix.r4dx",
  [string]$Python = 'D:\venvs\r4dx-qwen4-ref\Scripts\python.exe',
  [string]$OraclePython = '',
  [string]$Exe = '',
  [string]$Device = '1',
  [string]$Device0Ok = '',
  [int]$Threads = 32
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$env:PYTHONIOENCODING = 'utf-8'

# ---- paths ------------------------------------------------------------------------------------------
$qdir = $PSScriptRoot
if (-not $Tokens) { $Tokens = Join-Path $Root 'tokens_calib.json' }
if (-not $KvCalib) { $KvCalib = Join-Path $Root 'kvcalib.json' }
$venvRocm10 = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui\venv-rocm10\Scripts\python.exe"
if (-not $OraclePython) { $OraclePython = if (Test-Path $venvRocm10) { $venvRocm10 } else { $Python } }
if (-not $Exe) {
  foreach ($c in 'build\win-hip\src\convert\r4dx-convert.exe', 'build\win-hip-dry\src\convert\r4dx-convert.exe') {
    if (Test-Path (Join-Path $repo $c)) { $Exe = Join-Path $repo $c; break }
  }
  if (-not $Exe) { $Exe = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe' }
}
$logdir = Join-Path $Root 'logs'
$hessScript = 'tools\reference\qwen4exp\hessian_q4.py'
$allocScript = 'tools\reference\qwen4exp\alloc_q4.py'
$corpusScript = Join-Path $qdir 'corpus_q4.py'
$oracleScript = Join-Path $qdir 'oracle_q4.py'
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

# ---- helpers ----------------------------------------------------------------------------------------
function Q([string]$a) { if ($a -match '^[A-Za-z0-9_\-\.\\:/=,]+$') { $a } else { '"' + $a.Replace('"', '\"') + '"' } }
function Fmt([string]$exe, [string[]]$a) { (@((Q $exe)) + ($a | ForEach-Object { Q $_ })) -join ' ' }
function Need([bool]$ok, [string]$msg) { if (-not $ok) { $script:problems += $msg } }
function FreeGiB([string]$path) {
  $r = [IO.Path]::GetPathRoot($path)
  return [math]::Round((Get-PSDrive $r.Substring(0, 1)).Free / 1GB, 1)
}
function HessianDone([int]$L) { Test-Path (Join-Path $Root ("hessian\L{0:D2}\DONE.json" -f $L)) }

function Open-Gpu {
  if ($Device -notmatch '^[01]$') { $script:problems += "-Device must be 0 or 1, got '$Device'" }
  if ($Device -eq '0' -and -not $Device0Ok) { $script:problems += 'device 0 drives the desktop: the reference tools refuse it; pass -Device0Ok yes only with the user''s go-ahead (sets R4DX_ALLOW_GPU0)' }
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
function Step([string]$name, [string]$exe, [string[]]$a, [bool]$gpu) {
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
  if (-not $Run) { return }
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
  if ($rc -ne 0) { throw "[pipeline] $name failed (exit $rc) after $min min, see $log" }
  Write-Host "[pipeline] $name done in $min min (log $log)"
}

# ---- stages -----------------------------------------------------------------------------------------
function Stage-Corpus {
  Write-Host "`n[pipeline] ===== corpus (CPU, ~1-2 min)"
  if ((Test-Path $Tokens) -and -not $Force) { Write-Host "[pipeline] corpus: $Tokens exists, skipping (-Force redoes it)"; return }
  Need (Test-Path (Join-Path $Root 'sft\qwen38-agent-sft-combined-100pct.parquet')) "no SFT parquet: hf download ukisai/Qwen3.8-27B-multi-turn-agent-sft --repo-type dataset --local-dir $Root\sft"
  Need (Test-Path 'E:\models\r4dx\corpus-v2\samples.jsonl') 'no corpus-v2 samples.jsonl'
  Flush-Problems 'corpus'
  Step 'corpus' $Python @($corpusScript, 'build', '--out', $Tokens) $false
}

function Stage-Hessian([int]$L) {
  if ($propList -contains $L) {
    Open-Gpu
    Need (Test-Path $Tokens) "no $Tokens (run -Stage corpus)"
    Flush-Problems "propagate-L$L"
    Step "propagate-L$L" $Python @($hessScript, 'capture', '--layer', "$L", '--root', $Root, '--tokens', $Tokens, '--model-dir', $Model,
         '--ngram-sidecar', $Sidecar, '--budget-seconds', "$BudgetSeconds", '--propagate-only') $true
    return
  }
  Open-Gpu
  Need (Test-Path $Tokens) "no $Tokens (run -Stage corpus)"
  Need ((FreeGiB $Root) -ge 80) "less than 80 GiB free on $Root (a layer step needs two 20 GB residual boundaries + 14 GB of accumulators + 8.4 GB of Hessians)"
  if ((HessianDone $L) -and -not $Force) { Write-Host "[pipeline] hessian L${L}: DONE already, skipping (-Force redoes it)"; Flush-Problems "hessian-L$L"; return }
  Flush-Problems "hessian-L$L"
  $a = @($hessScript, 'capture', '--layer', "$L", '--root', $Root, '--tokens', $Tokens, '--model-dir', $Model,
         '--ngram-sidecar', $Sidecar, '--budget-seconds', "$BudgetSeconds")
  if ($Force) { $a += '--force' }
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
         '--ngram-sidecar', $Sidecar,
         '--layouts', 'w4a16', '--no-bf16', '--lm-head', 'w4a16', '--w4a16-group-rule', '^lm_head$=32',
         '--kv-calib', $KvCalib, '--threads', "$Threads")
  Need (Test-Path $Exe) "converter not found at $Exe (build target r4dx-convert through tools\with_lock.ps1 -Name build)"
  Need (Test-Path $manifest) "no mix manifest $manifest (run -Stage alloc)"
  Need (Test-Path $Sidecar) "no n-gram sidecar $Sidecar"
  Need (Test-Path $KvCalib) "no fp8 KV calibration $KvCalib (M1-15; without --kv-calib every descale is the placeholder 1.0)"
  Need ((FreeGiB (Split-Path -Parent $Output)) -ge 60) "less than 60 GiB free next to $Output"
  if ((Test-Path $Output) -and -not $Force) { Write-Host "[pipeline] convert: $Output exists, skipping (-Force redoes it)"; return }
  Flush-Problems 'convert'
  Step 'convert' $Exe $a $false
  if ($Run) { & $Python (Join-Path $repo 'tools\quant2\decode_bytes.py') $Output --by-class }
}

# ---- dispatch ---------------------------------------------------------------------------------------
if (-not $Run) { Write-Host '[pipeline] PLAN ONLY: no -Run, nothing is executed' }
Write-Host "[pipeline] root $Root, layers $($layerList -join ','), K in $($kList -join ','), cap $CapGib GiB per card, device $Device"
switch ($Stage) {
  'corpus'  { Stage-Corpus }
  'hessian' { foreach ($L in $layerList) { Stage-Hessian $L } }
  'oracle'  { foreach ($L in $layerList) { Stage-Oracle $L } }
  'verify'  { foreach ($L in $layerList) { Stage-Verify $L } }
  'alloc'   { Stage-Alloc }
  'convert' { Stage-Convert }
  'all' {
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
