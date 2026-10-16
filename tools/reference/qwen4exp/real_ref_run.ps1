<#
.SYNOPSIS
  Resumable, lock-per-step driver of the real-model qwen4_exp reference (plan M0-5 / M0-7): tools\reference\qwen4exp\real_ref.py.

.DESCRIPTION
  Jobsets (see REAL_REF.md): canon (4 canon segments x tie rules recent+hf), probe8k, probe32k, goldens, kvcalib.
  For each jobset: `real_ref.py prepare` (CPU, no lock) when its jobset json is missing, then a loop of bounded GPU steps.
  EVERY step is its own `tools\with_lock.ps1 -Name gpu` acquisition (device 1, HIP_VISIBLE_DEVICES=1), so other agents can use
  the GPU between steps. A step ends at a layer boundary after -MaxMinutes (default 11, under the ~15 minute rule), or after
  -MaxLayers layers, with the residuals of all its sequences checkpointed in <Root>\work\<jobset>\ (sha256 in state.json).
  Re-running the script continues where the last step stopped; finished jobsets return immediately.
  Exit code of a step: 0 jobset finished, 3 more steps needed, else error (the script stops).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\reference\qwen4exp\real_ref_run.ps1 -JobSet canon
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\reference\qwen4exp\real_ref_run.ps1 -JobSet all
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\reference\qwen4exp\real_ref_run.ps1 -JobSet probe32k -MaxMinutes 10
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("canon", "canon_p1", "truth_canon", "truth_probe8k", "truthw16_canon", "truthw16_probe8k", "truth_probe8k_hf", "probe8k", "probe32k", "probe8k_p1", "goldens", "kvcalib", "all")][string]$JobSet,
    [double]$MaxMinutes = 11,
    [int]$MaxLayers = 0,
    [int]$CkptEvery = 4,
    [double]$CacheGib = 10,
    [int]$MaxSteps = 400,
    [string]$Root = "E:\models\r4dx\q4exp\ref",
    [string]$Python = "D:\venvs\r4dx-qwen4-ref\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$lock = Join-Path $repo "tools\with_lock.ps1"
$script = Join-Path $PSScriptRoot "real_ref.py"
$prepareArg = @{ truth_canon = "truth"; truth_probe8k = "truth"; truthw16_canon = "truth"; truthw16_probe8k = "truth"; truth_probe8k_hf = "truth"; canon = "canon"; canon_p1 = "canon_p1"; probe8k = "probes"; probe32k = "probes"; probe8k_p1 = "probes"; goldens = "goldens"; kvcalib = "kvcalib" }
$order = if ($JobSet -eq "all") { @("goldens", "canon", "canon_p1", "probe8k", "probe8k_p1", "probe32k", "kvcalib") } else { @($JobSet) }

$env:HIP_VISIBLE_DEVICES = "1"
Set-Location $repo
foreach ($name in $order) {
    $jsonPath = Join-Path $Root "jobsets\$name.json"
    if (-not (Test-Path $jsonPath)) {
        Write-Host "[real_ref_run] prepare $($prepareArg[$name])"
        & $Python $script --root $Root prepare $prepareArg[$name]
        if ($LASTEXITCODE -ne 0) { Write-Error "prepare failed ($LASTEXITCODE)"; exit $LASTEXITCODE }
    }
    $extra = ""
    if ($MaxLayers -gt 0) { $extra = " --max-layers $MaxLayers" }
    $cmd = "$Python $script --root $Root step --jobset $name --max-minutes $MaxMinutes --ckpt-every $CkptEvery --cache-gib $CacheGib$extra"
    for ($n = 1; $n -le $MaxSteps; $n++) {
        Write-Host "[real_ref_run] ${name} step ${n} ($(Get-Date -Format 'HH:mm:ss')): waiting for the gpu lock"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $lock -Name gpu -Run $cmd
        $rc = $LASTEXITCODE
        if ($rc -eq 0) { Write-Host "[real_ref_run] ${name} finished after $n step(s)"; break }
        if ($rc -ne 3) { Write-Error "[real_ref_run] ${name} step $n failed with exit code $rc"; exit $rc }
        if ($MaxLayers -gt 0) { Write-Host "[real_ref_run] ${name}: -MaxLayers given, stopping after one step"; break }
    }
}
exit 0
