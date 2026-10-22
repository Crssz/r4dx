<#
.SYNOPSIS
  Runs tools\reference\qwen35moe\ref_run.py in GPU-lock slices until it finishes (exit 0), re-taking the lock after every exit 75.

.DESCRIPTION
  Each slice is one `tools\with_lock.ps1 -Name <Lock> -Run "set HIP_VISIBLE_DEVICES=<Device>&& python ref_run.py <RefArgs> --budget-seconds <Seconds>"`,
  so somebody else's short GPU test can run between two slices (ref_run.py saves its residuals after the layer that crosses the budget).
  The reference stack needs one card: -Device 1 / -Lock gpu is the pipeline's convention; r4dx itself needs BOTH cards, so never run this
  while r4dx is loaded.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\reference\qwen35moe\run_slices.ps1 `
    -RefArgs "kvcalib --tokens E:\models\r4dx\q35moe\tokens_kvcalib.json --out E:\models\r4dx\q35moe\kvcalib.json --work E:\models\r4dx\q35moe\work\kvcalib"
#>
param(
    [Parameter(Mandatory = $true)][string]$RefArgs,
    [string]$Lock = 'gpu',
    [string]$Device = '1',
    [int]$Seconds = 840,
    [int]$MaxSlices = 200
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$py = 'D:\venvs\r4dx-qwen4-ref\Scripts\python.exe'
$tool = Join-Path $PSScriptRoot 'ref_run.py'
$wl = Join-Path $repo 'tools\with_lock.ps1'
for ($n = 1; $n -le $MaxSlices; $n++) {
    $inner = "set HIP_VISIBLE_DEVICES=$Device&& `"$py`" -X faulthandler `"$tool`" $RefArgs --budget-seconds $Seconds"
    Write-Host ("[run_slices] slice {0} {1:u}" -f $n, (Get-Date))
    & powershell -NoProfile -ExecutionPolicy Bypass -File $wl -Name $Lock -Run $inner
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { Write-Host "[run_slices] finished after $n slice(s)"; exit 0 }
    if ($rc -ne 75) { Write-Host "[run_slices] ref_run.py failed with exit code $rc"; exit $rc }
}
Write-Host "[run_slices] gave up after $MaxSlices slices"
exit 1
