<#
.SYNOPSIS
  Run a command line while holding a machine-wide named mutex, so parallel agents do not collide.

.DESCRIPTION
  -Name gpu    every process that touches a GPU (tests, tools, reference scripts with --device cuda)
  -Name build  every cmake configure/build (one build tree, build\win-hip)
  -Run         the command line, executed with `cmd /c` in the caller's cwd.
  Waits up to -TimeoutMinutes for the lock (default 240) and returns the command's exit code
  (75 if the lock timed out). The mutex is released even if the command fails or this process dies.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name build -Run "powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1"
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\with_lock.ps1 -Name gpu -Run "build\win-hip\tests\model\qwen4exp\test_gpu_qsa.exe"
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("gpu", "build")][string]$Name,
    [Parameter(Mandatory = $true)][string]$Run,
    [int]$TimeoutMinutes = 240
)
$ErrorActionPreference = "Stop"

$m = New-Object System.Threading.Mutex($false, "Global\r4dx-lock-$Name")
$got = $false
try {
    try { $got = $m.WaitOne([TimeSpan]::FromMinutes($TimeoutMinutes)) }
    catch [System.Threading.AbandonedMutexException] { $got = $true }
    if (-not $got) { Write-Error "with_lock.ps1: timed out waiting for lock '$Name'"; exit 75 }
    & cmd.exe /c $Run
    exit $LASTEXITCODE
}
finally {
    if ($got) { $m.ReleaseMutex() }
}
