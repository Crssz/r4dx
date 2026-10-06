# tools/prefill/warm_delta.ps1 -- starts r4dx-server on one device (or -Tp 2), runs warm_delta.py
# (cold 32k/64k prompt, then a ~4k-token user turn appended through prefix reuse) and stops the server.
#
#   .\tools\prefill\warm_delta.ps1 -Device 1 -OutDir <models-root>\r4dx\prefill-m0\profile\warm-tp1
param(
  [int]$Device = 1,
  [ValidateSet(1, 2)][int]$Tp = 1,
  [string]$Bases = '32k,64k',
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx",
  [string]$Layout = 'trellis',
  [Parameter(Mandatory = $true)][string]$OutDir,
  [int]$Port = 8094,
  [int]$MaxCtx = 73728,
  [string]$Server = '',
  [string[]]$ServerArgs = @(),
  [string]$Python = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Server) { $Server = Join-Path $repo 'build\win-hip\src\server\r4dx-server.exe' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$argList = @('--model', $Model, '--layout', $Layout, '--port', "$Port", '--max-ctx', "$MaxCtx", '--vision', 'off')
if ($Tp -eq 2) { $env:HIP_VISIBLE_DEVICES = '0,1'; $argList += @('--tp', '2') } else { $env:HIP_VISIBLE_DEVICES = "$Device" }
$argList += $ServerArgs
$errLog = Join-Path $OutDir 'server.err.log'
Write-Output "[warm] HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES) $Server $($argList -join ' ')"
$proc = Start-Process -FilePath $Server -ArgumentList $argList -NoNewWindow -PassThru `
  -RedirectStandardOutput (Join-Path $OutDir 'server.out.log') -RedirectStandardError $errLog
$rc = 1
try {
  $t0 = Get-Date
  while ($true) {
    if ($proc.HasExited) { throw "[warm] server exited early (code $($proc.ExitCode)); see $errLog" }
    try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$Port/health").StatusCode -eq 200) { break } } catch { }
    if (((Get-Date) - $t0).TotalSeconds -gt 900) { throw '[warm] server not healthy after 900 s' }
    Start-Sleep -Seconds 3
  }
  & $Python (Join-Path $PSScriptRoot 'warm_delta.py') --port $Port --bases $Bases --out $OutDir
  $rc = $LASTEXITCODE
} finally {
  if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -Confirm:$false }
  Write-Output "[warm] server stopped; log $errLog"
}
exit $rc
