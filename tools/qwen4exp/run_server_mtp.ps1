param(
    [string]$Container = "E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx",
    [int]$Mtp = 1,
    [string]$Host = "127.0.0.1",
    [int]$Port = 8080,
    [int]$MaxCtx = 32768
)

$ErrorActionPreference = "Stop"

# Ensure both AMD GPUs are accessible
Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue

$serverPath = Join-Path $PSScriptRoot "..\..\build\win-hip\src\server\r4dx-server.exe"

$arguments = @(
    "--model", $Container,
    "--layout", "trellis",
    "--mtp", $Mtp,
    "--host", $Host,
    "--port", $Port,
    "--max-ctx", $MaxCtx
)

Write-Host "[r4dx-server MTP] Starting server on http://$Host`:$Port with --mtp $Mtp..." -ForegroundColor Green
& $serverPath $arguments
