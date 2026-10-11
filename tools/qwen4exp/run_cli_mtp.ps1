param(
    [string]$Container = "E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx",
    [int]$Mtp = 1,
    [string]$Prompt = "Explain quantum computing in three short bullet points.",
    [int]$MaxTokens = 256,
    [switch]$Chat
)

$ErrorActionPreference = "Stop"

# Ensure both AMD GPUs are accessible
Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue

$cliPath = Join-Path $PSScriptRoot "..\..\build\win-hip\src\cli\r4dx-cli.exe"

$arguments = @(
    "--model", $Container,
    "--layout", "trellis",
    "--mtp", $Mtp,
    "--max-tokens", $MaxTokens,
    "--temperature", "0",
    "--think", "off",
    "--stats"
)

if ($Chat) {
    $arguments += "--chat"
} else {
    $arguments += @("--prompt", $Prompt)
}

Write-Host "[r4dx-cli MTP] Starting with --mtp $Mtp on dual R9700 cards..." -ForegroundColor Cyan
& $cliPath $arguments
