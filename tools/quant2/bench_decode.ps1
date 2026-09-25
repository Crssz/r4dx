# quant2 decode benchmark (HIP device 1): the Milestone 11 protocol from docs/perf.md -- the
# standard haiku prompt, --vision off --think off --temperature 0 --max-tokens 256 --max-ctx 2048
# --stats, one fresh process per run, containers INTERLEAVED run by run so drift hits all of them
# alike. Each run does plain decode and --dflash k=7. Writes bench.json and bench.md to -OutDir.
#
#   .\tools\quant2\bench_decode.ps1 -Container v6=D:\models\r4dx\qwen38-27b-v6.r4dx,q1full=D:\...\q1full.r4dx
#
# Each entry is name=path or name=path@binary (a different r4dx-cli.exe, e.g. main's build, to check
# that the branch binary is speed-neutral on an unrotated container).
param(
  [Parameter(Mandatory = $true)][string[]]$Container,
  [int]$Runs = 3,
  [string]$Dflash = 'D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx',
  [string]$OutDir = 'D:\models\r4dx\quant2-bench',
  [string]$Layout = 'w4a16',
  [switch]$NoDflash
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[bench] stop r4dx-server first' }
$env:HIP_VISIBLE_DEVICES = '1'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$defaultCli = Join-Path $repo 'build\win-hip\src\cli\r4dx-cli.exe'
$prompt = 'Write a haiku about GPUs, then explain what a GPU is in two sentences.'

$entries = foreach ($c in $Container) {
  $name, $rest = $c -split '=', 2
  $path, $bin = $rest -split '@', 2
  if (-not $bin) { $bin = $defaultCli }
  if (-not (Test-Path $path)) { throw "[bench] ${name}: container $path not found" }
  [pscustomobject]@{ name = $name; path = $path; bin = $bin }
}
$modes = @('plain') + $(if ($NoDflash) { @() } else { @('dflash7') })
$results = @()

for ($r = 1; $r -le $Runs; $r++) {
  foreach ($e in $entries) {
    foreach ($m in $modes) {
      $log = Join-Path $OutDir ("{0}_{1}_run{2}.log" -f $e.name, $m, $r)
      $args = @('--model', $e.path, '--layout', $Layout, '--prompt', $prompt, '--vision', 'off',
                '--think', 'off', '--temperature', '0', '--max-tokens', '256', '--max-ctx', '2048', '--stats')
      if ($m -eq 'dflash7') { $args += @('--dflash', $Dflash, '--dflash-k', '7') }
      # PS 5.1 makes native stderr lines (the --stats output) error records; under 'Stop' the first
      # one would abort. Stringify everything and judge by the exit code.
      $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
      try { $out = @(& $e.bin @args 2>&1 | ForEach-Object { "$_" }) } finally { $ErrorActionPreference = $prev }
      $code = $LASTEXITCODE
      $out | Set-Content -Encoding utf8 $log
      $text = ($out | Where-Object { $_ -notmatch '^\[' -and $_ -notmatch 'NativeCommandError|CategoryInfo|^\s*At |^\s*\+ ' }) -join "`n"
      $row = [ordered]@{ name = $e.name; mode = $m; run = $r; exit = $code; decode_tok_s = $null; tokens = $null;
                         accept_pct = $null; tok_per_round = $null; vram_gib = $null
                         text_sha256 = [BitConverter]::ToString(
                           [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '').Substring(0, 16) }
      $s = ($out | Out-String)
      if ($s -match 'decode: (\d+) tok in [\d.]+s \(([\d.]+) tok/s\).*VRAM: ([\d.]+) GiB') {
        $row.tokens = [int]$Matches[1]; $row.decode_tok_s = [double]$Matches[2]; $row.vram_gib = [double]$Matches[3]
      }
      if ($s -match '\[stats\] dflash: .*\(([\d.]+)% acceptance, ([\d.]+) tok/round') {
        $row.accept_pct = [double]$Matches[1]; $row.tok_per_round = [double]$Matches[2]
      }
      $results += [pscustomobject]$row
      Write-Host ("[bench] run {0} {1,-10} {2,-8} {3,7} tok/s  {4} tok  exit {5}" -f $r, $e.name, $m, $row.decode_tok_s, $row.tokens, $code)
      $results | ConvertTo-Json -Depth 3 | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.json')
    }
  }
}

$md = @('| container | mode | decode tok/s (runs) | mean | tokens | dflash acc % / tok-round | text |', '|---|---|---|--:|--:|---|---|')
foreach ($e in $entries) {
  foreach ($m in $modes) {
    $rs = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m })
    $v = @($rs | ForEach-Object { $_.decode_tok_s })
    $mean = if ($v.Count) { [math]::Round((($v | Measure-Object -Average).Average), 2) } else { $null }
    $shas = @($rs | ForEach-Object { $_.text_sha256 } | Sort-Object -Unique)
    $acc = if ($m -eq 'dflash7') { "{0} / {1}" -f $rs[0].accept_pct, $rs[0].tok_per_round } else { '--' }
    $md += "| $($e.name) | $m | $($v -join ', ') | $mean | $($rs[0].tokens) | $acc | $(if ($shas.Count -eq 1) { $shas[0] } else { 'VARIES' }) |"
  }
}
$md | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.md')
$md
