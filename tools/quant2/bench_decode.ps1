# quant2 decode benchmark (HIP device 1): the Milestone 11 protocol from docs/perf.md --
# --vision off --think off --temperature 0 --max-tokens 256 --max-ctx 2048 --stats, one fresh process
# per run, containers INTERLEAVED run by run so drift hits all of them alike. Each run does plain decode
# and --dflash k=7 over every prompt in -PromptFile (default: tests/model/mtp_prompts.txt, whose first
# line is the standard haiku prompt). Writes bench.json and bench.md to -OutDir.
#
#   .\tools\quant2\bench_decode.ps1 -Container v6=D:\models\r4dx\qwen38-27b-v6.r4dx,q1full=D:\...\q1full.r4dx
#
# Each entry is name=path or name=path@binary (a different r4dx-cli.exe, e.g. main's build, to check
# that the branch binary is speed-neutral on an unrotated container).
#
# Decode speed on this box is sensitive to host CPU load (a concurrent r4dx-convert moved v6's plain
# decode by -6%), so the script first waits until no r4dx-convert / hessian capture is running.
param(
  [Parameter(Mandatory = $true)][string[]]$Container,
  [int]$Runs = 3,
  [string]$Dflash = 'D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx',
  [string]$OutDir = 'D:\models\r4dx\quant2-bench',
  [string]$Layout = 'w4a16',
  [string]$PromptFile = '',
  [switch]$NoDflash,
  [switch]$NoIdleWait
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[bench] stop r4dx-server first' }
$env:HIP_VISIBLE_DEVICES = '1'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$defaultCli = Join-Path $repo 'build\win-hip\src\cli\r4dx-cli.exe'
if (-not $PromptFile) { $PromptFile = Join-Path $repo 'tests\model\mtp_prompts.txt' }
$prompts = @(Get-Content $PromptFile | Where-Object { $_.Trim() })

if (-not $NoIdleWait) {
  while (Get-Process r4dx-convert -ErrorAction SilentlyContinue) {
    Write-Host '[bench] waiting for r4dx-convert to finish (CPU load skews decode timing)'
    Start-Sleep 30
  }
}

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
    for ($p = 0; $p -lt $prompts.Count; $p++) {
      foreach ($m in $modes) {
        $log = Join-Path $OutDir ("{0}_{1}_p{2}_run{3}.log" -f $e.name, $m, $p, $r)
        $args = @('--model', $e.path, '--layout', $Layout, '--prompt', $prompts[$p], '--vision', 'off',
                  '--think', 'off', '--temperature', '0', '--max-tokens', '256', '--max-ctx', '2048', '--stats')
        if ($m -eq 'dflash7') { $args += @('--dflash', $Dflash, '--dflash-k', '7') }
        # PS 5.1 makes native stderr lines (the --stats output) error records; under 'Stop' the first
        # one would abort. Stringify everything and judge by the exit code.
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        try { $out = @(& $e.bin @args 2>&1 | ForEach-Object { "$_" }) } finally { $ErrorActionPreference = $prev }
        $code = $LASTEXITCODE
        $out | Set-Content -Encoding utf8 $log
        $text = ($out | Where-Object { $_ -notmatch '^\[' -and $_ -notmatch 'NativeCommandError|CategoryInfo|^\s*At |^\s*\+ ' }) -join "`n"
        $row = [ordered]@{ name = $e.name; mode = $m; prompt = $p; run = $r; exit = $code; decode_tok_s = $null;
                           tokens = $null; decode_s = $null; accept_pct = $null; tok_per_round = $null; vram_gib = $null
                           text_sha256 = [BitConverter]::ToString(
                             [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '').Substring(0, 16) }
        $s = ($out | Out-String)
        if ($s -match 'decode: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\).*VRAM: ([\d.]+) GiB') {
          $row.tokens = [int]$Matches[1]; $row.decode_s = [double]$Matches[2]
          $row.decode_tok_s = [double]$Matches[3]; $row.vram_gib = [double]$Matches[4]
        }
        if ($s -match '\[stats\] dflash: .*\(([\d.]+)% acceptance, ([\d.]+) tok/round') {
          $row.accept_pct = [double]$Matches[1]; $row.tok_per_round = [double]$Matches[2]
        }
        $results += [pscustomobject]$row
        Write-Host ("[bench] run {0} {1,-10} p{2} {3,-8} {4,7} tok/s  {5} tok  exit {6}" -f $r, $e.name, $p, $m, $row.decode_tok_s, $row.tokens, $code)
        ConvertTo-Json -InputObject @($results) -Depth 3 | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.json')
      }
    }
  }
}

# Per container/mode: per-prompt mean tok/s, and the token-weighted aggregate (sum tokens / sum decode
# seconds over every prompt and run) -- the number to compare, since it does not over-weight short texts.
$hdr = '| container | mode | ' + (($prompts | ForEach-Object -Begin { $i = 0 } -Process { "p$i"; $i++ }) -join ' | ') + ' | aggregate tok/s | text stable |'
$md = @($hdr, ('|---|---|' + ('--:|' * ($prompts.Count + 1)) + '---|'))
foreach ($e in $entries) {
  foreach ($m in $modes) {
    $cells = @()
    $stable = $true
    for ($p = 0; $p -lt $prompts.Count; $p++) {
      $rs = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m -and $_.prompt -eq $p -and $_.decode_tok_s })
      $mean = if ($rs.Count) { [math]::Round((($rs | ForEach-Object { $_.decode_tok_s } | Measure-Object -Average).Average), 2) } else { '--' }
      $acc = if ($m -eq 'dflash7' -and $rs.Count) { " ($($rs[0].accept_pct)%)" } else { '' }
      $cells += "$mean$acc"
      if (@($rs | ForEach-Object { $_.text_sha256 } | Sort-Object -Unique).Count -gt 1) { $stable = $false }
    }
    $all = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m -and $_.decode_s })
    $tok = ($all | Measure-Object tokens -Sum).Sum
    $sec = ($all | Measure-Object decode_s -Sum).Sum
    $agg = if ($sec) { [math]::Round($tok / $sec, 2) } else { '--' }
    $md += "| $($e.name) | $m | $($cells -join ' | ') | **$agg** | $stable |"
  }
}
$md | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.md')
$md
