# quant2 decode benchmark (HIP device 1): the Milestone 11 protocol from docs/perf.md --
# --vision off --think off --temperature 0 --max-tokens 256 --max-ctx 2048 --stats, one fresh process
# per run, containers INTERLEAVED run by run so drift hits all of them alike. Each run does every mode
# of -Modes over every prompt in -PromptFile (default: tests/model/mtp_prompts.txt, whose first line is
# the standard haiku prompt), then one prefill run over -PrefillPromptFile. Writes bench.json and
# bench.md to -OutDir.
#
#   .\tools\quant2\bench_decode.ps1 -Container `
#       huihui=D:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx#trellis
#   (-Layout defaults to trellis, the production container's layout, so #trellis is optional; a w4a16
#   container needs #w4a16. Earlier records compared base w4a16 / K4m / mix4.5m containers, all
#   retired with the base checkpoint -- docs/huihui.md.)
#
# Each entry is name=path[@binary][#layout]: @binary runs a different r4dx-cli.exe (e.g. main's build,
# to check that the branch binary is speed-neutral on an unrotated container), #layout its own
# --layout (default -Layout), so one run can interleave w4a16 and trellis containers
# (docs/trellis-kernel.md 1, A3 / A3p). The first entry, or -Baseline, is what the tables compare
# against.
#
# Tensor parallel: an entry may end in !<tp> (name=path[@binary][#layout]!2), or -Tp sets the default
# for every entry (1). A TP=1 entry runs on HIP device 1 as before; a --tp 2 entry passes --tp 2
# --tp-mode real with HIP_VISIBLE_DEVICES unset (docs/tp.md 9.2: rank 0 on device 1, rank 1 on
# device 0), and bench.json gets each rank's VRAM (vram_rank0/1_gib device-wide, buffers_rank0/1_gib
# this process's). --tp 2 caps --mtp at 7. The same entry path can appear twice with different names
# (e.g. mix=...#trellis and mix_tp2=...#trellis!2) to interleave TP=1 and TP=2 on one binary.
#
# -Modes: plain, dflash<k> (--dflash -Dflash --dflash-k <k>) and mtp<k> (--mtp <k>); the default is
# A3's three, plain, dflash7 and mtp3. -NoDflash drops the dflash modes (a container without an MTP
# head needs -Modes plain,dflash7).
#
# Prefill (A3p: prompts of >= 256 tokens): one --chat process per entry and run, --max-tokens 8 --max-ctx
# 4096 -- the first decode prompt as a warm-up turn (a fresh process pays every kernel's first launch;
# it is not counted), then every line of -PrefillPromptFile (default tools/quant2/prefill_prompts.txt:
# 339-544 tokens of text each, 353-558 once chat-templated; ASCII, piped through stdin) as a turn of its own; each later turn prefills only
# its own new tokens, at the context the turns before it left. A turn whose prefill has fewer than 256
# tokens is recorded but left out of the tables. -NoPrefill skips it.
#
# Decode speed on this box is sensitive to host CPU load (a concurrent r4dx-convert moved v6's plain
# decode by -6%), so before every run the script waits until no other GPU job (r4dx-convert,
# r4dx-cli, tool_teacher_forced_logprobs, tool_trellis_gemm_bench, a python full_logits_golden /
# hessian_capture / trellis_quant / kl_report) is running and the CPU is below -MaxCpuPct (2 s of
# samples; -NoIdleWait skips both). An r4dx-server stops it outright.
param(
  [Parameter(Mandatory = $true)][string[]]$Container,
  [int]$Runs = 3,
  [string]$Dflash = 'D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx',
  [string]$OutDir = 'D:\models\r4dx\quant2-bench',
  [string]$Layout = 'trellis',
  [string]$PromptFile = '',
  [string[]]$Modes = @('plain', 'dflash7', 'mtp3'),
  [string]$PrefillPromptFile = '',
  [string]$Baseline = '',
  [int]$MaxCpuPct = 20,
  [ValidateSet(1, 2)][int]$Tp = 1,
  [switch]$NoDflash,
  [switch]$NoPrefill,
  [switch]$NoIdleWait
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[bench] stop r4dx-server first' }
$env:HIP_VISIBLE_DEVICES = '1'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$defaultCli = Join-Path $repo 'build\win-hip\src\cli\r4dx-cli.exe'
if (-not $PromptFile) { $PromptFile = Join-Path $repo 'tests\model\mtp_prompts.txt' }
if (-not $PrefillPromptFile) { $PrefillPromptFile = Join-Path $repo 'tools\quant2\prefill_prompts.txt' }
$prompts = @(Get-Content $PromptFile | Where-Object { $_.Trim() })
$prefillPrompts = @(Get-Content $PrefillPromptFile | Where-Object { $_.Trim() })

# Waits until no other GPU job runs and the CPU is quiet (see the header).
function Wait-Quiet {
  if ($NoIdleWait) { return }
  $jobs = 'r4dx-convert', 'r4dx-cli', 'tool_teacher_forced_logprobs', 'tool_trellis_gemm_bench'
  $py = 'full_logits_golden|hessian_capture|trellis_quant|kl_report'
  $said = $false
  $t0 = Get-Date
  while ($true) {
    if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[bench] an r4dx-server started; stop it first' }
    $busy = @(Get-Process $jobs -ErrorAction SilentlyContinue | ForEach-Object { $_.ProcessName })
    $busy += @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -match $py } | ForEach-Object { 'python' })
    $cpu = 0
    if (-not $busy) {
      $s = 1..2 | ForEach-Object { (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average; Start-Sleep 1 }
      $cpu = ($s | Measure-Object -Maximum).Maximum
    }
    if (-not $busy -and $cpu -lt $MaxCpuPct) { return }
    # Another GPU job is waited out for as long as it runs; a CPU that never quiets down (a steady
    # background load above -MaxCpuPct) only for 5 minutes, then the run goes ahead with a note.
    if (-not $busy -and ((Get-Date) - $t0).TotalMinutes -gt 5) {
      Write-Host "[bench] CPU still at $cpu% after 5 minutes; running anyway"
      return
    }
    if (-not $said) {
      Write-Host ("[bench] waiting: {0}" -f $(if ($busy) { ($busy | Sort-Object -Unique) -join ', ' } else { "CPU at $cpu%" }))
      $said = $true
    }
    Start-Sleep 10
  }
}

$entries = foreach ($c in $Container) {
  $name, $rest = $c -split '=', 2
  $rest, $tpStr = $rest -split '!', 2
  $rest, $lay = $rest -split '#', 2
  $path, $bin = $rest -split '@', 2
  if (-not $bin) { $bin = $defaultCli }
  if (-not $lay) { $lay = $Layout }
  $tpN = if ($tpStr) { [int]$tpStr } else { $Tp }
  if ($tpN -notin 1, 2) { throw "[bench] ${name}: tp must be 1 or 2, got '$tpStr'" }
  if (-not (Test-Path $path)) { throw "[bench] ${name}: container $path not found" }
  if (-not (Test-Path $bin)) { throw "[bench] ${name}: binary $bin not found" }
  [pscustomobject]@{ name = $name; path = $path; bin = $bin; layout = $lay; tp = $tpN }
}
foreach ($e in $entries) {
  foreach ($m in $Modes) {
    if ($e.tp -eq 2 -and $m -match '^mtp(\d+)$' -and [int]$Matches[1] -gt 7) { throw "[bench] $($e.name): --tp 2 caps --mtp at 7 ($m)" }
  }
}
# The extra flags and the device environment of one entry.
function TpArgs($e) { if ($e.tp -eq 2) { return @('--tp', '2', '--tp-mode', 'real') } else { return @() } }
if (-not $Baseline) { $Baseline = $entries[0].name }
if (-not ($entries | Where-Object { $_.name -eq $Baseline })) { throw "[bench] -Baseline $Baseline is not an entry" }
$modes = @($Modes | Where-Object { -not ($NoDflash -and $_ -match '^dflash') })
foreach ($m in $modes) {
  if ($m -notmatch '^(plain|dflash\d+|mtp\d+)$') { throw "[bench] unknown mode '$m' (plain, dflash<k>, mtp<k>)" }
}
$results = @()

# One r4dx-cli process; its output, stringified (PS 5.1 makes native stderr lines -- the --stats
# output -- error records, and under 'Stop' the first one would abort; judge by the exit code).
function Invoke-Cli($e, [string[]]$cliArgs, [string[]]$stdin, [string]$log) {
  Wait-Quiet
  if ($e.tp -eq 2) { Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue } else { $env:HIP_VISIBLE_DEVICES = '1' }
  $cliArgs = @($cliArgs) + @(TpArgs $e)
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try {
    if ($stdin) { $out = @($stdin | & $e.bin @cliArgs 2>&1 | ForEach-Object { "$_" }) }
    else { $out = @(& $e.bin @cliArgs 2>&1 | ForEach-Object { "$_" }) }
  } finally { $ErrorActionPreference = $prev }
  $code = $LASTEXITCODE
  $out | Set-Content -Encoding utf8 $log
  return [pscustomobject]@{ code = $code; out = $out }
}

# The generated text: the non-log lines after the "[stats] container load" line. Load logs come
# before it, and under --tp 2 the two rank threads' load lines can interleave mid-line on stderr,
# leaving fragments that do not start with '['.
function TextOf([string[]]$out) {
  $i = 0
  for ($k = 0; $k -lt $out.Count; $k++) { if ($out[$k] -match '^\[stats\] container load') { $i = $k + 1; break } }
  return (@($out | Select-Object -Skip $i) | Where-Object { $_ -notmatch '^\[' -and $_ -notmatch 'NativeCommandError|CategoryInfo|^\s*At |^\s*\+ ' }) -join "`n"
}

for ($r = 1; $r -le $Runs; $r++) {
  foreach ($e in $entries) {
    for ($p = 0; $p -lt $prompts.Count; $p++) {
      foreach ($m in $modes) {
        $log = Join-Path $OutDir ("{0}_{1}_p{2}_run{3}.log" -f $e.name, $m, $p, $r)
        $cliArgs = @('--model', $e.path, '--layout', $e.layout, '--prompt', $prompts[$p], '--vision', 'off',
                     '--think', 'off', '--temperature', '0', '--max-tokens', '256', '--max-ctx', '2048', '--stats')
        if ($m -match '^dflash(\d+)$') { $cliArgs += @('--dflash', $Dflash, '--dflash-k', $Matches[1]) }
        if ($m -match '^mtp(\d+)$') { $cliArgs += @('--mtp', $Matches[1]) }
        $res = Invoke-Cli $e $cliArgs $null $log
        $code = $res.code; $out = $res.out
        $text = TextOf $out
        $row = [ordered]@{ name = $e.name; layout = $e.layout; tp = $e.tp; mode = $m; prompt = $p; run = $r; exit = $code
                           decode_tok_s = $null; tokens = $null; decode_s = $null; prefill_tokens = $null
                           prefill_s = $null; accept_pct = $null; tok_per_round = $null; vram_gib = $null
                           text_sha256 = [BitConverter]::ToString(
                             [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '').Substring(0, 16) }
        $s = ($out | Out-String)
        if ($s -match 'prefill: (\d+) tok in ([\d.]+)s .*decode: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\).*VRAM: ([\d.]+) GiB') {
          $row.prefill_tokens = [int]$Matches[1]; $row.prefill_s = [double]$Matches[2]
          $row.tokens = [int]$Matches[3]; $row.decode_s = [double]$Matches[4]
          $row.decode_tok_s = [double]$Matches[5]; $row.vram_gib = [double]$Matches[6]
        }
        if ($s -match '\[stats\] (?:dflash|mtp): .*\(([\d.]+)% acceptance, ([\d.]+) tok/round') {
          $row.accept_pct = [double]$Matches[1]; $row.tok_per_round = [double]$Matches[2]
        }
        if ($e.tp -eq 2) {
          # One line per rank after load and again after generation; the last one of each rank counts.
          foreach ($l in $out) {
            if ($l -match '\[stats\] tp rank (\d) \(HIP device \d+\): VRAM used ([\d.]+) GiB.*buffers ([\d.]+) GiB') {
              $row["vram_rank$($Matches[1])_gib"] = [double]$Matches[2]; $row["buffers_rank$($Matches[1])_gib"] = [double]$Matches[3]
            }
          }
        }
        $results += [pscustomobject]$row
        Write-Host ("[bench] run {0} {1,-10} p{2} {3,-8} {4,7} tok/s  {5} tok  exit {6}" -f $r, $e.name, $p, $m, $row.decode_tok_s, $row.tokens, $code)
        ConvertTo-Json -InputObject @($results) -Depth 3 | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.json')
      }
    }
    if (-not $NoPrefill -and $prefillPrompts.Count) {
      $log = Join-Path $OutDir ("{0}_prefill_run{1}.log" -f $e.name, $r)
      $cliArgs = @('--model', $e.path, '--layout', $e.layout, '--chat', '--vision', 'off', '--think', 'off',
                   '--temperature', '0', '--max-tokens', '8', '--max-ctx', '4096', '--stats')
      $res = Invoke-Cli $e $cliArgs (@($prompts[0]) + $prefillPrompts) $log
      $code = $res.code; $out = $res.out
      $turns = @($out | Where-Object { $_ -match '\[stats\] prefill: (\d+) tok in ([\d.]+)s' } | ForEach-Object {
        $null = $_ -match '\[stats\] prefill: (\d+) tok in ([\d.]+)s'; , @([int]$Matches[1], [double]$Matches[2]) })
      $reprefill = [bool]($out | Where-Object { $_ -match 'did not extend the previous token prefix' })
      for ($t = 1; $t -lt $turns.Count; $t++) {   # turn 0 is the warm-up
        $row = [ordered]@{ name = $e.name; layout = $e.layout; tp = $e.tp; mode = 'prefill'; prompt = $t - 1; run = $r; exit = $code
                           prefill_tokens = $turns[$t][0]; prefill_s = $turns[$t][1]
                           prefill_tok_s = $(if ($turns[$t][1] -gt 0) { [math]::Round($turns[$t][0] / $turns[$t][1], 2) } else { $null })
                           counted = ($turns[$t][0] -ge 256); reprefill_warning = $reprefill }
        $results += [pscustomobject]$row
        Write-Host ("[bench] run {0} {1,-10} prefill p{2}: {3} tok {4,8} tok/s  exit {5}" -f $r, $e.name, ($t - 1), $row.prefill_tokens, $row.prefill_tok_s, $code)
      }
      if ($turns.Count -ne $prefillPrompts.Count + 1) {
        Write-Host ("[bench] {0} prefill run {1}: {2} turns reported, {3} expected (exit {4}; see {5})" -f $e.name, $r, $turns.Count, ($prefillPrompts.Count + 1), $code, $log)
      }
      ConvertTo-Json -InputObject @($results) -Depth 3 | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.json')
    }
  }
}

function Median([double[]]$v) {
  if (-not $v) { return $null }
  $s = @($v | Sort-Object); $n = $s.Count
  if ($n % 2) { return $s[($n - 1) / 2] } else { return ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
}
# Token-weighted tok/s of one run of one (container, mode): sum tokens / sum decode seconds over its
# prompts -- the number to compare, since it does not over-weight short texts. The tables give the
# aggregate over every run and the median over runs, and the median against -Baseline's.
function RunRate($rows) {
  $sec = ($rows | Measure-Object decode_s -Sum).Sum
  if ($sec) { return ($rows | Measure-Object tokens -Sum).Sum / $sec } else { return $null }
}
function LayoutCell($e) { if ($e.tp -eq 2) { return "$($e.layout), tp 2" } else { return $e.layout } }
$medRun = @{}
foreach ($e in $entries) {
  foreach ($m in $modes) {
    $rates = @(1..$Runs | ForEach-Object { $rn = $_
      RunRate @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m -and $_.run -eq $rn -and $_.decode_s }) } |
      Where-Object { $_ })
    $medRun["$($e.name)|$m"] = Median $rates
  }
}
$hdr = '| container | layout | mode | ' + (($prompts | ForEach-Object -Begin { $i = 0 } -Process { "p$i"; $i++ }) -join ' | ') +
       " | aggregate tok/s | median of runs | vs $Baseline | text stable |"
$md = @("Decode (A3: median of runs >= the baseline's - 0.5%)", '', $hdr, ('|---|---|---|' + ('--:|' * ($prompts.Count + 3)) + '---|'))
foreach ($e in $entries) {
  foreach ($m in $modes) {
    $cells = @()
    $stable = $true
    for ($p = 0; $p -lt $prompts.Count; $p++) {
      $rs = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m -and $_.prompt -eq $p -and $_.decode_tok_s })
      $mean = if ($rs.Count) { [math]::Round((($rs | ForEach-Object { $_.decode_tok_s } | Measure-Object -Average).Average), 2) } else { '--' }
      $acc = if ($m -ne 'plain' -and $rs.Count) { " ($($rs[0].accept_pct)%)" } else { '' }
      $cells += "$mean$acc"
      if (@($rs | ForEach-Object { $_.text_sha256 } | Sort-Object -Unique).Count -gt 1) { $stable = $false }
    }
    $agg = RunRate @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m -and $_.decode_s })
    $med = $medRun["$($e.name)|$m"]
    $base = $medRun["$Baseline|$m"]
    $vs = if ($med -and $base) { '{0:+0.00;-0.00}%' -f (100 * ($med / $base - 1)) } else { '--' }
    $md += "| $($e.name) | $(LayoutCell $e) | $m | $($cells -join ' | ') | **$(if ($agg) { [math]::Round($agg, 2) } else { '--' })** | $(if ($med) { [math]::Round($med, 2) } else { '--' }) | $vs | $stable |"
  }
}
$vramRows = @()
foreach ($e in $entries) {
  foreach ($m in $modes) {
    $rs = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq $m -and $_.decode_s })
    if (-not $rs.Count) { continue }
    $mx = { param($f) $v = @($rs | ForEach-Object { $_.$f } | Where-Object { $_ -ne $null }); if ($v.Count) { ($v | Measure-Object -Maximum).Maximum } else { '--' } }
    if ($e.tp -eq 2) {
      $vramRows += "| $($e.name) | $(LayoutCell $e) | $m | $(& $mx 'vram_rank0_gib') ($(& $mx 'buffers_rank0_gib')) | $(& $mx 'vram_rank1_gib') ($(& $mx 'buffers_rank1_gib')) |"
    } else {
      $vramRows += "| $($e.name) | $(LayoutCell $e) | $m | $(& $mx 'vram_gib') | -- |"
    }
  }
}
$md += @('', 'VRAM used, GiB (max over runs; tp 2: device-wide per rank, this process''s buffers in parentheses)', '',
         '| container | layout | mode | device 1 (rank 0) | device 0 (rank 1) |', '|---|---|---|--:|--:|') + $vramRows
if (-not $NoPrefill -and $prefillPrompts.Count) {
  $pre = @{}
  foreach ($e in $entries) {
    $rows = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq 'prefill' -and $_.counted -and $_.prefill_s })
    $sec = ($rows | Measure-Object prefill_s -Sum).Sum
    $pre[$e.name] = if ($sec) { ($rows | Measure-Object prefill_tokens -Sum).Sum / $sec } else { $null }
  }
  $md += @('', "Prefill of the >= 256-token prompts, warm (A3p: >= 0.80x the baseline); per prompt its tokens and mean tok/s", '',
           ('| container | layout | ' + (($prefillPrompts | ForEach-Object -Begin { $i = 0 } -Process { "p$i"; $i++ }) -join ' | ') +
            " | aggregate tok/s | x $Baseline |"),
           ('|---|---|' + ('--:|' * ($prefillPrompts.Count + 2))))
  foreach ($e in $entries) {
    $cells = @()
    for ($p = 0; $p -lt $prefillPrompts.Count; $p++) {
      $rs = @($results | Where-Object { $_.name -eq $e.name -and $_.mode -eq 'prefill' -and $_.prompt -eq $p -and $_.prefill_tok_s })
      $cells += if ($rs.Count) { "$($rs[0].prefill_tokens) tok: $([math]::Round((($rs | Measure-Object prefill_tok_s -Average).Average), 1))" } else { '--' }
    }
    $x = if ($pre[$e.name] -and $pre[$Baseline]) { '{0:0.000}' -f ($pre[$e.name] / $pre[$Baseline]) } else { '--' }
    $md += "| $($e.name) | $(LayoutCell $e) | $($cells -join ' | ') | **$(if ($pre[$e.name]) { [math]::Round($pre[$e.name], 1) } else { '--' })** | $x |"
  }
}
$md | Set-Content -Encoding utf8 (Join-Path $OutDir 'bench.md')
$md
