#requires -Version 5.1
<#
.SYNOPSIS
  TP=1 byte-identity guard (docs/tp.md 10.3, gate G2): the candidate build without --tp must
  reproduce the frozen baseline binaries' bytes on every row of the matrix.

.DESCRIPTION
  Runs each row with the baseline binaries (build\baseline, frozen at aa54c20 -- BASELINE.txt) and
  with the candidate build, both on HIP device 1 (HIP_VISIBLE_DEVICES=1) and without any --tp flag,
  and compares SHA-256:
    rows 1-5, 7-9  r4dx-cli: stdout (the generated text, captured byte for byte), the
                   --dump-token-ids file, and the timing-free stderr lines
                   "[stats] mtp: / dflash: / sampled:" (rounds, drafted, accepted, fallback rows --
                   the draft-dependent numbers; greedy verify and seeded sample-and-match emit
                   draft-independent tokens, so text and ids alone cannot see a drafter or MTP-head
                   byte change);
    row 6          tool_teacher_forced_logprobs on the 4-layer container, layouts bf16/w4a16/w4a8/
                   mxfp4, --layers 4, the kl_corpus tokens: every *.logprobs.f16 it writes.

    1  standard protocol, plain greedy
    2  standard protocol + --dflash <drafter> --dflash-k 7
    3  standard protocol + --mtp 3
    4  --temperature 0.7 --top-k 20 --top-p 0.8 --seed 1, plain
    5  row 4 + --dflash <drafter> --dflash-k 7
    6  tool_teacher_forced_logprobs, 4-layer container, all four layouts
    7  --vision on --image tools\reference\golden_out\vision_test_image.png
       --prompt "What is in this picture?", plain greedy (SKIPped when the gitignored golden image
       is absent -- pass -Image to point at another copy)
    8  row 4 + --mtp 3 (MTP sampled rounds)
    9  --chat, two user turns fed through stdin, greedy (the second turn prefills a suffix at
       pos > 0 on top of the first turn's state)

  Standard protocol (docs/perf.md): --layout w4a16 --vision off --think off --temperature 0
  --max-tokens 256 --max-ctx 2048 --stats, prompt "Write a haiku about GPUs, then explain what a
  GPU is in two sentences." (The rest of stderr -- timings, VRAM -- is kept for reference and not
  compared.)

  Pre-flight (docs/tp.md 9.2): refuses to run while an r4dx-server process exists -- the production
  server holds ~28 GiB of device 1, and every v6 row needs ~16-19 GiB.

  Exit code 0 when every requested row was compared and is equal; throws (exit 1) on any
  difference or failed run, and on a SKIPped row unless -AllowSkip is given.

.PARAMETER Baseline
  Directory of the frozen baseline binaries (flat: r4dx-cli.exe, tool_teacher_forced_logprobs.exe).

.PARAMETER Candidate
  The candidate: a flat directory like -Baseline, or a build tree (build\win-hip, where the
  binaries are src\cli\r4dx-cli.exe and tests\model\tool_teacher_forced_logprobs.exe).

.PARAMETER Rows
  Which rows to run (default: all, 1..9).

.PARAMETER AllowSkip
  Accept a SKIPped row (row 7 without the golden image) instead of failing the run as incomplete.

.PARAMETER OutDir
  Where every run's stdout/stderr/token ids/log-prob dumps go (default build\logs\tp1_identity).

.EXAMPLE
  powershell -File tools\tp\tp1_identity.ps1 -Baseline build\baseline -Candidate build\win-hip
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Candidate,
    [int[]]$Rows = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [string]$Model = "D:\models\r4dx\qwen38-27b-v6.r4dx",
    [string]$Dflash = "D:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx",
    [string]$TestModel = "D:\models\r4dx\g64\qwen38-27b-l4-allmtp.r4dx",
    [string]$Tokens = "tools\reference\kl_corpus\tokens.json",
    [string]$Image = "tools\reference\golden_out\vision_test_image.png",
    [string]$OutDir = "build\logs\tp1_identity",
    [switch]$AllowSkip
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Resolve-Full([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $RepoRoot $p)
}

# A flat directory (build\baseline) or a build tree (build\win-hip).
function Find-Binary([string]$Dir, [string]$Name, [string]$TreeSubdir) {
    $flat = Join-Path $Dir $Name
    if (Test-Path $flat) { return $flat }
    $tree = Join-Path (Join-Path $Dir $TreeSubdir) $Name
    if (Test-Path $tree) { return $tree }
    throw "tp1_identity: $Name not found in $Dir (looked at $flat and $tree)"
}

# Windows command-line quoting for one argument (CommandLineToArgvW rules).
function Format-Arg([string]$a) {
    if ($a -eq '') { return '""' }
    if ($a -notmatch '[\s"]') { return $a }
    $q = $a -replace '(\\*)"', '$1$1\"'
    $q = $q -replace '(\\+)$', '$1$1'
    return '"' + $q + '"'
}

# Runs $Exe with $ArgList, stdout captured BYTE FOR BYTE to $StdoutPath (PowerShell's own `>` would
# re-encode it), stderr to $StderrPath. $StdinText, when given, is written to stdin as UTF-8 (no
# BOM) and stdin is then closed. Returns the exit code.
function Invoke-Captured([string]$Exe, [string[]]$ArgList, [string]$StdoutPath, [string]$StderrPath,
                         [string]$StdinText = $null) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = (($ArgList | ForEach-Object { Format-Arg $_ }) -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $true
    $psi.WorkingDirectory = $RepoRoot
    $out = [System.IO.File]::Create($StdoutPath)
    $err = [System.IO.File]::Create($StderrPath)
    try {
        $p = [System.Diagnostics.Process]::Start($psi)
        $t1 = $p.StandardOutput.BaseStream.CopyToAsync($out)
        $t2 = $p.StandardError.BaseStream.CopyToAsync($err)
        if ($StdinText) {
            $bytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes($StdinText)
            $p.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
            $p.StandardInput.BaseStream.Flush()
        }
        $p.StandardInput.Close()
        $p.WaitForExit()
        $t1.Wait()
        $t2.Wait()
        return $p.ExitCode
    } finally {
        $out.Dispose()
        $err.Dispose()
    }
}

function Get-Sha([string]$Path) { return (Get-FileHash -Algorithm SHA256 $Path).Hash }

# ---- pre-flight -------------------------------------------------------------------------------
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) {
    throw "tp1_identity: an r4dx-server process is running -- stop the production server first " +
          "(it holds ~28 GiB of HIP device 1; docs/tp.md 9.2)"
}

$BaselineDir = Resolve-Full $Baseline
$CandidateDir = Resolve-Full $Candidate
$Out = Resolve-Full $OutDir
New-Item -ItemType Directory -Force $Out | Out-Null
$Bins = @{
    baseline  = @{
        cli = Find-Binary $BaselineDir "r4dx-cli.exe" "src\cli"
        tf  = Find-Binary $BaselineDir "tool_teacher_forced_logprobs.exe" "tests\model"
    }
    candidate = @{
        cli = Find-Binary $CandidateDir "r4dx-cli.exe" "src\cli"
        tf  = Find-Binary $CandidateDir "tool_teacher_forced_logprobs.exe" "tests\model"
    }
}
foreach ($side in "baseline", "candidate") {
    foreach ($k in "cli", "tf") {
        Write-Output ("[tp1_identity] {0,-9} {1,-3} {2}  sha256 {3}" -f $side, $k, $Bins[$side][$k],
                      (Get-Sha $Bins[$side][$k]))
    }
}

$Prompt = "Write a haiku about GPUs, then explain what a GPU is in two sentences."
$Std = @("--model", $Model, "--layout", "w4a16", "--vision", "off", "--think", "off",
         "--temperature", "0", "--max-tokens", "256", "--max-ctx", "2048", "--stats",
         "--prompt", $Prompt)
$Sampled = @("--model", $Model, "--layout", "w4a16", "--vision", "off", "--think", "off",
             "--temperature", "0.7", "--top-k", "20", "--top-p", "0.8", "--seed", "1",
             "--max-tokens", "256", "--max-ctx", "2048", "--stats", "--prompt", $Prompt)
$DflashArgs = @("--dflash", $Dflash, "--dflash-k", "7")
$ImagePath = Resolve-Full $Image
$CliRows = @{
    1 = $Std
    2 = $Std + $DflashArgs
    3 = $Std + @("--mtp", "3")
    4 = $Sampled
    5 = $Sampled + $DflashArgs
    7 = @("--model", $Model, "--layout", "w4a16", "--vision", "on", "--image", $ImagePath,
          "--think", "off", "--temperature", "0", "--max-tokens", "256", "--max-ctx", "2048",
          "--stats", "--prompt", "What is in this picture?")
    8 = $Sampled + @("--mtp", "3")
    9 = @("--model", $Model, "--layout", "w4a16", "--vision", "off", "--think", "off",
          "--temperature", "0", "--max-tokens", "256", "--max-ctx", "2048", "--stats", "--chat")
}
# Row 9's two user turns, one per stdin line.
$ChatStdin = "Write a haiku about GPUs.`nNow explain what a GPU is in two sentences.`n"

# The timing-free, draft-dependent stderr lines (src/cli/main.cpp --stats): MTP / DFlash rounds,
# drafted and accepted counts, and the sampler's fallback-row count.
function Get-StatsLines([string]$StderrPath) {
    $lines = [System.IO.File]::ReadAllLines($StderrPath) |
        Where-Object { $_ -match '^\[stats\] (mtp|dflash|sampled):' }
    return (@($lines) -join "`n")
}

$saved_hip = $env:HIP_VISIBLE_DEVICES
$env:HIP_VISIBLE_DEVICES = "1"
$results = New-Object System.Collections.Generic.List[object]
try {
    foreach ($row in $Rows) {
        if ($row -eq 6) {
            foreach ($layout in "bf16", "w4a16", "w4a8", "mxfp4") {
                $name = "row6_$layout"
                $dirs = @{}
                foreach ($side in "baseline", "candidate") {
                    $d = Join-Path $Out "$side\$name"
                    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
                    New-Item -ItemType Directory -Force $d | Out-Null
                    $dirs[$side] = $d
                    $argList = @("--model", $TestModel, "--layout", $layout, "--tokens",
                                 (Resolve-Full $Tokens), "--out-dir", $d, "--layers", "4", "--quiet")
                    Write-Output "[tp1_identity] $name $side ..."
                    $rc = Invoke-Captured $Bins[$side].tf $argList (Join-Path $d "stdout.txt") (Join-Path $d "stderr.txt")
                    if ($rc -ne 0) { throw "tp1_identity: $name $side exited $rc (see $d\stderr.txt)" }
                }
                $files = @(Get-ChildItem (Join-Path $dirs.baseline "*.logprobs.f16") | Sort-Object Name)
                if ($files.Count -eq 0) { throw "tp1_identity: $name produced no *.logprobs.f16" }
                $cand = @(Get-ChildItem (Join-Path $dirs.candidate "*.logprobs.f16") | Sort-Object Name)
                $equal = ($cand.Count -eq $files.Count)
                foreach ($f in $files) {
                    $c = Join-Path $dirs.candidate $f.Name
                    if (-not (Test-Path $c) -or (Get-Sha $f.FullName) -ne (Get-Sha $c)) { $equal = $false }
                }
                $results.Add([pscustomobject]@{ Row = $name; Result = $(if ($equal) { "EQUAL" } else { "DIFF" });
                                                Detail = "$($files.Count) logprobs.f16 file(s)" })
            }
            continue
        }
        if (-not $CliRows.ContainsKey($row)) { throw "tp1_identity: no row $row (rows are 1..9)" }
        $name = "row$row"
        if ($row -eq 7 -and -not (Test-Path $ImagePath)) {
            Write-Output "[tp1_identity] $name SKIP: $ImagePath is absent (gitignored golden_out)"
            $results.Add([pscustomobject]@{ Row = $name; Result = "SKIP"; Detail = "golden image absent" })
            continue
        }
        $sha = @{}
        foreach ($side in "baseline", "candidate") {
            $d = Join-Path $Out $side
            New-Item -ItemType Directory -Force $d | Out-Null
            $ids = Join-Path $d "$name.ids"
            if (Test-Path $ids) { Remove-Item -Force $ids }
            $argList = $CliRows[$row] + @("--dump-token-ids", $ids)
            $stdin = $(if ($row -eq 9) { $ChatStdin } else { $null })
            $errPath = Join-Path $d "$name.stderr.txt"
            Write-Output "[tp1_identity] $name $side ..."
            $rc = Invoke-Captured $Bins[$side].cli $argList (Join-Path $d "$name.stdout.txt") $errPath $stdin
            if ($rc -ne 0) { throw "tp1_identity: $name $side exited $rc (see $errPath)" }
            if (-not (Test-Path $ids)) { throw "tp1_identity: $name $side wrote no --dump-token-ids file" }
            $sha[$side] = @{ text = Get-Sha (Join-Path $d "$name.stdout.txt"); ids = Get-Sha $ids;
                             stats = Get-StatsLines $errPath }
        }
        $equal = ($sha.baseline.text -eq $sha.candidate.text) -and ($sha.baseline.ids -eq $sha.candidate.ids) -and
                 ($sha.baseline.stats -ceq $sha.candidate.stats)
        $statsNote = $(if ($sha.candidate.stats) { "stats $(@($sha.candidate.stats -split "`n").Count) line(s)" +
                                                    $(if ($sha.baseline.stats -cne $sha.candidate.stats) { " DIFFER" } else { "" }) }
                       else { "no stats lines" })
        $results.Add([pscustomobject]@{ Row = $name; Result = $(if ($equal) { "EQUAL" } else { "DIFF" });
                                        Detail = "text $($sha.candidate.text.Substring(0, 12)) ids $($sha.candidate.ids.Substring(0, 12)) $statsNote" })
    }
} finally {
    if ($null -eq $saved_hip) { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    else { $env:HIP_VISIBLE_DEVICES = $saved_hip }
}

$table = $results | Format-Table -AutoSize | Out-String -Width 200
Write-Output $table
$table | Out-File -Encoding utf8 (Join-Path $Out "summary.txt")
$diffs = @($results | Where-Object { $_.Result -eq "DIFF" })
if ($diffs.Count -gt 0) { throw "tp1_identity: G2 FAIL -- $($diffs.Count) row(s) differ from the baseline" }
$skips = @($results | Where-Object { $_.Result -eq "SKIP" })
$all = @(1, 2, 3, 4, 5, 6, 7, 8, 9)
$full = (@(Compare-Object ($Rows | Sort-Object -Unique) $all).Count -eq 0)
if ($skips.Count -gt 0) {
    $msg = "[tp1_identity] G2 INCOMPLETE: $(($skips | ForEach-Object { $_.Row }) -join ', ') SKIPped; " +
           "every compared row is byte-identical to the baseline"
    if (-not $AllowSkip) { throw "$msg (pass -AllowSkip to accept a skipped row)" }
    Write-Output "$msg (accepted: -AllowSkip)"
} elseif ($full) {
    Write-Output "[tp1_identity] G2 PASS: every row is byte-identical to the baseline"
} else {
    Write-Output ("[tp1_identity] rows {0} byte-identical to the baseline (a partial run, not the " +
                  "full G2 matrix)" -f ($Rows -join ','))
}
