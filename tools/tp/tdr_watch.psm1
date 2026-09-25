#requires -Version 5.1
<#
.SYNOPSIS
  Runs a two-GPU tool while watching for a Windows TDR, and stops it at the first one (docs/tp.md
  10.5, G8; Appendix B N44, N55, N64). Imported by tools\tp\soak.ps1 and tools\tp\ar_stress.ps1;
  tools\server\smoke.ps1 -Tp 2 uses Start-TdrWatchJob (end of this file) instead, since its run is a
  server plus the requests the script itself sends, not one tool process.

.DESCRIPTION
  Invoke-TdrWatched:
  1. Refuses to start while an r4dx-server process exists (docs/tp.md 9.2).
  2. Removes HIP_VISIBLE_DEVICES from the process environment (both GPUs); restored afterwards.
  3. Records the start time and starts the tool (Start-Process -NoNewWindow: its output goes to
     this console, unchanged).
  4. While the tool runs, every -PollSeconds (default 20) runs tdr_check.ps1 -Since <start> -Quiet.
     A device-0 TDR need not stop the workload -- P3's first one (18:49:44, 52 s into its stress)
     did not, and the second, 2.5 minutes later, reset the adapter and crashed a desktop app
     (N44) -- so at the FIRST TDR the tool is stopped (Stop-Process), never retried.
  5. After the tool ended (or was stopped): waits 30 s (Windows Error Reporting logs a TDR a few
     seconds late) and runs tdr_check.ps1 -Since <start> once more.
  6. Scans the JSON log lines this run wrote for "HIP error 719" / "unspecified launch failure":
     the P3 TDR reached HIP as error 719 (N44), so such an error counts as a suspected TDR.

  Returns an object: ExitCode (the tool's; $null if unknown), Tdr, SuspectedTdr, Stopped, Start,
  NewLines (the JSON log lines of this run). Messages go to the host (Write-Host), not the output.

.EXAMPLE
  Import-Module .\tools\tp\tdr_watch.psm1
  $r = Invoke-TdrWatched -Exe $exe -ToolArgs @('--iterations', '3', '--json', $log) -WorkingDirectory $root `
         -JsonLog $log -JsonAppends
#>

$ErrorActionPreference = "Stop"
$script:TdrCheck = Join-Path $PSScriptRoot "tdr_check.ps1"

# Windows command-line quoting for Start-Process -ArgumentList (PS 5.1 joins the array unquoted).
function ConvertTo-ArgString([string[]]$List) {
    $parts = foreach ($a in $List) {
        if ($a -eq '') { '""' }
        elseif ($a -match '"') { throw "tdr_watch: an argument containing a double quote is not supported: $a" }
        elseif ($a -match '\s') { '"' + ($a -replace '\\+$', '$0$0') + '"' }
        else { $a }
    }
    return ($parts -join ' ')
}

# tdr_check.ps1 in a child process: @{ Code; Output }.
function Invoke-TdrCheck([datetime]$Since, [switch]$Quiet) {
    $list = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $script:TdrCheck, '-Since',
              $Since.ToString('yyyy-MM-ddTHH:mm:ss'))
    if ($Quiet) { $list += '-Quiet' }
    $out = & powershell @list
    return [pscustomobject]@{ Code = $LASTEXITCODE; Output = @($out) }
}

# The lines of `Path` from byte `Offset` on (the file may still be open for appending). `Path` must
# be absolute: [IO.File] resolves a relative path against the process directory, not the location.
function Get-LinesFrom([string]$Path, [long]$Offset) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return @() }
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
                                 [System.IO.FileShare]::ReadWrite)
    try {
        if ($Offset -gt $fs.Length) { $Offset = 0 }
        [void]$fs.Seek($Offset, [System.IO.SeekOrigin]::Begin)
        $text = (New-Object System.IO.StreamReader($fs)).ReadToEnd()
    } finally {
        $fs.Dispose()
    }
    return @($text -split "`r?`n" | Where-Object { $_ -ne '' })
}

function Invoke-TdrWatched {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$ToolArgs = @(),
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        # Absolute path of the tool's JSON log. -JsonAppends: the tool appends to it (tool_tp_soak), so
        # only the bytes after the pre-run length are this run's; otherwise the tool rewrites it
        # (tool_tp_ar_stress) and the whole file counts if it was written after the start.
        [string]$JsonLog = '',
        [switch]$JsonAppends,
        [string]$Label = 'tdr_watch',
        [ValidateRange(5, 300)][int]$PollSeconds = 20,
        # Test hook: the start of the TDR window (default: the run's start). An earlier time makes an
        # old, already logged TDR trip the watch, which exercises the stop path without a GPU.
        [datetime]$CheckSince
    )
    if (Get-Process r4dx-server -ErrorAction SilentlyContinue) {
        throw "stop the production server first (r4dx-server is running; this run uses both GPUs)"
    }
    if (-not (Test-Path -LiteralPath $Exe)) { throw "not built: $Exe (run .\build.ps1)" }
    if ($JsonLog -and -not [System.IO.Path]::IsPathRooted($JsonLog)) {
        throw "tdr_watch: -JsonLog must be an absolute path, got $JsonLog"
    }
    $jsonOffset = 0L
    if ($JsonAppends -and $JsonLog -and (Test-Path -LiteralPath $JsonLog)) {
        $jsonOffset = (Get-Item -LiteralPath $JsonLog).Length
    }

    $r = [pscustomobject]@{ ExitCode = $null; Tdr = $false; SuspectedTdr = $false; Stopped = $false; Start = $null;
                            NewLines = @() }
    $savedHip = $env:HIP_VISIBLE_DEVICES
    try {
        Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        $start = Get-Date
        $r.Start = $start
        $since = if ($PSBoundParameters.ContainsKey('CheckSince')) { $CheckSince } else { $start }
        Write-Host ("[{0}] start {1:yyyy-MM-dd HH:mm:ss}: {2} {3}" -f $Label, $start, $Exe, ($ToolArgs -join ' '))
        Write-Host ("[{0}] TDR watch: tdr_check -Since {1:yyyy-MM-dd HH:mm:ss} every {2} s while it runs" -f `
                    $Label, $since, $PollSeconds)
        $sp = @{ FilePath = $Exe; WorkingDirectory = $WorkingDirectory; NoNewWindow = $true; PassThru = $true }
        $argString = ConvertTo-ArgString $ToolArgs
        if ($argString) { $sp.ArgumentList = $argString }
        $p = Start-Process @sp
        $null = $p.Handle  # PS 5.1: ExitCode stays empty unless a handle was taken while it ran
        while (-not $p.WaitForExit($PollSeconds * 1000)) {
            $c = Invoke-TdrCheck -Since $since -Quiet
            if ($c.Code -ne 0) {
                Write-Host ("[{0}] TDR at {1:yyyy-MM-dd HH:mm:ss} while the tool ran: stopping it (pid {2}); no retry" -f `
                            $Label, (Get-Date), $p.Id)
                foreach ($l in $c.Output) { Write-Host $l }
                try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch { Write-Host "[$Label] Stop-Process: $_" }
                $null = $p.WaitForExit(30000)
                $r.Tdr = $true
                $r.Stopped = $true
                break
            }
        }
        if ($p.HasExited) { $r.ExitCode = $p.ExitCode }
    } finally {
        if ($null -ne $savedHip) { $env:HIP_VISIBLE_DEVICES = $savedHip }
        else { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    }

    Write-Host ("[{0}] tool {1} at {2:yyyy-MM-dd HH:mm:ss} (exit code {3}); waiting 30 s for Windows Error Reporting" -f `
                $Label, $(if ($r.Stopped) { 'stopped' } else { 'exited' }), (Get-Date), $r.ExitCode)
    Start-Sleep -Seconds 30
    $c = Invoke-TdrCheck -Since $since -Quiet
    foreach ($l in $c.Output) { Write-Host $l }
    if ($c.Code -ne 0) { $r.Tdr = $true }

    if ($JsonLog -and (Test-Path -LiteralPath $JsonLog)) {
        # @(): a one-line result would otherwise unroll to a string (and [-1] index its last char).
        if ($JsonAppends) {
            $r.NewLines = @(Get-LinesFrom $JsonLog $jsonOffset)
        } elseif ((Get-Item -LiteralPath $JsonLog).LastWriteTime -ge $r.Start) {
            $r.NewLines = @(Get-LinesFrom $JsonLog 0)
        }
    }
    $hit = @($r.NewLines | Where-Object { $_ -match 'HIP error 719\b|unspecified launch failure' })
    if ($hit.Count -gt 0) {
        $r.SuspectedTdr = $true
        Write-Host ("[{0}] suspected TDR: the log reports HIP error 719 / unspecified launch failure (N44):" -f $Label)
        Write-Host ("    " + $hit[0])
    }
    if ($r.Stopped -and $r.NewLines.Count -gt 0) {
        Write-Host ("[{0}] last log line before the stop: {1}" -f $Label, $r.NewLines[-1])
    }
    return $r
}

# For a run that is not one tool process -- tools\server\smoke.ps1 -Tp 2 drives an r4dx-server with
# requests from the calling script (docs/tp.md P5, Appendix B N77): a background job that runs
# tdr_check.ps1 -Since <Since> -Quiet every -PollSeconds and, at the first TDR, writes the check's
# output to -Marker and stops -TargetPid at once (no retry). Returns the job; the caller removes it
# (Stop-Job / Remove-Job) when the run ends, then waits 30 s and runs the final check itself.
function Start-TdrWatchJob {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][datetime]$Since,
        [Parameter(Mandatory = $true)][int]$TargetPid,
        [Parameter(Mandatory = $true)][string]$Marker,
        [ValidateRange(5, 300)][int]$PollSeconds = 20
    )
    if (-not [System.IO.Path]::IsPathRooted($Marker)) { throw "tdr_watch: -Marker must be an absolute path, got $Marker" }
    Remove-Item -LiteralPath $Marker -ErrorAction SilentlyContinue
    return Start-Job -ScriptBlock {
        param($Check, $SinceText, $ProcId, $MarkerPath, $Poll)
        while ($true) {
            Start-Sleep -Seconds $Poll
            $out = & powershell -NoProfile -ExecutionPolicy Bypass -File $Check -Since $SinceText -Quiet
            if ($LASTEXITCODE -ne 0) {
                $lines = @($out) + @("[tdr_watch] TDR: stopped pid $ProcId at " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') +
                                     "; no retry")
                $lines | Set-Content -LiteralPath $MarkerPath
                Stop-Process -Id $ProcId -Force -ErrorAction SilentlyContinue
                return
            }
        }
    } -ArgumentList $script:TdrCheck, $Since.ToString('yyyy-MM-ddTHH:mm:ss'), $TargetPid, $Marker, $PollSeconds
}

Export-ModuleMember -Function Invoke-TdrWatched, Start-TdrWatchJob
