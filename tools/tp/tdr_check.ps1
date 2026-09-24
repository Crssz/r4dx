#requires -Version 5.1
<#
.SYNOPSIS
  Reports any Windows TDR (GPU timeout detection and recovery) since a given time (docs/tp.md 10.5,
  gate G8; Appendix B N44, N55).

.DESCRIPTION
  On this box a TDR shows up ONLY as an Application-log event from provider "Windows Error
  Reporting" whose message names "LiveKernelEvent" with "P1: 141" (VIDEO_ENGINE_TIMEOUT_DETECTED);
  the classic System event 4101 (Display) is checked too, but never appeared here (N44).

  Windows Error Reporting RE-LOGS its queued reports in bursts: every old LiveKernelEvent report is
  logged again, each time with a fresh TimeCreated (25 at once at 00:11, 09:31, 11:46, 15:12 ... on
  2026-09-24, the 2026-08-19 20:08 dump 242 times in total). A filter on TimeCreated alone would
  therefore report those as new TDRs. Each 141 event names its kernel dump,
  C:\WINDOWS\LiveKernelReports\WATCHDOG\WATCHDOG-<yyyyMMdd>-<HHmm>.dmp, stamped (local time, minute
  resolution) when the TDR happened: an event counts as a TDR since -Since only if that stamp is at
  or after -Since truncated to the minute (so a TDR in the same minute as -Since, just before it, is
  also reported -- conservative). An event whose message carries no dump name counts as a TDR.
  Re-logged old reports are listed as ignored.

  Exit code 0: no TDR since -Since. 1: at least one (each is printed).

.PARAMETER Since
  Start of the window (a [datetime], e.g. the time recorded just before a device-0 run).

.PARAMETER Quiet
  Print only TDRs (and the verdict line), not the ignored re-logged reports.

.EXAMPLE
  $t0 = Get-Date; <device-0 run>; Start-Sleep 30; .\tools\tp\tdr_check.ps1 -Since $t0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][datetime]$Since,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

function Get-EventsSince([hashtable]$filter) {
    try {
        return @(Get-WinEvent -FilterHashtable $filter -ErrorAction Stop)
    } catch {
        # Get-WinEvent throws when nothing matches; anything else is a real failure.
        if ($_.FullyQualifiedErrorId -match 'NoMatchingEventsFound') { return @() }
        throw
    }
}

$sinceMinute = Get-Date -Year $Since.Year -Month $Since.Month -Day $Since.Day -Hour $Since.Hour `
    -Minute $Since.Minute -Second 0 -Millisecond 0
$tdrs = New-Object System.Collections.Generic.List[string]
$ignored = @{}

$wer = Get-EventsSince @{ LogName = 'Application'; ProviderName = 'Windows Error Reporting'; StartTime = $Since }
foreach ($e in $wer) {
    $msg = $e.Message
    if ($msg -notmatch 'LiveKernelEvent' -or $msg -notmatch 'P1:\s*141\b') { continue }
    $m = [regex]::Match($msg, 'WATCHDOG-(\d{8})-(\d{4})\.dmp')
    if ($m.Success) {
        $stamp = [datetime]::ParseExact($m.Groups[1].Value + $m.Groups[2].Value, 'yyyyMMddHHmm',
                                        [System.Globalization.CultureInfo]::InvariantCulture)
        if ($stamp -lt $sinceMinute) {
            $key = "WATCHDOG-$($m.Groups[1].Value)-$($m.Groups[2].Value).dmp"
            if (-not $ignored.ContainsKey($key)) { $ignored[$key] = 0 }
            $ignored[$key] += 1
            continue
        }
        $line = ("{0:yyyy-MM-dd HH:mm:ss}  Application / Windows Error Reporting: LiveKernelEvent 141, dump " +
                 "WATCHDOG-{1}-{2}.dmp") -f $e.TimeCreated, $m.Groups[1].Value, $m.Groups[2].Value
        $tdrs.Add($line)
    } else {
        $line = ("{0:yyyy-MM-dd HH:mm:ss}  Application / Windows Error Reporting: LiveKernelEvent 141 " +
                 "(no dump name in the report)") -f $e.TimeCreated
        $tdrs.Add($line)
    }
}

$sys = Get-EventsSince @{ LogName = 'System'; Id = 4101; StartTime = $Since }
foreach ($e in $sys) {
    $first = ($e.Message -split "`r?`n")[0]
    $line = "{0:yyyy-MM-dd HH:mm:ss}  System event 4101 ({1}): {2}" -f $e.TimeCreated, $e.ProviderName, $first
    $tdrs.Add($line)
}

if (-not $Quiet) {
    foreach ($k in ($ignored.Keys | Sort-Object)) {
        Write-Output ("[tdr_check] ignored: {0} re-logged {1} time(s) (a TDR from before {2:yyyy-MM-dd HH:mm})" -f `
                      $k, $ignored[$k], $sinceMinute)
    }
}
foreach ($t in $tdrs) { Write-Output "[tdr_check] TDR: $t" }
if ($tdrs.Count -gt 0) {
    Write-Output ("[tdr_check] FAIL: {0} TDR event(s) since {1:yyyy-MM-dd HH:mm:ss}" -f $tdrs.Count, $Since)
    exit 1
}
Write-Output ("[tdr_check] OK: no TDR since {0:yyyy-MM-dd HH:mm:ss}" -f $Since)
exit 0
