<#
.SYNOPSIS
  Capture a NATIVE stack for the paddle GPU server's access violation.
  RUN FROM AN ELEVATED POWERSHELL. Diagnostic only - changes no shipped code.

.DESCRIPTION
  faulthandler already gave us the Python frame (no elevation needed):

      paddlex/inference/models/runners/paddle_static/runner.py:509 in _create
        -> predictor = paddle_inference.create_predictor(config)

  What it cannot give is the native stack INSIDE paddle's C++ inference engine, which is
  where the 0xC0000005 actually occurs. This script gets that.

  The crash reproduces on roughly 1 GPU server start in 5 and takes about 8 seconds, so a
  capture loop of ~15 starts is normally enough. Three methods, in order of preference:

    1. procdump  - smallest, most targeted. A full dump on the first AV.
    2. TTD       - time-travel trace; lets you step BACKWARDS from the fault. Best signal,
                   biggest artefact (hundreds of MB), needs the TTD feature installed.
    3. WER       - no extra tooling, but writes dumps for the whole machine while enabled.

  Analyse the resulting .dmp in WinDbg:  !analyze -v  then  k

.PARAMETER Method
  procdump (default) | ttd | wer

.PARAMETER Trials
  How many server starts to attempt before giving up. Default 15.

.EXAMPLE
  .\capture_native_crash.ps1 -Method procdump -ProcDump C:\tools\procdump64.exe
#>
[CmdletBinding()]
param(
    [ValidateSet('procdump', 'ttd', 'wer')]
    [string]$Method = 'procdump',
    [string]$ProcDump = '',
    [int]$Trials = 15,
    [string]$OutDir = 'C:\OCULI.pd1\TESTING\RESULTS\crashdumps',
    [string]$Python = 'C:\OCULI.pd1\TESTING\.venv-release\Scripts\python.exe',
    [string]$BkMain = 'C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main'
)

$ErrorActionPreference = 'Stop'

$isAdmin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Run this from an ELEVATED PowerShell (Run as administrator)."
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "output dir : $OutDir"
Write-Host "method     : $Method"
Write-Host "trials     : $Trials"

# The probe drives the server starts; we only arrange for a dump to be taken when one
# of them faults. Reusing it keeps the reproduction identical to the measured one.
$probe = 'C:\OCULI.pd1\TESTING\RESOURCES\server_reliability_probe.py'

switch ($Method) {

    'procdump' {
        if (-not $ProcDump) {
            $ProcDump = (Get-Command procdump64.exe, procdump.exe -ErrorAction SilentlyContinue |
                         Select-Object -First 1 -ExpandProperty Source)
        }
        if (-not $ProcDump -or -not (Test-Path -LiteralPath $ProcDump)) {
            Write-Error @"
procdump not found. Download Sysinternals ProcDump and pass -ProcDump <path>:
  https://learn.microsoft.com/sysinternals/downloads/procdump
Then re-run:  .\capture_native_crash.ps1 -ProcDump C:\tools\procdump64.exe
"@
            exit 2
        }
        # -e 1 : first-chance too;  -ma : full memory;  -x is not used because the server
        # is spawned by the probe, so we register procdump as a postmortem-style monitor
        # on any python.exe that faults.
        Write-Host "`nRegistering procdump to dump ANY faulting python.exe ..."
        & $ProcDump -accepteula -ma -i $OutDir
        try {
            Write-Host "Running $Trials server starts; waiting for one to fault ...`n"
            & $Python $probe --bk $BkMain --trials $Trials --timeout 60
        }
        finally {
            Write-Host "`nUnregistering procdump ..."
            & $ProcDump -accepteula -u
        }
    }

    'ttd' {
        # Time Travel Debugging: record the child from start to fault, then replay it
        # backwards from the access violation. -onLaunch attaches to children too.
        Write-Host "`nStarting TTD recording (large artefacts - watch disk) ...`n"
        & tttracer.exe -out $OutDir -children -accepteula `
            $Python $probe --bk $BkMain --trials $Trials --timeout 60
    }

    'wer' {
        # Machine-wide LocalDumps. Enabled only for the duration of this run and removed
        # afterwards, because it captures dumps for EVERY crashing python.exe while set.
        $key = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\python.exe'
        Write-Host "`nEnabling WER LocalDumps for python.exe ..."
        New-Item -Path $key -Force | Out-Null
        New-ItemProperty -Path $key -Name DumpFolder   -Value $OutDir -PropertyType ExpandString -Force | Out-Null
        New-ItemProperty -Path $key -Name DumpType     -Value 2       -PropertyType DWord        -Force | Out-Null
        New-ItemProperty -Path $key -Name DumpCount    -Value 5       -PropertyType DWord        -Force | Out-Null
        try {
            Write-Host "Running $Trials server starts ...`n"
            & $Python $probe --bk $BkMain --trials $Trials --timeout 60
        }
        finally {
            Write-Host "`nRemoving WER LocalDumps key (leaving it on dumps every python crash) ..."
            Remove-Item -Path $key -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "`n================ RESULT ================"
$dumps = Get-ChildItem $OutDir -Filter '*.dmp' -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending
if ($dumps) {
    $dumps | Select-Object @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }, LastWriteTime, Name |
        Format-Table -AutoSize
    Write-Host @"
Next, in WinDbg (or 'windbgx'):
    .symfix ; .reload
    !analyze -v
    k                      # native stack at the fault
    lm m paddle*           # confirm which paddle module owns the faulting address

The Python frame is already known, so the question for the dump is narrow:
what does paddle_inference.create_predictor() dereference that is null/stale?
"@
} else {
    Write-Warning @"
No dump captured. The crash is intermittent (~1 start in 5) - re-run with more trials:
    .\capture_native_crash.ps1 -Method $Method -Trials 40
If starts are succeeding every time, confirm the crash still reproduces first:
    $Python $probe --bk $BkMain --trials 20
and look for rc=3221225477 rows.
"@
}
