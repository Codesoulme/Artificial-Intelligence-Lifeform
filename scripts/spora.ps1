param(
    [Parameter(Mandatory=$true)][ValidateSet('ask','learn','info')][string]$Action,
    [int]$Key = 0,
    [int]$Value = 0
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $env:PYTHONUTF8 = '1'
    switch ($Action) {
        'info' { & '.\.venv\Scripts\python.exe' -m spora.cli info }
        'ask' { & '.\.venv\Scripts\python.exe' -m spora.cli ask $Key }
        'learn' { & '.\.venv\Scripts\python.exe' -m spora.cli learn $Key $Value }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Command failed; see the message above.' }
} finally { Pop-Location }
