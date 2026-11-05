param(
    [int]$Steps = 1000,
    [int]$Seed = 7,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $env:PYTHONUTF8 = '1'
    & '.\.venv\Scripts\python.exe' -m spora.train --steps $Steps --seed $Seed --device $Device --run-dir "runs/p1-seed$Seed" --checkpoint "checkpoints/p1-seed$Seed.pt"
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the traceback.' }
} finally { Pop-Location }
