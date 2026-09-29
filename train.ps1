# Windows equivalent of `make train` (Git Bash on Windows ships no make).
#
#   .\train.ps1              full pipeline from the default seed
#   .\train.ps1 -Seed 7      different seed
#   .\train.ps1 -Clean       wipe artifacts/ and data/synthetic first
#   .\train.ps1 -Test        run the test suite afterwards
#
# Reproduces everything in artifacts/ from a fixed seed (spec phase 1).

[CmdletBinding()]
param(
    [int]$Seed = 42,
    [switch]$Clean,
    [switch]$Test
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Torch's ONNX exporter prints emoji; a cp1252 console raises UnicodeEncodeError.
$env:PYTHONIOENCODING = 'utf-8'

$py = if (Test-Path '.venv\Scripts\python.exe') { '.venv\Scripts\python.exe' } else { 'python' }
if ($py -eq 'python') {
    Write-Warning 'No .venv found — using the global interpreter. Create one with: python -m venv .venv'
}

if ($Clean) {
    Write-Host '==> cleaning artifacts/ and data/synthetic' -ForegroundColor Yellow
    Remove-Item -Recurse -Force 'artifacts', 'data\synthetic' -ErrorAction SilentlyContinue
}

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)

    Write-Host "==> $Label" -ForegroundColor Cyan
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()

Invoke-Step 'generating synthetic CRM data'  @('training/generate_synthetic.py', '--seed', $Seed)
Invoke-Step 'training XGBoost classifier'    @('training/train_xgb.py',          '--seed', $Seed)
Invoke-Step 'training LSTM + ONNX export'    @('training/train_lstm.py',         '--seed', $Seed)
Invoke-Step 'writing metrics and model card' @('training/evaluate.py',           '--seed', $Seed)

Write-Host ''
Write-Host '==> artifacts/' -ForegroundColor Green
Get-ChildItem 'artifacts' | Select-Object Name, @{
    Name = 'Size'; Expression = { '{0,8:N0} B' -f $_.Length }
} | Format-Table -AutoSize

if ($Test) {
    Invoke-Step 'running tests' @('-m', 'pytest', '-q')
}

$sw.Stop()
Write-Host ("done in {0:N1}s (seed {1})" -f $sw.Elapsed.TotalSeconds, $Seed) -ForegroundColor Green
