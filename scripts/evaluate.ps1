# VoxShield Evaluation Script
#
# Usage:
#   .\scripts\evaluate.ps1 -DatasetDir "D:\Datasets\ASVspoof5"
#   .\scripts\evaluate.ps1 -DatasetDir "D:\Datasets\ASVspoof5" -Split dev
#   .\scripts\evaluate.ps1 -DatasetDir "D:\Datasets\ASVspoof5" -MaxSamples 5000

param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetDir,

    [string]$Checkpoint  = "checkpoints\asvspoof5\best.pt",
    [string]$Split       = "dev",
    [string]$OutputDir   = "reports\asvspoof5",
    [int]   $BatchSize   = 64,
    [int]   $MaxSamples  = 0,
    [switch]$SkipRobustness = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VoxShield — Evaluation" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Checkpoint : $Checkpoint" -ForegroundColor White
Write-Host "  Dataset    : $DatasetDir" -ForegroundColor White
Write-Host "  Split      : $Split" -ForegroundColor White
Write-Host "  Output dir : $OutputDir" -ForegroundColor White
Write-Host ""

if (-not (Test-Path $Checkpoint)) {
    Write-Host "ERROR: Checkpoint not found: $Checkpoint" -ForegroundColor Red
    Write-Host "  Train first: .\scripts\train_asvspoof5.ps1 -DatasetDir `"$DatasetDir`"" -ForegroundColor Yellow
    exit 1
}

$evalArgs = @(
    "-m", "evaluation.evaluator",
    "--checkpoint", $Checkpoint,
    "--asvspoof5-dir", $DatasetDir,
    "--split", $Split,
    "--output-dir", $OutputDir,
    "--batch-size", $BatchSize
)

if ($MaxSamples -gt 0) { $evalArgs += "--max-samples"; $evalArgs += $MaxSamples }
if ($SkipRobustness)   { $evalArgs += "--skip-robustness" }

python @evalArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Evaluation failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Reports written to: $OutputDir" -ForegroundColor Green
Write-Host ""
