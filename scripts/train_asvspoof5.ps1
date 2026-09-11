# VoxShield ASVspoof5 Training Script
#
# Usage:
#   .\scripts\train_asvspoof5.ps1 -DatasetDir "D:\Datasets\ASVspoof5"
#   .\scripts\train_asvspoof5.ps1 -DatasetDir "D:\Datasets\ASVspoof5" -Resume
#   .\scripts\train_asvspoof5.ps1 -DatasetDir "D:\Datasets\ASVspoof5" -Epochs 50 -BatchSize 64
#
# Checkpoint output: checkpoints\asvspoof5\best.pt

param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetDir,

    [string]$CheckpointDir = "checkpoints\asvspoof5",
    [int]   $Epochs        = 30,
    [int]   $BatchSize     = 32,
    [int]   $NumWorkers    = 4,
    [float] $LR            = 0.001,
    [int]   $Patience      = 8,
    [int]   $Seed          = 42,
    [switch]$Resume        = $false,
    [switch]$AMP           = $false,
    [switch]$PrepareOnly   = $false
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VoxShield — ASVspoof5 Training" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Dataset   : $DatasetDir" -ForegroundColor White
Write-Host "  Checkpoint: $CheckpointDir" -ForegroundColor White
Write-Host "  Epochs    : $Epochs" -ForegroundColor White
Write-Host "  Batch size: $BatchSize" -ForegroundColor White
Write-Host "  Workers   : $NumWorkers" -ForegroundColor White
Write-Host "  Resume    : $Resume" -ForegroundColor White
Write-Host ""

# ── Verify dataset exists ─────────────────────────────────────────────────────
if (-not (Test-Path $DatasetDir)) {
    Write-Host "ERROR: Dataset directory not found: $DatasetDir" -ForegroundColor Red
    exit 1
}

# ── Step 1: Prepare dataset index ─────────────────────────────────────────────
Write-Host "[1/2] Preparing dataset index..." -ForegroundColor Yellow
python -m training.prepare_asvspoof5 `
    --dataset-dir "$DatasetDir" `
    --output-json "reports\asvspoof5\dataset_info.json"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dataset preparation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Dataset verified." -ForegroundColor Green

if ($PrepareOnly) {
    Write-Host "  --PrepareOnly flag set. Stopping after preparation." -ForegroundColor Yellow
    exit 0
}

# ── Step 2: Train ─────────────────────────────────────────────────────────────
Write-Host "[2/2] Starting training..." -ForegroundColor Yellow
Write-Host "  Output: $CheckpointDir\best.pt" -ForegroundColor White
Write-Host ""

$trainArgs = @(
    "-m", "training.train",
    "--asvspoof5-dir", $DatasetDir,
    "--checkpoint-dir", $CheckpointDir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--lr", $LR,
    "--patience", $Patience,
    "--seed", $Seed
)

if ($Resume)  { $trainArgs += "--resume" }
if ($AMP)     { $trainArgs += "--amp" }

python @trainArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Training complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Best checkpoint: $CheckpointDir\best.pt" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  Evaluate : .\scripts\evaluate.ps1 -DatasetDir `"$DatasetDir`""
Write-Host "  Backend  : .\scripts\run_backend.ps1"
Write-Host ""
