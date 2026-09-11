# VoxShield Backend Launcher
#
# Usage:
#   .\scripts\run_backend.ps1
#   .\scripts\run_backend.ps1 -Port 8000 -Reload
#
# Environment variables:
#   VOXSHIELD_CHECKPOINT — override checkpoint path
#   VOXSHIELD_DB         — override SQLite database path
#   VOXSHIELD_MAX_UPLOAD_MB — max upload size (default 25)

param(
    [string]$Host   = "0.0.0.0",
    [int]   $Port   = 8000,
    [switch]$Reload = $false,
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VoxShield Backend" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Set project root
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

# Optional checkpoint override
if ($Checkpoint -ne "") {
    $env:VOXSHIELD_CHECKPOINT = $Checkpoint
    Write-Host "  Using checkpoint: $Checkpoint" -ForegroundColor Yellow
}

# Check for checkpoint
$ckptPaths = @(
    "checkpoints\asvspoof5\best.pt",
    "checkpoints\asvspoof5\last.pt",
    "checkpoints\best.pt"
)
$found = $false
foreach ($p in $ckptPaths) {
    if (Test-Path $p) {
        Write-Host "  Checkpoint: $p" -ForegroundColor Green
        $found = $true
        break
    }
}
if (-not $found) {
    Write-Host "  WARNING: No checkpoint found." -ForegroundColor Yellow
    Write-Host "  The API will start but return MODEL OFFLINE for predictions." -ForegroundColor Yellow
    Write-Host "  Train a model first: .\scripts\train_asvspoof5.ps1" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Starting on http://${Host}:${Port}" -ForegroundColor White
Write-Host "  API docs: http://localhost:${Port}/docs" -ForegroundColor White
Write-Host "  Press Ctrl+C to stop." -ForegroundColor White
Write-Host ""

$reloadFlag = if ($Reload) { "--reload" } else { "" }

if ($reloadFlag) {
    uvicorn backend.app:app --host $Host --port $Port --reload --log-level info
} else {
    uvicorn backend.app:app --host $Host --port $Port --log-level info
}
