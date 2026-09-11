# VoxShield SIH Demo Launcher
# Starts backend + opens browser to frontend
#
# Usage:
#   .\scripts\demo.ps1 -DatasetDir "D:\Datasets\ASVspoof5"

param(
    [string]$BackendPort  = "8000",
    [string]$FrontendPort = "3000",
    [string]$Checkpoint   = "checkpoints\asvspoof5\best.pt"
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         VOXSHIELD DEMO LAUNCHER           ║" -ForegroundColor Cyan
Write-Host "║     Voice Threat Intelligence System      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check checkpoint
if (Test-Path $Checkpoint) {
    Write-Host "  ✓ Checkpoint: $Checkpoint" -ForegroundColor Green
} else {
    Write-Host "  ⚠ No checkpoint found at $Checkpoint" -ForegroundColor Yellow
    Write-Host "    Dashboard will show MODEL OFFLINE." -ForegroundColor Yellow
    Write-Host "    Train with: .\scripts\train_asvspoof5.ps1 -DatasetDir <PATH>" -ForegroundColor Yellow
}

if ($Checkpoint) { $env:VOXSHIELD_CHECKPOINT = $Checkpoint }
$env:VITE_API_URL = "http://localhost:$BackendPort"

Write-Host ""
Write-Host "  Starting backend on port $BackendPort ..." -ForegroundColor White
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectRoot'; uvicorn backend.app:app --host 0.0.0.0 --port $BackendPort --log-level info"
)

Start-Sleep 2

Write-Host "  Starting frontend on port $FrontendPort ..." -ForegroundColor White
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$ProjectRoot\frontend'; `$env:VITE_API_URL='http://localhost:$BackendPort'; npx vite --port $FrontendPort --host"
)

Start-Sleep 3

Write-Host ""
Write-Host "  Opening dashboard..." -ForegroundColor Green
Start-Process "http://localhost:$FrontendPort"

Write-Host ""
Write-Host "  VoxShield is running:" -ForegroundColor Green
Write-Host "    Frontend : http://localhost:$FrontendPort" -ForegroundColor White
Write-Host "    API      : http://localhost:$BackendPort" -ForegroundColor White
Write-Host "    API docs : http://localhost:$BackendPort/docs" -ForegroundColor White
Write-Host ""
Write-Host "  Close the PowerShell windows to stop." -ForegroundColor Yellow
