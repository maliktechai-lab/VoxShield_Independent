# VoxShield Setup Script
# Run this once to install Python and Node dependencies.
#
# Usage (PowerShell):
#   .\scripts\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VoxShield Independent — Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── Check Python ──────────────────────────────────────────────────────────────
Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow
$pyver = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "  Found: $pyver" -ForegroundColor Green

# ── Check Node ────────────────────────────────────────────────────────────────
Write-Host "[2/4] Checking Node.js..." -ForegroundColor Yellow
$nodever = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Node.js not found. Install from https://nodejs.org" -ForegroundColor Red
    exit 1
}
Write-Host "  Found: Node $nodever" -ForegroundColor Green

# ── Install Python dependencies ───────────────────────────────────────────────
Write-Host "[3/4] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some Python packages may not have installed. Check requirements.txt." -ForegroundColor Yellow
}
Write-Host "  Python dependencies installed." -ForegroundColor Green

# ── Install frontend dependencies ─────────────────────────────────────────────
Write-Host "[4/4] Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: npm install failed." -ForegroundColor Red
    Set-Location ..
    exit 1
}
Set-Location ..
Write-Host "  Frontend dependencies installed." -ForegroundColor Green

# ── Create required directories ───────────────────────────────────────────────
@("checkpoints", "checkpoints/asvspoof5", "reports", "reports/asvspoof5") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Prepare dataset : python -m training.prepare_asvspoof5 --dataset-dir <PATH>"
Write-Host "  2. Train model     : .\scripts\train_asvspoof5.ps1 -DatasetDir <PATH>"
Write-Host "  3. Start backend   : .\scripts\run_backend.ps1"
Write-Host "  4. Start frontend  : .\scripts\run_frontend.ps1"
Write-Host ""
