# VoxShield Live Demo Launcher
# ==============================
# Starts backend + frontend, then opens the browser.
#
# Usage:
#   .\scripts\start_demo.ps1
#
# Requires:
#   - Python + uvicorn installed (pip install uvicorn)
#   - Node.js + npm installed
#   - checkpoints\asvspoof5\best.pt exists
#
# What this does:
#   1. Starts the VoxShield FastAPI backend on port 8000
#   2. Starts the Vite frontend dev server on port 3000
#   3. Opens http://localhost:3000 in the default browser
#   4. Live detection overlay is available immediately in the browser

param(
    [int]   $BackendPort  = 8000,
    [int]   $FrontendPort = 3000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     VOXSHIELD LIVE DEMO LAUNCHER          ║" -ForegroundColor Cyan
Write-Host "║   Voice Anti-Spoofing Detection System    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check checkpoint
$ckpt = "checkpoints\asvspoof5\best.pt"
if (Test-Path $ckpt) {
    Write-Host "  ✓ Checkpoint: $ckpt" -ForegroundColor Green
} else {
    Write-Host "  ✗ Checkpoint not found: $ckpt" -ForegroundColor Red
    Write-Host "    Dashboard will show MODEL OFFLINE." -ForegroundColor Yellow
}

# Set CORS to include both Vite ports
$env:VOXSHIELD_CORS_ORIGINS = "http://localhost:$FrontendPort,http://localhost:5173"
$env:VITE_API_URL = "http://localhost:$BackendPort"

Write-Host ""
Write-Host "  [1/3] Starting backend on port $BackendPort ..." -ForegroundColor White

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot'; " +
    "`$env:VOXSHIELD_CORS_ORIGINS='http://localhost:$FrontendPort,http://localhost:5173'; " +
    "Write-Host 'VoxShield Backend — http://localhost:$BackendPort' -ForegroundColor Cyan; " +
    "uvicorn backend.app:app --host 0.0.0.0 --port $BackendPort --log-level info"
)

Start-Sleep 4

Write-Host "  [2/3] Starting frontend on port $FrontendPort ..." -ForegroundColor White

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$ProjectRoot\frontend'; " +
    "`$env:VITE_API_URL='http://localhost:$BackendPort'; " +
    "Write-Host 'VoxShield Frontend — http://localhost:$FrontendPort' -ForegroundColor Cyan; " +
    "npx vite --port $FrontendPort --host"
)

Start-Sleep 4

Write-Host "  [3/3] Opening browser..." -ForegroundColor White
Start-Process "http://localhost:$FrontendPort"

Write-Host ""
Write-Host "  ✓ VoxShield is running" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard  : http://localhost:$FrontendPort" -ForegroundColor White
Write-Host "  Backend    : http://localhost:$BackendPort" -ForegroundColor White
Write-Host "  API docs   : http://localhost:$BackendPort/docs" -ForegroundColor White
Write-Host ""
Write-Host "  LIVE DETECTION:" -ForegroundColor Yellow
Write-Host "    1. Allow microphone access when the browser asks" -ForegroundColor White
Write-Host "    2. Click [Start Live Detection] in the bottom-right panel" -ForegroundColor White
Write-Host "    3. Speak — first result appears after 4 seconds" -ForegroundColor White
Write-Host "    4. Results update every ~2 seconds" -ForegroundColor White
Write-Host ""
Write-Host "  DEMO BONAFIDE TEST:" -ForegroundColor Yellow
Write-Host "    Upload a .wav/.flac file of natural human speech" -ForegroundColor White
Write-Host "    Expected: BONA FIDE · LOW threat" -ForegroundColor Green
Write-Host ""
Write-Host "  DEMO SPOOF TEST:" -ForegroundColor Yellow
Write-Host "    Upload a .wav/.flac file of TTS/voice-converted speech" -ForegroundColor White
Write-Host "    Expected: SPOOF DETECTED · HIGH/CRITICAL threat" -ForegroundColor Red
Write-Host ""
Write-Host "  Close the two PowerShell windows to stop all services." -ForegroundColor Yellow
Write-Host ""
