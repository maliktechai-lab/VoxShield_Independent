# VoxShield Live Demo Launcher
# ======================================================
# Starts the backend and frontend, then opens the browser.
#
# Usage:
#   cd D:\VoxShield_Independent
#   .\scripts\start_demo.ps1
#
# Optional parameters:
#   .\scripts\start_demo.ps1 -BackendPort 8000 -FrontendPort 3000
# ======================================================

param(
    [int] $BackendPort  = 8000,
    [int] $FrontendPort = 3000
)

$ErrorActionPreference = "Stop"

# Resolve project root from this script's directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "======================================================"
Write-Host "  VOXSHIELD LIVE DEMO LAUNCHER"
Write-Host "  Voice Anti-Spoofing Detection System"
Write-Host "======================================================"
Write-Host ""

# ----------------------------------------------------------
# Prerequisite checks
# ----------------------------------------------------------

# 1. Python / uvicorn
$PythonExe = $null
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "  [OK] Python  : $PythonExe (venv)"
} else {
    $global_py = Get-Command python -ErrorAction SilentlyContinue
    if ($global_py) {
        $PythonExe = $global_py.Source
        Write-Host "  [OK] Python  : $PythonExe (global)"
    } else {
        Write-Host "  [FAIL] Python not found. Install Python or create .venv." -ForegroundColor Red
        exit 1
    }
}

# Derive uvicorn path from the same Python environment
$UvicornExe = $null
$VenvUvicorn = Join-Path $ProjectRoot ".venv\Scripts\uvicorn.exe"
if (Test-Path $VenvUvicorn) {
    $UvicornExe = $VenvUvicorn
    Write-Host "  [OK] uvicorn : $UvicornExe"
} else {
    $global_uv = Get-Command uvicorn -ErrorAction SilentlyContinue
    if ($global_uv) {
        $UvicornExe = "uvicorn"
        Write-Host "  [OK] uvicorn : (global PATH)"
    } else {
        Write-Host "  [FAIL] uvicorn not found. Run: pip install uvicorn" -ForegroundColor Red
        exit 1
    }
}

# 2. Node / npm
$NpmExe = Get-Command npm -ErrorAction SilentlyContinue
if ($NpmExe) {
    Write-Host "  [OK] npm     : $($NpmExe.Source)"
} else {
    Write-Host "  [FAIL] npm not found. Install Node.js." -ForegroundColor Red
    exit 1
}

# 3. Checkpoint
$CkptPath = Join-Path $ProjectRoot "checkpoints\asvspoof5\best.pt"
if (Test-Path $CkptPath) {
    Write-Host "  [OK] Model   : checkpoints\asvspoof5\best.pt"
} else {
    Write-Host "  [WARN] Checkpoint not found: checkpoints\asvspoof5\best.pt" -ForegroundColor Yellow
    Write-Host "         Dashboard will show MODEL OFFLINE." -ForegroundColor Yellow
}

# 4. Frontend node_modules
$FrontendDir  = Join-Path $ProjectRoot "frontend"
$NodeModules  = Join-Path $FrontendDir "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "  [INFO] Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    npm install
    Pop-Location
}
Write-Host "  [OK] Frontend: $FrontendDir"

Write-Host ""

# ----------------------------------------------------------
# Build command strings (pure ASCII, no em-dashes, no Unicode)
# ----------------------------------------------------------

$CorsOrigins = "http://localhost:$FrontendPort,http://localhost:5173"
$ApiUrl      = "http://localhost:$BackendPort"

# Backend command: runs in a new PowerShell window
# Use -join to avoid multi-line string concatenation issues
$BackendCmd = @(
    "Set-Location '$ProjectRoot'",
    "`$env:VOXSHIELD_CORS_ORIGINS = '$CorsOrigins'",
    "Write-Host 'VoxShield Backend started on http://localhost:$BackendPort'",
    "& '$UvicornExe' backend.app:app --host 0.0.0.0 --port $BackendPort --log-level info"
) -join "; "

# Frontend command: runs in a new PowerShell window
$FrontendCmd = @(
    "Set-Location '$FrontendDir'",
    "`$env:VITE_API_URL = '$ApiUrl'",
    "Write-Host 'VoxShield Frontend started on http://localhost:$FrontendPort'",
    "npx vite --port $FrontendPort --host"
) -join "; "

# ----------------------------------------------------------
# Start backend
# ----------------------------------------------------------

Write-Host "  [1/3] Starting backend on port $BackendPort ..."
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $BackendCmd

Write-Host "        Waiting for backend to be ready..."
$MaxWait  = 30
$Interval = 2
$Elapsed  = 0
$BackendReady = $false

while ($Elapsed -lt $MaxWait) {
    Start-Sleep $Interval
    $Elapsed += $Interval
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$BackendPort/health" `
                                  -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $BackendReady = $true
            break
        }
    } catch {
        # Not ready yet - keep waiting
    }
}

if ($BackendReady) {
    Write-Host "        Backend is ready." -ForegroundColor Green
} else {
    Write-Host "        Backend did not respond within $MaxWait seconds." -ForegroundColor Yellow
    Write-Host "        Check the backend window for errors." -ForegroundColor Yellow
}

# ----------------------------------------------------------
# Start frontend
# ----------------------------------------------------------

Write-Host ""
Write-Host "  [2/3] Starting frontend on port $FrontendPort ..."
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $FrontendCmd

Write-Host "        Waiting for frontend to be ready..."
$Elapsed  = 0
$FrontendReady = $false

while ($Elapsed -lt $MaxWait) {
    Start-Sleep $Interval
    $Elapsed += $Interval
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$FrontendPort" `
                                  -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $FrontendReady = $true
            break
        }
    } catch {
        # Not ready yet
    }
}

if ($FrontendReady) {
    Write-Host "        Frontend is ready." -ForegroundColor Green
} else {
    Write-Host "        Frontend did not respond in time." -ForegroundColor Yellow
    Write-Host "        It may still be starting. Check the frontend window." -ForegroundColor Yellow
}

# ----------------------------------------------------------
# Open browser
# ----------------------------------------------------------

Write-Host ""
Write-Host "  [3/3] Opening browser at http://localhost:$FrontendPort ..."
Start-Process "http://localhost:$FrontendPort"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

$ModelStatus = if ($BackendReady) { "ready" } else { "unknown - check backend window" }

Write-Host ""
Write-Host "======================================================"
Write-Host "  VoxShield demo started"
Write-Host "  Backend  : http://localhost:$BackendPort"
Write-Host "  Frontend : http://localhost:$FrontendPort"
Write-Host "  Model    : $ModelStatus"
Write-Host "======================================================"
Write-Host ""
Write-Host "  LIVE DETECTION STEPS:"
Write-Host "    1. Open http://localhost:$FrontendPort in Chrome or Edge"
Write-Host "    2. Allow microphone access when the browser prompts"
Write-Host "    3. Click [Start Live Detection] in the bottom-right panel"
Write-Host "    4. Speak - first result appears after 4 seconds"
Write-Host "    5. Results update every 2 seconds"
Write-Host ""
Write-Host "  UPLOAD TEST (bonafide):"
Write-Host "    Drag a natural human speech WAV/FLAC file onto the upload area"
Write-Host "    Expected result: BONA FIDE, LOW threat"
Write-Host ""
Write-Host "  UPLOAD TEST (spoof):"
Write-Host "    Drag a TTS or voice-converted WAV/FLAC file onto the upload area"
Write-Host "    Expected result: SPOOF DETECTED, HIGH or CRITICAL threat"
Write-Host ""
Write-Host "  Close the two PowerShell windows to stop all services."
Write-Host ""
