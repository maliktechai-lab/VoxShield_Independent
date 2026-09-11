# VoxShield Frontend Launcher
#
# Usage:
#   .\scripts\run_frontend.ps1             # dev mode (hot reload)
#   .\scripts\run_frontend.ps1 -Prod       # build and serve production build
#   .\scripts\run_frontend.ps1 -ApiUrl http://192.168.1.100:8000

param(
    [switch]$Prod    = $false,
    [string]$ApiUrl  = "http://localhost:8000",
    [int]   $Port    = 3000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$FrontendDir = Join-Path $ProjectRoot "frontend"
Set-Location $FrontendDir

# Set API URL for frontend
$env:VITE_API_URL = $ApiUrl

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VoxShield Frontend" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  API URL : $ApiUrl" -ForegroundColor White
Write-Host "  Port    : $Port" -ForegroundColor White
Write-Host ""

if ($Prod) {
    Write-Host "  Building production bundle..." -ForegroundColor Yellow
    npm run build
    Write-Host "  Serving dist/ on http://localhost:$Port" -ForegroundColor Green
    npx serve dist --listen $Port
} else {
    Write-Host "  Starting dev server on http://localhost:$Port" -ForegroundColor Green
    Write-Host "  Press Ctrl+C to stop." -ForegroundColor White
    Write-Host ""
    npx vite --port $Port --host
}
