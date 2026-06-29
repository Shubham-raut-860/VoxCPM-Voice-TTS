#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "==> Voice Agent API — Setup" -ForegroundColor Cyan

# 1. Create venv
python -m venv venv
& .\venv\Scripts\Activate.ps1

# 2. (Optional) install torch with CUDA — comment out if already installed
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining deps
pip install --upgrade pip --quiet
pip install -r requirements.txt

# 4. Create .env if missing
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "==> Created .env from .env.example — edit before running" -ForegroundColor Yellow
}

# 5. Create storage dirs
New-Item -ItemType Directory -Force -Path "storage\voices"  | Out-Null
New-Item -ItemType Directory -Force -Path "storage\outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "storage\loras"   | Out-Null

Write-Host ""
Write-Host "==> Setup complete" -ForegroundColor Green
Write-Host "==> Run: venv\Scripts\uvicorn app.main:app --reload --port 8000" -ForegroundColor Green
Write-Host "==> Docs: http://localhost:8000/docs" -ForegroundColor Green
