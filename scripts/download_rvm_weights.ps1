# Download Robust Video Matting (RVM) MobileNetV3 weights
# Run:  .\scripts\download_rvm_weights.ps1

$ErrorActionPreference = "Stop"

$url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
$dst = Join-Path $PSScriptRoot "..\models\rvm_mobilenetv3.pth"
$dst = (Resolve-Path -LiteralPath (Split-Path $dst)).Path + "\rvm_mobilenetv3.pth"

if (Test-Path $dst) {
    Write-Host "Weights already exist at $dst" -ForegroundColor Yellow
    exit 0
}

Write-Host "Downloading $url ..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing
Write-Host "Saved to $dst" -ForegroundColor Green
