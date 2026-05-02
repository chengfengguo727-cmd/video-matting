# Install FFmpeg via winget (Gyan.FFmpeg build)
# Run from elevated PowerShell:  .\scripts\install_ffmpeg.ps1

$ErrorActionPreference = "Stop"

Write-Host "Installing FFmpeg via winget ..." -ForegroundColor Cyan
& winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
    throw "winget install failed (exit $LASTEXITCODE). You can also download from https://ffmpeg.org/download.html and add to PATH manually."
}

Write-Host ""
Write-Host "Restart this PowerShell window so PATH refreshes, then run:" -ForegroundColor Yellow
Write-Host "    ffmpeg -version" -ForegroundColor Yellow
