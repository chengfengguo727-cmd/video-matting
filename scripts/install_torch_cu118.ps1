# Install PyTorch (CUDA 11.8) + project deps into a project-local venv.
# Run from PowerShell:  .\scripts\install_torch_cu118.ps1

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDir     = Join-Path $projectRoot "venv"
$venvPython  = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv at $venvDir ..." -ForegroundColor Cyan
    & python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
}

Write-Host "Using interpreter: $venvPython" -ForegroundColor DarkCyan

Write-Host "Upgrading pip ..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "Installing PyTorch 2.1.2 + CUDA 11.8 ..." -ForegroundColor Cyan
& $venvPython -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Host "Installing project requirements ..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed" }

Write-Host ""
Write-Host "Verifying CUDA availability ..." -ForegroundColor Cyan
& $venvPython -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed" }

Write-Host ""
Write-Host "Done. To activate the venv in your shell, run:" -ForegroundColor Green
Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "Or invoke directly without activation:" -ForegroundColor Green
Write-Host "    .\venv\Scripts\python.exe -m src.cli all .\input.mp4 -o .\out.webm" -ForegroundColor Yellow
