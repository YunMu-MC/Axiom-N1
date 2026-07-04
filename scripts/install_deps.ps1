param(
    [switch]$Cuda
)
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CacheDir = Join-Path $ProjectRoot ".pip-cache"
$DownloadsDir = Join-Path $ProjectRoot "downloads"

if (-not (Test-Path $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
}

$env:PIP_CACHE_DIR = $CacheDir
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($Cuda) {
    New-Item -ItemType Directory -Force -Path $DownloadsDir | Out-Null
    $TorchWheel = Join-Path $DownloadsDir "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"
    if (-not (Test-Path $TorchWheel)) {
        $TorchUrl = "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl"
        curl.exe -L --fail --retry 20 --retry-delay 5 --continue-at - --output $TorchWheel $TorchUrl
    }
    & $VenvPython -m pip install $TorchWheel --no-index --find-links $DownloadsDir --timeout 300 --progress-bar off
    & $VenvPython -m pip install triton-windows -i "https://pypi.tuna.tsinghua.edu.cn/simple" --timeout 300 --retries 10 --progress-bar off
    & $VenvPython -m pip install numpy pyyaml tqdm pytest -i "https://pypi.tuna.tsinghua.edu.cn/simple" --timeout 180 --retries 10
} else {
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") -i "https://pypi.tuna.tsinghua.edu.cn/simple" --timeout 180 --retries 10
}
& $VenvPython -m pip install -e $ProjectRoot
if ($Cuda) {
    & $VenvPython (Join-Path $ProjectRoot "scripts\check_kernel.py") --config (Join-Path $ProjectRoot "configs\tiny_unit.yaml") --require-kernel
}
