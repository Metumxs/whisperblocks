@echo off
setlocal
chcp 65001 >nul
title WhisperBlocks - Setup

cd /d "%~dp0"

echo.
echo  ============================================================
echo   WhisperBlocks - one-time setup
echo  ============================================================
echo.

REM --- 1. Verify Python 3.12 is available via the py launcher ---
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python 3.12 not found.
    echo.
    echo  Install it first, then re-run this script:
    echo      py install 3.12
    echo  or  winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

REM --- 2. Create the virtual environment (reuse if it already exists) ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if errorlevel 1 (
        echo  [ERROR] Existing .venv must use Python 3.12.
        echo  Delete .venv manually, then re-run install.bat.
        echo  This script will not delete your environment automatically.
        pause
        exit /b 1
    )
    echo  [i] .venv already exists with Python 3.12 - reusing it.
    echo      Delete the .venv folder manually to start clean.
) else (
    echo  [1/5] Creating .venv with Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create .venv.
        pause
        exit /b 1
    )
)

REM --- 3. Upgrade pip inside the venv ---
REM    The "python -m pip" form is required on Windows when pip is
REM    being replaced by itself.
echo.
echo  [2/5] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo  [ERROR] pip upgrade failed.
    pause
    exit /b 1
)

REM --- 4. Install CUDA-enabled PyTorch FIRST ---
REM    PyPI ships the CPU-only torch build on Windows. The CUDA build
REM    lives on the PyTorch index. Skipping this step is the single
REM    most common setup failure - GPU transcription then dies with
REM    "Torch not compiled with CUDA enabled".
REM
REM    Version must be 2.8.0: WhisperX 3.8.6 pins torch~=2.8.0.
echo.
echo  [3/5] Installing CUDA-enabled PyTorch (~3.5 GB download)...
".venv\Scripts\python.exe" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo  [ERROR] PyTorch installation failed.
    echo.
    echo  If the cu128 index is unreachable or your NVIDIA driver is too
    echo  old, try cu126 instead by editing this file:
    echo      --index-url https://download.pytorch.org/whl/cu126
    echo.
    pause
    exit /b 1
)

REM --- 5. Install WhisperX and the rest of the dependencies ---
echo.
echo  [4/5] Installing WhisperX and remaining dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements\requirements.txt
if errorlevel 1 (
    echo  [ERROR] requirements\requirements.txt installation failed.
    pause
    exit /b 1
)

REM --- 6. Verify the whole stack is coherent ---
echo.
echo  [5/5] Verifying installation...
".venv\Scripts\python.exe" -c "import sys; import torch; import importlib.metadata as m; available = torch.cuda.is_available(); print('  torch         :', torch.__version__); print('  CUDA runtime  :', torch.version.cuda); print('  cuda available:', available); print('  whisperx      :', m.version('whisperx')); print('  [ERROR] CPU-only torch wheel installed. Reinstall the CUDA wheel from the PyTorch index.' if torch.version.cuda is None else ('  [WARNING] CUDA runtime installed, but no active NVIDIA GPU/driver is available. CPU fallback works: set device = ' + chr(34) + 'cpu' + chr(34) + ' in config/config.toml.' if not available else '  CUDA GPU ready.')); sys.exit(1 if torch.version.cuda is None else 0)"
if errorlevel 1 (
    echo  [ERROR] Verification failed - see the message above.
    echo  If a CPU-only torch wheel was reported, activate .venv and run:
    echo      python -m pip uninstall -y torch torchaudio
    echo      python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    echo  Then re-run install.bat.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Setup complete.
echo.
echo   Double-click run.bat to start transcribing.
echo  ============================================================
echo.
pause
exit /b 0
