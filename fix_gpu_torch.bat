@echo off
REM ============================================================
REM  Replaces CPU-only torch with a CUDA-enabled build. Tries
REM  several CUDA index tags in order (newest first) since
REM  Python 3.13 wheels aren't available for every tag yet.
REM ============================================================

echo Your GPU/driver:
nvidia-smi
if errorlevel 1 (
    echo [!] nvidia-smi failed -- no NVIDIA GPU detected, or driver not on PATH.
    pause
    exit /b 1
)

echo.
echo Removing the CPU-only torch build ...
python -m pip uninstall -y torch

for %%T in (cu130 cu128 cu126 cu124 cu121 cu118) do (
    echo.
    echo ==== Trying %%T ====
    python -m pip install torch --index-url https://download.pytorch.org/whl/%%T
    python -c "import torch" 2>nul
    if not errorlevel 1 (
        echo.
        echo Installed successfully using %%T.
        goto :check
    )
    echo %%T did not have a matching build for this Python version, trying the next one...
)

echo.
echo [!] None of the CUDA-tagged indexes had a build for your Python version.
echo     Tell Claude: "python --version" output and this failure.
pause
exit /b 1

:check
echo.
echo Verifying GPU is visible to torch ...
python -c "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo.
echo If it printed "CUDA available: True", re-run run_bert_shap.bat.
pause
