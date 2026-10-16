

echo Installing required Python packages (torch, transformers, sentencepiece, blobfile)...
python -m pip install --quiet torch transformers sentencepiece blobfile
if errorlevel 1 (
    echo.
    echo [!] pip install failed. Make sure Python is installed and on PATH, then re-run this file.
    pause
    exit /b 1
)

echo.
echo Downloading mBERT and IndicBERT. This can take a few minutes
echo depending on your connection (mBERT ~700MB, IndicBERT ~130MB).
echo.
python "%~dp0download_models.py"


pause
