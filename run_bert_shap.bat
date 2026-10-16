@echo off
REM ============================================================
REM  Fine-tunes mBERT and IndicBERT (already downloaded locally)
REM  on the same DravidianCodeMix sentiment task as the classical
REM  model, then computes PartitionSHAP and the Language-Segment
REM  SHAP breakdown for each. Run from this folder.
REM ============================================================

echo Installing required Python packages...
python -m pip install --quiet shap scikit-learn pyspellchecker pandas numpy
if errorlevel 1 (
    echo.
    echo [!] pip install failed. Make sure Python is installed and on PATH, then re-run this file.
    pause
    exit /b 1
)

echo.
echo ================= mBERT =================
python train_bert_and_shap.py --model mbert
if errorlevel 1 (
    echo [!] mBERT run failed -- see the error above.
)

echo.
echo ================ IndicBERT ================
python train_bert_and_shap.py --model indicbert
if errorlevel 1 (
    echo [!] IndicBERT run failed -- see the error above.
)

echo.
echo Done. Outputs in this folder:
echo   bert_mbert_language_segment_shap.csv / _summary.json        (DravidianCodeMix validation)
echo   bert_mbert_scraped_language_segment_shap.csv / _summary.json (REAL scraped comments -- the actual demo)
echo   bert_indicbert_* (same, for IndicBERT)
echo Tell Claude they're ready.
pause
