@echo off
REM ============================================================
REM  Launches the Streamlit explainability dashboard. Run this
REM  from inside the MLSHAP folder (same one with comments.csv,
REM  the classical_* / scraped_* / bert_* result files, etc.)
REM ============================================================

echo Installing required Python packages...
python -m pip install --quiet streamlit shap scikit-learn pyspellchecker joblib xgboost pandas numpy
if errorlevel 1 (
    echo.
    echo [!] pip install failed. Make sure Python is installed and on PATH, then re-run this file.
    pause
    exit /b 1
)

echo.
echo Starting the dashboard -- a browser tab should open automatically.
echo (Leave this window open while you use it; close it to stop the server.)
echo.
python -m streamlit run dashboard.py
pause
