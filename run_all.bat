@echo off
setlocal
cd /d "%~dp0"
REM ============================================================
REM  Runs the whole project end to end, then opens the dashboard.
REM
REM  Stages:
REM    0  dependencies
REM    1  Tamil-English sentiment  (classical model + SHAP)
REM    2  Language-Segment SHAP    (the novelty)
REM    3  Apply to scraped YouTube comments
REM    4  Causal ablation
REM    5  Malayalam fake news      (shared-task splits)
REM    6  Five-language fake news  + cross-lingual transfer
REM    7  Benchmark audit          (the leakage finding)
REM    8  Pull live news feeds
REM    9  Live-news leakage consequences + length-shift stats
REM   10  Dashboard
REM
REM  Safe to re-run. Stages that need a file produced earlier will
REM  say so and skip rather than crash.
REM ============================================================

echo ============================================================
echo  STAGE 0  Dependencies
echo ============================================================
python -m pip install --quiet pandas numpy scikit-learn xgboost shap joblib pyspellchecker scipy streamlit openpyxl
if errorlevel 1 (
    echo [!] pip install failed. Check Python is on PATH.
    pause
    exit /b 1
)

REM The Hindi/Telugu loaders read raw\*_lead.csv -- put them there if the
REM consolidation step left them in the folder root.
if not exist "raw" mkdir raw
if exist "hindi_fakenews_lead.csv"  if not exist "raw\hindi_fakenews_lead.csv"  copy /y "hindi_fakenews_lead.csv"  "raw\" >nul
if exist "telugu_fakenews_lead.csv" if not exist "raw\telugu_fakenews_lead.csv" copy /y "telugu_fakenews_lead.csv" "raw\" >nul

echo.
echo ============================================================
echo  STAGE 1  Tamil-English sentiment, classical model + SHAP
echo ============================================================
if exist "dravidian_sentiment_train.csv" (
    python train_classical.py
) else (
    echo [skip] dravidian_sentiment_train.csv missing
)

echo.
echo ============================================================
echo  STAGE 2  Language-Segment SHAP Attribution
echo ============================================================
if exist "classical_tree_shap_values.npy" (
    python language_segment_shap.py
) else (
    echo [skip] stage 1 outputs missing
)

echo.
echo ============================================================
echo  STAGE 3  Apply to our own scraped YouTube comments
echo ============================================================
if exist "comments.csv" (
    python apply_classical_to_scraped.py
) else (
    echo [skip] comments.csv missing
)

echo.
echo ============================================================
echo  STAGE 4  Causal ablation: language and entity masking
echo ============================================================
if exist "comments.csv" (
    python ablation_study.py
) else (
    echo [skip] comments.csv missing
)

echo.
echo ============================================================
echo  STAGE 5  Malayalam fake news, official shared-task splits
echo ============================================================
if exist "Task-1_trainset.csv" (
    python train_fakenews.py
) else (
    echo [skip] Task-1_trainset.csv missing
)

echo.
echo ============================================================
echo  STAGE 6  Five languages + cross-lingual transfer matrix
echo ============================================================
python prepare_multilingual.py
if exist "multilingual_fakenews.csv" (
    python train_multilingual.py
) else (
    echo [skip] prepare_multilingual.py produced nothing
)

echo.
echo ============================================================
echo  STAGE 7  Benchmark audit -- the leakage finding
echo ============================================================
if exist "multilingual_fakenews.csv" (
    python benchmark_audit.py
    python leakage_ablation.py
) else (
    echo [skip] stage 6 outputs missing
)

echo.
echo ============================================================
echo  STAGE 8  Pull live news feeds, all five languages
echo ============================================================
python pull_feeds.py

echo.
echo ============================================================
echo  STAGE 9  Do the leaks cause real-world false accusations?
echo ============================================================
if exist "feed_articles.csv" (
    python feed_leakage_test.py
    python length_shift_analysis.py
) else (
    echo [skip] no feed articles pulled
)

echo.
echo ============================================================
echo  STAGE 10  Dashboard
echo ============================================================
echo Everything is computed. Opening the dashboard in your browser.
echo Leave this window open while you use it; close it to stop the server.
echo.
python -m streamlit run dashboard.py

pause
