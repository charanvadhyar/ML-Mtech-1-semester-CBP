@echo off
REM ============================================================
REM  Downloads the Hindi and Telugu fake news datasets from
REM  Zenodo (blocked from the Claude sandbox, reachable from
REM  your own network). ~105MB total. Extracts into datasets\.
REM ============================================================

echo Downloading Hindi + Telugu fake news datasets from Zenodo...
echo (about 105MB total, give it a few minutes)
echo.
python "%~dp0download_datasets.py"

echo.
echo Done. Paste the file listing above to Claude.
pause
