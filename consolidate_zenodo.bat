@echo off
REM ============================================================
REM  Collapses the ~40,600 per-article .txt files in
REM  datasets\hindi and datasets\telugu into one CSV per
REM  language, so they can be moved and trained on.
REM ============================================================

echo Reading the Hindi and Telugu article files (about 40,600 of them).
echo This takes a couple of minutes.
echo.
python "%~dp0consolidate_zenodo.py"

echo.
pause
