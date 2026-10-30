@echo off
REM ============================================================
REM  Pulls live news from RSS feeds in English, Hindi, Tamil,
REM  Telugu and Malayalam. Standard library only, nothing to
REM  install. Writes feed_articles.csv + feed_status.csv.
REM ============================================================

echo Pulling live news feeds in all five languages...
echo (publisher feed URLs change often, so some may fail -- that's expected,
echo  the script reports which ones answered)
echo.
python "%~dp0pull_feeds.py"

echo.
pause
