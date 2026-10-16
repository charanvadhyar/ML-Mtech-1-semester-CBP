@echo off
REM ============================================================
REM  The very-new transformers release (5.x) has a bug where
REM  loading ANY model -- even plain BERT -- tries to import
REM  torchvision for an unrelated object-detection loss function,
REM  and crashes if torchvision doesn't exactly match your torch
REM  build. We don't need vision models at all, so pin to a
REM  stable, well-tested transformers release instead.
REM ============================================================

echo Removing the problematic transformers version ...
python -m pip uninstall -y transformers torchvision

echo.
echo Installing a stable transformers release (4.46.3) ...
python -m pip install "transformers==4.46.3"

echo.
echo Verifying BERT loads cleanly ...
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; import torch; print('transformers import OK'); print('CUDA still available:', torch.cuda.is_available())"

echo.
echo If it printed "transformers import OK" and "CUDA still available: True",
echo re-run run_bert_shap.bat.
pause
