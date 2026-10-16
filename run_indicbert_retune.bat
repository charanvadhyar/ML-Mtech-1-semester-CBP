@echo off
REM ============================================================
REM  Re-runs IndicBERT with settings suited to ALBERT-style models:
REM  higher LR (5e-5), 5 epochs, warmup + linear decay, grad
REM  clipping, more training data. Previous run sat at chance
REM  (48.4%), so its SHAP numbers were not meaningful.
REM  Overwrites bert_indicbert_* result files.
REM ============================================================

python train_bert_and_shap.py --model indicbert --lr 5e-5 --epochs 5 --n_train 8000 --batch_size 16

echo.
echo Check the "train acc" per epoch above: it should climb well past 0.5.
echo Then look at the final test accuracy. Paste the output to Claude.
pause
