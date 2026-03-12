@echo off
title TextMamba3D Quick Train
echo Starting WSL + TextMamba3D training...
echo.
wsl -d Ubuntu -e bash -c "cd /mnt/e/VSCode_Project/TextMamba3D && source .venv/bin/activate && python3 train.py --config configs/textbrats.yaml --max-samples 50"
echo.
echo Training finished.
pause
