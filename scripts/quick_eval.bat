@echo off
title TextMamba3D Evaluation
echo Starting WSL + TextMamba3D evaluation...
echo.
wsl -d Ubuntu -e bash -c "cd /mnt/e/VSCode_Project/TextMamba3D && source .venv/bin/activate && bash scripts/quick_eval.sh"
echo.
echo Evaluation finished.
pause
