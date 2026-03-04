@echo off
title TextMamba3D Evaluation
echo Starting WSL + TextMamba3D evaluation...
echo.
wsl -d Ubuntu -e bash -c "cd /mnt/e/VSCode_Project/BS6207/TextMamba3D && source .venv/bin/activate && bash quick_eval.sh"
echo.
echo Evaluation finished.
pause
