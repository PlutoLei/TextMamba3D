#!/bin/bash
cd /mnt/e/VSCode_Project/TextMamba3D
source .venv/bin/activate
nohup python3 -u train.py --config configs/textbrats.yaml --resume checkpoints/last.pth > training_log.txt 2>&1 &
echo "Training PID: $!"
