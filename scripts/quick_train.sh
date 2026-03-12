#!/bin/bash
cd /mnt/e/VSCode_Project/TextMamba3D
source .venv/bin/activate
python3 train.py --config configs/textbrats.yaml --max-samples 50
