#!/bin/bash
# Quick evaluation script for TextMamba3D
# Run in WSL2: bash quick_eval.sh
cd /mnt/e/VSCode_Project/BS6207/TextMamba3D
source .venv/bin/activate

echo "=== TextMamba3D Evaluation ==="
echo ""

# 1. WITH text guidance
echo ">>> Evaluating WITH text guidance (best.pth)..."
python3 evaluate.py \
    --config configs/textbrats.yaml \
    --checkpoint checkpoints/best.pth \
    --split val

echo ""
echo "---"
echo ""

# 2. WITHOUT text guidance
echo ">>> Evaluating WITHOUT text guidance (best_no_text.pth)..."
python3 evaluate.py \
    --config configs/textbrats.yaml \
    --checkpoint checkpoints/best_no_text.pth \
    --split val \
    --no-text

echo ""
echo "=== Done ==="
