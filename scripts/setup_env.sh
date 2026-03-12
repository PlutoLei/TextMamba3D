#!/bin/bash
cd /mnt/e/VSCode_Project/TextMamba3D
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm
pip install monai nibabel numpy scipy transformers pyyaml tensorboard tqdm einops
echo "=== Installation complete ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
