#!/bin/bash
cd /mnt/e/VSCode_Project/TextMamba3D
source .venv/bin/activate
echo "Starting mamba-ssm installation at $(date)" > install_mamba.log
pip install mamba-ssm >> install_mamba.log 2>&1
echo "Installation completed at $(date)" >> install_mamba.log
echo "Verifying installation..." >> install_mamba.log
python3 -c "from mamba_ssm import Mamba; print('Mamba import OK')" >> install_mamba.log 2>&1
echo "Done" >> install_mamba.log
