# TextMamba3D

Text-guided 3D medical image segmentation using unified Mamba architecture.

## Features

- **Full Mamba Architecture**: Unified Mamba blocks for image encoding, text encoding, and fusion
- **Text-Guided Segmentation**: Leverage diagnosis text to improve segmentation quality
- **Edge Enhancement**: Dedicated edge loss for sharper boundaries
- **Contrastive Learning**: Text-image alignment via contrastive loss

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Download BraTS 2021 dataset and organize as:

```
data/BraTS2021/
├── train/
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
├── val/
└── test/
```

## Training

```bash
python train.py --config configs/default.yaml
```

## Evaluation

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

## Architecture

```
Image (4ch MRI) ──► 3D Mamba Encoder ──┐
                                       ├──► Mamba Fusion ──► 3D Mamba Decoder ──► Segmentation
Diagnosis Text ──► Text Mamba Encoder ─┘
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{textmamba3d,
  title={TextMamba3D: Text-Guided 3D Medical Image Segmentation},
  year={2026}
}
```
