# data/brats_dataset.py
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
from .text_generator import DiagnosisTextGenerator


class BraTSDataset(Dataset):
    """BraTS 2021 dataset with text generation."""

    MODALITIES = ['t1', 't1ce', 't2', 'flair']

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        tokenizer = None,
        max_text_len: int = 128,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        self.text_generator = DiagnosisTextGenerator()

        # Find all cases
        self.cases = self._find_cases()

    def _find_cases(self) -> list:
        """Find all case directories."""
        cases = []
        split_dir = os.path.join(self.data_dir, self.split)

        if not os.path.exists(split_dir):
            return cases

        for case_name in sorted(os.listdir(split_dir)):
            case_dir = os.path.join(split_dir, case_name)
            if os.path.isdir(case_dir):
                cases.append(case_dir)

        return cases

    def _load_nifti(self, path: str) -> np.ndarray:
        """Load NIfTI file."""
        return nib.load(path).get_fdata()

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case_dir = self.cases[idx]
        case_name = os.path.basename(case_dir)

        # Load modalities
        images = []
        for mod in self.MODALITIES:
            path = os.path.join(case_dir, f'{case_name}_{mod}.nii.gz')
            img = self._load_nifti(path)
            images.append(img)

        image = np.stack(images, axis=0).astype(np.float32)  # [4, D, H, W]

        # Load segmentation
        seg_path = os.path.join(case_dir, f'{case_name}_seg.nii.gz')
        mask = self._load_nifti(seg_path).astype(np.int64)

        # Normalize image
        for i in range(image.shape[0]):
            img_i = image[i]
            nonzero = img_i[img_i > 0]
            if len(nonzero) > 0:
                mean, std = nonzero.mean(), nonzero.std()
                image[i] = (img_i - mean) / (std + 1e-8)

        # Convert to tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        # Generate diagnosis text
        text = self.text_generator.generate(mask)

        # Tokenize
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            text_ids = tokens['input_ids'].squeeze(0)
        else:
            # Simple character-level tokenization as fallback
            text_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            for i, c in enumerate(text[:self.max_text_len]):
                text_ids[i] = ord(c) % 30000

        return {
            'image': image,
            'mask': mask,
            'text': text,
            'text_ids': text_ids,
            'case_name': case_name,
        }
