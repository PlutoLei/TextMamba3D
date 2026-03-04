# data/brats_textbrats_dataset.py
"""
TextBraTS Dataset Loader
- 使用专家标注的文本描述（无信息泄露）
- 支持 BraTS2020 数据格式
"""
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import Optional, Callable, List


class TextBraTSDataset(Dataset):
    """
    TextBraTS dataset with expert-annotated text descriptions.

    Data structure:
        data_dir/
            BraTS20_Training_001/
                BraTS20_Training_001_flair.nii
                BraTS20_Training_001_t1.nii
                BraTS20_Training_001_t1ce.nii
                BraTS20_Training_001_t2.nii
                BraTS20_Training_001_seg.nii
                BraTS20_Training_001_flair_text.txt  # Expert text
                BraTS20_Training_001_flair_text.npy  # Text features (optional)
    """

    MODALITIES = ['t1', 't1ce', 't2', 'flair']

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        tokenizer=None,
        max_text_len: int = 128,
        use_text_features: bool = False,  # 是否使用预计算的文本特征
        train_ratio: float = 0.8,  # 训练集比例
        val_ratio: float = 0.0,  # 验证集比例（0 = 用 1-train_ratio）
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.use_text_features = use_text_features

        # Find all cases and split
        all_cases = self._find_cases()
        self.cases = self._split_cases(all_cases, split, train_ratio, val_ratio, seed)

        print(f"TextBraTS {split}: {len(self.cases)} samples")

    def _find_cases(self) -> List[str]:
        """Find all case directories with both MRI and text data."""
        cases = []

        if not os.path.exists(self.data_dir):
            print(f"Warning: data_dir not found: {self.data_dir}")
            return cases

        for case_name in sorted(os.listdir(self.data_dir)):
            case_dir = os.path.join(self.data_dir, case_name)
            if not os.path.isdir(case_dir):
                continue

            # Check required files exist
            text_file = os.path.join(case_dir, f"{case_name}_flair_text.txt")
            seg_file_nii = os.path.join(case_dir, f"{case_name}_seg.nii")
            seg_file_nii_gz = os.path.join(case_dir, f"{case_name}_seg.nii.gz")

            has_text = os.path.exists(text_file)
            has_seg = os.path.exists(seg_file_nii) or os.path.exists(seg_file_nii_gz)

            if has_text and has_seg:
                cases.append(case_dir)

        return cases

    def _split_cases(
        self,
        all_cases: List[str],
        split: str,
        train_ratio: float,
        val_ratio: float,
        seed: int
    ) -> List[str]:
        """Split cases into train/val/test sets.

        If val_ratio > 0: 3-way split (train / val / test).
        If val_ratio == 0: 2-way split (train / val), backward compatible.
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(all_cases))

        train_size = int(len(all_cases) * train_ratio)

        if val_ratio > 0:
            val_size = int(len(all_cases) * val_ratio)
            if split == 'train':
                selected_indices = indices[:train_size]
            elif split == 'val':
                selected_indices = indices[train_size:train_size + val_size]
            else:  # test
                selected_indices = indices[train_size + val_size:]
        else:
            if split == 'train':
                selected_indices = indices[:train_size]
            else:  # val
                selected_indices = indices[train_size:]

        return [all_cases[i] for i in selected_indices]

    def _load_nifti(self, path: str) -> np.ndarray:
        """Load NIfTI file (.nii or .nii.gz)."""
        if os.path.exists(path):
            return nib.load(path).get_fdata()
        elif os.path.exists(path + '.gz'):
            return nib.load(path + '.gz').get_fdata()
        elif path.endswith('.nii.gz') and os.path.exists(path[:-3]):
            return nib.load(path[:-3]).get_fdata()
        else:
            raise FileNotFoundError(f"NIfTI file not found: {path}")

    def _load_text(self, case_dir: str, case_name: str) -> str:
        """Load expert-annotated text description."""
        text_file = os.path.join(case_dir, f"{case_name}_flair_text.txt")

        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            return "Brain MRI scan with tumor."  # Fallback

    def _load_text_features(self, case_dir: str, case_name: str) -> Optional[np.ndarray]:
        """Load pre-computed text features if available."""
        npy_file = os.path.join(case_dir, f"{case_name}_flair_text.npy")

        if os.path.exists(npy_file):
            return np.load(npy_file)
        return None

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case_dir = self.cases[idx]
        case_name = os.path.basename(case_dir)

        # Load modalities
        images = []
        for mod in self.MODALITIES:
            # Try .nii first, then .nii.gz
            path_nii = os.path.join(case_dir, f'{case_name}_{mod}.nii')
            path_nii_gz = os.path.join(case_dir, f'{case_name}_{mod}.nii.gz')

            if os.path.exists(path_nii):
                img = self._load_nifti(path_nii)
            elif os.path.exists(path_nii_gz):
                img = self._load_nifti(path_nii_gz)
            else:
                raise FileNotFoundError(f"Modality {mod} not found for {case_name}")

            images.append(img)

        image = np.stack(images, axis=0).astype(np.float32)  # [4, D, H, W]

        # Load segmentation
        seg_path_nii = os.path.join(case_dir, f'{case_name}_seg.nii')
        seg_path_nii_gz = os.path.join(case_dir, f'{case_name}_seg.nii.gz')

        if os.path.exists(seg_path_nii):
            mask = self._load_nifti(seg_path_nii).astype(np.int64)
        else:
            mask = self._load_nifti(seg_path_nii_gz).astype(np.int64)

        # BraTS label mapping: 0->0, 1->1, 2->2, 4->3
        mask[mask == 4] = 3

        # Normalize image (per-modality z-score normalization)
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

        # Load expert text (NO information leakage!)
        text = self._load_text(case_dir, case_name)

        # Tokenize text
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            text_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            # Simple character-level tokenization as fallback
            text_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            for i, c in enumerate(text[:self.max_text_len]):
                text_ids[i] = ord(c) % 30000
            attention_mask = (text_ids > 0).long()

        result = {
            'image': image,
            'mask': mask,
            'text': text,
            'text_ids': text_ids,
            'attention_mask': attention_mask,
            'case_name': case_name,
        }

        # Optionally load pre-computed text features
        if self.use_text_features:
            text_features = self._load_text_features(case_dir, case_name)
            if text_features is not None:
                result['text_features'] = torch.from_numpy(text_features)

        return result


def get_textbrats_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 2,
    transform=None,
    tokenizer=None,
    max_text_len: int = 128,
    train_ratio: float = 0.8,
):
    """Create train and validation dataloaders for TextBraTS."""
    from torch.utils.data import DataLoader

    train_dataset = TextBraTSDataset(
        data_dir=data_dir,
        split='train',
        transform=transform,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        train_ratio=train_ratio,
    )

    val_dataset = TextBraTSDataset(
        data_dir=data_dir,
        split='val',
        transform=transform,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        train_ratio=train_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
