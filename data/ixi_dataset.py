"""IXI dataset loader for SSL pretraining. Loads preprocessed 4-channel .npz files."""
import os, glob, torch, numpy as np
from torch.utils.data import Dataset


class IXIDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.transform = transform
        split_dir = os.path.join(data_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*', 'data.npz')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = torch.from_numpy(data['image'].astype(np.float32))  # [4, D, H, W]
        mask = torch.zeros(image.shape[1:], dtype=torch.long)
        if self.transform:
            image, mask = self.transform(image, mask)
        return {'image': image, 'mask': mask,
                'text_ids': torch.zeros(1, dtype=torch.long),
                'attention_mask': torch.zeros(1, dtype=torch.long)}
