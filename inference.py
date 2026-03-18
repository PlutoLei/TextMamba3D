# inference.py
"""
Inference script for TextMamba3D.

Performs inference on new MRI scans without requiring ground truth masks.
Uses learnable default text embeddings for text-free inference.

Usage:
    python inference.py --checkpoint best_model.pth --input /path/to/case_dir
    python inference.py --checkpoint best_model.pth --input /path/to/cases --batch
    python inference.py --checkpoint best_model.pth --input /path/to/case --text "clinical text"
"""

import argparse
import os
from pathlib import Path
from typing import Iterator, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models import TextMamba3D
from models.text_encoder import TextMambaEncoder
from utils.sliding_window import gaussian_weight_3d
from utils.tta import tta_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TextMamba3D Inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./predictions')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--save_prob', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (flip-based)')
    parser.add_argument('--tta-flips', type=int, default=8, choices=[4, 8],
                        help='Number of TTA flip augmentations (4 or 8)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP (mixed-precision) inference')
    parser.add_argument('--no-gaussian', action='store_true',
                        help='Disable Gaussian weighting for sliding window')
    return parser.parse_args()


class InferenceDataLoader:
    """Data loader for inference without ground truth."""

    MODALITIES = ['t1', 't1ce', 't2', 'flair']

    def __init__(self, patch_size: tuple[int, int, int] = (96, 96, 96)) -> None:
        self.patch_size = patch_size

    def load_case(self, case_dir: str) -> dict:
        """Load a single case for inference."""
        case_dir = Path(case_dir)
        case_name = case_dir.name

        images = []
        for mod in self.MODALITIES:
            img_path = self._find_modality_file(case_dir, case_name, mod)
            images.append(nib.load(str(img_path)).get_fdata())

        image = np.stack(images, axis=0).astype(np.float32)
        image = self._normalize(image)

        return {
            'image': torch.from_numpy(image),
            'case_name': case_name,
            'original_shape': image.shape[1:],
            'case_dir': str(case_dir),
        }

    def _find_modality_file(self, case_dir: Path, case_name: str, mod: str) -> Path:
        """Find modality file with various naming conventions."""
        patterns = [
            f'{case_name}_{mod}.nii.gz',
            f'{case_name}_{mod}.nii',
            f'{mod}.nii.gz',
            f'{mod}.nii',
        ]
        for pattern in patterns:
            candidate = case_dir / pattern
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find {mod} modality in {case_dir}")

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Z-score normalization per modality."""
        for i in range(image.shape[0]):
            nonzero = image[i][image[i] > 0]
            if len(nonzero) > 0:
                image[i] = (image[i] - nonzero.mean()) / (nonzero.std() + 1e-8)
        return image

    def get_patches(
        self, image: torch.Tensor, overlap: float = 0.5
    ) -> Iterator[tuple[torch.Tensor, tuple[int, int, int]]]:
        """Extract overlapping patches for sliding window inference."""
        _, D, H, W = image.shape
        pd, ph, pw = self.patch_size

        stride = tuple(int(p * (1 - overlap)) for p in self.patch_size)

        def get_coords(size: int, patch_size: int, stride: int) -> list[int]:
            coords = list(range(0, max(size - patch_size + 1, 1), stride))
            if coords[-1] + patch_size < size:
                coords.append(size - patch_size)
            return coords

        d_coords = get_coords(D, pd, stride[0])
        h_coords = get_coords(H, ph, stride[1])
        w_coords = get_coords(W, pw, stride[2])

        for d in d_coords:
            for h in h_coords:
                for w in w_coords:
                    yield image[:, d:d+pd, h:h+ph, w:w+pw], (d, h, w)


class TextMamba3DInference:
    """Inference wrapper for TextMamba3D model."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
        use_tta: bool = False,
        tta_flips: int = 8,
        use_amp: bool = True,
        use_gaussian: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_tta = use_tta
        self.tta_flips = tta_flips
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.use_gaussian = use_gaussian

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        text_max_len = self.config['model'].get('text_max_len', 128)
        use_pretrained_text = self.config['model'].get('use_pretrained_text', True)

        self.model = TextMamba3D(
            img_size=tuple(self.config['model']['img_size']),
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels'],
            embed_dim=self.config['model']['embed_dim'],
            depths=self.config['model']['depths'],
            text_embed_dim=self.config['model']['text_embed_dim'],
            text_max_len=text_max_len,
            use_pretrained_text=use_pretrained_text,
            use_text_gate=self.config['model'].get('use_text_gate', False),
            use_cross_scale_skip=self.config['model'].get('use_cross_scale_skip', False),
            text_gate_init_bias=self.config['model'].get('text_gate_init_bias', 2.0),
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f'Model loaded from {checkpoint_path}')
        print(f'Device: {self.device}')

        self.data_loader = InferenceDataLoader(
            patch_size=tuple(self.config['data']['patch_size'])
        )

        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize tokenizer with fallback."""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(TextMambaEncoder.PUBMEDBERT_NAME)
        except ImportError:
            print('Warning: transformers not installed, using fallback tokenization')
            return None

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        max_len = self.config['model'].get('text_max_len', 128)

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            return tokens['input_ids'].to(self.device)

        text_ids = torch.zeros(1, max_len, dtype=torch.long)
        for i, c in enumerate(text[:max_len]):
            text_ids[0, i] = ord(c) % 30000
        return text_ids.to(self.device)

    @torch.no_grad()
    def predict_patch(
        self, patch: torch.Tensor, text_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict on a single patch.

        Always returns softmax probabilities of shape [num_classes, D, H, W],
        regardless of whether TTA is enabled.
        """
        patch = patch.unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            if self.use_tta:
                pred = tta_predict(
                    self.model, patch, text_ids,
                    use_text=text_ids is not None,
                    num_flips=self.tta_flips,
                )
            else:
                pred = self.model(patch, text_ids, use_text=text_ids is not None)
                pred = F.softmax(pred, dim=1)

        return pred.squeeze(0)

    @torch.no_grad()
    def predict_volume(
        self,
        image: torch.Tensor,
        text: Optional[str] = None,
        use_sliding_window: bool = True,
        overlap: float = 0.5,
    ) -> torch.Tensor:
        """Predict segmentation for a full volume."""
        _, D, H, W = image.shape
        num_classes = self.config['model']['out_channels']
        patch_size = tuple(self.config['data']['patch_size'])
        text_ids = self.tokenize_text(text) if text else None

        fits_in_patch = D <= patch_size[0] and H <= patch_size[1] and W <= patch_size[2]
        if not use_sliding_window or fits_in_patch:
            return self._predict_direct(image, text_ids, patch_size)

        return self._predict_sliding_window(image, text_ids, num_classes, patch_size, overlap)

    def _predict_direct(
        self, image: torch.Tensor, text_ids: Optional[torch.Tensor], patch_size: tuple
    ) -> torch.Tensor:
        """Direct inference with padding if necessary."""
        _, D, H, W = image.shape
        pad_d = max(0, patch_size[0] - D)
        pad_h = max(0, patch_size[1] - H)
        pad_w = max(0, patch_size[2] - W)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0
            )

        pred = self.predict_patch(
            image[:, :patch_size[0], :patch_size[1], :patch_size[2]], text_ids
        )
        return pred[:, :D, :H, :W]

    def _predict_sliding_window(
        self, image: torch.Tensor, text_ids: Optional[torch.Tensor],
        num_classes: int, patch_size: tuple, overlap: float
    ) -> torch.Tensor:
        """Sliding window inference with optional Gaussian weighting."""
        _, D, H, W = image.shape
        prediction = torch.zeros(num_classes, D, H, W, device='cpu')
        count_map = torch.zeros(1, D, H, W, device='cpu')

        patches = list(self.data_loader.get_patches(image, overlap))
        pd, ph, pw = patch_size

        # Pre-compute Gaussian weight map (or uniform fallback)
        if self.use_gaussian:
            sigma_scale = self.config.get('inference', {}).get('gaussian_sigma_scale', 0.125)
            gauss_weight = gaussian_weight_3d(patch_size, sigma_scale=sigma_scale)
        else:
            gauss_weight = None

        for patch, (d, h, w) in tqdm(patches, desc='Sliding window', leave=False):
            pred = self.predict_patch(patch, text_ids).cpu()

            if gauss_weight is not None:
                # gauss_weight: [D, H, W] -> unsqueeze(0) -> [1, D, H, W]
                prediction[:, d:d+pd, h:h+ph, w:w+pw] += pred * gauss_weight.unsqueeze(0)
                count_map[:, d:d+pd, h:h+ph, w:w+pw] += gauss_weight.unsqueeze(0)
            else:
                prediction[:, d:d+pd, h:h+ph, w:w+pw] += pred
                count_map[:, d:d+pd, h:h+ph, w:w+pw] += 1

        return prediction / count_map.clamp(min=1e-8)

    def predict_case(
        self, case_dir: str, text: Optional[str] = None, return_prob: bool = False
    ) -> dict:
        """Run inference on a single case."""
        data = self.data_loader.load_case(case_dir)
        overlap = self.config.get('inference', {}).get('overlap', 0.5)
        prob = self.predict_volume(data['image'], text=text, overlap=overlap)

        prediction = prob.numpy() if return_prob else prob.argmax(dim=0).numpy().astype(np.uint8)

        return {
            'prediction': prediction,
            'case_name': data['case_name'],
            'original_shape': data['original_shape'],
        }


def save_prediction(
    prediction: np.ndarray, output_path: str, reference_nii: Optional[str] = None
) -> None:
    """Save prediction as NIfTI file."""
    if reference_nii and os.path.exists(reference_nii):
        ref = nib.load(reference_nii)
        nii = nib.Nifti1Image(prediction, ref.affine, ref.header)
    else:
        nii = nib.Nifti1Image(prediction, np.eye(4))
    nib.save(nii, output_path)


def get_case_dirs(input_path: str, batch_mode: bool) -> list[str]:
    """Get list of case directories to process."""
    if not batch_mode:
        return [input_path]
    return [
        os.path.join(input_path, d)
        for d in sorted(os.listdir(input_path))
        if os.path.isdir(os.path.join(input_path, d))
    ]


def main() -> None:
    args = parse_args()

    inference = TextMamba3DInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_tta=args.tta,
        tta_flips=args.tta_flips,
        use_amp=not args.no_amp,
        use_gaussian=not args.no_gaussian,
    )

    os.makedirs(args.output, exist_ok=True)
    case_dirs = get_case_dirs(args.input, args.batch)
    print(f'Processing {len(case_dirs)} case(s)...')

    for case_dir in tqdm(case_dirs, desc='Cases'):
        try:
            result = inference.predict_case(
                case_dir=case_dir,
                text=args.text,
                return_prob=args.save_prob,
            )

            case_name = result['case_name']
            ref_nii = os.path.join(case_dir, f'{case_name}_t1.nii.gz')
            ref_nii = ref_nii if os.path.exists(ref_nii) else None

            if args.save_prob:
                output_path = os.path.join(args.output, f'{case_name}_prob.npy')
                np.save(output_path, result['prediction'])
            else:
                output_path = os.path.join(args.output, f'{case_name}_pred.nii.gz')
                save_prediction(result['prediction'], output_path, ref_nii)

            print(f'  Saved: {output_path}')
        except Exception as e:
            print(f'  Error processing {case_dir}: {e}')

    print(f'\nInference complete. Results saved to {args.output}')


if __name__ == '__main__':
    main()
