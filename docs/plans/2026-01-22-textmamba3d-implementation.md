# TextMamba3D Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a text-guided 3D medical image segmentation model using unified Mamba architecture to improve edge quality.

**Architecture:** Full Mamba architecture with 3D image encoder, text encoder, Mamba fusion module, and 3D decoder. Text features guide image segmentation through sequence concatenation and Mamba layers.

**Tech Stack:** PyTorch, Mamba (mamba-ssm), monai, nibabel, transformers (tokenizer only)

---

## Project Structure

```
TextMamba3D/
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── brats_dataset.py
│   ├── transforms.py
│   └── text_generator.py
├── models/
│   ├── __init__.py
│   ├── mamba_block.py
│   ├── encoder_3d.py
│   ├── text_encoder.py
│   ├── fusion.py
│   ├── decoder_3d.py
│   └── textmamba3d.py
├── losses/
│   ├── __init__.py
│   ├── dice_loss.py
│   ├── edge_loss.py
│   └── contrastive_loss.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_losses.py
├── requirements.txt
└── README.md
```

---

## Task 1: Project Initialization

**Files:**
- Create: `TextMamba3D/requirements.txt`
- Create: `TextMamba3D/configs/default.yaml`
- Create: `TextMamba3D/.gitignore`

**Step 1: Create requirements.txt**

```txt
torch>=2.0.0
mamba-ssm>=1.2.0
monai>=1.3.0
nibabel>=5.0.0
numpy>=1.24.0
scipy>=1.10.0
transformers>=4.30.0
pyyaml>=6.0
tensorboard>=2.14.0
tqdm>=4.65.0
einops>=0.7.0
pytest>=7.4.0
```

**Step 2: Create default config**

```yaml
# configs/default.yaml
data:
  data_dir: "./data/BraTS2021"
  patch_size: [96, 96, 96]
  batch_size: 2
  num_workers: 4

model:
  img_size: [96, 96, 96]
  in_channels: 4
  out_channels: 4  # background, necrotic, edema, enhancing
  embed_dim: 96
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]
  text_embed_dim: 256
  text_max_len: 128

loss:
  dice_weight: 1.0
  ce_weight: 1.0
  edge_weight: 1.0
  contrastive_weight: 0.5
  temperature: 0.07

training:
  epochs: 300
  lr: 1e-4
  weight_decay: 1e-5
  warmup_epochs: 10

eval:
  metrics: ["dice", "hd95", "assd"]
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
data/BraTS*/
checkpoints/
logs/
*.pt
*.pth
.DS_Store
.idea/
```

**Step 4: Initialize git repository**

Run:
```bash
cd TextMamba3D && git init && git add . && git commit -m "chore: initialize project structure"
```

---

## Task 2: Mamba Block Implementation

**Files:**
- Create: `TextMamba3D/models/__init__.py`
- Create: `TextMamba3D/models/mamba_block.py`
- Create: `TextMamba3D/tests/__init__.py`
- Create: `TextMamba3D/tests/test_models.py`

**Step 1: Create models/__init__.py**

```python
from .mamba_block import MambaBlock, MambaLayer
```

**Step 2: Create tests/__init__.py**

```python
# tests package
```

**Step 3: Write the failing test for MambaBlock**

```python
# tests/test_models.py
import torch
import pytest


class TestMambaBlock:
    def test_mamba_block_output_shape(self):
        """Test MambaBlock maintains input shape."""
        from models.mamba_block import MambaBlock

        block = MambaBlock(dim=96, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 1000, 96)  # [B, L, D]
        out = block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_mamba_layer_output_shape(self):
        """Test MambaLayer with multiple blocks."""
        from models.mamba_block import MambaLayer

        layer = MambaLayer(dim=96, depth=2, d_state=16)
        x = torch.randn(2, 1000, 96)
        out = layer(x)
        assert out.shape == x.shape
```

**Step 4: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestMambaBlock -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 5: Write MambaBlock implementation**

```python
# models/mamba_block.py
import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        if Mamba is not None:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simple linear layer for testing without mamba-ssm
            self.mamba = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
        Returns:
            [B, L, D] output tensor
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaLayer(nn.Module):
    """Stack of Mamba blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
```

**Step 6: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestMambaBlock -v`
Expected: PASS

**Step 7: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add MambaBlock and MambaLayer"
```

---

## Task 3: 3D Patch Embedding and Encoder

**Files:**
- Modify: `TextMamba3D/models/__init__.py`
- Create: `TextMamba3D/models/encoder_3d.py`
- Modify: `TextMamba3D/tests/test_models.py`

**Step 1: Write failing test for 3D encoder**

```python
# Add to tests/test_models.py

class TestEncoder3D:
    def test_patch_embed_3d_output_shape(self):
        """Test 3D patch embedding."""
        from models.encoder_3d import PatchEmbed3D

        embed = PatchEmbed3D(
            img_size=(96, 96, 96),
            patch_size=(4, 4, 4),
            in_channels=4,
            embed_dim=96,
        )
        x = torch.randn(2, 4, 96, 96, 96)  # [B, C, D, H, W]
        out = embed(x)
        # Expected: [B, (96/4)^3, 96] = [B, 13824, 96]
        expected_seq_len = (96 // 4) ** 3
        assert out.shape == (2, expected_seq_len, 96)

    def test_encoder_3d_output_shape(self):
        """Test full 3D Mamba encoder."""
        from models.encoder_3d import MambaEncoder3D

        encoder = MambaEncoder3D(
            img_size=(96, 96, 96),
            in_channels=4,
            embed_dim=96,
            depths=[2, 2],
            patch_size=(4, 4, 4),
        )
        x = torch.randn(2, 4, 96, 96, 96)
        features = encoder(x)

        # Should return multi-scale features
        assert isinstance(features, list)
        assert len(features) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestEncoder3D -v`
Expected: FAIL

**Step 3: Write 3D encoder implementation**

```python
# models/encoder_3d.py
import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import MambaLayer


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (4, 4, 4),
        in_channels: int = 4,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * \
                          (img_size[1] // patch_size[1]) * \
                          (img_size[2] // patch_size[2])

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)
        return x


class PatchMerging3D(nn.Module):
    """Patch merging for downsampling."""

    def __init__(self, dim: int, spatial_dims: tuple):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D*H*W, C]
        Returns:
            [B, D*H*W/8, 2*C]
        """
        B, L, C = x.shape
        D, H, W = self.spatial_dims

        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H, w=W)

        # Merge 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = rearrange(x, 'b d h w c -> b (d h w) c')

        x = self.norm(x)
        x = self.reduction(x)
        return x


class MambaEncoder3D(nn.Module):
    """3D Mamba Encoder with hierarchical features."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        patch_size: tuple = (4, 4, 4),
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Calculate spatial dimensions at each stage
        d, h, w = img_size[0] // patch_size[0], \
                  img_size[1] // patch_size[1], \
                  img_size[2] // patch_size[2]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, depth in enumerate(depths):
            # Mamba stage
            dim = embed_dim * (2 ** i)
            stage = MambaLayer(
                dim=dim,
                depth=depth,
                d_state=d_state,
                dropout=dropout,
            )
            self.stages.append(stage)

            # Downsample (except last stage)
            if i < len(depths) - 1:
                downsample = PatchMerging3D(
                    dim=dim,
                    spatial_dims=(d // (2 ** i), h // (2 ** i), w // (2 ** i)),
                )
                self.downsamples.append(downsample)

        self.spatial_dims = [
            (d // (2 ** i), h // (2 ** i), w // (2 ** i))
            for i in range(len(depths))
        ]

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            List of features at each stage
        """
        x = self.patch_embed(x)

        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return features
```

**Step 4: Update models/__init__.py**

```python
# models/__init__.py
from .mamba_block import MambaBlock, MambaLayer
from .encoder_3d import PatchEmbed3D, MambaEncoder3D
```

**Step 5: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestEncoder3D -v`
Expected: PASS

**Step 6: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add 3D patch embedding and Mamba encoder"
```

---

## Task 4: Text Mamba Encoder

**Files:**
- Create: `TextMamba3D/models/text_encoder.py`
- Modify: `TextMamba3D/models/__init__.py`
- Modify: `TextMamba3D/tests/test_models.py`

**Step 1: Write failing test**

```python
# Add to tests/test_models.py

class TestTextEncoder:
    def test_text_encoder_output_shape(self):
        """Test text Mamba encoder."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=30522,
            embed_dim=256,
            max_len=128,
            depth=2,
        )
        # Simulated token ids
        input_ids = torch.randint(0, 30522, (2, 64))
        out = encoder(input_ids)

        assert out.shape == (2, 64, 256)

    def test_text_encoder_global_feature(self):
        """Test global feature extraction."""
        from models.text_encoder import TextMambaEncoder

        encoder = TextMambaEncoder(
            vocab_size=30522,
            embed_dim=256,
            max_len=128,
            depth=2,
        )
        input_ids = torch.randint(0, 30522, (2, 64))
        out = encoder(input_ids)
        global_feat = encoder.get_global_feature(out)

        assert global_feat.shape == (2, 256)
```

**Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestTextEncoder -v`
Expected: FAIL

**Step 3: Write Text Mamba Encoder**

```python
# models/text_encoder.py
import torch
import torch.nn as nn
from .mamba_block import MambaLayer


class TextMambaEncoder(nn.Module):
    """Text encoder using Mamba architecture."""

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 256,
        max_len: int = 128,
        depth: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Mamba layers
        self.mamba_layers = MambaLayer(
            dim=embed_dim,
            depth=depth,
            d_state=d_state,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices
        Returns:
            [B, L, embed_dim] text features
        """
        B, L = input_ids.shape

        # Embed tokens
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :L, :]
        x = self.dropout(x)

        # Mamba encoding
        x = self.mamba_layers(x)
        x = self.norm(x)

        return x

    def get_global_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature via mean pooling.

        Args:
            x: [B, L, D] sequence features
        Returns:
            [B, D] global feature
        """
        return x.mean(dim=1)
```

**Step 4: Update models/__init__.py**

```python
# models/__init__.py
from .mamba_block import MambaBlock, MambaLayer
from .encoder_3d import PatchEmbed3D, MambaEncoder3D
from .text_encoder import TextMambaEncoder
```

**Step 5: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestTextEncoder -v`
Expected: PASS

**Step 6: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add Text Mamba encoder"
```

---

## Task 5: Mamba Fusion Module

**Files:**
- Create: `TextMamba3D/models/fusion.py`
- Modify: `TextMamba3D/models/__init__.py`
- Modify: `TextMamba3D/tests/test_models.py`

**Step 1: Write failing test**

```python
# Add to tests/test_models.py

class TestFusion:
    def test_mamba_fusion_output_shape(self):
        """Test Mamba fusion module."""
        from models.fusion import MambaFusion

        fusion = MambaFusion(
            img_dim=192,
            text_dim=256,
            hidden_dim=192,
            depth=2,
        )

        img_feat = torch.randn(2, 1000, 192)   # [B, N, D_img]
        text_feat = torch.randn(2, 64, 256)    # [B, M, D_text]

        out = fusion(img_feat, text_feat)

        # Output should match image feature shape
        assert out.shape == img_feat.shape
```

**Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestFusion -v`
Expected: FAIL

**Step 3: Write Mamba Fusion module**

```python
# models/fusion.py
import torch
import torch.nn as nn
from .mamba_block import MambaLayer


class MambaFusion(nn.Module):
    """Fuse image and text features using Mamba.

    Strategy: Concatenate [text, image] tokens, process with Mamba,
    then extract image portion. Text at the front guides image features
    through Mamba's causal nature.
    """

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        hidden_dim: int,
        depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project to common dimension
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Mamba fusion layers
        self.mamba_fusion = MambaLayer(
            dim=hidden_dim,
            depth=depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Project back to image dimension
        self.out_proj = nn.Linear(hidden_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_feat: [B, N, D_img] image features
            text_feat: [B, M, D_text] text features
        Returns:
            [B, N, D_img] fused image features
        """
        B, N, _ = img_feat.shape
        M = text_feat.shape[1]

        # Project to common space
        img_h = self.img_proj(img_feat)    # [B, N, hidden_dim]
        text_h = self.text_proj(text_feat)  # [B, M, hidden_dim]

        # Concatenate: [text, image] - text guides image
        concat = torch.cat([text_h, img_h], dim=1)  # [B, M+N, hidden_dim]

        # Mamba fusion
        fused = self.mamba_fusion(concat)  # [B, M+N, hidden_dim]

        # Extract image portion
        img_fused = fused[:, M:, :]  # [B, N, hidden_dim]

        # Project back and residual
        out = self.out_proj(img_fused)
        out = self.norm(out + img_feat)

        return out
```

**Step 4: Update models/__init__.py**

```python
# models/__init__.py
from .mamba_block import MambaBlock, MambaLayer
from .encoder_3d import PatchEmbed3D, MambaEncoder3D
from .text_encoder import TextMambaEncoder
from .fusion import MambaFusion
```

**Step 5: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestFusion -v`
Expected: PASS

**Step 6: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add Mamba fusion module"
```

---

## Task 6: 3D Mamba Decoder

**Files:**
- Create: `TextMamba3D/models/decoder_3d.py`
- Modify: `TextMamba3D/models/__init__.py`
- Modify: `TextMamba3D/tests/test_models.py`

**Step 1: Write failing test**

```python
# Add to tests/test_models.py

class TestDecoder3D:
    def test_decoder_output_shape(self):
        """Test 3D Mamba decoder."""
        from models.decoder_3d import MambaDecoder3D

        decoder = MambaDecoder3D(
            img_size=(96, 96, 96),
            patch_size=(4, 4, 4),
            out_channels=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
        )

        # Simulated encoder features (4 stages)
        features = [
            torch.randn(2, 24*24*24, 96),    # Stage 1
            torch.randn(2, 12*12*12, 192),   # Stage 2
            torch.randn(2, 6*6*6, 384),      # Stage 3
            torch.randn(2, 3*3*3, 768),      # Stage 4 (bottleneck)
        ]

        out = decoder(features)

        # Output should be [B, out_channels, D, H, W]
        assert out.shape == (2, 4, 96, 96, 96)
```

**Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestDecoder3D -v`
Expected: FAIL

**Step 3: Write 3D Mamba Decoder**

```python
# models/decoder_3d.py
import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import MambaLayer


class PatchExpanding3D(nn.Module):
    """Patch expanding for upsampling."""

    def __init__(self, dim: int, out_dim: int, spatial_dims: tuple):
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.expand = nn.Linear(dim, 8 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D*H*W, C]
        Returns:
            [B, 8*D*H*W, C/2]
        """
        B, L, C = x.shape
        D, H, W = self.spatial_dims

        x = self.expand(x)  # [B, L, 8*out_dim]
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c',
                     d=D, h=H, w=W, p1=2, p2=2, p3=2)
        x = rearrange(x, 'b d h w c -> b (d h w) c')
        x = self.norm(x)

        return x


class MambaDecoder3D(nn.Module):
    """3D Mamba Decoder with skip connections."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (4, 4, 4),
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_stages = len(depths)

        # Calculate spatial dimensions
        d, h, w = img_size[0] // patch_size[0], \
                  img_size[1] // patch_size[1], \
                  img_size[2] // patch_size[2]

        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        # Build decoder stages (reverse order)
        for i in range(len(depths) - 1, -1, -1):
            dim = embed_dim * (2 ** i)

            # Mamba stage
            stage = MambaLayer(
                dim=dim,
                depth=depths[i],
                d_state=d_state,
                dropout=dropout,
            )
            self.stages.append(stage)

            # Upsample (except first decoder stage)
            if i > 0:
                spatial = (d // (2 ** i), h // (2 ** i), w // (2 ** i))
                upsample = PatchExpanding3D(
                    dim=dim,
                    out_dim=dim // 2,
                    spatial_dims=spatial,
                )
                self.upsamples.append(upsample)

                # Skip connection projection
                skip_proj = nn.Linear(dim, dim // 2)
                self.skip_projs.append(skip_proj)

        # Final projection to output
        self.final_expand = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * (patch_size[0] ** 3)),
            nn.GELU(),
        )
        self.final_proj = nn.Conv3d(embed_dim, out_channels, 1)

        self.patch_size = patch_size
        self.base_spatial = (d, h, w)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [stage1, stage2, ..., bottleneck]
        Returns:
            [B, out_channels, D, H, W]
        """
        # Start from bottleneck
        x = features[-1]

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i < len(self.upsamples):
                x = self.upsamples[i](x)

                # Skip connection
                skip_idx = len(features) - 2 - i
                if skip_idx >= 0:
                    skip = self.skip_projs[i](features[skip_idx])
                    x = x + skip

        # Final expansion to original resolution
        B, L, C = x.shape
        d, h, w = self.base_spatial
        p = self.patch_size[0]

        x = self.final_expand(x)  # [B, L, C * p^3]
        x = rearrange(x, 'b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)',
                     d=d, h=h, w=w, p1=p, p2=p, p3=p, c=C)
        x = self.final_proj(x)

        return x
```

**Step 4: Update models/__init__.py**

```python
# models/__init__.py
from .mamba_block import MambaBlock, MambaLayer
from .encoder_3d import PatchEmbed3D, MambaEncoder3D
from .text_encoder import TextMambaEncoder
from .fusion import MambaFusion
from .decoder_3d import MambaDecoder3D
```

**Step 5: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestDecoder3D -v`
Expected: PASS

**Step 6: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add 3D Mamba decoder"
```

---

## Task 7: Complete TextMamba3D Model

**Files:**
- Create: `TextMamba3D/models/textmamba3d.py`
- Modify: `TextMamba3D/models/__init__.py`
- Modify: `TextMamba3D/tests/test_models.py`

**Step 1: Write failing test**

```python
# Add to tests/test_models.py

class TestTextMamba3D:
    def test_full_model_forward(self):
        """Test complete TextMamba3D model."""
        from models.textmamba3d import TextMamba3D

        model = TextMamba3D(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            text_embed_dim=256,
            text_depth=2,
        )

        img = torch.randn(2, 4, 96, 96, 96)
        text_ids = torch.randint(0, 30522, (2, 64))

        out = model(img, text_ids)

        assert out.shape == (2, 4, 96, 96, 96)

    def test_model_get_features_for_contrastive(self):
        """Test feature extraction for contrastive loss."""
        from models.textmamba3d import TextMamba3D

        model = TextMamba3D(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=4,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            text_embed_dim=256,
            text_depth=2,
        )

        img = torch.randn(2, 4, 96, 96, 96)
        text_ids = torch.randint(0, 30522, (2, 64))

        out, img_feat, text_feat = model(img, text_ids, return_features=True)

        assert img_feat.shape == (2, 256)  # Global image feature
        assert text_feat.shape == (2, 256)  # Global text feature
```

**Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestTextMamba3D -v`
Expected: FAIL

**Step 3: Write complete TextMamba3D model**

```python
# models/textmamba3d.py
import torch
import torch.nn as nn
from .encoder_3d import MambaEncoder3D
from .text_encoder import TextMambaEncoder
from .fusion import MambaFusion
from .decoder_3d import MambaDecoder3D


class TextMamba3D(nn.Module):
    """Text-guided 3D medical image segmentation with Mamba architecture."""

    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 2, 2],
        patch_size: tuple = (4, 4, 4),
        text_embed_dim: int = 256,
        text_max_len: int = 128,
        text_depth: int = 4,
        fusion_depth: int = 2,
        d_state: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 3D Image Encoder
        self.img_encoder = MambaEncoder3D(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            patch_size=patch_size,
            d_state=d_state,
            dropout=dropout,
        )

        # Text Encoder
        self.text_encoder = TextMambaEncoder(
            embed_dim=text_embed_dim,
            max_len=text_max_len,
            depth=text_depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Fusion at bottleneck
        bottleneck_dim = embed_dim * (2 ** (len(depths) - 1))
        self.fusion = MambaFusion(
            img_dim=bottleneck_dim,
            text_dim=text_embed_dim,
            hidden_dim=bottleneck_dim,
            depth=fusion_depth,
            d_state=d_state,
            dropout=dropout,
        )

        # Decoder
        self.decoder = MambaDecoder3D(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            d_state=d_state,
            dropout=dropout,
        )

        # Feature projection for contrastive loss
        self.img_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
        )

    def forward(
        self,
        img: torch.Tensor,
        text_ids: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Args:
            img: [B, C, D, H, W] input image
            text_ids: [B, L] text token indices
            return_features: whether to return features for contrastive loss
        Returns:
            seg_output: [B, out_channels, D, H, W]
            img_feat: [B, text_embed_dim] (if return_features)
            text_feat: [B, text_embed_dim] (if return_features)
        """
        # Encode image
        img_features = self.img_encoder(img)

        # Encode text
        text_features = self.text_encoder(text_ids)

        # Fuse at bottleneck
        bottleneck = img_features[-1]
        fused_bottleneck = self.fusion(bottleneck, text_features)

        # Replace bottleneck with fused features
        decoder_features = img_features[:-1] + [fused_bottleneck]

        # Decode
        seg_output = self.decoder(decoder_features)

        if return_features:
            # Global features for contrastive loss
            img_global = self.img_proj(bottleneck.mean(dim=1))
            text_global = self.text_encoder.get_global_feature(text_features)
            return seg_output, img_global, text_global

        return seg_output
```

**Step 4: Update models/__init__.py**

```python
# models/__init__.py
from .mamba_block import MambaBlock, MambaLayer
from .encoder_3d import PatchEmbed3D, MambaEncoder3D
from .text_encoder import TextMambaEncoder
from .fusion import MambaFusion
from .decoder_3d import MambaDecoder3D
from .textmamba3d import TextMamba3D
```

**Step 5: Run test to verify it passes**

Run: `cd TextMamba3D && python -m pytest tests/test_models.py::TestTextMamba3D -v`
Expected: PASS

**Step 6: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add complete TextMamba3D model"
```

---

## Task 8: Loss Functions

**Files:**
- Create: `TextMamba3D/losses/__init__.py`
- Create: `TextMamba3D/losses/dice_loss.py`
- Create: `TextMamba3D/losses/edge_loss.py`
- Create: `TextMamba3D/losses/contrastive_loss.py`
- Create: `TextMamba3D/tests/test_losses.py`

**Step 1: Write failing tests**

```python
# tests/test_losses.py
import torch
import pytest


class TestDiceLoss:
    def test_dice_loss_perfect_prediction(self):
        """Perfect prediction should give loss close to 0."""
        from losses.dice_loss import DiceLoss

        loss_fn = DiceLoss()
        pred = torch.ones(2, 4, 16, 16, 16) * 10  # High logits
        target = torch.ones(2, 16, 16, 16, dtype=torch.long)

        loss = loss_fn(pred, target)
        assert loss.item() < 0.1


class TestEdgeLoss:
    def test_edge_loss_output(self):
        """Test edge loss computation."""
        from losses.edge_loss import EdgeLoss

        loss_fn = EdgeLoss()
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))

        loss = loss_fn(pred, target)
        assert loss.item() >= 0


class TestContrastiveLoss:
    def test_contrastive_loss_identical_features(self):
        """Identical normalized features should give low loss."""
        from losses.contrastive_loss import ContrastiveLoss

        loss_fn = ContrastiveLoss(temperature=0.07)
        feat = torch.randn(4, 256)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        loss = loss_fn(feat, feat)
        assert loss.item() < 1.0
```

**Step 2: Run tests to verify they fail**

Run: `cd TextMamba3D && python -m pytest tests/test_losses.py -v`
Expected: FAIL

**Step 3: Write Dice Loss**

```python
# losses/dice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] logits
            target: [B, D, H, W] class indices
        Returns:
            Scalar loss
        """
        num_classes = pred.shape[1]

        # Softmax
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes)  # [B, D, H, W, C]
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        # Skip background if specified
        start_idx = 0 if self.include_background else 1

        dice_scores = []
        for i in range(start_idx, num_classes):
            pred_i = pred[:, i]
            target_i = target_onehot[:, i]

            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return 1 - torch.stack(dice_scores).mean()
```

**Step 4: Write Edge Loss**

```python
# losses/edge_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """Edge-enhanced loss for sharper boundaries."""

    def __init__(self, edge_weight: float = 2.0):
        super().__init__()
        self.edge_weight = edge_weight

        # Sobel kernels for 3D
        self.register_buffer('sobel_x', self._create_sobel_kernel(0))
        self.register_buffer('sobel_y', self._create_sobel_kernel(1))
        self.register_buffer('sobel_z', self._create_sobel_kernel(2))

    def _create_sobel_kernel(self, axis: int) -> torch.Tensor:
        """Create 3D Sobel kernel for given axis."""
        kernel = torch.zeros(1, 1, 3, 3, 3)

        if axis == 0:  # x (depth)
            kernel[0, 0, 0, :, :] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, 2, :, :] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        elif axis == 1:  # y (height)
            kernel[0, 0, :, 0, :] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, :, 2, :] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        else:  # z (width)
            kernel[0, 0, :, :, 0] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, :, :, 2] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

        return kernel / 16.0  # Normalize

    def get_edge_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Extract edge mask from segmentation target.

        Args:
            target: [B, D, H, W] class indices
        Returns:
            [B, 1, D, H, W] edge mask
        """
        # Convert to float and add channel dim
        target_float = target.float().unsqueeze(1)

        # Compute gradients
        gx = F.conv3d(target_float, self.sobel_x, padding=1)
        gy = F.conv3d(target_float, self.sobel_y, padding=1)
        gz = F.conv3d(target_float, self.sobel_z, padding=1)

        # Gradient magnitude
        edge = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)

        # Normalize to [0, 1]
        edge = edge / (edge.max() + 1e-8)

        return edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W] logits
            target: [B, D, H, W] class indices
        Returns:
            Scalar loss
        """
        # Get edge mask
        edge_mask = self.get_edge_mask(target)  # [B, 1, D, H, W]

        # Weight mask: higher weight on edges
        weight = 1 + self.edge_weight * edge_mask  # [B, 1, D, H, W]
        weight = weight.squeeze(1)  # [B, D, H, W]

        # Weighted cross entropy
        loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (loss * weight).mean()

        return weighted_loss
```

**Step 5: Write Contrastive Loss**

```python
# losses/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for text-image alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_feat: [B, D] image global features
            text_feat: [B, D] text global features
        Returns:
            Scalar loss
        """
        # Normalize
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        # Similarity matrix
        logits = img_feat @ text_feat.T / self.temperature  # [B, B]

        # Labels: diagonal is positive pairs
        B = img_feat.shape[0]
        labels = torch.arange(B, device=img_feat.device)

        # Bidirectional contrastive loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
```

**Step 6: Create losses/__init__.py**

```python
# losses/__init__.py
from .dice_loss import DiceLoss
from .edge_loss import EdgeLoss
from .contrastive_loss import ContrastiveLoss


class CombinedLoss(nn.Module):
    """Combined loss for TextMamba3D."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        edge_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.edge_weight = edge_weight
        self.contrastive_weight = contrastive_weight

        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss()
        self.contrastive_loss = ContrastiveLoss(temperature)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        img_feat: torch.Tensor = None,
        text_feat: torch.Tensor = None,
    ) -> dict:
        """
        Returns:
            dict with 'total', 'dice', 'ce', 'edge', 'contrastive' losses
        """
        import torch.nn.functional as F

        losses = {}

        # Dice loss
        losses['dice'] = self.dice_loss(pred, target)

        # Cross entropy
        losses['ce'] = F.cross_entropy(pred, target)

        # Edge loss
        losses['edge'] = self.edge_loss(pred, target)

        # Contrastive loss
        if img_feat is not None and text_feat is not None:
            losses['contrastive'] = self.contrastive_loss(img_feat, text_feat)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=pred.device)

        # Total
        losses['total'] = (
            self.dice_weight * losses['dice'] +
            self.ce_weight * losses['ce'] +
            self.edge_weight * losses['edge'] +
            self.contrastive_weight * losses['contrastive']
        )

        return losses
```

**Step 7: Run tests to verify they pass**

Run: `cd TextMamba3D && python -m pytest tests/test_losses.py -v`
Expected: PASS

**Step 8: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add loss functions (dice, edge, contrastive)"
```

---

## Task 9: Data Module - BraTS Dataset

**Files:**
- Create: `TextMamba3D/data/__init__.py`
- Create: `TextMamba3D/data/brats_dataset.py`
- Create: `TextMamba3D/data/text_generator.py`
- Create: `TextMamba3D/data/transforms.py`
- Create: `TextMamba3D/tests/test_dataset.py`

**Step 1: Write failing test**

```python
# tests/test_dataset.py
import torch
import pytest


class TestTextGenerator:
    def test_generate_diagnosis_text(self):
        """Test diagnosis text generation from mask."""
        from data.text_generator import DiagnosisTextGenerator

        generator = DiagnosisTextGenerator()

        # Create fake mask
        mask = torch.zeros(96, 96, 96, dtype=torch.long)
        mask[40:60, 40:60, 40:60] = 1  # Necrotic
        mask[35:65, 35:65, 35:65] = 2  # Edema (surrounding)
        mask[45:55, 45:55, 45:55] = 4  # Enhancing (core)

        text = generator.generate(mask)

        assert isinstance(text, str)
        assert len(text) > 10
```

**Step 2: Run test to verify it fails**

Run: `cd TextMamba3D && python -m pytest tests/test_dataset.py -v`
Expected: FAIL

**Step 3: Write Text Generator**

```python
# data/text_generator.py
import torch
import numpy as np
from typing import Dict, Tuple


class DiagnosisTextGenerator:
    """Generate diagnosis text from segmentation mask."""

    # Brain region mapping (simplified)
    BRAIN_REGIONS = {
        (0.0, 0.5): {"x": "右侧", "y": "后", "z": "下"},
        (0.5, 1.0): {"x": "左侧", "y": "前", "z": "上"},
    }

    LOBE_MAP = {
        "前上": "额叶",
        "前下": "额叶",
        "后上": "顶叶",
        "后下": "枕叶",
    }

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing

    def _get_region(self, centroid: np.ndarray, shape: tuple) -> str:
        """Map centroid to brain region."""
        norm = centroid / np.array(shape)

        x_side = "左侧" if norm[0] > 0.5 else "右侧"
        y_pos = "前" if norm[1] > 0.5 else "后"
        z_pos = "上" if norm[2] > 0.5 else "下"

        lobe = self.LOBE_MAP.get(y_pos + z_pos, "脑实质")

        return f"{x_side}{lobe}"

    def _compute_volumes(self, mask: torch.Tensor) -> Dict[str, float]:
        """Compute volumes of each region in cm³."""
        voxel_vol = np.prod(self.voxel_spacing) / 1000  # mm³ to cm³

        volumes = {
            "necrotic": (mask == 1).sum().item() * voxel_vol,
            "edema": (mask == 2).sum().item() * voxel_vol,
            "enhancing": (mask == 4).sum().item() * voxel_vol,
        }
        volumes["total"] = sum(volumes.values())

        return volumes

    def _analyze_boundary(self, mask: torch.Tensor) -> str:
        """Analyze boundary characteristics."""
        # Simple gradient-based analysis
        tumor_mask = (mask > 0).float()

        # Compute gradient magnitude
        grad_x = torch.diff(tumor_mask, dim=0).abs()
        grad_y = torch.diff(tumor_mask, dim=1).abs()
        grad_z = torch.diff(tumor_mask, dim=2).abs()

        edge_sum = grad_x.sum() + grad_y.sum() + grad_z.sum()
        tumor_surface = edge_sum.item()
        tumor_volume = tumor_mask.sum().item()

        if tumor_volume == 0:
            return "未见明显病灶"

        # Surface to volume ratio indicates boundary complexity
        ratio = tumor_surface / (tumor_volume ** (2/3) + 1e-8)

        if ratio > 10:
            return "边界不规则，呈浸润性生长"
        elif ratio > 6:
            return "边界欠清，与周围组织分界不明确"
        else:
            return "边界尚清"

    def _get_grade(self, volumes: Dict[str, float]) -> str:
        """Estimate tumor grade based on volumes."""
        enhancing_ratio = volumes["enhancing"] / (volumes["total"] + 1e-8)

        if enhancing_ratio > 0.3:
            return "高级别胶质瘤(HGG)"
        else:
            return "考虑低级别胶质瘤(LGG)可能"

    def generate(self, mask: torch.Tensor) -> str:
        """
        Generate diagnosis text from segmentation mask.

        Args:
            mask: [D, H, W] segmentation mask
                0: background
                1: necrotic/non-enhancing
                2: edema
                4: enhancing tumor
        Returns:
            Diagnosis text string
        """
        if mask.sum() == 0:
            return "MRI平扫未见明显异常信号。"

        # Find tumor centroid
        tumor_coords = torch.nonzero(mask > 0).float()
        centroid = tumor_coords.mean(dim=0).numpy()

        # Get region
        region = self._get_region(centroid, mask.shape)

        # Compute volumes
        volumes = self._compute_volumes(mask)

        # Analyze boundary
        boundary_desc = self._analyze_boundary(mask)

        # Get grade
        grade = self._get_grade(volumes)

        # Generate text
        text = (
            f"MRI示{region}占位性病变，"
            f"大小约{volumes['total']:.1f}cm³，"
            f"其中强化区域约{volumes['enhancing']:.1f}cm³，"
            f"周围水肿区域约{volumes['edema']:.1f}cm³。"
            f"{boundary_desc}，"
            f"{grade}。"
        )

        return text

    def generate_batch(self, masks: torch.Tensor) -> list:
        """Generate texts for a batch of masks."""
        return [self.generate(m) for m in masks]
```

**Step 4: Write BraTS Dataset**

```python
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
```

**Step 5: Write Transforms**

```python
# data/transforms.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class RandomCrop3D:
    """Random 3D crop."""

    def __init__(self, size: Tuple[int, int, int]):
        self.size = size

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, D, H, W = image.shape
        td, th, tw = self.size

        d = np.random.randint(0, max(1, D - td + 1))
        h = np.random.randint(0, max(1, H - th + 1))
        w = np.random.randint(0, max(1, W - tw + 1))

        image = image[:, d:d+td, h:h+th, w:w+tw]
        mask = mask[d:d+td, h:h+th, w:w+tw]

        return image, mask


class RandomFlip3D:
    """Random 3D flip."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for axis in [1, 2, 3]:  # D, H, W
            if np.random.random() < self.prob:
                image = torch.flip(image, [axis])
                mask = torch.flip(mask, [axis - 1])

        return image, mask


class Compose:
    """Compose transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_train_transforms(patch_size: Tuple[int, int, int]):
    return Compose([
        RandomCrop3D(patch_size),
        RandomFlip3D(prob=0.5),
    ])


def get_val_transforms(patch_size: Tuple[int, int, int]):
    return Compose([
        RandomCrop3D(patch_size),  # Center crop would be better for val
    ])
```

**Step 6: Create data/__init__.py**

```python
# data/__init__.py
from .brats_dataset import BraTSDataset
from .text_generator import DiagnosisTextGenerator
from .transforms import (
    RandomCrop3D,
    RandomFlip3D,
    Compose,
    get_train_transforms,
    get_val_transforms,
)
```

**Step 7: Run tests to verify they pass**

Run: `cd TextMamba3D && python -m pytest tests/test_dataset.py -v`
Expected: PASS

**Step 8: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add BraTS dataset and text generator"
```

---

## Task 10: Training Script

**Files:**
- Create: `TextMamba3D/train.py`
- Create: `TextMamba3D/utils/__init__.py`
- Create: `TextMamba3D/utils/metrics.py`

**Step 1: Write metrics utilities**

```python
# utils/metrics.py
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """Compute Dice score per class."""
    pred_argmax = pred.argmax(dim=1)

    scores = {}
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred_argmax == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection) / (union + 1e-8)
        scores[f'dice_class_{c}'] = dice.item()

    scores['dice_mean'] = np.mean(list(scores.values()))
    return scores


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance."""
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan

    # Distance transform
    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)

    # Surface distances
    pred_surface = pred_dist[target > 0]
    target_surface = target_dist[pred > 0]

    all_distances = np.concatenate([pred_surface, target_surface])

    return np.percentile(all_distances, 95)
```

**Step 2: Create utils/__init__.py**

```python
# utils/__init__.py
from .metrics import dice_score, hausdorff_distance_95
```

**Step 3: Write training script**

```python
# train.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import TextMamba3D
from losses import CombinedLoss
from data import BraTSDataset, get_train_transforms, get_val_transforms
from utils.metrics import dice_score


def parse_args():
    parser = argparse.ArgumentParser(description='Train TextMamba3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)

        optimizer.zero_grad()

        # Forward
        pred, img_feat, text_feat = model(image, text_ids, return_features=True)

        # Loss
        losses = criterion(pred, mask, img_feat, text_feat)
        loss = losses['total']

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_dice = []

    for batch in tqdm(loader, desc='Validating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)

        pred, img_feat, text_feat = model(image, text_ids, return_features=True)
        losses = criterion(pred, mask, img_feat, text_feat)

        total_loss += losses['total'].item()

        # Dice score
        dice = dice_score(pred, mask, num_classes=4)
        all_dice.append(dice['dice_mean'])

    return total_loss / len(loader), np.mean(all_dice)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_transform = get_train_transforms(tuple(config['data']['patch_size']))
    val_transform = get_val_transforms(tuple(config['data']['patch_size']))

    train_dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        transform=train_transform,
    )
    val_dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # Model
    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        text_embed_dim=config['model']['text_embed_dim'],
    ).to(device)

    # Loss
    criterion = CombinedLoss(
        dice_weight=config['loss']['dice_weight'],
        ce_weight=config['loss']['ce_weight'],
        edge_weight=config['loss']['edge_weight'],
        contrastive_weight=config['loss']['contrastive_weight'],
        temperature=config['loss']['temperature'],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )

    # Tensorboard
    writer = SummaryWriter('logs')

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    best_dice = 0
    for epoch in range(start_epoch, config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}')

        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
            }, 'checkpoints/best.pth')

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'checkpoints/last.pth')

    writer.close()


if __name__ == '__main__':
    import numpy as np
    main()
```

**Step 4: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add training script and metrics"
```

---

## Task 11: Evaluation Script

**Files:**
- Create: `TextMamba3D/evaluate.py`

**Step 1: Write evaluation script**

```python
# evaluate.py
import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import TextMamba3D
from data import BraTSDataset, get_val_transforms
from utils.metrics import dice_score, hausdorff_distance_95


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TextMamba3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_pred', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, save_pred=False, save_dir='predictions'):
    model.eval()

    all_dice = {f'class_{i}': [] for i in range(1, 4)}
    all_hd95 = {f'class_{i}': [] for i in range(1, 4)}

    if save_pred:
        os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(loader, desc='Evaluating'):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        text_ids = batch['text_ids'].to(device)
        case_name = batch['case_name'][0]

        pred = model(image, text_ids)
        pred_argmax = pred.argmax(dim=1)

        # Dice scores
        dice = dice_score(pred, mask, num_classes=4)
        for c in range(1, 4):
            all_dice[f'class_{c}'].append(dice[f'dice_class_{c}'])

        # HD95
        pred_np = pred_argmax[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()

        for c in range(1, 4):
            pred_c = (pred_np == c).astype(np.uint8)
            mask_c = (mask_np == c).astype(np.uint8)
            hd95 = hausdorff_distance_95(pred_c, mask_c)
            if not np.isnan(hd95):
                all_hd95[f'class_{c}'].append(hd95)

        # Save prediction
        if save_pred:
            np.save(os.path.join(save_dir, f'{case_name}_pred.npy'), pred_np)

    # Summary
    print('\n=== Evaluation Results ===')
    print('\nDice Scores:')
    for c in range(1, 4):
        scores = all_dice[f'class_{c}']
        print(f'  Class {c}: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

    mean_dice = np.mean([np.mean(v) for v in all_dice.values()])
    print(f'  Mean: {mean_dice:.4f}')

    print('\nHD95:')
    for c in range(1, 4):
        scores = all_hd95[f'class_{c}']
        if scores:
            print(f'  Class {c}: {np.mean(scores):.2f} ± {np.std(scores):.2f}')

    return {'dice': mean_dice}


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = get_val_transforms(tuple(config['data']['patch_size']))
    dataset = BraTSDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    model = TextMamba3D(
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        text_embed_dim=config['model']['text_embed_dim'],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded checkpoint from {args.checkpoint}')

    # Evaluate
    results = evaluate(model, loader, device, save_pred=args.save_pred)


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
cd TextMamba3D && git add . && git commit -m "feat: add evaluation script"
```

---

## Task 12: Final Project Setup

**Files:**
- Create: `TextMamba3D/README.md`

**Step 1: Write README**

```markdown
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
```

**Step 2: Final commit**

```bash
cd TextMamba3D && git add . && git commit -m "docs: add README"
```

---

## Verification

After completing all tasks, verify the implementation:

1. **Run all tests:**
   ```bash
   cd TextMamba3D && python -m pytest tests/ -v
   ```

2. **Check model can forward:**
   ```bash
   cd TextMamba3D && python -c "
   import torch
   from models import TextMamba3D
   model = TextMamba3D(img_size=(96,96,96), in_channels=4, out_channels=4)
   img = torch.randn(1, 4, 96, 96, 96)
   text = torch.randint(0, 30000, (1, 64))
   out = model(img, text)
   print(f'Output shape: {out.shape}')
   "
   ```

3. **Verify loss computation:**
   ```bash
   cd TextMamba3D && python -c "
   import torch
   from losses import CombinedLoss
   loss_fn = CombinedLoss()
   pred = torch.randn(2, 4, 32, 32, 32)
   target = torch.randint(0, 4, (2, 32, 32, 32))
   img_feat = torch.randn(2, 256)
   text_feat = torch.randn(2, 256)
   losses = loss_fn(pred, target, img_feat, text_feat)
   print(f'Total loss: {losses[\"total\"].item():.4f}')
   "
   ```
