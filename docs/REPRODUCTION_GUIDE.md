# TextMamba3D 完整复现指南

> 文本引导的3D医学图像分割 - 基于统一Mamba架构

---

## 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [数据集准备](#数据集准备)
4. [项目结构](#项目结构)
5. [核心代码](#核心代码)
6. [训练与评估](#训练与评估)
7. [设计原理](#设计原理)
8. [常见问题](#常见问题)

---

## 项目概述

### 核心创新

1. **全Mamba统一架构**：图像编码器、文本编码器、融合模块、解码器全部使用Mamba
2. **文本引导分割**：通过诊断文本信息加强3D病灶分割的边缘质量
3. **对比学习对齐**：使用对比损失实现文本-图像特征对齐
4. **边缘增强损失**：专门的边缘损失函数提升边界清晰度

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      输入层                              │
├─────────────────┬───────────────────────────────────────┤
│   3D MRI 影像    │         构造的诊断文本                 │
│   [B,4,D,H,W]   │         (从mask生成)                   │
└────────┬────────┴──────────────┬────────────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────────────┐
│  3D Mamba       │    │  Text Mamba Encoder     │
│  Encoder        │    │  (Token→特征序列)        │
│  (分层特征)      │    └───────────┬─────────────┘
└────────┬────────┘                │
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────────────────┐
│           Mamba Fusion Module (Bottleneck)              │
│   [text_tokens, img_tokens] → Mamba → 融合特征           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              3D Mamba Decoder (Skip Connections)        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              分割输出 [B, 4, D, H, W]                    │
│              (背景, 坏死, 水肿, 强化)                     │
└─────────────────────────────────────────────────────────┘
```

### 损失函数

```
L_total = λ₁·L_dice + λ₂·L_ce + λ₃·L_edge + λ₄·L_contrastive

其中：
- L_dice: Dice损失，整体分割质量
- L_ce: 交叉熵损失，像素级分类
- L_edge: 边缘增强损失（3D Sobel + 加权CE）
- L_contrastive: 对比损失，文本-图像对齐
```

---

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- GPU显存 >= 8GB (推荐12GB+)

### 安装步骤

```bash
# 1. 克隆项目（或复制整个目录）
git clone <your-repo-url> TextMamba3D
cd TextMamba3D

# 2. 创建虚拟环境
conda create -n textmamba3d python=3.10
conda activate textmamba3d

# 3. 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装mamba-ssm（需要CUDA）
pip install mamba-ssm

# 5. 安装其他依赖
pip install -r requirements.txt
```

### requirements.txt

```
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

### 验证安装

```bash
# 运行测试
pytest tests/ -v

# 预期输出：14 passed
```

---

## 数据集准备

### BraTS 2021 数据集下载

BraTS (Brain Tumor Segmentation) 是脑肿瘤分割的标准数据集。

**下载方式1：Synapse官方渠道（推荐）**

1. 访问 https://www.synapse.org/#!Synapse:syn25829067
2. 注册Synapse账号并登录
3. 加入BraTS 2021挑战赛
4. 下载 `BraTS2021_Training_Data.tar`

**下载方式2：Kaggle镜像**

1. 访问 https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
2. 登录Kaggle账号
3. 下载数据集

### 数据组织

下载后，按以下结构组织数据：

```
TextMamba3D/
└── data/
    └── BraTS2021/
        ├── train/
        │   ├── BraTS2021_00000/
        │   │   ├── BraTS2021_00000_t1.nii.gz
        │   │   ├── BraTS2021_00000_t1ce.nii.gz
        │   │   ├── BraTS2021_00000_t2.nii.gz
        │   │   ├── BraTS2021_00000_flair.nii.gz
        │   │   └── BraTS2021_00000_seg.nii.gz
        │   ├── BraTS2021_00001/
        │   └── ...
        ├── val/
        │   └── ...
        └── test/
            └── ...
```

### 数据预处理说明

数据加载时自动执行以下预处理：

1. **模态堆叠**: 4个模态(T1, T1ce, T2, FLAIR) → [4, D, H, W]
2. **Z-score归一化**: 每个模态独立归一化 (仅对非零体素)
3. **随机裁剪**: 训练时裁剪为 96×96×96 的patch
4. **文本生成**: 从分割mask自动生成诊断文本

### 分割标签说明

```
0: 背景
1: 坏死/非强化肿瘤核心 (NCR/NET)
2: 水肿区域 (ED)
4: 强化肿瘤 (ET)
```

---

## 项目结构

```
TextMamba3D/
├── configs/
│   └── default.yaml          # 配置文件
├── models/
│   ├── __init__.py           # 模块导出
│   ├── mamba_block.py        # Mamba基础模块
│   ├── encoder_3d.py         # 3D图像编码器
│   ├── text_encoder.py       # 文本Mamba编码器
│   ├── fusion.py             # Mamba融合模块
│   ├── decoder_3d.py         # 3D解码器
│   └── textmamba3d.py        # 完整模型
├── losses/
│   ├── __init__.py           # CombinedLoss
│   ├── dice_loss.py          # Dice损失
│   ├── edge_loss.py          # 边缘增强损失
│   └── contrastive_loss.py   # 对比损失
├── data/
│   ├── __init__.py           # 数据模块导出
│   ├── brats_dataset.py      # BraTS数据集
│   ├── text_generator.py     # 诊断文本生成器
│   └── transforms.py         # 3D数据增强
├── utils/
│   ├── __init__.py
│   └── metrics.py            # 评估指标
├── tests/
│   └── test_models.py        # 单元测试
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── requirements.txt          # 依赖
└── README.md
```

---

## 核心代码

### 1. Mamba基础模块 (models/mamba_block.py)

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

### 2. 3D图像编码器 (models/encoder_3d.py)

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

### 3. 文本Mamba编码器 (models/text_encoder.py)

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

### 4. Mamba融合模块 (models/fusion.py)

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

### 5. 3D解码器 (models/decoder_3d.py)

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
                skip_proj = nn.Linear(dim // 2, dim // 2)
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

### 6. 完整模型 (models/textmamba3d.py)

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

### 7. 损失函数 (losses/)

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
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

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
        self.register_buffer('sobel_x', self._create_sobel_kernel(0))
        self.register_buffer('sobel_y', self._create_sobel_kernel(1))
        self.register_buffer('sobel_z', self._create_sobel_kernel(2))

    def _create_sobel_kernel(self, axis: int) -> torch.Tensor:
        kernel = torch.zeros(1, 1, 3, 3, 3)
        if axis == 0:
            kernel[0, 0, 0, :, :] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, 2, :, :] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        elif axis == 1:
            kernel[0, 0, :, 0, :] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, :, 2, :] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        else:
            kernel[0, 0, :, :, 0] = torch.tensor([[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]])
            kernel[0, 0, :, :, 2] = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        return kernel / 16.0

    def get_edge_mask(self, target: torch.Tensor) -> torch.Tensor:
        target_float = target.float().unsqueeze(1)
        gx = F.conv3d(target_float, self.sobel_x, padding=1)
        gy = F.conv3d(target_float, self.sobel_y, padding=1)
        gz = F.conv3d(target_float, self.sobel_z, padding=1)
        edge = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        edge = edge / (edge.max() + 1e-8)
        return edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge_mask = self.get_edge_mask(target)
        weight = 1 + self.edge_weight * edge_mask
        weight = weight.squeeze(1)
        loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (loss * weight).mean()
        return weighted_loss
```

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

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        logits = img_feat @ text_feat.T / self.temperature
        B = img_feat.shape[0]
        labels = torch.arange(B, device=img_feat.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
```

```python
# losses/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, pred, target, img_feat=None, text_feat=None) -> dict:
        losses = {}
        losses['dice'] = self.dice_loss(pred, target)
        losses['ce'] = F.cross_entropy(pred, target)
        losses['edge'] = self.edge_loss(pred, target)
        if img_feat is not None and text_feat is not None:
            losses['contrastive'] = self.contrastive_loss(img_feat, text_feat)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=pred.device)
        losses['total'] = (
            self.dice_weight * losses['dice'] +
            self.ce_weight * losses['ce'] +
            self.edge_weight * losses['edge'] +
            self.contrastive_weight * losses['contrastive']
        )
        return losses
```

### 8. 数据模块 (data/)

```python
# data/text_generator.py
import torch
import numpy as np
from typing import Dict, Tuple


class DiagnosisTextGenerator:
    """Generate diagnosis text from segmentation mask."""

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
        voxel_vol = np.prod(self.voxel_spacing) / 1000
        volumes = {
            "necrotic": (mask == 1).sum().item() * voxel_vol,
            "edema": (mask == 2).sum().item() * voxel_vol,
            "enhancing": (mask == 4).sum().item() * voxel_vol,
        }
        volumes["total"] = sum(volumes.values())
        return volumes

    def _analyze_boundary(self, mask: torch.Tensor) -> str:
        """Analyze boundary characteristics."""
        tumor_mask = (mask > 0).float()
        grad_x = torch.diff(tumor_mask, dim=0).abs()
        grad_y = torch.diff(tumor_mask, dim=1).abs()
        grad_z = torch.diff(tumor_mask, dim=2).abs()
        edge_sum = grad_x.sum() + grad_y.sum() + grad_z.sum()
        tumor_surface = edge_sum.item()
        tumor_volume = tumor_mask.sum().item()

        if tumor_volume == 0:
            return "未见明显病灶"

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
        """Generate diagnosis text from segmentation mask."""
        if mask.sum() == 0:
            return "MRI平扫未见明显异常信号。"

        tumor_coords = torch.nonzero(mask > 0).float()
        centroid = tumor_coords.mean(dim=0).numpy()
        region = self._get_region(centroid, mask.shape)
        volumes = self._compute_volumes(mask)
        boundary_desc = self._analyze_boundary(mask)
        grade = self._get_grade(volumes)

        text = (
            f"MRI示{region}占位性病变，"
            f"大小约{volumes['total']:.1f}cm³，"
            f"其中强化区域约{volumes['enhancing']:.1f}cm³，"
            f"周围水肿区域约{volumes['edema']:.1f}cm³。"
            f"{boundary_desc}，"
            f"{grade}。"
        )
        return text
```

```python
# data/brats_dataset.py
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import Optional, Callable
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
        self.cases = self._find_cases()

    def _find_cases(self) -> list:
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
        image = np.stack(images, axis=0).astype(np.float32)

        # Load segmentation
        seg_path = os.path.join(case_dir, f'{case_name}_seg.nii.gz')
        mask = self._load_nifti(seg_path).astype(np.int64)

        # Normalize
        for i in range(image.shape[0]):
            img_i = image[i]
            nonzero = img_i[img_i > 0]
            if len(nonzero) > 0:
                mean, std = nonzero.mean(), nonzero.std()
                image[i] = (img_i - mean) / (std + 1e-8)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        # Generate text
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

```python
# data/transforms.py
import torch
import numpy as np
from typing import Tuple


class RandomCrop3D:
    """Random 3D crop."""

    def __init__(self, size: Tuple[int, int, int]):
        self.size = size

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
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

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        for axis in [1, 2, 3]:
            if np.random.random() < self.prob:
                image = torch.flip(image, [axis])
                mask = torch.flip(mask, [axis - 1])
        return image, mask


class Compose:
    """Compose transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
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
        RandomCrop3D(patch_size),
    ])
```

### 9. 配置文件 (configs/default.yaml)

```yaml
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

---

## 训练与评估

### 训练

```bash
# 基础训练
python train.py --config configs/default.yaml

# 从checkpoint恢复
python train.py --config configs/default.yaml --resume checkpoints/last.pth
```

### 评估

```bash
# 评估最佳模型
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth

# 保存预测结果
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth --save_pred
```

### TensorBoard监控

```bash
tensorboard --logdir logs
```

### 评估指标

| 指标 | 说明 |
|-----|------|
| Dice Score | 分割重叠度 (0-1, 越高越好) |
| HD95 | 95%百分位Hausdorff距离 (mm, 越低越好) |
| ASSD | 平均对称表面距离 (mm, 越低越好) |

---

## 设计原理

### 为什么选择Mamba？

1. **线性复杂度**: O(n) vs Transformer的O(n²)
2. **长序列建模**: 3D医学图像token数量巨大 (96³/4³ = 13824)
3. **因果建模**: 天然适合序列到序列的融合

### 文本引导的原理

```
文本: "MRI示左侧额叶占位性病变..."
      ↓ 编码
文本特征: [T₁, T₂, ..., Tₘ]

图像特征: [I₁, I₂, ..., Iₙ]
      ↓
拼接: [T₁...Tₘ, I₁...Iₙ]
      ↓ Mamba (因果扫描)
融合特征: 文本信息传递到图像
```

### 边缘增强的原理

1. **3D Sobel算子**: 检测3个方向的梯度
2. **边缘掩码**: 梯度大的区域为边缘
3. **加权损失**: 边缘区域的损失权重更高

---

## 常见问题

### Q: mamba-ssm安装失败？

```bash
# 确保CUDA版本正确
nvcc --version

# 从源码安装
pip install mamba-ssm --no-build-isolation
```

### Q: 显存不足？

1. 减小batch_size: 2 → 1
2. 减小patch_size: 96 → 64
3. 使用梯度累积

### Q: 训练loss不下降？

1. 检查数据路径是否正确
2. 降低学习率: 1e-4 → 5e-5
3. 增加warmup epochs

### Q: 如何使用自己的数据？

1. 继承`BraTSDataset`类
2. 修改`MODALITIES`和数据加载逻辑
3. 调整`out_channels`匹配类别数

---

## 参考文献

1. U-Mamba: https://github.com/bowang-lab/U-Mamba
2. SegMamba: https://github.com/ge-xing/SegMamba
3. Mamba: https://github.com/state-spaces/mamba
4. BraTS Challenge: https://www.synapse.org/brats

---

**生成日期**: 2026-01-22

**项目版本**: 1.0.0
