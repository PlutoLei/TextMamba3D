#!/usr/bin/env python3
"""
验证TextMamba3D安装是否正确

运行: python scripts/verify_installation.py
"""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def main():
    print("="*60)
    print("TextMamba3D 安装验证")
    print("="*60)
    print()

    # Check required packages
    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("nibabel", "nibabel"),
        ("scipy", "scipy"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("einops", "einops"),
        ("tensorboard", "tensorboard"),
    ]

    all_ok = True
    print("1. 检查必需依赖...")
    for module, package in packages:
        ok, error = check_import(module)
        status = "✓" if ok else "✗"
        print(f"   {status} {package}")
        if not ok:
            all_ok = False
            print(f"      安装: pip install {package}")

    # Check mamba-ssm
    print("\n2. 检查Mamba依赖...")
    ok, error = check_import("mamba_ssm")
    if ok:
        print("   ✓ mamba-ssm (GPU加速)")
    else:
        print("   ○ mamba-ssm (未安装，将使用fallback)")
        print("      可选安装: pip install mamba-ssm")

    # Check CUDA
    print("\n3. 检查GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"      显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ○ CUDA不可用，将使用CPU (训练会很慢)")
    except:
        print("   ✗ 无法检测GPU")

    # Check TextMamba3D models
    print("\n4. 检查TextMamba3D模块...")
    try:
        from models import TextMamba3D
        print("   ✓ TextMamba3D模型")
    except ImportError as e:
        print(f"   ✗ TextMamba3D模型: {e}")
        all_ok = False

    try:
        from losses import CombinedLoss
        print("   ✓ 损失函数")
    except ImportError as e:
        print(f"   ✗ 损失函数: {e}")
        all_ok = False

    try:
        from data import BraTSDataset
        print("   ✓ 数据集")
    except ImportError as e:
        print(f"   ✗ 数据集: {e}")
        all_ok = False

    # Test forward pass
    print("\n5. 测试模型前向传播...")
    try:
        import torch
        from models import TextMamba3D

        model = TextMamba3D(
            img_size=(32, 32, 32),  # 小尺寸测试
            in_channels=4,
            out_channels=4,
            embed_dim=48,
            depths=[1, 1],
            text_embed_dim=64,
            text_depth=1,
        )

        img = torch.randn(1, 4, 32, 32, 32)
        text_ids = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            out = model(img, text_ids)

        print(f"   ✓ 前向传播成功")
        print(f"      输入: image {tuple(img.shape)}, text {tuple(text_ids.shape)}")
        print(f"      输出: {tuple(out.shape)}")
    except Exception as e:
        print(f"   ✗ 前向传播失败: {e}")
        all_ok = False

    # Check data directory
    print("\n6. 检查数据目录...")
    import os
    data_dir = "data/BraTS2021"
    if os.path.exists(data_dir):
        train_dir = os.path.join(data_dir, "train")
        if os.path.exists(train_dir):
            cases = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"   ✓ 数据集已准备: {len(cases)} training cases")
        else:
            print(f"   ○ 数据目录存在但train子目录为空")
    else:
        print(f"   ○ 数据目录不存在: {data_dir}")
        print("      运行: python scripts/prepare_brats.py --help")

    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✓ 所有检查通过！可以开始训练。")
        print()
        print("下一步:")
        print("  1. 准备数据: python scripts/prepare_brats.py --input <数据路径>")
        print("  2. 开始训练: python train.py --config configs/default.yaml")
    else:
        print("✗ 部分检查未通过，请修复上述问题。")
    print("="*60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
