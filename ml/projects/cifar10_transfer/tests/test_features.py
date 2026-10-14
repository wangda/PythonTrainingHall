"""features.py 的单元测试：transforms 的确定性/随机性、ImageNet 归一化常数、loader 划分。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.py 的 build_transforms / make_loaders 后全部变绿。
骨架状态下以 NotImplementedError 失败——失败信息就是你要实现的东西。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="本项目需要 torch/torchvision")
from PIL import Image
from torchvision import transforms

from features import IMAGENET_MEAN, IMAGENET_STD, build_transforms

# 与 config.yaml 对应的最小测试配置（不读真实配置文件，避免测试被配置改动影响）
CONFIG = {
    "data": {"raw_dir": "data/raw", "val_size": 5000, "random_state": 42},
    "features": {"image_size": 224, "augment": True},
    "train": {"batch_size": 64, "num_workers": 0},
}


@pytest.fixture
def constant_image():
    """纯色测试图：所有像素 (128,128,128)——Resize/Crop 都改变不了它，
    任何输出差异只能来自 Normalize 的常数。"""
    return Image.new("RGB", (300, 300), color=(128, 128, 128))


# ---------------------------------------------------------------- build_transforms 评估版


def test_评估版_确定性(constant_image):
    tf = build_transforms(CONFIG, train=False)
    out1 = tf(constant_image)
    out2 = tf(constant_image)
    assert torch.equal(out1, out2), "评估版 transform 必须是全确定性的（项目文档 §3）"


def test_评估版_输出形状与值域(constant_image):
    out = build_transforms(CONFIG, train=False)(constant_image)
    assert out.shape == (3, 224, 224)          # NCHW 的 CHW（前置文档 §1.3）


def test_评估版_用的是ImageNet常数(constant_image):
    """纯色 128 图归一化后的精确期望值：(128/255 − mean) / std，逐通道验证。
    换成 CIFAR 自己的均值方差会立刻变红（前置文档 §6.5 的铁律）。"""
    out = build_transforms(CONFIG, train=False)(constant_image)
    x = 128 / 255
    expected = [(x - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    for c in range(3):
        assert out[c].mean().item() == pytest.approx(expected[c], abs=1e-3)


def test_ImageNet常数_就是文档里那组数():
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- build_transforms 训练版


def test_训练版_包含随机增强(constant_image):
    tf = build_transforms(CONFIG, train=True)
    types = [type(t) for t in tf.transforms]
    assert transforms.RandomCrop in types, "训练版必须有 RandomCrop（位置扰动）"
    assert transforms.RandomHorizontalFlip in types, "训练版必须有水平翻转（方向扰动）"


def test_训练版_与评估版不同(constant_image):
    """两条流水线必须是两套东西（项目文档 §3 的纪律）。"""
    train_tf = build_transforms(CONFIG, train=True)
    eval_tf = build_transforms(CONFIG, train=False)
    assert len(train_tf.transforms) != len(eval_tf.transforms)


def test_训练版_增强关闭时退化为评估版(constant_image):
    config = {**CONFIG, "features": {**CONFIG["features"], "augment": False}}
    tf = build_transforms(config, train=True)
    out1, out2 = tf(constant_image), tf(constant_image)
    assert torch.equal(out1, out2), "augment=False 时训练版也必须确定性"


def test_训练版_尺寸跟着config走():
    """lenet 路线 image_size=32：transforms 输出必须跟着变（项目文档 §7 的坑）。"""
    config32 = {**CONFIG, "features": {**CONFIG["features"], "image_size": 32}}
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    out = build_transforms(config32, train=False)(img)
    assert out.shape == (3, 32, 32)


# ---------------------------------------------------------------- make_loaders


def test_加载器_划分大小与可复现():
    """需要 CIFAR-10 已下载（先跑 python data/download.py），否则跳过。"""
    import os
    from pathlib import Path

    if not (Path("data/raw") / "cifar-10-batches-py").exists():
        pytest.skip("CIFAR-10 未缓存：先运行 python data/download.py")

    from features import make_loaders

    train1, val1, test1 = make_loaders(CONFIG)
    assert len(train1.dataset) == 45000 and len(val1.dataset) == 5000
    assert len(test1.dataset) == 10000

    train2, val2, _ = make_loaders(CONFIG)
    assert val1.dataset.indices == val2.dataset.indices, \
        "同种子下验证集划分必须可复现（random_split 要传固定种子的 Generator）"


def test_加载器_训练与验证挂不同transform():
    from pathlib import Path

    if not (Path("data/raw") / "cifar-10-batches-py").exists():
        pytest.skip("CIFAR-10 未缓存")

    from features import make_loaders

    train_loader, val_loader, _ = make_loaders(CONFIG)
    train_tf = train_loader.dataset.transform
    val_tf = val_loader.dataset.transform
    types_train = [type(t) for t in train_tf.transforms]
    types_val = [type(t) for t in val_tf.transforms]
    assert transforms.RandomHorizontalFlip in types_train
    assert transforms.RandomHorizontalFlip not in types_val, \
        "验证集绝不许有随机增强（项目文档 §3）"
