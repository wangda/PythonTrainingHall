"""features.py 的单元测试：展平形状、归一化值域与无状态性、划分可复现。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.py 的四个函数后全部变绿。
骨架状态下应全部以 NotImplementedError 失败（而不是导入错误）——
失败信息就是你要实现的东西。
"""

import numpy as np
import pytest

from features import build_features, flatten_images, normalize_pixels, split_data

# 与 config.yaml 对应的最小测试配置（不读真实配置文件，避免测试被配置改动影响）
CONFIG = {
    "data": {"test_size": 0.25, "random_state": 42},
    "features": {"normalize": True},
}
# 两个数据集的元信息（与 data/download.py 的约定一致）
META_DIGITS = {"dataset": "digits", "image_shape": (8, 8), "pixel_max": 16.0, "classes": list(range(10))}
META_MNIST = {"dataset": "mnist", "image_shape": (28, 28), "pixel_max": 255.0, "classes": list(range(10))}


@pytest.fixture
def digits_like():
    """200 张模拟 digits 图像：(200, 8, 8)，像素 0~16；标签在 0~9 循环。"""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 17, (200, 8, 8)).astype(float)
    y = np.arange(200) % 10
    return X, y


@pytest.fixture
def mnist_like():
    """200 张模拟 mnist 图像：已展平 (200, 784)，像素 0~255。"""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (200, 784)).astype(float)
    y = np.arange(200) % 10
    return X, y


# ---------- flatten_images ----------

def test_展平_三维图像变二维(digits_like):
    X, _ = digits_like
    out = flatten_images(X, META_DIGITS)
    assert out.shape == (200, 64)                    # 8×8 = 64 个像素 = 64 个特征
    assert np.array_equal(out[0], X[0].reshape(-1))  # 展平不重排像素值


def test_展平_已展平的mnist原样通过(mnist_like):
    X, _ = mnist_like
    out = flatten_images(X, META_MNIST)
    assert out.shape == (200, 784)
    assert np.array_equal(out, X)


def test_展平_尺寸不符报错(digits_like):
    X, _ = digits_like
    with pytest.raises(AssertionError):
        flatten_images(X, META_MNIST)  # 8×8 的图喂给 28×28 的 meta，必须当场报错


# ---------- normalize_pixels ----------

def test_归一化_值域压到0到1(digits_like):
    X, _ = digits_like
    X_flat = flatten_images(X, META_DIGITS)
    out = normalize_pixels(X_flat, META_DIGITS, CONFIG)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_归一化_系数取自meta的pixel_max():
    X = np.array([[16.0, 8.0]])
    out = normalize_pixels(X, META_DIGITS, CONFIG)
    assert np.allclose(out, [[1.0, 0.5]])          # digits 上限 16：16/16=1
    out = normalize_pixels(X, META_MNIST, CONFIG)
    assert np.allclose(out, [[16 / 255, 8 / 255]])  # mnist 上限 255：系数必须跟着数据集走


def test_归一化_无状态所以无泄漏(digits_like):
    """像素归一化除以的是先验常数，与批次里有哪些样本无关——
    单条处理和整批处理结果必须逐元素一致（模块 1 的 StandardScaler 做不到这点）。"""
    X, _ = digits_like
    X_flat = flatten_images(X, META_DIGITS)
    batch = normalize_pixels(X_flat, META_DIGITS, CONFIG)
    single = normalize_pixels(X_flat[[0]], META_DIGITS, CONFIG)
    assert np.allclose(batch[0], single[0])


def test_归一化_开关关闭时原样返回(digits_like):
    X, _ = digits_like
    X_flat = flatten_images(X, META_DIGITS)
    config = {**CONFIG, "features": {"normalize": False}}
    out = normalize_pixels(X_flat, META_DIGITS, config)
    assert np.array_equal(out, X_flat)


# ---------- split_data ----------

def test_划分_比例与形状(digits_like):
    X, y = digits_like
    X_flat = flatten_images(X, META_DIGITS)
    X_train, X_test, y_train, y_test = split_data(X_flat, y, CONFIG)
    assert X_train.shape == (150, 64)   # test_size=0.25 → 200 × 0.75
    assert X_test.shape == (50, 64)
    assert y_train.shape == (150,) and y_test.shape == (50,)


def test_划分_同种子可复现(digits_like):
    X, y = digits_like
    X_flat = flatten_images(X, META_DIGITS)
    split1 = split_data(X_flat, y, CONFIG)
    split2 = split_data(X_flat, y, CONFIG)
    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)  # random_state 固定 → 两次划分逐元素一致


def test_划分_保持类别比例(digits_like):
    X, y = digits_like
    X_flat = flatten_images(X, META_DIGITS)
    _, _, y_train, y_test = split_data(X_flat, y, CONFIG)
    # stratify=y：每类在训练/测试集中的比例应与全集一致（全集每类恰好 20 个）
    assert np.all(np.bincount(y_train) == 15)
    assert np.all(np.bincount(y_test) == 5)


# ---------- build_features ----------

def test_build_features_端到端(digits_like):
    X, y = digits_like
    X_out, y_out = build_features(X, y, META_DIGITS, CONFIG)
    assert X_out.shape == (200, 64)
    assert y_out.shape == (200,)
    assert X_out.min() >= 0.0 and X_out.max() <= 1.0  # 展平 + 归一化一步到位
