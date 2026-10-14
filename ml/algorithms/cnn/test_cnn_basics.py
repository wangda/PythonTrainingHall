"""CNN 手写练习的测试：小例子对照文档手算值，完整实现与 PyTorch 对答案。

运行：在本目录执行 `../../../.venv/bin/python -m pytest test_cnn_basics.py -v`

骨架状态下全部以 NotImplementedError 失败——失败信息就是你要实现的东西。
与 PyTorch 的对答案测试在未安装 torch 时自动跳过（但本模块项目需要 torch，建议装上）。
"""

import numpy as np
import pytest

from cnn_basics import conv2d_forward, correlate2d, max_pool2d, output_size

torch = pytest.importorskip("torch", reason="对答案需要 torch（pip install torch）")
import torch.nn.functional as F


# ---------------------------------------------------------------- output_size（文档 §2.3）


def test_输出尺寸_文档手算值():
    assert output_size(4, 3) == 2              # §2.2：4×4 ⊗ 3×3 → 2×2
    assert output_size(32, 3, p=1) == 32       # 黄金组合：k=3, p=1, s=1 尺寸不变
    assert output_size(32, 5) == 28            # LeNet Conv1
    assert output_size(28, 2, s=2) == 14       # LeNet Pool1
    assert output_size(14, 5) == 10            # LeNet Conv2


def test_输出尺寸_stride与padding组合():
    assert output_size(224, 7, p=3, s=2) == 112    # ResNet 第一层卷积的经典配置
    assert output_size(7, 3, p=1) == 7


# ---------------------------------------------------------------- correlate2d（文档 §2.2）


def test_互相关_竖直边缘探测器():
    """文档 §2.2 的手算例子：左边亮右边暗的 4×4 图 ⊗ Sobel 竖直边缘核。"""
    image = np.array([[10, 10, 10, 0],
                      [10, 10, 10, 0],
                      [10, 10, 10, 0],
                      [10, 10, 10, 0]], dtype=float)
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], dtype=float)
    out = correlate2d(image, kernel)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[0, 30], [0, 30]])   # 中间一列被点亮


def test_互相关_随机数据和PyTorch对答案():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(6, 7))
    kernel = rng.normal(size=(3, 3))
    ref = F.conv2d(torch.tensor(image[None, None]), torch.tensor(kernel[None, None]))
    assert np.allclose(correlate2d(image, kernel), ref.numpy()[0, 0], atol=1e-10)


# ---------------------------------------------------------------- conv2d_forward（文档 §2.4）


def test_卷积前向_单通道等价于互相关():
    """Cin=Cout=1 时 conv2d_forward 应退化为 correlate2d。"""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 1, 5, 5))
    w = rng.normal(size=(1, 1, 3, 3))
    out = conv2d_forward(x, w)
    assert out.shape == (2, 1, 3, 3)
    assert np.allclose(out[0, 0], correlate2d(x[0, 0], w[0, 0]), atol=1e-10)


def test_卷积前向_偏置被加到每个位置():
    x = np.zeros((1, 1, 4, 4))
    w = np.ones((1, 1, 3, 3))
    out = conv2d_forward(x, w, bias=np.array([2.5]))
    assert np.allclose(out, 2.5)          # 全零输入：输出处处等于偏置


def test_卷积前向_多通道与PyTorch对答案():
    """N=2, Cin=3, Cout=4, k=3, padding=1：文档 §2.4 的立体核规则全面检验。"""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 8, 8))
    w = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=(4,))
    out = conv2d_forward(x, w, b, stride=1, padding=1)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b),
                   stride=1, padding=1)
    assert out.shape == (2, 4, 8, 8)      # k=3, p=1, s=1 → 尺寸不变（黄金组合）
    assert np.allclose(out, ref.numpy(), atol=1e-10)


def test_卷积前向_stride2与PyTorch对答案():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 2, 9, 9))
    w = rng.normal(size=(5, 2, 3, 3))
    out = conv2d_forward(x, w, stride=2, padding=1)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), stride=2, padding=1)
    assert out.shape == (1, 5, 5, 5)      # (9+2−3)/2+1 = 5
    assert np.allclose(out, ref.numpy(), atol=1e-10)


# ---------------------------------------------------------------- max_pool2d（文档 §3）


def test_池化_文档手算例子():
    """文档 §3.2：4×4 输入 2×2 池化 stride=2 → [[5,4],[3,6]]。"""
    x = np.array([[1, 3, 2, 0],
                  [5, 2, 1, 4],
                  [0, 1, 6, 2],
                  [2, 3, 0, 1]], dtype=float)[None, None]   # 包成 (1,1,4,4)
    out = max_pool2d(x, kernel_size=2, stride=2)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out[0, 0], [[5, 4], [3, 6]])


def test_池化_通道独立不混合():
    """池化对每个通道独立进行——与卷积"通道相加"的本质区别。"""
    x = np.zeros((1, 2, 2, 2))
    x[0, 0] = [[1, 2], [3, 4]]
    x[0, 1] = [[10, 20], [30, 40]]
    out = max_pool2d(x, kernel_size=2, stride=2)
    assert out[0, 0, 0, 0] == 4 and out[0, 1, 0, 0] == 40   # 各通道各取各的 max


def test_池化_随机数据和PyTorch对答案():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 3, 8, 6))
    ref = F.max_pool2d(torch.tensor(x), kernel_size=2, stride=2)
    assert np.allclose(max_pool2d(x, 2, 2), ref.numpy(), atol=1e-10)
