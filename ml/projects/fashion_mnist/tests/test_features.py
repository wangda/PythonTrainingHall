"""features.py 的单元测试：三集合划分无泄漏、归一化值域、DataLoader 形状与 shuffle。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.build_dataloaders 后全部变绿。
骨架状态下应全部以 NotImplementedError 失败（而不是导入错误）——
失败信息就是你要实现的东西。
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from features import build_dataloaders

# 与 config.yaml 对应的最小测试配置（不读真实配置文件，避免测试被配置改动影响）
CONFIG = {
    "data": {"val_size": 0.2, "random_state": 42},
    "features": {"normalize": True, "standardize": False},
    "train": {"batch_size": 32},
}


@pytest.fixture
def toy_data():
    """玩具数据：训练 200 张（每类 20 张），测试 50 张，28×28 像素 0~255。"""
    rng = np.random.default_rng(0)
    X_train = rng.integers(0, 256, (200, 28, 28), dtype=np.uint8)
    y_train = np.arange(200, dtype=np.uint8) % 10
    X_test = rng.integers(0, 256, (50, 28, 28), dtype=np.uint8)
    y_test = np.arange(50, dtype=np.uint8) % 10
    return X_train, y_train, X_test, y_test


def _collect(loader):
    """把 DataLoader 里所有样本收集成 numpy（泄漏检查用）。"""
    xs, ys = [], []
    for X, y in loader:
        xs.append(X.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def test_返回三个DataLoader(toy_data):
    train_loader, val_loader, test_loader = build_dataloaders(*toy_data, CONFIG)
    for loader in (train_loader, val_loader, test_loader):
        assert isinstance(loader, DataLoader)


def test_划分_验证集从训练集切(toy_data):
    X_train, _, X_test, _ = toy_data
    train_loader, val_loader, test_loader = build_dataloaders(*toy_data, CONFIG)
    # val_size=0.2：训练子集 160 + 验证集 40 = 原训练集 200；测试集 50 原样
    assert len(train_loader.dataset) == 160
    assert len(val_loader.dataset) == 40
    assert len(test_loader.dataset) == 50


def test_划分_测试集与训练验证零重叠(toy_data):
    """数据泄漏检查：测试集的任何一个样本都不许出现在 train/val 里。"""
    _, _, X_test, _ = toy_data
    train_loader, val_loader, test_loader = build_dataloaders(*toy_data, CONFIG)
    X_test_flat = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0
    for loader in (train_loader, val_loader):
        xs, _ = _collect(loader)
        # 逐样本比对：测试集任一行不应与 train/val 任一行逐元素相等
        for row in X_test_flat:
            assert not np.any(np.all(np.isclose(xs, row, atol=1e-6), axis=1))


def test_划分_同种子可复现(toy_data):
    loaders1 = build_dataloaders(*toy_data, CONFIG)
    loaders2 = build_dataloaders(*toy_data, CONFIG)
    # train_loader 开了 shuffle，遍历顺序每次不同，排序后比集合；val/test 顺序稳定，直接比
    x1, x2 = _collect(loaders1[0])[0], _collect(loaders2[0])[0]
    assert np.array_equal(np.sort(x1, axis=0), np.sort(x2, axis=0))
    for l1, l2 in zip(loaders1[1:], loaders2[1:]):
        assert np.array_equal(_collect(l1)[0], _collect(l2)[0])


def test_归一化_值域压到0到1(toy_data):
    for loader in build_dataloaders(*toy_data, CONFIG):
        xs, _ = _collect(loader)
        assert xs.min() >= 0.0 and xs.max() <= 1.0


def test_归一化_开关关闭时值域不变(toy_data):
    config = {**CONFIG, "features": {"normalize": False, "standardize": False}}
    for loader in build_dataloaders(*toy_data, config):
        xs, _ = _collect(loader)
        assert xs.max() > 1.0      # 没归一化，原始像素最大应远大于 1


def test_DataLoader_batch形状与类型(toy_data):
    train_loader, val_loader, test_loader = build_dataloaders(*toy_data, CONFIG)
    X, y = next(iter(train_loader))
    assert X.shape == (32, 784)                # batch_size=32，28×28 展平成 784
    assert y.shape == (32,)
    assert X.dtype == torch.float32            # 特征 float32
    assert y.dtype == torch.long               # CrossEntropyLoss 只认 long 标签


def test_DataLoader_只有训练集shuffle(toy_data):
    train_loader, val_loader, test_loader = build_dataloaders(*toy_data, CONFIG)
    assert isinstance(train_loader.sampler, RandomSampler)        # 训练必须乱序
    assert isinstance(val_loader.sampler, SequentialSampler)      # 评估乱序没意义
    assert isinstance(test_loader.sampler, SequentialSampler)
