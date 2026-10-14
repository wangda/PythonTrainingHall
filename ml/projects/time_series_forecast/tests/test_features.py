"""features.py 的单元测试：窗口对齐、按时间切分、无泄漏、DataLoader 形状。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.py 的四个函数后全部变绿。
骨架状态下应全部以 NotImplementedError 失败（而不是导入错误）——
失败信息就是你要实现的东西。
"""

import numpy as np
import pandas as pd
import pytest
import torch

from features import (
    build_datasets,
    fit_scaler_on_train,
    make_windows,
    split_by_time,
)

# 与 config.yaml 对应的最小测试配置（不读真实配置文件，避免测试被配置改动影响）
CONFIG = {
    "data": {
        "feature_cols": ["HUFL", "HULL", "OT"],
        "target_col": "OT",
        "window_size": 4,
        "horizon": 2,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42,
        "batch_size": 8,
    },
    "train": {"batch_size": 8},
}


@pytest.fixture
def series():
    """120 行 × 3 列的确定性序列：值 = 行号，逐元素可追踪。"""
    return np.arange(120 * 3, dtype=np.float64).reshape(120, 3)


# ---------- split_by_time ----------

def test_按时间切分_比例与顺序(series):
    train, val, test = split_by_time(series, val_ratio=0.2, test_ratio=0.2)
    assert len(train) == 72 and len(val) == 24 and len(test) == 24
    # 时间顺序：train 最早（行号最小），test 最晚，段间无重叠
    assert train[0, 0] < train[-1, 0] < val[0, 0] < val[-1, 0] < test[0, 0] < test[-1, 0]


def test_按时间切分_总和不变(series):
    train, val, test = split_by_time(series, val_ratio=0.15, test_ratio=0.15)
    assert len(train) + len(val) + len(test) == len(series)


# ---------- make_windows ----------

def test_滑动窗口_形状与样本数(series):
    X, y = make_windows(series, window_size=4, horizon=2, target_col=2)
    # n_samples = 120 - 4 - 2 + 1 = 115
    assert X.shape == (115, 4, 3)
    assert y.shape == (115, 2)
    assert X.dtype == np.float32 and y.dtype == np.float32


def test_滑动窗口_对齐关系逐索引断言(series):
    """第 i 个窗口的最后一行 = series[i+3]，目标第一行 = series[i+4]——绝不偷看。"""
    X, y = make_windows(series, window_size=4, horizon=2, target_col=2)
    for i in [0, 1, 50, 114]:
        assert np.array_equal(X[i], series[i : i + 4])          # 全部特征
        assert np.array_equal(y[i], series[i + 4 : i + 6, 2])   # 只取目标列


def test_滑动窗口_目标只取目标列(series):
    X, y = make_windows(series, window_size=4, horizon=2, target_col=0)
    assert np.array_equal(y[0], series[4:6, 0])  # target_col=0 时目标来自第 0 列


# ---------- fit_scaler_on_train ----------

def test_标准化_只用于训练段(series):
    train, _, _ = split_by_time(series, 0.2, 0.2)
    scaler = fit_scaler_on_train(train)
    # scaler 的统计量必须等于【训练段】的统计量，而不是全量数据的
    assert np.allclose(scaler.mean_, train.mean(axis=0))
    assert np.allclose(scaler.scale_, train.std(axis=0))
    assert not np.allclose(scaler.mean_, series.mean(axis=0))  # 全量均值必不同（泄漏探针）


def test_标准化_transform不改scaler(series):
    train, val, _ = split_by_time(series, 0.2, 0.2)
    scaler = fit_scaler_on_train(train)
    mean_before = scaler.mean_.copy()
    scaler.transform(val)  # val 只许 transform
    assert np.array_equal(scaler.mean_, mean_before)


# ---------- build_datasets（端到端） ----------

@pytest.fixture
def df(series):
    return pd.DataFrame({
        "date": pd.date_range("2016-07-01", periods=120, freq="1h"),
        "HUFL": series[:, 0],
        "HULL": series[:, 1],
        "OT": series[:, 2],
    })


def test_build_datasets_形状与划分(df):
    train_loader, val_loader, test_loader, scaler = build_datasets(df, CONFIG)
    # 窗口样本数：各段独立窗口化，n - window - horizon + 1
    assert len(train_loader.dataset) == 72 - 4 - 2 + 1   # 66
    assert len(val_loader.dataset) == 24 - 4 - 2 + 1     # 18
    assert len(test_loader.dataset) == 24 - 4 - 2 + 1    # 18
    X, y = next(iter(test_loader))
    assert X.shape == (8, 4, 3) and y.shape == (8, 2)    # (batch, window, features) / (batch, horizon)


def test_build_datasets_val_test不shuffle(df):
    """val/test 的 DataLoader 不 shuffle：两次迭代顺序完全一致。"""
    _, val_loader, test_loader, _ = build_datasets(df, CONFIG)
    batches1 = [X.clone() for X, _ in val_loader]
    batches2 = [X.clone() for X, _ in val_loader]
    for a, b in zip(batches1, batches2):
        assert torch.equal(a, b)


def test_build_datasets_标准化无泄漏(df):
    """scaler 必须只在训练段 fit：均值 ≠ 全量均值（时序趋势下必然有差异）。"""
    _, _, _, scaler = build_datasets(df, CONFIG)
    full_mean = df[CONFIG["data"]["feature_cols"]].values.mean(axis=0)
    assert not np.allclose(scaler.mean_, full_mean)
