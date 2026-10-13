"""features.py 的单元测试：形状、无 NaN、新特征、防数据泄漏、划分可复现。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.py 的三个函数后全部变绿。
骨架状态下应全部以 NotImplementedError 失败（而不是导入错误）——
失败信息就是你要实现的东西。
"""

import numpy as np
import pandas as pd
import pytest

from features import (
    ENGINEERED_COLS,
    FEATURE_COLS,
    TARGET_COL,
    add_engineered_features,
    build_features,
    split_and_scale,
)

# 与 config.yaml 对应的最小测试配置（不读真实配置文件，避免测试被配置改动影响）
CONFIG = {
    "data": {"test_size": 0.25, "random_state": 42},
    "features": {"target_col": TARGET_COL, "add_engineered_features": True},
}


@pytest.fixture
def raw_df():
    """200 行模拟数据：8 个原始特征 + 目标列，取值范围贴近真实数据。"""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "MedInc": rng.uniform(0.5, 15, n),
        "HouseAge": rng.uniform(1, 52, n),
        "AveRooms": rng.uniform(2, 10, n),
        "AveBedrms": rng.uniform(0.5, 2, n),
        "Population": rng.uniform(3, 5000, n),
        "AveOccup": rng.uniform(1, 6, n),
        "Latitude": rng.uniform(32.5, 42, n),
        "Longitude": rng.uniform(-124.3, -114, n),
    })
    df[TARGET_COL] = 0.5 * df["MedInc"] + rng.normal(0, 0.3, n)
    return df


# ---------- add_engineered_features ----------

def test_组合特征_新增三列且数值正确(raw_df):
    out = add_engineered_features(raw_df, CONFIG)
    for col in ENGINEERED_COLS:
        assert col in out.columns, f"缺少组合特征列 {col}"
    assert np.allclose(out["BedroomRatio"], raw_df["AveBedrms"] / raw_df["AveRooms"])
    assert np.allclose(out["RoomsPerPerson"], raw_df["AveRooms"] / raw_df["AveOccup"])
    assert np.allclose(out["LogMedInc"], np.log(raw_df["MedInc"]))


def test_组合特征_不修改原始入参(raw_df):
    original = raw_df.copy()
    add_engineered_features(raw_df, CONFIG)
    pd.testing.assert_frame_equal(raw_df, original)  # 入参必须原样不动


def test_组合特征_开关关闭时原样返回(raw_df):
    config = {**CONFIG, "features": {**CONFIG["features"], "add_engineered_features": False}}
    out = add_engineered_features(raw_df, config)
    for col in ENGINEERED_COLS:
        assert col not in out.columns


# ---------- build_features ----------

def test_build_features_形状与列(raw_df):
    X, y = build_features(raw_df, CONFIG)
    assert X.shape == (200, len(FEATURE_COLS) + len(ENGINEERED_COLS))  # 8 + 3 = 11
    assert y.shape == (200,)
    assert TARGET_COL not in X.columns  # 答案不能混进特征
    for col in FEATURE_COLS + ENGINEERED_COLS:
        assert col in X.columns


def test_build_features_无缺失值(raw_df):
    X, y = build_features(raw_df, CONFIG)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_build_features_开关关闭时只有原始特征(raw_df):
    config = {**CONFIG, "features": {**CONFIG["features"], "add_engineered_features": False}}
    X, _ = build_features(raw_df, config)
    assert X.shape == (200, len(FEATURE_COLS))


# ---------- split_and_scale ----------

def test_划分_比例与形状(raw_df):
    X, y = build_features(raw_df, CONFIG)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, CONFIG)
    assert X_train.shape == (150, X.shape[1])   # test_size=0.25 → 200 × 0.75
    assert X_test.shape == (50, X.shape[1])
    assert y_train.shape == (150,) and y_test.shape == (50,)
    assert isinstance(X_train, np.ndarray)


def test_划分_同种子可复现(raw_df):
    X, y = build_features(raw_df, CONFIG)
    split1 = split_and_scale(X, y, CONFIG)
    split2 = split_and_scale(X, y, CONFIG)
    for a, b in zip(split1[:4], split2[:4]):
        assert np.array_equal(a, b)  # random_state 固定 → 两次划分逐元素一致


def test_标准化_训练集均值为0标准差为1(raw_df):
    X, y = build_features(raw_df, CONFIG)
    X_train, X_test, _, _, scaler = split_and_scale(X, y, CONFIG)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-8)


def test_标准化_无数据泄漏(raw_df):
    from sklearn.model_selection import train_test_split

    X, y = build_features(raw_df, CONFIG)
    _, _, _, _, scaler = split_and_scale(X, y, CONFIG)

    # 用相同参数在测试侧独立重放一次划分，拿到【未缩放】的训练集
    X_train_raw = train_test_split(
        X, y, test_size=CONFIG["data"]["test_size"], random_state=CONFIG["data"]["random_state"]
    )[0]

    # scaler 的均值必须等于【训练集】的均值，而不是全量数据的均值——
    # 若有人图省事在划分前对全量数据 fit，这里会挂（两组均值因抽样噪声必然有差异）
    assert np.allclose(scaler.mean_, X_train_raw.mean(axis=0).values, atol=1e-8)
    assert not np.allclose(scaler.mean_, X.mean(axis=0).values, atol=1e-8)
