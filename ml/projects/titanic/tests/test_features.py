"""features.py 的单元测试：划分可复现、状态只来自训练集（防泄漏）、编码与组合特征正确。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.py 的三个函数后全部变绿。
骨架状态下应全部以 NotImplementedError 失败——失败信息就是你要实现的东西。
"""

import numpy as np
import pandas as pd
import pytest

from features import (
    BASE_FEATURES,
    EMBARKED_CATEGORIES,
    fit_preprocessor,
    split_data,
    transform,
)

# 与 config.yaml 对应的最小测试配置（不读真实配置文件，避免测试被配置改动影响）
CONFIG = {
    "data": {"test_size": 0.25, "random_state": 42},
    "features": {"add_family_features": True},
}


@pytest.fixture
def toy_df():
    """40 行玩具泰坦尼克数据：含缺失值、泄漏列、高基数千文本列——真实数据的微缩版。"""
    rng = np.random.default_rng(0)
    n = 40
    ages = rng.uniform(1, 60, n)
    ages[::5] = np.nan                       # 每 5 行一个年龄缺失
    fares = rng.uniform(5, 100, n)
    fares[3] = np.nan                        # 一条票价缺失
    embarked = rng.choice(["C", "Q", "S"], n).astype(object)
    embarked[7] = None                       # 一条登船港口缺失
    return pd.DataFrame({
        "pclass": rng.integers(1, 4, n),
        "sex": rng.choice(["male", "female"], n),
        "age": ages,
        "sibsp": rng.integers(0, 3, n),
        "parch": rng.integers(0, 2, n),
        "fare": fares,
        "embarked": embarked,
        "survived": np.tile([0, 1], n // 2),     # 恰好 20/20 平衡，方便验证分层
        # 泄漏列与高基数文本列（真实数据里就有，features 必须删掉）
        "boat": np.where(np.arange(n) % 2 == 0, "13", None),
        "body": np.where(np.arange(n) % 2 == 1, 42.0, np.nan),
        "name": [f"Passenger {i}" for i in range(n)],
        "ticket": [f"PC-{i}" for i in range(n)],
        "cabin": [f"C{i}" for i in range(n)],
        "home.dest": ["Somewhere"] * n,
    })


# ---------------------------------------------------------------- split_data


def test_划分_比例与形状(toy_df):
    train_df, test_df = split_data(toy_df, CONFIG)
    assert len(train_df) == 30 and len(test_df) == 10     # test_size=0.25
    assert set(train_df.columns) == set(toy_df.columns)   # 只分行，不动列


def test_划分_同种子可复现(toy_df):
    train1, test1 = split_data(toy_df, CONFIG)
    train2, test2 = split_data(toy_df, CONFIG)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_划分_保持类别比例(toy_df):
    _, test_df = split_data(toy_df, CONFIG)
    # stratify=survived：全集 20/20 平衡 → 测试集 10 行必须恰好 5/5
    assert test_df["survived"].sum() == 5


# ------------------------------------------------------- fit_preprocessor + transform


def test_fit状态_统计量只来自训练集():
    """防泄漏核心测试（项目文档 §3）：6 行训练（age 中位数 25.0）+ 2 行测试（age=1000），
    测试行的缺失年龄必须填训练集的中位数 25.0——填了任何别的都说明统计量看错了地方。"""
    df = pd.DataFrame({
        "pclass": [1, 2, 3, 1, 2, 3, 1, 2],
        "sex": ["male", "female"] * 4,
        "age": [10.0, 20.0, 30.0, 40.0, np.nan, np.nan, np.nan, np.nan],
        "sibsp": [0] * 8,
        "parch": [0] * 8,
        "fare": [10.0] * 7 + [np.nan],
        "embarked": ["C", "S", "S", "S", "Q", None, "S", "S"],
        "survived": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    train_df, test_df = df.iloc[:6], df.iloc[6:]
    _, _, state = fit_preprocessor(train_df, CONFIG)
    assert state["age_median"] == 25.0        # (10,20,30,40) 的中位数，与测试行无关
    assert state["fare_median"] == 10.0
    assert state["embarked_mode"] == "S"      # 训练集众数
    X_test, _ = transform(test_df, state, CONFIG)
    assert (X_test["age"] == 25.0).all()      # 测试行的 NaN 填的是训练集统计量
    assert X_test["fare"].iloc[-1] == 10.0


def test_组合特征_数学正确(toy_df):
    train_df, _ = split_data(toy_df, CONFIG)
    X, _, state = fit_preprocessor(train_df, CONFIG)
    raw = train_df.fillna({"age": 0})  # sibsp/parch 无缺失，直接对原始列验证
    assert np.allclose(X["family_size"], raw["sibsp"] + raw["parch"] + 1)
    assert np.array_equal(X["is_alone"].values,
                          ((raw["sibsp"] + raw["parch"] + 1) == 1).astype(float).values)


def test_编码_sex与embarked(toy_df):
    train_df, _ = split_data(toy_df, CONFIG)
    X, _, _ = fit_preprocessor(train_df, CONFIG)
    assert set(X["sex_male"].unique()) <= {0.0, 1.0}
    for c in EMBARKED_CATEGORIES:
        assert f"embarked_{c}" in X.columns
    # one-hot 每行恰好一个 1（缺失已填众数，不会出现全 0 行）
    oh = X[[f"embarked_{c}" for c in EMBARKED_CATEGORIES]]
    assert (oh.sum(axis=1) == 1).all()


def test_onehot列清单稳定_子集缺类别也不断列():
    """测试集里可能恰好没有 Q 港乘客——one-hot 列必须仍然齐全（否则模型对不上列）。"""
    df = pd.DataFrame({
        "pclass": [1, 2], "sex": ["male", "female"],
        "age": [30.0, 25.0], "sibsp": [0, 0], "parch": [0, 0],
        "fare": [50.0, 20.0], "embarked": ["C", "C"],   # 只有 C，没有 Q/S
        "survived": [0, 1],
    })
    _, _, state = fit_preprocessor(df, CONFIG)
    X, _ = transform(df, state, CONFIG)
    assert list(X.columns) == state["feature_names"]
    assert (X["embarked_Q"] == 0).all() and (X["embarked_S"] == 0).all()


def test_泄漏列与高基数列被删除(toy_df):
    train_df, _ = split_data(toy_df, CONFIG)
    X, _, _ = fit_preprocessor(train_df, CONFIG)
    for col in ["boat", "body", "name", "ticket", "cabin", "home.dest"]:
        assert col not in X.columns


def test_出口_无缺失且全数值(toy_df):
    train_df, test_df = split_data(toy_df, CONFIG)
    X_train, y_train, state = fit_preprocessor(train_df, CONFIG)
    X_test, y_test = transform(test_df, state, CONFIG)
    for X in [X_train, X_test]:
        assert not X.isna().any().any(), "transform 出口不许有缺失值"
        assert all(np.issubdtype(t, np.number) for t in X.dtypes), "所有列必须是数值型"
    assert X_train.shape[1] == len(BASE_FEATURES) + 2     # 基础 9 列 + family_size + is_alone
    assert set(np.unique(y_train)) == {0, 1} and y_test.shape == (10,)


def test_开关关闭_不生成组合特征(toy_df):
    config = {**CONFIG, "features": {"add_family_features": False}}
    train_df, _ = split_data(toy_df, config)
    X, _, state = fit_preprocessor(train_df, config)
    assert "family_size" not in X.columns and "is_alone" not in X.columns
    assert state["feature_names"] == BASE_FEATURES


def test_纯函数_同状态同输入同输出(toy_df):
    """transform 是纯函数：不许偷偷统计新量（项目文档 §3）。"""
    train_df, test_df = split_data(toy_df, CONFIG)
    _, _, state = fit_preprocessor(train_df, CONFIG)
    X1, _ = transform(test_df, state, CONFIG)
    X2, _ = transform(test_df, state, CONFIG)
    pd.testing.assert_frame_equal(X1, X2)


def test_无目标列时_y为None(toy_df):
    """predict.py 场景：新乘客没有 survived 列。"""
    train_df, test_df = split_data(toy_df, CONFIG)
    _, _, state = fit_preprocessor(train_df, CONFIG)
    X, y = transform(test_df.drop(columns=["survived"]), state, CONFIG)
    assert y is None and len(X) == len(test_df)
