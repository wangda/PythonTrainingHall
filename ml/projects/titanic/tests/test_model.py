"""model.py 的单元测试：工厂创建正确模型、未知名报错、训练一行跑通。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import numpy as np
import pandas as pd
import pytest

from model import MODEL_NAMES, create_model, train_model

TINY_X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                       "f2": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]})
TINY_Y = np.array([0, 0, 0, 1, 1, 1])


def _config(name, params=None):
    return {"model": {"name": name, "params": params or {}}}


def test_工厂_决策树():
    from sklearn.tree import DecisionTreeClassifier
    model = create_model(_config("decision_tree", {"max_depth": 3, "random_state": 0}))
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 3


def test_工厂_随机森林():
    from sklearn.ensemble import RandomForestClassifier
    model = create_model(_config("random_forest", {"n_estimators": 10, "random_state": 0}))
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10


def test_工厂_lightgbm():
    lgb = pytest.importorskip("lightgbm")     # 未安装则跳过（先装：pip install lightgbm）
    model = create_model(_config("lightgbm", {"n_estimators": 10, "verbose": -1}))
    assert isinstance(model, lgb.LGBMClassifier)
    assert model.n_estimators == 10


def test_工厂_xgboost():
    xgb = pytest.importorskip("xgboost")      # 未安装则跳过
    model = create_model(_config("xgboost", {"n_estimators": 10}))
    assert isinstance(model, xgb.XGBClassifier)


def test_工厂_未知模型名报错():
    with pytest.raises(ValueError, match="未知模型"):
        create_model(_config("gpt5"))


def test_train_model_跑通并能预测():
    model = create_model(_config("decision_tree", {"random_state": 0}))
    model = train_model(model, TINY_X, TINY_Y)
    pred = model.predict(TINY_X)
    assert pred.shape == (6,)
    # 玩具数据线性可分（f1>2.5 → 1），树必须学满分
    assert (pred == TINY_Y).all()


def test_模型名单_与文档一致():
    assert set(MODEL_NAMES) == {"decision_tree", "random_forest", "lightgbm", "xgboost"}
