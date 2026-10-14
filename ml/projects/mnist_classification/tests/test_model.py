"""model.py 的单元测试：模型工厂 create_model 已完整实现，这部分测试应当直接通过。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from model import create_model, train_model


def test_工厂_默认创建逻辑回归():
    config = {"model": {"name": "logistic_regression", "params": {}}}
    model = create_model(config)
    assert isinstance(model, LogisticRegression)
    assert model.C == 1.0
    assert model.max_iter == 1000


def test_工厂_透传配置里的超参数():
    config = {"model": {"name": "logistic_regression", "params": {"C": 0.1, "max_iter": 200}}}
    model = create_model(config)
    assert model.C == 0.1
    assert model.max_iter == 200


def test_工厂_未知模型名报错():
    config = {"model": {"name": "gpt5", "params": {}}}
    with pytest.raises(ValueError, match="未知模型"):
        create_model(config)


def test_train_model_拟合后可分数据():
    # 三类线性可分的模拟数据，train_model 实现后应在测试点上全部分对
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    X = np.vstack([c + rng.normal(0, 0.3, (50, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 50)
    config = {"model": {"name": "logistic_regression", "params": {}}}
    model = train_model(create_model(config), X, y)
    assert (model.predict(centers) == [0, 1, 2]).all()
    # 分类模型还要能输出概率：每行和为 1（前置文档 §5 softmax）
    assert np.allclose(model.predict_proba(centers).sum(axis=1), 1.0)
