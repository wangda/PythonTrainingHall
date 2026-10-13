"""model.py 的单元测试：模型工厂 create_model 已完整实现，这部分测试应当直接通过。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from model import create_model, train_model


def test_工厂_默认创建线性回归():
    config = {"model": {"name": "linear_regression", "params": {}}}
    model = create_model(config)
    assert isinstance(model, LinearRegression)


def test_工厂_ridge_使用配置里的_alpha():
    config = {"model": {"name": "ridge", "params": {"alpha": 2.5}}}
    model = create_model(config)
    assert isinstance(model, Ridge)
    assert model.alpha == 2.5


def test_工厂_未知模型名报错():
    config = {"model": {"name": "gpt5", "params": {}}}
    with pytest.raises(ValueError, match="未知模型"):
        create_model(config)


def test_train_model_拟合后参数合理():
    # y = 2x + 1 的无噪声数据，train_model 实现后应学到 coef≈2, intercept≈1
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (100, 1))
    y = 2 * X[:, 0] + 1
    config = {"model": {"name": "linear_regression", "params": {}}}
    model = train_model(create_model(config), X, y)
    assert np.allclose(model.coef_, [2.0], atol=1e-6)
    assert np.allclose(model.intercept_, 1.0, atol=1e-6)
