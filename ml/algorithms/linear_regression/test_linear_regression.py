"""手写线性回归的测试：正确性对照已知解析值与 sklearn，通过标准 = 全部 allclose。

测试数据与前置文档 §1/§3/§4 同分布：generate_sample_data(seed=42) 生成的房价数据。
运行：在本目录执行 `../../../.venv/bin/python -m pytest test_linear_regression.py -v`
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score as sklearn_r2_score

from linear_regression import (
    closed_form,
    generate_sample_data,
    gradient_descent,
    mae,
    mse,
    r2_score,
    rmse,
)


@pytest.fixture(scope="module")
def housing_data():
    """文档 §1 的房价数据（seed=42，可复现）。"""
    return generate_sample_data(seed=42)


@pytest.fixture(scope="module")
def design_matrix(housing_data):
    """增广矩阵 X_b：第一列全 1 吸收截距 b（文档 §3.2）。"""
    x, _ = housing_data
    return np.column_stack([np.ones(len(x)), x])


def test_数据生成器_形状与可复现性():
    # 已实现完整的辅助函数：形状正确，同 seed 结果一致
    x, y = generate_sample_data(seed=42)
    assert x.shape == (50,) and y.shape == (50,)
    x2, y2 = generate_sample_data(seed=42)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    # 与文档 §1 打印的前 5 个值核对（文档已实际运行验证）
    assert np.allclose(x[:5], [132.9, 92.7, 143.0, 123.7, 51.3], atol=0.05)
    assert np.allclose(y[:5], [236.1, 169.7, 247.4, 221.8, 92.4], atol=0.05)


def test_闭式解_对照已知解析值(design_matrix, housing_data):
    # 文档 §3.3 已验证：b ≈ 27.2841, w ≈ 1.5101（真值 30 / 1.5）
    _, y = housing_data
    theta = closed_form(design_matrix, y)
    assert theta.shape == (2,)
    assert np.allclose(theta, [27.2841, 1.5101], atol=1e-3)


def test_闭式解_对照_sklearn(design_matrix, housing_data):
    # sklearn LinearRegression 内部也是闭式解路线（文档 §8），两边应 allclose
    x, y = housing_data
    theta = closed_form(design_matrix, y)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    assert np.allclose(theta[0], model.intercept_)
    assert np.allclose(theta[1], model.coef_[0])


def test_mse_对照_sklearn(design_matrix, housing_data):
    _, y = housing_data
    theta = np.linalg.lstsq(design_matrix, y, rcond=None)[0]  # 测试侧独立求解，避免依赖待实现函数
    y_pred = design_matrix @ theta
    # 文档 §7.1：MSE ≈ 57.348
    assert np.allclose(mse(y, y_pred), mean_squared_error(y, y_pred))
    assert np.allclose(mse(y, y_pred), 57.348, atol=1e-2)


def test_mse_返回标量浮点数(housing_data):
    _, y = housing_data
    result = mse(y, y + 1.0)
    assert np.ndim(result) == 0
    assert np.allclose(result, 1.0)  # 预测全部偏 1，MSE 必为 1


def test_rmse_mae_对照_sklearn(design_matrix, housing_data):
    _, y = housing_data
    theta = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    y_pred = design_matrix @ theta
    assert np.allclose(rmse(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)))
    assert np.allclose(mae(y, y_pred), mean_absolute_error(y, y_pred))
    # 文档 §7.1 参考值：RMSE ≈ 7.573，MAE ≈ 6.289
    assert np.allclose(rmse(y, y_pred), 7.573, atol=1e-2)
    assert np.allclose(mae(y, y_pred), 6.289, atol=1e-2)


def test_r2_完美预测等于_1(housing_data):
    # 预测值完全等于真实值时 R² 必须恰为 1（文档 §7.2）
    _, y = housing_data
    assert np.allclose(r2_score(y, y.copy()), 1.0)


def test_r2_猜均值等于_0(housing_data):
    # 全部猜均值 = 瞎猜基线，R² 应为 0（文档 §7.2）
    _, y = housing_data
    assert np.allclose(r2_score(y, np.full_like(y, y.mean())), 0.0)


def test_r2_对照_sklearn(design_matrix, housing_data):
    _, y = housing_data
    theta = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    y_pred = design_matrix @ theta
    # 文档 §7.2 / §8：R² ≈ 0.9779
    assert np.allclose(r2_score(y, y_pred), sklearn_r2_score(y, y_pred))
    assert np.allclose(r2_score(y, y_pred), 0.9779, atol=1e-3)


def test_梯度下降_收敛且与闭式解一致(housing_data):
    # 文档 §4.3：特征缩放（x/100）后 lr=0.1 跑 2000 步收敛到闭式解
    x, y = housing_data
    x_s = x / 100
    X_b = np.column_stack([np.ones(len(x)), x_s])
    theta_gd, loss_history = gradient_descent(X_b, y, lr=0.1, n_iter=2000)

    theta_cf = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # 同一缩放矩阵上的闭式解
    assert theta_gd.shape == (2,)
    assert np.allclose(theta_gd, theta_cf, atol=1e-3)
    # 文档 §4.3：w 缩过 100 倍，除回去后应还原出 b≈27.2841, w≈1.5101
    assert np.allclose(theta_gd[0], 27.2841, atol=1e-3)
    assert np.allclose(theta_gd[1] / 100, 1.5101, atol=1e-3)


def test_梯度下降_loss历史单调下降并收敛(housing_data):
    # 文档 §4.3：loss 12222 → 57.43 → 57.35，单调下降然后躺平
    x, y = housing_data
    X_b = np.column_stack([np.ones(len(x)), x / 100])
    _, loss_history = gradient_descent(X_b, y, lr=0.1, n_iter=2000)

    assert isinstance(loss_history, np.ndarray)
    assert loss_history.shape == (2000,)  # 每次更新记录一个 MSE
    assert np.all(np.diff(loss_history) <= 1e-9)  # 单调不增（允许浮点误差）
    assert loss_history[0] > 1000  # 起点远离最优
    assert np.allclose(loss_history[-1], 57.3479, atol=1e-2)  # 收敛到闭式解的损失
