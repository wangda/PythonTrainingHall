"""手写 MLP 的测试：梯度检查（对照数值梯度）是灵魂，外加端到端训练 sanity check。

测试数据：digits（load_digits_data(seed=42)，✅ 已实现）+ 测试内现造的随机小数据。
运行：在本目录执行 `../../../.venv/bin/python -m pytest test_mlp.py -v`
"""

import numpy as np
import pytest

from mlp import (
    backward,
    cross_entropy_loss,
    forward,
    init_params,
    load_digits_data,
    numerical_gradient,
    one_hot,
    predict,
    relu,
    relu_grad,
    softmax,
    train,
    update_params,
)


@pytest.fixture(scope="module")
def digits_data():
    """digits 数据集（✅ 辅助函数，seed=42，可复现）：1437 训练 / 360 测试，64 维。"""
    return load_digits_data(seed=42)


def _build_small_net():
    """梯度检查用的小网络与随机小数据（📝 函数在测试体内调用，骨架态下测试 FAIL 而非 ERROR）。

    sizes=[8, 5, 3]，n=10，seed=0。返回 (X, Y, params, cache, grads)。
    """
    rng = np.random.default_rng(0)
    sizes = [8, 5, 3]
    n = 10
    X = rng.normal(0, 1, (n, sizes[0]))
    Y = one_hot(rng.integers(0, sizes[-1], n), sizes[-1])
    params = init_params(sizes, seed=0)
    _, cache = forward(X, params)
    grads = backward(params, cache, X, Y)
    return X, Y, params, cache, grads


def _numerical_grad_of(param_key):
    """对小网络上某个参数算中心差分数值梯度（loss 关于该参数）。"""
    X, Y, params, _, _ = _build_small_net()

    def loss_of(p):
        p2 = {k: v.copy() for k, v in params.items()}
        p2[param_key] = p
        probs, _ = forward(X, p2)
        return cross_entropy_loss(probs, Y)

    return numerical_gradient(loss_of, params[param_key].copy())


# ---------------------------------------------------------------- ✅ 辅助函数（骨架态下即绿）


def test_数据加载_形状与可复现性(digits_data):
    X_train, X_test, y_train, y_test = digits_data
    assert X_train.shape == (1437, 64) and X_test.shape == (360, 64)
    assert X_train.dtype == np.float64
    assert X_train.min() >= 0 and X_train.max() <= 1  # 已 /16 归一化
    assert set(np.unique(y_train)) == set(range(10))  # 10 个数字类别
    again = load_digits_data(seed=42)
    assert all(np.array_equal(a, b) for a, b in zip(digits_data, again))


def test_one_hot_形状与值():
    Y = one_hot(np.array([0, 2, 1]), 3)
    assert Y.shape == (3, 3)
    assert np.array_equal(Y, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.allclose(Y.sum(axis=1), 1)


def test_softmax_行和为1且数值稳定():
    P = softmax(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1)
    assert np.allclose(P[0], [0.090031, 0.244728, 0.665241], atol=1e-6)
    P_big = softmax(np.array([[1000.0, 1001.0, 1002.0]]))  # 减行最大值后不溢出
    assert np.isfinite(P_big).all() and np.allclose(P_big.sum(), 1)


def test_数值梯度_对照已知导数():
    # f(x) = sum(x²) 的解析梯度是 2x，中心差分应逼近到 1e-8 量级
    x = np.array([1.0, -2.0, 3.0])
    grad = numerical_gradient(lambda v: (v ** 2).sum(), x)
    assert np.allclose(grad, 2 * x, atol=1e-8)


# ---------------------------------------------------------------- relu（§3）


def test_relu_已知值():
    z = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(relu(z), [0.0, 0.0, 3.0])
    assert relu(z).shape == z.shape


def test_relu_grad_已知值且零点取0():
    # z=0 处数学上不可导，工程约定取 0（文档 §3）
    z = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(relu_grad(z), [0.0, 0.0, 1.0])
    assert relu_grad(z).dtype == np.float64


# ---------------------------------------------------------------- init_params（§4）


def test_初始化_形状与偏置全零():
    params = init_params([64, 32, 10], seed=42)
    assert params["W1"].shape == (64, 32) and params["b1"].shape == (32,)
    assert params["W2"].shape == (32, 10) and params["b2"].shape == (10,)
    assert np.array_equal(params["b1"], np.zeros(32))
    assert np.array_equal(params["b2"], np.zeros(10))


def test_初始化_方差符合He且可复现():
    # He 初始化：W_l ~ N(0, 2/fan_in)。实测（seed=42）：
    # W1.std()=0.1769 vs 理论 sqrt(2/64)=0.1768；W2.std()=0.2559 vs 理论 0.2500
    # （W2 只有 320 个元素，样本方差有波动，容差取 rtol=0.05）
    params = init_params([64, 32, 10], seed=42)
    assert np.allclose(params["W1"].std(), np.sqrt(2 / 64), rtol=0.05)
    assert np.allclose(params["W2"].std(), np.sqrt(2 / 32), rtol=0.05)
    again = init_params([64, 32, 10], seed=42)
    assert all(np.array_equal(params[k], again[k]) for k in params)


def test_初始化_相邻层权重不全等():
    # 防全零对称性灾难的行为断言（文档 §4）：各层 W 必须互不相同且非零
    params = init_params([64, 32, 10], seed=42)
    assert not np.array_equal(params["W1"][:32, :10], params["W2"])  # 同形状切块对比
    assert (params["W1"] != 0).all() and (params["W2"] != 0).all()


# ---------------------------------------------------------------- forward / loss（§4/§5）


def test_前向传播_输出形状与概率性质(digits_data):
    X_train, _, _, _ = digits_data
    params = init_params([64, 32, 10], seed=42)
    probs, cache = forward(X_train[:20], params)
    assert probs.shape == (20, 10)
    assert np.allclose(probs.sum(axis=1), 1)  # 每行是概率分布
    assert np.isfinite(probs).all()  # 不爆 nan/inf
    assert cache["A0"].shape == (20, 64) and cache["Z1"].shape == (20, 32)


def test_交叉熵_已知值与均匀分布基线():
    # 3 分类全部猜均匀分布 = 瞎猜基线 L = ln 3 ≈ 1.0986
    Y = one_hot(np.array([0, 1, 2]), 3)
    assert np.allclose(cross_entropy_loss(np.full((3, 3), 1 / 3), Y), np.log(3), atol=1e-6)
    # 预测全对 → L ≈ 0；probs 含 0 和 1 也不能炸出 inf/nan（clip 防护）
    assert np.allclose(cross_entropy_loss(Y.copy(), Y), 0.0, atol=1e-12)
    loss = cross_entropy_loss(1.0 - Y, Y)  # 全错且极其自信
    assert np.isfinite(loss) and loss > 30


# ---------------------------------------------------------------- backward（§5）——梯度检查是灵魂


def test_反向传播_梯度形状与params一致():
    _, _, params, _, grads = _build_small_net()
    assert set(grads.keys()) == set(params.keys())
    for k in params:
        assert grads[k].shape == params[k].shape


def test_反向传播_数值梯度检查_W1():
    _, _, _, _, grads = _build_small_net()
    # 实测解析 vs 数值最大绝对差 ≈ 2e-11，容差留足余量（rtol=1e-4, atol=1e-6）
    assert np.allclose(grads["W1"], _numerical_grad_of("W1"), rtol=1e-4, atol=1e-6)


def test_反向传播_数值梯度检查_b1():
    _, _, _, _, grads = _build_small_net()
    # 实测最大绝对差 ≈ 6e-12
    assert np.allclose(grads["b1"], _numerical_grad_of("b1"), rtol=1e-4, atol=1e-6)


def test_反向传播_数值梯度检查_W2():
    _, _, _, _, grads = _build_small_net()
    # 实测最大绝对差 ≈ 2e-11
    assert np.allclose(grads["W2"], _numerical_grad_of("W2"), rtol=1e-4, atol=1e-6)


def test_反向传播_数值梯度检查_b2():
    _, _, _, _, grads = _build_small_net()
    # 实测最大绝对差 ≈ 1e-11
    assert np.allclose(grads["b2"], _numerical_grad_of("b2"), rtol=1e-4, atol=1e-6)


def test_反向传播_整体相对误差():
    # loss 关于全部参数的整体相对误差断言：实测 ≈ 3.6e-11，阈值 1e-5 留足余量
    _, _, params, _, grads = _build_small_net()
    for k in params:
        num = _numerical_grad_of(k)
        ana = grads[k]
        rel = np.abs(ana - num).max() / (np.abs(ana).max() + np.abs(num).max())
        assert rel < 1e-5


# ---------------------------------------------------------------- update_params（§5）


def test_参数更新_一步SGD():
    params = {"W1": np.array([[1.0, 2.0], [3.0, 4.0]]), "b1": np.array([0.5, -0.5])}
    grads = {"W1": np.array([[0.1, 0.2], [0.3, 0.4]]), "b1": np.array([1.0, -1.0])}
    out = update_params(params, grads, lr=0.1)
    assert out is params  # 原地更新并返回同一个 dict
    assert np.allclose(params["W1"], [[0.99, 1.98], [2.97, 3.96]])
    assert np.allclose(params["b1"], [0.4, -0.4])


# ---------------------------------------------------------------- train / predict（§4/§5）端到端


def test_训练_digits_loss下降且测试准确率达标(digits_data):
    # 实测（CPU 约 0.3 秒）：loss 1.9701 → 0.0060，
    # 前 5 个 epoch 均值 0.9267，后 5 个 0.0057，测试准确率 0.9694
    X_train, X_test, y_train, y_test = digits_data
    params, loss_history = train(X_train, y_train, sizes=[64, 32, 10],
                                 lr=0.5, n_epochs=200, batch_size=128, seed=42)
    assert isinstance(loss_history, np.ndarray)
    assert loss_history.shape == (200,)  # 每 epoch 一个值
    assert loss_history[:5].mean() > loss_history[-5:].mean()  # 整体下降趋势（允许小抖动）
    assert np.allclose(loss_history[-1], 0.006, atol=1e-3)
    acc = (predict(X_test, params) == y_test).mean()
    assert acc > 0.9  # 实测 0.9694，阈值留余量


def test_训练_过拟合32样本到满分(digits_data):
    # 经典 sanity check：小数据 full-batch 高 lr，训练准确率必须能到 1.0——
    # 到不了说明梯度流断了（文档 §6 调试清单第一条）
    # 实测：lr=0.5、100 epoch，loss 2.4226 → 0.0223，训练准确率 1.0
    X_train, _, y_train, _ = digits_data
    X_small, y_small = X_train[:32], y_train[:32]
    params, _ = train(X_small, y_small, sizes=[64, 32, 10],
                      lr=0.5, n_epochs=100, batch_size=32, seed=42)
    assert (predict(X_small, params) == y_small).mean() == 1.0


def test_训练_学会XOR():
    # XOR 线性不可分，是"为什么需要隐藏层"的最小例证（文档 §1）
    # 实测（seed=0）：loss 1.1309 → 0.0009，准确率 1.0
    # 注意：只有 4 个隐藏单元的 XOR 对初始化敏感，实测 20 个 seed 约一半收敛，
    # 这里固定 seed=0 保证稳定；想随便换 seed 都收敛可把隐藏层加宽到 16
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    params, _ = train(X, y, sizes=[2, 4, 2], lr=0.5, n_epochs=2000, batch_size=4, seed=0)
    assert (predict(X, params) == y).mean() == 1.0


def test_predict_返回合法标签(digits_data):
    X_train, _, y_train, _ = digits_data
    params, _ = train(X_train, y_train, sizes=[64, 32, 10],
                      lr=0.5, n_epochs=5, batch_size=128, seed=42)
    y_pred = predict(X_train[:50], params)
    assert y_pred.shape == (50,)
    assert set(np.unique(y_pred)) <= set(range(10))
