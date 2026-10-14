"""手写逻辑回归的测试：正确性对照已知解析值与 sklearn，通过标准 = 全部 allclose。

测试数据与前置文档 §1/§5 同分布：generate_sample_data(seed=42) 与 generate_multiclass_data(seed=0)。
运行：在本目录执行 `../../../.venv/bin/python -m pytest test_logistic_regression.py -v`
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score

from logistic_regression import (
    accuracy_score,
    confusion_matrix,
    cross_entropy,
    f1_score,
    generate_multiclass_data,
    generate_sample_data,
    gradient_descent,
    one_hot,
    precision_score,
    predict,
    predict_proba,
    recall_score,
    sigmoid,
    softmax,
    softmax_cross_entropy,
    softmax_gradient_descent,
)


def _fit_logistic_no_penalty(X, y):
    """无正则逻辑回归（对答案必须无正则，否则系数对不上手写版，文档 §8）。

    sklearn 1.9 的官方写法：C=np.inf（C=1/λ，λ=0 即无正则）。旧写法
    penalty='none'（字符串）直接报错，penalty=None 带 FutureWarning（1.10 将移除）——
    版本细节见文档 §8。
    """
    return LogisticRegression(C=np.inf).fit(X, y)


# ---------------------------------------------------------------- 数据 fixture

@pytest.fixture(scope="module")
def binary_data():
    """文档 §1 的贯穿例子"考试通过预测"数据（seed=42，可复现）。"""
    return generate_sample_data(seed=42)


@pytest.fixture(scope="module")
def binary_design_scaled(binary_data):
    """增广矩阵 X_b：第一列全 1 吸收截距；第 0 列特征 hours 缩放 /10 对齐量纲。"""
    X, _ = binary_data
    X_s = X.copy()
    X_s[:, 0] /= 10
    return np.column_stack([np.ones(len(X_s)), X_s])


@pytest.fixture(scope="module")
def sklearn_binary(binary_data):
    """sklearn 无正则逻辑回归，作为二分类的标准答案。"""
    X, y = binary_data
    return _fit_logistic_no_penalty(X, y)


@pytest.fixture(scope="module")
def sklearn_theta(sklearn_binary):
    """sklearn 学出的参数拼成手写约定的 theta = [b, w1, w2]。"""
    return np.concatenate([sklearn_binary.intercept_, sklearn_binary.coef_[0]])


@pytest.fixture(scope="module")
def multi_data():
    """文档 §5 的三簇二维高斯数据（seed=0，可复现）。"""
    return generate_multiclass_data(seed=0)


@pytest.fixture(scope="module")
def sklearn_multi(multi_data):
    """sklearn 无正则多分类逻辑回归（multinomial），作为多分类的标准答案。"""
    X, y = multi_data
    return _fit_logistic_no_penalty(X, y)


# ---------------------------------------------------------------- 数据生成器

def test_数据生成器_形状与可复现性():
    # 已实现完整的辅助函数：形状正确，同 seed 结果一致
    X, y = generate_sample_data(seed=42)
    assert X.shape == (200, 2) and y.shape == (200,)
    X2, y2 = generate_sample_data(seed=42)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    # 与文档 §1 打印的前 5 个值核对（文档已实际运行验证）
    assert np.allclose(
        X[:5],
        [[7.7396, 0.8887], [4.3888, 0.9859], [8.5860, 0.7504],
         [6.9737, 0.5719], [0.9418, 0.5070]],
        atol=0.05,
    )
    assert y.sum() == 124  # 200 人里 124 通过、76 挂科


def test_多分类数据生成器_形状与可复现性():
    # 已实现完整的辅助函数：三簇各 100 点，标签 0/1/2
    X, y = generate_multiclass_data(seed=0)
    assert X.shape == (300, 2) and y.shape == (300,)
    assert np.bincount(y).tolist() == [100, 100, 100]
    X2, y2 = generate_multiclass_data(seed=0)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    # 与文档 §5 打印的前 3 个值核对（文档已实际运行验证）
    assert np.allclose(
        X[:3],
        [[0.1006, -0.1057], [0.5123, 0.0839], [-0.4285, 0.2893]],
        atol=0.05,
    )


# ---------------------------------------------------------------- sigmoid（§2）

def test_sigmoid_对照已知解析值():
    # 文档 §2：σ(0)=0.5，σ(2)≈0.8808，σ(-2)≈0.1192
    assert np.allclose(sigmoid(0.0), 0.5)
    assert np.allclose(sigmoid(2.0), 0.880797, atol=1e-6)
    assert np.allclose(sigmoid(-2.0), 0.119203, atol=1e-6)


def test_sigmoid_对称性():
    # 文档 §2 性质：σ(-z) = 1 - σ(z)
    z = np.linspace(-10, 10, 21)
    assert np.allclose(sigmoid(-z), 1 - sigmoid(z))


def test_sigmoid_大数不溢出():
    # 文档 §2 数值坑：naive 实现里 np.exp(1000) 会溢出，必须处理
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 有 RuntimeWarning(overflow) 直接判失败
        result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(result).all()
    assert np.allclose(result, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------- 交叉熵（§3）

def test_交叉熵_手算已知值():
    # 文档 §3 手算例子：-[log0.9 + log0.8 + log0.7 + log0.9] / 4 ≈ 0.1976
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1])
    assert np.allclose(cross_entropy(y_true, y_prob), 0.197635, atol=1e-6)


def test_交叉熵_完美预测接近零且容忍极端概率():
    # 预测全对时 L≈0；p 含 0 和 1 也不能炸出 inf/nan（log(0) 防护）
    y_true = np.array([1.0, 0.0])
    assert np.allclose(cross_entropy(y_true, [1.0, 0.0]), 0.0, atol=1e-12)
    loss = cross_entropy(y_true, [0.0, 1.0])  # 全错且极其自信
    assert np.isfinite(loss) and loss > 30


def test_交叉熵_对照_sklearn(binary_data, sklearn_binary):
    # sklearn.metrics.log_loss 与文档 §3 的定义一致
    X, y = binary_data
    p = sklearn_binary.predict_proba(X)[:, 1]
    assert np.allclose(cross_entropy(y, p), log_loss(y, p))


# ---------------------------------------------------------------- 梯度下降（§4）

def test_梯度下降_loss历史单调下降并收敛(binary_design_scaled, binary_data):
    # 缩放特征 + lr=1.0 跑 10000 步：loss 0.6886 → 0.3667，单调下降然后躺平
    _, y = binary_data
    _, loss_history = gradient_descent(binary_design_scaled, y, lr=1.0, n_iter=10000)

    assert isinstance(loss_history, np.ndarray)
    assert loss_history.shape == (10000,)  # 每次更新记录一个交叉熵
    assert np.all(np.diff(loss_history) <= 1e-9)  # 单调不增（允许浮点误差）
    assert loss_history[0] > 0.6  # 起点远离最优
    assert np.allclose(loss_history[-1], 0.36668, atol=1e-4)  # 收敛到 sklearn 的最优损失


def test_梯度下降_收敛后与sklearn系数一致(binary_design_scaled, binary_data, sklearn_theta):
    # 文档 §4/§8：无正则下，手写 GD 收敛后应逼近 sklearn 的 MLE 解
    # 实测最大偏差 ≈ 0.008（交叉熵收敛慢），容差放宽到 atol=0.01
    _, y = binary_data
    theta_gd, _ = gradient_descent(binary_design_scaled, y, lr=1.0, n_iter=10000)
    # 手写用的特征 hours 缩放过 10 倍，权重除回去再比较
    theta_unscaled = theta_gd.copy()
    theta_unscaled[1] /= 10
    assert theta_gd.shape == (3,)
    assert np.allclose(theta_unscaled, sklearn_theta, atol=0.01)
    # 文档 §4 参考值：b≈-10.05, w_hours≈0.80, w_attendance≈9.48
    assert np.allclose(theta_unscaled, [-10.0534, 0.7991, 9.4790], atol=0.01)


# ---------------------------------------------------------------- 预测（§1/§6）

def test_predict_proba_形状与值域且对照sklearn(binary_data, sklearn_binary, sklearn_theta):
    X, _ = binary_data
    X_b = np.column_stack([np.ones(len(X)), X])
    p = predict_proba(X_b, sklearn_theta)
    assert p.shape == (200,)
    assert (p > 0).all() and (p < 1).all()  # sigmoid 输出严格在 (0, 1) 内
    # 同一个 theta 下必须与 sklearn predict_proba 完全一致
    assert np.allclose(p, sklearn_binary.predict_proba(X)[:, 1])


def test_predict_默认阈值与sklearn一致(binary_data, sklearn_binary, sklearn_theta):
    # 文档 §6：threshold=0.5 时手写 predict 应与 sklearn.predict 逐样本一致
    X, _ = binary_data
    X_b = np.column_stack([np.ones(len(X)), X])
    y_pred = predict(X_b, sklearn_theta)
    assert y_pred.shape == (200,)
    assert set(np.unique(y_pred)) <= {0, 1}
    assert np.array_equal(y_pred, sklearn_binary.predict(X))


def test_predict_阈值越高正例越少(binary_data, sklearn_theta):
    # 文档 §6 行为断言：阈值越高越严格，预测出的正例数单调不增
    # 实测：threshold=0.9 → 70 个正例，0.5 → 128 个，0.1 → 178 个
    X, _ = binary_data
    X_b = np.column_stack([np.ones(len(X)), X])
    n_pos_low = predict(X_b, sklearn_theta, threshold=0.1).sum()
    n_pos_mid = predict(X_b, sklearn_theta, threshold=0.5).sum()
    n_pos_high = predict(X_b, sklearn_theta, threshold=0.9).sum()
    assert n_pos_high <= n_pos_mid <= n_pos_low
    assert n_pos_high < n_pos_low  # 本数据上严格成立


# ---------------------------------------------------------------- 评估指标（§7）

def test_分类指标_对照sklearn(binary_data, sklearn_binary):
    # 用 sklearn 的预测作输入，独立验证四个指标公式
    # 文档 §7 参考值：accuracy≈0.84, precision≈0.8594, recall≈0.8871, F1≈0.8730
    X, y = binary_data
    y_pred = sklearn_binary.predict(X)
    assert np.allclose(accuracy_score(y, y_pred), sklearn_accuracy_score(y, y_pred))
    assert np.allclose(precision_score(y, y_pred), sklearn_precision_score(y, y_pred))
    assert np.allclose(recall_score(y, y_pred), sklearn_recall_score(y, y_pred))
    assert np.allclose(f1_score(y, y_pred), sklearn_f1_score(y, y_pred))


def test_混淆矩阵_对照sklearn(binary_data, sklearn_binary):
    # 文档 §7 约定与 sklearn 一致：行是真实、列是预测，[[TN, FP], [FN, TP]]
    # 文档 §7 参考值：[[58, 18], [14, 110]]
    X, y = binary_data
    y_pred = sklearn_binary.predict(X)
    cm = confusion_matrix(y, y_pred)
    assert cm.shape == (2, 2)
    assert np.array_equal(cm, sklearn_confusion_matrix(y, y_pred))
    assert np.array_equal(cm, [[58, 18], [14, 110]])


# ---------------------------------------------------------------- softmax 多分类（§5）

def test_one_hot_形状与值():
    # 文档 §5：y = [0, 2, 1] → [[1,0,0],[0,0,1],[0,1,0]]
    y = np.array([0, 2, 1])
    Y = one_hot(y, 3)
    assert Y.shape == (3, 3)
    assert np.array_equal(Y, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.allclose(Y.sum(axis=1), 1)  # 每行恰有一个 1


def test_softmax_行和为1且数值稳定():
    # 文档 §5：每行是概率分布；大 logits 必须靠"减行最大值"稳住不溢出
    z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    P = softmax(z)
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1)
    assert (P > 0).all() and (P < 1).all()
    assert np.allclose(P[0], [0.090031, 0.244728, 0.665241], atol=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 有 RuntimeWarning(overflow) 直接判失败
        P_big = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.isfinite(P_big).all()
    assert np.allclose(P_big.sum(), 1)


def test_softmax交叉熵_已知值():
    # 文档 §5：3 分类全部猜均匀分布 = 瞎猜基线 L = ln 3 ≈ 1.0986
    y = np.array([0, 1, 2])
    Y = one_hot(y, 3)
    P_uniform = np.full((3, 3), 1 / 3)
    assert np.allclose(softmax_cross_entropy(Y, P_uniform), np.log(3), atol=1e-6)
    # 预测全对 → L ≈ 0
    assert np.allclose(softmax_cross_entropy(Y, Y.copy()), 0.0, atol=1e-12)


def test_softmax梯度下降_loss历史单调下降并收敛(multi_data):
    # 文档 §5：默认 lr=0.5、n_iter=2000，loss 0.6972 → 0.1177，单调下降
    X, y = multi_data
    X_b = np.column_stack([np.ones(len(X)), X])
    _, loss_history = softmax_gradient_descent(X_b, one_hot(y, 3))

    assert isinstance(loss_history, np.ndarray)
    assert loss_history.shape == (2000,)
    assert np.all(np.diff(loss_history) <= 1e-9)
    assert np.allclose(loss_history[0], 0.6972, atol=1e-3)
    assert np.allclose(loss_history[-1], 0.1177, atol=1e-3)


def test_softmax梯度下降_三簇准确率(multi_data):
    # 文档 §5：三簇基本线性可分，训练准确率应 > 0.9（实测 0.9533）
    X, y = multi_data
    X_b = np.column_stack([np.ones(len(X)), X])
    Theta, _ = softmax_gradient_descent(X_b, one_hot(y, 3))
    assert Theta.shape == (3, 3)  # (d=3, K=3)
    y_pred = (X_b @ Theta).argmax(axis=1)  # softmax 单调，直接对 logits 取 argmax
    assert (y_pred == y).mean() > 0.9


def test_softmax梯度下降_预测概率与sklearn近似(multi_data, sklearn_multi):
    # 文档 §5/§8：无正则下与 sklearn（multinomial）的 predict_proba 近似一致
    # 默认参数（2000 步）实测最大偏差 ≈ 0.014，容差取 atol=0.05；
    # 想更严可把 n_iter 提到 5000（偏差降到 0.001 量级）
    X, y = multi_data
    X_b = np.column_stack([np.ones(len(X)), X])
    Theta, _ = softmax_gradient_descent(X_b, one_hot(y, 3))
    P_hand = softmax(X_b @ Theta)
    P_sklearn = sklearn_multi.predict_proba(X)
    assert np.allclose(P_hand, P_sklearn, atol=0.05)
