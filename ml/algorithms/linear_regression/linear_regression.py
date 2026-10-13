"""线性回归手写练习骨架。

仅用 NumPy 实现：闭式解（正规方程）、批量梯度下降、评估指标（MSE/RMSE/MAE/R²）。
所有公式与前置文档《01-线性回归前置学习文档.md》一一对应，docstring 中标注了章节。

使用流程：
    x, y = generate_sample_data()
    X_b = np.column_stack([np.ones(len(x)), x])   # 增广矩阵，文档 §3.2
    theta = closed_form(X_b, y)
    ...
"""

import numpy as np


def generate_sample_data(seed=42):
    """生成文档 §1 的示例房价数据（已实现完整，直接可用）。

    真实规律：price = 1.5 * area + 30 + 噪声（噪声 ~ N(0, 10²)）。

    参数:
        seed: int，随机种子，固定后结果可复现（文档默认 42）。

    返回:
        (area, price): 两个形状均为 (50,) 的 np.ndarray。
            area: 房屋面积（平米），均匀分布在 [40, 160)；
            price: 成交价（万）。
    """
    rng = np.random.default_rng(seed)
    n = 50
    area = rng.uniform(40, 160, n)
    price = 1.5 * area + 30 + rng.normal(0, 10, n)
    return area, price


def closed_form(X_b, y):
    """闭式解（正规方程）一步求最优参数，对应文档 §3。

    公式：θ̂ = (XᵀX)⁻¹Xᵀy

    参数:
        X_b: np.ndarray，形状 (n, d) 的增广矩阵——第一列全 1（吸收截距 b），
             其余每列一个特征。构造方式见文档 §3.2：
             np.column_stack([np.ones(n), x])
        y:   np.ndarray，形状 (n,)，真实目标值。

    返回:
        theta: np.ndarray，形状 (d,)，最优参数向量。
               theta[0] 是截距 b，theta[1:] 是各特征权重 w。

    参考输出（用 generate_sample_data() 的房价数据）:
        b ≈ 27.2841, w ≈ 1.5101（真值 b=30, w=1.5）
    """
    raise NotImplementedError


def mse(y_true, y_pred):
    """均方误差 MSE，对应文档 §2。

    公式：MSE = (1/n) · Σᵢ (yᵢ - ŷᵢ)²

    参数:
        y_true: np.ndarray，形状 (n,)，真实值。
        y_pred: np.ndarray，形状 (n,)，预测值。

    返回:
        float，标量，误差平方的平均。
    """
    raise NotImplementedError


def rmse(y_true, y_pred):
    """均方根误差 RMSE，对应文档 §7.1。

    公式：RMSE = √MSE

    参数:
        y_true: np.ndarray，形状 (n,)，真实值。
        y_pred: np.ndarray，形状 (n,)，预测值。

    返回:
        float，标量。单位和 y 一致，业务汇报首选。
    """
    raise NotImplementedError


def mae(y_true, y_pred):
    """平均绝对误差 MAE，对应文档 §7.1。

    公式：MAE = (1/n) · Σᵢ |yᵢ - ŷᵢ|

    参数:
        y_true: np.ndarray，形状 (n,)，真实值。
        y_pred: np.ndarray，形状 (n,)，预测值。

    返回:
        float，标量。比 RMSE 抗异常值。
    """
    raise NotImplementedError


def r2_score(y_true, y_pred):
    """决定系数 R²，对应文档 §7.2。

    公式：R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
    （ȳ 是 y_true 的均值；分母是"直接猜均值"的总误差，充当基线）

    参数:
        y_true: np.ndarray，形状 (n,)，真实值。
        y_pred: np.ndarray，形状 (n,)，预测值。

    返回:
        float，标量。1 = 完美预测；0 = 和猜均值一样；< 0 = 比瞎猜还差。
    """
    raise NotImplementedError


def gradient_descent(X_b, y, lr=0.1, n_iter=2000):
    """批量梯度下降训练线性回归，对应文档 §4。

    每步用全部 n 个样本计算梯度（Batch GD）：
        梯度：∂L/∂θ = -(2/n) · Xᵀ(y - ŷ)，其中 ŷ = Xθ（推导见文档 §9）
        更新：θ ← θ - η · ∂L/∂θ

    参数:
        X_b:    np.ndarray，形状 (n, d) 的增广矩阵（第一列全 1）。
                注意：特征尺度大时先做缩放（如 x / 100，文档 §4.3），否则 lr 难调。
        y:      np.ndarray，形状 (n,)，真实目标值。
        lr:     float，学习率 η。太大发散、太小收敛慢（文档 §4.4）。
        n_iter: int，迭代次数。

    返回:
        (theta, loss_history):
            theta:        np.ndarray，形状 (d,)，训练完成后的参数。
            loss_history: np.ndarray，形状 (n_iter,)，每一次参数更新后的 MSE 值。
                          正常工作时应单调下降并趋于平稳（文档 §4.3 的 loss 曲线）。
    """
    raise NotImplementedError
