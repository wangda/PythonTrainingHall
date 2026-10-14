"""逻辑回归手写练习骨架。

仅用 NumPy 实现：sigmoid、交叉熵损失、二分类梯度下降、预测与阈值、
分类评估指标、one-hot、softmax、多分类交叉熵、softmax 梯度下降。
所有公式与前置文档《03-逻辑回归与分类前置学习文档.md》一一对应，docstring 中标注了章节。

使用流程：
    X, y = generate_sample_data()                       # 文档 §1 的贯穿例子
    X_b = np.column_stack([np.ones(len(X)), X])         # 增广矩阵（与模块 1 约定一致）
    theta, loss_history = gradient_descent(X_b, y)      # 文档 §4
    p = predict_proba(X_b, theta)
    y_pred = predict(X_b, theta, threshold=0.5)         # 文档 §6
    print(accuracy_score(y, y_pred), f1_score(y, y_pred))
"""

import numpy as np


def generate_sample_data(seed=42):
    """生成文档 §1 的贯穿例子"考试通过预测"数据（已实现完整，直接可用）。

    真实规律：logit = 0.9 * hours + 10 * attendance - 11，
    通过概率 p = σ(logit)，真实结果按 p 抽样（0=挂科, 1=通过）。

    参数:
        seed: int，随机种子，固定后结果可复现（文档默认 42）。

    返回:
        (X, y):
            X: np.ndarray，形状 (200, 2)，第 0 列是每周学习时长（0~10 小时），
               第 1 列是出勤率（0.5~1.0）；
            y: np.ndarray，形状 (200,)，取值 0.0 / 1.0。
    """
    rng = np.random.default_rng(seed)
    n = 200
    hours = rng.uniform(0, 10, n)           # 每周学习时长（小时）
    attendance = rng.uniform(0.5, 1.0, n)   # 出勤率
    logit = 0.9 * hours + 10 * attendance - 11        # 真实规律（对数几率）
    p = 1 / (1 + np.exp(-logit))                      # 通过概率
    y = (rng.uniform(0, 1, n) < p).astype(float)      # 按概率抽样出真实结果（0=挂科, 1=通过）
    X = np.column_stack([hours, attendance])
    return X, y


def generate_multiclass_data(seed=0):
    """生成 softmax 多分类练习用的三簇二维高斯数据（已实现完整，直接可用）。

    对应文档 §5。3 个簇中心分别为 [0,0] / [3,0] / [1.5,3]，每簇 100 个点，
    标准差 0.8，三类基本线性可分（手写 softmax 回归训练准确率约 95%）。

    参数:
        seed: int，随机种子，固定后结果可复现（文档默认 0）。

    返回:
        (X, y):
            X: np.ndarray，形状 (300, 2)，二维特征；
            y: np.ndarray，形状 (300,)，整型标签 0 / 1 / 2（每类各 100 个）。
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 3.0]])
    X = np.vstack([rng.normal(c, 0.8, (100, 2)) for c in centers])
    y = np.repeat(np.arange(3), 100)
    return X, y


def sigmoid(z):
    """Sigmoid 函数，对应文档 §2。

    公式：σ(z) = 1 / (1 + e⁻ᶻ)

    参数:
        z: float 或 np.ndarray，任意形状，对数几率（logit）。

    返回:
        与 z 同形状的 float / np.ndarray，值域 (0, 1)。

    参考输出:
        σ(0) = 0.5；σ(2) ≈ 0.8808；σ(-2) ≈ 0.1192；恒有 σ(-z) = 1 - σ(z)。

    实现提示:
        - 一行 `1 / (1 + np.exp(-z))` 就能跑，但有数值溢出坑：
          z = -1000 时 np.exp(1000) 会溢出为 inf 并抛 RuntimeWarning。
        - 稳妥写法：按 z 正负分两支计算（z >= 0 用 1/(1+e⁻ᶻ)，
          z < 0 用 eᶻ/(1+eᶻ)，两者都只会算 e 的负指数），可用布尔掩码索引；
          或者简单 np.clip(z, -500, 500)。测试会用 ±1000 验证不溢出。
    """
    raise NotImplementedError


def cross_entropy(y_true, y_prob):
    """二元交叉熵损失（均值），对应文档 §3。

    公式：L = -(1/n) · Σᵢ [yᵢ·log pᵢ + (1-yᵢ)·log(1-pᵢ)]

    参数:
        y_true: np.ndarray，形状 (n,)，真实标签（0 或 1）。
        y_prob: np.ndarray，形状 (n,)，预测为正类的概率 p ∈ [0, 1]。

    返回:
        float，标量，逐样本损失的均值。

    参考输出:
        y_true = [1, 1, 0, 0]，y_prob = [0.9, 0.8, 0.3, 0.1] → L ≈ 0.1976；
        预测全对（p 与 y 完全一致）→ L ≈ 0。

    实现提示:
        - log(0) = -inf，必须先做防护：np.clip(y_prob, eps, 1 - eps)，
          eps 取 1e-15 即可。测试会传入含 0 和 1 的概率验证不炸。
        - 与 sklearn.metrics.log_loss 的定义一致，可对照。
    """
    raise NotImplementedError


def gradient_descent(X_b, y, lr=0.1, n_iter=2000):
    """批量梯度下降训练二分类逻辑回归，对应文档 §4。

    模型：p = σ(X_b @ θ)（文档 §1）
    每步用全部 n 个样本计算梯度（Batch GD）：
        梯度：∂L/∂θ = (1/n) · X_bᵀ(p - y)  —— 与线性回归形式一致（推导见文档 §9）
        更新：θ ← θ - η · ∂L/∂θ

    参数:
        X_b:    np.ndarray，形状 (n, d) 的增广矩阵——第一列全 1（吸收截距 b），
                与模块 1 约定一致：np.column_stack([np.ones(n), X])。
        y:      np.ndarray，形状 (n,)，真实标签（0 或 1）。
        lr:     float，学习率 η。太大发散、太小收敛慢（文档 §4）。
        n_iter: int，迭代次数。

    返回:
        (theta, loss_history):
            theta:        np.ndarray，形状 (d,)，训练完成后的参数。
                          theta[0] 是截距 b，theta[1:] 是各特征权重 w。
            loss_history: np.ndarray，形状 (n_iter,)，每次参数更新后的交叉熵值。
                          正常工作时应单调下降并趋于平稳。

    参考输出（generate_sample_data() 数据）:
        先把第 0 列特征缩放（hours / 10，量纲对齐），再 lr=1.0、n_iter=10000：
            theta ≈ [-10.05, 7.99, 9.47]，loss 从 0.6886 降到 0.3667。
        与 sklearn LogisticRegression(penalty=None) 的系数一致（截距 -10.0534，
        两个权重 0.7991 和 9.4790——注意缩放后 hours 的权重差 10 倍）。

    实现提示:
        - theta 初始化为 np.zeros(d)。
        - 每一步都要把更新后的 loss 追加进 loss_history。
        - 交叉熵损失收敛比线性回归的 MSE 慢得多：默认 lr=0.1 跑 2000 步
          只能爬到半路（loss ≈ 0.43），属正常现象，不是实现错了；
          测试里用的是 lr=1.0、n_iter=10000。
    """
    raise NotImplementedError


def predict_proba(X_b, theta):
    """预测正类概率 p = σ(X_b @ θ)，对应文档 §1 / §4。

    参数:
        X_b:   np.ndarray，形状 (n, d) 的增广矩阵（第一列全 1）。
        theta: np.ndarray，形状 (d,)，已训练的参数。

    返回:
        np.ndarray，形状 (n,)，每个样本预测为正类（y=1）的概率，值域 (0, 1)。

    参考行为:
        用 sklearn 学出的 theta 计算，结果与 sklearn 的 predict_proba(X)[:, 1]
        完全 allclose。
    """
    raise NotImplementedError


def predict(X_b, theta, threshold=0.5):
    """按阈值把概率切成 0/1 类别，对应文档 §6。

    规则：p >= threshold → 1，否则 → 0。阈值即决策边界的位置：
    调高阈值 = 更严格（正例变少，精确率升、召回率降），调低则相反。

    参数:
        X_b:       np.ndarray，形状 (n, d) 的增广矩阵（第一列全 1）。
        theta:     np.ndarray，形状 (d,)，已训练的参数。
        threshold: float，分类阈值，默认 0.5，取值范围 [0, 1]。

    返回:
        np.ndarray，形状 (n,)，取值 0 / 1 的预测标签。

    参考行为（generate_sample_data() + sklearn 学出的 theta）:
        threshold=0.9 时预测出 70 个正例，0.5 时 128 个，0.1 时 178 个——
        阈值越高正例越少，单调不增。

    实现提示:
        一行：(predict_proba(X_b, theta) >= threshold).astype(int)。
    """
    raise NotImplementedError


def accuracy_score(y_true, y_pred):
    """准确率：预测对的比例，对应文档 §7。

    公式：accuracy = (TP + TN) / n

    参数:
        y_true: np.ndarray，形状 (n,)，真实标签（0 或 1）。
        y_pred: np.ndarray，形状 (n,)，预测标签（0 或 1）。

    返回:
        float，标量，值域 [0, 1]。

    参考输出（generate_sample_data() 上 sklearn 模型的预测）:
        accuracy ≈ 0.84。
    """
    raise NotImplementedError


def precision_score(y_true, y_pred):
    """精确率：预测为正的样本里有多少是真的正，对应文档 §7。

    公式：precision = TP / (TP + FP)

    参数:
        y_true: np.ndarray，形状 (n,)，真实标签（0 或 1）。
        y_pred: np.ndarray，形状 (n,)，预测标签（0 或 1）。

    返回:
        float，标量，值域 [0, 1]。

    参考输出（generate_sample_data() 上 sklearn 模型的预测）:
        precision ≈ 0.8594（TP=110, FP=18）。

    实现提示:
        一个正例都没预测出来时（TP + FP = 0）按 0.0 处理，避免除零。
    """
    raise NotImplementedError


def recall_score(y_true, y_pred):
    """召回率：真实为正的样本里有多少被找出来了，对应文档 §7。

    公式：recall = TP / (TP + FN)

    参数:
        y_true: np.ndarray，形状 (n,)，真实标签（0 或 1）。
        y_pred: np.ndarray，形状 (n,)，预测标签（0 或 1）。

    返回:
        float，标量，值域 [0, 1]。

    参考输出（generate_sample_data() 上 sklearn 模型的预测）:
        recall ≈ 0.8871（TP=110, FN=14）。
    """
    raise NotImplementedError


def f1_score(y_true, y_pred):
    """F1：精确率与召回率的调和平均，对应文档 §7。

    公式：F1 = 2 · P · R / (P + R)

    参数:
        y_true: np.ndarray，形状 (n,)，真实标签（0 或 1）。
        y_pred: np.ndarray，形状 (n,)，预测标签（0 或 1）。

    返回:
        float，标量，值域 [0, 1]。P 和 R 都为 0 时返回 0.0。

    参考输出（generate_sample_data() 上 sklearn 模型的预测）:
        F1 ≈ 0.8730。

    实现提示:
        复用你写的 precision_score 和 recall_score。
    """
    raise NotImplementedError


def confusion_matrix(y_true, y_pred):
    """二分类混淆矩阵，对应文档 §7。

    约定（与 sklearn.metrics.confusion_matrix 一致）：行是真实、列是预测，
        [[TN, FP],
         [FN, TP]]

    参数:
        y_true: np.ndarray，形状 (n,)，真实标签（0 或 1）。
        y_pred: np.ndarray，形状 (n,)，预测标签（0 或 1）。

    返回:
        np.ndarray，形状 (2, 2)，整型，即 [[TN, FP], [FN, TP]]。

    参考输出（generate_sample_data() 上 sklearn 模型的预测）:
        [[58, 18],
         [14, 110]]
    """
    raise NotImplementedError


def one_hot(y, n_classes):
    """整型标签转 one-hot 矩阵，对应文档 §5。

    例：y = [0, 2, 1]，n_classes = 3 →
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]]

    参数:
        y:         np.ndarray，形状 (n,)，整型标签，取值 0 ~ n_classes-1。
        n_classes: int，类别总数 K。

    返回:
        np.ndarray，形状 (n, K)，每行恰有一个 1，其余为 0。

    实现提示:
        先 np.zeros((n, K))，再用花式索引 Y[np.arange(n), y] = 1。
    """
    raise NotImplementedError


def softmax(z):
    """Softmax：把 K 个分数转成和为 1 的概率分布，对应文档 §5。

    公式：softmax(z)ₖ = eᶻᵏ / Σⱼ eᶻʲ（对每行独立计算）

    参数:
        z: np.ndarray，形状 (n, K)，n 个样本、K 个类别的原始分数（logits）。

    返回:
        np.ndarray，形状 (n, K)，每行是一个概率分布（行和为 1，值域 (0, 1)）。

    参考输出:
        softmax([[1, 2, 3]]) ≈ [[0.0900, 0.2447, 0.6652]]。

    实现提示（必须做，测试会验证）:
        e 的指数极易溢出：z = [1000, 1001, 1002] 直接 np.exp 全变 inf。
        利用 softmax 对"整行同减一个常数"不变的性质，先逐行减去行最大值：
            z = z - z.max(axis=1, keepdims=True)
        再取 exp、按行归一化（axis=1, keepdims=True）。
    """
    raise NotImplementedError


def softmax_cross_entropy(Y_onehot, P):
    """多分类交叉熵损失（均值），对应文档 §5。

    公式：L = -(1/n) · Σᵢ Σₖ Yᵢₖ · log Pᵢₖ
    （one-hot 下每行只剩真实类别那一项：-log Pᵢ,真实类）

    参数:
        Y_onehot: np.ndarray，形状 (n, K)，真实标签的 one-hot 矩阵。
        P:        np.ndarray，形状 (n, K)，softmax 输出的预测概率。

    返回:
        float，标量，逐样本损失的均值。

    参考输出:
        3 分类全部猜均匀分布 P = 1/3 → L = ln 3 ≈ 1.0986（瞎猜基线）；
        预测全对（P 与 Y_onehot 一致）→ L ≈ 0。

    实现提示:
        同二元版一样要防 log(0)：np.clip(P, 1e-15, 1)。
    """
    raise NotImplementedError


def softmax_gradient_descent(X_b, Y_onehot, lr=0.5, n_iter=2000):
    """批量梯度下降训练 softmax 多分类回归，对应文档 §5。

    模型：P = softmax(X_b @ Θ)，参数是矩阵 Θ（每列对应一个类别）。
    每步用全部 n 个样本计算梯度：
        梯度：∂L/∂Θ = (1/n) · X_bᵀ(P - Y)  —— 又是熟悉的形式（推导见文档 §9）
        更新：Θ ← Θ - η · ∂L/∂Θ

    参数:
        X_b:      np.ndarray，形状 (n, d) 的增广矩阵——第一列全 1（吸收各类别截距）。
        Y_onehot: np.ndarray，形状 (n, K)，真实标签的 one-hot 矩阵。
        lr:       float，学习率 η，默认 0.5。
        n_iter:   int，迭代次数，默认 2000。

    返回:
        (Theta, loss_history):
            Theta:        np.ndarray，形状 (d, K)，训练完成后的参数矩阵。
                          Theta[0, :] 是各类别的截距。
            loss_history: np.ndarray，形状 (n_iter,)，每次参数更新后的多分类交叉熵值。
                          正常工作时应单调下降并趋于平稳。

    参考输出（generate_multiclass_data() 数据，默认参数 lr=0.5、n_iter=2000）:
        loss 从 0.6972 降到 0.1177；训练准确率 ≈ 0.9533；
        预测概率与 sklearn LogisticRegression(penalty=None) 的 predict_proba
        最大偏差 ≈ 0.014（迭代越多越接近，n_iter=5000 时偏差降到 0.001 量级）。

    实现提示:
        - Theta 初始化为 np.zeros((d, K))。
        - 每一步复用你写的 softmax 和 softmax_cross_entropy。
        - 预测类别直接取 logits 的行内 argmax：(X_b @ Theta).argmax(axis=1)，
          不必过 softmax（softmax 单调，不改变 argmax）。
    """
    raise NotImplementedError
