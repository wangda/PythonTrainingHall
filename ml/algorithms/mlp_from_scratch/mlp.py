"""MLP（多层感知机）手写练习骨架——阶段二核心里程碑。

仅用 NumPy 实现：任意层数的全连接网络（隐藏层 ReLU + 输出层 softmax + 交叉熵）、
He 初始化、前向传播、反向传播（链式法则逐层推）、mini-batch SGD 训练循环。
所有公式与前置文档《07-MLP与反向传播前置学习文档.md》一一对应，docstring 中标注了章节。

架构约定（与文档 §4/§5 一致，全练习统一遵守）：
    - sizes = [64, 32, 10] 表示 64 维输入 → 隐藏层 32 个神经元（ReLU）→ 输出层 10 类（softmax）。
      隐藏层激活函数统一 ReLU，输出层 softmax + 交叉熵（合体梯度 δ = P - Y，文档 §5）。
    - 参数用 dict 表示，键为 "W1", "b1", "W2", "b2", ...；
      W_l 形状 (sizes[l-1], sizes[l])，b_l 形状 (sizes[l],)。
    - 前向传播返回 (probs, cache)，cache 是 dict：
          cache["A0"] = X（输入），cache["Z1"], cache["A1"], ..., cache["ZL"], cache["AL"]
      其中 Z_l = A_{l-1} @ W_l + b_l，A_l = relu(Z_l)（l < L），A_L = softmax(Z_L) = probs。
      backward 靠 cache 里的 A 和 Z 逐层回传梯度。

使用流程：
    X_train, X_test, y_train, y_test = load_digits_data()
    params, loss_history = train(X_train, y_train, sizes=[64, 32, 10],
                                 lr=0.5, n_epochs=200, batch_size=128)
    acc = (predict(X_test, params) == y_test).mean()
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ============================================================ 已实现完整的辅助函数（✅）


def load_digits_data(seed=42):
    """加载 sklearn 自带的 digits 手写数字数据集（8×8 小图版 MNIST），已实现完整。

    1797 张 8×8 灰度图，像素值范围 0~16，这里除以 16 归一化到 [0, 1]——
    注意与 MNIST 的 /255 不同（digits 像素上限是 16，不是 255）。
    按 80/20 切分训练/测试集（random_state=seed），10 个类别（数字 0~9）分布均匀。

    参数:
        seed: int，随机种子，固定后切分结果可复现（默认 42）。

    返回:
        (X_train, X_test, y_train, y_test):
            X_train: np.ndarray，形状 (1437, 64)，float64，取值 [0, 1]；
            X_test:  np.ndarray，形状 (360, 64)，float64；
            y_train / y_test: np.ndarray，形状 (1437,) / (360,)，整型标签 0~9。
    """
    digits = load_digits()
    X = digits.data / 16.0  # digits 像素范围 0~16，不是 MNIST 的 0~255
    y = digits.target
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def one_hot(y, n_classes):
    """整型标签转 one-hot 矩阵，已实现完整（模块 2 已练过，这里直接给出）。

    例：y = [0, 2, 1]，n_classes = 3 →
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]]

    参数:
        y:         np.ndarray，形状 (n,)，整型标签，取值 0 ~ n_classes-1。
        n_classes: int，类别总数 K。

    返回:
        np.ndarray，形状 (n, K)，每行恰有一个 1，其余为 0。
    """
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def softmax(z):
    """Softmax：把 K 个分数转成和为 1 的概率分布，已实现完整（模块 2 已练过）。

    公式：softmax(z)ₖ = eᶻᵏ / Σⱼ eᶻʲ（对每行独立计算）
    数值稳定处理已做好：先逐行减去行最大值再取 exp（softmax 对整行同减常数不变，
    但能避免 np.exp(1000) 溢出成 inf，见文档 §4）。

    参数:
        z: np.ndarray，形状 (n, K)，n 个样本、K 个类别的原始分数（logits）。

    返回:
        np.ndarray，形状 (n, K)，每行是一个概率分布（行和为 1，值域 (0, 1)）。

    参考输出:
        softmax([[1, 2, 3]]) ≈ [[0.0900, 0.2447, 0.6652]]。
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================ 留给你手写（📝）


def relu(z):
    """ReLU 激活函数，对应文档 §3。

    公式：relu(z) = max(0, z)（逐元素）

    参数:
        z: np.ndarray，任意形状，隐藏层的加权输入（pre-activation）。

    返回:
        np.ndarray，与 z 同形状，负数截断为 0，非负原样通过。

    参考输出:
        relu([-2.0, 0.0, 3.0]) = [0.0, 0.0, 3.0]。

    实现提示:
        一行：np.maximum(0, z)。
    """
    raise NotImplementedError


def relu_grad(z):
    """ReLU 的导数，对应文档 §3。

    公式：relu′(z) = 1（z > 0）；0（z <= 0）。
    z = 0 处数学上不可导，工程约定取 0（测试按此约定断言）。

    ⚠️ 输入是 z（加权输入，激活前），不是 a（激活后）！
    反向传播里求 δ_l 时，relu′ 必须吃 cache["Z_l"]。这是最常见的 bug 之一：
    概念上导数是"对加权输入 z 求导"，请坚持传 z，养成正确的心智模型。

    参数:
        z: np.ndarray，任意形状，隐藏层的加权输入（激活前的 Z，不是激活后的 A）。

    返回:
        np.ndarray，与 z 同形状，取值 0.0 / 1.0（float）。

    参考输出:
        relu_grad([-2.0, 0.0, 3.0]) = [0.0, 0.0, 1.0]（注意 z=0 处取 0）。

    实现提示:
        一行：(z > 0).astype(float)。
    """
    raise NotImplementedError


def init_params(sizes, seed=42):
    """He 初始化网络参数，对应文档 §4。

    为什么不能用全零：所有权重相同 → 同一层的所有神经元输出相同、梯度相同、
    更新相同，永远学不出不同特征（对称性灾难，文档 §4）。
    为什么用 He：ReLU 会截断一半输入，He 初始化按 W_l ~ N(0, 2/sizes[l-1])
    缩放方差，让各层激活值方差在深层网络中保持稳定（文档 §4）。

    参数:
        sizes: list[int]，各层神经元个数，含输入层和输出层。
               如 [64, 32, 10] = 输入 64 维 → 隐藏层 32 → 输出 10 类。
               支持任意层数（[64, 32, 16, 10] 就是两个隐藏层）。
        seed:  int，随机种子，固定后参数可复现。用 np.random.default_rng(seed)。

    返回:
        params: dict，键 "W1", "b1", ..., "WL", "bL"（L = len(sizes) - 1）：
            W_l: np.ndarray，形状 (sizes[l-1], sizes[l])，
                 元素 ~ N(0, 2/sizes[l-1])，即 rng.normal(0, sqrt(2/sizes[l-1]), 形状)；
            b_l: np.ndarray，形状 (sizes[l],)，全零（偏置不需要随机打破对称）。

    参考输出（sizes=[64, 32, 10], seed=42）:
        W1.std() ≈ 0.1769（理论 sqrt(2/64) ≈ 0.1768），
        W2.std() ≈ 0.2559（理论 sqrt(2/32) = 0.2500，元素少时样本方差有波动）。
    """
    raise NotImplementedError


def forward(X, params):
    """前向传播：逐层算 Z 和 A，输出 softmax 概率，对应文档 §4。

    对 L 层网络（L = len(params) // 2）：
        A0 = X
        Z_l = A_{l-1} @ W_l + b_l          （l = 1..L）
        A_l = relu(Z_l)                    （l = 1..L-1，隐藏层）
        A_L = softmax(Z_L)                 （输出层）

    参数:
        X:      np.ndarray，形状 (n, sizes[0])，n 个样本的输入特征。
        params: dict，init_params 返回的参数。

    返回:
        (probs, cache):
            probs: np.ndarray，形状 (n, K)，softmax 输出的预测概率（即 A_L），
                   每行和为 1；
            cache: dict，backward 要用的中间结果，键值约定（全练习统一）：
                   "A0" = X（输入本身），
                   "Z1", "A1", ..., "ZL", "AL"（每层的加权输入和激活输出，
                   "AL" 就是 probs）。

    参考行为（sizes=[64, 32, 10]，digits 数据）:
        probs.shape == (n, 10)；每行和为 1；不出现 nan/inf。

    实现提示:
        - 层数从 params 推导：L = len(params) // 2，不要写死两层。
        - 别忘了把 A0 = X 也放进 cache，backward 算 ∂L/∂W1 要用它。
    """
    raise NotImplementedError


def cross_entropy_loss(probs, Y):
    """多分类交叉熵损失（均值），对应文档 §5。

    公式：L = -(1/n) · Σᵢ Σₖ Yᵢₖ · log Pᵢₖ
    （one-hot 下每行只剩真实类别那一项：-log Pᵢ,真实类）

    参数:
        probs: np.ndarray，形状 (n, K)，softmax 输出的预测概率 P。
        Y:     np.ndarray，形状 (n, K)，真实标签的 one-hot 矩阵。

    返回:
        float，标量，逐样本损失的均值。

    参考输出:
        K 分类全部猜均匀分布 P = 1/K → L = ln K（瞎猜基线，3 类时 ≈ 1.0986）；
        预测全对（P 与 Y 一致）→ L ≈ 0；
        小网络 [8,5,3]、seed=0 的随机数据上，初始 loss ≈ 1.3918。

    实现提示:
        log(0) = -inf，必须先 clip：np.log(np.clip(probs, 1e-15, 1))。
    """
    raise NotImplementedError


def backward(params, cache, X, Y):
    """反向传播：从输出层逐层回传误差，算出所有参数的梯度，对应文档 §5。核心中的核心。

    链式法则逐层推（L 层网络，n = 批大小）：
        输出层：δ_L = P - Y                        （softmax + 交叉熵的合体梯度，形状 (n, K)）
        回传：  δ_l = (δ_{l+1} @ W_{l+1}ᵀ) ⊙ relu′(Z_l)   （l = L-1 .. 1）
        梯度：  ∂L/∂W_l = A_{l-1}ᵀ @ δ_l / n
                ∂L/∂b_l = mean(δ_l, axis=0)        （对批内样本取平均）

    以 sizes=[64, 32, 10]（L=2）为例，各张量形状：
        P, Y:      (n, 10)
        δ₂:        (n, 10)
        ∂L/∂W₂:    (32, 10)   = A₁ᵀ(n, 32) @ δ₂(n, 10) / n
        ∂L/∂b₂:    (10,)
        δ₁:        (n, 32)    = (δ₂ @ W₂ᵀ(n, 32)) ⊙ relu′(Z₁)
        ∂L/∂W₁:    (64, 32)
        ∂L/∂b₁:    (32,)

    参数:
        params: dict，当前参数（回传 δ 时要用各层 W 的当前值）。
        cache:  dict，forward 返回的缓存（"A0".."AL"、"Z1".."ZL"）。
        X:      np.ndarray，形状 (n, sizes[0])，本批输入（即 cache["A0"]）。
        Y:      np.ndarray，形状 (n, K)，本批标签的 one-hot 矩阵。

    返回:
        grads: dict，与 params 完全同键同形状（"W1" 对 "W1"……），
               每个元素是对应参数的梯度 ∂L/∂θ。

    参考行为（梯度检查，文档 §6）:
        小网络 [8,5,3]、n=10 随机数据上，解析梯度与 numerical_gradient 中心差分
        数值梯度的最大绝对差 ≈ 2e-11，相对误差 ≈ 3.6e-11——梯度写对了就是这个量级。

    实现提示:
        - 本函数只算梯度、不更新参数（更新是 update_params 的事）。train 里
          必须先算完所有梯度再统一更新，否则算 δ 时用到的 W 已被改掉。
        - 每个梯度算完顺手 assert 形状：grads["W1"].shape == params["W1"].shape。
          numpy 广播会把形状错位静默蒙混过去，assert 是最便宜的排错手段。
        - relu′ 的输入是 cache 里的 Z_l，不是 A_l（见 relu_grad 的 docstring）。
    """
    raise NotImplementedError


def update_params(params, grads, lr):
    """SGD 参数更新一步，对应文档 §5。

    公式：θ ← θ - η · ∂L/∂θ（对每个参数）

    参数:
        params: dict，当前参数。
        grads:  dict，backward 返回的梯度，与 params 同键同形状。
        lr:     float，学习率 η。

    返回:
        params: 原地更新后的同一个 dict（键不变，值已更新）。

    实现提示:
        对每个键 params[k] -= lr * grads[k]，循环三行以内。
    """
    raise NotImplementedError


def train(X, y, sizes, lr=0.5, n_epochs=200, batch_size=128, seed=42):
    """mini-batch SGD 训练循环，对应文档 §4/§5。

    每个 epoch：
        1. 用 rng.permutation(n) 打乱样本顺序；
        2. 按 batch_size 切批，逐批 forward → 算 loss → backward → update_params；
        3. 把本 epoch 内各批 loss 的均值追加进 loss_history。

    参数:
        X:          np.ndarray，形状 (n, sizes[0])，训练特征。
        y:          np.ndarray，形状 (n,)，整型标签 0 ~ K-1（函数内部转 one-hot）。
        sizes:      list[int]，网络结构，如 [64, 32, 10]。
        lr:         float，学习率，默认 0.5。
        n_epochs:   int，训练轮数，默认 200。
        batch_size: int，批大小，默认 128；batch_size >= n 即退化为 full-batch。
        seed:       int，随机种子（同时用于 init_params 和每个 epoch 的打乱）。

    返回:
        (params, loss_history):
            params:       dict，训练完成的参数；
            loss_history: np.ndarray，形状 (n_epochs,)，每 epoch 的批 loss 均值。

    参考输出（digits 训练集 1437 样本，sizes=[64,32,10]，默认参数，CPU 约 0.3 秒）:
        loss 1.9701 → 0.0060（前 5 个 epoch 均值 0.9267，后 5 个 0.0057）；
        训练准确率 1.0000，测试准确率 0.9694。
    """
    raise NotImplementedError


def predict(X, params):
    """预测类别：前向传播后取概率最大的类别，对应文档 §4。

    参数:
        X:      np.ndarray，形状 (n, sizes[0])，输入特征。
        params: dict，已训练的参数。

    返回:
        np.ndarray，形状 (n,)，整型预测标签 0 ~ K-1。

    实现提示:
        一行：forward(X, params)[0].argmax(axis=1)。
    """
    raise NotImplementedError


# ============================================================ 对答案工具（✅ 已实现完整）


def numerical_gradient(f, x, eps=1e-5):
    """中心差分数值梯度，对应文档 §6——这是给你对答案的工具，已实现完整，直接用。

    公式：∂f/∂xᵢ ≈ (f(x + eps·eᵢ) - f(x - eps·eᵢ)) / (2·eps)（逐元素扰动）

    用法（梯度检查）：写 backward 后，拿它的输出和本函数对比，
    两者 allclose（相对误差 1e-5 量级以内）说明梯度推对了。文档 §6 有完整示范。

    参数:
        f:   callable，接收与 x 同形状的 np.ndarray，返回 float 标量。
        x:   np.ndarray，任意形状，要求导的点（float 类型）。
        eps: float，扰动步长，默认 1e-5。中心差分误差 ~ O(eps²)，
             eps 太小会被浮点舍入误差淹没，1e-5 是常用甜区。

    返回:
        np.ndarray，与 x 同形状，逐元素的数值梯度。

    参考输出:
        f = lambda v: (v ** 2).sum()，x = [1.0, -2.0, 3.0] →
        数值梯度 ≈ [2.0, -4.0, 6.0]（= 2x，与解析导数一致）。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        f1 = f(x)
        x[i] = orig - eps
        f2 = f(x)
        x[i] = orig
        grad[i] = (f1 - f2) / (2 * eps)
        it.iternext()
    return grad
