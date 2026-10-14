"""决策树手写练习骨架。

仅用 NumPy 实现：熵、基尼系数、分裂增益、最佳分裂穷举、递归构建决策树
（含 predict / predict_proba 与四个预剪枝旋钮）。
所有公式与前置文档《05-决策树与集成学习前置学习文档.md》一一对应，docstring 中标注了章节。

使用流程：
    X, y = generate_circle_data()                      # 文档 §1 的同心圆数据
    tree = DecisionTree(max_depth=5).fit(X, y)         # 文档 §4 / §5
    y_pred = tree.predict(X)
    print((y_pred == y).mean())                        # 训练准确率
    # 和 sklearn 对答案：
    from sklearn.tree import DecisionTreeClassifier
    sk = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y)
    print(sk.score(X, y))                              # 应与你的准确率接近（文档 §7）
"""

import numpy as np


def generate_sample_data(seed=42):
    """生成文档 §1 的贯穿例子"考试通过预测"数据（已实现完整，直接可用）。

    与模块 2（逻辑回归）的练习数据**完全相同**——同一份数据，模块 2 用线性边界学，
    本模块用树学，方便直接对比两种模型的行为差异。

    真实规律：logit = 0.9 * hours + 10 * attendance - 11，
    通过概率 p = σ(logit)，真实结果按 p 抽样（0=挂科, 1=通过）。

    参数:
        seed: int，随机种子，固定后结果可复现（文档默认 42）。

    返回:
        (X, y):
            X: np.ndarray，形状 (200, 2)，第 0 列每周学习时长（0~10），第 1 列出勤率（0.5~1.0）；
            y: np.ndarray，形状 (200,)，取值 0 / 1。
    """
    rng = np.random.default_rng(seed)
    n = 200
    hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(0.5, 1.0, n)
    logit = 0.9 * hours + 10 * attendance - 11
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    X = np.column_stack([hours, attendance])
    return X, y


def generate_circle_data(seed=0, n=400):
    """生成同心圆数据：内圆（半径 ≤1，类 0）+ 外环（半径 1.5~2.5，类 1）（已实现完整）。

    对应文档 §1：线性分类器在这份数据上结构性无解（边界只能是直线），
    决策树用矩形切割轻松逼近圆形边界——这是树的主场。

    参数:
        seed: int，随机种子，固定后结果可复现（文档默认 0）。
        n:    int，总样本数（内圆 ⌊n/2⌋ 个，外环其余），默认 400。

    返回:
        (X, y):
            X: np.ndarray，形状 (n, 2)，平面坐标；
            y: np.ndarray，形状 (n,)，整型标签 0（内圆）/ 1（外环）。
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    r0 = rng.uniform(0.0, 1.0, n0)
    a0 = rng.uniform(0, 2 * np.pi, n0)
    X0 = np.column_stack([r0 * np.cos(a0), r0 * np.sin(a0)])
    n1 = n - n0
    r1 = rng.uniform(1.5, 2.5, n1)
    a1 = rng.uniform(0, 2 * np.pi, n1)
    X1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    return X, y


def entropy(y):
    """信息熵（单位 bit），对应文档 §2.2。

    公式：H = − Σₖ pₖ · log₂ pₖ，其中 pₖ 是类别 k 在节点内的占比。

    参数:
        y: np.ndarray，形状 (n,)，某个节点内的标签（非空）。

    返回:
        float，标量，值域 [0, 1]（二分类时）。

    参考输出（全部为精确值）:
        纯节点（如 [1,1,1]）→ 0.0；
        五五开（[0,0,1,1]）→ 1.0；
        [0,0,0,1]（占比 0.75/0.25）→ 0.8112781...

    实现提示:
        - 用 np.unique(y, return_counts=True) 数各类别个数——它只数**出现过**的类，
          p 恒大于 0，天然规避了 log₂(0)；
        - 熵只和比例有关、和样本数无关：8 人五五开和 800 人五五开都是 1 bit。
    """
    raise NotImplementedError


def gini(y):
    """基尼不纯度，对应文档 §3。

    公式：Gini = 1 − Σₖ pₖ²
    概率解读：从节点里（放回地）随机抽两个样本，类别不同的概率。

    参数:
        y: np.ndarray，形状 (n,)，某个节点内的标签（非空）。

    返回:
        float，标量，值域 [0, 0.5]（二分类时）。

    参考输出（全部为精确值）:
        纯节点 → 0.0；五五开 → 0.5；[0,0,0,1] → 0.375。
    """
    raise NotImplementedError


def impurity(y, criterion="gini"):
    """不纯度调度器：按 criterion 分派到 entropy / gini，对应文档 §2/§3。

    参数:
        y:         np.ndarray，形状 (n,)，节点内标签。
        criterion: str，"gini" 或 "entropy"（sklearn 同款参数名）。

    返回:
        float，标量，不纯度。

    异常:
        ValueError: criterion 不是 "gini" / "entropy"。
    """
    raise NotImplementedError


def split_gain(y, y_left, y_right, criterion="gini"):
    """一次二分裂的增益 = 父不纯度 − 加权子不纯度，对应文档 §2.3。

    公式：gain = I(父) − [ (n左/n)·I(左) + (n右/n)·I(右) ]，I 为 impurity。

    参数:
        y:         np.ndarray，形状 (n,)，父节点标签。
        y_left:    np.ndarray，形状 (n左,)，左子节点标签（非空）。
        y_right:   np.ndarray，形状 (n右,)，右子节点标签（非空）。
        criterion: str，"gini" 或 "entropy"。

    返回:
        float，标量，分裂增益（恒 ≥ 0，数学上不纯度的凹性保证）。

    参考输出（文档 §2.3 / §3.2 的手算例子，y=[1]*5+[0]*3 按"出勤率 0.8"切）:
        y_left=[1,0,0,0], y_right=[1,1,1,1]:
        criterion="entropy" → 0.5487949...
        criterion="gini"    → 0.28125

    实现提示:
        复用你的 impurity()，注意权重是样本数占比，不是 (0.5, 0.5)。
    """
    raise NotImplementedError


def best_split(X, y, criterion="gini", min_samples_leaf=1):
    """穷举所有（特征, 阈值）候选，返回增益最大的一刀，对应文档 §4.2。

    候选阈值的取法：对每个特征，把样本按该特征排序，取**相邻不同值的中点**——
    阈值在两个相邻样本值之间滑动时分裂结果不变，所以只有这些中点值得试。

    分裂规则约定（与 sklearn 一致）：X[:, j] <= threshold → 左子节点，否则 → 右子节点。

    参数:
        X:                 np.ndarray，形状 (n, d)，特征矩阵。
        y:                 np.ndarray，形状 (n,)，标签。
        criterion:         str，"gini" 或 "entropy"。
        min_samples_leaf:  int，预剪枝约束——分裂后每个子节点至少这么多样本，
                           不满足的候选直接跳过（文档 §5.2）。

    返回:
        (feature, threshold, gain):
            feature:   int，最佳分裂的特征下标；
            threshold: float，最佳阈值（某对相邻值的中点）；
            gain:      float，该分裂的增益。
        若不存在增益严格大于 0 且满足 min_samples_leaf 的分裂，返回 None。

    参考输出（文档 §4.2 的四点例子，X=[[1],[2],[3],[4]], y=[0,0,1,1]）:
        criterion="entropy" → (0, 2.5, 1.0)：一刀下去两边全纯，增益满分 1 bit；
        criterion="gini"    → (0, 2.5, 0.5)。

    实现提示:
        - 排序用 np.argsort(X[:, j])，遍历 i 时跳过 x_sorted[i] == x_sorted[i+1]
          （相邻值相同，中间没有有效阈值）；
        - 最佳记录用"严格大于"更新（gain > best_gain），平局时保留先发现的候选，
          保证结果确定性（测试依赖这一点）；
        - 初始 best_gain 设为 0.0，即"没有正收益就不切"（文档 §4.4 的 XOR 讨论）；
        - 别忘了用 split_gain 之前先检查两边样本数是否都 >= min_samples_leaf。
    """
    raise NotImplementedError


class DecisionTree:
    """手写决策树分类器，对应文档 §4（构建）与 §5（预剪枝）。

    节点用嵌套 dict 表示（约定如下，predict 时要按这个结构递归下行）:

    叶子节点:
        {"is_leaf": True,
         "prediction": int,        # 叶内多数类（平局取较小类别号）
         "proba": np.ndarray,      # 叶内各类别占比，形状 (n_classes,)，predict_proba 用
         "n_samples": int}

    内部节点:
        {"is_leaf": False,
         "feature": int,           # 分裂特征下标
         "threshold": float,       # 分裂阈值：X[:, feature] <= threshold → 左
         "gain": float,            # 该分裂的增益
         "left": dict,             # 左子树（嵌套节点）
         "right": dict}            # 右子树

    属性（fit 后可用，测试会检查）:
        _tree:      dict，根节点；
        depth_:     int，树的实际深度（根深度为 0，纯叶子的树深度为 0）；
        n_leaves_:  int，叶子数量；
        classes_:   np.ndarray，训练集里出现过的类别（升序）。
    """

    def __init__(self, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0):
        """初始化超参数（与 sklearn 同名同义，文档 §5.2）。

        参数:
            criterion:              str，"gini" 或 "entropy"，分裂标准。
            max_depth:              int 或 None，最大深度（根为 0）；None = 不限。
            min_samples_split:      int，节点样本数达到多少才允许再分裂，默认 2。
            min_samples_leaf:       int，每个叶子至少多少样本，默认 1。
            min_impurity_decrease:  float，增益至少达到多少才允许分裂，默认 0.0。
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        """递归构建决策树（贪心：每节点选当前增益最大的一刀，文档 §4.1）。

        参数:
            X: np.ndarray，形状 (n, d)，特征矩阵。
            y: np.ndarray，形状 (n,)，整型标签（0 / 1 或 0..K-1）。

        返回:
            self（sklearn 惯例，支持链式 DecisionTree(...).fit(X, y)）。

        停止条件（文档 §4.4 / §5.2，满足任一即做成叶子）:
            1. 节点纯净（只剩一类）；
            2. 深度达到 max_depth；
            3. 样本数 < min_samples_split；
            4. best_split 返回 None（无正增益 / 不满足 min_samples_leaf）；
            5. 最佳增益 < min_impurity_decrease。

        实现提示:
            - 写成 self._build(X, y, depth) 递归辅助函数，返回节点 dict；
            - 叶子预测用 np.argmax(np.bincount(y))——平局时自动取较小类别号，
              保证确定性（测试依赖这一点）；
            - 叶子的 proba 用 np.bincount(y, minlength=len(self.classes_)) / len(y)；
            - 递归时左右子集用布尔掩码 X[:, j] <= t 划分；
            - fit 末尾别忘了统计 self.depth_ 和 self.n_leaves_
              （可以在 _build 递归返回时顺带返回深度/叶子数，或建完树再遍历一遍）。
        """
        raise NotImplementedError

    def predict(self, X):
        """对每个样本预测类别：从根到叶递归下行，返回叶子 prediction，对应文档 §4.3。

        参数:
            X: np.ndarray，形状 (n, d)，特征矩阵（d 必须与训练时一致）。

        返回:
            np.ndarray，形状 (n,)，整型预测标签。

        实现提示:
            写 self._predict_one(x, node) 递归：内部节点按
            x[node["feature"]] <= node["threshold"] 决定走左还是走右，
            到叶子返回 node["prediction"]；外层对每行调用。
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """对每个样本输出各类别概率 = 所落叶子的类别占比，对应文档 §4.3。

        参数:
            X: np.ndarray，形状 (n, d)，特征矩阵。

        返回:
            np.ndarray，形状 (n, n_classes)，每行和为 1；
            列顺序与 self.classes_ 一致。

        参考行为:
            纯叶子输出的概率是 one-hot（如 [0.0, 1.0]）——不剪枝的树会把
            训练集切到全纯，predict_proba 全是 0/1，这是"过拟合的树过度自信"
            的直接证据（文档 §5.1）。
        """
        raise NotImplementedError
