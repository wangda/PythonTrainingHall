"""手写决策树的测试：不纯度/增益对照精确手算值，树的行为对照已知结构，最终与 sklearn 对答案。

测试数据与前置文档同源：generate_sample_data(seed=42)（文档 §1 考试例子）、
generate_circle_data(seed=0)（文档 §1 同心圆例子）。
运行：在本目录执行 `../../../.venv/bin/python -m pytest test_decision_tree.py -v`

骨架状态下：数据生成器 2 条测试通过，其余以 NotImplementedError 失败——
失败信息就是你要实现的东西。
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from decision_tree import (
    DecisionTree,
    best_split,
    entropy,
    generate_circle_data,
    generate_sample_data,
    gini,
    impurity,
    split_gain,
)

# ---------------------------------------------------------------- 数据 fixture


@pytest.fixture(scope="module")
def exam_data():
    """文档 §1 的贯穿例子"考试通过预测"（与模块 2 同一份数据，seed=42）。"""
    return generate_sample_data(seed=42)


@pytest.fixture(scope="module")
def circle_data():
    """文档 §1 的同心圆数据（seed=0，可复现）。"""
    return generate_circle_data(seed=0)


@pytest.fixture
def and_data():
    """AND 玩具数据：4 个角点，y = (x0==1 且 x1==1)。贪心树两步可完美分开，
    且每步的分裂结果都是确定性的（平局一律取先发现的特征 / 较小类别号）。"""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    return X, y


# ---------------------------------------------------------------- 数据生成器


def test_数据生成器_形状与可复现性():
    X, y = generate_sample_data(seed=42)
    assert X.shape == (200, 2) and y.shape == (200,)
    X2, y2 = generate_sample_data(seed=42)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert set(np.unique(y)) == {0, 1}


def test_同心圆数据_半径与类别严格对应(circle_data):
    X, y = circle_data
    assert X.shape == (400, 2) and y.shape == (400,)
    r = np.sqrt((X ** 2).sum(axis=1))
    assert (r[y == 0] <= 1.0).all()          # 类 0 全部在内圆
    assert (r[y == 1] >= 1.5).all()          # 类 1 全部在外环——两类之间有一圈真空带
    X2, y2 = generate_circle_data(seed=0)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)


# ---------------------------------------------------------------- 熵（文档 §2）


def test_熵_纯节点为零():
    assert entropy(np.array([1, 1, 1, 1])) == pytest.approx(0.0, abs=1e-12)
    assert entropy(np.array([0])) == pytest.approx(0.0, abs=1e-12)


def test_熵_五五开为一比特():
    assert entropy(np.array([0, 0, 1, 1])) == pytest.approx(1.0, abs=1e-12)


def test_熵_已知值():
    # 文档 §2.2 手算：-(0.25·log2 0.25 + 0.75·log2 0.75) = 0.8112781...
    assert entropy(np.array([0, 0, 0, 1])) == pytest.approx(0.8112781, abs=1e-6)


def test_熵_只和比例有关与样本量无关():
    a = entropy(np.array([0, 0, 1, 1]))          # 4 人五五开
    b = entropy(np.array([0] * 400 + [1] * 400))  # 800 人五五开
    assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------- 基尼（文档 §3）


def test_基尼_纯节点为零():
    assert gini(np.array([0, 0, 0])) == pytest.approx(0.0, abs=1e-12)


def test_基尼_五五开为半():
    assert gini(np.array([0, 0, 1, 1])) == pytest.approx(0.5, abs=1e-12)


def test_基尼_已知值():
    # 文档 §3.1 手算：1 - (0.25² + 0.75²) = 0.375
    assert gini(np.array([0, 0, 0, 1])) == pytest.approx(0.375, abs=1e-12)


# ---------------------------------------------------------------- 调度与增益


def test_不纯度调度器_与两个实现一致():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    assert impurity(y, "gini") == pytest.approx(gini(y))
    assert impurity(y, "entropy") == pytest.approx(entropy(y))
    with pytest.raises(ValueError):
        impurity(y, "mse")  # 未知标准必须报错，不能静默


def test_分裂增益_熵_文档手算例子():
    # 文档 §2.3：8 人 5 过 3 挂，按出勤率 0.8 切 → 增益 0.5488 bit
    y = np.array([1] * 5 + [0] * 3)
    y_left = np.array([1, 0, 0, 0])
    y_right = np.array([1, 1, 1, 1])
    assert split_gain(y, y_left, y_right, "entropy") == pytest.approx(0.5487949, abs=1e-6)


def test_分裂增益_基尼_文档手算例子():
    # 文档 §3.2：同一分裂换基尼 → 增益 0.28125
    y = np.array([1] * 5 + [0] * 3)
    y_left = np.array([1, 0, 0, 0])
    y_right = np.array([1, 1, 1, 1])
    assert split_gain(y, y_left, y_right, "gini") == pytest.approx(0.28125, abs=1e-12)


def test_分裂增益_默认参数是基尼():
    y = np.array([1] * 5 + [0] * 3)
    y_left = np.array([1, 0, 0, 0])
    y_right = np.array([1, 1, 1, 1])
    assert split_gain(y, y_left, y_right) == pytest.approx(
        split_gain(y, y_left, y_right, "gini"))


# ---------------------------------------------------------------- best_split（文档 §4.2）


def test_best_split_四点例子_阈值是中点且增益满分():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    feature, threshold, gain = best_split(X, y, criterion="entropy")
    assert feature == 0
    assert threshold == pytest.approx(2.5)     # 相邻值 2 和 3 的中点
    assert gain == pytest.approx(1.0)          # 一刀下去两边全纯
    _, _, gain_gini = best_split(X, y, criterion="gini")
    assert gain_gini == pytest.approx(0.5)     # 父 gini 0.5，子全纯


def test_best_split_纯节点返回None():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 1, 1])                  # 没有正增益的一刀
    assert best_split(X, y) is None


def test_best_split_叶子样本数约束():
    # y = [0,1,1,0]：唯一有正增益的分裂是切掉端点（子节点 1:3），
    # min_samples_leaf=2 后所有候选要么违规要么零增益 → None
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 1, 0])
    assert best_split(X, y, min_samples_leaf=2) is None
    feature, threshold, gain = best_split(X, y, min_samples_leaf=1)
    assert threshold == pytest.approx(1.5)     # 平局（1.5 与 3.5 同增益）保留先发现的
    assert gain == pytest.approx(1 / 6, abs=1e-12)  # 0.5 − 3/4·(4/9)


def test_best_split_跳过相邻相同值():
    # 特征值 [1,1,2]：1 和 1 之间没有有效阈值，合法阈值只有 1.5
    X = np.array([[1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])
    feature, threshold, gain = best_split(X, y)
    assert threshold == pytest.approx(1.5)
    # 父 gini = 1 − ((2/3)² + (1/3)²) = 4/9，左右子节点全纯 → 增益 = 4/9
    assert gain == pytest.approx(4 / 9, abs=1e-12)


# ---------------------------------------------------------------- DecisionTree（文档 §4/§5）


def test_树_AND数据_完美拟合(and_data):
    X, y = and_data
    tree = DecisionTree().fit(X, y)
    assert np.array_equal(tree.predict(X), y)    # 训练准确率 100%
    assert tree.depth_ == 2                       # 根切一刀（depth 1），右侧再切一刀（depth 2）
    assert tree.n_leaves_ == 3                    # 左纯叶 + 右侧两个纯叶


def test_树_根分裂用第一个特征(and_data):
    # AND 数据上 x0 和 x1 的首刀增益相同（0.125 gini），严格大于的更新规则
    # 保证平局时选先发现的特征 0——这个测试把"确定性"钉死
    X, y = and_data
    tree = DecisionTree(max_depth=1).fit(X, y)
    assert tree._tree["feature"] == 0
    assert tree._tree["threshold"] == pytest.approx(0.5)


def test_树_最大深度限制(and_data):
    X, y = and_data
    tree = DecisionTree(max_depth=1).fit(X, y)
    assert tree.depth_ == 1
    assert tree.n_leaves_ == 2
    # 右叶 [0,1] 平局 → 取较小类别号 0 → 全部预测 0 → 准确率 3/4
    assert (tree.predict(X) == y).mean() == pytest.approx(0.75)


def test_树_同心圆数据_不剪枝背下训练集(circle_data):
    X, y = circle_data
    tree = DecisionTree().fit(X, y)
    train_acc = (tree.predict(X) == y).mean()
    assert train_acc >= 0.99                      # 文档 §5.1：不剪枝的树≈背下训练集


def test_树_同心圆数据_剪枝限制容量(circle_data):
    X, y = circle_data
    shallow = DecisionTree(max_depth=2).fit(X, y)
    deep = DecisionTree().fit(X, y)
    assert shallow.depth_ <= 2
    assert shallow.n_leaves_ < deep.n_leaves_
    # 深度 2 只能切一个矩形框，逼近不了圆环 → 训练准确率显著低于满分
    assert (shallow.predict(X) == y).mean() < 0.95


def test_树_最小增益门槛_设得巨大则拒绝任何分裂(circle_data):
    X, y = circle_data
    tree = DecisionTree(min_impurity_decrease=1.0).fit(X, y)
    assert tree.n_leaves_ == 1 and tree.depth_ == 0   # 树根直接变叶子
    # 同心圆 200/200 平局 → 多数类取 0 → 训练准确率恰好 0.5
    assert (tree.predict(X) == y).mean() == pytest.approx(0.5)


def test_树_min_samples_split_过小节点不再分(and_data):
    X, y = and_data
    tree = DecisionTree(min_samples_split=4).fit(X, y)   # 根 4 个样本能切，子节点 2 个不能
    assert tree.n_leaves_ == 2
    assert tree.depth_ == 1


def test_树_predict_输出形状与取值(and_data, circle_data):
    X, y = and_data
    tree = DecisionTree().fit(X, y)
    pred = tree.predict(circle_data[0])
    assert pred.shape == (400,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_树_predict_proba_纯叶子输出onehot(and_data):
    X, y = and_data
    tree = DecisionTree().fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # 不剪枝的树叶全纯 → 每行都是 one-hot（过拟合的树过度自信的直接证据，文档 §5.1）
    assert ((proba == 0.0) | (proba == 1.0)).all()


def test_树_与sklearn对答案(circle_data):
    """同数据、同剪枝参数：手写树的测试准确率应与 sklearn 接近（不要求逐节点相同，
    平局打破与浮点细节不同，结构允许差异——文档 §7）。"""
    X, y = circle_data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    ours = DecisionTree(max_depth=5).fit(X_tr, y_tr)
    sk = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_tr, y_tr)
    acc_ours = (ours.predict(X_te) == y_te).mean()
    acc_sk = sk.score(X_te, y_te)
    assert acc_ours >= 0.85                       # 同心圆 + 深度 5，两家都应该轻松过 85%
    assert abs(acc_ours - acc_sk) <= 0.1


def test_树_考试数据_学出合理边界(exam_data):
    """与模块 2 同一份数据：树应达到与逻辑回归（0.83~0.84）同一量级的准确率。"""
    X, y = exam_data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    tree = DecisionTree(max_depth=5).fit(X_tr, y_tr)
    acc = (tree.predict(X_te) == y_te).mean()
    assert acc >= 0.70    # 标签本身按概率抽样（有运气成分），上限摆在那里
