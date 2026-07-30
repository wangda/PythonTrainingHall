"""
06 - 随机数生成
==============
机器学习离不开随机数：数据打乱、参数初始化、模拟采样。

运行：python 06_random.py
"""

import numpy as np

# ============================================================
# 1. 随机种子
# ============================================================
print("=" * 50)
print("1. 随机种子 — 保证可复现")
print("=" * 50)

# 设置种子后，每次运行结果一样
# randn() : 生成正态分布的随机数
np.random.seed(42)
print("seed(42):", np.random.randn(3), "\n", np.random.randn(3))

np.random.seed(42)
print("seed(42) again:", np.random.randn(3), "\n", np.random.randn(3))   # 完全一样

# 不设种子 → 每次不同
print("\nno seed:", np.random.randn(3))
print("no seed:", np.random.randn(3))           # 不同

# 最佳实践：在脚本开头设一次种子
# np.random.seed(42)  # ← 加在 import 后面
# 应该是每次randn()后，使用新生成的随机数作为种子了？


# ============================================================
# 2. 均匀分布
# ============================================================
print("\n" + "=" * 50)
print("2. 均匀分布")
print("=" * 50)

# rand — [0, 1) 均匀分布（高维）
print("rand(3):", np.random.rand(3))           # 一维
print("rand(2, 3):\n", np.random.rand(2, 3))   # 二维

# uniform — [low, high) 均匀分布
print("\nuniform(0, 10, 5):", np.random.uniform(0, 10, 5))
print("uniform(-1, 1, (2, 3)):\n", np.random.uniform(-1, 1, (2, 3)))


# ============================================================
# 3. 正态分布
# ============================================================
print("\n" + "=" * 50)
print("3. 正态分布")
print("=" * 50)

# randn — 标准正态分布（均值 0，方差 1）
print("randn(5):", np.random.randn(5))
print("randn(2, 3):\n", np.random.randn(2, 3))

# normal — 指定均值和标准差
print("\nnormal(mean=0, std=1, size=5):", np.random.normal(0, 1, 5))
print("normal(mean=5, std=2, size=5):", np.random.normal(5, 2, 5))

# 验证大数定律
np.random.seed(42)
samples = np.random.normal(0, 1, 1000000)
print(f"\n100 万样本：均值={samples.mean():.4f}，标准差={samples.std():.4f}")
print("理论值：均值=0，标准差=1")


# ============================================================
# 4. 整数随机
# ============================================================
print("\n" + "=" * 50)
print("4. 整数随机")
print("=" * 50)

# randint — [low, high) 随机整数
print("randint(0, 10, 10):", np.random.randint(0, 10, 10))
print("randint(0, 10, (3, 4)):\n", np.random.randint(0, 10, (3, 4)))

# 只给一个参数是 [0, low)
print("randint(5, size=8):", np.random.randint(5, size=8))


# ============================================================
# 5. 随机采样
# ============================================================
print("\n" + "=" * 50)
print("5. 采样与打乱")
print("=" * 50)

# choice — 从数组中随机选取
arr = np.array([10, 20, 30, 40, 50])

# 采样一个
print("choice(arr):", np.random.choice(arr))

# 采样多个（可重复）
print("choice(arr, size=3):", np.random.choice(arr, size=3))

# 不重复采样
print("choice(arr, size=3, replace=False):", np.random.choice(arr, size=3, replace=False))

# 带权重采样
weights = [0.1, 0.1, 0.1, 0.1, 0.6]   # 最后一个概率最高
samples = np.random.choice(arr, size=1000, p=weights)
print("\n带权重采样 1000 次:")
for val in arr:
    print(f"  {val}: {(samples == val).sum()} 次 (权重 {weights[list(arr).index(val)]})")


# ============================================================
# 6. 数组打乱
# ============================================================
print("\n" + "=" * 50)
print("6. shuffle / permutation")
print("=" * 50)

# shuffle — 就地打乱（修改原数组）
arr = np.arange(10)
print("打乱前:", arr)
np.random.shuffle(arr)
print("shuffle 后:", arr)

# permutation — 返回打乱后的副本（不修改原数组）
arr2 = np.arange(10)
perm = np.random.permutation(arr2)
print("\n原数组:", arr2)
print("permutation:", perm)
print("原数组不变:", arr2)

# 多维 shuffle：只打乱第一维（行）
mat = np.arange(15).reshape(5, 3)
print("\n原矩阵:\n", mat)
np.random.shuffle(mat)
print("shuffle 行后:\n", mat)  # 行的顺序变了，行内不变


# ============================================================
# 7. 其他分布
# ============================================================
print("\n" + "=" * 50)
print("7. 其他常用分布")
print("=" * 50)

# 伯努利 / 二项分布
binom = np.random.binomial(n=10, p=0.5, size=10)
print("二项分布(10次抛硬币):", binom)

# 指数分布（等待时间）
exp = np.random.exponential(scale=1.0, size=10)
print("指数分布:", exp)

# Beta 分布（常用于贝叶斯）
beta = np.random.beta(a=2, b=5, size=10)
print("Beta(2,5) 分布:", beta)

# 多项分布
multi = np.random.multinomial(n=100, pvals=[0.3, 0.5, 0.2], size=5)
print("\n多项分布 (5次试验，100个样本):\n", multi)
print("每行之和:", multi.sum(axis=1))  # 都是 100


# ============================================================
# 8. 实际应用：数据集划分
# ============================================================
print("\n" + "=" * 50)
print("8. 实战：训练/测试集划分")
print("=" * 50)

np.random.seed(42)
N = 1000
X = np.random.randn(N, 10)       # 1000 个样本，10 个特征
y = np.random.randint(0, 2, N)   # 二分类标签

# 8:2 划分
indices = np.random.permutation(N)
split = int(N * 0.8)

train_idx = indices[:split]
test_idx = indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"总样本: {N}")
print(f"训练集: {len(X_train)} ({len(X_train)/N*100:.0f}%)")
print(f"测试集: {len(X_test)} ({len(X_test)/N*100:.0f}%)")
print(f"训练集标签分布: {np.bincount(y_train)}")
print(f"测试集标签分布: {np.bincount(y_test)}")


# ============================================================
# ★ 动手练习
# ============================================================
# ============================================================
# 9. 动手练习
# ============================================================
print("\n" + "#" * 50)
print("# 动手练习")
print("#" * 50)

# 练习 1：蒙特卡洛估计 π
# 在单位正方形内随机撒点，统计落在单位圆内的比例
# 提示：x = uniform(-1, 1), y = uniform(-1, 1)
# 判断 x² + y² <= 1
# 尝试 N = 1000, 10000, 100000 看看精度变化
# ↓ 你来实现
np.random.seed(42)
N = 100000000
x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)
hit = x**2 + y**2 <= 1
print("Pi：", (np.sum(hit) / N) * 4)

# 练习 2：模拟掷骰子
# 模拟掷一个 6 面骰子 10000 次，统计各面出现频率
# 理论概率都是 1/6 ≈ 0.1667
# ↓ 你来实现
np.random.seed(42)
dice = np.random.randint(1, 7, 10000)
print(np.bincount(dice) / 10000)

# 练习 3：K-Fold 交叉验证索引生成
# 给定 N=100, K=5，生成 5 组 (train_idx, val_idx)
# 每组验证集大小 = 20
# 提示：用 shuffle + reshape
# ↓ 你来实现
N, K = 100, 5
indices = np.random.permutation(N).reshape(K, -1)

result = []
for i in range(K):
    val_idx = indices[i]
    train_idx = np.concatenate([indices[j] for j in range(K) if j != i])
    result.append((train_idx, val_idx))

print( result)

# 练习 4：模拟随机游走
# 初始位置 0，每一步 +1 或 -1 概率各 50%
# 走 1000 步，记录路径
# 最后离原点多远？
# ↓ 你来实现
distance = 0
for i in range(1000):
    step: int = np.random.choice([-1, 1])
    distance += step
print(f"最后距离原点距离：{distance}")