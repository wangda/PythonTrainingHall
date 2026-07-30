"""
07 - 综合练习
=============
把前面学的内容串起来，做几个有实际意义的练习。

参考答案在 07_solutions.py（先动手做，实在卡住再看）。

运行：python 07_exercises.py
"""

import numpy as np

print("=" * 50)
print("综合练习 — 先自己动手，卡住了再参考答案")
print("=" * 50)


# ============================================================
# 练习 1：手写 KMeans 聚类（仅用 NumPy）
# ============================================================
print("\n" + "-" * 50)
print("练习 1：KMeans 聚类")
print("-" * 50)
print("""
KMeans 算法：
  1. 随机初始化 K 个中心点
  2. 重复直到收敛：
     a. 每个点分配到最近的中心
     b. 更新中心为所属点的均值
""")

def kmeans(X, K, max_iters=100, tol=1e-4, seed=42):
    """
    手动实现 KMeans 聚类

    参数:
      X: shape (N, D) — N 个 D 维数据点
      K: 聚类数
      max_iters: 最大迭代次数
      tol: 中心点变化小于此值则停止

    返回:
      centroids: shape (K, D) — 最终中心点
      labels: shape (N,) — 每个点的聚类分配
    """
    np.random.seed(seed)

    # 1. 从数据中随机选 K 个点作为初始中心
    # ↓ 你的代码（提示：用 random.choice）
    # centroids = ...

    # 2. 迭代
    for i in range(max_iters):
        # 2a. 计算每个点到各中心的距离（欧几里得）
        #     利用广播：(N,1,D) - (K,D) → (N,K,D) → 平方和 → (N,K)
        # ↓ 你的代码
        # distances = ...

        # 2b. 分配最近的中心
        # ↓ 你的代码（用 argmin）
        # labels = ...

        # 2c. 更新中心：每个簇的均值
        #     注意：可能有的簇没有点
        # ↓ 你的代码
        # new_centroids = ...

        # 2d. 检查收敛
        # diff = np.linalg.norm(new_centroids - centroids)
        # if diff < tol:
        #     break
        # centroids = new_centroids
        pass

    # ↓ 返回最终结果
    # return centroids, labels

# 生成测试数据：三个明显分离的簇
np.random.seed(42)
cluster1 = np.random.randn(50, 2) + np.array([0, 0])
cluster2 = np.random.randn(50, 2) + np.array([5, 0])
cluster3 = np.random.randn(50, 2) + np.array([2.5, 5])
X = np.vstack([cluster1, cluster2, cluster3])

print("数据形状:", X.shape)
print("包含 3 个簇，每个 50 个点")

# 取消下面注释来测试你的实现
# centroids, labels = kmeans(X, K=3)
# print("中心点:\n", centroids)
# print("标签分布:", np.bincount(labels))


# ============================================================
# 练习 2：图像作为矩阵操作
# ============================================================
print("\n" + "-" * 50)
print("练习 2：图像处理（矩阵操作）")
print("-" * 50)

# 生成一个简单的"图像"（28x28 的随机矩阵模拟手写数字）
np.random.seed(42)
image = np.random.rand(28, 28)

# 2a. 将图像二值化（大于均值的置 1，否则置 0）
# ↓ 你的代码（提示：广播 + 布尔索引）
# binary = ...

# 2b. 对图像做 2x2 池化（取每个 2x2 块的最大值）
#     28x28 → 14x14
# 提示：reshape 成 (14, 2, 14, 2)，然后在轴 1 和轴 3 上取 max
# ↓ 你的代码
# pooled = ...

# 2c. Sobel 边缘检测（简化版）
# 水平梯度：Gx = image[:-2, 1:-1] - image[2:, 1:-1]
# 垂直梯度：Gy = image[1:-1, :-2] - image[1:-1, 2:]
# 梯度幅值：G = sqrt(Gx² + Gy²)
# ↓ 你的代码
# edges = ...


# ============================================================
# 练习 3：多元线性回归（正规方程 + 梯度下降两种实现）
# ============================================================
print("\n" + "-" * 50)
print("练习 3：线性回归")
print("-" * 50)

# 生成数据：y = 3*x1 - 2*x2 + 1.5*x3 + noise
np.random.seed(42)
N = 200
X = np.random.randn(N, 3)
true_w = np.array([3.0, -2.0, 1.5])
true_b = 0.5
y = X @ true_w + true_b + 0.2 * np.random.randn(N)

# 3a. 正规方程
# 添加偏置项：X_bias = [ones(N, 1), X]
# θ = (X^T X)^{-1} X^T y
# ↓ 你的代码
# X_bias = ...
# theta = ...
# w_hat, b_hat = ...
# print("真实 w:", true_w, "估计 w:", w_hat)
# print("真实 b:", true_b, "估计 b:", b_hat)

# 3b. 梯度下降（不用框架）
def linear_regression_gd(X, y, lr=0.01, epochs=100):
    """
    批量梯度下降求解线性回归
    """
    N, D = X.shape
    # 添加偏置
    X_b = np.c_[np.ones(N), X]   # shape (N, D+1)

    # 初始化参数
    # ↓ 你的代码
    # theta = ...

    # 训练循环
    for epoch in range(epochs):
        # 预测
        # y_pred = ...

        # 计算损失（MSE）
        # loss = ...

        # 计算梯度
        # grad = ...

        # 更新参数
        # theta -= lr * grad

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {loss:.6f}")

    return theta

# 取消注释测试
# theta = linear_regression_gd(X, y)
# print("梯度下降结果:", theta)


# ============================================================
# 练习 4：PCA 降维
# ============================================================
print("\n" + "-" * 50)
print("练习 4：PCA 降维")
print("-" * 50)

def pca(X, n_components):
    """
    主成分分析（PCA）

    参数:
      X: shape (N, D)
      n_components: 降维目标维度

    步骤：
      1. 数据中心化（减均值）
      2. 计算协方差矩阵
      3. 特征值分解
      4. 取前 k 个特征向量
      5. 投影

    返回:
      X_reduced: shape (N, n_components) — 降维后的数据
      explained_ratio: 每个主成分解释的方差比例
    """
    # ↓ 你的代码
    # 1. 中心化
    # ...

    # 2. 协方差矩阵
    # ...

    # 3. 特征值分解 (np.linalg.eig)
    # ...

    # 4. 按特征值排序，取前 k 个
    # ...

    # 5. 投影
    # ...

    # return X_reduced, explained_ratio

np.random.seed(42)
# 生成 100 个三维数据，但实际内在维度只有 2
X_high = np.random.randn(100, 3)
# 让第三维和前两维高度相关
X_high[:, 2] = 0.5 * X_high[:, 0] + 0.3 * X_high[:, 1] + 0.1 * np.random.randn(100)

# 用 PCA 降到 2 维，看方差保留了多少
# X_reduced, ratio = pca(X_high, 2)
# print("解释方差比例:", ratio)
# print("累计解释方差:", ratio.sum())


# ============================================================
# 练习 5：交叉验证
# ============================================================
print("\n" + "-" * 50)
print("练习 5：K-Fold 交叉验证")
print("-" * 50)

def kfold_split(N, K=5, shuffle=True, seed=42):
    """
    生成 K-Fold 交叉验证的 train/val 索引

    参数:
      N: 样本数
      K: 折数
      shuffle: 是否先打乱

    返回:
      list of (train_idx, val_idx) tuples，长度为 K
    """
    # ↓ 你的代码
    # 提示：indices → shuffle(可选的) → reshape(K, N//K) → 每折拿一块当 val
    # ...

    pass

# 验证：K 折索引不重叠，合并后等于全集
# splits = kfold_split(100, K=5)
# for i, (train_idx, val_idx) in enumerate(splits):
#     print(f"Fold {i}: train={len(train_idx)}, val={len(val_idx)}")
#     assert len(set(train_idx) & set(val_idx)) == 0, "overlap!"
# print("所有折验证通过 OK")


# ============================================================
# ★ 挑战题（可选）
# ============================================================
print("\n" + "#" * 50)
print("# 挑战题（限 NumPy，不能导入其他库）")
print("#" * 50)

# 挑战 1：手写 Softmax 回归（多分类逻辑回归）
# 挑战 2：用 NumPy 实现简单的三层全连接神经网络（前向 + 反向传播）
# 挑战 3：手写 t-SNE 的核心步骤（计算高维空间的联合概率）
# 挑战 4：用 NumPy 实现卷积操作（im2col 方法）
