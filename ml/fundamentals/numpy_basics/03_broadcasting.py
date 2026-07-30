"""
03 - 广播机制（Broadcasting）
=============================
广播是 NumPy 最强大也最容易踩坑的特性。
理解它 = 告别 for 循环。

核心规则：从尾部维度开始比较，维度大小要相同或其中一个为 1。

运行：python 03_broadcasting.py
"""

import numpy as np

# ============================================================
# 1. 标量 + 数组
# ============================================================
print("=" * 50)
print("1. 标量与数组的运算")
print("=" * 50)

# arr = np.array([1, 2, 3, 4])
# print(f"{arr} + 10 = {arr + 10}")
# print(f"{arr} * 2  = {arr * 2}")
# print(f"{arr} ** 2 = {arr ** 2}")
# print(f"10 - {arr} = {10 - arr}")   # 广播对调也成立
arr: np.ndarray = np.array([1, 2, 3, 4])
print(f"arr = {arr}")
print(f"arr + 10 = {arr + 10}")
print(f"arr * 2 = {arr * 2}")
print(f"arr ** 2 = {arr ** 2}")
print(f"10 - arr = {10 - arr}")

arr2: np.ndarray = np.arange(15, 19)
print(f"arr2 = {arr2}")
print(f"arr + arr2 = {arr + arr2}")

try:
    arr2: np.ndarray = np.arange(15, 25)
    print(f"{arr} + {arr2} = {arr + arr2}")
except Exception as e:
    print("报错：", e)

# 二维也适用
# mat = np.ones((3, 3))
# print("\nmat:\n", mat)
# print("mat + 5:\n", mat + 5)
mat = np.ones((3, 3))
print(f"mat = {mat}")
print(f"\n mat + 5= \n {mat + 5}")


# ============================================================
# 2. 一维 + 二维（最典型场景）
# ============================================================
print("\n" + "=" * 50)
print("2. 一维数组 + 二维矩阵")
print("=" * 50)

# 每一行加同一个向量
# mat = np.arange(12).reshape(3, 4)
# row = np.array([10, 20, 30, 40])
# print("mat:\n", mat)
# print("row:", row)
# print("mat + row:\n", mat + row)
mat: np.ndarray = np.arange(12).reshape(3, 4)
row: np.ndarray[int] = np.array([10, 20, 30 , 40])
print("mat:\n", mat)
print("row: ", row)
print(f"mat + row = {mat + row}")


# 每一列加同一个向量 — 需要 shape 匹配
# col = np.array([100, 200, 300])      # shape (3,)
# print("\ncol:", col, "shape:", col.shape)
col: np.ndarray[int] = np.array([100, 200, 300])
print(f"\ncol: {col}, shape: {col.shape}")

# mat + col 会怎样？→ shape (3,4) vs (3,) → 尾部对齐比较
# (3, 4) vs (3,) → (3, 4) vs (1, 3)？不对，(3,) 对齐到 4 和 3 比较...
# 实际上是：(3, 4) vs (3,) → 比较 (4 vs 3) → 不匹配且无 1 → 报错
# try:
#     result = mat + col
#     print("mat + col:", result)
# except ValueError as e:
#     print("mat + col 报错:", e)
try:
    result = mat + col
    print("mat + col = ", (mat + col))
except ValueError as e:
    print("mat + col 错误：", e)


# 正确的做法：reshape 为 (3, 1) → (3,4) vs (3,1) → OK
# col_reshaped = col.reshape(-1, 1)     # shape (3, 1)
# print("\ncol.reshape(-1, 1):\n", col_reshaped, "shape:", col_reshaped.shape)
# print("mat + col_reshaped:\n", mat + col_reshaped)
col_reshaped: np.ndarray[int] = col.reshape(-1, 1)
print("\ncol.reshape(-1, 1):\n", col_reshaped, "shape:", col_reshaped.shape)
print(f"mat + col_reshaped = {mat + col_reshaped}")


# ============================================================
# 3. 广播规则详解
# ============================================================
print("\n" + "=" * 50)
print("3. 广播规则")
print("=" * 50)

# """
# 规则：从右往左比较两个数组的 shape，逐个维度检查：
#   - 维度大小相同 → 没问题
#   - 其中一个为 1 → 广播扩展到这个维度大小
#   - 都不满足 → 报错 ValueError
#
# 例子：
#   shape A:      (3, 4, 5)
#   shape B:         (4, 5)
#   对齐:        (3, 4, 5)
#               (   4, 5)  → B 自动补成 (1, 4, 5)
#   比较:         维2: 3 vs 1  → OK, 广播 B 的 3 份
#                维1: 4 vs 4  → OK
#                维0: 5 vs 5  → OK
#   结果: (3, 4, 5)
# """
#
examples = [
    ((3, 4),   (4,)),       # 行广播
    ((3, 4),   (3, 1)),     # 列广播
    ((1, 4),   (3, 1)),     # 双方都广播
    ((3, 1, 5), (4, 1)),    # 2维 vs 3维
    ((3, 4),   (4, 1)),     # OK
    ((3, 4),   (3,)),       # 会报错
]
#
# for a, b in examples:
#     try:
#         A = np.zeros(a)
#         B = np.ones(b)
#         result_shape = (A + B).shape
#         print(f"  {str(a):12s} + {str(b):12s} -> {str(result_shape):12s} OK")
#     except ValueError as e:
#         print(f"  {str(a):12s} + {str(b):12s} -> 报错 X")

for left, right in examples:
    try:
        A = np.zeros(left)
        B = np.ones(right)
        result_shape = (A + B).shape
        print(f"  {str(left):12s} + {str(right):12s} -> {str(result_shape):12s}  value = {A + B} \nOK")
    except ValueError as e:
        print(f"  {str(left):12s} + {str(right):12s} -> 错误 X")

# ============================================================
# 4. 可视化理解广播
# ============================================================
print("\n" + "=" * 50)
print("4. 广播的实际效果")
print("=" * 50)

#  矩阵         向量(行)       结果
#  [1 2 3]     [10 20 30]    [11 22 33]
#  [4 5 6]  +            =   [14 25 36]
#  [7 8 9]                  [17 28 39]
#
#  向量被"复制"成了 3 行（逻辑上），但 NumPy 不会真的复制数据

# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# v = np.array([10, 20, 30])
# print("A:\n", A)
# print("v:", v)
# print("A + v（每行加 v）:\n", A + v)

A = np.arange(9).reshape(3, 3)
V = np.array([10, 20, 30])
print("A:\n", A)
print("V:", V)
print(f"A + V = {A + V}")


# # 给每列加一个值：用 reshape
# c = np.array([1, 2, 3])       # 我们希望加在每列上
# c_col = c.reshape(-1, 1)      # (3,1)
# print("\nc_col:\n", c_col)
# print("A + c_col（每列加 c）:\n", A + c_col)
c = np.array([10, 20, 30])
c_col = c.reshape(-1, 1)
print("\nc_col: \n ", c_col)
print("A+c_col(每列加c)：\n", A + c_col)


# 更直观的写法：用 np.newaxis
# print("\n用 np.newaxis:")
# print("A + c[:, np.newaxis]:\n", A + c[:, np.newaxis])
# c[:, np.newaxis] 等价于 c.reshape(-1, 1), 但语义更清晰
print("\n用 np.newaxis: ")
print("A + c[:, np.newaxis]:\n", A + c[:, np.newaxis])

print("\n 部分数据加新维度：")
print(f"c[1:2, np.newaxis]:\n{c[1:3, np.newaxis]}")



# ============================================================
# 5. 常见用途
# ============================================================
print("\n" + "=" * 50)
print("5. 常见用途")
print("=" * 50)

# 5a. 标准化（减去均值，除以标准差）
# data = np.random.randn(4, 3)   # 4个样本，3个特征
# mean = data.mean(axis=0)       # 每个特征的均值 (3,)
# std = data.std(axis=0)         # 每个特征的标准差 (3,)
data: np.ndarray = np.random.randn(4, 3)
mean = data.mean(axis = 0)
mean2 = data.mean(axis = 1)

std = data.std(axis=0)
std2 = data.std(axis=1)

print("原始数据：\n", data)
print("\n均值:", mean)
print("均值2:", mean2)
print("\n标准差:", std)
print("标准差2:", std2)


# normalized = (data - mean) / std   # 广播自动应用到每一行
# print("原始数据:\n", data)
# print("\n均值:", mean)
# print("标准差:", std)
# print("\n标准化后的数据:\n", normalized)
# print("标准化后均值 ≈", normalized.mean(axis=0).round(6))
# print("标准化后标准差 ≈", normalized.std(axis=0).round(6))
normalized = (data - mean) / std
print("\n原始数据：\n", data)
print("\n均值：\n", mean)
print("\n标准差：\n", std)
print("\n标准化后的数据：\n", normalized)
print("标准化后均值 ≈", normalized.mean(axis=0).round(6))
print("标准化后标准差 ≈", normalized.std(axis=0).round(6))

# 5b. 外积
# x = np.array([1, 2, 3])
# y = np.array([10, 20, 30, 40])
# outer = x[:, np.newaxis] * y[np.newaxis, :]

x = np.array([1, 2, 3])
y = np.array([10, 20, 30, 40])
outer = x[:, np.newaxis] * y[np.newaxis, :]
print("x:", x)
print("y:", y)
print("x * y(外积):\n", outer)


# 等价于 np.outer(x, y)
# print("\n外积 x × y:\n", outer)
# print("np.outer 验证:\n", np.outer(x, y))
print("\n外积 x * y = \n", outer)
print("np.outer 验证：\n", np.outer(x, y))



# ============================================================
# 6. ★ 广播陷阱
# ============================================================
print("\n" + "=" * 50)
print("6. 常见陷阱")
print("=" * 50)

# 陷阱 1：误把 (n,) 当成行向量或列向量
# a = np.array([1, 2, 3])
# print("(3,) shape:", a.shape)      # (3,) — 一维，不是行也不是列
# print("行向量:", a[np.newaxis, :].shape)  # (1, 3)
# print("列向量:", a[:, np.newaxis].shape)  # (3, 1)
a: np.ndarray = np.array([1, 2, 3])
print("a.shape = ", a.shape)
print("a转行向量：", a[np.newaxis, :].shape)
print("a转列向量：", a[:, np.newaxis].shape)

# 陷阱 2：reduce 操作忘记 keepdims
# mat = np.random.randn(2, 3)
# sum0 = mat.sum(axis=0)           # shape (3,)
# sum0_keep = mat.sum(axis=0, keepdims=True)  # shape (1, 3)
# print("\nsum(axis=0):", sum0.shape)
# print("sum(axis=0, keepdims=True):", sum0_keep.shape)
mat = np.random.randn(2, 3)
print("\n mat: \n", mat)
sum0 = mat.sum(axis=0)
print(f"sum0={sum0} \n sum0.shape = {sum0.shape}")
sum0_keep = mat.sum(axis = 0, keepdims=True)
print(f"sum0_keep={sum0_keep} \n sum0_keep.shape = {sum0_keep.shape}")


# 有 keepdims 更方便后续广播
print("有 keepdims 可以直接广播:", (mat - sum0_keep).shape)  # OK
try:
    print("无 keepdims:", (mat - sum0).shape)  # (2,3) vs (3,) → OK?
    # 咦？这居然不报错，因为 (2,3) vs (3,) → 广播 (3,) 到 (2,3)
    # 但语义可能不合适，结果可能不是你要的
    print("  虽然不报错，但语义要确认")
except ValueError:
    pass

# 陷阱 3：字符串和数组运算可能不是你想的那样
# print(np.array(["1", "2"]) + np.array(["3", "4"]))  # TypeError
print(np.array(["1", "2"]))
print(np.array(["3", "4"]))
np.array(["1", "2"]) + np.array(["3", "4"])

# ============================================================
# ★ 动手练习
# ============================================================
# ============================================================
# 7. 动手练习
# ============================================================
print("\n" + "#" * 50)
print("# 动手练习")
print("#" * 50)

# # 练习 1：不用 for 循环，对矩阵的每一行加不同的值
mat = np.arange(12).reshape(3, 4)
row_add = np.array([1, 2, 3])       # 第0行+1，第1行+2，第2行+3
# 提示：需要变成 (3, 1)
# ↓ 你写：
row_add = row_add[:, np.newaxis]
print(f"row_add={row_add}")
result = mat + row_add
print(f"result={result}")

# 练习 2：计算矩阵每一行的 L2 范数（欧几里得距离）
# 公式：sqrt(sum(x_i^2))
vectors = np.random.randn(5, 3)     # 5 个三维向量
# 不用 for 循环，用广播
# ↓ 你写：
print(vectors * vectors)
norms = np.sqrt(np.sum(vectors * vectors, axis=-1))
print(f"\nnorms={norms}")

# 练习 3：欧几里得距离矩阵
# 给定两组点，计算两两之间的距离
points_a = np.array([[0, 0], [1, 1], [2, 2]])     # 3 个点
points_b = np.array([[0, 0], [1, 1]])              # 2 个点
# 期望输出 shape (3, 2)，即每个 a 点与每个 b 点的距离
# 提示：(a - b)² → sum(axis=-1) → sqrt
# 再提示：用 np.newaxis 扩展维度
# 答案公式：(a[:, np.newaxis, :] - b[np.newaxis, :, :]) 的 L2
# ↓ 你实现：
a = points_a[:, np.newaxis, :]
b = points_b[np.newaxis, :, :]
dif = a - b
dist = np.sqrt(np.sum(dif**2, axis=-1))
print(dist)
