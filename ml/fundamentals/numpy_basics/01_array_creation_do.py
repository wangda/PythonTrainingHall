"""
01 - 数组创建
=============
NumPy 的核心是 ndarray（N-dimensional array）。
先用这几种方式创建数组，感受和 Python list 的区别。

运行：python 01_array_creation.py
"""
import numpy as np


# ============================================================
# 1. 从 Python list 创建
# ============================================================
print("*" * 50)
print("1. 从 list 创建")
print("*" * 50)

# 一维数组
# N-Dimensional（N 维）
arr1 = np.array([2,3,4,5,6,7,8,100])
print(f"一维:{arr1}")
print(f"形状：{arr1.shape}")
print(f"维度：{arr1.ndim}")
print(f"数据类型：{arr1.dtype}")

# 二维数组（矩阵）
arr2 = np.array([[1,2,3], [4,5,6], [7,3,9]])
print(f"\n二维：{arr2} ")
print(f"形状：{arr2.shape}")
print(f"维度：{arr2.ndim}")
print(f"数据类型：{arr2.dtype}")

# ============================================================
# 2. 全零 / 全一 / 空数组
# ============================================================
print("\n" + "*" * 50)
print("2. zeros, ones, empty")
print("*" * 50)

zeros: np.ndarray = np.zeros(shape=(3,4), dtype=np.int64)
print(f"zeros: \n {zeros}")

ones: np.ndarray = np.ones(shape=(2,3))
print(f"ones: \n {ones}")

empty: np.ndarray = np.empty((4,5))
print(f"empty:\n {empty}")

# ============================================================
# 3. arange — 类似 range，但返回数组
# ============================================================
print("\n" + "*" * 50)
print("3. arange")
print("*" * 50)

a: np.ndarray = np.arange(10)
print(f"arange(10): {a}")

b: np.ndarray = np.arange(2, 10, 2)
print(f"arange(2, 10, 2): {b}")

c: np.ndarray = np.arange(0, 1, 0.2)
print(f"arange(0, 1, 0.2): {c}")

# ============================================================
# 4. linspace — 等间距取 N 个点（推荐）
# ============================================================
print("*" * 50)
print("4. linspace")
print("*" * 50)

ls = np.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {ls}")

ls2 = np.linspace(0, 10, 13)
print(f"linspace(1, 10, 13): {ls2}")

# ============================================================
# 5. reshape — 改变形状
# ============================================================
print("\n" + "*" * 50)
print("5. reshape")
print("*" * 50)

x = np.arange(12)
print(f"原始，shape:{x.shape}, 数据：{x}")

# reshape 成 3x4 矩阵
y = x.reshape(3, 4)
print(f"reshape(3,4): \nshape={y.shape}, \n 数据：{y}")

# -1 表示自动推断维度
z = x.reshape(2, -1)
print(f"reshape(2, -1): \nshape={z.shape}, \n 数据：{z}")

# flatten / ravel — 展平
print(f"\nflatten: {y.flatten()}")
print("ravel: ", y.ravel())
# 区别：flatten 返回副本，ravel 返回视图（修改会影响原数组）

# ⚠️ reshape 必须保证元素总数不变，否则报错：
try:
    x.reshape(3, 8)
except ValueError as e:
    print("reshape报错：", e)

# ============================================================
# 6. 数组属性总结
# ============================================================
print("*" * 50)
print("6. 常用属性")
print("*" * 50)

arr: np.ndarray = np.random.randn(3,4, 5)

print("数组：", arr)
print("形状：", arr.shape)
print("维度：", arr.ndim)
print("数据类型：",arr.dtype)
print("元素数量：", arr.size)
print("每个元素字节大小:", arr.itemsize)
print("数组占总字节大小：", arr.nbytes)


# ============================================================
# 7. ★ 动手练习
# ============================================================
# ============================================================
# 7. 动手练习（在下面补充代码，然后运行验证）
# ============================================================
print("\n" + "#" * 50)
print("# 动手练习")
print("#" * 50)

# 练习 1：创建一个 5x5 的全一矩阵，数据类型为 int32
# ↓ 在下面写你的代码
my_ones = np.ones(shape=(5, 5), dtype=np.int32)

# 练习 2：用 arange 创建一个从 0 到 20（不含）的数组，再 reshape 成 4x5
# ↓ 在下面写你的代码


# 练习 3：用 linspace 在 [-π, π] 之间取 100 个点（用于后面画 sin 曲线）
# ↓ 在下面写你的代码

# 练习 4：创建一个 3x3 的"棋盘"矩阵（交替 0 和 1）
# 预期结果：
# [[0 1 0]
#  [1 0 1]
#  [0 1 0]]
# 提示：用 np.eye + 某种操作，或用 np.zeros + 花式索引（下节学）
# ↓ 试试看
