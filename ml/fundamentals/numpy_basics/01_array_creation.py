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
print("=" * 50)
print("1. 从 list 创建")
print("=" * 50)

# 一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print("一维:", arr1)
print("形状:", arr1.shape)     # (5,)
print("维度:", arr1.ndim)      # 1
print("数据类型:", arr1.dtype) # int64（由内容推断）

# 二维数组（矩阵）
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\n二维:\n", arr2)
print("形状:", arr2.shape)     # (2, 3)
print("维度:", arr2.ndim)      # 2

# 指定数据类型
arr3 = np.array([1, 2, 3], dtype=np.float32)
print("\n指定 dtype float32:", arr3, "→ dtype:", arr3.dtype)


# ============================================================
# 2. 全零 / 全一 / 空数组
# ============================================================
print("\n" + "=" * 50)
print("2. zeros / ones / empty")
print("=" * 50)

zeros = np.zeros((3, 4))          # 3行4列全零
print("zeros:\n", zeros)

ones = np.ones((2, 3))            # 2行3列全一
print("\nones:\n", ones)

empty = np.empty((2, 2))          # 分配内存但不初始化（值为内存垃圾）
print("\nempty（不要依赖初始值）:\n", empty)

# 单位矩阵
eye = np.eye(4)                   # 4x4 单位矩阵
print("\neye(4):\n", eye)

# 对角矩阵
diag = np.diag([1, 2, 3, 4])
print("\ndiag([1,2,3,4]):\n", diag)

# full — 全为指定值
full = np.full((2, 3), 7)
print("\nfull (all=7):\n", full)


# ============================================================
# 3. arange — 类似 range，但返回数组
# ============================================================
print("\n" + "=" * 50)
print("3. arange")
print("=" * 50)

a = np.arange(10)                  # 0 ~ 9
print("arange(10):", a)

b = np.arange(2, 10, 2)           # 2, 4, 6, 8
print("arange(2, 10, 2):", b)

c = np.arange(0, 1, 0.2)          # 浮点数步长
print("arange(0, 1, 0.2):", c)
# 注意：浮点数步长可能因精度问题末尾元素不准确，下面 linspace 更可靠


# ============================================================
# 4. linspace — 等间距取 N 个点（推荐）
# ============================================================
print("\n" + "=" * 50)
print("4. linspace")
print("=" * 50)

ls = np.linspace(0, 1, 5)         # [0, 1] 之间等距取 5 个点
print("linspace(0, 1, 5):", ls)

ls2 = np.linspace(0, 10, 11)      # 和 arange(0, 11) 等价但更安全
print("linspace(0, 10, 11):", ls2)

# endpoint=False 排除终点
ls3 = np.linspace(0, 1, 4, endpoint=False)
print("linspace(0, 1, 4, endpoint=False):", ls3)


# ============================================================
# 5. reshape — 改变形状
# ============================================================
print("\n" + "=" * 50)
print("5. reshape")
print("=" * 50)

x = np.arange(12)
print("原始:", x, "shape:", x.shape)

# reshape 成 3x4 矩阵
y = x.reshape(3, 4)
print("reshape(3, 4):\n", y)

# -1 表示自动推断维度
z = x.reshape(2, -1)              # 2行，列自动计算
print("reshape(2, -1):\n", z)

# flatten / ravel — 展平
print("flatten:", y.flatten())
print("ravel:  ", y.ravel())
# 区别：flatten 返回副本，ravel 返回视图（修改会影响原数组）

# ⚠️ reshape 必须保证元素总数不变，否则报错：
try:
    x.reshape(3, 5)
except ValueError as e:
    print("\nreshape 错误:", e)


# ============================================================
# 6. 数组属性总结
# ============================================================
print("\n" + "=" * 50)
print("6. 常用属性")
print("=" * 50)
arr = np.random.randn(3, 4, 5)    # 3×4×5 三维数组
print("数组:\n", arr)
print("shape:", arr.shape)        # (3, 4, 5)
print("ndim:", arr.ndim)          # 3（维度数）
print("size:", arr.size)          # 60（总元素数）
print("dtype:", arr.dtype)        # float64
print("itemsize:", arr.itemsize, "bytes")  # 每个元素字节数
print("nbytes:", arr.nbytes, "bytes")      # 总字节数


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
print(my_ones)
print("第一题结束 \n ")

# 练习 2：用 arange 创建一个从 0 到 20（不含）的数组，再 reshape 成 4x5
# ↓ 在下面写你的代码
arr: np.ndarray = np.arange(0, 20)
arr1 = arr.reshape(4, 5)
print(arr1)
print("第二题结束 \n ")

# 练习 3：用 linspace 在 [-π, π] 之间取 100 个点（用于后面画 sin 曲线）
# ↓ 在下面写你的代码
arr2 = np.linspace(-np.pi, np.pi, 100)
print(arr2)
print("第三题结束 \n ")

# 练习 4：创建一个 3x3 的"棋盘"矩阵（交替 0 和 1）
# 预期结果：
# [[0 1 0]
#  [1 0 1]
#  [0 1 0]]
# 提示：用 np.eye + 某种操作，或用 np.zeros + 花式索引（下节学）
# ↓ 试试看
arr3: np.ndarray = np.eye(3, 3, k=1)
arr3 = arr3 + np.eye(3, 3, k=-1)
print(arr3)