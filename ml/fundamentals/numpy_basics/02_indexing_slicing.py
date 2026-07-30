"""
02 - 索引与切片
===============
NumPy 的索引比 Python list 强大得多：
  基本切片、花式索引、布尔索引。

运行：python 02_indexing_slicing.py
"""

import numpy as np

# ============================================================
# 1. 基本索引（和 Python list 类似）
# ============================================================
print("=" * 50)
print("1. 基本索引")
print("=" * 50)

arr = np.array([10, 20, 30, 40, 50])
print("arr:", arr)
print("arr[0]:", arr[0])      # 10
print("arr[-1]:", arr[-1])    # 50
print("arr[-2]:", arr[-2])    # 40

# arr: np.ndarray = np.array([10, 20, 30, 40, 50])
# print("arr=", arr)
# print("arr首个=", arr[0])
# print("arr末个=", arr[-1])
# print("arr倒第二个", arr[-2])

# 二维
# mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print("\nmat:\n", mat)
# print("mat[0, 1]:", mat[0, 1])   # 第0行第1列 → 2
# print("mat[1]:", mat[1])         # 第1行 → [4 5 6]
# print("mat[1, :]:", mat[1, :])   # 同上，明确表示"第1行所有列"

mat: np.ndarray = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(f"mat:\n{mat}\n")
print(f"第0行 第1列 mat[0,1] = ", mat[0, 1])
print(f"第2行 mat[1]: {mat[1]}")
print(f"第3行 mat[2, :]: {mat[2, :]}")

# ============================================================
# 2. 切片 — 和 list 一样 [start:stop:step]
# ============================================================
print("\n" + "=" * 50)
print("2. 切片")
print("=" * 50)

# a = np.arange(10)
# print("a:", a)
# print("a[2:5]:", a[2:5])         # [2 3 4]
# print("a[:5]:", a[:5])           # [0 1 2 3 4]
# print("a[::2]:", a[::2])         # 每隔一个取
# print("a[::-1]:", a[::-1])       # 反转
a: np.ndarray = np.arange(10)
print(f"a: {a}")
print(f"a[2:5]: {a[2:5]}")
print(f"a[:5]: {a[:5]}")
print(f"a[::2]: {a[::2]}")
print(f"a[::3]: {a[::3]}")
print(f"a[::]: {a[::]}")
print(f"a[:]: {a[:]}")
print(f"a[::-1]: {a[::-1]}")


# 二维切片 — 先行后列
# mat = np.arange(16).reshape(4, 4)
# print("\nmat:\n", mat)
# print("\nmat[:2, :3]:\n", mat[:2, :3])   # 前2行，前3列
# print("\nmat[1:3, 1:3]:\n", mat[1:3, 1:3])  # 中心子矩阵

mat: np.ndarray = np.arange(16).reshape(4, 4)
print(f"矩阵mat:\n {mat}")
print(f"\n mat[:2, :3]:\n{mat[:2, :3]}")
print(f"\n mat[1:3, 1:3]: \n {mat[1:3, 1:3]}")


# 隔行采样
# print("\nmat[::2, ::2]:\n", mat[::2, ::2])  # 每隔一行一列
print(f"\nmat[::2, ::2]: \n {mat[::2, ::2]}")

# 行/列提取
# print("\nmat[:, 1]:", mat[:, 1])     # 第1列（变成一维）
# print("\nmat[:, [1]]:\n", mat[:, [1]])  # 第1列（保持二维，后面解释）
print(f"\n mat[:, 1]: \n{mat[:, 1]}")
print(f"\n mat[:, [1]]: \n{mat[:, [1]]}")

# ============================================================
# 3. ★ 视图 vs 副本（重要！）
# ============================================================
print("\n" + "=" * 50)
print("3. 视图 vs 副本 — 切片返回视图！")
print("=" * 50)

# arr = np.array([1, 2, 3, 4, 5])
# view = arr[1:4]               # ← 这是视图，不是副本
# view[0] = 999                 # 修改视图会影响原数组
# print("arr after modifying view:", arr)   # [1 999 3 4 5]
arr: np.ndarray = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
view[0] = 1024
print(f"arr: {arr}")
print(f"view: {view}")

# 要复制用 .copy()
# arr2 = np.array([1, 2, 3, 4, 5])
# copy = arr2[1:4].copy()
# copy[0] = 999
# print("arr2 after modifying copy:", arr2)  # [1 2 3 4 5] ← 不受影响
arr2 = np.array([1, 2, 3, 4, 5])
copy = arr2[1:4].copy()
copy[0] = 2048
print(f"arr2: {arr2}")
print(f"copy: {copy}")

# 切片返回视图是性能设计（不复制数据），
# 但如果要独立修改，记得 .copy()


# ============================================================
# 4. 花式索引（Fancy Indexing）— 用整数数组索引
# ============================================================
print("\n" + "=" * 50)
print("4. 花式索引")
print("=" * 50)

# a = np.array([10, 20, 30, 40, 50, 60])
# indices = [0, 2, 4]
# print("a:", a)
# print("a[[0, 2, 4]]:", a[indices])   # [10 30 50]
a: np.ndarray = np.array([10, 20, 30, 40, 50, 60])
indices:list[int] = [0, 2, 4]
print(f"a: {a}")
print(f"a[indices]: {a[indices]}")
a[indices] = 1
print(f"a: {a}")


# 取多个不连续的行
# mat = np.arange(20).reshape(5, 4)
# print("\nmat:\n", mat)
# print("\nmat[[0, 2, 4]]:\n", mat[[0, 2, 4]])  # 第0, 2, 4行
mat: np.ndarray = np.arange(20).reshape(5, 4)
print(f"\n mat: \n{mat}")
print(f"\n mat[[0, 2, 4]]: \n {mat[[0, 2, 4]]}")

# 同时指定行和列的索引
# print("\nmat[[0, 1, 2], [0, 1, 2]]:", mat[[0, 1, 2], [0, 1, 2]])
# 对应取：(0,0), (1,1), (2,2) → [0, 5, 10]

# 和切片不同，花式索引总是返回副本


# ============================================================
# 5. 布尔索引 — 最强大的筛选方式
# ============================================================
print("\n" + "=" * 50)
print("5. 布尔索引")
print("=" * 50)

# a = np.array([1, 2, 3, 4, 5, 6])
a: np.ndarray[int] = np.array([1, 2, 3, 4, 5, 6])

# 条件比较生成布尔数组
# mask = a > 3
# print("a:", a)
# print("a > 3:", mask)
# print("a[a > 3]:", a[a > 3])         # [4 5 6]
mask = a > 3
print(f"a: {a}")
print(f"a>3: {mask}")
print(f"a[a>3]: {a[a > 3]}   {a[mask]}")

# 组合条件：用 & | ~ （不能用 and or not）
# print("a[(a > 2) & (a < 5)]:", a[(a > 2) & (a < 5)])  # [3 4]
# print("a[~(a > 3)]:", a[~(a > 3)])   # 取反 → [1 2 3]
print(f"a[(a>2) & (a<5)]: {a[(a>2) & (a<5)]}")
print(f"a[(a>2) & (a<1)]: {a[(a>2) & (a<1)]}")


# 实战：替换满足条件的值
# a[a % 2 == 0] = -1
# print("\na (偶数替换为 -1):", a)
a[a%2 == 0] = -1
print(f"a = {a}")

# 二维布尔索引
# mat = np.arange(12).reshape(3, 4)
# print("\nmat:\n", mat)
# print("\nmat[mat > 5]:", mat[mat > 5])   # 返回一维数组
# 注意：布尔索引二维数组结果是展平的一维
mat: np.ndarray[ int] = np.arange(12).reshape(3, 4)
print(f"\nmat:\n{mat}")
print(f"\n mat[mat > 5]: {mat[mat > 5]}")


# ============================================================
# 6. np.where — 条件索引的瑞士军刀
# ============================================================
print("\n" + "=" * 50)
print("6. np.where")
print("=" * 50)

# a = np.array([1, 3, 2, 5, 4, 6])
a: np.ndarray = np.array([1, 3, 2, 5, 4, 6])

# where(条件) → 返回符合条件的索引
# indices = np.where(a > 3)
# print("a:", a)
# print("np.where(a > 3):", indices)
# print("a[np.where(a > 3)]:", a[indices])
indices = np.where(a > 3)
print(f"\na={a}")
print(f"np.where(a>3): {indices}")
print(f"a[where(a>3)]: {a[indices]}")

# where(条件, x, y) → 满足条件选 x，否则选 y
# result = np.where(a > 3, 100, 0)
# print("np.where(a > 3, 100, 0):", result)
print(f"np.where(a>3, 100, 0): {np.where(a>3, 100, 0)}")

# 实战：找最大值位置
arr = np.array([[3, 7, 1], [8, 2, 5], [4, 9, 6]])
max_idx = np.where(arr == arr.max())
print("\narr:\n", arr)
print("最大值位置:", max_idx, "→ 值:", arr[max_idx])


# ============================================================
# 7. 索引总结对照表
# ============================================================
print("\n" + "=" * 50)
print("7. 索引方式总结")
print("=" * 50)

arr = np.array([10, 20, 30, 40, 50])
print(f"arr = {arr}")
print(f"  arr[1]          = {arr[1]}          # 整数索引")
print(f"  arr[[0, 2, 4]]  = {arr[[0, 2, 4]]}  # 花式索引")
print(f"  arr[1:4]        = {arr[1:4]}        # 切片")
print(f"  arr[arr > 30]   = {arr[arr > 30]}   # 布尔索引")


# ============================================================
# ★ 动手练习
# ============================================================
# ============================================================
# 8. 动手练习
# ============================================================
print("\n" + "#" * 50)
print("# 动手练习")
print("#" * 50)

# 准备数据
data = np.arange(25).reshape(5, 5)

# 练习 1：提取 data 的边框（第一行、最后一行、第一列、最后一列）
# 预期：
# 边框值 = [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]
# 提示：可以用花式索引 + np.concatenate，或用切片 + .copy()
# 注意不要漏掉角落被重复计算
# ↓ 你的代码：
print(f"\ndata:\n {data}")
top = data[0]
bottom = data[-1]
left = data[1:-1, 0]
right = data[1:-1, -1]
print(f"left: {left}, right: {right}")
mid = np.column_stack((left, right))

border = np.concatenate((top, mid.flatten(), bottom))
print(f"data的边框值：{border}")

# 练习 2：提取 data 中所有能被 3 整除的元素
# ↓ 你的代码：
copy = data.copy()
div3 = copy[copy % 3 == 0]
print(f"\ndiv3中能被3整除的数字：{div3}")

# 练习 3：将 data 中所有小于 10 或大于 20 的元素替换为 -1
# ↓ 你的代码：
data_filtered = data.copy()
indices = np.where((data_filtered<10) | (data_filtered>20))
data_filtered[indices] = -1
print(f"\ndata 中所有小于 10 或大于 20 的元素替换为 -1:\n {data_filtered}")

# 练习 4：用 np.where 将 data 中的偶数替换为 0，奇数替换为 1
# ↓ 你的代码：
print(f"\n用 np.where 将 data 中的偶数替换为 0，奇数替换为 1\n")
binary = data.copy()
binary = np.where(binary%2==0, 0, 1)
print(f"indices: \n{binary}")
