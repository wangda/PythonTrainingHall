"""
05 - 统计函数与 axis 理解
=========================
数据预处理离不开统计量。
这份文件拆成小步子，每步先讲概念再敲代码。

运行：python 05_statistics.py
"""

import numpy as np

# ============================================================
# 第一步：描述性统计量——用几个数概括一堆数
# ============================================================
# 给你一堆数据，比如全班考试成绩：
#   mean（均值）    = 总分 ÷ 人数        → 代表"平均水平"
#   std（标准差）   = 每个分数离均值多远的平均   → 代表"分散程度"
#   var（方差）     = 标准差的平方        → 同上，量纲不同
#   min / max       = 最小值和最大值
#   median（中位数） = 排序后在中间的数   → 比均值更能抵抗异常值
#
# 区分：均值易受极端值影响（班里一个考 100 一个考 0，均值 50）
#       中位数更稳健（排序后取中间）

print("=" * 55)
print("第一步：描述性统计量")
print("=" * 55)

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print("data:", data)
print("总和 sum:", np.sum(data))           # 21
print("均值 mean:", np.mean(data))          # 3.5
print("中位数 median:", np.median(data))    # 3.5
print("方差 var:", np.var(data))            # 2.92
print("标准差 std:", np.std(data))          # 1.71
print("最小值 min:", np.min(data))
print("最大值 max:", np.max(data))
print("最小值的索引 argmin:", np.argmin(data))
print("最大值的索引 argmax:", np.argmax(data))

# --- 均值 vs 中位数 的区别 ---
# 假设薪资数据：绝大部分人月薪 5k-10k，但 CEO 月薪 100w
salaries = np.array([5, 6, 7, 8, 9, 100])   # 单位：千
print("\n薪资数据（含 CEO）:", salaries)
print("均值:", np.mean(salaries), "（被 CEO 拉高了）")
print("中位数:", np.median(salaries), "（更能代表普通员工）")
# 这就是为什么新闻常说"平均工资"不靠谱——中位数更真实


# ============================================================
# 第二步：总体方差 vs 样本方差（ddof 参数）
# ============================================================
# np.var 和 np.std 默认算的是"总体方差"（除以 n）。
# 但你拿到的数据通常只是"样本"（从总体里抽出来的）。
# 样本方差要除以 (n-1)，因为用了一个自由度去估计均值。
#
# ddof = 0（默认）→ 总体方差，除以 n
# ddof = 1        → 样本方差，除以 (n-1)

print("\n" + "=" * 55)
print("第二步：总体 vs 样本标准差")
print("=" * 55)

data = np.array([2, 4, 6, 8, 10])
print("data:", data)
print("总体标准差 ddof=0:", np.std(data, ddof=0))   # 除以 5
print("样本标准差 ddof=1:", np.std(data, ddof=1))   # 除以 4

# 当数据量很大时，两者差别很小，可以忽略。小数据集要注意。

# 在机器学习中，标准化（Z-score）一般用总体标准差 ddof=0：
# z = (x - mean) / std    ← 默认 ddof=0


# ============================================================
# 第三步：★ axis 参数——新手最容易蒙的地方
# ============================================================
# axis = 沿着哪个轴"压缩"
#
#   axis=0 → 沿着行的方向压缩 → 结果是对"每一列"操作
#   axis=1 → 沿着列的方向压缩 → 结果是对"每一行"操作
#
# 想象一个 3×4 的矩阵：
#        col0  col1  col2  col3
# row0   [1     2     3     4]
# row1   [5     6     7     8]
# row2   [9    10    11    12]
#
# sum(axis=0)  → 每个列求和 → [15 18 21 24]   shape (4,)
# sum(axis=1)  → 每个行求和 → [10 26 42]       shape (3,)

print("\n" + "=" * 55)
print("第三步：axis 参数（重点）")
print("=" * 55)

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("arr:\n", arr)

print("\n--- axis=0（跨行，对每一列操作）---")
print("sum(axis=0):", arr.sum(axis=0))     # [12 15 18]
print("mean(axis=0):", arr.mean(axis=0))   # [4. 5. 6.]
print("max(axis=0):", arr.max(axis=0))     # [7 8 9]

print("\n--- axis=1（跨列，对每一行操作）---")
print("sum(axis=1):", arr.sum(axis=1))     # [6 15 24]
print("mean(axis=1):", arr.mean(axis=1))   # [2. 5. 8.]
print("max(axis=1):", arr.max(axis=1))     # [3 6 9]

# ────────────────────────────────────────────────────
# 更直观的理解：
#   axis=0 → "垂直方向" → 结果 shape 少了"行"
#   axis=1 → "水平方向" → 结果 shape 少了"列"
# ────────────────────────────────────────────────────
# 也可以这么记：axis = 你要压缩（求和）的那个维度

# --- 继续：三维数组 ---
three_d = np.arange(24).reshape(2, 3, 4)
print("\n三维数组 shape:", three_d.shape)
# axis=0 → 压掉第 0 维 → (3, 4)
print("sum(axis=0) → shape (3, 4):")
print(three_d.sum(axis=0))

# axis=1 → 压掉第 1 维 → (2, 4)
print("\nsum(axis=1) → shape (2, 4):")
print(three_d.sum(axis=1))

# axis=2 → 压掉第 2 维 → (2, 3)
print("\nsum(axis=2) → shape (2, 3):")
print(three_d.sum(axis=2))


# ============================================================
# 第四步：keepdims——保留维度，方便继续运算
# ============================================================
# 不加 keepdims：shape 从 (3, 4) 变成 (4,)
# 加 keepdims：  shape 从 (3, 4) 变成 (1, 4)
#
# 为什么需要 keepdims？
# 因为标准化时：(data - mean) / std
# 如果 mean 的 shape 和 data 不匹配，广播会出问题。

print("\n" + "=" * 55)
print("第四步：keepdims（保持维度）")
print("=" * 55)

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("arr shape:", arr.shape)

s0 = arr.sum(axis=0)
print("\nsum(axis=0) shape:", s0.shape)     # (3,)  — 维度没了

s0_k = arr.sum(axis=0, keepdims=True)
print("sum(axis=0, keepdims=True) shape:", s0_k.shape)  # (1, 3)

s1_k = arr.sum(axis=1, keepdims=True)
print("sum(axis=1, keepdims=True) shape:", s1_k.shape)  # (2, 1)

# 实际应用：数据标准化
print("\n标准化示例（利用 keepdims + 广播）:")
data = np.random.randn(4, 3)
mean = data.mean(axis=0, keepdims=True)    # shape (1, 3)
std = data.std(axis=0, keepdims=True)      # shape (1, 3)
normalized = (data - mean) / std           # (4,3) - (1,3) → 广播 OK
print("标准化后均值 ≈", normalized.mean(axis=0).round(6))
print("标准化后标准差 ≈", normalized.std(axis=0).round(6))

# 如果不加 keepdims：
# mean = data.mean(axis=0)  → shape (3,)
# data - mean  → (4,3) - (3,) → 广播也 OK
# 但语义不清晰。keepdims 是好的工程习惯。


# ============================================================
# 第五步：累计统计 cumsum / cumprod
# ============================================================
# cumsum = 累加：当前位置之前所有元素的和
# cumprod = 累乘
#
# 应用：计算累计增长率、滑动窗口均值

print("\n" + "=" * 55)
print("第五步：累计统计")
print("=" * 55)

arr = np.array([1, 2, 3, 4, 5, 6])
print("arr:", arr)
print("cumsum（累加）:", np.cumsum(arr))   # [1, 3, 6, 10, 15, 21]
print("cumprod（累乘）:", np.cumprod(arr)) # [1, 2, 6, 24, 120, 720]

# 二维累计
mat = np.arange(9).reshape(3, 3)
print("\nmat:\n", mat)
print("cumsum(axis=0)（逐行累加）:\n", np.cumsum(mat, axis=0))
print("cumsum(axis=1)（逐列累加）:\n", np.cumsum(mat, axis=1))


# ============================================================
# 第六步：百分位数与五数概括
# ============================================================
# 中位数就是 50% 分位数。
# 常用百分位数：
#   25% 分位数（Q1）→ 四分之一分位点
#   50% 分位数（Q2）→ 中位数
#   75% 分位数（Q3）→ 四分之三分位点
#
# 五数概括 = [min, Q1, median, Q3, max]
# 箱线图就是基于五数概括画的

print("\n" + "=" * 55)
print("第六步：百分位数与五数概括")
print("=" * 55)

# 生成 1000 个正态分布数据
np.random.seed(42)
data = np.random.randn(1000)

print("数据分布（标准正态分布，理论均值 0，标准差 1）:")
for q in [5, 25, 50, 75, 95]:
    print(f"  {q}% 分位数: {np.percentile(data, q):.3f}")

# 五数概括
min_val = np.min(data)
q1 = np.percentile(data, 25)
med = np.median(data)
q3 = np.percentile(data, 75)
max_val = np.max(data)
print("\n五数概括:")
print(f"  Min={min_val:.3f}, Q1={q1:.3f}, Med={med:.3f}, Q3={q3:.3f}, Max={max_val:.3f}")


# ============================================================
# 第七步：相关性——两个变量之间的关系
# ============================================================
# 相关系数的范围是 [-1, 1]：
#   > 0  → 正相关（x 越大 y 越大）
#   < 0  → 负相关（x 越大 y 越小）
#   = 0  → 无关

print("\n" + "=" * 55)
print("第七步：相关性")
print("=" * 55)

np.random.seed(42)
x = np.random.randn(100)
y_pos = 2 * x + 0.5 * np.random.randn(100)           # 正相关
y_neg = -2 * x + 0.5 * np.random.randn(100)          # 负相关
y_rand = np.random.randn(100)                         # 无关

corr_matrix_p = np.corrcoef(x, y_pos)
corr_matrix_n = np.corrcoef(x, y_neg)
corr_matrix_r = np.corrcoef(x, y_rand)

print("y = 2x + noise  相关系数:", corr_matrix_p[0, 1].round(3))
print("y = -2x + noise 相关系数:", corr_matrix_n[0, 1].round(3))
print("y = random      相关系数:", corr_matrix_r[0, 1].round(3))


# ============================================================
# 第八步：唯一值与计数
# ============================================================
print("\n" + "=" * 55)
print("第八步：唯一值与计数")
print("=" * 55)

labels = np.array([1, 0, 1, 2, 1, 0, 1, 2, 2, 0, 1])

unique, counts = np.unique(labels, return_counts=True)
print("类别:", unique)
print("计数:", counts)

# bincount 更快（只适用于非负整数）
print("bincount:", np.bincount(labels))


# ============================================================
# 动手练习
# ============================================================
print("\n" + "#" * 55)
print("# 动手练习")
print("#" * 55)

# 练习 1：数据标准化
# 生成 100×5 的矩阵，对每个特征（列）做 Z-score 标准化
# 用广播 + keepdims，不用 for 循环
np.random.seed(42)
X = np.random.randn(100, 5)
# ↓ 你的代码
# X_norm = ...

# 练习 2：查找离群点
# 找出 data 中超过"均值 ± 3 倍标准差"的元素
data = np.random.randn(1000)
data[500] = 100   # 手动加异常值
# ↓ 用布尔索引找出离群点的索引

# 练习 3：滚动窗口均值（滑动平均）
# 用 cumsum 实现，不用 for 循环
signal = np.sin(np.linspace(0, 10, 100))
window_size = 5
# 提示：rolling[i] = (cumsum[i] - cumsum[i-5]) / 5
# ↓ 你实现

# 练习 4：混淆矩阵
# 手动计算每个类别的准确率
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 0, 2, 1, 1, 2])
# 对每个类别，正确预测数 / 总数
# ↓ 你的代码
