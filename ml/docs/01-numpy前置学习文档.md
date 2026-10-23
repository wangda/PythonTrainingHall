# 01 - NumPy 前置学习文档（教程 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：数组创建、索引与切片、形状操作、广播、ufunc、统计聚合（axis）、矩阵运算（dot / linalg / einsum）、随机数。
> **本文环境**：NumPy 2.x（所有示例已用 NumPy 2.2 实际运行验证）。

---

## 本文怎么用

1. **当教程读**：从头读一遍，每个代码块亲手敲进解释器跑一遍。重点章节（§5 广播、§7 axis）务必动手。
2. **当速查手册用**：忘了某个函数的参数，直接查文末【完整速查表】或跳到对应大节。

**先决知识**：Python 基础你已具备；数学概念不熟的先读前置文档：`ml/docs/00-线性代数前置知识.md`（矩阵乘法、点积、范数）、`ml/docs/00-概率与数理统计前置知识.md`（均值、方差、正态分布）、`ml/docs/00-微积分前置知识.md`（阶段二反向传播才用，本文不涉及）。

**阅读约定**：🟢 必须掌握（KMeans 和阶段二手写算法天天用）；🟡 了解即可（看得懂别人代码就行）。

---

## 1. 核心对象 ndarray 🟢

### 1.1 ndarray 是什么

`ndarray`（N 维数组）是 NumPy 的核心对象，和 Python list 对比：

| 对比项     | Python list                  | NumPy ndarray            |
| ---------- | ---------------------------- | ------------------------ |
| 元素类型   | 可混装 int/str/对象          | **统一类型**（dtype）    |
| 内存布局   | 存对象指针，元素散落各处     | **连续内存块**           |
| 速度       | 解释器逐条执行，慢           | 底层 C 循环，快几十倍    |
| `+` 的含义 | 拼接 `[1,2]+[3,4]=[1,2,3,4]` | 相加 `[1,2]+[3,4]=[4,6]` |

**为什么 ML 全用它**：机器学习的数据都是"数字表格"——图片是 `(H, W, 3)`，文本是 `(seq_len,)`，一批样本是 `(batch, features)`。NumPy 是整条科学计算链的地基：Matplotlib 收 ndarray，Pandas 底层是 ndarray，PyTorch 的 Tensor API 照抄 NumPy。**学好 NumPy，形状思维直接迁移到后面所有工具。**

### 1.2 dtype：数组的类型

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([1.0, 2, 3])                 # 有浮点就整体提升
c = np.array([1, 2, 3], dtype=np.float32)
print(a.dtype, b.dtype, c.dtype, c.astype(np.int32).dtype)
# int64 float64 float32 int32 —— astype 转换类型（返回新数组）
```

常用 dtype：`int32/int64`、`float32/float64`（ML 训练一般用 float32 省内存）、`bool`。**规则：一个数组只有一个 dtype**，混进 float 全体变 float。

### 1.3 核心属性

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape, a.ndim, a.size, a.dtype)   # (2, 3) 2 6 int64
# shape = 形状（最常用），ndim = 维度数，size = 元素总数，dtype = 类型
```

ML 代码里的 `(batch_size, features)` 形状标注对应的就是 `.shape`；遇到维度报错，第一反应打印双方的 `.shape`。

**本节小结**：ndarray = 统一类型 + 连续内存的 N 维数组，比 list 快几十倍。背四个属性：`.shape` / `.ndim` / `.size` / `.dtype`。

---

## 2. 数组创建 🟢

### 2.1 从数据创建

**`np.array(object, dtype=None)`** — 从 list/tuple 创建。

| 参数     | 含义                      | 默认值           |
| -------- | ------------------------- | ---------------- |
| `object` | 源数据（嵌套 list/tuple） | 必填             |
| `dtype`  | 指定类型                  | None（自动推断） |

```python
a = np.array([1, 2, 3])                    # 一维，shape (3,)
b = np.array([[1, 2], [3, 4]])             # 二维，shape (2, 2)
c = np.array([1, 2, 3], dtype=np.float32)  # 打印为 [1. 2. 3.]
```

### 2.2 造固定值的数组

| 函数                                   | 一句话说明                                   |
| -------------------------------------- | -------------------------------------------- |
| `np.zeros(shape, dtype=float)`         | 全 0                                         |
| `np.ones(shape, dtype=float)`          | 全 1                                         |
| `np.full(shape, fill_value)`           | 全是指定值                                   |
| `np.empty(shape)`                      | 不初始化（内容是内存残留，用完必须自己填满） |
| `np.zeros_like(a)` / `np.ones_like(a)` | 仿照已有数组的形状和类型                     |

```python
print(np.zeros((2, 3)))    # 全 0 的 2 行 3 列：[[0. 0. 0.] [0. 0. 0.]]（实际分两行打印）
print(np.ones(3))          # [1. 1. 1.]
print(np.full((2, 2), 7))  # [[7 7] [7 7]]（实际分两行打印）
print(np.zeros_like(np.ones((2, 2))).shape)   # (2, 2)
```

**ML 用途**：`np.zeros(n, dtype=int)` 初始化 KMeans 的簇标签数组；`np.zeros_like(x)` 初始化梯度。

### 2.3 造序列

**`np.arange(start, stop, step)`** — 类似 `range`，步长可以是小数；**`np.linspace(start, stop, num=50)`** — 区间内均匀取 num 个点（含两端）。

```python
print(np.arange(5))           # [0 1 2 3 4]
print(np.arange(1, 10, 2))    # [1 3 5 7 9]
print(np.arange(0, 1, 0.2))   # [0.  0.2 0.4 0.6 0.8]
print(np.linspace(0, 1, 5))   # [0.   0.25 0.5  0.75 1.  ]
```

**区别记忆**：`arange` 控制步长，`linspace` 控制点数。画图造 x 轴用 `linspace` 更稳（不受浮点步长误差影响）。

### 2.4 特殊矩阵

```python
print(np.eye(3))   # 单位矩阵（对角线 1，其余 0）：[[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]（实际分三行打印）
```

**本节速查表**：

| 需求                 | 函数                                |
| -------------------- | ----------------------------------- |
| 从 list 建数组       | `np.array`                          |
| 全 0 / 全 1 / 指定值 | `np.zeros` / `np.ones` / `np.full`  |
| 仿照已有数组         | `np.zeros_like` / `np.ones_like`    |
| 等差序列（按步长）   | `np.arange`                         |
| 均匀取点（按点数）   | `np.linspace`                       |
| 单位矩阵             | `np.eye`                            |
| 随机数组             | 见 §9，用 `np.random.default_rng()` |

---

## 3. 索引与切片 🟢

### 3.1 一维索引与切片

和 list 完全一样的语法 `a[start:stop:step]`，左闭右开：

```python
a = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]
print(a[0], a[-1], a[2:5], a[::2], a[::-1])
# 0 9 [2 3 4] [0 2 4 6 8] [9 8 7 6 5 4 3 2 1 0]
# 依次是：第 0 个、倒数第 1 个、切片 [2,5)、步长 2、反转
```

### 3.2 多维索引

**用逗号分隔各维度**，`a[行, 列]`：

```python
a = np.arange(12).reshape(3, 4)   # [[0 1 2 3] [4 5 6 7] [8 9 10 11]]（实际分三行打印）
print(a[1, 2])       # 6 —— 第 1 行第 2 列
print(a[0, :])       # [0 1 2 3] —— 第 0 行（: 表示"这一维全要"）
print(a[:, 1])       # [1 5 9] —— 第 1 列
print(a[0:2, 1:3])   # [[1 2] [5 6]] —— 前两行 × 第 1~2 列的子块
print(a[-1, :])      # [ 8  9 10 11] —— 最后一行
```

### 3.3 布尔索引（条件筛选）🟢

**用布尔数组当索引，选出 True 位置的元素**，ML 里筛数据的日常操作：

```python
a = np.array([1, 5, 3, 8, 2])
mask = a > 3            # [False  True False  True False]
print(a[mask])          # [5 8]
a[a > 3] = 0            # 条件赋值：大于 3 的全改成 0
print(a)                # [1 0 3 0 2]
```

**KMeans 真实用法**：`X[labels == k]` 取出所有属于第 k 簇的样本，再求均值得到新中心。多条件组合用 `&` / `|`，**每个条件必须加括号**：`a[(a > 1) & (a < 5)]`。

### 3.4 花式索引（整数数组索引）

**用整数数组按位置取元素，可以重复、可以乱序**：

```python
a = np.array([10, 20, 30, 40, 50])
print(a[[0, 2, 4]])     # [10 30 50]
print(a[[4, 0, 0]])     # [50 10 10] —— 乱序、重复

b = np.arange(12).reshape(3, 4)
print(b[[0, 1, 2], [3, 0, 2]])   # [3 4 10] —— 行列索引一一配对取点 (0,3)(1,0)(2,2)
```

### 3.5 视图 vs 拷贝（重点陷阱）🟢

**切片是"视图"不是拷贝——改切片会改到原数组！** 这是和 Python list 切片最大的不同：

```python
a = np.arange(5)          # [0 1 2 3 4]
b = a[1:3]                # b 是 a 的视图，共享内存
b[0] = 999
print(a)                  # [  0 999   2   3   4] —— 原数组被改了！

c = a[1:3].copy()         # .copy() 得到独立拷贝
c[0] = -1
print(a)                  # [  0 999   2   3   4] —— 不受影响
```

**口诀**：基本切片（`a[1:3]`、`a[:, 0]`）→ 视图，改它动原数组；布尔索引、花式索引 → 拷贝。拿不准就 `.copy()`，想确认就查 `b.base is a`。

**本节小结**：多维索引 `a[行, 列]`，`:` 整维全取；布尔索引筛数据 `X[labels == k]`；花式索引按位置取 `a[[0, 2, 4]]`。**切片是视图，要独立副本用 `.copy()`**。

---

## 4. 形状操作 🟢

### 4.1 reshape：改形状不改数据

**`a.reshape(new_shape)`** — 元素总数必须不变；`-1` 表示"这一维帮我算"。

```python
a = np.arange(12)
print(a.reshape(3, 4).shape, a.reshape(2, -1).shape, a.reshape(2, 2, 3).shape)
# (3, 4) (2, 6) (2, 2, 3) —— -1 自动推断；可升三维
```

### 4.2 ravel vs flatten：拉平

| 函数          | 返回 | 改返回值影响原数组？ |
| ------------- | ---- | -------------------- |
| `a.ravel()`   | 视图 | **会**               |
| `a.flatten()` | 拷贝 | 不会                 |

```python
a = np.arange(6).reshape(2, 3)
r = a.ravel()       # [0 1 2 3 4 5]
r[0] = 99
print(a[0, 0])      # 99 —— ravel 是视图，原数组被改
```

只读数据用 `ravel()`（快），要独立副本用 `flatten()`。

### 4.3 转置：`.T`

```python
a = np.arange(6).reshape(2, 3)
print(a.T)         # [[0 3] [1 4] [2 5]]（实际分三行打印）
print(a.T.shape)   # (3, 2)
```

二维下 `.T` 就是行列互换，矩阵乘前对齐形状天天用：`X.T @ X`。

### 4.4 expand_dims / squeeze：增删长度为 1 的维

```python
a = np.array([1, 2, 3])                 # shape (3,)
b = np.expand_dims(a, axis=0)           # (1, 3) —— 加了"行"维
c = np.expand_dims(a, axis=1)           # (3, 1) —— 加了"列"维
print(np.squeeze(b).shape)              # (3,) —— 去掉长度 1 的维
```

**用途**：广播对不齐形状时手动补一维（见 §5）。等价写法 `a[:, None]`（`None` = 这里加一维）。

### 4.5 拼接与切分

**`np.concatenate([a, b], axis=0)`** 沿已有轴拼接（不增维度，其余轴形状必须一致）；**`np.stack([a, b], axis=0)`** 沿**新轴**堆叠（维度 +1）。

```python
a = np.array([[1, 2], [3, 4]]); b = np.array([[5, 6]])
print(np.concatenate([a, b], axis=0))   # [[1 2] [3 4] [5 6]] —— 竖着拼
print(np.concatenate([a, a], axis=1))   # [[1 2 1 2] [3 4 3 4]] —— 横着拼

x = np.array([1, 2, 3]); y = np.array([4, 5, 6])
print(np.stack([x, y], axis=0).shape, np.vstack([x, y]).shape)   # (2, 3) (2, 3) —— 竖着堆叠
print(np.hstack([x, y]).shape, np.column_stack([x, y]).shape)    # (6,) (3, 2) —— 横着接 / 当两列
```

**`np.split(a, sections, axis=0)`** — 拼接的逆操作：

```python
a = np.arange(8)
print(np.split(a, 2))    # [array([0, 1, 2, 3]), array([4, 5, 6, 7])] —— 均分 2 份
print(np.split(a, [3]))  # [array([0, 1, 2]), array([3, 4, 5, 6, 7])] —— 在下标 3 处切开
```

**本节速查表**：

| 需求             | 函数                                               |
| ---------------- | -------------------------------------------------- |
| 改形状           | `reshape`，`-1` 自动算                             |
| 拉平             | `ravel()`（视图）/ `flatten()`（拷贝）             |
| 转置             | `.T`                                               |
| 加/删长度 1 的维 | `np.expand_dims` / `np.squeeze`（或 `a[:, None]`） |
| 沿已有轴拼接     | `np.concatenate` / `vstack` / `hstack`             |
| 沿新轴堆叠       | `np.stack`                                         |
| 切分             | `np.split`                                         |

---

## 5. 广播机制 Broadcasting 🟢（重点）

### 5.1 什么是广播

形状不同的数组做逐元素运算时，NumPy 自动把小的"拉伸"成大的形状（不实际复制数据），这就是广播：

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a + 10)   # [[11 12 13] [14 15 16]] —— 标量 10 被"拉伸"成 a 的形状
```

### 5.2 广播规则（背下来）

**从右往左对齐两个数组的 shape，逐位比较：**

1. 某一位**相等** → OK；
2. 某一位**其中一个是 1** → 拉伸到另一个的大小；
3. 某一位**既不相等也没有 1** → 报错；
4. 维数不够的数组，**左边自动补 1** 再比。

图示式例子：

```
例1（最常见）二维 + 一维：a: (2, 3)  b: (3,) → 左边补 1 变 (1,3) → 第 0 维拉伸成 2 → 结果 (2, 3)
  b = [10,20,30] 被想象成两行重复（逻辑上的，不占内存）
例2 列向量 + 行向量：a: (3, 1)  b: (1, 4) → 结果 (3, 4)，两个都被拉伸
例3 失败：a: (2, 3)  b: (4,) → 右对齐 3 vs 4，不等且无 1 → ValueError
```

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])      # shape (3,)
print(a + b)                    # [[11 22 33] [14 25 36]]

col = np.array([[100], [200]])  # shape (2, 1)，列向量
print(a + col)                  # [[101 102 103] [204 205 206]] —— 每行分别加 100 和 200
```

### 5.3 常见报错怎么排查

报错 `ValueError: operands could not be broadcast together with shapes (2,3) (4,)`，排查三步：

1. 打印两边的 `.shape`；
2. 从右往左对齐，找到第一位"既不相等也没有 1"的维度——就是它出的事；
3. 修正：reshape 数据，或用 `np.expand_dims` / `[:, None]` 把 1 补在正确位置。

```python
a = np.ones((2, 3)); b = np.ones((4,))
try:
    a + b
except ValueError as e:
    print(e)   # operands could not be broadcast together with shapes (2,3) (4,)
print((a + np.ones((2, 1))).shape)   # (2, 3) ✓ —— 想"按行加"b 应是 (3,)，"按列加"应是 (2,1)
```

### 5.4 为什么这是 ML 的核心机制

- 数据归一化 `X - X.mean(axis=0)`：(n, d) 减 (d,)。
- 加偏置 `X @ W + b`：(n, out) 加 (out,)。
- **KMeans 算距离**：(n, 1, d) 样本减 (k, d) 中心 → (n, k, d)，纯靠广播一步算完所有样本到所有中心的差。

**本节小结**：规则一句话——**从右往左对齐，相等或有一方为 1 就行，维数不够左边补 1**。报错就打印 shape，从右往左找出事的维度。

---

## 6. 逐元素运算与 ufunc 🟢

### 6.1 算术与比较运算

所有运算符（`+ - * / // % **`）在数组间都是**逐元素**的（形状相同或能广播）。注意：`*` 是逐元素乘，**不是矩阵乘**（矩阵乘用 `@`，见 §8）。

```python
a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
print(a + b, a * b, a ** 2)   # [5 7 9] [ 4 10 18] [1 4 9] —— * 是逐元素乘！
print(a > 1)                  # [False  True  True] —— 比较返回布尔数组
```

### 6.2 常用 ufunc 速查

ufunc（通用函数）对整个数组逐元素作用，一次调用替代 for 循环：

| 函数                                    | 作用                                             |
| --------------------------------------- | ------------------------------------------------ |
| `np.sqrt(x)`                            | 平方根                                           |
| `np.exp(x)`                             | e 的 x 次方（softmax、sigmoid 必用）             |
| `np.log(x)`                             | 自然对数（交叉熵损失必用）                       |
| `np.abs(x)`                             | 绝对值                                           |
| `np.maximum(a, b)` / `np.minimum(a, b)` | 逐元素取较大/较小者（ReLU = `np.maximum(x, 0)`） |
| `np.clip(x, min, max)`                  | 截断到区间（防梯度爆炸、防 log(0)）              |
| `np.square(x)` / `np.sign(x)`           | 平方 / 符号                                      |
| `np.floor` / `np.ceil` / `np.round`     | 取整                                             |

```python
x = np.array([-1.0, 0.5, 3.0])
print(np.maximum(x, 0), np.clip(x, 0, 1))   # [0.  0.5 3. ] [0.  0.5 1. ] —— maximum(x,0) 就是 ReLU
print(np.exp(np.array([0, 1])), np.sqrt(np.array([1, 4, 9])))   # [1. 2.71828183] [1. 2. 3.]
```

阶段二要手写的 sigmoid 就是 ufunc 的组合，全程逐元素、任意形状通用：

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(np.array([-1.0, 0, 1.0])))   # [0.26894142 0.5        0.73105858]
```

**本节小结**：运算符都是逐元素的（`*` ≠ 矩阵乘）。记住 `exp / log / maximum / clip` 四个，阶段二的激活函数和损失函数全靠它们。

---

## 7. 统计与聚合函数 🟢（重点中的重点）

### 7.1 常用函数一览

| 函数                          | 作用                                                             |
| ----------------------------- | ---------------------------------------------------------------- |
| `np.sum` / `np.mean`          | 求和 / 均值                                                      |
| `np.std` / `np.var`           | 标准差 / 方差（概念详见 `ml/docs/00-概率与数理统计前置知识.md`） |
| `np.min` / `np.max`           | 最小/最大值                                                      |
| `np.argmin` / `np.argmax`     | **最值所在的下标**（KMeans 分配簇必用 argmin）                   |
| `np.median` / `np.percentile` | 中位数 / 分位数                                                  |
| `np.cumsum`                   | 累计和                                                           |

```python
a = np.array([3, 1, 4, 1, 5])
print(np.sum(a), np.mean(a), np.std(a), np.median(a))   # 14 2.8 1.6（约） 3.0
print(np.argmin(a), np.argmax(a))                       # 1 4 —— 最值所在的下标
print(np.cumsum(a))                                     # [ 3  4  8  9 14]
```

### 7.2 axis 参数：往哪个方向压缩

**核心理解：`axis=k` = "沿第 k 个轴的方向走一遍，把这一路走到的值聚合成一个值"，这一维被压掉消失。**

二维数组 shape `(行, 列)`：

- `axis=0`：沿**行方向**走（竖着往下），每**列**压成一个值 → shape `(m, n)` → `(n,)`；
- `axis=1`：沿**列方向**走（横着往右），每**行**压成一个值 → shape `(m, n)` → `(m,)`。

```python
a = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
print(np.sum(a))              # 21 —— 不写 axis 全部聚合
print(np.sum(a, axis=0))      # [5 7 9] —— 每列的和：1+4, 2+5, 3+6，shape (3,)
print(np.sum(a, axis=1))      # [ 6 15] —— 每行的和：1+2+3, 4+5+6，shape (2,)
```

记忆技巧（二选一）：① **"结果 shape 里被压的那一维消失"**：(2,3) 按 axis=0 → (3,)；② **"axis 指向走的方向"**：axis=0 沿第 0 维（行）的方向边走边收集。

三维数组同理，shape `(batch, 行, 列)`：

```python
b = np.arange(24).reshape(2, 3, 4)   # shape (2, 3, 4)
print(np.sum(b, axis=0).shape, np.sum(b, axis=1).shape, np.sum(b, axis=2).shape)
# (3, 4) (2, 4) (2, 3) —— 被压的那维消失
print(np.sum(b, axis=(1, 2)).shape)  # (2,) —— axis 可给元组，一次压多维
```

**`argmin/argmax` 加 axis**：返回沿该轴最值的**下标**：

```python
d = np.array([[1, 5, 2], [4, 0, 6]])
print(np.argmin(d, axis=1))   # [0 1] —— 第 0 行最小值在下标 0，第 1 行在下标 1
```

**KMeans 真实场景**：`distances` 是 (n 样本, k 中心) 距离表，`labels = np.argmin(distances, axis=1)` 一步得到每个样本最近的中心。

### 7.3 keepdims：保住被压掉的维度

默认聚合后被压的维直接消失；`keepdims=True` 把它保留成长度 1——**目的是让结果还能和原数组广播**：

```python
a = np.array([[1, 2, 3], [4, 5, 6]])               # shape (2, 3)
print(np.mean(a, axis=0).shape)                    # (3,)
print(np.mean(a, axis=0, keepdims=True).shape)     # (1, 3)

# 经典用途：按行归一化（每行减自己的均值）
row_mean = np.mean(a, axis=1, keepdims=True)       # shape (2, 1)
print(a - row_mean)   # [[-1. 0. 1.] [-1. 0. 1.]] —— (2,3)-(2,1) 广播成功
```

不用 keepdims 的话 `(2,3) - (2,)` 会把 `(2,)` 补成 `(1,2)`，对不上直接报错——这就是 keepdims 存在的意义。

**本节小结**：`axis=k` 沿第 k 维聚合，结果 shape 里第 k 维消失（`(2,3)`：axis=0→`(3,)`，axis=1→`(2,)`）；`keepdims=True` 保住长度 1 的维以便广播。KMeans 两行背下来：`np.argmin(dist, axis=1)` 分配簇；`X[labels == k].mean(axis=0)` 更新中心。

---

## 8. 矩阵运算 🟢

> 矩阵乘法的数学规则（(m,k)@(k,n)→(m,n)、点积含义）详见 `ml/docs/00-线性代数前置知识.md`，这里只讲 NumPy 怎么写。

### 8.1 np.dot / @ / matmul 的区别

| 写法              | 说明                                                        |
| ----------------- | ----------------------------------------------------------- |
| `a @ b`（推荐）   | 矩阵乘运算符，等价 `np.matmul`，支持 batch                  |
| `np.matmul(a, b)` | 同上；高维时把最后两维当矩阵、前面的维当批                  |
| `np.dot(a, b)`    | 一维是点积、二维等同 matmul，**高维行为不同且反直觉，别用** |

**统一建议：一律写 `@`。**

```python
a = np.array([[1, 2], [3, 4]]); b = np.array([[5, 6], [7, 8]])
print(a @ b)   # [[19 22] [43 50]] —— 19 = 1*5 + 2*7
v = np.array([1, 2]); w = np.array([3, 4])
print(v @ w)   # 11 —— 一维@一维 = 点积 1*3+2*4
print(a @ v)   # [ 5 11] —— (2,2)@(2,) 矩阵乘向量
batch = np.arange(24).reshape(2, 3, 4)   # 2 个 (3,4) 矩阵
other = np.arange(8).reshape(4, 2)       # 一个 (4,2) 矩阵
print((batch @ other).shape)             # (2, 3, 2) —— 每个 batch 各自做矩阵乘
```

**形状规则（报错自查）**：`(m, k) @ (k, n) → (m, n)`，中间两个 k 必须相等。报 `matmul ... mismatch` 时打印两边 shape，不对就用 `.T` / `reshape` 调整。

### 8.2 np.linalg 常用函数 🟡（了解即可）

| 函数                    | 作用            | 一句话直觉                                   |
| ----------------------- | --------------- | -------------------------------------------- |
| `np.linalg.norm(x)`     | 范数（长度）    | 向量的"长度"，KMeans 算距离的收尾一步        |
| `np.linalg.inv(A)`      | 逆矩阵          | 矩阵版本的"倒数"                             |
| `np.linalg.solve(A, b)` | 解方程 Ax=b     | 比 `inv(A) @ b` 更快更准，工程上永远用 solve |
| `np.linalg.eig(A)`      | 特征值/特征向量 | 方阵的"固有方向和拉伸倍数"（PCA 用）         |
| `np.linalg.det(A)`      | 行列式          | 矩阵对面积的缩放倍数                         |

```python
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))   # 5.0 —— 勾股定理 sqrt(3²+4²)，这就是"长度"
A = np.array([[3.0, 1.0], [1.0, 2.0]]); b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)  # 解 3x+y=9, x+2y=8
print(x, A @ x)            # [2. 3.] [9. 8.] —— 验证 A@x 确实等于 b
```

范数、特征值的数学解释详见 `ml/docs/00-线性代数前置知识.md`。

### 8.3 einsum 入门 🟢（学习路径明确要求）

**`np.einsum('下标记号', 数组1, 数组2, ...)`** — 给每个维度起个字母名来描述运算，一行表达矩阵乘、batch 乘、求迹等。

**下标记号规则（三条）**：

1. 每个数组的每一维起一个字母，逗号分隔不同数组，`->` 后是结果的维度字母；
2. **在输入里出现、但没在 `->` 右边的字母 = 沿该维求和**（被加总消掉）；
3. 不同数组里**同名字母**的维度长度必须相等（它们两两相乘）。

**例 1：矩阵乘法 `'ij,jk->ik'`** — 读法：`C[i,k] = 对 j 求和(A[i,j] * B[j,k])`，j 不在 `->` 右边所以是求和维（字母只是名字，`'ab,bc->ac'` 等价）：

```python
A = np.array([[1, 2], [3, 4]]); B = np.array([[5, 6], [7, 8]])
print(np.einsum('ij,jk->ik', A, B))   # [[19 22] [43 50]] —— 和 A @ B 一模一样
```

**例 2：batch 矩阵乘 `'bij,bjk->bik'`** — b 在结果里所以保留（每个 batch 独立算），j 求和：

```python
batchA = np.arange(8).reshape(2, 2, 2)      # 2 个 (2,2)
batchB = np.arange(8, 16).reshape(2, 2, 2)
D = np.einsum('bij,bjk->bik', batchA, batchB)
print(D.shape)                               # (2, 2, 2)
print(np.allclose(D, batchA @ batchB))       # True —— 和 @ 的 batch 语义一致
```

**例 3：对角线求和（矩阵的迹）`'ii->'`** — 同一数组里 i 出现两次 = 取下标相等的元素（对角线），i 不在右边所以求和：

```python
M = np.array([[1, 2], [3, 4]])
print(np.einsum('ii->', M))    # 5 —— 对角线 1+4
```

**什么时候用 einsum**：需要在一个表达式里同时转置+相乘+求和时最简洁。入门阶段 `@` + `sum` 组合也够用，einsum 看得懂即可。

**本节小结**：矩阵乘一律写 `@`（形状规则 `(m,k)@(k,n)→(m,n)`，`*` 是逐元素乘别搞混）；`linalg` 记 `norm`（长度）和 `solve`（解方程）；einsum 中 `->` 右边没有的字母 = 求和维。

---

## 9. 随机数 🟢

### 9.1 现代写法：Generator API

```python
rng = np.random.default_rng(seed=42)   # 创建随机数生成器
```

| 方法                            | 作用                               | 示例                               |
| ------------------------------- | ---------------------------------- | ---------------------------------- |
| `rng.random(size)`              | [0,1) 均匀浮点                     | `rng.random((2, 3))`               |
| `rng.uniform(low, high, size)`  | [low,high) 均匀                    | `rng.uniform(-1, 1, 5)`            |
| `rng.normal(loc, scale, size)`  | 正态分布（均值 loc，标准差 scale） | `rng.normal(0, 1, 3)`              |
| `rng.integers(low, high, size)` | [low,high) 随机整数                | `rng.integers(0, 10, 5)`           |
| `rng.choice(a, size, replace)`  | 从数组抽样                         | `rng.choice(10, 3, replace=False)` |
| `rng.shuffle(x)`                | **原地**打乱（无返回值）           | `rng.shuffle(x)`                   |
| `rng.permutation(x)`            | 返回打乱的**拷贝**                 | `rng.permutation(10)`              |

```python
rng = np.random.default_rng(42)
print(rng.random(3))                      # [0.77395605 0.43887844 0.85859792]
print(rng.integers(0, 10, 5))             # 5 个 [0,10) 的整数
print(rng.normal(0, 1, 3))                # 3 个标准正态样本
print(rng.choice(10, 3, replace=False))   # 从 0~9 不重复抽 3 个
x = np.arange(5); rng.shuffle(x)          # 原地打乱，注意无返回值
```

**KMeans 真实用法**：初始化中心 `centers = X[rng.choice(len(X), k, replace=False)]`；划分训练/测试集前 `idx = rng.permutation(len(X))` 打乱。

### 9.2 为什么要设 seed

设了 seed，"随机"序列就是**确定的**，每次运行一模一样（两个 `default_rng(42)` 的 `random()` 输出逐项相等，本文已实测验证）。作用：**实验可复现**（别人能跑出一模一样的结果）、**可调试**（bug 能稳定复现）、**可公平对比**（只改一个超参数，其他随机因素不变）。

### 9.3 旧写法（看懂旧代码用）🟡

| 旧写法                        | 等价的新写法                                             |
| ----------------------------- | -------------------------------------------------------- |
| `np.random.rand(2, 3)`        | `rng.random((2, 3))`（旧写法参数是分开的维度，不是元组） |
| `np.random.randn(3)`          | `rng.standard_normal(3)`                                 |
| `np.random.randint(0, 10, 5)` | `rng.integers(0, 10, 5)`                                 |
| `np.random.seed(42)`          | `rng = np.random.default_rng(42)`                        |

新代码统一用 `default_rng`，旧代码看得懂就行。

**本节小结**：`rng = np.random.default_rng(seed)`，一切随机操作走 `rng.xxx`；记 `random / integers / normal / choice / permutation` 五个。seed = 可复现实验。

---

## 10. 常用杂项

### 10.1 np.where：数组版 if-else 🟢

```python
a = np.array([1, -2, 3, -4])
print(np.where(a > 0, a, 0))   # [1 0 3 0] —— 条件真取 a，否则取 0（又一个 ReLU 写法）
print(np.where(a > 0))         # (array([0, 2]),) —— 只传条件时返回满足条件的下标
```

### 10.2 np.unique：去重与计数

```python
labels = np.array([0, 1, 1, 2, 0, 1])
vals, counts = np.unique(labels, return_counts=True)
print(vals, counts)   # [0 1 2] [2 3 1] —— 每个值出现几次（KMeans 检查各簇样本数）
```

### 10.3 np.argsort：排序后的下标

```python
a = np.array([30, 10, 20])
idx = np.argsort(a)
print(idx, a[idx])             # [1 2 0] [10 20 30] —— 下标顺序，花式索引可还原排序结果
print(np.argsort(a)[::-1])     # [0 2 1] —— 反转即降序（找 top-K 用）
```

### 10.4 NaN 与 Inf 处理

真实数据常有缺失值 NaN，**NaN 会污染一切聚合**（任何数遇 NaN 都得 NaN），所以有 nan 系函数跳过它：

```python
a = np.array([1.0, np.nan, 3.0])
print(np.isnan(a))                                # [False  True False] —— 找 NaN 位置
print(np.isinf(np.array([1.0, np.inf])))          # [False  True]
print(np.sum(a), np.nansum(a), np.nanmean(a))     # nan 4.0 2.0 —— 被污染 vs 跳过 NaN
a[np.isnan(a)] = 0                                # 常见清洗：NaN 替换成 0
print(a)                                          # [1. 0. 3.]
```

nan 系还有 `nanstd / nanmin / nanmax / nanmedian`。

### 10.5 np.allclose：浮点数比较 🟢

浮点有精度误差，`==` 比较几乎总是错的，用 `allclose`（默认容差 rtol=1e-05, atol=1e-08）：

```python
print(0.1 + 0.2 == 0.3)                              # False —— 浮点精度坑
print(np.allclose(0.1 + 0.2, 0.3))                   # True
print(np.allclose([1.0, 2.0], [1.0, 2.000000001]))   # True
```

**手写 ML 算法的调试标配**：实现完和参考实现对拍 `assert np.allclose(my_result, expected)`。

**本节小结**：`where` 条件选择、`unique(return_counts=True)` 计数、`argsort` 排序下标；数据有缺失先 `isnan`、聚合用 nan 系；浮点比较永远 `allclose`。

---

## 11. 性能与避坑

### 11.1 向量化 vs for 循环

```python
import numpy as np
import time

n = 1_000_000
a = np.random.default_rng(0).random(n)
b = np.random.default_rng(1).random(n)
t0 = time.perf_counter()
c1 = a * b + 1                    # 向量化
t1 = time.perf_counter()
c2 = np.empty(n)
for i in range(n):                # for 循环
    c2[i] = a[i] * b[i] + 1
t2 = time.perf_counter()
print(np.allclose(c1, c2))        # True —— 结果一样
print(f"向量化: {(t1-t0)*1000:.1f} ms, for循环: {(t2-t1)*1000:.1f} ms")
# 典型输出：向量化 ~4 ms, for循环 ~200 ms —— 差 50~100 倍
```

**结论：写 NumPy 代码时，一旦出现 `for i in range(len(a))` 逐个处理元素的冲动，先停下来想"能不能用切片/广播/ufunc 表达"。** KMeans 要求的"纯 NumPy"，核心就是把双重 for 换成广播 + argmin。

### 11.2 四大陷阱速查

| 陷阱          | 现象                                    | 对策                             |
| ------------- | --------------------------------------- | -------------------------------- |
| 切片是视图    | 改切片原数组跟着变                      | 独立副本用 `.copy()`（见 §3.5）  |
| dtype 溢出    | int8 的 127+1 变 -128                   | 注意范围，必要时 astype 更大类型 |
| 隐式类型转换  | 往 int 数组塞 float，小数被**静默砍掉** | 赋值前确认 dtype                 |
| 整数/浮点除法 | `/` 总是浮点除，`//` 才是整除           | 同 Python，注意区分              |

```python
x = np.array([127], dtype=np.int8)
print(x + 1)    # [-128] + RuntimeWarning —— 溢出绕回
y = np.array([1, 2, 3]); y[0] = 3.9
print(y)        # [3 2 3] —— 3.9 被静默截断进 int 数组，不报错！
print(0.1 + 0.2 == 0.3, np.allclose(0.1 + 0.2, 0.3))   # False True
```

---

## 完整速查表

**创建**：`np.array(list, dtype=)` 从数据建数组 ｜ `zeros/ones/full(shape)` 全 0/全 1/指定值 ｜ `zeros_like/ones_like(a)` 仿形状 ｜ `arange(start,stop,step)` 等差序列 ｜ `linspace(start,stop,num)` 均匀取点 ｜ `eye(n)` 单位矩阵

**属性与形状**：`a.shape/ndim/size/dtype` 形状/维数/总数/类型 ｜ `reshape(-1)` 改形状，-1 自动算 ｜ `ravel()/flatten()` 拉平（视图/拷贝） ｜ `.T` 转置 ｜ `expand_dims(a,axis)/squeeze(a)` 增删长度 1 的维 ｜ `concatenate/vstack/hstack` 沿已有轴拼接 ｜ `stack` 沿新轴堆叠 ｜ `split` 切分

**索引**：`a[i,j]` 多维索引 ｜ `a[行切片, 列切片]` 切片（视图！） ｜ `a[a>0]` 布尔索引 ｜ `a[[0,2,4]]` 花式索引 ｜ `a.copy()` 独立拷贝

**运算与 ufunc**：`+ - * / **` 逐元素运算（`*` 不是矩阵乘） ｜ `sqrt/exp/log/abs` 逐元素数学函数 ｜ `maximum/minimum` 逐元素取最值 ｜ `clip(x,lo,hi)` 截断

**统计聚合**：`sum/mean/std/var` 和/均值/标准差/方差 ｜ `min/max` 最值 ｜ `argmin/argmax` 最值下标（KMeans 分配簇） ｜ `median/percentile` 中位数/分位数 ｜ `cumsum` 累计和 ｜ `axis=k` 沿第 k 维聚合，该维消失 ｜ `keepdims=True` 保留长度 1 的维以便广播

**矩阵运算**：`a @ b` 矩阵乘（含 batch） ｜ `np.dot` 点积（高维有坑，别用） ｜ `linalg.norm/inv/solve/eig/det` 范数/逆/解方程/特征值/行列式 ｜ `einsum('ij,jk->ik',A,B)` 下标运算，`->` 外字母求和

**随机数**：`np.random.default_rng(seed)` 建生成器 ｜ `rng.random/uniform/normal/integers` 均匀/区间均匀/正态/整数 ｜ `rng.choice/shuffle/permutation` 抽样/原地打乱/打乱拷贝

**杂项**：`where(cond,x,y)` 条件选择 ｜ `unique(a,return_counts=True)` 去重计数 ｜ `argsort` 排序下标 ｜ `isnan/isinf` 找 NaN/Inf ｜ `nanmean` 等跳过 NaN 聚合 ｜ `allclose` 浮点近似相等

---

## 自测题（附答案）

1. **（视图）** 执行 `a = np.arange(6); b = a[::2]; b[0] = 99` 后，`a` 是什么？为什么？
2. **（广播）** 判断每个运算的结果 shape 或报错：a) `(3,1,4)` 与 `(5,4)`；b) `(2,3)` 与 `(2,)`；c) `(3,)` 与 `(1,)`。
3. **（axis）** `x` 的 shape 是 `(100, 20)`（100 个样本、每个 20 维特征）。求"每个特征的均值"和"每个样本的均值"分别怎么写？结果 shape 各是什么？
4. **（KMeans 核心一步）** 样本矩阵 `X`（shape `(100, 2)`）、3 个中心 `centers`（shape `(3, 2)`）。用广播 + argmin 计算每个样本到每个中心的欧氏距离，并给出每个样本最近的中心下标。（提示：把 X 变成 `(100, 1, 2)`）
5. **（einsum）** 写出与 `A @ B` 等价的 einsum 表达式；再写一个求 A 主对角线元素之和的表达式。
6. **（随机数）** 从 200 个样本里可复现地随机不重复抽 5 个作为 KMeans 初始中心，写代码。
7. **（调试）** 你手写的 `sigmoid(z)` 和参考答案"看起来一样"但 `==` 全是 False，最可能的原因是什么？怎么验证？
8. **（性能）** 逐样本算距离 `for i in range(len(X)): dists[i] = np.sqrt(np.sum((X[i] - center) ** 2))` 有什么问题？怎么改写？

### 参考答案

**1.** `a` 变成 `[99, 1, 2, 3, 4, 5]`。`a[::2]` 是切片即视图，`b` 与 `a` 共享内存，改 `b[0]` 就是改 `a[0]`。避免：`b = a[::2].copy()`。

**2.** a) `(3,5,4)`——右对齐 `(3,1,4)` vs `(_,5,4)`：4=4，1 拉伸到 5，补 1 拉伸到 3。b) 报错——右对齐 3 vs 2 不等且无 1；修正：`b.reshape(2,1)` 或 `b[:, None]`。c) `(3,)`——1 拉伸到 3。

**3.** 每个特征的均值 `x.mean(axis=0)` → `(20,)`；每个样本的均值 `x.mean(axis=1)` → `(100,)`。

**4.**

```python
diff = X[:, None, :] - centers[None, :, :]   # (100,1,2)-(1,3,2) 广播 → (100,3,2)
dist = np.linalg.norm(diff, axis=2)          # 沿最后一维求长度 → (100, 3)
labels = np.argmin(dist, axis=1)             # 每个样本最近的中心下标 → (100,)
```

（`dist` 也可写 `np.sqrt(np.sum(diff**2, axis=2))`，等价。）

**5.** 矩阵乘 `np.einsum('ij,jk->ik', A, B)`；对角线求和 `np.einsum('ii->', A)`。

**6.**

```python
rng = np.random.default_rng(42)          # 固定 seed 保证可复现
idx = rng.choice(200, 5, replace=False)
initial_centers = X[idx]
```

**7.** 浮点精度误差：不同计算路径得到的浮点结果末位有微小差异，`==` 逐位比较必然失败。验证：`np.allclose(my_sigmoid(z), expected)` 应为 True，或打印 `np.max(np.abs(my - expected))` 看误差量级。

**8.** for 循环逐样本计算，慢几十到上百倍且完全没必要。向量化改写（`X:(n,d)` 与 `center:(d,)` 广播，一行算完）：`dists = np.linalg.norm(X - center, axis=1)`

---

## 与学习路径的对应关系

| 本文章节     | 学习清单条目           | 阶段一 KMeans 用到哪里                   | 阶段二用到哪里                 |
| ------------ | ---------------------- | ---------------------------------------- | ------------------------------ |
| §1 ndarray   | 基础                   | 数据全是 ndarray                         | 权重、梯度全是 ndarray         |
| §2 数组创建  | 数组创建               | `zeros` 初始化 labels                    | 初始化权重/梯度                |
| §3 索引切片  | 索引与切片             | `X[labels==k]` 按簇取样                  | 数据集切分、mini-batch 取数    |
| §4 形状操作  | （支撑技能）           | reshape 数据                             | 对齐矩阵乘维度                 |
| §5 广播      | 广播机制               | `(n,1,d)-(k,d)` 一步算全距离             | `X@W+b` 加偏置、归一化         |
| §6 ufunc     | （支撑技能）           | `diff**2`                                | sigmoid/softmax/ReLU、log 损失 |
| §7 统计聚合  | 统计函数（axis、聚合） | `argmin` 分配簇、`mean(axis=0)` 更新中心 | 损失求均值、batch 梯度平均     |
| §8 矩阵运算  | dot/linalg/einsum      | `norm` 求距离                            | `X@W`、反向传播链式矩阵乘      |
| §9 随机数    | 随机数生成             | `choice` 初始化中心、`permutation` 打乱  | 权重初始化、mini-batch 抽样    |
| §10 杂项     | （支撑技能）           | `unique` 检查簇分布                      | `allclose` 梯度检查            |
| §11 性能避坑 | （贯穿始终）           | for 循环换向量化                         | 反向传播全是向量化矩阵运算     |

**下一步**：本文 → Matplotlib（画损失曲线、KMeans 聚类散点图）→ Pandas，然后回到 NumPy 手写 KMeans 完成阶段一里程碑。
