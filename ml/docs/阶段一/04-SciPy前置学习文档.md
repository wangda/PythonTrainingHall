# 04 - SciPy 前置学习文档（直觉导览 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：SciPy 模块地图、`scipy.stats` 概率分布与假设检验、`scipy.optimize` 最小化与曲线拟合、`scipy.spatial` 距离计算、`scipy.sparse` 稀疏矩阵、`scipy.linalg` 与 NumPy 的关系、`interpolate`/`integrate` 知道存在即可、官方文档查阅方法。
> **本文环境**：SciPy 1.18（所有示例已用 SciPy 1.18.1 + NumPy 2.x 实际运行验证）。
> **定位**：你的学习计划对 SciPy 的要求是"按需，不单独学，用到什么查什么"。本文就是为此写的——**建立直觉 + 留一张地图**，让你知道 SciPy 里有什么、每个模块解决什么问题、用到时能想起来查哪里。

---

## 本文怎么用

**本文不用逐节练习，通读一遍建立直觉即可，用到再回来查。** 和前三份文档（NumPy / Matplotlib / Pandas 都要动手敲）不同，SciPy 的正确打开方式是"知道它有什么"。

建议读法：

1. **通读全文**，代码示例看懂逻辑、愿意的话跑一遍有个感觉就行，不要求默写参数。
2. 标【要有直觉】的章节（§2~§5）认真读，这几块在 ML 里真的会遇到；
3. 标【用到再查】的章节（§6~§8）混个脸熟，记住"有这么个东西"就达标；
4. 以后真用到某个功能，先翻文末【速查表】定位到模块和函数，再按 §9 的方法查官方文档。

**先决知识**：NumPy（SciPy 的输入输出基本都是 ndarray），见 `ml/docs/01-numpy前置学习文档.md`。文中涉及的数学概念（正态分布、p 值、导数等）一律浅讲，需要深入时引用 `ml/docs/00-概率与数理统计前置知识.md`、`ml/docs/00-微积分前置知识.md`、`ml/docs/00-线性代数前置知识.md`。

**阅读约定**：【要有直觉】= ML 里会遇到，要理解它在干什么；【用到再查】= 知道存在即可，真用到再翻文档。

**全文约定**：示例默认已 `import numpy as np`。

---

## 1. SciPy 是什么

**一句话：SciPy 是建立在 NumPy 之上的科学计算工具箱。** NumPy 管"数组怎么存、怎么算"（数据结构 + 基础运算），SciPy 管"各科学领域的现成算法"（统计、优化、积分、信号处理……）。NumPy 给你砖头和水泥，SciPy 给你预制板——盖房子不用从烧砖开始。

官方原话是 SciPy 为数学、科学、工程领域提供基础算法。对你来说重要的是：**这些算法都是几十年验证过的成熟实现，正确性和性能都比自己写好，遇到对应问题直接用，不要自己造轮子。**

### 1.1 模块地图

SciPy 按领域组织成十几个子模块，用的时候 `from scipy import 模块名`。全表如下，重点看"ML 里哪里会遇到"这一列：

| 模块 | 干什么用 | ML 里哪里会遇到 | 重要度 |
|---|---|---|---|
| `stats` | 概率分布、统计检验 | 采样、理解数据分布、A/B 测试、假设检验 | ★★★【要有直觉】 |
| `optimize` | 求函数最小值、曲线拟合 | 所有"训练模型"的本质就是 minimize；逻辑回归等传统模型内部就调它 | ★★★【要有直觉】 |
| `sparse` | 稀疏矩阵（只存非零元素） | 文本词袋、推荐系统、sklearn 的 CountVectorizer 输出 | ★★【用到再查】 |
| `spatial` | 距离计算、空间数据结构 | KMeans / KNN 算样本间距离 | ★★【要有直觉】 |
| `linalg` | 线性代数（NumPy linalg 的超集） | 偶尔见到 `pinv`、`block_diag` 等 | ★【用到再查】 |
| `interpolate` | 插值（已知几个点，估计中间的点） | 数据预处理偶尔用 | ★【知道存在】 |
| `integrate` | 数值积分 | ML 里基本不用手写 | ★【知道存在】 |
| `signal` | 信号处理（滤波等） | 音频/传感器领域才用 | ★【知道存在】 |
| `special` | 特殊数学函数（gamma 等） | 偶尔在公式实现里见到 | ★【知道存在】 |
| `fftpack` / `ndimage` / `io` 等 | 傅里叶变换 / 图像处理 / 文件读写 | 各有专门领域，ML 入门基本不碰 | ★【知道存在】 |

**记住前四个加粗的就够了**：`stats`、`optimize`、`sparse`、`spatial`。后面的模块很多人工作几年都用不到一次，知道它们存在、真有需求时能搜到就行。

**本节小结**：SciPy = NumPy 之上的"各领域算法成品库"。模块按领域组织，ML 常用的是 stats / optimize / sparse / spatial 四个，其余知道存在即可。

---

## 2. scipy.stats：概率分布【要有直觉】

这是 SciPy 里和 ML 关系最密的模块。概率分布的概念（什么是分布、正态分布、均值方差）详见 `ml/docs/00-概率与数理统计前置知识.md`，这里只讲**怎么在代码里用**。

### 2.1 统一接口四件套

`scipy.stats` 里最妙的设计：**所有分布的用法一模一样**。不管你用正态分布、均匀分布还是二项分布，都是这四个方法：

| 方法 | 一句话说明 | 生活类比 |
|---|---|---|
| `rvs(size=n)` | 随机采样 n 个数 | 从这个分布里"抽盲盒" |
| `pdf(x)` | x 处的概率密度（曲线在 x 处多高） | "身高 170cm 这个位置，分布曲线有多高" |
| `cdf(x)` | 累计概率：随机取一个值，≤ x 的概率 | "身高不超过 170cm 的人占多大比例" |
| `ppf(q)` | cdf 的反函数：累计概率 q 对应哪个 x | "前 97.5% 的人身高不超过多少" |

`pdf` 和 `cdf` 是一对：`pdf` 是曲线的高度，`cdf` 是曲线下从 -∞ 到 x 的面积。`ppf` 是 `cdf` 的倒着查。这三者的关系详见概率文档，这里只要会调用。

用标准正态分布 `norm` 完整演示一次：

```python
from scipy import stats

# 标准正态分布：均值 0，标准差 1
std_norm = stats.norm(loc=0, scale=1)

samples = std_norm.rvs(size=5, random_state=42)   # 抽 5 个样本（random_state 固定结果）
print(samples)
# [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]

print(std_norm.pdf(0))      # 0.3989422804014327 —— 钟形曲线在 0 点最高
print(std_norm.cdf(0))      # 0.5 —— 一半的值小于 0（均值），完全对称
print(std_norm.cdf(1.96))   # 0.9750021048517795 —— 约 97.5% 的值 ≤ 1.96
print(std_norm.ppf(0.975))  # 1.959963984540054 —— 反查回来正好是 1.96

print(std_norm.mean(), std_norm.std())   # 0.0 1.0 —— 还能直接查均值、标准差
```

读代码见到 `cdf(1.96) ≈ 0.975` 这种数别慌——它就是"正态分布里 97.5% 的值落在 1.96 以内"，是概率文档里"钟形曲线下面积"的代码版。

### 2.2 两种写法：冻结对象 vs 每次传参

上面的 `stats.norm(loc=0, scale=1)` 先造出一个"参数固定好的分布对象"，再调它的方法，这叫**冻结（frozen）写法**。也可以不建对象，每次调用时把参数传进去：

```python
# 冻结写法：建一次对象，反复用
std_norm = stats.norm(loc=0, scale=1)
std_norm.pdf(0)          # 0.3989...

# 每次传参写法：用一下就扔
stats.norm.pdf(0, loc=0, scale=1)   # 0.3989...（结果完全一样）
```

两种写法结果一样。**同一个分布要反复用时用冻结写法**（参数只写一次，不易错）；一次性计算就用每次传参。别人的代码里两种都会见到，认出来是同一回事就行。

### 2.3 常用分布速览

| 分布 | 一句话场景 | 调用示例 |
|---|---|---|
| `norm` | 正态分布：自然界默认分布（身高、测量误差） | `norm.rvs(loc=170, scale=10, size=100)` |
| `uniform` | 均匀分布：每个值一样可能（掷骰子的连续版） | `uniform.rvs(loc=0, scale=1, size=100)`（0~1 均匀） |
| `binom` | 二项分布：抛 n 次硬币，正面朝上几次 | `binom.pmf(7, n=10, p=0.5)` → 0.117（抛 10 次正好 7 次正面的概率） |
| `bernoulli` | 伯努利分布：抛一次硬币（二项分布 n=1 的特例） | `bernoulli.rvs(p=0.3, size=10)`（用户点不点击） |
| `t` | t 分布：样本少时的"谨慎版"正态分布 | 假设检验内部用（见 §3），一般不直接调 |
| `chi2` | 卡方分布：分类数据的检验用 | 同上，`chi2_contingency` 内部用 |

注意离散分布（binom、bernoulli）用 `pmf`（概率质量函数，"恰好等于 x 的概率"）代替 `pdf`，其余三个方法通用。

**ML 里的用途**：生成模拟数据验证算法、理解模型的概率假设（逻辑回归假设伯努利分布、朴素贝叶斯假设正态分布）、初始化神经网络权重（`norm.rvs`）。

**本节小结**：所有分布共享四件套 `rvs / pdf(pmf) / cdf / ppf`。`rvs` 采样、`pdf` 曲线高度、`cdf` 累计概率、`ppf` 反查分位数。冻结写法和每次传参写法等价。常用分布认个脸熟：norm、uniform、binom、bernoulli。

---

## 3. 假设检验直觉：p 值是什么【要有直觉】

场景：**A/B 测试**。你给网站换了个新按钮，旧按钮 A 组 100 人平均点击率 5.0%，新按钮 B 组 100 人平均 5.4%。B 组高 0.4 个百分点——**这是新按钮真的好，还是纯粹运气好？**

**p 值（p-value）的直觉定义**：

> 假设两组其实**没差别**（新按钮啥用没有），纯粹靠运气，出现"这么大甚至更大差异"的概率。

- p 值**小**（惯例 < 0.05）：没差别的话很难靠运气出现这么大差异 → 倾向于认为"真有差别"；
- p 值**大**：这点差异靠运气就能出现 → 没有证据证明有差别。

0.05 是业界惯例的"显著性水平"，不是什么自然常数。原理推导（零假设、检验统计量）详见 `ml/docs/00-概率与数理统计前置知识.md`，**这里的要求只有一条：会读 p 值**。

### 3.1 t 检验：比较两组连续数据的均值

**`stats.ttest_ind(a, b)`** — 独立双样本 t 检验，比较两组数据的均值是否有显著差异。

| 参数 | 含义 |
|---|---|
| `a`, `b` | 两组样本（一维数组） |
| 返回值 | `(t统计量, p值)`，一般只接 p 值 |

```python
from scipy import stats
import numpy as np

rng = np.random.default_rng(42)
clicks_a = rng.normal(5.0, 1.0, size=100)   # 旧按钮：100 人，真实均值 5.0
clicks_b = rng.normal(5.4, 1.0, size=100)   # 新按钮：100 人，真实均值 5.4

t_stat, p_value = stats.ttest_ind(clicks_a, clicks_b)
print(t_stat)    # -3.5169057797820913 —— 检验统计量，不用管含义
print(p_value)   # 0.000541344811667655 —— p ≈ 0.0005 << 0.05
```

p ≈ 0.0005，远小于 0.05："如果两个按钮真没差别，靠运气出现这么大差异的概率只有万分之五" → 有理由相信新按钮更好。

### 3.2 卡方检验：比较分类数据的比例

上面比较的是"平均点击率"（连续值）。如果数据本身就是"点击/未点击"两种类别，数人头做检验，用卡方检验。

**`stats.chi2_contingency(table)`** — 列联表卡方检验，判断"分组"和"结果"两个分类变量是否相关。

| 参数 | 含义 |
|---|---|
| `table` | 二维计数表：行 = 分组，列 = 结果 |
| 返回值 | `(卡方统计量, p值, 自由度, 期望频数表)`，一般只接 p 值 |

```python
table = np.array([[40, 60],    # 旧按钮：40 人点击，60 人未点击
                  [55, 45]])   # 新按钮：55 人点击，45 人未点击

chi2, p, dof, expected = stats.chi2_contingency(table)
print(p)         # 0.04743689282486553 —— p ≈ 0.047 < 0.05
print(expected)
# [[47.5 52.5]
#  [47.5 52.5]] —— "假如两组没差别"时每个格子理论上应该有的人数
```

p ≈ 0.047，勉强小于 0.05：有一点点证据说新旧按钮的点击率不同，但远不如 t 检验那个例子铁。这也顺便展示了"0.05 只是个惯例"——0.047 和 0.053 没有本质区别，别把 p 值当圣旨。

**本节小结**：p 值 = "两组其实没差别时，纯靠运气出现这么大差异的概率"；p < 0.05 惯例上认为"差异显著"。连续数据比均值用 `ttest_ind`，分类数据数人头用 `chi2_contingency`，都只读返回的 p 值即可。

---

## 4. scipy.optimize：给函数找最小值【要有直觉】

### 4.1 minimize 是什么

**一句话：给任何一个函数，`minimize` 自动帮你找它的最小值（和取得最小值的 x）。**

这就是梯度下降、牛顿法这些算法的"成品版"。你在 `ml/docs/00-微积分前置知识.md` 里学过的"导数为 0 的地方是极值""沿负梯度方向走能下山"，`minimize` 内部全帮你做了——你不用手写迭代循环，不用自己调学习率。

**为什么这是核心**：**训练模型的本质就是 minimize**。损失函数 = "模型预测得有多烂"，训练 = 调整参数让损失最小。你阶段二手写梯度下降是为了学原理；实际中逻辑回归、SVM 等传统模型的内部就是调用这样的优化器。

**`optimize.minimize(fun, x0)`** — 求函数最小值。

| 参数 | 含义 |
|---|---|
| `fun` | 要最小化的函数，输入是一个数组 x（哪怕只有一个自变量也传数组），返回一个数 |
| `x0` | 从哪个点开始找（初始猜测），形状和 x 一致 |
| 返回值 | 结果对象：`.x` 最优解、`.fun` 最小值、`.success` 是否成功、`.message` 说明 |

```python
from scipy import optimize

def f(x):
    return (x[0] - 3) ** 2 + 5     # 抛物线，肉眼可见 x=3 时最小，最小值 5

result = optimize.minimize(f, x0=[0.0])   # 从 x=0 出发找
print(result.x)        # [3.00000003] —— 找到了 x≈3
print(result.fun)      # 5.000000000000001 —— 最小值≈5
print(result.success)  # True
print(result.message)  # Optimization terminated successfully.
```

注意两点：① 哪怕只有一个自变量，函数也要写成收数组 `x[0]` 的形式；② 结果是 3.00000003 不是精确的 3——数值优化都是近似解，够用就行。

### 4.2 curve_fit：给数据找最佳参数

场景：你有一堆带噪声的 (x, y) 数据点，猜它们是条直线 y = ax + b，**但 a、b 是多少？** `curve_fit` 帮你找。

**`optimize.curve_fit(f, xdata, ydata)`** — 曲线拟合：给定函数形式和数据，自动找最优参数。

| 参数 | 含义 |
|---|---|
| `f` | 猜的函数形式，第一个参数是自变量，后面是待求参数：`f(x, a, b, ...)` |
| `xdata`, `ydata` | 观测数据 |
| 返回值 | `(popt, pcov)`：popt = 最优参数数组；pcov = 参数的协方差（入门不用管） |

```python
def line(x, a, b):
    return a * x + b               # 我猜数据是条直线，a、b 未知

x_data = np.linspace(0, 10, 20)
rng = np.random.default_rng(42)
y_data = 2.0 * x_data + 1.0 + rng.normal(0, 0.5, size=20)  # 真实直线 + 噪声

popt, pcov = optimize.curve_fit(line, x_data, y_data)
print(popt)   # [2.02411524 0.86295774] —— 拟合出 a≈2.02, b≈0.86
```

真实的 a=2.0、b=1.0，拟合出 2.02 和 0.86——有噪声所以不完全准，但很接近。**这其实就是最小二乘版的线性回归**：sklearn 的 `LinearRegression` 干的就是这件事。

### 4.3 和 ML 训练的关系

| 你在 ML 里做的事 | 对应的 optimize 概念 |
|---|---|
| 定义损失函数（MSE、交叉熵） | `fun` |
| 随机初始化权重 | `x0` |
| 训练 = 迭代更新权重 | `minimize` 内部的迭代 |
| 手写梯度下降（阶段二） | 原理学习版；成品优化器内部就是梯度类算法 |

**本节小结**：`minimize(fun, x0)` 给任何函数找最小值，函数收数组返回标量，结果读 `.x` / `.fun` / `.success`。`curve_fit(f, x, y)` 给数据找最佳参数，本质是最小二乘回归。训练模型 = minimize 损失函数，这是理解 ML 训练的一把总钥匙。

---

## 5. scipy.spatial：距离计算【要有直觉】

KMeans 的核心一步：每个样本离哪个聚类中心最近，就归哪类。这就是"算距离"，也正是你的阶段一里程碑。

### 5.1 cdist：两组点之间的距离矩阵

**`spatial.distance.cdist(XA, XB)`** — 算 XA 里每个点到 XB 里每个点的距离，返回距离矩阵。

| 参数 | 含义 |
|---|---|
| `XA` | 点集，形状 `(m, 特征数)` |
| `XB` | 点集，形状 `(n, 特征数)` |
| 返回值 | `(m, n)` 矩阵，`[i, j]` = XA 第 i 个点到 XB 第 j 个点的欧氏距离 |

```python
from scipy import spatial

X = np.array([[1, 2], [2, 1], [8, 9], [9, 8]])   # 4 个样本（二维点）
centers = np.array([[1, 1], [9, 9]])              # 2 个聚类中心

dist = spatial.distance.cdist(X, centers)
print(dist)
# [[ 1.         10.63014581]    # 样本0：离中心0距离1，离中心1距离10.6
#  [ 1.         10.63014581]    # 样本1：同上
#  [10.63014581  1.        ]    # 样本2：离中心1近
#  [10.63014581  1.        ]]

labels = dist.argmin(axis=1)     # 每个样本取距离最小的中心的下标
print(labels)   # [0 0 1 1] —— 前两个样本归簇0，后两个归簇1
```

**这就是 KMeans 的分配步骤，一行搞定。** 你在 NumPy 文档（`ml/docs/01-numpy前置学习文档.md` §5 广播）里见过的手写写法是：

```python
dist_manual = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
```

两者结果完全一样。手写广播版帮你理解原理，`cdist` 是日常用的成品。同一家族的还有 `pdist(X)`（算一组点内部两两之间的距离，返回压缩成一维的结果），KMeans 用不到，做层次聚类时会遇到。

### 5.2 单个距离：euclidean / cosine

```python
a = np.array([1, 2]); b = np.array([4, 6])
print(spatial.distance.euclidean(a, b))   # 5.0 —— sqrt((4-1)² + (6-2)²) = sqrt(9+16)

u = np.array([1, 0]); v = np.array([0, 1]); w = np.array([1, 1])
print(spatial.distance.cosine(u, v))   # 1.0 —— 两个向量垂直，毫无相似
print(spatial.distance.cosine(u, w))   # 0.29289321881345254 —— 方向有重叠
```

注意：**cosine 算的是"距离"不是"相似度"**——值越小方向越像（完全同向 = 0，垂直 = 1）。文本相似度、推荐系统常用余弦；看到 `1 - cosine(...)` 就是在把它换算回相似度。

**本节小结**：`cdist(A, B)` 算两组点两两距离（KMeans 分配步骤），`pdist(X)` 算一组点内部两两距离，`euclidean` / `cosine` 算两个向量间的单个距离。cosine 输出的是距离，越小越像。

---

## 6. scipy.sparse：稀疏矩阵【用到再查】

### 6.1 直觉：为什么需要它

想象一张 **100 万用户 × 10 万商品** 的评分表。每个用户只评过几十个商品，表里 99.999% 的格子是空的（0）。

```python
n_rows, n_cols = 1_000_000, 100_000
print(n_rows * n_cols * 8)   # 800000000000 —— 全存下来要 800 GB 内存，直接爆炸
```

**稀疏矩阵的思路：只存非零元素**——"第几行第几列是什么值"，其余默认是 0。3 个非零值就只花 3 个值的内存，800 GB 变几 KB。

**ML 里哪里遇到**：

- **文本词袋 / TF-IDF**：sklearn 的 `CountVectorizer` / `TfidfVectorizer` 输出的就是稀疏矩阵（几万个词，一篇文章只用到几十个）；
- **推荐系统**：上面说的用户 × 商品评分表；
- 看到 sklearn 报错或输出里出现 `csr_matrix` 字样，就是它了。

### 6.2 csr_matrix 基本用法

`csr_matrix` 是最常用的稀疏格式（CSR = 按行压缩存储，取行很快）。创建、转回普通数组、取非零个数：

```python
from scipy import sparse

dense = np.array([[1, 0, 0, 2],
                  [0, 0, 3, 0],
                  [0, 0, 0, 0]])
csr = sparse.csr_matrix(dense)    # 从普通数组创建
print(csr)
# <Compressed Sparse Row sparse matrix of dtype 'int64'
#  	with 3 stored elements and shape (3, 4)>
#    Coords	Values
#    (0, 0)	1
#    (0, 3)	2
#    (1, 2)	3 —— 只存了 3 个非零元素的位置和值

print(csr.nnz)           # 3 —— 非零元素个数（number of non-zeros）
print(csr.shape)         # (3, 4)
print(csr.toarray())     # 转回普通二维数组（矩阵大时千万别这么干，内存会炸）
print(csr[1].toarray())  # [[0 0 3 0]] —— 取第 1 行（CSR 取行快）
```

稀疏矩阵支持和普通数组一样的矩阵乘法（`csr @ 向量`），sklearn 模型可以直接吃稀疏矩阵，**大多数时候你不需要手动创建它**——`CountVectorizer.fit_transform()` 直接返回给你。你要做的是：见到 `csr_matrix` 不陌生，知道 `.toarray()` 能转回普通数组（但大矩阵别转），知道 `.nnz` / `.shape` 怎么看。

**本节小结**：稀疏矩阵只存非零元素，解决"巨大但几乎全空"的矩阵内存爆炸问题。ML 里主要作为 `CountVectorizer` / `TfidfVectorizer` 的输出被动遇到。认三个属性：`.nnz`（非零个数）、`.shape`、`.toarray()`（转回普通数组）。

---

## 7. scipy.linalg 与 numpy.linalg 的关系【用到再查】

**一句话：`scipy.linalg` 是 `numpy.linalg` 的超集，功能更全；日常用 `numpy.linalg` 就够。**

`numpy.linalg` 有的（`inv`、`solve`、`norm`、`eig`、`svd`），`scipy.linalg` 全都有且同名同用法；`scipy.linalg` 还多一些 NumPy 没有的分解和工具。日常 ML 代码里你见到的基本都是 `np.linalg`，见到 `scipy.linalg` 时知道它是"更全的版本"就行。

举两个 NumPy 里没有的，**知道存在即可**：

```python
from scipy import linalg

# lu 分解：把矩阵拆成 下三角 × 上三角（解方程组的内部步骤）
A = np.array([[2., 1.], [4., 3.]])
P, L, U = linalg.lu(A)
print(L)
# [[1.  0. ]
#  [0.5 1. ]]

# block_diag：把几个矩阵沿对角线拼成一个大矩阵，其余位置补 0
bd = linalg.block_diag([1, 2], [[3, 4], [5, 6]])
print(bd)
# [[1 2 0 0]
#  [0 0 3 4]
#  [0 0 5 6]]
```

类似还有 `pinv`（伪逆，NumPy 其实也有 `np.linalg.pinv`）、`expm`（矩阵指数）等。用到哪个查哪个，不用提前学。线性代数概念（逆矩阵、分解）详见 `ml/docs/00-线性代数前置知识.md`。

**本节小结**：`scipy.linalg` ⊇ `numpy.linalg`，同名函数用法一样。日常用 `np.linalg`；见到 `scipy.linalg` 的 `lu`、`block_diag` 等，知道是"NumPy 没有的补充工具"即可。

---

## 8. interpolate 与 integrate【知道存在即可】

这两个模块 ML 入门基本用不到，各给一句话 + 一个例子混脸熟。

### 8.1 interpolate：已知几个点，估计中间的点

**`interpolate.interp1d(x, y)`** — 已知 x 处的 y 值，造一个函数，可以查任意中间 x 的估计值（默认直线连接相邻点）。

```python
from scipy import interpolate

x_known = np.array([0, 1, 2, 3])
y_known = np.array([0, 1, 4, 9])     # 其实来自 y = x²，但 interp1d 不知道

f = interpolate.interp1d(x_known, y_known)
print(f(1.5))           # 2.5 —— 1 和 2 之间直线连出来的估计（真值是 2.25）
print(f([0.5, 2.5]))    # [0.5 6.5] —— 一次查多个点
```

用途：数据预处理时补全缺失的采样点。更平滑的版本是三次样条 `CubicSpline`，用到再查。

### 8.2 integrate：算定积分

**`integrate.quad(f, a, b)`** — 数值计算定积分 ∫ₐᵇ f(x)dx（曲线下面积，概念详见 `ml/docs/00-微积分前置知识.md`）。

```python
from scipy import integrate

area, err = integrate.quad(lambda x: x ** 2, 0, 3)
print(area, err)   # 9.000000000000002 9.99...e-14 —— ∫₀³ x²dx = 9，误差约 10⁻¹³
```

返回 `(积分值, 估计误差)`。ML 里基本不用手算积分（期望、概率计算都有现成的解析形式或采样方法），知道有这功能就行。

**本节小结**：`interp1d` 已知若干点估计中间点；`integrate.quad` 算定积分。两者在 ML 入门阶段都很少直接用，知道存在即可。

---

## 9. 怎么查 SciPy

SciPy 太大，没人能全记住。**会用文档比记住函数重要。**

### 9.1 官方文档结构

官网 `docs.scipy.org/doc/scipy/` 的 API 参考就是**按模块组织的**，和本文 §1.1 的模块地图一一对应。每个模块页面列出该模块所有函数，点进函数页面有：参数说明、返回值、Examples（可照抄的示例，**先看这个**）。

### 9.2 help() 与 docstring

```python
from scipy import stats
help(stats.norm.pdf)     # 或 print(stats.norm.pdf.__doc__)
```

docstring 末尾的 Examples 部分可以直接抄进解释器跑。Jupyter 里用 `stats.norm.pdf?` 更快捷。

### 9.3 遇到需求时的查找路径

三步走：

1. **先想属于哪个领域** → 对照 §1.1 模块地图（或文末速查表）定位到模块。例："要算两个文本向量的相似度" → 距离问题 → `scipy.spatial`。
2. **在模块里找函数** → 官方文档模块页扫一遍函数名单，或用搜索引擎搜 "scipy 你的需求关键词"（如 `scipy chi square test`）。
3. **读 docstring 的 Examples** → 照抄改成自己的数据，先跑通再理解参数。

**本节小结**：官方文档按模块组织；`help(函数)` 看 docstring，重点抄 Examples；查找路径 = 定领域 → 找模块 → 搜函数 → 抄示例。

---

## 速查表（需求 → 模块 → 函数）

| 我想干什么 | 模块 | 函数 |
|---|---|---|
| 从正态/均匀/二项分布采样 | `stats` | `norm.rvs / uniform.rvs / binom.rvs`（加 `random_state` 固定结果） |
| 查分布曲线某点的高度 | `stats` | `norm.pdf(x)`（离散分布用 `.pmf(x)`） |
| 查"≤ x 的概率" | `stats` | `norm.cdf(x)` |
| 反查"前 q% 不超过多少" | `stats` | `norm.ppf(q)` |
| 算两组连续数据的均值是否有显著差异 | `stats` | `ttest_ind(a, b)` → 读 p 值 |
| 算分类数据两组比例是否有显著差异 | `stats` | `chi2_contingency(table)` → 读 p 值 |
| 给任意函数找最小值 | `optimize` | `minimize(fun, x0)` → 读 `.x` / `.fun` / `.success` |
| 给数据拟合参数（如直线 a、b） | `optimize` | `curve_fit(f, xdata, ydata)` → 读 `popt` |
| 算每个样本到每个聚类中心的距离（KMeans） | `spatial` | `distance.cdist(X, centers)` |
| 算一组点内部两两距离 | `spatial` | `distance.pdist(X)` |
| 算两个向量的欧氏/余弦距离 | `spatial` | `distance.euclidean(a, b)` / `distance.cosine(a, b)` |
| 巨大但几乎全 0 的矩阵 | `sparse` | `csr_matrix(dense)`；`.nnz` / `.toarray()` |
| NumPy 没有的矩阵分解/拼接 | `linalg` | `lu`、`block_diag`（日常 `np.linalg` 够用） |
| 已知几个点估计中间值 | `interpolate` | `interp1d(x, y)` |
| 算定积分 | `integrate` | `quad(f, a, b)` |
| 查某个函数怎么用 | — | `help(函数)`，看 docstring 的 Examples |

**分布四件套**：`rvs` 采样 ｜ `pdf/pmf` 曲线高度 ｜ `cdf` 累计概率 ｜ `ppf` 反查分位数

---

## 自测题（附答案）

以"知道该用哪个函数"为主，不要求推导。

1. **（stats）** 你想生成 1000 个"均值 170、标准差 10"的模拟身高数据做实验，用哪个模块哪个函数？写出代码（要求结果可复现）。
2. **（stats 四件套）** `norm.cdf(0) == 0.5` 说明了标准正态分布的什么性质？如果想反过来查"前 90% 的值不超过多少"，用哪个方法？
3. **（假设检验）** 运营同学给你一张表：旧版页面 200 个访客 30 人下单，新版页面 200 个访客 45 人下单，问"新版是不是真的更好"。你该调用哪个函数？返回值里看哪个数？这个数满足什么条件时你可以说"有显著差异"？
4. **（optimize）** 你在读一份源码，看到 `optimize.minimize(loss_fn, w0)`，其中 `loss_fn` 收一个权重数组返回一个数。这行代码在干什么？它在 ML 里对应什么过程？
5. **（spatial + KMeans）** KMeans 迭代中，已知样本矩阵 `X`（形状 `(100, 2)`）和 3 个聚类中心 `centers`（形状 `(3, 2)`），写出"给每个样本分配最近中心"的两行代码。

### 参考答案

**1.**

```python
from scipy import stats
heights = stats.norm.rvs(loc=170, scale=10, size=1000, random_state=42)
```

正态分布用 `norm`，采样用 `rvs`，`loc` 是均值、`scale` 是标准差，`random_state` 固定随机结果保证可复现。

**2.** `cdf(0) = 0.5` 说明：标准正态分布里恰好有一半的值 ≤ 0，即以均值 0 为中心左右对称（cdf 是"≤ x 的累计概率"）。反查"前 90% 不超过多少"用 `ppf`：`norm.ppf(0.9)`。

**3.** 分类数据（下单/未下单）数人头 → 卡方检验：

```python
from scipy import stats
table = [[30, 170], [45, 155]]          # [下单, 未下单] × [旧版, 新版]
chi2, p, dof, expected = stats.chi2_contingency(table)
```

看返回值里的 **p 值**；当 **p < 0.05**（惯例）时说差异显著——含义是"如果新旧版其实没差别，纯靠运气出现这么大下单率差距的概率不到 5%"。

**4.** 这行代码在**自动寻找让 `loss_fn`（损失函数）取值最小的权重**：从初始权重 `w0` 出发，迭代调整权重直到损失降不下去。它对应 ML 里的**模型训练**过程——手写梯度下降时你自己写的那个迭代循环，`minimize` 内部帮你做了（用的是更成熟的优化算法）。

**5.**

```python
from scipy import spatial
dist = spatial.distance.cdist(X, centers)   # (100, 3)，每行是该样本到 3 个中心的距离
labels = dist.argmin(axis=1)                # (100,)，每个样本最近中心的下标
```

这就是 KMeans 的分配步骤，等价于 NumPy 文档里的广播写法 `np.sqrt(((X[:, None] - centers) ** 2).sum(axis=2)).argmin(axis=1)`。

---

## 与 ML 学习路径的对应关系

| 本文章节 | 学习清单条目 | 后面哪里用到 |
|---|---|---|
| §1 模块地图 | （全局定位） | 贯穿始终：遇到需求先定位模块，再查文档 |
| §2 stats 概率分布 | 概率分布按需查 | 生成模拟数据验证 KMeans；理解朴素贝叶斯/逻辑回归的分布假设；神经网络权重初始化 |
| §3 假设检验 | （理解直觉即可） | 阶段二项目里比较模型效果、分析特征与标签的相关性；看懂别人 EDA 报告里的 p 值 |
| §4 optimize | optimize.minimize 按需查 | 理解"训练 = 最小化损失"这把总钥匙；阶段二手写梯度下降时对照；传统模型内部实现 |
| §5 spatial 距离 | （支撑 KMeans） | **阶段一 KMeans 里程碑的分配步骤**；KNN、文本相似度（cosine） |
| §6 sparse | （用到再查） | 阶段二文本类项目：`CountVectorizer` / `TfidfVectorizer` 的输出；推荐系统 |
| §7 linalg | （知道存在） | 读论文/源码时偶尔见到；日常继续用 `np.linalg` |
| §8 interpolate / integrate | （知道存在） | 数据预处理补采样点；基本不直接用 |
| §9 查文档 | （贯穿始终） | SciPy 全靠查；这个技能本身比任何单个函数都重要 |

**下一步**：回到阶段一里程碑，用 NumPy 手写 KMeans（分配步骤可以用本文 §5 的 `cdist` 一行替换，也可以坚持广播手写练功底）；之后阶段二引入 sklearn 时，本文的 stats（分布假设）、optimize（训练本质）、sparse（文本特征）会反复回响。
