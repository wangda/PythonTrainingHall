# 02 - Matplotlib 前置学习文档（教程 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：Figure/Axes 心智模型、折线图、散点图、条形图、直方图、imshow、子图布局、图表修饰、中文乱码、保存与杂项。
> **本文环境**：Matplotlib 3.10（所有示例已用 Matplotlib 3.10.3 实际运行验证，Agg 后端 + savefig 全部生成成功）。
> **定位**：够用就行。颜色映射（cmap）和热力图只留"了解即可"的一小节，用到再查官方文档。

---

## 本文怎么用

1. **当教程读**：从头读一遍，每个代码块亲手敲进 `.py` 文件跑一遍（画图必须跑脚本，在交互式终端里看图不方便）。重点章节（§3 折线图、§6 子图布局）务必动手。
2. **当速查手册用**：忘了某个参数，直接查文末【完整速查表】或跳到对应大节。

**先决知识**：NumPy（画图的数据都是 ndarray），见 `ml/docs/01-numpy前置学习文档.md`；本文造示例数据用的 `rng.normal`、`argmin` 等都在那篇里讲过。

**阅读约定**：🟢 必须掌握（ML 阶段天天用）；🟡 了解即可（看得懂、用到会查就行）。

**全文约定**：所有示例统一用**面向对象风格**（`fig, ax = plt.subplots()` 开头），原因见 §1.2。示例里 `import numpy as np` 和 `import matplotlib.pyplot as plt` 只在首次出现，后续默认已导入。

---

## 1. 先建立心智模型 🟢

### 1.1 Figure / Axes / Axis 三层对象

类比"一张纸上贴几张图"：

| 对象 | 类比 | 职责 |
|---|---|---|
| `Figure` | 一张纸（画布） | 整张图，管大小（figsize）、保存（savefig） |
| `Axes` | 纸上贴的一张图（子图） | **你 99% 的操作对象**：plot/scatter/标题/标签全在这层 |
| `Axis` | 一张图里的 x 轴或 y 轴 | 刻度、刻度标签（日常很少直接碰） |

一张 `Figure` 可以贴多个 `Axes`（子图）；每个 `Axes` 有 `xaxis` 和 `yaxis` 两个 `Axis`。

**最容易混的坑**：`Axes`（子图）不是 `Axis`（轴）。设置标题、画数据用 `ax.plot()` / `ax.set_title()`；调刻度才用 `ax.xaxis.xxx`。

### 1.2 两种 API 风格：pyplot 状态式 vs 面向对象

```python
# 状态式（pyplot）：操作的是"当前图"，plt 内部记住当前是哪个子图
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])        # 画到"当前 Axes"
plt.title('demo')
plt.subplot(1, 2, 2)       # 切换当前子图
plt.plot([3, 2, 1])        # 画到右边子图

# 面向对象（OO）：显式拿到对象，操作谁一目了然
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([1, 2, 3])
ax1.set_title('left')
ax2.plot([3, 2, 1])
```

**为什么推荐 OO 风格**：

- 多子图时，状态式靠 `plt.subplot(...)` 切换"当前"，代码顺序一乱就画错地方；OO 风格 `axes[0, 1].plot(...)` 指名道姓，永远不会混。
- ML 代码经常把画图封装成函数（传入 ax 参数），OO 风格天然支持，状态式做不到。
- 官方文档和成熟项目（sklearn、PyTorch 教程）的主流示例都是 OO 风格。

**注意命名的不对称**：状态式 `plt.title()` / `plt.xlabel()` 对应 OO 的 `ax.set_title()` / `ax.set_xlabel()`（多个 `set_` 前缀）。查资料时看到 `plt.xxx`，心里自动换成 `ax.set_xxx`。

**本节小结**：Figure = 画布，Axes = 子图（主战场），Axis = 轴。全文统一 OO 风格：`fig, ax = plt.subplots()` 起手，一切操作走 `ax.xxx`。

---

## 2. 第一张图：最小完整流程 🟢

一句话流程：**建画布 → 画数据 → 加修饰 → 显示或保存**。

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()                      # 1. 建画布+一个子图
x = np.arange(10)
loss = 2.0 * np.exp(-0.3 * x) + 0.1           # 模拟一条下降的 loss
ax.plot(x, loss, label='train loss')          # 2. 画折线，label 给图例用
ax.set_title('Training Loss')                 # 3. 标题 / 轴标签 / 图例
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend()
plt.show()                                    # 4a. 弹窗显示
fig.savefig('first.png', dpi=150, bbox_inches='tight')   # 4b. 或保存成文件
```

预期效果：一张从 2.1 左右指数下降到 0.1 的折线图，带标题、轴标签、右上角图例。

### 2.1 savefig 常用参数

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `fname`（第一个位置参数） | 文件名，扩展名决定格式 | 必填 | `'loss.png'` / `'loss.svg'` |
| `dpi` | 分辨率（每英寸点数） | 100（rcParams 配置） | `150`（论文/报告够用）、`300`（印刷） |
| `bbox_inches` | 边界裁剪方式 | None（按画布原样） | `'tight'`（**裁掉多余白边，几乎总是加它**） |

**顺序陷阱：`savefig` 必须在 `show` 之前调用。** `plt.show()` 会把当前 Figure 关掉并清空，先 show 再 savefig 存出来的是**空白图**。脚本里批量出图时，养成"先 savefig、再 close、不用 show"的习惯（`show` 只在本地交互调试时用）。

**本节小结**：四步走 `subplots → ax.plot → ax.set_* → savefig/show`；`savefig(..., dpi=150, bbox_inches='tight')` 是标配；**先 savefig 后 show**。

---

## 3. 折线图（核心）🟢

### 3.1 ax.plot 常用参数

**`ax.plot(x, y, ...)`** — 画折线；只传一个数组时把它当 y，x 自动取 0,1,2,...

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `color` | 颜色 | 自动循环 | `'tab:blue'`、`'red'`、`'#FF5733'`、`'g'`（详见 §8.1） |
| `linestyle` | 线型 | `'-'` 实线 | `'--'` 虚线、`'-.'` 点划线、`':'` 点线 |
| `linewidth` | 线宽 | 1.5 | `1`（密集曲线）、`2` |
| `marker` | 数据点标记 | None（不画） | `'o'` 圆点、`'s'` 方块、`'^'` 三角 |
| `markersize` | 标记大小 | 6 | `3`（点密集时调小） |
| `label` | 图例文字（配 `ax.legend()` 用） | 无 | `'train loss'` |
| `alpha` | 透明度 0~1 | 1 | `0.5`（多条线重叠时） |

```python
fig, ax = plt.subplots()
x = np.arange(30)
train_loss = 2.0 * np.exp(-0.15 * x) + 0.05
ax.plot(x, train_loss, color='tab:blue', linestyle='--', linewidth=1.5,
        marker='o', markersize=3, alpha=0.8, label='train')
ax.legend()
plt.show()
```

预期效果：蓝色虚线、每个 epoch 一个小圆点、半透明的下降曲线。

### 3.2 实战一：train loss vs val loss 多曲线对比

ML 里最高频的图，没有之一。判断过拟合就看它：train 还在降、val 开始升 = 过拟合。

```python
rng = np.random.default_rng(42)
epochs = np.arange(1, 31)
train_loss = 2.0 * np.exp(-0.12 * epochs) + 0.05 + rng.normal(0, 0.02, 30)
val_loss   = 2.2 * np.exp(-0.10 * epochs) + 0.15 + rng.normal(0, 0.03, 30)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, train_loss, label='train loss')
ax.plot(epochs, val_loss, label='val loss', linestyle='--')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Train vs Val Loss')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

预期效果：两条都下降、val（虚线）略高于 train 且抖动更大的曲线，带浅色网格。**要点：多次调用 `ax.plot` 就往同一张图里加线，靠 `label` + `legend()` 区分。**

### 3.3 实战二：双 y 轴 ax.twinx()（loss + accuracy 同图）

loss 是 0~2 量级，accuracy 是 0~1，画在一个 y 轴上会互相挤压看不清。`twinx()` 创建**共享 x 轴、独立右 y 轴**的姊妹子图：

```python
acc = 0.5 + 0.45 * (1 - np.exp(-0.1 * epochs)) + rng.normal(0, 0.01, 30)

fig, ax1 = plt.subplots()
ax1.plot(epochs, train_loss, color='tab:blue', label='loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')   # 左轴刻度也染蓝，一眼对上

ax2 = ax1.twinx()                                  # 姊妹轴：共享 x
ax2.plot(epochs, acc, color='tab:red', label='accuracy')
ax2.set_ylabel('accuracy', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.set_title('Loss & Accuracy')
plt.show()
```

预期效果：蓝线 loss 下降（读左轴），红线 accuracy 上升到 0.9 附近（读右轴）。**记忆点：轴和线同色是双 y 轴图的行规，不然读者分不清哪条线读哪个轴。**

**本节小结**：`plot` 参数记 `color/linestyle/marker/label/alpha` 五个；同图多线 = 多次 `ax.plot` + `legend`；量纲差太多用 `ax.twinx()`。

---

## 4. 散点图（核心）🟢

### 4.1 ax.scatter 常用参数

**`ax.scatter(x, y, ...)`** — 画散点，x、y 是等长数组。

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `s` | 点大小（面积） | 36 | `10`（点密集时）、`5` |
| `c` | 颜色（单个颜色，或与点等长的类别/数值数组） | 自动 | `'tab:blue'` 或 `labels` |
| `cmap` | 颜色映射（c 传数值数组时生效） | 'viridis' | `'coolwarm'`（用到再查，§5.3） |
| `alpha` | 透明度 | 1 | `0.5`（点重叠时必加，能看出密度） |
| `marker` | 点形状 | `'o'` | `'x'`、`'^'` |
| `label` | 图例文字 | 无 | `'cluster 0'` |

### 4.2 实战一：二维数据按类别着色（为 KMeans 可视化铺路）

两种画法，按需选：

```python
rng = np.random.default_rng(42)
center0 = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
center1 = rng.normal(loc=[3, 3], scale=0.5, size=(50, 2))
X = np.vstack([center0, center1])
labels = np.array([0] * 50 + [1] * 50)        # 类别标签

# 画法 A：按类别分开画，每类一个 label，图例最清楚（KMeans 结果推荐这种）
fig, ax = plt.subplots()
ax.scatter(X[labels == 0, 0], X[labels == 0, 1],
           c='tab:blue', alpha=0.6, label='cluster 0')
ax.scatter(X[labels == 1, 0], X[labels == 1, 1],
           c='tab:orange', alpha=0.6, label='cluster 1')
ax.legend()
plt.show()

# 画法 B：c 直接传标签数组 + cmap 上色，一行搞定，用 colorbar 显示数值含义
fig, ax = plt.subplots()
sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
fig.colorbar(sc, ax=ax, label='label')
plt.show()
```

预期效果：画法 A 看到左下蓝点一簇、右上橙点一簇；画法 B 同样两簇但颜色来自 coolwarm 映射，右侧有色条。**注意 `X[labels == 0, 0]` 就是 01-NumPy 文档 §3.3 布尔索引的实战**。

### 4.3 实战二：回归预测值 vs 真实值

判断回归模型好坏的经典图：点越贴紧 y=x 对角线，预测越准。

```python
y_true = rng.normal(0, 1, 100)
y_pred = y_true * 0.9 + rng.normal(0, 0.2, 100)

fig, ax = plt.subplots()
ax.scatter(y_true, y_pred, alpha=0.5)
ax.plot([-3, 3], [-3, 3], color='red', linestyle='--', label='y = x')  # 参考线
ax.set_xlabel('true')
ax.set_ylabel('predicted')
ax.legend()
plt.show()
```

预期效果：一群点沿红色虚线对角线分布、有轻微离散。**技巧：散点图里加参考线，用 `ax.plot` 画一条直线即可。**

**本节小结**：`scatter` 参数记 `s/c/alpha`；按类别着色两种画法——分类别多次 scatter（图例清楚）或 `c=labels, cmap=...`（一行搞定）；预测 vs 真实加 y=x 参考线。

---

## 5. 其他常用图型（够用即可）🟢

### 5.1 条形图 ax.bar / ax.barh：特征重要性

**`ax.bar(x, height)`** 竖向条形，**`ax.barh(y, width)`** 横向条形。

```python
features = ['age', 'income', 'score', 'visits']
importance = [0.42, 0.28, 0.18, 0.12]

fig, ax = plt.subplots()
ax.bar(features, importance)        # 竖向：x 是类别名，height 是数值
ax.set_ylabel('importance')
plt.show()

fig, ax = plt.subplots()
ax.barh(features, importance)       # 横向：特征名长时更好排版
ax.set_xlabel('importance')
plt.show()
```

预期效果：竖版四根从左到右递减的柱子；横版四根从上（visits）到下（age）递增的柱子（barh 默认从下往上排）。**经验：类别名长、类别多 → 用 barh。**

### 5.2 直方图 ax.hist：数据分布 / 残差分布

**`ax.hist(x, bins=...)`** — 把数据按区间分桶计数画成柱状，看分布形状。

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `x` | 一维数据 | 必填 | 残差、某列特征 |
| `bins` | 分桶数（或桶边界数组） | 10 | `20`~`50`，数据多就调大 |
| `edgecolor` | 柱边框颜色 | None | `'black'`（柱子挨一起时便于区分） |

```python
residuals = y_true - y_pred                     # 残差 = 真实 - 预测

fig, ax = plt.subplots()
ax.hist(residuals, bins=20, edgecolor='black')
ax.set_xlabel('residual')
plt.show()
```

预期效果：以 0 为中心的近似钟形分布。**ML 用途：残差直方图应以 0 为中心、近似对称——明显偏斜说明模型有系统性偏差。** bins 太小图太糙看不出形状，太大每个桶没几个点全是锯齿，一般 20~50 之间试。

### 5.3 ax.imshow：显示矩阵 / 图片（混淆矩阵）

**`ax.imshow(A)`** — 把二维矩阵（或图片数组）按数值大小映射成颜色显示。

```python
cm = np.array([[45, 3, 2],      # 3 分类的混淆矩阵：行=真实类，列=预测类
               [5, 38, 7],
               [1, 4, 40]])

fig, ax = plt.subplots()
im = ax.imshow(cm)                          # 矩阵直接变热力图
fig.colorbar(im, ax=ax)                     # 加色条显示数值-颜色对应
for i in range(3):                          # 每个格子里写上数字
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center')
ax.set_xlabel('predicted')
ax.set_ylabel('true')
plt.show()
```

预期效果：3×3 方格图，对角线格颜色最深（45/38/40 最大），格内有数字，右侧有色条。**读法：对角线深 = 分类准；非对角线上有深格 = 这两类容易混。**

> 🟡 **颜色映射与热力图：了解即可，用到再查。** `imshow` 和 `scatter(c=...)` 的 `cmap` 参数控制"数值→颜色"的映射，常用 `'viridis'`（默认）、`'coolwarm'`（正负对比）、`'gray'`（灰度图）；色条用 `fig.colorbar(返回的对象, ax=ax)`。ML 阶段只有混淆矩阵和注意力权重用得到，届时查官方 cmap 列表即可。

**本节速查表**：

| 需求 | 函数 |
|---|---|
| 特征重要性 | `ax.bar`（竖）/ `ax.barh`（横，名字长时用） |
| 数据/残差分布 | `ax.hist(x, bins=20, edgecolor='black')` |
| 混淆矩阵/矩阵可视化 | `ax.imshow(cm)` + `fig.colorbar` |
| 格子里写字 | `ax.text(x, y, s, ha='center', va='center')` |

---

## 6. 子图布局（核心）🟢

### 6.1 plt.subplots(nrows, ncols) 与 axes 索引

**`plt.subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False)`** — 建画布 + nrows×ncols 个子图，返回 `(fig, axes)`。

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `nrows` / `ncols` | 行数 / 列数 | 1 / 1 | `plt.subplots(2, 2)` |
| `figsize` | 画布尺寸（宽， 高），单位英寸 | (6.4, 4.8) | `(8, 4)`、`(10, 8)` |
| `sharex` / `sharey` | 子图间共享 x / y 轴刻度 | False | `True`（同量纲对比时） |

**axes 的索引规则（重点）**：

- `plt.subplots()`（1×1）→ 返回**单个 Axes**，直接用 `ax.plot(...)`；
- `plt.subplots(1, 3)`（一行/一列）→ 返回**一维数组**，`axes[0]`、`axes[1]`；
- `plt.subplots(2, 2)`（多行多列）→ 返回**二维数组**，`axes[行, 列]`（NumPy 索引）。

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].plot(train_loss)
axes[1].hist(residuals, bins=20)
axes[2].scatter(y_true, y_pred, s=5)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(train_loss)          # 左上
axes[0, 1].plot(acc)                 # 右上
axes[1, 0].scatter(y_true, y_pred, s=5)   # 左下
axes[1, 1].imshow(cm)                # 右下
plt.show()
```

**共享轴**：`sharey=True` 让几个子图共用同一套 y 刻度（内部重复的刻度标签自动隐藏），同量纲对比时避免视觉误导：

```python
fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].plot(train_loss)
axes[1].plot(val_loss)      # 和左图同一 y 尺度，高低一眼可比
plt.show()
```

### 6.2 调整间距：tight_layout / constrained_layout

子图多了标题和轴标签容易互相重叠，两选一解决：

| 方法 | 用法 | 说明 |
|---|---|---|
| `fig.tight_layout()` | 画完所有子图后调用一次 | 自动挤开重叠 |
| `constrained_layout=True` | `plt.subplots(2, 2, constrained_layout=True)` | 建图时声明，全程自动维护间距，**更省心，推荐** |

### 6.3 实战：2×2 训练监控面板

一张图同时看 loss、accuracy、数据分布、混淆矩阵——这就是 ML 实验报告的标配：

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

axes[0, 0].plot(epochs, train_loss, label='train')
axes[0, 0].plot(epochs, val_loss, label='val')
axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(epochs, acc, color='tab:green')
axes[0, 1].set_title('Accuracy'); axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=10)
axes[1, 0].set_title('Data')

im = axes[1, 1].imshow(cm)
fig.colorbar(im, ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix')

fig.savefig('dashboard.png', dpi=150)
plt.show()
```

预期效果：四宫格——左上双 loss 曲线、右上 accuracy 上升曲线、左下两簇彩色散点、右下带色条的混淆矩阵，间距整齐不重叠。

**本节小结**：`subplots(行, 列)` 返回的 axes 按"1×1 单个 → 一行/一列一维 → 多行多列二维"索引；同量纲对比加 `sharex/sharey=True`；间距问题用 `constrained_layout=True`；每个子图各自 `set_title/set_xlabel`。

---

## 7. 图表修饰速查 🟢

全是 `ax.set_*` 方法，记住一组通吃所有图型：

| 方法 | 作用 | 常用写法 |
|---|---|---|
| `ax.set_title(s)` | 子图标题 | `ax.set_title('Loss')` |
| `fig.suptitle(s)` | 整张大图标题 | `fig.suptitle('实验报告')` |
| `ax.set_xlabel(s)` / `set_ylabel(s)` | 轴标签 | `ax.set_xlabel('epoch')` |
| `ax.set_xlim(lo, hi)` / `set_ylim` | 轴显示范围 | `ax.set_ylim(0, 1)`（accuracy 固定 0~1 好看） |
| `ax.legend()` | 显示图例（依赖 plot 时的 label） | `ax.legend(loc='upper right')`，loc 默认自动选位 |
| `ax.grid()` | 网格线 | `ax.grid(alpha=0.3)`——**alpha 调淡，不抢数据的风头** |
| `ax.set_xticks(list)` / `set_yticks` | 手动指定刻度位置 | `ax.set_xticks([0, 10, 20, 30])` |
| `ax.axhline(y=...)` / `ax.axvline(x=...)` | 水平/垂直参考线 | `ax.axhline(y=0, color='red', linestyle='--')`（残差图必备） |

```python
fig, ax = plt.subplots()
ax.plot(epochs, train_loss)
ax.set_title('Training Loss')
ax.set_xlabel('epoch'); ax.set_ylabel('loss')
ax.set_xlim(0, 30); ax.set_ylim(0, 2.5)
ax.set_xticks([0, 10, 20, 30])
ax.grid(alpha=0.3)
plt.show()
```

### 中文乱码问题

Matplotlib 默认字体不含中文字形，标题/标签写中文会变成方框 `□□□`。两行配置解决：

```python
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 换中文字体
plt.rcParams['axes.unicode_minus'] = False                 # 修复负号 '-' 显示成方块
```

**检测本机有哪些中文字体**（配置前先查，名字抄错等于没配）：

```python
from matplotlib import font_manager
names = {f.name for f in font_manager.fontManager.ttflist}
print(sorted(n for n in names if any(k in n for k in ('CJK', 'Hei', 'WenQuanYi'))))
```

Linux 常见中文字体名：`'WenQuanYi Micro Hei'`（文泉驿）、`'Noto Sans CJK SC'`（思源黑体，有些发行版只装 JP 版 `'Noto Sans CJK JP'` 也能显示中文）。`font.sans-serif` 接受**列表**，按顺序找第一个存在的字体，写 `['Noto Sans CJK SC', 'WenQuanYi Micro Hei']` 做兜底。`axes.unicode_minus` 必须一起配，否则坐标轴负号变方块。

**本节小结**：修饰全是 `ax.set_*`；`grid(alpha=0.3)`、`axhline` 是 ML 高频；中文乱码两行 rcParams，字体名先用 `font_manager` 查本机。

---

## 8. 常用杂项 🟢

### 8.1 颜色指定方式

三种写法都合法，混着用：

```python
ax.plot(x, y, color='g')            # 单字母：r g b k(黑) w(白) y m c
ax.plot(x, y, color='red')          # 英文名：red/blue/orange/green/gray...
ax.plot(x, y, color='#FF5733')      # 十六进制（和设计稿对齐时用）
ax.plot(x, y, color='tab:blue')     # tab 调色板：tab:blue/orange/green/red...
```

默认颜色循环顺序就是 `tab:blue → tab:orange → tab:green → tab:red → ...`，多曲线时不指定 color 也够用。

### 8.2 线型与标记速查表

| linestyle | 效果 | marker | 效果 |
|---|---|---|---|
| `'-'` | 实线（默认） | `'o'` | 圆点 |
| `'--'` | 虚线 | `'s'` | 方块 |
| `'-.'` | 点划线 | `'^'` | 上三角 |
| `':'` | 点线 | `'x'` | 叉 |

偷懒写法：`ax.plot(x, y, 'o--')` 一个字符串同时给标记+线型+颜色顺序的三个符号（如 `'o--'` = 圆点+虚线）。**约定俗成：train 实线、val 虚线。**

### 8.3 plt.style 整体风格

`plt.style.use('ggplot')` 之类的预设主题一键换肤（灰底网格等）。只影响美观，**建议用 `with plt.style.context(...)` 限定作用范围**，避免全局污染：

```python
with plt.style.context('ggplot'):
    fig, ax = plt.subplots()
    ax.plot(train_loss)
    plt.show()
```

### 8.4 保存格式选择

| 格式 | 类型 | 适用 |
|---|---|---|
| `png` | 位图 | **默认选它**，报告/README/聊天工具通用 |
| `svg` | 矢量图 | 网页、后续还要编辑 |
| `pdf` | 矢量图 | 论文投稿（LaTeX） |

格式由 `savefig` 文件名的扩展名决定：`fig.savefig('loss.svg')`。

### 8.5 plt.close：批量存图防内存泄漏

脚本里循环生成几十上百张图时，Figure 对象一直占内存，攒多了会爆。每张存完就关：

```python
for epoch in range(10):
    fig, ax = plt.subplots()
    ax.plot(np.random.default_rng(epoch).normal(0, 1, 100).cumsum())
    fig.savefig(f'/tmp/fig_{epoch}.png')
    plt.close(fig)          # 关掉释放内存；plt.close('all') 关掉所有
```

交互式探索无所谓，**批量出图脚本必加**。

**本节小结**：颜色三写法（单字母/英文名/十六进制）；train 实线 val 虚线是行规；保存默认 png；批量出图循环里 `plt.close(fig)`。

---

## 完整速查表

**创建**：`fig, ax = plt.subplots()` 单图 ｜ `plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)` 子图网格 ｜ `axes[i]` / `axes[i, j]` 索引 ｜ `sharex/sharey=True` 共享轴 ｜ `ax.twinx()` 双 y 轴

**折线**：`ax.plot(x, y, color=, linestyle=, linewidth=, marker=, label=, alpha=)` ｜ 多线 = 多次 plot + `ax.legend()` ｜ `'-' '--' '-.' ':'` 线型

**散点**：`ax.scatter(x, y, s=, c=, cmap=, alpha=)` ｜ 按类别：多次 scatter 或 `c=labels, cmap='coolwarm'` ｜ `ax.plot` 画 y=x 参考线

**条形**：`ax.bar(x, height)` 竖 ｜ `ax.barh(y, width)` 横（名字长时用）

**直方**：`ax.hist(x, bins=20, edgecolor='black')`

**矩阵/图片**：`ax.imshow(A)` ｜ `fig.colorbar(im, ax=ax)` 色条 ｜ `ax.text(j, i, s, ha='center', va='center')` 格内写字

**修饰**：`set_title/set_xlabel/set_ylabel` ｜ `set_xlim/set_ylim` 范围 ｜ `legend(loc=)` 图例 ｜ `grid(alpha=0.3)` 网格 ｜ `set_xticks/set_yticks` 刻度 ｜ `axhline/axvline` 参考线 ｜ `fig.suptitle` 大标题 ｜ 中文：`rcParams['font.sans-serif']=['WenQuanYi Micro Hei']` + `rcParams['axes.unicode_minus']=False`

**保存**：`fig.savefig('x.png', dpi=150, bbox_inches='tight')`（png 默认/svg 网页/pdf 论文） ｜ **先 savefig 后 show** ｜ 批量出图循环里 `plt.close(fig)`

---

## 自测题（附答案）

1. **（心智模型）** `fig, axes = plt.subplots(2, 3)` 后，`axes` 是什么类型、shape 是什么？给右上角子图设置标题怎么写？
2. **（双 loss 曲线）** 给定 `epochs`、`train_loss`、`val_loss` 三个等长数组，画一张图：两条曲线（train 实线、val 虚线）、有图例、有 x/y 轴标签、加半透明网格。
3. **（残差图）** 给定回归模型的 `y_pred` 和 `residuals`（= y_true - y_pred），画"预测值-残差"散点图，并加一条 y=0 的红色虚线参考线。
4. **（子图+直方图）** 画 1×2 两个子图，分别显示两组数据的分布直方图，要求共享 y 轴便于对比。
5. **（条形图）** 给定 `features`（名字列表）和 `importance`（数值列表），画一张按重要性**从小到大**排列的水平条形图（提示：用 `np.argsort`）。
6. **（顺序陷阱）** 下面代码保存出来的图为什么是空白的？怎么改？

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
plt.show()
fig.savefig('out.png')
```

7. **（双 y 轴）** 想把 loss（0~2）和 accuracy（0~1）画在同一张图里且不互相挤压，用什么方法？写出核心三行代码。
8. **（中文乱码）** 同事说图上中文全是方框、负号也显示不对，你怎么修？需要确认本机有什么字体时怎么查？

### 参考答案

**1.** `axes` 是 shape 为 `(2, 3)` 的 NumPy 数组，元素是 Axes 对象。右上角是第 0 行第 2 列：`axes[0, 2].set_title('右上')`。

**2.**

```python
fig, ax = plt.subplots()
ax.plot(epochs, train_loss, label='train loss')
ax.plot(epochs, val_loss, label='val loss', linestyle='--')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

**3.**

```python
fig, ax = plt.subplots()
ax.scatter(y_pred, residuals, alpha=0.5)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('predicted')
ax.set_ylabel('residual')
plt.show()
```

**4.**

```python
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
plt.show()
```

**5.**

```python
idx = np.argsort(importance)
fig, ax = plt.subplots()
ax.barh([features[i] for i in idx], [importance[i] for i in idx])
ax.set_xlabel('importance')
plt.show()
```

**6.** `plt.show()` 会关闭并清空当前 Figure，之后再 savefig 存的是空白图。把 `savefig` 挪到 `show` 之前；脚本批量出图时干脆不用 `show`，savefig 后 `plt.close(fig)`。

**7.** 用 `ax.twinx()` 双 y 轴：

```python
ax2 = ax1.twinx()
ax1.plot(epochs, train_loss, color='tab:blue')
ax2.plot(epochs, acc, color='tab:red')
```

**8.** 开头加两行：`plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']`（列表兜底）和 `plt.rcParams['axes.unicode_minus'] = False`。查本机中文字体：

```python
from matplotlib import font_manager
print(sorted({f.name for f in font_manager.fontManager.ttflist
              if any(k in f.name for k in ('CJK', 'Hei', 'WenQuanYi'))}))
```

---

## 与 ML 学习路径的对应关系

| 本文章节 | 阶段一（KMeans 项目） | 后续 ML 阶段 |
|---|---|---|
| §1 心智模型 + §2 最小流程 | 所有画图的骨架 | 所有画图的骨架 |
| §3 折线图（多曲线/twinx） | — | loss/accuracy 训练曲线、学习曲线（train vs val 分数） |
| §4 散点图（类别着色） | **KMeans 聚类结果可视化**（按簇着色 + 中心点标记） | 数据分布散点、预测值 vs 真实值、残差图 |
| §5.1 bar/barh | — | 特征重要性（树模型 feature_importances_） |
| §5.2 hist | 看特征分布是否需要归一化 | 残差分布检查、数据分布探索 |
| §5.3 imshow | — | 混淆矩阵、注意力权重热力图 |
| §6 子图布局 | KMeans 迭代过程多帧对比 | 2×2 训练监控面板（loss/acc/数据/混淆矩阵） |
| §7 修饰 + 中文 | 图表标注、实验报告中文标题 | 同左 |
| §8 杂项（close/格式） | 批量保存迭代快照防内存泄漏 | 实验批量出图、报告导出 |

**下一步**：本文 → Pandas（`ml/docs/03-...`，数据加载与清洗）→ 回到 NumPy 手写 KMeans，并用本文的散点图（§4.2）画出聚类结果，完成阶段一里程碑。
