# 07 - MLP 与反向传播前置学习文档（教程 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：线性模型的天花板（为什么需要隐藏层）、神经元与感知机、激活函数（sigmoid/tanh/ReLU）、前向传播逐层矩阵推导与参数初始化（全零灾难、He/Xavier）、反向传播（链式法则逐层推梯度 + 微型网络手算 + 完整 numpy 实现）、梯度检查与调试、PyTorch 重写 MLP（autograd/nn.Module/训练循环五步法）、工程化 Trainer（checkpoint/早停/学习率调度）、手写练习预告。
> **本文环境**：NumPy 2.2 + scikit-learn 1.9 + PyTorch 2.14.1（所有示例已实际运行验证，输出逐项核对；numpy 手写的梯度用数值梯度验证过，torch autograd 的梯度和 numpy 手推结果 allclose 一致；torch 相关代码均固定 `torch.manual_seed`，输出可复现）。

---

## 本文怎么用

1. **当教程读**：从头读一遍，每个代码块亲手敲一遍跑一遍。§5 反向传播是本模块的心脏、也是整个学习路径的分水岭，务必动手——尤其是 §5.5 的微型网络手算，拿纸笔跟着算一遍，算完反向传播的"神秘感"就没了。
2. **当速查手册用**：手写练习时忘了公式或 API，直接查文末【完整速查表】或跳到对应大节。

**先决知识**（都在阶段一和本阶段模块 1、2，本文会直接引用）：

- `../阶段一/00-微积分前置知识.md`：导数、偏导数、**链式法则**（本文的心脏就靠它）、梯度下降的"下山"直觉。
- `../阶段一/00-线性代数前置知识.md`：矩阵乘法、转置、逐元素乘（Hadamard 积）。
- `../阶段一/01-numpy前置学习文档.md`：`@` 矩阵乘、广播、`np.maximum`、`default_rng`、`allclose`。
- `./01-线性回归前置学习文档.md`（下称"模块 1 文档"）：模型 `w·x + b`、梯度下降 `θ ← θ − η·∂L/∂θ`、学习率、epoch/batch/iteration、特征缩放、过拟合、正则化。
- `./03-逻辑回归与分类前置学习文档.md`（下称"模块 2 文档"）：sigmoid、交叉熵、softmax（含减最大值 trick）、one-hot、梯度 `(1/n)·Xᵀ(P−Y)`、分类评估指标。**本文默认你已经读完模块 1 和模块 2**，相同概念只引用不重讲。

**阅读约定**：🟢 必须掌握（手写练习直接考）；🟡 了解即可（看得懂、知道去哪查就行）。

**贯穿全文的例子**：主线上次项目用过的 **digits 数据集**（sklearn 内置 `load_digits`，1797 张 8×8 的手写数字灰度图，64 维输入、10 个类别）。我们从"一个神经元"出发，一步步搭到"两层隐藏层的 MLP"，全程真实训练、给出真实 loss 曲线——digits 秒级训练，文档里的每个训练循环你都能原样跑完。另配一个**微型例子**专门讲反向传播推导：单样本、2 维输入、一个 2 神经元隐藏层、2 类输出的小网络，§5.5 会用具体数字把前向+反向完整手算一遍。

```python
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape, y.shape)          # (1797, 64) (1797,)
print(X.min(), X.max())          # 0.0 16.0 —— 像素值 0~16
print(np.bincount(y))            # [178 182 177 183 181 182 181 179 174 180] —— 10 类基本平衡
```

---

## 1. 为什么需要隐藏层：线性模型的天花板 🟢

### 1.1 用数据说话（一）：逻辑回归在 digits 上已经很能打了

模块 2 你亲手写过 softmax 多分类，项目里又拿它做了手写数字识别。先把它在 digits 上的真实表现复现出来（训练/测试按 75%/25% 切分）：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print("测试准确率:", round(clf.score(X_te, y_te), 4))     # 0.9733
wrong = np.where(pred != y_te)[0]
print("分错:", len(wrong), "/", len(y_te))               # 12 / 450
print("前10个分错(真实→预测):", [(int(y_te[i]), int(pred[i])) for i in wrong[:10]])
# [(2, 3), (9, 3), (7, 9), (3, 5), (9, 8), (4, 1), (5, 9), (8, 5), (6, 5), (5, 6)]
```

97.33% 的准确率，450 张里只错 12 张——而且错的全是"长得像"的数字对（2→3、9→3、7→9、4→1……），都是笔画高度相似的混淆。可以把这 12 张图画出来亲眼看（都是真的写得潦草）：

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for ax, i in zip(axes.flat, wrong):
    ax.imshow(X_te[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'真{int(y_te[i])}→猜{int(pred[i])}', fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

所以第一个结论要公正：**在 digits 这种"干净小图"上，线性模型已经接近天花板**。那我们费这么大劲学神经网络图什么？换下一份数据就知道了。

### 1.2 用数据说话（二）：月牙形数据上，线性模型彻底失败

造一份"两个月牙"数据（`make_moons`）：两类点各弯成一个月牙，互相嵌套。**没有任何一条直线能把它们分开**——这不是模型参数没调好，是"直线"这个形状本身就不够：

```python
from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=42)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=42)

lr_moon = LogisticRegression().fit(Xm_tr, ym_tr)
print("逻辑回归  训练/测试:", round(lr_moon.score(Xm_tr, ym_tr), 4),
      round(lr_moon.score(Xm_te, ym_te), 4))        # 0.8429 0.8083
```

逻辑回归训练准确率卡在 84%——**欠拟合**：不是学不动，是模型的表达能力到顶了（模块 2 §6.1：逻辑回归的决策边界只能是直线）。现在只改一件事：给它加**一个 16 神经元的隐藏层**（怎么实现的正是本文 §4、§5 的内容，这里先当黑盒看结果）：

```python
from sklearn.neural_network import MLPClassifier

mlp_moon = MLPClassifier(hidden_layer_sizes=(16,), solver='lbfgs',
                         max_iter=2000, random_state=42).fit(Xm_tr, ym_tr)
print("MLP(16)   训练/测试:", round(mlp_moon.score(Xm_tr, ym_tr), 4),
      round(mlp_moon.score(Xm_te, ym_te), 4))        # 0.9107 0.9083
```

| 模型                     | 训练准确率 | 测试准确率 | 决策边界形状       |
| ------------------------ | ---------- | ---------- | ------------------ |
| 逻辑回归（无隐藏层）     | 0.8429     | 0.8083     | 只能是一条直线     |
| **MLP，1 个隐藏层(16)**  | **0.9107** | **0.9083** | 可以弯成月牙的弧线 |

同样的数据、同样的梯度下降、同样的交叉熵损失，**唯一差别就是中间多了一层"先变换特征再分类"**。测试准确率从 0.8083 跳到 0.9083——弯的边界学出来了。

### 1.3 天花板根源：特征组合要人手工造

模块 1 §5 其实教过一条不用神经网络也能画弯边界的路：**手工造特征**。把 x₁²、x₂²、x₁x₂……这些"特征的组合"当新特征喂给线性模型，弯的问题就变成高维空间里直的问题。这条路在月牙数据上同样有效：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly5 = make_pipeline(PolynomialFeatures(5), LogisticRegression(max_iter=5000)).fit(Xm_tr, ym_tr)
print("5次多项式+逻辑回归:", round(poly5.score(Xm_tr, ym_tr), 4),
      round(poly5.score(Xm_te, ym_te), 4))          # 0.8929 0.9083
```

0.9083，和 MLP 打平。那 MLP 的价值在哪？看代价：

- **多项式路线**：你要自己猜该造哪些组合（5 次？x₁x₂ 要不要？x₁³x₂² 呢？）。2 个特征 5 次多项式已经造出 20 个新特征；digits 的 64 个像素造 3 次组合就接近 5 万个，造 5 次直接上千万——**组合爆炸**。而且造什么全靠人对数据的理解，换一份数据全部重来。
- **MLP 路线**：隐藏层让机器**自己学**"哪些特征组合有用"。128 个隐藏神经元就是 128 个自动学出来的"新特征"，第二层再在这 128 个新特征上继续组合。**特征工程从"人肉穷举"变成"梯度下降顺便做的事"**。

回到 digits 上看一眼两者的差距（sklearn 黑盒版，本文后面会亲手实现）：

```python
mlp_d = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
mlp_d.fit(X_tr / 16.0, y_tr)
print("sklearn MLP(128,64) 测试准确率:", round(mlp_d.score(X_te / 16.0, y_te), 4))  # 0.9778
```

digits 上提升不大（0.9733 → 0.9778），因为这份数据对线性模型太友好了。但请记住月牙数据的教训：**线性模型能解的问题，隐藏层都能解；隐藏层能解的弯问题，线性模型给再久也学不会**。图像、语音、自然语言全是"弯问题"——这就是整个深度学习存在的原因。

**本节小结**：逻辑回归在 digits 上 0.9733 已经接近线性天花板；在月牙数据上线性模型彻底失败（0.8083），一个 16 神经元的隐藏层把它拉到 0.9083。天花板的根源是"特征组合要靠人手工造"，而手工造会组合爆炸；隐藏层 = 让机器用梯度下降自己学特征组合。

---

## 2. 神经元与感知机 🟢

### 2.1 是什么（直觉）：神经元 = w·x + b + 一个开关函数

模块 1 和模块 2 你已经把神经元的全部零件学完了：

```
z = w·x + b        ← 模块 1：线性加权求和（逻辑回归去掉 sigmoid 也是它）
a = f(z)           ← 模块 2：sigmoid 就是一个 f；本文换成 ReLU 等（§3）
```

一个**神经元**就是这两步：先把输入按权重加权求和加个偏置（算出一个"倾向分" z），再过一个**激活函数** f（决定这个神经元"点不点亮"，输出 a）。说白了：**一个神经元就是模块 2 的逻辑回归单元，只是换掉了 sigmoid、也不再把输出解释成概率**。

多个神经元并排 = 一**层**（layer）；层的输出再喂给下一层，就搭出了**多层感知机**（MLP，Multi-Layer Perceptron）——本文的主角。输入进、答案出的过程中间的层叫**隐藏层**（hidden layer）：它们的输出既不来自数据、也不直接给人看，是网络内部"自己学的特征"。

### 2.2 历史一句话

1958 年 Rosenblatt 提出**感知机**（perceptron）：w·x + b 之后过一个"阶跃函数"（z>0 输出 1，否则 0），是第一个能"从数据学权重"的机器，轰动一时。1969 年 Minsky 证明单层感知机连 XOR（异或）都学不会，神经网络研究就此冰封十几年——直到反向传播（本文 §5）让多层网络可训练才翻身。这段历史留下两个遗产：名字（MLP 里的"P"就是感知机）和教训（**单层线性能力有限，必须堆层；堆了层必须会训练**）。

### 2.3 关键论证：没有激活函数，堆多少层都等于一层

堆层的全部意义来自激活函数。证明就一行矩阵乘法结合律。设第一层 `z₁ = W₁x`，第二层 `z₂ = W₂z₁`（都没有激活函数），那么：

```
z₂ = W₂(W₁x) = (W₂W₁)x
```

**两个矩阵先乘起来，就是一个新矩阵**——两层"网络"和一个单层模型 `z = Wx`（W = W₂W₁）数学上严格相等。堆 100 层也一样：W₁₀₀…W₂W₁ 还是一个矩阵。用 numpy 实测，两行代码：

```python
rng = np.random.default_rng(0)
x  = np.array([1.0, 2.0, 3.0])
W1 = rng.normal(0, 1, (4, 3)).round(2)   # 第一层：3 输入 → 4 神经元
W2 = rng.normal(0, 1, (2, 4)).round(2)   # 第二层：4 → 2 输出
two_layers = W2 @ (W1 @ x)               # 两层连着算（都没有激活函数）
one_layer  = (W2 @ W1) @ x               # 先合并成一个矩阵再算
print("两层结果:", two_layers.round(6))   # [-3.823  -3.0332]
print("一层结果:", one_layer.round(6))   # [-3.823  -3.0332]
print("完全相等:", np.allclose(two_layers, one_layer))   # True
print("等效权重 W2@W1 的形状:", (W2 @ W1).shape)         # (2, 3)
```

逐字相等。结论：**没有激活函数的多层网络 ≡ 单层线性模型 ≡ 模块 2 的逻辑回归**，月牙数据照样学不会。激活函数是"非线性"的唯一来源——每层算完 z 之后过一个弯的函数，下一层就在一个"被掰弯过的空间"里继续组合，层叠起来才能逼近任意弯曲的边界。这就是为什么激活函数不是可选项，是命脉。

**本节小结**：神经元 = `w·x + b` 加激活函数，就是去掉概率解释的逻辑回归单元；并排成层、层叠成 MLP。感知机的历史教训：单层连 XOR 都学不会。没有激活函数时 W₂(W₁x) = (W₂W₁)x，堆多少层都等价于一层线性模型（numpy 实测逐字相等）——**激活函数是多层网络存在的理由**。

---

## 3. 激活函数 🟢

### 3.1 是什么（直觉）：每个神经元的"点亮开关"

激活函数 f 接在 `z = w·x + b` 后面，决定这个神经元对输入"作何反应"。三个主角一张表先看全局（导数 §5 反向传播要用，必须记）：

|            | 公式 f(z)              | 输出范围 | 导数 f′(z)             | 性格                             |
| ---------- | ---------------------- | -------- | ---------------------- | -------------------------------- |
| sigmoid    | 1/(1+e⁻ᶻ)              | (0, 1)   | σ(z)(1−σ(z))，最大 0.25 | 模块 2 学过；两端饱和→梯度消失   |
| tanh       | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)      | (−1, 1)  | 1−tanh²(z)，最大 1     | 零中心版 sigmoid；同样会饱和     |
| **ReLU**   | max(0, z)              | [0, +∞)  | z>0 时 **1**，z≤0 时 0 | **现代默认**；不饱和、计算零成本 |

拿一排 z 值把三个函数和导数全算出来，数字比图像更好记：

```python
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
def relu(z): return np.maximum(0, z)
def relu_prime(z): return (z > 0).astype(float)
def tanh_prime(z):
    t = np.tanh(z)
    return 1 - t**2

zs = np.array([-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0])
np.set_printoptions(suppress=True)       # 关掉科学计数法，小数以普通形式打印
print("z:      ", zs)
print("sigmoid:", sigmoid(zs).round(4))          # [0.     0.0067 0.1192 0.5    0.8808 0.9933 1.    ]
print("σ'(z):  ", sigmoid_prime(zs).round(6))    # [0.000045 0.006648 0.104994 0.25     0.104994 0.006648 0.000045]
print("tanh:   ", np.tanh(zs).round(4))          # [-1.     -0.9999 -0.964   0.      0.964   0.9999  1.    ]
print("tanh':  ", tanh_prime(zs).round(6))       # [0.       0.000182 0.070651 1.       0.070651 0.000182 0.      ]
print("relu:   ", relu(zs))                      # [ 0.  0.  0.  0.  2.  5. 10.]
print("relu':  ", relu_prime(zs))                # [0. 0. 0. 0. 1. 1. 1.]
```

### 3.2 sigmoid：老朋友，但深网络的毒药

sigmoid 在模块 2 是输出层主角（把分数压成概率），当隐藏层激活却有一个致命伤，就写在上面的数字里：**导数 σ′(z) = σ(z)(1−σ(z)) 的最大值只有 0.25（z=0 处），|z| 一大就趋近 0**（z=±10 时 0.000045）。

这和模块 2 §3 讲的"梯度消失"是同一个病，但在多层网络里病情指数放大：§5 会看到，反向传播是把每层的局部导数**连乘**回前面的层。每层乘一个 ≤0.25 的数，10 层就是 0.25¹⁰ ≈ 10⁻⁶，20 层就是 10⁻¹²——**靠近输入的层收到的梯度几乎为 0，参数根本不更新，等于白挂在那里**。§6.4 有实测表：20 层 sigmoid 网络，第一层梯度比输出端小 7 个数量级。

所以 sigmoid 现在的定位很清晰：**二分类输出层专用**（模块 2 的用法），隐藏层别用它。

### 3.3 tanh：零中心版 sigmoid

tanh 把 sigmoid 压扁平移成 (−1, 1) 区间、过原点：`tanh(z) = 2σ(2z) − 1`。输出以 0 中心（正负有对称的激活），数学性质比 sigmoid 顺一点，导数最大能到 1。但**两端同样饱和**（z=±10 时导数 0.0002），梯度消失的病一样不少。今天主要在老式 RNN 里还能见到它；全连接网络默认不用。🟡 了解即可。

### 3.4 ReLU：现代默认，简单到不像话

**ReLU（Rectified Linear Unit，整流线性单元）= max(0, z)**：正数原样通过，负数一刀切成 0。就这么简单，却是 2012 年后深度学习爆发的关键零件之一。为什么它能治梯度消失？看导数：

```
relu′(z) = 1   （z > 0 时）
relu′(z) = 0   （z < 0 时）
```

**正区间的梯度恒等于 1**——反向传播连乘时，乘 1 不衰减，梯度可以原封不动地穿过任意多层送抵第一层。这就是"缓解梯度消失"的全部机制。外加两个工程优点：计算是零成本的比较操作（sigmoid 要算 e 的指数）；负数一刀切带来稀疏激活（一部分神经元输出 0，天然抗过拟合）。

两个已知的坑，一句话了解（🟡）：

- **dying ReLU（神经元坏死）**：如果某个神经元的 z 长期为负，它输出恒 0、梯度恒 0，参数永远不再更新——这个神经元就"死"了。权重初始化太差或学习率太大时容易大面积发生；
- **LeakyReLU**：给负区间留一条小缝 `f(z) = max(0.01z, z)`，负数也有 0.01 的梯度，防坏死：`np.where(z > 0, z, 0.01*z)` 在 z=[−10, −1, 0, 1, 10] 上输出 `[−0.1, −0.01, 0, 1, 10]`。实战中线性区的默认值仍然是朴素 ReLU。

**本节小结**：sigmoid 导数最大 0.25、两端饱和，多层连乘导致梯度消失，只配当二分类输出层；tanh 是零中心版 sigmoid，同样的病。ReLU = max(0, z)，正区间梯度恒为 1，不饱和、零成本，是隐藏层的现代默认；代价是可能坏死，变体 LeakyReLU 留 0.01 的负区间梯度。

---

## 4. 前向传播：逐层矩阵推导 🟢

### 4.1 是什么（直觉）：数据像水流一样逐层被变换

"前向传播"就是模块 2 的 `P = softmax(X @ W + b)` 多叠几层。以本文主线的网络（64 输入 → 128 隐藏 → 64 隐藏 → 10 输出）为例，单个样本 x 的旅程：

```
x (64维)
  │  z₁ = W₁ᵀx + b₁        ← 第一层：64 个特征 → 128 个神经元的倾向分
  │  a₁ = relu(z₁)          ← 128 个"学出来的新特征"
  ▼
a₁ (128维)
  │  z₂ = W₂ᵀa₁ + b₂       ← 第二层：128 个新特征 → 64 个神经元的倾向分
  │  a₂ = relu(z₂)          ← 64 个"新特征的新特征"
  ▼
a₂ (64维)
  │  z₃ = W₃ᵀa₂ + b₃       ← 输出层：64 → 10 个类别的分数（logit）
  │  P  = softmax(z₃)       ← 模块 2 §5，压成 10 个概率
  ▼
P (10维) → 和 one-hot 标签算交叉熵 → 损失 L
```

每一层都是同一个模子：**线性变换（W·上层的输出 + b）→ 激活函数**。§1 说的"机器自己学特征"就发生在这里：第一层把 64 个像素组合成 128 个新特征，第二层再组合成 64 个，输出层在这 64 个高级特征上做它最擅长的线性分类。**MLP = 自动特征工程（隐藏层）+ 逻辑回归（输出层）**，这句话值回本文一半篇幅。

> 记号提醒：上面写成 `W₁ᵀx` 是按"每列一个神经元"的习惯；代码里我们用模块 2 的约定——**一行一个样本、X 形状 (n, d)、W 形状 (d_in, d_out)**，层计算统一写成 `z = A_in @ W + b`。两种写法只差一个整体转置，别混着用就行。本文全部用后者。

### 4.2 形状推导：初学者最大的坎，用一张表焊死

反向传播的 bug 十有八九是形状错位。养成习惯：**写每一层之前，先在纸上把每个张量的形状推一遍**。batch size = 32 时，主线网络的完整形状账本：

| 步骤 | 运算                    | 输入形状            | 输出形状    | 备注                             |
| ---- | ----------------------- | ------------------- | ----------- | -------------------------------- |
| 输入 | X                       |                     | (32, 64)    | 32 张图 × 64 像素                |
| 层1  | z₁ = X @ W₁ + b₁        | (32,64)@(64,128)+128 | (32, 128)   | b₁ 广播到每一行                  |
| 层1  | a₁ = relu(z₁)           | (32, 128)           | (32, 128)   | 逐元素，形状不变                 |
| 层2  | z₂ = a₁ @ W₂ + b₂       | (32,128)@(128,64)+64 | (32, 64)    |                                  |
| 层2  | a₂ = relu(z₂)           | (32, 64)            | (32, 64)    |                                  |
| 输出 | z₃ = a₂ @ W₃ + b₃       | (32,64)@(64,10)+10   | (32, 10)    | 每行 10 个 logit                 |
| 输出 | P = softmax(z₃)         | (32, 10)            | (32, 10)    | **按行**归一化（axis=1）         |

三条铁律：

1. **矩阵乘法的形状规则**：(m, k) @ (k, j) → (m, j)。内维 k 必须相等，外维 m、j 留给结果。W 的形状永远是 `(输入维度, 输出维度)`；
2. **batch 维度永远幸存**：第一维 32 从头带到尾，每层只是换"每行装什么"（64 像素 → 128 特征 → 64 特征 → 10 分数）；
3. **b 靠广播**：b 形状 (128,)，加在 (32, 128) 上自动复制 32 份——每个样本加同一份偏置（NumPy 文档的广播规则）。

### 4.3 怎么用：完整 numpy 前向代码

```python
rng = np.random.default_rng(42)

def he_init(d_in, d_out):
    # He 初始化：按输入维度缩放方差，§4.4 讲为什么
    return rng.normal(0, np.sqrt(2 / d_in), (d_in, d_out))

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)      # 模块 2 §5.5：减最大值防溢出
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

# 参数：64 → 128 → 64 → 10
W1, b1 = he_init(64, 128), np.zeros(128)
W2, b2 = he_init(128, 64), np.zeros(64)
W3, b3 = he_init(64, 10),  np.zeros(10)

batch = X[:32] / 16.0                       # 取 32 张图，像素缩到 0~1
z1 = batch @ W1 + b1;  a1 = relu(z1)
z2 = a1 @ W2 + b2;     a2 = relu(z2)
z3 = a2 @ W3 + b3;     P = softmax(z3)
for name, t in [('X', batch), ('z1', z1), ('a1', a1), ('z2', z2), ('a2', a2),
                ('z3', z3), ('P', P)]:
    print(f"{name}: {t.shape}")
# X: (32, 64)  z1: (32, 128)  a1: (32, 128)  z2: (32, 64)  a2: (32, 64)
# z3: (32, 10)  P: (32, 10)
print("每行概率和:", P.sum(axis=1).round(6)[:5])    # [1. 1. 1. 1. 1.]
print("参数量:", 64*128 + 128 + 128*64 + 64 + 64*10 + 10)   # 17226
```

和 §4.2 的表逐行对上。**17226 个参数**——这就是梯度下降要调的旋钮总数（§7 的 PyTorch 同结构网络参数数目一字不差）。反向传播的任务：算出 L 对这 17226 个数每一个的偏导数。

### 4.4 参数初始化：为什么不能全零

模块 1、2 里我们习惯 `theta = np.zeros(...)` 从全零出发，一直很健康。**多层网络里全零初始化是灾难**，而且灾难的原因可以亲手算出来。做一个单步实验——全零初始化一个小网络，算一次前向+反向，看梯度长什么样（反向公式 §5 才推，这里只看结果）：

```python
Xb = X[:32] / 16.0
Yb = np.eye(10)[y[:32]]
n = 32
W1 = np.zeros((64, 8));  b1 = np.zeros(8)
W2 = np.zeros((8, 10));  b2 = np.zeros(10)

z1 = Xb @ W1 + b1                        # 全 0
a1 = np.maximum(0, z1)                   # relu(0)=0，全 0
z2 = a1 @ W2 + b2                        # 全 0
P0 = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)   # 每类概率恒 0.1
print("loss =", -np.mean(np.log(P0[np.arange(n), y[:32]])))   # 2.3025850929940455 = ln(10)

d2 = (P0 - Yb) / n                       # 输出层误差（§5.3 的结论，先借用）
dW2 = a1.T @ d2                          # a1 全 0 → dW2 全 0
d1 = (d2 @ W2.T) * (z1 > 0)              # W2 全 0 → d1 全 0
dW1 = Xb.T @ d1                          # 全 0
print("dW1 最大绝对值:", np.abs(dW1).max())   # 0.0
print("dW2 最大绝对值:", np.abs(dW2).max())   # 0.0
print("db2 最大绝对值:", np.abs(d2.sum(axis=0)).max())   # 0.025 ← 唯一非零的梯度
```

读这行链条，**全零是一处会自我复制的癌症**：W=0 → 激活 a=0 → 对 W 的梯度 = aᵀδ = 0，对上一层的回传 = δWᵀ = 0。更新一步，W 纹丝不动，下一步一模一样——**W 永远是 0，只有输出层的 b 在学**（它只能学出"每个类别的先验频率"）。用 §5 的完整训练代码实测 300 个 epoch：loss 从 2.3026（恰好 ln 10，纯瞎猜的损失）开始，300 轮后仍卡在 **2.3017**，测试准确率 8.2% 不如瞎蒙。把 0 换成任何**所有神经元相同的值**（比如全 0.5）结果一样卡住 2.3017——这叫**对称性灾难**：同一层的神经元初始值相同，收到的梯度就相同，更新后仍然相同，128 个神经元永远是彼此的克隆，等价于每层只有 1 个神经元。

**打破对称的办法：随机初始化，而且方差要按输入维度缩放。** 直觉：z = w·x + b 是 d 个输入的加权和，输入越多 z 的方差越大（概率文档：独立随机变量求和，方差相加）。d=64 的层和 d=4096 的层若用同一方差的 w，后者的 z 会大 64 倍——激活值层层放大（爆炸）或缩小（消失）。所以权重标准差要随 d_in 反向缩放：

```
Xavier/Glorot（配 sigmoid/tanh）：std = sqrt(2 / (d_in + d_out))
He/Kaiming（配 ReLU）：           std = sqrt(2 / d_in)
```

主线网络第一层（64→128）：He 的 std = sqrt(2/64) = **0.1768**，Xavier 的 std = sqrt(2/192) = 0.1021。本文 numpy 版用 He（`he_init` 就是上面那行 `np.sqrt(2/d_in)`）；PyTorch 的 `nn.Linear` 默认用 U(−1/√d_in, 1/√d_in) 的均匀分布，效果同级。§6.4 的实测表会告诉你"初始化选对"和"选对激活函数"是同一场战争的两条战线。

**本节小结**：前向传播 = 每层重复"线性变换 + 激活"，MLP = 自动特征工程 + 逻辑回归输出层。形状铁律：(m,k)@(k,j)→(m,j)、batch 维幸存、b 广播。全零（或全同值）初始化会自我复制：梯度恒 0 或神经元互为克隆，loss 永远卡在 ln(10)=2.3026 附近；必须随机初始化且按输入维度缩放方差——ReLU 配 He（std=√(2/d_in)），sigmoid/tanh 配 Xavier（std=√(2/(d_in+d_out))）。

---

## 5. 反向传播：链式法则逐层推梯度 🟢（本模块心脏）

### 5.1 复习：链式法则

微积分文档（`../阶段一/00-微积分前置知识.md` §3.2）的核心结论：**函数套函数时，总变化率 = 每一层变化率的连乘**：

```
y = f(u),  u = g(x)   ⟹   dy/dx = dy/du · du/dx
```

中间的 du 一上一下"约掉"，像链条一环扣一环。该文档 §3.3 还预告过：**反向传播没有新数学，它就是链式法则在多层网络上的批量、自动化执行**。本节就是把这句话兑现。

梯度下降需要的还是那老三样（模块 1 §4）：算损失对参数的梯度 ∂L/∂θ，然后 `θ ← θ − η·∂L/∂θ`。新问题是：现在参数散在三层里（W₁, b₁, W₂, b₂, W₃, b₃），损失 L 到 W₁ 之间隔着 softmax、交叉熵、两层 relu、三次矩阵乘——怎么算 ∂L/∂W₁？答：沿着"L ← z₃ ← a₂ ← z₂ ← a₁ ← z₁ ← W₁"这条链，**从后往前**，一节一节把局部导数乘回去。

### 5.2 计算图：前向存中间值，反向逐层退

把 §4.1 的流程画成"计算图"，每个节点是一次运算、每条边是一个张量：

```
前向（从左到右算值，顺手把中间结果存进缓存）：
X → [@W₁+b₁] → z₁ → [relu] → a₁ → [@W₂+b₂] → z₂ → [relu] → a₂ → [@W₃+b₃] → z₃ → [softmax+CE] → L

反向（从右到左传梯度，每一节都用到前向存下的中间值）：
                    δ₁ ← δ₂ ←──────────── δ₃=(P−Y)/n ←──────────── dL/dL=1
```

为什么前向要存 z₁、a₁、z₂、a₂？因为反向每一节都要用它们：

- relu 的导数 relu′(z₁) = (z₁ > 0)，要用**前向存的 z₁**；
- ∂L/∂W₂ = a₁ᵀδ₂ 要用**前向存的 a₁**。

这就是反向传播的完整算法框架：**前向走一遍（存缓存）→ 反向走一遍（用缓存逐节乘链式法则）**。所有深度学习框架（PyTorch 的 autograd，§7）内部就是这张图。

### 5.3 输出层起点：δ₃ = P − Y（模块 2 的结论直接搬）

链条的第一节我们其实已经推过。模块 2 §5.6 的结论：**softmax + 交叉熵合体后，损失对 logit 的梯度恰好是"预测 − 真实"**：

```
∂L/∂z₃ = (1/n) · (P − Y)
```

- `P`：softmax 输出，(n, 10)，每行是 10 个类别的预测概率；
- `Y`：one-hot 标签，(n, 10)，每行正确类别是 1 其余是 0；
- `P − Y`：逐元素相减——正确类别位置上 = 预测概率 − 1（负的，欠了多少），其他位置 = 预测概率 − 0（正的，多给了多少）。**每个样本的"误差方向"一目了然**；
- `(1/n)`：损失是 n 个样本的平均，求导后 1/n 跟着下来（模块 2 §3.3）。

模块 2 只用到它来算 ∂L/∂W（单层，一步到头）。现在它是**反向传播的起点**——从这个 δ₃ 出发，一层层往回传。记 δ₃ = (P − Y)/n（字母 δ 读作 delta，深度学习文献里"传回某层的误差信号"的通用记号）。

### 5.4 逐层推导：三个公式，每个都配形状核验和一行 numpy

以倒数第二层（a₂ → z₃ → L）为例，把三个推导模式各推一遍，之后所有层都是复制粘贴。

**① 权重梯度：∂L/∂W₃ = a₂ᵀδ₃**

直觉先行：W₃[i, j] 是"第 i 个隐藏特征对第 j 个类别分数"的权重。它动一点，z₃[:, j] 动的幅度 = a₂[:, i]（**输入越大，同一个权重撬动的输出越大**——z₃ = a₂@W₃ + b₃，∂z₃/∂W₃[i,j] 就是 a₂ 的第 i 列）。链式法则：∂L/∂W₃[i,j] = （a₂ 第 i 列）·（δ₃ 第 j 列）。把 64×10 个格子一次性写成矩阵：

```
∂L/∂W₃ = a₂ᵀ @ δ₃
```

**形状核验**：a₂ᵀ 是 (64, n)，δ₃ 是 (n, 10) → 结果 (64, 10)，恰好和 W₃ 同形状 ✓（梯度必须和参数同形状，不然没法 `W -= lr * grad`，这条可以当万能校验器）。

**② 偏置梯度：∂L/∂b₃ = δ₃ 按列求和**

b₃[j] 同时加在所有样本的 z₃[:, j] 上（广播），所以它的梯度 = 所有样本贡献之和 = δ₃ 的第 j 列求和：`db3 = d3.sum(axis=0)`。形状：(10,)，和 b₃ 一致 ✓。注意**没有 1/n**——1/n 已经含在 δ₃ 里了。

**③ 回传给下一层的误差：δ₂ = (δ₃W₃ᵀ) ⊙ relu′(z₂)**

这一步是"穿层"，两个动作：

- **先过线性层**：a₂[j] 影响 L 的全部路径是"经过 z₃ 的 10 个分量"，每条路径的权重是 W₃[j, :]。把 10 条路径的贡献加起来 = δ₃ @ W₃ᵀ（形状 (n,10)@(10,64) → (n,64)，和 a₂ 同形状 ✓）；
- **再过激活函数**：a₂ = relu(z₂)，链式法则乘上 relu 的局部导数。relu′(z₂) = (z₂ > 0)，逐元素 0 或 1——**负数位置把梯度信号截断（门关着），正数位置原样放行**。

`⊙` 是逐元素乘（Hadamard 积，线性代数文档），numpy 里就是 `*`。合起来：

```
δ₂ = (δ₃ @ W₃ᵀ) * (z₂ > 0)
```

然后对第二层重复 ①②：∂L/∂W₂ = a₁ᵀδ₂，∂L/∂b₂ = δ₂.sum(axis=0)；再穿回第一层：δ₁ = (δ₂W₂ᵀ)⊙relu′(z₁)，∂L/∂W₁ = Xᵀδ₁，∂L/∂b₁ = δ₁.sum(axis=0)。**整条反向链只有三个模式在循环：权重梯度 = 上层激活的转置乘误差；偏置梯度 = 误差按列求和；误差回传 = 乘权重转置再 ⊙ 激活导数。**

汇总成一张表（背这张表就背下了反向传播）：

| 梯度    | 公式                        | 形状推导                       | numpy 一行                     |
| ------- | --------------------------- | ------------------------------ | ------------------------------ |
| δ₃      | (P − Y)/n                   | (n,10)                         | `d3 = (P - Y) / n`             |
| ∂L/∂W₃ | a₂ᵀδ₃                       | (64,n)@(n,10) → (64,10)        | `dW3 = a2.T @ d3`              |
| ∂L/∂b₃ | δ₃ 列和                     | (n,10)→(10,)                   | `db3 = d3.sum(axis=0)`         |
| δ₂      | (δ₃W₃ᵀ)⊙relu′(z₂)           | (n,10)@(10,64) → (n,64)        | `d2 = (d3 @ W3.T) * (z2 > 0)`  |
| ∂L/∂W₂ | a₁ᵀδ₂                       | (128,n)@(n,64) → (128,64)      | `dW2 = a1.T @ d2`              |
| ∂L/∂b₂ | δ₂ 列和                     | (n,64)→(64,)                   | `db2 = d2.sum(axis=0)`         |
| δ₁      | (δ₂W₂ᵀ)⊙relu′(z₁)           | (n,64)@(64,128) → (n,128)      | `d1 = (d2 @ W2.T) * (z1 > 0)`  |
| ∂L/∂W₁ | Xᵀδ₁                        | (64,n)@(n,128) → (64,128)      | `dW1 = X.T @ d1`               |
| ∂L/∂b₁ | δ₁ 列和                     | (n,128)→(128,)                 | `db1 = d1.sum(axis=0)`         |

### 5.5 微型例子：2-2-2 网络，拿纸笔完整手算一遍

公式表看懂了不算懂，亲手算一遍才算。造一个能装进口袋的网络：**2 维输入、1 个 2 神经元隐藏层（ReLU）、2 类输出（softmax），单样本**。参数故意取简单小数，每个中间值都能用草稿纸验算：

```
x = [1.0, 2.0]，真实标签 = 第 1 类，Y = [0, 1]

W₁ = [[0.5, −0.5],     b₁ = [0.1, −0.2]
      [0.3,  0.8]]
W₂ = [[0.6, −0.4],     b₂ = [0.0, 0.1]
      [−0.7, 0.2]]
```

**前向（跟着算）**：

```
z₁ = x·W₁ + b₁
   第1个神经元: 1.0×0.5 + 2.0×0.3 + 0.1 = 0.5+0.6+0.1 = 1.2
   第2个神经元: 1.0×(−0.5) + 2.0×0.8 + (−0.2) = −0.5+1.6−0.2 = 0.9
z₁ = [1.2, 0.9]
a₁ = relu(z₁) = [1.2, 0.9]          （两个都是正数，原样放行）

z₂ = a₁·W₂ + b₂
   类0: 1.2×0.6 + 0.9×(−0.7) + 0.0 = 0.72−0.63 = 0.09
   类1: 1.2×(−0.4) + 0.9×0.2 + 0.1 = −0.48+0.18+0.1 = −0.20
z₂ = [0.09, −0.20]

P = softmax(z₂)：减最大值 0.09 后算 e⁰=1、e^(−0.29)≈0.7483
   P = [1/1.7483, 0.7483/1.7483] = [0.5720, 0.4280]

L = −log(P[1]) = −log(0.4280) = 0.8486
```

读一下这个故事：正确答案是类 1，模型却只给了 0.428 的把握（还给错误的类 0 分了 0.572），损失 0.8486。模型"不太行"，接下来反向传播要算出**每个参数各自该为这 0.8486 负多少责**。

**反向（跟着算）**：

```
δ₂ = P − Y = [0.5720, 0.4280] − [0, 1] = [0.5720, −0.5720]
   （单样本 n=1，不除）
   读法：类0 的概率多了 0.572（要压下去），类1 的概率欠了 0.572（要抬上来）

∂L/∂W₂ = a₁ᵀδ₂（外积：a₁ 的每个元素 × δ₂ 的每个元素）
       = [[1.2×0.5720,  1.2×(−0.5720)],     = [[ 0.6864, −0.6864],
          [0.9×0.5720,  0.9×(−0.5720)]]        [ 0.5148, −0.5148]]
   读法：W₂[0,0]=0.6 的梯度 +0.6864 —— 更新时 W₂ -= lr×0.6864，这个"把 a₁[0]
   算到类0 头上"的权重被削小；W₂[0,1] 的梯度 −0.6864，会被加大。方向全对：
   压类0、抬类1。
∂L/∂b₂ = δ₂ = [0.5720, −0.5720]

回传到隐藏层：
δ₂·W₂ᵀ = [0.5720×0.6 + (−0.5720)×(−0.4),   0.5720×(−0.7) + (−0.5720)×0.2]
       = [0.3432+0.2288,                    −0.4004−0.1144]
       = [0.5720, −0.5148]
relu′(z₁) = [1, 1]  （z₁ 两个分量都是正数，门全开）
δ₁ = [0.5720, −0.5148] ⊙ [1, 1] = [0.5720, −0.5148]
   读法：隐藏神经元1 的输出被"算到类0 头上"用得多（W₂ 第1行 0.6 > −0.4），
   所以它的梯度为正——更新会压低它；神经元2 则相反。

∂L/∂W₁ = xᵀδ₁ = [[1.0×0.5720,  1.0×(−0.5148)],    = [[0.5720, −0.5148],
                 [2.0×0.5720,  2.0×(−0.5148)]]       [1.1440, −1.0296]]
∂L/∂b₁ = δ₁ = [0.5720, −0.5148]
```

注意 ∂L/∂W₁ 的第二行恰好是第一行的 2 倍——因为 x = [1, 2]，输入大 2 倍，对应的权重梯度就大 2 倍（"输入越大权重责任越大"，§5.4 ① 的直觉在数字里现身）。

**更新一步验证方向**：学习率 0.1 走一小步，所有参数 `p -= 0.1 × grad`，重新前向算损失——**L 从 0.8486 降到 0.4073**。一次小小的更新，损失腰斩，梯度的方向是对的。

以上每个数字都是真实计算的，下面这段代码就是手算过程的逐行翻译，跑出来逐字核对（文末还有用"数值梯度"对答案的版本，§6 讲）：

```python
x  = np.array([1.0, 2.0])
Y  = np.array([0.0, 1.0])          # 正确答案是第 1 类
W1 = np.array([[0.5, -0.5],
               [0.3,  0.8]])
b1 = np.array([0.1, -0.2])
W2 = np.array([[0.6, -0.4],
               [-0.7, 0.2]])
b2 = np.array([0.0, 0.1])

# ---- 前向 ----
z1 = x @ W1 + b1                   # [1.2 0.9]
a1 = np.maximum(0, z1)             # [1.2 0.9]
z2 = a1 @ W2 + b2                  # [ 0.09 -0.2 ]
e = np.exp(z2 - z2.max())
P = e / e.sum()                    # [0.572 0.428]
L = -np.log(P[1])                  # 0.8486

# ---- 反向（三个模式循环）----
d2 = P - Y                                   # [ 0.572 -0.572]  输出层误差
dW2 = np.outer(a1, d2)                       # [[ 0.6864 -0.6864] [ 0.5148 -0.5148]]
db2 = d2
d1 = (d2 @ W2.T) * (z1 > 0)                  # [ 0.572 -0.5148]  回传
dW1 = np.outer(x, d1)                        # [[ 0.572  -0.5148] [ 1.144  -1.0296]]
db1 = d1

lr = 0.1
W1n, b1n = W1 - lr * dW1, b1 - lr * db1
W2n, b2n = W2 - lr * dW2, b2 - lr * db2
z1n = x @ W1n + b1n; a1n = np.maximum(0, z1n)
z2n = a1n @ W2n + b2n
en = np.exp(z2n - z2n.max()); Pn = en / en.sum()
print("更新后 loss:", round(float(-np.log(Pn[1])), 4))   # 0.4073
```

**为什么这个例子重要**：深度学习面试和笔试里"手算一遍反向传播"是保留节目；更重要的是，你亲手算过之后，§5.4 那张表里的每一行都不再是符号，而是"就是刚才草稿纸上那一步"。

### 5.6 完整可运行的 numpy MLP：前向 + 反向 + 训练

把 §4.3 的前向和 §5.4 的反向装进函数，接上模块 1 的 mini-batch 训练循环（§4.5，唯一的变化是参数从 1 个 theta 变成 6 个矩阵），在 digits 上真实训练：

```python
from sklearn.model_selection import train_test_split

Xs = X / 16.0          # 特征缩放：像素 0~16 → 0~1（模块 1 §4.3 的标准动作）
X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.25, random_state=42)
Y_tr = np.eye(10)[y_tr]          # one-hot（模块 2 §5.3）

def forward(Xb, params):
    W1, b1, W2, b2, W3, b3 = params
    z1 = Xb @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2;  a2 = relu(z2)
    z3 = a2 @ W3 + b3
    P = softmax(z3)
    return P, (Xb, z1, a1, z2, a2)          # 缓存：反向要用（§5.2）

def backward(Yb, P, cache, params):
    W1, b1, W2, b2, W3, b3 = params
    Xb, z1, a1, z2, a2 = cache
    n = len(Xb)
    d3 = (P - Yb) / n                        # §5.3：输出层起点
    dW3 = a2.T @ d3;  db3 = d3.sum(axis=0)   # 模式①②
    d2 = (d3 @ W3.T) * (z2 > 0)              # 模式③：穿层
    dW2 = a1.T @ d2;  db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (z1 > 0)
    dW1 = Xb.T @ d1;  db1 = d1.sum(axis=0)
    return [dW1, db1, dW2, db2, dW3, db3]

def loss_acc(params, X_, y_):
    P, _ = forward(X_, params)
    loss = -np.mean(np.log(np.clip(P[np.arange(len(y_)), y_], 1e-15, 1)))
    return loss, (P.argmax(axis=1) == y_).mean()

rng = np.random.default_rng(42)
W1, W2, W3 = he_init(64, 128), he_init(128, 64), he_init(64, 10)
b1, b2, b3 = np.zeros(128), np.zeros(64), np.zeros(10)   # b 初始化为 0 是安全的（灾难只针对 W，§4.4）
params = [W1, b1, W2, b2, W3, b3]

lr, bs, n = 0.1, 64, len(X_tr)
for ep in range(300):
    idx = rng.permutation(n)                 # 每个 epoch 打乱（模块 1 §4.5 SGD）
    for s in range(0, n, bs):
        b = idx[s:s + bs]
        P, cache = forward(X_tr[b], params)
        grads = backward(Y_tr[b], P, cache, params)
        params = [p - lr * g for p, g in zip(params, grads)]   # θ ← θ − η·∂L/∂θ
    if ep % 60 == 0 or ep == 299:
        l_tr, a_tr = loss_acc(params, X_tr, y_tr)
        _, a_te = loss_acc(params, X_te, y_te)
        print(f"epoch {ep:3d}: loss={l_tr:.4f} 训练acc={a_tr:.4f} 测试acc={a_te:.4f}")
# epoch   0: loss=1.4614 训练acc=0.5330 测试acc=0.5333
# epoch  60: loss=0.0170 训练acc=0.9985 测试acc=0.9778
# epoch 120: loss=0.0054 训练acc=1.0000 测试acc=0.9778
# epoch 180: loss=0.0030 训练acc=1.0000 测试acc=0.9800
# epoch 240: loss=0.0020 训练acc=1.0000 测试acc=0.9800
# epoch 299: loss=0.0015 训练acc=1.0000 测试acc=0.9800
```

**你刚刚从零手写了一个神经网络，并看着它学会认手写数字**：loss 1.4614 → 0.0015，测试准确率 98.0%——超过 §1 逻辑回归的 0.9733，也追平了 §1 的 sklearn 黑盒（0.9778）。60 个 epoch 后训练准确率 100%、测试停在 98%——训练/测试之间那 2 个点的缝就是过拟合的雏形（模块 1 §5.3 的诊断表），§8 的早停和正则化就是为这条缝准备的。

对了，§4.4 提到的"全零初始化 loss 卡在 2.3017"就是把上面 `he_init` 换成 `np.zeros` 跑 300 轮的实测结果，现在你有代码了，可以自己复现。

**本节小结**：反向传播 = 链式法则从损失出发逐层往回乘，前向存缓存、反向用缓存。起点是模块 2 的结论 δ₃=(P−Y)/n；之后只有三个模式循环：dW = 上层激活的转置乘误差、db = 误差按列求和、误差回传 = 乘权重转置再 ⊙ 激活导数。形状核验（梯度必须和参数同形状）是最便宜的调试器。2-2-2 微型网络手算一遍前向+反向（loss 0.8486，更新一步降到 0.4073），再跑通完整 numpy MLP（digits 测试准确率 98.0%）。

---

## 6. 梯度检查与调试 🟢

### 6.1 是什么（直觉）：用定义给反向传播"对答案"

§5 的 backward 有 9 行公式，任何一个符号、转置、求和轴写错，loss 都可能照样下降一点（梯度大体方向对、细节错），这种 bug 最难抓。好在有一个**不依赖任何推导**的裁判：回到导数的定义本身。

微积分文档里导数的定义：f′(x) ≈ [f(x+ε) − f(x−ε)] / (2ε)（**中心差分**，比单侧差分 [f(x+ε)−f(x)]/ε 精度高一阶）。做法粗暴有效：**把每个参数单独 ±ε 扰动一下，看 loss 变多少**——这就是"数值梯度"。它只用前向（你确信写对的部分），不算任何链式法则。然后：

```
解析梯度（§5 backward 算的） ≈ 数值梯度（定义算的）  ⟹  backward 写对了
```

这就是**梯度检查**（gradient checking），是手写神经网络的**标配测试**，也是手写练习的核心通过手段。

### 6.2 怎么用：数值梯度实现 + 对答案

```python
def loss_fn(params, Xb, Yb):
    P, _ = forward(Xb, params)
    return -np.mean(np.sum(Yb * np.log(np.clip(P, 1e-15, 1)), axis=1))

def numerical_grad(params, Xb, Yb, eps=1e-6):
    """中心差分：把每个参数分别 ±eps，看 loss 变多少"""
    grads = []
    for p in params:
        g = np.zeros_like(p)
        it = np.nditer(p, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps;  lp = loss_fn(params, Xb, Yb)
            p[i] = old - eps;  lm = loss_fn(params, Xb, Yb)
            p[i] = old
            g[i] = (lp - lm) / (2 * eps)
            it.iternext()
        grads.append(g)
    return grads
```

两个工程细节：ε 取 1e-6 量级（太大截断误差大，太小浮点舍入误差大，1e-5~1e-7 之间都稳）；**数值梯度是 O(参数量×2) 次前向**，17226 个参数 × 2 × 每次前向——大网络上跑不起。所以标准姿势是**搭一个小号网络 + 一个小 batch 来检查**（公式和大小无关，5 个样本、8 神经元的网络就能验证全部 9 行推导）：

```python
rng6 = np.random.default_rng(42)
small = [he_init(64, 8), np.zeros(8), he_init(8, 4), np.zeros(4), he_init(4, 10), np.zeros(10)]
Xb, Yb = X_tr[:5], np.eye(10)[y_tr[:5]]
P, cache = forward(Xb, small)
ana = backward(Yb, P, cache, small)      # 解析梯度（§5 的 9 行）
num = numerical_grad(small, Xb, Yb)      # 数值梯度（定义算的）
for name, a_, n_ in zip(['dW1','db1','dW2','db2','dW3','db3'], ana, num):
    print(f"{name}: allclose={np.allclose(a_, n_, atol=1e-5)}  最大误差={np.abs(a_ - n_).max():.2e}")
# dW1: allclose=True  最大误差=4.04e-10
# db1: allclose=True  最大误差=2.46e-10
# dW2: allclose=True  最大误差=2.55e-10
# db2: allclose=True  最大误差=1.87e-10
# dW3: allclose=True  最大误差=4.67e-10
# db3: allclose=True  最大误差=3.13e-10
```

6 组参数全部对上，最大误差 10⁻¹⁰ 量级（中心差分的理论误差就是 O(ε²) = 10⁻¹² 级，加上浮点噪声）——§5 的推导一行没错。**记住这个工作流：先在小网络上过梯度检查，再上真实数据训练。** 顺序反了，你会花三天怀疑人生。

### 6.3 常见 bug 清单（梯度检查能抓住每一个）

| bug | 症状 | 例子 |
| --- | ---- | ---- |
| 梯度符号反了 | loss 不降反升（如果在训练） | `d1 = -(d2 @ W2.T) * ...` |
| 忘了除 n | 梯度全部大 n 倍，batch size 一变学习率就失灵 | `d3 = (P - Yb)` 漏了 `/n` |
| 激活导数用错输入 | 梯度数值细节错但训练照常（最阴险） | sigmoid 的 σ′ 吃 z（或由其输出 a 算 a(1−a)），拿错张量就全错 |
| 形状错位靠广播蒙混 | 不报错，结果静默错 | (n,10) 和 (10,) 相减时多写/少写 keepdims |
| softmax 忘减最大值 | loss 变 NaN | 模块 2 §5.5 |
| 缓存张量拿错层 | 某一层梯度对不上 | a1 和 a2 写串 |

前两个用梯度检查实测抓一遍（在小网络上各植入一个 bug）：

```python
def backward_buggy(Yb, P, cache, params, bug):
    W1, b1, W2, b2, W3, b3 = params
    _, z1, a1, z2, a2 = cache
    n = len(P)
    d3 = (P - Yb) if bug == 'forget_n' else (P - Yb) / n     # bug①：忘除 n
    dW3 = a2.T @ d3; db3 = d3.sum(axis=0)
    d2 = (d3 @ W3.T) * (z2 > 0)
    dW2 = a1.T @ d2; db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (z1 > 0)
    if bug == 'sign':                                        # bug③：符号反了
        d1 = -d1
    dW1 = cache[0].T @ d1
    db1 = d1.sum(axis=0)
    return [dW1, db1, dW2, db2, dW3, db3]

for bug in ['forget_n', 'sign']:
    g_bad = backward_buggy(Yb, P, cache, small, bug)
    ok = all(np.allclose(a_, n_, atol=1e-5) for a_, n_ in zip(g_bad, num))
    err = max(np.abs(a_ - n_).max() for a_, n_ in zip(g_bad, num))
    print(f"bug={bug:10s}: allclose={ok}  最大误差={err:.2e}")
# bug=forget_n  : allclose=False  最大误差=1.67e+00
# bug=sign      : allclose=False  最大误差=5.05e-01
```

allclose 立刻变 False，误差大到无法辩解——这就是梯度检查的价值：**把"好像能训练"的隐性 bug 变成"当场报警"的显性失败**。

### 6.4 梯度消失与爆炸：实测与对策

§3.2 讲过 sigmoid 梯度消失的原理（σ′≤0.25，多层连乘），§4.4 讲过初始化要按输入维度缩放。现在把这两件事放在同一张实验桌上：搭 width=8 的纯前馈网络，输出端注入单位梯度，测传回第一层的梯度大小（300 次随机试验取平均）：

| 配置                         | 深度 2 | 深度 5 | 深度 10 | 深度 20  |
| ---------------------------- | ------ | ------ | ------- | -------- |
| sigmoid + 普通 N(0,1) 初始化 | 1.5e−1 | 1.7e−2 | 4.5e−4  | **2.8e−7（消失）** |
| **ReLU + He 初始化**         | 9.5e−1 | 9.1e−1 | 6.9e−1  | **3.2e−1（稳定）** |
| ReLU + 普通 N(0,1) 初始化    | 3.8    | 29     | 7.0e+2  | **3.3e+5（爆炸）** |

三行各讲一个道理：

1. **sigmoid 行**：每过一层梯度 ×0.25，20 层后只剩千万分之一——第一层几乎收不到学习信号。这是 2012 年之前深层网络训练不动的主因；
2. **ReLU+He 行**：正区间梯度恒 1，加上按 √(2/d_in) 缩放的初始化让信号方差层层守恒——深度 20 梯度量级几乎不变。**这就是现代网络的默认配方**；
3. **ReLU+N(0,1) 行**：权重方差没缩放，每层信号放大 ~√8 倍，20 层后梯度 33 万——参数更新一步飞出十万八千里，loss 变 NaN。梯度爆炸的对策：初始化缩放（治本）、梯度裁剪（把梯度范数超过阈值的整体砍回去，治标，RNN 里常用）。

**本节小结**：数值梯度（中心差分，ε≈1e-6）是反向传播的裁判——只用前向、不依赖任何推导，解析梯度与它 allclose 才说明 backward 写对了；在小网络+小 batch 上检查（大网络算不起）。常见 bug：符号反、忘除 n、激活导数吃错输入、广播蒙混，梯度检查全能当场抓住。梯度消失（sigmoid 每层 ×0.25）与爆炸（权重方差没缩放）的实测对照：ReLU + He 初始化是现代默认配方。

---

## 7. PyTorch 重写 MLP 🟢

### 7.1 autograd 的直觉：你写前向，它自动反向

§5、§6 你手推了 9 行反向公式、又用数值梯度对了答案。一个自然的问题：每换一种网络结构（CNN、RNN、Transformer……）都要重新推一遍吗？不用——这就是 **autograd（自动微分）** 存在的意义：**你只写前向代码，框架在幕后把计算图（§5.2）记下来，`loss.backward()` 一行自动执行反向传播**。

但在信任它之前，先亲手证明它给的答案和你手推的一样。用 §5.5 的同款问题（输入@权重→softmax+交叉熵），torch 算一遍、numpy 手推导数算一遍：

```python
import torch
import torch.nn as nn

rng7 = np.random.default_rng(0)
x_np = rng7.normal(size=(3, 4))            # 3 个样本、4 维输入
W_np = rng7.normal(size=(4, 2))            # 一层权重：4 → 2
y_np = np.array([0, 1, 0])

# ---- torch：只写前向 ----
xt = torch.tensor(x_np)
Wt = torch.tensor(W_np, requires_grad=True)   # requires_grad：请帮我盯着这个参数
yt = torch.tensor(y_np)
logits = xt @ Wt
loss = nn.functional.cross_entropy(logits, yt)  # softmax+交叉熵合体（§5.3）
loss.backward()                                  # 自动反向传播，就这一行
print(Wt.grad.numpy().round(6))
# [[-0.09419   0.09419 ]
#  [ 0.128448 -0.128448]
#  [ 0.114092 -0.114092]
#  [ 0.139289 -0.139289]]

# ---- numpy 手推：∂L/∂W = Xᵀ(P−Y)/n（模块 2 §5.6 的结论）----
z_np = x_np @ W_np
e = np.exp(z_np - z_np.max(axis=1, keepdims=True))
P = e / e.sum(axis=1, keepdims=True)
dW_np = x_np.T @ (P - np.eye(2)[y_np]) / 3
print(np.allclose(Wt.grad.numpy(), dW_np, atol=1e-8))   # True
```

逐位一致。**autograd 没有魔法，它就是 §5.2 计算图的工程实现**：前向时每个算子（`@`、`cross_entropy`……）顺手登记"我的局部导数怎么算"，`backward()` 从 loss 出发沿链式法则逐节乘回去——和你 §5.4 手推的三个模式一模一样。理解了 §5，autograd 对你就不是黑盒，只是"不用自己动笔的反向传播"。

### 7.2 四件新武器：nn.Module / nn.Linear / DataLoader / 优化器

| numpy 手写版（§5.6）     | PyTorch 对应物                        | 职责                             |
| ------------------------ | ------------------------------------- | -------------------------------- |
| `W1, b1, ...` 手工管理   | `nn.Linear(64, 128)`                  | 一层 = 一个对象，参数自动注册    |
| `forward()` 手写         | `nn.Module` 子类的 `forward()`        | 前向还是你写（框架不管前向）     |
| `backward()` 手写        | `loss.backward()`                     | 反向全自动                       |
| 手写 permutation/slice   | `DataLoader(dataset, batch_size=64)`  | 打乱、分批、（可选）多进程加载   |
| `p - lr * g` 手写        | `torch.optim.SGD / Adam`              | 更新规则封装（Adam 自带自适应lr） |

### 7.3 训练循环五步法（背下来，所有 torch 训练都长这样）

```
for xb, yb in loader:
    logits = model(xb)        # ① 前向：算预测
    loss = loss_fn(logits, yb)# ② 算损失
    opt.zero_grad()           # ③ 清空旧梯度（.grad 默认累加，不清会越攒越多）
    loss.backward()           # ④ 反向：自动算所有参数的梯度
    opt.step()                # ⑤ 更新：θ ← θ − η·∂L/∂θ（一行替换 §5.6 的 zip 循环）
```

第③步是 torch 新手最容易忘的：`.grad` 的设计是**累加**（某些高级用法需要梯度累积），所以每步开始前必须清零。

### 7.4 完整训练：torch 版 MLP 在 digits 上

```python
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)          # 固定种子：初始化和 batch 打乱都可复现
np.random.seed(42)

Xtr = torch.tensor(X_tr, dtype=torch.float32);  ytr = torch.tensor(y_tr)
Xte = torch.tensor(X_te, dtype=torch.float32);  yte = torch.tensor(y_te)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(42))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
print("参数量:", sum(p.numel() for p in model.parameters()))   # 17226 —— 和 §4.3 手算一致
opt = torch.optim.Adam(model.parameters(), lr=0.001)   # Adam：自适应学习率的默认优化器
loss_fn = nn.CrossEntropyLoss()                        # softmax+交叉熵合体（不要自己先 softmax！）

for ep in range(300):
    model.train()
    for xb, yb in loader:
        logits = model(xb)               # ① 前向
        loss = loss_fn(logits, yb)       # ② 损失
        opt.zero_grad()                  # ③ 清梯度
        loss.backward()                  # ④ 反向
        opt.step()                       # ⑤ 更新
    if ep % 60 == 0 or ep == 299:
        model.eval()
        with torch.no_grad():            # 评估时关掉梯度记录，省内存省算力
            l_tr = loss_fn(model(Xtr), ytr).item()
            a_tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
            a_te = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {ep:3d}: loss={l_tr:.4f} 训练acc={a_tr:.4f} 测试acc={a_te:.4f}")
# epoch   0: loss=2.0851 训练acc=0.6110 测试acc=0.6089
# epoch  60: loss=0.0161 训练acc=0.9970 测试acc=0.9756
# epoch 120: loss=0.0021 训练acc=1.0000 测试acc=0.9822
# epoch 180: loss=0.0007 训练acc=1.0000 测试acc=0.9822
# epoch 240: loss=0.0002 训练acc=1.0000 测试acc=0.9800
# epoch 299: loss=0.0001 训练acc=1.0000 测试acc=0.9800
```

对照 §5.6 的 numpy 手写版（最终 0.9800）：loss 曲线形态一致、测试准确率打平——**两种实现殊途同归**，这正是 §7.1 证明过的事在整个网络上的重演。注意三个 torch 特有的细节：

- `nn.CrossEntropyLoss` 内部含 softmax（且数值稳定），**喂给它的是 logit 不是概率**——自己先 softmax 一遍是新手经典错误；
- `model.eval()` + `torch.no_grad()`：评估/推理时的标准搭档，前者关掉 dropout 之类的训练专属行为（本文没用到但项目会有），后者不记录计算图；
- 参数量 17226 与 §4.3 的算术完全一致——`nn.Linear(64,128)` 就是 64×128 的 W 加 128 的 b，框架里没有藏任何魔法。

**device 一句带过**：有 GPU 时 `device = 'cuda'`，然后 `model.to(device)`、每个 batch 的数据也 `.to(device)`，其余代码一字不改。digits 这种小数据 CPU 秒级跑完，本文全部用 CPU（输出也更好复现）；项目的 MNIST 全量数据上 GPU 会有几十倍加速。

**本节小结**：autograd = 计算图的工程实现，你写前向、`loss.backward()` 自动反向，实测它的梯度与手推 allclose 一致。四件武器：nn.Linear（层）、nn.Module（前向）、DataLoader（分批）、optim（更新）。训练循环五步法：forward → loss → zero_grad → backward → step（③最易忘，.grad 默认累加）。CrossEntropyLoss 吃 logit 不吃概率。torch 版 digits 测试 0.9800，与 numpy 手写版打平。

---

## 8. 工程化 Trainer：checkpoint / 早停 / 学习率调度 🟡

### 8.1 为什么训练代码要封装成类

§7.4 的训练循环只有 5 行，但真实项目里它会不停长肉：记录 loss 曲线、定期在验证集上评估、保存最优模型、早停、调学习率、打日志……散在脚本里很快就没法维护。所以项目里会把它们收进一个 `Trainer` 类（`fit()` 管训练循环、`evaluate()` 管评估、`save/load` 管存档），训练配置（lr、batch size、epochs）集中成参数。本节偏概念，逐个讲清三个组件解决什么问题，代码细节留给项目。

### 8.2 训练/验证/测试三集合：为什么两个不够了

模块 1、2 只有训练集和测试集。现在多了一个**验证集**，动机在 §5.6 的数字里已经露头：训练 acc 100%、测试 98%——**训练后期的epoch数、学习率、隐藏层大小这些超参数，本身就是"看着测试集调"会污染测试集**。测试集的角色是"假装未来的新数据"，一辈子只该用一次；反复瞄着它调参，等于间接把测试集学进去了（过拟合到测试集上）。

解法：再切一块**验证集**专门用来调超参数和做训练中的决策（早停、选最优 checkpoint），测试集封存到最后一次评估：

```python
X_tr, X_tmp, y_tr, y_tmp = train_test_split(Xs, y, test_size=0.4, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
print(len(X_tr), len(X_val), len(X_te))   # 1078 359 360 —— 60% / 20% / 20%
```

分工一句话：**训练集教模型，验证集教你调参，测试集只负责最后的宣判**。

### 8.3 checkpoint：存最优，不存最后

训练 300 个 epoch，验证准确率的高点可能在第 180 轮（之后开始过拟合缓慢下滑）。只存最后一轮 = 把过拟合的版本当成果。标准做法：**训练中每个 epoch 在验证集上评估，一旦刷新历史最优就落盘存档**：

```python
best_val_acc = 0.0
# 训练循环内，每个 epoch 末：
#     val_acc = evaluate(model, X_val, y_val)
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
torch.save({'epoch': 87,
            'model_state': model.state_dict(),      # state_dict：所有参数打包成的字典
            'best_val_acc': 0.96}, '/tmp/best.pt')

# 以后随时复活到"历史最佳状态"：
ckpt = torch.load('/tmp/best.pt', weights_only=False)
model.load_state_dict(ckpt['model_state'])
print("恢复到 epoch", ckpt['epoch'], "，当时验证准确率", ckpt['best_val_acc'])
# 恢复到 epoch 87 ，当时验证准确率 0.96
```

`state_dict` 是 torch 的存档格式：一个"参数名 → 张量"的普通字典，保存/加载都是它。训练大模型时还会把优化器状态（`opt.state_dict()`）一起存，方便断点续训。

### 8.4 早停（early stopping）：过拟合的刹车

验证集准确率不再进步时，继续训练只会把训练集背得越来越熟。给它一个**耐心值**（patience）：连续 patience 个 epoch 没刷新纪录就刹车，同时 checkpoint 机制保证你手里留的是刹车前最好的那一版：

```python
patience, wait, best = 3, 0, -np.inf
history = [0.90, 0.93, 0.95, 0.96, 0.961, 0.960, 0.959, 0.960, 0.959]  # 模拟验证准确率
for ep, va in enumerate(history):
    if va > best:
        best, wait = va, 0            # 刷新纪录：耐心回满
    else:
        wait += 1                     # 没进步：耐心 -1
        if wait >= patience:
            print(f"epoch {ep}: 连续 {wait} 轮没进步，早停。最优 {best} 在 epoch {history.index(best)}")
            break
# epoch 7: 连续 3 轮没进步，早停。最优 0.961 在 epoch 4
```

项目里 patience 常取 10~20。早停本质是一种**正则化**（模块 1 §6 的思想：主动限制模型"学太多"），零成本、效果稳，深度学习项目几乎必开。

### 8.5 学习率调度：越学步子越小

模块 1 §4.5 提过"SGD 会在最优点附近一直抖动，常用越学学习率越小来稳住"。这就是**学习率调度**（lr scheduler）：训练初期大步快跑，后期小步微调。常见三派：

- **StepLR**：每过固定 epoch 数砍一刀（如每 50 轮 ×0.1）——简单常用；
- **指数/余弦衰减**：连续平滑地变小——现代训练的主流（余弦退火）；
- **ReduceLROnPlateau**：盯着验证集指标，平台期就砍——和早停同款思路。

torch 里一行挂上：`sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)`，每个 epoch 末 `sched.step()`。概念到这就够，项目里会实际用。

**本节小结**：训练代码封装成 Trainer 类是为了收纳不断长肉的训练逻辑。三集合分工：训练集教模型、验证集调参和做训练中决策、测试集只宣判一次。三个组件：checkpoint 存验证集最优而非最后一轮（state_dict 落盘）；早停用耐心值在验证集平台期刹车（本质是正则化）；学习率调度让步子越学越小（StepLR/余弦/Plateau）。

---

## 9. 手写练习预告

接下来在 `ml/algorithms/mlp_from_scratch/` 里仅用 NumPy 手写本模块的全部核心，本文的公式就是设计图：

| 要手写的                        | 对应本文 | 核心一行代码                                   |
| ------------------------------- | -------- | ---------------------------------------------- |
| `relu(z)` 与 `relu_prime(z)`    | §3.4     | `np.maximum(0, z)` / `(z > 0)`                 |
| `he_init(d_in, d_out)`          | §4.4     | `rng.normal(0, np.sqrt(2/d_in), (d_in, d_out))` |
| `forward(X, params)`            | §4.3     | `z = a @ W + b; a = relu(z)` 逐层               |
| `backward(Y, P, cache, params)` | §5.4     | `d3=(P-Y)/n; dW=a.T@d; d=(d@W.T)*(z>0)`        |
| `numerical_grad(...)`           | §6.2     | `(f(p+eps) - f(p-eps)) / (2*eps)`              |
| 训练循环 `train(...)`           | §5.6     | `params = [p - lr*g for p, g in zip(...)]`     |

数据用 digits（`load_digits`，`X/16.0` 缩放，75/25 切分，random_state=42——与本文逐字一致）。**通过标准**：

1. **梯度检查全过**：所有参数的解析梯度与数值梯度 `np.allclose(atol=1e-5)`（§6.2 的完整流程）；
2. **digits 测试准确率 ≥ 0.97**（本文手写版跑到 0.9800，sklearn 黑盒 0.9778，低于 0.97 说明实现有问题）；
3. 全零初始化实验复现"loss 卡在 2.3017"（§4.4）。

先跑通梯度检查再上训练——§6.2 的工作流是纪律，不是建议。

---

## 完整速查表

**概念**：线性模型的天花板 = 决策边界只能直，弯问题（月牙）彻底失败 ｜ 隐藏层 = 机器自己学特征组合，MLP = 自动特征工程 + 逻辑回归输出层 ｜ 神经元 = w·x+b 加激活函数 ｜ 没有激活函数堆多少层都等价于一层（W₂(W₁x)=(W₂W₁)x） ｜ sigmoid 导数≤0.25 多层连乘→梯度消失，隐藏层默认 ReLU（正区间导数恒 1） ｜ 前向存缓存、反向用缓存逐节乘链式法则 ｜ 反向只有三个模式：dW=上层激活转置乘误差、db=误差列和、回传=乘权重转置⊙激活导数 ｜ 梯度必须和参数同形状（万能校验） ｜ 全零/全同值初始化自我复制，loss 卡 ln(10)=2.3026 ｜ He 初始化配 ReLU：std=√(2/d_in) ｜ 数值梯度（中心差分）是反向传播的裁判，小网络+小 batch 上检查 ｜ autograd=计算图工程实现，你写前向它自动反向 ｜ 训练五步法：forward→loss→zero_grad→backward→step ｜ CrossEntropyLoss 吃 logit 不吃概率 ｜ 训练/验证/测试三集合：教模型/调参/宣判 ｜ checkpoint 存最优、早停刹车、lr 调度步子渐小

**公式**：

| 名称            | 公式                                        | 一句话                             |
| --------------- | ------------------------------------------- | ---------------------------------- |
| 神经元          | a = f(w·x + b)                              | 线性加权 + 激活                    |
| 层（batch 版）  | Z = A_in @ W + b                            | W 形状 (d_in, d_out)，b 广播       |
| ReLU / 导数     | max(0,z) / z>0 ? 1 : 0                      | 正区间原样放行                     |
| sigmoid 导数    | σ(1−σ)，最大 0.25                           | 两端饱和，梯度消失之源             |
| He 初始化       | std = √(2/d_in)                             | 配 ReLU                            |
| Xavier 初始化   | std = √(2/(d_in+d_out))                     | 配 sigmoid/tanh                    |
| 输出层误差      | δ₃ = (P − Y)/n                              | 模块 2 结论，反向起点              |
| 权重梯度        | ∂L/∂W = A_inᵀ @ δ                           | 和 W 同形状                        |
| 偏置梯度        | ∂L/∂b = δ.sum(axis=0)                       | 广播的逆运算                       |
| 误差回传        | δ_prev = (δ @ Wᵀ) ⊙ f′(z_prev)              | 乘转置再过激活导数                 |
| 参数更新        | θ ← θ − η·∂L/∂θ                             | 模块 1 一字不改                    |
| 数值梯度        | [f(θ+ε) − f(θ−ε)] / 2ε                      | 中心差分，ε≈1e-6                   |

**代码**：

| 需求              | NumPy 手写                                          | PyTorch                                           |
| ----------------- | --------------------------------------------------- | ------------------------------------------------- |
| ReLU / 导数       | `np.maximum(0, z)` / `(z > 0)`                      | `nn.ReLU()`                                       |
| 一层前向          | `a @ W + b`                                         | `nn.Linear(d_in, d_out)`                          |
| softmax+交叉熵    | 先 `softmax`（减最大值）再 `-log(P[正确类])`        | `nn.CrossEntropyLoss()`（**喂 logit**）           |
| 反向              | `d3=(P-Y)/n; dW=a.T@d; d=(d@W.T)*(z>0)`             | `loss.backward()`                                 |
| 更新              | `p -= lr * g`                                       | `opt = optim.Adam(...); opt.step()`               |
| 清梯度            | ——                                                  | `opt.zero_grad()`（.grad 默认累加）               |
| 分批              | `rng.permutation(n)` 手动切片                       | `DataLoader(TensorDataset(X,y), batch_size, shuffle=True)` |
| 梯度检查          | 中心差分 + `np.allclose(atol=1e-5)`                 | 同样适用（`p.grad` 对比数值梯度）                 |
| 评估              | `(P.argmax(1) == y).mean()`                         | `model.eval()` + `with torch.no_grad():`          |
| 存档/读档         | `np.savez`                                          | `torch.save(model.state_dict(), ...)` / `load_state_dict` |
| 学习率调度        | 手动 `lr *= 0.1`                                    | `optim.lr_scheduler.StepLR(opt, ...)`             |
| 上 GPU            | ——                                                  | `model.to('cuda')`，数据也 `.to('cuda')`          |

---

## 自测题（附答案）

1. **（天花板）** 逻辑回归在 digits 上有 0.9733，在月牙数据上只有 0.8083。差距的根源是什么？多项式特征也能把月牙做到 0.9083，那隐藏层的价值在哪？
2. **（激活函数存在性）** 证明：没有激活函数的两层网络 W₂(W₁x) 和一个单层网络等价。这个结论为什么是"激活函数是多层网络存在的理由"？
3. **（ReLU）** sigmoid 的导数最大值是多少？ReLU 正区间的导数是多少？用这两个数解释为什么深层网络用 sigmoid 会梯度消失、用 ReLU 能缓解。
4. **（形状）** batch=32 的网络 64→128→64→10，写出 z1、a1、W2、dW2、δ₂（d2）各自的形状。∂L/∂W₂ = a₁ᵀδ₂ 的形状推导是什么？
5. **（初始化）** 全零初始化时，为什么只有输出层的 b 在学、W 永远不动？loss 会卡在哪个数附近？He 初始化的公式是什么、为什么按输入维度缩放？
6. **（反向传播三模式）** 默写反向传播的三个循环模式（dW、db、回传），并说明 §5.5 微型网络里 ∂L/∂W₁ 的第二行为什么恰好是第一行的 2 倍。
7. **（梯度检查）** 数值梯度的公式和 ε 的典型取值是什么？为什么要用"小网络+小 batch"做检查？它能抓住哪几类 bug？
8. **（autograd 与训练循环）** 写出 PyTorch 训练循环五步法。`zero_grad()` 忘了会怎样？`nn.CrossEntropyLoss` 的输入应该是 logit 还是概率？
9. **（三集合）** 有了训练/测试集为什么还要验证集？"盯着测试集调超参数"错在哪？
10. **（工程组件）** checkpoint 为什么存"验证集最优"而不是"最后一轮"？早停为什么本质是一种正则化？

### 参考答案

**1.** 根源：逻辑回归的决策边界只能是直线（模块 2 §6.1），月牙数据没有任何直线能分开——欠拟合，表达能力到顶。多项式路线有效但要人肉猜组合（5 次？交互项要不要？），且特征数组合爆炸（64 维的 3 次组合就 4 万多）；隐藏层让梯度下降自动学出有用的特征组合，换数据不用重造特征——特征工程从人肉穷举变成训练副产品。

**2.** 矩阵乘法结合律：W₂(W₁x) = (W₂W₁)x，括号里的乘积是一个新矩阵，整个两层网络就是单层线性模型。既然没有激活函数时堆层不增加任何表达能力（月牙数据照样学不会），那么多层网络超出单层的全部能力都只能来自层与层之间的非线性激活——这就是它存在的理由。

**3.** sigmoid 导数 σ(1−σ) 最大 0.25（z=0 处），两端趋近 0；ReLU 正区间导数恒为 1。反向传播是各层局部导数连乘：sigmoid 每层最多乘 0.25，10 层衰减到 10⁻⁶ 量级，前面的层收不到学习信号；ReLU 乘 1 不衰减，梯度可以穿任意多层（实测：20 层 sigmoid 第一层梯度 2.8e−7，ReLU+He 初始化仍有 0.32）。

**4.** z1 (32,128)、a1 (32,128)、W2 (128,64)、dW2 (128,64)、δ₂ (32,64)。∂L/∂W₂ = a₁ᵀδ₂：(128,32)@(32,64) → (128,64)，与 W2 同形状 ✓。铁律：(m,k)@(k,j)→(m,j)，梯度必须和参数同形状。

**5.** 全零时激活 a=0 → dW = aᵀδ = 0；回传误差 δWᵀ = 0 → 更浅的层梯度也是 0。唯一非零的是输出层偏置的梯度（δ₃ 列和），它只能学出类别先验频率；W 更新量为 0 于是永远为 0，自我复制。loss 卡在 ln(10)=2.3026 附近（实测 2.3017）。He：std=√(2/d_in)；因为 z 是 d_in 个输入的加权和，方差随 d_in 线性增长，权重方差按 1/d_in 缩放才能让各层信号方差守恒（ReLU 砍掉一半负值，所以分子是 2）。

**6.** ①权重梯度 = 上层激活的转置乘本层误差：dW = A_inᵀ @ δ；②偏置梯度 = 误差按列求和：db = δ.sum(axis=0)；③误差回传 = 乘权重转置再过激活导数：δ_prev = (δ @ Wᵀ) ⊙ f′(z_prev)。微型网络里 dW1 = xᵀδ₁（外积），x=[1,2]，第 j 列的梯度是 δ₁[j]·x，第二行（x₂=2）自然是第一行（x₁=1）的 2 倍——输入越大，对应权重对输出的撬动越大、责任越大。

**7.** [f(θ+ε)−f(θ−ε)]/2ε，ε≈1e-6（1e-5~1e-7 都稳）。数值梯度的代价是 2×参数量 次前向，大网络算不起；而推导公式与网络大小无关，小网络验证通过即可信。能抓住：符号反、忘除 n、激活导数用错输入、形状错位（广播蒙混）、缓存拿错层——凡是让解析梯度偏离真值的 bug 都会被 allclose=False 当场报警。

**8.** ①`logits = model(xb)` ②`loss = loss_fn(logits, yb)` ③`opt.zero_grad()` ④`loss.backward()` ⑤`opt.step()`。忘 zero_grad：.grad 默认累加，梯度越攒越大，等价于学习率不受控地翻倍，训练很快崩掉。CrossEntropyLoss 吃 logit（内部自带数值稳定的 softmax），先自己过一遍 softmax 是经典错误（损失数值会变、且丢了内部的稳定处理）。

**9.** 测试集的角色是"假装未来的新数据"，只能用一次；反复看它调超参数（epoch 数、lr、层大小）等于把测试集的信息间接学进模型，评估结果虚高。验证集专门承接"训练中的决策"（调参、早停、选 checkpoint），测试集封存到最后宣判。

**10.** 训练后期模型开始过拟合，最后一轮的验证表现往往不是最好的；存"验证集刷新纪录时"的快照，交付的才是泛化最好的版本。早停在验证指标不再进步时主动终止训练，限制了模型"把训练集背下来"的容量——和模块 1 的 Ridge/Lasso 一样，都是"主动约束模型别学太狠"，所以本质是正则化。

---

## 与后续模块的关系（学习路径对应表）

| 本文概念                       | CNN / RNN / Transformer 怎么复用                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| 层 = 线性变换 + 激活           | 卷积层、注意力层都只是**另一种"w·x+b"**（卷积是局部+共享权重的线性变换），激活照旧  |
| 反向传播三模式                 | **一字不改**。所有框架 autograd 替你执行，但读论文、调 bug 时脑子里跑的就是这张图   |
| softmax + 交叉熵输出层         | 图像分类、机器翻译（每个词一个 softmax）的标配输出，模块 2 的结论继续服役           |
| 梯度消失/爆炸、He 初始化       | Transformer 的残差连接、LayerNorm 都是这场战争的延伸武器                            |
| 训练五步法、nn.Module          | 所有 PyTorch 模型的训练循环**逐字相同**，只是 model 的定义变复杂                    |
| Trainer（checkpoint/早停/调度）| 直接复用。项目里的 Trainer 类会从 CNN 一路用到 Transformer                          |
| 梯度检查                       | 手写任何新算子（自定义层）后的标准验证手段                                          |

**直接衔接**：本文 → 在 `ml/algorithms/mlp_from_scratch/` 完成手写清单（§9 预告，梯度检查全过 + digits ≥0.97）→ 项目 `projects/fashion_mnist/`（PyTorch + 自定义 Trainer，把 §7、§8 全部落到实处）。想跑 MNIST 全量数据，模块 2 项目已缓存好：`ml/projects/mnist_classification/data/raw/mnist.npz`（X: float32 (70000,784) 0~255，y: int64），`np.load` 直接读，不用重新下载。
