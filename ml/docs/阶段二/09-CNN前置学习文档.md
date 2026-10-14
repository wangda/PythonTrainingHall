# 09 - CNN 前置学习文档（教程 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：为什么全连接网络搞不定图像、卷积层（互相关手算、stride/padding、多通道、参数量、im2col 概念）、池化层、从部件到网络（LeNet 逐层尺寸推导）、PyTorch 训练循环速成、迁移学习（冻结/微调策略、预处理匹配、学习率分配）、数据增强、手写练习预告与完整速查表。
> **本文环境**：代码按 PyTorch 2.x + torchvision 0.2x 的 API 编写。所有手算数值都是精确推导的（可用纸笔验证）；网络训练类示例请亲手运行——本模块的手写练习测试会帮你和 PyTorch 对答案。
> **优先级说明**：本模块是 **P1**（调库为主，手写关键组件一次）——卷积/池化的**前向**手写一遍理解本质，**不手写反向传播、不手写完整卷积层**，网络搭建和训练全部用 PyTorch。

---

## 本文怎么用

1. **当教程读**：从头读一遍，§2.2 的互相关手算和 §4 的 LeNet 尺寸推导务必在纸上跟着算一遍——CNN 的"神秘感"算完这两遍就没了。
2. **当速查手册用**：搭网络时忘了输出尺寸公式、参数量公式、迁移学习该冻结哪层，直接查文末【完整速查表】。

**先决知识**：

- `../阶段一/01-numpy前置学习文档.md`：多维数组切片、reshape、广播。
- `./03-逻辑回归与分类前置学习文档.md`：softmax 多分类、交叉熵、分类评估指标。
- 模块 4（MLP，如果已学）：PyTorch 的 `nn.Module` 和训练循环。**没学也不要紧**——§5 是一节自包含的 PyTorch 速成，够用。
- 本文不再使用梯度下降的手写实现，但模块 1/2 的"损失 → 梯度 → 更新"世界观会默认你已有。

**阅读约定**：🟢 必须掌握（手写练习或项目直接考）；🟡 了解即可。

**贯穿全文的例子**：CIFAR-10 数据集——32×32 彩色小图，10 个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车），训练集 5 万张。本模块的项目就是用它做迁移学习；本文的部件讲解（卷积、池化）用更小的手工数组，保证每个数字都能手算验证。

---

## 1. 为什么全连接网络搞不定图像 🟢

### 1.1 参数爆炸

模块 2 的项目里，手写数字是 8×8=64 或 28×28=784 个像素，展平后喂给逻辑回归没问题。换成一张普通的 224×224 彩色图试试：

```python
h, w, c = 224, 224, 3
n_pixels = h * w * c                      # 150,528 个输入
n_hidden = 1000                           # 假设第一个隐藏层 1000 个神经元
n_params = n_pixels * n_hidden + n_hidden # 每个神经元都要和每个像素连一条权重
print(n_params)                           # 150,528,000 —— 一亿五千万个参数！
```

一层就一亿五千万参数：内存爆炸、训练慢、数据量根本喂不饱这么多自由度的模型（必过拟合）。这是**结构性的浪费**——下面两节看浪费在哪。

### 1.2 两个被浪费的直觉：局部性与平移不变性

看一张猫的照片，你的视觉系统有两条先验，全连接网络一条都不知道：

1. **局部性**：认出"猫眼"只需要看眼睛附近的一小块像素，和照片对角另一头的像素无关。全连接让每个神经元看**全部**像素——用不上局部性。
2. **平移不变性**：猫在照片左边还是右边，都是猫。"猫眼探测器"在左上角学到的本领，应该能直接用在右下角。全连接在每个位置学**独立**的权重——同一只眼睛出现在不同位置，它当成完全不同的模式重学一遍。

卷积层就是把这两条先验**固化进网络结构**里：神经元只看局部一小块（局部性），同一个探测器扫遍全图、权重共享（平移不变性）。参数从"像素数 × 神经元数"骤降到"一小块窗口的大小"——这就是下一节的卷积。

### 1.3 从"表格"到"网格"：数据形状变了

前几个模块的数据是表格：一行一个样本，列之间没有空间关系（交换两列含义不变）。图像是**网格**：像素 (i, j) 的邻居是 (i±1, j±1)，打乱像素顺序图就毁了。张量约定（PyTorch）：

```
单张图：  (C, H, W)     通道 × 高 × 宽，如 (3, 224, 224)
一批图：  (N, C, H, W)  N 张图叠成四维，N 永远在最前面
```

记住 `NCHW` 这四个字母的顺序——CNN 时代所有报错里一半是它排的队不对。

**本节小结**：全连接处理图像会参数爆炸（224²×3 × 1000 ≈ 1.5 亿参数/层），且浪费图像的两条先验——局部性与平移不变性。卷积层把这两条固化进结构。图像张量形状 (N, C, H, W)。

---

## 2. 卷积层 🟢（本模块心脏之一）

### 2.1 是什么（直觉）：滑动的模式探测器

一个卷积核（kernel）就是一个小小的**模式模板**，比如 3×3 的"竖直边缘探测器"。卷积运算 = 拿着这个模板在图上**滑动**，每到一个位置，把模板盖在局部窗口上**逐元素相乘再求和**——输出一个数，表示"这个局部和模板的匹配程度"。扫完全图得到一张"匹配程度热力图"，叫**特征图（feature map）**。

边缘亮的地方输出大，平的地方输出小——模板滑到哪儿，就在哪儿找它的目标模式，权重全程共享（同一块模板），这就是 §1.2 的两条先验落地。

严格说 PyTorch 里的 `conv2d` 算的是**互相关**（cross-correlation，模板直接乘加），不是数学教材里的卷积（模板要先翻转 180°）。深度学习里不翻转——反正模板是学出来的，翻不翻无所谓。本文沿用工程习惯，把互相关就叫"卷积"。

### 2.2 手算一遍：竖直边缘探测器

输入 4×4 灰度图（左边亮 10，右边暗 0，中间一条竖直边缘）：

```
输入 X:                    卷积核 K（竖直边缘探测器）:
10  10  10   0              1   0  -1
10  10  10   0              1   0  -1
10  10  10   0              1   0  -1
10  10  10   0
```

3×3 窗口在 4×4 输入上有 2×2 = 4 个落点（窗口左上角可以在第 1~2 行、第 1~2 列）。算左上角落点：

```
窗口 =          逐元素乘 =         求和:
10 10 10        10  0 -10
10 10 10   →    10  0 -10    →   10+0-10 + 10+0-10 + 10+0-10 = 0
10 10 10        10  0 -10
```

窗口完全落在"全亮"区，左右抵消 → 0。窗口右移一格（骑在边缘上）：

```
窗口 =          逐元素乘 =
10  10   0      10   0   0
10  10   0  →   10   0   0   →   每行 10+0−0 = 10，三行共 30
10  10   0      10   0   0
```

整张特征图：

```
输出（特征图）:
 0  30
 0  30        ← 中间一列输出 30：边缘被"点亮"了，平坦区输出 0
```

**探测器确实找到了竖直边缘**。卷积神经网络里，这样的模板不是人设计的（像上面的 Sobel 核），而是训练学出来的——第一层学长这样探测边缘的核，更深的层学探测"眼睛""轮子"的核（§4.3 讲堆叠）。

### 2.3 输出尺寸公式与两个旋钮：padding 和 stride 🟢

窗口能落在哪些位置，决定输出多大。输入 n×n、核 k×k：

- **padding = p**：输入四周补 p 圈 0。作用有两个：保住边缘信息（不补的话边缘像素被扫到的次数少）、控制输出尺寸。
- **stride = s**：窗口每次滑 s 格（默认 1）。s=2 是"隔一格采一次"，输出尺寸减半，自带降采样。

输出边长（宽和高同理，分别算）：

```
out = ⌊(n + 2p − k) / s⌋ + 1
```

**背一个最常用的特例**：`k=3, p=1, s=1` → out = n——**尺寸不变**。现代 CNN 几乎全是这个组合（VGG 起的头），堆多少层尺寸都不变，降采样全部交给池化/stride=2 来做，尺寸账立刻变得好算。

验算两个：4×4 输入 k=3 p=0 s=1 → (4−3)/1+1 = 2 ✓（§2.2 的输出确实是 2×2）；32×32 输入 k=3 p=1 s=1 → (32+2−3)/1+1 = 32 ✓。

### 2.4 多通道：立体窗口 🟢

彩色图有 3 个通道，卷积核也得是立体的：输入 Cin 个通道时，一个卷积核的形状是 **(k, k, Cin)**——每个输入通道配一片 2D 模板，各通道分别乘加后**再全部加起来**，得到一个输出数。

```
输入 (Cin, H, W)  ×  一个核 (Cin, k, k)  →  一张特征图 (H', W')
```

要 Cout 个输出通道，就准备 Cout 个这样的立体核，各扫一遍，叠起来：

```
输入 (N, Cin, H, W)  →  conv2d(权重形状 (Cout, Cin, k, k))  →  输出 (N, Cout, H', W')
```

**参数量公式**（比全连接省多少，一算便知）：

```
参数量 = k × k × Cin × Cout + Cout      （末尾 +Cout 是每个输出通道一个偏置 b）
```

对比：3×3 核、3→64 通道 = 3·3·3·64 + 64 = **1792 个参数**——和 §1.1 的一亿五千万比，差了四个数量级。注意参数量**与图像尺寸无关**：图再大，核还是那么小（权重共享的功劳）。

直觉读法：Cin 是"看几种输入模式"，Cout 是"学几种新探测器"。第一层 Cin=3（RGB），学 64 个探测器 = 64 种基础模式（横边、竖边、红点、蓝块…）的 detector。

### 2.5 im2col：卷积其实是矩阵乘 🟡

工程实现里卷积不是真的滑窗口循环——太慢。技巧叫 **im2col**：把每个窗口的像素**展开成一列**，所有窗口的列拼成一个大矩阵；卷积核也展平成行；一次**矩阵乘法**算出整张特征图。

```
滑窗口乘加（慢）  →  窗口展开成列 + 核展平成行  →  一次 GEMM（快，吃满 GPU）
```

理解到"卷积 = 换了个姿势的矩阵乘"这层即可，**不要求手写实现**（练习里写朴素的滑窗口版，性能不管）。知道这个概念的价值：以后看底层优化文章、理解为什么 CNN 吃 GPU 带宽，都靠它。

**本节小结**：卷积 = 模板滑窗乘加，输出"匹配程度热力图"；4×4⊗竖直边缘核手算得 [[0,30],[0,30]]。输出尺寸 ⌊(n+2p−k)/s⌋+1，黄金组合 k=3,p=1,s=1 尺寸不变。多通道：核形状 (Cout, Cin, k, k)，参数量 k²·Cin·Cout+Cout，与图像尺寸无关。im2col 把卷积变矩阵乘（了解即可）。

---

## 3. 池化层 🟢

### 3.1 是什么：有损压缩 + 平移鲁棒

卷积层输出的特征图还是那么大。池化（pooling）对它做**降采样**：每 2×2 的小块压成一个数——max pooling 取最大值，average pooling 取均值。

为什么取 max 有讲究：特征图上的大数值 = "这里探测到了目标模式"。取 max 的意思是"**这个小区域里只要探测到了就行，具体在哪个像素无所谓**"——模式轻微平移一两个像素，max 的输出不变。这就是池化带来的**局部平移鲁棒性**，同时尺寸减半、计算量骤降。

### 3.2 手算一遍

输入 4×4，2×2 max pooling，stride=2（不重叠）：

```
输入:                  分块:                输出:
1  3  2  0            ┌1 3┐ ┌2 0┐          5  4
5  2  1  4            │5 2│ │1 4│
0  1  6  2        →   └─┴─┘ └─┴─┘     →    3  6
2  3  0  1            ┌0 1┐ ┌6 2┐
                      │2 3│ │0 1│
                      └─┴─┘ └─┴─┘
```

四个块各取 max：max(1,3,5,2)=5，max(2,0,1,4)=4，max(0,1,2,3)=3，max(6,2,0,1)=6。输出 2×2 = [[5,4],[3,6]]，尺寸减半。

输出尺寸公式和卷积同一个：⌊(n+2p−k)/s⌋+1。池化的惯例是 k=2, s=2, p=0（尺寸严格减半、无重叠）。

### 3.3 现代观点 🟡

池化没有可学习参数（纯规则运算），它干的"降采样"活，用 stride=2 的卷积也能干（还能顺便学东西）。所以 2014 年后的主流网络（ResNet 等）倾向于用 stride=2 卷积替代池化。但入门网络（LeNet）和直觉理解阶段，池化仍然是最干净的教具，而且 PyTorch 里写起来就一行。

**本节小结**：池化 = 分块压缩，max 版取块内最大值（"探测到了就行，位置无所谓"→ 平移鲁棒）；2×2/s2 把尺寸减半；无可学习参数。4×4 手算得 [[5,4],[3,6]]。现代网络常用 stride=2 卷积替代它。

---

## 4. 从部件到网络：LeNet 逐层推导 🟢

### 4.1 积木公式

CNN 的基本积木块是一句话：

```
Conv → ReLU → Pool，重复 N 次 → Flatten → Linear → … → Linear(类别数)
```

- **Conv**：学特征探测器（§2）；
- **ReLU**：`max(0, x)`，引入非线性（没有它，多少层卷积叠起来都等价于一层——线性函数的复合还是线性）；
- **Pool**：降尺寸、增鲁棒（§3）；
- **Flatten + Linear**：特征提取完了，最后几层全连接当分类头（模块 2 的 softmax 回归在这里等着你）。

### 4.2 LeNet-5 尺寸推导（1998 年的古董，2026 年仍是最好的教具）

LeNet 为 32×32 灰度图设计（MNIST 的 28×28 补 2 圈 0 而来）。跟着算一遍每一层的形状：

```
输入        (1, 32, 32)
Conv1: 6 个 5×5 核, p=0, s=1   → (6, 28, 28)     # (32−5)/1+1 = 28
Pool1: 2×2, s=2                 → (6, 14, 14)     # 28/2
Conv2: 16 个 5×5 核(×6 通道)    → (16, 10, 10)    # (14−5)+1 = 10
Pool2: 2×2, s=2                 → (16, 5, 5)      # 10/2
Flatten                         → 400             # 16·5·5
FC1: 400 → 120
FC2: 120 → 84
FC3: 84 → 10（类别数）
```

每一层的尺寸都能用 §2.3 的公式推出来——**以后搭任何 CNN，先在纸上把这张尺寸表列出来**，80% 的形状错误在写代码前就消灭了。

参数量也算一笔：Conv2 = 5·5·6·16 + 16 = **2416**；FC1 = 400·120 + 120 = **48120**。注意看：**卷积层参数少、全连接层参数多**——CNN 的参数量大头永远在最后的分类头。

### 4.3 深度在学什么：从边缘到零件到物体 🟡

为什么叠很多层？直观图景（Zeiler & Fergus 2014 的可视化经典结论）：

```
第 1 层：边缘、色块（§2.2 那种探测器）
第 2 层：纹理、拐角（边缘的组合）
第 3 层：零件（眼睛、轮子、花纹——纹理的组合）
第 4-5 层：物体级模式（整张猫脸、整辆车）
```

每一层都在前一层的基础上组合出更抽象的模式。这个"**层级特征**"图景是迁移学习（§6）的理论基础：浅层学的边缘/纹理探测器对**所有**图像任务都通用，所以别人在大数据集上学好的网络，前面几层可以拿来直接用。

感受野（receptive field）：第 N 层的一个输出像素，能"看到"输入的多大区域。层数越深、感受野越大——这也是为什么要深：浅层看局部，深层看全局。会算概念即可，不考递推公式。

**本节小结**：积木 = Conv→ReLU→Pool 重复 + Flatten + 全连接分类头。LeNet 尺寸链 (1,32,32)→(6,28,28)→(6,14,14)→(16,10,10)→(16,5,5)→400→120→84→10，搭网络前先在纸上推这张表。卷积层参数少、分类头参数多。深度 = 特征的层级组合（边缘→纹理→零件→物体），这是迁移学习的理论基础。

---

## 5. PyTorch 训练循环速成 🟢

（如果模块 4 已学，本节当复习；没学，这一节足够你完成本模块。）

### 5.1 世界观：NumPy 数组 + 自动求导

PyTorch 的 `Tensor` 就是一个**会记录自己怎么被算出来的 NumPy 数组**。你用 Tensor 做加减乘除，它在后台默默建一张计算图；最后对损失喊一声 `.backward()`，它沿图反向传播，把每个参数的梯度算好放进 `param.grad`——模块 1/2 你手推的梯度公式，现在全自动。

```python
import torch

x = torch.tensor([2.0], requires_grad=True)   # requires_grad=True：请帮我盯着它求导
y = 3 * x ** 2 + 1                            # y = 3x² + 1
y.backward()                                  # 反向传播：算 dy/dx
print(x.grad)                                 # tensor([12.]) —— dy/dx = 6x = 12 ✓ 手算对上
```

这就是全部魔法。剩下的都是固定套路。

### 5.2 训练循环五步（背下来，以后 90% 的训练代码都是它）

```python
for epoch in range(n_epochs):
    model.train()                          # 训练模式（Dropout/BatchNorm 等行为切换）
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)            # ① 前向：算预测
        loss = criterion(logits, y_batch)  # ② 算损失（交叉熵，模块 2 §5）
        optimizer.zero_grad()              # ③ 清零旧梯度（PyTorch 默认累加，不清零会叠加）
        loss.backward()                    # ④ 反向传播：算所有参数的梯度
        optimizer.step()                   # ⑤ 更新参数：θ ← θ − η·grad
```

三行仪式代码的对应关系：`backward()` 做的是模块 1/2 里"手推梯度公式"的活；`optimizer.step()` 就是那行 `theta = theta - lr * grad`；`zero_grad()` 是 PyTorch 的怪癖（梯度默认**累加**到 `.grad` 上，每步要先清零——忘了清零是最经典的新手 bug，损失会诡异地震荡不降）。

评估时要换一套仪式：

```python
model.eval()                               # 评估模式
with torch.no_grad():                      # 不建计算图（省内存、快）
    logits = model(X_test)
correct = (logits.argmax(dim=1) == y_test).float().mean()
```

### 5.3 三件套：nn.Module / DataLoader / optimizer

```python
import torch.nn as nn

class LeNet(nn.Module):                    # 网络 = 继承 nn.Module 的类
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(     # 卷积部分：按 §4.2 的尺寸表堆
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(),   # (1,32,32)→(6,28,28)
            nn.MaxPool2d(2, 2),                            # →(6,14,14)
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),  # →(16,10,10)
            nn.MaxPool2d(2, 2),                            # →(16,5,5)
        )
        self.classifier = nn.Sequential(   # 分类头
            nn.Flatten(),                  # (16,5,5)→400
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):                  # 前向逻辑；model(x) 时自动调用
        return self.classifier(self.features(x))

model = LeNet()
criterion = nn.CrossEntropyLoss()                          # softmax + 交叉熵二合一（模块 2 §5）
                                                         # 注意：喂 logits，不要自己先 softmax！
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam：自适应学习率的梯度下降
```

三个细节：

1. **`nn.CrossEntropyLoss` 吃 logits**：它内部自己做 softmax（数值稳定版，模块 2 §5.5 的减最大值 trick 内置）。如果你先手动 softmax 再喂它，等于做了两次 softmax，损失全错——经典坑。
2. **`Adam` 优化器**：你学的梯度下降是"全局一个学习率"；Adam 给每个参数维护自适应步长，工程上几乎总是比裸 SGD 收敛快、对 lr 不敏感。入门默认 `Adam(lr=1e-3)`，CNN  fine-tune 常用更小的 lr（§6.4）。
3. **DataLoader**：把数据集包成"自动打乱、按批切片"的迭代器：

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)   # 训练要打乱
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False) # 测试不用
```

### 5.4 device：把数据和模型都搬上 GPU（没有 GPU 也能跑）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# 训练循环里每个 batch：
X_batch, y_batch = X_batch.to(device), y_batch.to(device)
```

规则就一条：**模型和数据必须在同一个设备上**，否则报 `Expected all tensors to be on the same device`。CPU 上 CIFAR-10 迁移学习也能跑（慢，几分钟一个 epoch 量级），本项目默认配置按 CPU 友好设计。

**本节小结**：Tensor = 会记仇（计算图）的 NumPy 数组，`backward()` 自动求梯度。训练五步：前向 → 损失 → zero_grad → backward → step。`CrossEntropyLoss` 吃 logits 不许先 softmax；Adam 是默认优化器；`model.train()/eval()` + `no_grad()` 是训练/评估的两套仪式；模型和数据同一 device。

---

## 6. 迁移学习 🟢（本模块心脏之二，项目的核心）

### 6.1 为什么：你缺的不是模型，是数据

CIFAR-10 只有 5 万张训练图。从零训一个好用的 CNN 需要百万级图片（ResNet 是在 ImageNet 的 128 万张图上炼出来的）。但 §4.3 说了：CNN 的浅层学的是**通用**探测器（边缘、纹理），中层是通用零件——这些本领和"你的 10 类小图任务"无关，跟谁都能复用。

迁移学习 = **把别人在大数据上炼好的网络拿来，换掉最后一层分类头，接着训**。站在巨人肩膀上，5 万张图也能打出漂亮分数。

### 6.2 三种用法（按你的数据量从少到多）

```
数据很少（千级）：  冻结全部主干，只训新的分类头（特征提取器用法）
数据中等（万级）：  冻结浅层，微调深层 + 新分类头
数据较多：         全量微调（所有层都训，但主干用很小的学习率）
```

CIFAR-10（5 万）落在第二档——但项目里我们会**两种都跑**对比（冻结主干 vs 全量微调），亲眼看数据量怎么决定策略。

### 6.3 手术流程：以 ResNet18 为例

ResNet18 是 torchvision 自带的经典网络（18 层，残差连接——残差的思想是"让网络学修正量而不是完整映射"，解决深层网络梯度传不动的问题；本模块不手写它，会用即可）。在 ImageNet 上预训练好的权重一行下载：

```python
import torch.nn as nn
from torchvision import models

# ① 拿预训练网络（weights 首次下载约 45MB）
model = models.resnet18(weights="IMAGENET1K_V1")

# ② 换头：最后一层 fc 原本是 512→1000（ImageNet 一千类），换成 512→10（CIFAR-10）
print(model.fc)                    # Linear(in_features=512, out_features=1000)
model.fc = nn.Linear(512, 10)

# ③ 冻结主干（特征提取器用法）：所有参数的 requires_grad 关掉，再单独放开新头
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True     # 只有新头会被训练

# ④ 优化器只收"要训练的参数"——冻结的参数根本不传进去（省内存、省计算）
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

`requires_grad = False` 的直觉：反向传播时这些参数的梯度不算、不更新——它们被"速冻"在 ImageNet 学到的状态，只负责把图片翻译成 512 维特征向量，新头在这 512 维上学分类。

### 6.4 学习率分配：新头大踏步，主干小碎步 🟢

全量微调时，新分类头是随机初始化的（一无所知），主干是炼好的（身怀绝技）——用同一个学习率不公平也不安全：新头需要大步学，主干大步会"灾难性遗忘"（几脚就把 ImageNet 的好特征踹没了）。PyTorch 支持**分组学习率**：

```python
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},   # 深层主干：小碎步
    {"params": model.fc.parameters(),     "lr": 1e-3},   # 新分类头：大踏步
], lr=1e-5)                                              # 其余层（浅层）：默认 lr 最小
```

经验法则：**主干的学习率比新头小 10~100 倍**。

### 6.5 预处理必须匹配预训练模型 🟢（最容易翻车的点）

预训练网络是在**特定预处理**的数据上学的：ResNet18 的训练图都按 ImageNet 的均值/方差标准化过。你的图片进来时**必须用同一组数**，否则网络看到的"颜色分布"和它习惯的不一样，特征全乱：

```python
from torchvision import transforms

imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],      # ImageNet 三个通道的均值——照抄，不许自己算
    std=[0.229, 0.224, 0.225],       # ImageNet 三个通道的标准差
)
tf = transforms.Compose([
    transforms.Resize(224),          # ResNet18 习惯的输入尺寸（CIFAR 32×32 太小，放大）
    transforms.CenterCrop(224),
    transforms.ToTensor(),           # HWC 的 [0,255] uint8 → CHW 的 [0,1] float
    imagenet_normalize,
])
```

三个"必须"：尺寸必须匹配（224×224）、ToTensor 必须在 Normalize 之前（先归一化到 [0,1] 再做减均值除方差）、**均值方差必须用 ImageNet 的那组数**而不是你自己数据集的——因为权重是按那组数学的。项目测试会检查这件事。

### 6.6 数据增强：免费的"更多数据" 🟢

训练集只有 5 万张？每张图每次喂进来时随机变个样——随机裁剪、随机左右翻转——模型见到的就永远是"新图"，过拟合显著缓解：

```python
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),            # 随机裁一块：位置扰动
    transforms.RandomHorizontalFlip(),     # 左右翻转：方向扰动（猫照镜子还是猫）
    transforms.ToTensor(),
    imagenet_normalize,
])
```

两条铁律：

1. **只在训练集上增强**——测试/验证集用确定性的 Resize+CenterCrop，评估才稳定可比；
2. **增强必须保持语义**——猫翻转了还是猫 ✅；手写数字"6"翻转成"9" ❌（所以 MNIST 不做水平翻转）。增强策略是领域知识，不是无脑堆。

**本节小结**：迁移学习 = 拿 ImageNet 预训练网络换头续训。数据少→只训头；中等→微调深层；多→全量微调（主干 lr 小 10~100 倍）。手术四步：下载权重 → 换 `model.fc` → `requires_grad` 冻结 → 优化器只收活参数。预处理必须用 ImageNet 均值方差 + 224 尺寸。数据增强只加在训练集、必须保持语义。

---

## 7. 手写练习预告与考点清单

**手写练习**（`ml/algorithms/cnn/`，仅用 NumPy 写前向，和 PyTorch 对答案）：

1. `correlate2d` —— 单通道互相关（§2.2 的手算变成代码）；
2. `conv2d_forward` —— 多通道 + 多核 + padding + stride 的完整前向（§2.4）；
3. `max_pool2d` —— 池化前向（§3）；
4. `output_size` —— 尺寸公式（§2.3）。

**不手写**：反向传播、im2col 优化、完整训练——P1 模块的纪律（§0 优先级说明）。

**本模块考点清单**：

- [ ] 手算 4×4 输入 ⊗ 3×3 核的特征图（本文 §2.2 级别）
- [ ] 默写输出尺寸公式，并算出 k=3,p=1,s=1 时尺寸不变
- [ ] 算任意 conv 层的参数量（k²·Cin·Cout+Cout），说出为什么与图像尺寸无关
- [ ] 纸上推导 LeNet 的逐层尺寸表（§4.2）
- [ ] 默写 PyTorch 训练五步，解释 `zero_grad` 存在的理由
- [ ] 解释 `CrossEntropyLoss` 为什么不能先手动 softmax
- [ ] 迁移学习三种策略各自对应什么数据量级
- [ ] 解释为什么预处理必须用 ImageNet 均值方差、为什么主干学习率要比新头小
- [ ] 数据增强的两条铁律

---

## 完整速查表

### 公式

| 名称 | 公式 | 备注 |
|---|---|---|
| 卷积/池化输出尺寸 | `out = ⌊(n + 2p − k)/s⌋ + 1` | 宽、高分别算；k=3,p=1,s=1 → out=n |
| 卷积参数量 | `k × k × Cin × Cout + Cout` | +Cout 是偏置；与图像尺寸无关 |
| 卷积核形状 | `(Cout, Cin, k, k)` | 每个输出通道一个立体核 |
| 张量形状 | `(N, C, H, W)` | batch 永远在最前面 |

### PyTorch API

```python
# 部件
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.MaxPool2d(kernel_size, stride)     # 惯例 (2, 2)：尺寸减半
nn.Flatten()                          # (N,C,H,W) → (N, C·H·W)
nn.CrossEntropyLoss()                 # 吃 logits，别先 softmax！
torch.optim.Adam(params, lr=1e-3)

# 训练五步
model.train()
logits = model(X); loss = criterion(logits, y)
optimizer.zero_grad(); loss.backward(); optimizer.step()

# 评估仪式
model.eval()
with torch.no_grad():
    logits = model(X)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device); X = X.to(device); y = y.to(device)
```

### 迁移学习决策表

| 你的数据量 | 策略 | 冻结 | 学习率 |
|---|---|---|---|
| 千级 | 特征提取器 | 主干全冻结，只训新头 | 新头 1e-3 |
| 万级（CIFAR-10） | 微调深层 | 冻结浅层，放开深层+头 | 头 1e-3，深层 1e-4 |
| 十万级+ | 全量微调 | 不冻结 | 头 1e-3，主干 1e-5~1e-4 |

```python
# 换头 + 冻结模板
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, 10)
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# 预处理模板（必须匹配）
transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

---

## 下一步

1. 做手写练习：`ml/algorithms/cnn/`（README 里有实现顺序；写前向、和 PyTorch 对答案）；
2. 做项目：`ml/projects/cifar10_transfer/`（配套文档 `./10-CIFAR10迁移学习项目文档.md`）——LeNet 从零基线 vs ResNet18 冻结微调 vs 全量微调，亲眼感受"预训练 + 微调"的威力。
