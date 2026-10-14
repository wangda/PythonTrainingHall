# 08 - Fashion-MNIST 分类项目文档（模块 4 配套项目）

> **面向读者**：刚学完阶段二模块 4（MLP 与反向传播）的你。
> **前置阅读**：《07-MLP与反向传播前置学习文档.md》（下称"前置文档"）——线性模型天花板（§1）、神经元与激活函数（§2/§3）、前向传播与初始化（§4）、反向传播（§5）、梯度检查（§6）、PyTorch 重写 MLP（§7）、工程化 Trainer（§8）都在那里讲透了，本文只引用不重复。
> **同时推荐阅读**：《04-手写数字分类项目文档.md》——工程骨架的通用部分（七文件分工、config 分离、模型包思想）那里讲过，本文只讲差异。
> **项目位置**：`ml/projects/fashion_mnist/`（代码骨架已生成，核心组件留给你实现）。
> **本文数据**：文中所有数字都在当前环境实际运行验证过（下载走了点弯路，见 §2.3 的如实记录）。

---

## 本文怎么用

模块 2 的手写数字项目解决的是"**图像 + 多分类 + 概率输出**"；本项目的定位是**框架升级 + 训练工程化**——同类型的任务（28×28 灰度图十分类），但工具从 sklearn 换成 PyTorch，训练从"一行 `fit`"换成"自己掌控的训练循环"。建议的顺序：

1. 通读本文 §1~§4，搞清楚与模块 2 的差异点；
2. 按 §5 的任务清单动手填骨架（Trainer 是主战场）；
3. 写完先跑测试（§8），绿了再训练；
4. 踩坑了去 §7 对号入座。

---

## 1. 项目目标与产出

### 1.1 为什么要做这个项目

模块 2 用 `LogisticRegression` 一行 `fit` 就训完了——省心，但你不知道里面发生了什么，更控制不了它。前置文档 §1 已经论证了线性模型的天花板，§7 你手写过了 PyTorch 训练循环。这个项目把练习升级为工程，三个新维度：

1. **从 sklearn 到 PyTorch（工程框架切换）**：模型是 `nn.Module`，数据走 `DataLoader`，训练循环五步（清零梯度 → 前向 → 算损失 → 反向 → 更新）全在你手里。手写练习里的 NumPy 反向传播被 autograd 接管——你推导过梯度怎么算（前置文档 §5），现在让机器替你算，你的注意力转移到"训练过程怎么管"；
2. **从"训完就完"到 Trainer 封装（本项目的灵魂）**：checkpoint 存验证集最优而非最后一个 epoch、早停在 patience 耗尽时收工、学习率按调度器衰减——这三件事 sklearn 的 `fit` 替你做不了，而它们是真实训练不出事的底线（前置文档 §8）；
3. **从手写反向传播（练习）到 autograd（工程）**：前置文档 §5 你手推过链式法则、§6 做过梯度检查；本项目 `loss.backward()` 一行替代全部手工推导。理解原理之后用工具，和直接调包，是两回事——前者你知道 backward 里发生了什么。

### 1.2 产出物

做完这个项目，你将拥有：

1. 一个 Fashion-MNIST 训练流水线：`config.yaml` 控制全部旋钮，`train.py` 一条命令跑完；
2. 一个你自己实现的 `Trainer` 类：checkpoint / 早停 / lr 调度三件套，以后每个 PyTorch 项目都能复用；
3. 一张训练曲线图（train/val loss 双线）+ 一张混淆矩阵热力图 + 测试集四项指标；
4. 一个能对真实图片推理的命令行工具：`python predict.py --input my_image.png` 输出 top-3 类别与概率。

---

## 2. 数据集：Fashion-MNIST

### 2.1 与 MNIST 对比：同尺寸同规模，为什么它是更好的基准

| 属性 | MNIST | Fashion-MNIST |
|---|---|---|
| 样本数 | 60000 训练 + 10000 测试 | 完全相同（刻意设计） |
| 分辨率 | 28×28 灰度，784 像素 | 完全相同 |
| 类别数 | 10（数字 0~9） | 10（服饰类别） |
| 像素范围 | 0~255 | 0~255 |
| LogisticRegression 实测 | ≈ 0.92（模块 2 实测 0.9214） | ≈ 0.84 |
| MLP（本项目）实测 | ≈ 0.98 | 0.8878 |
| 获取方式 | sklearn fetch_openml / IDX 镜像 | torchvision.datasets.FashionMNIST / IDX 镜像 |

Fashion-MNIST（Zalando，2017）的设计意图就是**接替 MNIST 当基准数据集**：MNIST 太简单了——模块 2 你亲眼见过，一个纯线性模型（逻辑回归）不费力就到 0.92，它区分不出"模型真的进步了"和"数据太容易"。Fashion-MNIST 把数字换成服饰，同样的 28×28 灰度格子里装的是衬衫、球鞋、包——轮廓更复杂、类间相似度更高（Shirt 和 T-shirt 的差别比 4 和 9 微妙得多），逻辑回归立刻掉到 0.84 左右，模型优劣重新变得可分辨。API、文件格式（IDX）、规模全部与 MNIST 对齐，迁移成本为零。

### 2.2 十个类别长什么样

类别标签 0~9 对应：**T-shirt/top（T恤）、Trouser（裤子）、Pullover（套头衫）、Dress（连衣裙）、Coat（大衣）、Sandal（凉鞋）、Shirt（衬衫）、Sneaker（运动鞋）、Bag（包）、Ankle boot（短靴）**。

图像数据第一步永远是用眼睛看（下面代码实际跑过，输出存为 `reports/figures/fashion_mnist_samples.png`）：

```python
import matplotlib.pyplot as plt
from data.download import load_raw_data, CLASS_NAMES

x_train, y_train, _, _, _ = load_raw_data(config)
fig, axes = plt.subplots(2, 5, figsize=(9, 4))
for cls in range(10):
    ax = axes.flat[cls]
    ax.imshow(x_train[y_train == cls][0], cmap="gray")   # 每类取第一张
    ax.set_title(CLASS_NAMES[cls], fontsize=9)
    ax.axis("off")
```

![Fashion-MNIST 十类样例](../../projects/fashion_mnist/reports/figures/fashion_mnist_samples.png)

先记住两个直观感受：

1. **黑底白物**：和 MNIST 一样是背景 0、主体亮——`predict.py --invert` 存在的原因（§7.7）；
2. **上半身衣物彼此相似**：T-shirt、Pullover、Dress、Coat、Shirt 摆在一起，轮廓都是"一块布+两个袖子"——这预告了混淆矩阵的重灾区（§5 任务 4 实测验证）。

类别分布是教科书级均衡：训练集每类恰好 6000 张，测试集每类恰好 1000 张——accuracy 是诚实指标（模块 2 文档 §2.4 讨论过什么时候它会骗人）。

### 2.3 下载踩坑实录（如实记录）

`data/download.py` 是双通道设计，本环境实测情况如下：

- **通道 ① torchvision**：`datasets.FashionMNIST(download=True)` 能连上官方 S3 源（`fashion-mnist.s3-website.eu-central-1.amazonaws.com`），但实测下载速度只有 15~25 kB/s，26.4MB 的训练图像要拖约 25 分钟——能用，但慢得离谱；
- **通道 ② IDX 镜像**（本环境实际走通的路径）：直接拉 zalandoresearch/fashion-mnist 官方仓库的 GitHub 直链，按"raw 直连 → gh-proxy 代理 → S3 镜像"顺序逐个尝试。本环境实测：raw 直连时通时断（四个文件里前两个超时失败、后两个直连成功），**gh-proxy 代理以 ~200 kB/s 兜底跑通**，全部文件几分钟内下完，手工解析 IDX（魔数校验 + 大端 uint32 头 + uint8 像素体）后写入缓存。

两条通道的产物统一缓存为 `data/raw/fashion_mnist.npz`（np.savez_compressed，uint8 原始像素，约 25MB）：**数据来源可以换，接口不用动**——这和模块 2 的缓存设计是同一个思想。缓存一次，之后 `load_raw_data` 直接命中，训练实验零网络开销。

---

## 3. 工程化：与模块 2 项目的四个差异点

七文件骨架的职责划分、config 分离的理由、测试先行——模块 1/2 的项目文档已讲透，不再重复。这里只讲换到 PyTorch 后的差异。

### 3.1 差异一：三集合划分（train / val / test）

模块 2 只有训练集和测试集，因为 sklearn 的 `fit` 内部没有"训练中看表现"的概念。PyTorch 训练是按 epoch 迭代的，每个 epoch 结束你都有机会做决策：要不要停？存不存权重？这些决策的依据必须是**验证集**——从训练集里再切一块（本配置 10%，即 54000/6000/10000 三集合）。

铁律：**验证集只能从训练集切，测试集一眼都不许看**。从测试集切验证 = 训练决策偷看考题 = 评估结果虚假偏高且永远无法发现（前置文档 §8；本文 §7.2 有踩坑详解）。`tests/test_features.py` 的 `test_划分_测试集与训练验证零重叠` 专查这个。

### 3.2 差异二：Trainer 类封装训练循环

sklearn 把训练封装成 `model.fit(X, y)`，代价是你失去了对过程的控制。本项目的 `Trainer` 把前置文档 §7 的五步循环 + §8 的三个工程组件装在一起：

```
fit() 每个 epoch：
  _train_one_epoch()   五步法（前置文档 §7）
  _validate()          验证集评估（no_grad，不更新参数）
  ├── 验证 loss 创新低 → save_checkpoint()     存 models/best.pt
  ├── 否则 epochs_no_improve += 1
  ├── scheduler.step()                          lr 调度
  └── check_early_stopping() → break           早停
```

`__init__` 和 `_train_one_epoch`/`_validate` 已实现（机械动作），**`fit` / `save_checkpoint` / `check_early_stopping` / `_make_scheduler` 四个方法是你的任务**——测试用"剧本验证 loss"（继承 Trainer 覆写 `_validate` 喂一串预定 loss）秒级验证组件行为，不用等真实训练。

### 3.3 差异三：模型存 .pt 而不是 joblib

sklearn 模型用 joblib 整个 pickle；PyTorch 的惯例是**只存权重**（state_dict，纯 tensor 数据）：

```python
torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
            "val_loss": val_loss, "val_acc": val_acc, "config": config}, "models/best.pt")
# 加载时：先用 config 重建结构，再装入权重
model = create_model(ckpt["config"]); model.load_state_dict(ckpt["model_state_dict"])
```

为什么不一整个 pickle 模型对象：pickle 依赖类的定义位置，代码一重构（文件改名、类挪位置）就加载失败。state_dict 是纯数据，配合工厂函数重建结构，走到哪都能活。config 一起存的原因和模块 1/2 的"模型包"思想一致：**预测时要靠它复现预处理和模型结构**。

### 3.4 差异四：history.json 记录训练过程

sklearn 的 `fit` 是个黑箱，训完只剩最终模型。神经网络训练的**过程信息**（每个 epoch 的 train/val loss）是诊断过拟合的第一手证据，所以 `train.py` 把 `Trainer.fit()` 返回的 history 存成 `models/history.json`，`evaluate.py` 拿它画训练曲线。看曲线比看最终数字重要得多——§5 任务 3 的消融实验全靠它说话。

---

## 4. Pipeline 五步落点

| 步骤 | 干什么 | 本项目落点 |
|---|---|---|
| ① 加载数据 | 下载并缓存 Fashion-MNIST | `data/download.py` 的 `load_raw_data(config)`（双通道统一接口） |
| ② EDA | 看十类样例、确认类别均衡 | §2 已带你做过，结论（黑底白物、0~255）固化进了代码 |
| ③ 特征工程 | 切验证集、归一化、包 DataLoader | `features.py` 的 `build_dataloaders`（三集合！） |
| ④ 训练 | 训练循环 + checkpoint/早停/调度 | `model.py` + `trainer.py`（**你的主战场**）+ `train.py` |
| ⑤ 评估 | 测试集指标 + 混淆矩阵 + 训练曲线 | `evaluate.py` |

第 ⑥ 步推理服务化依然是 `predict.py`——输入一张 28×28 PNG，输出 top-3 类别与概率。

---

## 5. 本项目要做什么（任务清单）

按顺序做五件事，每件都有可衡量的验收标准。以下数字全部在当前环境实测（GPU 训练约 20 秒，CPU 几分钟内也能跑完）。

### 任务 1：默认 config 跑通 baseline

实现 `features.py` 的 `build_dataloaders`，跑通 `train.py`（此时 Trainer 四个方法也先写出最小可用版）。

**参考基线（已在当前环境实测，config 默认参数：hidden [128,64]、lr 1e-3、batch 128、epochs 30、patience 5、StepLR(step_size=10, gamma=0.5)）**：

```
三集合: train 54000 / val 6000 / test 10000（参数量 109,386）
早停在第 25 epoch 触发，验证集最优在第 20 epoch（val_loss=0.2801）
测试集 accuracy: 0.8878   macro F1: 0.8873
```

你的实现应复现这个量级。**本任务的可衡量目标：测试集 accuracy ≥ 0.87。**

### 任务 2：实现 Trainer 三组件，画训练曲线

把 `fit` / `save_checkpoint` / `check_early_stopping` / `_make_scheduler` 四个方法写完，先跑 `tests/test_trainer.py` 的 6 条测试（剧本验证 loss，秒级出结果），绿了再真训。然后用 `evaluate.py` 画出 `reports/figures/training_curves.png`：

![训练曲线](../../projects/fashion_mnist/reports/figures/training_curves.png)

读图要点（实测）：train_loss 一路单调下降（0.63 → 0.17）；val_loss 在第 20 epoch 触底（0.2801）后横盘微升——**红线（val_loss 最低点）之后 train_loss 还在降、val_loss 不降，这就是过拟合开始的分界线**。checkpoint 存的正是第 20 epoch 的权重，早停在第 25 epoch 收工（patience=5 跑满）。epoch 11 处 lr 从 1e-3 衰减到 5e-4（StepLR），train_loss 在 epoch 10→11 有一跳（0.268 → 0.244）——调度器生效的直接证据。

### 任务 3：消融实验

复制 config 改两处，各跑一次，和 baseline 对比：

**① 关掉早停跑 50 epoch**（`early_stopping.enabled: false`，`epochs: 50`）。实测：

```
train_loss: 0.28（epoch 20）→ 0.126（epoch 50）  一路降到底
val_loss:   0.2801（epoch 20，最低）→ 0.2991（epoch 50）  触底回升 +6.8%
```

![关早停 50 epoch](../../projects/fashion_mnist/reports/figures/ablation_no_early_stopping.png)

两条曲线张口越来越大——这就是教科书上的过拟合图。注意一个细节：**即使关掉早停，best.pt 里存的仍是第 20 epoch 的最优权重**——checkpoint 和早停是两个独立的保险，一个管"留最好的"，一个管"别浪费时间"。

**② 换 lr 调度器**（`lr_scheduler.type`）。实测三种设置的最优 val_loss：

| 调度器 | 最优 val_loss | 出现在 epoch |
|---|---|---|
| step（默认，10/0.5） | 0.2801 | 20 |
| cosine（t_max=30） | 0.2798 | 18 |
| none（恒定 1e-3） | 0.2925 | 18 |

step 和 cosine 基本打平，但都明显好过不调——恒定学习率后期在最优点附近来回跳，降不下去。结论：**lr 调度的收益是实的，具体选哪种倒不那么敏感**。

### 任务 4：测试集评估 + 混淆矩阵分析

实现 `evaluate.py` 的三个函数，生成 `reports/figures/confusion_matrix.png`。测试集 10000 张共误判 1122 张，最大的几对（实测）：

| 真实 → 预测 | 次数 | 真实 → 预测 | 次数 |
|---|---|---|---|
| Shirt → T-shirt/top | 133 | Shirt → Coat | 63 |
| T-shirt/top → Shirt | 101 | Coat → Shirt | 60 |
| Pullover → Coat | 87 | Pullover → Shirt | 58 |
| Coat → Pullover | 86 | Shirt → Pullover | 74 |

**误判全部集中在上半身衣物**——裤子、鞋、包几乎不错（Trouser 的 F1 高达 0.979），因为它们轮廓独特；Shirt 是全场最差（recall 0.688），被散抢到 T-shirt/Pullover/Coat 三个类上。看图说话——`reports/figures/misclassified_shirt.png`（实测生成）：

![Shirt 误判分析](../../projects/fashion_mnist/reports/figures/misclassified_shirt.png)

中间一行是被误判为 T-shirt 的 Shirt：清一色**短袖衬衫**——和第三行的真 T-shirt 摆在一起，人也得多看两眼（区别只剩领口和扣子的细节，28×28 分辨率下糊成一片）。这回答了"为什么准确率卡在 0.89"：**784 个展平像素 + MLP 的信息提取能力到头了**，类间差异本身就需要空间结构（领型、肩线）才能捕捉——这是模块 5 CNN 要解决的问题。

### 任务 5：predict.py 推理演示

拿真实图片走一遍推理。本环境实测：从测试集取一张真实标签为 Ankle boot 的图存成 PNG：

```bash
# ① 黑底白物（与训练数据同分布）：正确
../../../.venv/bin/python predict.py --input boot.png
# 样本 0: 预测 = Ankle boot（置信度 99.29%）
#     Ankle boot: 99.29%    Sneaker: 0.69%    Sandal: 0.02%

# ② 同一张图反相成白底黑物，不加 --invert：翻车且自信
../../../.venv/bin/python predict.py --input boot_inverted.png
# 样本 0: 预测 = Bag（置信度 100.00%）   ← 错得理直气壮

# ③ 加上 --invert：恢复正确
../../../.venv/bin/python predict.py --input boot_inverted.png --invert
# 样本 0: 预测 = Ankle boot（置信度 99.29%）
```

注意 ② 的翻车方式和模块 2 不同：不是低置信度乱猜，而是 **100% 置信度地错**——反相后的图在训练分布里根本没有对应物，softmax 照样给出极端概率。这再次印证《03-逻辑回归与分类前置学习文档.md》§6 那句话：概率不是结论，是依据；分布外的输入，概率再高也不可信。

---

## 6. 特征工程提示：图像任务的"特征工程"基本就是归一化

模块 1 的特征工程是造新特征；本项目的"特征工程"朴素到底——**784 个像素就是全部特征**，工程上只有两件事：

1. **/255 归一化**（默认开）：把像素压到 [0,1]。像素 0~255 的量纲会让梯度在不同特征上失衡（前置文档 §4），压到 [0,1] 后 Adam 收敛明显更稳。它是无状态变换，不存在数据泄漏（模块 2 文档 §3.1 论证过）；
2. **可选标准化**（`features.standardize: true`）：(x-均值)/标准差，让输入近似零均值。注意统计量**只能用切分后的训练子集计算**，再套到验证/测试集上——否则就是把 val/test 的分布信息泄漏进预处理（和模块 1 的 StandardScaler 同一个坑）。

可以讨论但不实现的方向：

- **数据增强**：随机水平翻转（服饰左右翻转不改变类别）、小幅度平移，凭空扩增训练集，通常能再抬 1~2 个百分点——`torchvision.transforms` 几行就能加进 DataLoader 管道；
- **把 MLP 换成 CNN**：展平像素丢弃了空间结构（"哪些像素相邻"），这是 MLP 在图像上的结构性天花板。本项目 0.87~0.89 的成绩，CNN 架构轻松上 0.92+——那是模块 5 的事，到时回看 `build_dataloaders` 里的 `reshape(n, -1)`，你会明白那就是天花板所在。

---

## 7. 常见坑（每个都值一次报错）

### 7.1 末层接 softmax + CrossEntropyLoss = 双重 softmax

`nn.CrossEntropyLoss` 内部已经包含 `log_softmax`（前置文档 §5）。模型末层再接一个 Softmax，相当于把概率又压了一遍——梯度被压扁，训练明显变差。实测对照（玩具数据各训 50 步）：裸 logits 的 loss 降到 **0.0**，双重 softmax 的 loss 卡在 **1.46** 降不动。

```python
# ❌ 错误：nn.Sequential(nn.Linear(64, 10), nn.Softmax(dim=1))
# ✅ 正确：末层就是 nn.Linear(64, 10)，输出裸 logits
#          需要概率时（predict.py 展示置信度）才显式 torch.softmax 一次
```

`tests/test_model.py` 的 `test_末层是Linear不接softmax` 专查这个。

### 7.2 验证集从测试集切 = 数据泄漏

Fashion-MNIST 官方给的 10000 张测试集，训练全程不许参与任何决策。如果图省事 `train_test_split(全部70000张)` 再切三份，验证集里就混着测试集的信息——早停、checkpoint 全被污染，报出去的 accuracy 虚假偏高，而且**没有任何测试能事后发现**。正确做法：`build_dataloaders` 只对 60000 张训练集做切分。`test_划分_测试集与训练验证零重叠` 专查这个。

### 7.3 checkpoint 存了最后一个 epoch 而不是最优

baseline 实测：验证集最优在第 **20** epoch（val_loss 0.2801），第 30 epoch 时 val_loss 已回升到 0.29 附近。如果 `fit` 结束后无脑存最后一个 epoch，存的就是一个轻微过拟合的权重——验证集白看了。正确姿势：**只在 val_loss 创历史新低时存**。测试 `test_checkpoint_存的是验证最优而非最后` 用"先降后升"的剧本 loss 专查这个。

### 7.4 DataLoader 的 shuffle 只在训练集开

训练集 shuffle 是 SGD 的基本要求（每个 epoch 的 batch 顺序不同，梯度估计不带周期性偏差）；验证/测试集 shuffle 毫无意义——评估不反向传播，顺序不影响结果，乱序反而让你没法对齐"第几张图预测成了什么"（画误判拼图时就是灾难）。`test_DataLoader_只有训练集shuffle` 用 sampler 类型断言专查这个。

### 7.5 torch 随机种子不止一个

只写 `torch.manual_seed(42)` 不够：python 的 `random`（某些库内部用）、numpy（切分数据用）、cuda（GPU 上的算子）各吃各的种子。`train.py` 的 `set_seed` 一次固定全部四个——这也是 baseline 数字能逐位复现的前提。

### 7.6 device 不一致：模型在 cuda、数据在 cpu

`config` 里 `device: auto` 时本机会选 cuda，如果你 forward 时忘了把 batch 也 `.to(device)`，报错长这样（实测）：

```
RuntimeError: Expected all tensors to be on the same device, but got
mat1 is on cpu, different from other tensors on cuda:0
```

规矩只有一条：**模型和每个 batch 都要搬到同一个 device**。`_train_one_epoch` 里的 `X.to(self.device), y.to(self.device)` 每次循环都要做，别在循环外做一次就以为万事大吉（DataLoader 每次产出的都是新 tensor）。

### 7.7 手绘/自拍图忘记 --invert

和模块 2 同一个坑，但翻车方式升级了：训练数据是黑底白物，白底图不反相直接喂进去，模型会**高置信度地错**（实测：反相的短靴被认成 Bag，置信度 100%，见 §5 任务 5）。判断方法不变：看图时背景是白的，就加 `--invert`。

---

## 8. 如何运行

所有命令都在项目根目录执行：

```bash
cd ml/projects/fashion_mnist

# ① 下载并缓存数据（已实现；torchvision 失败自动回退 IDX 镜像，见 §2.3）
../../../.venv/bin/python data/download.py

# ② 跑测试——骨架状态下 features/trainer 相关 14 条失败（NotImplementedError 就是你的任务清单）
../../../.venv/bin/python -m pytest tests/ -v

# ③ 实现 features.py + trainer.py 后，测试应变绿

# ④ 训练（GPU/CPU 自适应；最优权重存 models/best.pt，history 存 models/history.json）
../../../.venv/bin/python train.py

# ⑤ 评估（实现 evaluate.py 后）：测试集指标 + classification_report + 混淆矩阵 + 训练曲线
../../../.venv/bin/python evaluate.py

# ⑥ 命令行预测：28×28 PNG；白底黑物图加 --invert
../../../.venv/bin/python predict.py --input my_image.png --invert
```

**实验的正确姿势**与模块 1/2 相同：改 `config.yaml` → 跑 `train.py` → 跑 `evaluate.py` → 记录指标。消融实验（§5 任务 3）建议复制几份 config（如 `config_no_earlystop.yaml`），用 `train.py --config xxx.yaml` 指定。

---

## 与已学知识的对照表

| 本项目环节 | 知识出处 |
|---|---|
| 为什么要 MLP（线性模型天花板） | 前置文档 §1 |
| 神经元、ReLU 激活 | 前置文档 §2 / §3 |
| 前向传播、初始化 | 前置文档 §4 |
| 反向传播、梯度检查（手写练习的算法内功） | 前置文档 §5 / §6 |
| autograd、nn.Module、DataLoader、训练循环五步法 | 前置文档 §7 |
| checkpoint / 早停 / lr 调度 / 三集合 | 前置文档 §8 |
| accuracy / macro P/R/F1、混淆矩阵 | 《03-逻辑回归与分类前置学习文档.md》§7 |
| 七文件工程骨架、config 分离、模型包思想 | 《02-房价预测项目文档.md》§3、《04-手写数字分类项目文档.md》§3 |
| 图像归一化、--invert 的坑 | 《04-手写数字分类项目文档.md》§3/§7 |
| imshow、子图、热力图 | `../阶段一/02-Matplotlib前置学习文档.md` |

---

## 完成后检查单

- [ ] `pytest tests/ -v` 全绿（19 条；骨架态下 14 条失败、全是 NotImplementedError）
- [ ] baseline 复现：早停第 25 epoch 触发、最优第 20 epoch（val_loss 0.2801），测试集 accuracy = 0.8878（**可衡量目标：≥ 0.87**）
- [ ] 训练曲线已生成，能指出 val_loss 拐点（epoch 20）并解释"拐点之后发生了什么"
- [ ] 消融实验完成：关早停 50 epoch 观察到 val_loss 回升（0.2801 → 0.2991）；换 lr 调度器对比过（step 0.2801 / cosine 0.2798 / none 0.2925）
- [ ] 混淆矩阵已生成，能说出最容易混淆的类别对（Shirt → T-shirt/top，133 次）并看图解释原因（短袖衬衫 vs T恤，28×28 下轮廓无差别）
- [ ] 能解释 Shirt 的 recall（0.688）为什么全场最低，以及为什么这不怪模型
- [ ] `predict.py` 对一张图片推理成功，且亲眼见过白底图不加 `--invert` 时 100% 置信度翻车的现场
- [ ] 能说清三个组件各自防什么：checkpoint 防"存了过拟合的权重"、早停防"白训练还更差"、lr 调度防"后期在最优点附近来回跳"
- [ ] config.yaml 里的每个旋钮（val_size / hidden_sizes / lr / batch_size / patience / lr_scheduler）都能说出作用

全部勾上，模块 4 毕业——你在模块 1/2 的工程骨架上，完成了从 sklearn 到 PyTorch 的切换，并亲手实现了真实训练不出事的三件套。模块 5 的 CNN 会回答本项目留下的悬念：怎么突破展平像素 + MLP 的 0.89 天花板。
