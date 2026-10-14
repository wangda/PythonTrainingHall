# 10 - CIFAR-10 迁移学习项目文档

> **前置**：学完《09-CNN前置学习文档.md》（下称"前置文档"）+ 完成手写练习 `algorithms/cnn/`。
> **项目位置**：`ml/projects/cifar10_transfer/`
> **一句话**：同一个 CIFAR-10 数据集上，**LeNet 从零训练 vs ResNet18 冻结微调 vs ResNet18 全量微调**三条路线对比，亲手感受"预训练模型 + 微调"的威力，并把 PyTorch 训练循环、迁移学习手术、数据增强一次练全。

---

## 1. 这个项目练什么

| 技能 | 对应前置文档 | 在项目里的位置 |
|---|---|---|
| PyTorch 训练循环五步 | §5 | `train.py`（已实现的参考版，读熟） |
| 预处理匹配预训练模型 | §6.5 | `features.py`（**你的核心任务**） |
| 数据增强（只增强训练集） | §6.6 | `features.py` |
| 迁移学习手术（换头、冻结、分组 lr） | §6.3/§6.4 | `model.py`（**你的核心任务**） |
| 三路线对比实验 | §6.2 | 改 config 跑三次 + 填对比表 |
| 图像模型评估 | 模块 2 §7 | `evaluate.py`（已实现：accuracy + 混淆矩阵） |

和前三个项目最大的区别：**模型产物从 `joblib` 变成 `torch.save` 的 `.pt`**（PyTorch 模型的标准序列化方式），预处理状态从"中位数 dict"变成"transforms 配置"。

## 2. 数据集

CIFAR-10：32×32 彩色小图，10 类（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车），训练集 50000 张、测试集 10000 张，各类完全均衡。

`data/download.py` 已实现：`torchvision.datasets.CIFAR10` 一行下载（约 170MB，首次需联网），缓存到 `data/raw/`。

注意一个设计决策：**`download.py` 返回的是 PIL 图片数据集，不做任何张量化**——预处理（resize、增强、Normalize）全部在 `features.py` 的 transforms 里声明，由 DataLoader 在每次取数据时执行。这是 torchvision 的标准范式：增强的随机性因此**每个 epoch 都不同**（同一张图，这次裁左边、下次裁右边），这正是数据增强想要的（前置文档 §6.6）。

## 3. 核心工程课：两套 transforms 的纪律

本项目唯一的"新工程坑"：**训练集和验证/测试集必须用不同的 transforms**：

```
训练集：  Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ToTensor → Normalize   ← 有随机性
验证/测试：Resize(224) → CenterCrop(224) → ToTensor → Normalize                            ← 全确定性
```

- 随机增强只许出现在训练集——测试集加随机性，每次评估分数都不一样，实验没法对比；
- 两边的 ToTensor + Normalize（ImageNet 均值方差）必须**完全一致**——这是前置文档 §6.5 的"预处理匹配"，模块 2 项目里"预测必须复现预处理"的 CNN 版。

`features.py` 因此设计成一个函数两个分支：

```python
train_tf = build_transforms(config, train=True)
eval_tf  = build_transforms(config, train=False)
```

另一个纪律：**从训练集里切验证集**（train/val/test 三份）。测试集（CIFAR-10 官方那 1 万张）只在最后一次评估用；调参、选策略看验证集。切分用 `torch.utils.data.random_split` + 固定种子生成器。注意一个细节：random_split 出来的 train/val 共享同一个底层 dataset 对象，但各自要挂**不同的 transform**——用 `torch.utils.data.Dataset` 的包装子类（`features.py` 里给了 `TransformedSubset` 的实现提示）。

## 4. 三条实验路线（任务清单的骨架）

全部通过 `config.yaml` 的 `model.name` + `model.finetune` 切换，不改代码：

| 路线 | config | 学什么 | 预期（CPU 可接受量级） |
|---|---|---|---|
| A. LeNet 从零 | `name: lenet` | 亲手搭的经典 CNN 基线；输入保持 32×32 不放大 | 训练几个 epoch，test acc 60% 上下 |
| B. ResNet18 冻结微调 | `name: resnet18, finetune: head_only` | 主干冻成特征提取器，只训新头（§6.2 第一档） | 1~2 个 epoch 就冲 90%+ |
| C. ResNet18 全量微调 | `name: resnet18, finetune: full` | 分组学习率全量微调（§6.4） | 上限最高，也最吃算力 |

**A vs B 是本项目的灵魂对比**：LeNet 从零吭哧吭哧学到 60%，冻结的 ResNet18 第一分钟就 90%——"站在巨人肩膀上"从成语变成数字。B vs C 看数据量与微调深度的关系（§6.2）。

CPU 友好设计：默认 `epochs: 2`、`batch_size: 64`、B/C 路线图片放大到 224 是主要算力开销；有 GPU 就放开 epochs。训练日志每 100 个 batch 打一次 loss，eval 每个 epoch 末尾在验证集上算 accuracy——**训练曲线、验证分数全部打印并保存**，养成可视化习惯。

## 5. 模型手术：`model.py` 你要实现什么

`create_model(config)` 工厂的两个分支：

1. **`lenet`**：按前置文档 §4.2 的尺寸表用 `nn.Sequential` 堆出来（CIFAR 是 3 通道彩色图，Conv1 的 in_channels 从 1 改成 3，fc 输入维度自己用尺寸公式推——提示：32×32 输入走一遍 §4.2 的流程，Flatten 前是 (16, 5, 5)？还是别的？算一遍）。
2. **`resnet18`**：`models.resnet18(weights=...)` 下载预训练权重 → `model.fc = nn.Linear(512, 10)` 换头 → 按 `finetune` 冻结：
   - `head_only`：冻结全部，放开 `model.fc`；
   - `full`：全放开，但 `create_optimizer` 里分组给 lr（头 1e-3、layer4 1e-4、其余 1e-5）。
3. **`create_optimizer(model, config)`**：只收 `requires_grad=True` 的参数（§6.3 第④步）。

骨架里给了 `count_trainable_params(model)` 工具函数——每换一种冻结策略，打印可训练参数量看一眼：B 路线应该只有 fc 头的 5130 个（512×10+10），C 路线是 1100 万+。**这个数字比任何文字都更能让你理解"冻结"意味着什么。**

## 6. 任务清单

1. **跑通数据**：`python data/download.py` 下载缓存；`pytest tests/ -v` 看任务清单；
2. **实现 `features.py`**：`build_transforms`（两套）+ `make_loaders`（切验证集、挂不同 transform、DataLoader）→ 测试变绿（测试会检查：训练 transform 有随机性、评估 transform 是确定性的、Normalize 用的是 ImageNet 那组数、两个 loader 的 transform 不一样）；
3. **实现 `model.py`**：`create_model` 两个分支 + `create_optimizer`；
4. **路线 B**：`train.py` 跑冻结微调 → `evaluate.py` 出测试集分数 + 混淆矩阵；
5. **路线 A**：config 切 `lenet` 重训，对比 B——写一句观察；
6. **路线 C**（有 GPU 或耐心时）：全量微调，对比 B/C；
7. **推理 demo**：`predict.py --input my_cat.jpg` 预测一张网上找的图；
8. **填对比表**（README 里有模板）：三条路线的 best val acc / test acc / 训练时长 / 可训练参数量；
9. （进阶）混淆矩阵里找 CIFAR-10 最经典的一对难兄难弟（猫 vs 狗），去测试集里翻几张被错分的图看看——`evaluate.py` 会帮你把它们存下来。

## 7. 常见坑

- **忘了 `model.fc = nn.Linear(512, 10)` 在冻结之后要重新放开**——先冻结全部再换头，新头是冻着的，训了个寂寞（正确顺序：换头 → 冻结主干 → 放开 fc；或者冻结时跳过 fc）；
- **`CrossEntropyLoss` 前手动 softmax**（前置文档 §5.3 的坑，这里会再犯一次）；
- **Normalize 用了 CIFAR 自己的均值方差**——预训练模型只认 ImageNet 那组数（§6.5）；
- **评估时忘了 `model.eval()` + `no_grad()`**——结果能算出来但内存泄漏、且若网络含 Dropout/BN 行为会错；
- **DataLoader 的 `num_workers` 在 Windows 上设大会卡死**——本项目默认 0，Linux/Mac 上可调大加速；
- **`random_split` 不固定种子**——每次运行验证集都不一样，分数无法对比；用 `torch.Generator().manual_seed(...)`；
- LeNet 路线别放大到 224（白烧算力）——`image_size` 在 config 里按路线分开配；
- 训练中途想看形状，打印一句 `print(x.shape)` 比盯着报错猜快十倍——前置文档 §4.2 的尺寸表就是干这个的。

## 8. 与前置项目的结构差异速览

| | titanic（模块 3） | cifar10_transfer（本项目） |
|---|---|---|
| 数据 | 表格 csv | 图片数据集（torchvision） |
| 预处理 | 有状态统计量（中位数 dict） | 声明式 transforms（增强 + Normalize） |
| 模型 | sklearn/LightGBM `fit` | PyTorch 训练循环（`train.py` 参考实现） |
| 产物 | `model.joblib` | `model.pt`（`torch.save` 权重 + 配置快照） |
| 训练控制 | 早停 / CV | epoch 循环 + 验证集 |
