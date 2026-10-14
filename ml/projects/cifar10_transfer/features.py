"""特征工程（骨架）：PIL 图片 → DataLoader，配套文档《10-CIFAR10迁移学习项目文档.md》§3。

你要实现的两个函数：
    build_transforms  声明式预处理：训练版（带随机增强）与评估版（全确定性）
    make_loaders      切验证集（random_split 固定种子）→ 挂不同 transform → DataLoader

教学要点（项目文档 §3，前置文档 §6.5/§6.6）：
    - 训练集和验证/测试集必须用**不同**的 transforms：训练有随机增强，评估全确定性；
    - Normalize 的均值/方差必须用 ImageNet 那组数（预训练权重的"语言"），
      不许用 CIFAR 自己算的——这是迁移学习最容易翻车的点；
    - ToTensor 必须在 Normalize 之前（先把 [0,255] uint8 转成 [0,1] float）。
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# ImageNet 标准化常数（前置文档 §6.5：照抄，不许换成你自己数据集的统计量）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformedSubset(Dataset):
    """给 random_split 切出的子集单独挂 transform 的包装器（已完整实现）。

    为什么需要它：random_split 出来的 train/val 共享同一个底层 dataset 对象，
    但两者要挂**不同**的 transforms（训练增强 / 评估确定）——包一层即可。

    属性:
        indices: 底层数据集中被选中的下标列表（测试用来验证划分的可复现性）。
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        self.indices = list(subset.indices)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]          # x 是 PIL 图片（download.py 故意不做张量化）
        return self.transform(x), y


def build_transforms(config, train):
    """构建预处理流水线（你要实现的核心之一，项目文档 §3）。

    参数:
        config: dict，读取 config["features"]["image_size"] 和 ["augment"]。
        train:  bool，True = 训练版（随机增强），False = 评估版（全确定性）。

    返回:
        torchvision.transforms.Compose。

    训练版（train=True 且 augment 开启时）:
        Resize(image_size + 32) → RandomCrop(image_size) → RandomHorizontalFlip
        → ToTensor → Normalize(IMAGENET_MEAN, IMAGENET_STD)
        （先放大一圈再随机裁回原尺寸，制造位置扰动；翻转制造方向扰动）

    评估版（train=False，或 augment 关闭时）:
        Resize(image_size) → CenterCrop(image_size)（size 相同时等于无操作，写上更稳）
        → ToTensor → Normalize(IMAGENET_MEAN, IMAGENET_STD)
        全确定性：同一张图每次变换结果必须逐元素一致（测试会验证）。

    提示:
        - transforms.Compose([...]) 把步骤串起来；
        - image_size 从 config["features"]["image_size"] 读——lenet 路线配 32，
          resnet18 路线配 224，换路线时只改 config.yaml；
        - Normalize 用本文件顶部的 IMAGENET_MEAN / IMAGENET_STD 常量。
    """
    raise NotImplementedError


def make_loaders(config):
    """构建三个 DataLoader：train / val / test（你要实现的核心之二，项目文档 §3）。

    参数:
        config: dict，读取 data / features / train 三节。

    返回:
        (train_loader, val_loader, test_loader):
            train_loader: 训练子集（50000 − val_size 张），训练版 transform，shuffle=True；
            val_loader:   验证子集（val_size 张），评估版 transform，shuffle=False；
            test_loader:  官方测试集（10000 张），评估版 transform，shuffle=False。

    步骤提示:
        1. train_set, test_set, meta = load_dataset(config)（data/download.py 已实现）；
        2. g = torch.Generator().manual_seed(config["data"]["random_state"])，
           random_split(train_set, [50000 − val_size, val_size], generator=g)
           —— 固定种子的生成器保证每次运行划分一致（不固定 = 每次分数没法对比）；
        3. 用 TransformedSubset 给两个子集分别挂 build_transforms(config, True/False)；
        4. test_set 也要挂评估版 transform（它是完整 dataset，直接赋值 transform 属性
           即可：test_set.transform = build_transforms(config, train=False)）；
        5. DataLoader(batch_size=config["train"]["batch_size"],
                      num_workers=config["train"]["num_workers"], ...)。
    """
    raise NotImplementedError
