"""模型层：LeNet（从零基线）与 ResNet18（迁移学习）的创建与优化器分组。

LeNet 类已完整实现（前置文档 §5.3 的参考版，3 通道 CIFAR 输入）；
create_model / create_optimizer 是你要实现的迁移学习手术（项目文档 §5，
前置文档 §6.3/§6.4）。

工厂模式：调用方只说一句"按 config 给我一个模型"，切换路线只改 config.yaml。
"""

import torch
import torch.nn as nn

MODEL_NAMES = ("lenet", "resnet18")
FINETUNE_MODES = ("head_only", "full")


class LeNet(nn.Module):
    """LeNet-5 的 CIFAR 版（3 通道 32×32 输入，10 类输出），已完整实现。

    逐层尺寸（前置文档 §4.2 的尺寸推导，3 通道版）：
        (3,32,32) → Conv(5×5)→(6,28,28) → Pool→(6,14,14)
                  → Conv(5×5)→(16,10,10) → Pool→(16,5,5) → Flatten 400
                  → FC 400→120→84→10
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(),     # (3,32,32)→(6,28,28)
            nn.MaxPool2d(2, 2),                            # →(6,14,14)
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),    # →(16,10,10)
            nn.MaxPool2d(2, 2),                            # →(16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # →400
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_trainable_params(model):
    """统计可训练参数量（已完整实现）——冻结策略的直观体检表。

    head_only 的 ResNet18 应该只有 5130（=512×10+10，只有新 fc 头）；
    全量微调是 1100 万+。这个数字比任何文字都更能说明"冻结"意味着什么。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(config):
    """按 config 创建模型（你要实现的核心之一，前置文档 §6.3 的手术四步）。

    参数:
        config: dict，读取 config["model"] 的 name / num_classes；
                resnet18 分支还要读 pretrained 和 finetune。

    返回:
        nn.Module，未训练。

    name == "lenet":
        直接返回 LeNet(num_classes=config["model"]["num_classes"])。

    name == "resnet18"（迁移学习手术，顺序不要乱）:
        1. 取预训练网络：
           weights = "IMAGENET1K_V1" if config["model"]["pretrained"] else None
           model = torchvision.models.resnet18(weights=weights)
           （pretrained=True 首次会联网下载约 45MB 权重——这就是迁移学习的本体）
        2. 换头：model.fc = nn.Linear(model.fc.in_features, num_classes)
           —— 先换头再冻结，新头默认 requires_grad=True；
        3. 按 config["model"]["finetune"] 冻结：
           - "head_only"：for p in model.parameters(): p.requires_grad = False，
             然后单独放开 model.fc 的参数（for p in model.fc.parameters(): True）；
           - "full"：全部可训练，什么都不用冻；
        4. 创建后打印 count_trainable_params(model) 自检（5130 vs 1100 万+）。

    异常:
        ValueError: 未知 model.name 或未知 finetune 模式。
    """
    raise NotImplementedError


def create_optimizer(model, config):
    """按冻结策略创建优化器（你要实现的核心之二，前置文档 §6.3 第④步 / §6.4）。

    参数:
        model:  create_model 返回的模型（requires_grad 已按策略设置好）。
        config: dict，读取 config["train"] 的 lr_head / lr_backbone_deep /
                lr_backbone_shallow。

    返回:
        torch.optim.Adam 实例。

    规则:
        - 通用原则：优化器只收 requires_grad=True 的参数
          （filter(lambda p: p.requires_grad, model.parameters())），
          冻结参数传进去也不会更新，但会白白分配动量内存；
        - finetune == "head_only"（或 lenet）：单一学习率 lr_head；
        - finetune == "full"：分组学习率（前置文档 §6.4 的模板）——
          fc 头 lr_head、layer4 lr_backbone_deep、其余 lr_backbone_shallow：
              torch.optim.Adam([
                  {"params": model.layer4.parameters(), "lr": lr_backbone_deep},
                  {"params": model.fc.parameters(),     "lr": lr_head},
              ], lr=lr_backbone_shallow)
          （Adam 的默认 lr 作用于没进任何组的参数——浅层主干）。

    提示:
        - lenet 没有 layer4/fc 属性，分组逻辑只对 resnet18 + full 生效，
          分支判断用 isinstance(model, LeNet) 或读 config["model"]["name"]；
        - 读 model 的 finetune 策略：可以从 config["model"]["finetune"] 再读一次，
          也可以 create_model 时把策略存到 model.finetune 属性上（推荐，自检更方便）。
    """
    raise NotImplementedError
