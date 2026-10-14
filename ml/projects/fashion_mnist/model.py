"""模型层：按 config 动态堆叠 MLP（已完整实现，工厂模式示范）。

结构简单，教学点在前置文档 §7（nn.Module / autograd），这里不留手写项。
"""

import torch
from torch import nn


def create_model(config):
    """按 config["model"] 创建 MLP（已完整实现）。

    参数:
        config: dict，读取 config["model"]["input_dim"] / ["hidden_sizes"] /
                ["num_classes"]。hidden_sizes 是隐藏层宽度列表，如 [128, 64]。

    返回:
        torch.nn.Sequential：[Linear(input_dim, h1), ReLU, ..., Linear(h_last, num_classes)]。

    关键约定——末层不接 softmax:
        输出是 logits（原始打分，可正可负、和不为 1）。训练用
        nn.CrossEntropyLoss，它内部已经包含 log_softmax（前置文档 §5），
        再手动接 softmax 就是"双重 softmax"，梯度被压扁，训练会明显变差。
        需要概率时（如 predict.py 展示置信度）才显式 softmax 一次。
    """
    mcfg = config["model"]
    dims = [mcfg["input_dim"], *mcfg["hidden_sizes"], mcfg["num_classes"]]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:      # 隐藏层之间才接 ReLU，末层输出裸 logits
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def count_parameters(model):
    """统计可训练参数量（用于测试里手算断言的对照）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
