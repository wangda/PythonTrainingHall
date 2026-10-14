"""model.py 的单元测试：create_model 已完整实现，这部分测试应当直接通过。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import torch
from torch import nn

from model import count_parameters, create_model

CONFIG = {"model": {"input_dim": 784, "hidden_sizes": [128, 64], "num_classes": 10}}


def test_输出形状是batch乘10():
    model = create_model(CONFIG)
    logits = model(torch.randn(7, 784))
    assert logits.shape == (7, 10)      # 10 个类别的 logits


def test_末层是Linear不接softmax():
    model = create_model(CONFIG)
    assert isinstance(list(model.children())[-1], nn.Linear)
    assert not any(isinstance(m, (nn.Softmax, nn.LogSoftmax)) for m in model.modules())
    # logits 是原始打分：不做 softmax，各行概率和一般不为 1
    logits = model(torch.randn(4, 784))
    assert not torch.allclose(logits.softmax(dim=1), logits)


def test_参数量手算断言():
    # 784×128+128 + 128×64+64 + 64×10+10 = 100480 + 8256 + 650
    model = create_model(CONFIG)
    assert count_parameters(model) == 109_386


def test_hidden_sizes动态堆叠():
    config = {"model": {"input_dim": 784, "hidden_sizes": [256, 128, 64], "num_classes": 10}}
    model = create_model(config)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 4                                    # 3 隐层 + 1 输出层
    assert [m.out_features for m in linears] == [256, 128, 64, 10]


def test_无隐藏层退化为线性模型():
    config = {"model": {"input_dim": 784, "hidden_sizes": [], "num_classes": 10}}
    model = create_model(config)
    assert len(list(model.children())) == 1                     # 只有一个 Linear
    assert model(torch.randn(3, 784)).shape == (3, 10)
