"""model.py 的单元测试：LSTMForecaster 已完整实现，这部分测试应当直接通过。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import torch

from model import LSTMForecaster, count_parameters, create_model

CONFIG = {
    "data": {"horizon": 4},
    "model": {"input_size": 7, "hidden_size": 64, "num_layers": 2, "dropout": 0.1},
}


def test_工厂_按config创建():
    model = create_model(CONFIG)
    assert isinstance(model, LSTMForecaster)
    assert model.lstm.hidden_size == 64 and model.lstm.num_layers == 2
    assert model.head.out_features == 4  # 输出维度 = horizon


def test_前向_输出形状与无激活截断():
    model = LSTMForecaster(input_size=7, hidden_size=64, num_layers=2, dropout=0.1, horizon=4)
    model.eval()
    X = torch.randn(16, 96, 7)  # (batch, window, features)
    out = model(X)
    assert out.shape == (16, 4)
    assert not torch.all(out >= 0)  # 回归输出可正可负——末层绝不能接 ReLU/sigmoid 截断


def test_参数量_手算断言():
    """LSTM 参数手算（input=7, hidden=64, layers=2, horizon=4）：
    每层 4*hidden*(input+hidden) + 2*4*hidden（torch 的 b_ih 和 b_hh 两个偏置）：
        layer1: 4*64*(7+64)  + 512 = 18688
        layer2: 4*64*(64+64) + 512 = 33280
    head: 64*4 + 4 = 260。合计 52228。
    """
    model = LSTMForecaster(input_size=7, hidden_size=64, num_layers=2, dropout=0.1, horizon=4)
    assert count_parameters(model) == 18688 + 33280 + 260


def test_单层时dropout不生效():
    model = LSTMForecaster(input_size=7, hidden_size=64, num_layers=1, dropout=0.5, horizon=4)
    assert model.lstm.dropout == 0.0  # 单层 LSTM 层间 dropout 无处安放，torch 要求为 0
