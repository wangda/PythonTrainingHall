"""模型层：nn.LSTM 多变量时序预测（已完整实现，工厂模式示范）。

结构一句话：LSTM 编码整个窗口 → 取最后一个时间步的隐状态 → Linear 映射到
horizon 个预测值。原理见前置文档 §4（LSTM 公式）与 §5（nn.LSTM 实战：
输出形状、batch_first），这里不留手写项。
"""

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """多变量输入 → LSTM → 最后时间步 → Linear → horizon 步预测。

    约定:
        - 输入 X 形状 (batch, window_size, input_size)，batch_first=True——
          batch 维在最前，和 DataLoader 产出的形状天然一致（前置文档 §5）；
        - nn.LSTM 返回 (output, (h_n, c_n))：output 形状
          (batch, window_size, hidden_size)，是每个时间步的顶层隐状态序列；
          取 output[:, -1, :] 即"看完整个窗口后"的隐状态；
        - 末层是 Linear(hidden_size, horizon)，【无激活函数】：回归任务的
          输出是任意实数，接 ReLU/sigmoid 都会把输出范围错误地截断
          （对照：分类任务末层也不接 softmax，由 CrossEntropyLoss 内部处理）。
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # 层间 dropout，单层时无意义
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        output, _ = self.lstm(x)        # output: (batch, window_size, hidden_size)
        last = output[:, -1, :]         # 取最后一个时间步: (batch, hidden_size)
        return self.head(last)          # (batch, horizon)


def create_model(config):
    """按 config["model"] 创建 LSTMForecaster（已完整实现）。

    参数:
        config: dict，读取 config["model"]["input_size"] / ["hidden_size"] /
                ["num_layers"] / ["dropout"]，以及 config["data"]["horizon"]
                （输出维度 = 预测步长）。

    返回:
        LSTMForecaster 实例。输出形状 (batch, horizon)，裸数值、无激活。
    """
    mcfg = config["model"]
    return LSTMForecaster(
        input_size=mcfg["input_size"],
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        dropout=mcfg.get("dropout", 0.0),
        horizon=config["data"]["horizon"],
    )


def count_parameters(model):
    """统计可训练参数量（用于测试里手算断言的对照）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
