"""LSTM（长短期记忆网络）手写练习骨架——阶段二模块 6。

仅用 NumPy 手写 LSTM 的三件套：切门、单步前向、多步展开。所有公式与前置文档
《11-LSTM与时间序列前置学习文档.md》一一对应，docstring 中标注了章节。
对答案工具是 torch.nn.LSTMCell / torch.nn.LSTM（同权重比较输出），
本文件已实现完整的 make_torch_lstm_cell 负责把 numpy 参数灌进 torch。

约定（与文档 §4 一致，全练习统一遵守）：
    - 单步前向：lstm_cell_forward(x_t, h_prev, c_prev, params) -> (h_t, c_t)。
      x_t 形状 (batch, input_size)，h_prev / c_prev 形状 (batch, hidden_size)。
    - 参数用 dict 表示，完全对齐 torch.nn.LSTMCell 的参数布局：
          params["W_ih"]: 形状 (4*hidden_size, input_size)   —— 输入到门的权重
          params["W_hh"]: 形状 (4*hidden_size, hidden_size)  —— 隐状态到门的权重
          params["b_ih"]: 形状 (4*hidden_size,)              —— 输入侧偏置
          params["b_hh"]: 形状 (4*hidden_size,)              —— 隐状态侧偏置
      四个门的权重沿第 0 轴拼在一起。

⚠️⚠️ 最大的坑：门的拼接顺序是 i, f, g, o ⚠️⚠️
    PyTorch 官方顺序：输入门 i、遗忘门 f、候选记忆 g̃、输出门 o。
    很多教材按 f, i, o, g 的顺序讲公式——顺序记错，对答案时一切全错。
    本练习（和 torch、和测试）一律按 i, f, g, o。

公式（文档 §4，单步）：
    z = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh     形状 (batch, 4*hidden)
    z 切成 4 段：i, f, g, o（各 (batch, hidden)）
    i = σ(i)，f = σ(f)，o = σ(o)，g = tanh(g)
    c_t = f ⊙ c_prev + i ⊙ g        （遗忘旧的 + 写入新的）
    h_t = o ⊙ tanh(c_t)             （输出门控制暴露多少细胞状态）

使用流程：
    params = init_params(input_size=3, hidden_size=4, seed=0)
    H, C = lstm_forward(X, params)          # X 形状 (batch, seq_len, input_size)
    cell = make_torch_lstm_cell(params, 3, 4)  # 同权重的 torch 版，用来对答案
"""

import numpy as np
import torch


# ============================================================ 已实现完整的辅助函数（✅）


def sigmoid(x):
    """Sigmoid 激活函数，已实现完整（模块 2/4 已练过，这里直接给出）。

    公式：σ(x) = 1 / (1 + e^{-x})（逐元素），把任意实数压到 (0, 1)。
    LSTM 里三道门（i, f, o）都用 sigmoid：输出在 0~1 之间，天然是"开度"。

    参数:
        x: np.ndarray，任意形状。

    返回:
        np.ndarray，与 x 同形状，值域 (0, 1)。

    参考输出:
        sigmoid([-2.0, 0.0, 2.0]) ≈ [0.1192, 0.5000, 0.8808]。
    """
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    """双曲正切激活函数，已实现完整（直接调 np.tanh）。

    tanh 把任意实数压到 (-1, 1)。LSTM 里"内容"用 tanh：候选记忆 g̃ = tanh(g)，
    以及 h_t = o ⊙ tanh(c_t)——sigmoid 管门（开度），tanh 管内容（数值）。

    参数:
        x: np.ndarray，任意形状。

    返回:
        np.ndarray，与 x 同形状，值域 (-1, 1)。

    参考输出:
        tanh([-2.0, 0.0, 2.0]) ≈ [-0.9640, 0.0000, 0.9640]。
    """
    return np.tanh(x)


def init_params(input_size, hidden_size, seed=42):
    """按 torch 默认初始化生成 LSTM 参数，已实现完整。

    torch.nn.LSTMCell 的默认初始化：所有权重和偏置 ~ 均匀分布 U(-k, k)，
    k = 1/sqrt(hidden_size)。这里用 numpy 复刻同样的分布（RNG 不同，数值不同，
    但分布一致；对答案靠的是把本函数生成的参数灌进 torch，见 make_torch_lstm_cell）。

    参数:
        input_size:  int，输入特征维度。
        hidden_size: int，隐藏状态维度。
        seed:        int，随机种子，固定后参数可复现。用 np.random.default_rng(seed)。

    返回:
        params: dict，四个键（对齐 torch 布局）：
            "W_ih": np.ndarray，形状 (4*hidden_size, input_size)；
            "W_hh": np.ndarray，形状 (4*hidden_size, hidden_size)；
            "b_ih": np.ndarray，形状 (4*hidden_size,)；
            "b_hh": np.ndarray，形状 (4*hidden_size,)；
            全部元素 ~ U(-1/√hidden_size, 1/√hidden_size)。

    参考输出（input_size=3, hidden_size=4, seed=0）:
        理论 k = 1/√4 = 0.5；W_ih 形状 (16, 3)，元素全部落在 [-0.5, 0.5]。
    """
    rng = np.random.default_rng(seed)
    k = 1.0 / np.sqrt(hidden_size)
    return {
        "W_ih": rng.uniform(-k, k, (4 * hidden_size, input_size)),
        "W_hh": rng.uniform(-k, k, (4 * hidden_size, hidden_size)),
        "b_ih": rng.uniform(-k, k, (4 * hidden_size,)),
        "b_hh": rng.uniform(-k, k, (4 * hidden_size,)),
    }


def make_torch_lstm_cell(params, input_size, hidden_size):
    """把 numpy 参数灌进 torch.nn.LSTMCell 并返回它——对答案的桥，已实现完整。

    返回的 cell 与你手写的 lstm_cell_forward 吃同样的 (x_t, h_prev, c_prev)，
    只是 h 和 c 要作为 tuple (h, c) 传入、以 tuple 返回：
        h_t, c_t = cell(torch.from_numpy(x_t), (torch.from_numpy(h_prev),
                                                 torch.from_numpy(c_prev)))
    转成 float64（.double()）再灌权重，保证与 numpy 计算逐位可比（偏差 1e-15 量级）。

    参数:
        params:      dict，init_params 返回的参数（或你自己构造的同布局 dict）。
        input_size:  int，输入特征维度。
        hidden_size: int，隐藏状态维度。

    返回:
        torch.nn.LSTMCell，权重已替换为 params 的值，dtype=float64。
    """
    cell = torch.nn.LSTMCell(input_size, hidden_size).double()
    with torch.no_grad():
        cell.weight_ih.copy_(torch.from_numpy(params["W_ih"]))
        cell.weight_hh.copy_(torch.from_numpy(params["W_hh"]))
        cell.bias_ih.copy_(torch.from_numpy(params["b_ih"]))
        cell.bias_hh.copy_(torch.from_numpy(params["b_hh"]))
    return cell


def generate_demo_sequence(seed=42):
    """生成一条 200 步的合成正弦 + 噪声序列，供多步展开演示，已实现完整。

    y_t = sin(2π·t / 50) + 0.1·ε_t，ε_t ~ N(0, 1)，t = 0..199。
    周期 50 步的正弦叠加小幅噪声——典型的"有长期规律 + 短期扰动"时序，
    正好用来直观感受 LSTM 的记忆能力（文档 §1/§6）。

    参数:
        seed: int，随机种子，固定后序列可复现。

    返回:
        np.ndarray，形状 (200, 1)，float64——(seq_len, input_size=1) 布局，
        想喂给 lstm_forward 记得加 batch 维：seq[np.newaxis] → (1, 200, 1)。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 50) + 0.1 * rng.normal(0, 1, 200)
    return y.reshape(-1, 1)


# ============================================================ 留给你手写（📝）


def split_gates(z, hidden_size):
    """把拼好的门向量切成四段，对应文档 §4——热身题。

    ⚠️ 顺序是 i, f, g, o（PyTorch 官方顺序），不是某些教材的 f, i, o, g！

    参数:
        z:           np.ndarray，形状 (batch, 4*hidden_size)，四个门拼在一起的
                     加权输入（沿第 1 轴拼接）。
        hidden_size: int，隐藏状态维度，每段门的宽度。

    返回:
        (i, f, g, o): 四个 np.ndarray，各形状 (batch, hidden_size)，依次是
                      输入门、遗忘门、候选记忆、输出门的加权输入（未过激活）。

    参考输出（hidden_size=2，z = [[0, 1, 2, 3, 4, 5, 6, 7]]）:
        i = [[0, 1]]，f = [[2, 3]]，g = [[4, 5]]，o = [[6, 7]]。

    实现提示:
        沿第 1 轴切片：z[:, 0:h]、z[:, h:2*h]、z[:, 2*h:3*h]、z[:, 3*h:4*h]。
    """
    raise NotImplementedError


def lstm_cell_forward(x_t, h_prev, c_prev, params):
    """LSTM 单步前向传播，对应文档 §4——本练习的核心。

    公式（文档 §4）：
        z = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh   形状 (batch, 4*hidden)
        i, f, g, o = split_gates(z, hidden_size)           各 (batch, hidden)
        i = σ(i)，f = σ(f)，o = σ(o)，g = tanh(g)
        c_t = f ⊙ c_prev + i ⊙ g
        h_t = o ⊙ tanh(c_t)

    参数:
        x_t:    np.ndarray，形状 (batch, input_size)，当前时间步的输入。
        h_prev: np.ndarray，形状 (batch, hidden_size)，上一步的隐状态。
        c_prev: np.ndarray，形状 (batch, hidden_size)，上一步的细胞状态。
        params: dict，init_params 返回的参数（"W_ih"/"W_hh"/"b_ih"/"b_hh"）。

    返回:
        (h_t, c_t):
            h_t: np.ndarray，形状 (batch, hidden_size)，当前步隐状态；
            c_t: np.ndarray，形状 (batch, hidden_size)，当前步细胞状态。

    参考输出（params=init_params(3, 4, seed=0)，batch=2，x_t/h_prev/c_prev
    用 default_rng(1) 标准正态生成）:
        与同权重 torch.nn.LSTMCell 的最大绝对偏差 ≈ 1e-16（float64 纯浮点
        运算顺序差异）；h_t 第一行前两个值 ≈ [-0.1571, 0.1958]。

    实现提示:
        - 注意 W_ih 要转置：x_t @ W_ih.T，(batch,input)@(input,4*hidden)。
        - sigmoid 管门（i, f, o），tanh 管内容（g 和最后的 tanh(c_t)），别串。
        - hidden_size 从 params["W_hh"].shape[1] 推导，不要新增参数。
    """
    raise NotImplementedError


def lstm_forward(X, params):
    """LSTM 多步展开前向传播，对应文档 §4/§5。

    从零初始化 h0 = 0、c0 = 0（对齐 torch.nn.LSTM 默认行为），逐时间步：
        (h_t, c_t) = lstm_cell_forward(X[:, t, :], h_prev, c_prev, params)
    把每一步的 h_t 收集起来。

    参数:
        X:      np.ndarray，形状 (batch, seq_len, input_size)，整个序列
                （batch_first 布局，对齐 torch.nn.LSTM(batch_first=True)）。
        params: dict，init_params 返回的参数。

    返回:
        (H, C):
            H: np.ndarray，形状 (batch, seq_len, hidden_size)，所有时间步的
               隐状态（对齐 nn.LSTM 的 output，H[:, t, :] 就是第 t 步的 h_t）；
            C: np.ndarray，形状 (batch, hidden_size)，最后一个时间步的细胞状态
               （对齐 nn.LSTM 返回的 c_n.squeeze(0)）。

    参考输出（params=init_params(3, 4, seed=0)，X 用 default_rng(1) 生成，
    形状 (2, 5, 3)）:
        与同权重 torch.nn.LSTM(batch_first=True) 对比：H 全程最大绝对偏差
        ≈ 5.6e-17，最终 c_n ≈ 2.8e-17（1e-16 量级）——逐步传递没有错位
        就是这个精度。

    实现提示:
        - 隐状态维度：hidden_size = params["W_hh"].shape[1]。
        - 上一步输出的 (h_t, c_t) 就是下一步输入的 (h_prev, c_prev)——
          h 和 c 都要传，只传 h 不传 c 是最典型的错位 bug。
        - 收集 H：可以每步 append 到列表，最后 np.stack(list, axis=1)。
    """
    raise NotImplementedError
