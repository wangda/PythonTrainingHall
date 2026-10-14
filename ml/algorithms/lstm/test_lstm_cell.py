"""手写 LSTM 的测试：与 torch.nn.LSTMCell / torch.nn.LSTM 同权重对答案是灵魂。

测试数据：init_params 生成的随机参数 + 测试内现造的随机输入。
运行：在本目录执行 `../../../.venv/bin/python -m pytest test_lstm_cell.py -v`
"""

import numpy as np
import torch

from lstm_cell import (
    generate_demo_sequence,
    init_params,
    lstm_cell_forward,
    lstm_forward,
    make_torch_lstm_cell,
    sigmoid,
    split_gates,
    tanh,
)


def _单步对答案素材(input_size=3, hidden_size=4, batch=5, seed=0):
    """随机参数 + 随机输入（📝 函数在测试体内调用，骨架态下测试 FAIL 而非 ERROR）。

    返回 (x_t, h_prev, c_prev, params)。
    """
    params = init_params(input_size, hidden_size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    x_t = rng.normal(0, 1, (batch, input_size))
    h_prev = rng.normal(0, 1, (batch, hidden_size))
    c_prev = rng.normal(0, 1, (batch, hidden_size))
    return x_t, h_prev, c_prev, params


# ---------------------------------------------------------------- ✅ 辅助函数（骨架态下即绿）


def test_sigmoid_已知值():
    out = sigmoid(np.array([-2.0, 0.0, 2.0]))
    assert np.allclose(out, [0.1192029, 0.5, 0.8807970], atol=1e-6)


def test_tanh_已知值():
    out = tanh(np.array([-2.0, 0.0, 2.0]))
    assert np.allclose(out, [-0.9640276, 0.0, 0.9640276], atol=1e-6)


def test_初始化_形状范围与可复现():
    params = init_params(input_size=3, hidden_size=4, seed=0)
    assert params["W_ih"].shape == (16, 3)  # 4*hidden × input
    assert params["W_hh"].shape == (16, 4)  # 4*hidden × hidden
    assert params["b_ih"].shape == (16,) and params["b_hh"].shape == (16,)
    k = 1.0 / np.sqrt(4)  # torch 默认初始化 U(-1/√hidden, 1/√hidden)
    for v in params.values():
        assert v.min() >= -k and v.max() <= k
    again = init_params(input_size=3, hidden_size=4, seed=0)
    assert all(np.array_equal(params[key], again[key]) for key in params)


def test_torch桥_权重确实灌进去了():
    params = init_params(input_size=3, hidden_size=4, seed=0)
    cell = make_torch_lstm_cell(params, 3, 4)
    assert np.array_equal(cell.weight_ih.detach().numpy(), params["W_ih"])
    assert np.array_equal(cell.weight_hh.detach().numpy(), params["W_hh"])
    assert np.array_equal(cell.bias_ih.detach().numpy(), params["b_ih"])
    assert np.array_equal(cell.bias_hh.detach().numpy(), params["b_hh"])


def test_演示序列_形状与可复现():
    seq = generate_demo_sequence(seed=42)
    assert seq.shape == (200, 1)
    assert np.array_equal(seq, generate_demo_sequence(seed=42))
    assert np.isfinite(seq).all()


# ---------------------------------------------------------------- split_gates（§4）热身


def test_切门_形状与内容():
    h = 2
    z = np.arange(2 * 4 * h, dtype=float).reshape(2, 4 * h)
    i, f, g, o = split_gates(z, h)
    for gate in (i, f, g, o):
        assert gate.shape == (2, h)
    assert np.array_equal(i, z[:, 0:h])
    assert np.array_equal(f, z[:, h:2 * h])
    assert np.array_equal(g, z[:, 2 * h:3 * h])
    assert np.array_equal(o, z[:, 3 * h:4 * h])


def test_切门_顺序是i_f_g_o():
    # 已知构造值：四段分别是 10, 20, 30, 40——顺序错了立刻现形
    h = 3
    z = np.concatenate([np.full((1, h), 10.0), np.full((1, h), 20.0),
                        np.full((1, h), 30.0), np.full((1, h), 40.0)], axis=1)
    i, f, g, o = split_gates(z, h)
    assert (i == 10).all() and (f == 20).all() and (g == 30).all() and (o == 40).all()


# ---------------------------------------------------------------- lstm_cell_forward（§4）核心


def test_单步前向_形状():
    x_t, h_prev, c_prev, params = _单步对答案素材()
    h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev, params)
    assert h_t.shape == (5, 4) and c_t.shape == (5, 4)


def test_单步前向_对答案torch():
    # 与 torch.nn.LSTMCell 同权重比较：实测最大绝对偏差 1e-16 量级（float64）
    x_t, h_prev, c_prev, params = _单步对答案素材()
    h_t, c_t = lstm_cell_forward(x_t, h_prev, c_prev, params)
    cell = make_torch_lstm_cell(params, 3, 4)
    h_ref, c_ref = cell(torch.from_numpy(x_t),
                        (torch.from_numpy(h_prev), torch.from_numpy(c_prev)))
    assert np.allclose(h_t, h_ref.detach().numpy(), atol=1e-6)
    assert np.allclose(c_t, c_ref.detach().numpy(), atol=1e-6)


# ---------------------------------------------------------------- 门的行为断言（教学性，§3/§4）


def _全零权重组装(hidden=3):
    """只留偏置起作用的 params：W 全零、b_hh 全零，方便用 b_ih 精确控制各门。"""
    params = {
        "W_ih": np.zeros((4 * hidden, 2)),
        "W_hh": np.zeros((4 * hidden, hidden)),
        "b_ih": np.zeros(4 * hidden),
        "b_hh": np.zeros(4 * hidden),
    }
    return params, hidden


def test_遗忘门大开_细胞状态原样记住():
    # b_ih 的 f 段（第 hidden..2*hidden 个）设 +10 → f ≈ σ(10) ≈ 0.99995 ≈ 1
    # i 段设 -10 → i ≈ 0，新记忆写不进；于是 c_t ≈ c_prev（完全记住过去）
    params, hidden = _全零权重组装()
    params["b_ih"][hidden:2 * hidden] = 10.0   # f 大开
    params["b_ih"][0:hidden] = -10.0           # i 关闭，排除新记忆干扰
    rng = np.random.default_rng(0)
    c_prev = rng.normal(0, 1, (4, hidden))
    _, c_t = lstm_cell_forward(np.zeros((4, 2)), np.zeros((4, hidden)), c_prev, params)
    assert np.allclose(c_t, c_prev, atol=1e-3)  # f≈1 时记忆几乎无损


def test_遗忘门关闭_过去被清零():
    # f 段设 -10 → f ≈ 0，c_prev 被彻底遗忘；i 也关闭 → c_t ≈ 0
    params, hidden = _全零权重组装()
    params["b_ih"][hidden:2 * hidden] = -10.0  # f 关闭
    params["b_ih"][0:hidden] = -10.0           # i 关闭
    rng = np.random.default_rng(0)
    c_prev = rng.normal(0, 1, (4, hidden))
    _, c_t = lstm_cell_forward(np.zeros((4, 2)), np.zeros((4, hidden)), c_prev, params)
    assert np.allclose(c_t, 0.0, atol=1e-3)


def test_输入门关闭_新记忆进不来():
    # i 关闭、g 拉满（b=+10 → tanh 饱和到 ≈1）、f 关闭 → c_t ≈ i⊙g ≈ 0
    params, hidden = _全零权重组装()
    params["b_ih"][0:hidden] = -10.0               # i 关闭
    params["b_ih"][2 * hidden:3 * hidden] = 10.0   # g 内容拉满
    params["b_ih"][hidden:2 * hidden] = -10.0      # f 关闭，排除旧记忆干扰
    rng = np.random.default_rng(0)
    c_prev = rng.normal(0, 1, (4, hidden))
    _, c_t = lstm_cell_forward(np.zeros((4, 2)), np.zeros((4, hidden)), c_prev, params)
    assert np.allclose(c_t, 0.0, atol=1e-3)


# ---------------------------------------------------------------- lstm_forward（§4/§5）多步展开


def _多步对答案素材(input_size=3, hidden_size=4, batch=2, seq_len=5, seed=0):
    params = init_params(input_size, hidden_size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = rng.normal(0, 1, (batch, seq_len, input_size))
    return X, params


def test_多步展开_形状():
    X, params = _多步对答案素材()
    H, C = lstm_forward(X, params)
    assert H.shape == (2, 5, 4)  # (batch, seq_len, hidden)
    assert C.shape == (2, 4)


def test_多步展开_对答案torchLSTM():
    # 与 torch.nn.LSTM(batch_first=True) 同权重逐步比较：
    # H 全程 allclose + 最终 c_n allclose，同时验证 h/c 逐步传递没有错位。
    # 实测 H 与 c_n 的最大绝对偏差均为 1e-16 量级（float64）。
    X, params = _多步对答案素材()
    H, C = lstm_forward(X, params)
    lstm = torch.nn.LSTM(3, 4, batch_first=True).double()
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(torch.from_numpy(params["W_ih"]))
        lstm.weight_hh_l0.copy_(torch.from_numpy(params["W_hh"]))
        lstm.bias_ih_l0.copy_(torch.from_numpy(params["b_ih"]))
        lstm.bias_hh_l0.copy_(torch.from_numpy(params["b_hh"]))
    output, (_, c_n) = lstm(torch.from_numpy(X))
    assert np.allclose(H, output.detach().numpy(), atol=1e-6)
    assert np.allclose(C, c_n.detach().numpy()[0], atol=1e-6)  # c_n 形状 (1, batch, hidden)
