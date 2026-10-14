"""手写 Transformer 注意力的测试：与 torch 对答案（SDPA / MultiheadAttention）是灵魂。

测试数据：make_demo_inputs（✅ 已实现，seed 固定可复现）+ 测试内现造的小数据。
运行：在本目录执行 `../../../.venv/bin/python -m pytest test_attention.py -v`
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from attention import (
    causal_mask,
    combine_heads,
    make_demo_inputs,
    make_torch_mha,
    multi_head_attention,
    positional_encoding,
    scaled_dot_product_attention,
    softmax,
    split_heads,
)


def _make_params(d_model=8, seed=1):
    """MHA 测试用的随机参数（📝 函数在测试体内调用，骨架态下测试 FAIL 而非 ERROR）。"""
    rng = np.random.default_rng(seed)
    params = {}
    for k in ["W_q", "W_k", "W_v", "W_o"]:
        params[k] = rng.normal(0, 0.5, (d_model, d_model))
    for k in ["b_q", "b_k", "b_v", "b_o"]:
        params[k] = rng.normal(0, 0.5, d_model)
    return params


# ---------------------------------------------------------------- ✅ 辅助函数（骨架态下即绿）


def test_softmax_行和为1且数值稳定():
    P = softmax(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=-1), 1)
    assert np.allclose(P[0], [0.090031, 0.244728, 0.665241], atol=1e-6)
    P_big = softmax(np.array([[[1000.0, 1001.0, 1002.0]]]))  # 三维输入沿最后一维
    assert np.isfinite(P_big).all() and np.allclose(P_big.sum(), 1)


def test_演示输入_形状与可复现性():
    X = make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42)
    assert X.shape == (2, 4, 8)
    assert X.dtype == np.float64
    again = make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42)
    assert np.array_equal(X, again)


def test_torch桥_参数装载成功():
    params = _make_params(d_model=8, seed=1)
    mha = make_torch_mha(params, d_model=8, n_heads=2)
    assert mha.in_proj_weight.shape == (24, 8)  # (3*d_model, d_model)，q,k,v 拼接
    assert mha.in_proj_bias.shape == (24,)
    assert mha.out_proj.weight.shape == (8, 8)
    # torch Linear 是 y = x @ Wᵀ + b，装载时转过置：in_proj_weight 前 8 行 == W_q.T
    assert np.allclose(mha.in_proj_weight.detach().numpy()[:8], params["W_q"].T)


# ---------------------------------------------------------------- causal_mask（§4）


def test_causal_mask_形状与上下三角():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    assert m.dtype == bool
    assert np.array_equal(np.tril(np.ones((4, 4), dtype=bool)), m)  # 下三角含对角线允许
    assert not np.triu(m, k=1).any()  # 严格上三角全禁止
    assert np.array_equal(
        causal_mask(3),
        [[True, False, False], [True, True, False], [True, True, True]],
    )


# ---------------------------------------------------------------- scaled_dot_product_attention（§3/§4）


def test_SDPA_输出与权重形状():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, 3, 8))  # lq=3
    K = rng.normal(size=(2, 5, 8))  # lk=5（交叉注意力时 lq != lk）
    V = rng.normal(size=(2, 5, 6))  # d_v=6 可与 d_k 不同
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 3, 6)
    assert weights.shape == (2, 3, 5)


def test_SDPA_手算已知值():
    # lq=lk=2, d_k=d_v=2，scores = QKᵀ/√2 ≈ [[0.7071, 0], [0, 0.7071]]
    Q = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    K = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    V = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    out, weights = scaled_dot_product_attention(Q, K, V)
    # 实测：weights ≈ [[0.6698, 0.3302], [0.3302, 0.6698]]
    assert np.allclose(weights[0], [[0.66976155, 0.33023845], [0.33023845, 0.66976155]], atol=1e-6)
    # output = weights @ V ≈ [[1.6605, 2.6605], [2.3395, 3.3395]]
    assert np.allclose(out[0], [[1.66047690, 2.66047690], [2.33952310, 3.33952310]], atol=1e-6)


def test_SDPA_权重行和为1():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, 3, 8))
    K = rng.normal(size=(2, 5, 8))
    V = rng.normal(size=(2, 5, 6))
    _, weights = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(weights.sum(axis=-1), 1)  # 每个 query 对各 key 的关注之和为 1


def test_SDPA_mask后被禁位置权重为0():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, 2, 4))
    K = rng.normal(size=(2, 2, 4))
    V = rng.normal(size=(2, 2, 4))
    mask = np.zeros((2, 2), dtype=bool)
    mask[:, 0] = True  # 只允许关注 key 0
    out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert np.allclose(weights[..., 1], 0)  # 被禁位置权重为 0（exp(-1e9) 下溢到精确的 0）
    assert np.allclose(out, V[:, :1, :].repeat(2, axis=1))  # 只剩 key 0 的 V 被完整复制


def test_SDPA_因果mask下权重严格下三角():
    rng = np.random.default_rng(0)
    l = 5
    Q = rng.normal(size=(2, l, 8))
    K = rng.normal(size=(2, l, 8))
    V = rng.normal(size=(2, l, 8))
    _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(l))
    for b in range(2):
        assert np.allclose(np.triu(weights[b], k=1), 0)  # 位置 i 看不到 j > i
        assert np.allclose(weights[b].sum(axis=-1), 1)  # 行和仍为 1


def test_SDPA_对答案_随机输入对照torch():
    # 实测 numpy vs torch.nn.functional.scaled_dot_product_attention 最大绝对差 ≈ 4e-16
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, 3, 8))
    K = rng.normal(size=(2, 5, 8))
    V = rng.normal(size=(2, 5, 6))
    out, _ = scaled_dot_product_attention(Q, K, V)
    out_torch = F.scaled_dot_product_attention(
        torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V)
    ).numpy()
    assert np.allclose(out, out_torch, atol=1e-9)


def test_SDPA_对答案_因果mask对照torch():
    # torch 侧 attn_mask 布尔语义与本文一致：True = 允许关注
    # 实测最大绝对差 ≈ 3e-16
    rng = np.random.default_rng(0)
    l = 5
    Q = rng.normal(size=(2, l, 8))
    K = rng.normal(size=(2, l, 8))
    V = rng.normal(size=(2, l, 8))
    mask = causal_mask(l)
    out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
    out_torch = F.scaled_dot_product_attention(
        torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V),
        attn_mask=torch.from_numpy(mask),
    ).numpy()
    assert np.allclose(out, out_torch, atol=1e-9)


# ---------------------------------------------------------------- positional_encoding（§6）


def test_位置编码_pos0已知值():
    # pos=0：sin(0)=0、cos(0)=1 → 偶数位全 0、奇数位全 1
    pe = positional_encoding(max_len=4, d_model=8)
    assert np.allclose(pe[0, 0::2], 0)
    assert np.allclose(pe[0, 1::2], 1)


def test_位置编码_形状与已知值():
    pe = positional_encoding(max_len=10, d_model=8)
    assert pe.shape == (10, 8)
    assert pe.dtype == np.float64
    # PE(1, 0)=sin(1)≈0.8415，PE(1, 1)=cos(1)≈0.5403，PE(1, 2)=sin(1/10000^(2/8))≈0.0998
    assert np.allclose(pe[1, 0], np.sin(1), atol=1e-12)
    assert np.allclose(pe[1, 1], np.cos(1), atol=1e-12)
    assert np.allclose(pe[1, 2], 0.09983341664682815, atol=1e-12)


def test_位置编码_不同位置行不同():
    pe = positional_encoding(max_len=10, d_model=8)
    for i in range(10):
        for j in range(i + 1, 10):
            assert not np.allclose(pe[i], pe[j])  # 每个位置的"指纹"独一无二


# ---------------------------------------------------------------- split_heads / combine_heads（§5）


def test_split_heads_形状与head内容正确():
    b, l, d_model, n_heads = 2, 3, 8, 2
    X = np.arange(b * l * d_model).reshape(b, l, d_model)
    heads = split_heads(X, n_heads)
    d_k = d_model // n_heads
    assert heads.shape == (b, n_heads, l, d_k)
    for h in range(n_heads):  # 第 h 个头的每个位置恰是原向量的第 h 段切片
        assert np.array_equal(heads[:, h, :, :], X[:, :, h * d_k:(h + 1) * d_k])


def test_combine_heads_与split互逆():
    X = make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42)
    for n_heads in [1, 2, 4, 8]:
        assert np.array_equal(combine_heads(split_heads(X, n_heads)), X)


# ---------------------------------------------------------------- multi_head_attention（§5）——对答案是灵魂


def test_MHA_输出形状():
    X = make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42)
    params = _make_params(d_model=8, seed=1)
    out = multi_head_attention(X, params, n_heads=2)
    assert out.shape == (2, 4, 8)
    assert np.isfinite(out).all()


def test_MHA_对答案_对照torch():
    # 同权重（make_torch_mha 装载）对照 torch.nn.MultiheadAttention
    # 实测双精度下最大绝对差 ≈ 2e-15（n_heads=2）/ 9e-16（n_heads=4），纯浮点累加顺序差异
    X = make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42)
    params = _make_params(d_model=8, seed=1)
    tX = torch.from_numpy(X)
    for n_heads in [2, 4]:
        out = multi_head_attention(X, params, n_heads)
        mha = make_torch_mha(params, d_model=8, n_heads=n_heads)
        with torch.no_grad():
            out_torch, _ = mha(tX, tX, tX)
        assert np.allclose(out, out_torch.numpy(), atol=1e-9)
