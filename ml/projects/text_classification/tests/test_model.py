"""model.py 的单元测试：TextClassifier 已完整实现，这部分测试应当直接通过。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import torch

from model import TextClassifier, count_parameters, create_model

CONFIG = {
    "features": {"max_len": 16, "vocab_size": 100},
    "model": {
        "d_model": 32, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 64, "dropout": 0.0, "num_classes": 4,
    },
}


def test_工厂_按config创建():
    model = create_model(CONFIG)
    assert isinstance(model, TextClassifier)
    assert model.encoder.num_layers == 2
    assert model.head.out_features == 4


def test_前向_输出形状():
    model = create_model(CONFIG)
    model.eval()
    ids = torch.randint(3, 100, (4, 16))
    mask = torch.zeros(4, 16, dtype=torch.bool)
    out = model(ids, padding_mask=mask)
    assert out.shape == (4, 4)   # (batch, num_classes) 裸 logits


def test_padding_mask生效_补pad输出不变():
    """同一句话，后面补不同数量的 [PAD]，[CLS] 位输出必须不变——
    这是 src_key_padding_mask 正确性的强测试：注意力不许关注空白。"""
    torch.manual_seed(0)
    model = create_model(CONFIG)
    model.eval()
    ids_full = torch.tensor([[3, 4, 5, 6, 7, 0, 0, 0]])   # 5 个词 + 3 个 [PAD]
    mask_full = torch.tensor([[False] * 5 + [True] * 3])
    ids_short = ids_full[:, :5]                           # 同一句话，不带 PAD
    mask_short = torch.tensor([[False] * 5])
    with torch.no_grad():
        out_full = model(ids_full, padding_mask=mask_full)
        out_short = model(ids_short, padding_mask=mask_short)
    assert torch.allclose(out_full, out_short, atol=1e-4)  # float32 数值误差量级


def test_位置编码已注入():
    """pe 是 buffer：随 state_dict 保存、形状 (max_len, d_model)、pos=0 行为正确。"""
    model = create_model(CONFIG)
    assert model.pe.shape == (16, 32)
    assert torch.allclose(model.pe[0, 0::2], torch.zeros(16))   # sin(0)=0
    assert torch.allclose(model.pe[0, 1::2], torch.ones(16))    # cos(0)=1


def test_参数量_手算断言():
    """vocab=100, d_model=32, nhead=4, layers=2, ffn=64, classes=4, max_len=16：
        embedding: 100*32 = 3200
        每层 EncoderLayer：
            self_attn in_proj 3*32*32 + 3*32 = 3168；out_proj 32*32+32 = 1056 → 4224
            linear1 32*64+64 = 2112；linear2 64*32+32 = 2080 → 4192
            两个 LayerNorm: 2*(32+32) = 128
            合计每层 4224+4192+128 = 8544；两层 17088
        head: 32*4+4 = 132
        位置编码是 buffer 不参与训练。合计 3200+17088+132 = 20420。
    """
    model = create_model(CONFIG)
    assert count_parameters(model) == 3200 + 2 * 8544 + 132
