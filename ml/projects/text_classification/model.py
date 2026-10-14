"""模型层：Transformer Encoder 文本分类器（已完整实现，工厂模式）。

结构一句话：Embedding → 正弦位置编码 → TransformerEncoder → 取 [CLS] 位 → Linear 分类头。
原理见前置文档 §6（位置编码）与 §7（PyTorch 搭 Transformer Encoder 做分类），
这里不留手写项——手写注意力在 ml/algorithms/transformer_attention/ 练过了。
"""

import math

import numpy as np
import torch
from torch import nn


class TextClassifier(nn.Module):
    """词 id 序列 → TransformerEncoder → [CLS] 位向量 → 4 类 logits。

    约定:
        - 输入 ids 形状 (batch, max_len) int64，padding_mask (batch, max_len) bool
          （True = padding，torch 的 src_key_padding_mask 约定是"True 忽略"）；
        - 位置编码用正弦版（前置文档 §6 的公式），注册为 buffer：不参与训练、
          随 state_dict 保存、device 跟着模型走；
        - 取第 0 位（[CLS]）的输出向量做分类——它因为因果上"看得到全文"，
          被训练成一个"全文摘要向量"（BERT 同款套路）；
        - 末层 Linear 输出裸 logits，不接 softmax：CrossEntropyLoss 内部已含
          softmax（训练时）；预测时想要概率再显式 torch.softmax 一次。
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, num_classes, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.head = nn.Linear(d_model, num_classes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 正弦位置编码（公式与前置文档 §6 完全一致，向量化实现）
        pos = np.arange(max_len, dtype=np.float32).reshape(-1, 1)
        i = np.arange(d_model) // 2
        angles = pos / 10000 ** (2 * i / d_model)
        pe = np.where(np.arange(d_model) % 2 == 0, np.sin(angles), np.cos(angles))
        self.register_buffer("pe", torch.from_numpy(pe.astype(np.float32)))

    def forward(self, ids, padding_mask=None):
        x = self.embedding(ids) * math.sqrt(self.d_model)   # 缩放：让嵌入盖过位置编码
        x = x + self.pe[: ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(x[:, 0, :])                        # 取 [CLS] 位


def create_model(config):
    """按 config 创建 TextClassifier（已完整实现）。

    参数:
        config: dict，读取 config["model"] 的结构字段，词表大小取
                config["features"]["vocab_size"]（词表是特征工程的产物），
                max_len 同理来自 config["features"]["max_len"]。

    返回:
        TextClassifier 实例，输出 (batch, num_classes) 裸 logits。
    """
    mcfg = config["model"]
    return TextClassifier(
        vocab_size=config["features"]["vocab_size"],
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        num_layers=mcfg["num_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"],
        num_classes=mcfg["num_classes"],
        max_len=config["features"]["max_len"],
    )


def count_parameters(model):
    """统计可训练参数量（用于测试里手算断言的对照）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
