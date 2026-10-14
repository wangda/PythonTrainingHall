"""features.py 的单元测试：分词、词表防泄漏、编码对齐、DataLoader 行为。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

通过标准：实现 features.py 的三个函数后全部变绿。
骨架状态下应全部以 NotImplementedError 失败（而不是导入错误）。
"""

import numpy as np
import pytest
import torch

from features import (
    SPECIAL_TOKENS,
    build_dataloaders,
    build_vocab,
    encode,
    simple_tokenize,
)

CONFIG = {
    "data": {"val_size": 0.2, "random_state": 42, "train_subset": None},
    "features": {"max_len": 8, "vocab_size": 50, "min_freq": 1},
    "train": {"batch_size": 4},
}

TRAIN_TEXTS = [
    "the team won the championship game last night",
    "government election results shocked the nation today",
    "stocks rallied as the company reported record profit",
    "new smartphone chip is faster than the old one",
] * 3
TEST_TEXTS = [
    "quarterly earnings beat wall street expectations",
    "the striker scored two goals in the final",
]


# ---------- simple_tokenize（已实现，直接绿） ----------

def test_分词_已知值():
    assert simple_tokenize("Wall St. Bears Claw Back! (Reuters)") == \
        ["wall", "st", "bears", "claw", "back", "reuters"]
    assert simple_tokenize("iPhone 15 sold 1M units") == ["iphone", "15", "sold", "1m", "units"]


# ---------- build_vocab ----------

def test_词表_特殊token与编号():
    vocab = build_vocab(["the cat sat on the mat"], vocab_size=50, min_freq=1)
    assert vocab["[PAD]"] == 0 and vocab["[UNK]"] == 1 and vocab["[CLS]"] == 2
    assert vocab["the"] == 3          # 词频最高（出现 2 次）的普通词排第 3
    assert vocab["cat"] == 4          # 其余按 most_common 顺序（同频按首次出现序）


def test_词表_min_freq过滤():
    vocab = build_vocab(["common common common rare"], vocab_size=50, min_freq=2)
    assert "common" in vocab and "rare" not in vocab  # 词频 1 < min_freq=2


def test_词表_上限():
    vocab = build_vocab(["a b c d e f g h"], vocab_size=6, min_freq=1)
    assert len(vocab) == 6            # 3 特殊 + 3 普通词


def test_词表_只用传入的文本_测试集词汇进不来():
    """词表是训练段的训练产物：测试集专有的词必须查无此人（防泄漏）。"""
    train = ["football match on sunday"]
    test = ["quantum entanglement experiment"]
    vocab = build_vocab(train, vocab_size=50, min_freq=1)
    assert "quantum" not in vocab and "entanglement" not in vocab
    ids, _ = encode(test, vocab, max_len=6)
    # 测试集的新词全部落进 [UNK]=1（[CLS] 占第 0 位，后两位是 padding=0）
    assert (ids[0, 1:4] == 1).all() and (ids[0, 4:] == 0).all()


# ---------- encode ----------

def test_编码_形状与CLS句首():
    vocab = build_vocab(["hello world"], vocab_size=50, min_freq=1)
    ids, mask = encode(["hello world"], vocab, max_len=6)
    assert ids.shape == (1, 6) and mask.shape == (1, 6)
    assert ids[0, 0] == SPECIAL_TOKENS["[CLS]"]       # 句首恒为 [CLS]
    assert ids.dtype == np.int64


def test_编码_截断与补齐():
    vocab = build_vocab(["a b c d e f g"], vocab_size=50, min_freq=1)
    long_text = "a b c d e f g extra words beyond limit"
    ids, mask = encode([long_text], vocab, max_len=5)
    assert ids.shape == (1, 5)
    # max_len=5：[CLS] + 截断到 4 个词 → 全长无 padding，mask 全 False
    assert not mask.any()
    short_ids, short_mask = encode(["a"], vocab, max_len=5)
    assert short_mask[0].sum() == 3                      # [CLS] + 1 个词，其余 3 位全 pad


def test_编码_padding掩码与ids一致():
    vocab = build_vocab(["x y z"], vocab_size=50, min_freq=1)
    ids, mask = encode(["x y z"], vocab, max_len=6)
    assert np.array_equal(mask, ids == 0)                # True 恰好在 [PAD] 位置


# ---------- build_dataloaders（端到端） ----------

def test_build_dataloaders_形状与划分():
    loaders = build_dataloaders(TRAIN_TEXTS, [0, 1, 2, 3] * 3,
                                TEST_TEXTS, [2, 1], CONFIG)
    train_loader, val_loader, test_loader, vocab = loaders
    n_train = int(len(TRAIN_TEXTS) * 0.8)                # val_size=0.2
    assert len(train_loader.dataset) == n_train
    assert len(val_loader.dataset) == len(TRAIN_TEXTS) - n_train
    assert len(test_loader.dataset) == 2
    ids, mask, labels = next(iter(test_loader))
    assert ids.shape == (2, 8) and mask.shape == (2, 8) and labels.shape == (2,)
    assert ids.dtype == torch.int64 and mask.dtype == torch.bool


def test_build_dataloaders_val_test不shuffle():
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        TRAIN_TEXTS, [0, 1, 2, 3] * 3, TEST_TEXTS, [2, 1], CONFIG)
    b1 = [ids.clone() for ids, _, _ in test_loader]
    b2 = [ids.clone() for ids, _, _ in test_loader]
    for a, b in zip(b1, b2):
        assert torch.equal(a, b)                          # 两次迭代顺序一致


def test_build_dataloaders_词表只用训练段():
    """词表来自训练段的 train 部分；val/test 里的专有词不该在词表里。"""
    _, _, _, vocab = build_dataloaders(
        TRAIN_TEXTS, [0, 1, 2, 3] * 3,
        ["zebra zonkey zzzz"], [0], CONFIG)
    assert "zebra" not in vocab and "zzzz" not in vocab   # 测试集词汇进不来
    assert "championship" in vocab                        # 训练段词汇在
