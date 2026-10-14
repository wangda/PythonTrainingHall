"""文本特征工程（骨架）：原始新闻文本 → 词表 → 定长 id 序列 → 三个 DataLoader。

配套文档《14-文本分类项目文档.md》§3/§5；原理见前置文档 §7（Transformer Encoder 做分类）。

没有 transformers / torchtext——分词和词表自己实现，这恰是本项目的教学点：
文本的"特征工程"就是 分词 → 建词表 → 编码成 id 序列，每一步都有泄漏陷阱：

    ① build_vocab：词表只能用【训练段】的文本统计。测试集里的词混进词表，
       等于模型提前见过测试数据（虽然是间接的，性质和 StandardScaler 泄漏一样）；
    ② encode：每条文本变成长度固定 max_len 的 id 序列——句首 [CLS]、超出截断、
       不足补 [PAD]。padding 位置要生成 mask 传给模型（注意力不许关注空白，
       见 model.py 的 src_key_padding_mask）；
    ③ 预测时必须复现同一个词表——train.py 把 vocab.json 和模型一起保存，
       predict.py 直接加载它，绝不重新统计。

约定：
    - 特殊 token：[PAD]=0、[UNK]=1、[CLS]=2（普通词从 3 开始编号）；
    - [PAD]=0 让 padding 掩码可以直接用 ids != PAD 算出来；
    - Embedding 的 padding_idx=0 会把 [PAD] 的向量固定成全零。
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}


def simple_tokenize(text):
    """极简英文分词：小写 + 正则切词（已完整实现——它太简单，不值得你手写）。

    规则：转小写后，把连续的字母/数字段当作一个词（标点、空白都是分隔符）。
    AG News 是标准英文新闻，这个粒度够用；中文分词是另一门学问，不在此展开。

    参数:
        text: str，原始新闻文本（标题 + 摘要）。

    返回:
        list[str]，词序列。

    参考输出:
        simple_tokenize("Wall St. Bears Claw Back! (Reuters)") 
        → ['wall', 'st', 'bears', 'claw', 'back', 'reuters']
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(texts, vocab_size, min_freq):
    """从【训练段】文本统计词表（本项目防泄漏的关键函数）。

    参数:
        texts:      list[str]，训练段原始文本。
        vocab_size: int，词表上限（含 3 个特殊 token）。
        min_freq:   int，词频低于该值的词不进词表。

    返回:
        dict，word → id。[PAD]/[UNK]/[CLS] 占 0/1/2，普通词按词频从高到低
        从 3 开始编号，总数不超过 vocab_size。

    为什么只喂训练段:
        词表是"训练产物"。测试集的词混进词表 = 模型在训练时知道
        "测试集里会出现这些词"——评估虚高且毫无察觉（项目文档 §7.1）。
    """
    raise NotImplementedError


def encode(texts, vocab, max_len):
    """把文本列表编码成定长 id 矩阵 + padding 掩码。

    参数:
        texts:   list[str]，原始文本。
        vocab:   dict，build_vocab 的返回（或 vocab.json 加载回来的）。
        max_len: int，目标长度：句首 [CLS] 占 1 位，超出截断、不足补 [PAD]。

    返回:
        (ids, padding_mask):
            ids:          np.ndarray，(n, max_len)，int64。第 0 位恒为 [CLS]=2；
            padding_mask: np.ndarray，(n, max_len)，bool——True = 该位置是 [PAD]，
                          会被传给 TransformerEncoder 的 src_key_padding_mask
                          （True 表示"忽略我"，torch 的约定）。

    提示:
        - 词表里没有的词 → [UNK]=1；
        - 一行 np.full + 切片赋值就能写完，别写双重循环。
    """
    raise NotImplementedError


def build_dataloaders(train_texts, train_labels, test_texts, test_labels, config):
    """串起三步：切验证集 → 训练段建词表 → 编码 → DataLoader 三件套。

    参数:
        train_texts / train_labels: 训练段文本与标签（0~3）。
        test_texts  / test_labels:  官方测试集——只许评估，不许参与建词表。
        config: dict，读取 config["data"] 的 val_size / random_state / train_subset
                已在上游用掉，这里用不到），config["features"] 的 max_len /
                vocab_size / min_freq，config["train"] 的 batch_size。

    返回:
        (train_loader, val_loader, test_loader, vocab):
            前三个是 DataLoader，batch 是 (ids int64 (batch, max_len),
            padding_mask bool (batch, max_len), labels int64 (batch,))；
            vocab: dict，训练段词表（train.py 会存成 vocab.json）。

    正确顺序（顺序错了就是泄漏）:
        1. 用固定种子把训练段切成 train/val 两部分（val 从训练段来，
           绝不碰测试集——早停/checkpoint 的依据必须是"没见过的数据"）；
        2. build_vocab 只喂 train 部分；
        3. encode 三部分文本（同一个词表！）；
        4. DataLoader：train shuffle=True，val/test False。
    """
    raise NotImplementedError


def save_vocab(vocab, path):
    """词表存成 json（已完整实现——predict.py 靠它复现分词编码）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    print(f"词表已保存到 {path}（{len(vocab)} 个词）")


def load_vocab(path):
    """加载 vocab.json（已完整实现）。注意 json 的键是 str，id 要转回 int。"""
    with open(path, encoding="utf-8") as f:
        return {w: int(i) for w, i in json.load(f).items()}
