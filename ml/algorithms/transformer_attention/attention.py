"""Transformer 注意力手写练习骨架——阶段二模块 7（P1：Scaled Dot-Product Attention + Multi-Head）。

仅用 NumPy 实现：Scaled Dot-Product Attention（含掩码）、因果掩码、正弦位置编码、
多头拆分/合并、Multi-Head Attention 组装。
所有公式与前置文档《13-Transformer注意力机制前置学习文档.md》一一对应，docstring 中标注了章节。

架构约定（与文档 §3/§5 一致，全练习统一遵守）：
    - 全程 NumPy，张量形状用三维 (batch, seq_len, d) 维护（对齐 torch 的 batch_first 习惯）。
      split_heads 之后临时出现四维 (batch, n_heads, seq_len, d_k)，combine_heads 后回到三维。
    - scaled_dot_product_attention(Q, K, V, mask=None)：
          Q (b, lq, d_k)、K (b, lk, d_k)、V (b, lk, d_v)
          → 返回 (output (b, lq, d_v), weights (b, lq, lk))。
      mask 语义：可广播到 scores 形状 (b, lq, lk) 的布尔/0-1 数组，**0 / False 表示禁止关注**。
      实现上在 softmax 之前把被禁位置的分数置为 -1e9（不是 -inf：若某一行所有 key 全被 mask，
      -inf 会让 softmax 整行 nan；-1e9 下该行退化为均匀分布，虽无意义但至少不炸。本练习的
      掩码（因果、padding）每行至少保留一个 key，两种写法结果一致，选更稳的）。
    - multi_head_attention 是自注意力（Q/K/V 都来自同一个 X），params dict 约定键：
          "W_q", "W_k", "W_v", "W_o"：各 (d_model, d_model)；
          "b_q", "b_k", "b_v", "b_o"：各 (d_model,)。

使用流程：
    X = make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42)
    pe = positional_encoding(max_len=4, d_model=8)
    X = X + pe                                    # 注入位置信息（文档 §6）
    out = multi_head_attention(X, params, n_heads=2)
"""

import numpy as np
import torch


# ============================================================ 已实现完整的辅助函数（✅）


def softmax(z, axis=-1):
    """Softmax：把分数转成和为 1 的概率分布，已实现完整（模块 2/4 已练过，这里直接给出）。

    公式：softmax(z)ₖ = eᶻᵏ / Σⱼ eᶻʲ（沿 axis 独立计算）
    数值稳定处理已做好：先减去沿 axis 的最大值再取 exp（softmax 对同减常数不变，
    但能避免 np.exp(1000) 溢出成 inf）。

    与模块 2/4 的版本相比多了 axis 参数：注意力里 softmax 沿最后一维（key 维）做，
    三维/四维张量都用 axis=-1，默认参数即可。

    参数:
        z:    np.ndarray，任意形状，原始分数（logits）。
        axis: int，沿哪一维归一化，默认 -1（最后一维）。

    返回:
        np.ndarray，与 z 同形状，沿 axis 和为 1，值域 (0, 1)。

    参考输出:
        softmax([[1, 2, 3]]) ≈ [[0.0900, 0.2447, 0.6652]]。
    """
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_demo_inputs(batch=2, seq_len=4, d_model=8, seed=42):
    """固定 seed 的随机演示输入，已实现完整。

    参数:
        batch:   int，批大小，默认 2。
        seq_len: int，序列长度（一句话几个词），默认 4。
        d_model: int，模型维度（每个词的嵌入维度），默认 8。
        seed:    int，随机种子，固定后输入可复现（默认 42）。

    返回:
        np.ndarray，形状 (batch, seq_len, d_model)，float64，~ N(0, 1)。
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (batch, seq_len, d_model))


# ============================================================ 留给你手写（📝）


def causal_mask(seq_len):
    """因果掩码（下三角允许、上三角禁止），对应文档 §4。热身题。

    自回归场景（如 GPT 逐词生成）中，位置 i 的词只能看到 0..i 的词，
    不允许"偷看未来"。掩码 M 满足：M[i, j] = True（j <= i），False（j > i）。

    参数:
        seq_len: int，序列长度 l。

    返回:
        np.ndarray，形状 (seq_len, seq_len)，bool 类型，下三角（含对角线）为 True，
        严格上三角为 False。不带 batch 维——靠广播应用到 (b, l, l) 的 scores 上。

    参考输出（seq_len=3）:
        [[ True, False, False],
         [ True,  True, False],
         [ True,  True,  True]]

    实现提示:
        一行：np.tril(np.ones((seq_len, seq_len), dtype=bool))。
    """
    raise NotImplementedError


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention，对应文档 §3（公式）与 §4（掩码）。核心中的核心。

    公式：Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

        scores  = Q @ Kᵀ / √d_k        （每对 (query, key) 的原始相关度）
        weights = softmax(scores)       （沿最后一维归一化 → 每个 query 对各 key 的关注比例）
        output  = weights @ V           （按关注比例对 value 加权求和）

    为什么要除 √d_k：d_k 大时点积的方差随之变大，softmax 被推到饱和区（梯度趋近 0），
    除以 √d_k 把分数拉回 softmax 的敏感区间（文档 §3）。注意除的是 √d_k（每个头的维度），
    多头场景下 d_k = d_model / n_heads，不是 √d_model！

    掩码：mask 不为 None 时，在 softmax 之前把被禁位置的分数置为 -1e9（见模块 docstring
    的取舍说明）。掩码加在 softmax 之前，不是之后——之后再清零会破坏行和为 1 的性质。

    参数:
        Q:    np.ndarray，形状 (b, lq, d_k)，b 个序列、各 lq 个 query。
        K:    np.ndarray，形状 (b, lk, d_k)，各 lk 个 key（自注意力时 lq == lk）。
        V:    np.ndarray，形状 (b, lk, d_v)，各 lk 个 value（d_v 可与 d_k 不同）。
        mask: np.ndarray 或 None，可广播到 (b, lq, lk) 的布尔/0-1 数组，
              0 / False 表示禁止关注。None 表示不掩码。

    返回:
        (output, weights):
            output:  np.ndarray，形状 (b, lq, d_v)；
            weights: np.ndarray，形状 (b, lq, lk)，沿最后一维和为 1，
                     被 mask 的位置为 0。

    参考输出（手算对照，lq=lk=2, d_k=d_v=2）:
        Q = [[[1, 0], [0, 1]]]，K 同 Q，V = [[[1, 2], [3, 4]]]，无 mask：
        scores = QKᵀ/√2 ≈ [[0.7071, 0], [0, 0.7071]]
        weights ≈ [[0.6698, 0.3302],
                   [0.3302, 0.6698]]
        output  ≈ [[1.6605, 2.6605],
                   [2.3395, 3.3395]]

    实现提示:
        - K 转置用 K.swapaxes(-2, -1)：只交换最后两维，前面的 batch（以及 MHA 里的
          多头）维度不动，三维四维通吃。写 K.transpose(0, 2, 1) 只适用于三维。
        - softmax 直接用本文件 ✅ 版，沿 axis=-1。
        - 整个函数不超过 6 行。
    """
    raise NotImplementedError


def positional_encoding(max_len, d_model):
    """正弦位置编码，对应文档 §6。

    注意力本身对词序不敏感（打乱输入行的顺序，输出只是同样被打乱），
    必须把位置信息注入词向量。正弦编码直接按公式算出、无需学习：

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    偶数维用 sin、奇数维用 cos，同一对 (2i, 2i+1) 共用一个频率。
    每个位置得到一串独一无二的"位置指纹"，直接加到词嵌入上（X + PE）。

    参数:
        max_len: int，最大序列长度（行数）。
        d_model: int，模型维度（列数，练习中取偶数）。

    返回:
        np.ndarray，形状 (max_len, d_model)，float64。

    参考输出（max_len=4, d_model=8）:
        PE[0] = [0, 1, 0, 1, 0, 1, 0, 1]（pos=0：sin(0)=0、cos(0)=1）；
        PE[1, 0] = sin(1) ≈ 0.8415，PE[1, 1] = cos(1) ≈ 0.5403，
        PE[1, 2] = sin(1/10000^(2/8)) ≈ 0.0998。

    实现提示:
        - 禁止 Python 双重循环，用 broadcasting 一次算出：
          pos 列向量 (max_len, 1) 与维度行向量 (1, d_model) 相除得角度矩阵 (max_len, d_model)，
          再按列下标奇偶分别套 sin / cos（np.where 一行收尾）。
        - 指数是 2i/d_model，其中 i = 列下标 // 2（同一对奇偶列共用频率）。
    """
    raise NotImplementedError


def split_heads(X, n_heads):
    """把 d_model 维切成 n_heads 个头，对应文档 §5。

    多头的本质：不放大计算量，而是把 d_model 维均分成 n_heads 份，
    每个头在自己的 d_k = d_model / n_heads 维子空间里独立做注意力，
    各自捕捉不同类型的关联（有的头看语法、有的头看指代……）。

    轴变换（全部难点都在这里）：
        (b, l, d_model) --reshape--> (b, l, n_heads, d_k) --transpose--> (b, n_heads, l, d_k)
    reshape 把最后一维 d_model 拆成 (n_heads, d_k) 两维；
    transpose(0, 2, 1, 3) 把 n_heads 维提到 seq_len 前面，
    让每个头拿到完整的 (l, d_k) 序列，之后就能按三维注意力批量计算。

    参数:
        X:       np.ndarray，形状 (b, l, d_model)。
        n_heads: int，头数，必须整除 d_model。

    返回:
        np.ndarray，形状 (b, n_heads, l, d_k)，d_k = d_model // n_heads。

    参考行为:
        X = np.arange(1*2*8).reshape(1, 2, 8)，n_heads=2 →
        返回形状 (1, 2, 2, 4)，且 head 0 的内容是 X[..., 0:4]、head 1 是 X[..., 4:8]。

    实现提示:
        两行：reshape 后 transpose(0, 2, 1, 3)。transpose 的轴序写错不会报错，
        只会静默给出错误结果——用上面的 arange 例子自查。
    """
    raise NotImplementedError


def combine_heads(X):
    """split_heads 的逆操作：把多头输出拼回 d_model 维，对应文档 §5。

    轴变换（与 split_heads 正好相反）：
        (b, n_heads, l, d_k) --transpose--> (b, l, n_heads, d_k) --reshape--> (b, l, d_model)

    参数:
        X: np.ndarray，形状 (b, n_heads, l, d_k)。

    返回:
        np.ndarray，形状 (b, l, n_heads * d_k)。

    参考行为:
        对任意 X 与整除其最后一维的 n_heads：combine_heads(split_heads(X, n_heads)) == X。

    实现提示:
        两行：transpose(0, 2, 1, 3) 后 reshape。transpose 后内存不连续不影响 reshape
        （numpy 会自动处理），不需要手动 np.ascontiguousarray。
    """
    raise NotImplementedError


def multi_head_attention(X, params, n_heads):
    """Multi-Head Attention（自注意力），对应文档 §5。组装件：把前面几件拼起来。

    流程（自注意力：Q/K/V 都来自同一个 X）：
        1. 线性投影：Q = X @ W_q + b_q，K = X @ W_k + b_k，V = X @ W_v + b_v
           （形状均 (b, l, d_model)）；
        2. split_heads：Q/K/V 各拆成 (b, n_heads, l, d_k)；
        3. scaled_dot_product_attention：每个头独立做注意力（前四维批量算）；
        4. combine_heads：拼回 (b, l, d_model)；
        5. 输出投影：out = 拼接结果 @ W_o + b_o → (b, l, d_model)。

    参数:
        X:       np.ndarray，形状 (b, l, d_model)。
        params:  dict，键 "W_q"/"W_k"/"W_v"/"W_o"（各 (d_model, d_model)）
                 与 "b_q"/"b_k"/"b_v"/"b_o"（各 (d_model,)）。
        n_heads: int，头数，必须整除 d_model。

    返回:
        np.ndarray，形状 (b, l, d_model)。

    参考行为:
        与 torch.nn.MultiheadAttention（同权重，用 make_torch_mha 装载）输出一致，
        双精度下最大绝对差约 2e-15（纯浮点累加顺序差异）。

    实现提示:
        - 本函数不写新数学，只是按上面 5 步调用已写好的函数，约 8 行。
        - 第 3 步不需要 mask（自注意力全连接）；想加因果掩码就把 causal_mask(l)
          传给 scaled_dot_product_attention——(l, l) 会广播到 (b, n_heads, l, l)。
    """
    raise NotImplementedError


# ============================================================ 对答案工具（✅ 已实现完整）


def make_torch_mha(params, d_model, n_heads):
    """把 numpy 的 W_q/W_k/W_v/W_o 灌进 torch.nn.MultiheadAttention——对答案的桥，已实现完整。

    torch 侧的参数布局（大坑，装载顺序错一位答案就全对不上）：
        - in_proj_weight 形状 (3*d_model, d_model)，是 q、k、v 三个投影矩阵**按 q,k,v 顺序**
          纵向拼接成的一大块；in_proj_bias 同理，(3*d_model,)，q,k,v 拼接。
        - torch 的 Linear 按 y = x @ Wᵀ + b 计算（W 形状 (out, in)），
          而本练习 numpy 侧按 X @ W + b（W 形状 (d_model, d_model)），
          所以装载时每个矩阵都要**转置**。
        - out_proj 是普通的 (d_model, d_model) Linear，同样要转置。

    参数:
        params:  dict，multi_head_attention 同款参数（W_q/W_k/W_v/W_o 各 (d_model, d_model)，
                 b_q/b_k/b_v/b_o 各 (d_model,)），float64。
        d_model: int，模型维度。
        n_heads: int，头数。

    返回:
        torch.nn.MultiheadAttention（batch_first=True，float64），权重已装载，
        调用 mha(x, x, x) 即得自注意力输出（x 为 torch 张量 (b, l, d_model)）。

    用法:
        mha = make_torch_mha(params, d_model=8, n_heads=2)
        with torch.no_grad():
            out_torch, _ = mha(torch.from_numpy(X), torch.from_numpy(X), torch.from_numpy(X))
    """
    mha = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True).double()
    in_proj_w = np.concatenate([params["W_q"].T, params["W_k"].T, params["W_v"].T], axis=0)
    in_proj_b = np.concatenate([params["b_q"], params["b_k"], params["b_v"]])
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.from_numpy(in_proj_w))
        mha.in_proj_bias.copy_(torch.from_numpy(in_proj_b))
        mha.out_proj.weight.copy_(torch.from_numpy(params["W_o"].T))
        mha.out_proj.bias.copy_(torch.from_numpy(params["b_o"]))
    return mha
