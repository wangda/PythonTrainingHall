# 13 - Transformer 注意力机制前置学习文档（教程 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：从 RNN 到注意力、Q/K/V 直觉、Scaled Dot-Product Attention（手写对象）、padding/因果掩码、Multi-Head Attention、位置编码、PyTorch 搭 Transformer Encoder 做分类。
> **本文环境**：NumPy 2.2 + PyTorch 2.14（所有示例已实际运行验证，输出逐项核对）。
> **模块定位**：P1——手写 Scaled Dot-Product Attention 与 Multi-Head（`algorithms/transformer_attention/`），Encoder 调库（不手写完整 Encoder）。

---

## 本文怎么用

1. **当教程读**：从头读一遍，每个代码块亲手敲一遍跑一遍。§3（SDPA）是本模块的心脏，务必手算一遍。
2. **当速查手册用**：写练习和项目时忘了公式或 API，直接查文末【完整速查表】。

**先决知识**（直接引用，不再重复讲）：

- `11-LSTM与时间序列前置学习文档.md`（下称"模块 6 文档"）：RNN/LSTM 的串行结构、长程依赖的代价——本文的动机全部由此而来；
- `07-MLP与反向传播前置学习文档.md`：softmax（模块 2 也讲过）、训练循环五步法（§7）；
- `03-逻辑回归与分类前置学习文档.md`：softmax 与交叉熵。

**阅读约定**：🟢 必须掌握（手写练习直接考）；🟡 了解即可。

**贯穿全文的例子**：一个四个词的迷你句子 **"我 爱 机器 学习"**。每个词先随便给一个 8 维向量（假装是"词义"，真词向量是模块 7 项目的 Embedding 学出来的），全程用它把注意力权重矩阵（4×4）算出来、看清楚"哪个词关注哪个词"：

```python
import numpy as np

words = ["我", "爱", "机器", "学习"]
rng = np.random.default_rng(42)
X = rng.normal(0, 1, (1, 4, 8))       # (batch=1, seq_len=4, d_model=8)
print(X[0, 0, :4].round(3))           # [ 0.305 -1.04   0.75   0.941]
```

形状约定全文统一：`(batch, seq_len, d_model)` 三维——和 torch 的 `batch_first=True` 习惯一致（模块 6 文档 §5 踩过的坑这里提前规避）。

---

## 1. 从 RNN 到注意力：为什么需要注意力 🟢

### 1.1 RNN/LSTM 的两个死穴（承接模块 6）

模块 6 文档讲透了 LSTM 如何用门控缓解梯度消失，但两个结构性问题门也救不了：

1. **必须串行**：RNN/LSTM 逐时间步读序列，第 t 步必须等第 t−1 步算完——序列长 1000 就要串行 1000 步。GPU 的强项是并行，这里全用不上，**训练速度被串行锁死**；
2. **长程依赖靠"接力传话"**：位置 1 的信息要影响位置 500，必须经过 499 次隐状态传递。LSTM 的门控让传递"不那么容易忘"，但路径长度不变——距离越远，信息衰减越厉害。

### 1.2 注意力的破局思想：开卷考试

注意力的思路换了个赛道：**不搞"逐句背诵"（压缩成隐状态），直接"开卷考试"——要回答某个位置的问题时，把整个序列摊开，自己挑相关的看**。

| | RNN/LSTM | 注意力 |
|---|---|---|
| 读序列方式 | 逐位置串行，维护一个"记忆" | 所有位置**同时**互相看（并行！） |
| 远距离信息路径 | O(距离) 次传递 | **O(1)**：一步直达 |
| 计算复杂度 | O(seq_len) 串行步数 | O(seq_len²) 一次算完（平方贵，但能并行） |

"任意两个位置一步直达"就是**长程依赖的解药**，"所有位置同时算"就是**并行化的钥匙**。Transformer（2017，Attention Is All You Need）= 堆叠的注意力层 + 前馈层，完全没有循环。

**本节小结**：RNN 的串行与长程依赖是结构性死穴；注意力让每个位置直接看所有位置——一步直达 + 天然并行，代价是 seq² 的计算量。

---

## 2. 注意力的直觉：Query / Key / Value 🟢

### 2.1 数据库检索类比

把注意力想成一次**模糊检索**。数据库的三个角色：

| 角色 | 数据库里是什么 | 注意力里是什么 | 形状 |
|---|---|---|---|
| **Query（查询）** | 你在搜索框里敲的关键词 | "我（当前位置）想找什么信息" | 每个位置一个，(seq, d_k) |
| **Key（键）** | 每条记录的索引字段 | "我（每个位置）能提供什么线索" | (seq, d_k) |
| **Value（值）** | 每条记录的实际内容 | "匹配上了，我实际给你这些内容" | (seq, d_v) |

检索过程：Q 和每个 K **算相似度**（点积）→ 相似度过 softmax 变成"关注度权重"（和为 1）→ 按权重**对 V 加权求和**——匹配度越高，那条记录的内容占比越大。模糊就模糊在：不是返回一条记录，而是**按相似度配比所有记录**。

### 2.2 自注意力：自己查自己

**自注意力（self-attention）= Q、K、V 全部来自同一个序列**：序列里的每个位置都发起一次查询，检索的对象是包括自己在内的全部位置。"我"这个位置的 Query 和"机器"的 Key 点积大，说明"我"需要重点参考"机器"的信息。

**本节小结**：Q=我想找什么、K=我的索引、V=我的内容；点积算相似度 → softmax 成权重 → 加权求和 V。自注意力 = Q/K/V 同源，序列内部互相检索。

---

## 3. Scaled Dot-Product Attention 🟢（核心，手写对象）

### 3.1 公式逐符号翻译

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

| 符号 | 含义 | 形状（batch 维省略） |
|---|---|---|
| Q | 查询矩阵 | (lq, d_k) |
| K | 键矩阵 | (lk, d_k) |
| Kᵀ | K 的转置 | (d_k, lk) |
| **QKᵀ** | 所有两两点积 = 相似度分数矩阵 | (lq, lk) |
| **√d_k** | 缩放因子（d_k 是 Q/K 的维度） | 标量 |
| softmax(·) | 沿最后一维（key 维）归一化 → 权重，行和为 1 | (lq, lk) |
| **· V** | 按权重对 value 加权求和 | (lq, d_v) |

三步流程：**算分（QKᵀ）→ 归一（softmax，行和为 1）→ 汇总（加权平均 V）**。权重矩阵 (lq, lk) 的第 i 行第 j 列 = "第 i 个位置对第 j 个位置的关注度"。

### 3.2 裸自注意力跑一遍（贯穿例子）

先不搞投影矩阵，直接拿词向量 X 同时当 Q、K、V（最裸的自注意力）：

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)      # 数值稳定（模块 2 的老规矩）
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def sdpa(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)   # (b, lq, lk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)        # §4：禁看的置 -1e9
    w = softmax(scores)                              # 行和为 1 的关注度
    return w @ V, w

out, w = sdpa(X, X, X)        # X 就是 §0 的四词句
print(w[0].round(3))          # 4×4 注意力权重矩阵
# [[0.77  0.06  0.128 0.042]
#  [0.135 0.439 0.159 0.266]
#  [0.268 0.147 0.339 0.246]
#  [0.069 0.194 0.195 0.542]]
print(w.sum(-1).round(3))     # [[1. 1. 1. 1.]] —— 每行和为 1
```

读这张 4×4 表：第 0 行是"我"对四个词的关注度 [0.77, 0.06, 0.128, 0.042]——随机向量下它主要关注自己（0.77），这没语义，**但读表的方法是通用的**：真实训练后，"它"的行会在"机器"列上出现高权重（指代消解），这就是注意力可解释性的由来。

### 3.3 为什么要除 √d_k（高频面试题，实测说话）

点积的量级随维度增长：两个 d 维随机向量的点积，标准差约 √d。d_k=512 时点积分数动辄 ±30——softmax 对大分数极端敏感，会**饱和**成 one-hot（一个 1 其余全 0），梯度几乎为零（模块 2 讲过的 softmax 饱和问题）。除以 √d_k 把分数拉回 softmax 的敏感区间。实测（4 个 key，随机向量）：

| d_k | 点积分数 std（不缩放） | 缩放后 std | softmax 最大权重：不缩放 vs 缩放 |
|---|---|---|---|
| 8 | 1.2 | 0.43 | 0.758 vs **0.442** |
| 512 | 30.0 | 1.32 | **1.000（完全饱和）** vs 0.778 |

不缩放的 d_k=512 直接饱和成 one-hot（最大权重=1.000）。**注意除的是 √d_k，不是 √d_model**——多头场景下每个头的 d_k = d_model / n_heads（§5），这是手写练习的第一大坑。

### 3.4 手算一遍（两个位置的最小例子）

```python
Q = np.array([[[1.0, 0.0], [0.0, 1.0]]])   # 两个 query（d_k=2）
V = np.array([[[1.0, 2.0], [3.0, 4.0]]])   # 两个 value
out, w = sdpa(Q, Q, V)                      # K = Q
print(w.round(6))
# [[[0.669762 0.330238]
#   [0.330238 0.669762]]]      scores = QKᵀ/√2 = [[0.707, 0], [0, 0.707]] → softmax
print(out.round(6))
# [[[1.660477 2.660477]
#   [2.339523 3.339523]]]     output = w @ V，第一行 = 0.6698·[1,2] + 0.3302·[3,4]
```

手算核对第一行：`0.6698×1 + 0.3302×3 = 1.6605`，`0.6698×2 + 0.3302×4 = 2.6605`——公式和数字严丝合缝。**手写练习的测试就埋着这道题**。

**本节小结**：SDPA 三步 = QKᵀ 算分 → 除 √d_k 防饱和 → softmax 行归一 → 加权求和 V。手写约 6 行 numpy。

---

## 4. 掩码：padding mask 与因果 mask 🟢

### 4.1 机制：softmax 前把禁看位置置 −∞

掩码不是新公式，是一个小技巧：**在 softmax 之前**把不许关注的位置的分数设成 −∞，`exp(−∞)=0`，权重自然为零。实测：

```python
print(softmax(np.array([[1.0, 2.0, 3.0]])).round(4))        # [[0.09   0.2447 0.6652]]
print(softmax(np.array([[1.0, 2.0, -1e9]])).round(4))       # [[0.2689 0.7311 0.    ]]
```

第 3 位权重精确变 0，其余两位的相对比例不变（[0.269 : 0.731] = [1 : 2.72]）。**掩码必须加在 softmax 之前**——softmax 之后强行清零会破坏"行和为 1"。工程上用 −1e9 代替 −inf（−1e9 过 softmax 同样精确得 0，且避免某行全被 mask 时 −inf 产生 nan——本练习的约定，docstring 里有取舍说明）。

### 4.2 padding mask：空白位置不许看

batch 内句子长短不一，短句补 [PAD]（模块 7 项目）。[PAD] 是无意义占位——如果不加掩码，每个词的注意力会"分心"去看空白，句子被污染。padding mask 把所有 [PAD] 位置的 **key** 列禁掉（每行的行和仍为 1，只是在真实词上重新分配）。这就是项目里 `src_key_padding_mask` 的原理（True = 忽略，torch 的约定）。

### 4.3 因果 mask：不许偷看未来

自回归生成（GPT 逐词预测下一个词）的铁律：**位置 i 只能看 0..i，不许看 i+1..**（考场上不许翻后面的考卷）。实现：把权重矩阵的**严格上三角**禁掉。

```python
causal = np.tril(np.ones((4, 4), dtype=bool))    # 下三角 True = 允许
_, w_c = sdpa(X, X, X, mask=causal)
print(w_c[0, 2].round(3))                        # 第 3 个词的行：[0.355 0.194 0.45  0.  ]
print(bool((np.triu(w_c[0], k=1) == 0).all()))   # True —— 严格上三角全零
```

位置 2 对位置 3 的权重精确为 0；它把原来分给"未来"的关注度**重新分配**给了允许的位置（行和仍是 1）。没有因果 mask，"预测下一个词"就是抄答案——模型能学但完全作弊。

**本节小结**：掩码 = softmax 前置 −1e9。padding mask 禁 [PAD] 的 key 列；因果 mask 禁严格上三角（不许偷看未来）。

---

## 5. Multi-Head Attention 🟢

### 5.1 为什么一头不够

一组 Q/K/V 只能学出**一种**关注模式（一张 4×4 权重表）。但语言里的关联是多层次的："它"既在语法上指代前文某名词（一头该管这个），又可能与远处的语义呼应（另一头管这个）。**多头 = 并排开多组独立的检索系统，各学各的关注模式，最后把各组结果拼起来**。

### 5.2 结构与形状账

关键设计：**d_model 不变，每个头在 d_model/n_heads 维的子空间里工作**——总计算量几乎不变：

```
d_model = 512, n_heads = 8 → 每头 d_k = 64
Q = X·W_q  (l, 512) → split_heads → 8 个头各 (l, 64)
每头独立 SDPA（用自己的 64 维空间算相似度）
→ 拼回 (l, 512) → 输出投影 W_o
```

```python
def split_heads(Xm, n_heads):
    b, l, d = Xm.shape
    return Xm.reshape(b, l, n_heads, d // n_heads).transpose(0, 2, 1, 3)  # (b, h, l, d_k)

def combine_heads(Xm):
    b, h, l, dk = Xm.shape
    return Xm.transpose(0, 2, 1, 3).reshape(b, l, h * dk)                 # 逆操作

Xh = np.arange(1*4*8).reshape(1, 4, 8)
heads = split_heads(Xh, 2)
print(heads.shape)                                   # (1, 2, 4, 4) —— d_k = 4
print(np.array_equal(heads[:, 0], Xh[..., 0:4]))     # True：head0 恰是前 4 维切片
print(np.array_equal(combine_heads(heads), Xh))      # True：拆了能拼回去
```

**轴变换是手写练习的全部难点**：`(b, l, d_model) → reshape → (b, l, h, d_k) → transpose(0,2,1,3) → (b, h, l, d_k)`。写错轴不报错、只给错答案——用 arange 例子自查。

### 5.3 多头自注意力组装（练习的最终形态）

```python
p = np.random.default_rng(1)
P = {k: p.normal(0, 0.5, (8, 8)) for k in ["Wq", "Wk", "Wv", "Wo"]}
Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]      # 三个投影（这就是"参数"所在）
outs = []
for h in range(2):                                    # 每个头独立做 SDPA
    o_h, _ = sdpa(split_heads(Q, 2)[:, h], split_heads(K, 2)[:, h], split_heads(V, 2)[:, h])
    outs.append(o_h)
mha_out = np.concatenate(outs, axis=-1) @ P["Wo"]     # 拼接 + 输出投影
print(mha_out.shape)                                  # (1, 4, 8)
print(mha_out[0, 0, :3].round(3))                     # [-1.018  1.468 -0.839]
```

流程五步：投影 → 拆头 → 逐头 SDPA → 拼接 → 输出投影。**对答案**：把同一组参数灌进 `torch.nn.MultiheadAttention`（练习的 `make_torch_mha` 已实现装载），输出逐元素 allclose（实测偏差 2e-15 量级）。

**本节小结**：多头 = 把 d_model 切成 h 份子空间各学一种关注模式，总计算量不变；五步组装 + 轴变换是手写核心；对答案靠同权重灌进 torch MHA。

---

## 6. 位置编码 🟢

### 6.1 注意力天生不知道词序（实测置换等变性）

SDPA 里每个位置的 Query 只通过**内容**（点积）匹配 Key——把输入的顺序打乱，输出只是跟着打乱，内容一模一样。实测：

```python
out_raw, _ = sdpa(X, X, X)
perm = np.array([2, 0, 3, 1])                          # 打乱顺序
out_shuf, _ = sdpa(X[:, perm], X[:, perm], X[:, perm])
print(np.allclose(out_raw[:, perm], out_shuf[0][None]))   # True —— 置换等变
```

**注意力是"集合"运算，不是"序列"运算**。但"我 爱 机器 学习"和"学习 机器 爱 我"意思完全不同——必须把位置信息**注入**词向量。

### 6.2 正弦位置编码

原始论文的方案：给每个位置生成一行独一无二的"位置指纹"，直接**加**到词向量上：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**逐符号翻译**：`pos` 是位置编号（第几个词），`i` 是维度对的编号，偶数维用 sin、奇数维用 cos，同一对 (2i, 2i+1) 共用一个频率。不同维度对的频率从 1 到 1/10000 指数递减——**低频维像"个位数"、高频维像"千位数"**，组合起来每个位置得到一串独一无二的编码，类似二进制计数器的实数版。

```python
def positional_encoding(max_len, d_model):
    pos = np.arange(max_len, dtype=float).reshape(-1, 1)
    i = np.arange(d_model) // 2                       # 同一对奇偶列共用频率
    angles = pos / 10000 ** (2 * i / d_model)
    return np.where(np.arange(d_model) % 2 == 0, np.sin(angles), np.cos(angles))

pe = positional_encoding(4, 8)
print(pe[0].round(1))        # [0. 1. 0. 1. 0. 1. 0. 1.]  （pos=0：sin(0)=0, cos(0)=1）
print(pe[1, 0].round(4), pe[1, 1].round(4), pe[1, 2].round(4))
# 0.8415 0.5403 0.0998       （PE(1,·) 逐维的指纹）
print(all(not np.allclose(pe[i], pe[j]) for i in range(4) for j in range(i+1, 4)))
# True —— 每个位置的指纹独一无二
```

一句话记住变体：**正弦编码（公式算出、不用学、能外推更长序列）** vs **可学习位置编码（一个 nn.Parameter，跟着训练学）**——两者效果相近，项目里用的是后者（一行 `nn.Parameter`），练习里手写的是前者。

**本节小结**：注意力置换等变（实测）→ 必须注入位置；正弦 PE = 不同频率 sin/cos 的位置指纹，加进词向量。

---

## 7. PyTorch 搭 Transformer Encoder 做分类 🟢

### 7.1 组件全家福

一个分类用的 Transformer Encoder 从下到上：

```
词 id → Embedding(vocab_size, d_model)      词 → 向量（可学习）
      → + 位置编码                           §6：正弦或可学习
      → nn.TransformerEncoder（× num_layers） 每层 = 自注意力 + 前馈网络（FFN）
      → 取 [CLS] 位（或均值池化）              全文摘要向量
      → Linear(d_model, num_classes)          分类头（裸 logits，交叉熵内部含 softmax）
```

两个一句话概念：**残差连接**（每个子层的输出 = 子层输入 + 子层结果，信息有高速公路，深网络才训得动）与 **LayerNorm**（每个位置的向量归一化，稳定训练）——它们都在 `nn.TransformerEncoderLayer` 内部打包好了，知道存在即可。

### 7.2 完整实战：合成序列分类

任务：序列里含特殊 token id=7 → 正类，否则负类（2000 条训练 / 1000 验证，秒级）：

```python
import torch
from torch import nn
torch.manual_seed(42)

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20, 32, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, 16, 32))    # 可学习位置编码
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(32, 2)
    def forward(self, ids, mask=None):
        x = self.emb(ids) + self.pos[:, :ids.shape[1]]
        x = self.enc(x, src_key_padding_mask=mask)         # padding 掩码传这里
        return self.head(x[:, 0, :])                       # [CLS] 位

# 训练循环五步法（模块 4 文档 §7），Adam(lr=1e-3)，15 epoch
# 实测 train_loss: 0.5314 → 0.2426(ep3) → 0.0057(ep6) → 0.0017(ep14)
# 验证集准确率: 1.0 —— "找特殊 token"恰好是注意力最擅长的模式
```

三个必记的调用细节：

1. **`batch_first=True` 显式写**（模块 6 同款坑，EncoderLayer 同样默认 False）；
2. **`src_key_padding_mask` 传 bool 掩码，True = 忽略**——模块 7 项目里 `features.encode` 产出的掩码直接喂这里；
3. **[CLS] 位**：句首放一个特殊 token，它的输出"看得到全文"，被训练成摘要向量（BERT 同款）；均值池化（`x.mean(dim=1)` 带 mask）是等价替代。

**本节小结**：Encoder 分类器五层积木（Embedding → 位置 → Encoder → [CLS]/池化 → Linear）；batch_first、padding mask、[CLS] 三个调用细节刻死。

---

## 8. 手写练习预告

接下来在 `ml/algorithms/transformer_attention/` 里仅用 NumPy 手写五件套（本文公式就是设计图）：

| 要手写的函数 | 对应本文 | 核心一行代码 |
|---|---|---|
| `causal_mask` | §4.3 | `np.tril(np.ones((l, l), dtype=bool))` |
| `scaled_dot_product_attention` | §3 | `softmax(Q@K.swapaxes(-2,-1)/√d_k) @ V` |
| `positional_encoding` | §6.2 | broadcasting + `np.where(偶列, sin, cos)` |
| `split_heads / combine_heads` | §5.2 | `reshape + transpose(0,2,1,3)` 与其逆 |
| `multi_head_attention` | §5.3 | 五步组装（不写新数学） |

对答案方式：SDPA 对照 `torch.nn.functional.scaled_dot_product_attention`（实测偏差 4e-16）；MHA 同权重灌进 `torch.nn.MultiheadAttention`（练习的 `make_torch_mha` 已实现装载，实测偏差 2e-15）。通过标准 = 测试全绿。

---

## 完整速查表

**概念**：注意力 = 开卷考试（一步直达 + 并行）｜ Q 查询 / K 索引 / V 内容 ｜ 自注意力 = QKV 同源 ｜ SDPA 三步 = 算分 → 防饱和（÷√d_k）→ 归一加权 ｜ 掩码 = softmax 前置 −1e9 ｜ 多头 = 子空间各学一种关注模式 ｜ 注意力置换等变 → 必须注入位置

**公式**：

| 名称 | 公式 | 一句话 |
|---|---|---|
| SDPA | **softmax(QKᵀ/√d_k)·V** | 模糊检索三步曲 |
| 因果掩码 | 权重严格上三角 = 0 | 不许偷看未来 |
| padding 掩码 | [PAD] 的 key 列权重 = 0 | 空白不许看 |
| 多头 | concat(head₁..head_h)·W_o，每头 d_k=d_model/h | 子空间并行检索 |
| 正弦 PE | PE(pos,2i)=sin(pos/10000^(2i/d))，奇列 cos | 位置指纹 |

**代码**（PyTorch）：

| 需求 | 写法 |
|---|---|
| 手写 SDPA 对照 | `F.scaled_dot_product_attention(Q, K, V, attn_mask=...)` |
| 手写 MHA 对照 | `nn.MultiheadAttention(d_model, nhead, batch_first=True)`（in_proj 是 q,k,v 纵向拼接） |
| Encoder 层 | `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)` |
| Encoder 堆叠 | `nn.TransformerEncoder(layer, num_layers=N)` |
| padding 掩码 | `encoder(x, src_key_padding_mask=mask)`（True = 忽略） |
| 位置编码 | 正弦（buffer）或 `nn.Parameter`（可学习） |
| [CLS] 分类头 | `head(x[:, 0, :])` |

---

## 自测题（附答案）

1. **（动机）** RNN/LSTM 的两个结构性死穴是什么？注意力分别怎么解决？
2. **（Q/K/V）** 用数据库检索类比说清 Q、K、V 各是什么。"自注意力"特殊在哪？
3. **（形状账）** Q (4, 8)、K (4, 8)、V (4, 8) 做 SDPA（无掩码），QKᵀ、权重、输出各是什么形状？
4. **（√d_k）** 为什么要除 √d_k？除的是 √d_k 还是 √d_model？多头时 d_k 怎么算？
5. **（掩码）** 因果掩码 mask 掉的是权重矩阵的哪部分？掩码为什么必须加在 softmax 之前？
6. **（多头）** d_model=512、n_heads=8 时每头的 d_k 是多少？为什么多头几乎不增加计算量？
7. **（位置编码）** 注意力为什么是"集合运算"？实测怎么验证？正弦 PE 和可学习 PE 的取舍？
8. **（分类头）** [CLS] 位的输出为什么能当"全文摘要"？等价替代是什么？

### 参考答案

**1.** ①串行计算（第 t 步等第 t−1 步）→ 注意力全位置同时算，并行；②长程依赖靠隐状态接力传递，距离越远衰减越厉害 → 注意力任意两位置一步直达（O(1) 路径）。

**2.** Q=搜索关键词（当前位置想找什么）、K=记录索引（每个位置能提供什么线索）、V=记录内容（匹配上实际给什么）；点积算相似度、softmax 配比、加权求和 V。自注意力 = Q/K/V 都来自同一序列，序列内部互相检索。

**3.** QKᵀ (4,4)；softmax 后权重 (4,4) 行和为 1；输出 (4,8)。

**4.** 点积方差随维度增长（std≈√d），大分数使 softmax 饱和成 one-hot、梯度归零；除 √d_k 拉回敏感区间（实测 d_k=512 不缩放时最大权重=1.000 完全饱和）。除的是 √d_k；多头时 d_k = d_model / n_heads（512/8=64）。

**5.** 严格上三角（位置 i 禁看 j>i）。softmax 之后清零会破坏行和为 1 的归一化性质；之前置 −1e9 则 exp 后精确为 0、行和在允许位置上重新归一。

**6.** 64。把 d_model 均分成 h 份，每个头在自己的子空间里做注意力，总维度不变（参数量与单头 full 维相当），只是多了一次 reshape/transpose 和输出投影。

**7.** 每个位置的输出只由内容点积决定，与位置编号无关——打乱输入行，输出同样打乱（实测 `allclose(out[:, perm], out_shuffled)` 为 True）。正弦 PE 免训练、可外推更长序列；可学习 PE 是 nn.Parameter、随任务学、不能超过训练长度。效果相近。

**8.** [CLS] 作为序列第一个 token，它的 Query 会关注全文所有位置，训练后其输出向量被塑造成"全文摘要"，直接接分类头（BERT 同款）。等价替代：对全部位置做带 padding mask 的均值池化。

---

## 与后续模块的关系（学习路径对应表）

| 本文概念 | 后面哪里复用 |
|---|---|
| SDPA + 多头 | 一切 Transformer 架构（BERT/GPT/LLM）的唯一注意力原语 |
| 因果掩码 | GPT 式自回归生成的核心机制 |
| [CLS] 分类头 | BERT 微调做分类的标准姿势 |
| Encoder 堆叠 + 残差 + LayerNorm | 深度 Transformer 的积木（层数 × 宽度的缩放律） |
| pytorch 训练循环五步法 | 模块 4 文档 §7 原封不动，最后一次重复 |

**直接衔接**：本文 → 在 `algorithms/transformer_attention/` 手写注意力五件套并与 torch 对答案（§8）→ 项目 `projects/text_classification/`（AG News 真实新闻，自己实现分词/词表/padding mask，走完"文本 → 定长 id 序列 → Encoder → 分类"全流程）。
