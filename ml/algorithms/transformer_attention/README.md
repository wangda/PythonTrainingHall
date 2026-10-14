# 手写练习：Transformer 注意力机制

> 前置文档：`ml/docs/阶段二/13-Transformer注意力机制前置学习文档.md`（下称"文档"）。本练习的所有公式都在文档 §3~§6 里，这里只做导引，不重复讲解。

## 目标

仅用 NumPy，在 `attention.py` 中手写实现注意力五件套：

1. **causal_mask**（文档 §4）——因果掩码：下三角允许、上三角禁止（热身）
2. **scaled_dot_product_attention**（文档 §3/§4）——**核心中的核心**。`softmax(QKᵀ/√d_k)V`，支持掩码
3. **positional_encoding**（文档 §6）——正弦位置编码，broadcasting 向量化
4. **split_heads / combine_heads**（文档 §5）——多头拆分与合并，轴变换是全部难点
5. **multi_head_attention**（文档 §5）——组装件：投影 → 拆头 → 逐头注意力 → 合并 → 输出投影

对答案方式：**与 torch 同权重/同输入比较**——
- SDPA 对照 `torch.nn.functional.scaled_dot_product_attention`（实测最大绝对差 ≈ 4e-16）；
- MHA 对照 `torch.nn.MultiheadAttention`，参数装载用已实现完整的 `make_torch_mha`（实测 ≈ 2e-15）。

已实现完整、直接可用的辅助函数：
- `softmax(z, axis=-1)`：减最大值稳定版，模块 2/4 已练过，直接给。
- `make_demo_inputs(batch, seq_len, d_model, seed)`：固定 seed 的随机演示输入。
- `make_torch_mha(params, d_model, n_heads)`：对答案的桥，numpy 参数 → torch MHA（注意 in_proj 是 q,k,v 纵向拼接、每个矩阵要转置——docstring 里写清了）。

## 实现步骤建议

| 顺序 | 函数 | 对应文档 | 说明 |
| ---- | ---- | ---- | ---- |
| 1 | `causal_mask` | §4 | 一行 `np.tril`，热身 |
| 2 | `scaled_dot_product_attention` | §3/§4 | 核心。不超过 6 行；K 转置用 `swapaxes(-2, -1)`；mask 在 softmax **之前**置 -1e9 |
| 3 | 对答案 | §3 | `test_SDPA_对答案_随机输入对照torch` 绿了再继续；带因果 mask 再对一次 |
| 4 | `positional_encoding` | §6 | 禁止双重循环；偶数列 sin、奇数列 cos，`np.where` 一行收尾 |
| 5 | `split_heads` / `combine_heads` | §5 | reshape + transpose(0,2,1,3)；写错轴不报错只给错结果，用 arange 例子自查 |
| 6 | `multi_head_attention` | §5 | 纯组装，约 8 行，不写新数学 |
| 7 | 终验 | §5 | `test_MHA_对答案_对照torch`（n_heads=2 和 4 都要绿） |

## 通过标准

```bash
cd ml/algorithms/transformer_attention
../../../.venv/bin/python -m pytest test_attention.py -v
```

全部测试通过（绿）。骨架状态下 15 条测试以 `NotImplementedError` 失败、3 条 ✅ 辅助函数测试通过；每实现一个函数，对应的测试就转绿。

## 常见坑提示

- **除的是 √d_k，不是 √d_model**：多头场景下 d_k = d_model / n_heads。除错对象不报错，但对答案测试立刻红。
- **mask 加在 softmax 之前**：之后清零会破坏"行和为 1"的性质。置 -1e9 而不是 -inf（某行全被 mask 时 -inf 会 nan，模块 docstring 有取舍说明）。
- **K 的转置方向**：用 `K.swapaxes(-2, -1)`，只交换最后两维——三维、四维（多头）通吃。`transpose(0, 2, 1)` 只适用三维，进多头就崩。
- **split_heads 的 transpose 轴序**：`(b, l, n_heads, d_k) → transpose(0, 2, 1, 3) → (b, n_heads, l, d_k)`。轴序写错**不报错**、静默给错答案——测试里那条 arange 构造的 head 内容断言就是抓它的。
- **torch in_proj 的拼接顺序**：q、k、v 纵向拼成 (3*d_model, d_model)，且 torch Linear 按 `x @ Wᵀ + b` 算、numpy 侧按 `X @ W + b` 算，装载时每个矩阵都要转置——`make_torch_mha` 已替你处理，但值得读一遍源码。
- **浮点比较**：永远 `np.allclose`，不要 `==`（对答案容差 atol=1e-9，纯浮点累加顺序差异在 1e-15 量级）。

## 玩耍建议（测试之外）

- 可视化注意力权重：4 个词的句子过一遍 SDPA，`plt.imshow(weights[0])` 画 4×4 热力图，看看"哪个词关注哪个词"。
- 把 n_heads 从 1 调到 8，对同一份输入分别画出每个头的权重矩阵——不同头关注模式不同（文档 §5 的直觉实证）。
- 画位置编码热力图：`positional_encoding(64, 64)` 过 `plt.imshow`，看"位置指纹"的低频/高频条纹。
- 试试加 padding mask：把序列后几位 mask 掉，观察被 pad 位置的输出是否完全由真实 token 决定（这也是项目里 `src_key_padding_mask` 的原理）。
