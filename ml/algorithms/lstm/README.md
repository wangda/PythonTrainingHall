# 手写练习：LSTM（长短期记忆网络）

> 前置文档：`ml/docs/阶段二/11-LSTM与时间序列前置学习文档.md`（下称"文档"）。本练习的所有公式都在文档 §4 里，这里只做导引，不重复讲解。

## 目标

仅用 NumPy，在 `lstm_cell.py` 中手写实现 LSTM 的三件套：

1. **split_gates**（文档 §4）——把拼好的门向量切成 i, f, g, o 四段（热身）
2. **lstm_cell_forward**（文档 §4）——**核心**。单步前向：三门 + 细胞状态更新
3. **lstm_forward**（文档 §4/§5）——多步展开：逐时间步调单步前向，收集所有隐状态

对答案方式：**与 torch 同权重比较**——`make_torch_lstm_cell`（已实现完整）把你的 numpy 参数灌进 `torch.nn.LSTMCell`，你的手写输出与 torch 输出逐元素 allclose（实测偏差 1e-16 量级，纯 float64 浮点运算顺序差异）才算对。多步展开再与 `torch.nn.LSTM(batch_first=True)` 比全程 H 和最终 c_n。

已实现完整、直接可用的辅助函数：
- `sigmoid` / `tanh`：激活函数，模块 2/4 已练过，直接给。
- `init_params(input_size, hidden_size, seed)`：按 torch 默认初始化（均匀分布 U(−1/√hidden, 1/√hidden)）生成四个参数。
- `make_torch_lstm_cell(params, ...)`：对答案的桥，numpy 参数 → torch.nn.LSTMCell。
- `generate_demo_sequence(seed)`：200 步正弦+噪声合成序列，供多步展开演示。

## 实现步骤建议

| 顺序 | 函数 | 对应文档 | 说明 |
| ---- | ---- | ---- | ---- |
| 1 | `split_gates` | §4 | 热身。沿第 1 轴切 4 段，顺序 **i, f, g, o** |
| 2 | `lstm_cell_forward` | §4 | 核心。`z = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh`，切门，i/f/o 过 sigmoid、g 过 tanh，`c_t = f⊙c_prev + i⊙g`，`h_t = o⊙tanh(c_t)` |
| 3 | 对答案 | §4 | 跑 `test_单步前向_对答案torch`，绿了再继续 |
| 4 | `lstm_forward` | §4/§5 | h0/c0 从零初始化，逐步调用单步前向，H 用 `np.stack(list, axis=1)` 收集 |
| 5 | 对答案 | §5 | 跑 `test_多步展开_对答案torchLSTM`，H 全程 + c_n 全绿才算完 |

## 通过标准

```bash
cd ml/algorithms/lstm
../../../.venv/bin/python -m pytest test_lstm_cell.py -v
```

全部测试通过（绿）。骨架状态下 9 条测试以 `NotImplementedError` 失败、5 条 ✅ 辅助函数测试通过；每实现一个函数，对应的测试就转绿。

## 常见坑提示

- **门顺序是 i, f, g, o**：PyTorch 官方顺序是输入门、遗忘门、候选记忆 g̃、输出门。很多教材按 f, i, o, g 讲公式——顺序记错，对答案时一切全错。本练习（和 torch、和测试）一律 i, f, g, o。
- **[h, x] 拼接维度**：torch 把 W_ih 和 W_hh 分开存、分别与 x_t 和 h_prev 相乘再相加，等价于把 [x_t, h_prev] 拼起来乘一个大矩阵。本练习对齐 torch 的分开写法：`x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh`，别自己拼出一个 (batch, input+hidden) 再对不上权重形状。
- **sigmoid 管门，tanh 管内容**：i, f, o 三道门过 sigmoid（输出 0~1，是"开度"）；候选记忆 g̃ 和最后的 tanh(c_t) 过 tanh（输出 −1~1，是"内容"）。串了激活函数，数值不会报错但答案全错。
- **上一步的 (h, c) 都要传**：多步展开时，上一步输出的 h_t **和** c_t 都是下一步的输入。只传 h 不传 c（比如每步把 c 重置为零），单步测试照样绿、多步测试立刻红——这正是 `test_多步展开_对答案torchLSTM` 存在的原因。
- **W_ih 的转置方向**：W_ih 形状是 (4*hidden, input)，与 x_t (batch, input) 相乘要写成 `x_t @ W_ih.T`。写反了形状对不上会直接报错（好事），但 W_hh 是方阵 (4*hidden, hidden)，写反了**不报错**只给错答案——对答案测试会抓住它。

## 玩耍建议（测试之外）

- 在 `generate_demo_sequence()` 上跑多步展开：`seq[np.newaxis]` 加成 (1, 200, 1) 喂给 `lstm_forward`，画 H 里某几个隐藏单元随时间的变化曲线，看看随机权重的 LSTM "关注"什么。
- 调 f 门 bias 看"记忆长度"：把 `params["b_ih"][hidden:2*hidden]` 整体加大（f≈1，健忘变长情）或调小（f≈0，每步清零重来），对比同一条序列上 H 的变化幅度。
- 和 torch 对完答案后，试试 `torch.nn.LSTM(..., num_layers=2, batch_first=True)` 的形状：output 还是 (batch, seq_len, hidden)，但 h_n/c_n 变成 (2, batch, hidden)——想想第二层吃什么输入。
- 故意制造 bug 再抓 bug：把门顺序改成 f, i, g, o，或把 W_hh 的转置写反（方阵不报错！），看对答案测试怎么报警——体会"同权重对答案是 LSTM 的照妖镜"。
