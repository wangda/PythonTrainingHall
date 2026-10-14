# 手写练习：逻辑回归与 softmax 多分类

> 前置文档：`ml/docs/阶段二/03-逻辑回归与分类前置学习文档.md`（下称"文档"）。本练习的所有公式都在文档里，这里只做导引，不重复讲解。

## 目标

仅用 NumPy，在 `logistic_regression.py` 中手写实现：

**二分类主线**
1. **sigmoid**（文档 §2）——注意数值溢出
2. **交叉熵损失**（文档 §3）
3. **梯度下降**（文档 §4）——核心，梯度与线性回归同形：`(1/n)·Xᵀ(p-y)`
4. **predict_proba / predict**（文档 §1 / §6）——概率输出与可配阈值
5. **评估指标**：accuracy / precision / recall / F1 / 混淆矩阵（文档 §7）

**多分类扩展**
6. **one_hot / softmax / softmax_cross_entropy / softmax_gradient_descent**（文档 §5）

最后用 sklearn（文档 §8）对答案：手写结果与 sklearn 结果 `np.allclose` 为 True。

## 实现步骤建议

| 顺序 | 函数 | 对应文档 | 说明 |
| ---- | ---- | ---- | ---- |
| 1 | `sigmoid` | §2 | 热身。先写 naive 版跑过已知值测试，再处理 ±1000 不溢出 |
| 2 | `cross_entropy` | §3 | 一行均值公式，先记得 clip 防 log(0) |
| 3 | `predict_proba` / `predict` | §1/§6 | 各一行：`σ(X_b @ θ)`、按阈值切 0/1 |
| 4 | 四个指标 + `confusion_matrix` | §7 | 数 TP/FP/FN/TN，四个指标全部对照 sklearn.metrics |
| 5 | `gradient_descent` | §4 | 核心。梯度 `(1/n)·X_bᵀ(p-y)`，循环更新并记录 loss 历史 |
| 6 | `one_hot` / `softmax` | §5 | softmax 必须先减行最大值，否则大 logits 全溢出 |
| 7 | `softmax_cross_entropy` | §5 | one-hot 下每行只剩 `-log P[真实类]` |
| 8 | `softmax_gradient_descent` | §5 | 参数是矩阵 Θ(d,K)，梯度 `(1/n)·X_bᵀ(P-Y)`，和二分类同一套骨架 |
| 9 | 对答案 | §8 | 跑测试，和 sklearn 对照 |

数据生成器都已实现完整，直接用：
- `generate_sample_data(seed=42)`：文档 §1 贯穿例子"考试通过预测"，200 个样本（学习时长、出勤率 → 是否通过）。
- `generate_multiclass_data(seed=0)`：文档 §5 三簇二维高斯，300 个点、3 类各 100。

## 通过标准

```bash
cd ml/algorithms/logistic_regression
../../../.venv/bin/python -m pytest test_logistic_regression.py -v
```

全部测试通过（绿）。骨架状态下 19 条测试全部以 `NotImplementedError` 失败、2 条数据生成器测试通过；每实现一个函数，对应的测试就转绿。

## 常见坑提示

- **sigmoid 溢出**：`np.exp(1000)` 直接变 inf。按 z 正负分两支算（都只用 e 的负指数），或 `np.clip(z, -500, 500)`。测试会把 RuntimeWarning 当失败处理，光结果对还不够，警告也不能有。
- **log(0)**：交叉熵里 p=0 或 p=1 会炸出 -inf。先 `np.clip(p, 1e-15, 1 - 1e-15)`。
- **增广矩阵**：截距 b 照旧靠拼全 1 列吸收进 θ：`np.column_stack([np.ones(n), X])`。sklearn 不用拼，手写要拼（与模块 1 约定一致）。
- **特征尺度**：hours（0~10）和 attendance（0.5~1.0）差 10 倍，直接训练学习率难调。测试里的做法：hours / 10 再拼增广列，训练完把对应权重除回去再和 sklearn 比。
- **交叉熵收敛慢**：比线性回归的 MSE 慢得多。默认 lr=0.1 跑 2000 步只能到半路（loss≈0.43），不是实现错了——测试用 lr=1.0、n_iter=10000（缩放后的特征上仍单调下降）。
- **softmax 必须减行最大值**：`z - z.max(axis=1, keepdims=True)`，softmax 对整行同减常数不变，但数值上天差地别。
- **混淆矩阵约定**：行是真实、列是预测，`[[TN, FP], [FN, TP]]`，与 sklearn 一致。写反了指标全错。
- **浮点比较**：永远用 `np.allclose`，不要用 `==`。
- **对答案必须无正则**：sklearn 默认带 L2 正则，系数会对不上手写版。用 `LogisticRegression(C=np.inf)`（C=1/λ，λ=0 即无正则；旧写法 `penalty='none'` 字符串已报错、`penalty=None` 带 FutureWarning，版本细节见文档 §8；多分类不用再设 `multi_class`，该参数已废弃，非二分类自动走 multinomial）。

## 玩耍建议（测试之外）

- 画 sigmoid 曲线：`z = np.linspace(-10, 10)` 过一遍你写的 `sigmoid`，观察 σ(0)=0.5 和两端饱和（文档 §2）。
- 画梯度下降的 loss 历史曲线，换几个学习率对比收敛速度（文档 §4）。
- 画决策边界：散点图按 y 着色，再叠加 `w·x + b = 0` 的直线（文档 §6）。
- 调阈值看此消彼长：threshold 从 0.1 扫到 0.9，画 precision / recall 曲线，观察"越严格精确率越高、召回率越低"（文档 §6/§7）。
- 多分类画三簇散点 + 三条决策边界，直观感受 softmax 如何把平面切成三块（文档 §5）。
