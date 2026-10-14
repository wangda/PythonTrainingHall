# 手写练习：MLP 与反向传播（★ 核心里程碑）

> 前置文档：`ml/docs/阶段二/07-MLP与反向传播前置学习文档.md`（下称"文档"）。本练习的所有公式都在文档里，这里只做导引，不重复讲解。

## 目标

仅用 NumPy，在 `mlp.py` 中手写实现一个**任意层数**的全连接网络（隐藏层 ReLU + 输出层 softmax + 交叉熵）：

1. **relu / relu_grad**（文档 §3）——激活函数及其导数
2. **init_params**（文档 §4）——He 初始化，理解"全零对称性灾难"
3. **forward**（文档 §4）——逐层矩阵前向传播，顺手存好 cache
4. **cross_entropy_loss**（文档 §5）——多分类交叉熵均值
5. **backward**（文档 §5）——**核心中的核心**，链式法则逐层回传梯度
6. **update_params**（文档 §5）——SGD 更新一步
7. **train**（文档 §4/§5）——mini-batch 训练循环
8. **predict**（文档 §4）——argmax 出类别

对答案方式与之前不同：**梯度检查**（文档 §6）——用已实现完整的 `numerical_gradient`（中心差分）逐参数验证你 backward 推的梯度，相对误差 1e-5 量级以内才算对。

已实现完整、直接可用的辅助函数：
- `load_digits_data(seed=42)`：sklearn digits 数据集（8×8 小图，1437 训练 / 360 测试，64 维，像素已 /16 归一化）。
- `one_hot` / `softmax`：模块 2 已练过，这里直接给（softmax 的"减行最大值"数值稳定处理已做好）。
- `numerical_gradient`：梯度检查工具，是手段不是考察对象。

## 实现步骤建议

| 顺序 | 函数 | 对应文档 | 说明 |
| ---- | ---- | ---- | ---- |
| 1 | `relu` / `relu_grad` | §3 | 各一行热身。relu_grad 吃 z 不吃 a，z=0 处约定取 0 |
| 2 | `init_params` | §4 | He 初始化：`rng.normal(0, sqrt(2/sizes[l-1]), 形状)`，b 全零 |
| 3 | `forward` | §4 | 逐层 `Z = A @ W + b`，隐藏层 ReLU、输出层 softmax；cache 存 "A0".."AL"、"Z1".."ZL" |
| 4 | `cross_entropy_loss` | §5 | 均值公式，先 clip 防 log(0) |
| 5 | `backward` | §5 | 核心。δ_L = P − Y，逐层回传，每个梯度 assert 形状 |
| 6 | 梯度检查 | §6 | 跑数值梯度检查测试，绿了再继续——梯度不对后面全是白练 |
| 7 | `update_params` | §5 | `params[k] -= lr * grads[k]` |
| 8 | `train` / `predict` | §4/§5 | 打乱→分批→forward→loss→backward→update；每 epoch 记录批 loss 均值 |

## 通过标准

```bash
cd ml/algorithms/mlp_from_scratch
../../../.venv/bin/python -m pytest test_mlp.py -v
```

全部测试通过（绿）。骨架状态下 18 条测试以 `NotImplementedError` 失败、4 条 ✅ 辅助函数测试通过；每实现一个函数，对应的测试就转绿。

## 常见坑提示

- **relu_grad 吃 z 不吃 a**：反向传播求 δ_l 时，relu′ 的输入是 cache 里的 `Z_l`（加权输入，激活前），不是 `A_l`（激活后）。这是本练习最高频的 bug。
- **忘了 /n**：交叉熵是"均值"，所以 ∂L/∂W_l = Aᵀδ/**n**、∂L/∂b_l = mean(δ)。漏掉 /n 梯度大 n 倍，loss 直接起飞。
- **softmax 减最大值已帮你做了**：`softmax()` 是完整实现的，直接用，不用自己再处理溢出。
- **先算完所有梯度再更新**：backward 回传 δ 时用的是各层 W 的**当前值**。如果在循环里边算梯度边更新 W，后面层的 δ 就用错了。本骨架里 backward 只算不更新、update_params 统一更新，天然避开这个坑。
- **形状错位被广播静默蒙混**：numpy 广播不会报错，只会给你错得离谱的结果。每个梯度算完顺手 `assert grads["W1"].shape == params["W1"].shape`，是最便宜的排错手段。
- **梯度检查的甜区**：`numerical_gradient` 的 eps=1e-5 是中心差分的常用甜区——太大截断误差大，太小被浮点舍入淹没。写对了 backward，解析 vs 数值的相对误差应在 1e-9 量级。
- **XOR 对初始化敏感**：4 个隐藏单元的 XOR 网络实测 20 个 seed 只有约一半收敛（梯度流全通也会卡在局部极小），测试固定了 seed=0。这不是 bug，是 §1"非凸优化"的亲身体验。

## 玩耍建议（测试之外）

- 画 digits 训练的 loss 曲线，换 lr（0.05 / 0.5 / 2.0）对比收敛速度与震荡。
- 换网络结构：`[64, 16, 10]`、`[64, 64, 32, 10]`（三层！forward/backward 写对了就直接支持），对比准确率与训练时间。
- 跑真 MNIST：模块 2 项目里缓存了 `ml/projects/mnist_classification/data/raw/mnist.npz`（`np.load` 读取，X 是 0~255 的像素，**记得 /255**，和 digits 的 /16 不同）。sizes 换成 `[784, 128, 10]`，和 digits 对比准确率。
- 故意制造 bug 再抓 bug：把 backward 里的 relu′ 输入从 Z 换成 A、或把 /n 删掉，看梯度检查测试怎么报警——体会"梯度检查是反向传播的照妖镜"。
- 加宽 XOR 隐藏层（[2, 16, 2]），换 20 个 seed 试试，验证"宽一层就稳了"。
