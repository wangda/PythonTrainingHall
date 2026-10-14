# 项目：Fashion-MNIST 分类（PyTorch MLP + 自定义 Trainer）

> 配套文档：`ml/docs/阶段二/08-FashionMNIST分类项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 4（MLP 与反向传播）后做这个项目：工程骨架沿用模块 1/2
> （`ml/projects/house_prices/`、`ml/projects/mnist_classification/`），
> 本项目新增三个维度：**PyTorch 工程框架 + 自定义 Trainer（checkpoint/早停/lr 调度）+ autograd**。

## 目录结构

```
fashion_mnist/
├── config.yaml          # 所有旋钮：三集合划分、模型结构、lr/batch_size/epochs、早停、lr 调度
├── data/
│   ├── download.py      # ✅ 已实现：torchvision 优先、IDX 镜像回退，统一缓存 fashion_mnist.npz
│   └── raw/             # fashion_mnist.npz 缓存（首次运行自动生成）
├── features.py          # 📝 骨架：从训练集切验证集、归一化、包 DataLoader（三集合是你的第一课）
├── model.py             # ✅ create_model 已实现（按 hidden_sizes 动态堆叠 MLP，末层不接 softmax）
├── trainer.py           # 📝 骨架：Trainer 类——checkpoint / 早停 / lr 调度（本项目的灵魂）
├── train.py             # 🔧 流程框架已写好，调用上面的 TODO
├── evaluate.py          # 📝 骨架：测试集指标 + classification_report + 混淆矩阵 + 训练曲线
├── predict.py           # ✅ 已实现：命令行推理，--input 接受 28×28 PNG，支持 --invert
├── tests/               # ✅ 已实现：features / model / trainer 的单元测试
├── models/              # 训练产物：best.pt（验证集最优 checkpoint）+ history.json
├── reports/figures/     # 评估图表（混淆矩阵、训练曲线）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/fashion_mnist

# 0. 下载并缓存数据（已实现，直接跑；torchvision 失败会自动回退 IDX 镜像）
../../../.venv/bin/python data/download.py

# 1. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 2. 实现 features.py + trainer.py → 测试变绿

# 3. 训练（GPU/CPU 自适应；仓库已附带训练好的 models/best.pt，可直接跳到第 5 步体验）
../../../.venv/bin/python train.py

# 4. 实现 evaluate.py → 测试集指标 + 混淆矩阵 + 训练曲线
../../../.venv/bin/python evaluate.py

# 5. 命令行推理（白底图记得加 --invert）
../../../.venv/bin/python predict.py --input my_image.png --invert
```

## 任务清单（详见项目文档 §5）

1. **默认 config 跑通 baseline**：测试集 accuracy 目标 ≥ 0.87；
2. **实现 Trainer 三组件**：用 history.json 画训练/验证曲线，指出早停触发点；
3. **消融实验**：关掉早停跑 50 epoch 观察 val loss 回升（过拟合现场）；换 lr scheduler 对比；
4. **测试集评估 + 混淆矩阵分析**：找出最容易混淆的类别对并看图解释；
5. **predict.py 推理**：对真实图片预测，体会预处理复现与 `--invert` 的坑。

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（07-MLP与反向传播） |
|---|---|---|
| `data/download.py` | §2 数据集 | —— |
| `features.py` | §3 工程化差异点、§5 任务清单 | §7 DataLoader / §8 三集合 |
| `model.py` | §3 工程化差异点 | §7 nn.Module |
| `trainer.py` | §1 项目目标、§5 任务 2/3 | §7 训练循环五步法 / §8 工程化 Trainer |
| `train.py` | §4 Pipeline | —— |
| `evaluate.py` | §4 Pipeline、§5 任务 4 | 03 前置文档 §7 评估指标 |
| `predict.py` | §5 任务 5、§7 常见坑 | §5 softmax |
