# 项目：手写数字分类（digits / MNIST）

> 配套文档：`ml/docs/阶段二/04-手写数字分类项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 2（逻辑回归与多分类）后做这个项目：工程骨架复用模块 1
> （`ml/projects/house_prices/`，见《02-房价预测项目文档.md》），
> 本项目新增三个维度：**图像数据 + 多分类 + 概率输出**。

## 目录结构

```
mnist_classification/
├── config.yaml          # 所有旋钮：dataset 选择、test_size、random_state、归一化开关、C/max_iter
├── data/
│   ├── download.py      # ✅ 已实现：load_dataset(config) 统一接口；digits 内置直读，mnist 缓存 npz
│   └── raw/             # mnist.npz 缓存（首次运行自动生成，digits 无需缓存）
├── features.py          # 📝 骨架：图像展平、像素归一化、划分（你要实现的核心）
├── model.py             # ✅ create_model 已实现（工厂模式）；📝 train_model 骨架
├── train.py             # 🔧 流程框架已写好，调用上面的 TODO 函数
├── evaluate.py          # 📝 骨架：accuracy + macro P/R/F1 + 混淆矩阵热力图
├── predict.py           # ✅ 已实现：命令行推理，--input 接受 PNG 图片或像素 csv，支持 --invert
├── tests/               # ✅ 已实现：features / model 的单元测试
├── models/              # 训练产物（train.py 运行后生成）
├── reports/figures/     # 评估图表（evaluate 实现后生成）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/mnist_classification

# 0. digits 内置无需下载；mnist 缓存 data/raw/mnist.npz 已随仓库预置
#    （若删除后需重新下载：../../../.venv/bin/python data/download.py）

# 1. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 2. 实现 features.py 四个函数 + model.train_model → 测试变绿

# 3. 训练（内部会调你实现的函数，默认 digits 秒级完成）
../../../.venv/bin/python train.py

# 4. 实现 evaluate.py → 指标 + 混淆矩阵图
../../../.venv/bin/python evaluate.py

# 5. 命令行预测（手绘白底黑字图记得加 --invert）
../../../.venv/bin/python predict.py --input my_digit.png --invert
```

## 任务清单（详见项目文档 §5）

1. **digits baseline**：跑通全流程，测试集 accuracy 目标 ≥ 0.95；
2. **混淆矩阵分析**：找出最容易混淆的数字对，对照图像解释原因；
3. **切 mnist 全量重训**：config.yaml 里 `data.dataset: mnist`，对比 accuracy；
4. **predict.py 推理**：对自己画/找的数字图片预测，体会"预处理必须复现"和 `--invert` 的坑。

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（03-逻辑回归与分类） |
|---|---|---|
| `data/download.py` | §2 数据集 | —— |
| `features.py` | §3 工程化差异点、§5 任务清单、§6 特征工程提示 | §1 sigmoid 模型 |
| `model.py` | §3 工程化流程 | §3 交叉熵 / §4 梯度下降 / §5 softmax |
| `train.py` | §3 工程化流程、§4 Pipeline | —— |
| `evaluate.py` | §4 Pipeline、§5 目标 | §7 评估指标 |
| `predict.py` | §3 工程化流程、§7 常见坑 | §5 softmax / §6 阈值 |
