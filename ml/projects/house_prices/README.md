# 项目：房价预测（California Housing）

> 配套文档：`ml/docs/阶段二/02-房价预测项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 1（线性回归）后做这个项目：手写已在 `ml/algorithms/linear_regression/`
> 练过，这里重点走一遍**工业界 ML 项目的工程化流程**。

## 目录结构

```
house_prices/
├── config.yaml          # 所有旋钮：数据路径、test_size、random_state、模型类型与超参数
├── data/
│   ├── download.py      # ✅ 已实现：下载 California Housing 并缓存 csv / 读缓存
│   └── raw/             # 数据缓存（首次运行自动生成）
├── features.py          # 📝 骨架：组合特征、组装 X/y、划分+标准化（你要实现的核心）
├── model.py             # ✅ create_model 已实现（工厂模式）；📝 train_model 骨架
├── train.py             # 🔧 流程框架已写好，调用上面的 TODO 函数
├── evaluate.py          # 📝 骨架：MSE/RMSE/MAE/R² + 残差图 + 预测对比图
├── predict.py           # ✅ 已实现：命令行推理（依赖 features.py 完成后可用）
├── tests/               # ✅ 已实现：features / model 的单元测试
├── models/              # 训练产物（train.py 运行后生成）
├── reports/figures/     # 评估图表（evaluate 实现后生成）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/house_prices

# 0. 下载并缓存数据（已实现，可直接跑）
../../../.venv/bin/python data/download.py

# 1. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 2. 实现 features.py 三个函数（文档 §6 有特征构造思路）→ 测试变绿

# 3. 训练（内部会调你实现的函数）
../../../.venv/bin/python train.py

# 4. 实现 evaluate.py → 评估 + 出图
../../../.venv/bin/python evaluate.py

# 5. 命令行预测
../../../.venv/bin/python predict.py --input "8.3252,41,6.98,1.02,322,2.55,37.88,-122.23"
```

## 任务清单（详见项目文档 §5）

1. **Baseline**：原始 8 特征 + LinearRegression，记录测试集 R²（目标 > 0.6）；
2. **特征工程**：加 BedroomRatio / RoomsPerPerson / LogMedInc，对比 R² 变化；
3. **Ridge 对比**：config.yaml 里 `model.name` 改成 `ridge`，调 `alpha`；
4. **评估可视化**：残差图 + 预测 vs 真实散点图，按前置文档 §7.3 的诊断表分析。

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（01-线性回归） |
|---|---|---|
| `data/download.py` | §2 数据集 | —— |
| `features.py` | §5 任务清单、§6 特征工程提示 | §5 特征扩展 |
| `model.py` | §3 工程化流程 | §3 闭式解 / §6 正则化 |
| `train.py` | §3 工程化流程、§4 Pipeline | —— |
| `evaluate.py` | §4 Pipeline、§5 目标 | §7 模型评估 |
| `predict.py` | §3 工程化流程 | §8 sklearn 速查 |
