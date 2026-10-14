# 项目：Titanic 生存预测 + LightGBM 调参报告

> 配套文档：`ml/docs/阶段二/06-Titanic生存预测项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 3（决策树与集成学习，见《05-决策树与集成学习前置学习文档.md》）
> 并完成手写练习 `algorithms/decision_tree/` 后做这个项目。
> 工程骨架复用前两个项目（`house_prices` / `mnist_classification`），
> 本项目新增三个维度：**脏表格数据（缺失值/类别特征/泄漏列）+ 有状态预处理 + 调参报告**。

## 目录结构

```
titanic/
├── config.yaml          # 所有旋钮：数据路径、特征开关、四模型选择、调参网格
├── data/
│   ├── download.py      # ✅ 已实现：load_dataset(config)；OpenML 拉取 + csv 缓存
│   └── raw/             # titanic.csv 缓存（首次运行自动生成，需联网一次）
├── features.py          # 📝 骨架：split_data / fit_preprocessor / transform（你要实现的核心）
├── model.py             # ✅ 已实现：create_model 工厂（决策树/随机森林/LightGBM/XGBoost）
├── train.py             # 🔧 流程框架已写好，调用上面的 TODO 函数
├── evaluate.py          # ✅ 已实现：accuracy/P/R/F1/ROC-AUC + 混淆矩阵 + ROC + 特征重要性对比
├── tune.py              # 📝 核心骨架：调参四轮法，stage 0 已实现，stage 1~4 待你完成
├── predict.py           # ✅ 已实现：命令行推理（单乘客参数 / 批量 csv）
├── tests/               # ✅ 已实现：features / model 的单元测试
├── models/              # 训练产物（train.py 运行后生成）
├── reports/             # 评估图表 + tuning_report.md（tune.py 运行后生成）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/titanic

# 0. 安装依赖（lightgbm 必装，xgboost 可选）
pip install -r requirements.txt

# 1. 预热数据缓存（仅首次需要联网，约几百 KB）
../../../.venv/bin/python data/download.py

# 2. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 3. 实现 features.py 三个函数 → 测试变绿
#    （核心课：先划分、状态只在训练集 fit、transform 是纯函数）

# 4. 训练（默认 lightgbm；改 config.yaml 的 model.name 可切换四模型）
../../../.venv/bin/python train.py

# 5. 完整评估：指标 + 混淆矩阵 + ROC + 特征重要性对比图
../../../.venv/bin/python evaluate.py

# 6. 调参四轮法（实现 tune.py 的 stage 1~4）→ reports/tuning_report.md
../../../.venv/bin/python tune.py

# 7. 推理 demo
../../../.venv/bin/python predict.py --pclass 3 --sex male --age 22 --sibsp 1 --parch 0 --fare 7.25 --embarked S
```

## 任务清单（详见项目文档 §7/§8）

1. **跑通基线**：实现 `features.py` → 测试变绿 → `train.py` 出分；
2. **泄漏实验**（必做）：暂时不删 `boat`/`body` 重训一次，亲眼看"作弊的分数"，改回来；
3. **三代对比**：decision_tree → random_forest → lightgbm → xgboost 各训一遍，记录指标；
4. **调参报告**：完成 `tune.py` stage 1~4，补填"观察与结论"，产出 `reports/tuning_report.md`；
5. **特征重要性**：看 `reports/figures/feature_importance.png`，回答三种口径排序是否一致；
6. （进阶）从 `name` 提取 Title 特征，重跑 CV 对比。

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（05-决策树与集成学习） |
|---|---|---|
| `data/download.py` | §2 数据集 | —— |
| `features.py` | §3 有状态预处理、§4 特征工程、§5 泄漏列 | §1.4 树不需要缩放 |
| `model.py` | §6 三代模型对比 | §5 剪枝 / §9 随机森林 / §11 LightGBM / §12 XGBoost |
| `train.py` | §3 流程铁律 | —— |
| `evaluate.py` | §8 任务 5 | §9.5 / §11.6 特征重要性 |
| `tune.py` | §7 调参四轮法 | §11.3 早停 / §11.4 交叉验证 / §11.5 调参顺序 |
| `predict.py` | §3 state 复现 | —— |
