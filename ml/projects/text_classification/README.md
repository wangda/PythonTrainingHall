# 项目：文本分类（AG News，Transformer Encoder）

> 配套文档：`ml/docs/阶段二/14-文本分类项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 7（Transformer 注意力机制）后做这个项目：手写注意力已在
> `ml/algorithms/transformer_attention/` 练过，这里用 **PyTorch 调库**（nn.TransformerEncoder）
> 走完一条完整的文本分类流水线——分词、词表、padding mask、[CLS] 分类头，全手工。

## 目录结构

```
text_classification/
├── config.yaml          # ✅ 所有旋钮：子集大小、max_len、词表、模型结构、训练超参
├── data/
│   ├── download.py      # ✅ 已实现：AG News 三通道下载（raw 直连 → gh-proxy → 合成兜底）+ 缓存
│   └── raw/             # 数据缓存（官方全量 train 120000 / test 7600 已就位，可离线跑）
├── features.py          # 📝 骨架：分词、建词表（防泄漏核心）、编码、DataLoader 三件套
├── model.py             # ✅ 已实现：TextClassifier（Embedding + 正弦位置编码 + Encoder + [CLS] 头）
├── trainer.py           # 📝 骨架：fit / save_checkpoint / check_early_stopping / _make_scheduler
├── train.py             # 🔧 流程框架已写好，调用上面的 TODO 函数
├── evaluate.py          # 📝 骨架：accuracy / macro F1 + 混淆矩阵热力图 + 训练曲线
├── predict.py           # ✅ 已实现：命令行推理（一条英文新闻 → top-2 类别与概率）
├── tests/               # ✅ 已实现：features / model / trainer 的单元测试
├── models/              # 训练产物（best.pt + vocab.json + history.json，示范产物已保留）
├── reports/figures/     # 混淆矩阵、训练曲线（已生成示范）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/text_classification

# 0. 数据已缓存（官方全量 AG News），首次运行如无缓存会自动三通道下载
../../../.venv/bin/python data/download.py

# 1. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 2. 实现 features.py 三个函数 + trainer.py 四个方法（项目文档 §5）→ 测试变绿

# 3. 训练（GPU 半分钟内）
../../../.venv/bin/python train.py

# 4. 评估：测试集指标 + 混淆矩阵 + 训练曲线
../../../.venv/bin/python evaluate.py

# 5. 命令行预测
../../../.venv/bin/python predict.py --input "Apple unveils new M5 chip for laptops"
```

## 任务清单（详见项目文档 §5）

1. **Baseline**：默认 config 跑通（实测测试集 accuracy 0.8261）；
2. **Trainer 三组件**：checkpoint 存验证最优、早停、cosine 调度，画训练曲线；
3. **TF-IDF 对照**：跑一遍 sklearn TF-IDF+LogisticRegression（实测 0.8963——
   2 万样本下经典方法更强，想明白为什么，这比刷分重要）；
4. **混淆矩阵分析**：找出最大混淆对（实测 Business↔Sci/Tech），看样例解释为什么；
5. **predict 演示**：正例 + 一个易混例，体会"词表和模型必须成对保存"。

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（13-Transformer） |
|---|---|---|
| `data/download.py` | §2 数据集 | —— |
| `features.py` | §3 差异点、§5 任务清单 | ——（文本特征工程专属） |
| `model.py` | §4 Pipeline | §6 位置编码、§7 搭 Encoder |
| `trainer.py` | §5 任务清单 | §7 训练循环五步法 |
| `evaluate.py` | §5 任务 3、§7 常见坑 | ——（指标见模块 2 文档 §7） |
| `predict.py` | §3 差异点（词表随模型保存） | —— |
