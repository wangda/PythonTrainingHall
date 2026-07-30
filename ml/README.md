# Machine Learning 学习路径（优化版）

从 Python 基础到深度学习，系统掌握 ML 理论 + 工程实践。

---

## 目录结构

```
ml/
├── fundamentals/       # 基础库练习（numpy, matplotlib, pandas）
├── algorithms/         # 算法从零实现（每算法一个文件夹）
├── projects/           # 完整项目（工程化标准流程）
│   └── _template/      # 项目模板
├── d2l_exercises/      # 《动手学深度学习》课后练习
└── notebooks/          # 探索性 notebook
```

---

## 优先级说明

每个模块标注了 **P0 / P1 / P2**：

| 优先级 | 含义 | 要求 |
|---|---|---|
| **P0** | 核心必学，投入最大精力 | 手写 + 调库 + 项目验证 |
| **P1** | 重要，调库为主，手写关键组件 | 手写核心组件 1 次，其余调库 |
| **P2** | 了解概念，知道是什么、用在哪儿 | 读文档 + 跑 demo，不手写 |

---

## 阶段一：工具链（3 周）

**目标**：掌握向量化编程思维，能流畅处理数据。

### NumPy — 一切的基础

- [x] ~~数组创建~~（已完成）
- [x] ~~索引与切片~~（已完成）
- [X] 广播机制 ← 你在这里
- [ ] 矩阵运算（dot, linalg, einsum）
- [ ] 统计函数（axis 理解、聚合）
- [ ] 随机数生成

**里程碑**：手写 KMeans（仅 NumPy）

### Matplotlib — 够用就行

- [ ] 折线图/散点图（够画 loss 曲线即可）
- [ ] 子图布局（subplots）
- [ ] ~~颜色映射、热力图~~ ← **砍掉，用到再查**

### Pandas — 特征工程命脉

- [ ] DataFrame 创建与索引
- [ ] groupby + agg 分组聚合
- [ ] 缺失值处理（dropna, fillna）
- [ ] merge / join 多表操作
- [ ] 读取 CSV / Excel

### SciPy（按需，不单独学）

用到什么查什么：概率分布、optimize.minimize

---

## 阶段二：主干算法（4 个月）

### P0 — 核心，必须手写 + 项目

#### 模块 1：线性回归
- [ ] 闭式解（正规方程）
- [ ] 批量梯度下降 / SGD
- [ ] 多项式回归 + 特征扩展
- [ ] Ridge / Lasso 正则化
- [ ] 评估：MSE、R²、残差分析

**项目**：房价预测（`projects/house_prices/`）

#### 模块 2：逻辑回归与分类
- [ ] Sigmoid + 交叉熵损失
- [ ] 梯度下降训练二分类器
- [ ] Softmax 多分类
- [ ] 评估：准确率、精确率、召回率、F1、ROC-AUC、混淆矩阵

**项目**：手写数字分类（`projects/mnist_classification/`）

#### 模块 3：决策树与集成学习
- [ ] 手写决策树（信息增益 / 基尼系数分裂）
- [ ] 预剪枝
- [ ] 随机森林（调库）
- [ ] LightGBM 调参实战（早停、交叉验证、特征重要性）
- [ ] XGBoost（对比 LightGBM 差异）

**项目**：Titanic 生存预测 + LightGBM 调参报告

#### 模块 4：MLP 与反向传播 ★ 核心里程碑
- [ ] 感知机
- [ ] 手写 3 层 MLP + backward（纯 NumPy）
- [ ] 在 MNIST 上跑通
- [ ] PyTorch 重写 MLP
- [ ] 封装可复用的 Trainer 类（checkpoint、早停、lr 调度）

> 做完这个，你就有了"精通反向传播底层实现"的底气。

**项目**：Fashion-MNIST 分类 + 自定义 Trainer

### P1 — 重要，调库为主

#### 模块 5：CNN
- [ ] 卷积层理解（im2col 概念，不手写完整实现）
- [ ] 池化层
- [ ] LeNet 搭建 + 训练
- [ ] 迁移学习：用 torchvision ResNet18 做 fine-tune
- [ ] ~~ResNet/AlexNet 手写~~ ← **砍掉**

**项目**：CIFAR-10 分类（迁移学习），感受"预训练模型 + 微调"的威力

#### 模块 6：LSTM 与时间序列
- [ ] RNN 结构理解（循环含义）
- [ ] **手写 LSTM Cell 一次**（理解 gate 机制）
- [ ] PyTorch nn.LSTM 调库使用（重点搞懂 batch_first、hidden_state 维度）
- [ ] 滑动窗口构造时序样本
- [ ] 多变量时序预测

**项目**：工业传感器预测 / 销量预测（`projects/time_series_forecast/`）

#### 模块 7：Transformer 注意力机制
- [ ] **手写 Scaled Dot-Product Attention**
- [ ] Multi-Head Attention 理解
- [ ] 位置编码
- [ ] PyTorch 调库搭 Transformer Encoder（不手写完整 Encoder）
- [ ] ~~完整 Transformer 从零实现~~ ← **砍掉**

### P2 — 了解概念，不做深度投入

| 模块 | 投入 | 说明 |
|---|---|---|
| SVM | 2 天 | 理解"核函数把低维不可分映射到高维可分"即可，不手推对偶问题，不写 SMO |
| 聚类 | 只写 KMeans | DBSCAN / 层次聚类读文档了解概念，不手写 |
| PCA | 手写 1 次 | PCA 手写（特征值分解），t-SNE / UMAP 调库了解 |
| Word2Vec | 1 天 | 理解分布式语义表示思想即可，不用手写，后续直接用 transformers 库 |

---

## 阶段三：实战项目（贯穿，约 5 个月）

### 工程化标准（新增）

利用你的软件工程背景，每个项目强制包含：

```
projects/project_name/
├── config.yaml          # 超参数配置（learning_rate, batch_size 等）
├── data/
│   └── download.py      # 数据下载与预处理
├── features.py          # 特征工程
│   └── test_features.py # 特征处理单元测试
├── model.py             # 模型定义
├── train.py             # 训练脚本
├── evaluate.py          # 评估 + 可视化
├── predict.py           # 命令行推理 demo
│                        # python predict.py --input "..." --output result.csv
├── requirements.txt
└── Dockerfile           # （后期加，非必须）
```

### 推荐项目清单

| 项目 | 核心技能 | 优先级 |
|---|---|---|
| 房价预测 | 线性回归、特征工程、R² 评估 | P0 |
| Titanic | 决策树、LightGBM、调参 | P0 |
| Fashion-MNIST | MLP、反向传播、Trainer 封装 | P0 |
| CIFAR-10 迁移学习 | CNN、预训练模型 fine-tune | P1 |
| 时序预测 | LSTM、滑动窗口、多变量预测 | P1 |
| 文本分类 | Transformer、Tokenizer、分类头 | P1 |

选 **P0 项目先做**，学完对应模块就做。P1 项目选 2 个做。

---

## 时间线（总计约 10-12 个月）

```
第 1-3 周   阶段一：NumPy + Matplotlib + Pandas
第 4-7 周   线性回归 → 逻辑回归（手写 + 调库）
第 8-12 周  决策树 → LightGBM（调参实战）
第 13-18 周 MLP 手写反向传播 → PyTorch Trainer 封装 ★
第 19-22 周 LSTM + 时序预测
第 23-26 周 Transformer + 文本分类
第 27-40 周 2-3 个完整工程化项目 + 查漏补缺
```

---

## 学习原则

| 原则 | 说明 |
|---|---|
| **P0 手写，P1 调库** | 核心算法手写一次再调库，非核心直接调库 |
| **做项目再学，不为学而学** | 每个模块学完立即做一个项目巩固 |
| **工程化是壁垒** | config.yaml + 单元测试 + predict.py，这是你的差异化优势 |
| **可视化每一件事** | 训练曲线、混淆矩阵、特征重要性——养成 plot 习惯 |
| **For 循环是敌人** | 尽量用向量化操作替代循环 |
| **用 git 管理进度** | 每个模块一个 commit，进步可追溯 |
| **不要贪多求全** | P2 模块了解即可，留时间给项目和工程化 |

---

## 推荐的参考资源

| 资源 | 用途 |
|---|---|
| 《动手学深度学习》(d2l) | 主线教材，理论 + 代码 |
| Scikit-Learn 文档 | 算法 API 与使用示例 |
| PyTorch 官方教程 | 深度学习工程实践 |
| LightGBM 文档 | 工业级树模型调参 |
| 李沐《动手学深度学习》视频 | d2l 配套讲解 |

---

## 进度跟踪

```
fundamentals/
├── numpy_basics/    → 01 02 已完成，03 进行中
├── matplotlib/
└── pandas/

algorithms/
├── linear_regression/    # 手写 + 调库 + test
├── logistic_regression/
├── decision_tree/
├── mlp_from_scratch/     # ★ 核心里程碑
└── kmeans/

projects/
├── house_prices/
├── titanic/
├── fashion_mnist/
├── cifar10_transfer/
├── time_series_forecast/
└── text_classification/
```
