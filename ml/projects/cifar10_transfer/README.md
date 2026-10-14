# 项目：CIFAR-10 迁移学习（LeNet 从零 vs ResNet18 微调）

> 配套文档：`ml/docs/阶段二/10-CIFAR10迁移学习项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 5（CNN，见《09-CNN前置学习文档.md》）并完成手写练习
> `algorithms/cnn/` 后做这个项目。工程骨架复用前几个项目，
> 本项目新增三个维度：**图片数据集 + PyTorch 训练循环 + 预训练模型微调**。

## 目录结构

```
cifar10_transfer/
├── config.yaml          # 所有旋钮：路线切换（lenet/resnet18）、finetune 策略、lr 分组、增强开关
├── data/
│   ├── download.py      # ✅ 已实现：load_dataset(config)，torchvision 下载 + 缓存（PIL 原图）
│   └── raw/             # CIFAR-10 缓存（首次运行自动下载，约 170MB，需联网）
├── features.py          # 📝 骨架：build_transforms（两套纪律）+ make_loaders（切验证集）
├── model.py             # ✅ LeNet 已实现；📝 create_model / create_optimizer 骨架（迁移手术）
├── train.py             # ✅ 已实现：训练五步参考版 + 验证集评估 + 最优权重保存
├── evaluate.py          # ✅ 已实现：测试集 accuracy + 分类报告 + 混淆矩阵 + 错分样本图
├── predict.py           # ✅ 已实现：任意图片的 top-3 预测
├── tests/               # ✅ 已实现：features / model 的单元测试
├── models/              # 训练产物 model.pt（权重 + config 快照 + meta）
├── reports/figures/     # 评估图表（evaluate.py 运行后生成）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/cifar10_transfer

# 0. 安装依赖（torch/torchvision 体积较大，CPU 版即可）
pip install -r requirements.txt

# 1. 预热数据缓存（首次需联网，约 170MB）
../../../.venv/bin/python data/download.py

# 2. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 3. 实现 features.py（两套 transforms 的纪律）+ model.py（迁移手术）→ 测试变绿

# 4. 路线 B：ResNet18 冻结微调（默认配置，CPU 也能跑）
../../../.venv/bin/python train.py

# 5. 测试集评估 + 混淆矩阵 + 错分样本图
../../../.venv/bin/python evaluate.py

# 6. 路线 A：config.yaml 里 model.name 改 lenet、features.image_size 改 32，重跑 4-5

# 7. 路线 C（有 GPU 或耐心）：model.finetune 改 full，重跑 4-5

# 8. 推理 demo
../../../.venv/bin/python predict.py --input my_cat.jpg
```

## 三路线对比表（任务 8：跑完填这张表）

| 路线 | best val acc | test acc | 训练时长 | 可训练参数 | 一句话观察 |
|---|---|---|---|---|---|
| A. LeNet 从零 | | | | | |
| B. ResNet18 冻结微调 | | | | 5,130 | |
| C. ResNet18 全量微调 | | | | 11M+ | |

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（09-CNN） |
|---|---|---|
| `data/download.py` | §2 数据集 | §1.3 NCHW |
| `features.py` | §3 两套 transforms 的纪律 | §6.5 预处理匹配 / §6.6 数据增强 |
| `model.py` | §5 迁移手术 | §4 LeNet / §6.3 换头冻结 / §6.4 分组 lr |
| `train.py` | §4 三条路线 | §5 训练五步（参考实现） |
| `evaluate.py` | §6 任务 9 错分分析 | 模块 2 §7 评估指标 |
| `predict.py` | —— | §6.5 预处理复现 |
