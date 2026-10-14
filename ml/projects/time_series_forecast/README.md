# 项目：时序预测（ETTh1 工业传感器）

> 配套文档：`ml/docs/阶段二/12-时序预测项目文档.md`（下称"项目文档"）。
> 学完阶段二模块 6（LSTM 与时间序列）后做这个项目：手写 LSTM Cell 已在
> `ml/algorithms/lstm/` 练过，这里用 **PyTorch nn.LSTM** 做多变量工业传感器预测，
> 重点是把模块 4 的 Trainer 工程化经验迁移到回归 + 时序场景。

## 目录结构

```
time_series_forecast/
├── config.yaml          # ✅ 所有旋钮：窗口长、预测步长、按时间切分比例、模型与训练超参
├── data/
│   ├── download.py      # ✅ 已实现：ETTh1 三通道下载（raw 直连 → gh-proxy → 合成兜底）+ 缓存
│   └── raw/             # 数据缓存（ETTh1 真实数据已就位，可离线跑）
├── features.py          # 📝 骨架：滑动窗口、按时间切分、训练段标准化（时序泄漏防护是核心）
├── model.py             # ✅ 已实现：LSTMForecaster（nn.LSTM + 取最后时间步 + Linear）
├── trainer.py           # 📝 骨架：fit / save_checkpoint / check_early_stopping / _make_scheduler
├── train.py             # 🔧 流程框架已写好，调用上面的 TODO 函数
├── evaluate.py          # 📝 骨架：反归一化 MAE/RMSE + naive baseline 对比 + 两张图
├── predict.py           # ✅ 已实现：命令行推理（最近 96 小时 → 未来 4 小时油温）
├── tests/               # ✅ 已实现：features / model / trainer 的单元测试
├── models/              # 训练产物（best.pt + history.json，示范产物已保留）
├── reports/figures/     # 评估图表（预测对比曲线、训练曲线，已生成示范）
└── requirements.txt
```

## 使用流程

```bash
cd ml/projects/time_series_forecast

# 0. 数据已缓存（ETTh1 真实数据），首次运行如无缓存会自动三通道下载
../../../.venv/bin/python data/download.py

# 1. 看测试失败——失败信息就是任务清单
../../../.venv/bin/python -m pytest tests/ -v

# 2. 实现 features.py 四个函数 + trainer.py 四个方法（项目文档 §5）→ 测试变绿

# 3. 训练（GPU 几秒 / CPU 一两分钟）
../../../.venv/bin/python train.py

# 4. 评估：反归一化指标 + naive baseline 对比 + 出图
../../../.venv/bin/python evaluate.py

# 5. 命令行预测（最近 96 小时传感器数据 csv）
../../../.venv/bin/python predict.py --input recent.csv
```

## 任务清单（详见项目文档 §5）

1. **Baseline**：默认 config 跑通，记录测试集 MAE/RMSE（反归一化后）；
2. **Trainer 三组件**：checkpoint 存验证最优、早停、StepLR，用 history.json 画训练曲线；
3. **naive baseline 对战**：测试集上和"预测=窗口最后值"正面对比（实测 LSTM RMSE 小胜、
   MAE 打平——把"为什么 naive 这么顽固"想明白，这是本项目最值钱的一课）；
4. **消融实验**：window_size（24/48/96）、horizon（1/4/12）扫一遍，观察对 naive 优势的变化；
5. **评估可视化**：预测 vs 真实曲线 + 训练曲线，按前置文档 §7 分析。

## 与文档的对应关系

| 文件 | 项目文档章节 | 前置文档（11-LSTM） |
|---|---|---|
| `data/download.py` | §2 数据集 | —— |
| `features.py` | §3 差异点、§5 任务清单 | §6 滑动窗口与时序泄漏 |
| `model.py` | §4 Pipeline | §5 nn.LSTM 实战 |
| `trainer.py` | §5 任务清单 | §7 评估（MSE loss） |
| `evaluate.py` | §5 任务 3、§7 常见坑 | §7 多步预测与评估 |
| `predict.py` | §3 工程化差异点 | §5 输出形状 |
