"""工程化 Trainer（骨架）——与 fashion_mnist 项目同构，本项目的灵魂。

配套文档《12-时序预测项目文档.md》§5 任务 2；原理见前置文档 §5（nn.LSTM 实战）
与 §7（多步预测与评估）。三个工程组件与 fashion_mnist 项目完全同构：

    ① save_checkpoint：验证集最优时存 models/best.pt——是【最优】不是【最后】；
    ② check_early_stopping：验证 loss 连续 patience 个 epoch 不创新低就停；
    ③ lr 调度：本项目只支持 StepLR / none（聚焦时序本身，调度器不是主角）。

与分类版 Trainer 的差异只有两处：
    - criterion 换成 nn.MSELoss()（回归任务，前置文档 §7）；
    - 验证只算 loss，不算 accuracy（回归没有"准确率"，对外指标是反归一化
      后的 MAE/RMSE，那是 evaluate.py 的事）。

用法（train.py 已串好）：
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit()   # 返回每 epoch 的 train_loss/val_loss/lr 序列
"""

from pathlib import Path

import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parent

# ETTh1 + LSTM 的实测参考（config 默认参数 hidden 64 / layers 2 / window 96 /
# horizon 4 / lr 1e-3 / batch 64 / StepLR(10, 0.5) / patience 5，seed=42，详见项目文档 §5）：
#   早停在第 9 epoch 触发，验证集最优在第 4 epoch（val_loss=0.0134）；
#   测试集（反归一化后）LSTM MAE 0.7370 / RMSE 1.0439，
#   naive baseline（预测 = 窗口最后一个值）MAE 0.7287 / RMSE 1.0752——
#   RMSE 小胜、MAE 基本打平：小时级 ETTh1 上 naive 极其顽固，"打过 baseline"
#   本身就是本项目最有价值的功课（前置文档 §7；项目文档 §5 任务 3 的消融实验）。


class Trainer:
    """把"训练循环 + 三个工程组件"封装成一个类。"""

    def __init__(self, model, train_loader, val_loader, config):
        """组装训练所需的一切（已完整实现——初始化只是记账，没有教学点）。

        参数:
            model:        create_model 返回的 nn.Module（输出 (batch, horizon) 回归值）；
            train_loader: 训练集 DataLoader（shuffle=True）；
            val_loader:   验证集 DataLoader（不 shuffle）；
            config:       dict，读取 config["train"] 下的 lr / weight_decay /
                          epochs / device / early_stopping / lr_scheduler /
                          model_dir / model_filename。
        """
        tcfg = config["train"]
        self.config = config
        self.device = self._resolve_device(tcfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.MSELoss()   # 回归任务：均方误差（前置文档 §7）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg.get("weight_decay", 0.0),
        )
        self.scheduler = self._make_scheduler()

        self.epochs = tcfg["epochs"]
        es = tcfg.get("early_stopping", {})
        self.early_stopping_enabled = es.get("enabled", True)
        self.patience = es.get("patience", 5)

        # 训练过程记账：best_* 记录验证集最优，history 记录每 epoch 序列
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        model_dir = Path(tcfg["model_dir"])
        self.ckpt_dir = model_dir if model_dir.is_absolute() else PROJECT_ROOT / model_dir
        self.ckpt_path = self.ckpt_dir / tcfg["model_filename"]

    @staticmethod
    def _resolve_device(name):
        """device 自适应：auto 时有 cuda 用 cuda，否则 cpu（已完整实现）。"""
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    # ---------- 训练循环的两个基本动作（已实现） ----------

    def _train_one_epoch(self):
        """在训练集上跑一个 epoch，返回平均 train loss。

        每个 batch 的五步：清零梯度 → 前向 → 算 MSE → 反向 → 更新。
        与分类版唯一的区别是 criterion 从 CrossEntropyLoss 换成 MSELoss，
        且 y 的形状从 (batch,) long 标签变成 (batch, horizon) float 目标值。
        """
        self.model.train()
        total_loss = 0.0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)   # 数据和模型必须在同一 device
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(X)
        return total_loss / len(self.train_loader.dataset)

    def _validate(self):
        """在验证集上评估，返回平均 MSE loss。不反向传播，不更新参数。"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                total_loss += self.criterion(self.model(X), y).item() * len(X)
        return total_loss / len(self.val_loader.dataset)

    # ---------- 你的三个任务 ----------

    def fit(self):
        """训练主循环：每 epoch 训练 + 验证 + 记账 + 三个组件联动。

        返回:
            dict，三个键（每 epoch 一个值，长度 = 实际训练的 epoch 数）：
                "train_loss": list[float]，训练集平均 MSE；
                "val_loss":   list[float]，验证集平均 MSE（早停/checkpoint 的依据）；
                "lr":         list[float]，该 epoch 使用的学习率。

        每 epoch 的正确顺序（顺序错了三个组件全是摆设）：
            1. _train_one_epoch() 训练一轮；
            2. _validate() 拿到 val_loss，与 lr 一起记入 history；
            3. val_loss 创历史新低 → 更新 best_val_loss / best_epoch /
               epochs_no_improve 归零，并调 save_checkpoint()；
               否则 epochs_no_improve += 1；
            4. 有 scheduler 就 scheduler.step()；
            5. 打印一行进度（epoch / train_loss / val_loss / lr）；
            6. check_early_stopping() 为真 → 打印提示并 break。
        """
        raise NotImplementedError

    def save_checkpoint(self):
        """把【当前】模型权重存到 models/best.pt——只在验证集最优时被 fit 调用。

        保存内容（torch.save 一个 dict）：
            "epoch":            当前 epoch 编号（定位"最优发生在哪一轮"）；
            "model_state_dict": self.model.state_dict()（只存权重，不存整个对象）；
            "val_loss":         该 epoch 的验证集 MSE；
            "config":           self.config（预测时要靠它重建模型结构、复现预处理）。

        提示:
            - 先 self.ckpt_dir.mkdir(parents=True, exist_ok=True) 再存；
            - 只存 state_dict 的原因与 fashion_mnist 项目一致：纯数据配合
              create_model 重建结构，代码重构后仍能加载；
            - scaler 的均值/方差不在 Trainer 的职责内——train.py 会在训练
              结束后把它们补写进这个 checkpoint。
        """
        raise NotImplementedError

    def check_early_stopping(self):
        """判断该不该提前收工：验证 loss 连续 patience 个 epoch 没创新低就停。

        返回:
            bool，True 表示早停条件达成（fit 应 break）。

        提示:
            - 一行核心逻辑：self.early_stopping_enabled and
              self.epochs_no_improve >= self.patience；
            - 别拿 train_loss 当早停依据——训练 loss 一路降到底恰恰是过拟合，
              验证 loss 才是泛化能力的信号。
        """
        raise NotImplementedError

    def _make_scheduler(self):
        """按 config["train"]["lr_scheduler"] 创建学习率调度器（self.optimizer 已就位）。

        返回:
            torch.optim.lr_scheduler 实例，或 None（type 为 "none" 时）。

        提示:
            - "step": optim.lr_scheduler.StepLR(self.optimizer,
                      step_size=配置里的 step_size, gamma=配置里的 gamma)；
            - "none" 或缺省: 返回 None，fit 里记得判空再 step()；
            - 本项目只支持这两种（调度器不是时序的主角）；
              未知 type 抛 ValueError，把合法选项写进报错信息。
        """
        raise NotImplementedError
