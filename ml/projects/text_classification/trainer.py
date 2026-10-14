"""工程化 Trainer（骨架）——与 fashion_mnist / time_series_forecast 项目同构。

配套文档《14-文本分类项目文档.md》§5 任务 2；原理见前置文档 §7（训练循环五步法）
与模块 4 文档 §8（checkpoint / 早停 / lr 调度）。三个工程组件与前两个项目完全同构：

    ① save_checkpoint：验证集最优时存 models/best.pt——是【最优】不是【最后】；
    ② check_early_stopping：验证 loss 连续 patience 个 epoch 不创新低就停；
    ③ lr 调度：本项目支持 step / cosine / none（Transformer 常用 cosine 平滑衰减）。

与回归版（time_series_forecast）的差异：
    - criterion 是 CrossEntropyLoss（多分类，前置文档 §7 的训练循环）；
    - 验证除了 loss 还算 accuracy（分类任务有"准确率"这个直观指标）。

用法（train.py 已串好）：
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit()   # 返回每 epoch 的 train_loss/val_loss/val_acc/lr 序列
"""

from pathlib import Path

import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parent

# AG News + TransformerEncoder 的实测参考（config 默认参数：训练子集 20000 / max_len 64 /
# vocab 20000 / d_model 128 / nhead 4 / layers 2 / lr 5e-4 / cosine(t_max=10) /
# patience 3，seed=42，详见项目文档 §5）：
#   早停在第 7 epoch 触发，验证集最优在第 4 epoch（val_loss=0.4886，val_acc=0.8215）；
#   测试集 accuracy 0.8261 / macro F1 0.8263；TF-IDF+LogisticRegression 对照 0.8963
#   （2 万训练样本下经典特征工程更强——小数据上 Transformer 优势出不来，项目文档 §5 任务 3）。
#   最容易混淆的类对：Business ↔ Sci/Tech（282/195 次，混淆矩阵里的最大非对角块）。


class Trainer:
    """把"训练循环 + 三个工程组件"封装成一个类。"""

    def __init__(self, model, train_loader, val_loader, config):
        """组装训练所需的一切（已完整实现——初始化只是记账，没有教学点）。

        参数:
            model:        create_model 返回的 nn.Module（输出 (batch, num_classes) logits）；
            train_loader: 训练集 DataLoader（shuffle=True），batch 是 (ids, padding_mask, labels)；
            val_loader:   验证集 DataLoader（不 shuffle）；
            config:       dict，读取 config["train"] 下的 lr / weight_decay / epochs /
                          device / early_stopping / lr_scheduler / model_dir / model_filename。
        """
        tcfg = config["train"]
        self.config = config
        self.device = self._resolve_device(tcfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()   # 多分类标准损失（内部已含 softmax）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg.get("weight_decay", 0.0),
        )
        self.scheduler = self._make_scheduler()

        self.epochs = tcfg["epochs"]
        es = tcfg.get("early_stopping", {})
        self.early_stopping_enabled = es.get("enabled", True)
        self.patience = es.get("patience", 3)

        # 训练过程记账：best_* 记录验证集最优，history 记录每 epoch 序列
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

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

        每个 batch 的五步（前置文档 §7）：清零梯度 → 前向 → 算交叉熵 → 反向 → 更新。
        注意 batch 是三元组 (ids, padding_mask, labels)，padding_mask 必须传给模型。
        """
        self.model.train()
        total_loss = 0.0
        for ids, mask, labels in self.train_loader:
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(ids, padding_mask=mask), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(ids)
        return total_loss / len(self.train_loader.dataset)

    def _validate(self):
        """在验证集上评估，返回 (平均 loss, accuracy)。不反向传播，不更新参数。"""
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for ids, mask, labels in self.val_loader:
                ids, mask, labels = ids.to(self.device), mask.to(self.device), labels.to(self.device)
                logits = self.model(ids, padding_mask=mask)
                total_loss += self.criterion(logits, labels).item() * len(ids)
                correct += (logits.argmax(dim=1) == labels).sum().item()
        n = len(self.val_loader.dataset)
        return total_loss / n, correct / n

    # ---------- 你的三个任务 ----------

    def fit(self):
        """训练主循环：每 epoch 训练 + 验证 + 记账 + 三个组件联动。

        返回:
            dict，四个键（每 epoch 一个值，长度 = 实际训练的 epoch 数）：
                "train_loss": list[float]，训练集平均交叉熵；
                "val_loss":   list[float]，验证集平均交叉熵（早停/checkpoint 的依据）；
                "val_acc":    list[float]，验证集准确率；
                "lr":         list[float]，该 epoch 使用的学习率。

        每 epoch 的正确顺序（顺序错了三个组件全是摆设）：
            1. _train_one_epoch() 训练一轮；
            2. _validate() 拿到 (val_loss, val_acc)，与 lr 一起记入 history；
            3. val_loss 创历史新低 → 更新 best_val_loss / best_epoch /
               epochs_no_improve 归零，并调 save_checkpoint()；
               否则 epochs_no_improve += 1；
            4. 有 scheduler 就 scheduler.step()；
            5. 打印一行进度（epoch / train_loss / val_loss / val_acc / lr）；
            6. check_early_stopping() 为真 → 打印提示并 break。
        """
        raise NotImplementedError

    def save_checkpoint(self):
        """把【当前】模型权重存到 models/best.pt——只在验证集最优时被 fit 调用。

        保存内容（torch.save 一个 dict）：
            "epoch":            当前 epoch 编号；
            "model_state_dict": self.model.state_dict()；
            "val_loss":         该 epoch 的验证集交叉熵；
            "val_acc":          该 epoch 的验证集准确率；
            "config":           self.config（预测时要靠它重建模型结构）。

        提示:
            - 先 self.ckpt_dir.mkdir(parents=True, exist_ok=True) 再存；
            - vocab 不在这里存——train.py 训练结束后把词表单独存成 vocab.json
              并补写进 checkpoint（词表和模型是一对，缺一个预测就是错的）。
        """
        raise NotImplementedError

    def check_early_stopping(self):
        """判断该不该提前收工：验证 loss 连续 patience 个 epoch 没创新低就停。

        返回:
            bool，True 表示早停条件达成（fit 应 break）。

        提示:
            - 一行核心逻辑：self.early_stopping_enabled and
              self.epochs_no_improve >= self.patience；
            - 别拿 train_loss 当早停依据——训练 loss 一路降到底恰恰是过拟合。
        """
        raise NotImplementedError

    def _make_scheduler(self):
        """按 config["train"]["lr_scheduler"] 创建学习率调度器（self.optimizer 已就位）。

        返回:
            torch.optim.lr_scheduler 实例，或 None（type 为 "none" 时）。

        提示:
            - "step": StepLR(step_size=配置里的 step_size, gamma=配置里的 gamma)；
            - "cosine": CosineAnnealingLR(T_max=配置里的 t_max)——Transformer
              常用的平滑衰减（前置文档 §7）；
            - "none" 或缺省: 返回 None，fit 里记得判空再 step()；
            - 未知 type 抛 ValueError，把合法选项写进报错信息。
        """
        raise NotImplementedError
