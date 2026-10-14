"""工程化 Trainer（骨架）——本项目的灵魂，你的主战场。

配套文档《08-FashionMNIST分类项目文档.md》§5 任务 2；原理见前置文档 §7（训练
循环五步法）与 §8（工程化 Trainer：checkpoint / 早停 / lr 调度 / 三集合）。

为什么需要 Trainer 而不是一坨裸循环：
    手写训练循环五个 epoch 看不出问题，但真实训练动辄几十上百 epoch——
    "训到一半验证集已经开始退步了，后面全是浪费时间还过拟合"、"第 37 个 epoch
    验证集最好，但程序只留了最后一个 epoch 的权重"、"学习率从头到尾不变，后期
    在最优点附近来回跳"。这三个问题的工程解法就是本类的三个核心组件：

    ① save_checkpoint：验证集最优时存 models/best.pt——是【最优】不是【最后】。
       最后一个 epoch 往往已经过拟合，存它等于白瞎了验证集存在的意义；
    ② check_early_stopping：验证 loss 连续 patience 个 epoch 不创新低就停。
       省时间只是表面收益，本质收益是"把模型停在泛化最好的位置"；
    ③ lr 调度：按 config 选 StepLR / CosineAnnealingLR / 无。前期大步快跑，
       后期小步微调，是收敛质量的直接来源。

用法（train.py 已串好）：
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit()   # 返回每 epoch 的 train_loss/val_loss/val_acc 序列
"""

from pathlib import Path

import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parent

# Fashion-MNIST + MLP 的实测参考（config 默认参数 hidden [128,64] / lr 1e-3 /
# batch 128 / StepLR(10, 0.5) / patience 5，详见项目文档 §5）：
#   早停在第 25 epoch 触发，验证集最优在第 20 epoch（val_loss=0.2801）；
#   最终测试集 accuracy 0.8878，macro F1 0.8873；
#   关掉早停跑满 50 epoch：train_loss 一路降到 0.126，val_loss 从 0.2801 回升到
#   0.2991——val loss 触底回升就是过拟合的现场。


class Trainer:
    """把"训练循环 + 三个工程组件"封装成一个类。"""

    def __init__(self, model, train_loader, val_loader, config):
        """组装训练所需的一切（已完整实现——初始化只是记账，没有教学点）。

        参数:
            model:        create_model 返回的 nn.Module（输出 logits，未接 softmax）；
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

        self.criterion = nn.CrossEntropyLoss()   # 内含 log_softmax，模型只输出 logits
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

    # ---------- 训练循环的两个基本动作（已实现，前置文档 §7 五步法） ----------

    def _train_one_epoch(self):
        """在训练集上跑一个 epoch，返回平均 train loss。

        每个 batch 的五步（前置文档 §7）：
            ① optimizer.zero_grad()  清上一步的梯度（PyTorch 默认累加）；
            ② logits = model(X)      前向传播；
            ③ loss = criterion(logits, y)  算损失；
            ④ loss.backward()        autograd 反向传播；
            ⑤ optimizer.step()       更新参数。
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
        """在验证集上评估，返回 (平均 loss, accuracy)。不反向传播，不更新参数。"""
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                total_loss += self.criterion(logits, y).item() * len(X)
                correct += (logits.argmax(dim=1) == y).sum().item()
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
            2. _validate() 拿到 val_loss / val_acc，记入 history（lr 也记）；
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
            "epoch":            当前 epoch 编号（定位"最优发生在哪一轮"）；
            "model_state_dict": self.model.state_dict()（只存权重，不存整个对象）；
            "val_loss"/"val_acc": 该 epoch 的验证集表现；
            "config":           self.config（预测时要靠它重建模型结构、复现预处理）。

        提示:
            - 先 self.ckpt_dir.mkdir(parents=True, exist_ok=True) 再存；
            - 为什么存 state_dict 而不是整个 model：后者 pickle 了类定义，
              换个目录结构就加载失败；state_dict 是纯数据，配合 create_model
              重建结构再 load_state_dict，走到哪都能活；
            - 常见错误：在 fit 结束后无脑存最后一个 epoch——那往往已轻微过拟合，
              验证集最低点在第 15 epoch，你存的却是第 30 epoch 的权重。
        """
        raise NotImplementedError

    def check_early_stopping(self):
        """判断该不该提前收工：验证 loss 连续 patience 个 epoch 没创新低就停。

        返回:
            bool，True 表示早停条件达成（fit 应 break）。

        提示:
            - 一行核心逻辑：self.early_stopping_enabled and
              self.epochs_no_improve >= self.patience；
            - epochs_no_improve 的更新在 fit 里：创新低归零，否则 +1；
            - 别拿 train_loss 当早停依据——训练 loss 可以一路降到底，恰恰是在
              过拟合；验证 loss 才是泛化能力的信号（前置文档 §8）。
        """
        raise NotImplementedError

    def _make_scheduler(self):
        """按 config["train"]["lr_scheduler"] 创建学习率调度器（self.optimizer 已就位）。

        返回:
            torch.optim.lr_scheduler 实例，或 None（type 为 "none" 时）。

        提示:
            - "step":   optim.lr_scheduler.StepLR(self.optimizer,
                        step_size=配置里的 step_size, gamma=配置里的 gamma)
              ——每 step_size 个 epoch，lr 乘 gamma；
            - "cosine": optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                        T_max=配置里的 t_max)——按余弦曲线平滑降到 0；
            - "none" 或缺省: 返回 None，fit 里记得判空再 step()；
            - 未知 type 抛 ValueError，把合法选项写进报错信息。
        """
        raise NotImplementedError
