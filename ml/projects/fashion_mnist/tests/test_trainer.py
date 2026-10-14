"""trainer.py 的单元测试：checkpoint 存最优、早停触发、lr 调度生效。

用小玩具数据 + 可控的"剧本验证 loss"快速测三个组件——不等真实训练，
直接继承 Trainer 覆写 _validate 喂一串预先写好的 val loss，观察组件行为。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

骨架状态下应全部以 NotImplementedError 失败（FAIL 而非 ERROR）。
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import Trainer


def _toy_loader(n=64):
    """玩具 DataLoader：随机特征 + 随机标签，只为让训练循环能动起来。"""
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(n, 784)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 10, n), dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=32)


def _make_config(tmp_path, **overrides):
    """最小训练配置：checkpoint 指到 pytest 临时目录，1 个 epoch 起步。"""
    train = {
        "lr": 0.01, "weight_decay": 0.0, "epochs": 6, "device": "cpu",
        "early_stopping": {"enabled": True, "patience": 2},
        "lr_scheduler": {"type": "none"},
        "model_dir": str(tmp_path), "model_filename": "best.pt",
    }
    train.update(overrides)
    return {"train": train}


class ScriptedTrainer(Trainer):
    """把 _validate 换成剧本：按预定序列返回 val loss，训练动作打折扣。"""

    def __init__(self, *args, scripted_val_losses=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._script = list(scripted_val_losses)

    def _train_one_epoch(self):
        return 0.0                       # 不真训练，loss 打个折

    def _validate(self):
        return self._script.pop(0), 0.0  # 按剧本返回 val loss，acc 固定 0


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(784, 16), nn.ReLU(), nn.Linear(16, 10))


def test_checkpoint_存的是验证最优而非最后(tmp_path, model):
    """val loss 剧本：0.50 → 0.40（最优）→ 0.45 → 0.50，
    checkpoint 必须记录第 2 个 epoch，而不是最后第 4 个。"""
    config = _make_config(tmp_path, epochs=4)
    trainer = ScriptedTrainer(model, _toy_loader(), _toy_loader(), config,
                              scripted_val_losses=[0.50, 0.40, 0.45, 0.50])
    history = trainer.fit()
    assert len(history["val_loss"]) == 4
    assert trainer.best_epoch == 2
    ckpt = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2                      # 存的是验证最优的那一轮
    assert ckpt["val_loss"] == pytest.approx(0.40)


def test_早停_patience耗尽时触发(tmp_path, model):
    """val loss 剧本：0.50 → 0.40（最优）→ 0.41 → 0.42，patience=2，
    第 4 个 epoch 时已连续 2 轮不创新低，应提前收工（不再跑第 5、6 轮）。"""
    config = _make_config(tmp_path, epochs=6)
    trainer = ScriptedTrainer(model, _toy_loader(), _toy_loader(), config,
                              scripted_val_losses=[0.50, 0.40, 0.41, 0.42, 0.43, 0.44])
    history = trainer.fit()
    assert len(history["val_loss"]) == 4           # 提前停在第 4 epoch
    assert trainer.best_epoch == 2


def test_早停_关闭时跑满全程(tmp_path, model):
    config = _make_config(tmp_path, epochs=4,
                          early_stopping={"enabled": False, "patience": 1})
    trainer = ScriptedTrainer(model, _toy_loader(), _toy_loader(), config,
                              scripted_val_losses=[0.50, 0.60, 0.70, 0.80])
    history = trainer.fit()
    assert len(history["val_loss"]) == 4           # 一路变差也不停


def test_lr调度_step按config生效(tmp_path, model):
    """StepLR：lr=0.1、step_size=2、gamma=0.5，4 个 epoch 后 lr 应为 0.1×0.5×0.5。"""
    config = _make_config(tmp_path, epochs=4,
                          early_stopping={"enabled": False, "patience": 99},
                          lr_scheduler={"type": "step", "step_size": 2, "gamma": 0.5})
    config["train"]["lr"] = 0.1
    trainer = ScriptedTrainer(model, _toy_loader(), _toy_loader(), config,
                              scripted_val_losses=[0.50, 0.45, 0.40, 0.35])
    history = trainer.fit()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5 * 0.5)
    assert history["lr"] == pytest.approx([0.1, 0.1, 0.05, 0.05])


def test_lr调度_无调度时lr不变(tmp_path, model):
    config = _make_config(tmp_path, epochs=3,
                          early_stopping={"enabled": False, "patience": 99},
                          lr_scheduler={"type": "none"})
    config["train"]["lr"] = 0.01
    trainer = ScriptedTrainer(model, _toy_loader(), _toy_loader(), config,
                              scripted_val_losses=[0.5, 0.4, 0.3])
    trainer.fit()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_lr调度_未知类型报错(tmp_path, model):
    config = _make_config(tmp_path, lr_scheduler={"type": "magic"})
    with pytest.raises(ValueError):
        Trainer(model, _toy_loader(), _toy_loader(), config)
