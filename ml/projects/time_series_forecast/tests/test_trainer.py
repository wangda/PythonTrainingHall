"""trainer.py 的单元测试：用"剧本验证 loss"快测三个工程组件（秒级，不真训）。

技巧：子类覆写 _train_one_epoch / _validate，把验证 loss 按剧本播出——
让 checkpoint / 早停 / lr 调度在可控场景下触发，断言行为而不是碰运气。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import numpy as np
import pytest
import torch
from torch import nn

from model import create_model
from trainer import Trainer

CONFIG = {
    "data": {"horizon": 2},
    "model": {"input_size": 3, "hidden_size": 4, "num_layers": 1, "dropout": 0.0},
    "train": {
        "lr": 0.01,
        "weight_decay": 0.0,
        "epochs": 20,
        "device": "cpu",
        "early_stopping": {"enabled": True, "patience": 3},
        "lr_scheduler": {"type": "step", "step_size": 2, "gamma": 0.5},
        "model_dir": None,        # 测试里由 tmp_path 注入（绝对路径）
        "model_filename": "best.pt",
    },
}


def _make_dataloaders(n=16):
    X = torch.randn(n, 5, 3)
    y = torch.randn(n, 2)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    return loader, loader


class ScriptedTrainer(Trainer):
    """按剧本播 val_loss 的 Trainer：train 恒 0.5，val 逐 epoch 取 scripted 列表。"""

    def __init__(self, *args, scripted_val, **kwargs):
        super().__init__(*args, **kwargs)
        self.scripted_val = list(scripted_val)

    def _train_one_epoch(self):
        return 0.5

    def _validate(self):
        return self.scripted_val.pop(0) if self.scripted_val else 9.9


def _build(tmp_path, scripted_val, es_enabled=True):
    config = {**CONFIG, "train": {**CONFIG["train"],
                                  "model_dir": str(tmp_path),
                                  "early_stopping": {"enabled": es_enabled, "patience": 3}}}
    model = create_model(config)
    train_loader, val_loader = _make_dataloaders()
    return ScriptedTrainer(model, train_loader, val_loader, config,
                           scripted_val=list(scripted_val)), config


def test_checkpoint_存的是验证最优不是最后(tmp_path):
    """剧本 [0.5, 0.4, 0.45, ...]：最优在第 2 epoch（0.4），checkpoint 必须记着它。"""
    trainer, _ = _build(tmp_path, scripted_val=[0.5, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6, 0.6])
    trainer.fit()
    ckpt = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2                       # 最优 epoch，不是最后一个
    assert np.isclose(ckpt["val_loss"], 0.4)
    assert "model_state_dict" in ckpt and "config" in ckpt
    assert trainer.best_epoch == 2


def test_早停_patience耗尽触发(tmp_path):
    """剧本 [0.5, 0.4, 0.45, 0.5, 0.55, 0.6, ...]：patience=3，第 5 epoch 应停。"""
    trainer, _ = _build(tmp_path, scripted_val=[0.5, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6, 0.6])
    history = trainer.fit()
    assert len(history["train_loss"]) == 5          # 第 5 epoch 播完即停（0.55 后连续 3 次未创新低）
    assert history["val_loss"] == [0.5, 0.4, 0.45, 0.5, 0.55]


def test_早停关闭_跑满全部epoch(tmp_path):
    trainer, config = _build(tmp_path, scripted_val=[0.5] * 8, es_enabled=False)
    history = trainer.fit()
    assert len(history["train_loss"]) == config["train"]["epochs"]


def test_调度器_按config生效(tmp_path):
    """StepLR(step_size=2, gamma=0.5)：每 2 个 epoch lr 减半，history 里可核对。"""
    trainer, _ = _build(tmp_path, scripted_val=[0.5] * 5, es_enabled=False)
    history = trainer.fit()
    lrs = history["lr"]
    assert lrs[0] == pytest.approx(0.01)
    assert lrs[2] == pytest.approx(0.005)           # 2 个 epoch 后减半
    assert lrs[4] == pytest.approx(0.0025)


def test_调度器_未知类型报错(tmp_path):
    trainer, config = _build(tmp_path, scripted_val=[0.5])
    bad = {**config, "train": {**config["train"],
                               "lr_scheduler": {"type": "cosine"}}}
    model = create_model(bad)
    train_loader, val_loader = _make_dataloaders()
    with pytest.raises(ValueError, match="step / none"):
        Trainer(model, train_loader, val_loader, bad)
