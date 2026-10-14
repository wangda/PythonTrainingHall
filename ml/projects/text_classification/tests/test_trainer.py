"""trainer.py 的单元测试：用"剧本验证"快测三个工程组件（秒级，不真训）。

技巧：子类覆写 _train_one_epoch / _validate，把验证 loss 按剧本播出——
让 checkpoint / 早停 / lr 调度在可控场景下触发，断言行为而不是碰运气。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`
"""

import numpy as np
import pytest
import torch

from model import create_model
from trainer import Trainer

CONFIG = {
    "features": {"max_len": 8, "vocab_size": 20},
    "model": {"d_model": 16, "nhead": 2, "num_layers": 1,
              "dim_feedforward": 16, "dropout": 0.0, "num_classes": 4},
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


def _make_dataloaders(n=12):
    ids = torch.randint(3, 20, (n, 8))
    mask = torch.zeros(n, 8, dtype=torch.bool)
    labels = torch.randint(0, 4, (n,))
    ds = torch.utils.data.TensorDataset(ids, mask, labels)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    return loader, loader


class ScriptedTrainer(Trainer):
    """按剧本播 (val_loss, val_acc) 的 Trainer：train 恒 0.5。"""

    def __init__(self, *args, scripted_val, **kwargs):
        super().__init__(*args, **kwargs)
        self.scripted_val = list(scripted_val)

    def _train_one_epoch(self):
        return 0.5

    def _validate(self):
        if self.scripted_val:
            return self.scripted_val.pop(0)
        return 9.9, 0.0


def _build(tmp_path, scripted_val, es_enabled=True, scheduler=None):
    train_cfg = {**CONFIG["train"], "model_dir": str(tmp_path),
                 "early_stopping": {"enabled": es_enabled, "patience": 3}}
    if scheduler is not None:
        train_cfg["lr_scheduler"] = scheduler
    config = {**CONFIG, "train": train_cfg}
    model = create_model(config)
    train_loader, val_loader = _make_dataloaders()
    return ScriptedTrainer(model, train_loader, val_loader, config,
                           scripted_val=list(scripted_val)), config


def test_checkpoint_存的是验证最优不是最后(tmp_path):
    """剧本 [(0.5,.8), (0.4,.85), (0.45,.82), ...]：最优在第 2 epoch，checkpoint 必须记着它。"""
    script = [(0.5, 0.80), (0.4, 0.85), (0.45, 0.82), (0.5, 0.83), (0.55, 0.84), (0.6, 0.83), (0.6, 0.83)]
    trainer, _ = _build(tmp_path, scripted_val=script)
    trainer.fit()
    ckpt = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2                       # 最优 epoch，不是最后一个
    assert np.isclose(ckpt["val_loss"], 0.4)
    assert np.isclose(ckpt["val_acc"], 0.85)
    assert "model_state_dict" in ckpt and "config" in ckpt
    assert trainer.best_epoch == 2


def test_早停_patience耗尽触发(tmp_path):
    script = [(0.5, 0.8), (0.4, 0.85), (0.45, 0.82), (0.5, 0.83), (0.55, 0.84), (0.6, 0.83), (0.6, 0.83)]
    trainer, _ = _build(tmp_path, scripted_val=script)
    history = trainer.fit()
    assert len(history["train_loss"]) == 5          # 第 5 epoch 播完即停（0.4 后连续 3 次未创新低）
    assert history["val_acc"] == [0.80, 0.85, 0.82, 0.83, 0.84]


def test_早停关闭_跑满全部epoch(tmp_path):
    trainer, config = _build(tmp_path, scripted_val=[(0.5, 0.8)] * 8, es_enabled=False)
    history = trainer.fit()
    assert len(history["train_loss"]) == config["train"]["epochs"]


def test_调度器_按config生效(tmp_path):
    """StepLR(step_size=2, gamma=0.5)：每 2 个 epoch lr 减半，history 里可核对。"""
    trainer, _ = _build(tmp_path, scripted_val=[(0.5, 0.8)] * 6,
                        es_enabled=False, scheduler={"type": "step", "step_size": 2, "gamma": 0.5})
    history = trainer.fit()
    lrs = history["lr"]
    assert lrs[0] == pytest.approx(0.01)
    assert lrs[2] == pytest.approx(0.005)
    assert lrs[4] == pytest.approx(0.0025)


def test_调度器_cosine生效(tmp_path):
    """CosineAnnealingLR(t_max=10)：lr 沿余弦曲线下降（起点 0.01，epoch 2 ≈ 0.00976）。"""
    trainer, _ = _build(tmp_path, scripted_val=[(0.5, 0.8)] * 6,
                        es_enabled=False, scheduler={"type": "cosine", "t_max": 10})
    history = trainer.fit()
    assert history["lr"][0] == pytest.approx(0.01)
    assert history["lr"][1] == pytest.approx(0.01 * (1 + np.cos(np.pi / 10)) / 2, abs=1e-4)
    assert history["lr"][1] < history["lr"][0]


def test_调度器_未知类型报错(tmp_path):
    config = {**CONFIG, "train": {**CONFIG["train"], "model_dir": str(tmp_path),
                                  "lr_scheduler": {"type": "warp"}}}
    model = create_model(config)
    train_loader, val_loader = _make_dataloaders()
    with pytest.raises(ValueError, match="step / cosine / none"):
        Trainer(model, train_loader, val_loader, config)
