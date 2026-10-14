"""训练入口（流程框架已写好，Trainer/fit 与 build_dataloaders 待你在对应模块实现）。

用法：
    python train.py                  # 用默认 config.yaml 训练
    python train.py --config my.yaml # 换一份配置做实验

流程：加载配置 → 固定随机种子 → 加载数据 → DataLoader → 建模型 → Trainer.fit()
     → 保存 history.json。最优权重在训练过程中由 Trainer 存到 models/best.pt。
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.download import load_raw_data
from features import build_dataloaders
from model import create_model, count_parameters
from trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """加载 yaml 配置文件（已完整实现）。"""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """固定全部随机源：python / numpy / torch（含 cuda）。

    torch 的随机性藏得比 sklearn 深——权重初始化、DataLoader shuffle、
    dropout 各吃各的种子，只固定一个等于没固定。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history, config):
    """把每 epoch 的 train_loss/val_loss/val_acc 序列存成 json（画训练曲线用）。"""
    path = PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["history_filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST 分类：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    # 1. 加载配置 + 固定种子：改实验不改代码，且结果可复现
    config = load_config(args.config)
    set_seed(config["data"]["random_state"])

    # 2. 加载数据（data/download.py 已实现：有缓存读缓存，没有自动双通道下载）
    x_train, y_train, x_test, y_test, meta = load_raw_data(config)
    print(f"数据集: {meta['dataset']}，训练 {x_train.shape}，测试 {x_test.shape}")

    # 3. 特征工程 —— TODO: 实现 features.build_dataloaders（三集合！）
    train_loader, val_loader, test_loader = build_dataloaders(
        x_train, y_train, x_test, y_test, config)
    print(f"三集合: train {len(train_loader.dataset)} / "
          f"val {len(val_loader.dataset)} / test {len(test_loader.dataset)}")

    # 4. 建模型（model.create_model 已实现）
    model = create_model(config)
    print(f"模型: {config['model']['hidden_sizes']}，参数量 {count_parameters(model):,}")

    # 5. 训练 —— TODO: 实现 trainer.Trainer 的 fit / save_checkpoint /
    #    check_early_stopping / _make_scheduler
    trainer = Trainer(model, train_loader, val_loader, config)
    print(f"设备: {trainer.device}")
    history = trainer.fit()
    save_history(history, config)
    print(f"训练结束：共 {len(history['train_loss'])} epoch，"
          f"验证集最优在第 {trainer.best_epoch} epoch（val_loss={trainer.best_val_loss:.4f}），"
          f"权重已存 {trainer.ckpt_path}")


if __name__ == "__main__":
    main()
