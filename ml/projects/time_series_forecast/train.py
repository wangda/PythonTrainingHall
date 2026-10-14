"""训练入口（流程框架已写好，features 与 Trainer 待你在对应模块实现后即可跑通）。

用法：
    python train.py                  # 用默认 config.yaml 训练
    python train.py --config my.yaml # 换一份配置做实验（如改 window_size）

流程：加载配置 → 固定随机种子 → 加载数据 → 窗口化+DataLoader → 建模型
     → Trainer.fit() → 把 scaler 补写进 checkpoint → 保存 history.json。
最优权重由 Trainer 在训练中存到 models/best.pt（验证集最优，不是最后一个 epoch）。
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.download import load_raw_data
from features import build_datasets
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
    """把每 epoch 的 train_loss/val_loss/lr 序列存成 json（画训练曲线用）。"""
    path = PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["history_filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="ETTh1 时序预测：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    # 1. 加载配置 + 固定种子
    config = load_config(args.config)
    set_seed(config["data"]["random_state"])

    # 2. 加载数据（data/download.py 已实现：有缓存读缓存，没有自动三通道下载）
    df, meta = load_raw_data(config)
    print(f"数据来源: {meta['source']}，{df.shape[0]} 个小时点 × {len(meta['feature_cols'])} 路传感器")

    # 3. 特征工程 —— TODO: 实现 features 的 split_by_time / make_windows /
    #    fit_scaler_on_train / build_datasets（窗口化 + 时序泄漏防护是核心）
    train_loader, val_loader, test_loader, scaler = build_datasets(df, config)

    # 4. 建模型（model.create_model 已实现）
    model = create_model(config)
    print(f"模型: LSTM(hidden={config['model']['hidden_size']}, "
          f"layers={config['model']['num_layers']})，参数量 {count_parameters(model):,}")

    # 5. 训练 —— TODO: 实现 trainer.Trainer 的 fit / save_checkpoint /
    #    check_early_stopping / _make_scheduler
    trainer = Trainer(model, train_loader, val_loader, config)
    print(f"设备: {trainer.device}")
    history = trainer.fit()
    save_history(history, config)

    # 6. 把 scaler 补写进 checkpoint（预测时必须复现训练段统计量的标准化）
    ckpt = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    ckpt["scaler"] = scaler
    ckpt["feature_cols"] = list(config["data"]["feature_cols"])
    ckpt["target_col"] = config["data"]["target_col"]
    torch.save(ckpt, trainer.ckpt_path)
    print(f"训练结束：共 {len(history['train_loss'])} epoch，"
          f"验证集最优在第 {trainer.best_epoch} epoch（val_loss={trainer.best_val_loss:.4f}），"
          f"scaler 已补写进 {trainer.ckpt_path}")


if __name__ == "__main__":
    main()
