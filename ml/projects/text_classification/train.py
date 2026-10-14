"""训练入口（流程框架已写好，features 与 Trainer 待你在对应模块实现后即可跑通）。

用法：
    python train.py                  # 用默认 config.yaml 训练
    python train.py --config my.yaml # 换一份配置做实验（如改 d_model / max_len）

流程：加载配置 → 固定随机种子 → 加载数据 → 分词/词表/DataLoader → 建模型
     → Trainer.fit() → 存 vocab.json + 补写进 checkpoint → 保存 history.json。
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
from features import build_dataloaders, save_vocab
from model import create_model, count_parameters
from trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """加载 yaml 配置文件（已完整实现）。"""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """固定全部随机源：python / numpy / torch（含 cuda）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history, config):
    """把每 epoch 的 train_loss/val_loss/val_acc/lr 序列存成 json（画训练曲线用）。"""
    path = PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["history_filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="AG News 文本分类：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    # 1. 加载配置 + 固定种子
    config = load_config(args.config)
    set_seed(config["data"]["random_state"])

    # 2. 加载数据（data/download.py 已实现：有缓存读缓存，没有自动三通道下载）
    train_texts, train_labels, test_texts, test_labels, meta = load_raw_data(config)
    print(f"类别: {meta['class_names']}")

    # 3. 特征工程 —— TODO: 实现 features 的 build_vocab / encode / build_dataloaders
    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        train_texts, train_labels, test_texts, test_labels, config)

    # 4. 建模型（model.create_model 已实现）
    model = create_model(config)
    print(f"模型: TransformerEncoder(d_model={config['model']['d_model']}, "
          f"nhead={config['model']['nhead']}, layers={config['model']['num_layers']})，"
          f"参数量 {count_parameters(model):,}")

    # 5. 训练 —— TODO: 实现 trainer.Trainer 的 fit / save_checkpoint /
    #    check_early_stopping / _make_scheduler
    trainer = Trainer(model, train_loader, val_loader, config)
    print(f"设备: {trainer.device}")
    history = trainer.fit()
    save_history(history, config)

    # 6. 词表和模型是一对：单独存 vocab.json，并补写进 checkpoint
    vocab_path = (PROJECT_ROOT / config["train"]["model_dir"]
                  / config["train"]["vocab_filename"])
    save_vocab(vocab, vocab_path)
    ckpt = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
    ckpt["vocab_path"] = str(vocab_path)
    torch.save(ckpt, trainer.ckpt_path)
    print(f"训练结束：共 {len(history['train_loss'])} epoch，"
          f"验证集最优在第 {trainer.best_epoch} epoch"
          f"（val_loss={trainer.best_val_loss:.4f}，val_acc={history['val_acc'][trainer.best_epoch - 1]:.4f}）")


if __name__ == "__main__":
    main()
