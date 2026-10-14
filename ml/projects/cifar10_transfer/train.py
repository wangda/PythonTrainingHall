"""训练入口（已完整实现的参考版——前置文档 §5 训练五步的实战样板，读熟它）。

用法：
    python train.py                  # 用默认 config.yaml 训练（resnet18 + head_only）
    python train.py --config my.yaml # 换一份配置（如 lenet 从零 / full 全量微调）

流程：加载配置 → DataLoader → 创建模型 + 优化器 → epoch 循环
     （训练五步 + 每个 epoch 末尾验证集评估）→ 保存验证集最优的权重。

产物：models/model.pt —— torch.save 的 dict，三个键：
    "model_state": 验证集最优时的 state_dict；
    "config":      训练时的 config 快照（predict/evaluate 靠它复现模型结构和预处理）；
    "meta":        数据集元信息（类别名等）。
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from features import make_loaders
from model import count_trainable_params, create_model, create_optimizer

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """加载 yaml 配置文件。"""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    """有 GPU 用 GPU，没有回退 CPU（前置文档 §5.4）。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval):
    """训练一个 epoch：前置文档 §5.2 的训练五步循环（参考实现）。

    返回:
        float，本 epoch 的平均 loss。
    """
    model.train()                        # 训练模式（前置文档 §5.2 的仪式）
    total_loss, total_n = 0.0, 0
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        logits = model(X)                # ① 前向（logits——CrossEntropyLoss 不许先 softmax）
        loss = criterion(logits, y)      # ② 损失
        optimizer.zero_grad()            # ③ 清零旧梯度（忘了它 = 经典新手 bug）
        loss.backward()                  # ④ 反向传播
        optimizer.step()                 # ⑤ 更新
        total_loss += loss.item() * len(y)
        total_n += len(y)
        if (i + 1) % log_interval == 0:
            print(f"    batch {i + 1:4d}: loss = {total_loss / total_n:.4f}")
    return total_loss / total_n


def evaluate_val(model, loader, criterion, device):
    """在验证集上评估：eval 模式 + no_grad（前置文档 §5.2 的评估仪式）。

    返回:
        (val_loss, val_acc)。
    """
    model.eval()
    total_loss, correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += len(y)
    return total_loss / total_n, correct / total_n


def save_checkpoint(model_state, config, meta):
    """保存权重 + config 快照 + meta（复现模型结构和预处理的唯一依据）。"""
    model_dir = PROJECT_ROOT / config["train"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / config["train"]["model_filename"]
    torch.save({"model_state": model_state, "config": config, "meta": meta}, path)
    print(f"模型已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 迁移学习：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    print(f"device: {device}")

    # 1. 数据 —— TODO: 实现 features.build_transforms / make_loaders
    train_loader, val_loader, _ = make_loaders(config)
    print(f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)}")

    # 2. 模型 + 优化器 —— TODO: 实现 model.create_model / create_optimizer
    model = create_model(config).to(device)
    n_trainable = count_trainable_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_trainable:,} / {n_total:,}"
          f"（{n_trainable / n_total:.1%}——冻结策略的体检表，项目文档 §5）")
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # 3. epoch 循环：训练 + 验证，保存验证集最优权重
    best_val_acc = 0.0
    best_state = None
    for epoch in range(config["train"]["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, config["train"]["log_interval"])
        val_loss, val_acc = evaluate_val(model, val_loader, criterion, device)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  ← 新的最优，已记录"
        print(f"epoch {epoch + 1}/{config['train']['epochs']}: "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time() - t0:.0f}s){marker}")

    # 4. 保存（config 快照必须一起存——evaluate/predict 靠它复现模型和预处理）
    from data.download import CLASSES
    meta = {"dataset": "cifar10", "classes": CLASSES, "best_val_acc": best_val_acc}
    save_checkpoint(best_state, config, meta)
    print(f"训练完成，最优验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
