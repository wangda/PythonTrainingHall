"""评估与可视化（已完整实现）：测试集 accuracy + 分类报告 + 混淆矩阵 + 错分样本图。

用法（train.py 跑完、模型已保存后）：
    python evaluate.py

铁律：官方测试集（1 万张）只在这里用一次——调参和选策略看的是训练时的验证集分数。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from features import build_transforms, make_loaders
from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def load_checkpoint(path, device):
    """加载 train.py 保存的模型包，并重建可推理的模型。

    返回:
        (model, bundle): model 已加载权重并切到 eval 模式；
        bundle 含训练时的 config 快照和 meta（复现预处理用 bundle["config"]，
        而不是当前 config.yaml——训练后改了配置也不影响评估的复现）。
    """
    bundle = torch.load(path, map_location=device, weights_only=False)
    # 重建结构时关掉预训练下载：真正的权重在 checkpoint 里，load_state_dict 会覆盖；
    # 若按原 config 的 pretrained=True 重建，会白白联网下载 45MB（离线时直接报错）
    import copy
    cfg = copy.deepcopy(bundle["config"])
    cfg["model"]["pretrained"] = False
    model = create_model(cfg)                  # 按训练时的 config 快照重建结构
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    return model, bundle


def predict_all(model, loader, device):
    """在整个 loader 上推理，返回 (y_true, y_pred, y_proba) 三个 numpy 数组。"""
    ys, preds, probas = [], [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            proba = torch.softmax(logits, dim=1)   # 评估展示用 softmax 没问题
                                                   #（只有喂 CrossEntropyLoss 时禁止）
            ys.append(y.numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probas.append(proba.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probas)


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """画 10 类混淆矩阵热力图（行=真实，列=预测）。对角线外深色格 = 最难分的类别对。"""
    from sklearn.metrics import confusion_matrix

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, classes, rotation=45, ha="right")
    ax.set_yticks(labels, classes)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("CIFAR-10 混淆矩阵（行=真实，列=预测）")
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"混淆矩阵已保存到 {save_path}")


def save_misclassified(model, config, classes, save_dir, max_n=16):
    """从测试集里翻出被错分的样本，存成网格图（项目文档 §6 任务 9）。

    注意：这里直接用 build_transforms 重建评估 transform——内容确定性，
    与训练时 bundle 里的 config 等价（评估版无随机性）。
    """
    from data.download import load_dataset

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
    _, test_set, _ = load_dataset(config)
    test_set.transform = build_transforms(config, train=False)

    wrong = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(len(test_set)):
            x, y = test_set[i]
            pred = model(x.unsqueeze(0).to(device)).argmax(dim=1).item()
            if pred != y:
                wrong.append((i, y, pred))
            if len(wrong) >= max_n:
                break
    if not wrong:
        print("前若干张没有错分样本，模型太强了")
        return

    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    for ax, (idx, y, pred) in zip(np.ravel(axes), wrong):
        ax.imshow(test_set.data[idx])    # 原始 32×32 像素（未增强），直接看
        ax.set_title(f"真:{classes[y]} 猜:{classes[pred]}", fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[len(wrong):]:
        ax.axis("off")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "misclassified.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"错分样本图已保存到 {path}（前 {len(wrong)} 张错分图）")


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 迁移学习：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, bundle = load_checkpoint(model_path, device)

    # 测试集 loader：用 bundle 里的 config 快照复现预处理（训练后改了 config.yaml 也不怕）
    _, _, test_loader = make_loaders(bundle["config"])
    y_true, y_pred, _ = predict_all(model, test_loader, device)

    from sklearn.metrics import accuracy_score, classification_report
    print(f"测试集 accuracy: {accuracy_score(y_true, y_pred):.4f}"
          f"（训练时验证集最优: {bundle['meta'].get('best_val_acc', float('nan')):.4f}）")
    print(classification_report(y_true, y_pred, target_names=bundle["meta"]["classes"]))

    figures_dir = PROJECT_ROOT / config["evaluate"]["figures_dir"]
    plot_confusion_matrix(y_true, y_pred, bundle["meta"]["classes"],
                          figures_dir / "confusion_matrix.png")
    save_misclassified(model, bundle["config"], bundle["meta"]["classes"], figures_dir)


if __name__ == "__main__":
    main()
