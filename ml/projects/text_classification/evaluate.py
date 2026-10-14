"""评估与可视化（骨架）：测试集 accuracy / macro F1 + 混淆矩阵热力图 + 训练曲线。

原理见前置文档 §7 与模块 2 文档 §7（分类评估指标）；四类的 class_report 用
真实类别名（World / Sports / Business / Sci-Tech）。

用法（train.py 跑完、models/best.pt 与 vocab.json 已存在后）：
    python evaluate.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.download import CLASS_NAMES, load_raw_data
from features import build_dataloaders, load_vocab
from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def load_checkpoint(path, device):
    """加载 checkpoint + 词表并重建模型（已完整实现）。

    返回:
        (model, vocab, ckpt): 模型已 load_state_dict 并 .to(device).eval()；
            vocab 从 checkpoint 记录的 vocab_path 加载（词表和模型是一对）。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    vocab = load_vocab(ckpt["vocab_path"])
    return model, vocab, ckpt


def evaluate_on_test(model, test_loader, device):
    """在测试集上评估分类指标。

    参数:
        model:       load_checkpoint 返回的模型（eval 模式）。
        test_loader: 测试集 DataLoader，batch 是 (ids, padding_mask, labels)。
        device:      torch.device。

    返回:
        dict，四个键：
            "accuracy":          float，整体准确率；
            "macro_f1":          float，四类 F1 的算术平均（类别均衡时≈accuracy）；
            "y_true" / "y_pred": np.ndarray，(n,) 的 0~3 标签（画混淆矩阵用）。

    提示:
        - 逐 batch torch.no_grad() 前向，argmax 得预测类，np.concatenate 收集；
        - accuracy / macro_f1 用 sklearn.metrics 的 accuracy_score / f1_score(average="macro")。
    """
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, save_path, class_names):
    """画混淆矩阵热力图（行 = 真实，列 = 预测）。

    提示:
        - sklearn.metrics.confusion_matrix 算矩阵；
        - plt.imshow(cm, cmap="Blues") + 每格标数字 + x/y 轴刻度用 class_names；
        - 保存用 fig.savefig(save_path)，plt.close()；
        - 看图找"最大非对角块"——那就是模型最爱的混淆对（本项目预期是
          Business ↔ Sci/Tech，项目文档 §5 任务 3）。
    """
    raise NotImplementedError


def plot_training_curves(history_path, save_path):
    """从 history.json 画 train/val loss + val_acc 双子图。

    提示:
        - plt.subplots(1, 2)：左图 train/val loss 双线（val 回升点 = 过拟合开始）；
          右图 val_acc 单线（直观的"学得怎么样"）；
        - 保存用 fig.savefig(save_path)，plt.close()。
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="AG News 文本分类：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (PROJECT_ROOT / config["train"]["model_dir"]
                  / config["train"]["model_filename"])
    model, _, _ = load_checkpoint(model_path, device)

    # 与训练时完全一致的数据管线（同 config → 同切分 → 同词表语义）
    train_texts, train_labels, test_texts, test_labels, _ = load_raw_data(config)
    _, _, test_loader, _ = build_dataloaders(
        train_texts, train_labels, test_texts, test_labels, config)

    result = evaluate_on_test(model, test_loader, device)
    print(f"测试集（{len(result['y_true'])} 条）:")
    print(f"  accuracy = {result['accuracy']:.4f}")
    print(f"  macro F1 = {result['macro_f1']:.4f}")

    figures_dir = PROJECT_ROOT / config["evaluate"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(result["y_true"], result["y_pred"],
                          figures_dir / "confusion_matrix.png", CLASS_NAMES)
    history_path = (PROJECT_ROOT / config["train"]["model_dir"]
                    / config["train"]["history_filename"])
    if history_path.exists():
        plot_training_curves(history_path, figures_dir / "training_curves.png")
    print(f"图表已保存到 {figures_dir}/")


if __name__ == "__main__":
    main()
