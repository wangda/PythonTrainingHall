"""评估与可视化（骨架）：加载 best.pt 在【测试集】上评估 + 混淆矩阵 + 训练曲线。

配套文档《08-FashionMNIST分类项目文档.md》§5 任务 4；指标含义见前置文档（03 文档 §7）。
注意三件事：
    1. 评估用的是官方测试集——训练全程它没被碰过，这里的数字才是对外汇报的成绩；
    2. 加载的是验证集最优的 checkpoint（best.pt），不是训练最后一步的权重；
    3. 训练曲线从 models/history.json 读（train.py 存的），曲线上的拐点就是
       早停该停的位置——对着图能讲清"什么时候模型开始过拟合"。

用法（train.py 跑完、best.pt 已保存后）：
    python evaluate.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.download import load_raw_data
from features import build_dataloaders
from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def load_checkpoint(path, config):
    """加载 Trainer 保存的 checkpoint，重建模型并装入权重（已完整实现）。

    参数:
        path:   str 或 Path，best.pt 文件路径。
        config: dict，当前 config.yaml 的内容（模型结构以 checkpoint 里存的为准，
                这里只用来解析 device）。

    返回:
        (model, ckpt): model 是装入权重、切到 eval 模式的 nn.Module；
                       ckpt 是原始 checkpoint dict（含 epoch / val_loss / val_acc /
                       config，打印信息用）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"])   # 结构以 checkpoint 存的 config 为准
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def evaluate_on_test(model, test_loader, class_names, config):
    """在测试集上计算 accuracy + macro P/R/F1，打印 classification_report。

    参数:
        model:       装入 best.pt 权重的 nn.Module（已 eval）。
        test_loader: 测试集 DataLoader（不 shuffle）。
        class_names: list[str]，十个类别的真实名称，用于报告和混淆矩阵坐标轴。
        config:      dict，读取 config["evaluate"]["figures_dir"] 保存混淆矩阵图。

    返回:
        dict，四个键：accuracy / precision_macro / recall_macro / f1_macro。

    提示:
        - 前向用 torch.no_grad()，logits.argmax(dim=1) 得预测类别；
        - device 从 next(model.parameters()).device 取，batch 要 .to(device)；
        - 指标用 sklearn.metrics 的 accuracy_score / precision_score /
          recall_score / f1_score（average="macro"），报告用 classification_report
          （传 target_names=class_names）；
        - 收集全部 y_true / y_pred 后调本文件的 plot_confusion_matrix 存图。
    """
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """画混淆矩阵热力图：行=真实类别，列=预测类别。

    参数:
        y_true/y_pred: np.ndarray，形状 (n,)。
        class_names:   list[str]，类别名（T-shirt/top … Ankle boot）。
        save_path:     str 或 Path，图片保存路径（父目录不存在要创建）。

    返回:
        None，图片保存到 save_path。

    提示:
        - 矩阵用 sklearn.metrics.confusion_matrix(y_true, y_pred)；
        - plt.imshow(cm, cmap="Blues")，格子里用 ax.text 标数字；类别名较长，
          x 轴刻度旋转 45 度（plt.xticks(rotation=45, ha="right")）；
        - 对角线外的深色格就是最容易混淆的类别对——实测最大的几对都出在
          上半身衣物之间（Shirt / T-shirt/top / Pullover / Coat），对着图解释；
        - fig.savefig(save_path, dpi=150, bbox_inches="tight")，然后 plt.close()。
    """
    raise NotImplementedError


def plot_training_curves(history, save_path):
    """画训练/验证 loss 双线图 + 验证准确率曲线（从 history.json 读的数据）。

    参数:
        history:   dict，train.py 存的 history.json 内容，含 train_loss /
                   val_loss / val_acc 三个等长序列。
        save_path: str 或 Path，图片保存路径。

    返回:
        None，图片保存到 save_path。

    提示:
        - 左右两个子图：左边 train_loss 与 val_loss 双线（加图例），右边 val_acc；
        - 用 axvline 或散点标出 val_loss 最低点——那就是 checkpoint 存的 epoch，
          也是"再往后训就是过拟合"的分界线；
        - 这张图是 §5 任务 2 和任务 3 的分析对象：早停触发点、关早停后 val loss
          回升的幅度，全靠它说话。
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST 分类：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    parser.add_argument("--model", default=None, help="checkpoint 路径，默认取 config 里的 train.model_dir/model_filename")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    model, ckpt = load_checkpoint(model_path, config)
    print(f"加载 checkpoint：第 {ckpt['epoch']} epoch，"
          f"val_loss={ckpt['val_loss']:.4f}，val_acc={ckpt['val_acc']:.4f}")

    # 数据划分必须与训练时完全一致（同一份 config，同一个 random_state）
    x_train, y_train, x_test, y_test, meta = load_raw_data(config)
    _, _, test_loader = build_dataloaders(x_train, y_train, x_test, y_test, config)

    metrics = evaluate_on_test(model, test_loader, meta["class_names"], config)
    for k, v in metrics.items():
        print(f"{k:>16s}: {v:.4f}")

    # 训练曲线（train.py 存的 history.json）
    history_path = PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["history_filename"]
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        figures_dir = PROJECT_ROOT / config["evaluate"]["figures_dir"]
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(history, figures_dir / "training_curves.png")
        print(f"训练曲线已保存到 {figures_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
