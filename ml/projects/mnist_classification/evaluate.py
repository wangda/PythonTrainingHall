"""评估与可视化（骨架）：分类指标 + 混淆矩阵热力图。

指标含义见前置文档 §7（准确率 / 精确率 / 召回率 / F1），这里直接用
sklearn.metrics 的官方实现；混淆矩阵用 matplotlib imshow 画热力图。

用法（train.py 跑完、模型已保存后）：
    python evaluate.py
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from data.download import load_dataset
from features import build_features, split_data

PROJECT_ROOT = Path(__file__).resolve().parent


def load_model(path):
    """加载 train.py 保存的模型包（已完整实现）。

    参数:
        path: str 或 Path，.joblib 文件路径。

    返回:
        dict，含两个键：
            "model": 训练好的 sklearn 模型；
            "meta":  dict，数据集元信息（dataset / image_shape / pixel_max /
                     classes / normalize），复现预处理要用。
    """
    return joblib.load(path)


def evaluate(model, X_test, y_test, meta=None, config=None):
    """在测试集上计算分类指标（accuracy + macro 精确率/召回率/F1）。

    参数:
        model:  训练好的 sklearn 分类模型（有 predict 方法）。
        X_test: np.ndarray，形状 (n_test, d)，归一化后的测试特征。
        y_test: np.ndarray，形状 (n_test,)，测试集真实标签。
        meta:   dict，可选；提供时额外打印 classification_report，并在 config
                也提供时把混淆矩阵热力图保存到 config["evaluate"]["figures_dir"]。
        config: dict，可选，见上。

    返回:
        dict，四个键（前置文档 §7 的指标定义，手写练习里你用 NumPy 推过一遍）：
            "accuracy":        准确率，类别均衡时的首选总指标；
            "precision_macro": 宏平均精确率（各类等权，不被大类主导）；
            "recall_macro":    宏平均召回率；
            "f1_macro":        宏平均 F1。

    提示:
        - y_pred = model.predict(X_test)；
        - 指标用 sklearn.metrics 的 accuracy_score / precision_score /
          recall_score / f1_score，后三个传 average="macro"；
        - 打印报告用 classification_report(y_test, y_pred)；
        - config 与 meta 都不为 None 时调用本文件的 plot_confusion_matrix，
          图片存到 PROJECT_ROOT / config["evaluate"]["figures_dir"]。
    """
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """画混淆矩阵热力图：行=真实类别，列=预测类别。

    参数:
        y_true:    np.ndarray，形状 (n,)，真实标签。
        y_pred:    np.ndarray，形状 (n,)，预测标签。
        classes:   list，类别列表（[0..9]），决定坐标轴刻度。
        save_path: str 或 Path，图片保存路径（父目录不存在要创建）。

    返回:
        None，图片保存到 save_path。

    提示:
        - 矩阵用 sklearn.metrics.confusion_matrix(y_true, y_pred, labels=classes)；
        - plt.imshow(cm, cmap="Blues")，在格子里标注数字（ax.text），
          对角线越深越好；对角线外的深色格就是"最容易混淆的数字对"；
        - 保存用 fig.savefig(save_path, dpi=150)，记得 plt.close() 释放内存。
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="手写数字分类：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    bundle = load_model(model_path)
    X, y, meta = load_dataset(config)
    X, y = build_features(X, y, meta, config)
    _, X_test, _, y_test = split_data(X, y, config)  # 同 random_state → 与训练时完全一致的划分

    metrics = evaluate(bundle["model"], X_test, y_test, meta, config)
    for k, v in metrics.items():
        print(f"{k:>16s}: {v:.4f}")


if __name__ == "__main__":
    main()
