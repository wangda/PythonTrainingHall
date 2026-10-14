"""评估与可视化（已完整实现）：分类指标 + ROC-AUC + 混淆矩阵 + 特征重要性对比。

指标定义见模块 2 前置文档 §7；类别不平衡（约 38% 获救）下重点关注
precision / recall / F1 / ROC-AUC，别只看 accuracy（前置文档 §11.7）。

用法（train.py 跑完、模型包已保存后）：
    python evaluate.py
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from data.download import load_dataset
from features import split_data, transform

PROJECT_ROOT = Path(__file__).resolve().parent


def load_bundle(path):
    """加载 train.py 保存的模型包（已完整实现）。

    返回:
        dict，三个键："model"（训练好的模型）、"state"（预处理状态）、"meta"（元信息）。
    """
    return joblib.load(path)


def evaluate(model, X_test, y_test):
    """在测试集上计算分类指标（已完整实现）。

    参数:
        model:  训练好的分类模型（需有 predict 和 predict_proba）。
        X_test: pd.DataFrame，transform 输出的测试特征。
        y_test: np.ndarray，测试集真实标签 0/1。

    返回:
        dict，五个键（模块 2 前置文档 §7 的定义，正类 = 获救）：
            "accuracy":  准确率（不平衡数据仅供参考）；
            "precision": 预测获救的人里真获救的比例；
            "recall":    真获救的人里被找出的比例；
            "f1":        精确率与召回率的调和平均；
            "roc_auc":   ROC 曲线下面积（0.5=瞎猜，1=完美）。
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    """画二分类混淆矩阵热力图（行=真实，列=预测），已完整实现。"""
    from sklearn.metrics import confusion_matrix

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["遇难", "获救"])
    ax.set_yticks([0, 1], ["遇难", "获救"])
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title("混淆矩阵（行=真实，列=预测）")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"混淆矩阵已保存到 {save_path}")


def plot_roc_curve(y_test, y_proba, save_path):
    """画 ROC 曲线（扫阈值的 (FPR, TPR) 轨迹，模块 2 前置文档 §7.4），已完整实现。"""
    from sklearn.metrics import auc, roc_curve

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="瞎猜基线")
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC 曲线")
    ax.legend()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC 曲线已保存到 {save_path}")


def plot_feature_importance(model, X_test, y_test, feature_names, save_path):
    """画特征重要性对比图：模型自带口径 vs 置换重要性（已完整实现）。

    模型自带口径（前置文档 §11.6）：
        - LightGBM：gain（总增益）与 split（分裂次数）两条；
        - sklearn 树/森林：不纯度重要性（feature_importances_）；
    置换重要性（permutation_importance，前置文档 §9.5）：在测试集上打乱某特征
    看分数掉多少——三种口径排在一起，亲眼对比排序差异。

    参数:
        model:         训练好的模型。
        X_test:        pd.DataFrame，测试特征（列名 = 特征名）。
        y_test:        np.ndarray，测试标签。
        feature_names: list[str]，特征名（取自 state["feature_names"]）。
        save_path:     图片保存路径。
    """
    from sklearn.inspection import permutation_importance

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # 模型自带重要性（归一化到和为 1，方便和置换重要性并排比较）
    builtin = {}
    booster = getattr(model, "booster_", None)
    if booster is not None:  # LightGBM
        for imp_type in ["gain", "split"]:
            imp = model.booster_.feature_importance(importance_type=imp_type)
            imp = np.asarray(imp, dtype=float)
            builtin[imp_type] = imp / imp.sum() if imp.sum() > 0 else imp
    elif hasattr(model, "feature_importances_"):  # sklearn 树 / 森林
        imp = np.asarray(model.feature_importances_, dtype=float)
        builtin["impurity"] = imp / imp.sum() if imp.sum() > 0 else imp

    perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    builtin["permutation"] = perm.importances_mean

    n = len(feature_names)
    n_bars = len(builtin)
    x = np.arange(n)
    width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5))
    for k, (label, values) in enumerate(builtin.items()):
        ax.bar(x + k * width, values, width, label=label)
    ax.set_xticks(x + width * (n_bars - 1) / 2, feature_names, rotation=45, ha="right")
    ax.set_ylabel("重要性（自带口径已归一化）")
    ax.set_title("特征重要性对比：自带口径 vs 置换重要性")
    ax.legend()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"特征重要性对比图已保存到 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Titanic 生存预测：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    bundle = load_bundle(model_path)
    df, meta = load_dataset(config)
    train_df, test_df = split_data(df, config)   # 同 random_state → 与训练时完全一致的划分
    X_test, y_test = transform(test_df, bundle["state"], config)

    model = bundle["model"]
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"{k:>12s}: {v:.4f}")

    from sklearn.metrics import classification_report
    print(classification_report(y_test, model.predict(X_test), target_names=["遇难", "获救"]))

    figures_dir = PROJECT_ROOT / config["evaluate"]["figures_dir"]
    plot_confusion_matrix(y_test, model.predict(X_test), figures_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], figures_dir / "roc_curve.png")
    plot_feature_importance(model, X_test, y_test, bundle["state"]["feature_names"],
                            figures_dir / "feature_importance.png")


if __name__ == "__main__":
    main()
