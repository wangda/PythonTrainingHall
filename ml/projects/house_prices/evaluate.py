"""评估与可视化（骨架）：指标计算 + 残差图 + 预测 vs 真实散点图。

指标公式与手写练习 ml/algorithms/linear_regression/linear_regression.py 完全对应
（前置文档 §7），这里直接用 sklearn.metrics 的官方实现。

用法（train.py 跑完、模型已保存后）：
    python evaluate.py
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from data.download import load_raw_data
from features import build_features, split_and_scale

PROJECT_ROOT = Path(__file__).resolve().parent


def load_model(path):
    """加载 train.py 保存的模型包（已完整实现）。

    参数:
        path: str 或 Path，.joblib 文件路径。

    返回:
        dict，含三个键：
            "model":         训练好的 sklearn 模型；
            "scaler":        训练集上 fit 的 StandardScaler；
            "feature_names": list[str]，训练时的特征列顺序。
    """
    return joblib.load(path)


def evaluate(model, X_test, y_test, config=None):
    """在测试集上计算四个回归指标。

    参数:
        model:  训练好的 sklearn 模型（有 predict 方法）。
        X_test: np.ndarray，形状 (n_test, d)，标准化后的测试特征。
        y_test: np.ndarray，形状 (n_test,)，测试集真实目标。
        config: dict，可选；提供时额外把残差图、预测对比图保存到
                config["evaluate"]["figures_dir"]。

    返回:
        dict，四个键（思路和手写练习的 mse/rmse/mae/r2_score 一致）：
            "mse":  均方误差；
            "rmse": 均方根误差（单位和目标一致，汇报首选）；
            "mae":  平均绝对误差；
            "r2":   决定系数（1 完美，0 等于猜均值，< 0 比瞎猜差）。

    提示:
        - y_pred = model.predict(X_test)；
        - 指标用 sklearn.metrics 的 mean_squared_error / mean_absolute_error /
          r2_score，RMSE 对 MSE 开方即可（前置文档 §8：新版 sklearn 的
          mean_squared_error 没有 squared 参数）；
        - config 不为 None 时调用本文件的 plot_residuals / plot_prediction_vs_true，
          图片存到 PROJECT_ROOT / config["evaluate"]["figures_dir"]。
    """
    raise NotImplementedError


def plot_residuals(y_true, y_pred, save_path):
    """画残差图：预测值 vs 残差（真实 - 预测），加 y=0 参考线。

    参数:
        y_true:    np.ndarray，形状 (n,)，真实值。
        y_pred:    np.ndarray，形状 (n,)，预测值。
        save_path: str 或 Path，图片保存路径（父目录不存在要创建）。

    返回:
        None，图片保存到 save_path。

    提示:
        - 残差 resid = y_true - y_pred；
        - plt.scatter(y_pred, resid) + plt.axhline(0, color="red")；
        - 健康标准：点随机分布在 0 线上下、无形状、无喇叭口（前置文档 §7.3）；
        - 保存用 fig.savefig(save_path)，记得 plt.close() 释放内存。
    """
    raise NotImplementedError


def plot_prediction_vs_true(y_true, y_pred, save_path):
    """画"预测 vs 真实"散点图，加 y=x 对角线（完美预测的落点）。

    参数:
        y_true:    np.ndarray，形状 (n,)，真实值。
        y_pred:    np.ndarray，形状 (n,)，预测值。
        save_path: str 或 Path，图片保存路径（父目录不存在要创建）。

    返回:
        None，图片保存到 save_path。

    提示:
        - plt.scatter(y_true, y_pred)，再画一条从最小值到最大值的对角参考线；
        - 点越贴近对角线越好；整体向下弯说明高房价被系统性低估（目标有上限截断）。
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="房价预测：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    bundle = load_model(model_path)
    df = load_raw_data(PROJECT_ROOT / config["data"]["raw_path"])
    X, y = build_features(df, config)
    _, X_test, _, y_test, _ = split_and_scale(X, y, config)  # 同 random_state → 与训练时完全一致的划分

    metrics = evaluate(bundle["model"], X_test, y_test, config)
    for k, v in metrics.items():
        print(f"{k:>5s}: {v:.4f}")


if __name__ == "__main__":
    main()
