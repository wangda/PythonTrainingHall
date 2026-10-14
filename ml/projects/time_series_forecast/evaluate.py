"""评估与可视化（骨架）：测试集 MAE/RMSE（反归一化）+ naive baseline 对比 + 两张图。

原理见前置文档 §7（多步预测与评估）：时序预测的汇报铁律——
① 指标必须反归一化回原始单位（"油温差 0.41 度"才有人看得懂，0.0284 没人懂）；
② 必须跟 naive baseline（预测 = 窗口最后一个值）比，打不过 naive 就是白干。

用法（train.py 跑完、models/best.pt 已存在后）：
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
from features import build_datasets
from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def load_checkpoint(path, device):
    """加载 checkpoint 并重建模型（已完整实现）。

    返回:
        (model, ckpt): 模型已 load_state_dict 并 .to(device)；
            ckpt 里带 scaler / feature_cols / target_col / config。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def evaluate_on_test(model, test_loader, scaler, target_col_idx, device):
    """在测试集上评估，返回（反归一化后的）MAE / RMSE 与预测、真实值数组。

    参数:
        model:          load_checkpoint 返回的模型（eval 模式）。
        test_loader:    测试集 DataLoader，batch 是 (X, y)，y 形状 (batch, horizon)。
        scaler:         训练段上 fit 的 StandardScaler（反归一化用）。
        target_col_idx: 目标列 OT 在特征列里的下标（反归一化只还原这一列）。
        device:         torch.device。

    返回:
        dict，四个键：
            "mae":  float，反归一化后的平均绝对误差；
            "rmse": float，反归一化后的均方根误差；
            "y_true": np.ndarray，(n_samples, horizon)，原始单位的真实值；
            "y_pred": np.ndarray，(n_samples, horizon)，原始单位的预测值。

    提示:
        - 模型输出 y_pred_scaled 是【标准化空间】的值。反归一化只针对目标列：
          y = y_pred_scaled * scaler.scale_[target_col_idx] + scaler.mean_[target_col_idx]
          （标准化的逆变换，别把 7 列全反了——只有 OT 那一列是目标）；
        - 收集全测试集：逐 batch torch.no_grad() 前向，np.concatenate；
        - MAE = mean(|y_true - y_pred|)，RMSE = sqrt(mean((y_true - y_pred)^2))。
    """
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            preds.append(model(X.to(device)).cpu().numpy())
            trues.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    mean, scale = scaler.mean_[target_col_idx], scaler.scale_[target_col_idx]
    y_true = y_true * scale + mean
    y_pred = y_pred * scale + mean
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "y_true": y_true,
        "y_pred": y_pred,
    }



def naive_baseline(test_loader, scaler, target_col_idx):
    """naive baseline：预测 = 窗口最后一个时间点的 OT 值（已完整实现，对照组）。

    "明天 = 今天"是时序预测最顽固的对手（前置文档 §7）。它不需要任何模型——
    直接把每个窗口最后一行的 OT 值复制 horizon 份当预测。
    """
    y_true_list, y_pred_list = [], []
    for X, y in test_loader:
        # X 的最后一行、目标列 = 窗口末尾的 OT（标准化空间）
        last_ot_scaled = X[:, -1, target_col_idx].numpy()
        y_pred_list.append(np.repeat(last_ot_scaled[:, None], y.shape[1], axis=1))
        y_true_list.append(y.numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    # 反归一化到原始单位
    mean, scale = scaler.mean_[target_col_idx], scaler.scale_[target_col_idx]
    return y_true * scale + mean, y_pred * scale + mean


def plot_prediction_vs_true(y_true, y_pred, save_path, n_points=480):
    """画测试集头部 n_points 个点的"预测 vs 真实"曲线对比图。

    参数:
        y_true / y_pred: np.ndarray，(n_samples, horizon)，原始单位（evaluate_on_test 返回）。
        save_path:       str 或 Path，图片保存路径（父目录不存在要创建）。
        n_points:        取测试集前多少个窗口（每个窗口画其第 1 步预测，折线更可读）。

    提示:
        - 真实值一条线、预测值一条虚线，加图例和轴标签（单位：度）；
        - fig.savefig 后 plt.close() 释放内存。
    """
    from pathlib import Path as _Path

    save_path = _Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n_points, len(y_true))
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:n, 0], label="true OT", lw=1)
    plt.plot(y_pred[:n, 0], label="predicted OT (+1h)", lw=1, ls="--")
    plt.xlabel("时间（小时点）")
    plt.ylabel("油温 OT（度）")
    plt.title(f"测试集预测 vs 真实（前 {n} 个窗口，每窗口取 +1h 步）")
    plt.legend()
    fig = plt.gcf()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_training_curves(history_path, save_path):
    """从 history.json 画 train/val loss 双线图（早停效果一目了然）。

    参数:
        history_path: str 或 Path，train.py 存的 history.json（含 train_loss/val_loss/lr）。
        save_path:    str 或 Path，图片保存路径。

    提示:
        - val_loss 回升的位置就是过拟合开始的地方，早停箭头可以标在最低点；
        - 保存用 fig.savefig(save_path)，plt.close()。
    """
    from pathlib import Path as _Path

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    save_path = _Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history["train_loss"], label="train MSE")
    plt.plot(epochs, history["val_loss"], label="val MSE")
    best = int(np.argmin(history["val_loss"])) + 1
    plt.axvline(best, color="red", ls=":", label=f"best epoch = {best}")
    plt.xlabel("epoch")
    plt.ylabel("MSE（标准化空间）")
    plt.title("训练 / 验证 loss 曲线（红线 = 验证最优，早停守护它）")
    plt.legend()
    fig = plt.gcf()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="ETTh1 时序预测：评估入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (PROJECT_ROOT / config["train"]["model_dir"]
                  / config["train"]["model_filename"])
    model, ckpt = load_checkpoint(model_path, device)

    # 与训练时完全一致的数据管线（同 config → 同划分 → 同 scaler 语义）
    df, _ = load_raw_data(config)
    _, _, test_loader, scaler = build_datasets(df, config)
    target_col_idx = ckpt["feature_cols"].index(ckpt["target_col"])

    result = evaluate_on_test(model, test_loader, scaler, target_col_idx, device)
    nt_true, nt_pred = naive_baseline(test_loader, scaler, target_col_idx)
    naive_mae = np.mean(np.abs(nt_true - nt_pred))
    naive_rmse = np.sqrt(np.mean((nt_true - nt_pred) ** 2))

    print(f"测试集（反归一化后，{len(result['y_true'])} 个窗口 × "
          f"{config['data']['horizon']} 步）:")
    print(f"  LSTM : MAE = {result['mae']:.4f}, RMSE = {result['rmse']:.4f}")
    print(f"  naive: MAE = {naive_mae:.4f}, RMSE = {naive_rmse:.4f}")
    win = "LSTM 胜出" if result["mae"] < naive_mae else "⚠️ 打不过 naive，模型白干"
    print(f"  对比: {win}")

    figures_dir = PROJECT_ROOT / config["evaluate"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_prediction_vs_true(result["y_true"], result["y_pred"],
                            figures_dir / "prediction_vs_true.png",
                            n_points=config["evaluate"].get("plot_points", 480))
    history_path = (PROJECT_ROOT / config["train"]["model_dir"]
                    / config["train"]["history_filename"])
    if history_path.exists():
        plot_training_curves(history_path, figures_dir / "training_curves.png")
    print(f"图表已保存到 {figures_dir}/")


if __name__ == "__main__":
    main()
