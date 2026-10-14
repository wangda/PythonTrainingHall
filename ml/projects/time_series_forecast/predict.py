"""命令行推理入口（已完整实现）。

用法：
    # 最近 96 个小时点的 7 列传感器数据（csv，含表头，列序 = feature_cols）
    python predict.py --input recent.csv

    # 指定 checkpoint
    python predict.py --input recent.csv --model models/best.pt

注意：预测必须复现训练时的预处理——csv 只含原始传感器值，
标准化用 checkpoint 里存的训练段 scaler（只 transform，绝不重新 fit）。
输出未来 horizon 个小时点的 OT 预测（原始单位：度）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="ETTh1 时序预测：命令行推理")
    parser.add_argument("--input", required=True,
                        help="最近 window_size 行 × 7 列传感器数据的 csv 路径（列序 = 训练时 feature_cols）")
    parser.add_argument("--model", default=None, help="checkpoint 路径，默认 models/best.pt")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def load_window(csv_path, feature_cols, window_size):
    """读最近一段传感器数据并校验形状。

    异常:
        ValueError: 行数不足 window_size，或缺少特征列。
    """
    df = pd.read_csv(csv_path)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"csv 缺少特征列: {missing}，需要 {feature_cols}")
    values = df[feature_cols].values.astype(np.float64)
    if len(values) < window_size:
        raise ValueError(
            f"需要最近 {window_size} 行数据（训练时窗口长），只收到 {len(values)} 行。"
            f"请提供更长的历史数据。")
    return values[-window_size:]          # 只取最近 window_size 行


def main():
    args = parse_args()

    import yaml
    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or (PROJECT_ROOT / config["train"]["model_dir"]
                                / config["train"]["model_filename"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"])       # 结构以 checkpoint 存的 config 为准
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    feature_cols = ckpt["feature_cols"]
    window_size = ckpt["config"]["data"]["window_size"]
    horizon = ckpt["config"]["data"]["horizon"]
    target_col_idx = feature_cols.index(ckpt["target_col"])

    window = load_window(args.input, feature_cols, window_size)
    X = ckpt["scaler"].transform(window)[np.newaxis].astype(np.float32)  # (1, window, 7)

    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X).to(device)).cpu().numpy()  # (1, horizon)
    # 反归一化：只有目标列，用训练段统计量（标准化空间 → 度）
    mean, scale = ckpt["scaler"].mean_[target_col_idx], ckpt["scaler"].scale_[target_col_idx]
    pred = pred_scaled[0] * scale + mean

    print(f"输入: 最近 {window_size} 个小时点的 {len(feature_cols)} 路传感器数据")
    print(f"预测未来 {horizon} 个小时点的 {ckpt['target_col']}（油温，度）:")
    for i, v in enumerate(pred, 1):
        print(f"  +{i}h: {v:.3f}")


if __name__ == "__main__":
    main()
