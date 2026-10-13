"""命令行推理入口（已完整实现，依赖 features.add_engineered_features 完成后可用）。

用法：
    # 单条预测：8 个特征值，逗号分隔，顺序与 FEATURE_COLS 一致
    python predict.py --input "8.3252,41,6.98,1.02,322,2.55,37.88,-122.23"

    # 批量预测：csv 文件需包含 8 个原始特征列
    python predict.py --input new_houses.csv

    # 指定模型文件
    python predict.py --input "..." --model models/ridge.joblib

注意：预测必须与训练走同一套特征逻辑——先 add_engineered_features 补组合特征，
再按训练时的 feature_names 对齐列顺序，最后用过训练集 fit 的 scaler.transform。
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml

from features import FEATURE_COLS, add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="房价预测：命令行推理")
    parser.add_argument(
        "--input",
        required=True,
        help="8 个逗号分隔的特征值（顺序: " + ",".join(FEATURE_COLS) + "），或含这 8 列的 csv 路径",
    )
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def load_bundle(model_path):
    """加载模型包：model + scaler + feature_names（train.py 保存的三件套）。"""
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["scaler"], bundle["feature_names"]


def parse_input(input_str):
    """把 --input 解析成 DataFrame（8 个原始特征列）。

    参数:
        input_str: str，csv 路径（以 .csv 结尾）或逗号分隔的 8 个数值。

    返回:
        pd.DataFrame，列 = FEATURE_COLS；csv 可能含多行，单条输入为 1 行。
    """
    if input_str.endswith(".csv"):
        df = pd.read_csv(input_str)
        missing = set(FEATURE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"csv 缺少特征列: {missing}，需要 {FEATURE_COLS}")
        return df[FEATURE_COLS]

    values = [float(v) for v in input_str.split(",")]
    if len(values) != len(FEATURE_COLS):
        raise ValueError(f"需要 {len(FEATURE_COLS)} 个特征值，收到 {len(values)} 个")
    return pd.DataFrame([values], columns=FEATURE_COLS)


def predict(input_df, model, scaler, feature_names, config):
    """对输入数据走"特征工程 → 列对齐 → 缩放 → 预测"全流程。

    参数:
        input_df:      pd.DataFrame，含 8 个原始特征列。
        model:         训练好的 sklearn 模型。
        scaler:        训练集上 fit 的 StandardScaler。
        feature_names: list[str]，训练时的特征列顺序。
        config:        dict（add_engineered_features 开关要和训练时一致）。

    返回:
        np.ndarray，每行一个预测值，单位 10 万美元。
    """
    df = add_engineered_features(input_df, config)   # 与训练完全相同的特征逻辑
    X = df[feature_names]                            # 按训练时的列顺序对齐
    X_scaled = scaler.transform(X)                   # 只 transform，不 fit
    return model.predict(X_scaled)


def main():
    args = parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    model, scaler, feature_names = load_bundle(model_path)
    input_df = parse_input(args.input)
    preds = predict(input_df, model, scaler, feature_names, config)

    for i, p in enumerate(preds):
        print(f"样本 {i}: 预测房价中位数 = {p:.4f}（×10 万美元）≈ {p * 10:.2f} 万美元")


if __name__ == "__main__":
    main()
