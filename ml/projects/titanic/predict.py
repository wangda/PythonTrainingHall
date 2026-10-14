"""命令行推理入口（已完整实现）。

用法：
    # 单个乘客：用命令行参数描述
    python predict.py --pclass 3 --sex male --age 22 --sibsp 1 --parch 0 --fare 7.25 --embarked S

    # 批量：csv 文件，每行一个乘客（列：pclass,sex,age,sibsp,parch,fare,embarked）
    python predict.py --input passengers.csv --output result.csv

    # 指定模型包
    python predict.py --model models/model.joblib --pclass 1 --sex female --age 30 ...

注意：预测必须复现训练时的预处理——填补值、编码、组合特征、列顺序，
全部来自模型包里的 state（fit_preprocessor 在训练集上学出的那份），
不需要也不应该重新统计任何东西（项目文档 §3）。
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml

from features import transform

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]


def parse_args():
    parser = argparse.ArgumentParser(description="Titanic 生存预测：命令行推理")
    parser.add_argument("--input", default=None,
                        help="乘客 csv 路径（列：%s；可含 survived 列，会被忽略）" % ",".join(RAW_COLUMNS))
    parser.add_argument("--output", default=None, help="批量预测结果输出 csv 路径（默认打印到屏幕）")
    # 单乘客模式的七个字段
    parser.add_argument("--pclass", type=int, choices=[1, 2, 3])
    parser.add_argument("--sex", choices=["male", "female"])
    parser.add_argument("--age", type=float)
    parser.add_argument("--sibsp", type=int)
    parser.add_argument("--parch", type=int)
    parser.add_argument("--fare", type=float)
    parser.add_argument("--embarked", choices=["C", "Q", "S"])
    parser.add_argument("--model", default=None, help="模型包路径，默认取 config 里的 train.model_dir/model_filename")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def load_bundle(model_path):
    """加载模型包：model + state + meta（train.py 保存的三件套）。"""
    return joblib.load(model_path)


def load_passengers(args):
    """从 --input csv 或命令行字段构造原始乘客 DataFrame。

    返回:
        pd.DataFrame，列 = RAW_COLUMNS（允许缺列——缺失字段交给 transform 的填补逻辑）。
    """
    if args.input:
        df = pd.read_csv(args.input)
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"csv 缺少列: {missing}（需要 {RAW_COLUMNS}）")
        return df[RAW_COLUMNS]

    fields = {c: getattr(args, c) for c in RAW_COLUMNS}
    if any(v is None for v in fields.values()):
        missing = [c for c, v in fields.items() if v is None]
        raise SystemExit(f"单乘客模式需要全部字段，缺少: {missing}（或用 --input csv）")
    return pd.DataFrame([fields])


def predict(df, bundle):
    """复现预处理并预测（state 是唯一的预处理依据，项目文档 §3）。

    参数:
        df:     pd.DataFrame，原始乘客表格。
        bundle: dict，模型包（"model" / "state" / "meta"）。

    返回:
        (proba, pred): 获救概率数组与 0/1 预测数组。
    """
    X, _ = transform(df, bundle["state"])
    model = bundle["model"]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return proba, pred


def main():
    args = parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    bundle = load_bundle(model_path)
    df = load_passengers(args)
    proba, pred = predict(df, bundle)

    if args.output:
        out = df.copy()
        out["survival_proba"] = proba.round(4)
        out["survived_pred"] = pred
        out.to_csv(args.output, index=False)
        print(f"预测结果已写入 {args.output}")
    else:
        for i, (p, y_hat) in enumerate(zip(proba, pred)):
            label = "获救" if y_hat == 1 else "遇难"
            print(f"乘客 {i}: 预测 = {label}（获救概率 {p:.1%}）")


if __name__ == "__main__":
    main()
