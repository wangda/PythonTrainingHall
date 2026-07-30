"""
命令行推理入口

用法：
  python predict.py --input data/sample.csv --output predictions.csv
  python predict.py --input "feature1=1.5,feature2=abc"  # 单条预测
"""

import argparse
import pandas as pd
import yaml
import joblib


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--input", required=True, help="输入文件路径或单条数据")
    parser.add_argument("--output", default="predictions.csv", help="输出路径")
    parser.add_argument("--model_path", default="models/model.pkl", help="模型文件路径")
    return parser.parse_args()


def load_model(path):
    return joblib.load(path)


def preprocess(df):
    """特征预处理（与训练时一致）"""
    # TODO: 与 features.py 保持一致
    return df


def predict(model, features):
    return model.predict(features)


def main():
    args = parse_args()
    config = load_config()
    model = load_model(args.model_path)

    # 判断输入是文件还是单条数据
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        # 单条解析: "key1=val1,key2=val2"
        data = {}
        for pair in args.input.split(","):
            k, v = pair.split("=")
            data[k.strip()] = v.strip()
        df = pd.DataFrame([data])

    features = preprocess(df)
    preds = predict(model, features)

    df["prediction"] = preds
    df.to_csv(args.output, index=False)
    print(f"预测完成，结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
