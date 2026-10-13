"""训练入口（流程框架已写好，标注 TODO 的步骤待你在对应模块中实现）。

用法：
    python train.py                  # 用默认 config.yaml 训练
    python train.py --config my.yaml # 换一份配置（如把 model.name 改成 ridge）

流程：加载配置 → 加载数据 → 特征工程 → 训练 → 保存模型 → 测试集评估。
产物：models/model.joblib —— 一个 dict，打包了 model + scaler + 特征列名，
     预测时三者缺一不可（scaler 用来 transform 新数据，列名用来对齐特征顺序）。
"""

import argparse
from pathlib import Path

import joblib
import yaml

from data.download import load_raw_data
from features import build_features, split_and_scale
from model import create_model, train_model

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """加载 yaml 配置文件（已完整实现）。

    参数:
        path: str 或 Path，配置文件路径；None 时用项目根目录下的 config.yaml。

    返回:
        dict，配置内容。
    """
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_model(model, scaler, feature_names, config):
    """把模型 + scaler + 特征列名打包保存到 config 指定的目录（已完整实现）。

    参数:
        model:         训练完成的 sklearn 模型。
        scaler:        在训练集上 fit 过的 StandardScaler。
        feature_names: list[str]，训练时 X 的列名顺序（预测时必须按同样顺序喂数据）。
        config:        dict，读取 config["train"]["model_dir"] / ["model_filename"]。

    返回:
        Path，模型文件路径。
    """
    model_dir = PROJECT_ROOT / config["train"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / config["train"]["model_filename"]
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_names": list(feature_names)},
        path,
    )
    print(f"模型已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="房价预测：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    # 1. 加载配置：所有旋钮都在 yaml 里，改实验不改代码
    config = load_config(args.config)

    # 2. 加载数据：有本地缓存读缓存，没有自动下载（data/download.py 已实现）
    df = load_raw_data(PROJECT_ROOT / config["data"]["raw_path"])
    print(f"数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")

    # 3. 特征工程 —— TODO: 实现 features.build_features / split_and_scale
    X, y = build_features(df, config)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, config)
    print(f"特征矩阵: {X.shape}；训练/测试 = {X_train.shape[0]} / {X_test.shape[0]}")

    # 4. 训练 —— TODO: 实现 model.train_model
    model = train_model(create_model(config), X_train, y_train)

    # 5. 保存模型（scaler 和特征列名必须一起存，预测时复用）
    save_model(model, scaler, X.columns, config)

    # 6. 测试集快速评估 —— TODO: 实现 evaluate.evaluate 后取消注释
    # from evaluate import evaluate
    # metrics = evaluate(model, X_test, y_test, config)
    # print(f"测试集指标: {metrics}")


if __name__ == "__main__":
    main()
