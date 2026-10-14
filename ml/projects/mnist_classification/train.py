"""训练入口（流程框架已写好，标注 TODO 的步骤待你在对应模块中实现）。

用法：
    python train.py                  # 用默认 config.yaml 训练（默认 digits 数据集）
    python train.py --config my.yaml # 换一份配置（如把 data.dataset 改成 mnist）

流程：加载配置 → 加载数据 → 特征工程 → 训练 → 保存模型 → 测试集评估。
产物：models/model.joblib —— 一个 dict，打包了 model + 数据集元信息 meta
     （图像形状、归一化系数 pixel_max、类别列表、归一化开关），
     预测时必须靠 meta 复现训练时的预处理，缺一不可。
"""

import argparse
from pathlib import Path

import joblib
import yaml

from data.download import load_dataset
from features import build_features, split_data
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


def save_model(model, meta, config):
    """把模型 + 数据集元信息打包保存到 config 指定的目录（已完整实现）。

    参数:
        model:  训练完成的 sklearn 模型。
        meta:   dict，数据集元信息（dataset / image_shape / pixel_max / classes），
                预测时靠它复现预处理（尺寸校验、归一化系数），再补上归一化开关。
        config: dict，读取 config["train"]["model_dir"] / ["model_filename"]
                和 config["features"]["normalize"]。

    返回:
        Path，模型文件路径。
    """
    model_dir = PROJECT_ROOT / config["train"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / config["train"]["model_filename"]
    bundle_meta = {**meta, "normalize": bool(config["features"].get("normalize", True))}
    joblib.dump({"model": model, "meta": bundle_meta}, path)
    print(f"模型已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="手写数字分类：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    # 1. 加载配置：所有旋钮都在 yaml 里，改实验不改代码
    config = load_config(args.config)

    # 2. 加载数据：digits 内置直接读；mnist 有缓存读缓存，没有自动下载（data/download.py 已实现）
    X, y, meta = load_dataset(config)
    print(f"数据集: {meta['dataset']}，X {X.shape}，图像 {meta['image_shape']}，像素上限 {meta['pixel_max']}")

    # 3. 特征工程 —— TODO: 实现 features.flatten_images / normalize_pixels / split_data
    X, y = build_features(X, y, meta, config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    print(f"特征矩阵: {X.shape}；训练/测试 = {X_train.shape[0]} / {X_test.shape[0]}")

    # 4. 训练 —— TODO: 实现 model.train_model
    model = train_model(create_model(config), X_train, y_train)

    # 5. 保存模型（数据集元信息必须一起存，预测时复现预处理）
    save_model(model, meta, config)

    # 6. 测试集快速评估 —— TODO: 实现 evaluate.evaluate 后取消注释
    # from evaluate import evaluate
    # metrics = evaluate(model, X_test, y_test, meta, config)
    # print(f"测试集指标: {metrics}")


if __name__ == "__main__":
    main()
