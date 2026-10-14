"""训练入口（流程框架已写好，标注 TODO 的步骤待你在 features.py 中实现）。

用法：
    python train.py                  # 用默认 config.yaml 训练（默认 lightgbm）
    python train.py --config my.yaml # 换一份配置（如把 model.name 改成 decision_tree）

流程：加载配置 → 加载数据 → 划分 → fit 预处理（只在训练集上）→ transform →
     训练 → 打包保存（模型 + 预处理状态 + 元信息）→ 测试集快速评估。

产物：models/model.joblib —— 一个 dict，三个键：
    "model": 训练好的模型；
    "state": 预处理状态（中位数/众数/特征列清单/开关快照），predict.py 靠它复现预处理；
    "meta":  数据集元信息。
"""

import argparse
from pathlib import Path

import joblib
import yaml

from data.download import load_dataset
from features import fit_preprocessor, split_data, transform
from model import create_model, train_model

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """加载 yaml 配置文件（已完整实现）。"""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_bundle(model, state, meta, config):
    """把 模型 + 预处理状态 + 元信息 打包保存到 config 指定的目录（已完整实现）。

    参数:
        model:  训练完成的模型。
        state:  dict，fit_preprocessor 返回的预处理状态——预测时复现预处理的唯一依据。
        meta:   dict，数据集元信息。
        config: dict，读取 config["train"]["model_dir"] / ["model_filename"]。

    返回:
        Path，模型文件路径。
    """
    model_dir = PROJECT_ROOT / config["train"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / config["train"]["model_filename"]
    joblib.dump({"model": model, "state": state, "meta": meta}, path)
    print(f"模型包已保存到 {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Titanic 生存预测：训练入口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    args = parser.parse_args()

    # 1. 加载配置：所有旋钮都在 yaml 里，改实验不改代码
    config = load_config(args.config)

    # 2. 加载数据：有缓存读缓存，没有自动从 OpenML 下载（data/download.py 已实现）
    df, meta = load_dataset(config)
    print(f"数据集: {meta['dataset']}，{meta['n_rows']} 行，"
          f"获救率 {meta['n_survived']/meta['n_rows']:.1%}")

    # 3. 划分 + 有状态预处理 —— TODO: 实现 features.split_data / fit_preprocessor / transform
    #    顺序铁律（项目文档 §3）：先划分，状态只在训练集上 fit，再用同一状态变换两边
    train_df, test_df = split_data(df, config)
    X_train, y_train, state = fit_preprocessor(train_df, config)
    X_test, y_test = transform(test_df, state, config)
    print(f"特征: {state['feature_names']}")
    print(f"训练/测试 = {len(X_train)} / {len(X_test)}")

    # 4. 训练（model.create_model / train_model 已实现，工厂模式按 config 创建）
    model = train_model(create_model(config), X_train, y_train)

    # 5. 打包保存：预处理状态必须随模型一起存，预测时缺一不可
    save_bundle(model, state, meta, config)

    # 6. 测试集快速评估（evaluate.py 已实现，训练完顺手看一眼）
    from evaluate import evaluate
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"{k:>12s}: {v:.4f}")


if __name__ == "__main__":
    main()
