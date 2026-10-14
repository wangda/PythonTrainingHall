"""命令行推理入口（已完整实现）。

用法：
    # 单张 PNG 图片（任意来源：自己画的、网上找的）
    python predict.py --input my_digit.png

    # 白底黑字的手绘图需要反相（MNIST/digits 都是黑底白字）
    python predict.py --input my_digit.png --invert

    # 像素值 csv：每行 h*w 个像素值（digits 64 列 / mnist 784 列，与训练数据集一致）
    python predict.py --input samples.csv

    # 指定模型文件
    python predict.py --input my_digit.png --model models/model.joblib

注意：预测必须复现训练时的预处理——尺寸校验、灰度化、按数据集元信息里的
pixel_max 归一化，全部来自模型包里的 meta，不需要也不应该再读 config.yaml。
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="手写数字分类：命令行推理")
    parser.add_argument(
        "--input",
        required=True,
        help="PNG 图片路径，或像素值 csv 路径（每行 h*w 个像素值）",
    )
    parser.add_argument("--invert", action="store_true",
                        help="反相（白底黑字 → 黑底白字）。手绘图通常是白底黑字，需要加上")
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def load_bundle(model_path):
    """加载模型包：model + meta（train.py 保存的两件套，meta 是复现预处理的依据）。"""
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["meta"]


def load_image(path, meta, invert=False):
    """读一张图片，转成与训练数据同格式的像素行向量。

    参数:
        path:   str 或 Path，图片路径（PNG 等 matplotlib.image 支持的格式）。
        meta:   dict，模型包里的数据集元信息，读取 image_shape 和 pixel_max。
        invert: bool，True 时反相（白底黑字 → 黑底白字）。

    返回:
        np.ndarray，形状 (1, h*w)，已按 pixel_max 归一化到 [0, 1]。

    异常:
        ValueError: 图片尺寸与训练数据集的 image_shape 不一致。
    """
    import matplotlib.image as mpimg

    h, w = meta["image_shape"]
    img = mpimg.imread(path)                       # PNG → float [0,1]，形状 (H,W) 或 (H,W,3/4)
    if img.ndim == 3:                              # 彩色/带透明通道 → 转灰度
        img = img[..., :3].mean(axis=2)
    if img.shape != (h, w):
        raise ValueError(
            f"图片尺寸应为 {h}×{w}（训练数据集 {meta['dataset']} 的规格），收到 {img.shape}。"
            f"请先缩放到 {h}×{w} 再预测。"
        )
    if invert:
        img = 1.0 - img
    # imread 读 PNG 得到的是 [0,1] 浮点像素，与训练时 /pixel_max 的归一化天然等价
    return img.reshape(1, -1)


def load_csv(path, meta):
    """读像素值 csv：每行 h*w 个像素值（原始量纲，函数内负责归一化）。

    参数:
        path: str 或 Path，csv 路径（无表头，每行 h*w 个数值）。
        meta: dict，模型包里的数据集元信息。

    返回:
        np.ndarray，形状 (n, h*w)，已归一化到 [0, 1]。

    异常:
        ValueError: 列数与训练数据集的像素数不一致。
    """
    X = np.loadtxt(path, delimiter=",", ndmin=2)
    h, w = meta["image_shape"]
    if X.shape[1] != h * w:
        raise ValueError(f"csv 每行应有 {h * w} 个像素值（{meta['dataset']} 数据集），收到 {X.shape[1]} 个")
    return X / meta["pixel_max"]


def predict_top3(X, model, meta):
    """输出每个样本的 top-3 预测类别及概率。

    参数:
        X:     np.ndarray，形状 (n, d)，已归一化的像素矩阵。
        model: 训练好的 sklearn 分类模型（需有 predict_proba）。
        meta:  dict，读取 meta["classes"] 作为类别标签。

    返回:
        list[list[tuple]]，每个样本一个 top-3 列表，元素为 (类别, 概率)，按概率降序。
    """
    classes = np.asarray(meta["classes"])
    proba = model.predict_proba(X)                 # (n, 10)，softmax 概率（前置文档 §5）
    top3_idx = np.argsort(proba, axis=1)[:, ::-1][:, :3]
    return [[(int(classes[i]), float(proba[row, i])) for i in top3_idx[row]] for row in range(len(X))]


def main():
    args = parse_args()

    import yaml
    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    model, meta = load_bundle(model_path)

    if args.input.endswith(".csv"):
        X = load_csv(args.input, meta)
    else:
        X = load_image(args.input, meta, invert=args.invert)

    for i, top3 in enumerate(predict_top3(X, model, meta)):
        print(f"样本 {i}: 预测 = {top3[0][0]}（置信度 {top3[0][1]:.2%}）")
        for cls, p in top3:
            print(f"    类别 {cls}: {p:.2%}")


if __name__ == "__main__":
    main()
