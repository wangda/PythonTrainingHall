"""命令行推理入口（已完整实现）。

用法：
    # 单张 28×28 PNG 图片
    python predict.py --input my_image.png

    # 白底黑字的图需要反相（Fashion-MNIST 是黑底白物）
    python predict.py --input my_image.png --invert

    # 指定 checkpoint
    python predict.py --input my_image.png --model models/best.pt

注意：预测必须复现训练时的预处理——尺寸校验（28×28）、/255 归一化、可选标准化，
全部来自 checkpoint 里存的 config，不需要也不应该再读 config.yaml。
softmax 只在预测时显式做一次（训练时 CrossEntropyLoss 内部已含，见 model.py）。
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST 分类：命令行推理")
    parser.add_argument("--input", required=True, help="PNG 图片路径（28×28）")
    parser.add_argument("--invert", action="store_true",
                        help="反相（白底黑物 → 黑底白物）。自己拍/画的图通常是白底，需要加上")
    parser.add_argument("--model", default=None, help="checkpoint 路径，默认 models/best.pt")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def load_image(path, invert=False):
    """读一张 28×28 图片，转成 (1, 784) 的 [0,1] 浮点向量。

    异常:
        ValueError: 图片尺寸不是 28×28。
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(path)                       # PNG → float [0,1]，(H,W) 或 (H,W,3/4)
    if img.ndim == 3:                              # 彩色/带透明通道 → 转灰度
        img = img[..., :3].mean(axis=2)
    if img.shape != (28, 28):
        raise ValueError(
            f"图片尺寸应为 28×28（Fashion-MNIST 的规格），收到 {img.shape}。"
            f"请先缩放到 28×28 再预测。"
        )
    if invert:
        img = 1.0 - img
    return img.reshape(1, -1).astype(np.float32)


def predict_top3(model, X, device):
    """输出 top-3 类别及概率（softmax 在这里显式做一次，训练时不需要）。"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    top3_idx = np.argsort(proba, axis=1)[:, ::-1][:, :3]
    return [[(CLASS_NAMES[i], float(proba[row, i])) for i in top3_idx[row]] for row in range(len(X))]


def main():
    args = parse_args()

    import yaml
    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"])           # 结构以 checkpoint 存的 config 为准
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    X = load_image(args.input, invert=args.invert)
    fcfg = ckpt["config"].get("features", {})
    if fcfg.get("standardize", False):
        if "standardize_stats" in ckpt:
            mean, std = ckpt["standardize_stats"]
            X = (X - mean) / std
        else:
            print("警告：训练时开启了标准化，但 checkpoint 里没有统计量，预测未做标准化！")

    for i, top3 in enumerate(predict_top3(model, X, device)):
        print(f"样本 {i}: 预测 = {top3[0][0]}（置信度 {top3[0][1]:.2%}）")
        for name, p in top3:
            print(f"    {name}: {p:.2%}")


if __name__ == "__main__":
    main()
