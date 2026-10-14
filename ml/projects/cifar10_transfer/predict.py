"""命令行推理入口（已完整实现）。

用法：
    python predict.py --input my_cat.jpg
    python predict.py --input a.jpg b.jpg c.jpg        # 多张
    python predict.py --input my_cat.jpg --model models/model.pt

注意：预测必须复现训练时的预处理——图片尺寸、Normalize 常数，
全部来自模型包里的 config 快照（train.py 保存的那份），不读当前 config.yaml。
"""

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image

from features import build_transforms
from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 迁移学习：命令行推理")
    parser.add_argument("--input", nargs="+", required=True, help="图片路径（可多张）")
    parser.add_argument("--model", default=None, help="模型文件路径，默认取 config 里的 train.model_dir/model_filename")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def load_bundle(model_path, device):
    """加载模型包并重建模型（复现结构用包里的 config 快照）。"""
    bundle = torch.load(model_path, map_location=device, weights_only=False)
    # 同上：重建结构时关掉预训练下载，权重以 checkpoint 为准
    import copy
    cfg = copy.deepcopy(bundle["config"])
    cfg["model"]["pretrained"] = False
    model = create_model(cfg)
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    return model, bundle


def predict_image(path, model, bundle, device):
    """对一张图片输出 top-3 预测类别及概率。

    参数:
        path:   str 或 Path，图片路径（任意尺寸，transforms 会缩放到训练规格）。
        model:  重建好的模型（eval 模式）。
        bundle: dict，模型包（复现预处理用 bundle["config"]）。

    返回:
        list[tuple]，(类别名, 概率) 按概率降序的前 3 名。
    """
    tf = build_transforms(bundle["config"], train=False)   # 评估版 transform（确定性）
    img = Image.open(path).convert("RGB")                  # 统一成 RGB（防灰度/透明通道）
    x = tf(img).unsqueeze(0).to(device)                    # 加 batch 维：(1,3,H,W)
    with torch.no_grad():
        proba = torch.softmax(model(x), dim=1)[0]
    top3 = proba.topk(3)
    classes = bundle["meta"]["classes"]
    return [(classes[i], float(p)) for p, i in zip(top3.values, top3.indices)]


def main():
    args = parse_args()

    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or PROJECT_ROOT / config["train"]["model_dir"] / config["train"]["model_filename"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, bundle = load_bundle(model_path, device)

    for path in args.input:
        top3 = predict_image(path, model, bundle, device)
        print(f"{path}: 预测 = {top3[0][0]}（置信度 {top3[0][1]:.1%}）")
        for cls, p in top3:
            print(f"    {cls:>10s}: {p:.1%}")


if __name__ == "__main__":
    main()
