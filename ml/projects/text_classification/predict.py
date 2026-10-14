"""命令行推理入口（已完整实现）。

用法：
    python predict.py --input "Apple unveils new chip for laptops"
    python predict.py --input "Oil prices surge as OPEC cuts output"
    python predict.py --input "..." --model models/best.pt

注意：预测必须复现训练时的预处理——分词用 features.simple_tokenize（同一个函数），
编码用 checkpoint 记录的 vocab.json（同一个词表，绝不重新统计）。softmax 只在
预测时显式做一次（训练时 CrossEntropyLoss 内部已含）。
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from features import encode
from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="AG News 文本分类：命令行推理")
    parser.add_argument("--input", required=True, help="一条英文新闻文本（标题，或标题+摘要）")
    parser.add_argument("--model", default=None, help="checkpoint 路径，默认 models/best.pt")
    parser.add_argument("--config", default=None, help="配置文件路径，默认项目根目录 config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml
    with open(args.config or PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = args.model or (PROJECT_ROOT / config["train"]["model_dir"]
                                / config["train"]["model_filename"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    from features import load_vocab
    vocab = load_vocab(ckpt["vocab_path"])
    max_len = ckpt["config"]["features"]["max_len"]
    class_names = ckpt["config"]["model"].get("class_names") or [
        "World", "Sports", "Business", "Sci/Tech"]

    ids, mask = encode([args.input], vocab, max_len)      # 与训练完全相同的编码逻辑
    with torch.no_grad():
        logits = model(torch.from_numpy(ids).to(device),
                       padding_mask=torch.from_numpy(mask).to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top2 = np.argsort(proba)[::-1][:2]
    print(f"输入: {args.input!r}")
    print(f"预测: {class_names[top2[0]]}（置信度 {proba[top2[0]]:.2%}）")
    for i in top2:
        print(f"    {class_names[i]}: {proba[i]:.2%}")


if __name__ == "__main__":
    main()
