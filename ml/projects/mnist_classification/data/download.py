"""数据下载与加载（已完整实现，直接可用）。

双数据源，由 config.yaml 的 data.dataset 一行切换：
    digits  sklearn 内置 load_digits()：1797 张 8×8 灰度图，像素范围 0~16，
            零下载、秒级训练，用于快速验证全流程；
    mnist   fetch_openml("mnist_784")：70000 张 28×28 灰度图，像素范围 0~255，
            首次下载后用 np.savez_compressed 缓存为 data/raw/mnist.npz
            （70000×784 存 csv 会有几百 MB，压缩 npz 实测约 20MB）。

用法：
    python data/download.py            # 命令行直接运行：下载并缓存 mnist（digits 无需下载）
    from data.download import load_dataset
    X, y, meta = load_dataset(config)  # 有缓存读缓存，没有则先下载
"""

from pathlib import Path

import numpy as np
import yaml

# 项目根目录：本文件在 <root>/data/download.py，向上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "mnist.npz"


def _load_digits():
    """digits 数据源：sklearn 内置，零下载。

    返回 (X, y, meta)：X 为 (1797, 8, 8) 的图像数组（未展平，展平是 features.py 的职责）。
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    meta = {
        "dataset": "digits",
        "image_shape": (8, 8),     # 每张图的高、宽
        "pixel_max": 16.0,         # 像素值上限（注意：不是 255！归一化要用它）
        "classes": list(range(10)),
    }
    return digits.images.astype(np.float64), digits.target.astype(int), meta


def _load_mnist(cache_path):
    """mnist 数据源：有 npz 缓存读缓存，没有则 fetch_openml 下载并缓存。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        npz = np.load(cache_path)
        print(f"从缓存加载 mnist：{cache_path}")
        X, y = npz["data"], npz["target"]
    else:
        print("本地缓存不存在，开始从 OpenML 下载 mnist_784（arff 几十 MB，首次较慢）……")
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
        X = X.astype(np.float64)          # (70000, 784)，已展平
        y = y.astype(int)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, data=X, target=y)
        print(f"数据已缓存到 {cache_path}（{X.shape[0]} 行 × {X.shape[1]} 列）")
    meta = {
        "dataset": "mnist",
        "image_shape": (28, 28),
        "pixel_max": 255.0,
        "classes": list(range(10)),
    }
    return X, y, meta


def load_dataset(config):
    """按 config 加载数据集，统一返回 (X, y, meta)。

    参数:
        config: dict，读取 config["data"]["dataset"]（"digits" / "mnist"）；
                mnist 时还会读 config["data"]["raw_path"] 作为缓存路径。

    返回:
        (X, y, meta):
            X:    np.ndarray。digits 为 (1797, 8, 8) 图像数组；mnist 为 (70000, 784)
                  已展平的像素矩阵（OpenML 原始格式，展平与否由数据集自身决定）。
            y:    np.ndarray，形状 (n,)，类别标签 0~9。
            meta: dict，数据集元信息——image_shape（高宽）、pixel_max（归一化系数）、
                  classes（类别列表）。预测时要靠它复现预处理，会随模型一起保存。
    """
    name = config["data"]["dataset"]
    if name == "digits":
        return _load_digits()
    if name == "mnist":
        raw_path = config["data"].get("raw_path")
        cache_path = (PROJECT_ROOT / raw_path) if raw_path else DEFAULT_RAW_PATH
        return _load_mnist(cache_path)
    raise ValueError(f"未知数据集: {name}（config.yaml 的 data.dataset 只支持 digits / mnist）")


if __name__ == "__main__":
    # 命令行直接运行：预热 mnist 缓存（digits 内置无需下载）
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    X, y, meta = load_dataset({**config, "data": {**config["data"], "dataset": "mnist"}})
    print(f"mnist: X {X.shape}, y {y.shape}, meta = {meta}")
    print("各类别样本数:", np.bincount(y))
