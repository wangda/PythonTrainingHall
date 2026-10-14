"""Fashion-MNIST 下载与缓存（已完整实现，直接可用）。

双通道，按顺序尝试：
    ① torchvision.datasets.FashionMNIST(download=True)：官方推荐方式，
      内部同样是拉 IDX 文件；
    ② 手工解析 IDX：网络对 torchvision 不友好时，回退到 zalandoresearch/fashion-mnist
      官方仓库的 GitHub 直链（raw 直连 → gh-proxy 代理 → S3 镜像，依次尝试），
      用 urllib 下载 gzip 压缩的 IDX 文件并手工解析。

两条通道的产物统一缓存为 data/raw/fashion_mnist.npz（np.savez_compressed）：
    x_train/y_train/x_test/y_test，uint8 原始像素 0~255，
    x_* 形状为 (n, 28, 28)——归一化、展平、切验证集都是 features.py 的职责。

用法：
    python data/download.py              # 命令行直接运行：下载并缓存
    from data.download import load_raw_data
    x_train, y_train, x_test, y_test, meta = load_raw_data(config)
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import yaml

# 项目根目录：本文件在 <root>/data/download.py，向上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "fashion_mnist.npz"

# 官方 IDX 镜像直链（zalandoresearch/fashion-mnist 仓库 data/fashion/ 下的原始文件）
# 按顺序尝试：GitHub raw 直连 → gh-proxy 代理 → 官方 S3 镜像（torchvision 同款，慢但稳）
IDX_URL_TEMPLATES = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}",
    "https://gh-proxy.com/https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{name}",
]
IDX_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

# Fashion-MNIST 十个类别的真实名称（下标 = 标签 0~9）
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _make_meta():
    """数据集元信息：预测时靠它复现预处理，会随 checkpoint 一起保存。"""
    return {
        "dataset": "fashion_mnist",
        "image_shape": (28, 28),     # 每张图的高、宽
        "pixel_max": 255.0,          # 像素值上限，归一化系数
        "classes": list(range(10)),
        "class_names": list(CLASS_NAMES),
    }


# ---------- 通道 ②：IDX 手工解析 ----------

def _parse_idx_images(data):
    """解析 IDX 图像文件：魔数 2051，随后是样本数、行数、列数（大端 uint32）。"""
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 2051, f"不是合法的 IDX 图像文件（魔数 {magic}，应为 2051）"
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols).copy()


def _parse_idx_labels(data):
    """解析 IDX 标签文件：魔数 2049，随后是样本数。"""
    magic, n = struct.unpack(">II", data[:8])
    assert magic == 2049, f"不是合法的 IDX 标签文件（魔数 {magic}，应为 2049）"
    return np.frombuffer(data[8:], dtype=np.uint8, count=n).copy()


def _fetch_idx_file(filename):
    """按 IDX_URL_TEMPLATES 顺序尝试下载一个 gzip 文件，返回解压后的字节。"""
    last_exc = None
    for template in IDX_URL_TEMPLATES:
        url = template.format(name=filename)
        try:
            print(f"  下载 {url}")
            with urllib.request.urlopen(url, timeout=120) as resp:
                return gzip.decompress(resp.read())
        except Exception as exc:  # 网络问题形态各异，换下一个镜像
            print(f"    失败：{type(exc).__name__}: {exc}")
            last_exc = exc
    raise RuntimeError(f"所有 IDX 镜像都下载失败（{filename}）") from last_exc


def _download_from_idx_mirror(cache_path):
    """从官方 IDX 镜像下载四个 gzip 文件，解析后写入 npz 缓存。"""
    print("torchvision 通道失败，回退到官方 IDX 镜像……")
    parsed = {}
    for key, filename in IDX_FILES.items():
        raw = _fetch_idx_file(filename)
        if "images" in filename:
            parsed[key] = _parse_idx_images(raw)
        else:
            parsed[key] = _parse_idx_labels(raw)
    _save_cache(cache_path, parsed)
    return parsed["x_train"], parsed["y_train"], parsed["x_test"], parsed["y_test"]


# ---------- 通道 ①：torchvision ----------

def _download_with_torchvision(cache_path):
    """用 torchvision.datasets.FashionMNIST 下载，转成 uint8 数组后写缓存。"""
    print("尝试 torchvision.datasets.FashionMNIST 通道……")
    from torchvision import datasets

    torch_cache = PROJECT_ROOT / "data" / "raw" / "torchvision"
    train_set = datasets.FashionMNIST(root=str(torch_cache), train=True, download=True)
    test_set = datasets.FashionMNIST(root=str(torch_cache), train=False, download=True)
    parsed = {
        "x_train": train_set.data.numpy().astype(np.uint8),
        "y_train": train_set.targets.numpy().astype(np.uint8),
        "x_test": test_set.data.numpy().astype(np.uint8),
        "y_test": test_set.targets.numpy().astype(np.uint8),
    }
    _save_cache(cache_path, parsed)
    return parsed["x_train"], parsed["y_train"], parsed["x_test"], parsed["y_test"]


# ---------- 统一入口 ----------

def _save_cache(cache_path, parsed):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **parsed)
    print(f"数据已缓存到 {cache_path}（训练 {parsed['x_train'].shape[0]} 张，"
          f"测试 {parsed['x_test'].shape[0]} 张）")


def load_raw_data(config):
    """按 config 加载 Fashion-MNIST，统一返回 (x_train, y_train, x_test, y_test, meta)。

    参数:
        config: dict，读取 config["data"]["raw_path"] 作为缓存路径。

    返回:
        x_train: np.ndarray，(60000, 28, 28)，uint8 原始像素 0~255（未归一化、未展平）；
        y_train: np.ndarray，(60000,)，uint8 标签 0~9；
        x_test:  np.ndarray，(10000, 28, 28)，官方测试集——只许评估用，不许参与任何训练决策；
        y_test:  np.ndarray，(10000,)；
        meta:    dict，数据集元信息（image_shape / pixel_max / classes / class_names）。
    """
    raw_path = config["data"].get("raw_path")
    cache_path = (PROJECT_ROOT / raw_path) if raw_path else DEFAULT_RAW_PATH

    if cache_path.exists():
        print(f"从缓存加载 Fashion-MNIST：{cache_path}")
        npz = np.load(cache_path)
        return npz["x_train"], npz["y_train"], npz["x_test"], npz["y_test"], _make_meta()

    try:
        arrays = _download_with_torchvision(cache_path)
    except Exception as exc:  # 网络问题形态各异（超时/重置/HTTP 错误），统一回退
        print(f"torchvision 通道失败：{type(exc).__name__}: {exc}")
        arrays = _download_from_idx_mirror(cache_path)
    return *arrays, _make_meta()


if __name__ == "__main__":
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    x_train, y_train, x_test, y_test, meta = load_raw_data(config)
    print(f"Fashion-MNIST: x_train {x_train.shape} {x_train.dtype}, "
          f"x_test {x_test.shape}, meta = {meta}")
    print("训练集各类别样本数:", np.bincount(y_train))
