"""数据下载与加载（已完整实现）。

数据源：CIFAR-10（32×32 彩色图，10 类，训练 50000 张 / 测试 10000 张）。
首次运行需联网（约 170MB），缓存到 config["data"]["raw_dir"]。

用法：
    python data/download.py          # 预热缓存 + 打印数据集概况

设计决策（项目文档 §2）：这里返回的是**未张量化的 PIL 图片数据集**
（transform=None）——预处理全部交给 features.py 的 transforms 声明，
由 DataLoader 每次取数时执行，保证训练增强每个 epoch 都随机。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CIFAR-10 官方类别名（下标即标签编号）
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_dataset(config):
    """加载 CIFAR-10：有缓存读缓存，没有自动下载。

    参数:
        config: dict，读取 config["data"]["raw_dir"]。

    返回:
        (train_set, test_set, meta):
            train_set: torchvision.datasets.CIFAR10，50000 张，transform=None（PIL 图）；
            test_set:  同上，10000 张官方测试集；
            meta:      dict，数据集元信息（dataset / classes / image_shape / n_train / n_test）。
    """
    from torchvision.datasets import CIFAR10

    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_set = CIFAR10(root=raw_dir, train=True, download=True, transform=None)
    test_set = CIFAR10(root=raw_dir, train=False, download=True, transform=None)
    meta = {
        "dataset": "cifar10",
        "classes": CLASSES,
        "image_shape": (3, 32, 32),
        "n_train": len(train_set),
        "n_test": len(test_set),
    }
    return train_set, test_set, meta


def main():
    import yaml

    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_set, test_set, meta = load_dataset(config)
    print(f"数据集: {meta['dataset']}，训练 {meta['n_train']} 张 / 测试 {meta['n_test']} 张")
    print(f"类别: {meta['classes']}")
    img, label = train_set[0]
    print(f"样本 0: {type(img).__name__} {img.size}，标签 {label}（{meta['classes'][label]}）")


if __name__ == "__main__":
    main()
