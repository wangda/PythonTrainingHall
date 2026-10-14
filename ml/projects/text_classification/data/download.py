"""AG News 下载与缓存（已完整实现，直接可用）。

三通道，按顺序尝试：
    ① GitHub raw 直连：raw.githubusercontent.com/mhjabreel/CharCnn_Keras 仓库
      data/ag_news_csv/ 下的 train.csv / test.csv；
    ② gh-proxy 代理：同一份文件的代理前缀变体（国内网络对 GitHub raw 不友好时
      实测可行）；
    ③ 内置极小合成数据集兜底：全部网络通道失败时，按四类新闻的典型词汇拼
      800 条合成样本——只够让流程跑通，指标没有参考意义（项目文档 §2.3 如实注明）。

AG News 数据集（Zhang et al., 2015）：
    新闻标题+摘要的文本分类基准，4 类（World / Sports / Business / Sci-Tech），
    官方切分 train 120000 条 / test 7600 条。CSV 无表头，三列：
    类别编号(1~4), 标题, 摘要——本项目把标题和摘要拼在一起作为输入文本，
    类别编号减 1 转成 0~3。

产物缓存为 data/raw/ag_news_train.csv / ag_news_test.csv：
    数据来源可以换（甚至兜底成合成数据），接口不用动。

用法：
    python data/download.py              # 命令行直接运行：下载并缓存
    from data.download import load_raw_data
    train_texts, train_labels, test_texts, test_labels, meta = load_raw_data(config)
"""

import csv
import random
import urllib.request
from pathlib import Path

import yaml

# 项目根目录：本文件在 <root>/data/download.py，向上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILENAME = "ag_news_train.csv"
TEST_FILENAME = "ag_news_test.csv"

# 按顺序尝试的下载前缀：GitHub raw 直连 → gh-proxy 代理变体
URL_PREFIXES = [
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/",
    "https://gh-proxy.com/https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/",
]
REMOTE_FILES = {TRAIN_FILENAME: "train.csv", TEST_FILENAME: "test.csv"}

# AG News 四个类别的真实名称（CSV 里的编号 1~4 → 下标 0~3）
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# 通道 ③ 兜底用的合成数据模板：每类若干典型句式，随机拼接词汇生成
_SYNTH_TEMPLATES = {
    "World": ["government election peace treaty crisis protest", "president minister war united nations border"],
    "Sports": ["team wins championship match goals league", "player score tournament coach season victory"],
    "Business": ["stocks market shares profit earnings company", "economy bank merger investors revenue trade"],
    "Sci/Tech": ["software computer internet technology research", "chip data smartphone science digital launch"],
}


def _make_meta():
    """数据集元信息：类别名等，会随 checkpoint 一起保存。"""
    return {
        "dataset": "ag_news",
        "num_classes": 4,
        "class_names": list(CLASS_NAMES),
    }


def _download_file(filename, remote_name, cache_dir):
    """按 URL_PREFIXES 顺序尝试下载一个文件，落盘缓存后返回。"""
    last_exc = None
    for prefix in URL_PREFIXES:
        url = prefix + remote_name
        try:
            print(f"  下载 {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / filename).write_bytes(data)
            print(f"    成功：{len(data) / 1024 / 1024:.1f} MB")
            return
        except Exception as exc:  # 网络问题形态各异，换下一个通道
            print(f"    失败：{type(exc).__name__}: {exc}")
            last_exc = exc
    raise RuntimeError(f"所有下载通道都失败（{remote_name}）") from last_exc


def _write_synthetic_csv(cache_dir, filename, n_per_class=100, seed=42):
    """通道 ③：四类典型词汇随机拼接，生成极小合成数据集兜底。"""
    rng = random.Random(seed)
    path = cache_dir / filename
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for label, name in enumerate(CLASS_NAMES, start=1):
            vocab = " ".join(_SYNTH_TEMPLATES[name]).split()
            for _ in range(n_per_class):
                words = [rng.choice(vocab) for _ in range(rng.randint(5, 12))]
                writer.writerow([label, words[0].capitalize(), " ".join(words)])
    return path


def _generate_synthetic(cache_dir):
    """全部网络通道失败时，写两份合成 CSV 并打印醒目警告。"""
    print("警告：所有下载通道失败，使用内置合成数据集兜底！")
    print("  合成数据每类只有典型词汇随机拼接的样本，流程能跑通，")
    print("  但指标没有参考意义——请检查网络后删除 data/raw/ 重新下载。")
    _write_synthetic_csv(cache_dir, TRAIN_FILENAME, n_per_class=200)
    _write_synthetic_csv(cache_dir, TEST_FILENAME, n_per_class=50)


def _read_csv(path):
    """读一份 AG News CSV，返回 (texts, labels)：texts 是标题+摘要，labels 是 0~3。"""
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            labels.append(int(row[0]) - 1)              # 1~4 → 0~3
            texts.append((row[1] + " " + row[2]).strip())
    return texts, labels


def load_raw_data(config):
    """按 config 加载 AG News，统一返回 (train_texts, train_labels, test_texts, test_labels, meta)。

    参数:
        config: dict，读取 config["data"]["raw_dir"]（缓存目录）与
                config["data"]["train_subset"]（训练子集大小，None = 全量）、
                config["data"]["random_state"]。

    返回:
        train_texts:  list[str]，标题+摘要拼成的原始文本（未分词、未截断）；
        train_labels: list[int]，标签 0~3；
        test_texts/test_labels: 官方测试集——只许评估用，不许参与任何训练决策
                      （包括建词表！见 features.py 的 docstring）；
        meta:         dict，数据集元信息（num_classes / class_names）。

    注意:
        train_subset 只裁训练集，测试集永远全量 7600 条；取子集用固定种子的
        随机抽样（各类别比例大致保持均衡），保证实验可复现。
    """
    dcfg = config["data"]
    cache_dir = PROJECT_ROOT / dcfg.get("raw_dir", "data/raw")
    train_csv, test_csv = cache_dir / TRAIN_FILENAME, cache_dir / TEST_FILENAME

    if train_csv.exists() and test_csv.exists():
        print(f"从缓存加载 AG News：{cache_dir}")
    else:
        try:
            for filename, remote_name in REMOTE_FILES.items():
                if not (cache_dir / filename).exists():
                    _download_file(filename, remote_name, cache_dir)
        except Exception as exc:
            print(f"下载失败：{type(exc).__name__}: {exc}")
            _generate_synthetic(cache_dir)

    train_texts, train_labels = _read_csv(train_csv)
    test_texts, test_labels = _read_csv(test_csv)

    subset = dcfg.get("train_subset")
    if subset and subset < len(train_texts):
        rng = random.Random(dcfg.get("random_state", 42))
        idx = rng.sample(range(len(train_texts)), subset)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]
        print(f"训练集取子集：{subset} 条（官方全量 120000 条，config data.train_subset 可调）")

    print(f"AG News: train {len(train_texts)} 条 / test {len(test_texts)} 条，4 类 {CLASS_NAMES}")
    return train_texts, train_labels, test_texts, test_labels, _make_meta()


if __name__ == "__main__":
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_texts, train_labels, test_texts, test_labels, meta = load_raw_data(config)
    from collections import Counter
    print("训练子集各类别样本数:", sorted(Counter(train_labels).items()))
    print("样例：", train_texts[0][:120])
