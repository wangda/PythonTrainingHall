"""数据下载与加载（已完整实现）。

数据源：OpenML 的 Titanic 数据集（1309 名乘客，目标列 survived）。
首次运行需要联网，下载后缓存到 config["data"]["raw_path"]（data/raw/titanic.csv），
之后离线可用。

用法：
    python data/download.py          # 预热缓存 + 打印数据集概况
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 项目文档 §5：这两列是"答案的另一种写法"（救生艇编号→获救，遗体编号→遇难），
# 这里只在下载层保留它们（方便你做"泄漏实验"），features.py 负责在特征工程时删除。
LEAKAGE_COLS = ["boat", "body"]


def fetch_titanic():
    """从 OpenML 拉取 Titanic 数据集并做最低限度的类型规整（联网，仅首次调用）。

    返回:
        pd.DataFrame，1309 行，含目标列 "survived"（int，1=获救 0=遇难）。

    规整内容（只动类型，不动信息）:
        - pclass：OpenML 各版本里可能是 "1st"/"2nd"/"3rd" 或 "1"/"2"/"3"，
          统一提取数字转成 int 1/2/3；
        - survived：字符串 "0"/"1" → int；
        - sex / embarked / cabin 等类别列 → object（保留缺失值 NaN）；
        - age / fare 等数值列 → float。
    """
    from sklearn.datasets import fetch_openml

    print("正在从 OpenML 下载 Titanic 数据集（仅首次需要联网）...")
    bunch = fetch_openml("titanic", version=1, as_frame=True)
    df = bunch.frame.copy()

    df["pclass"] = df["pclass"].astype(str).str.extract(r"(\d)")[0].astype(int)
    # survived 在 OpenML 里是 categorical("0"/"1")，先转字符串再转 int 最稳
    df["survived"] = df["survived"].astype(str).astype(int)
    for col in ["sex", "embarked", "name", "ticket", "cabin", "boat", "home.dest"]:
        if col in df.columns:
            df[col] = df[col].astype("object")
    for col in ["age", "sibsp", "parch", "fare", "body"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_dataset(config):
    """加载数据集：优先读本地缓存，没有则联网下载并缓存。

    参数:
        config: dict，读取 config["data"]["raw_path"]（缓存位置，相对项目根目录）。

    返回:
        (df, meta):
            df:   pd.DataFrame，原始表格（含目标列 "survived"）；
            meta: dict，数据集元信息（dataset / target / n_rows / n_survived）。
    """
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        try:
            df = fetch_titanic()
        except Exception as e:
            raise RuntimeError(
                f"下载 Titanic 数据集失败（需要联网）：{e}\n"
                f"也可以手动下载 OpenML titanic v1 的 csv 放到 {raw_path}"
            ) from e
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"数据集已缓存到 {raw_path}")

    meta = {
        "dataset": "titanic",
        "target": "survived",
        "n_rows": int(len(df)),
        "n_survived": int(df["survived"].sum()),
    }
    return df, meta


def main():
    import yaml

    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    df, meta = load_dataset(config)
    print(f"数据集: {meta['dataset']}，共 {meta['n_rows']} 行，"
          f"获救 {meta['n_survived']} 人（{meta['n_survived']/meta['n_rows']:.1%}）")
    print(f"列: {list(df.columns)}")
    print(f"缺失值 Top5:\n{df.isna().sum().sort_values(ascending=False).head()}")


if __name__ == "__main__":
    main()
