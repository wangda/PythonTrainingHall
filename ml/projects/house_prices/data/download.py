"""数据下载与加载（已完整实现，直接可用）。

数据集：California Housing（20640 个加州街区、8 个特征、目标为街区房价中位数，
单位 10 万美元），是经典 Boston 房价数据集的现代替代品。

下载策略（双通道，任一可用即可）：
    1. 首选 sklearn 的 fetch_california_housing —— 数据托管在 figshare，
       部分网络环境会返回 403；
    2. 失败时回退到 OpenML 的原始数据集（data_id=537，Pace & Barry 1997），
       按 sklearn 源码同样的规则做列变换，产出的数据与通道 1 完全一致。

用法：
    python data/download.py            # 命令行直接运行：下载并缓存 csv
    from data.download import load_raw_data
    df = load_raw_data()               # 有缓存读缓存，没有则先下载
"""

from pathlib import Path

import pandas as pd

# 项目根目录：本文件在 <root>/data/download.py，向上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "california_housing.csv"


def _fetch_from_sklearn():
    """通道 1：sklearn 官方 fetcher（数据托管在 figshare）。"""
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True).frame  # frame 已含目标列


def _fetch_from_openml():
    """通道 2：OpenML 原始数据 + 手工复刻 sklearn 的列变换。

    变换规则照抄 sklearn/datasets/_california_housing.py 源码：
    户均值 = 总量 / households，目标值除以 100000（单位变成 10 万美元）。
    """
    from sklearn.datasets import fetch_openml

    raw = fetch_openml(data_id=537, as_frame=True, parser="auto").frame  # "houses" 数据集
    df = pd.DataFrame({
        "MedInc": raw["median_income"],
        "HouseAge": raw["housing_median_age"],
        "AveRooms": raw["total_rooms"] / raw["households"],
        "AveBedrms": raw["total_bedrooms"] / raw["households"],
        "Population": raw["population"],
        "AveOccup": raw["population"] / raw["households"],
        "Latitude": raw["latitude"],
        "Longitude": raw["longitude"],
        "MedHouseVal": raw["median_house_value"] / 100000.0,
    })
    return df.astype(float)


def download_data(save_path=None):
    """下载 California Housing 数据集并缓存为 csv。

    参数:
        save_path: str 或 Path，csv 保存路径；None 时使用 DEFAULT_RAW_PATH
                   （项目根目录下 data/raw/california_housing.csv）。

    返回:
        df: pd.DataFrame，形状 (20640, 9)——8 个特征列 + 目标列 MedHouseVal。
            特征：MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
                  Latitude, Longitude；目标：MedHouseVal（单位 10 万美元）。
    """
    save_path = Path(save_path) if save_path else DEFAULT_RAW_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = _fetch_from_sklearn()
        print("数据源：sklearn（figshare）")
    except Exception as e:  # figshare 被墙/限流时走 OpenML
        print(f"sklearn 通道失败（{type(e).__name__}: {e}），改用 OpenML 通道……")
        df = _fetch_from_openml()
        print("数据源：OpenML（data_id=537）")

    df.to_csv(save_path, index=False)
    print(f"数据已缓存到 {save_path}（{df.shape[0]} 行 × {df.shape[1]} 列）")
    return df


def load_raw_data(path=None):
    """读取本地缓存的原始数据；缓存不存在时自动下载。

    参数:
        path: str 或 Path，csv 路径；None 时使用 DEFAULT_RAW_PATH。

    返回:
        df: pd.DataFrame，同 download_data 的返回。
    """
    path = Path(path) if path else DEFAULT_RAW_PATH
    if not path.exists():
        print(f"本地缓存 {path} 不存在，开始下载……")
        return download_data(path)
    return pd.read_csv(path)


if __name__ == "__main__":
    df = download_data()
    print(df.head())
    print(df.describe().round(2))
