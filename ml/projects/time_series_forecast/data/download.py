"""ETTh1 下载与缓存（已完整实现，直接可用）。

三通道，按顺序尝试：
    ① GitHub raw 直连：zhouhaoyi/ETDataset 官方仓库的 ETT-small/ETTh1.csv；
    ② gh-proxy 代理：网络对 GitHub raw 不友好时，走 gh-proxy / mirror.ghproxy
       前缀代理（同一文件，多个代理变体依次尝试）；
    ③ 内置合成数据兜底：全部网络通道失败时，现场生成一份多变量合成时序
       （温度 + 功率 + 湿度三条相关序列，叠加趋势 + 双周期 + 噪声，展开成与
       ETTh1 同构的 7 列 + date 列）。兜底保证流程永远能跑通，但文档会如实
       注明走的是合成通道。

产物统一缓存为 data/raw/etth1.csv（pandas DataFrame，date + 7 列传感器数据）。
缓存一次，之后 load_raw_data 直接命中，训练实验零网络开销。

用法：
    python data/download.py              # 命令行直接运行：下载并缓存
    from data.download import load_raw_data
    df, meta = load_raw_data(config)
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import yaml

# 项目根目录：本文件在 <root>/data/download.py，向上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "etth1.csv"

# ETTh1 的 7 个传感器列（HUFL/LUFL = 高/低有用负载，HULL/LULL = 高/低无用负载，
# MUFL/MULL = 中有用/无用负载，OT = 油温，即预测目标）
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COL = "OT"

# 通道 ①② 的候选 URL：raw 直连 → gh-proxy 代理变体，按序尝试
ETTH1_URLS = [
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "https://gh-proxy.com/https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "https://ghproxy.net/https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
]


def _make_meta(source):
    """数据集元信息：随流程传递，文档/汇报时如实说明数据来源通道。"""
    return {
        "dataset": "ETTh1",
        "source": source,               # "raw直连" / "gh-proxy代理" / "内置合成数据"
        "target_col": TARGET_COL,
        "feature_cols": list(FEATURE_COLS),
        "freq": "1h",                   # 采样频率：1 小时一个点（实测确认；15 分钟级的是 ETTm1）
    }


# ---------- 通道 ③：内置合成数据兜底 ----------

def _make_synthetic(n=17420, seed=42):
    """生成与 ETTh1 同构的合成多变量时序（趋势 + 双周期 + 相关噪声）。

    三条隐因子序列（温度基线、功率负载、湿度）互相相关，再展开成 7 列：
        - 趋势项：缓慢漂移（模拟季节/工况变化）；
        - 双周期：日周期（96 点 = 24 小时）+ 周周期（672 点 = 7 天）；
        - 噪声：AR(1) 自相关噪声（时序的噪声不是白噪声，前后相关）。
    形状与列名与真实 ETTh1 完全一致，保证下游代码零改动。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = 2 * np.pi * t / 24        # 日周期（24 个小时点 = 1 天）
    weekly = 2 * np.pi * t / (24 * 7) # 周周期（168 个小时点 = 7 天）

    def ar1_noise(phi=0.9, scale=1.0):
        eps = rng.normal(0, scale, n)
        out = np.zeros(n)
        for i in range(1, n):
            out[i] = phi * out[i - 1] + eps[i]
        return out

    temp_base = 20 + 8 * np.sin(daily) + 4 * np.sin(weekly) + 0.0003 * t + ar1_noise()
    power = 50 + 20 * np.sin(daily - 0.8) + 6 * np.sin(weekly) + ar1_noise(scale=2)
    humidity = 60 - 0.6 * temp_base + 5 * np.sin(daily + 1.5) + ar1_noise(scale=1.5)

    cols = {
        "HUFL": power + rng.normal(0, 0.5, n),
        "HULL": power * 0.3 + rng.normal(0, 0.5, n),
        "MUFL": power * 0.6 + 0.2 * temp_base + rng.normal(0, 0.5, n),
        "MULL": power * 0.15 + rng.normal(0, 0.5, n),
        "LUFL": power * 0.45 - 0.1 * humidity + rng.normal(0, 0.5, n),
        "LULL": power * 0.1 + rng.normal(0, 0.5, n),
        "OT":   0.6 * temp_base + 0.15 * power + ar1_noise(phi=0.95, scale=0.3),
    }
    dates = pd.date_range("2016-07-01", periods=n, freq="1h")
    return pd.DataFrame({"date": dates, **cols})


# ---------- 统一入口 ----------

def _download_etth1(cache_path):
    """按 ETTH1_URLS 顺序尝试下载，全部失败则回退合成数据。返回 (df, source)。"""
    for url in ETTH1_URLS:
        try:
            print(f"  下载 {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                content = resp.read()
            df = pd.read_csv(pd.io.common.BytesIO(content))
            assert list(df.columns) == ["date", *FEATURE_COLS], f"列名不符：{df.columns}"
            assert len(df) > 10000, f"行数异常：{len(df)}"
            source = "raw直连" if "gh-proxy" not in url and "ghproxy" not in url else "gh-proxy代理"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"数据已缓存到 {cache_path}（{len(df)} 行）")
            return df, source
        except Exception as exc:  # 网络问题形态各异，换下一个通道
            print(f"    失败：{type(exc).__name__}: {exc}")
    print("所有网络通道失败，回退到内置合成数据（文档需如实注明）")
    df = _make_synthetic()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"合成数据已缓存到 {cache_path}（{len(df)} 行）")
    return df, "内置合成数据"


def load_raw_data(config):
    """按 config 加载 ETTh1，统一返回 (df, meta)。

    参数:
        config: dict，读取 config["data"]["raw_path"] 作为缓存路径。

    返回:
        df:   pd.DataFrame，columns = ["date", *FEATURE_COLS]，按时间升序，
              7 个传感器列均为 float，无任何预处理（窗口化/标准化是 features.py 的职责）；
        meta: dict，数据集元信息（source / target_col / feature_cols / freq）。
    """
    raw_path = config["data"].get("raw_path")
    cache_path = (PROJECT_ROOT / raw_path) if raw_path else DEFAULT_RAW_PATH

    if cache_path.exists():
        print(f"从缓存加载 ETTh1：{cache_path}")
        df = pd.read_csv(cache_path)
        return df, _make_meta("缓存（来源见首次下载日志）")

    df, source = _download_etth1(cache_path)
    return df, _make_meta(source)


if __name__ == "__main__":
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    df, meta = load_raw_data(config)
    print(f"ETTh1: {df.shape}，列 = {list(df.columns)}")
    print(f"来源: {meta['source']}，时间范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(df[FEATURE_COLS].describe().round(2))
