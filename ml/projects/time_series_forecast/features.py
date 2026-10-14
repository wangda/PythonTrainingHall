"""特征工程（骨架）：传感器长序列 → 滑动窗口 → 三个 DataLoader。

配套文档《12-时序预测项目文档.md》§3/§5；原理见前置文档 §6（滑动窗口与时序泄漏）。

本模块的三个函数对应时序特征工程的三步，每一步都有一个"泄漏陷阱"：

    ① split_by_time：按【时间先后】切 train/val/test，禁止 shuffle。
       时序数据的样本前后相关，随机划分会让"未来"混进训练集——模型等于
       提前看过考题，评估虚假偏高且永远无法发现（前置文档 §6）；
    ② make_windows：滑动窗口把长序列切成 (window_size 历史 → horizon 未来)
       的监督学习样本。窗口必须在【各段内部】构造——跨越 train/val 边界的
       窗口，输入里带着训练段末尾、目标却在验证段，同样是泄漏；
    ③ fit_scaler_on_train：标准化的均值/方差只用【训练段】计算，再套到
       val/test 上。全量数据算统计量 = 把未来的分布信息泄漏进预处理
       （和模块 1 的 StandardScaler 同一个坑）。

约定：
    - train_loader 可以 shuffle=True：窗口化之后，每个窗口是独立样本，
      打乱【样本顺序】不影响任何窗口内部的时间结构——这与序列级 shuffle
      （打乱原始时间点的先后）是两回事，后者是灾难；
    - val/test 不 shuffle（评估要能对齐"第几个窗口预测成了什么"）；
    - dtype：特征与目标均 float32。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def split_by_time(values, val_ratio, test_ratio):
    """按时间先后把序列切成 train/val/test 三段（禁止 shuffle）。

    参数:
        values:     np.ndarray，(n, n_features)，按时间升序的长序列；
        val_ratio:  float，验证集占比（时间轴上倒数第二段）；
        test_ratio: float，测试集占比（时间轴上最后一段）。

    返回:
        (train, val, test): 三个 np.ndarray，行数比例为
            (1 - val_ratio - test_ratio, val_ratio, test_ratio)，
            时间顺序：train 最早，test 最晚，段间无重叠。

    为什么禁止 shuffle（前置文档 §6）:
        时序样本前后相关：第 t 个点和第 t+1 个点往往只差一点点。随机划分后，
        训练集里全是测试点的前后邻居，模型靠"背邻居"就能蒙对——评估数字
        漂亮得虚假。按时间切，测试集才是真正的"未来"。
    """
    raise NotImplementedError


def make_windows(series, window_size, horizon, target_col):
    """滑动窗口：把 (n, n_features) 序列切成监督学习样本。

    参数:
        series:      np.ndarray，(n, n_features)，按时间升序（单段内部，
                     不许跨 train/val/test 边界）；
        window_size: int，输入窗口长（用过去多少个点）；
        horizon:     int，预测步长（预测未来多少个点）；
        target_col:  int，目标列在 series 列轴上的下标（ETTh1 里 OT 是第 6 列）。

    返回:
        (X, y):
            X: np.ndarray，(n_samples, window_size, n_features) float32，
               第 i 个窗口 = series[i : i + window_size, :]（全部特征）；
            y: np.ndarray，(n_samples, horizon) float32，
               第 i 个目标 = series[i + window_size : i + window_size + horizon, target_col]
               ——只用目标列，因为我们要预测的是 OT 的未来；
            n_samples = n - window_size - horizon + 1。

    对齐关系（测试会逐索引断言这个）:
        第 i 个窗口的最后一行是 series[i + window_size - 1]，
        第 i 个目标的第一行是 series[i + window_size]——恰好是窗口最后一行
        的下一个时间点。预测从不偷看窗口内部的信息。

    提示:
        - 用列表收集再 np.stack/np.asarray，或预分配后按行填充；
        - 输出转 np.float32（torch 的默认浮点）。
    """
    raise NotImplementedError


def fit_scaler_on_train(train_values):
    """在【训练段】上 fit 标准化器（统计量绝不许碰 val/test）。

    参数:
        train_values: np.ndarray，(n_train, n_features)，仅训练段。

    返回:
        sklearn.preprocessing.StandardScaler，已 fit。
        对 val/test 只调 transform，绝不再 fit——全量数据算均值方差，
        等于把未来的分布信息泄漏进预处理（前置文档 §6）。

    提示:
        - scaler.mean_ / scaler.scale_ 会被 train.py 存进 checkpoint，
          predict.py 靠它复现预处理。
    """
    raise NotImplementedError


def build_datasets(df, config):
    """把上面三步串成 DataLoader 三件套（train 可 shuffle，val/test 不行）。

    参数:
        df:     pd.DataFrame，load_raw_data 的返回（date + 7 个传感器列）；
        config: dict，读取 config["data"] 的 feature_cols / target_col /
                window_size / horizon / val_ratio / test_ratio，
                以及 config["train"]["batch_size"]。

    返回:
        (train_loader, val_loader, test_loader, scaler):
            前三个是 DataLoader，每个 batch 是 (X, y)：
                X 形状 (batch, window_size, n_features) float32，
                y 形状 (batch, horizon) float32；
            scaler 是在训练段上 fit 的 StandardScaler（反归一化评估和
            predict.py 复现预处理都要用它）。

    正确顺序（顺序错了就是泄漏）:
        1. 取 df[feature_cols].values，按 split_by_time 切成三段；
        2. fit_scaler_on_train 只用训练段 fit，再 transform 三段；
        3. 各段【内部】分别 make_windows（跨段窗口 = 泄漏）；
        4. 包 TensorDataset + DataLoader：train shuffle=True，val/test False。

    提示:
        - target_col 是列名在 feature_cols 里的下标：
          feature_cols.index(config["data"]["target_col"])；
        - train 的 shuffle 打乱的是【窗口样本的顺序】，不是窗口内部的时间——
          窗口化之后每个样本独立，打乱是 SGD 的正常要求，与序列级 shuffle
          的灾难不是一回事。
    """
    raise NotImplementedError
