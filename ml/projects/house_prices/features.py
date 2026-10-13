"""特征工程（骨架）：原始特征 → 模型输入，配套文档《02-房价预测项目文档.md》§5/§6。

三个函数对应 Pipeline 的"特征工程"一步：
    add_engineered_features   构造组合特征（文档 §6 的思路 1~3）
    build_features            组装最终特征矩阵 X 和目标向量 y
    split_and_scale           划分训练/测试集 + 标准化（scaler 只在训练集 fit！）

约定：
    - 原始特征 8 列见 FEATURE_COLS，目标列见 TARGET_COL；
    - 工程特征 3 列见 ENGINEERED_COLS，是否添加由 config["features"]["add_engineered_features"] 控制；
    - 训练和预测必须走同一套特征逻辑，predict.py 会直接复用本模块。
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "MedInc",       # 街区收入中位数（万美元）
    "HouseAge",     # 街区房龄中位数（年）
    "AveRooms",     # 户均房间数
    "AveBedrms",    # 户均卧室数
    "Population",   # 街区人口
    "AveOccup",     # 户均居住人数
    "Latitude",     # 纬度
    "Longitude",    # 经度
]
TARGET_COL = "MedHouseVal"                          # 目标：街区房价中位数（10 万美元）
ENGINEERED_COLS = ["BedroomRatio", "RoomsPerPerson", "LogMedInc"]


def add_engineered_features(df, config):
    """按 config 决定是否添加组合特征，返回新 DataFrame（不修改入参）。

    参数:
        df:     pd.DataFrame，至少包含 FEATURE_COLS 这 8 列。
        config: dict，读取 config["features"]["add_engineered_features"]；
                为 False 时原样返回（仍返回副本）。

    返回:
        pd.DataFrame，开启时在原 8 列基础上新增 ENGINEERED_COLS 三列：
            BedroomRatio    = AveBedrms / AveRooms   卧室占房间的比例
            RoomsPerPerson  = AveRooms / AveOccup    人均房间数
            LogMedInc       = log(MedInc)            收入右偏，取对数压平长尾

    提示:
        - 先 df.copy() 再增列，避免修改调用方的数据；
        - 除法注意分母为 0 的极端行（可用 np.where 或加 eps 保护）；
        - 对数用 np.log。
    """
    raise NotImplementedError


def build_features(df, config):
    """组装特征矩阵 X 和目标向量 y。

    参数:
        df:     pd.DataFrame，包含 FEATURE_COLS + TARGET_COL。
        config: dict，同 add_engineered_features。

    返回:
        (X, y):
            X: pd.DataFrame，形状 (n, n_features)。add_engineered_features 开启时
               为 8 + 3 = 11 列，否则 8 列；列顺序 = FEATURE_COLS + 新增列。
            y: pd.Series，形状 (n,)，即 TARGET_COL 列。

    提示:
        - 内部先调 add_engineered_features，再按列名拆 X / y；
        - X 中不能包含 TARGET_COL（否则等于把答案告诉模型）。
    """
    raise NotImplementedError


def split_and_scale(X, y, config):
    """划分训练/测试集并标准化特征（本项目防数据泄漏的核心函数）。

    参数:
        X:      pd.DataFrame 或 np.ndarray，形状 (n, d)，特征矩阵。
        y:      pd.Series 或 np.ndarray，形状 (n,)，目标向量。
        config: dict，读取 config["data"]["test_size"] 和 ["random_state"]。

    返回:
        (X_train, X_test, y_train, y_test, scaler):
            X_train/X_test: np.ndarray，标准化后的特征，行数比例 = 1-test_size : test_size；
            y_train/y_test: np.ndarray，与 X 对应的划分；
            scaler: 已在 X_train 上 fit 的 StandardScaler（要随模型一起保存，
                    预测新数据时只能 transform，不能重新 fit）。

    提示:
        - 划分：sklearn.model_selection.train_test_split，把 test_size 和
          random_state 从 config 里读出来传进去；
        - 标准化：sklearn.preprocessing.StandardScaler，
          scaler.fit(X_train) → scaler.transform(X_train) / scaler.transform(X_test)。
          铁律：scaler 只用训练集 fit，绝不能让测试集参与统计量的计算（文档 §7 常见坑）。
    """
    raise NotImplementedError
