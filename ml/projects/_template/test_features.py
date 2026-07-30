"""
特征工程单元测试

运行：python -m pytest test_features.py -v
"""

import numpy as np
import pandas as pd
from features import (  # noqa: F401 — 在项目里实现后取消注释
    # fill_missing,
    # normalize,
    # encode_categorical,
)


def make_dummy_data():
    """生成测试用 dummy 数据"""
    return pd.DataFrame({
        "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat_col": ["a", "b", "a", "c", None],
        "target": [0, 1, 0, 1, 0],
    })


# TODO: 在 features.py 实现对应函数后取消注释
# class TestFeatureEngineering:
#     def test_fill_missing_numeric(self):
#         df = make_dummy_data()
#         result = fill_missing(df, "numeric_col", strategy="mean")
#         assert result["numeric_col"].isnull().sum() == 0
#         assert result.loc[2, "numeric_col"] == pytest.approx(3.0, abs=0.01)
#
#     def test_fill_missing_categorical(self):
#         df = make_dummy_data()
#         result = fill_missing(df, "cat_col", strategy="mode")
#         assert result["cat_col"].isnull().sum() == 0

#     def test_normalize(self):
#         data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
#         normalized = normalize(data)
#         assert normalized.shape == data.shape
#         assert np.abs(normalized.mean(axis=0)).max() < 1e-10
#         assert np.abs(normalized.std(axis=0) - 1.0).max() < 1e-10
