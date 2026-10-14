"""特征工程（骨架）：原始表格 → 模型输入，配套文档《06-Titanic生存预测项目文档.md》§3/§4。

三个函数构成"有状态预处理"的完整流程（和模块 2 的无状态归一化有本质区别，文档 §3）：
    split_data        ① 先划分训练/测试集（stratify 保持获救比例）
    fit_preprocessor  ② 只在训练集上 fit 出预处理状态（中位数/众数/列清单）
    transform         ③ 用同一份状态变换任意数据（测试集、预测时的新乘客）

教学要点：
    - age 中位数、embarked 众数都是"从数据里统计出来的量"，只能看训练集——
      先全量填补再划分 = 数据泄漏（文档 §3）；
    - state 必须随模型打包保存（train.py 负责），predict.py 靠它复现预处理；
    - 全程不做特征缩放：树模型只看排序不看尺度（前置文档 §1.4），缩放是多余动作；
    - boat / body 是泄漏列（文档 §5），本模块负责删除——做泄漏实验时注释掉
      LEAKAGE_COLS 的引用即可，但记得改回来。
"""

import numpy as np
import pandas as pd

# 泄漏列：答案的另一种写法（项目文档 §5）
LEAKAGE_COLS = ["boat", "body"]
# 高基数文本列：入门版弃用（进阶可提取 Title / 甲板，项目文档 §4）
DROP_COLS = ["name", "ticket", "cabin", "home.dest"]

TARGET = "survived"

# transform 输出的特征列（add_family_features=True 时再追加 family_size / is_alone）
BASE_FEATURES = [
    "pclass", "age", "sibsp", "parch", "fare",
    "sex_male", "embarked_C", "embarked_Q", "embarked_S",
]
EMBARKED_CATEGORIES = ["C", "Q", "S"]   # one-hot 列清单固定，防止子集缺类别时列对不上


def split_data(df, config):
    """划分训练/测试集（保持类别比例）。

    参数:
        df:     pd.DataFrame，load_dataset 返回的原始表格（含目标列 survived）。
        config: dict，读取 config["data"]["test_size"] 和 ["random_state"]。

    返回:
        (train_df, test_df): pd.DataFrame，
            行数比例 = 1-test_size : test_size；stratify=df["survived"] 保证两集
            获救比例一致（约 38% 获救，不平衡数据必须分层，前置文档 §11.7）。

    提示:
        - sklearn.model_selection.train_test_split，传 test_size / random_state /
          stratify=df[TARGET]；
        - 这是对**行**的划分，不动任何列；
        - 划分必须先于一切统计型预处理（中位数/众数）——顺序反了就是泄漏（文档 §3）。
    """
    raise NotImplementedError


def fit_preprocessor(train_df, config):
    """在训练集上 fit 预处理状态，并返回变换后的训练特征（文档 §3 第②步）。

    参数:
        train_df: pd.DataFrame，split_data 返回的训练集（含目标列）。
        config:   dict，读取 config["features"]["add_family_features"]。

    返回:
        (X_train, y_train, state):
            X_train: pd.DataFrame，形状 (n_train, n_features)，全数值、无缺失；
            y_train: np.ndarray，形状 (n_train,)，0/1；
            state:   dict，预处理状态，键至少包括：
                "age_median":    float，训练集 age 中位数；
                "fare_median":   float，训练集 fare 中位数；
                "embarked_mode": str，训练集 embarked 众数（"C"/"Q"/"S"）；
                "add_family_features": bool，开关快照（预测时复现同一套特征）；
                "feature_names": list[str]，transform 输出的列顺序。

    提示:
        - 中位数用 train_df["age"].median()（中位数比均值抗离群值——票价有天价票）；
        - 众数用 train_df["embarked"].mode()[0]；
        - 算出 state 后，直接 return (*transform(train_df, state, config), state)
          ——fit 和 transform 共享同一套变换逻辑，不写两份代码。
    """
    raise NotImplementedError


def transform(df, state, config=None):
    """用 fit 好的状态变换数据（训练集、测试集、预测时的新乘客共用），文档 §3 第③步。

    参数:
        df:     pd.DataFrame，原始表格（列与 load_dataset 输出一致；预测时无目标列）。
        state:  dict，fit_preprocessor 返回的预处理状态。
        config: dict，可选（特征开关以 state["add_family_features"] 为准）。

    返回:
        (X, y):
            X: pd.DataFrame，形状 (n, len(state["feature_names"]))，
               列顺序严格等于 state["feature_names"]，全数值、无缺失；
            y: np.ndarray 或 None，df 含目标列时是 0/1 标签，否则 None（预测场景）。

    处理步骤（顺序不要乱）:
        1. 复制 df，丢弃 LEAKAGE_COLS 和 DROP_COLS 中存在的列；
        2. 缺失值填补：age ← state["age_median"]，fare ← state["fare_median"]，
           embarked ← state["embarked_mode"]（全部来自训练集统计量）；
        3. 编码：sex_male = (sex == "male").astype(int)；
           embarked one-hot 成 embarked_C / embarked_Q / embarked_S 三列——
           用 `(embarked == c).astype(int)` 逐类别生成，保证列永远齐全
           （不要用 pd.get_dummies 后不 reindex，测试集会缺列）；
        4. 组合特征（state["add_family_features"] 为 True 时）：
           family_size = sibsp + parch + 1；is_alone = (family_size == 1).astype(int)；
        5. 按 state["feature_names"] 选列排序，astype(float) 收尾（LightGBM 要全数值）。

    提示:
        - 出口处断言：X 没有缺失值、所有列都是数值型——泄漏和编码错误都在这里现形；
        - 这个函数是"纯函数"：同样的 (df, state) 必须给出同样的 X，不许偷偷统计新量。
    """
    raise NotImplementedError
