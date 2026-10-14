"""模型层：按配置创建模型、训练（已完整实现，工厂模式示范）。

四个名字对应前置文档的三代模型对比（项目文档 §6）：
    decision_tree  → sklearn DecisionTreeClassifier（手写练习的调库版，基线）
    random_forest  → sklearn RandomForestClassifier（bagging 路线，前置文档 §9）
    lightgbm       → LGBMClassifier（boosting 主力，前置文档 §11）
    xgboost        → XGBClassifier（对比用，前置文档 §12）

工厂模式：调用方只说一句"按 config 给我一个模型"，切换模型只改 config.yaml。
"""

MODEL_NAMES = ("decision_tree", "random_forest", "lightgbm", "xgboost")


def create_model(config):
    """按 config 创建模型实例（已完整实现）。

    参数:
        config: dict，读取 config["model"]["name"] 和 ["params"]。
                params 透传给对应模型的构造函数（切换 name 时记得在
                config.yaml 里同步换 params）。

    返回:
        未 fit 的模型实例（sklearn / lightgbm / xgboost 统一 fit/predict 接口）。

    异常:
        ValueError:   未知模型名；
        ImportError:  选了 lightgbm / xgboost 但环境未安装（报错信息里给安装命令）。
    """
    name = config["model"]["name"]
    params = dict(config["model"].get("params", {}))

    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "未安装 lightgbm：pip install lightgbm（或先切回 random_forest）"
            ) from e
        return lgb.LGBMClassifier(**params)

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "未安装 xgboost：pip install xgboost（或先切回 random_forest）"
            ) from e
        return XGBClassifier(**params)

    raise ValueError(f"未知模型: {name}（config.yaml 的 model.name 只支持 {MODEL_NAMES}）")


def train_model(model, X_train, y_train):
    """用训练集拟合模型（已完整实现）。

    参数:
        model:   create_model 返回的模型实例（未 fit）。
        X_train: pd.DataFrame，形状 (n_train, d)，transform 输出的训练特征。
        y_train: np.ndarray，形状 (n_train,)，训练标签 0/1。

    返回:
        训练完成的模型实例（fit 返回 self，可直接 return model.fit(...)）。

    提示:
        一行就够——统一 API：model.fit(X, y)。
        LightGBM 的早停版本在 tune.py 里（需要额外切验证集，前置文档 §11.3），
        这里的默认训练不需要。
    """
    return model.fit(X_train, y_train)
