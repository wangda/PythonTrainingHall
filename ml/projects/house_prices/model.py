"""模型层：按配置创建模型、训练。create_model 已完整实现（工厂模式示范）。

工厂模式：调用方不关心具体是 LinearRegression 还是 Ridge，只说一句
"按 config 给我一个模型"，切换模型只改 config.yaml，不动代码。
"""

from sklearn.linear_model import LinearRegression, Ridge


def create_model(config):
    """按 config 创建模型实例（已完整实现）。

    参数:
        config: dict，读取 config["model"]["name"] 和 ["params"]。
                name 可选 "linear_regression" / "ridge"；
                ridge 时用 params["alpha"] 作为正则化强度（前置文档 §6 的 λ）。

    返回:
        sklearn 回归模型实例（LinearRegression 或 Ridge），尚未 fit。

    异常:
        ValueError: config["model"]["name"] 不是已知的模型名。
    """
    name = config["model"]["name"]
    params = config["model"].get("params", {})
    if name == "linear_regression":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0))
    raise ValueError(f"未知模型: {name}（config.yaml 的 model.name 只支持 linear_regression / ridge）")


def train_model(model, X_train, y_train):
    """用训练集拟合模型。

    参数:
        model:   create_model 返回的 sklearn 模型实例（未 fit）。
        X_train: np.ndarray，形状 (n_train, d)，标准化后的训练特征。
        y_train: np.ndarray，形状 (n_train,)，训练目标。

    返回:
        训练完成的模型实例（sklearn 的 fit 返回 self，可直接 return model.fit(...)）。

    提示:
        一行就够——sklearn 统一 API（前置文档 §8）：model.fit(X, y)。
        LinearRegression 内部走闭式解路线，正好和手写练习的正规方程对上。
    """
    raise NotImplementedError
