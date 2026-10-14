"""模型层：按配置创建模型、训练。create_model 已完整实现（工厂模式示范）。

工厂模式：调用方不关心具体超参数怎么传，只说一句"按 config 给我一个模型"，
切换超参数只改 config.yaml，不动代码。
"""

from sklearn.linear_model import LogisticRegression


def create_model(config):
    """按 config 创建模型实例（已完整实现）。

    参数:
        config: dict，读取 config["model"]["name"] 和 ["params"]。
                name 可选 "logistic_regression"；
                params 透传 C（正则化强度的倒数 C=1/λ，01 前置文档 §6）
                和 max_iter（lbfgs 迭代上限，前置文档 §4）。

    返回:
        sklearn.linear_model.LogisticRegression 实例，尚未 fit。
        多分类说明：sklearn 1.9 的 lbfgs 求解器默认就是 softmax（multinomial）
        路线，对应前置文档 §5，无需也不应再传已废弃的 multi_class 参数。

    异常:
        ValueError: config["model"]["name"] 不是已知的模型名。
    """
    name = config["model"]["name"]
    params = config["model"].get("params", {})
    if name == "logistic_regression":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=params.get("max_iter", 1000))
    raise ValueError(f"未知模型: {name}（config.yaml 的 model.name 只支持 logistic_regression）")


def train_model(model, X_train, y_train):
    """用训练集拟合模型。

    参数:
        model:   create_model 返回的 sklearn 模型实例（未 fit）。
        X_train: np.ndarray，形状 (n_train, d)，归一化后的训练特征。
        y_train: np.ndarray，形状 (n_train,)，训练标签 0~9。

    返回:
        训练完成的模型实例（sklearn 的 fit 返回 self，可直接 return model.fit(...)）。

    提示:
        一行就够——sklearn 统一 API（前置文档 §8）：model.fit(X, y)。
        LogisticRegression 内部走梯度下降路线最小化交叉熵损失，
        正好和前置文档 §3/§4 的推导对上。
    """
    raise NotImplementedError
