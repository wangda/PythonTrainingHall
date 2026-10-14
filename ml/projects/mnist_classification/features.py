"""特征工程（骨架）：图像像素 → 模型输入，配套文档《04-手写数字分类项目文档.md》§5/§6。

四个函数对应 Pipeline 的"特征工程"一步：
    flatten_images      图像展平：(n, 8, 8) → (n, 64)；已展平的 (n, 784) 原样通过
    normalize_pixels    像素归一化：除以该数据集的像素最大值（digits 16 / mnist 255）
    split_data          划分训练/测试集（stratify=y 保持类别比例）
    build_features      总装：展平 + 归一化（train.py / evaluate.py 的入口）

教学要点（和模块 1 的本质区别）：
    像素归一化是【无状态】的——除以的是一个先验常数（数据集固有的像素上限），
    不从数据里统计任何量。模块 1 的 StandardScaler 要从训练集学出均值/方差，
    所以必须"先划分再 fit"防数据泄漏；而 /255 对每个样本独立生效，先除后除、
    全量除还是单条除，结果完全一样，不存在泄漏问题。

约定：
    - 归一化系数取自 meta["pixel_max"]（由 data/download.py 随数据集给出），
      不能拍脑袋写 255——digits 的像素上限是 16；
    - 是否归一化由 config["features"]["normalize"] 控制；
    - 训练和预测必须走同一套预处理，predict.py 靠模型包里的 meta 复现本模块逻辑。
"""

import numpy as np


def flatten_images(X, meta):
    """把图像数组展平成二维特征矩阵。

    参数:
        X:    np.ndarray，(n, h, w) 的图像数组（digits）或 (n, h*w) 已展平矩阵（mnist）。
        meta: dict，读取 meta["image_shape"] = (h, w)，用于校验输入尺寸。

    返回:
        np.ndarray，形状 (n, h*w)，dtype 不变。已是二维的输入直接原样返回。

    提示:
        - 三维时用 X.reshape(len(X), -1)；二维时先校验列数等于 h*w 再返回；
        - 线性模型没有空间概念，64/784 个像素就是 64/784 个特征，展平即可。
    """
    raise NotImplementedError


def normalize_pixels(X, meta, config):
    """像素归一化：除以该数据集的像素最大值，把像素压到 [0, 1]。

    参数:
        X:      np.ndarray，形状 (n, d)，展平后的像素矩阵。
        meta:   dict，读取 meta["pixel_max"]（digits=16.0，mnist=255.0）。
        config: dict，读取 config["features"]["normalize"]；为 False 时原样返回。

    返回:
        np.ndarray，形状不变；开启归一化时值域压到 [0, 1]。

    提示:
        - 一行就够：X / meta["pixel_max"]；
        - 这是无状态变换（除以常数），和 StandardScaler 有本质区别，不存在数据泄漏；
        - 系数搞错是静默错误：digits 若误除 255，所有特征缩到 1/16，模型照样跑但精度受损。
    """
    raise NotImplementedError


def split_data(X, y, config):
    """划分训练/测试集（保持类别比例，无标准化——归一化已在 build_features 完成）。

    参数:
        X:      np.ndarray，形状 (n, d)，特征矩阵。
        y:      np.ndarray，形状 (n,)，类别标签 0~9。
        config: dict，读取 config["data"]["test_size"] 和 ["random_state"]。

    返回:
        (X_train, X_test, y_train, y_test): np.ndarray，
            行数比例 = 1-test_size : test_size；stratify=y 保证两集类别比例一致。

    提示:
        - sklearn.model_selection.train_test_split，传 test_size / random_state /
          stratify=y（分层抽样：保证每个类别按比例同时进入训练集和测试集）；
        - 不需要 scaler——像素归一化无状态，没有"在训练集上 fit"的对象要返回。
    """
    raise NotImplementedError


def build_features(X, y, meta, config):
    """组装最终特征矩阵：展平 + 归一化（Pipeline 特征工程一步的总装函数）。

    参数:
        X:      np.ndarray，load_dataset 返回的原始图像（(n,8,8) 或 (n,784)）。
        y:      np.ndarray，形状 (n,)，类别标签。
        meta:   dict，数据集元信息（image_shape / pixel_max）。
        config: dict，读取 config["features"]["normalize"]。

    返回:
        (X_out, y):
            X_out: np.ndarray，形状 (n, h*w)，展平并按 config 归一化后的特征矩阵；
            y:     np.ndarray，原样返回（分类任务不对标签做变换）。

    提示:
        内部依次调 flatten_images → normalize_pixels 即可。
    """
    raise NotImplementedError
