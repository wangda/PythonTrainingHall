"""特征工程（骨架）：像素数组 → 三个 DataLoader，配套文档《08-FashionMNIST分类项目文档.md》§5/§6。

本模块只有一个函数，但它承担着本项目最容易出错的一个决定：
**验证集必须从训练集里切，绝对不能碰测试集。**

为什么（前置文档 §8 的三集合思想）：
    - 训练集：模型反向传播真正学习的数据；
    - 验证集：训练过程中用来"看着办"的数据——早停什么时候触发、checkpoint
      存哪一个 epoch、lr 调度效果如何，这些【训练决策】全靠它；
    - 测试集：训练全部结束后才允许看一眼的数据，用于对外汇报成绩。
    如果从测试集切验证集，等于让训练决策偷看了考题——测试集就不再是
    "没见过的数据"，报出去的准确率虚假偏高，而且没有任何办法再发现这一点。
    Fashion-MNIST 官方给的是 60000 训练 + 10000 测试，验证集只能从那 60000
    里再切一块出来。

约定：
    - 归一化（/255）是无状态变换；可选标准化（(x-均值)/标准差）的统计量
      只能用切分后的【训练子集】计算，再套到验证/测试集上；
    - train_loader 必须 shuffle=True，val/test 不 shuffle（评估不需要乱序，
      乱序反而让两次评估结果不可比）；
    - 返回 torch.Tensor 包装的 DataLoader，dtype：特征 float32、标签 long。
"""

import numpy as np


def build_dataloaders(X_train, y_train, X_test, y_test, config):
    """从训练集切验证集、归一化、转 Tensor、包 DataLoader。

    参数:
        X_train: np.ndarray，(60000, 28, 28) uint8，官方训练集（验证集从这里切）；
        y_train: np.ndarray，(60000,)，标签 0~9；
        X_test:  np.ndarray，(10000, 28, 28) uint8，官方测试集（不许参与任何决策）；
        y_test:  np.ndarray，(10000,)；
        config:  dict，读取：
            config["data"]["val_size"] / ["random_state"]  验证集占比与种子；
            config["features"]["normalize"] / ["standardize"]  归一化/标准化开关；
            config["train"]["batch_size"]                  DataLoader 批大小。

    返回:
        (train_loader, val_loader, test_loader): torch.utils.data.DataLoader 三元组。
            每个 batch 是 (features, labels)：features 形状 (batch, 784) float32，
            labels 形状 (batch,) long；train shuffle=True，val/test shuffle=False。

    提示:
        - 切分用 sklearn.model_selection.train_test_split(X_train, y_train,
          test_size=val_size, random_state=..., stratify=y_train)；
        - 展平 reshape(n, -1) 后除以 255.0（先转 float32 再除）；
        - 标准化若开启：均值/标准差只在训练子集上算，val/test 复用同一组统计量；
        - 数据集包装用 torch.utils.data.TensorDataset + DataLoader；
        - 标签要转 torch.long（CrossEntropyLoss 只认 long 标签）。
    """
    raise NotImplementedError
