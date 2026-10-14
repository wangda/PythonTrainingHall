"""CNN 手写练习骨架：卷积与池化的前向计算（仅前向，不写反向——P1 模块纪律）。

仅用 NumPy 实现：单通道互相关、多通道卷积前向、最大池化、输出尺寸公式。
所有公式与前置文档《09-CNN前置学习文档.md》一一对应，docstring 中标注了章节。

使用流程：
    from cnn_basics import correlate2d, conv2d_forward, max_pool2d, output_size
    # 写完一个函数就跑测试，和 PyTorch 的 F.conv2d / F.max_pool2d 对答案
"""

import numpy as np


def output_size(n, k, p=0, s=1):
    """卷积/池化的输出边长，对应文档 §2.3。

    公式：out = ⌊(n + 2p − k) / s⌋ + 1

    参数:
        n: int，输入边长（宽、高分别调用本函数各算一次）。
        k: int，核（或池化窗口）边长。
        p: int，padding 圈数，默认 0。
        s: int，stride 步长，默认 1。

    返回:
        int，输出边长。

    参考输出（全部为精确值）:
        output_size(4, 3)          == 2    （文档 §2.2 的 4×4⊗3×3 例子）
        output_size(32, 3, p=1)    == 32   （黄金组合 k=3,p=1,s=1 尺寸不变）
        output_size(32, 5)         == 28   （LeNet Conv1，文档 §4.2）
        output_size(28, 2, s=2)    == 14   （LeNet Pool1）

    实现提示:
        一行：整除公式照搬。注意用整数除法 //（公式里的 ⌊⌋ 就是向下取整）。
    """
    raise NotImplementedError


def correlate2d(image, kernel):
    """单通道 2D 互相关（valid，无 padding，stride=1），对应文档 §2.1/§2.2。

    把 kernel 盖在 image 的每个合法位置上，逐元素相乘再求和，得到一个输出数；
    扫完全图得到特征图。深度学习里习惯把互相关叫"卷积"（模板不翻转，文档 §2.1）。

    参数:
        image:  np.ndarray，形状 (H, W)，单通道图。
        kernel: np.ndarray，形状 (k, k)，卷积核（本函数简化为正方形核）。

    返回:
        np.ndarray，形状 (H−k+1, W−k+1)，特征图。

    参考输出（文档 §2.2 的手算例子）:
        image = [[10,10,10,0], [10,10,10,0], [10,10,10,0], [10,10,10,0]]
        kernel = [[1,0,-1], [1,0,-1], [1,0,-1]]   （竖直边缘探测器）
        → [[0, 30], [0, 30]]   （中间一列被点亮）

    实现提示:
        - 双重循环遍历窗口左上角 (i, j)，窗口 = image[i:i+k, j:j+k]，
          输出 = (window * kernel).sum()——教学版允许 for 循环（性能不管，
          工业实现用 im2col 变矩阵乘，文档 §2.5）；
        - 先算输出形状 (H−k+1, W−k+1) 建好数组再填。
    """
    raise NotImplementedError


def conv2d_forward(x, weight, bias=None, stride=1, padding=0):
    """多通道卷积前向（NCHW 格式），对应文档 §2.4。和 PyTorch F.conv2d 完全对齐。

    参数:
        x:      np.ndarray，形状 (N, Cin, H, W)——一批 N 张图，每张 Cin 个通道。
        weight: np.ndarray，形状 (Cout, Cin, k, k)——Cout 个立体核，
                每个核覆盖全部 Cin 个输入通道。
        bias:   np.ndarray 或 None，形状 (Cout,)，每个输出通道一个偏置。
        stride: int，步长（本练习只要求正方形核 + 宽高同 stride）。
        padding: int，四周补 0 的圈数。

    返回:
        np.ndarray，形状 (N, Cout, H_out, W_out)，
        H_out = output_size(H, k, padding, stride)，W_out 同理。

    计算规则（对第 n 张图、第 co 个输出通道、位置 (i, j)）:
        out[n, co, i, j] = Σ_ci Σ窗口 x_pad[n, ci, 窗口] * weight[co, ci]
                           + bias[co]
        即：窗口在**每个输入通道**上分别和对应核片乘加，再把 Cin 个结果全加起来。

    参考行为:
        与 torch.nn.functional.conv2d 的输出 np.allclose（测试里对答案）；
        文档 §2.4 的参数量公式：weight.size = k²·Cin·Cout，bias.size = Cout。

    实现提示:
        - padding 用 np.pad(x, ((0,0),(0,0),(p,p),(p,p))) 在四维数组的 H/W 轴上补 0；
        - 四层循环（n, co, i, j）+ 窗口乘加最直白；窗口在原图上的位置是
          (i*stride, j*stride) 起、取 k×k——别忘了乘 stride；
        - 想少一层循环：对固定的 (n, co, i, j)，窗口块 x_pad[n, :, r:r+k, c:c+k]
          和 weight[co] 形状相同，直接 (窗口 * weight[co]).sum()。
    """
    raise NotImplementedError


def max_pool2d(x, kernel_size=2, stride=2):
    """最大池化前向（NCHW 格式），对应文档 §3。

    对每个通道独立分块，每块取最大值。池化没有可学习参数（纯规则运算）。

    参数:
        x:           np.ndarray，形状 (N, C, H, W)。
        kernel_size: int，池化窗口边长（惯例 2）。
        stride:      int，步长（惯例等于 kernel_size，块不重叠）。

    返回:
        np.ndarray，形状 (N, C, H_out, W_out)，
        H_out = output_size(H, kernel_size, 0, stride)。

    参考输出（文档 §3.2 的手算例子，单张单通道）:
        x = [[1,3,2,0], [5,2,1,4], [0,1,6,2], [2,3,0,1]]（包成 (1,1,4,4)）
        → [[5,4], [3,6]]

    实现提示:
        - 和 conv2d_forward 同一套骨架，只是窗口操作从"乘加"变".max()"；
        - 无核、无偏置、通道之间不混合（每个通道独立池化——这是和卷积的本质区别）。
    """
    raise NotImplementedError
