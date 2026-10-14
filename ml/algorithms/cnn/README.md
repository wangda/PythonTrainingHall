# 手写练习：卷积与池化（前向计算）

> 前置文档：`ml/docs/阶段二/09-CNN前置学习文档.md`（下称"文档"）。本练习的所有公式都在文档里，这里只做导引，不重复讲解。
> **P1 纪律**：只手写**前向**（理解本质），不手写反向传播、不手写完整训练——网络搭建和训练全部用 PyTorch（项目里练）。

## 目标

仅用 NumPy，在 `cnn_basics.py` 中手写实现：

1. **output_size**（文档 §2.3）——输出尺寸公式 `⌊(n+2p−k)/s⌋+1`，一行
2. **correlate2d**（文档 §2.2）——单通道互相关，双重循环滑窗口
3. **conv2d_forward**（文档 §2.4）——多通道 + 多核 + padding + stride 的完整前向，核心
4. **max_pool2d**（文档 §3）——最大池化，通道独立不混合

最后用 PyTorch 的 `F.conv2d` / `F.max_pool2d` 对答案：随机数据上 `np.allclose` 为 True。

## 实现步骤建议

| 顺序 | 函数 | 对应文档 | 说明 |
| ---- | ---- | ---- | ---- |
| 1 | `output_size` | §2.3 | 热身，一行公式。整数除法 `//` 就是向下取整 |
| 2 | `correlate2d` | §2.2 | 双重循环 + `(window * kernel).sum()`，先过文档的手算例子测试 |
| 3 | `conv2d_forward` | §2.4 | 核心。先跑通无 padding/stride=1，再加 padding（`np.pad`），最后加 stride（窗口起点 `i*stride`） |
| 4 | `max_pool2d` | §3 | 复用 conv 的循环骨架，窗口操作换成 `.max()` |
| 5 | 对答案 | 测试文件 | 随机数据上和 PyTorch `allclose` |

## 通过标准

```bash
cd ml/algorithms/cnn
../../../.venv/bin/python -m pytest test_cnn_basics.py -v
```

全部测试通过（绿）。骨架状态下全部以 `NotImplementedError` 失败；每实现一个函数，对应的测试就转绿。测试需要 torch（本模块项目也要用，`pip install torch torchvision`）。

## 常见坑提示

- **stride 别忘了乘**：窗口左上角的坐标是 `(i*stride, j*stride)`，不是 `(i, j)`——stride=1 时两者相同，bug 只有在 stride=2 的测试里才现形；
- **padding 补在哪个轴**：四维数组 `(N, C, H, W)` 只在 H/W 两轴补 0：`np.pad(x, ((0,0),(0,0),(p,p),(p,p)))`；
- **多通道是"分别乘加再相加"**：窗口块 `x_pad[n, :, r:r+k, c:c+k]` 和 `weight[co]` 形状都是 `(Cin, k, k)`，对应位置乘起来**全部**求和——不是逐通道输出；
- **池化没有通道混合**：卷积把 Cin 个通道加在一起，池化每个通道各池各的——`test_池化_通道独立不混合` 钉死了这个区别；
- **浮点比较**：永远 `np.allclose`；
- **教学版允许 for 循环**：四层循环在 224×224 的图上会慢得离谱——这正是 im2col 存在的理由（文档 §2.5），练习数据都很小，性能不管。

## 玩耍建议（测试之外）

- 拿一张真实图片（`matplotlib.image.imread` 转灰度），用你写的 `correlate2d` 跑竖直边缘核和水平边缘核，把两张特征图 `plt.imshow` 出来——亲眼看"边缘被点亮"（文档 §2.2）；
- 写个 `avg_pool2d`（一行之差），对比 max 和 avg 池化后的图片观感；
- 用 `output_size` 把文档 §4.2 的 LeNet 尺寸表逐层验证一遍，再推一遍 CIFAR 版（3 通道 32×32 输入）的尺寸表——项目里搭 LeNet 要用；
- （挑战，🟡）把 `correlate2d` 用 `np.lib.stride_tricks.sliding_window_view` 重写成无显式双重循环版，体会 im2col "窗口展开 → 矩阵乘" 的思想（文档 §2.5）。
