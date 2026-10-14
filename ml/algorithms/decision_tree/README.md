# 手写练习：决策树（含预剪枝）

> 前置文档：`ml/docs/阶段二/05-决策树与集成学习前置学习文档.md`（下称"文档"）。本练习的所有公式都在文档里，这里只做导引，不重复讲解。

## 目标

仅用 NumPy，在 `decision_tree.py` 中手写实现：

1. **entropy**（文档 §2）——信息熵，注意 log₂(0) 的规避思路
2. **gini**（文档 §3）——基尼不纯度
3. **impurity** —— 按 criterion 分派的调度器
4. **split_gain**（文档 §2.3）——父不纯度 − 加权子不纯度
5. **best_split**（文档 §4.2）——排序取相邻中点穷举，本练习的性能瓶颈与核心
6. **DecisionTree 类**（文档 §4/§5）——递归 `fit` / `predict` / `predict_proba`，支持四个预剪枝旋钮

最后用 sklearn 的 `DecisionTreeClassifier`（文档 §7）对答案：同数据同参数，测试准确率差距 ≤ 0.1。

## 实现步骤建议

| 顺序 | 函数 | 对应文档 | 说明 |
| ---- | ---- | ---- | ---- |
| 1 | `entropy` / `gini` | §2/§3 | 热身，各三行。先过已知值测试（纯节点、五五开、文档手算值） |
| 2 | `impurity` | §2/§3 | 调度器 + 非法 criterion 抛 ValueError |
| 3 | `split_gain` | §2.3 | 复用 impurity，权重是样本数占比 |
| 4 | `best_split` | §4.2 | 核心。排序 → 相邻中点 → 逐个算增益，严格大于才更新最佳 |
| 5 | `DecisionTree.fit` | §4/§5 | 递归 `_build(X, y, depth)`，返回嵌套 dict 节点；五条停止条件缺一不可 |
| 6 | `DecisionTree.predict` / `predict_proba` | §4.3 | 递归下行到叶子，返回 prediction / proba |
| 7 | 对答案 | §7 | 跑测试，和 sklearn 对照 |

数据生成器都已实现完整，直接用：

- `generate_sample_data(seed=42)`：与模块 2 完全相同的"考试通过预测"数据——同一份数据，用树再学一遍，对比两种模型的行为；
- `generate_circle_data(seed=0)`：同心圆 400 点，内圆类 0、外环类 1——线性模型结构性无解、树的主场（文档 §1）。

## 通过标准

```bash
cd ml/algorithms/decision_tree
../../../.venv/bin/python -m pytest test_decision_tree.py -v
```

全部测试通过（绿）。骨架状态下 2 条数据生成器测试通过、其余以 `NotImplementedError` 失败；每实现一个函数，对应的测试就转绿。

## 常见坑提示

- **log₂(0)**：`np.log2(0)` 是 `-inf`。用 `np.unique(y, return_counts=True)` 数类别——它只数**出现过**的类，p 恒大于 0，天然规避。不要先 bincount 再过滤零，多一道工序还多一个坑。
- **平局打破规则**：两处平局必须确定性处理，测试钉死了结果——`best_split` 用**严格大于**更新最佳（平局保留先发现的候选）；叶子多数类用 `np.argmax(np.bincount(y))`（平局自动取较小类别号）。
- **相邻相同值要跳过**：特征值 [1, 1, 2] 只有一个有效阈值 1.5；在相同值中间切，会造出一个空子节点。
- **分裂方向**：`X[:, j] <= threshold → 左`，与 sklearn 一致，写反了不影响正确性但和文档图示相反。
- **五条停止条件**：纯净 / max_depth / min_samples_split / best_split 返回 None / 增益不足 min_impurity_decrease——少一条，就有测试变红（比如忘了第 5 条，`min_impurity_decrease=1.0` 的测试会炸）。
- **递归深度**：同心圆 400 点不剪枝时树会很深（几百层），Python 默认递归上限 1000——本练习数据规模内安全；如果做超大数据玩耍时碰到 RecursionError，知道是这回事即可。
- **XOR 盲区**：y=[0,1,1,0] 的四个角点，贪心树在根节点就找不到正增益的一刀，直接躺平成叶子——这不是你写错了，是贪心算法的结构性盲区（文档 §4.4）。测试里 `min_samples_leaf=2` 的用例正是利用了这个现象。
- **浮点比较**：永远 `pytest.approx` / `np.allclose`，不要 `==`。

## 玩耍建议（测试之外）

- 画决策边界：同心圆散点按 y 着色，用 `np.meshgrid` 铺网格过一遍你的 `predict`，`plt.contourf` 画出树的矩形切割，和模块 2 逻辑回归的直线边界对比（文档 §1/§3）。
- 画验证曲线：`max_depth` 从 1 扫到 20，记录 train/test 准确率，亲眼看"训练满分、测试先升后降"的过拟合曲线（文档 §5.3）。
- 换 `criterion="entropy"` 重训，对比树的深度和准确率——验证文档 §3.2 的结论"两者效果几乎一样"。
- 在考试数据上对比：你的手写树、sklearn 树、模块 2 手写的逻辑回归——同一份数据的三种世界观。
- 打印你的树：写一个递归函数把节点 dict 打印成 if-else 文本（sklearn `export_text` 的手写版），体会"可解释性"这个超能力。
