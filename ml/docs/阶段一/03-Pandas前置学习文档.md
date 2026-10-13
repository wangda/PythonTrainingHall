# 03 - Pandas 前置学习文档（教程 + 速查手册）

> **面向读者**：Python 熟练、数学只到初中水平的程序员。
> **覆盖范围**：Series/DataFrame 心智模型、创建与查看、选取数据（loc/iloc/布尔筛选）、数据清洗（缺失值/重复值/类型转换/列操作）、排序与统计、groupby 分组聚合、merge/concat 多表操作、读写文件（CSV/Excel）、与 ML 的衔接（特征工程套路）、性能与避坑。
> **本文环境**：pandas 2.3（所有示例已用 pandas 2.3.0 实际运行验证）。
> **定位**：Pandas 是你学习计划里的"特征工程命脉"——阶段二房价预测、Titanic 项目的数据加载和特征工程全靠它。

---

## 本文怎么用

1. **当教程读**：从头读一遍，每个代码块亲手敲进 `.py` 文件或解释器跑一遍。重点章节（§3 选取数据、§6 groupby、§7 merge）务必动手。
2. **当速查手册用**：忘了某个函数的参数，直接查文末【完整速查表】或跳到对应大节。

**先决知识**：NumPy（Pandas 底层就是 ndarray，`axis`、广播、布尔索引的概念全部通用），见 `ml/docs/01-numpy前置学习文档.md`；画图相关见 `ml/docs/02-Matplotlib前置学习文档.md`。

**阅读约定**：🟢 必须掌握（阶段二特征工程天天用）；🟡 了解即可（用到再查）。

**全文约定**：所有示例统一以 `import pandas as pd` 和 `import numpy as np` 开头（只在首次出现，后续默认已导入）。

---

## 1. 先建立心智模型：Series 与 DataFrame 🟢

### 1.1 Series：一列带标签的数据

**Series = 带标签（index）的一维数组**。可以理解为"NumPy 一维数组 + 每个元素一个名字"，也可以理解为"Excel 里的一列"或"一个有序字典"：

```python
import pandas as pd
import numpy as np

s = pd.Series([90, 85, 88], index=['语文', '数学', '英语'], name='分数')
print(s)
# 语文    90
# 数学    85
# 英语    88
# Name: 分数, dtype: int64

print(s.values, type(s.values))   # [90 85 88] <class 'numpy.ndarray'> —— 底层就是 ndarray
print(s.index)                    # Index(['语文', '数学', '英语'], dtype='object')
print(s['数学'])                  # 85 —— 用标签取值，像字典
print(s + 10)                     # 每科各加 10 分（向量化运算，和 NumPy 一样）
```

### 1.2 DataFrame：一张带行标签、列标签的表

**DataFrame = 若干共用同一个行标签的 Series 拼成的表**。两个类比：

| 类比 | 对应关系 |
|---|---|
| Excel 表格 | DataFrame = 一张工作表；columns = 列头；index = 行号；每个单元格 = 一个值 |
| 数据库表 | DataFrame = 一张表；columns = 字段；index ≈ 主键；`merge` = JOIN、`groupby` = GROUP BY |

```python
df = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'age':  [28, 35, 42],
}, index=['e1', 'e2', 'e3'])
print(df)
#     name  age
# e1  张三   28
# e2  李四   35
# e3  王五   42
print(df.index)     # Index(['e1', 'e2', 'e3'], dtype='object') —— 行标签
print(df.columns)   # Index(['name', 'age'], dtype='object') —— 列标签
print(df.shape)     # (3, 2) —— 3 行 2 列，和 NumPy 的 shape 一样
print(df['name'])   # 取出一列，得到的是 Series（name 列、e1/e2/e3 为 index）
```

**index（行标签）** 和 **columns（列标签）** 是 Pandas 的灵魂——所有选取、对齐、合并操作都围绕标签进行。行标签可以换：

```python
df2 = df.set_index('name')        # 把 name 列变成行标签（返回新表）
print(df2.index)                  # Index(['张三', '李四', '王五'], dtype='object')
print(df2.reset_index())          # 逆操作：行标签还原成普通列，index 变回 0,1,2
```

**注意**：Pandas 几乎所有方法都**返回新对象、不改原表**（`set_index`、`drop`、`rename`、`fillna`……），想保留结果必须接回赋值：`df = df.set_index('name')`。这和 NumPy 的 `reshape` 等函数一个风格。

### 1.3 与 NumPy ndarray 的关系：多了标签和自动对齐

DataFrame 底层数据就是 NumPy ndarray（`df.to_numpy()` 能取出来，见 §9.1），运算规则（向量化、广播、`axis`）也照搬 NumPy。**核心区别只有一个：Pandas 按标签自动对齐，NumPy 按位置对齐。**

```python
a = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
b = pd.Series([10, 20], index=['y', 'z'])
print(a + b)
# x     NaN    ← b 里没有 x，对不齐的位置填 NaN（缺失值）
# y    12.0
# z    23.0
# dtype: float64
```

同样形状的 NumPy 数组直接按位置加；Pandas 先按标签"对表"再运算。**好处**：merge、groupby 之后结果自动按标签对齐，不用关心顺序；**坑**：两个 index 不一致的 Series/DataFrame 运算会悄悄产生一堆 NaN——遇到莫名其妙的 NaN，先检查两边 index。

### 1.4 为什么 ML 特征工程离不开它

机器学习的数据绝大多数是**表格数据**（CSV、数据库导出、Excel）：一行一个样本，一列一个特征。特征工程的全流程——读进来 → 看分布 → 填缺失 → 类别编码 → 分箱 → 归一化 → 多表拼接 → 导出成 NumPy 喂给模型——每一步 Pandas 都有一句对应的写法。NumPy 只管"数字矩阵怎么算"，**"原始表格变成数字矩阵"这前半程全是 Pandas 的活**。阶段二的房价预测、Titanic 项目，代码里一半以上都会是 Pandas。

**本节小结**：Series = 带标签的一列，DataFrame = 带行/列标签的一张表（Excel 工作表 / 数据库表）。方法默认返回新对象。与 NumPy 的关键区别是**按标签自动对齐**，对不齐产生 NaN。

---

## 2. 创建与查看 🟢

### 2.1 创建 DataFrame 的三种方式

```python
# 方式一：字典（列名 -> 列数据），最常用
df = pd.DataFrame({
    'name':   ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
    'dept':   ['技术', '销售', '技术', '人事', '技术', '销售'],
    'age':    [28, 35, 42, 31, 26, 39],
    'salary': [15000, 12000, 22000, 10000, 13500, 14000],
}, index=['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])

# 方式二：list of dict（每行一个字典），适合从 JSON/API 数据建表
rows = [{'name': '张三', 'age': 28}, {'name': '李四', 'age': 35}]
print(pd.DataFrame(rows))
#   name  age
# 0  张三   28
# 1  李四   35

# 方式三：从 NumPy 数组建（columns 指定列名）
arr = np.array([[1, 2], [3, 4]])
print(pd.DataFrame(arr, columns=['a', 'b']))
#    a  b
# 0  1  2
# 1  3  4
```

这张 6 行的员工表 `df` 是本章的主角，后面各节反复用它。真实工作里 90% 的表不是手建的，而是 `pd.read_csv()` 读进来的（见 §8）。

### 2.2 查看数据五件套

拿到任何一张新表，**固定先跑这五个**，30 秒摸清数据底细：

```python
print(df.head(3))      # 前 3 行（默认 5 行）；df.tail(2) 看后 2 行
print(df.shape)        # (6, 4) —— 行数、列数
print(df.dtypes)       # 每列类型：name object / dept object / age int64 / salary int64
df.info()              # 行数、每列非缺失个数、类型、内存占用（直接打印，返回 None）
print(df.describe())   # 数值列的统计摘要
```

`df.describe()` 输出（**每一行统计量都要会读**）：

```text
             age        salary
count   6.000000      6.000000   ← 非缺失值个数（比行数少 = 有缺失）
mean   33.500000  14416.666667   ← 均值
std     6.284903   4103.859972   ← 标准差（数据散开程度）
min    26.000000  10000.000000   ← 最小值
25%    28.750000  12375.000000   ← 第 1 四分位数：25% 的数据比它小
50%    33.000000  13750.000000   ← 中位数：一半数据比它小
75%    38.000000  14750.000000   ← 第 3 四分位数：75% 的数据比它小
max    42.000000  22000.000000   ← 最大值
```

**ML 用途**：`describe()` 是第一道体检——`count` 不齐说明有缺失；`min` 出现负数年龄、-999 之类说明有脏数据；`mean` 和 `50%`（中位数）差很远说明分布偏斜（房价、收入类特征常见），可能要 log 变换（见 §9.5）。只想看类别列加 `df.describe(include='object')`。

**本节小结**：建表三种方式（字典 / list of dict / ndarray）；新表上手五件套 `head / shape / dtypes / info / describe`，describe 的 8 行统计量逐个会读。

---

## 3. 选取数据（重点，最容易混乱）🟢

Pandas 选取数据有**三套并存的方式**，新手最大的混乱来源。先记住一句话：**按列名取列用 `df['col']`；按标签选行/块用 `.loc`；按位置选行/块用 `.iloc`。**

### 3.1 按列选：`df['col']` 与 `df[['col1', 'col2']]`

```python
print(df['age'])          # 取一列 → 返回 Series
print(df[['name', 'salary']])   # 取多列（注意是双层中括号，内层是个 list）→ 返回 DataFrame
#     name  salary
# e1  张三   15000
# e2  李四   12000
# ...（共 6 行）
```

**记忆点**：单层中括号给**一个**列名 → Series；给**列名列表** → DataFrame。`df.name` 点号写法也能取列，但列名含空格/中文或与方法重名时会炸，工程代码里统一用中括号。

### 3.2 .loc[行标签, 列名]：按标签选

**`.loc` 认的是标签（index 的值、列名），不认位置。** 我们的 df 行标签是 `'e1'..'e6'`：

```python
print(df.loc['e2'])                    # 取 e2 这一行 → Series
print(df.loc['e2', 'salary'])          # 12000 —— 行标签 + 列名，取单个值
print(df.loc[['e1', 'e3'], ['name', 'age']])   # 多行多列 → DataFrame
#     name  age
# e1  张三   28
# e3  王五   42

print(df.loc['e1':'e3', 'age'])        # 标签切片取 e1 到 e3 的 age 列
# e1    28
# e2    35
# e3    42
```

**大坑警告：`.loc` 的切片是"闭区间"，两端都包含！** `df.loc['e1':'e3']` 包含 e1、e2、e3 三行——和 Python/NumPy 切片"左闭右开"相反，这是新手必踩的坑。

### 3.3 .iloc[行位置, 列位置]：按位置选

**`.iloc` 认的是整数位置（第几行、第几列，从 0 开始）**，行为和 NumPy 完全一致（左闭右开）：

```python
print(df.iloc[0])          # 第 0 行（不管行标签叫什么）
print(df.iloc[0, 3])       # 15000 —— 第 0 行第 3 列
print(df.iloc[0:3, 1:3])   # 前 3 行 × 第 1~2 列的子块（dept、age 两列，3 行）
print(df.iloc[-1])         # 最后一行
```

**对照记**：`loc` = **l**abel（标签），`iloc` = **i**nteger（整数位置）。行标签恰好是 0,1,2,... 的默认整数索引时两者结果常常一样，但语义不同——切片时 loc 闭区间、iloc 左闭右开，务必分清。

### 3.4 布尔条件筛选（日常最高频）

**用一个布尔 Series 当"筛子"，选出 True 的行**，和 NumPy 布尔索引一模一样：

```python
print(df[df['age'] > 30])                  # age 大于 30 的行（李四、王五、赵六、孙八 共 4 行）

# 多条件：&（且）、|（或），每个条件必须加括号！（运算符优先级的原因）
print(df[(df['age'] > 30) & (df['dept'] == '技术')])   # 王五 1 行
print(df[(df['age'] < 28) | (df['salary'] > 20000)])   # 钱七、王五 2 行

# isin：值在某个集合里（相当于多个 == 的或）
print(df[df['dept'].isin(['技术', '人事'])])           # 张三、王五、赵六、钱七 4 行

# str.contains：字符串包含（文本列筛选神器）
print(df[df['name'].str.contains('张')])               # 张三 1 行
```

**筛选 + 选列的正确写法：条件放 loc 的行位，列名放列位**：

```python
print(df.loc[df['age'] > 30, ['name', 'salary']])
#     name  salary
# e2  李四   12000
# e3  王五   22000
# e4  赵六   10000
# e5  孙八   14000
```

### 3.5 链式赋值陷阱：SettingWithCopyWarning 🟢

**千万不要这样改数据**：

```python
df[df['age'] > 30]['salary'] = 0     # ❌ 链式赋值：先筛再取列再赋值
```

`df[df['age'] > 30]` 返回的可能是**拷贝**而不是原表的视图，第二段 `['salary'] = 0` 改的是那份临时拷贝——原表纹丝不动，Pandas 会抛出 `SettingWithCopyWarning` 警告你（有时还不抛）。改没改成全靠运气，这是 Pandas 最著名的坑。

**正确做法：用 `.loc` 一步定位到"原表的哪些行哪些列"，一次完成赋值**：

```python
df.loc[df['age'] > 30, 'salary'] = 0      # ✅ 条件 + 列名一次写完，直接改原表
print(df['salary'].tolist())   # [15000, 0, 0, 0, 13500, 0]
df.loc[df['age'] > 30, 'salary'] = [12000, 22000, 10000, 14000]   # 改回去，后面还要用
```

同理，`sub = df[df['age'] > 30]; sub['x'] = 1` 这种"先筛出来再改"的写法也会触发警告——要么 `sub = df[...].copy()` 明确要独立副本，要么用 `.loc` 改原表。详见 §10.4。

### 3.6 .at / .iat：取单个值的快速通道 🟡

只取/改**一个格子**时，`at`（按标签）/ `iat`（按位置）比 loc/iloc 快，专用于标量：

```python
print(df.at['e1', 'age'])    # 28 —— 等价 df.loc['e1', 'age']，但更快
print(df.iat[0, 2])          # 28 —— 等价 df.iloc[0, 2]
df.at['e1', 'age'] = 29      # 改单个值也用它
df.at['e1', 'age'] = 28      # 改回去
```

**本节速查表**：

| 需求 | 写法 |
|---|---|
| 取一列 / 多列 | `df['col']` / `df[['a', 'b']]` |
| 按标签选 | `df.loc[行标签, 列名]`（切片**闭区间**） |
| 按位置选 | `df.iloc[行位置, 列位置]`（切片左闭右开） |
| 条件筛选 | `df[条件]`；多条件 `df[(条件1) & (条件2)]` |
| 筛选后选列 | `df.loc[条件, ['a', 'b']]` |
| 集合 / 包含 | `df['c'].isin([...])` / `df['c'].str.contains('x')` |
| 改满足条件的某列 | `df.loc[条件, 'col'] = 值`（**禁止链式赋值**） |
| 取/改单个值 | `df.at[标签, 列]` / `df.iat[行, 列]` |

---

## 4. 数据清洗 🟢

真实数据一定是脏的：有缺失、有重复、类型不对。本节是特征工程的第一道工序。新示例表（故意弄脏的）：

```python
dirty = pd.DataFrame({
    'name':   ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
    'dept':   ['技术', '销售', '技术', '人事', '技术', '销售'],
    'age':    [28.0, np.nan, 42.0, 31.0, np.nan, 39.0],      # 2 个缺失
    'salary': [15000, 12000, np.nan, 10000, 13500, np.nan],  # 2 个缺失
})
```

### 4.1 缺失值：先看清楚有多少

**`isna()` / `isnull()`** 完全等价（两个名字而已），返回同形状的布尔表；配合 `sum()` 统计**每列缺失几个**——这是拿到新表的标准动作：

```python
print(dirty.isna())          # 每个格子 True = 缺失
print(dirty.isna().sum())    # 每列缺失数：name 0, dept 0, age 2, salary 2
print(dirty.isna().sum().sum())   # 4 —— 全表缺失总数
```

### 4.2 dropna：扔掉缺失的

**`df.dropna(axis=0, how='any', subset=None)`** — 删除含缺失的行/列，返回新表。

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `axis` | 删行（0）还是删列（1） | 0 | `1`（某列缺太多时整列放弃） |
| `how` | `'any'` 有一个缺失就删 / `'all'` 全缺失才删 | `'any'` | `'all'` |
| `subset` | 只看哪些列的缺失 | None（全部列） | `['age']` |

```python
print(dirty.dropna())                        # 只剩张三、赵六 2 行（李四、王五、钱七、孙八各有缺失）
print(dirty.dropna(subset=['age']))          # 只要求 age 不缺失：4 行（salary 缺的留下了）
print(dirty.dropna(axis=1, how='any'))       # 有缺失的列整列删掉：只剩 name、dept 两列
```

**经验**：样本珍贵时别急着 dropna，先考虑 fillna；某列缺失过半，通常整列 drop（`axis=1`）或直接弃用。

### 4.3 fillna：填上缺失的

**`df.fillna(value)`** — 用指定值填缺失，返回新表。

```python
print(dirty['age'].fillna(0))                          # 缺失填 0
print(dirty['age'].fillna(dirty['age'].mean()))        # 填均值：35.0（ML 数值特征最常用）
print(dirty['age'].fillna(dirty['age'].median()))      # 填中位数：35.0（有极端值时比均值稳）
print(dirty.ffill())                                   # 用"上一行"的值往下填（时间序列常用）
print(dirty.bfill())                                   # 用"下一行"的值往上填
```

**注意**：老教程里的 `df.fillna(method='ffill')` 写法在 pandas 2.1 起**已弃用**，现在直接写 `df.ffill()` / `df.bfill()`。

**按列填不同的值**（传字典）：

```python
filled = dirty.fillna({'age': dirty['age'].median(), 'salary': 0})
print(filled['age'].tolist())      # [28.0, 35.0, 42.0, 31.0, 35.0, 39.0]
print(filled['salary'].tolist())   # [15000.0, 12000.0, 0.0, 10000.0, 13500.0, 0.0]
```

**ML 经验**：数值特征填中位数（抗极端值）；类别特征填众数或单独一个 `'Unknown'`；更讲究的做法是**按组填**（比如按性别分组填各自的中位数），见 §6.4 transform 和文末自测题。

### 4.4 重复值：duplicated / drop_duplicates

```python
d = pd.DataFrame({'name': ['张三', '李四', '张三'], 'age': [28, 35, 28]})
print(d.duplicated())          # [False, False, True] —— 逐行判断是否和前面某行完全重复
print(d.drop_duplicates())     # 删掉重复行，剩张三、李四 2 行
```

**`drop_duplicates(subset=None, keep='first')`** — 常用参数：`subset=['name']` 只按指定列判重；`keep='last'` 保留最后一次出现的（`keep=False` 全部删掉）。

### 4.5 类型转换：astype / to_datetime / to_numeric

CSV 读进来的列类型经常不对：日期是字符串、金额混着文字。三个转换工具：

```python
# astype：换 dtype
s = pd.Series(['1', '2', '3'])
print(s.astype(int).tolist())        # [1, 2, 3]

# to_datetime：字符串 → 日期时间
d = pd.to_datetime(pd.Series(['2024-01-05', '2024-03-12']))
print(d.dt.month.tolist())           # [1, 3] —— 转完之后 .dt 取出年/月/日/星期等部件

# to_numeric：字符串 → 数字；errors='coerce' 把转不动的变 NaN（脏数据救星）
s2 = pd.Series(['100', '200', 'abc', '300'])
print(pd.to_numeric(s2, errors='coerce').tolist())   # [100.0, 200.0, nan, 300.0] —— 'abc' 变 NaN 再统一处理
```

**`errors` 参数三选一**：`'raise'`（默认，转不动就报错）、`'coerce'`（转不动变 NaN）、`'ignore'`（已弃用，别用）。

### 4.6 列操作：新增、改名、删除、apply/map/replace

```python
df2 = df.copy()

# 新增列：直接赋值一个表达式（向量化，自动按行算）
df2['salary_k'] = df2['salary'] / 1000          # 工资改以"千"为单位
df2['senior'] = df2['age'] >= 35                # 布尔列也常见

# 改列名
df2 = df2.rename(columns={'salary_k': 'salary_kilo'})   # 传字典，只改要改的

# 删列 / 删行
df2 = df2.drop(columns=['senior'])              # 删列用 columns= 或 axis=1
df2 = df2.drop(index=['e6'])                    # 删行用 index=

# apply：对一列逐元素套函数（lambda 最常用）
print(df['name'].apply(lambda x: x + '_员工').tolist())
# ['张三_员工', '李四_员工', '王五_员工', '赵六_员工', '钱七_员工', '孙八_员工']

# map：按字典映射（类别编码常用），也可以传函数
level_map = {'技术': 'T', '销售': 'S', '人事': 'H'}
print(df['dept'].map(level_map).tolist())       # ['T', 'S', 'T', 'H', 'T', 'S']

# replace：值替换（脏数据清洗常用）
print(df['dept'].replace({'技术': '研发'}).tolist())   # ['研发', '销售', '研发', '人事', '研发', '销售']
```

**apply vs map 的区别**：`map` 逐元素、支持字典映射（`dept` 列这种）；`apply` 也是逐元素但只收函数，更通用。DataFrame 级别的 `df.apply(func, axis=1)` 可以**逐行**处理（`row` 是一整行 Series），灵活但慢，能向量化就别用（见 §10.1）。

**本节速查表**：

| 需求 | 写法 |
|---|---|
| 统计每列缺失 | `df.isna().sum()` |
| 删缺失行 / 列 | `df.dropna()` / `df.dropna(axis=1)` |
| 填缺失 | `df.fillna(值/字典)`、`df.ffill()`、`df.bfill()` |
| 去重 | `df.drop_duplicates(subset=[...], keep='first')` |
| 转类型 | `astype` / `pd.to_datetime` / `pd.to_numeric(errors='coerce')` |
| 新增列 | `df['新列'] = 表达式` |
| 改名 / 删除 | `rename(columns={})` / `drop(columns=[])` |
| 逐元素变换 / 映射 | `apply(lambda)` / `map(dict)` / `replace(dict)` |

---

## 5. 排序与统计 🟢

### 5.1 排序：sort_values / sort_index

**`df.sort_values(by, ascending=True)`** — 按列值排序；**`df.sort_index()`** — 按行标签排序。都返回新表。

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `by` | 按哪列排（可多列传 list） | 必填 | `'salary'`、`['dept', 'salary']` |
| `ascending` | 升序？ | True | `False`（降序）、多列时传等长 list |

```python
print(df.sort_values('salary', ascending=False))          # 按工资从高到低：王五、张三、孙八...
print(df.sort_values(['dept', 'age'], ascending=[True, False]))   # 先按部门升序，同部门按年龄降序
print(df.sort_index(ascending=False))                     # 按行标签 e6 → e1 倒排
```

### 5.2 类别列神器：value_counts / unique / nunique

```python
print(df['dept'].value_counts())          # 每个类别出现几次：技术 3, 销售 2, 人事 1
print(df['dept'].value_counts(normalize=True))   # 改成占比：技术 0.5, 销售 0.333..., 人事 0.166...
print(df['dept'].unique())                # ['技术' '销售' '人事'] —— 去重后的取值
print(df['dept'].nunique())               # 3 —— 有几个不同取值
```

**ML 用途**：拿到类别特征先 `value_counts()`——Titanic 里看男女比例、房价里看各城区样本量；`nunique()` 太大（比如几千个取值）说明不适合直接 one-hot。

### 5.3 聚合统计：mean / median / std / min / max

```python
print(df['age'].mean(), df['age'].median(), df['age'].std())   # 33.5 33.0 6.2849...
print(df.mean(numeric_only=True))      # 对每列聚合：age 33.5, salary 14416.67（numeric_only 跳过文本列）
print(df.max(numeric_only=True))       # 每列最大值
```

**与 NumPy 的两点重要差异**（01 文档 §7 的 axis 概念照搬，但默认值不同）：

1. **默认方向不同**：NumPy 的 `np.mean(a)` 不写 axis 是把**所有元素**压成一个数；Pandas 的 `df.mean()` 默认 `axis=0` **按列**聚合，每列一个结果。想按行聚合写 `df.mean(axis=1, numeric_only=True)`。
2. **默认跳过 NaN**：Pandas 聚合默认忽略缺失值（NaN 不参与计算），NumPy 的 `np.mean` 遇到 NaN 直接返回 NaN。`dirty['age'].mean()` = 35.0（自动跳过 2 个缺失），不需要像 NumPy 那样换 `nanmean`。

```python
print(dirty['age'].mean())    # 35.0 —— NaN 被自动跳过
```

**本节小结**：排序 `sort_values(by, ascending)`；类别列 `value_counts()`；聚合默认按列（axis=0）且自动跳过 NaN——与 NumPy 的两大差异记牢。

---

## 6. groupby 分组聚合（重点，学习清单核心）🟢

### 6.1 心智模型：拆分 → 应用 → 合并

`groupby` 干的事用一句话说：**按某列把表拆成几组，每组各自算统计量，再把结果合并回一张表**（split-apply-combine）。

类比"按部门统计平均工资"：把员工表按 dept 列拆成"技术/销售/人事"三摞 → 每摞各自算 salary 均值 → 三摞的结果拼回一张小表。写过 SQL 的话，`groupby('dept')['salary'].mean()` 就是 `SELECT dept, AVG(salary) FROM emp GROUP BY dept`。

### 6.2 基本用法与 agg

```python
print(df.groupby('dept')['salary'].mean())
# dept
# 人事    10000.000000
# 技术    16833.333333     ← (15000+22000+13500)/3
# 销售    13000.000000
# Name: salary, dtype: float64
```

**`groupby('col').agg(...)`** — 一组算多个统计量：

```python
# 写法一：传函数名列表
print(df.groupby('dept')['salary'].agg(['mean', 'max', 'count']))

# 写法二：字典，对不同列用不同聚合（老写法，简单场景够用）
print(df.groupby('dept').agg({'salary': 'mean', 'age': 'max'}))

# 写法三：命名聚合（推荐，结果列名自己起，最清晰）
print(df.groupby('dept').agg(
    avg_salary=('salary', 'mean'),      # 新列名 = ('源列', 聚合函数)
    max_age=('age', 'max'),
    n=('name', 'count'),
))
#       avg_salary  max_age  n
# dept
# 人事    10000.000       31  1
# 技术   16833.333       42  3
# 销售   13000.000       39  2
```

常用聚合函数名：`'mean' 'median' 'std' 'min' 'max' 'sum' 'count' 'nunique'`。

### 6.3 as_index=False：分组键要不要变成行标签

默认情况下分组键（dept）会变成结果的**行标签**；加 `as_index=False` 让它保留为**普通列**、行标签用回 0,1,2,...：

```python
print(df.groupby('dept', as_index=False)['salary'].mean())
#   dept        salary
# 0  人事  10000.000000
# 1  技术  16833.333333
# 2  销售  13000.000000
```

效果等价于 `df.groupby('dept')['salary'].mean().reset_index()`。**建议**：结果还要继续 merge 或导出时加 `as_index=False`，后续处理更顺手。

### 6.4 transform：分组统计"贴回"原表 🟢

`agg` 把每组压成一行；**`transform` 把每组的统计量广播回原表的每一行**（形状和原表一致）。这是特征工程的重武器：

```python
# 例 1：加一列"本部门平均工资"（每行都贴上自己部门的均值）
df2 = df.copy()
df2['dept_avg_salary'] = df2.groupby('dept')['salary'].transform('mean')
print(df2[['name', 'dept', 'salary', 'dept_avg_salary']])
# 张三/技术 → 16833.33，李四/销售 → 13000，王五/技术 → 16833.33，...

# 例 2：组内标准化（每行 = 自己减本组均值再除以本组标准差）—— ML 分组特征工程经典套路
g = df2.groupby('dept')['salary']
df2['salary_z'] = (df2['salary'] - g.transform('mean')) / g.transform('std')
print(df2.loc[['e1', 'e3'], ['salary', 'salary_z']])
#      salary  salary_z
# e1  15000   -0.404095   ← 张三在技术部内部偏低
# e3  22000    1.138813   ← 王五在技术部内部偏高
```

另一个高频用途：**按组填缺失值**（同组中位数填该组缺失），见文末自测题第 3 题。

### 6.5 size vs count 的区别

```python
print(dirty.groupby('dept').size())     # 每组行数（不管缺不缺失）：人事 1, 技术 3, 销售 2
print(dirty.groupby('dept').count())    # 每组每列的"非缺失个数"：技术组 age 只有 2（有 1 个缺失）
```

**记法**：`size` = 这组一共几行；`count` = 这列有几个非缺失值。想统计"每组样本数"用 `size()`。

### 6.6 实战：Titanic 风格数据按舱位/性别统计存活率

阶段二 Titanic 项目的预演。构造 10 行小数据（`survived` 1=存活 0=遇难）：

```python
titanic = pd.DataFrame({
    'pclass':   [1, 1, 2, 2, 3, 3, 1, 2, 3, 3],
    'sex':      ['female', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'male'],
    'age':      [29.0, 35.0, 22.0, 40.0, 18.0, 45.0, 30.0, 26.0, 33.0, 50.0],
    'fare':     [100.0, 80.0, 30.0, 25.0, 10.0, 8.0, 90.0, 20.0, 12.0, 7.0],
    'survived': [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
})

# 按舱位统计存活率（survived 是 0/1，均值就是存活率）
print(titanic.groupby('pclass')['survived'].mean())
# pclass
# 1    0.666667    ← 头等舱 2/3 存活
# 2    0.333333    ← 二等舱 1/3
# 3    0.250000    ← 三等舱 1/4

# 按性别统计
print(titanic.groupby('sex')['survived'].mean())
# sex
# female    0.8   ← 女性 4/5 存活
# male      0.0   ← 男性全部遇难

# 两个维度交叉：按 舱位×性别 统计存活率和平均票价
print(titanic.groupby(['pclass', 'sex']).agg(
    存活率=('survived', 'mean'),
    平均票价=('fare', 'mean'),
))
#               存活率    平均票价
# pclass sex
# 1      female  1.0   95.0
#        male    0.0   80.0
# 2      female  0.5   25.0
#        male    0.0   25.0
# 3      female  1.0   10.0
#        male    0.0    9.0
```

一眼看出"舱位越高、女性，存活率越高"——这就是 EDA（探索性数据分析）的日常。**技巧**：标签是 0/1 时，`mean` 直接就是"比率"，算转化率、点击率同理。

**本节小结**：心智模型"拆分-应用-合并"；`groupby('col').agg(...)` 记三种写法（列表 / 字典 / 命名聚合）；`as_index=False` 让分组键保留为列；`transform` 把组统计量贴回原表（组内标准化、按组填缺失）；`size()` 数行数、`count()` 数非缺失；0/1 标签的 mean = 比率。

---

## 7. merge / join 多表操作（重点，学习清单核心）🟢

### 7.1 心智模型：两张表"按钥匙对钥匙"拼接

`pd.merge` 就是 SQL 的 JOIN：**指定一把"钥匙"（共同的列），左表每一行拿着自己的钥匙去右表里找，配上对就拼成一行**。本节用两张小表完整演示——用户表和订单表：

```python
users = pd.DataFrame({
    'user_id': [1, 2, 3, 4],
    'city':    ['北京', '上海', '北京', '广州'],
})
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'user_id':  [1, 1, 2, 5, 5],
    'amount':   [99.0, 49.0, 150.0, 200.0, 80.0],
})
```

注意构造时故意埋的两种情况：用户 3、4 **没有订单**；订单 104、105 的 user_id=5 **在用户表里不存在**。四种 join 的差异全在这几个"对不上"的行上。

### 7.2 how 参数：四种连接方式

**`pd.merge(left, right, how='inner', on=None)`** — 按钥匙拼接两张表。

| how | 含义 | 本例结果 |
|---|---|---|
| `'inner'`（默认） | 只保留**两边都有**的钥匙 | 3 行：用户 1 的 2 单 + 用户 2 的 1 单 |
| `'left'` | **左表的钥匙一个不漏**，右表配不上填 NaN；右表一键多行时结果会相应变多 | 5 行：用户 1 有 2 单占 2 行；用户 3、4 的订单列是 NaN |
| `'right'` | **右表一行不丢** | 5 行：订单 104、105 的 city 是 NaN |
| `'outer'` | 两边全集，配不上都填 NaN | 7 行：用户 3、4 和订单 104、105 都保留 |

```python
print(pd.merge(users, orders, on='user_id'))                    # inner（默认）：3 行
print(pd.merge(users, orders, on='user_id', how='left'))        # 5 行，user 3/4 的 order_id、amount 为 NaN
print(pd.merge(users, orders, on='user_id', how='right'))       # 5 行，order 104/105 的 city 为 NaN
print(pd.merge(users, orders, on='user_id', how='outer'))       # 7 行，两边配不上的全保留

# how='left' 的输出（重点核对 NaN 位置）：
#    user_id city  order_id  amount
# 0        1   北京     101.0    99.0
# 1        1   北京     102.0    49.0
# 2        2   上海     103.0   150.0
# 3        3   北京       NaN     NaN     ← 用户 3 没下过单
# 4        4   广州       NaN     NaN     ← 用户 4 没下过单
```

**ML 场景**：用户信息表 + 行为日志表、商品表 + 销量表……特征工程经常要 left join 把宽表拼出来。**"以谁为准"就选谁放左边 + how='left'**——左表的钥匙一个不漏。但注意：左表有、右表有多行匹配时，结果行数会**超过**左表行数（本例用户 1 占了 2 行）；想让"每个用户恰好一行"，得先 groupby 聚合右表再 merge（自测题第 5 题就是这么干的）。

### 7.3 on / left_on / right_on / suffixes

| 参数 | 含义 | 默认值 |
|---|---|---|
| `on` | 两表同名的钥匙列 | None（自动用所有同名列，**建议显式写**） |
| `left_on` / `right_on` | 两表钥匙列名**不同**时分别指定 | None |
| `suffixes` | 两表有同名非钥匙列时，结果列名的后缀 | `('_x', '_y')` |

```python
# 两表钥匙列名不同
orders2 = orders.rename(columns={'user_id': 'uid'})
m = pd.merge(users, orders2, left_on='user_id', right_on='uid')
print(m.shape)        # (3, 5) —— 结果里 user_id 和 uid 两列都在，按需 drop 一个
```

### 7.4 validate：防止意外多对多爆炸 🟢

如果钥匙在两表里都有重复，merge 会做**笛卡尔积**——左表某钥匙 3 行 × 右表同钥匙 3 行 = 结果 9 行。数据一脏，几万行悄悄变几百万行（"行数爆炸"）。**`validate` 参数让 pandas 帮你检查钥匙关系，不符合就报错**：

| validate 取值 | 含义 |
|---|---|
| `'1:1'` | 两表钥匙都唯一 |
| `'1:m'` | 左表钥匙唯一，右表可重复（用户→订单就是这种） |
| `'m:1'` | 左表可重复，右表唯一 |

```python
m = pd.merge(users, orders, on='user_id', validate='1:m')   # 用户唯一、订单可重复：通过
print(m.shape)   # (3, 4) —— 3 行（用户 1 的 2 单 + 用户 2 的 1 单），4 列（user_id/city/order_id/amount）

bad = pd.DataFrame({'user_id': [1, 1], 'tag': ['a', 'b']})  # 假设右表 user_id 意外重复
try:
    pd.merge(users, bad, on='user_id', validate='1:1')      # 左表也宣称唯一，但右表 user 1 有 2 行
except pd.errors.MergeError as e:
    print('MergeError:', e)   # MergeError: Merge keys are not unique in right dataset; not a one-to-one merge
```

**工程习惯**：merge 之后立刻 `print(m.shape)` 检查行数是否符合预期；重要 merge 加上 `validate`。这两招能挡住 90% 的"多表拼接翻车"。

### 7.5 concat：上下堆叠 vs 左右拼接

**`pd.concat([df1, df2, ...], axis=0, ignore_index=False)`** — 沿着轴把表"摞"起来，不做钥匙匹配。

```python
df_a = pd.DataFrame({'name': ['张三'], 'age': [28]})
df_b = pd.DataFrame({'name': ['李四'], 'age': [35]})

print(pd.concat([df_a, df_b]))                      # 上下堆叠（默认 axis=0），行标签是 0、0（保留了原标签）
print(pd.concat([df_a, df_b], ignore_index=True))   # 重新编行标签 0、1 —— 堆叠时几乎都该加它
print(pd.concat([df_a, df_b], axis=1))              # 左右拼接：按行标签对齐，列变多
```

**merge vs concat 怎么选**：需要"按钥匙配对"用 merge（订单配对用户）；只是"结构一样的表摞一起 / 并排摆"用 concat（多个月份的 CSV 纵向合并、两批特征横向拼）。`axis=1` 的 concat 是按行标签硬对齐，标签对不上就出 NaN——多数情况下你想要的是 merge。

**本节速查表**：

| 需求 | 写法 |
|---|---|
| 按键拼接 | `pd.merge(left, right, on='key', how=...)` |
| 两表键名不同 | `left_on=..., right_on=...` |
| 防行数爆炸 | `validate='1:1' / '1:m' / 'm:1'` + 检查 `m.shape` |
| 上下堆叠 | `pd.concat([df1, df2], ignore_index=True)` |
| 左右并排 | `pd.concat([df1, df2], axis=1)`（按行标签对齐，慎用于键配对场景） |

---

## 8. 读写文件 🟢

### 8.1 read_csv：最常用的参数

**`pd.read_csv(filepath_or_buffer, ...)`** — 读 CSV 成 DataFrame。先把员工表存成文件当素材：

```python
df.to_csv('/tmp/pd_demo_emp.csv', index=False)     # index=False：别把行标签 e1..e6 存进去
df2 = pd.read_csv('/tmp/pd_demo_emp.csv')
print(df2.shape)          # (6, 4)
print(df2.dtypes)         # name/dept 是 object，age/salary 是 int64 —— 类型自动推断
```

| 参数 | 含义 | 默认值 | 常用取值 |
|---|---|---|---|
| `sep` | 分隔符 | `','` | `'\t'`（TSV）、`';'` |
| `header` | 第几行当列名 | 0（首行） | `None`（没有表头，配 `names=[...]` 自己起） |
| `usecols` | 只读哪些列 | None（全读） | `['name', 'age']`（大文件省内存） |
| `encoding` | 文件编码 | `'utf-8'` | `'gbk'`（国内老 Excel 导出的 CSV 常见） |
| `parse_dates` | 哪些列解析成日期 | False | `['order_date']` |
| `na_values` | 哪些字符串算缺失 | 内置一组 | `['NA', '无', '-']`（脏数据里的假值） |
| `nrows` | 只读前 N 行 | None | `1000`（大文件先抽样看结构） |
| `dtype` | 手动指定列类型 | None（自动推断） | `{'id': str}`（防止编号被当数字） |

```python
# 综合示例：只读两列、前 3 行
df3 = pd.read_csv('/tmp/pd_demo_emp.csv', usecols=['name', 'age'], nrows=3)
print(df3.shape)     # (3, 2)
```

**两个最常见的坑**：① 中文乱码 → 加 `encoding='gbk'`（或 `'gb18030'`）再试；② 日期列读进来是字符串 → 加 `parse_dates=['日期列']`，或读完后 `pd.to_datetime(df['日期列'])`。

### 8.2 to_csv：index=False 的重要性

**`df.to_csv(path, index=True)`** — 存 CSV。**几乎总是写 `index=False`**：默认会把行标签（0,1,2,...）存成文件的第一列，下次读回来就多出一列莫名其妙的 `Unnamed: 0`。只有行标签本身有业务含义（比如用户 ID 被 set_index 了）时才保留。

```python
df.to_csv('/tmp/pd_demo_out.csv', index=False)
back = pd.read_csv('/tmp/pd_demo_out.csv')
print(back.columns.tolist())   # ['name', 'dept', 'age', 'salary'] —— 干净，没有 Unnamed: 0
```

### 8.3 read_excel 简述

**`pd.read_excel(path, sheet_name=0)`** — 读 Excel（需要装 openpyxl，本文环境已装）。`sheet_name` 给工作表名或序号；其余参数（`header/usecols/nrows`）和 read_csv 一致。对应写出 `df.to_excel(path, index=False)`：

```python
df.to_excel('/tmp/pd_demo_emp.xlsx', index=False)
dfx = pd.read_excel('/tmp/pd_demo_emp.xlsx', sheet_name=0)
print(dfx.shape)     # (6, 4)
```

### 8.4 大数据量：parquet 与 chunksize 🟡

- **parquet 格式**：列式存储，比 CSV **快好几倍、体积小好几倍、还能保住列类型**（CSV 存不了类型，日期读回来全变字符串）。`df.to_parquet('x.parquet')` / `pd.read_parquet('x.parquet')`（需要装 pyarrow）。数据量上了百万行就换它，用到再查。
- **chunksize 分块读**：文件大到内存装不下时，`read_csv(..., chunksize=10000)` 返回一个迭代器，每次给你 1 万行的小表，逐块处理后汇总：

```python
# 造一个 10 行的文件演示（真实场景是几千万行）
titanic.to_csv('/tmp/pd_demo_titanic.csv', index=False)
total = 0
for chunk in pd.read_csv('/tmp/pd_demo_titanic.csv', chunksize=4):
    total += chunk['survived'].sum()     # 每块 4 行，逐块累加
print(total)     # 4 —— 不用把整个文件读进内存也能算总和
```

**本节小结**：读表主力 `pd.read_csv`，记 `encoding/sep/usecols/parse_dates/na_values/nrows`；写出 `to_csv(index=False)`；Excel 用 `read_excel`；大文件换 parquet 或 `chunksize` 分块。

---

## 9. 与 ML 的衔接 🟢

特征工程的终点是"一张干净的数字矩阵"。本节把 Pandas 和 NumPy/sklearn 接起来。

### 9.1 DataFrame 转 NumPy：.to_numpy()

**`df.to_numpy()`** — 取出底层 ndarray，喂给 sklearn 或手写算法（阶段一 KMeans 就用它）：

```python
X = titanic[['age', 'fare']].to_numpy()      # 特征矩阵
y = titanic['survived'].to_numpy()           # 标签向量
print(X.shape, X.dtype)    # (10, 2) float64
print(y.shape, y.dtype)    # (10,) int64
```

**注意**：选取的列类型必须统一（都是数值），混着文本列会转成 object 类型，模型直接报错。文本/类别列先做编码（§9.2）再 to_numpy。

### 9.2 类别编码：pd.get_dummies（one-hot）

模型只认数字，类别列（性别、舱位、城市）要编码成数字。**one-hot 编码**：一个 k 取值的类别列 → k 个 0/1 列。

```python
print(pd.get_dummies(titanic, columns=['sex'], dtype=int))
#    pclass  age  fare  survived  sex_female  sex_male
# 0       1   29   100         1           1         0
# 1       1   35    80         0           0         1
# ...（sex 列消失，变成 sex_female / sex_male 两列 0/1）
```

**`pd.get_dummies(df, columns=[...], dtype=int)`** — 对指定列做 one-hot。注意 `dtype=int`：pandas 2.x 默认生成 **True/False 布尔列**，加 `dtype=int` 得到 0/1，喂模型更直观。

**Titanic 项目用途**：`sex`、`embarked`（登船港口）列都必须这样编码后才能进模型。类别取值特别多的列（几百上千种）不适合 one-hot（列数爆炸），那是更后面的话题（目标编码、Embedding）。

### 9.3 分箱：pd.cut / pd.qcut

把连续数值切成几个区间（离散化），让模型按区间学规律：

```python
# cut：按"等宽边界"切（你自己定边界）
titanic['age_bin'] = pd.cut(titanic['age'], bins=[0, 18, 35, 60], labels=['少年', '青年', '中年'])
print(titanic['age_bin'].value_counts())    # 青年 6, 中年 3, 少年 1（含边界 18 算少年）

# qcut：按"等频"切（每桶样本数差不多），不用自己想边界
titanic['fare_q'] = pd.qcut(titanic['fare'], q=3, labels=['低', '中', '高'])
print(titanic['fare_q'].value_counts())     # 高 4, 低 3, 中 3 —— 自动按分位数切（样本少时做不到完全均匀）
```

**记法**：`cut` 你定边界（年龄 0-18/18-35/35-60 这种业务区间），`qcut` 它定边界（保证每桶一样多）。分箱结果也是类别，后续同样要 one-hot。

### 9.4 归一化 / 标准化

不同特征量纲差太大（年龄 20~60、票价 7~100），直接喂模型，大数值特征会"压住"小数值特征。两种常见变换（都是纯向量化，Pandas/NumPy 都行）：

```python
cols = ['age', 'fare']

# 标准化（z-score）：减均值除标准差 → 每列均值 0、标准差 1
z = (titanic[cols] - titanic[cols].mean()) / titanic[cols].std()
print(z.mean().round(6).tolist())    # [0.0, -0.0]（均值变为 0；-0.0 就是 0）
print(z.std().tolist())              # [1.0000000000000002, 0.9999999999999998]（标准差变为 1，末位是浮点误差）

# 最小-最大归一化：(x - min) / (max - min) → 压到 [0, 1]
mm = (titanic[cols] - titanic[cols].min()) / (titanic[cols].max() - titanic[cols].min())
print(mm.min().tolist(), mm.max().tolist())   # [0.0, 0.0] [1.0, 1.0]
```

阶段二会学到 sklearn 的 `StandardScaler`/`MinMaxScaler` 干同样的事（还能记住训练集的统计量、原样应用到测试集——**测试集必须用训练集的均值/标准差，不能自己重新算**，这点手写时容易错）。

### 9.5 偏斜数据的 log 变换

房价、收入这类特征常常"大部分挤在左边，少数天价拖出长尾巴"（右偏）。直接进模型，几个极端值带歪学习。**取 log 能把长尾巴压回来**（log(10)=2.3、log(100)=4.6、log(1000)=6.9——大数被压得狠）：

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].hist(titanic['fare'], bins=8, edgecolor='black')
axes[0].set_title('fare raw (right-skewed)')   # 中文标题需先配字体，见 02-Matplotlib 文档 §7
axes[1].hist(np.log1p(titanic['fare']), bins=8, edgecolor='black')   # log1p = log(1+x)，防 log(0)
axes[1].set_title('after log1p')
fig.savefig('/tmp/pd_demo_log.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

**房价项目用途**：`SalePrice`（目标值）和 `GrLivArea`（面积）这类特征通常都要 log 变换。**用 `np.log1p`（即 log(1+x)）而不是 `np.log`**，数据里有 0 时不会算出 -inf；预测完记得用 `np.expm1` 变回去。

**本节小结**：进模型前 `df[特征列].to_numpy()`；类别列 `pd.get_dummies(dtype=int)`；连续特征按需 `cut/qcut` 分箱、标准化或 min-max 归一化；右偏特征 `np.log1p`。这就是阶段二每个项目开头那段特征工程代码的全部套路。

---

## 10. 性能与避坑 🟢

### 10.1 优先向量化/内置方法，别用 for 循环逐行

Pandas 底层是 NumPy，向量化操作比逐行快几十上百倍（原理见 01 文档 §11.1）。**看到 `for` 或 `iterrows()` 逐行处理，先停下来想能不能用列运算表达**：

```python
# ❌ 慢：逐行循环
# total = 0
# for _, row in df.iterrows():
#     total += row['age'] * 2

# ✅ 快：列运算一句话
total = (df['age'] * 2).sum()
print(total)     # 402
```

常见的"循环改列"都有向量化替代：`df['年薪'] = df['salary'] * 12`（算术）、`df['长名'] = df['name'].str.len() > 2`（字符串 `.str` 系列）、`np.where(df['age'] > 30, '老', '少')`（条件二选一）。确实绕不开逐行时，用 `df.apply(func, axis=1)` 至少比 `iterrows` 规范。

### 10.2 链式赋值陷阱（再强调）

§3.5 讲过，这里重申因为它太常见：**`df[条件]['col'] = 值` 改的可能是一份临时拷贝**，原表没动，还会收到 `SettingWithCopyWarning` 警告。正确写法永远是 **`df.loc[条件, 'col'] = 值`**——行、列一次定位，直接作用在原表上。

### 10.3 inplace 参数：不推荐

很多方法有 `inplace=True` 参数（直接改原表、无返回值），比如 `df.dropna(inplace=True)`。老教程爱用，**现在不推荐**，两个理由：

1. **链式写法直接断掉**：`df.dropna(inplace=True).head()` 会报错——inplace 方法返回 `None`，后面什么都接不了。而 `df = df.dropna().reset_index()` 这种流水线写法是 Pandas 的主流风格。
2. **可读性差**：`df = df.fillna(0)` 一眼看出"新表替换旧表"；`df.fillna(0, inplace=True)` 藏在句尾的参数容易被看漏。

**统一习惯：不用 inplace，全部用"接回赋值"**：`df = df.dropna()`。性能上没有损失（inplace 内部多数也是先拷贝再替换）。

### 10.4 SettingWithCopyWarning 出现时的正确应对

看到这个警告，按三步处理：

1. **先判断意图**：你是想改原表，还是想改筛出来的子表？
2. 想改**原表** → 合并成一步 `.loc`：`df.loc[条件, 'col'] = 值`；
3. 想改**子表**（筛出来当新数据集用）→ 筛选时显式 `.copy()`：`sub = df[df['age'] > 30].copy()`，之后随便改 sub 不会再警告。

**不要做的事**：用 `pd.set_option('mode.chained_assignment', None)` 把警告全局关掉——警告是对的，代码才是错的。

**本节小结**：列运算替代 for/iterrows；改数据统一 `.loc`；拒绝 inplace、用接回赋值；SettingWithCopyWarning 用 `.loc` 或 `.copy()` 正面解决。

---

## 完整速查表

**心智模型**：`pd.Series(data, index=)` 带标签的一列 ｜ `pd.DataFrame(dict)` 带行/列标签的表 ｜ `df.index / df.columns / df.shape` ｜ `set_index('col')` / `reset_index()` ｜ 方法默认返回新表，接回赋值 ｜ 按标签自动对齐，对不齐产生 NaN

**创建与查看**：`pd.DataFrame(dict / list of dict / ndarray, columns=)` ｜ `head(n) / tail(n)` ｜ `shape` ｜ `dtypes` ｜ `info()` ｜ `describe()`（count/mean/std/min/25%/50%/75%/max）

**选取**：`df['col']` 单列 Series ｜ `df[['a','b']]` 多列 ｜ `df.loc[行标签, 列名]`（切片闭区间） ｜ `df.iloc[行位置, 列位置]`（左闭右开） ｜ `df[条件]` 布尔筛选，多条件 `(A) & (B)` ｜ `df.loc[条件, ['a']]` 筛选+选列 ｜ `isin([...])` ｜ `str.contains('x')` ｜ 改值 `df.loc[条件, 'col'] = v`（**禁止 `df[..][..] = v`**） ｜ 单值 `at` / `iat`

**清洗**：`isna().sum()` 每列缺失数 ｜ `dropna(axis=, how=, subset=)` ｜ `fillna(值/字典)` ｜ `ffill() / bfill()` ｜ `drop_duplicates(subset=, keep=)` ｜ `astype` / `pd.to_datetime` / `pd.to_numeric(errors='coerce')` ｜ 新列 `df['c'] = 表达式` ｜ `rename(columns={})` ｜ `drop(columns=) / drop(index=)` ｜ `apply(lambda)` / `map(dict)` / `replace(dict)`

**排序统计**：`sort_values(by, ascending=)` ｜ `sort_index()` ｜ `value_counts(normalize=)` ｜ `unique() / nunique()` ｜ `mean/median/std/min/max`（默认按列、自动跳过 NaN）

**groupby**：`groupby('col')['x'].mean()` ｜ `agg(['mean','max'])` / `agg({'a':'mean'})` / `agg(新名=('列','函数'))` ｜ `as_index=False` ｜ `transform('mean')` 组统计贴回原表 ｜ `size()` 组行数 vs `count()` 非缺失数 ｜ 0/1 列的 mean = 比率

**多表**：`pd.merge(l, r, on='key', how='inner/left/right/outer')` ｜ `left_on= / right_on=` 键名不同 ｜ `validate='1:1'/'1:m'/'m:1'` 防爆炸 ｜ merge 后检查 `.shape` ｜ `pd.concat([a, b], ignore_index=True)` 上下堆叠 ｜ `pd.concat([a, b], axis=1)` 左右并排（按行标签对齐）

**读写**：`pd.read_csv(f, encoding=, sep=, header=, usecols=, parse_dates=, na_values=, nrows=)` ｜ `df.to_csv(f, index=False)` ｜ `pd.read_excel(f, sheet_name=)` ｜ 大文件 `chunksize=N` 分块 ｜ 更快更省 `to_parquet / read_parquet`（需 pyarrow）

**ML 衔接**：`df[cols].to_numpy()` 转特征矩阵 ｜ `pd.get_dummies(df, columns=[], dtype=int)` one-hot ｜ `pd.cut(x, bins=, labels=)` 等宽分箱 ｜ `pd.qcut(x, q=)` 等频分箱 ｜ 标准化 `(df-df.mean())/df.std()` ｜ min-max `(df-df.min())/(df.max()-df.min())` ｜ 右偏 `np.log1p`，还原 `np.expm1`

**避坑**：向量化替代 iterrows ｜ 链式赋值 `df[..][..]=v` 禁止，用 `.loc` ｜ 不用 `inplace=True`，接回赋值 ｜ SettingWithCopyWarning → `.loc` 或 `.copy()`

---

## 自测题（附答案）

1. **（选取）** 用员工表 `df`，选出年龄大于 30 的人的 `name` 和 `salary` 两列，写出两种写法（布尔筛选、.loc）。
2. **（loc vs iloc）** `df.loc['e1':'e3']` 和 `df.iloc[0:3]` 各返回几行？如果行标签是默认的 0,1,2,...，`df.loc[0:2]` 和 `df.iloc[0:2]` 行数一样吗？
3. **（按组填缺失，Titanic 真实场景）** `titanic` 表里假设有乘客 `age` 缺失，要求**按性别分组、用各组的中位数**填缺失（男乘客用男性中位数，女乘客用女性中位数）。写出代码。（提示：transform）
4. **（groupby）** 用员工表 `df`，按部门统计：平均工资（列名 avg_salary）、最高年龄（列名 max_age）、人数（列名 n），且部门保留为普通列。
5. **（merge + 算比率）** 用 `users` 和 `orders` 表：以用户表为准拼接（每个用户一行不丢），然后算出**每个用户的订单总金额**（没下过单的用户金额记 0），最后算"有订单用户的平均客单价（总金额/订单数）"。写出核心代码步骤。
6. **（one-hot）** 把 `titanic` 表的 `sex` 和 `pclass` 两列都做 one-hot 编码，要求结果是 0/1 整数列，且删除原始列。写出代码。
7. **（排错）** 同事写了下面代码想给年龄大于 30 的员工工资打 9 折，运行后原表没变还弹了警告。指出问题并改写。

```python
df[df['age'] > 30]['salary'] = df[df['age'] > 30]['salary'] * 0.9
```

8. **（综合：Pandas → NumPy）** 从 `titanic` 表提取 `age`、`fare` 两列作为特征矩阵 X、`survived` 作为标签 y（都是 NumPy 数组），并对 X 做标准化。写出代码。

### 参考答案

**1.**

```python
df[df['age'] > 30][['name', 'salary']]        # 写法一：布尔筛选再取列（只读可以这样）
df.loc[df['age'] > 30, ['name', 'salary']]    # 写法二：.loc 一步完成（推荐，后续要改值时必须用它）
```

**2.** `df.loc['e1':'e3']` 返回 3 行（e1、e2、e3，标签切片**闭区间**）；`df.iloc[0:3]` 返回 3 行（位置 0、1、2，左闭右开）。行标签是 0,1,2,... 时**不一样**：`df.loc[0:2]` 是 3 行（0、1、2），`df.iloc[0:2]` 是 2 行（0、1）。这是 loc/iloc 最隐蔽的区别。

**3.**

```python
titanic['age'] = titanic['age'].fillna(
    titanic.groupby('sex')['age'].transform('median'))
```

`transform('median')` 算出每个乘客所属性别组的中位数并贴回每一行，`fillna` 只填缺失的位置——男乘客缺失处填男性中位数，女乘客填女性中位数。

**4.**

```python
result = df.groupby('dept', as_index=False).agg(
    avg_salary=('salary', 'mean'),
    max_age=('age', 'max'),
    n=('name', 'count'),
)
```

**5.**

```python
m = pd.merge(users, orders, on='user_id', how='left')        # 以用户为准，5 行（用户 1 有 2 单占 2 行）
per_user = m.groupby('user_id').agg(
    total=('amount', 'sum'),        # NaN 在 sum 里被跳过，没订单的用户 total=0
    n_orders=('order_id', 'count'), # count 只数非缺失：没订单的用户为 0
).reset_index()
avg_per_order = per_user.loc[per_user['n_orders'] > 0, 'total'].sum() / \
                per_user.loc[per_user['n_orders'] > 0, 'n_orders'].sum()
print(avg_per_order)    # 99.3333... ← (99+49+150)/(2+1)
```

要点：`sum` 跳过 NaN 自动实现"没订单记 0"；`count` 只数非缺失值，正好数订单数。

**6.**

```python
titanic_enc = pd.get_dummies(titanic, columns=['sex', 'pclass'], dtype=int)
```

`columns=[...]` 指定要编码的列，编码后原始列自动被替换掉；`dtype=int` 让结果是 0/1 而不是 True/False。若担心完全共线性（两个 0/1 列信息冗余），可加 `drop_first=True` 每类少生成一列。

**7.** 这是链式赋值：`df[df['age'] > 30]` 先产生一个临时拷贝，`['salary'] = ...` 改的是拷贝，原表不变，并触发 `SettingWithCopyWarning`。改为 `.loc` 一步定位：

```python
df.loc[df['age'] > 30, 'salary'] = df.loc[df['age'] > 30, 'salary'] * 0.9
```

**8.**

```python
X = titanic[['age', 'fare']].to_numpy()          # (10, 2)
y = titanic['survived'].to_numpy()               # (10,)
X = (X - X.mean(axis=0)) / X.std(axis=0)         # 按列标准化（axis=0 = 每个特征一列）
```

到这一步，数据已经是纯 NumPy，可以直接喂给阶段一的 KMeans 或阶段二的 sklearn 模型。

---

## 与 ML 学习路径的对应关系

| 本文章节 | 学习清单条目 | 阶段二项目里用到哪里 |
|---|---|---|
| §1 心智模型 | DataFrame 创建与索引 | 理解所有表格操作的底层逻辑；index 对齐保你不被 NaN 坑 |
| §2 创建与查看 | DataFrame 创建与索引 | 每个项目开头：`read_csv` 后立刻 `head/info/describe` 做 EDA 体检 |
| §3 选取数据 | DataFrame 创建与索引 | 选特征列、按条件筛样本、`.loc` 修脏数据 |
| §4 数据清洗 | 缺失值处理（dropna/fillna） | Titanic 的 Age/Cabin 缺失处理；房价项目的几十处缺失列 |
| §5 排序与统计 | （支撑技能） | EDA 看分布、`value_counts` 看类别比例 |
| §6 groupby | groupby + agg 分组聚合 | Titanic 按舱位/性别看存活率；按组填缺失（transform）；构造"组内均值"类特征 |
| §7 merge/concat | merge/join 多表操作 | 房价项目合并多个数据源；用户表+行为表拼宽表；训练/测试特征 concat 后统一编码再拆开 |
| §8 读写文件 | 读取 CSV/Excel | 每个项目第一步 `pd.read_csv('train.csv')`；提交结果 `to_csv(index=False)` |
| §9 与 ML 衔接 | （支撑技能） | `to_numpy` 喂模型；`get_dummies` 类别编码；`cut/qcut` 分箱；标准化；房价 SalePrice 的 log1p |
| §10 性能与避坑 | （贯穿始终） | 大数据量特征工程不卡死；避开链式赋值这个新手第一坑 |

**下一步**：本文学完后回到 NumPy 手写 KMeans 完成阶段一里程碑（数据可以用 Pandas 读进来再 `.to_numpy()`）；之后阶段二的房价预测、Titanic 项目会直接套用本文 §4/§6/§7/§9 的特征工程套路。
