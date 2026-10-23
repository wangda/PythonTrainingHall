# T1-3 Pythonic 惯用法

> 前置：T1-2 的 ruff 已配置好。本主题没有新知识点，全是"换脑子"——把 Java 的肌肉记忆翻译成 Python 的。

## 1. comprehension：循环的最惯用形态

```python
# list
doubled = [x * 2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

# dict
name_to_len = {name: len(name) for name in names}

# set
unique_tags = {tag.lower() for tag in tags}

# generator：惰性，处理大数据不占内存
total = sum(x * x for x in range(10_000_000))
```

判断标准：**一行能说清"从哪来、怎么变、留哪些"就用 comprehension；需要多层嵌套或中途 break 就老实用 for**。超过两层 `[...]` 套 `[...]` 的，宁可拆成 for——可读性优先，comprehension 不是炫技。

Java 对照：`stream().map().filter().toList()`。区别是 comprehension 是语言级语法，没有流的开销；而"惰性管线"在 Python 里对应 generator 表达式（T3-1 深入）。

## 2. EAFP vs LBYL

Java 是 LBYL 世界（Look Before You Leap）：先判空、先 containsKey、先 exists。Python 的文化是 EAFP（Easier to Ask Forgiveness than Permission）：

```python
# LBYL（Java 味）
if key in config:
    value = config[key]
else:
    value = DEFAULT

# EAFP（Pythonic）
try:
    value = config[key]
except KeyError:
    value = DEFAULT

# 但 dict 场景最惯用的其实是 .get() / .setdefault()——先想想标准库有没有直接给答案
value = config.get(key, DEFAULT)
```

适用判断：

- **EAFP**：操作大概率成功、竞争条件下检查与使用之间有时间窗（TOCTOU）、异常路径不频繁。
- **LBYL**：失败是常态、异常开销显著、except 无法精确圈定错误来源。
- 不确定时，先查标准库：`dict.get`、`dict.setdefault`、`next(iter, default)` 这类"带默认值的 API"往往直接消灭这个问题。

## 3. 鸭子类型：能力判断，不是血统判断

```python
# Java 味：先问血统
if isinstance(saver, FileSaver):
    saver.save(data)

# Pythonic：直接用能力（配合 T1-1 的 Protocol 做静态约束）
saver.save(data)
```

静态层面用 Protocol 约束，运行层面信任鸭子类型。isinstance 的合法场景：真正需要分支处理不同类别时（如 `isinstance(x, (int, float))` 做数值归一）。

## 4. 解包与并发迭代

```python
# 交换
a, b = b, a

# 星号收集
head, *rest = [1, 2, 3, 4]           # head=1, rest=[2, 3, 4]
first, *mid, last = "python"

# 合并序列 / 字典
merged = [*xs, *ys]
combined = {**defaults, **overrides}  # 或 3.9+ 的 defaults | overrides

# enumerate：需要下标时的唯一正解
for i, item in enumerate(items, start=1):
    ...

# zip：并行遍历两个序列
for name, score in zip(names, scores):
    ...

# 函数签名中的 *args / **kwargs
def log(level, *values, timestamp=None): ...
```

Java 对照：`for (int i = 0; i < list.size(); i++)` 在 Python 里几乎总是错误的起点。

## 5. 函数是一等公民

```python
# 高阶函数：把行为当参数传（对应 Java 的函数式接口/方法引用）
prices.sort(key=lambda item: item.price)
max(orders, key=lambda o: o.total())
handlers = {"json": parse_json, "csv": parse_csv}

# 克制原则：lambda 只用在 key= 这类"一行表达式"的位置。
# 超过一行、需要分支、需要复用 → 定义命名函数，哪怕只有三行。
```

## 6. pathlib：路径不是字符串

```python
from pathlib import Path

data_dir = Path("modern_python") / "tier1" / "exercises"   # / 运算符拼路径
config_file = data_dir / "config.toml"

text = config_file.read_text(encoding="utf-8")             # 直接读写
data_dir.mkdir(parents=True, exist_ok=True)
for py_file in data_dir.glob("**/*.py"):
    print(py_file.stem, py_file.suffix)
```

Java 对照：`Path.of("a").resolve("b")`，但 Python 的 Path 顺带包办了读写、遍历、glob——大多数场景不再需要开流。`os.path` 拼接写法全部过时。

## 7. 真值判断与 None 处理

```python
if not items:            # 空容器/None/0 都是假——判断"没有数据"
if item is None:         # 判断"恰好是 None"，必须用 is
if item is not None:     # 不要 if item == None，更不要 if not item（0 会被误伤）
```

规则一句话：**判 None 用 `is`，判"空"用真值，两者混用是 bug 来源**（`quantity=0` 会被 `if not quantity` 吞掉）。

## 8. Java 味反模式对照表

| Java 味写法 | Pythonic 写法 |
|-------------|---------------|
| `for i in range(len(xs))` + `xs[i]` | `for x in xs`；要下标就 `enumerate(xs)` |
| `if len(xs) == 0` | `if not xs` |
| `if s == "a" \|\| s == "b"`（Python 里 `s == "a" or s == "b"`） | `if s in ("a", "b")` |
| 手写字符串拼接循环 | `"\n".join(parts)` |
| 全套 getter/setter | 直接属性；需要逻辑时再上 `@property` |
| `new Point()` 后逐个 set | 构造参数 / dataclass 一次成型 |
| camelCase | snake_case（类名仍 PascalCase，常量 UPPER_CASE） |
| `System.out.println` 调试 | logging / breakpoint()（T2、T3 展开） |
| 接口 + 工厂 + 实现 | 协议 + 直接构造，有真实需求再抽象 |
| `Optional.ofNullable(x).orElse(default)` | `x if x is not None else default` 或 `x or default`（注意 0 的陷阱） |

## 9. 自检清单

- [ ] 四种 comprehension 各写出一个不查资料的例子
- [ ] 能说出一个 EAFP 合适、LBYL 合适的场景各一个
- [ ] `range(len(...))` 出现在自己代码里时会有警觉
- [ ] 知道 `x or default` 在 `x=0` 时的坑
- [ ] 路径操作第一反应是 pathlib
