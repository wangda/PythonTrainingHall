# modern_python 学习计划

面向 Java 转 Python 开发者的现代 Python 学习路线。基于本项目环境：Python 3.12 + uv + pyproject.toml。

## 总体目标

1. 写出"Pythonic"而非"Java 味"的 Python 代码
2. 掌握现代 Python 工具链（类型检查、ruff、pytest）并应用到日常开发
3. 熟悉事实标准库（Pydantic、httpx、Polars 等），能读懂现代开源项目的惯用写法

## 环境准备（开始前一次性完成）

- [ ] 安装 ruff、pyright：`uv add --dev ruff pyright`（或 `uvx` 直接运行）
- [ ] VS Code 装 Pylance，检查模式调到 `basic`（第 3 周后升 `strict`）
- [ ] `pyproject.toml` 中加入 `[tool.ruff]` 基础配置（line-length=100, target-version="py312"）
- [ ] 确认 `uv run pytest` 可执行（pytest 加入 dev 依赖）

## 目录规划

```
modern_python/
├── README.md                  # 本计划
├── tier1/                     # 第一梯队：立即改变写码方式
│   ├── 01_类型标注.md
│   ├── 01_练习.md
│   ├── 02_工具链_ruff与pyright.md
│   ├── 02_练习.md
│   ├── 03_Pythonic惯用法.md
│   ├── 03_练习.md
│   └── exercises/             # 练习代码（t1_1/、t1_2/、t1_3/ 子目录）
├── tier2/                     # 第二梯队：日常开发核心
│   ├── 0x_*.md
│   └── exercises/
└── tier3/                     # 第三梯队：进阶机制
    ├── 0x_*.md
    └── exercises/
```

约定：每个主题一份学习文档（含 Java 对比表），一份练习文档（题目 + 验收标准），代码放 `exercises/`。

---

## 第一梯队：立即改变写码方式（约 3 周）

### T1-1 类型标注系统

**学习内容**
- 函数/变量标注基础：`def f(x: int) -> str`
- 内置泛型：`list[int]`、`dict[str, int]`、`tuple[int, ...]`
- 联合类型 `int | None`（3.10+ 新写法，弃用 `Optional`/`List`）
- `Protocol`：结构子类型 vs Java 名义子类型的思维差异（本梯队最重要概念）
- `dataclass`、`Enum`、`TypedDict`、`Literal`

**练习题目**
1. 给一段无标注的"Java 味"代码补全类型标注，直到 pyright strict 通过
2. 用 `Protocol` 重构一个依赖具体类的模块（对比 Java 的 interface + implements）
3. 用 `dataclass` + `Enum` 建模一个订单/配置领域对象（对照 Java record）

**验收标准**：练习 1 的代码 `pyright` strict 模式零报错

### T1-2 现代工具链

**学习内容**
- uv 全套：`uv add / remove / sync / run / lock`，锁文件的作用
- ruff：lint 规则分级、`--fix` 自动修复、格式化（替代 black + isort）
- pyright / mypy 的定位差异
- pre-commit 钩子
- pipx：装全局 CLI 工具

**练习题目**
1. 对本项目现有代码（如 `ml/` 下任一模块）跑 ruff，逐条理解并修复告警
2. 写一个 ruff 配置，把 `E501`（行宽）改成 100 并验证生效
3. 给练习 1 的修复过程配 pre-commit，验证"提交被拦下"的效果

**验收标准**：`ruff check .` 零告警；能说清 ruff 一条规则为什么有价值

### T1-3 Pythonic 惯用法

**学习内容**
- comprehension（list/dict/set/generator 四种）
- EAFP vs LBYL（`try/except KeyError` vs 先判存在）
- 鸭子类型：不问"是什么类型"，问"有没有这能力"
- 解包：`*args` / `**kwargs`、星号解包、enumerate、zip
- 函数是一等公民、高阶函数、lambda 的克制使用
- `pathlib.Path` 替代字符串路径

**练习题目**
1. "翻译题"：给 10 段 Java 风格循环代码，改写成 comprehension / zip / enumerate 版本
2. 把一段 `if file.exists(): ...` 改成 EAFP 风格，并说明两种写法各自的适用场景
3. 用 pathlib 重写一段 `os.path` 拼接代码

**验收标准**：改写后的代码过 ruff 的 pyupgrade 规则；能口头解释每处改动为什么更 Pythonic

---

## 第二梯队：日常开发核心（约 4 周）

### T2-1 Pydantic（事实标准）

**学习内容**
- `BaseModel`：字段定义、校验器、默认值
- 序列化：`model_dump` / `model_validate` / JSON 往返
- 与 `dataclass` 的分工：需要校验/JSON 用 Pydantic，纯内存结构用 dataclass
- `pydantic-settings`：环境变量配置管理（对应 Java @ConfigurationProperties）
- 与 LangChain/LangGraph 生态的关系（贴合本项目依赖）

**练习题目**
1. 定义一个带嵌套模型 + 校验规则（min/max/pattern）的配置模型，构造非法数据验证报错信息
2. 用 `pydantic-settings` 替换一段硬编码配置
3. Pydantic vs dataclass 对比实验：同一领域对象两种写法，总结选择标准

**验收标准**：非法输入触发 `ValidationError` 且信息可读；JSON 往返无精度/类型丢失

### T2-2 新语法（3.10–3.13）

**学习内容**
- `match/case`：解构嵌套结构、守卫条件（对比 Java 21 switch 模式匹配）
- 海象运算符 `:=`
- `zoneinfo` 替代 pytz
- 异常组 `ExceptionGroup` / `except*`（为 T2-3 异步铺垫）
- 3.12 类型参数语法 `def f[T](x: T) -> T`

**练习题目**
1. 用 `match/case` 实现一个 AST/命令解析器（如计算器表达式解构）
2. 用海象运算符改写一段"边读边判"的循环
3. 用 3.12 泛型语法写一个 `Result[T, E]` 风格容器

**验收标准**：每题至少一处"旧写法明显更啰嗦"的对比注释

### T2-3 异步编程

**学习内容**
- 协程心智模型：显式 async/await（对比 Java 虚拟线程的透明并发）
- 事件循环：`asyncio.run`、Task、gather
- `asyncio.TaskGroup`（3.11+ 结构化并发）
- `httpx`：同步/异步统一 API
- GIL 与适用边界：IO 密集用 asyncio，CPU 密集用多进程/NumPy 向量化

**练习题目**
1. 用 `httpx` + `asyncio.gather` 并发抓取 10 个 URL，对比串行耗时
2. 用 `TaskGroup` 重写练习 1，故意让一个任务抛异常，观察取消行为
3. CPU 密集实验：asyncio 跑纯计算 vs 多进程 vs NumPy 向量化，记录耗时并解释

**验收标准**：能说清"什么任务不该用 asyncio"并给出实验数据

### T2-4 pytest 测试

**学习内容**
- 基础：assert、`@pytest.mark.parametrize`、异常测试
- fixture：作用域、依赖注入式组织（对比 JUnit @BeforeAll 的区别）
- conftest.py 共享机制
- mock：`unittest.mock` / pytest-mock
- 覆盖率 pytest-cov；属性测试 hypothesis（对应 jqwik）

**练习题目**
1. 给 tier1 练习代码补测试，覆盖率到 90%
2. parametrize 重构 5 个仅参数不同的测试函数
3. 用 fixture 组织"临时文件 + 构造数据"的测试上下文
4. hypothesis 挑战：对自己的 `Result[T, E]` 容器验证恒等性质

**验收标准**：`uv run pytest --cov` 达标；测试无重复样板

---

## 第三梯队：进阶机制（约 3 周，按需深入）

### T3-1 语言底层机制

**学习内容**
- 装饰器：语法糖本质、带参装饰器、functools.wraps（对照 Java 注解 + AOP 代理）
- 生成器 / 迭代器协议：`yield`、惰性求值、数据管线（ML 场景高频）
- 上下文管理器：`with`、`contextlib.contextmanager`、`ExitStack`
- 描述符协议：理解 `@property`、方法绑定、`__getattr__` vs `__getattribute__`
- import 系统与模块组织：`_` 前缀约定、`__init__.py` 的 re-export 惯例

**练习题目**
1. 手写一个计时/重试装饰器（不查资料，凭理解）
2. 生成器管线：读大文件 → 过滤 → 聚合，全程惰性，内存峰值对比一次性加载版本
3. 自定义上下文管理器实现"计时日志"，分别用类和 `@contextmanager` 两种写法

**验收标准**：能解释装饰器执行顺序、生成器 `send/close` 的行为

### T3-2 数据/AI 方向现代库（贴合本项目）

**学习内容**
- Polars：lazy frame、表达式 API，与 pandas 的取舍
- NumPy 向量化思维（告别 Python for 循环）
- structlog 结构化日志；Rich 终端输出（训练日志美化）

**练习题目**
1. 同一份 CSV：pandas 与 Polars 各写一遍清洗+聚合，对比 API 风格与耗时
2. 把一个 for 循环数值计算改成 NumPy 向量化，记录加速比
3. Rich 重做训练循环的进度输出

**验收标准**：Polars 题能说清 lazy 执行在什么场景有收益

### T3-3 项目结构与打包

**学习内容**
- `src/` 布局、可安装包、`pyproject.toml` 打包配置
- 入口脚本、`if __name__ == "__main__":`
- `uv build` / `uv publish`

**练习题目**
1. 把本目录改造成 `src/` 布局的可安装包，`uv run` 里以包名 import
2. 定义一个 CLI 入口（argparse 或 Typer），`uv build` 产出 wheel 并安装验证

**验收标准**：wheel 安装到临时 venv 后命令可用

---

## 建议路线与节奏

| 周次 | 内容 | 里程碑 |
|------|------|--------|
| 1–2 | T1-1 + T1-2 | pyright strict 通过 + ruff 零告警 |
| 3 | T1-3 | 惯用法翻译题全部完成 |
| 4 | T2-1 | Pydantic 配置模型落地 |
| 5 | T2-2 | match/case 解析器完成 |
| 6 | T2-3 | 并发抓取实验数据出来 |
| 7 | T2-4 | 覆盖率 90% 里程碑 |
| 8+ | T3 按需 | 挑 1–2 个主题深入 |

**推进方式**：每完成一个主题，在本 README 对应小节打勾，然后继续下一个。练习文档先自己写，卡住再看学习文档对照。

## 不碰清单（当前阶段）

- Django（重、偏老式全家桶）
- 多继承 MRO 细节、元类（除非具体项目需要）
- Web 框架（若后续要做模型服务化，届时直接学 FastAPI + Pydantic 组合）
