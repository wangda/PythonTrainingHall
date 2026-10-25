# T1-2 现代工具链：uv / ruff / pyright / pre-commit

> 前置：T1-1 已开始使用 pyright。本主题把工具链补完整，之后所有练习都跑在这些工具之上。

## 1. uv 全套（不只是 pip 替代品）

你项目已经在用 uv，这里把常用面补齐：

| 命令 | 作用 | Java 类比 |
|------|------|-----------|
| `uv add requests` | 加依赖并写进 pyproject + lock | Maven 加 dependency |
| `uv add --dev pytest` | 加开发期依赖 | scope=test |
| `uv remove requests` | 移除依赖 | |
| `uv sync` | 按 lock 文件精确还原环境 | `mvn install` 的确定性版 |
| `uv run pytest` | 在项目 venv 里执行命令 | |
| `uv lock --upgrade` | 升级锁定版本 | versions 插件 |
| `uv python pin 3.12` | 固定解释器版本（写 `.python-version`） | toolchains 插件 |
| `uv tool install ruff` | 全局安装 CLI 工具（隔离环境） | `npm install -g` / sdkman |

关键观念：**`uv.lock` 必须提交**。它是可复现构建的来源，对应 Java 世界 lockfile（Gradle verification metadata / Maven enforcer 的意义）。

## 2. pyproject.toml：唯一配置入口

现代 Python 项目**没有 setup.py**。一个文件管所有：

```toml
[project]              # PEP 621 标准元数据（你项目已有）
name = "pythontraininghall"
requires-python = ">=3.12"
dependencies = [...]

[dependency-groups]    # 开发依赖（uv 的分组，旧式写法是 optional-dependencies）
dev = ["ruff>=0.8", "pyright>=1.1", "pytest>=8"]

[tool.ruff]            # 每个工具的配置都挂在 [tool.<name>] 下
line-length = 100

[tool.pyright]
typeCheckingMode = "strict"

[tool.pytest.ini_options]
testpaths = ["modern_python"]
```

Java 类比：`[project]` ≈ pom 的坐标与依赖，`[tool.*]` ≈ 各 maven 插件的 `<configuration>`。区别是它同时还是构建后端配置——一份文件，没有 XML。

## 3. ruff：lint + format 二合一

一句话定位：**管"写得像不像地道 Python"**。用 Rust 实现，一个二进制吃掉一整排老工具：flake8（lint）+ isort（import 排序）+ pyupgrade（旧语法升级）+ black（格式化）。

```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint + 自动修复
uv run ruff format .         # 格式化（替代 black）
```

"现代 Python 转换"就是 `UP` 规则组 + `--fix` 的组合：自动把 `typing.List[int]` 改成 `list[int]`、`Optional[X]` 改成 `X | None`、`%` 格式化改成 f-string。可以把它理解为"检查风格时顺手把旧语法翻译成现代语法"。

规则组按前缀选择，常用的：

| 前缀 | 来源 | 抓什么 | 对 Java 人的价值 |
|------|------|--------|------------------|
| `E` `F` | pycodestyle/pyflakes | 基础错误：未用变量、未定义名、f-string 缺占位符 | 相当于编译警告 |
| `I` | isort | import 排序分组 | 自动化，无脑开 |
| `UP` | pyupgrade | 旧式写法 → 现代写法（`%` → f-string、`List` → `list`） | 学新语法的免费老师 |
| `B` | bugbear | 常见 bug 模式（可变默认参数、`==` 比较 None） | 类似 SpotBugs |
| `SIM` | simplify | 可简化的冗余代码 | 消灭 Java 味 |

推荐起步配置（写进根 `pyproject.toml`）：

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.ruff.lint.isort]
known-first-party = ["d2l_v2", "ml", "modern_python"]
```

规则详情 `uv run ruff rule B006` 或查 https://docs.astral.sh/ruff/rules/ 。**每条报错都查一下规则说明再修**，这是 T1 阶段学惯用法的主要途径之一。

## 4. pyright：静态类型检查器

一句话定位：**管"类型说得通说不通"**——检查对象是你的类型标注：标注一致不一致、`None` 有没有处理、泛型推导对不对。

它和 Java 编译器的关键差别：**pyright 报错不会阻止程序运行**。`def f(x: int) -> str` 传了 list，pyright 会标红，但 Python 解释器照跑不误。类型错误从"编译失败"降级为"检查器警告"，纪律得靠工作流撑（见第 6 节）。反过来注意：dataclass、Pydantic 这类框架会**在运行时读你的标注**做校验和序列化——那是框架行为，不是语言行为（T1-1 第 1 节的"两个关键认知"）。

### pyright vs mypy

| | pyright | mypy |
|---|---------|------|
| 实现 | TypeScript，Pylance 的内核 | Python |
| 速度 | 快 | 慢一些 |
| IDE 集成 | VS Code 原生（Pylance） | 需要扩展 |
| 配置方式 | `pyrightconfig.json` 或 `[tool.pyright]` | `[tool.mypy]` |
| 生态 | VS Code 用户主流 | 库存根类型存根（stubs）更全 |

本项目统一用 pyright，CLI + Pylance 同引擎，所见即所得。

## 5. pre-commit：把检查钉在提交上

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
uv tool install pre-commit   # 或 uv add --dev pre-commit
pre-commit install           # 挂到 git hook
pre-commit run --all-files   # 全量跑一次验证
```

Java 类比：git hook 版的 Maven enforcer + CI 门禁前置。被它拦下的提交说明检查器在起作用——这是 T1-2 练习 3 要亲眼验证的事。

## 6. 推荐工作流

```bash
# 日常写码循环
uv run ruff check --fix . && uv run ruff format . && uv run pyright && uv run pytest
```

可以把这条链写进 `justfile` / `Makefile`，或直接靠 pre-commit 兜底。核心原则：**检查器不通过就不算写完**，和 Java 的"编译不过不算写完"同一纪律，只是工具代替了编译器——pyright 的类型错误不像 Java 编译错误那样天然挡住运行，这条纪律就是那道挡板。

两个工具分工互补、互不覆盖，**必须都开**：

- ruff 管"写得地道"：风格、惯用法、旧语法、常见 bug 模式。
- pyright 管"类型说得通"：标注一致性、None 处理、泛型推导。
- 交叉盲区：ruff 不做类型检查——它的 `ANN` 规则组只能查"函数有没有**写**标注"，查不了标注**对不对**；pyright 则完全不管格式和风格。所以少跑任何一个，都会漏掉另一个独管的那半边。

## 7. 自检清单

- [ ] 能说清 uv.lock 为什么必须提交
- [ ] 能不看文档写出 `[tool.ruff]` 起步配置
- [ ] ruff 六个规则组各说出一条规则名和它抓的问题
- [ ] 能说清 ruff 和 pyright 各管什么、对方管不到的盲区在哪
- [ ] pre-commit 曾真的拦下过一次提交
