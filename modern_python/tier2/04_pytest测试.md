# T2-4 pytest 测试

> 前置：前三个主题的练习代码就是本主题的被测对象。

## 1. pytest vs JUnit 5

| JUnit 5 | pytest | 差异要点 |
|---------|--------|----------|
| `@Test` 注解 | 函数名 `test_` 前缀 | 约定优于配置，没有注解噪音 |
| `@BeforeEach` / `@BeforeAll` | fixture（`scope="function"/"session"`） | fixture 是**按名字注入的依赖**，可组合、可复用、可参数化 |
| `assertEquals(a, b)` | 裸 `assert a == b` | 失败时 pytest 重写 assert，自动展示两边的详细值 |
| `@ParameterizedTest` + `@MethodSource` | `@pytest.mark.parametrize` | 一行搞定，参数直接可读 |
| `assertThrows(...)` | `with pytest.raises(...)` | 上下文管理器写法，还能拿到异常对象继续断言 |
| `@Nested` 类 | 测试类分组（可选） | 大多数项目直接用文件 + 函数命名组织 |
| Maven Surefire | `uv run pytest` | |

最大的观念差异：**JUnit 靠注解和类结构组织测试，pytest 靠"普通函数 + 按名注入的 fixture"**。fixture 机制本质是依赖注入，但没有容器、没有注解。

## 2. 基础三板斧

```python
import pytest

from modern_python.tier2.exercises.t2_2.result import Err, Ok

def test_unwrap_ok():
    assert Ok(3).unwrap() == 3

def test_unwrap_err_raises():
    with pytest.raises(RuntimeError, match="boom"):
        Err("boom").unwrap()

@pytest.mark.parametrize("raw, expected", [
    ("90", 90.0),
    (" 85 ", 85.0),
    ("abc", None),
    ("", None),
    ("150", None),
])
def test_normalize(raw: str, expected: float | None):
    assert normalize_score(raw) is expected  # is None / == 数值 的合写技巧
```

parametrize 的价值：**用例只是数据**。加一个边界用例 = 加一行，而不是复制一个测试函数。

## 3. fixture：按名注入

```python
import pytest

@pytest.fixture
def sample_order() -> Order:                      # 函数名 = 注入用的名字
    return Order(order_id="o-1", items=[OrderItem(sku="a", unit_price=5.0, quantity=2)])

@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:           # 整个会话共享，创建一次
    d = tmp_path_factory.mktemp("data")
    (d / "config.toml").write_text("...", encoding="utf-8")
    return d

def test_total(sample_order: Order):              # 声明参数名，pytest 自动注入
    assert sample_order.total() == 10.0
```

规则：

- fixture 可以依赖其他 fixture（组合）；`yield` 之后写 teardown。
- `scope` 默认 function（每个测试新建）；session/module 级用于创建昂贵资源。**用了 session 级的东西被测试改了状态，是共享 fixture 的经典事故**——默认 function，有理由再放大。
- `tmp_path`（每个测试独立的临时目录）、`monkeypatch`（打补丁/改环境变量，测试后自动还原）、`capsys`（抓 stdout）是内置三宝，先找内置再自造。

## 4. conftest.py：fixture 的共享层

`conftest.py` 放在目录里，其中的 fixture 对**该目录及子目录**所有测试自动可见，不需要 import。层级规则与作用域同构：根 conftest 放通用 fixture，子目录 conftest 放局部 fixture，就近覆盖。

注意：测试文件之间**禁止互相 import**（共享逻辑放 conftest 或普通模块），这是 pytest 测试组织的纪律。

## 5. mock：先 monkeypatch，后 unittest.mock

```python
def test_fetch_handles_http_error(monkeypatch: pytest.MonkeyPatch):
    def fake_get(url: str) -> object:
        raise httpx.ConnectTimeout("offline")
    monkeypatch.setattr(httpx, "get", fake_get)      # 测试结束自动还原
    ...
```

`monkeypatch` 覆盖 90% 场景且无 mock 框架的心智负担。需要断言"调用了几次、传了什么参数"时再上 `unittest.mock.MagicMock`。mock 的铁律与 Java 相同：**mock 边界（网络、文件系统、时钟），不 mock 你正在测的逻辑**。

## 6. 覆盖率与运行

```bash
uv add --dev pytest pytest-cov
uv run pytest modern_python/tier2 --cov=modern_python/tier2 --cov-report=term-missing
```

`term-missing` 直接告诉你哪些行没测到。注意：覆盖率是底线指标不是目标——90% 但全测 getter 和 10% 但全测边界，选后者。

## 7. hypothesis：性质测试（对应 jqwik）

```python
from hypothesis import given, strategies as st

@given(st.integers())
def test_ok_roundtrip(value: int):
    r = Ok(value)
    assert r.map(lambda x: x).unwrap() == value      # 不写具体例子，写"对任意 x 成立"
```

你提供**性质**（不变量、往返、幂等），hypothesis 负责生成大量随机输入（含边界值）去找反例。单元测试证明"我想到的例子是对的"，性质测试攻击"我没想到的例子"。

## 8. 自检清单

- [ ] 能不看资料写 parametrize 和 raises
- [ ] 能解释 fixture 与 @BeforeEach 的本质区别（按名注入 vs 固定挂钩）
- [ ] 知道 session scope 的风险和默认选 function 的原因
- [ ] monkeypatch 的自动还原机制说得出为什么重要
- [ ] 能为一个函数写出至少两条"性质"
