# T2-1 Pydantic：标注终于"算数"了

> 前置：T1-1 的类型标注。Pydantic 是"运行时消费标注"框架的代表——这是它和 dataclass 的本质分野。

## 1. 定位：它对应 Java 的什么

Pydantic v2 ≈ **Bean Validation + Jackson + @ConfigurationProperties** 三合一：

| Java 世界 | Pydantic |
|-----------|----------|
| Bean Validation 注解（`@NotNull` `@Min` `@Pattern`） | `Field()` 约束 + 校验器 |
| Jackson `ObjectMapper` 反序列化 | `model_validate_json()`，直接从标注推导 schema |
| `@ConfigurationProperties` | `pydantic-settings` 的 `BaseSettings` |
| OpenAPI 手写/生成 | FastAPI 自动从模型生成（T3 之后的事） |

核心差异：Java 的校验靠注解处理器，Pydantic 靠**读你的类型标注本身**——标注写对，校验和序列化就免费获得。

## 2. BaseModel 基础

```python
from pydantic import BaseModel, Field

class OptimizerConfig(BaseModel):
    name: Literal["adamw", "sgd"]
    learning_rate: float = Field(gt=0, le=1)      # 约束直接挂在 Field 上
    weight_decay: float = Field(ge=0, default=0.01)

class TrainingConfig(BaseModel):
    run_name: str = Field(min_length=3, max_length=32)
    epochs: int = Field(ge=1, le=1000)
    device: Literal["cpu", "cuda", "mps"]
    optimizer: OptimizerConfig                    # 嵌套模型，自动递归校验
    tags: list[str] = Field(default_factory=list)
```

要点：

- 字段无默认值 = 必填；缺了直接 `ValidationError`。
- 默认是**宽松模式**：`"32"` 会被转成 `32`，这是"边界数据清洗"的设计取向。要严格就 `model_config = ConfigDict(strict=True)`。
- 违规时抛 `ValidationError`，`e.errors()` 给出每个字段的结构化错误列表——对应 Java 里 BindingResult 的逐字段错误。

## 3. 自定义校验器

```python
from pydantic import field_validator, model_validator

class TrainingConfig(BaseModel):
    ...
    @field_validator("run_name")
    @classmethod
    def no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("run_name 不能含空格")
        return v

    @model_validator(mode="after")                # 跨字段校验，拿整个模型
    def check_device(self) -> "TrainingConfig":
        if self.device == "cuda" and self.optimizer.name == "sgd":
            raise ValueError("sgd + cuda 组合当前不支持")
        return self
```

规则：能用 `Field` 约束表达的不写校验器；`field_validator` 管单字段；`model_validator(mode="after")` 管跨字段。

## 4. 序列化与 JSON 往返

```python
cfg = TrainingConfig.model_validate_json(raw_json)   # JSON 字符串 -> 模型
d = cfg.model_dump()                                  # -> dict（mode="python"）
j = cfg.model_dump_json()                             # -> JSON 字符串
same = TrainingConfig.model_validate_json(j) == cfg   # 往返一致性
```

对照 Java：`model_validate` ≈ Jackson readValue，`model_dump` ≈ writeValueAsString 的对象版。Pydantic 不需要 `@JsonProperty` 这类注解——字段名即键名，别名用 `Field(alias=...)`。

## 5. 模型配置

```python
from pydantic import ConfigDict

class TrainingConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,        # 不可变（对应 Java record 的取向）
        extra="forbid",     # 多余的键直接报错——防手滑配置项拼错被静默忽略
        strict=True,        # 不做隐式类型转换
    )
```

`extra="forbid"` 强烈推荐用在配置模型上：`"epoch"` 拼错成 `"epoche"` 时，默认行为是静默丢弃，这是配置系统的经典事故源。

## 6. pydantic-settings：环境变量配置

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_", env_file=".env")

    learning_rate: float = 0.001
    batch_size: int = 32
    device: Literal["cpu", "cuda"] = "cpu"
```

环境变量 `TRAIN_LEARNING_RATE=0.01` 自动映射到 `learning_rate`，**带完整类型转换和校验**。对照 Java 的 `@ConfigurationProperties(prefix = "train")`，少了 YAML、少了 setter，类型安全还更强（坏值启动即报错而不是运行中 NPE）。

查找优先级：初始化参数 > 环境变量 > `.env` 文件 > 默认值。

## 7. dataclass vs Pydantic 分工

| | dataclass | Pydantic BaseModel |
|---|-----------|--------------------|
| 校验 | 无 | 完整 |
| 序列化 | 需手写或 `asdict` | 内置 JSON 往返 |
| 运行时开销 | 极低 | 实例化有校验成本 |
| 不可变 | `frozen=True` | `frozen=True` |
| 适用 | 纯内存结构、内部数据传递、高频创建 | 一切跨边界的数据：配置、API 输入输出、文件/网络反序列化 |

一句话：**进出系统边界的数据用 Pydantic，系统内部流转的结构用 dataclass**。你的 order_domain（T1-1）是内部模型，dataclass 合适；训练配置从环境变量/YAML 来，Pydantic 合适。

## 8. 生态位置

- **FastAPI**：请求/响应体就是 Pydantic 模型，校验 + OpenAPI 文档全自动。
- **LangChain / LangGraph**（你项目已依赖）：消息、工具参数、图状态大量使用 Pydantic 模型——会写 Pydantic 是读懂这两个库源码的前置条件。

## 9. 自检清单

- [ ] 能不查资料写出一个含嵌套 + 三种 Field 约束的模型
- [ ] 能说清宽松模式和 strict 模式的取舍
- [ ] 知道 `extra="forbid"` 防的是什么事故
- [ ] 能背出"边界用 Pydantic、内部用 dataclass"的判断依据
