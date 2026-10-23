# Tool Calling（工具调用）

> 阶段 0 理论文档 01。本文档不含代码——只建立心智模型，代码在 exercises/ 里手写。
> 证据等级标注：【核实】= 从本地 openai SDK 2.38.0 源码读出；【通识】= 业界公认、多来源一致；【待核】= 依据训练知识，未联网核实，需练习或查官方文档确认。

## 1. 它解决什么问题

LLM API 本质是"文本进、文本出"的函数。它没有计算器、查不了数据库、动不了你的文件系统。

Tool Calling 解决的问题是：**让模型能"表达调用意图"，而执行权始终在你手里**。流程是模型输出一段结构化的"我想调 foo(参数为 X)"，你的代码看到后决定是否真的调用 foo。

这个分工是整个 agent 安全模型的地基：**模型永远不执行任何东西，它只产生意图；执行权、权限、兜底全部属于调用方代码**。后面所有框架（LangChain/LangGraph）不管包装成什么样子，都没有改变这个分工。

## 2. 底层机制：API 层面到底发生了什么【核实】

一次完整的工具调用是**两阶段往返协议**，不是一次请求：

```
阶段一：请求（你 → API）
  messages: 对话历史
  tools:    工具定义列表（name + description + parameters 的 JSON Schema）

阶段一：响应（API → 你）
  模型可能返回 tool_calls（而不是普通 content）：
    每个 tool_call = { id, type: "function", function: { name, arguments } }

阶段二：你执行工具，把结果回填，再发一次请求
  新增一条消息：role="tool", tool_call_id=<对应id>, content=<执行结果字符串>

阶段二：响应（API → 你）
  模型基于工具结果继续生成，可能返回 content、也可能再次 tool_calls（循环）
```

从本地安装的 openai SDK 源码（`openai/types/chat/`）核实到的关键细节：

- **`arguments` 是 JSON 字符串，不是对象**。SDK 官方注释原文：模型并不总能生成合法 JSON，且可能幻觉出 schema 里不存在的参数——**校验责任在调用方**。这是协议设计的一部分，不是实现缺陷。
- **回填靠 `tool_call_id` 配对**。工具结果消息的 `tool_call_id` 是必填字段。assistant 消息带了几个 tool_calls，下一轮请求就必须有几条 role="tool" 消息与之一一配对。
- **`tool_choice` 控制模型的调用意愿**：`"none"`（禁止）/ `"auto"`（模型自己决定，默认）/ `"required"`（强制至少调一个）/ 指定具体工具（强制调某个）。
- **`parallel_tool_calls` 是显式开关**：允许模型在一轮响应里同时返回多个 tool_calls。
- **消息列表无状态**：每一轮请求都要把完整对话历史（包括工具定义、工具结果）全量重发。服务端不记得上一轮——"上下文管理"之所以是 agent 工程的核心难题，根源就在这里。

【通识】还有一个容易忽略的细节：消息角色集合是 `system`（新版本叫 `developer`）、`user`、`assistant`、`tool` 四种。工具结果不是普通文本，是独立角色。

## 3. 各家差异【待核，练习时顺带验证】

协议本质同构：**意图结构化 + id 配对回填**。差异在容器格式和消息角色设计：

| 维度 | OpenAI 系 | Anthropic 系 | Gemini 系 |
|---|---|---|---|
| 调用意图 | assistant 消息的 `tool_calls` 数组 | content 里的 `tool_use` block（可与文本 block 混排） | 候选内容里的 `functionCall` part |
| 结果回填 | 独立角色 `role="tool"` + `tool_call_id` | **复用 `role="user"`**，content 里放 `tool_result` block | 无独立工具角色，用 `functionResponse` part |
| 触发信号 | finish_reason = `tool_calls` | stop_reason = `tool_use` | 无统一字段，看 parts 内容 |
| 多工具/轮 | 一轮多个 tool_calls | 一轮多个 tool_use blocks | 一轮多个 functionCall parts |

实操层面的差异：国产及开源端点（DeepSeek、Qwen、Kimi 等）绝大多数**兼容 OpenAI 协议**，但 `tool_choice="required"`、并行调用、严格 schema 模式的支持程度参差不齐——这正是开放问题清单里要动手验证的。

## 4. 生产中的真实问题【通识】

按踩坑频率排序：

1. **参数不可信**。幻觉参数、非法 JSON 是常态而非异常。生产代码的标准做法：JSON 解析失败或校验失败时，把错误信息作为工具结果回填给模型让它重试——而不是抛异常让整个流程崩掉。
2. **回填完整性**。一轮有 N 个 tool_calls 就必须回填 N 条结果，漏一条下一轮请求直接 400。
3. **并行调用的执行策略**。多个 tool_calls 之间无依赖的可以真并行（asyncio.gather），有依赖的必须串行——模型并不知道你的工具哪些能并行，这个判断在你这边。
4. **工具执行的超时与重试**。工具可能调外部 API，会挂。重试的前提是工具幂等；不幂等的工具失败后怎么向模型交代（如实说失败原因 vs 换个说法），影响模型下一步决策质量。
5. **工具即权限边界**。给模型的工具集合 = 模型的能力边界。危险操作（删数据、发邮件、花权限）应该有人工确认层——这是阶段 2 `interrupt()` 要解决的问题的理论原型。
6. **成本**。tools 定义和全部历史每轮重发，工具越多、对话越长，每次请求越贵。这是"上下文裁剪"需求的第一来源。
7. **该调不调、不该调乱调**。靠 tool_description 质量 + system prompt 约束 + tool_choice 硬控制三层手段。

## 5. 业界成熟方案怎么处理【通识】

分层看，框架做的事 = 你手写循环做的事 + 工程化加固：

- **裸 SDK**（本阶段练习目标）：自己写两阶段循环、自己配对回填、自己兜底。
- **LangChain**：`@tool` 装饰器从函数签名和 docstring 自动生成 JSON Schema；统一消息模型（ToolMessage）抹平各家差异。
- **LangGraph**：把"模型节点 → 条件边 → 工具节点 → 回到模型节点"这个循环固化为图结构（ToolNode + 条件边），内置了"工具首次执行失败则把错误回填给模型重试"的策略。
- **更上层（create_agent / Deep Agents）**：整张图也不让你写了，只留 prompt 和工具集两个口子。

学完阶段 0 你会发现：**框架没有任何魔法，它只是把你手写的 while 循环拆成了可编排、可持久化、可观测的节点**。

## 6. 开放问题清单（练习后回填）

> 规则：练习时逐条验证，回填格式 `✅ Q1：结论 + 复现方式`。

- ⏳ **Q1** 一轮返回多个 tool_calls 时，`tool_call_id` 配对错乱 / 漏回填一条，API 具体报什么错？（→ e03/e04 破坏实验）
- ⏳ **Q2** `arguments` 为非法 JSON 时，模型端能否通过"回填错误信息"自我修复？成功率如何？（→ e03）
- ⏳ **Q3** `tool_choice="required"` 在 OpenAI 官方端点与国产兼容端点（DeepSeek/Qwen/Kimi 任选）行为是否一致？（→ e02 双端点对比）
- ⏳ **Q4** 并行 tool_calls 在兼容端点上默认开启还是关闭？`parallel_tool_calls=False` 是否被尊重？（→ e04）
- ⏳ **Q5** 工具定义（尤其长 description）在一轮请求里的 token 占比实测多少？（→ e06 顺带观察 usage 字段）
- ⏳ **Q6** 模型在"工具结果与它预期矛盾"时会坚持还是改口？这决定工具结果该怎么写措辞。（→ e03 选一个工具故意返回矛盾结果）
