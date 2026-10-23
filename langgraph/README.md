# LangGraph + Agent 开发系统学习计划

> 制定日期：2026-10-23
> 依据：官方文档已迁移至 https://docs.langchain.com/oss/python/langgraph/overview ，当前为 LangGraph 1.x，提供 Graph API（节点/边/状态图）与 Functional API（`@entrypoint`/`@task`）两套写法，预置高层封装为 `create_agent`。
> 总周期约 3~4 个月（按每天 1~2 小时 + 周末估）。

## 核心原则

1. **每个概念都手写验证过才算学会。**
2. 全程尽量不让 AI 代写；遇到不确定先查官方文档、再读源码（LangGraph 源码量不大，`StateGraph` 主路径几千行，可通读）。
3. 每个案例留三样东西：**图结构图（mermaid）、踩坑记录、评测集**——找工作时这三样比代码本身更有说服力。

---

## 阶段 0：前置地基（约 1 周）

不学这个，后面全是黑盒。

- **Function Calling / Tool Calling 原理**：不经过任何框架，用裸 SDK 手写一个工具调用循环（`while` 循环 + `tool_calls` 解析 + 结果回填 message）。这是所有 agent 框架的底层，LangGraph 只是把这件事工程化。
- **Structured Output**：Pydantic 模型约束 LLM 输出，理解 JSON mode 与 tool-call 式结构化输出的区别。
- **Streaming 与 Async**：异步调用、token 级流式输出。LangGraph 全栈是 async-first 的。
- **理论主线**：什么是 Agent？ReAct 论文的思想（Reason + Act 循环）、Agent = LLM + 工具 + 循环 + 状态。

**验收**：不用任何框架，100 行以内手写一个能调工具、能多轮的命令行 agent。

## 阶段 1：LangGraph 核心机制（约 2 周）

目标：把图的三大件（State / Node / Edge）和它们的类、API 摸透。

- 核心类与关系：`StateGraph` → `add_node` / `add_edge` / `add_conditional_edges` → `compile()` → `CompiledGraph.invoke/stream`
- **State 与 Reducer**（最重要的概念，没有之一）：`Annotated[list, add_messages]` 这类 reducer 如何决定状态合并方式。坑点高发区：节点返回值是**全量覆盖还是按 reducer 合并**，90% 的初学者 bug 出在这。
- 条件边 vs `Command`（节点内直接 `goto`）两种路由写法的取舍
- `START`/`END`、消息模型、节点返回 `dict` 的语义
- 预置件 `create_agent` 用来对照：它内部就是图 API 的封装，读它的源码验证你手写的东西
- 理论主线：工作流（workflow）与智能体（agent）的边界——控制流是人写的还是 LLM 决定的

**验收**：不看教程，纯手写一个带工具循环的 ReAct agent（自己定义 State、model 节点、tool 节点、条件边），行为与 `create_agent` 一致。能画出自己这张图的 mermaid 结构。

## 阶段 2：持久化、可控性、记忆（约 1.5 周）

这是 LangGraph 区别于"裸写 while 循环"的核心价值，也是面试重点。

- **Checkpointer**：`MemorySaver` → `SqliteSaver`/`PostgresSaver`，理解 thread_id、checkpoint 的写入时机、状态快照结构
- **Human-in-the-loop**：`interrupt()` 暂停 → 拿到人工输入 → `Command(resume=...)` 恢复。坑点：resume 后是**从暂停节点重新执行**，节点要设计成幂等/可重入
- **Time travel**：从历史 checkpoint 回溯重放（调试神器）
- **记忆**：短期记忆（thread 内消息）vs 长期记忆（cross-thread store），`InMemoryStore` 的用法
- **递归限制**：`recursion_limit` 为什么存在、怎么报错（典型的坑）
- 理论主线：durable execution（持久化执行）——为什么长任务 agent 必须有 checkpoint

**验收**：给阶段 1 的 agent 加上 SqliteSaver 和一个人工确认工具调用的环节；杀掉进程重启后能从断点续跑。

## 阶段 3：实战一 —— 智能问数 NL2SQL（约 2 周）

业界最主流的落地场景之一，流程天然带循环，最适合第一个大案例。

- 图结构：`问题改写 → 生成SQL → 执行 → 出错则带报错回炉重造（循环）→ 结果解读`
- 技能点：
  - 把 schema/DDL 注入 prompt 的上下文工程
  - 工具封装（只读连接、行数限制、超时）
  - SQL 校验节点（确定性代码节点和 LLM 节点混编——LangGraph 官方强调的 "mix deterministic and agentic steps"）
- 坑点：SQL 注入与只读防护、错误循环兜不住时必须有人工逃生口、schema 太大撑爆上下文（引入局部 schema 检索）
- 理论主线：agent 的可靠性工程——确定性护栏怎么包住不确定性

**验收**：针对一个真实 SQLite/Postgres 库（建几张有关联的表、造脏数据），问数准确率可复现，错误 SQL 能自动修复，死循环有人工兜底。可流式展示中间步骤。

## 阶段 4：实战二 —— Agentic RAG 智能问答（约 2 周）

- 图结构：`路由（要不要检索/查哪个库）→ 检索 → 检索质量自评 → 不够则改写查询重查（循环）→ 生成 → 幻觉自检`
- 技能点：
  - embedding + 向量库基础、hybrid search
  - 引用溯源
  - self-RAG / corrective-RAG 两个业界主流模式
- 坑点：chunk 策略对效果的影响远大于 agent 结构、评测怎么做（不用 eval 只靠肉眼 = 调不明白）
- 理论主线：agentic RAG 与朴素 RAG 的分界；RAG 评测基本法（retrieval hit rate + answer faithfulness）
- 配套：接入 **LangSmith**，学会看 trace 定位问题——这是后面所有案例的调试基建

**验收**：对自己灌入的一批真实文档（比如某产品手册/法规），实现"答得出、引得出、答不了会明说"，并用一个 20 题的评测集给出量化指标。

## 阶段 5：实战三 —— 经营分析报告生成（约 2 周）

考编排能力的案例：多分支、并行、子图、结构化产出。

- 图结构：`需求解析 → 并行取数（多个子图/分支各查一个业务主题）→ 汇总分析 → 生成图表 → 撰写章节 → 审查节点打回重写（循环）→ 结构化输出（markdown/邮件/PDF）`
- 技能点：
  - **`Send` API 实现动态 map-reduce**（章节数不固定，按解析结果动态分发）
  - **subgraph** 组织子任务
  - 并行分支的状态合并冲突（没有 reducer 的 key 在并行下会报错——经典坑）
- 坑点：并行写状态冲突、长报告的上下文预算管理、中间产物走 checkpoint 而不是全塞 message
- 理论主线：multi-agent 的两种主流拓扑——supervisor（主管分派）与 swarm（平权交接），本案例用 supervisor

**验收**：输入一句"给我看上月销售情况分析"，产出一份带数据、带图表、结构完整的多章节报告；中途 kill 进程能续跑。

## 阶段 6：实战四 —— 编码 Agent（约 2~3 周）

难度最高，也是面试最能打的案例。

- 图结构：`理解需求 → 探索代码库 → 制定计划 → 人工审批计划 → 逐步写代码 → 跑测试 → 失败带报错回炉（循环）→ 汇报`
- 技能点：
  - plan-and-execute 模式
  - 文件读写工具与工作区隔离（子进程沙箱执行测试）
  - **上下文管理**（长会话消息裁剪/压缩——coding agent 的生死线）
  - 多子代理（探索 agent 与编码 agent 分离）
- 坑点：上下文爆炸、测试输出太长怎么截断喂回模型、危险命令的权限门（human-in-the-loop 的真实用武之地）
- 参考：读 LangChain 官方 **deep-agents** 项目源码（planning、子代理、文件系统、上下文管理就是在 LangGraph 上这么搭的），对照自己写的找差距

**验收**：给它一个真实小仓库的 bug 或小需求，能自主完成"探索→计划→改码→测试通过"，全过程有 checkpoint、计划变更有人工确认。

## 阶段 7：生产化收尾（约 2 周）

- **部署**：`langgraph dev` / LangGraph Server 的 API 化、流式推给前端（SSE）、独立部署方案
- **评测与可观测**：LangSmith eval / 自建 eval 集、trace 分析、A/B prompt
- **生产常见问题清单**（整理成文档）：幂等与重试、LLM 超时与降级、成本控制、状态膨胀、并发 thread 的 checkpoint 隔离
- 把四个案例里沉淀的通用模式（工具封装、护栏、评测集）整理成自己的脚手架

**验收**：任选一个案例部署成带流式 UI 的服务，写一份生产问题清单文档。这套东西就是简历上的项目。

---

## 贯穿始终的四条纪律

1. **古法编程**：每阶段先手写，再看官方示例找差距；`create_agent`、prebuilt 组件当"参考答案"读，不当 starting point 用。
2. **读源码**：LangGraph 核心源码不大，`StateGraph.compile`、`add_messages`、`interrupt` 的实现都值得通读，读完框架就没有黑盒了。
3. **案例渐进**：每个新案例必须用上前面积累的机制（checkpoint、interrupt、Send……），最后四个案例合起来覆盖 LangGraph 全部核心 API。
4. **每个案例留三样东西**：图结构图（mermaid）、踩坑记录、评测集。

## 目录约定

一个阶段一个目录：`stage0/` ~ `stage7/`。每个阶段内部统一为：

```
stageN/
├── README.md      # 本阶段目标、模块清单、验收标准
├── theory/        # 练前理论文档（不含代码，含开放问题清单，练后回填）
├── exercises/     # 练习模块，每个小主题一个 .py（骨架含任务书，实现自己手写）
└── notes/         # 踩坑记录（短平快，随手记）
```

## 进度追踪

| 阶段 | 内容 | 预估 | 状态 |
|---|---|---|---|
| 0 | 前置地基（裸 SDK 工具循环 / 结构化输出 / 流式与异步） | 1 周 | 进行中（stage0/ 已就绪） |
| 1 | LangGraph 核心机制（State/Node/Edge/Reducer） | 2 周 | 未开始 |
| 2 | 持久化、可控性、记忆（Checkpointer / interrupt / 记忆） | 1.5 周 | 未开始 |
| 3 | 实战一：智能问数 NL2SQL | 2 周 | 未开始 |
| 4 | 实战二：Agentic RAG 智能问答 | 2 周 | 未开始 |
| 5 | 实战三：经营分析报告生成 | 2 周 | 未开始 |
| 6 | 实战四：编码 Agent | 2~3 周 | 未开始 |
| 7 | 生产化收尾（部署 / 评测 / 生产问题清单） | 2 周 | 未开始 |
