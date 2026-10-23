# 阶段 0：前置地基

> 目标：不借助任何 agent 框架，手写出一个能调工具、能多轮的命令行 agent。
> 对应总计划见 [../README.md](../README.md) 阶段 0。理论文档在 `theory/`，练习模块在 `exercises/`，**每个小主题一个模块**。

## 学习流程（每个模块都走这四步）

```
1. 读理论   → 对应 theory/ 文档（不带代码，只带心智模型和开放问题）
2. 手写     → 关掉一切参考，凭理解写 exercises/ 里的 TODO
3. 破坏     → 故意改坏：漏回填、传错 id、喂非法 JSON……观察报错
4. 回填     → 回 theory/ 文档的开放问题清单，⏳ 改 ✅ + 结论
```

## 运行配置

练习统一走 OpenAI 兼容协议（`openai` SDK 已安装），通过环境变量指定供应商，不把供应商写死在代码里：

```
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.deepseek.com/v1"   # 或其他 OpenAI 兼容端点
export LLM_MODEL="deepseek-chat"
```

国内可选端点：DeepSeek / Qwen（DashScope 兼容模式）/ Kimi（Moonshot）。三者的 key 当前都不在本机环境中，选定后补配。

## 模块清单

| 模块 | 主题 | 任务一句话 | 验收 |
|---|---|---|---|
| e01_chat_basics.py | 消息模型与多轮对话 | 用裸 SDK 维护一个多轮对话 | 模型能记住 5 轮前的信息；能解释"为什么它能记住" |
| e02_single_tool.py | 单工具两阶段协议 | 定义一个 get_weather 工具，手写完整两阶段往返 | 一次工具调用全流程走通；tool_choice 三种模式行为可演示 |
| e03_tool_loop.py | 多工具 agent 循环 | 多个工具 + while 循环 + 错误回填重试 | **本阶段 capstone**：模型自动决定调哪个、调几次，直到给出最终回答 |
| e04_parallel_calls.py | 并行工具调用 | 一轮多个 tool_calls 的执行与回填 | 能处理一轮多调用；验证 id 配对错误时的真实报错（Q1/Q4） |
| e05_structured_output.py | 结构化输出 | 用 JSON Schema / Pydantic 约束输出 | 模型输出可被 Pydantic 解析；对比 tool-call 式与 JSON mode 两种做法 |
| e06_streaming.py | 流式输出 | token 级流式打印 + usage 统计 | 流式与工具调用并存时输出顺序可解释（Q5） |
| e07_async_agent.py | 异步版最终 agent | e03 的 asyncio 改造 + 真并行工具执行 | 多个无依赖工具真实并发执行；100 行以内 |

## 阶段验收标准（硬标准）

1. e07 通过：一个 async 命令行 agent，多工具、真并行、错误自愈、流式输出，≤100 行有效代码。
2. theory/t01_tool_calling.md 的 6 个开放问题全部回填 ✅，每个有复现方式。
3. 能不看资料画出"两阶段工具调用协议"的完整时序图（手画，画在 notes/ 里）。
4. 能口头回答：如果模型一轮调了 3 个工具，你的消息列表下一轮必须长什么样？
