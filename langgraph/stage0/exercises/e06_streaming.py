"""e06 流式输出。

任务
    把 e02 的工具调用场景改成 token 级流式输出。

要求
    1. stream=True 收流，逐 chunk 打印 delta.content。
    2. 处理流式场景下的 tool_calls：tool call 的增量也是分片到达的
       （index / id / function.name / function.arguments 分散在多个 chunk），
       需要自己拼装完整 tool_call 再执行。
    3. 打印每次请求的 usage（注意：流式默认可能不返回 usage，查最后一个 chunk 或
       SDK 的 stream_options 参数）。

必须亲手验证的问题
    - 流式下"先看到文本还是先看到 tool_calls 分片"？
    - usage 里 prompt_tokens 大头是什么？（把工具定义加长一倍对比）→ 回填 theory Q5
    - 流式对你 e03 的 while 循环架构提出什么改造要求？（想清楚即可，不用现在实现）

验收
    - 一次"纯文本回答"和一次"工具调用"都在流式下走通。
"""

# TODO: 实现
