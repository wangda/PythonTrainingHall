"""e02 单工具的两阶段协议。

任务
    定义一个 get_weather(city: str) 工具（本地写死的假数据即可），
    手写完整的"两阶段往返"，让模型回答"北京今天适合穿什么"。

要求
    1. 用 JSON Schema 描述工具（name / description / parameters），写进请求的 tools 参数。
    2. 收到响应后判断：是 content 还是 tool_calls？
    3. 是 tool_calls 则：本地执行 → 构造 role="tool" 消息回填 → 再次请求 → 拿到最终回答。
    4. arguments 是 JSON 字符串，记得 json.loads + 校验（理论文档 §2：幻觉参数是常态）。
    5. 分别用 tool_choice="auto" / "required" / "none" 各跑一次，观察差异。

必须亲手验证的问题
    - 不写 tools 参数 vs tools=None，请求有区别吗？
    - tool_choice="none" 时模型面对"查天气"类问题会怎么回答？
    - 故意把回填消息的 tool_call_id 写错，报错原文是什么？（记入 notes/，这是 Q1 的一半）

验收
    - 三种 tool_choice 模式行为可复现，结论回填 theory 文档 Q3。

提示
    - 温馨提示：get_weather 的 description 写得好不好，直接影响模型调不调它——试两个版本对比。
"""

# TODO: 实现
