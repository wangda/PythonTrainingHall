"""e05 结构化输出。

任务
    让模型从一段自由文本中抽取订单信息（订单号/金额/状态/日期），
    输出必须能被 Pydantic 模型解析成功。

要求
    1. 方式一：把 JSON Schema 塞进 system prompt（纯 prompt 约束），解析失败要重试。
    2. 方式二：定义一个"伪工具"（如 submit_order_info），靠 tool_choice 强制调用，
       从 arguments 里拿结构化数据。
    3. 两种方式各跑 10 次对比：失败率、代码量、schema 变更时的维护成本。

必须亲手验证的问题
    - 方式一遇到模型输出带 ```json 围栏、前后缀废话怎么办？谁负责清洗？
    - 方式二里"工具"根本不会被执行，它的"执行结果"不存在——这个语义别扭在哪？
    - SDK 里 response_format 参数（json_object / json_schema 模式）与上面两种方式是什么关系？
      （查本地 SDK 源码 openai/types/shared/response_format* 后回答）

验收
    - 两种方式都跑通且可对比；结论写清"什么场景选哪种"。
"""

# TODO: 实现
