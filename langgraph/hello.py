from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# =======================================================
# 1. 配置 MiniMax 的 OpenAI 兼容接口
# =======================================================
# 请将这里的替换为你自己的真实 API Key
MINIMAX_API_KEY = "sk-cp-vZMCfbKpyms3A3GXqmPfkVaoyobeEAljdon9XXoxEROLe4wbsu2-nDpOLTSga5OhVIA384ggEQsBg5r-pcYM-b2JxfdORvoY0WLgDu5JdQwpLtxQppd4thU"
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1" # 或者是你平时使用的 MiniMax 官方/中转 URL
MODEL_NAME = "MiniMax-M2.7" # 替换为你想使用的 MiniMax 模型名称

llm = ChatOpenAI(
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    model=MODEL_NAME,
    temperature=0
)


# =======================================================
# 2. 定义工具 (Tools)
# =======================================================
@tool
def get_weather(location: str) -> str:
    """获取指定城市的实时天气。"""
    print(f"\n⚙️ [工具执行] 正在查询 {location} 的天气...")
    if "北京" in location:
        return "北京天气晴，25度，微风。"
    elif "上海" in location:
        return "上海大雨，21度，注意防汛。"
    return "阴天，20度。"


tools = [get_weather]
tool_node = ToolNode(tools)

# 关键步骤：把工具“绑定”给大模型，让模型知道它拥有这些超能力
model_with_tools = llm.bind_tools(tools)


# =======================================================
# 3. 定义状态 (State)
# =======================================================
class AgentState(TypedDict):
    # add_messages 让图能够自动追加对话历史，保留上下文
    messages: Annotated[list[BaseMessage], add_messages]


# =======================================================
# 4. 定义节点逻辑 (Nodes)
# =======================================================
def call_model(state: AgentState):
    """大模型节点：负责思考和决策"""
    print("\n🧠 [思考中] 大模型正在审视当前状态并做出决策...")
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# =======================================================
# 5. 定义路由逻辑 (Conditional Edge)
# =======================================================
def should_continue(state: AgentState):
    """条件路由：检查大模型上一步是想‘说话’还是‘调工具’"""
    last_message = state['messages'][-1]

    # 如果模型返回的消息里包含 tool_calls，说明它想借用外部工具
    if last_message.tool_calls:
        print(f"🎯 [决策结果] 大模型决定调用工具: {last_message.tool_calls[0]['name']}")
        return "continue"

    print("🏁 [决策结果] 大模型回答完毕，流程即将结束。")
    return "end"


# =======================================================
# 6. 编排图并编译
# =======================================================
workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 连线
workflow.add_edge(START, "agent")
workflow.add_edge("action", "agent")  # 工具跑完再回炉给模型

# 路由
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

app = workflow.compile()

# =======================================================
# 7. 测试运行
# =======================================================
if __name__ == "__main__":
    print("🚀 启动 LangGraph 极简 Agent...")

    # 初始化输入状态：给大模型提一个它无法独立回答的问题
    initial_input = {"messages": [HumanMessage(content="北京今天天气怎么样？")]}

    # 运行图并打印每一步的状态变化
    # app.stream 会把图在流转过程中，每个节点执行完后的 State 增量吐出来
    for chunk in app.stream(initial_input):
        print("\n--- 执行完了一个Node: {}", chunk)
        for node_name, state_update in chunk.items():
            print(f"\n--- 节点 '{node_name}' 执行完毕 ---")
            if 'messages' in state_update:
                last_msg = state_update['messages'][-1]
                # 区分打印：是大模型吐出的消息，还是工具吐出的消息
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"节点输出 -> 意图调用工具: {last_msg.tool_calls}")
                else:
                    print(f"节点输出 -> {last_msg.content}")