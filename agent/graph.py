"""Agent 的显式 LangGraph 工作流。"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode


class AgentGraphState(TypedDict):
    """一次 Agent 请求在图节点间流转的状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    trace_id: str


def build_agent_graph(*, model, tools, required_tool_name: str | None = None):
    """构建“模型判断 -> 工具执行 -> 再判断”的最小 Agent 图。"""
    model_with_tools = model.bind_tools(tools)
    required_tool = next(
        (tool for tool in tools if tool.name == required_tool_name), None
    )
    model_with_required_tool = (
        model.bind_tools([required_tool], tool_choice="required")
        if required_tool is not None else None
    )

    def agent_reason(state: AgentGraphState) -> dict:
        """让模型基于现有消息决定调用工具或给出最终回答。"""
        first_required_call = (
            model_with_required_tool is not None
            and not any(isinstance(message, ToolMessage) for message in state["messages"])
        )
        active_model = model_with_required_tool if first_required_call else model_with_tools
        response = active_model.invoke(state["messages"])
        return {"messages": [response]}

    def route_after_reason(state: AgentGraphState) -> str:
        """模型产生 tool_calls 时进入工具节点；否则结束本轮图执行。"""
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    graph = StateGraph(AgentGraphState)
    graph.add_node("agent_reason", agent_reason)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent_reason")
    graph.add_conditional_edges(
        "agent_reason",
        route_after_reason,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent_reason")
    return graph.compile()
