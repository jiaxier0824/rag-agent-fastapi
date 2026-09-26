"""LangGraph 主循环测试：模型调工具后会回到模型节点，再结束。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.graph import build_agent_graph


@tool
def echo_course_name(course_id: str) -> str:
    """返回课程代码。"""
    return f"course={course_id}"


@tool
def save_learning_preferences(daily_study_hours: float) -> str:
    """保存明确给出的每日学习时长。"""
    return f"saved={daily_study_hours}"


class FakeToolCallingModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="已根据工具结果完成回答。")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "echo_course_name",
                    "args": {"course_id": "INFS7410"},
                    "id": "tool-call-1",
                }
            ],
        )


class FakeRequiredToolModel:
    def __init__(self):
        self.bindings = []

    def bind_tools(self, tools, **kwargs):
        self.bindings.append(([tool.name for tool in tools], kwargs))
        return BoundRequiredToolModel(kwargs)


class BoundRequiredToolModel:
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        if self.kwargs.get("tool_choice") == "required":
            return AIMessage(content="", tool_calls=[{
                "name": "save_learning_preferences",
                "args": {"daily_study_hours": 2.0},
                "id": "save-call-1",
            }])
        assert any(isinstance(message, ToolMessage) for message in messages)
        return AIMessage(content="保存工具已返回。")


class LangGraphFlowTest(unittest.TestCase):
    def test_graph_runs_model_tool_model_loop(self):
        graph = build_agent_graph(
            model=FakeToolCallingModel(),
            tools=[echo_course_name],
        )

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="查询 INFS7410")],
                "session_id": "test-session",
                "trace_id": "test-trace",
            }
        )

        self.assertIsInstance(result["messages"][-2], ToolMessage)
        self.assertEqual(result["messages"][-2].content, "course=INFS7410")
        self.assertEqual(result["messages"][-1].content, "已根据工具结果完成回答。")

    def test_explicit_preference_forces_save_tool_only_on_first_reason_step(self):
        model = FakeRequiredToolModel()
        graph = build_agent_graph(
            model=model,
            tools=[save_learning_preferences, echo_course_name],
            required_tool_name="save_learning_preferences",
        )

        result = graph.invoke({
            "messages": [HumanMessage(content="请记住每天学习 2 小时")],
            "session_id": "test-session",
            "trace_id": "test-trace",
        })

        self.assertEqual(result["messages"][-2].content, "saved=2.0")
        self.assertEqual(result["messages"][-1].content, "保存工具已返回。")
        self.assertEqual(model.bindings[1], (["save_learning_preferences"], {"tool_choice": "required"}))


if __name__ == "__main__":
    unittest.main()
