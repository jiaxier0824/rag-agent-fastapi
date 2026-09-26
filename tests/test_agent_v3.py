import json
import unittest

from agent.execution import AgentExecutionContext
from agent.memory import AgentMemoryStore
from agent.rag_client import RagQueryResult
from agent.study_plan_store import StudyPlanStore
from agent.tools import build_tools


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def lpush(self, key, *values):
        self.lists.setdefault(key, [])
        for value in values:
            self.lists[key].insert(0, value)

    def ltrim(self, key, start, end):
        self.lists[key] = self.lists.get(key, [])[start : end + 1]

    def lrange(self, key, start, end):
        return self.lists.get(key, [])[start : end + 1]

    def expire(self, *_args):
        return True

    def scan_iter(self, match):
        prefix = match[:-1]
        return [key for key in self.values if key.startswith(prefix)]


class FakeRagClient:
    def ask(self, **_kwargs):
        return RagQueryResult(
            answer="课程资料答案",
            sources=["outline.md"],
            rag_trace_id="rag-1",
            cache_hit=False,
        )


class FakeMcpClient:
    def call_tool(self, name, arguments):
        if name == "get_study_plan":
            return {
                "ok": True,
                "code": "OK",
                "message": "学习计划读取完成。",
                "data": {"course_id": arguments["course_id"], "plan": "IR 计划"},
            }
        raise AssertionError(f"unexpected MCP tool: {name}")


class AgentV3Test(unittest.TestCase):
    def test_context_blocks_duplicate_tool_call(self):
        context = AgentExecutionContext(max_tool_calls=2)
        self.assertIsNone(context.begin_tool_call("search", {"question": "A"}))
        self.assertEqual(
            context.begin_tool_call("search", {"question": "A"}),
            "DUPLICATE_TOOL_CALL",
        )
        self.assertIsNone(context.begin_tool_call("get_plan", {"course_id": "INFS7410"}))
        self.assertEqual(
            context.begin_tool_call("list_plans", {}),
            "TOOL_CALL_LIMIT_EXCEEDED",
        )
        self.assertEqual(context.tools_called, ["search", "get_plan"])

    def test_memory_keeps_recent_turns_and_explicit_profile(self):
        store = AgentMemoryStore(short_memory_max_turns=1)
        store.redis_client = FakeRedis()
        store.append_turn("s1", "第一个问题", "第一个回答")
        store.append_turn("s1", "第二个问题", "第二个回答")

        self.assertEqual(
            store.get_recent_messages("s1"),
            [
                {"role": "user", "content": "第二个问题"},
                {"role": "assistant", "content": "第二个回答"},
            ],
        )
        profile = store.update_profile("s1", daily_study_hours=2.5)
        self.assertEqual(profile, {"daily_study_hours": 2.5})

    def test_multi_course_plan_isolation_and_tool_response(self):
        redis_client = FakeRedis()
        plans = StudyPlanStore()
        plans.redis_client = redis_client
        plans.save("s1", "INFS7410", "IR 计划")
        plans.save("s1", "COMP9001", "编程计划")
        self.assertEqual(plans.get("s1", "INFS7410"), "IR 计划")
        self.assertEqual(plans.list_plans("s1"), ["COMP9001", "INFS7410"])

        memory = AgentMemoryStore()
        memory.redis_client = redis_client
        context = AgentExecutionContext(max_tool_calls=2)
        tools = build_tools(
            session_id="s1",
            trace_id="t1",
            execution_context=context,
            mcp_client=FakeMcpClient(),
        )
        get_plan = next(item for item in tools if item.name == "get_study_plan")
        payload = json.loads(get_plan.invoke({"course_id": "INFS7410"}))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["plan"], "IR 计划")


if __name__ == "__main__":
    unittest.main()
