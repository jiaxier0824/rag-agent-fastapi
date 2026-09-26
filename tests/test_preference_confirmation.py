"""偏好写入只有收到 MCP 成功回执后才能向用户确认。"""

import json
import unittest

from agent.execution import AgentExecutionContext
from agent.service import AgentService
from agent.tools import build_tools


class FakeMcpClient:
    def __init__(self, ok: bool):
        self.ok = ok

    def call_tool(self, name, arguments):
        assert name == "save_learning_preferences"
        assert arguments["daily_study_hours"] == 2.0
        return {
            "ok": self.ok,
            "code": "OK" if self.ok else "PROFILE_SAVE_FAILED",
            "message": "已保存" if self.ok else "保存失败",
            "data": {"daily_study_hours": 2.0} if self.ok else None,
        }


def fake_service() -> AgentService:
    service = object.__new__(AgentService)
    service.model = "primary-model"
    service.fallback_model = "fallback-model"
    service.model_name = "primary-model"
    service.fallback_model_name = "fallback-model"
    service.trace_logger = None
    return service


class PreferenceConfirmationTest(unittest.TestCase):
    def test_mcp_success_receipt_is_separate_from_tool_attempt(self):
        for ok in (False, True):
            with self.subTest(ok=ok):
                context = AgentExecutionContext()
                save_tool = next(
                    tool for tool in build_tools(
                        session_id="s1", trace_id="t1", execution_context=context,
                        mcp_client=FakeMcpClient(ok),
                    ) if tool.name == "save_learning_preferences"
                )
                payload = json.loads(save_tool.invoke({"daily_study_hours": 2.0}))
                self.assertEqual(payload["ok"], ok)
                self.assertEqual(context.tools_called, ["save_learning_preferences"])
                self.assertEqual(context.profile_save_succeeded, ok)

    def test_sync_cannot_claim_success_without_receipt(self):
        service = fake_service()
        service._invoke_agent = lambda **_kwargs: "我已保存您的长期学习偏好。"

        result = service.execute(
            question="请记住我的长期学习偏好：每天学习 2 小时。",
            session_id="s1", trace_id="t1",
        )

        self.assertEqual(result.answer, "学习偏好尚未保存，请稍后重试。")

    def test_sync_keeps_success_after_receipt(self):
        service = fake_service()

        def invoke(*, execution_context, **_kwargs):
            execution_context.profile_save_succeeded = True
            return "我已保存您的长期学习偏好。"

        service._invoke_agent = invoke
        result = service.execute(
            question="请记住我的长期学习偏好：每天学习 2 小时。",
            session_id="s1", trace_id="t1",
        )
        self.assertEqual(result.answer, "我已保存您的长期学习偏好。")

    def test_sync_cannot_claim_success_when_mcp_rejects_write(self):
        service = fake_service()

        def invoke(*, execution_context, **_kwargs):
            save_tool = next(
                tool for tool in build_tools(
                    session_id="s1", trace_id="t1", execution_context=execution_context,
                    mcp_client=FakeMcpClient(False),
                ) if tool.name == "save_learning_preferences"
            )
            save_tool.invoke({"daily_study_hours": 2.0})
            return "我已保存您的长期学习偏好。"

        service._invoke_agent = invoke
        result = service.execute(
            question="请记住我的长期学习偏好：每天学习 2 小时。",
            session_id="s1", trace_id="t1",
        )
        self.assertEqual(result.tools_called, ["save_learning_preferences"])
        self.assertEqual(result.answer, "学习偏好尚未保存，请稍后重试。")

    def test_stream_never_emits_unconfirmed_claim(self):
        service = fake_service()
        service._stream_agent = lambda **_kwargs: iter(["我已", "保存您的长期学习偏好。"])

        events = list(service.stream_execute(
            question="请记住我的长期学习偏好：每天学习 2 小时。",
            session_id="s1", trace_id="t1",
        ))

        contents = [event["content"] for event in events if event["type"] == "content"]
        self.assertEqual(contents, ["学习偏好尚未保存，请稍后重试。"])

    def test_stream_emits_confirmed_answer_after_tool_success(self):
        service = fake_service()

        def stream(*, execution_context, **_kwargs):
            execution_context.record_tool("save_learning_preferences")
            execution_context.profile_save_succeeded = True
            yield "已"
            yield "保存。"

        service._stream_agent = stream
        events = list(service.stream_execute(
            question="请记住我的长期学习偏好：每天学习 2 小时。",
            session_id="s1", trace_id="t1",
        ))
        contents = [event["content"] for event in events if event["type"] == "content"]
        self.assertEqual(contents, ["已保存。"])
        self.assertEqual(events[-1]["tools_called"], ["save_learning_preferences"])

    def test_negative_save_request_is_not_treated_as_a_write(self):
        service = fake_service()
        service._stream_agent = lambda **_kwargs: iter(["不会保存这次临时安排。"])
        events = list(service.stream_execute(
            question="今天临时学习 2 小时，不要保存为长期偏好。",
            session_id="s1", trace_id="t1",
        ))
        contents = [event["content"] for event in events if event["type"] == "content"]
        self.assertEqual(contents, ["不会保存这次临时安排。"])


if __name__ == "__main__":
    unittest.main()
