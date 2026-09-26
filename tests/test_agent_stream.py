import unittest

from agent.service import AgentService


class AgentStreamTest(unittest.TestCase):
    def test_stream_does_not_bypass_agent_when_course_answer_is_cached(self) -> None:
        service = self._build_service()

        class CachedRagClient:
            @staticmethod
            def get_cached(**_kwargs):
                self.fail("Agent 入口不应读取课程缓存")

        service.rag_client = CachedRagClient()

        def stream_plan(*, execution_context, on_status=None, **_kwargs):
            execution_context.record_tool("search_course_knowledge")
            execution_context.record_rag_result(
                sources=["INFS7410_outline.md"],
                rag_trace_id="cached-stream-trace",
                cache_hit=True,
            )
            execution_context.record_tool("create_or_update_study_plan")
            yield "学习计划已保存。"

        service._stream_agent = stream_plan

        events = list(
            service.stream_execute(
                question="请根据课程资料创建学习计划",
                session_id="stream-test",
                trace_id="trace-stream-cache-hit",
            )
        )

        self.assertEqual(events[0], {"type": "status", "content": "正在分析你的问题"})
        self.assertIn({"type": "content", "content": "学习计划已保存。"}, events)
        self.assertEqual(events[-1]["tools_called"], ["search_course_knowledge", "create_or_update_study_plan"])

    def _build_service(self) -> AgentService:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = None
        return service

    def test_streams_tool_status_before_content(self) -> None:
        service = self._build_service()

        def fake_stream(
            *,
            model,
            question,
            session_id,
            trace_id,
            execution_context,
            on_status=None,
        ):
            self.assertEqual(model, "primary-model")
            self.assertIsNotNone(on_status)
            execution_context.record_tool("search_course_knowledge")
            execution_context.record_rag_result(
                sources=["INFS7410_outline.md"],
                rag_trace_id="rag-trace-stream",
                cache_hit=False,
            )
            on_status("正在检索课程资料")
            yield "课程资料显示，"
            yield "作业截止日期是 9 月 4 日。"

        service._stream_agent = fake_stream

        events = list(
            service.stream_execute(
                question="作业什么时候截止？",
                session_id="stream-test",
                trace_id="trace-stream-test",
            )
        )

        self.assertEqual(
            events,
            [
                {"type": "status", "content": "正在分析你的问题"},
                {"type": "status", "content": "正在检索课程资料"},
                {"type": "status", "content": "正在生成回答"},
                {"type": "content", "content": "课程资料显示，"},
                {"type": "content", "content": "作业截止日期是 9 月 4 日。"},
                {
                    "type": "metadata",
                    "content": "",
                    "sources": ["INFS7410_outline.md"],
                    "rag_trace_ids": ["rag-trace-stream"],
                    "tools_called": ["search_course_knowledge"],
                    "blocked_tool_calls": [],
                    "short_memory_turns_loaded": 0,
                    "profile_memory_loaded": False,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
