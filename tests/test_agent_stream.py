import unittest

from agent.service import AgentService


class AgentStreamTest(unittest.TestCase):
    def _build_service(self) -> AgentService:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        return service

    def test_streams_tool_status_before_content(self) -> None:
        service = self._build_service()

        def fake_stream(
            *,
            model,
            question,
            session_id,
            trace_id,
            on_status=None,
        ):
            self.assertEqual(model, "primary-model")
            self.assertIsNotNone(on_status)
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
            ],
        )


if __name__ == "__main__":
    unittest.main()
