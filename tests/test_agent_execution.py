import unittest

from agent.rag_client import RagQueryResult
from agent.service import AgentService


class FakeTraceLogger:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class AgentExecutionTest(unittest.TestCase):
    def test_execute_returns_cached_rag_result_without_invoking_model(self) -> None:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = FakeTraceLogger()

        class CachedRagClient:
            @staticmethod
            def get_cached(*, question, session_id):
                self.assertEqual(question, "课程作业怎么计分？")
                self.assertEqual(session_id, "agent-test")
                return RagQueryResult(
                    answer="缓存中的课程答案",
                    sources=["INFS7410_outline.md"],
                    rag_trace_id="cached-rag-trace",
                    cache_hit=True,
                )

        service.rag_client = CachedRagClient()

        def must_not_invoke(**_kwargs):
            self.fail("缓存命中后不应调用 Agent 模型")

        service._invoke_agent = must_not_invoke

        result = service.execute(
            question="课程作业怎么计分？",
            session_id="agent-test",
            trace_id="agent-trace-cache-hit",
        )

        self.assertEqual(result.answer, "缓存中的课程答案")
        self.assertTrue(result.rag_cache_hit)
        self.assertEqual(result.model_used, "cache")
        self.assertEqual(result.tools_called, ["search_course_knowledge"])
        self.assertEqual(service.trace_logger.events[0]["model_used"], "cache")

    def test_execute_returns_rag_sources_and_writes_trace(self) -> None:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = FakeTraceLogger()

        def fake_invoke(*, model, question, session_id, trace_id, execution_context):
            self.assertEqual(model, "primary-model")
            self.assertEqual(question, "课程作业怎么计分？")
            self.assertEqual(session_id, "agent-test")
            self.assertEqual(trace_id, "agent-trace-001")
            execution_context.record_tool("search_course_knowledge")
            execution_context.record_rag_result(
                sources=["INFS7410_outline.md"],
                rag_trace_id="agent-trace-001",
                cache_hit=False,
            )
            return "课程资料显示，作业占 60%。"

        service._invoke_agent = fake_invoke

        result = service.execute(
            question="课程作业怎么计分？",
            session_id="agent-test",
            trace_id="agent-trace-001",
        )

        self.assertEqual(result.answer, "课程资料显示，作业占 60%。")
        self.assertEqual(result.sources, ["INFS7410_outline.md"])
        self.assertEqual(result.rag_trace_ids, ["agent-trace-001"])
        self.assertEqual(result.tools_called, ["search_course_knowledge"])
        self.assertFalse(result.degraded)
        self.assertEqual(len(service.trace_logger.events), 1)
        self.assertEqual(
            service.trace_logger.events[0]["sources"],
            ["INFS7410_outline.md"],
        )

    def test_execute_falls_back_to_secondary_model_and_records_degradation(self) -> None:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = FakeTraceLogger()

        def fake_invoke(*, model, **_kwargs):
            if model == "primary-model":
                raise RuntimeError("primary unavailable")
            return "备用模型回答"

        service._invoke_agent = fake_invoke

        result = service.execute(
            question="给我一个学习建议",
            session_id="fallback-test",
            trace_id="agent-trace-fallback",
        )

        self.assertEqual(result.answer, "备用模型回答")
        self.assertTrue(result.degraded)
        self.assertEqual(result.model_used, "fallback-model")
        self.assertTrue(service.trace_logger.events[0]["degraded"])


if __name__ == "__main__":
    unittest.main()
