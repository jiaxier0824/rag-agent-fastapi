import unittest

from agent.service import AgentService


class FakeTraceLogger:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class AgentExecutionTest(unittest.TestCase):
    def test_cached_course_answer_does_not_skip_plan_orchestration(self) -> None:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = FakeTraceLogger()

        class CachedRagClient:
            @staticmethod
            def get_cached(*, question, session_id):
                self.fail("Agent 入口不应读取课程缓存")

        service.rag_client = CachedRagClient()

        def invoke_plan(*, execution_context, **_kwargs):
            execution_context.record_tool("search_course_knowledge")
            execution_context.record_rag_result(
                sources=["INFS7410_outline.md"],
                rag_trace_id="cached-rag-trace",
                cache_hit=True,
            )
            execution_context.record_tool("create_or_update_study_plan")
            return "学习计划已保存。"

        service._invoke_agent = invoke_plan

        result = service.execute(
            question="请根据课程资料创建学习计划",
            session_id="agent-test",
            trace_id="agent-trace-cache-hit",
        )

        self.assertEqual(result.answer, "学习计划已保存。")
        self.assertTrue(result.rag_cache_hit)
        self.assertEqual(result.model_used, "primary-model")
        self.assertEqual(result.tools_called, ["search_course_knowledge", "create_or_update_study_plan"])
        self.assertEqual(service.trace_logger.events[0]["model_used"], "primary-model")

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

    def test_execute_keeps_course_question_in_agent_tool_loop(self) -> None:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = FakeTraceLogger()

        class EmptyCacheClient:
            @staticmethod
            def get_cached(*, question, session_id):
                return None

        service.rag_client = EmptyCacheClient()

        def fake_invoke(*, model, question, **_kwargs):
            self.assertEqual(model, "primary-model")
            self.assertEqual(question, "INFS7410 的课程资料怎么说？")
            return "Agent 通过 search_course_knowledge 得到课程资料答案。"

        service._invoke_agent = fake_invoke

        result = service.execute(
            question="INFS7410 的课程资料怎么说？",
            session_id="course-guard-test",
            trace_id="course-guard-trace",
        )

        self.assertEqual(result.answer, "Agent 通过 search_course_knowledge 得到课程资料答案。")
        self.assertEqual(result.model_used, "primary-model")

    def test_course_plan_request_keeps_agent_tool_orchestration(self) -> None:
        service = object.__new__(AgentService)
        service.model = "primary-model"
        service.fallback_model = "fallback-model"
        service.model_name = "primary-model"
        service.fallback_model_name = "fallback-model"
        service.trace_logger = FakeTraceLogger()

        class NoDirectRagClient:
            @staticmethod
            def get_cached(**_kwargs):
                return None

            @staticmethod
            def ask(**_kwargs):
                self.fail("创建计划不能绕过 Agent，直接调用 RAG")

        service.rag_client = NoDirectRagClient()

        def fake_invoke(*, model, question, **_kwargs):
            self.assertEqual(model, "primary-model")
            self.assertIn("创建学习计划", question)
            return "Agent 已编排课程检索和计划工具。"

        service._invoke_agent = fake_invoke

        result = service.execute(
            question="请为 INFS7410 创建学习计划",
            session_id="plan-task-test",
            trace_id="plan-task-trace",
        )

        self.assertEqual(result.model_used, "primary-model")
        self.assertEqual(result.answer, "Agent 已编排课程检索和计划工具。")

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
