import unittest

from fastapi.testclient import TestClient

from agent.execution import AgentExecutionResult
from api import app
from dependencies import get_agent_service


class FakeAgentService:
    def execute(self, *, question: str, session_id: str, trace_id: str):
        assert question == "课程作业怎么计分？"
        assert session_id == "api-test"
        assert trace_id
        return AgentExecutionResult(
            answer="课程资料显示，作业占 60%。",
            sources=["INFS7410_outline.md"],
            rag_trace_ids=[trace_id],
            tools_called=["search_course_knowledge"],
            rag_cache_hit=False,
            model_used="primary-model",
            degraded=False,
        )


def build_client() -> TestClient:
    app.dependency_overrides[get_agent_service] = FakeAgentService
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


class AgentApiTest(unittest.TestCase):
    def tearDown(self) -> None:
        teardown_function()

    def test_chat_returns_answer_and_rag_sources(self) -> None:
        response = build_client().post(
            "/api/agent/chat",
            json={
                "question": "课程作业怎么计分？",
                "session_id": "api-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "课程资料显示，作业占 60%。")
        self.assertEqual(
            response.json()["sources"],
            [{"filename": "INFS7410_outline.md"}],
        )
        self.assertTrue(response.json()["trace_id"])

    def test_allows_browser_requests_from_rag_frontend(self) -> None:
        response = build_client().options(
            "/api/agent/chat",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://127.0.0.1:8000",
        )
