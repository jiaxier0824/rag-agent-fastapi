import unittest
from unittest.mock import patch

import httpx

from agent.cache import CachedRagResponse
from agent.rag_client import RagApiClient


class FakeCache:
    def __init__(self, cached_response: CachedRagResponse | None = None):
        self.cached_response = cached_response
        self.saved_answer: dict | None = None
        self.requested_question: str | None = None

    def get(self, question: str, session_id: str) -> CachedRagResponse | None:
        self.requested_question = question
        return self.cached_response

    def set(self, **kwargs) -> None:
        self.saved_answer = kwargs


class SuccessResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "answer": "课程资料答案",
            "trace_id": "rag-trace-001",
            "sources": [{"filename": "INFS7410_outline.md"}],
        }


class RagApiClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = "COMP9001 的作业要求是什么？"
        self.session_id = "rag-client-test"

    def test_returns_cached_answer_without_http_request(self) -> None:
        cache = FakeCache(
            cached_response=CachedRagResponse(
                answer="缓存中的课程答案",
                sources=["cached_source.md"],
                rag_trace_id="rag-trace-cached",
            )
        )
        client = RagApiClient(cache=cache, base_url="http://test")

        with patch("agent.rag_client.httpx.post") as mock_post:
            result = client.ask(self.question, self.session_id, "agent-trace-001")

        self.assertEqual(result.answer, "缓存中的课程答案")
        self.assertEqual(result.sources, ["cached_source.md"])
        self.assertEqual(result.rag_trace_id, "rag-trace-cached")
        self.assertTrue(result.cache_hit)
        mock_post.assert_not_called()

    def test_retries_after_network_error_then_caches_answer(self) -> None:
        cache = FakeCache()
        client = RagApiClient(
            cache=cache,
            base_url="http://test",
            max_retries=1,
            retry_interval_seconds=0,
        )
        request = httpx.Request("POST", "http://test/api/rag/chat")

        with patch(
            "agent.rag_client.httpx.post",
            side_effect=[
                httpx.ConnectError("temporary network error", request=request),
                SuccessResponse(),
            ],
        ) as mock_post:
            result = client.ask(self.question, self.session_id, "agent-trace-002")

        self.assertEqual(result.answer, "课程资料答案")
        self.assertEqual(result.sources, ["INFS7410_outline.md"])
        self.assertEqual(result.rag_trace_id, "rag-trace-001")
        self.assertFalse(result.cache_hit)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(
            cache.saved_answer,
            {
                "question": self.question,
                "session_id": self.session_id,
                "answer": "课程资料答案",
                "sources": ["INFS7410_outline.md"],
                "rag_trace_id": "rag-trace-001",
            },
        )
        self.assertEqual(
            mock_post.call_args.kwargs["headers"],
            {"X-Trace-ID": "agent-trace-002"},
        )

    def test_uses_original_user_question_as_cache_key(self) -> None:
        """模型改写工具查询时，缓存仍应绑定 Router 收到的原问题。"""
        cache = FakeCache()
        client = RagApiClient(
            cache=cache,
            base_url="http://test",
            max_retries=0,
        )

        with patch(
            "agent.rag_client.httpx.post",
            return_value=SuccessResponse(),
        ):
            client.ask(
                question="请解释 Precision、Recall 和 MRR",
                cache_question=self.question,
                session_id=self.session_id,
                trace_id="agent-trace-original-question",
            )

        self.assertEqual(cache.requested_question, self.question)
        self.assertEqual(cache.saved_answer["question"], self.question)

    def test_tool_reads_cache_before_http_call(self) -> None:
        cache = FakeCache()
        client = RagApiClient(cache=cache, base_url="http://test", max_retries=0)

        with patch("agent.rag_client.httpx.post", return_value=SuccessResponse()) as mock_post:
            client.ask(
                self.question,
                self.session_id,
                "agent-trace-cache-prechecked",
            )

        self.assertEqual(cache.requested_question, self.question)
        self.assertFalse(mock_post.call_args.kwargs["json"]["use_history"])

    def test_does_not_retry_for_4xx_request_error(self) -> None:
        cache = FakeCache()
        client = RagApiClient(
            cache=cache,
            base_url="http://test",
            max_retries=2,
            retry_interval_seconds=0,
        )
        request = httpx.Request("POST", "http://test/api/rag/chat")
        response = httpx.Response(400, request=request)
        error = httpx.HTTPStatusError(
            "bad request",
            request=request,
            response=response,
        )

        with patch("agent.rag_client.httpx.post", side_effect=error) as mock_post:
            result = client.ask(self.question, self.session_id, "agent-trace-003")

        self.assertEqual(result.answer, "课程资料请求参数异常，请检查后重试。")
        self.assertEqual(result.sources, [])
        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
