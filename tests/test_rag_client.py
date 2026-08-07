import unittest
from unittest.mock import patch

import httpx

from agent.rag_client import RagApiClient


class FakeCache:
    def __init__(self, cached_answer: str | None = None):
        self.cached_answer = cached_answer
        self.saved_answer: tuple[str, str, str] | None = None

    def get(self, question: str, session_id: str) -> str | None:
        return self.cached_answer

    def set(self, question: str, session_id: str, answer: str) -> None:
        self.saved_answer = (question, session_id, answer)


class SuccessResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"answer": "课程资料答案"}


class RagApiClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = "COMP9001 的作业要求是什么？"
        self.session_id = "rag-client-test"

    def test_returns_cached_answer_without_http_request(self) -> None:
        cache = FakeCache(cached_answer="缓存中的课程答案")
        client = RagApiClient(cache=cache, base_url="http://test")

        with patch("agent.rag_client.httpx.post") as mock_post:
            answer = client.ask(self.question, self.session_id)

        self.assertEqual(answer, "缓存中的课程答案")
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
            answer = client.ask(self.question, self.session_id)

        self.assertEqual(answer, "课程资料答案")
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(
            cache.saved_answer,
            (self.question, self.session_id, "课程资料答案"),
        )

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
            answer = client.ask(self.question, self.session_id)

        self.assertEqual(answer, "课程资料请求参数异常，请检查后重试。")
        self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
