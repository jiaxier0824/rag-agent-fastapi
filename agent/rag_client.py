import logging
from dataclasses import dataclass
from time import sleep

import httpx

from agent.cache import RagCache
from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagQueryResult:
    answer: str
    sources: list[str]
    rag_trace_id: str | None
    cache_hit: bool


class RagApiClient:
    def __init__(
        self,
        cache: RagCache,
        base_url: str = settings.rag_api_base_url,
        timeout_seconds: float = settings.rag_api_timeout_seconds,
        max_retries: int = settings.rag_api_max_retries,
        retry_interval_seconds: float = settings.rag_api_retry_interval_seconds,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_interval_seconds = retry_interval_seconds
        self.cache = cache

    def ask(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> RagQueryResult:
        cached_answer = self.cache.get(
            question=question,
            session_id=session_id,
        )

        if cached_answer is not None:
            logger.info(
                "RAG 缓存命中：session_id=%s",
                session_id,
            )
            return RagQueryResult(
                answer=cached_answer.answer,
                sources=cached_answer.sources,
                rag_trace_id=cached_answer.rag_trace_id,
                cache_hit=True,
            )

        logger.info(
            "RAG 缓存未命中：session_id=%s",
            session_id,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/rag/chat",
                    json={
                        "question": question,
                        "session_id": session_id,
                    },
                    headers={"X-Trace-ID": trace_id},
                    timeout=self.timeout_seconds,
                )

                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = [
                    source["filename"]
                    for source in data.get("sources", [])
                    if isinstance(source, dict) and source.get("filename")
                ]
                rag_trace_id = data.get("trace_id")

                self.cache.set(
                    question=question,
                    session_id=session_id,
                    answer=answer,
                    sources=sources,
                    rag_trace_id=rag_trace_id,
                )

                return RagQueryResult(
                    answer=answer,
                    sources=sources,
                    rag_trace_id=rag_trace_id,
                    cache_hit=False,
                )

            except httpx.HTTPStatusError as error:
                status_code = error.response.status_code

                if 400 <= status_code < 500:
                    logger.warning(
                        "RAG 请求参数异常，不重试：status_code=%s, session_id=%s",
                        status_code,
                        session_id,
                    )
                    return RagQueryResult(
                        answer="课程资料请求参数异常，请检查后重试。",
                        sources=[],
                        rag_trace_id=None,
                        cache_hit=False,
                    )

                error_message = "课程资料服务返回异常，请稍后重试。"

            except httpx.TimeoutException:
                error_message = "课程资料查询超时，请稍后重试。"

            except httpx.RequestError:
                error_message = "暂时无法连接课程资料服务，请稍后重试。"

            except (ValueError, KeyError):
                logger.exception(
                    "RAG 返回数据格式异常：session_id=%s",
                    session_id,
                )
                return RagQueryResult(
                    answer="课程资料服务返回的数据格式异常，请稍后重试。",
                    sources=[],
                    rag_trace_id=None,
                    cache_hit=False,
                )

            if attempt == self.max_retries:
                logger.warning(
                    "RAG 请求最终失败：attempts=%s, session_id=%s",
                    attempt + 1,
                    session_id,
                )
                return RagQueryResult(
                    answer=error_message,
                    sources=[],
                    rag_trace_id=None,
                    cache_hit=False,
                )

            wait_seconds = self.retry_interval_seconds * (attempt + 1)

            logger.warning(
                "RAG 请求失败，准备重试：attempt=%s/%s, wait_seconds=%s, session_id=%s",
                attempt + 1,
                self.max_retries + 1,
                wait_seconds,
                session_id,
            )

            sleep(wait_seconds)
