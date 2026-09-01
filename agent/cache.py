import hashlib
import json
import logging
from dataclasses import dataclass

import redis

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedRagResponse:
    answer: str
    sources: list[str]
    rag_trace_id: str | None

class RagCache:
    def __init__(
            self,
            redis_url: str = settings.redis_url,
            ttl_seconds: int = settings.rag_cache_ttl_seconds,
    ):

        self.redis_client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )
        self.ttl_seconds = ttl_seconds

    def get(
            self,
            question: str,
            session_id: str,
    ) -> CachedRagResponse | None:
        cache_key = self._build_cache_key(
            question=question,
            session_id=session_id,
        )

        try:
            value = self.redis_client.get(cache_key)
            if value is None:
                return None

            payload = json.loads(value)
            return CachedRagResponse(
                answer=payload["answer"],
                sources=payload.get("sources", []),
                rag_trace_id=payload.get("rag_trace_id"),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Redis 缓存格式过期，本次跳过缓存。")
            return None
        except redis.RedisError:
            logger.warning("Redis 读取失败，本次跳过缓存。")
            return None

    def set(
            self,
        question: str,
        session_id: str,
        answer: str,
        sources: list[str],
        rag_trace_id: str | None,
    ) -> None:
        cache_key = self._build_cache_key(
            question=question,
            session_id=session_id,
        )

        try:
            self.redis_client.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(
                    {
                        "answer": answer,
                        "sources": sources,
                        "rag_trace_id": rag_trace_id,
                    },
                    ensure_ascii=False,
                ),
            )
        except redis.RedisError:
            logger.warning("Redis 写入失败，本次跳过缓存。")

    def _build_cache_key(
            self,
            question: str,
            session_id: str,
    ) -> str:
        raw_key = f"{session_id}:{question}"
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        return f"rag_cache:{key_hash}"
