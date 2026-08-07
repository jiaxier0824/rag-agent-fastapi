import hashlib
import logging

import redis

from config.settings import settings

logger = logging.getLogger(__name__)

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
    ) -> str | None:
        cache_key = self._build_cache_key(
            question=question,
            session_id=session_id,
        )

        try:
            return self.redis_client.get(cache_key)
        except redis.RedisError:
            logger.warning("Redis 读取失败，本次跳过缓存。")
            return None

    def set(
            self,
            question: str,
            session_id: str,
            answer: str,
    ) -> None:
        cache_key = self._build_cache_key(
            question=question,
            session_id=session_id,
        )

        try:
            self.redis_client.setex(
                cache_key,
                self.ttl_seconds,
                answer,
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