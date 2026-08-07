import logging

import redis

from config.settings import settings

logger = logging.getLogger(__name__)


class StudyPlanStore:
    def __init__(
            self,
            redis_url: str = settings.redis_url,
            ttl_seconds: int = settings.study_plan_ttl_seconds,
    ):

        self.redis_client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )
        self.ttl_seconds = ttl_seconds

    def save(
            self,
            session_id: str,
            plan: str,
    ) -> None:
        try:
            self.redis_client.setex(
                self._build_key(session_id),
                self.ttl_seconds,
                plan,
            )
        except redis.RedisError:
            logger.warning("学习计划保存失败，本次跳过保存")

    def get(
            self,
            session_id: str,
            )->str | None:
        try:
            return self.redis_client.get(
                self._build_key(session_id),
            )

        except redis.RedisError:
            logger.warning("学习计划读取失败，本次跳过读取")
            return None

    def _build_key(self, session_id: str) -> str:
        return f"study_plan:{session_id}"
