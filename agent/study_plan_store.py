import logging
import re

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
            course_id: str,
            plan: str,
    ) -> bool:
        try:
            self.redis_client.setex(
                self._build_key(session_id, course_id),
                self.ttl_seconds,
                plan,
            )
            return True
        except redis.RedisError:
            logger.warning("学习计划保存失败，本次跳过保存")
            return False

    def get(
            self,
            session_id: str,
            course_id: str,
            )->str | None:
        try:
            return self.redis_client.get(
                self._build_key(session_id, course_id),
            )

        except redis.RedisError:
            logger.warning("学习计划读取失败，本次跳过读取")
            return None

    def list_plans(self, session_id: str) -> list[str]:
        prefix = f"study_plan:{session_id}:"
        course_ids: list[str] = []
        try:
            for key in self.redis_client.scan_iter(match=f"{prefix}*"):
                plan = self.redis_client.get(key)
                if plan is not None:
                    course_ids.append(key.removeprefix(prefix))
            return sorted(course_ids)
        except redis.RedisError:
            logger.warning("学习计划列表读取失败，本次返回空列表")
            return []

    def _build_key(self, session_id: str, course_id: str) -> str:
        normalized_course_id = course_id.strip().upper()
        if not re.fullmatch(r"[A-Z0-9_-]{2,32}", normalized_course_id):
            raise ValueError("课程编号只能包含字母、数字、下划线或连字符。")
        return f"study_plan:{session_id}:{normalized_course_id}"
