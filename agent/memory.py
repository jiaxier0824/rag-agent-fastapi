"""Agent 的短期会话记忆与显式长期偏好记忆。"""

import json
import logging

import redis

from config.settings import settings

logger = logging.getLogger(__name__)


class AgentMemoryStore:
    """Redis 中只保留有限短期对话和用户明确保存的稳定偏好。"""

    def __init__(
        self,
        redis_url: str = settings.redis_url,
        short_memory_ttl_seconds: int = settings.agent_short_memory_ttl_seconds,
        short_memory_max_turns: int = settings.agent_short_memory_max_turns,
        profile_ttl_seconds: int = settings.agent_profile_ttl_seconds,
    ):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.short_memory_ttl_seconds = short_memory_ttl_seconds
        self.short_memory_max_turns = short_memory_max_turns
        self.profile_ttl_seconds = profile_ttl_seconds

    def get_recent_messages(self, session_id: str) -> list[dict[str, str]]:
        try:
            items = self.redis_client.lrange(
                self._history_key(session_id),
                0,
                self.short_memory_max_turns * 2 - 1,
            )
            messages = [json.loads(item) for item in reversed(items)]
            return [
                message
                for message in messages
                if message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
            ]
        except (redis.RedisError, json.JSONDecodeError, TypeError, AttributeError):
            logger.warning("Agent 短期记忆读取失败，本次跳过历史")
            return []

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        try:
            # LPUSH 后最新 assistant 在最前；读取时 reverse 回时间正序。
            self.redis_client.lpush(
                self._history_key(session_id),
                json.dumps({"role": "user", "content": question}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False),
            )
            self.redis_client.ltrim(
                self._history_key(session_id),
                0,
                self.short_memory_max_turns * 2 - 1,
            )
            self.redis_client.expire(
                self._history_key(session_id),
                self.short_memory_ttl_seconds,
            )
        except redis.RedisError:
            logger.warning("Agent 短期记忆写入失败，本次跳过保存")

    def get_profile(self, session_id: str) -> dict[str, str]:
        try:
            value = self.redis_client.get(self._profile_key(session_id))
            if value is None:
                return {}
            payload = json.loads(value)
            return {
                str(key): str(item)
                for key, item in payload.items()
                if isinstance(key, str) and isinstance(item, (str, int, float))
            }
        except (redis.RedisError, json.JSONDecodeError, TypeError, AttributeError):
            logger.warning("Agent 长期偏好读取失败，本次跳过偏好")
            return {}

    def update_profile(self, session_id: str, **updates: object) -> dict[str, object]:
        profile = self.get_profile(session_id)
        profile.update({key: value for key, value in updates.items() if value is not None})
        try:
            self.redis_client.setex(
                self._profile_key(session_id),
                self.profile_ttl_seconds,
                json.dumps(profile, ensure_ascii=False),
            )
        except redis.RedisError:
            logger.warning("Agent 长期偏好写入失败，本次跳过保存")
        return profile

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"agent_memory:history:{session_id}"

    @staticmethod
    def _profile_key(session_id: str) -> str:
        return f"agent_memory:profile:{session_id}"
