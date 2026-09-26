"""学习计划和偏好保存失败时，不能向用户报告成功。"""

import unittest
from datetime import date, timedelta
from unittest.mock import patch

import redis

from agent.mcp_server import create_or_update_study_plan, save_learning_preferences
from agent.memory import AgentMemoryStore
from agent.study_plan import build_study_plan
from agent.study_plan_store import StudyPlanStore


class FailingRedis:
    def get(self, _key):
        return None

    def setex(self, *_args):
        raise redis.ConnectionError("simulated Redis outage")


class PersistenceEdgeTest(unittest.TestCase):
    def test_one_day_plan_has_no_negative_or_empty_phase(self):
        deadline = (date.today() + timedelta(days=1)).isoformat()
        plan = build_study_plan("复习", deadline)

        self.assertIn("剩余天数：1 天", plan)
        self.assertNotIn("最后 -1 天", plan)
        self.assertNotIn("最后 0 天", plan)
        self.assertNotIn("第 2-", plan)

    def test_plan_save_failure_is_reported_by_mcp(self):
        store = StudyPlanStore()
        store.redis_client = FailingRedis()
        deadline = (date.today() + timedelta(days=3)).isoformat()

        with patch("agent.mcp_server.get_study_plan_store", return_value=store):
            response = create_or_update_study_plan(
                course_id="INFS7410",
                task="复习",
                deadline=deadline,
                daily_study_hours=2.0,
                session_id="test-session",
                trace_id="test-trace",
            )

        self.assertFalse(response["ok"])
        self.assertEqual(response["code"], "PLAN_SAVE_FAILED")

    def test_profile_save_failure_is_reported_by_mcp(self):
        store = AgentMemoryStore()
        store.redis_client = FailingRedis()

        with patch("agent.mcp_server.get_agent_memory_store", return_value=store):
            response = save_learning_preferences(
                session_id="test-session",
                trace_id="test-trace",
                daily_study_hours=2.0,
            )

        self.assertFalse(response["ok"])
        self.assertEqual(response["code"], "PROFILE_SAVE_FAILED")


if __name__ == "__main__":
    unittest.main()
