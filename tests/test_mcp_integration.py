"""MCP 协议边界测试：真实 stdio Client 能调用真实 Server。"""

import unittest
from unittest.mock import patch

from agent.mcp_client import McpToolClient


class McpIntegrationTest(unittest.TestCase):
    def test_server_receives_only_required_connection_settings(self):
        with patch.dict("os.environ", {
            "RAG_API_BASE_URL": "http://host.docker.internal:8000",
            "REDIS_URL": "redis://redis:6379/0",
            "DASHSCOPE_API_KEY": "secret-not-needed-by-mcp",
        }):
            environment = McpToolClient()._server_parameters().env
        self.assertEqual(environment["RAG_API_BASE_URL"], "http://host.docker.internal:8000")
        self.assertEqual(environment["REDIS_URL"], "redis://redis:6379/0")
        self.assertNotIn("DASHSCOPE_API_KEY", environment)

    def test_server_registers_all_five_learning_tools(self):
        tool_names = McpToolClient().list_tool_names()
        self.assertSetEqual(
            tool_names,
            {
                "search_course_knowledge",
                "create_or_update_study_plan",
                "get_study_plan",
                "list_study_plans",
                "save_learning_preferences",
            },
        )

    def test_list_study_plans_over_stdio_mcp(self):
        payload = McpToolClient().call_tool(
            "list_study_plans",
            {"session_id": "mcp-test-session", "trace_id": "mcp-test-trace"},
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["code"], "OK")
        self.assertIn("course_ids", payload["data"])


if __name__ == "__main__":
    unittest.main()
