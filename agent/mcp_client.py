"""Agent 侧的同步 MCP stdio 客户端。"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class McpToolClientError(RuntimeError):
    """MCP Server 不可用或返回非预期结果。"""


class McpToolClient:
    """每次工具调用使用 MCP stdio 协议连接学习助手 MCP Server。"""

    def __init__(self, python_executable: str | None = None):
        self.python_executable = python_executable or sys.executable

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, Any]:
        return asyncio.run(self._call_tool(name, arguments))

    def list_tool_names(self) -> set[str]:
        return asyncio.run(self._list_tool_names())

    def _server_parameters(self) -> StdioServerParameters:
        project_root = str(Path(__file__).resolve().parent.parent)
        # MCP SDK 默认只继承安全的系统变量；业务连接地址必须显式传给子进程。
        inherited_settings = (
            "RAG_API_BASE_URL", "REDIS_URL", "RAG_API_TIMEOUT_SECONDS",
            "RAG_API_MAX_RETRIES", "RAG_API_RETRY_INTERVAL_SECONDS",
            "RAG_CACHE_TTL_SECONDS", "STUDY_PLAN_TTL_SECONDS",
            "AGENT_PROFILE_TTL_SECONDS",
        )
        return StdioServerParameters(
            command=self.python_executable,
            args=["-m", "agent.mcp_server"],
            cwd=project_root,
            env={name: os.environ[name] for name in inherited_settings if name in os.environ},
        )

    async def _call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, Any]:
        server = self._server_parameters()
        try:
            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.call_tool(name, arguments=arguments)
        except Exception as error:
            raise McpToolClientError(f"MCP 工具 {name} 不可用") from error

        if response.isError:
            raise McpToolClientError(f"MCP 工具 {name} 返回错误")

        for content in response.content:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as error:
                    raise McpToolClientError(f"MCP 工具 {name} 返回了非 JSON 内容") from error
                if isinstance(payload, dict):
                    return payload

        raise McpToolClientError(f"MCP 工具 {name} 未返回可读取的数据")

    async def _list_tool_names(self) -> set[str]:
        try:
            async with stdio_client(self._server_parameters()) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()
        except Exception as error:
            raise McpToolClientError("无法读取 MCP 工具清单") from error
        return {item.name for item in response.tools}
