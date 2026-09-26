"""LangGraph 可见的工具代理：业务实现通过 MCP Server 执行。"""

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from agent.execution import AgentExecutionContext
from agent.mcp_client import McpToolClient, McpToolClientError


logger = logging.getLogger(__name__)


def build_tools(
    *,
    session_id: str,
    trace_id: str,
    execution_context: AgentExecutionContext,
    on_status: Callable[[str], None] | None = None,
    cache_question: str | None = None,
    mcp_client: McpToolClient | None = None,
):
    """创建请求专属 MCP 工具代理，避免会话和执行记录串到其他请求。"""
    client = mcp_client or McpToolClient()

    def emit_status(content: str) -> None:
        if on_status is not None:
            on_status(content)

    def tool_response(*, ok: bool, code: str, message: str, data: object | None = None) -> str:
        return json.dumps({"ok": ok, "code": code, "message": message, "data": data}, ensure_ascii=False)

    def begin(tool_name: str, arguments: dict[str, object]) -> str | None:
        code = execution_context.begin_tool_call(tool_name, arguments)
        if code is not None:
            logger.warning("工具调用被保护规则拦截：tool=%s, code=%s, trace_id=%s", tool_name, code, trace_id)
        return code

    def call_mcp(tool_name: str, arguments: dict[str, object]) -> dict[str, Any]:
        request_arguments = {**arguments, "session_id": session_id, "trace_id": trace_id}
        try:
            return client.call_tool(tool_name, request_arguments)
        except McpToolClientError:
            logger.exception("MCP 工具调用失败：tool=%s, trace_id=%s", tool_name, trace_id)
            return {"ok": False, "code": "MCP_UNAVAILABLE", "message": "学习工具服务暂时不可用。", "data": None}

    def public_response(payload: dict[str, Any]) -> str:
        return tool_response(
            ok=bool(payload.get("ok")),
            code=str(payload.get("code") or "MCP_RESPONSE_INVALID"),
            message=str(payload.get("message") or "MCP 工具返回格式异常。"),
            data=payload.get("data"),
        )

    @tool(description="查询 UQ 课程、作业、截止日期和已上传课程资料。涉及课程事实时必须使用此工具。")
    def search_course_knowledge(question: str) -> str:
        if not question.strip():
            return tool_response(ok=False, code="INVALID_QUESTION", message="检索问题不能为空。")
        blocked = begin("search_course_knowledge", {"question": question})
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")
        emit_status("正在通过 MCP 检索课程资料")
        payload = call_mcp(
            "search_course_knowledge",
            {
                "question": question,
                "cache_question": cache_question or question,
            },
        )
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        execution_context.record_rag_result(
            sources=list(meta.get("sources") or []),
            rag_trace_id=meta.get("rag_trace_id"),
            cache_hit=bool(meta.get("cache_hit")),
        )
        return public_response(payload)

    @tool(description="为指定课程创建或更新学习计划。course_id 例如 INFS7410；deadline 为 YYYY-MM-DD。")
    def create_or_update_study_plan(course_id: str, task: str, deadline: str, daily_study_hours: float = 2.0) -> str:
        arguments = {"course_id": course_id.upper(), "task": task, "deadline": deadline, "daily_study_hours": daily_study_hours}
        blocked = begin("create_or_update_study_plan", arguments)
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")
        emit_status("正在通过 MCP 生成并保存学习计划")
        return public_response(call_mcp("create_or_update_study_plan", arguments))

    @tool(description="读取指定课程已经保存的学习计划。必须提供 course_id。")
    def get_study_plan(course_id: str) -> str:
        arguments = {"course_id": course_id.upper()}
        blocked = begin("get_study_plan", arguments)
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")
        emit_status("正在通过 MCP 读取已保存的学习计划")
        return public_response(call_mcp("get_study_plan", arguments))

    @tool(description="列出当前会话已保存计划的课程代码。用户不确定保存过哪些计划时使用。")
    def list_study_plans() -> str:
        blocked = begin("list_study_plans", {})
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")
        emit_status("正在通过 MCP 列出已保存的学习计划")
        return public_response(call_mcp("list_study_plans", {}))

    @tool(description="仅在用户明确说出稳定学习偏好时保存偏好，例如每日学习时长或复习缓冲天数。")
    def save_learning_preferences(daily_study_hours: float | None = None, review_buffer_days: int | None = None) -> str:
        arguments = {"daily_study_hours": daily_study_hours, "review_buffer_days": review_buffer_days}
        blocked = begin("save_learning_preferences", arguments)
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")
        payload = call_mcp("save_learning_preferences", arguments)
        if payload.get("ok") is True:
            execution_context.profile_save_succeeded = True
        return public_response(payload)

    return [search_course_knowledge, create_or_update_study_plan, get_study_plan, list_study_plans, save_learning_preferences]
