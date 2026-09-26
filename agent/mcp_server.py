"""UQ 学习助手 MCP Server：暴露全部五个业务工具。"""

import re
from datetime import date, datetime

from mcp.server.fastmcp import FastMCP

from agent.study_plan import build_study_plan
from dependencies import get_agent_memory_store, get_rag_api_client, get_study_plan_store


mcp = FastMCP("UQ Study Assistant")
_COURSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


def _response(*, ok: bool, code: str, message: str, data: object | None = None, meta: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"ok": ok, "code": code, "message": message, "data": data}
    if meta is not None:
        payload["meta"] = meta
    return payload


@mcp.tool(description="查询 UQ 课程、作业、截止日期和已上传课程资料。")
def search_course_knowledge(
    question: str,
    session_id: str,
    trace_id: str,
    cache_question: str | None = None,
) -> dict[str, object]:
    if not question.strip():
        return _response(ok=False, code="INVALID_QUESTION", message="检索问题不能为空。")
    result = get_rag_api_client().ask(
        question=question,
        session_id=session_id,
        trace_id=trace_id,
        cache_question=cache_question or question,
    )
    meta = {"sources": result.sources, "rag_trace_id": result.rag_trace_id, "cache_hit": result.cache_hit}
    if not result.success:
        return _response(ok=False, code=result.error_code or "RAG_UNAVAILABLE", message=result.answer, meta=meta)
    return _response(ok=True, code="OK", message="课程资料检索完成。", data={"answer": result.answer, "sources": result.sources}, meta=meta)


@mcp.tool(description="为指定课程创建或更新学习计划，并按 course_id 保存。")
def create_or_update_study_plan(course_id: str, task: str, deadline: str, daily_study_hours: float, session_id: str, trace_id: str) -> dict[str, object]:
    del trace_id
    if not _COURSE_ID_PATTERN.fullmatch(course_id):
        return _response(ok=False, code="INVALID_COURSE_ID", message="course_id 格式不合法。")
    if not task.strip():
        return _response(ok=False, code="INVALID_TASK", message="任务不能为空。")
    if not 0 < daily_study_hours <= 16:
        return _response(ok=False, code="INVALID_DAILY_HOURS", message="每日学习时长必须在 0 到 16 小时之间。")
    try:
        deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
    except ValueError:
        return _response(ok=False, code="INVALID_DEADLINE", message="deadline 必须使用 YYYY-MM-DD 格式。")
    if deadline_date <= date.today():
        return _response(ok=False, code="PAST_DEADLINE", message="截止日期必须晚于今天。")
    plan = build_study_plan(task=task, deadline=deadline, daily_study_hours=daily_study_hours)
    if not get_study_plan_store().save(session_id=session_id, course_id=course_id, plan=plan):
        return _response(ok=False, code="PLAN_SAVE_FAILED", message="学习计划保存失败，请稍后重试。")
    return _response(ok=True, code="OK", message="学习计划已保存。", data={"course_id": course_id.upper(), "plan": plan})


@mcp.tool(description="读取指定课程已经保存的学习计划。")
def get_study_plan(course_id: str, session_id: str, trace_id: str) -> dict[str, object]:
    del trace_id
    if not _COURSE_ID_PATTERN.fullmatch(course_id):
        return _response(ok=False, code="INVALID_COURSE_ID", message="course_id 格式不合法。")
    plan = get_study_plan_store().get(session_id=session_id, course_id=course_id)
    if plan is None:
        return _response(ok=False, code="PLAN_NOT_FOUND", message="该课程还没有保存学习计划。")
    return _response(ok=True, code="OK", message="学习计划读取完成。", data={"course_id": course_id.upper(), "plan": plan})


@mcp.tool(description="列出当前会话已保存计划的课程代码。")
def list_study_plans(session_id: str, trace_id: str) -> dict[str, object]:
    del trace_id
    course_ids = get_study_plan_store().list_plans(session_id=session_id)
    return _response(ok=True, code="OK", message="课程计划列表读取完成。", data={"course_ids": course_ids})


@mcp.tool(description="仅在用户明确表达稳定学习偏好时保存偏好。")
def save_learning_preferences(session_id: str, trace_id: str, daily_study_hours: float | None = None, review_buffer_days: int | None = None) -> dict[str, object]:
    del trace_id
    if daily_study_hours is None and review_buffer_days is None:
        return _response(ok=False, code="EMPTY_PREFERENCES", message="至少提供一项明确偏好。")
    if daily_study_hours is not None and not 0 < daily_study_hours <= 16:
        return _response(ok=False, code="INVALID_DAILY_HOURS", message="每日学习时长必须在 0 到 16 小时之间。")
    if review_buffer_days is not None and not 0 <= review_buffer_days <= 30:
        return _response(ok=False, code="INVALID_REVIEW_BUFFER", message="复习缓冲天数必须在 0 到 30 天之间。")
    profile = get_agent_memory_store().update_profile(session_id=session_id, daily_study_hours=daily_study_hours, review_buffer_days=review_buffer_days)
    if profile is None:
        return _response(ok=False, code="PROFILE_SAVE_FAILED", message="学习偏好保存失败，请稍后重试。")
    return _response(ok=True, code="OK", message="明确的学习偏好已保存。", data=profile)


if __name__ == "__main__":
    mcp.run(transport="stdio")
