"""供模型选择的学习助手工具，以及工具级保护规则。"""

import json
import logging
import re
from datetime import date, datetime
from collections.abc import Callable

from langchain_core.tools import tool

from agent.execution import AgentExecutionContext
from agent.memory import AgentMemoryStore
from agent.rag_client import RagApiClient
from agent.study_plan import build_study_plan
from agent.study_plan_store import StudyPlanStore


logger = logging.getLogger(__name__)
_COURSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


def build_tools(
    rag_client: RagApiClient,
    session_id: str,
    trace_id: str,
    study_plan_store: StudyPlanStore,
    memory_store: AgentMemoryStore,
    execution_context: AgentExecutionContext,
    on_status: Callable[[str], None] | None = None,
    cache_question: str | None = None,
):
    """创建本请求专属工具，避免 session 和执行记录串到其他用户。"""

    def emit_status(content: str) -> None:
        if on_status is not None:
            on_status(content)

    def tool_response(
        *,
        ok: bool,
        code: str,
        message: str,
        data: object | None = None,
    ) -> str:
        return json.dumps(
            {"ok": ok, "code": code, "message": message, "data": data},
            ensure_ascii=False,
        )

    def begin(tool_name: str, arguments: dict[str, object]) -> str | None:
        code = execution_context.begin_tool_call(tool_name, arguments)
        if code is not None:
            logger.warning(
                "工具调用被保护规则拦截：tool=%s, code=%s, trace_id=%s",
                tool_name,
                code,
                trace_id,
            )
        return code

    @tool(
        description=(
            "查询 UQ 课程、作业、截止日期和已上传课程资料。"
            "涉及课程事实时必须使用此工具。若工具返回 ok=false，不能编造资料答案。"
        )
    )
    def search_course_knowledge(question: str) -> str:
        if not question.strip():
            return tool_response(
                ok=False,
                code="INVALID_QUESTION",
                message="检索问题不能为空。",
            )
        blocked = begin("search_course_knowledge", {"question": question})
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")

        emit_status("正在检索课程资料")
        result = rag_client.ask(
            question=question,
            session_id=session_id,
            trace_id=trace_id,
            cache_question=cache_question or question,
        )
        execution_context.record_rag_result(
            sources=result.sources,
            rag_trace_id=result.rag_trace_id,
            cache_hit=result.cache_hit,
        )
        if not result.success:
            return tool_response(
                ok=False,
                code=result.error_code or "RAG_UNAVAILABLE",
                message=result.answer,
            )
        return tool_response(
            ok=True,
            code="OK",
            message="课程资料检索完成。",
            data={"answer": result.answer, "sources": result.sources},
        )

    @tool(
        description=(
            "为指定课程创建或更新学习计划，并按 course_id 保存。"
            "course_id 使用课程代码，例如 INFS7410；deadline 必须是 YYYY-MM-DD；"
            "daily_study_hours 必须在 0 到 16 之间。"
        )
    )
    def create_or_update_study_plan(
        course_id: str,
        task: str,
        deadline: str,
        daily_study_hours: float = 2.0,
    ) -> str:
        if not _COURSE_ID_PATTERN.fullmatch(course_id):
            return tool_response(ok=False, code="INVALID_COURSE_ID", message="course_id 格式不合法。")
        if not task.strip():
            return tool_response(ok=False, code="INVALID_TASK", message="任务不能为空。")
        if not 0 < daily_study_hours <= 16:
            return tool_response(ok=False, code="INVALID_DAILY_HOURS", message="每日学习时长必须在 0 到 16 小时之间。")
        try:
            deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
        except ValueError:
            return tool_response(ok=False, code="INVALID_DEADLINE", message="deadline 必须使用 YYYY-MM-DD 格式。")
        if deadline_date <= date.today():
            return tool_response(ok=False, code="PAST_DEADLINE", message="截止日期必须晚于今天。")

        arguments = {
            "course_id": course_id.upper(),
            "task": task,
            "deadline": deadline,
            "daily_study_hours": daily_study_hours,
        }
        blocked = begin("create_or_update_study_plan", arguments)
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")

        emit_status("正在生成并保存学习计划")
        try:
            plan = build_study_plan(
                task=task,
                deadline=deadline,
                daily_study_hours=daily_study_hours,
            )
        except ValueError as error:
            return tool_response(ok=False, code="INVALID_PLAN_INPUT", message=str(error))

        try:
            study_plan_store.save(session_id=session_id, course_id=course_id, plan=plan)
        except Exception:
            logger.exception("保存学习计划失败：trace_id=%s", trace_id)
            return tool_response(ok=False, code="PLAN_STORE_UNAVAILABLE", message="学习计划暂时无法保存。")
        return tool_response(ok=True, code="OK", message="学习计划已保存。", data={"course_id": course_id.upper(), "plan": plan})

    @tool(description="读取指定课程已经保存的学习计划。必须提供 course_id。")
    def get_study_plan(course_id: str) -> str:
        if not _COURSE_ID_PATTERN.fullmatch(course_id):
            return tool_response(ok=False, code="INVALID_COURSE_ID", message="course_id 格式不合法。")
        blocked = begin("get_study_plan", {"course_id": course_id.upper()})
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")

        emit_status("正在读取已保存的学习计划")
        try:
            plan = study_plan_store.get(session_id=session_id, course_id=course_id)
        except Exception:
            logger.exception("读取学习计划失败：trace_id=%s", trace_id)
            return tool_response(ok=False, code="PLAN_STORE_UNAVAILABLE", message="学习计划暂时无法读取。")
        if plan is None:
            return tool_response(ok=False, code="PLAN_NOT_FOUND", message="该课程还没有保存学习计划。")
        return tool_response(ok=True, code="OK", message="学习计划读取完成。", data={"course_id": course_id.upper(), "plan": plan})

    @tool(description="列出当前会话已保存计划的课程代码。用户不确定保存过哪些计划时使用。")
    def list_study_plans() -> str:
        blocked = begin("list_study_plans", {})
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")
        emit_status("正在列出已保存的学习计划")
        try:
            course_ids = study_plan_store.list_plans(session_id=session_id)
        except Exception:
            logger.exception("列出学习计划失败：trace_id=%s", trace_id)
            return tool_response(ok=False, code="PLAN_STORE_UNAVAILABLE", message="学习计划暂时无法读取。")
        return tool_response(ok=True, code="OK", message="课程计划列表读取完成。", data={"course_ids": course_ids})

    @tool(
        description=(
            "仅在用户明确说出稳定学习偏好时保存偏好，例如每日学习时长或复习缓冲天数。"
            "不要从普通聊天中自行推断或保存偏好。"
        )
    )
    def save_learning_preferences(
        daily_study_hours: float | None = None,
        review_buffer_days: int | None = None,
    ) -> str:
        if daily_study_hours is None and review_buffer_days is None:
            return tool_response(ok=False, code="EMPTY_PREFERENCES", message="至少提供一项明确偏好。")
        if daily_study_hours is not None and not 0 < daily_study_hours <= 16:
            return tool_response(ok=False, code="INVALID_DAILY_HOURS", message="每日学习时长必须在 0 到 16 小时之间。")
        if review_buffer_days is not None and not 0 <= review_buffer_days <= 30:
            return tool_response(ok=False, code="INVALID_REVIEW_BUFFER", message="复习缓冲天数必须在 0 到 30 天之间。")
        arguments = {"daily_study_hours": daily_study_hours, "review_buffer_days": review_buffer_days}
        blocked = begin("save_learning_preferences", arguments)
        if blocked:
            return tool_response(ok=False, code=blocked, message="本次请求已阻止重复或过多工具调用。")

        try:
            profile = memory_store.update_profile(session_id=session_id, **arguments)
        except Exception:
            logger.exception("保存学习偏好失败：trace_id=%s", trace_id)
            return tool_response(ok=False, code="PROFILE_STORE_UNAVAILABLE", message="学习偏好暂时无法保存。")
        return tool_response(ok=True, code="OK", message="明确的学习偏好已保存。", data=profile)

    return [
        search_course_knowledge,
        create_or_update_study_plan,
        get_study_plan,
        list_study_plans,
        save_learning_preferences,
    ]
