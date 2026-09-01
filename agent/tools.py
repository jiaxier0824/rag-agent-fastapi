import logging
from collections.abc import Callable

from langchain_core.tools import tool

from agent.rag_client import RagApiClient
from agent.execution import AgentExecutionContext
from agent.study_plan import build_study_plan
from agent.study_plan_store import StudyPlanStore


logger = logging.getLogger(__name__)


def build_tools(
    rag_client: RagApiClient,
    session_id: str,
    trace_id: str,
    study_plan_store: StudyPlanStore,
    execution_context: AgentExecutionContext,
    on_status: Callable[[str], None] | None = None,
):
    def emit_status(content: str) -> None:
        if on_status is not None:
            on_status(content)

    @tool(
        description=(
            "查询 UQ 课程，作业，截止日期和已上传课程资料。"
            "涉及课程事实时必须使用此工具"
        )
    )
    def search_course_knowledge(question: str) -> str:
        emit_status("正在检索课程资料")
        execution_context.record_tool("search_course_knowledge")
        logger.info(
            "调用课程资料工具：trace_id=%s, session_id=%s, question=%s",
            trace_id,
            session_id,
            question,
        )

        result = rag_client.ask(
            question=question,
            session_id=session_id,
            trace_id=trace_id,
        )
        execution_context.record_rag_result(
            sources=result.sources,
            rag_trace_id=result.rag_trace_id,
            cache_hit=result.cache_hit,
        )

        logger.info(
            "课程资料工具调用完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        return result.answer

    @tool(
        description=(
            "根据任务、截止日期和每日学习时长生成学习计划。"
            "deadline 必须使用 YYYY-MM-DD 格式。"
        )
    )
    def create_study_plan(
        task: str,
        deadline: str,
        daily_study_hours: float = 2.0,
    ) -> str:
        emit_status("正在生成并保存学习计划")
        execution_context.record_tool("create_study_plan")
        logger.info(
            "调用学习计划工具：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        plan = build_study_plan(
            task=task,
            deadline=deadline,
            daily_study_hours=daily_study_hours,
        )

        study_plan_store.save(
            session_id=session_id,
            plan=plan,
        )

        logger.info(
            "学习计划生成并保存完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        return plan

    @tool(
        description=(
            "读取当前用户会话已经保存的学习计划。"
            "当用户询问当前计划、之前的计划或已保存计划时必须使用。"
        )
    )
    def get_current_study_plan() -> str:
        emit_status("正在读取已保存的学习计划")
        execution_context.record_tool("get_current_study_plan")
        logger.info(
            "读取学习计划：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        plan = study_plan_store.get(
            session_id=session_id,
        )

        if plan is None:
            return "当前会话还没有保存学习计划，请先让我为你生成一份计划。"

        logger.info(
            "学习计划读取完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        return plan

    return [
        search_course_knowledge,
        create_study_plan,
        get_current_study_plan,
    ]
