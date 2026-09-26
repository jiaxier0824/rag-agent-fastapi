"""LangGraph 的构建、消息上下文和同步/流式执行。"""

from collections.abc import Callable, Iterator

from langchain_community.chat_models.tongyi import ChatTongyi

from agent.execution import AgentExecutionContext
from agent.graph import build_agent_graph
from agent.memory import AgentMemoryStore
from agent.preference_confirmation import preference_save_requested
from agent.tools import build_tools
from config.settings import settings


AGENT_SYSTEM_PROMPT = (
    "你是 UQ 学习助手。"
    "涉及课程、作业、截止日期或上传资料的事实问题时，"
    "必须调用 search_course_knowledge。"
    "课程代码本身不代表需要检索；只有回答需要核实课程资料中的事实或概念时，"
    "才先调用 search_course_knowledge，不能凭模型记忆回答课程事实。"
    "即使概念属于通识，只要用户问的是某门课程如何讲述或要求依据课程材料回答，"
    "也必须先检索，不能凭通识记忆直接作答。"
    "如果用户已经给出课程代码、任务、截止日期和每日学习时长，并明确要求创建或保存计划，"
    "直接调用 create_or_update_study_plan，不必检索课程资料，也不要再询问是否确认。"
    "当用户要求生成学习计划时，如果任务或截止日期需要从课程资料确认，"
    "先调用 search_course_knowledge，再调用 create_or_update_study_plan。"
    "调用 create_or_update_study_plan 时，必须提供 course_id，deadline 必须使用 YYYY-MM-DD 格式。"
    "不要编造工具没有返回的课程信息。"
    "当课程资料工具返回 ok=false 时，明确说明资料不可用，不要补写事实。"
    "当用户询问某课程已保存计划时调用 get_study_plan；不确定有哪些计划时调用 list_study_plans。"
    "用户明确说‘记住’或‘保存’每日学习时长、复习缓冲天数等长期偏好时，"
    "必须调用 save_learning_preferences；只有该工具返回 ok=true 才能声称已保存。"
    "用户没有明确要求保存时不要调用该工具。"
    "工具返回 DUPLICATE_TOOL_CALL 或 TOOL_CALL_LIMIT_EXCEEDED 后，"
    "不要再次调用同一工具，应向用户说明限制。"
)

class AgentGraphRuntime:
    """只负责准备 LangGraph 输入并运行图，不处理缓存、降级或日志。"""

    def __init__(self, memory_store: AgentMemoryStore | None):
        self.memory_store = memory_store

    def invoke(
        self,
        *,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
    ) -> str:
        graph = self._build_graph(
            model=model,
            session_id=session_id,
            trace_id=trace_id,
            execution_context=execution_context,
            cache_question=question,
        )
        result = graph.invoke(
            self._initial_state(question, session_id, trace_id, execution_context),
            config={"recursion_limit": settings.agent_max_tool_calls * 2 + 4},
        )
        return result["messages"][-1].content

    def stream(
        self,
        *,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
        on_status: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        graph = self._build_graph(
            model=model,
            session_id=session_id,
            trace_id=trace_id,
            execution_context=execution_context,
            on_status=on_status,
            cache_question=question,
        )
        for message, metadata in graph.stream(
            self._initial_state(question, session_id, trace_id, execution_context),
            config={"recursion_limit": settings.agent_max_tool_calls * 2 + 4},
            stream_mode="messages",
        ):
            if (
                metadata.get("langgraph_node") == "agent_reason"
                and isinstance(message.content, str)
                and message.content
            ):
                yield message.content

    def _build_graph(
        self,
        *,
        model: ChatTongyi,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
        on_status: Callable[[str], None] | None = None,
        cache_question: str | None = None,
    ):
        return build_agent_graph(
            model=model,
            tools=build_tools(
                session_id=session_id,
                trace_id=trace_id,
                execution_context=execution_context,
                on_status=on_status,
                cache_question=cache_question,
            ),
            required_tool_name=(
                "save_learning_preferences"
                if preference_save_requested(cache_question or "") else None
            ),
        )

    def _initial_state(
        self,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
    ) -> dict[str, object]:
        return {
            "messages": self._build_messages(question, session_id, execution_context),
            "session_id": session_id,
            "trace_id": trace_id,
        }

    def _build_messages(
        self,
        question: str,
        session_id: str,
        execution_context: AgentExecutionContext,
    ) -> list[dict[str, str]]:
        if self.memory_store is None:
            short_memory: list[dict[str, str]] = []
            profile: dict[str, str] = {}
        else:
            short_memory = self.memory_store.get_recent_messages(session_id=session_id)
            profile = self.memory_store.get_profile(session_id=session_id)

        execution_context.record_memory(
            short_memory_turns=len(short_memory) // 2,
            profile_loaded=bool(profile),
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        if profile:
            preferences = "；".join(f"{key}={value}" for key, value in profile.items())
            messages.append({"role": "system", "content": f"用户明确保存的学习偏好：{preferences}。"})
        messages.extend(short_memory)
        messages.append({"role": "user", "content": question})
        return messages
