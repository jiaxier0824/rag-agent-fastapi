"""Agent 用例编排：主备模型和同步/流式请求生命周期。"""

import logging
from collections.abc import Callable, Iterator
from time import perf_counter

from langchain_community.chat_models.tongyi import ChatTongyi

from agent.execution import AgentExecutionContext, AgentExecutionResult
from agent.memory import AgentMemoryStore
from agent.observability import AgentTraceLogger
from agent.preference_confirmation import confirmed_preference_answer, preference_save_requested
from agent.run_support import (
    metadata_event,
    persist_short_memory,
    result_from_context,
    write_stream_trace,
    write_trace,
)
from agent.runtime import AgentGraphRuntime
from agent.stream_bridge import stream_model_events
from config.settings import settings


logger = logging.getLogger(__name__)


class AgentService:
    """协调一次 Agent 请求；具体图执行、存储和日志分别下沉到专用模块。"""

    def __init__(
        self,
        memory_store: AgentMemoryStore,
        trace_logger: AgentTraceLogger | None = None,
        model_name: str = settings.agent_model_name,
        fallback_model_name: str = settings.agent_fallback_model_name,
    ):
        self.memory_store = memory_store
        self.trace_logger = trace_logger
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.model = ChatTongyi(model=model_name)
        self.fallback_model = ChatTongyi(model=fallback_model_name)
        self.runtime = AgentGraphRuntime(memory_store=memory_store)

    def _invoke_agent(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
    ) -> str:
        return self.runtime.invoke(
            model=model,
            question=question,
            session_id=session_id,
            trace_id=trace_id,
            execution_context=execution_context,
        )

    def execute(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> AgentExecutionResult:
        execution_context = AgentExecutionContext(max_tool_calls=settings.agent_max_tool_calls)
        started_at = perf_counter()
        degraded = False
        model_used = self.model_name
        logger.info(
            "Agent 使用主模型开始处理：model=%s, trace_id=%s, session_id=%s",
            self.model_name,
            trace_id,
            session_id,
        )
        try:
            answer = self._invoke_agent(
                model=self.model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
                execution_context=execution_context,
            )
        except Exception:
            degraded = True
            model_used = self.fallback_model_name
            logger.warning(
                "主模型调用失败，切换备用模型：primary=%s, fallback=%s, trace_id=%s",
                self.model_name,
                self.fallback_model_name,
                trace_id,
                exc_info=True,
            )
            try:
                answer = self._invoke_agent(
                    model=self.fallback_model,
                    question=question,
                    session_id=session_id,
                    trace_id=trace_id,
                    execution_context=execution_context,
                )
            except Exception:
                logger.exception("备用模型调用也失败：trace_id=%s", trace_id)
                failed_result = result_from_context(
                    answer="",
                    execution_context=execution_context,
                    model_used=model_used,
                    degraded=degraded,
                )
                write_trace(
                    getattr(self, "trace_logger", None),
                    trace_id=trace_id,
                    session_id=session_id,
                    mode="sync",
                    result=failed_result,
                    elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
                    success=False,
                )
                raise

        answer = confirmed_preference_answer(question, answer, execution_context)
        logger.info("Agent 处理完成：trace_id=%s, session_id=%s", trace_id, session_id)
        result = result_from_context(
            answer=answer,
            execution_context=execution_context,
            model_used=model_used,
            degraded=degraded,
        )
        persist_short_memory(
            getattr(self, "memory_store", None),
            session_id=session_id,
            question=question,
            answer=answer,
        )
        write_trace(
            getattr(self, "trace_logger", None),
            trace_id=trace_id,
            session_id=session_id,
            mode="sync",
            result=result,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
            success=True,
        )
        return result

    def stream_execute(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> Iterator[dict[str, object]]:
        execution_context = AgentExecutionContext(max_tool_calls=settings.agent_max_tool_calls)
        started_at = perf_counter()
        model_used = self.model_name
        degraded = False
        logger.info(
            "Agent 使用主模型开始流式处理：model=%s, trace_id=%s, session_id=%s",
            self.model_name,
            trace_id,
            session_id,
        )
        yield {"type": "status", "content": "正在分析你的问题"}
        hold_preference_answer = preference_save_requested(question)
        has_emitted_content = False
        answer_parts: list[str] = []

        try:
            for event in self._stream_model_events(
                model=self.model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
                execution_context=execution_context,
            ):
                if event["type"] == "content":
                    answer_parts.append(str(event["content"]))
                    if hold_preference_answer:
                        continue
                if event["type"] == "content" and not has_emitted_content:
                    yield {"type": "status", "content": "正在生成回答"}
                    has_emitted_content = True
                yield event
        except Exception:
            if has_emitted_content:
                logger.exception("主模型流式输出中断，无法切换备用模型以避免重复内容：trace_id=%s", trace_id)
                yield {"type": "error", "content": "回答生成中断，请重新提问。"}
                self._finish_stream(
                    trace_id=trace_id,
                    session_id=session_id,
                    execution_context=execution_context,
                    model_used=model_used,
                    degraded=degraded,
                    started_at=started_at,
                    success=False,
                )
                yield metadata_event(execution_context)
                return

            logger.warning(
                "主模型流式调用失败，切换备用模型：primary=%s, fallback=%s, trace_id=%s",
                self.model_name,
                self.fallback_model_name,
                trace_id,
                exc_info=True,
            )
            yield {"type": "status", "content": "主模型暂时不可用，正在切换备用模型"}
            degraded = True
            model_used = self.fallback_model_name
            yield {"type": "status", "content": "正在使用备用模型生成回答"}
            if hold_preference_answer:
                answer_parts.clear()
            try:
                for event in self._stream_model_events(
                    model=self.fallback_model,
                    question=question,
                    session_id=session_id,
                    trace_id=trace_id,
                    execution_context=execution_context,
                ):
                    if event["type"] == "content":
                        answer_parts.append(str(event["content"]))
                        if hold_preference_answer:
                            continue
                    yield event
            except Exception:
                logger.exception("备用模型流式调用也失败：trace_id=%s", trace_id)
                yield {"type": "error", "content": "当前模型服务暂时不可用，请稍后重试。"}
                self._finish_stream(
                    trace_id=trace_id,
                    session_id=session_id,
                    execution_context=execution_context,
                    model_used=model_used,
                    degraded=degraded,
                    started_at=started_at,
                    success=False,
                )
                yield metadata_event(execution_context)
                return

        if hold_preference_answer:
            answer = confirmed_preference_answer(question, "".join(answer_parts), execution_context)
            answer_parts = [answer]
            yield {"type": "status", "content": "正在生成回答"}
            yield {"type": "content", "content": answer}

        logger.info("Agent 流式处理完成：trace_id=%s, session_id=%s", trace_id, session_id)
        persist_short_memory(
            getattr(self, "memory_store", None),
            session_id=session_id,
            question=question,
            answer="".join(answer_parts),
        )
        self._finish_stream(
            trace_id=trace_id,
            session_id=session_id,
            execution_context=execution_context,
            model_used=model_used,
            degraded=degraded,
            started_at=started_at,
            success=True,
        )
        yield metadata_event(execution_context)

    def _stream_agent(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
        on_status: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        yield from self.runtime.stream(
            model=model,
            question=question,
            session_id=session_id,
            trace_id=trace_id,
            execution_context=execution_context,
            on_status=on_status,
        )

    def _stream_model_events(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
    ) -> Iterator[dict[str, object]]:
        yield from stream_model_events(
            lambda publish_status: self._stream_agent(
                model=model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
                execution_context=execution_context,
                on_status=publish_status,
            )
        )

    def _finish_stream(
        self,
        *,
        trace_id: str,
        session_id: str,
        execution_context: AgentExecutionContext,
        model_used: str,
        degraded: bool,
        started_at: float,
        success: bool,
    ) -> None:
        write_stream_trace(
            getattr(self, "trace_logger", None),
            trace_id=trace_id,
            session_id=session_id,
            execution_context=execution_context,
            model_used=model_used,
            degraded=degraded,
            started_at=started_at,
            success=success,
        )
