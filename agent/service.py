import logging
from collections.abc import Callable, Iterator
from queue import Queue
from threading import Thread
from time import perf_counter

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi

from agent.rag_client import RagApiClient
from agent.execution import AgentExecutionContext, AgentExecutionResult
from agent.observability import AgentTraceLogger
from agent.tools import build_tools
from config.settings import settings
from agent.study_plan_store import StudyPlanStore
logger = logging.getLogger(__name__)

class AgentService:
    def __init__(
        self,
        rag_client: RagApiClient,
        study_plan_store: StudyPlanStore,
        trace_logger: AgentTraceLogger | None = None,
        model_name: str = settings.agent_model_name,
        fallback_model_name: str = settings.agent_fallback_model_name,
    ):
        self.rag_client = rag_client
        self.study_plan_store = study_plan_store
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.trace_logger = trace_logger

        self.model = ChatTongyi(model=model_name)
        self.fallback_model = ChatTongyi(
            model=fallback_model_name,
        )

    def _build_agent(
        self,
        session_id: str,
        trace_id: str,
        model: ChatTongyi,
        execution_context: AgentExecutionContext,
        on_status: Callable[[str], None] | None = None,
    ):
        return create_agent(
            model=model,
            tools=build_tools(
                rag_client=self.rag_client,
                session_id=session_id,
                trace_id=trace_id,
                study_plan_store=self.study_plan_store,
                execution_context=execution_context,
                on_status=on_status,
            ),
            system_prompt=(
                "你是 UQ 学习助手。"
                "涉及课程、作业、截止日期或上传资料的事实问题时，"
                "必须调用 search_course_knowledge。"
                "当用户要求生成学习计划时，如果任务或截止日期需要从课程资料确认，"
                "先调用 search_course_knowledge，再调用 create_study_plan。"
                "调用 create_study_plan 时，deadline 必须使用 YYYY-MM-DD 格式。"
                "不要编造工具没有返回的课程信息。"
                "当用户询问当前、之前或已保存的学习计划时，"
                "必须调用 get_current_study_plan。"
            ),
        )

    def _invoke_agent(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
    ) -> str:
        agent = self._build_agent(
            session_id=session_id,
            trace_id=trace_id,
            model=model,
            execution_context=execution_context,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )
        return result["messages"][-1].content

    def execute(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> AgentExecutionResult:
        execution_context = AgentExecutionContext()
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
                logger.exception(
                    "备用模型调用也失败：trace_id=%s",
                    trace_id,
                )
                failed_result = AgentExecutionResult(
                    answer="",
                    sources=execution_context.sources,
                    rag_trace_ids=execution_context.rag_trace_ids,
                    tools_called=execution_context.tools_called,
                    rag_cache_hit=execution_context.rag_cache_hit,
                    model_used=model_used,
                    degraded=degraded,
                )
                self._write_trace(
                    trace_id=trace_id,
                    session_id=session_id,
                    mode="sync",
                    result=failed_result,
                    elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
                    success=False,
                )
                raise

        logger.info(
            "Agent 处理完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        result = AgentExecutionResult(
            answer=answer,
            sources=execution_context.sources,
            rag_trace_ids=execution_context.rag_trace_ids,
            tools_called=execution_context.tools_called,
            rag_cache_hit=execution_context.rag_cache_hit,
            model_used=model_used,
            degraded=degraded,
        )
        self._write_trace(
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
        execution_context = AgentExecutionContext()
        started_at = perf_counter()
        model_used = self.model_name
        degraded = False
        logger.info(
            "Agent 使用主模型开始流式处理：model=%s, trace_id=%s, session_id=%s",
            self.model_name,
            trace_id,
            session_id,
        )

        yield {
            "type": "status",
            "content": "正在分析你的问题",
        }

        has_emitted_content = False

        try:
            for event in self._stream_model_events(
                model=self.model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
                execution_context=execution_context,
            ):
                if event["type"] == "content" and not has_emitted_content:
                    yield {
                        "type": "status",
                        "content": "正在生成回答",
                    }
                    has_emitted_content = True

                yield event

        except Exception:
            if has_emitted_content:
                logger.exception(
                    "主模型流式输出中断，无法切换备用模型以避免重复内容：trace_id=%s",
                    trace_id,
                )
                yield {
                    "type": "error",
                    "content": "回答生成中断，请重新提问。",
                }
                self._write_stream_trace(
                    trace_id=trace_id,
                    session_id=session_id,
                    execution_context=execution_context,
                    model_used=model_used,
                    degraded=degraded,
                    started_at=started_at,
                    success=False,
                )
                yield self._metadata_event(execution_context)
                return

            logger.warning(
                "主模型流式调用失败，切换备用模型：primary=%s, fallback=%s, trace_id=%s",
                self.model_name,
                self.fallback_model_name,
                trace_id,
                exc_info=True,
            )

            yield {
                "type": "status",
                "content": "主模型暂时不可用，正在切换备用模型",
            }
            degraded = True
            model_used = self.fallback_model_name

            yield {
                "type": "status",
                "content": "正在使用备用模型生成回答",
            }

            try:
                for event in self._stream_model_events(
                    model=self.fallback_model,
                    question=question,
                    session_id=session_id,
                    trace_id=trace_id,
                    execution_context=execution_context,
                ):
                    yield event
            except Exception:
                logger.exception(
                    "备用模型流式调用也失败：trace_id=%s",
                    trace_id,
                )
                yield {
                    "type": "error",
                    "content": "当前模型服务暂时不可用，请稍后重试。",
                }
                self._write_stream_trace(
                    trace_id=trace_id,
                    session_id=session_id,
                    execution_context=execution_context,
                    model_used=model_used,
                    degraded=degraded,
                    started_at=started_at,
                    success=False,
                )
                yield self._metadata_event(execution_context)
                return

        logger.info(
            "Agent 流式处理完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )
        self._write_stream_trace(
            trace_id=trace_id,
            session_id=session_id,
            execution_context=execution_context,
            model_used=model_used,
            degraded=degraded,
            started_at=started_at,
            success=True,
        )
        yield self._metadata_event(execution_context)

    def _stream_agent(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
        on_status: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        agent = self._build_agent(
            session_id=session_id,
            trace_id=trace_id,
            execution_context=execution_context,
            model=model,
            on_status=on_status,
        )

        for message, metadata in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            stream_mode="messages",
        ):

            if (
                metadata["langgraph_node"] == "model"
                and isinstance(message.content, str)
                and message.content
            ):
                yield message.content

    def _stream_model_events(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        execution_context: AgentExecutionContext,
    ) -> Iterator[dict[str, object]]:
        event_queue = Queue()
        stream_finished = object()

        def publish_status(content: str) -> None:
            event_queue.put({
                "type": "status",
                "content": content,
            })

        def run_agent_stream() -> None:
            try:
                for content in self._stream_agent(
                    model=model,
                    question=question,
                    session_id=session_id,
                    trace_id=trace_id,
                    execution_context=execution_context,
                    on_status=publish_status,
                ):
                    event_queue.put({
                        "type": "content",
                        "content": content,
                    })
            except Exception as error:
                event_queue.put(error)
            finally:
                event_queue.put(stream_finished)

        Thread(target=run_agent_stream, daemon=True).start()

        while True:
            event = event_queue.get()

            if event is stream_finished:
                return

            if isinstance(event, Exception):
                raise event

            yield event

    @staticmethod
    def _metadata_event(execution_context: AgentExecutionContext) -> dict:
        return {
            "type": "metadata",
            "content": "",
            "sources": execution_context.sources,
            "rag_trace_ids": execution_context.rag_trace_ids,
        }

    def _write_stream_trace(
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
        result = AgentExecutionResult(
            answer="",
            sources=execution_context.sources,
            rag_trace_ids=execution_context.rag_trace_ids,
            tools_called=execution_context.tools_called,
            rag_cache_hit=execution_context.rag_cache_hit,
            model_used=model_used,
            degraded=degraded,
        )
        self._write_trace(
            trace_id=trace_id,
            session_id=session_id,
            mode="stream",
            result=result,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
            success=success,
        )

    def _write_trace(
        self,
        *,
        trace_id: str,
        session_id: str,
        mode: str,
        result: AgentExecutionResult,
        elapsed_ms: float,
        success: bool,
    ) -> None:
        if self.trace_logger is None:
            return

        self.trace_logger.write(
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "mode": mode,
                "success": success,
                "model_used": result.model_used,
                "degraded": result.degraded,
                "tools_called": result.tools_called,
                "rag_cache_hit": result.rag_cache_hit,
                "rag_trace_ids": result.rag_trace_ids,
                "sources": result.sources,
                "total_ms": elapsed_ms,
            }
        )
