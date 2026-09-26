"""一次 Agent 请求结束时的结果、记忆和可观测性处理。"""

from time import perf_counter

from agent.execution import AgentExecutionContext, AgentExecutionResult
from agent.memory import AgentMemoryStore
from agent.observability import AgentTraceLogger


def result_from_context(
    *,
    answer: str,
    execution_context: AgentExecutionContext,
    model_used: str,
    degraded: bool,
) -> AgentExecutionResult:
    return AgentExecutionResult(
        answer=answer,
        sources=execution_context.sources,
        rag_trace_ids=execution_context.rag_trace_ids,
        tools_called=execution_context.tools_called,
        rag_cache_hit=execution_context.rag_cache_hit,
        model_used=model_used,
        degraded=degraded,
        blocked_tool_calls=execution_context.blocked_tool_calls,
        short_memory_turns_loaded=execution_context.short_memory_turns_loaded,
        profile_memory_loaded=execution_context.profile_memory_loaded,
    )


def metadata_event(execution_context: AgentExecutionContext) -> dict[str, object]:
    return {
        "type": "metadata",
        "content": "",
        "sources": execution_context.sources,
        "rag_trace_ids": execution_context.rag_trace_ids,
        "tools_called": execution_context.tools_called,
        "blocked_tool_calls": execution_context.blocked_tool_calls,
        "short_memory_turns_loaded": execution_context.short_memory_turns_loaded,
        "profile_memory_loaded": execution_context.profile_memory_loaded,
    }


def persist_short_memory(
    memory_store: AgentMemoryStore | None,
    *,
    session_id: str,
    question: str,
    answer: str,
) -> None:
    if answer.strip() and memory_store is not None:
        memory_store.append_turn(session_id=session_id, question=question, answer=answer)


def write_trace(
    trace_logger: AgentTraceLogger | None,
    *,
    trace_id: str,
    session_id: str,
    mode: str,
    result: AgentExecutionResult,
    elapsed_ms: float,
    success: bool,
) -> None:
    if trace_logger is None:
        return
    trace_logger.write(
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
            "blocked_tool_calls": result.blocked_tool_calls,
            "short_memory_turns_loaded": result.short_memory_turns_loaded,
            "profile_memory_loaded": result.profile_memory_loaded,
            "total_ms": elapsed_ms,
        }
    )


def write_stream_trace(
    trace_logger: AgentTraceLogger | None,
    *,
    trace_id: str,
    session_id: str,
    execution_context: AgentExecutionContext,
    model_used: str,
    degraded: bool,
    started_at: float,
    success: bool,
) -> None:
    result = result_from_context(
        answer="",
        execution_context=execution_context,
        model_used=model_used,
        degraded=degraded,
    )
    write_trace(
        trace_logger,
        trace_id=trace_id,
        session_id=session_id,
        mode="stream",
        result=result,
        elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
        success=success,
    )
