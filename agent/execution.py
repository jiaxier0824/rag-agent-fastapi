"""Agent 单次请求的运行时上下文与结果对象。"""

from dataclasses import dataclass, field
import json


@dataclass
class AgentExecutionContext:
    """仅属于一次 Agent 调用，不能放到全局单例中。"""

    tools_called: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rag_trace_ids: list[str] = field(default_factory=list)
    rag_cache_hit: bool | None = None
    max_tool_calls: int = 6
    blocked_tool_calls: list[dict[str, str]] = field(default_factory=list)
    short_memory_turns_loaded: int = 0
    profile_memory_loaded: bool = False
    profile_save_succeeded: bool = False
    _tool_signatures: set[str] = field(default_factory=set, repr=False)

    def record_tool(self, tool_name: str) -> None:
        self.tools_called.append(tool_name)

    def begin_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> str | None:
        """记录一次允许的工具调用；被保护规则拦截时返回错误码。"""
        signature = json.dumps(
            {"tool": tool_name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if signature in self._tool_signatures:
            self.blocked_tool_calls.append(
                {"tool": tool_name, "code": "DUPLICATE_TOOL_CALL"}
            )
            return "DUPLICATE_TOOL_CALL"
        if len(self.tools_called) >= self.max_tool_calls:
            self.blocked_tool_calls.append(
                {"tool": tool_name, "code": "TOOL_CALL_LIMIT_EXCEEDED"}
            )
            return "TOOL_CALL_LIMIT_EXCEEDED"

        self._tool_signatures.add(signature)
        self.record_tool(tool_name)
        return None

    def record_memory(self, *, short_memory_turns: int, profile_loaded: bool) -> None:
        self.short_memory_turns_loaded = short_memory_turns
        self.profile_memory_loaded = profile_loaded

    def record_rag_result(
        self,
        *,
        sources: list[str],
        rag_trace_id: str | None,
        cache_hit: bool,
    ) -> None:
        self.sources = list(dict.fromkeys([*self.sources, *sources]))
        if rag_trace_id and rag_trace_id not in self.rag_trace_ids:
            self.rag_trace_ids.append(rag_trace_id)
        self.rag_cache_hit = cache_hit


@dataclass(frozen=True)
class AgentExecutionResult:
    answer: str
    sources: list[str]
    rag_trace_ids: list[str]
    tools_called: list[str]
    rag_cache_hit: bool | None
    model_used: str
    degraded: bool
    blocked_tool_calls: list[dict[str, str]] = field(default_factory=list)
    short_memory_turns_loaded: int = 0
    profile_memory_loaded: bool = False
