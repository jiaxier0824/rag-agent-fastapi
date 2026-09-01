"""Agent 单次请求的运行时上下文与结果对象。"""

from dataclasses import dataclass, field


@dataclass
class AgentExecutionContext:
    """仅属于一次 Agent 调用，不能放到全局单例中。"""

    tools_called: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rag_trace_ids: list[str] = field(default_factory=list)
    rag_cache_hit: bool | None = None

    def record_tool(self, tool_name: str) -> None:
        self.tools_called.append(tool_name)

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
