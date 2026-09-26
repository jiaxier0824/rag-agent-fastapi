from functools import lru_cache

from agent.cache import RagCache
from agent.memory import AgentMemoryStore
from agent.observability import AgentTraceLogger
from agent.rag_client import RagApiClient
from agent.service import AgentService
from agent.study_plan_store import StudyPlanStore


@lru_cache
def get_rag_cache() -> RagCache:
    return RagCache()


@lru_cache
def get_study_plan_store() -> StudyPlanStore:
    return StudyPlanStore()


@lru_cache
def get_agent_memory_store() -> AgentMemoryStore:
    return AgentMemoryStore()


@lru_cache
def get_rag_api_client() -> RagApiClient:
    return RagApiClient(
        cache=get_rag_cache(),
    )


@lru_cache
def get_agent_trace_logger() -> AgentTraceLogger:
    from config.settings import settings

    return AgentTraceLogger(settings.agent_observability_log_path)


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        memory_store=get_agent_memory_store(),
        trace_logger=get_agent_trace_logger(),
    )
