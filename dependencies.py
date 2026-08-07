from functools import lru_cache

from agent.cache import RagCache
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
def get_rag_api_client() -> RagApiClient:
    return RagApiClient(
        cache=get_rag_cache(),
    )


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        rag_client=get_rag_api_client(),
        study_plan_store=get_study_plan_store(),
    )