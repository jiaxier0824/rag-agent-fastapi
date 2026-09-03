from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    agent_model_name: str = "qwen-turbo"
    agent_fallback_model_name: str = "qwen-plus"

    rag_api_base_url: str = "http://127.0.0.1:8000"
    rag_api_timeout_seconds: float = 30.0
    rag_api_max_retries: int = 2
    rag_api_retry_interval_seconds: float = 1.0
    agent_observability_log_path: str = "./logs/agent_requests.jsonl"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_url: str = "redis://127.0.0.1:6379/0"
    rag_cache_ttl_seconds: int = 300

    study_plan_ttl_seconds: int = 2_592_000
    agent_short_memory_ttl_seconds: int = 604_800
    agent_short_memory_max_turns: int = 4
    agent_profile_ttl_seconds: int = 7_776_000
    agent_max_tool_calls: int = 6

settings = Settings()
