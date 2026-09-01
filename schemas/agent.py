from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    filename: str = Field(
        description="本次 Agent 回答使用的 RAG 来源文件名",
    )

class AgentChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="用户问题",
    )

    session_id: str = Field(
        default="user_001",
        description="用户会话编号",
    )

class AgentChatResponse(BaseModel):
    answer: str = Field(
        description="Agent 最终回答",
    )

    session_id: str = Field(
        description="用户会话编号",
    )

    trace_id: str = Field(
        description="本次 Agent 请求追踪号"
    )

    sources: list[SourceReference] = Field(
        default_factory=list,
        description="本次回答使用的知识库来源",
    )
