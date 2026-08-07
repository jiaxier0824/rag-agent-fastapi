import json
from collections.abc import Iterator
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agent.service import AgentService
from dependencies import get_agent_service
from schemas.agent import AgentChatRequest, AgentChatResponse

router = APIRouter(
    prefix="/api/agent",
    tags=["Agent"],
)


@router.post("/chat", response_model=AgentChatResponse)
def chat_with_agent(
    request: AgentChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentChatResponse:
    trace_id = uuid4().hex
    answer = agent_service.execute(
        question=request.question,
        session_id=request.session_id,
        trace_id=trace_id,
    )

    return AgentChatResponse(
        answer=answer,
        session_id=request.session_id,
        trace_id=trace_id,
    )


@router.post("/chat/stream")
def stream_chat_with_agent(
    request: AgentChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    trace_id = uuid4().hex

    def event_generator() -> Iterator[str]:
        for event in agent_service.stream_execute(
            question=request.question,
            session_id=request.session_id,
            trace_id=trace_id,
        ):
            data = {
                "type": event["type"],
                "content": event["content"],
                "session_id": request.session_id,
                "trace_id": trace_id,
            }

            yield (
                "event: message\n"
                f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            )

        done_data = {
            "type": "done",
            "content": "",
            "session_id": request.session_id,
            "trace_id": trace_id,
        }

        yield (
            "event: message\n"
            f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
