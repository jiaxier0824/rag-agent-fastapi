import logging

from fastapi import FastAPI

from routers.agent import router as agent_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


app = FastAPI(
    title="RAG Agent FastAPI",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(agent_router)