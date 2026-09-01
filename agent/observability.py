"""Agent 调用链的本地 JSONL 日志。"""

import json
from datetime import datetime, timezone
from pathlib import Path


class AgentTraceLogger:
    """每次请求写入一行，不记录 API Key、问题正文和回答正文。"""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)

    def write(self, event: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
