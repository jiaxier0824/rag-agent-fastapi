"""把后台 LangGraph 同步迭代器桥接为可发送的状态/内容事件。"""

from collections.abc import Callable, Iterator
from queue import Queue
from threading import Thread


def stream_model_events(
    run_stream: Callable[[Callable[[str], None]], Iterator[str]],
) -> Iterator[dict[str, object]]:
    event_queue: Queue[object] = Queue()
    stream_finished = object()

    def publish_status(content: str) -> None:
        event_queue.put({"type": "status", "content": content})

    def run() -> None:
        try:
            for content in run_stream(publish_status):
                event_queue.put({"type": "content", "content": content})
        except Exception as error:
            event_queue.put(error)
        finally:
            event_queue.put(stream_finished)

    Thread(target=run, daemon=True).start()
    while True:
        event = event_queue.get()
        if event is stream_finished:
            return
        if isinstance(event, Exception):
            raise event
        yield event
