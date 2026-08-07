import logging
from collections.abc import Callable, Iterator
from queue import Queue
from threading import Thread

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi

from agent.rag_client import RagApiClient
from agent.tools import build_tools
from config.settings import settings
from agent.study_plan_store import StudyPlanStore
logger = logging.getLogger(__name__)

class AgentService:
    def __init__(
        self,
        rag_client: RagApiClient,
        study_plan_store: StudyPlanStore,
        model_name: str = settings.agent_model_name,
        fallback_model_name: str = settings.agent_fallback_model_name,
    ):
        self.rag_client = rag_client
        self.study_plan_store = study_plan_store
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name

        self.model = ChatTongyi(model=model_name)
        self.fallback_model = ChatTongyi(
            model=fallback_model_name,
        )

    def _build_agent(
        self,
        session_id: str,
        trace_id: str,
        model: ChatTongyi,
        on_status: Callable[[str], None] | None = None,
    ):
        return create_agent(
            model=model,
            tools=build_tools(
                rag_client=self.rag_client,
                session_id=session_id,
                trace_id=trace_id,
                study_plan_store=self.study_plan_store,
                on_status=on_status,
            ),
            system_prompt=(
                "你是 UQ 学习助手。"
                "涉及课程、作业、截止日期或上传资料的事实问题时，"
                "必须调用 search_course_knowledge。"
                "当用户要求生成学习计划时，如果任务或截止日期需要从课程资料确认，"
                "先调用 search_course_knowledge，再调用 create_study_plan。"
                "调用 create_study_plan 时，deadline 必须使用 YYYY-MM-DD 格式。"
                "不要编造工具没有返回的课程信息。"
                "当用户询问当前、之前或已保存的学习计划时，"
                "必须调用 get_current_study_plan。"
            ),
        )

    def _invoke_agent(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> str:
        agent = self._build_agent(
            session_id=session_id,
            trace_id=trace_id,
            model=model,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )
        return result["messages"][-1].content

    def execute(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> str:
        logger.info(
            "Agent 使用主模型开始处理：model=%s, trace_id=%s, session_id=%s",
            self.model_name,
            trace_id,
            session_id,
        )

        try:
            answer = self._invoke_agent(
                model=self.model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
            )
        except Exception:
            logger.warning(
                "主模型调用失败，切换备用模型：primary=%s, fallback=%s, trace_id=%s",
                self.model_name,
                self.fallback_model_name,
                trace_id,
                exc_info=True,
            )
            answer = self._invoke_agent(
                model=self.fallback_model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
            )

        logger.info(
            "Agent 处理完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

        return answer

    def stream_execute(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> Iterator[dict[str, str]]:
        logger.info(
            "Agent 使用主模型开始流式处理：model=%s, trace_id=%s, session_id=%s",
            self.model_name,
            trace_id,
            session_id,
        )

        yield {
            "type": "status",
            "content": "正在分析你的问题",
        }

        has_emitted_content = False

        try:
            for event in self._stream_model_events(
                model=self.model,
                question=question,
                session_id=session_id,
                trace_id=trace_id,
            ):
                if event["type"] == "content" and not has_emitted_content:
                    yield {
                        "type": "status",
                        "content": "正在生成回答",
                    }
                    has_emitted_content = True

                yield event

        except Exception:
            if has_emitted_content:
                logger.exception(
                    "主模型流式输出中断，无法切换备用模型以避免重复内容：trace_id=%s",
                    trace_id,
                )
                yield {
                    "type": "error",
                    "content": "回答生成中断，请重新提问。",
                }
                return

            logger.warning(
                "主模型流式调用失败，切换备用模型：primary=%s, fallback=%s, trace_id=%s",
                self.model_name,
                self.fallback_model_name,
                trace_id,
                exc_info=True,
            )

            yield {
                "type": "status",
                "content": "主模型暂时不可用，正在切换备用模型",
            }

            yield {
                "type": "status",
                "content": "正在使用备用模型生成回答",
            }

            try:
                for event in self._stream_model_events(
                    model=self.fallback_model,
                    question=question,
                    session_id=session_id,
                    trace_id=trace_id,
                ):
                    yield event
            except Exception:
                logger.exception(
                    "备用模型流式调用也失败：trace_id=%s",
                    trace_id,
                )
                yield {
                    "type": "error",
                    "content": "当前模型服务暂时不可用，请稍后重试。",
                }
                return

        logger.info(
            "Agent 流式处理完成：trace_id=%s, session_id=%s",
            trace_id,
            session_id,
        )

    def _stream_agent(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
        on_status: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        agent = self._build_agent(
            session_id=session_id,
            trace_id=trace_id,
            model=model,
            on_status=on_status,
        )

        for message, metadata in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            stream_mode="messages",
        ):

            if (
                metadata["langgraph_node"] == "model"
                and isinstance(message.content, str)
                and message.content
            ):
                yield message.content

    def _stream_model_events(
        self,
        model: ChatTongyi,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> Iterator[dict[str, str]]:
        event_queue = Queue()
        stream_finished = object()

        def publish_status(content: str) -> None:
            event_queue.put({
                "type": "status",
                "content": content,
            })

        def run_agent_stream() -> None:
            try:
                for content in self._stream_agent(
                    model=model,
                    question=question,
                    session_id=session_id,
                    trace_id=trace_id,
                    on_status=publish_status,
                ):
                    event_queue.put({
                        "type": "content",
                        "content": content,
                    })
            except Exception as error:
                event_queue.put(error)
            finally:
                event_queue.put(stream_finished)

        Thread(target=run_agent_stream, daemon=True).start()

        while True:
            event = event_queue.get()

            if event is stream_finished:
                return

            if isinstance(event, Exception):
                raise event

            yield event
