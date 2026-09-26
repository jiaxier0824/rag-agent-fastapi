"""偏好保存请求的识别与写入结果确认。"""

from agent.execution import AgentExecutionContext


def preference_save_requested(question: str) -> bool:
    """只识别明确要求保存长期学习偏好的请求，不影响学习计划。"""
    if any(phrase in question for phrase in ("不要保存", "不保存", "无需保存", "别保存")):
        return False
    asks_to_save = any(phrase in question for phrase in ("记住", "保存"))
    mentions_preference = any(
        phrase in question for phrase in ("偏好", "每天学习", "每日学习", "复习缓冲")
    )
    is_plan_only = "学习计划" in question and "偏好" not in question and "记住" not in question
    return asks_to_save and mentions_preference and not is_plan_only


def confirmed_preference_answer(
    question: str, answer: str, execution_context: AgentExecutionContext,
) -> str:
    if preference_save_requested(question) and not execution_context.profile_save_succeeded:
        return "学习偏好尚未保存，请稍后重试。"
    return answer
