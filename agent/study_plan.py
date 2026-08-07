from datetime import date, datetime

def build_study_plan(
    task: str,
    deadline: str,
    daily_study_hours: float = 2.0,
) -> str:
    try:
        deadline_date = datetime.strptime(
            deadline,
            "%Y-%m-%d",
        ).date()
    except ValueError:
        return "截止日期格式错误，请使用 YYYY-MM-DD，例如 2026-08-20。"

    if daily_study_hours <= 0:
        return "每日学习时长必须大于 0。"

    remaining_days = (deadline_date - date.today()).days

    if remaining_days <= 0:
        return "截止日期已到或已过，无法生成新的学习计划。"

    learning_days = max(1, int(remaining_days * 0.5))
    practice_days = max(1, int(remaining_days * 0.3))
    review_days = remaining_days - learning_days - practice_days

    return (
        f"任务：{task}\n"
        f"截止日期：{deadline}\n"
        f"剩余天数：{remaining_days} 天\n"
        f"每日学习时长：{daily_study_hours} 小时\n\n"
        f"第 1-{learning_days} 天：学习核心知识并整理笔记。\n"
        f"第 {learning_days + 1}-{learning_days + practice_days} 天：完成练习和任务初稿。\n"
        f"最后 {review_days} 天：检查、修改并预留提交缓冲时间。"
    )
