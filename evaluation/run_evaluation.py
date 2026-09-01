import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent.service import AgentService
from dependencies import get_agent_service


EVALUATION_DIR = Path(__file__).parent
CASES_PATH = EVALUATION_DIR / "cases.json"
RESULTS_PATH = EVALUATION_DIR / "results.json"


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(
    agent_service: AgentService,
    case: dict,
) -> dict:
    session_id = f"evaluation-{case['id']}-{uuid4().hex}"
    trace_id = uuid4().hex
    started_at = perf_counter()

    events = list(
        agent_service.stream_execute(
            question=case["question"],
            session_id=session_id,
            trace_id=trace_id,
        )
    )

    elapsed_seconds = round(perf_counter() - started_at, 3)
    answer = "".join(
        event["content"]
        for event in events
        if event["type"] == "content"
    )
    statuses = [
        event["content"]
        for event in events
        if event["type"] == "status"
    ]
    metadata = next(
        (
            event
            for event in events
            if event["type"] == "metadata"
        ),
        {"sources": []},
    )
    sources = metadata["sources"]

    normalized_answer = normalize_text(answer)
    keyword_correct = all(
        normalize_text(keyword) in normalized_answer
        for keyword in case["expected_keywords"]
    )
    tool_correct = case["expected_status"] in statuses
    expected_sources = case.get("expected_sources", [])
    source_correct = all(source in sources for source in expected_sources)

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "statuses": statuses,
        "keyword_correct": keyword_correct,
        "tool_correct": tool_correct,
        "sources": sources,
        "source_correct": source_correct,
        "elapsed_seconds": elapsed_seconds,
        "trace_id": trace_id,
    }


def build_report(results: list[dict]) -> dict:
    total_cases = len(results)
    keyword_correct_cases = sum(result["keyword_correct"] for result in results)
    tool_correct_cases = sum(result["tool_correct"] for result in results)
    source_correct_cases = sum(result["source_correct"] for result in results)
    average_latency_seconds = round(
        sum(result["elapsed_seconds"] for result in results) / total_cases,
        3,
    ) if total_cases else 0.0

    return {
        "summary": {
            "total_cases": total_cases,
            "keyword_accuracy": round(keyword_correct_cases / total_cases, 3) if total_cases else 0.0,
            "tool_selection_accuracy": round(tool_correct_cases / total_cases, 3) if total_cases else 0.0,
            "source_attribution_accuracy": round(source_correct_cases / total_cases, 3) if total_cases else 0.0,
            "average_latency_seconds": average_latency_seconds,
        },
        "results": results,
    }


def main() -> None:
    cases = load_cases()
    agent_service = get_agent_service()
    results = []

    for case in cases:
        print(f"正在评测：{case['id']}")
        results.append(evaluate_case(agent_service, case))

    report = build_report(results)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    summary = report["summary"]
    print("\n评测完成")
    print(f"题目数量：{summary['total_cases']}")
    print(f"关键事实正确率：{summary['keyword_accuracy']:.1%}")
    print(f"工具选择正确率：{summary['tool_selection_accuracy']:.1%}")
    print(f"来源透传正确率：{summary['source_attribution_accuracy']:.1%}")
    print(f"平均响应时间：{summary['average_latency_seconds']} 秒")
    print(f"详细结果：{RESULTS_PATH}")


if __name__ == "__main__":
    main()
