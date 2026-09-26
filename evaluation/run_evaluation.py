"""Shared case loading and scoring for the isolated final evaluation runner."""

import json
import re
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent.service import AgentService


EVALUATION_DIR = Path(__file__).parent
CASES_PATH = EVALUATION_DIR / "cases.json"


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def contains_expected(answer: str, option: str) -> bool:
    """英文和数字按完整词匹配，避免把课程号 INFS7410 当成答案 10。"""
    if option.isascii() and any(character.isalnum() for character in option):
        pattern = re.escape(option.strip().lower()).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", answer.lower()) is not None
    return normalize_text(option) in normalize_text(answer)


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(
    agent_service: AgentService,
    case: dict,
) -> dict:
    session_id = f"evaluation-{case['id']}-{uuid4().hex}"
    trace_id = uuid4().hex

    deadline = (date.today() + timedelta(days=14)).isoformat()
    setup_tools: list[str] = []
    if setup_question := case.get("setup_question"):
        setup_events = list(
            agent_service.stream_execute(
                question=setup_question.replace("{deadline}", deadline),
                session_id=session_id,
                trace_id=uuid4().hex,
            )
        )
        setup_tools = next(
            (event.get("tools_called", []) for event in setup_events if event["type"] == "metadata"),
            [],
        )

    question = case["question"].replace("{deadline}", deadline)
    started_at = perf_counter()

    events = list(
        agent_service.stream_execute(
            question=question,
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
        {"sources": [], "tools_called": [], "blocked_tool_calls": []},
    )
    sources = metadata["sources"]

    keyword_correct = all(
        any(contains_expected(answer, option) for option in (term if isinstance(term, list) else [term]))
        for term in case.get("expected_keywords", [])
    )
    expected_tools = case.get("expected_tools", [])
    tools_called = metadata.get("tools_called", [])
    expected_status = case.get("expected_status")
    status_correct = (
        expected_status is None
        or any(
            status == expected_status
            or expected_status.removeprefix("正在") in status
            for status in statuses
        )
    )
    tool_correct = sorted(tools_called) == sorted(expected_tools)
    setup_correct = sorted(setup_tools) == sorted(case.get("expected_setup_tools", []))
    expected_sources = case.get("expected_sources", [])
    source_correct = all(source in sources for source in expected_sources)
    profile_memory_loaded = metadata.get("profile_memory_loaded", False)
    profile_memory_correct = (
        "expected_profile_memory_loaded" not in case
        or profile_memory_loaded == case["expected_profile_memory_loaded"]
    )

    return {
        "id": case["id"],
        "category": case.get("category", "knowledge"),
        "question": question,
        "answer": answer,
        "statuses": statuses,
        "status_correct": status_correct,
        "keyword_correct": keyword_correct,
        "tool_correct": tool_correct,
        "setup_tools_called": setup_tools,
        "setup_correct": setup_correct,
        "tools_called": tools_called,
        "blocked_tool_calls": metadata.get("blocked_tool_calls", []),
        "short_memory_turns_loaded": metadata.get("short_memory_turns_loaded", 0),
        "profile_memory_loaded": profile_memory_loaded,
        "profile_memory_correct": profile_memory_correct,
        "sources": sources,
        "source_correct": source_correct,
        "case_pass": keyword_correct and tool_correct and setup_correct and source_correct and status_correct and profile_memory_correct,
        "elapsed_seconds": elapsed_seconds,
        "trace_id": trace_id,
    }


def build_report(results: list[dict]) -> dict:
    total_cases = len(results)
    answer_cases = [result for result in results if result["category"] == "knowledge"]
    keyword_correct_cases = sum(result["keyword_correct"] for result in answer_cases)
    tool_correct_cases = sum(result["tool_correct"] for result in results)
    source_correct_cases = sum(result["source_correct"] for result in answer_cases)
    average_latency_seconds = round(
        sum(result["elapsed_seconds"] for result in results) / total_cases,
        3,
    ) if total_cases else 0.0

    return {
        "summary": {
            "total_cases": total_cases,
            "knowledge_cases": len(answer_cases),
            "tool_cases": total_cases - len(answer_cases),
            "keyword_accuracy": round(keyword_correct_cases / len(answer_cases), 3) if answer_cases else 0.0,
            "tool_selection_accuracy": round(tool_correct_cases / total_cases, 3) if total_cases else 0.0,
            "source_attribution_accuracy": round(source_correct_cases / len(answer_cases), 3) if answer_cases else 0.0,
            "case_pass_rate": round(sum(result["case_pass"] for result in results) / total_cases, 3) if total_cases else 0.0,
            "average_latency_seconds": average_latency_seconds,
        },
        "results": results,
    }
