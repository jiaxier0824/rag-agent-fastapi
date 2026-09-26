"""Run each real case once, recording failures and timeouts without losing the sweep."""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from evaluation.run_evaluation import build_report, load_cases


RESULTS_PATH = Path(
    os.environ.get("EVALUATION_RESULTS_PATH", Path(__file__).with_name("results_latest.json"))
)
CASE_TIMEOUT_SECONDS = 90
RESULT_MARKER = "__EVAL_RESULT__"
CASE_COMMAND = (
    "import json, logging, sys; "
    "logging.disable(logging.CRITICAL); "
    "from dependencies import get_agent_service; "
    "from evaluation.run_evaluation import evaluate_case, load_cases; "
    "case = next(item for item in load_cases() if item['id'] == sys.argv[1]); "
    f"print('{RESULT_MARKER}' + json.dumps(evaluate_case(get_agent_service(), case), ensure_ascii=False), flush=True)"
)


def failed_result(case: dict, elapsed_seconds: float, error: str) -> dict:
    return {
        "id": case["id"],
        "category": case.get("category", "knowledge"),
        "question": case["question"],
        "answer": "",
        "statuses": [],
        "status_correct": False,
        "keyword_correct": False,
        "tool_correct": False,
        "setup_tools_called": [],
        "setup_correct": False,
        "tools_called": [],
        "blocked_tool_calls": [],
        "short_memory_turns_loaded": 0,
        "profile_memory_loaded": False,
        "profile_memory_correct": False,
        "sources": [],
        "source_correct": False,
        "case_pass": False,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "trace_id": None,
        "error": error,
    }


def run_case(case: dict) -> dict:
    started_at = perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-c", CASE_COMMAND, case["id"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=CASE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return failed_result(case, perf_counter() - started_at, "TIMEOUT")

    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line.removeprefix(RESULT_MARKER))
    print(output[-1000:], flush=True)
    return failed_result(case, perf_counter() - started_at, f"PROCESS_EXIT_{process.returncode}")


def main() -> None:
    cases = load_cases()
    results = []
    started_at = datetime.now(timezone.utc).isoformat()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        result = run_case(case)
        results.append(result)
        report = build_report(results)
        report.update({
            "complete": index == len(cases),
            "planned_cases": len(cases),
            "completed_cases": index,
            "started_at_utc": started_at,
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
        })
        RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {'PASS' if result['case_pass'] else 'FAIL'} {result['elapsed_seconds']}s", flush=True)

    print(f"完成：{report['summary']}\n结果：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
