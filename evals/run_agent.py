import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.logging_config import configure_logging
from app.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMResponseParsingError,
    complete_with_tools,
)


console = Console()


@dataclass(frozen=True)
class AgentEvalCase:
    id: str
    prompt: str
    expected_tools: list[str]


@dataclass(frozen=True)
class AgentEvalResult:
    id: str
    prompt: str
    passed: bool
    expected_tools: list[str]
    actual_tools: list[str]
    latency_ms: int | None
    answer_preview: str | None
    failures: list[str]


def load_eval_cases(path: str) -> list[AgentEvalCase]:
    payload: list[dict[str, Any]] = json.loads(
        Path(path).read_text(encoding="utf-8"))
    return [
        AgentEvalCase(
            id=item["id"],
            prompt=item["prompt"],
            expected_tools=item.get("expected_tools", []),
        )
        for item in payload
    ]


def tool_count(tools: list[str]) -> Counter[str]:
    return Counter(tools)


def evaluate_case(llm: LLMClient, case: AgentEvalCase) -> AgentEvalResult:
    failures: list[str] = []

    try:
        response = complete_with_tools(llm, case.prompt)
    except (LLMConfigurationError, LLMResponseParsingError):
        raise
    except Exception as error:
        return AgentEvalResult(
            id=case.id,
            prompt=case.prompt,
            passed=False,
            expected_tools=case.expected_tools,
            actual_tools=[],
            latency_ms=None,
            answer_preview=None,
            failures=[f"Agent call failed: {error}"],
        )

    actual_tools = response.tool_names
    expected_tool_count = tool_count(case.expected_tools)
    actual_tool_count = tool_count(actual_tools)

    if actual_tool_count != expected_tool_count:
        missing_tools = list(
            (expected_tool_count - actual_tool_count).elements())
        unexpected_tools = list(
            (actual_tool_count - expected_tool_count).elements())

        if missing_tools:
            failures.append(f"Missing expected tools: {missing_tools}")
        if unexpected_tools:
            failures.append(f"Unexpected tools: {unexpected_tools}")
        if not missing_tools and not unexpected_tools:
            failures.append(
                f"Tool count mismatch: expected {expected_tool_count} and got {actual_tool_count}"
            )

    if not response.parsed.answer.strip():
        failures.append("Answer was empty")

    answer_preview = response.parsed.answer.replace("\n", " ")[:120]

    return AgentEvalResult(
        id=case.id,
        prompt=case.prompt,
        passed=len(failures) == 0,
        expected_tools=case.expected_tools,
        actual_tools=actual_tools,
        latency_ms=response.latency_ms,
        answer_preview=answer_preview,
        failures=failures,
    )


def render_results(results: list[AgentEvalResult]) -> None:
    table = Table(title="Agent Tool Eval Results", show_lines=True)
    table.add_column("Status", justify="center")
    table.add_column("Case")
    table.add_column("Expected Tools")
    table.add_column("Actual Tools")
    table.add_column("Latency", justify="center")
    table.add_column("Answer Preview")
    table.add_column("Failures")

    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        latency = f"{result.latency_ms} ms" if result.latency_ms is not None else "N/A"
        failures = "\n".join(result.failures) if result.failures else "N/A"

        table.add_row(
            status,
            result.id,
            ", ".join(result.expected_tools),
            ", ".join(result.actual_tools),
            latency,
            result.answer_preview,
            failures,
        )

    console.print(table)


def agent_metrics(results: list[AgentEvalResult]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {}

    passed = sum(1 for result in results if result.passed)
    expected_tool_calls = sum(len(result.expected_tools) for result in results)
    actual_tool_calls = sum(len(result.actual_tools) for result in results)

    matched_tool_calls = 0
    for result in results:
        expected_count = tool_count(result.expected_tools)
        actual_count = tool_count(result.actual_tools)
        matched_tool_calls += sum((expected_count & actual_count).values())

    precision = (
        matched_tool_calls / actual_tool_calls if actual_tool_calls > 0 else 1.0
    )
    recall = (
        matched_tool_calls / expected_tool_calls if expected_tool_calls > 0 else 1.0
    )

    return {
        "accuracy": passed / total,
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    eval_cases = load_eval_cases("evals/agent_dataset.json")
    llm = LLMClient(settings)
    results = [evaluate_case(llm, case) for case in eval_cases]

    render_results(results)

    metrics = agent_metrics(results)
    if metrics:
        console.print(
            "[bold]Agent metrics[/bold] "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Tool precision={metrics['precision']:.4f}, "
            f"Tool recall={metrics['recall']:.4f}"
        )

    passed = sum(1 for result in results if result.passed)
    total = len(results)

    console.print(
        f"[bold green]Passed {passed} / {total} cases[/bold green]"
    )

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
