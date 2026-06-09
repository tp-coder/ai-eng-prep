import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.embeddings import EmbeddingClient
from app.logging_config import configure_logging
from app.index import SearchResult
from app.vector_store import QdrantVectorStore


console = Console()


@dataclass(frozen=True)
class RagEvalCase:
    id: str
    question: str
    should_find_relevant_context: bool
    expected_sources_contains: list[str]
    expected_top_score_min: float | None = None
    expected_top_score_max: float | None = None


@dataclass(frozen=True)
class RagEvalResult:
    id: str
    question: str
    passed: bool
    top_score: float | None
    filtered_count: int
    top_source: str | None
    failures: list[str]


def load_eval_cases(path: str) -> list[RagEvalCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    return [
        RagEvalCase(
            id=item["id"],
            question=item["question"],
            should_find_relevant_context=item["should_find_relevant_context"],
            expected_sources_contains=item.get(
                "expected_sources_contains", []),
            expected_top_score_min=item.get("expected_top_score_min"),
            expected_top_score_max=item.get("expected_top_score_max"),
        )
        for item in payload
    ]


def filter_results_by_score(
    results: list[SearchResult],
    min_score: float,
) -> list[SearchResult]:
    return [result for result in results if result.score >= min_score]


def sources_matches_expectations(
    source_path: str,
    expected_sources_contains: list[str],
) -> bool:
    return any(expected in source_path for expected in expected_sources_contains)


def evaluate_case(
    case: RagEvalCase,
    results: list[SearchResult],
    min_score: float,
) -> RagEvalResult:
    failures: list[str] = []

    filtered_results = filter_results_by_score(results, min_score=min_score)

    top_result = results[0] if results else None
    top_score = top_result.score if top_result else None
    top_source = top_result.chunk.source_path if top_result else None

    found_relevant_context = len(filtered_results) > 0

    if case.should_find_relevant_context and not found_relevant_context:
        failures.append(
            f"Expected relevant context but none passed the min threshold ({min_score:.4f})"
        )

    if not case.should_find_relevant_context and found_relevant_context:
        failures.append(
            f"Expected no relevant context but {len(filtered_results)} results passed the min threshold"
        )

    if case.expected_top_score_min is not None:
        if top_score is None or top_score < case.expected_top_score_min:
            failures.append(
                f"Expected top score >= {case.expected_top_score_min:.4f} but got {top_score:.4f}"
            )

    if case.expected_top_score_max is not None:
        if top_score is None or top_score >= case.expected_top_score_max:
            failures.append(
                f"Expected top score < {case.expected_top_score_max:.4f} but got {top_score:.4f}"
            )

    if case.expected_sources_contains and filtered_results:
        filtered_sources = [
            result.chunk.source_path for result in filtered_results]

        if not any(
            sources_matches_expectations(
                source, case.expected_sources_contains)
            for source in filtered_sources
        ):
            failures.append(
                f"Expected at least one filtered source to contain {case.expected_sources_contains} but got {filtered_sources}"
            )

    return RagEvalResult(
        id=case.id,
        question=case.question,
        passed=len(failures) == 0,
        top_score=top_score,
        filtered_count=len(filtered_results),
        top_source=top_source,
        failures=failures,
    )


def render_results(results: list[RagEvalResult]) -> None:
    table = Table(title="RAG Retrieval Eval Results", show_lines=True)
    table.add_column("Status", justify="center")
    table.add_column("Case")
    table.add_column("Top Score", justify="center")
    table.add_column("Filtered", justify="center")
    table.add_column("Top Source")
    table.add_column("Failures", justify="center")

    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        top_score = f"{result.top_score:.4f}" if result.top_score is not None else "N/A"
        failures = "\n".join(result.failures) if result.failures else "N/A"

        table.add_row(
            status,
            result.id,
            top_score,
            str(result.filtered_count),
            result.top_source or "N/A",
            failures,
        )

    console.print(table)


def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    eval_cases = load_eval_cases("evals/rag_dataset.json")
    store = QdrantVectorStore(settings)
    if store.count() == 0:
        console.print(
            f"[bold red]No vectors found in Qdrant collection '{settings.qdrant_collection}'[/bold red]\n"
            "Run: uv run python -m app.ingest"
        )
        raise SystemExit(1)

    embedding_client = EmbeddingClient(settings)
    questions = [case.question for case in eval_cases]
    query_embeddings = embedding_client.embed_texts(questions)

    results: list[RagEvalResult] = []
    for case, query_embedding in zip(eval_cases, query_embeddings, strict=True):
        search_results = store.search(
            query_embedding=query_embedding,
            top_k=settings.retrieval_top_k,
        )

        result = evaluate_case(
            case=case,
            results=search_results,
            min_score=settings.retrieval_min_score,
        )

        results.append(result)

    render_results(results)
    passed = sum(1 for result in results if result.passed)
    total = len(results)

    console.print(
        f"[bold green]Passed {passed} / {total} cases[/bold green]"
    )

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
