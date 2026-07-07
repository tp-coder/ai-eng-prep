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
from app.pg_vector_store import PgVectorStore


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
    rank: int | None
    reciprocal_rank: float | None
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


def first_relevant_rank(
    case: RagEvalCase,
    results: list[SearchResult],
) -> int | None:
    # 1-indexed rank of the first retrieved result that matches the expected sources
    for rank, result in enumerate(results, start=1):
        if sources_matches_expectations(result.chunk.source_path, case.expected_sources_contains):
            return rank

    return None


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

    rank = first_relevant_rank(
        case, results) if case.should_find_relevant_context else None
    reciprocal_rank = (1.0 / rank) if rank else 0.0

    return RagEvalResult(
        id=case.id,
        question=case.question,
        passed=len(failures) == 0,
        top_score=top_score,
        filtered_count=len(filtered_results),
        top_source=top_source,
        rank=rank,
        reciprocal_rank=reciprocal_rank,
        failures=failures,
    )


def render_results(results: list[RagEvalResult]) -> None:
    table = Table(title="RAG Retrieval Eval Results", show_lines=True)
    table.add_column("Status", justify="center")
    table.add_column("Case")
    table.add_column("Top Score", justify="center")
    table.add_column("Filtered", justify="center")
    table.add_column("Top Source")
    table.add_column("Rank", justify="center")
    table.add_column("Reciprocal Rank", justify="center")
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
            str(result.rank) if result.rank is not None else "N/A",
            f"{result.reciprocal_rank:.4f}",
            failures,
        )

    console.print(table)


def retrieval_metrics(
    cases: list[RagEvalCase],
    results: list[RagEvalResult],
    k: int,
) -> dict[str, float]:
    relevant = [result for case, result in zip(
        cases, results, strict=True) if case.should_find_relevant_context]
    relevant_count = len(relevant)

    if relevant_count == 0:
        return {}

    return {
        "MRR": sum(relevance.reciprocal_rank for relevance in relevant) / relevant_count,
        "Hit@1": sum(1 for result in relevant if result.rank == 1) / relevant_count,
        f"Hit@{k}": sum(1 for result in relevant if result.rank is not None and result.rank <= k) / relevant_count,
        "relevant_count": relevant_count,
    }


def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    eval_cases = load_eval_cases("evals/rag_dataset.json")
    store = PgVectorStore(settings)
    if store.count() == 0:
        console.print(
            f"[bold red]No vectors found in pgvector table '{settings.pgvector_table}'[/bold red]\n"
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

    metrics = retrieval_metrics(
        eval_cases, results, k=settings.retrieval_top_k)
    if metrics:
        k = settings.retrieval_top_k
        console.print(
            f"[bold]Retrieval metrics[/bold] (over {int(metrics['relevant_count'])} relevant cases, top_k={k}) "
            f"MRR={metrics['MRR']:.4f}, Hit@1={metrics['Hit@1']:.4f}, Hit@{k}={metrics[f'Hit@{k}']:.4f}"
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
