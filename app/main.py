import argparse
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.embeddings import EmbeddingClient
from app.config import get_settings
from app.llm import LLMClient, LLMConfigurationError, LLMResponseParsingError
from app.logging_config import configure_logging
from app.index import SearchResult, load_index, search_index
from app.schemas import AssistantResponse


console = Console()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Engineering Prep CLI")

    parser.add_argument("prompt", nargs="?",
                        help="Prompt to send to the configured LLM provider")

    parser.add_argument(
        "--allow-llm-general",
        action="store_true",
        help="Allow the LLM to generate content that is not in the retrieved context"
    )

    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip retrieval-augmented generation (RAG) and call the LLM without local document context"
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks before calling the LLM"
    )

    return parser.parse_args()


def render_items(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "none"


def format_retrieved_context(results: list[SearchResult]) -> str:
    blocks: list[str] = []

    for result in results:
        source_label = f"{result.chunk.source_path}::chunk-{result.chunk.chunk_index}"
        blocks.append(
            f"""
            Source: {source_label}
            Similarity score: {result.score:.4f}
            Context: {result.chunk.text}
            """.strip()
        )

    return "\n\n---\n\n".join(blocks)


def filter_results_by_score(
    results: list[SearchResult],
    min_score: float,
) -> list[SearchResult]:
    return [result for result in results if result.score >= min_score]


def render_retrieval_debug(results: list[SearchResult]) -> None:
    table = Table(title="Retrieved context", show_lines=True)
    table.add_column("Score", justify="right")
    table.add_column("source")
    table.add_column("Preview")

    for result in results:
        source_label = f"{result.chunk.source_path}::chunk-{result.chunk.chunk_index}"
        preview = result.chunk.text[:240] + \
            ("..." if len(result.chunk.text) > 240 else "")
        table.add_row(f"{result.score:.4f}", source_label, preview)

    console.print(table)


def build_no_context_response(
    min_score: float,
    top_score: float | None,
) -> AssistantResponse:
    top_score_text = f"{top_score:.4f}" if top_score is not None else "none"

    return AssistantResponse(
        answer=(
            "I could not answer your question from the local document index because no review chunks passed the minimum relevance threshold."
        ),
        confidence="high",
        missing_context=[
            "No local document chunk matched the question strongly enough.",
            f"Retrieval minimum score is {min_score:.4f}",
            f"Highest retrieved score was {top_score_text}.",
        ],
        next_actions=[
            "Add relevant documents to data/docs and re-run ingestion process.",
            "Lower RETRIEVAL_MIN_SCORE if you want lighter context threshold - but this can produce false positives.",
            "Use --allow-llm-general to allow the LLM to generate content that is not in the retrieved context.",
            "Use --no-rag if you want to bypass retrieval completely."
        ],
        source_references=[],
    )


def render_assistant_response(
    parsed: AssistantResponse,
    model: str,
    latency_ms: int,
) -> None:
    table = Table(show_header=False, box=None)
    table.add_row("[bold]Answer[/bold]", parsed.answer)
    table.add_row("[bold]Confidence[/bold]", parsed.confidence)
    table.add_row("[bold]Missing context[/bold]",
                  render_items(parsed.missing_context))
    table.add_row("[bold]Next actions[/bold]",
                  render_items(parsed.next_actions))
    table.add_row("[bold]Source references[/bold]",
                  render_items(parsed.source_references))

    console.print(
        Panel.fit(
            table,
            title=f"{model} . {latency_ms}ms",
            border_style="green",
        )
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings)

    if not args.prompt:
        console.print("[bold green]AI Engineering Prep ready.[/bold green]")
        console.print(f"App: {settings.app_name}")
        console.print(f"Environment: {settings.app_env}")
        console.print(f"Log level: {settings.log_level}")
        console.print(f"LLM model: {settings.openai_model}")
        console.print(f"Embedding model: {settings.openai_embedding_model}")
        console.print(f"Retrieval top K: {settings.retrieval_top_k}")
        console.print(f"Retrieval min score: {settings.retrieval_min_score}")

        return

    retrieved_context: str | None = None

    try:
        if not args.no_rag:
            indexed_chunks = load_index(settings.index_path)

            if indexed_chunks:
                embedding_client = EmbeddingClient(settings)
                query_embedding = embedding_client.embed_texts([args.prompt])[
                    0]

                raw_results = search_index(
                    indexed_chunks=indexed_chunks,
                    query_embedding=query_embedding,
                    top_k=settings.retrieval_top_k,
                )

                results = filter_results_by_score(
                    results=raw_results,
                    min_score=settings.retrieval_min_score,
                )

                top_score = raw_results[0].score if raw_results else None

                logger.info(
                    "retrieval_completed raw_result_count=%s filtered_result_count=%s top_score=%s min_score=%s",
                    len(raw_results),
                    len(results),
                    top_score,
                    settings.retrieval_min_score,
                )

                if args.show_context and raw_results:
                    render_retrieval_debug(raw_results)

                if results:
                    retrieved_context = format_retrieved_context(results)
                else:
                    logger.warning(
                        "retrieval_context_skipped reason=no_results_above_threshold min_score=%s",
                        settings.retrieval_min_score,
                    )

                    if not args.allow_llm_general:
                        parsed = build_no_context_response(
                            min_score=settings.retrieval_min_score,
                            top_score=top_score,
                        )
                        render_assistant_response(
                            parsed=parsed,
                            model='local-rag-guard',
                            latency_ms=0,
                        )
                        return

            else:
                logger.warning(
                    "retrieval_skipped reason=index_not_found_or_empty")

                if not args.allow_llm_general:
                    parsed = AssistantResponse(
                        answer="I could not answer your question because the local document index is empty.",
                        confidence="high",
                        missing_context=[
                            f"No usable index was found at {settings.index_path}",
                        ],
                        next_actions=[
                            "Add documents to data/docs.",
                            "Run uv run python -m app.ingest.",
                            "Use --allow-llm-general to deliberately allow an LLM non-grounded answer."
                        ],
                        source_references=[],
                    )
                    return

        llm = LLMClient(settings)
        response = llm.complete(
            args.prompt, retrieved_context=retrieved_context)

    except LLMConfigurationError as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        raise SystemExit(1)

    except LLMResponseParsingError as error:
        console.print(f"[bold red]Response parsing error:[/bold red] {error}")
        raise SystemExit(1)

    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted by user. [/bold yellow]")
        raise SystemExit(130)

    except Exception as error:
        console.print(f"[bold red]LLM call error:[/bold red] {error}")
        raise SystemExit(1)

    render_assistant_response(
        parsed=response.parsed,
        model=response.model,
        latency_ms=response.latency_ms,
    )


if __name__ == "__main__":
    main()
