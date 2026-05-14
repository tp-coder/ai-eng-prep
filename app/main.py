import argparse
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from app import embeddings
from app.config import get_settings
from app.llm import LLMClient, LLMConfigurationError, LLMResponseParsingError
from app.logging_config import configure_logging
from app.index import SearchResult, load_index, search_index
from app.embeddings import EmbeddingClient


console = Console()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Engineering Prep CLI")
    parser.add_argument("prompt", nargs="?",
                        help="Prompt to send to the configured LLM provider")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip retrieval-augmented generation (RAG) and call the LLM without local document context"
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

        return

    retrieved_context: str | None = None

    try:
        if not args.no_rag:
            indexed_chunks = load_index(settings.index_path)
            if indexed_chunks:
                embedding_client = EmbeddingClient(settings)
                query_embedding = embedding_client.embed_texts([args.prompt])[
                    0]
                results = search_index(
                    indexed_chunks=indexed_chunks,
                    query_embedding=query_embedding,
                    top_k=settings.retrieval_top_k,
                )

                logger.info(
                    "retrieval_completed result_count=%s top_score=%s",
                    len(results),
                    results[0].score if results else None,
                )

                retrieved_context = format_retrieved_context(results)

            else:
                logger.warning(
                    "retrieval_skipped reason=index_not_found_or_empty")

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
    except Exception as error:
        console.print(f"[bold red]LLM call error:[/bold red] {error}")
        raise SystemExit(1)

    parsed = response.parsed

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
            title=f"{response.model} - {response.latency_ms}ms",
            border_style="green"
        )
    )


if __name__ == "__main__":
    main()
