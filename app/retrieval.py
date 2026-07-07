from app.config import Settings, get_settings
from app.embeddings import EmbeddingClient
from app.index import SearchResult
from app.pg_vector_store import PgVectorStore


def filter_results_by_score(
    results: list[SearchResult],
    min_score: float,
) -> list[SearchResult]:
    return [result for result in results if result.score >= min_score]


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


def retrieve_context(query: str, settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    store = PgVectorStore(settings)
    if store.count() == 0:
        return None

    embedding = EmbeddingClient(settings).embed_texts([query])[0]
    raw_results = store.search(embedding, top_k=settings.retrieval_top_k)
    results = filter_results_by_score(
        results=raw_results,
        min_score=settings.retrieval_min_score,
    )

    return format_retrieved_context(results) if results else None
