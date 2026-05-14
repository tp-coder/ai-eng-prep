import json
import math
from dataclasses import dataclass
from pathlib import Path
from app.documents import DocumentChunk


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    source_path: str
    text: str
    chunk_index: int
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    chunk: IndexedChunk
    score: float


def save_index(index_path: str, chunks: list[IndexedChunk]) -> None:
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "id": chunk.id,
            "source_path": chunk.source_path,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "embedding": chunk.embedding,
        }
        for chunk in chunks
    ]

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(index_path: str) -> list[IndexedChunk]:
    path = Path(index_path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))

    return [
        IndexedChunk(
            id=item["id"],
            source_path=item["source_path"],
            text=item["text"],
            chunk_index=item["chunk_index"],
            embedding=item["embedding"],
        )
        for item in payload
    ]


def build_index(
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> list[IndexedChunk]:

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    return [
        IndexedChunk(
            id=chunk.id,
            source_path=chunk.source_path,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("left and right vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a**2 for a in left))
    right_norm = math.sqrt(sum(b**2 for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def search_index(
    indexed_chunks: list[IndexedChunk],
    query_embedding: list[float],
    top_k: int,
) -> list[SearchResult]:

    results = [
        SearchResult(
            chunk=chunk,
            score=cosine_similarity(chunk.embedding, query_embedding)
        )
        for chunk in indexed_chunks
    ]

    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
