from dataclasses import dataclass


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
