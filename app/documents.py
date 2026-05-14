from dataclasses import dataclass
from pathlib import Path
from tracemalloc import start


@dataclass(frozen=True)
class SourceDocument:
    path: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source_path: str
    text: str
    chunk_index: int


def load_documents(docs_path: str) -> list[SourceDocument]:
    base_path = Path(docs_path)
    if not base_path.exists():
        return []

    documents: list[SourceDocument] = []
    for file_path in sorted(base_path.rglob("*")):
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            SourceDocument(
                path=str(file_path),
                text=text,
            )
        )

    return documents


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> list[str]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized_text = " ".join(text.split())
    if len(normalized_text) <= chunk_size:
        return [normalized_text]

    chunks: list[str] = []
    start_idx = 0

    while start_idx < len(normalized_text):
        end_idx = start_idx + chunk_size
        chunk = normalized_text[start_idx:end_idx].strip()

        if chunk:
            chunks.append(chunk)

        if end_idx >= len(normalized_text):
            break

        start_idx = end_idx - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:

    chunks: list[DocumentChunk] = []

    for document in documents:
        text_chunks = chunk_text(
            text=document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_idx, text in enumerate(text_chunks):
            chunks.append(
                DocumentChunk(
                    id=f"{document.path}::{chunk_idx}",
                    source_path=document.path,
                    text=text,
                    chunk_index=chunk_idx,
                )
            )

    return chunks
