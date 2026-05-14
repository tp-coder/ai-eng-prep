import pytest
from app.documents import chunk_text


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    chunks = chunk_text("hello world", chunk_size=100, chunk_overlap=10)
    assert chunks == ["hello world"]


def test_chunk_text_rejects_overlap_larger_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        chunk_text("hello world", chunk_size=100, chunk_overlap=100)


def test_chunk_text_creates_overlapping_chunks() -> None:
    text = "a" * 250
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 3
