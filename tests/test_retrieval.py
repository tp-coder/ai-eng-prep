from app.index import IndexedChunk, SearchResult
from app.main import filter_results_by_score, build_no_context_response


def make_result(score: float) -> SearchResult:
    return SearchResult(
        chunk=IndexedChunk(
            id=f"chunk-{score}",
            source_path="test.md",
            text="test",
            chunk_index=0,
            embedding=[1.0, 0.0],
        ),
        score=score,
    )


def test_filter_results_by_score_keeps_results_at_or_above_threshold() -> None:
    results = [
        make_result(0.9),
        make_result(0.25),
        make_result(0.1),
    ]

    filtered = filter_results_by_score(results, min_score=0.25)

    assert [result.score for result in filtered] == [0.9, 0.25]


def test_filter_results_by_score_returns_empty_list_when_no_match() -> None:
    results = [
        make_result(0.2),
        make_result(0.1),
    ]

    filtered = filter_results_by_score(results, min_score=0.25)

    assert filtered == []


def test_build_no_context_explains_retrieval_failure() -> None:
    response = build_no_context_response(min_score=0.25, top_score=0.07)

    assert "I could not answer your question from the local Qdrant collection" in response.answer
    assert response.confidence == "high"
    assert "Retrieval minimum score is 0.2500" in response.missing_context
    assert "Highest retrieved score was 0.0700." in response.missing_context
    assert response.source_references == []
