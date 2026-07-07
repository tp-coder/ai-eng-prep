import logging

import psycopg
from psycopg import sql

from app.config import Settings, get_settings
from app.documents import DocumentChunk
from app.index import IndexedChunk, SearchResult

logger = logging.getLogger(__name__)


def _vector_literal(embedding: list[float]) -> str:
    # pgvector accepts a text literal like "[0.1, 0.2, 0.3]" cast with ::vector
    return "[" + ",".join(str(e) for e in embedding) + "]"


class PgVectorStore:
    """ Postgres + pgvector store to replace QdrantVectorStore
    Consolidates vectors into the same Postgres DB used for traces
    Vectors persist across sessions and do not require re-ingestion
    This swap also allows for concurrent connections, meaning CLI and MCP calls can be made in parallel
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db_url = self.settings.database_url
        self.table = self.settings.pgvector_table
        self.dim = self.settings.embedding_dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with psycopg.connect(self.db_url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {table} ("
                    " chunk_id TEXT PRIMARY KEY,"
                    " source_path TEXT,"
                    " text TEXT,"
                    " chunk_index INT,"
                    " embedding vector({dim})"
                    ")"
                ).format(table=sql.Identifier(self.table), dim=sql.Literal(self.dim))
            )
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                    "USING hnsw (embedding vector_cosine_ops)"
                ).format(idx=sql.Identifier(f"{self.table}_embedding_idx"), table=sql.Identifier(self.table))
            )
        logger.info("pgvector_ready table=%s dim=%s distance=cosine",
                    self.table, self.dim)

    def reset_collection(self) -> None:
        with psycopg.connect(self.db_url) as conn:
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=sql.Identifier(self.table)))
        self._ensure_schema()

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        rows = [
            (chunk.id, chunk.source_path, chunk.text,
             chunk.chunk_index, _vector_literal(embedding))
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        query = sql.SQL(
            "INSERT INTO {table} (chunk_id, source_path, text, chunk_index, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            " source_path = EXCLUDED.source_path,"
            " text = EXCLUDED.text,"
            " chunk_index = EXCLUDED.chunk_index,"
            " embedding = EXCLUDED.embedding"
        ).format(table=sql.Identifier(self.table))

        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, rows)

        logger.info("pgvector_upserted table=%s row_count=%s",
                    self.table, len(rows))

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        query_vector = _vector_literal(query_embedding)
        query = sql.SQL(
            "SELECT chunk_id, source_path, text, chunk_index, 1 - (embedding <=> %s::vector) AS score "
            "FROM {table} "
            "ORDER BY embedding <=> %s::vector "
            "LIMIT %s"
        ).format(table=sql.Identifier(self.table))

        with psycopg.connect(self.db_url) as conn:
            rows = conn.execute(
                query, (query_vector, query_vector, top_k)).fetchall()

        return [
            SearchResult(
                chunk=IndexedChunk(
                    id=chunk_id,
                    source_path=source_path or "",
                    text=text or "",
                    chunk_index=chunk_index or 0,
                    embedding=[],  # not needed back from the store
                ),
                score=float(score),
            )
            for chunk_id, source_path, text, chunk_index, score in rows
        ]

    def count(self) -> int:
        with psycopg.connect(self.db_url) as conn:
            row = conn.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {table}").format(table=sql.Identifier(self.table))
            ).fetchone()
        return row[0] if row else 0
