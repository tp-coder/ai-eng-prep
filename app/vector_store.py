import logging
from typing import Collection
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import Settings, get_settings
from app.documents import DocumentChunk
from app.index import IndexedChunk, SearchResult

logger = logging.getLogger(__name__)

# deterministic namespace -> the same chunk_id always maps to the same point_id
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "ai-eng-prep.qdrant")


class QdrantVectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # for now, use a local on-disk instance: same client/API with zero infra
        # note: the path is locked to ONE process at a time
        self.client = QdrantClient(path=self.settings.qdrant_path)
        self.collection = self.settings.qdrant_collection

    def reset_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "qdrant_collection_ready collection=%s dim=%s distance=cosine",
            self.collection,
            self.settings.embedding_dim,
        )

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, chunk.id)),
                vector=embedding,
                payload={
                    "chunk_id": chunk.id,
                    "source_path": chunk.source_path,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self.client.upsert(Collection_name=self.collection, points=points)
        logger.info("qdrant_upserted collection=%s point_count=%s",
                    self.collection, len(points))

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    chunk=IndexedChunk(
                        id=payload.get("chunk_id", str(point.id)),
                        source_path=payload.get("source_path", ""),
                        text=payload.get("text", ""),
                        chunk_index=payload.get("chunk_index", 0),
                        embedding=[]  # not needed back from the store
                    ),
                    score=point.score,
                )
            )
        return results
