import logging

from rich.console import Console

from app.config import get_settings
from app.documents import chunk_documents, load_documents
from app.embeddings import EmbeddingClient
from app.logging_config import configure_logging
from app.vector_store import QdrantVectorStore


console = Console()
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    documents = load_documents(settings.docs_path)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        console.print(
            "[bold yellow]No documents found to index.[/bold yellow]")
        return

    console.print(f"Loaded {len(documents)} documents")
    console.print(f"Created {len(chunks)} document chunks")

    # Refactored to use Qdrant Vector Store
    embedding_client = EmbeddingClient(settings)
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])
    store = QdrantVectorStore(settings)
    store.reset_collection()
    store.upsert(chunks, embeddings)
    console.print(
        f"[bold green]Indexed {len(chunks)} chunks into Qdrant collection "
        f"'{settings.qdrant_collection}'[/bold green]"
    )


if __name__ == "__main__":
    main()
