import logging
from rich.console import Console
from app.config import get_settings
from app.documents import chunk_documents, load_documents
from app.embeddings import EmbeddingClient
from app.index import build_index, save_index
from app.logging_config import configure_logging


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

    embedding_client = EmbeddingClient(settings)
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])
    indexed_chunks = build_index(chunks, embeddings)

    save_index(settings.index_path, indexed_chunks)
    console.print(
        f"[bold green]Saved index to {settings.index_path}[/bold green]")


if __name__ == "__main__":
    main()
