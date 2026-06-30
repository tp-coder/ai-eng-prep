from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    app_name: str = "ai-engineering-prep"
    log_level: str = "INFO"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: int = 30
    openai_max_retries: int = 2

    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "documents"
    embedding_dim: int = 1536  # text-embedding-3-small dimension

    docs_path: str = "data/docs"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    retrieval_min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    database_url: str = "postgresql://aiprep:aiprep@localhost:5433/aiprep"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> "Settings":
    return Settings()
