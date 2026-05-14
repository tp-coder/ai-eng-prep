import logging
from openai import OpenAI
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingConfigurationError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY is missing. Add it to your .env file before trying again."
            )

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=self.settings.openai_max_retries,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.info(
            "embedding_request_started model=%s text_count=%s",
            self.settings.openai_embedding_model,
            len(texts),
        )

        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        logger.info(
            "embedding_request_completed model=%s embedding_count=%s",
            self.settings.openai_embedding_model,
            len(embeddings),
        )

        return embeddings
