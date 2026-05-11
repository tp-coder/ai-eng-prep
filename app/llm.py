import time
from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from app.config import Settings, get_settings
from app.schemas import AssistantResponse


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseParsingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    parsed: AssistantResponse
    model: str
    latency_ms: int


SYSTEM_PROMPT = """
You are an AI engineering assistant.
Your job is to answer clearly and pragmatically.
Rules:
- Be direct.
- Do not invent missing facts.
- If the request lacks context, say what was missing.
- Use source_references only when  explicit source materials are provided.
- Keep next_actions practical and short.
""".strip()


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        if not self.settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is missing. Add it to your .env file before trying again."
            )

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=self.settings.openai_max_retries,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def complete(self, prompt: str) -> LLMResponse:
        started_at = time.perf_counter()

        response = self.client.responses.parse(
            model=self.settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            text_format=AssistantResponse,
        )

        latency_ms = int((time.perf_counter() - started_at) * 1000)

        return LLMResponse(
            parsed=response.output_parsed,
            model=self.settings.openai_model,
            latency_ms=latency_ms,
        )
