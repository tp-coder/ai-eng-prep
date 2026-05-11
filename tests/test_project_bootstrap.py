from app.config import get_settings
from app.llm import LLMConfigurationError, LLMClient
from app.schemas import AssistantResponse
from app.logging_config import configure_logging


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.app_name == "ai-engineering-prep"
    assert settings.app_env == "local"


def test_llm_client_require_api_key() -> None:
    class FakeSettings:
        openai_api_key = None
        openai_model = "gpt-5-mini"
        openai_timeout_seconds = 30
        openai_max_retries = 2

    try:
        LLMClient(FakeSettings())
    except LLMConfigurationError as error:
        assert "OPENAI_API_KEY is missing" in str(error)
    else:
        raise AssertionError("Expected LLMConfigurationError")


def test_assistant_response_schema_accepts_valid_response() -> None:
    response = AssistantResponse(
        answer="The project skeleton is alive.",
        confidence="high",
        missing_context=[],
        next_actions=["Add structured output support."],
        source_references=[],
    )
    assert response.answer == "The project skeleton is alive."
    assert response.confidence == "high"
    assert response.next_actions == ["Add structured output support."]


def configure_logging_does_not_crash() -> None:
    settings = get_settings(log_level="INFO")
    configure_logging(settings)
