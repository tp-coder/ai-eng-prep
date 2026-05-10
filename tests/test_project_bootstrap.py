from app.config import get_settings
from app.llm import LLMConfigurationError, LLMClient


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.app_name == "ai-engineering-prep"
    assert settings.app_env == "local"


def test_llm_client_require_api() -> None:
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
