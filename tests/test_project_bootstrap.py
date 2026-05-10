from app.config import get_settings


def test_project_bootstrap():
    settings = get_settings()
    assert settings.app_name == "ai-engineering-prep"
    assert settings.app_env == "local"
