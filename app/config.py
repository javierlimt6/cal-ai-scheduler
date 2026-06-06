"""Application configuration loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cal_api_key: str = ""
    cal_api_base_url: str = "https://api.cal.com/v2"
    cal_username: str = ""

    llm_provider: str = "mock"
    llm_model: str = ""  # empty -> the provider's default (anthropic: claude-opus-4-8)
    anthropic_api_key: str = ""

    # Empty -> resolved from the cal.com /me profile at startup, else UTC
    timezone: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
