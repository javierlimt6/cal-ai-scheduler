"""Test-wide fixtures.

Tests must pass with no env vars, no network, and regardless of whatever a
developer has in their local ``.env`` (which pydantic-settings would otherwise
read). Real env vars take precedence over the dotenv file, so pinning them
here makes the suite hermetic.
"""

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("CAL_API_KEY", "")
    monkeypatch.setenv("CAL_USERNAME", "")
    monkeypatch.setenv("TIMEZONE", "UTC")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
