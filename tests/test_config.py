"""Tests for application configuration."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.config import Settings, settings


def test_settings_defaults() -> None:
    """Default settings should be populated from env or fallbacks."""
    assert settings.database_url.startswith("postgresql://")
    assert isinstance(settings.paper_trading_bankroll, Decimal)
    assert isinstance(settings.min_value_edge, Decimal)
    assert settings.log_level == "INFO"


def test_settings_immutable() -> None:
    """Settings dataclass is frozen and should not allow mutation."""
    with pytest.raises(AttributeError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_settings_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should read from environment variables."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PAPER_TRADING_BANKROLL", "5000.00")
    monkeypatch.setenv("MIN_VALUE_EDGE", "0.10")
    monkeypatch.setenv("ODDS_API_KEY", "test-key")

    s = Settings()
    assert s.database_url == "postgresql://u:p@host/db"
    assert s.log_level == "DEBUG"
    assert s.paper_trading_bankroll == Decimal("5000.00")
    assert s.min_value_edge == Decimal("0.10")
    assert s.odds_api_key == "test-key"
