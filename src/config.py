"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/valuebets"
        )
    )
    odds_api_key: str = field(default_factory=lambda: _env("ODDS_API_KEY", ""))
    football_data_api_key: str = field(default_factory=lambda: _env("FOOTBALL_DATA_API_KEY", ""))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    paper_trading_bankroll: Decimal = field(
        default_factory=lambda: Decimal(_env("PAPER_TRADING_BANKROLL", "1000.00"))
    )
    min_value_edge: Decimal = field(default_factory=lambda: Decimal(_env("MIN_VALUE_EDGE", "0.05")))


settings = Settings()
