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
    api_football_key: str = field(default_factory=lambda: _env("API_FOOTBALL_KEY", ""))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    min_prediction_confidence: Decimal = field(
        default_factory=lambda: Decimal(_env("MIN_PREDICTION_CONFIDENCE", "0.65"))
    )
    paper_trading_bankroll: Decimal = field(
        default_factory=lambda: Decimal(_env("PAPER_TRADING_BANKROLL", "1000.00"))
    )
    min_value_edge: Decimal = field(default_factory=lambda: Decimal(_env("MIN_VALUE_EDGE", "0.10")))
    odds_scan_interval_seconds: int = field(
        default_factory=lambda: int(_env("ODDS_SCAN_INTERVAL_SECONDS", "300"))
    )
    kelly_fraction: Decimal = field(
        default_factory=lambda: Decimal(_env("KELLY_FRACTION", "0.50"))
    )
    model_data_dir: str = field(default_factory=lambda: _env("MODEL_DATA_DIR", "data/models"))
    historical_data_dir: str = field(
        default_factory=lambda: _env("HISTORICAL_DATA_DIR", "data/historical")
    )
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))
    telegram_daily_summary_hour: int = field(
        default_factory=lambda: int(_env("TELEGRAM_DAILY_SUMMARY_HOUR", "22"))
    )


settings = Settings()
