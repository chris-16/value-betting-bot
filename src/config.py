"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/valuebets")
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    football_data_api_key: str = os.getenv("FOOTBALL_DATA_API_KEY", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    paper_trading_bankroll: Decimal = Decimal(os.getenv("PAPER_TRADING_BANKROLL", "1000.00"))
    min_value_edge: Decimal = Decimal(os.getenv("MIN_VALUE_EDGE", "0.05"))


settings = Settings()
