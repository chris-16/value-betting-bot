"""Football-data.co.uk historical CSV loader.

Downloads and parses CSV files with match results and Bet365 closing odds
for backtesting purposes.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests
from sqlalchemy.orm import Session

from src.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# League code mapping for football-data.co.uk URLs
LEAGUE_CSV_MAP: dict[str, str] = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "Argentina Primera División": "ARG",
}


@dataclass
class HistoricalMatch:
    """Parsed match from a football-data.co.uk CSV."""

    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str  # "H", "D", "A"
    b365_home: Decimal | None
    b365_draw: Decimal | None
    b365_away: Decimal | None


def _safe_decimal(value: str | None) -> Decimal | None:
    """Convert a CSV value to Decimal, returning None on failure."""
    if value is None or value.strip() == "":
        return None
    try:
        return Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None


def _safe_int(value: str | None) -> int:
    """Convert a CSV value to int, returning 0 on failure."""
    if value is None or value.strip() == "":
        return 0
    try:
        return int(value.strip())
    except (ValueError, TypeError):
        return 0


def download_season_csv(
    league: str,
    season: str,
    data_dir: str | None = None,
) -> Path:
    """Download a season CSV from football-data.co.uk.

    Args:
        league: Our league name (e.g. "Premier League").
        season: Season code (e.g. "2425" for 2024/25).
        data_dir: Directory to save CSVs (default: data/historical).

    Returns:
        Path to the downloaded CSV file.
    """
    csv_code = LEAGUE_CSV_MAP.get(league)
    if csv_code is None:
        raise ValueError(f"League '{league}' not mapped for football-data.co.uk")

    if data_dir is None:
        data_dir = settings.historical_data_dir

    dest = Path(data_dir)
    dest.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/{season}/{csv_code}.csv"
    file_path = dest / f"{csv_code}_{season}.csv"

    if file_path.exists():
        logger.info("CSV already exists: %s", file_path)
        return file_path

    logger.info("Downloading %s → %s", url, file_path)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    file_path.write_text(resp.text, encoding="utf-8")
    logger.info("Downloaded %d bytes to %s", len(resp.text), file_path)
    return file_path


def parse_csv(path: Path) -> list[HistoricalMatch]:
    """Parse a football-data.co.uk CSV into structured matches.

    Args:
        path: Path to the CSV file.

    Returns:
        List of HistoricalMatch records.
    """
    matches: list[HistoricalMatch] = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Some CSVs have different column name cases
            home = row.get("HomeTeam") or row.get("HT", "")
            away = row.get("AwayTeam") or row.get("AT", "")
            date = row.get("Date", "")
            result = row.get("FTR", "")

            if not home or not away:
                continue

            matches.append(
                HistoricalMatch(
                    date=date,
                    home_team=home.strip(),
                    away_team=away.strip(),
                    home_goals=_safe_int(row.get("FTHG")),
                    away_goals=_safe_int(row.get("FTAG")),
                    result=result.strip(),
                    b365_home=_safe_decimal(row.get("B365H")),
                    b365_draw=_safe_decimal(row.get("B365D")),
                    b365_away=_safe_decimal(row.get("B365A")),
                )
            )

    logger.info("Parsed %d matches from %s", len(matches), path)
    return matches


def load_historical_data(
    session: Session,
    league: str,
    seasons: list[str],
    data_dir: str | None = None,
) -> int:
    """Download and parse historical data for multiple seasons.

    Args:
        session: SQLAlchemy session (currently unused, for future DB loading).
        league: League name.
        seasons: List of season codes (e.g. ["2324", "2425"]).
        data_dir: Override data directory.

    Returns:
        Total number of matches loaded.
    """
    total = 0
    for season in seasons:
        try:
            path = download_season_csv(league, season, data_dir)
            matches = parse_csv(path)
            total += len(matches)
        except Exception:
            logger.exception("Failed to load %s season %s", league, season)

    logger.info("Loaded %d historical matches for %s", total, league)
    return total
