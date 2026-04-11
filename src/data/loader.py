"""Data loader — CLI tool for refreshing xG stats and loading historical data.

Usage:
    python -m src.data.loader [--xg] [--historical] [--all]
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings
from src.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Default seasons to load for historical data
DEFAULT_SEASONS = ["2324", "2425"]

# Leagues to scrape xG for
XG_LEAGUES = ["Premier League", "La Liga"]

# Leagues to load historical CSVs for
HISTORICAL_LEAGUES = ["Premier League", "La Liga"]


def refresh_xg_stats(season: str = "2025") -> int:
    """Scrape current season xG from Understat and persist to DB.

    Args:
        season: Understat season year (e.g. "2025" for 2025/26).

    Returns:
        Total number of team records upserted.
    """
    from src.scrapers.understat import UnderstatScraper, persist_team_xg

    session = SessionLocal()
    try:
        scraper = UnderstatScraper()
        total = 0

        for league in XG_LEAGUES:
            try:
                data = scraper.fetch_team_xg(league, season)
                if data:
                    count = persist_team_xg(session, data)
                    total += count
                    logger.info("Loaded xG for %d teams in %s", count, league)
            except Exception:
                logger.exception("Failed to fetch xG for %s", league)

        return total
    finally:
        session.close()


def load_all_historical(seasons: list[str] | None = None) -> int:
    """Load historical CSVs from football-data.co.uk.

    Args:
        seasons: List of season codes. Defaults to DEFAULT_SEASONS.

    Returns:
        Total number of matches loaded.
    """
    from src.scrapers.football_data_uk import load_historical_data

    if seasons is None:
        seasons = DEFAULT_SEASONS

    session = SessionLocal()
    try:
        total = 0
        for league in HISTORICAL_LEAGUES:
            try:
                count = load_historical_data(session, league, seasons)
                total += count
            except Exception:
                logger.exception("Failed to load historical data for %s", league)
        return total
    finally:
        session.close()


def main() -> None:
    """CLI entrypoint for data loading."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Value Betting Bot — Data Loader")
    parser.add_argument("--xg", action="store_true", help="Refresh xG stats from Understat")
    parser.add_argument(
        "--historical", action="store_true", help="Load historical CSVs"
    )
    parser.add_argument("--all", action="store_true", help="Run all data loading tasks")
    parser.add_argument(
        "--season", default="2025", help="Understat season year (default: 2025)"
    )
    args = parser.parse_args()

    if not any([args.xg, args.historical, args.all]):
        args.all = True

    if args.xg or args.all:
        logger.info("Refreshing xG stats...")
        count = refresh_xg_stats(args.season)
        logger.info("xG refresh complete: %d teams updated", count)

    if args.historical or args.all:
        logger.info("Loading historical data...")
        count = load_all_historical()
        logger.info("Historical load complete: %d matches", count)


if __name__ == "__main__":
    main()
