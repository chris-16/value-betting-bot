"""Value Betting Bot - main loop for scanning odds and placing paper bets."""

from __future__ import annotations

import logging
import time

from src.config import settings
from src.db.session import get_session
from src.scrapers.odds_api import OddsAPIClient, OddsAPIError, scan_all_leagues

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run() -> None:
    """Main bot loop: fetch odds, run model, detect value, place paper bets."""
    logger.info("Value Betting Bot starting (paper trading mode)")
    logger.info(
        "Bankroll: %s | Min edge: %s | Scan interval: %ds",
        settings.paper_trading_bankroll,
        settings.min_value_edge,
        settings.odds_scan_interval_seconds,
    )

    # Pre-validate API key at startup
    try:
        client = OddsAPIClient()
    except OddsAPIError:
        logger.exception("Failed to initialise Odds API client — check ODDS_API_KEY")
        return

    while True:
        try:
            logger.info("Scanning for value bets...")

            # Fetch and persist odds from all leagues
            session_gen = get_session()
            session = next(session_gen)
            try:
                results = scan_all_leagues(session, client)
                total = sum(results.values())
                logger.info(
                    "Odds scan complete — %d new odds rows across %d leagues",
                    total,
                    len(results),
                )
                for sport_key, count in results.items():
                    logger.info("  %s: %d rows", sport_key, count)
            finally:
                try:
                    next(session_gen)
                except StopIteration:
                    pass

            # TODO: run ML predictions
            # TODO: compare predictions vs odds, detect value
            # TODO: place paper bets

            logger.info(
                "Scan complete. Sleeping %ds...", settings.odds_scan_interval_seconds
            )
        except Exception:
            logger.exception("Error during scan cycle")

        time.sleep(settings.odds_scan_interval_seconds)


if __name__ == "__main__":
    run()
