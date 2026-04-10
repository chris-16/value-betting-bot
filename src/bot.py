"""Value Betting Bot - main loop for scanning odds and placing paper bets."""

from __future__ import annotations

import logging
import time

from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 300  # 5 minutes


def run() -> None:
    """Main bot loop: fetch odds, run model, detect value, place paper bets."""
    logger.info("Value Betting Bot starting (paper trading mode)")
    logger.info(
        "Bankroll: %s | Min edge: %s",
        settings.paper_trading_bankroll,
        settings.min_value_edge,
    )

    while True:
        try:
            logger.info("Scanning for value bets...")
            # TODO: fetch upcoming matches
            # TODO: fetch odds from bookmakers
            # TODO: run ML predictions
            # TODO: compare predictions vs odds, detect value
            # TODO: place paper bets
            logger.info("Scan complete. Sleeping %ds...", SCAN_INTERVAL_SECONDS)
        except Exception:
            logger.exception("Error during scan cycle")

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
