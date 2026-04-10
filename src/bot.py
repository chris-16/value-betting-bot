"""Value Betting Bot - main loop for scanning odds and placing paper bets."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime

from src.config import settings
from src.db.session import get_session
from src.scrapers.odds_api import OddsAPIClient, OddsAPIError, scan_all_leagues
from src.strategies.value_engine import place_paper_bet, scan_for_value
from src.telegram_alerts import send_daily_summary, send_value_bet_alert
from src.telegram_bot import start_bot_thread

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _should_send_daily_summary(last_summary_date: date | None) -> bool:
    """Check if it's time to send the daily summary.

    Returns True if the current hour matches the configured hour and
    the summary hasn't been sent today yet.
    """
    now = datetime.now()
    if now.hour != settings.telegram_daily_summary_hour:
        return False
    if last_summary_date == now.date():
        return False
    return True


def run() -> None:
    """Main bot loop: fetch odds, run model, detect value, place paper bets."""
    logger.info("Value Betting Bot starting (paper trading mode)")
    logger.info(
        "Bankroll: %s | Min edge: %s | Scan interval: %ds",
        settings.paper_trading_bankroll,
        settings.min_value_edge,
        settings.odds_scan_interval_seconds,
    )

    # Start Telegram bot in background thread (if configured)
    start_bot_thread()

    # Pre-validate API key at startup
    try:
        client = OddsAPIClient()
    except OddsAPIError:
        logger.exception("Failed to initialise Odds API client — check ODDS_API_KEY")
        return

    last_summary_date: date | None = None

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

                # Detect value bets and place paper bets
                value_bets = scan_for_value(session)
                for vb in value_bets:
                    bet = place_paper_bet(session, vb)
                    logger.info("Paper bet placed: %s", bet)

                    # Send Telegram alert for each value bet
                    try:
                        send_value_bet_alert(vb, session)
                    except Exception:
                        logger.exception("Failed to send Telegram alert for bet %s", bet.id)

                # Daily P&L summary
                if _should_send_daily_summary(last_summary_date):
                    try:
                        send_daily_summary(session)
                        last_summary_date = datetime.now().date()
                        logger.info("Daily summary sent")
                    except Exception:
                        logger.exception("Failed to send daily summary")

            finally:
                try:
                    next(session_gen)
                except StopIteration:
                    pass

            logger.info("Scan complete. Sleeping %ds...", settings.odds_scan_interval_seconds)
        except Exception:
            logger.exception("Error during scan cycle")

        time.sleep(settings.odds_scan_interval_seconds)


if __name__ == "__main__":
    run()
