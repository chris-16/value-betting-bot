"""Value Betting Bot - main loop for scanning odds and placing paper bets."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime

from src.config import settings
from src.db.session import get_session
from src.models.prediction import predict_upcoming_matches
from src.scrapers.odds_api import OddsAPIClient, OddsAPIError, scan_all_leagues
from src.scrapers.result_updater import update_results
from src.strategies.value_engine import place_paper_bet, scan_for_value
from src.telegram_alerts import send_bet_notification, send_daily_summary

# Weekly retrain: Monday at 6 AM
_RETRAIN_DAY = 0  # Monday
_RETRAIN_HOUR = 6

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Track bets already placed (match_id, selection) to avoid duplicates
_placed: set[tuple[int, str]] = set()
_last_retrain_date: date | None = None


def _should_send_daily_summary(last_summary_date: date | None) -> bool:
    now = datetime.now()
    if now.hour != settings.telegram_daily_summary_hour:
        return False
    if last_summary_date == now.date():
        return False
    return True


def run() -> None:
    """Main bot loop: fetch odds, run model, detect value, auto-place paper bets."""
    logger.info("Value Betting Bot starting (paper trading mode)")
    logger.info(
        "Bankroll: %s | Min edge: %s | Scan interval: %ds",
        settings.paper_trading_bankroll,
        settings.min_value_edge,
        settings.odds_scan_interval_seconds,
    )

    try:
        client = OddsAPIClient()
    except OddsAPIError:
        logger.exception("Failed to initialise Odds API client — check ODDS_API_KEY")
        return

    last_summary_date: date | None = None

    while True:
        try:
            logger.info("Scanning for value bets...")

            session_gen = get_session()
            session = next(session_gen)
            try:
                # Update results for finished matches and settle bets
                try:
                    updated = update_results(session)
                    if updated:
                        logger.info("Updated %d match results", updated)
                except Exception:
                    logger.exception("Error updating match results")

                results = scan_all_leagues(session, client)
                total = sum(results.values())
                logger.info(
                    "Odds scan complete — %d new odds rows across %d leagues",
                    total,
                    len(results),
                )

                # Weekly model retrain (Monday 6AM)
                global _last_retrain_date
                now = datetime.now()
                if (
                    now.weekday() == _RETRAIN_DAY
                    and now.hour == _RETRAIN_HOUR
                    and _last_retrain_date != now.date()
                ):
                    try:
                        from src.models.learning import retrain_model

                        retrain_model(session)
                        _last_retrain_date = now.date()
                        logger.info("Weekly model retrain complete")
                    except Exception:
                        logger.exception("Error during weekly model retrain")

                # Generate predictions for matches that don't have them yet
                try:
                    predicted = predict_upcoming_matches(session, limit=10)
                    logger.info("Generated predictions for %d matches", len(predicted))
                except Exception:
                    logger.exception("Error generating predictions")

                # Detect value bets, auto-place, and notify
                value_bets = scan_for_value(session)
                for vb in value_bets:
                    key = (vb.match_id, vb.selection)
                    if key in _placed:
                        continue

                    bet = place_paper_bet(session, vb)
                    _placed.add(key)
                    logger.info("Paper bet placed: %s", bet)

                    try:
                        send_bet_notification(vb, session)
                    except Exception:
                        logger.exception("Failed to send Telegram notification")

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
