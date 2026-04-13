"""Value Betting Bot - main loop for scanning odds and placing paper bets."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import func, select

from src.config import settings
from src.db.models import Bet, BetOutcome
from src.db.session import get_session
from src.models.prediction import predict_upcoming_matches
from src.scrapers.odds_api import OddsAPIClient, OddsAPIError, scan_all_leagues
from src.scrapers.result_updater import update_results
from src.strategies.paper_trading import get_current_bankroll
from src.strategies.value_engine import place_paper_bet, scan_for_value
from src.telegram_alerts import send_daily_summary, send_message

# Weekly retrain: Monday at 6 AM
_RETRAIN_DAY = 0  # Monday
_RETRAIN_HOUR = 6

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_last_retrain_date: date | None = None
_last_analysis_date: date | None = None


def _get_existing_bet_keys(session) -> set[tuple[int, str]]:
    """Load (match_id, selection) for all non-void bets from the database."""
    rows = session.execute(
        select(Bet.match_id, Bet.selection).where(
            Bet.outcome != BetOutcome.VOID,
        )
    ).all()
    return {(r.match_id, r.selection) for r in rows}


def _count_pending_bets(session) -> int:
    """Count currently pending bets."""
    return session.execute(
        select(func.count(Bet.id)).where(Bet.outcome == BetOutcome.PENDING)
    ).scalar_one()


def _total_pending_stakes(session) -> Decimal:
    """Sum of stakes for all pending bets."""
    result = session.execute(
        select(func.coalesce(func.sum(Bet.stake), Decimal("0.00"))).where(
            Bet.outcome == BetOutcome.PENDING
        )
    ).scalar_one()
    return Decimal(str(result))


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

                # Weekly model retrain + xG refresh (Monday 6AM)
                global _last_retrain_date
                now = datetime.now()
                if (
                    now.weekday() == _RETRAIN_DAY
                    and now.hour == _RETRAIN_HOUR
                    and _last_retrain_date != now.date()
                ):
                    # Refresh xG data from Understat
                    try:
                        from src.scrapers.understat import persist_team_xg

                        season = str(now.year - 1) if now.month < 8 else str(now.year)
                        for league in ["Premier League", "La Liga"]:
                            count = persist_team_xg(session, league, season)
                            logger.info("xG refresh: %s — %d teams", league, count)
                    except Exception:
                        logger.exception("Error refreshing xG data")

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

                # Detect and place value bets with all safety limits
                existing_keys = _get_existing_bet_keys(session)
                pending_count = _count_pending_bets(session)
                current_bankroll = get_current_bankroll(
                    session, settings.paper_trading_bankroll
                )

                value_bets = scan_for_value(
                    session, bankroll=current_bankroll
                )
                placed_count = 0
                for vb in value_bets:
                    # DB-level dedup: skip if bet already exists for this match+selection
                    key = (vb.match_id, vb.selection)
                    if key in existing_keys:
                        continue

                    # Enforce max pending bets
                    if pending_count >= settings.max_pending_bets:
                        logger.info(
                            "Max pending bets reached (%d) — stopping",
                            settings.max_pending_bets,
                        )
                        break

                    # Enforce max bets per cycle
                    if placed_count >= settings.max_bets_per_cycle:
                        logger.info(
                            "Max bets per cycle reached (%d) — stopping",
                            settings.max_bets_per_cycle,
                        )
                        break

                    # Enforce max bankroll exposure
                    pending_stakes = _total_pending_stakes(session)
                    exposure = pending_stakes / settings.paper_trading_bankroll
                    if exposure >= settings.max_bankroll_exposure:
                        logger.info(
                            "Max bankroll exposure reached (%.0f%%) — stopping",
                            exposure * 100,
                        )
                        break

                    bet = place_paper_bet(session, vb)
                    existing_keys.add(key)
                    pending_count += 1
                    placed_count += 1
                    logger.info("Paper bet placed: %s", bet)

                if placed_count:
                    logger.info("Placed %d new bets this cycle", placed_count)

                # Daily P&L summary
                if _should_send_daily_summary(last_summary_date):
                    try:
                        send_daily_summary(session)
                        last_summary_date = datetime.now().date()
                        logger.info("Daily summary sent")
                    except Exception:
                        logger.exception("Failed to send daily summary")

                # Daily AI analysis
                global _last_analysis_date
                if (
                    now.hour == settings.ai_analysis_hour
                    and _last_analysis_date != now.date()
                ):
                    try:
                        from src.models.daily_analysis import (
                            format_analysis_telegram,
                            run_daily_analysis,
                        )

                        analysis = run_daily_analysis(session)
                        if analysis:
                            _last_analysis_date = now.date()

                            # Auto-tune parameters from recommendations
                            try:
                                from src.strategies.auto_tune import (
                                    apply_ai_parameter_tuning,
                                )

                                new_version = apply_ai_parameter_tuning(session, analysis)
                                if new_version:
                                    try:
                                        send_message(
                                            f"Auto-tuned to <b>{new_version.name}</b>\n"
                                            f"{new_version.description}"
                                        )
                                    except Exception:
                                        pass
                                    logger.info("Auto-tuned to %s", new_version.name)
                            except Exception:
                                logger.exception("Auto-tune failed (non-blocking)")

                            try:
                                msg = format_analysis_telegram(analysis)
                                send_message(msg)
                            except Exception:
                                logger.exception("Failed to send AI analysis notification")
                            logger.info("Daily AI analysis complete")
                    except Exception:
                        logger.exception("Error during daily AI analysis")

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
