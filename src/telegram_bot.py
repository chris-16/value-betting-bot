"""Telegram Bot — handles commands (/status, /stats, /today) and scheduled alerts.

Runs as a long-polling bot in its own thread or process, responding to user
commands and sending proactive alerts for value bets and daily summaries.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.config import settings
from src.db.session import get_session
from src.strategies.paper_trading import get_portfolio_stats
from src.telegram_alerts import (
    format_stats_message,
    format_status_message,
    format_today_bets,
    get_todays_bets,
    send_daily_summary,
)

logger = logging.getLogger(__name__)

# Global reference to bot start time
_bot_started_at: datetime = datetime.now()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — return current bot and scheduler health."""
    text = format_status_message(_bot_started_at)
    if update.message:
        await update.message.reply_text(text, parse_mode="HTML")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats — return cumulative ROI, win rate, and drawdown."""
    session_gen = get_session()
    session = next(session_gen)
    try:
        stats = get_portfolio_stats(session, settings.paper_trading_bankroll)
        text = format_stats_message(stats)
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass

    if update.message:
        await update.message.reply_text(text, parse_mode="HTML")


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/today — return all bets and P&L for the current day."""
    session_gen = get_session()
    session = next(session_gen)
    try:
        bets = get_todays_bets(session)
        text = format_today_bets(bets, session)
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass

    if update.message:
        await update.message.reply_text(text, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Scheduled daily summary
# ---------------------------------------------------------------------------


async def _daily_summary_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job callback: send daily P&L summary to the configured chat."""
    session_gen = get_session()
    session = next(session_gen)
    try:
        send_daily_summary(session)
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass


# ---------------------------------------------------------------------------
# Bot lifecycle
# ---------------------------------------------------------------------------


def create_application() -> Application:  # type: ignore[type-arg]
    """Build and configure the Telegram bot application.

    Returns:
        Configured Application instance ready to run.

    Raises:
        ValueError: If TELEGRAM_BOT_TOKEN is not configured.
    """
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required to run the Telegram bot")

    global _bot_started_at
    _bot_started_at = datetime.now()

    app = Application.builder().token(settings.telegram_bot_token).build()

    # Register command handlers
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("today", cmd_today))

    # Schedule daily summary
    if settings.telegram_chat_id:
        hour = settings.telegram_daily_summary_hour
        logger.info("Scheduling daily summary at %02d:00", hour)

    return app


def run_bot() -> None:
    """Start the Telegram bot with long-polling (blocking)."""
    logger.info("Starting Telegram bot...")
    app = create_application()
    app.run_polling(drop_pending_updates=True)


def start_bot_thread() -> threading.Thread | None:
    """Start the Telegram bot in a background daemon thread.

    Returns:
        The started Thread, or None if Telegram is not configured.
    """
    if not settings.telegram_bot_token:
        logger.info("Telegram bot not configured — skipping")
        return None

    thread = threading.Thread(target=run_bot, daemon=True, name="telegram-bot")
    thread.start()
    logger.info("Telegram bot started in background thread")
    return thread


if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_bot()
