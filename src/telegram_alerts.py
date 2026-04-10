"""Telegram alert formatting and sending for value bet notifications and daily summaries."""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import Bet, BetOutcome, Match, Team
from src.strategies.paper_trading import PortfolioStats, get_portfolio_stats
from src.strategies.value_engine import ValueBet

logger = logging.getLogger(__name__)

_SEND_MESSAGE_URL = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SECONDS = 10


class TelegramError(Exception):
    """Raised when a Telegram API call fails."""


def _is_configured() -> bool:
    """Check whether Telegram bot token and chat ID are both set."""
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message via the Telegram Bot API.

    Args:
        text: Message body (supports HTML formatting).
        parse_mode: Telegram parse mode (default HTML).

    Returns:
        True if the message was sent successfully, False otherwise.

    Raises:
        TelegramError: If the Telegram API returns an error response.
    """
    if not _is_configured():
        logger.warning("Telegram not configured — skipping message")
        return False

    url = _SEND_MESSAGE_URL.format(token=settings.telegram_bot_token)
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise TelegramError(f"Telegram API error: {data.get('description', 'unknown')}")
        return True
    except requests.RequestException:
        logger.exception("Failed to send Telegram message")
        return False


# ---------------------------------------------------------------------------
# Value bet alert
# ---------------------------------------------------------------------------


def format_value_bet_alert(
    value_bet: ValueBet,
    home_team: str,
    away_team: str,
    league: str,
    kickoff: datetime,
) -> str:
    """Format a value bet detection into a Telegram alert message.

    Args:
        value_bet: The detected value bet.
        home_team: Home team name.
        away_team: Away team name.
        league: League / competition name.
        kickoff: Match kickoff time.

    Returns:
        Formatted HTML message string.
    """
    edge_pct = (value_bet.edge * Decimal("100")).quantize(Decimal("0.01"))
    pred_pct = (value_bet.predicted_probability * Decimal("100")).quantize(Decimal("0.1"))
    impl_pct = (value_bet.implied_probability * Decimal("100")).quantize(Decimal("0.1"))

    return (
        f"<b>Value Bet Detected</b>\n"
        f"\n"
        f"<b>{home_team} vs {away_team}</b>\n"
        f"{league} | {kickoff.strftime('%d %b %Y %H:%M')}\n"
        f"\n"
        f"Selection: <b>{value_bet.selection.upper()}</b>\n"
        f"Odds: <b>{value_bet.odds_price}</b>\n"
        f"Model: {pred_pct}% | Market: {impl_pct}%\n"
        f"Edge: <b>+{edge_pct}%</b>\n"
        f"Stake: <b>${value_bet.recommended_stake}</b>\n"
    )


def send_value_bet_alert(
    value_bet: ValueBet,
    session: Session,
) -> bool:
    """Look up match details and send a value bet alert via Telegram.

    Args:
        value_bet: The detected value bet.
        session: SQLAlchemy session for looking up match details.

    Returns:
        True if sent successfully.
    """
    match = session.get(Match, value_bet.match_id)
    if match is None:
        logger.error("Match %d not found — cannot send alert", value_bet.match_id)
        return False

    home_team = session.get(Team, match.home_team_id)
    away_team = session.get(Team, match.away_team_id)

    home_name = home_team.name if home_team else "Unknown"
    away_name = away_team.name if away_team else "Unknown"
    league_name = match.league.name if match.league else "Unknown"

    text = format_value_bet_alert(value_bet, home_name, away_name, league_name, match.kickoff)
    return send_message(text)


# ---------------------------------------------------------------------------
# Daily P&L summary
# ---------------------------------------------------------------------------


def format_daily_summary(stats: PortfolioStats, today_bets: list[Bet]) -> str:
    """Format a daily P&L summary message.

    Args:
        stats: Current portfolio statistics.
        today_bets: Bets placed or settled today.

    Returns:
        Formatted HTML message string.
    """
    pnl_sign = "+" if stats.total_pnl >= 0 else "-"
    roi_sign = "+" if stats.roi >= 0 else ""
    pnl_abs = abs(stats.total_pnl)

    settled_today = [b for b in today_bets if b.settled_at is not None]
    today_pnl = sum(
        (b.pnl for b in settled_today if b.pnl is not None),
        Decimal("0.00"),
    )
    today_pnl_sign = "+" if today_pnl >= 0 else ""

    wins_today = sum(1 for b in settled_today if b.outcome == BetOutcome.WIN)
    losses_today = sum(1 for b in settled_today if b.outcome == BetOutcome.LOSS)

    return (
        f"<b>Daily Summary</b>\n"
        f"\n"
        f"<b>Today's P&L:</b> {today_pnl_sign}${today_pnl}\n"
        f"Bets placed: {len(today_bets)} | Settled: {len(settled_today)}\n"
        f"W/L: {wins_today}/{losses_today}\n"
        f"\n"
        f"<b>Portfolio</b>\n"
        f"Bankroll: ${stats.current_bankroll}\n"
        f"Total P&L: {pnl_sign}${pnl_abs}\n"
        f"ROI: {roi_sign}{stats.roi}%\n"
        f"Win Rate: {stats.win_rate}%\n"
        f"Max Drawdown: {stats.max_drawdown_pct}%\n"
        f"Pending: {stats.pending_bets} bets\n"
    )


def send_daily_summary(session: Session) -> bool:
    """Compute today's stats and send a daily P&L summary via Telegram.

    Args:
        session: SQLAlchemy session.

    Returns:
        True if sent successfully.
    """
    stats = get_portfolio_stats(session, settings.paper_trading_bankroll)
    today_bets = get_todays_bets(session)
    text = format_daily_summary(stats, today_bets)
    return send_message(text)


# ---------------------------------------------------------------------------
# Query helpers for commands
# ---------------------------------------------------------------------------


def get_todays_bets(session: Session) -> list[Bet]:
    """Fetch all bets placed or settled today.

    Args:
        session: SQLAlchemy session.

    Returns:
        List of Bet objects from today.
    """
    today_start = datetime.combine(date.today(), datetime.min.time())

    return list(
        session.execute(
            select(Bet).where(
                (Bet.placed_at >= today_start)
                | ((Bet.settled_at.is_not(None)) & (Bet.settled_at >= today_start))
            )
        )
        .scalars()
        .all()
    )


def format_today_bets(bets: list[Bet], session: Session) -> str:
    """Format today's bets into a readable Telegram message.

    Args:
        bets: List of today's bets.
        session: SQLAlchemy session for match lookups.

    Returns:
        Formatted HTML message string.
    """
    if not bets:
        return "<b>Today's Bets</b>\n\nNo bets placed or settled today."

    today_pnl = sum(
        (b.pnl for b in bets if b.pnl is not None and b.settled_at is not None),
        Decimal("0.00"),
    )
    pnl_sign = "+" if today_pnl >= 0 else ""

    lines = [f"<b>Today's Bets</b> ({len(bets)} total)\n"]

    for bet in bets:
        match = session.get(Match, bet.match_id)
        if match:
            home = session.get(Team, match.home_team_id)
            away = session.get(Team, match.away_team_id)
            match_label = f"{home.name if home else '?'} vs {away.name if away else '?'}"
        else:
            match_label = f"Match #{bet.match_id}"

        outcome_emoji = {
            BetOutcome.WIN: "W",
            BetOutcome.LOSS: "L",
            BetOutcome.VOID: "V",
            BetOutcome.PENDING: "...",
        }
        status = outcome_emoji.get(bet.outcome, "?")
        pnl_str = f" ({'+' if bet.pnl >= 0 else ''}{bet.pnl})" if bet.pnl is not None else ""

        lines.append(
            f"[{status}] {match_label} | {bet.selection.upper()} "
            f"@ {bet.odds_price} | ${bet.stake}{pnl_str}"
        )

    lines.append(f"\n<b>Today P&L:</b> {pnl_sign}${today_pnl}")
    return "\n".join(lines)


def format_status_message(bot_started_at: datetime) -> str:
    """Format a /status response showing bot health.

    Args:
        bot_started_at: Timestamp when the bot started.

    Returns:
        Formatted HTML message string.
    """
    uptime = datetime.now() - bot_started_at
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"<b>Bot Status</b>\n"
        f"\n"
        f"Status: <b>RUNNING</b>\n"
        f"Uptime: {hours}h {minutes}m {seconds}s\n"
        f"Scan interval: {settings.odds_scan_interval_seconds}s\n"
        f"Min edge: {settings.min_value_edge}\n"
        f"Bankroll: ${settings.paper_trading_bankroll}\n"
    )


def format_stats_message(stats: PortfolioStats) -> str:
    """Format a /stats response with cumulative performance metrics.

    Args:
        stats: Current portfolio statistics.

    Returns:
        Formatted HTML message string.
    """
    roi_sign = "+" if stats.roi >= 0 else ""
    pnl_sign = "+" if stats.total_pnl >= 0 else ""

    return (
        f"<b>Cumulative Stats</b>\n"
        f"\n"
        f"ROI: <b>{roi_sign}{stats.roi}%</b>\n"
        f"Win Rate: <b>{stats.win_rate}%</b>\n"
        f"Max Drawdown: <b>{stats.max_drawdown_pct}%</b> (${stats.max_drawdown})\n"
        f"\n"
        f"Total Bets: {stats.total_bets}\n"
        f"W/L/V: {stats.wins}/{stats.losses}/{stats.voids}\n"
        f"Total P&L: {pnl_sign}${stats.total_pnl}\n"
        f"Bankroll: ${stats.current_bankroll}\n"
        f"Avg Odds: {stats.avg_odds}\n"
        f"Avg Edge: {stats.avg_edge}\n"
        f"Best Day: +${stats.best_day_pnl}\n"
        f"Worst Day: ${stats.worst_day_pnl}\n"
    )
