"""Telegram alerts — simple Spanish notifications for auto-placed bets and daily summaries."""

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
    pass


def _is_configured() -> bool:
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not _is_configured():
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
# Bet notification (auto-placed)
# ---------------------------------------------------------------------------


def _selection_label(selection: str, home_team: str, away_team: str) -> str:
    if selection == "home":
        return home_team
    if selection == "away":
        return away_team
    return "Empate"


def send_bet_notification(value_bet: ValueBet, session: Session) -> bool:
    match = session.get(Match, value_bet.match_id)
    if match is None:
        return False

    home_team = session.get(Team, match.home_team_id)
    away_team = session.get(Team, match.away_team_id)
    home_name = home_team.name if home_team else "?"
    away_name = away_team.name if away_team else "?"
    league_name = match.league.name if match.league else "?"

    confidence_pct = (value_bet.predicted_probability * Decimal("100")).quantize(Decimal("0.1"))
    edge_pct = (value_bet.edge * Decimal("100")).quantize(Decimal("0.1"))
    winner = _selection_label(value_bet.selection, home_name, away_name)
    potential_win = (value_bet.recommended_stake * (value_bet.odds_price - Decimal("1"))).quantize(
        Decimal("0.01")
    )

    text = (
        f"<b>Apuesta realizada</b>\n"
        f"\n"
        f"<b>{home_name} vs {away_name}</b>\n"
        f"{league_name} — {match.kickoff.strftime('%d/%m %H:%M')}\n"
        f"\n"
        f"Aposté a <b>{winner}</b> ({confidence_pct}% de certeza, ventaja {edge_pct}%)\n"
        f"Paga <b>{value_bet.odds_price}x</b>\n"
        f"Monto: <b>${value_bet.recommended_stake}</b> → ganancia potencial <b>${potential_win}</b>\n"
    )

    return send_message(text)


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------


def format_daily_summary(stats: PortfolioStats, today_bets: list[Bet]) -> str:
    placed_today = [b for b in today_bets if b.placed_at and b.placed_at.date() == date.today()]
    settled_today = [b for b in today_bets if b.settled_at and b.settled_at.date() == date.today()]
    today_pnl = sum(
        (b.pnl for b in settled_today if b.pnl is not None),
        Decimal("0.00"),
    )
    pnl_sign = "+" if today_pnl >= 0 else ""

    wins_today = sum(1 for b in settled_today if b.outcome == BetOutcome.WIN)
    losses_today = sum(1 for b in settled_today if b.outcome == BetOutcome.LOSS)

    roi_sign = "+" if stats.roi >= 0 else ""

    return (
        f"<b>Resumen del día</b>\n"
        f"\n"
        f"Nuevas apuestas: {len(placed_today)}\n"
        f"Resueltas hoy: {wins_today} ganadas, {losses_today} perdidas\n"
        f"Balance del día: <b>{pnl_sign}${today_pnl}</b>\n"
        f"\n"
        f"<b>Balance total</b>\n"
        f"Bankroll: ${stats.current_bankroll}\n"
        f"Rendimiento: {roi_sign}{stats.roi}%\n"
        f"Aciertos: {stats.win_rate}%\n"
        f"Pendientes: {stats.pending_bets} apuestas\n"
    )


def send_daily_summary(session: Session) -> bool:
    stats = get_portfolio_stats(session, settings.paper_trading_bankroll)
    today_bets = get_todays_bets(session)
    text = format_daily_summary(stats, today_bets)
    return send_message(text)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_todays_bets(session: Session) -> list[Bet]:
    """Return bets placed today or settled today (deduplicated)."""
    today_start = datetime.combine(date.today(), datetime.min.time())
    placed = set(
        session.execute(
            select(Bet).where(Bet.placed_at >= today_start)
        ).scalars().all()
    )
    settled = set(
        session.execute(
            select(Bet).where(
                Bet.settled_at.is_not(None),
                Bet.settled_at >= today_start,
            )
        ).scalars().all()
    )
    return list(placed | settled)


def format_today_bets(bets: list[Bet], session: Session) -> str:
    if not bets:
        return "<b>Apuestas de hoy</b>\n\nNinguna apuesta hoy."

    today_pnl = sum(
        (b.pnl for b in bets if b.pnl is not None and b.settled_at is not None),
        Decimal("0.00"),
    )
    pnl_sign = "+" if today_pnl >= 0 else ""

    lines = [f"<b>Apuestas de hoy</b> ({len(bets)})\n"]

    for bet in bets:
        match = session.get(Match, bet.match_id)
        if match:
            home = session.get(Team, match.home_team_id)
            away = session.get(Team, match.away_team_id)
            match_label = f"{home.name if home else '?'} vs {away.name if away else '?'}"
        else:
            match_label = f"Partido #{bet.match_id}"

        status_map = {
            BetOutcome.WIN: "Ganada",
            BetOutcome.LOSS: "Perdida",
            BetOutcome.VOID: "Anulada",
            BetOutcome.PENDING: "Pendiente",
        }
        status = status_map.get(bet.outcome, "?")
        pnl_str = f" ({'+' if bet.pnl >= 0 else ''}{bet.pnl})" if bet.pnl is not None else ""

        lines.append(f"• {match_label} — ${bet.stake} a {bet.odds_price}x [{status}]{pnl_str}")

    lines.append(f"\n<b>Balance:</b> {pnl_sign}${today_pnl}")
    return "\n".join(lines)


def format_status_message(bot_started_at: datetime) -> str:
    uptime = datetime.now() - bot_started_at
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"<b>Estado del bot</b>\n"
        f"\n"
        f"Funcionando hace {hours}h {minutes}m\n"
        f"Escaneo cada {settings.odds_scan_interval_seconds}s\n"
        f"Plata disponible: ${settings.paper_trading_bankroll}\n"
    )


def format_stats_message(stats: PortfolioStats) -> str:
    roi_sign = "+" if stats.roi >= 0 else ""
    pnl_sign = "+" if stats.total_pnl >= 0 else ""

    return (
        f"<b>Estadísticas</b>\n"
        f"\n"
        f"Rendimiento: <b>{roi_sign}{stats.roi}%</b>\n"
        f"Aciertos: <b>{stats.win_rate}%</b>\n"
        f"\n"
        f"Total apuestas: {stats.total_bets}\n"
        f"Ganadas/Perdidas: {stats.wins}/{stats.losses}\n"
        f"Balance: {pnl_sign}${stats.total_pnl}\n"
        f"Plata disponible: ${stats.current_bankroll}\n"
        f"Pendientes: {stats.pending_bets}\n"
    )
