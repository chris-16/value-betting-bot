"""Tests for Telegram alerts and bot command formatting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import (
    Bet,
    BetOutcome,
    League,
    MarketType,
    Match,
    MatchStatus,
    Team,
)
from src.db.session import Base
from src.strategies.paper_trading import PortfolioStats
from src.strategies.value_engine import ValueBet
from src.telegram_alerts import (
    format_daily_summary,
    format_stats_message,
    format_status_message,
    format_today_bets,
    format_value_bet_alert,
    get_todays_bets,
    send_message,
    send_value_bet_alert,
)

INITIAL_BANKROLL = Decimal("1000.00")


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def _create_match(
    session: Session,
    status: MatchStatus = MatchStatus.SCHEDULED,
    home_goals: int | None = None,
    away_goals: int | None = None,
) -> Match:
    """Helper: create a match with required parent entities."""
    league = League(name="Premier League", country="England")
    home = Team(name="Arsenal")
    away = Team(name="Chelsea")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        status=status,
        home_goals=home_goals,
        away_goals=away_goals,
    )
    session.add(match)
    session.flush()
    return match


def _create_bet(
    session: Session,
    match: Match,
    selection: str = "home",
    odds_price: str = "2.5000",
    stake: str = "50.00",
    outcome: BetOutcome = BetOutcome.PENDING,
    pnl: str | None = None,
    settled_at: datetime | None = None,
    placed_at: datetime | None = None,
) -> Bet:
    """Helper: create a bet for testing."""
    bet = Bet(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection=selection,
        odds_price=Decimal(odds_price),
        stake=Decimal(stake),
        model_probability=Decimal("0.550000"),
        implied_probability=Decimal("0.400000"),
        value_edge=Decimal("0.150000"),
        outcome=outcome,
        pnl=Decimal(pnl) if pnl is not None else None,
        settled_at=settled_at,
    )
    session.add(bet)
    session.flush()

    # Override placed_at if specified (server_default won't apply in SQLite)
    if placed_at is not None:
        bet.placed_at = placed_at
        session.flush()

    return bet


def _sample_value_bet(match_id: int = 1) -> ValueBet:
    """Create a sample ValueBet for formatting tests."""
    return ValueBet(
        match_id=match_id,
        market=MarketType.MATCH_WINNER,
        selection="home",
        predicted_probability=Decimal("0.550000"),
        implied_probability=Decimal("0.400000"),
        odds_price=Decimal("2.5000"),
        edge=Decimal("0.150000"),
        kelly_fraction=Decimal("0.100000"),
        recommended_stake=Decimal("100.00"),
        bookmaker_id=1,
    )


def _sample_portfolio_stats() -> PortfolioStats:
    """Create sample PortfolioStats for formatting tests."""
    return PortfolioStats(
        initial_bankroll=Decimal("1000.00"),
        current_bankroll=Decimal("1150.00"),
        total_bets=20,
        settled_bets=18,
        pending_bets=2,
        wins=10,
        losses=7,
        voids=1,
        win_rate=Decimal("58.82"),
        total_staked=Decimal("800.00"),
        total_pnl=Decimal("150.00"),
        roi=Decimal("15.00"),
        max_drawdown=Decimal("75.00"),
        max_drawdown_pct=Decimal("6.98"),
        best_day_pnl=Decimal("120.00"),
        worst_day_pnl=Decimal("-45.00"),
        avg_odds=Decimal("2.3500"),
        avg_edge=Decimal("0.085000"),
    )


# ---------- format_value_bet_alert ----------


class TestFormatValueBetAlert:
    def test_contains_match_info(self) -> None:
        """Alert includes team names and league."""
        vb = _sample_value_bet()
        text = format_value_bet_alert(
            vb,
            "Arsenal",
            "Chelsea",
            "Premier League",
            datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        )
        assert "Arsenal vs Chelsea" in text
        assert "Premier League" in text

    def test_contains_bet_details(self) -> None:
        """Alert includes selection, odds, edge, and stake."""
        vb = _sample_value_bet()
        text = format_value_bet_alert(
            vb,
            "Arsenal",
            "Chelsea",
            "Premier League",
            datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        )
        assert "HOME" in text
        assert "2.5000" in text
        assert "+15.00%" in text
        assert "$100.00" in text

    def test_contains_probabilities(self) -> None:
        """Alert includes model and market probabilities."""
        vb = _sample_value_bet()
        text = format_value_bet_alert(
            vb,
            "Arsenal",
            "Chelsea",
            "Premier League",
            datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        )
        assert "55.0%" in text
        assert "40.0%" in text

    def test_html_formatting(self) -> None:
        """Alert uses HTML bold tags."""
        vb = _sample_value_bet()
        text = format_value_bet_alert(
            vb,
            "Arsenal",
            "Chelsea",
            "Premier League",
            datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        )
        assert "<b>Value Bet Detected</b>" in text
        assert "<b>HOME</b>" in text


# ---------- format_daily_summary ----------


class TestFormatDailySummary:
    def test_contains_portfolio_info(self) -> None:
        """Summary includes bankroll, ROI, and win rate."""
        stats = _sample_portfolio_stats()
        text = format_daily_summary(stats, [])
        assert "$1150.00" in text
        assert "+15.00%" in text
        assert "58.82%" in text

    def test_contains_drawdown(self) -> None:
        """Summary includes max drawdown percentage."""
        stats = _sample_portfolio_stats()
        text = format_daily_summary(stats, [])
        assert "6.98%" in text

    def test_today_with_bets(self) -> None:
        """Summary reflects today's settled bets."""
        session = _make_session()
        match = _create_match(session, status=MatchStatus.FINISHED, home_goals=2, away_goals=1)
        now = datetime.now()
        bet = _create_bet(
            session,
            match,
            outcome=BetOutcome.WIN,
            pnl="75.00",
            settled_at=now,
            placed_at=now,
        )
        session.commit()

        stats = _sample_portfolio_stats()
        text = format_daily_summary(stats, [bet])
        assert "+$75.00" in text
        assert "Settled: 1" in text

    def test_negative_pnl_formatting(self) -> None:
        """Summary handles negative total P&L correctly."""
        stats = PortfolioStats(
            initial_bankroll=Decimal("1000.00"),
            current_bankroll=Decimal("850.00"),
            total_bets=10,
            settled_bets=10,
            pending_bets=0,
            wins=3,
            losses=7,
            voids=0,
            win_rate=Decimal("30.00"),
            total_staked=Decimal("500.00"),
            total_pnl=Decimal("-150.00"),
            roi=Decimal("-15.00"),
            max_drawdown=Decimal("200.00"),
            max_drawdown_pct=Decimal("18.18"),
            best_day_pnl=Decimal("50.00"),
            worst_day_pnl=Decimal("-120.00"),
            avg_odds=Decimal("2.1000"),
            avg_edge=Decimal("0.060000"),
        )
        text = format_daily_summary(stats, [])
        assert "-$150.00" in text  # Total P&L with sign prefix
        assert "-15.00%" in text  # ROI


# ---------- format_status_message ----------


class TestFormatStatusMessage:
    def test_shows_running_status(self) -> None:
        """Status message indicates bot is running."""
        text = format_status_message(datetime.now() - timedelta(hours=2, minutes=30))
        assert "RUNNING" in text
        assert "2h 30m" in text

    def test_shows_config(self) -> None:
        """Status message includes scan interval and bankroll."""
        text = format_status_message(datetime.now())
        assert str(settings.odds_scan_interval_seconds) in text
        assert str(settings.paper_trading_bankroll) in text


# ---------- format_stats_message ----------


class TestFormatStatsMessage:
    def test_contains_key_metrics(self) -> None:
        """Stats message includes ROI, win rate, and drawdown."""
        stats = _sample_portfolio_stats()
        text = format_stats_message(stats)
        assert "+15.00%" in text
        assert "58.82%" in text
        assert "6.98%" in text
        assert "$75.00" in text

    def test_contains_bet_counts(self) -> None:
        """Stats message includes W/L/V breakdown."""
        stats = _sample_portfolio_stats()
        text = format_stats_message(stats)
        assert "10/7/1" in text
        assert "Total Bets: 20" in text


# ---------- format_today_bets ----------


class TestFormatTodayBets:
    def test_no_bets(self) -> None:
        """Shows appropriate message when no bets today."""
        session = _make_session()
        text = format_today_bets([], session)
        assert "No bets" in text

    def test_with_bets(self) -> None:
        """Formats today's bets with match details."""
        session = _make_session()
        match = _create_match(session)
        bet = _create_bet(session, match, selection="home", outcome=BetOutcome.PENDING)
        session.commit()

        text = format_today_bets([bet], session)
        assert "Arsenal vs Chelsea" in text
        assert "HOME" in text
        assert "2.5000" in text
        assert "[...]" in text  # pending indicator

    def test_settled_bet_shows_pnl(self) -> None:
        """Settled bets show P&L in the output."""
        session = _make_session()
        match = _create_match(session, status=MatchStatus.FINISHED, home_goals=2, away_goals=0)
        bet = _create_bet(
            session,
            match,
            outcome=BetOutcome.WIN,
            pnl="75.00",
            settled_at=datetime.now(),
        )
        session.commit()

        text = format_today_bets([bet], session)
        assert "[W]" in text
        assert "+75.00" in text


# ---------- send_message ----------


class TestSendMessage:
    @patch("src.telegram_alerts.settings")
    def test_skips_when_not_configured(self, mock_settings: MagicMock) -> None:
        """Returns False when Telegram is not configured."""
        mock_settings.telegram_bot_token = ""
        mock_settings.telegram_chat_id = ""
        assert send_message("test") is False

    @patch("src.telegram_alerts.requests.post")
    @patch("src.telegram_alerts.settings")
    def test_sends_message_successfully(
        self, mock_settings: MagicMock, mock_post: MagicMock
    ) -> None:
        """Successfully sends message via Telegram API."""
        mock_settings.telegram_bot_token = "test-token"
        mock_settings.telegram_chat_id = "12345"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        assert send_message("Hello!") is True

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["text"] == "Hello!"
        assert call_args.kwargs["json"]["chat_id"] == "12345"

    @patch("src.telegram_alerts.requests.post")
    @patch("src.telegram_alerts.settings")
    def test_handles_api_error(self, mock_settings: MagicMock, mock_post: MagicMock) -> None:
        """Returns False on HTTP error."""
        mock_settings.telegram_bot_token = "test-token"
        mock_settings.telegram_chat_id = "12345"

        import requests

        mock_post.side_effect = requests.RequestException("Connection failed")

        assert send_message("Hello!") is False


# ---------- send_value_bet_alert ----------


class TestSendValueBetAlert:
    @patch("src.telegram_alerts.send_message")
    def test_sends_alert_with_match_details(self, mock_send: MagicMock) -> None:
        """Alert includes match team names from DB lookup."""
        mock_send.return_value = True
        session = _make_session()
        match = _create_match(session)
        session.commit()

        vb = _sample_value_bet(match_id=match.id)
        result = send_value_bet_alert(vb, session)

        assert result is True
        mock_send.assert_called_once()
        text = mock_send.call_args[0][0]
        assert "Arsenal vs Chelsea" in text

    @patch("src.telegram_alerts.send_message")
    def test_returns_false_for_missing_match(self, mock_send: MagicMock) -> None:
        """Returns False when the match is not found."""
        session = _make_session()
        vb = _sample_value_bet(match_id=9999)
        result = send_value_bet_alert(vb, session)

        assert result is False
        mock_send.assert_not_called()


# ---------- get_todays_bets ----------


class TestGetTodaysBets:
    def test_returns_todays_bets(self) -> None:
        """Fetches bets placed today."""
        session = _make_session()
        match = _create_match(session)
        now = datetime.now()
        bet = _create_bet(session, match, placed_at=now)
        session.commit()

        bets = get_todays_bets(session)
        assert len(bets) == 1
        assert bets[0].id == bet.id

    def test_excludes_old_bets(self) -> None:
        """Does not include bets from previous days."""
        session = _make_session()
        match = _create_match(session)
        yesterday = datetime.now() - timedelta(days=2)
        _create_bet(session, match, placed_at=yesterday)
        session.commit()

        bets = get_todays_bets(session)
        assert len(bets) == 0


# ---------- _should_send_daily_summary ----------


class TestShouldSendDailySummary:
    def test_returns_false_if_already_sent_today(self) -> None:
        """Returns False when summary was already sent today."""
        from datetime import date as real_date

        from src.bot import _should_send_daily_summary

        # Always false if last_summary_date is today (regardless of hour)
        today = real_date.today()
        assert _should_send_daily_summary(today) is False

    def test_returns_false_at_wrong_hour(self) -> None:
        """Returns False when current hour doesn't match the configured hour."""
        from src.bot import _should_send_daily_summary

        now = datetime.now()
        # If the current hour doesn't match the configured hour, should be False
        if now.hour != settings.telegram_daily_summary_hour:
            assert _should_send_daily_summary(None) is False

    def test_logic_with_none_last_date(self) -> None:
        """With no last_summary_date, result depends only on current hour."""
        from src.bot import _should_send_daily_summary

        now = datetime.now()
        result = _should_send_daily_summary(None)
        # Should only be True if current hour matches
        expected = now.hour == settings.telegram_daily_summary_hour
        assert result is expected
