"""Tests for the Paper Trading Simulator — settlement, bankroll, and analytics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

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
from src.strategies.paper_trading import (
    PortfolioStats,
    _calculate_bet_pnl,
    _determine_outcome,
    calculate_max_drawdown,
    calculate_roi,
    calculate_win_rate,
    get_cumulative_pnl,
    get_current_bankroll,
    get_daily_pnl,
    get_portfolio_stats,
    settle_bet,
    settle_pending_bets,
)

INITIAL_BANKROLL = Decimal("1000.00")


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def _create_match(
    session: Session,
    status: MatchStatus = MatchStatus.FINISHED,
    home_goals: int | None = 2,
    away_goals: int | None = 1,
    kickoff_offset_days: int = 0,
) -> Match:
    """Helper: create a match with required parent entities."""
    league = League(name="Premier League", country="England")
    home = Team(name=f"Home_{id(league)}")
    away = Team(name=f"Away_{id(league)}")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 10, 15, 0, tzinfo=UTC) + timedelta(days=kickoff_offset_days),
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
    odds_price: str = "2.0000",
    stake: str = "50.00",
    outcome: BetOutcome = BetOutcome.PENDING,
    pnl: str | None = None,
    settled_at: datetime | None = None,
) -> Bet:
    """Helper: create a bet for testing."""
    bet = Bet(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection=selection,
        odds_price=Decimal(odds_price),
        stake=Decimal(stake),
        model_probability=Decimal("0.600000"),
        implied_probability=Decimal("0.500000"),
        value_edge=Decimal("0.100000"),
        outcome=outcome,
        pnl=Decimal(pnl) if pnl is not None else None,
        settled_at=settled_at,
    )
    session.add(bet)
    session.flush()
    return bet


# ---------- _determine_outcome ----------


def test_determine_outcome_home_win() -> None:
    """Home selection wins when home_goals > away_goals."""
    session = _make_session()
    match = _create_match(session, home_goals=3, away_goals=1)
    bet = _create_bet(session, match, selection="home")
    assert _determine_outcome(bet, match) == BetOutcome.WIN


def test_determine_outcome_home_loss() -> None:
    """Home selection loses when away_goals > home_goals."""
    session = _make_session()
    match = _create_match(session, home_goals=0, away_goals=2)
    bet = _create_bet(session, match, selection="home")
    assert _determine_outcome(bet, match) == BetOutcome.LOSS


def test_determine_outcome_draw_win() -> None:
    """Draw selection wins when home_goals == away_goals."""
    session = _make_session()
    match = _create_match(session, home_goals=1, away_goals=1)
    bet = _create_bet(session, match, selection="draw")
    assert _determine_outcome(bet, match) == BetOutcome.WIN


def test_determine_outcome_away_win() -> None:
    """Away selection wins when away_goals > home_goals."""
    session = _make_session()
    match = _create_match(session, home_goals=0, away_goals=3)
    bet = _create_bet(session, match, selection="away")
    assert _determine_outcome(bet, match) == BetOutcome.WIN


def test_determine_outcome_void_no_goals() -> None:
    """Outcome is VOID when match has no goal data."""
    session = _make_session()
    match = _create_match(session, home_goals=None, away_goals=None)
    bet = _create_bet(session, match, selection="home")
    assert _determine_outcome(bet, match) == BetOutcome.VOID


# ---------- _calculate_bet_pnl ----------


def test_pnl_win() -> None:
    """WIN P&L = stake * (odds - 1)."""
    session = _make_session()
    match = _create_match(session)
    bet = _create_bet(session, match, odds_price="2.5000", stake="100.00")
    pnl = _calculate_bet_pnl(bet, BetOutcome.WIN)
    assert pnl == Decimal("150.00")


def test_pnl_loss() -> None:
    """LOSS P&L = -stake."""
    session = _make_session()
    match = _create_match(session)
    bet = _create_bet(session, match, stake="75.00")
    pnl = _calculate_bet_pnl(bet, BetOutcome.LOSS)
    assert pnl == Decimal("-75.00")


def test_pnl_void() -> None:
    """VOID P&L = 0 (stake returned)."""
    session = _make_session()
    match = _create_match(session)
    bet = _create_bet(session, match, stake="50.00")
    pnl = _calculate_bet_pnl(bet, BetOutcome.VOID)
    assert pnl == Decimal("0.00")


# ---------- settle_bet ----------


def test_settle_bet_win() -> None:
    """Settling a winning bet updates outcome, pnl, and settled_at."""
    session = _make_session()
    match = _create_match(session, home_goals=2, away_goals=0)
    bet = _create_bet(session, match, selection="home", odds_price="2.0000", stake="100.00")
    session.commit()

    settled = settle_bet(session, bet)

    assert settled.outcome == BetOutcome.WIN
    assert settled.pnl == Decimal("100.00")
    assert settled.settled_at is not None


def test_settle_bet_loss() -> None:
    """Settling a losing bet sets negative P&L."""
    session = _make_session()
    match = _create_match(session, home_goals=0, away_goals=1)
    bet = _create_bet(session, match, selection="home", stake="50.00")
    session.commit()

    settled = settle_bet(session, bet)

    assert settled.outcome == BetOutcome.LOSS
    assert settled.pnl == Decimal("-50.00")


def test_settle_bet_already_settled_raises() -> None:
    """Cannot settle a bet that is already settled."""
    session = _make_session()
    match = _create_match(session)
    bet = _create_bet(session, match, outcome=BetOutcome.WIN, pnl="50.00")
    session.commit()

    with pytest.raises(ValueError, match="already settled"):
        settle_bet(session, bet)


def test_settle_bet_match_not_finished_raises() -> None:
    """Cannot settle a bet for a match that hasn't finished."""
    session = _make_session()
    match = _create_match(session, status=MatchStatus.SCHEDULED, home_goals=None, away_goals=None)
    bet = _create_bet(session, match, selection="home")
    session.commit()

    with pytest.raises(ValueError, match="not finished"):
        settle_bet(session, bet)


# ---------- settle_pending_bets ----------


def test_settle_pending_bets_auto() -> None:
    """Auto-settle finds and settles all pending bets for finished matches."""
    session = _make_session()

    # Finished match with pending bet
    match1 = _create_match(session, home_goals=3, away_goals=0)
    _create_bet(session, match1, selection="home", stake="100.00")

    # Scheduled match with pending bet (should NOT be settled)
    match2 = _create_match(
        session,
        status=MatchStatus.SCHEDULED,
        home_goals=None,
        away_goals=None,
        kickoff_offset_days=5,
    )
    _create_bet(session, match2, selection="away", stake="50.00")

    session.commit()

    settled = settle_pending_bets(session)

    assert len(settled) == 1
    assert settled[0].outcome == BetOutcome.WIN
    assert settled[0].pnl == Decimal("100.00")


# ---------- get_current_bankroll ----------


def test_current_bankroll_no_bets() -> None:
    """Bankroll equals initial when there are no bets."""
    session = _make_session()
    assert get_current_bankroll(session, INITIAL_BANKROLL) == INITIAL_BANKROLL


def test_current_bankroll_with_settled_bets() -> None:
    """Bankroll reflects settled P&L."""
    session = _make_session()
    match = _create_match(session)

    _create_bet(
        session,
        match,
        outcome=BetOutcome.WIN,
        pnl="100.00",
        stake="100.00",
        settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    _create_bet(
        session,
        match,
        selection="away",
        outcome=BetOutcome.LOSS,
        pnl="-50.00",
        stake="50.00",
        settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    session.commit()

    # 1000 + 100 - 50 = 1050
    assert get_current_bankroll(session, INITIAL_BANKROLL) == Decimal("1050.00")


def test_current_bankroll_pending_stakes_deducted() -> None:
    """Pending stakes are subtracted from bankroll (money at risk)."""
    session = _make_session()
    match = _create_match(session, status=MatchStatus.SCHEDULED, home_goals=None, away_goals=None)
    _create_bet(session, match, stake="200.00")
    session.commit()

    # 1000 - 200 pending = 800
    assert get_current_bankroll(session, INITIAL_BANKROLL) == Decimal("800.00")


# ---------- get_daily_pnl ----------


def test_daily_pnl_groups_by_date() -> None:
    """P&L is grouped by settlement date."""
    session = _make_session()
    match = _create_match(session)

    day1 = datetime(2026, 4, 8, 18, 0, tzinfo=UTC)
    day2 = datetime(2026, 4, 9, 18, 0, tzinfo=UTC)

    _create_bet(
        session, match, outcome=BetOutcome.WIN, pnl="100.00",
        stake="100.00", settled_at=day1,
    )
    _create_bet(
        session, match, selection="draw", outcome=BetOutcome.LOSS, pnl="-50.00",
        stake="50.00", settled_at=day1,
    )
    _create_bet(
        session, match, selection="away", outcome=BetOutcome.WIN, pnl="75.00",
        stake="75.00", settled_at=day2,
    )
    session.commit()

    daily = get_daily_pnl(session)

    assert len(daily) == 2
    # Day 1: +100 - 50 = +50
    assert daily[0][1] == Decimal("50.00")
    # Day 2: +75
    assert daily[1][1] == Decimal("75.00")


# ---------- get_cumulative_pnl ----------


def test_cumulative_pnl() -> None:
    """Cumulative P&L is a running total of daily P&L."""
    session = _make_session()
    match = _create_match(session)

    _create_bet(
        session, match, outcome=BetOutcome.WIN, pnl="100.00",
        stake="100.00", settled_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
    )
    _create_bet(
        session, match, selection="draw", outcome=BetOutcome.LOSS, pnl="-30.00",
        stake="30.00", settled_at=datetime(2026, 4, 9, 18, 0, tzinfo=UTC),
    )
    session.commit()

    cumulative = get_cumulative_pnl(session)

    assert len(cumulative) == 2
    assert cumulative[0][1] == Decimal("100.00")
    assert cumulative[1][1] == Decimal("70.00")


# ---------- calculate_roi ----------


def test_roi_positive() -> None:
    """ROI for profitable trading."""
    session = _make_session()
    match = _create_match(session)
    _create_bet(
        session, match, outcome=BetOutcome.WIN, pnl="50.00",
        stake="100.00", settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    session.commit()

    roi = calculate_roi(session, INITIAL_BANKROLL)
    assert roi == Decimal("5.00")  # 50 / 1000 * 100


def test_roi_negative() -> None:
    """ROI for losing trading."""
    session = _make_session()
    match = _create_match(session)
    _create_bet(
        session, match, outcome=BetOutcome.LOSS, pnl="-100.00",
        stake="100.00", settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    session.commit()

    roi = calculate_roi(session, INITIAL_BANKROLL)
    assert roi == Decimal("-10.00")


def test_roi_no_bets() -> None:
    """ROI is 0 when no bets are settled."""
    session = _make_session()
    assert calculate_roi(session, INITIAL_BANKROLL) == Decimal("0.00")


# ---------- calculate_max_drawdown ----------


def test_max_drawdown_no_bets() -> None:
    """Max drawdown is 0 with no settled bets."""
    session = _make_session()
    dd, dd_pct = calculate_max_drawdown(session, INITIAL_BANKROLL)
    assert dd == Decimal("0.00")
    assert dd_pct == Decimal("0.00")


def test_max_drawdown_with_losses() -> None:
    """Max drawdown captures the largest peak-to-trough decline."""
    session = _make_session()
    match = _create_match(session)

    # Win +100 (bankroll: 1100), then lose -200 (bankroll: 900)
    # Drawdown = 1100 - 900 = 200
    _create_bet(
        session, match, outcome=BetOutcome.WIN, pnl="100.00",
        stake="100.00", settled_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
    )
    _create_bet(
        session, match, selection="away", outcome=BetOutcome.LOSS, pnl="-200.00",
        stake="200.00", settled_at=datetime(2026, 4, 9, 18, 0, tzinfo=UTC),
    )
    session.commit()

    dd, dd_pct = calculate_max_drawdown(session, INITIAL_BANKROLL)
    assert dd == Decimal("200.00")
    # 200 / 1100 * 100 = 18.18%
    assert dd_pct == Decimal("18.18")


# ---------- calculate_win_rate ----------


def test_win_rate() -> None:
    """Win rate is computed from decided bets only."""
    session = _make_session()
    match = _create_match(session)

    _create_bet(
        session, match, outcome=BetOutcome.WIN, pnl="50.00", stake="50.00",
        settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    _create_bet(
        session, match, selection="draw", outcome=BetOutcome.WIN, pnl="50.00", stake="50.00",
        settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    _create_bet(
        session, match, selection="away", outcome=BetOutcome.LOSS, pnl="-50.00", stake="50.00",
        settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    # VOID bet should not count toward win rate
    _create_bet(
        session, match, outcome=BetOutcome.VOID, pnl="0.00", stake="50.00",
        settled_at=datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    session.commit()

    rate = calculate_win_rate(session)
    assert rate == Decimal("66.67")  # 2 / 3 * 100


def test_win_rate_no_bets() -> None:
    """Win rate is 0 when no bets are decided."""
    session = _make_session()
    assert calculate_win_rate(session) == Decimal("0.00")


# ---------- get_portfolio_stats ----------


def test_portfolio_stats_comprehensive() -> None:
    """Portfolio stats returns all metrics correctly."""
    session = _make_session()
    match = _create_match(session)

    _create_bet(
        session, match, outcome=BetOutcome.WIN, pnl="100.00",
        odds_price="3.0000", stake="100.00",
        settled_at=datetime(2026, 4, 8, 18, 0, tzinfo=UTC),
    )
    _create_bet(
        session, match, selection="away", outcome=BetOutcome.LOSS, pnl="-50.00",
        odds_price="2.0000", stake="50.00",
        settled_at=datetime(2026, 4, 9, 18, 0, tzinfo=UTC),
    )
    session.commit()

    stats = get_portfolio_stats(session, INITIAL_BANKROLL)

    assert isinstance(stats, PortfolioStats)
    assert stats.total_bets == 2
    assert stats.settled_bets == 2
    assert stats.pending_bets == 0
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.total_pnl == Decimal("50.00")
    assert stats.roi == Decimal("5.00")
    assert stats.win_rate == Decimal("50.00")
    assert stats.current_bankroll == Decimal("1050.00")
    assert stats.best_day_pnl == Decimal("100.00")
    assert stats.worst_day_pnl == Decimal("-50.00")


def test_portfolio_stats_empty() -> None:
    """Portfolio stats handles no bets gracefully."""
    session = _make_session()
    stats = get_portfolio_stats(session, INITIAL_BANKROLL)

    assert stats.total_bets == 0
    assert stats.current_bankroll == INITIAL_BANKROLL
    assert stats.roi == Decimal("0.00")
    assert stats.win_rate == Decimal("0.00")
    assert stats.max_drawdown == Decimal("0.00")
