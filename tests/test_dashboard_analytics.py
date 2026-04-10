"""Tests for dashboard analytics — ROI by league/team and model accuracy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import (
    Bet,
    BetOutcome,
    League,
    MarketType,
    Match,
    MatchStatus,
    Prediction,
    Team,
)
from src.db.session import Base
from src.strategies.paper_trading import (
    get_model_accuracy,
    get_prediction_details,
    get_roi_by_league,
    get_roi_by_team,
)

INITIAL_BANKROLL = Decimal("1000.00")


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def _create_league(
    session: Session, name: str = "Premier League", country: str = "England",
) -> League:
    """Helper: create a league."""
    league = League(name=name, country=country)
    session.add(league)
    session.flush()
    return league


def _create_team(session: Session, name: str) -> Team:
    """Helper: create a team."""
    team = Team(name=name)
    session.add(team)
    session.flush()
    return team


def _create_match(
    session: Session,
    league: League,
    home: Team,
    away: Team,
    status: MatchStatus = MatchStatus.FINISHED,
    home_goals: int | None = 2,
    away_goals: int | None = 1,
    kickoff_offset_days: int = 0,
) -> Match:
    """Helper: create a match."""
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
    outcome: BetOutcome = BetOutcome.WIN,
    pnl: str = "50.00",
    settled_at: datetime | None = None,
) -> Bet:
    """Helper: create a settled bet."""
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
        pnl=Decimal(pnl),
        settled_at=settled_at or datetime(2026, 4, 10, 18, 0, tzinfo=UTC),
    )
    session.add(bet)
    session.flush()
    return bet


def _create_prediction(
    session: Session,
    match: Match,
    selection: str = "home",
    probability: str = "0.600000",
) -> Prediction:
    """Helper: create a prediction."""
    pred = Prediction(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection=selection,
        probability=Decimal(probability),
        model_version="w5-claude-v1",
    )
    session.add(pred)
    session.flush()
    return pred


# ---------- get_roi_by_league ----------


def test_roi_by_league_single() -> None:
    """ROI for a single league is calculated correctly."""
    session = _make_session()
    league = _create_league(session, "La Liga", "Spain")
    home = _create_team(session, "Barcelona")
    away = _create_team(session, "Real Madrid")
    match = _create_match(session, league, home, away, home_goals=3, away_goals=1)

    _create_bet(
        session, match, selection="home", stake="100.00",
        pnl="100.00", outcome=BetOutcome.WIN,
    )
    _create_bet(
        session, match, selection="away", stake="50.00",
        pnl="-50.00", outcome=BetOutcome.LOSS,
    )
    session.commit()

    result = get_roi_by_league(session)

    assert len(result) == 1
    assert result[0].name == "La Liga"
    assert result[0].total_bets == 2
    assert result[0].wins == 1
    assert result[0].losses == 1
    assert result[0].total_pnl == Decimal("50.00")
    # ROI = 50 / 150 * 100 = 33.33%
    assert result[0].roi == Decimal("33.33")


def test_roi_by_league_multiple() -> None:
    """ROI across multiple leagues, sorted by P&L descending."""
    session = _make_session()
    epl = _create_league(session, "Premier League", "England")
    la_liga = _create_league(session, "La Liga", "Spain")

    team_a = _create_team(session, "Team A")
    team_b = _create_team(session, "Team B")
    team_c = _create_team(session, "Team C")
    team_d = _create_team(session, "Team D")

    match_epl = _create_match(session, epl, team_a, team_b, home_goals=2, away_goals=0)
    match_liga = _create_match(
        session, la_liga, team_c, team_d, home_goals=0, away_goals=1, kickoff_offset_days=1,
    )

    _create_bet(
        session, match_epl, selection="home", stake="100.00",
        pnl="100.00", outcome=BetOutcome.WIN,
    )
    _create_bet(
        session, match_liga, selection="home", stake="100.00",
        pnl="-100.00", outcome=BetOutcome.LOSS,
    )
    session.commit()

    result = get_roi_by_league(session)

    assert len(result) == 2
    # EPL first (positive P&L)
    assert result[0].name == "Premier League"
    assert result[0].roi == Decimal("100.00")
    # La Liga second (negative P&L)
    assert result[1].name == "La Liga"
    assert result[1].roi == Decimal("-100.00")


def test_roi_by_league_empty() -> None:
    """Returns empty list when no settled bets exist."""
    session = _make_session()
    result = get_roi_by_league(session)
    assert result == []


# ---------- get_roi_by_team ----------


def test_roi_by_team_attributes_both_teams() -> None:
    """Each bet is attributed to both home and away teams."""
    session = _make_session()
    league = _create_league(session)
    home = _create_team(session, "Arsenal")
    away = _create_team(session, "Chelsea")
    match = _create_match(session, league, home, away, home_goals=2, away_goals=0)

    _create_bet(
        session, match, selection="home", stake="100.00",
        pnl="100.00", outcome=BetOutcome.WIN,
    )
    session.commit()

    result = get_roi_by_team(session)

    assert len(result) == 2
    names = {r.name for r in result}
    assert names == {"Arsenal", "Chelsea"}
    # Both teams should show the same bet
    for seg in result:
        assert seg.total_bets == 1


def test_roi_by_team_empty() -> None:
    """Returns empty list when no settled bets exist."""
    session = _make_session()
    result = get_roi_by_team(session)
    assert result == []


# ---------- get_model_accuracy ----------


def test_model_accuracy_all_correct() -> None:
    """100% accuracy when all predictions match results."""
    session = _make_session()
    league = _create_league(session)
    home = _create_team(session, "Team A")
    away = _create_team(session, "Team B")
    match = _create_match(session, league, home, away, home_goals=2, away_goals=0)

    _create_prediction(session, match, selection="home", probability="0.650000")
    session.commit()

    stats = get_model_accuracy(session)

    assert stats.total_predictions == 1
    assert stats.correct_predictions == 1
    assert stats.accuracy_pct == Decimal("100.00")


def test_model_accuracy_mixed() -> None:
    """Accuracy reflects correct proportion of predictions."""
    session = _make_session()
    league = _create_league(session)
    home = _create_team(session, "Team A")
    away = _create_team(session, "Team B")

    # Match: home win (2-0)
    match = _create_match(session, league, home, away, home_goals=2, away_goals=0)

    # Correct prediction
    _create_prediction(session, match, selection="home", probability="0.550000")
    # Incorrect prediction
    _create_prediction(session, match, selection="away", probability="0.300000")
    # Incorrect prediction
    _create_prediction(session, match, selection="draw", probability="0.150000")
    session.commit()

    stats = get_model_accuracy(session)

    assert stats.total_predictions == 3
    assert stats.correct_predictions == 1
    assert stats.accuracy_pct == Decimal("33.33")


def test_model_accuracy_no_predictions() -> None:
    """Returns zero stats when no predictions exist."""
    session = _make_session()
    stats = get_model_accuracy(session)

    assert stats.total_predictions == 0
    assert stats.correct_predictions == 0
    assert stats.accuracy_pct == Decimal("0.00")


def test_model_accuracy_ignores_unfinished() -> None:
    """Only evaluates predictions for finished matches."""
    session = _make_session()
    league = _create_league(session)
    home = _create_team(session, "Team A")
    away = _create_team(session, "Team B")

    # Scheduled match (not finished)
    match = _create_match(
        session, league, home, away,
        status=MatchStatus.SCHEDULED, home_goals=None, away_goals=None,
    )
    _create_prediction(session, match, selection="home", probability="0.700000")
    session.commit()

    stats = get_model_accuracy(session)

    assert stats.total_predictions == 0


# ---------- get_prediction_details ----------


def test_prediction_details_returns_tuples() -> None:
    """Returns correct detail tuples for finished match predictions."""
    session = _make_session()
    league = _create_league(session)
    home = _create_team(session, "Liverpool")
    away = _create_team(session, "Everton")
    match = _create_match(session, league, home, away, home_goals=1, away_goals=1)

    _create_prediction(session, match, selection="draw", probability="0.350000")
    _create_prediction(session, match, selection="home", probability="0.400000")
    session.commit()

    details = get_prediction_details(session)

    assert len(details) == 2
    # Each tuple: (selection, match_label, probability, is_correct)
    selections = {d[0] for d in details}
    assert "draw" in selections
    assert "home" in selections

    # Draw prediction should be correct (1-1)
    for sel, label, prob, correct in details:
        if sel == "draw":
            assert correct is True
            assert label == "Liverpool vs Everton"
        elif sel == "home":
            assert correct is False


def test_prediction_details_empty() -> None:
    """Returns empty list when no finished match predictions exist."""
    session = _make_session()
    details = get_prediction_details(session)
    assert details == []
