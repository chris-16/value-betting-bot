"""Tests for SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import (
    Bet,
    BetOutcome,
    Bookmaker,
    League,
    MarketType,
    Match,
    MatchStatus,
    Odds,
    Prediction,
    Team,
)
from src.db.session import Base


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_create_league() -> None:
    session = _make_session()
    league = League(name="Premier League", country="England")
    session.add(league)
    session.commit()

    result = session.get(League, league.id)
    assert result is not None
    assert result.name == "Premier League"
    assert result.country == "England"
    session.close()


def test_create_team() -> None:
    session = _make_session()
    team = Team(name="Arsenal FC")
    session.add(team)
    session.commit()

    result = session.get(Team, team.id)
    assert result is not None
    assert result.name == "Arsenal FC"
    session.close()


def test_create_match_with_relationships() -> None:
    session = _make_session()
    league = League(name="La Liga", country="Spain")
    home = Team(name="Barcelona")
    away = Team(name="Real Madrid")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 15, 20, 0, tzinfo=UTC),
        status=MatchStatus.SCHEDULED,
    )
    session.add(match)
    session.commit()

    result = session.get(Match, match.id)
    assert result is not None
    assert result.league.name == "La Liga"
    assert result.home_team.name == "Barcelona"
    assert result.away_team.name == "Real Madrid"
    assert result.status == MatchStatus.SCHEDULED
    session.close()


def test_create_bookmaker_and_odds() -> None:
    session = _make_session()
    league = League(name="Serie A", country="Italy")
    home = Team(name="AC Milan")
    away = Team(name="Inter Milan")
    bookmaker = Bookmaker(name="Bet365", key="bet365")
    session.add_all([league, home, away, bookmaker])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 20, 18, 0, tzinfo=UTC),
        status=MatchStatus.SCHEDULED,
    )
    session.add(match)
    session.flush()

    odds = Odds(
        match_id=match.id,
        bookmaker_id=bookmaker.id,
        market=MarketType.MATCH_WINNER,
        selection="home",
        price=Decimal("2.1000"),
        implied_probability=Decimal("0.476190"),
    )
    session.add(odds)
    session.commit()

    result = session.get(Odds, odds.id)
    assert result is not None
    assert result.price == Decimal("2.1000")
    assert result.bookmaker.name == "Bet365"
    session.close()


def test_create_prediction() -> None:
    session = _make_session()
    league = League(name="Bundesliga", country="Germany")
    home = Team(name="Bayern Munich")
    away = Team(name="Dortmund")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 5, 1, 15, 30, tzinfo=UTC),
        status=MatchStatus.SCHEDULED,
    )
    session.add(match)
    session.flush()

    pred = Prediction(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection="home",
        probability=Decimal("0.620000"),
        model_version="v0.1.0",
    )
    session.add(pred)
    session.commit()

    result = session.get(Prediction, pred.id)
    assert result is not None
    assert result.probability == Decimal("0.620000")
    assert result.model_version == "v0.1.0"
    session.close()


def test_create_bet_with_pnl() -> None:
    session = _make_session()
    league = League(name="Ligue 1", country="France")
    home = Team(name="PSG")
    away = Team(name="Lyon")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        status=MatchStatus.FINISHED,
        home_goals=3,
        away_goals=1,
    )
    session.add(match)
    session.flush()

    bet = Bet(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection="home",
        odds_price=Decimal("1.8000"),
        stake=Decimal("50.00"),
        model_probability=Decimal("0.650000"),
        implied_probability=Decimal("0.555556"),
        value_edge=Decimal("0.094444"),
        outcome=BetOutcome.WIN,
        pnl=Decimal("40.00"),
    )
    session.add(bet)
    session.commit()

    result = session.get(Bet, bet.id)
    assert result is not None
    assert result.outcome == BetOutcome.WIN
    assert result.pnl == Decimal("40.00")
    assert result.stake == Decimal("50.00")
    session.close()


def test_match_relationships_cascade() -> None:
    """Verify that match has correct relationship lists."""
    session = _make_session()
    league = League(name="EPL", country="England")
    home = Team(name="Liverpool")
    away = Team(name="Chelsea")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        status=MatchStatus.SCHEDULED,
    )
    session.add(match)
    session.flush()

    pred = Prediction(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection="home",
        probability=Decimal("0.500000"),
        model_version="v0.1.0",
    )
    session.add(pred)
    session.commit()

    assert len(match.predictions) == 1
    assert match.predictions[0].selection == "home"
    session.close()


def test_model_repr() -> None:
    league = League(name="Test League", country="Testland")
    assert "Test League" in repr(league)

    team = Team(name="Test FC")
    assert "Test FC" in repr(team)

    bookmaker = Bookmaker(name="TestBookie", key="testbookie")
    assert "TestBookie" in repr(bookmaker)
