"""Tests for the Value Engine — edge calculation, Kelly Criterion, and bet filtering."""

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
from src.strategies.value_engine import (
    ValueBet,
    calculate_edge,
    find_value_bets,
    kelly_criterion,
    place_paper_bet,
    scan_for_value,
)


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def _create_match(session: Session, status: MatchStatus = MatchStatus.SCHEDULED) -> Match:
    """Helper: create a match with required parent entities."""
    league = League(name="Premier League", country="England")
    home = Team(name="Arsenal")
    away = Team(name="Spurs")
    session.add_all([league, home, away])
    session.flush()

    match = Match(
        league_id=league.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
        status=status,
    )
    session.add(match)
    session.flush()
    return match


def _add_prediction(
    session: Session,
    match_id: int,
    selection: str,
    probability: str,
) -> Prediction:
    pred = Prediction(
        match_id=match_id,
        market=MarketType.MATCH_WINNER,
        selection=selection,
        probability=Decimal(probability),
        model_version="v0.1.0",
    )
    session.add(pred)
    session.flush()
    return pred


def _add_odds(
    session: Session,
    match_id: int,
    bookmaker_id: int,
    selection: str,
    price: str,
    implied_prob: str,
) -> Odds:
    odds = Odds(
        match_id=match_id,
        bookmaker_id=bookmaker_id,
        market=MarketType.MATCH_WINNER,
        selection=selection,
        price=Decimal(price),
        implied_probability=Decimal(implied_prob),
    )
    session.add(odds)
    session.flush()
    return odds


# ---------- calculate_edge ----------


def test_calculate_edge_positive() -> None:
    """Edge is positive when model thinks outcome is more likely than bookmaker."""
    edge = calculate_edge(Decimal("0.60"), Decimal("0.50"))
    assert edge == Decimal("0.100000")


def test_calculate_edge_zero() -> None:
    """Edge is zero when model and bookmaker agree."""
    edge = calculate_edge(Decimal("0.50"), Decimal("0.50"))
    assert edge == Decimal("0.000000")


def test_calculate_edge_negative() -> None:
    """Edge is negative when model thinks outcome is less likely than bookmaker."""
    edge = calculate_edge(Decimal("0.40"), Decimal("0.50"))
    assert edge == Decimal("-0.100000")


def test_calculate_edge_precision() -> None:
    """Edge is rounded to 6 decimal places."""
    edge = calculate_edge(Decimal("0.6543219"), Decimal("0.1234567"))
    assert edge == Decimal("0.530865")


# ---------- kelly_criterion ----------


def test_kelly_full() -> None:
    """Full Kelly for a clear value bet."""
    # Odds 2.50 (b=1.5), predicted prob 0.60
    # Full Kelly = (1.5 * 0.6 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333...
    frac = kelly_criterion(Decimal("0.60"), Decimal("2.50"), fraction=Decimal("1.0"))
    assert frac == Decimal("0.333333")


def test_kelly_quarter() -> None:
    """Quarter Kelly reduces stake by 75%."""
    frac = kelly_criterion(Decimal("0.60"), Decimal("2.50"), fraction=Decimal("0.25"))
    # 0.333... * 0.25 = 0.083333
    assert frac == Decimal("0.083333")


def test_kelly_no_edge() -> None:
    """Kelly returns 0 when there is no edge."""
    # Odds 2.00 (b=1.0), predicted prob 0.50
    # Full Kelly = (1.0 * 0.5 - 0.5) / 1.0 = 0 → returns 0
    frac = kelly_criterion(Decimal("0.50"), Decimal("2.00"))
    assert frac == Decimal("0")


def test_kelly_negative_edge() -> None:
    """Kelly returns 0 when edge is negative (don't bet)."""
    frac = kelly_criterion(Decimal("0.30"), Decimal("2.00"))
    assert frac == Decimal("0")


def test_kelly_odds_at_one() -> None:
    """Kelly returns 0 when odds equal 1.0 (b=0)."""
    frac = kelly_criterion(Decimal("0.90"), Decimal("1.00"))
    assert frac == Decimal("0")


def test_kelly_high_edge() -> None:
    """Kelly handles strong value correctly."""
    # Odds 5.00 (b=4.0), predicted prob 0.40
    # Full Kelly = (4.0 * 0.4 - 0.6) / 4.0 = (1.6 - 0.6) / 4.0 = 0.25
    frac = kelly_criterion(Decimal("0.40"), Decimal("5.00"), fraction=Decimal("1.0"))
    assert frac == Decimal("0.250000")


# ---------- find_value_bets ----------


def test_find_value_bets_returns_qualifying_bets() -> None:
    """Bets exceeding minimum edge are returned."""
    session = _make_session()
    match = _create_match(session)
    bookmaker = Bookmaker(name="Bet365", key="bet365")
    session.add(bookmaker)
    session.flush()

    # Model says 60% home, bookmaker implies 47.6% (odds 2.10)
    _add_prediction(session, match.id, "home", "0.600000")
    _add_odds(session, match.id, bookmaker.id, "home", "2.1000", "0.476190")
    session.commit()

    bets = find_value_bets(
        session,
        match.id,
        min_edge=Decimal("0.05"),
        bankroll=Decimal("1000.00"),
    )

    assert len(bets) == 1
    vb = bets[0]
    assert vb.selection == "home"
    assert vb.edge == Decimal("0.123810")  # 0.60 - 0.47619
    assert vb.recommended_stake > Decimal("0")
    assert vb.bookmaker_id == bookmaker.id


def test_find_value_bets_filters_below_threshold() -> None:
    """Bets below minimum edge are excluded."""
    session = _make_session()
    match = _create_match(session)
    bookmaker = Bookmaker(name="Betfair", key="betfair")
    session.add(bookmaker)
    session.flush()

    # Model says 50%, bookmaker implies 48% — edge is only 2%
    _add_prediction(session, match.id, "home", "0.500000")
    _add_odds(session, match.id, bookmaker.id, "home", "2.0833", "0.480000")
    session.commit()

    bets = find_value_bets(
        session,
        match.id,
        min_edge=Decimal("0.05"),
        bankroll=Decimal("1000.00"),
    )

    assert len(bets) == 0


def test_find_value_bets_multiple_selections() -> None:
    """Multiple selections can independently qualify as value bets."""
    session = _make_session()
    match = _create_match(session)
    bookmaker = Bookmaker(name="Bet365", key="bet365")
    session.add(bookmaker)
    session.flush()

    # Home: model 60%, bookmaker 47.6% → edge 12.4% (qualifies)
    _add_prediction(session, match.id, "home", "0.600000")
    _add_odds(session, match.id, bookmaker.id, "home", "2.1000", "0.476190")

    # Draw: model 25%, bookmaker 28% → edge -3% (doesn't qualify)
    _add_prediction(session, match.id, "draw", "0.250000")
    _add_odds(session, match.id, bookmaker.id, "draw", "3.5714", "0.280000")

    # Away: model 20%, bookmaker 12% → edge 8% (qualifies)
    _add_prediction(session, match.id, "away", "0.200000")
    _add_odds(session, match.id, bookmaker.id, "away", "8.3333", "0.120000")

    session.commit()

    bets = find_value_bets(
        session,
        match.id,
        min_edge=Decimal("0.05"),
        bankroll=Decimal("1000.00"),
    )

    assert len(bets) == 2
    selections = {vb.selection for vb in bets}
    assert selections == {"home", "away"}
    # Sorted by edge descending
    assert bets[0].edge >= bets[1].edge


def test_find_value_bets_no_predictions() -> None:
    """Returns empty list when no predictions exist for the match."""
    session = _make_session()
    match = _create_match(session)
    session.commit()

    bets = find_value_bets(session, match.id, min_edge=Decimal("0.05"))
    assert bets == []


def test_find_value_bets_no_odds() -> None:
    """Returns empty list when no odds exist for the match."""
    session = _make_session()
    match = _create_match(session)
    _add_prediction(session, match.id, "home", "0.600000")
    session.commit()

    bets = find_value_bets(session, match.id, min_edge=Decimal("0.05"))
    assert bets == []


def test_find_value_bets_stake_capped_at_bankroll() -> None:
    """Recommended stake never exceeds the bankroll."""
    session = _make_session()
    match = _create_match(session)
    bookmaker = Bookmaker(name="Bet365", key="bet365")
    session.add(bookmaker)
    session.flush()

    # Very high edge to push Kelly stake high
    _add_prediction(session, match.id, "home", "0.950000")
    _add_odds(session, match.id, bookmaker.id, "home", "2.0000", "0.500000")
    session.commit()

    small_bankroll = Decimal("10.00")
    bets = find_value_bets(
        session,
        match.id,
        min_edge=Decimal("0.01"),
        bankroll=small_bankroll,
        kelly_fraction=Decimal("1.0"),
    )

    assert len(bets) == 1
    assert bets[0].recommended_stake <= small_bankroll


# ---------- scan_for_value ----------


def test_scan_for_value_scheduled_only() -> None:
    """Only scheduled matches are scanned, not finished ones."""
    session = _make_session()

    # Scheduled match with value
    match1 = _create_match(session, status=MatchStatus.SCHEDULED)
    bk = Bookmaker(name="Bet365", key="bet365")
    session.add(bk)
    session.flush()
    _add_prediction(session, match1.id, "home", "0.700000")
    _add_odds(session, match1.id, bk.id, "home", "2.0000", "0.500000")

    # Finished match with value (should be excluded)
    league2 = League(name="La Liga", country="Spain")
    h2 = Team(name="Barcelona")
    a2 = Team(name="Real Madrid")
    session.add_all([league2, h2, a2])
    session.flush()
    match2 = Match(
        league_id=league2.id,
        home_team_id=h2.id,
        away_team_id=a2.id,
        kickoff=datetime(2026, 4, 10, 20, 0, tzinfo=UTC),
        status=MatchStatus.FINISHED,
        home_goals=2,
        away_goals=1,
    )
    session.add(match2)
    session.flush()
    _add_prediction(session, match2.id, "home", "0.700000")
    _add_odds(session, match2.id, bk.id, "home", "2.0000", "0.500000")
    session.commit()

    bets = scan_for_value(
        session,
        min_edge=Decimal("0.05"),
        bankroll=Decimal("1000.00"),
    )

    match_ids = {vb.match_id for vb in bets}
    assert match1.id in match_ids
    assert match2.id not in match_ids


# ---------- place_paper_bet ----------


def test_place_paper_bet_persists() -> None:
    """Paper bet is persisted to the database with correct fields."""
    session = _make_session()
    match = _create_match(session)
    session.commit()

    vb = ValueBet(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection="home",
        predicted_probability=Decimal("0.600000"),
        implied_probability=Decimal("0.476190"),
        odds_price=Decimal("2.1000"),
        edge=Decimal("0.123810"),
        kelly_fraction=Decimal("0.083333"),
        recommended_stake=Decimal("83.33"),
        bookmaker_id=1,
    )

    bet = place_paper_bet(session, vb)

    assert bet.id is not None
    assert bet.match_id == match.id
    assert bet.selection == "home"
    assert bet.odds_price == Decimal("2.1000")
    assert bet.stake == Decimal("83.33")
    assert bet.model_probability == Decimal("0.600000")
    assert bet.implied_probability == Decimal("0.476190")
    assert bet.value_edge == Decimal("0.123810")
    assert bet.outcome == BetOutcome.PENDING

    # Verify it's actually in the DB
    result = session.get(Bet, bet.id)
    assert result is not None
    assert result.stake == Decimal("83.33")
    session.close()


def test_place_paper_bet_appears_in_match_relationship() -> None:
    """Placed bet appears in the match's bets relationship."""
    session = _make_session()
    match = _create_match(session)
    session.commit()

    vb = ValueBet(
        match_id=match.id,
        market=MarketType.MATCH_WINNER,
        selection="away",
        predicted_probability=Decimal("0.350000"),
        implied_probability=Decimal("0.250000"),
        odds_price=Decimal("4.0000"),
        edge=Decimal("0.100000"),
        kelly_fraction=Decimal("0.033333"),
        recommended_stake=Decimal("33.33"),
        bookmaker_id=1,
    )

    place_paper_bet(session, vb)
    session.refresh(match)

    assert len(match.bets) == 1
    assert match.bets[0].selection == "away"
    session.close()
