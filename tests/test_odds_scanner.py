"""Tests for the Odds API scanner / persistence layer."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import Bookmaker, League, MarketType, Match, MatchStatus, Odds, Team
from src.db.session import Base
from src.scrapers.odds_api import (
    LEAGUE_SPORT_KEYS,
    OddsAPIClient,
    OddsAPIError,
    _safe_decimal,
    persist_odds,
    scan_all_leagues,
)


def _make_session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


# ---------------------------------------------------------------------------
# Sample API response fixture
# ---------------------------------------------------------------------------

SAMPLE_EVENT: dict[str, Any] = {
    "id": "abc123",
    "sport_key": "soccer_epl",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "commence_time": "2026-04-15T15:00:00Z",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "Chelsea", "price": 3.50},
                        {"name": "Draw", "price": 3.20},
                    ],
                }
            ],
        },
        {
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.15},
                        {"name": "Chelsea", "price": 3.40},
                        {"name": "Draw", "price": 3.25},
                    ],
                }
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Tests: persist_odds
# ---------------------------------------------------------------------------


def test_persist_odds_creates_entities() -> None:
    """Leagues, teams, bookmakers, matches, and odds should all be created."""
    session = _make_session()
    events = [SAMPLE_EVENT]

    count = persist_odds(session, "soccer_epl", events)

    # 2 bookmakers × 3 outcomes = 6 odds rows
    assert count == 6

    assert session.query(League).count() == 1
    assert session.query(Team).count() == 2
    assert session.query(Bookmaker).count() == 2
    assert session.query(Match).count() == 1
    assert session.query(Odds).count() == 6

    league = session.query(League).first()
    assert league is not None
    assert league.name == "Premier League"
    assert league.external_id == "soccer_epl"

    match = session.query(Match).first()
    assert match is not None
    assert match.status == MatchStatus.SCHEDULED
    assert match.external_id == "abc123"

    session.close()


def test_persist_odds_no_duplicates() -> None:
    """Running persist_odds twice with the same data should not create duplicate odds.

    Note: duplicates are prevented per retrieved_at timestamp. Since
    persist_odds uses datetime.now(UTC) internally, two rapid calls will
    have different timestamps but the same match/teams/bookmakers are reused.
    """
    session = _make_session()
    events = [SAMPLE_EVENT]

    persist_odds(session, "soccer_epl", events)

    # Second call should still create 6 new odds (different retrieved_at)
    # but should NOT create duplicate leagues/teams/bookmakers/matches
    persist_odds(session, "soccer_epl", events)

    assert session.query(League).count() == 1
    assert session.query(Team).count() == 2
    assert session.query(Bookmaker).count() == 2
    assert session.query(Match).count() == 1
    session.close()


def test_persist_odds_skips_invalid_events() -> None:
    """Events with missing required fields should be skipped."""
    session = _make_session()
    bad_event: dict[str, Any] = {"id": "", "home_team": "", "away_team": "", "commence_time": ""}
    count = persist_odds(session, "soccer_epl", [bad_event])
    assert count == 0
    session.close()


def test_persist_odds_correct_implied_probability() -> None:
    """Implied probability should be 1 / price."""
    session = _make_session()
    persist_odds(session, "soccer_epl", [SAMPLE_EVENT])

    odds = session.query(Odds).filter(Odds.selection == "home").first()
    assert odds is not None
    expected = Decimal("1") / Decimal("2.10")
    # Allow small rounding difference
    assert abs(odds.implied_probability - expected) < Decimal("0.0001")
    session.close()


def test_persist_odds_market_type() -> None:
    """All h2h odds should map to MATCH_WINNER market type."""
    session = _make_session()
    persist_odds(session, "soccer_epl", [SAMPLE_EVENT])

    for odds in session.query(Odds).all():
        assert odds.market == MarketType.MATCH_WINNER

    session.close()


def test_persist_odds_selections() -> None:
    """Selections should be 'home', 'away', and 'draw'."""
    session = _make_session()
    persist_odds(session, "soccer_epl", [SAMPLE_EVENT])

    selections = {o.selection for o in session.query(Odds).all()}
    assert selections == {"home", "away", "draw"}
    session.close()


# ---------------------------------------------------------------------------
# Tests: _safe_decimal
# ---------------------------------------------------------------------------


def test_safe_decimal_float() -> None:
    assert _safe_decimal(2.10) == Decimal("2.1")


def test_safe_decimal_int() -> None:
    assert _safe_decimal(3) == Decimal("3")


def test_safe_decimal_string() -> None:
    assert _safe_decimal("1.95") == Decimal("1.95")


def test_safe_decimal_invalid() -> None:
    assert _safe_decimal("not_a_number") == Decimal("0")


# ---------------------------------------------------------------------------
# Tests: OddsAPIClient
# ---------------------------------------------------------------------------


def test_client_requires_api_key() -> None:
    """Client should raise OddsAPIError if no API key is provided."""
    with patch("src.scrapers.odds_api.settings") as mock_settings:
        mock_settings.odds_api_key = ""
        try:
            OddsAPIClient(api_key="")
            assert False, "Expected OddsAPIError"
        except OddsAPIError:
            pass


def test_client_fetch_odds_calls_api() -> None:
    """fetch_odds should make a GET request with correct params."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [SAMPLE_EVENT]
    mock_response.headers = {"x-requests-remaining": "450", "x-requests-used": "50"}

    with patch("src.scrapers.odds_api.requests.get", return_value=mock_response) as mock_get:
        client = OddsAPIClient(api_key="test-key-123")
        result = client.fetch_odds("soccer_epl")

    assert result == [SAMPLE_EVENT]
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert "soccer_epl" in call_args[0][0]
    assert call_args[1]["params"]["apiKey"] == "test-key-123"
    assert call_args[1]["params"]["markets"] == "h2h"


def test_client_raises_on_http_error() -> None:
    """Client should raise OddsAPIError on non-200 response."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_response.headers = {}

    with patch("src.scrapers.odds_api.requests.get", return_value=mock_response):
        client = OddsAPIClient(api_key="bad-key")
        try:
            client.fetch_odds("soccer_epl")
            assert False, "Expected OddsAPIError"
        except OddsAPIError as exc:
            assert "401" in str(exc)


# ---------------------------------------------------------------------------
# Tests: scan_all_leagues
# ---------------------------------------------------------------------------


def test_scan_all_leagues_iterates_all_keys() -> None:
    """scan_all_leagues should fetch odds for every configured league."""
    session = _make_session()
    mock_client = MagicMock(spec=OddsAPIClient)
    mock_client.fetch_odds.return_value = [SAMPLE_EVENT]

    results = scan_all_leagues(session, mock_client)

    assert len(results) == len(LEAGUE_SPORT_KEYS)
    assert mock_client.fetch_odds.call_count == len(LEAGUE_SPORT_KEYS)
    # Each league call should have persisted some odds
    for sport_key in LEAGUE_SPORT_KEYS:
        assert sport_key in results
    session.close()


def test_scan_all_leagues_handles_api_error() -> None:
    """If one league fails, others should still be processed."""
    session = _make_session()
    mock_client = MagicMock(spec=OddsAPIClient)

    call_count = 0

    def side_effect(sport_key: str) -> list[dict[str, Any]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OddsAPIError("Rate limited")
        return [SAMPLE_EVENT]

    mock_client.fetch_odds.side_effect = side_effect

    results = scan_all_leagues(session, mock_client)

    # First league should have 0 (error), rest should have data
    assert len(results) == len(LEAGUE_SPORT_KEYS)
    zero_count = sum(1 for v in results.values() if v == 0)
    assert zero_count == 1
    session.close()


# ---------------------------------------------------------------------------
# Tests: League configuration
# ---------------------------------------------------------------------------


def test_league_sport_keys_has_five_leagues() -> None:
    """We should have exactly the top 5 European leagues configured."""
    assert len(LEAGUE_SPORT_KEYS) == 5
    assert "soccer_epl" in LEAGUE_SPORT_KEYS
    assert "soccer_spain_la_liga" in LEAGUE_SPORT_KEYS
    assert "soccer_germany_bundesliga" in LEAGUE_SPORT_KEYS
    assert "soccer_italy_serie_a" in LEAGUE_SPORT_KEYS
    assert "soccer_france_ligue_one" in LEAGUE_SPORT_KEYS
