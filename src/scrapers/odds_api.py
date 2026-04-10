"""Client for The Odds API — fetches pre-match odds for European football leagues."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import Bookmaker, League, MarketType, Match, MatchStatus, Odds, Team

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# Top 5 European football leagues — The Odds API sport keys
LEAGUE_SPORT_KEYS: dict[str, dict[str, str]] = {
    "soccer_epl": {"name": "Premier League", "country": "England"},
    "soccer_spain_la_liga": {"name": "La Liga", "country": "Spain"},
    "soccer_germany_bundesliga": {"name": "Bundesliga", "country": "Germany"},
    "soccer_italy_serie_a": {"name": "Serie A", "country": "Italy"},
    "soccer_france_ligue_one": {"name": "Ligue 1", "country": "France"},
}

# Mapping from The Odds API market keys to our MarketType enum
MARKET_KEY_MAP: dict[str, MarketType] = {
    "h2h": MarketType.MATCH_WINNER,
}

# Mapping from The Odds API outcome names to our selection strings
OUTCOME_SELECTION_MAP: dict[str, str] = {
    "Home": "home",
    "Away": "away",
    "Draw": "draw",
}


class OddsAPIError(Exception):
    """Raised when The Odds API returns an error response."""


class OddsAPIClient:
    """HTTP client for The Odds API v4."""

    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        self.api_key = api_key or settings.odds_api_key
        self.timeout = timeout
        if not self.api_key:
            raise OddsAPIError("ODDS_API_KEY is not configured")

    def _get(self, path: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Make authenticated GET request to The Odds API."""
        url = f"{BASE_URL}{path}"
        request_params = {"apiKey": self.api_key}
        if params:
            request_params.update(params)

        response = requests.get(url, params=request_params, timeout=self.timeout)

        remaining = response.headers.get("x-requests-remaining", "?")
        used = response.headers.get("x-requests-used", "?")
        logger.info("Odds API quota — used: %s, remaining: %s", used, remaining)

        if response.status_code != 200:
            raise OddsAPIError(
                f"Odds API returned {response.status_code}: {response.text[:500]}"
            )

        return response.json()  # type: ignore[no-any-return]

    def fetch_odds(self, sport_key: str) -> list[dict[str, Any]]:
        """Fetch pre-match h2h odds for a given sport key.

        Returns raw JSON events list from The Odds API.
        """
        return self._get(
            f"/sports/{sport_key}/odds",
            params={
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )


def _get_or_create_league(session: Session, sport_key: str) -> League:
    """Get existing league by external_id or create a new one."""
    league = session.query(League).filter(League.external_id == sport_key).first()
    if league is None:
        info = LEAGUE_SPORT_KEYS[sport_key]
        league = League(name=info["name"], country=info["country"], external_id=sport_key)
        session.add(league)
        session.flush()
        logger.info("Created league: %s", league)
    return league


def _get_or_create_team(session: Session, team_name: str) -> Team:
    """Get existing team by name or create a new one."""
    team = session.query(Team).filter(Team.name == team_name).first()
    if team is None:
        team = Team(name=team_name)
        session.add(team)
        session.flush()
        logger.debug("Created team: %s", team)
    return team


def _get_or_create_bookmaker(session: Session, key: str, title: str) -> Bookmaker:
    """Get existing bookmaker by key or create a new one."""
    bookmaker = session.query(Bookmaker).filter(Bookmaker.key == key).first()
    if bookmaker is None:
        bookmaker = Bookmaker(name=title, key=key)
        session.add(bookmaker)
        session.flush()
        logger.debug("Created bookmaker: %s", bookmaker)
    return bookmaker


def _get_or_create_match(
    session: Session,
    league: League,
    home_team: Team,
    away_team: Team,
    kickoff: datetime,
    external_id: str,
) -> Match:
    """Get existing match by external_id or create a new one."""
    match = session.query(Match).filter(Match.external_id == external_id).first()
    if match is None:
        match = Match(
            league_id=league.id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            kickoff=kickoff,
            status=MatchStatus.SCHEDULED,
            external_id=external_id,
        )
        session.add(match)
        session.flush()
        logger.debug("Created match: %s vs %s", home_team.name, away_team.name)
    return match


def _safe_decimal(value: float | int | str) -> Decimal:
    """Convert a price value to Decimal safely."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _odds_exists(
    session: Session,
    match_id: int,
    bookmaker_id: int,
    market: MarketType,
    selection: str,
    retrieved_at: datetime,
) -> bool:
    """Check if an identical odds snapshot already exists."""
    return (
        session.query(Odds)
        .filter(
            Odds.match_id == match_id,
            Odds.bookmaker_id == bookmaker_id,
            Odds.market == market,
            Odds.selection == selection,
            Odds.retrieved_at == retrieved_at,
        )
        .first()
        is not None
    )


def persist_odds(session: Session, sport_key: str, events: list[dict[str, Any]]) -> int:
    """Parse API events and persist odds to the database.

    Returns the number of new odds rows inserted.
    """
    league = _get_or_create_league(session, sport_key)
    now = datetime.now(UTC)
    inserted = 0

    for event in events:
        event_id: str = event.get("id", "")
        home_name: str = event.get("home_team", "")
        away_name: str = event.get("away_team", "")
        commence_str: str = event.get("commence_time", "")

        if not all([event_id, home_name, away_name, commence_str]):
            logger.warning("Skipping event with missing fields: %s", event_id)
            continue

        # Parse ISO 8601 kickoff time
        try:
            kickoff = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            logger.warning("Skipping event with unparseable time: %s", commence_str)
            continue

        home_team = _get_or_create_team(session, home_name)
        away_team = _get_or_create_team(session, away_name)
        match = _get_or_create_match(session, league, home_team, away_team, kickoff, event_id)

        bookmakers_data: list[dict[str, Any]] = event.get("bookmakers", [])
        for bm_data in bookmakers_data:
            bm_key: str = bm_data.get("key", "")
            bm_title: str = bm_data.get("title", bm_key)
            if not bm_key:
                continue

            bookmaker = _get_or_create_bookmaker(session, bm_key, bm_title)

            markets: list[dict[str, Any]] = bm_data.get("markets", [])
            for market_data in markets:
                market_key: str = market_data.get("key", "")
                market_type = MARKET_KEY_MAP.get(market_key)
                if market_type is None:
                    continue

                outcomes: list[dict[str, Any]] = market_data.get("outcomes", [])
                for outcome in outcomes:
                    raw_name: str = outcome.get("name", "")
                    price_raw = outcome.get("price", 0)

                    # Map outcome name; for h2h the name may be the team name
                    selection = OUTCOME_SELECTION_MAP.get(raw_name, "")
                    if not selection:
                        # Try matching team names for h2h
                        if raw_name == home_name:
                            selection = "home"
                        elif raw_name == away_name:
                            selection = "away"
                        else:
                            selection = raw_name.lower()

                    price = _safe_decimal(price_raw)
                    if price <= 0:
                        continue

                    implied_prob = Decimal("1") / price

                    if _odds_exists(
                        session, match.id, bookmaker.id, market_type, selection, now
                    ):
                        continue

                    odds_row = Odds(
                        match_id=match.id,
                        bookmaker_id=bookmaker.id,
                        market=market_type,
                        selection=selection,
                        price=price,
                        implied_probability=implied_prob,
                        retrieved_at=now,
                    )
                    session.add(odds_row)
                    inserted += 1

    session.commit()
    return inserted


def scan_all_leagues(session: Session, client: OddsAPIClient | None = None) -> dict[str, int]:
    """Fetch and persist odds for all configured European leagues.

    Returns a dict mapping sport_key → number of odds rows inserted.
    """
    if client is None:
        client = OddsAPIClient()

    results: dict[str, int] = {}

    for sport_key in LEAGUE_SPORT_KEYS:
        try:
            logger.info("Fetching odds for %s ...", sport_key)
            events = client.fetch_odds(sport_key)
            count = persist_odds(session, sport_key, events)
            results[sport_key] = count
            logger.info("Persisted %d odds rows for %s", count, sport_key)
        except OddsAPIError:
            logger.exception("API error fetching %s", sport_key)
            results[sport_key] = 0
        except Exception:
            logger.exception("Unexpected error fetching %s", sport_key)
            results[sport_key] = 0

    return results
