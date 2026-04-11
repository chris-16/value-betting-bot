"""Understat xG scraper — fetches expected goals data from understat.com.

Understat embeds JSON in <script> tags. No API key required.
Covers EPL + La Liga. Leagues not on Understat fall back to actual goals.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from src.db.models import League, Match, Team, TeamXGStats

logger = logging.getLogger(__name__)

BASE_URL = "https://understat.com"

# Mapping from our league names to Understat league slugs
LEAGUE_SLUG_MAP: dict[str, str] = {
    "Premier League": "EPL",
    "La Liga": "La_Liga",
}

# Reverse mapping for display
SLUG_LEAGUE_MAP: dict[str, str] = {v: k for k, v in LEAGUE_SLUG_MAP.items()}


@dataclass
class TeamXGData:
    """Per-team xG statistics for a season."""

    team_name: str
    league: str
    season: str
    xg: Decimal
    xga: Decimal
    matches_played: int
    xg_per_match: Decimal
    xga_per_match: Decimal


@dataclass
class MatchXGData:
    """Per-match xG data."""

    home_team: str
    away_team: str
    home_xg: Decimal
    away_xg: Decimal
    home_goals: int
    away_goals: int


def _extract_json_var(html: str, var_name: str) -> list[dict] | dict:
    """Extract JSON data from Understat's embedded <script> variables.

    Understat stores data like: var teamsData = JSON.parse('...');
    """
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html)
    if not match:
        raise ValueError(f"Could not find variable '{var_name}' in page")

    # Understat encodes the JSON with hex escapes
    raw = match.group(1)
    decoded = raw.encode("utf-8").decode("unicode_escape")
    return json.loads(decoded)


class UnderstatScraper:
    """Scraper for Understat xG data."""

    def __init__(self, timeout: int = 30) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self._timeout = timeout

    def fetch_team_xg(self, league: str, season: str) -> list[TeamXGData]:
        """Fetch per-team xG stats for a league and season.

        Args:
            league: Our league name (e.g. "Premier League").
            season: Season year (e.g. "2025" for 2025/26).

        Returns:
            List of TeamXGData for each team in the league.
        """
        slug = LEAGUE_SLUG_MAP.get(league)
        if slug is None:
            logger.warning("League '%s' not available on Understat", league)
            return []

        url = f"{BASE_URL}/league/{slug}/{season}"
        logger.info("Fetching team xG from %s", url)

        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()

        teams_data = _extract_json_var(resp.text, "teamsData")
        if not isinstance(teams_data, dict):
            raise ValueError("Unexpected teamsData format")

        results: list[TeamXGData] = []
        for _team_id, team_info in teams_data.items():
            title = team_info.get("title", "Unknown")
            history = team_info.get("history", [])

            if not history:
                continue

            total_xg = Decimal("0")
            total_xga = Decimal("0")
            matches_played = len(history)

            for match_day in history:
                total_xg += Decimal(str(match_day.get("xG", 0)))
                total_xga += Decimal(str(match_day.get("xGA", 0)))

            xg_per_match = (
                (total_xg / matches_played).quantize(Decimal("0.0001"))
                if matches_played > 0
                else Decimal("0")
            )
            xga_per_match = (
                (total_xga / matches_played).quantize(Decimal("0.0001"))
                if matches_played > 0
                else Decimal("0")
            )

            results.append(
                TeamXGData(
                    team_name=title,
                    league=league,
                    season=season,
                    xg=total_xg.quantize(Decimal("0.01")),
                    xga=total_xga.quantize(Decimal("0.01")),
                    matches_played=matches_played,
                    xg_per_match=xg_per_match,
                    xga_per_match=xga_per_match,
                )
            )

        logger.info("Fetched xG for %d teams in %s %s", len(results), league, season)
        return results

    def fetch_match_xg(self, league: str, season: str) -> list[MatchXGData]:
        """Fetch per-match xG data for a league and season.

        Args:
            league: Our league name (e.g. "Premier League").
            season: Season year (e.g. "2025" for 2025/26).

        Returns:
            List of MatchXGData for each match.
        """
        slug = LEAGUE_SLUG_MAP.get(league)
        if slug is None:
            logger.warning("League '%s' not available on Understat", league)
            return []

        url = f"{BASE_URL}/league/{slug}/{season}"
        logger.info("Fetching match xG from %s", url)

        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()

        dates_data = _extract_json_var(resp.text, "datesData")
        if not isinstance(dates_data, list):
            raise ValueError("Unexpected datesData format")

        results: list[MatchXGData] = []
        for match_info in dates_data:
            if not match_info.get("isResult", False):
                continue

            results.append(
                MatchXGData(
                    home_team=match_info.get("h", {}).get("title", ""),
                    away_team=match_info.get("a", {}).get("title", ""),
                    home_xg=Decimal(str(match_info.get("xG", {}).get("h", 0))),
                    away_xg=Decimal(str(match_info.get("xG", {}).get("a", 0))),
                    home_goals=int(match_info.get("goals", {}).get("h", 0)),
                    away_goals=int(match_info.get("goals", {}).get("a", 0)),
                )
            )

        logger.info("Fetched %d match xG records for %s %s", len(results), league, season)
        return results


def persist_team_xg(session: Session, data: list[TeamXGData]) -> int:
    """Persist team xG data to the database.

    Creates or updates TeamXGStats rows. Matches teams by name.

    Returns:
        Number of rows upserted.
    """
    count = 0
    for item in data:
        # Find the team by name
        team = session.query(Team).filter(Team.name == item.team_name).first()
        if team is None:
            logger.debug("Team '%s' not found in DB — skipping xG", item.team_name)
            continue

        # Find the league
        league = session.query(League).filter(League.name == item.league).first()
        if league is None:
            logger.debug("League '%s' not found in DB — skipping", item.league)
            continue

        # Upsert
        existing = (
            session.query(TeamXGStats)
            .filter(
                TeamXGStats.team_id == team.id,
                TeamXGStats.league_id == league.id,
                TeamXGStats.season == item.season,
            )
            .first()
        )

        if existing:
            existing.xg = item.xg
            existing.xga = item.xga
            existing.xg_per_match = item.xg_per_match
            existing.xga_per_match = item.xga_per_match
            existing.matches_played = item.matches_played
        else:
            row = TeamXGStats(
                team_id=team.id,
                league_id=league.id,
                season=item.season,
                xg=item.xg,
                xga=item.xga,
                xg_per_match=item.xg_per_match,
                xga_per_match=item.xga_per_match,
                matches_played=item.matches_played,
            )
            session.add(row)

        count += 1

    session.commit()
    logger.info("Persisted xG for %d teams", count)
    return count
