"""Feature engineering — computes match context from API-Football data.

Builds a structured context string to feed into Claude predictions,
replacing the empty additional_context with real statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import Match, Team
from src.models.elo import EloPrediction, get_team_rating, predict_match
from src.scrapers.api_football import (
    APIFootballClient,
    APIFootballError,
    get_league_id,
    _current_season,
)

logger = logging.getLogger(__name__)


@dataclass
class TeamContext:
    """Collected stats for one team."""

    name: str
    league_position: str = "?"
    points: str = "?"
    played: str = "?"
    form_l5: str = "?"  # e.g. "W W D L W"
    form_points: str = "?"  # e.g. "10/15"
    home_record: str = "?"
    away_record: str = "?"
    goals_scored_avg: str = "?"
    goals_conceded_avg: str = "?"
    elo_rating: str = "?"
    xg_per_match: str = "?"
    xga_per_match: str = "?"
    days_since_last_match: str = "?"
    momentum: str = "?"  # e.g. "↑" / "→" / "↓"


@dataclass
class MatchContext:
    """Full context for a match prediction."""

    home: TeamContext
    away: TeamContext
    h2h_summary: str = "Sin datos"
    elo_prediction: str = ""


def _form_string(fixtures: list[dict], team_id: int) -> tuple[str, str]:
    """Convert API-Football fixtures into form string like 'G G E D G' and '10/15'."""
    if not fixtures:
        return "?", "?"

    results = []
    points = 0
    for fix in fixtures[-5:]:
        teams = fix.get("teams", {})
        goals = fix.get("goals", {})

        home_goals = goals.get("home", 0) or 0
        away_goals = goals.get("away", 0) or 0

        is_home = teams.get("home", {}).get("id") == team_id
        if is_home:
            if home_goals > away_goals:
                results.append("G")
                points += 3
            elif home_goals == away_goals:
                results.append("E")
                points += 1
            else:
                results.append("D")
        else:
            if away_goals > home_goals:
                results.append("G")
                points += 3
            elif away_goals == home_goals:
                results.append("E")
                points += 1
            else:
                results.append("D")

    total = len(results) * 3
    return " ".join(results), f"{points}/{total}"


def _goals_avg(fixtures: list[dict], team_id: int) -> tuple[str, str]:
    """Calculate average goals scored and conceded from fixtures."""
    if not fixtures:
        return "?", "?"

    scored = 0
    conceded = 0
    count = 0

    for fix in fixtures:
        teams = fix.get("teams", {})
        goals = fix.get("goals", {})
        home_goals = goals.get("home", 0) or 0
        away_goals = goals.get("away", 0) or 0

        is_home = teams.get("home", {}).get("id") == team_id
        if is_home:
            scored += home_goals
            conceded += away_goals
        else:
            scored += away_goals
            conceded += home_goals
        count += 1

    if count == 0:
        return "?", "?"

    return f"{scored / count:.1f}", f"{conceded / count:.1f}"


def _h2h_summary(fixtures: list[dict], home_name: str, away_name: str) -> str:
    """Summarize H2H record from API-Football fixtures."""
    if not fixtures:
        return "Sin datos"

    home_wins = 0
    draws = 0
    away_wins = 0

    for fix in fixtures:
        goals = fix.get("goals", {})
        teams = fix.get("teams", {})
        h = goals.get("home", 0) or 0
        a = goals.get("away", 0) or 0

        fix_home = teams.get("home", {}).get("name", "")
        if fix_home == home_name:
            if h > a:
                home_wins += 1
            elif h == a:
                draws += 1
            else:
                away_wins += 1
        else:
            if a > h:
                home_wins += 1
            elif h == a:
                draws += 1
            else:
                away_wins += 1

    total = len(fixtures)
    return f"Últimos {total}: {home_name} {home_wins}G {draws}E {away_wins}D"


def _standing_for_team(standings: list[dict], team_name: str) -> dict | None:
    """Find a team in standings by name (fuzzy)."""
    for entry in standings:
        team = entry.get("team", {})
        if team.get("name", "") == team_name:
            return entry
    return None


def _find_api_team_id(client: APIFootballClient, team_name: str) -> int | None:
    """Search for API-Football team ID by name."""
    try:
        results = client.search_team(team_name)
        if results:
            return results[0].get("team", {}).get("id")
    except Exception:
        logger.debug("Could not find API-Football ID for %s", team_name)
    return None


def build_match_context(
    session: Session,
    match: Match,
) -> MatchContext:
    """Build full statistical context for a match.

    Collects data from API-Football and ELO ratings.
    Returns MatchContext with all available data.
    """
    home_team = session.get(Team, match.home_team_id)
    away_team = session.get(Team, match.away_team_id)
    home_name = home_team.name if home_team else "?"
    away_name = away_team.name if away_team else "?"
    league_name = match.league.name if match.league else ""

    home_ctx = TeamContext(name=home_name)
    away_ctx = TeamContext(name=away_name)

    # ELO ratings
    home_elo = get_team_rating(session, match.home_team_id)
    away_elo = get_team_rating(session, match.away_team_id)
    home_ctx.elo_rating = f"{home_elo:.0f}"
    away_ctx.elo_rating = f"{away_elo:.0f}"

    elo_pred = predict_match(home_elo, away_elo)
    elo_str = (
        f"Local {elo_pred.home_prob * 100:.0f}%, "
        f"Empate {elo_pred.draw_prob * 100:.0f}%, "
        f"Visitante {elo_pred.away_prob * 100:.0f}%"
    )

    h2h_summary = "Sin datos"

    # Try to get data from API-Football
    if settings.api_football_key:
        try:
            client = APIFootballClient()
            season = _current_season()
            league_id = get_league_id(league_name)

            # Standings
            if league_id:
                standings = client.get_standings(league_id, season)
                home_st = _standing_for_team(standings, home_name)
                away_st = _standing_for_team(standings, away_name)

                if home_st:
                    home_ctx.league_position = str(home_st.get("rank", "?"))
                    home_ctx.points = str(home_st.get("points", "?"))
                    home_ctx.played = str(home_st.get("all", {}).get("played", "?"))
                    h = home_st.get("home", {})
                    home_ctx.home_record = (
                        f"{h.get('win', '?')}G {h.get('draw', '?')}E {h.get('lose', '?')}D"
                    )
                if away_st:
                    away_ctx.league_position = str(away_st.get("rank", "?"))
                    away_ctx.points = str(away_st.get("points", "?"))
                    away_ctx.played = str(away_st.get("all", {}).get("played", "?"))
                    a = away_st.get("away", {})
                    away_ctx.away_record = (
                        f"{a.get('win', '?')}G {a.get('draw', '?')}E {a.get('lose', '?')}D"
                    )

            # Form + goals (requires API-Football team IDs)
            home_api_id = _find_api_team_id(client, home_name)
            away_api_id = _find_api_team_id(client, away_name)

            if home_api_id:
                home_fixtures = client.get_team_form(home_api_id, last=10)
                home_ctx.form_l5, home_ctx.form_points = _form_string(home_fixtures, home_api_id)
                home_ctx.goals_scored_avg, home_ctx.goals_conceded_avg = _goals_avg(
                    home_fixtures, home_api_id
                )

            if away_api_id:
                away_fixtures = client.get_team_form(away_api_id, last=10)
                away_ctx.form_l5, away_ctx.form_points = _form_string(away_fixtures, away_api_id)
                away_ctx.goals_scored_avg, away_ctx.goals_conceded_avg = _goals_avg(
                    away_fixtures, away_api_id
                )

            # H2H
            if home_api_id and away_api_id:
                h2h_fixtures = client.get_h2h(home_api_id, away_api_id, last=6)
                h2h_summary = _h2h_summary(h2h_fixtures, home_name, away_name)

        except APIFootballError:
            logger.warning("API-Football unavailable — using ELO only")
        except Exception:
            logger.exception("Error fetching API-Football data")

    return MatchContext(
        home=home_ctx,
        away=away_ctx,
        h2h_summary=h2h_summary,
        elo_prediction=elo_str,
    )


def format_context_for_prompt(ctx: MatchContext) -> str:
    """Format MatchContext into a string for Claude's additional_context."""
    h = ctx.home
    a = ctx.away

    return (
        f"Datos estadísticos (verificados — usá estos números, no inventes):\n"
        f"\n"
        f"LOCAL: {h.name}\n"
        f"  - Posición: {h.league_position}° ({h.points} pts en {h.played} partidos)\n"
        f"  - Últimos 5: {h.form_l5} ({h.form_points} pts)\n"
        f"  - En casa: {h.home_record}\n"
        f"  - Goles últimos 10: {h.goals_scored_avg} a favor, {h.goals_conceded_avg} en contra\n"
        f"  - xG/partido: {h.xg_per_match} | xGA/partido: {h.xga_per_match}\n"
        f"  - Días desde último partido: {h.days_since_last_match} | Momentum: {h.momentum}\n"
        f"  - ELO: {h.elo_rating}\n"
        f"\n"
        f"VISITANTE: {a.name}\n"
        f"  - Posición: {a.league_position}° ({a.points} pts en {a.played} partidos)\n"
        f"  - Últimos 5: {a.form_l5} ({a.form_points} pts)\n"
        f"  - De visitante: {a.away_record}\n"
        f"  - Goles últimos 10: {a.goals_scored_avg} a favor, {a.goals_conceded_avg} en contra\n"
        f"  - xG/partido: {a.xg_per_match} | xGA/partido: {a.xga_per_match}\n"
        f"  - Días desde último partido: {a.days_since_last_match} | Momentum: {a.momentum}\n"
        f"  - ELO: {a.elo_rating}\n"
        f"\n"
        f"H2H: {ctx.h2h_summary}\n"
        f"\n"
        f"MODELO BASELINE ELO: {ctx.elo_prediction}\n"
    )
