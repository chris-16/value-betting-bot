"""Poisson prediction model — xG-based independent Poisson for football match outcomes.

Implements attack/defense strength ratings from xG data (falling back to actual goals),
then uses independent Poisson distributions to estimate P(home_win), P(draw), P(away_win).

Based on academic research: Walsh & Joshi 2024, Wharton 2023.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal

from scipy.stats import poisson
from sqlalchemy.orm import Session

from src.db.models import Match, MatchStatus, TeamXGStats

logger = logging.getLogger(__name__)

# Maximum goals to enumerate in the Poisson grid (0..MAX_GOALS)
MAX_GOALS = 7

# Default home advantage multiplier
HOME_ADVANTAGE = Decimal("1.25")

# Exponential decay half-life in matches (recent matches weighted more)
DECAY_HALF_LIFE = 19  # ~half a season


@dataclass
class TeamStrength:
    """Attack and defense strength ratings for a team."""

    team_id: int
    team_name: str
    attack: float  # >1 means better than average attack
    defense: float  # <1 means better than average defense
    matches_used: int = 0


@dataclass
class GoalProbabilities:
    """Poisson model output — predicted probabilities for a match."""

    home_win: float
    draw: float
    away_win: float
    home_lambda: float
    away_lambda: float
    score_matrix: list[list[float]] = field(default_factory=list)

    def as_decimal_tuple(self) -> tuple[Decimal, Decimal, Decimal]:
        """Return (home, draw, away) as Decimals quantized to 6 places."""
        return (
            Decimal(str(round(self.home_win, 6))),
            Decimal(str(round(self.draw, 6))),
            Decimal(str(round(self.away_win, 6))),
        )


class PoissonPredictor:
    """Independent Poisson model for football match prediction.

    Computes team-level attack/defense ratings from xG data (or actual goals
    as fallback), then predicts match outcomes using independent Poisson
    distributions for home and away goal counts.
    """

    def __init__(self, home_advantage: float = float(HOME_ADVANTAGE)) -> None:
        self._home_advantage = home_advantage
        self._team_strengths: dict[int, TeamStrength] = {}
        self._league_avg: float = 1.35  # default average goals per team per match

    @property
    def team_strengths(self) -> dict[int, TeamStrength]:
        """Access fitted team strength ratings."""
        return self._team_strengths

    def fit(self, session: Session, league_id: int | None = None, season: str | None = None) -> None:
        """Compute attack/defense strength ratings from xG data or match results.

        Uses xG stats from TeamXGStats when available, falls back to actual
        goals from finished matches.

        Args:
            session: Active SQLAlchemy session.
            league_id: Optional league filter.
            season: Optional season filter for xG stats.
        """
        # Try xG-based ratings first
        xg_query = session.query(TeamXGStats)
        if league_id is not None:
            xg_query = xg_query.filter(TeamXGStats.league_id == league_id)
        if season is not None:
            xg_query = xg_query.filter(TeamXGStats.season == season)

        xg_stats = xg_query.all()

        if xg_stats:
            self._fit_from_xg(xg_stats)
        else:
            self._fit_from_goals(session, league_id)

    def _fit_from_xg(self, xg_stats: list[TeamXGStats]) -> None:
        """Compute ratings from xG statistics."""
        if not xg_stats:
            return

        # Calculate league averages
        total_xg = sum(float(s.xg_per_match) for s in xg_stats)
        total_xga = sum(float(s.xga_per_match) for s in xg_stats)
        n_teams = len(xg_stats)

        if n_teams == 0:
            return

        avg_xg = total_xg / n_teams
        avg_xga = total_xga / n_teams
        self._league_avg = (avg_xg + avg_xga) / 2

        if self._league_avg <= 0:
            self._league_avg = 1.35

        for stat in xg_stats:
            xg_pm = float(stat.xg_per_match)
            xga_pm = float(stat.xga_per_match)

            attack = xg_pm / avg_xg if avg_xg > 0 else 1.0
            defense = xga_pm / avg_xga if avg_xga > 0 else 1.0

            self._team_strengths[stat.team_id] = TeamStrength(
                team_id=stat.team_id,
                team_name=f"team_{stat.team_id}",
                attack=attack,
                defense=defense,
                matches_used=stat.matches_played,
            )

        logger.info(
            "Fitted Poisson from xG: %d teams, league avg=%.3f",
            len(self._team_strengths),
            self._league_avg,
        )

    def _fit_from_goals(self, session: Session, league_id: int | None = None) -> None:
        """Fallback: compute ratings from actual goals in finished matches."""
        query = session.query(Match).filter(
            Match.status == MatchStatus.FINISHED,
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
        )
        if league_id is not None:
            query = query.filter(Match.league_id == league_id)

        matches = query.order_by(Match.kickoff).all()
        if not matches:
            logger.warning("No finished matches found for Poisson fit")
            return

        # Accumulate goals per team with exponential decay
        team_scored: dict[int, float] = {}
        team_conceded: dict[int, float] = {}
        team_matches: dict[int, float] = {}

        n_matches = len(matches)
        for i, m in enumerate(matches):
            # Exponential decay weight: more recent matches count more
            age = n_matches - i - 1
            weight = math.exp(-math.log(2) * age / DECAY_HALF_LIFE)

            hg = float(m.home_goals or 0)
            ag = float(m.away_goals or 0)

            for tid, scored, conceded in [
                (m.home_team_id, hg, ag),
                (m.away_team_id, ag, hg),
            ]:
                team_scored[tid] = team_scored.get(tid, 0) + scored * weight
                team_conceded[tid] = team_conceded.get(tid, 0) + conceded * weight
                team_matches[tid] = team_matches.get(tid, 0) + weight

        # League average
        total_scored = sum(team_scored.values())
        total_weight = sum(team_matches.values())
        self._league_avg = total_scored / total_weight if total_weight > 0 else 1.35

        # Per-team ratings
        for tid in team_scored:
            w = team_matches.get(tid, 1)
            avg_scored = team_scored[tid] / w
            avg_conceded = team_conceded[tid] / w

            attack = avg_scored / self._league_avg if self._league_avg > 0 else 1.0
            defense = avg_conceded / self._league_avg if self._league_avg > 0 else 1.0

            self._team_strengths[tid] = TeamStrength(
                team_id=tid,
                team_name=f"team_{tid}",
                attack=attack,
                defense=defense,
                matches_used=int(team_matches.get(tid, 0)),
            )

        logger.info(
            "Fitted Poisson from goals: %d teams, league avg=%.3f",
            len(self._team_strengths),
            self._league_avg,
        )

    def _fit_from_goals_list(self, matches: list[Match]) -> None:
        """Fit from a pre-fetched list of Match objects (used by backtester)."""
        if not matches:
            return

        team_scored: dict[int, float] = {}
        team_conceded: dict[int, float] = {}
        team_matches: dict[int, float] = {}

        n_matches = len(matches)
        for i, m in enumerate(matches):
            age = n_matches - i - 1
            weight = math.exp(-math.log(2) * age / DECAY_HALF_LIFE)

            hg = float(m.home_goals or 0)
            ag = float(m.away_goals or 0)

            for tid, scored, conceded in [
                (m.home_team_id, hg, ag),
                (m.away_team_id, ag, hg),
            ]:
                team_scored[tid] = team_scored.get(tid, 0) + scored * weight
                team_conceded[tid] = team_conceded.get(tid, 0) + conceded * weight
                team_matches[tid] = team_matches.get(tid, 0) + weight

        total_scored = sum(team_scored.values())
        total_weight = sum(team_matches.values())
        self._league_avg = total_scored / total_weight if total_weight > 0 else 1.35

        for tid in team_scored:
            w = team_matches.get(tid, 1)
            avg_scored = team_scored[tid] / w
            avg_conceded = team_conceded[tid] / w

            attack = avg_scored / self._league_avg if self._league_avg > 0 else 1.0
            defense = avg_conceded / self._league_avg if self._league_avg > 0 else 1.0

            self._team_strengths[tid] = TeamStrength(
                team_id=tid,
                team_name=f"team_{tid}",
                attack=attack,
                defense=defense,
                matches_used=int(team_matches.get(tid, 0)),
            )

    def predict(self, home_team_id: int, away_team_id: int) -> GoalProbabilities:
        """Predict match outcome using independent Poisson model.

        Args:
            home_team_id: DB ID of the home team.
            away_team_id: DB ID of the away team.

        Returns:
            GoalProbabilities with P(home_win), P(draw), P(away_win).
        """
        home_strength = self._team_strengths.get(home_team_id)
        away_strength = self._team_strengths.get(away_team_id)

        # Default to average team if not found
        home_attack = home_strength.attack if home_strength else 1.0
        home_defense = home_strength.defense if home_strength else 1.0
        away_attack = away_strength.attack if away_strength else 1.0
        away_defense = away_strength.defense if away_strength else 1.0

        # Expected goals
        home_lambda = home_attack * away_defense * self._league_avg * self._home_advantage
        away_lambda = away_attack * home_defense * self._league_avg

        # Clamp to reasonable range
        home_lambda = max(0.2, min(5.0, home_lambda))
        away_lambda = max(0.2, min(5.0, away_lambda))

        # Build score probability matrix using independent Poisson
        score_matrix: list[list[float]] = []
        for i in range(MAX_GOALS + 1):
            row = []
            for j in range(MAX_GOALS + 1):
                p = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
                row.append(float(p))
            score_matrix.append(row)

        # Sum probabilities for each outcome
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for i in range(MAX_GOALS + 1):
            for j in range(MAX_GOALS + 1):
                p = score_matrix[i][j]
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        # Normalise to account for truncation
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total

        return GoalProbabilities(
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            score_matrix=score_matrix,
        )

    def predict_score_probs(
        self, home_team_id: int, away_team_id: int, top_n: int = 5
    ) -> list[tuple[int, int, float]]:
        """Return the top N most likely exact scores.

        Returns:
            List of (home_goals, away_goals, probability) tuples.
        """
        result = self.predict(home_team_id, away_team_id)
        scores: list[tuple[int, int, float]] = []

        for i in range(MAX_GOALS + 1):
            for j in range(MAX_GOALS + 1):
                scores.append((i, j, result.score_matrix[i][j]))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_n]
