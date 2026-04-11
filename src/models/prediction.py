"""Prediction wrapper — Poisson xG model with ELO ensemble.

Replaces the previous Claude LLM consensus approach with a statistical
Poisson model based on xG data, blended with ELO as a minor ensemble.

Architecture:
    1. Primary: PoissonPredictor (xG-based attack/defense ratings)
    2. Secondary: ELO baseline (15% weight)
    3. Calibration: Isotonic regression on historical predictions
    4. Fusion: 85% Poisson + 15% ELO
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import MarketType, Match, Prediction

logger = logging.getLogger(__name__)

MODEL_VERSION = "poisson-v1"


@dataclass(frozen=True)
class MatchPrediction:
    """Predicted outcome probabilities for a football match."""

    home_prob: Decimal
    draw_prob: Decimal
    away_prob: Decimal
    confidence: Decimal
    model_version: str
    reasoning: str

    def __post_init__(self) -> None:
        total = self.home_prob + self.draw_prob + self.away_prob
        # Allow small floating-point tolerance
        if abs(total - Decimal("1")) > Decimal("0.01"):
            raise ValueError(
                f"Probabilities must sum to ~1.0, got {total} "
                f"(home={self.home_prob}, draw={self.draw_prob}, away={self.away_prob})"
            )


def _safe_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely, defaulting to 0."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def _normalise_probabilities(
    home: Decimal, draw: Decimal, away: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Normalise probabilities to sum to exactly 1.0."""
    total = home + draw + away
    if total <= 0:
        return Decimal("0.333333"), Decimal("0.333334"), Decimal("0.333333")
    return (
        (home / total).quantize(Decimal("0.000001")),
        (draw / total).quantize(Decimal("0.000001")),
        (away / total).quantize(Decimal("0.000001")),
    )


def predict_match(
    session: Session,
    home_team_id: int,
    away_team_id: int,
    league_id: int | None = None,
) -> MatchPrediction:
    """Predict match outcome using Poisson model + ELO ensemble.

    Args:
        session: Active SQLAlchemy session.
        home_team_id: DB ID of the home team.
        away_team_id: DB ID of the away team.
        league_id: Optional league filter for model fitting.

    Returns:
        MatchPrediction with home/draw/away probabilities.
    """
    from src.models.calibration import ProbabilityCalibrator
    from src.models.elo import get_team_rating, predict_match as elo_predict
    from src.models.poisson import PoissonPredictor

    # 1. Poisson prediction
    predictor = PoissonPredictor()
    predictor.fit(session, league_id=league_id)
    poisson_pred = predictor.predict(home_team_id, away_team_id)

    poisson_home = poisson_pred.home_win
    poisson_draw = poisson_pred.draw
    poisson_away = poisson_pred.away_win

    # 2. Calibrate Poisson output
    calibrator = ProbabilityCalibrator()
    calibrator.load()
    if calibrator.is_fitted:
        poisson_home, poisson_draw, poisson_away = calibrator.calibrate_triple(
            poisson_home, poisson_draw, poisson_away
        )

    # 3. ELO prediction
    home_elo = get_team_rating(session, home_team_id)
    away_elo = get_team_rating(session, away_team_id)
    elo_pred = elo_predict(home_elo, away_elo)

    # 4. Ensemble: 85% Poisson + 15% ELO
    w_poisson = Decimal("0.85")
    w_elo = Decimal("0.15")

    fused_home = w_poisson * Decimal(str(poisson_home)) + w_elo * Decimal(str(elo_pred.home_prob))
    fused_draw = w_poisson * Decimal(str(poisson_draw)) + w_elo * Decimal(str(elo_pred.draw_prob))
    fused_away = w_poisson * Decimal(str(poisson_away)) + w_elo * Decimal(str(elo_pred.away_prob))

    fused_home, fused_draw, fused_away = _normalise_probabilities(
        fused_home, fused_draw, fused_away
    )

    reasoning = (
        f"Poisson(λH={poisson_pred.home_lambda:.2f}, λA={poisson_pred.away_lambda:.2f}) "
        f"→ [{poisson_pred.home_win:.0%}/{poisson_pred.draw:.0%}/{poisson_pred.away_win:.0%}] | "
        f"ELO({home_elo:.0f} vs {away_elo:.0f}) "
        f"→ [{elo_pred.home_prob:.0%}/{elo_pred.draw_prob:.0%}/{elo_pred.away_prob:.0%}]"
    )

    prediction = MatchPrediction(
        home_prob=fused_home,
        draw_prob=fused_draw,
        away_prob=fused_away,
        confidence=Decimal("0.75"),  # Poisson always produces output; static confidence
        model_version=MODEL_VERSION,
        reasoning=reasoning,
    )

    logger.info(
        "Prediction: home=%s draw=%s away=%s (%s)",
        fused_home,
        fused_draw,
        fused_away,
        MODEL_VERSION,
    )

    return prediction


def predict_and_store(
    session: Session,
    match_id: int,
    *,
    additional_context: str | None = None,
    model: str = "",
) -> MatchPrediction:
    """Predict match outcome and persist results to the database.

    Uses the Poisson + ELO ensemble model. The `model` and `additional_context`
    parameters are kept for API compatibility but are not used by the Poisson model.

    Args:
        session: Active SQLAlchemy session.
        match_id: ID of the Match to predict.
        additional_context: Unused (kept for compatibility).
        model: Unused (kept for compatibility).

    Returns:
        MatchPrediction with the predicted probabilities.

    Raises:
        ValueError: If match_id is not found.
    """
    match = session.get(Match, match_id)
    if match is None:
        raise ValueError(f"Match with id={match_id} not found")

    home_team_name = match.home_team.name
    away_team_name = match.away_team.name

    prediction = predict_match(
        session,
        match.home_team_id,
        match.away_team_id,
        league_id=match.league_id,
    )

    logger.info(
        "Poisson ensemble: [%s/%s/%s] for %s vs %s",
        prediction.home_prob * 100,
        prediction.draw_prob * 100,
        prediction.away_prob * 100,
        home_team_name,
        away_team_name,
    )

    # Persist predictions — one row per selection (home, draw, away)
    selections = [
        ("home", prediction.home_prob),
        ("draw", prediction.draw_prob),
        ("away", prediction.away_prob),
    ]

    for selection, probability in selections:
        # Check for existing prediction to avoid duplicate constraint violations
        existing = (
            session.query(Prediction)
            .filter(
                Prediction.match_id == match_id,
                Prediction.market == MarketType.MATCH_WINNER,
                Prediction.selection == selection,
                Prediction.model_version == MODEL_VERSION,
            )
            .first()
        )

        if existing is not None:
            # Update existing prediction
            existing.probability = probability
            logger.debug(
                "Updated prediction for match %d: %s = %s",
                match_id,
                selection,
                probability,
            )
        else:
            pred_row = Prediction(
                match_id=match_id,
                market=MarketType.MATCH_WINNER,
                selection=selection,
                probability=probability,
                model_version=MODEL_VERSION,
            )
            session.add(pred_row)
            logger.debug(
                "Created prediction for match %d: %s = %s",
                match_id,
                selection,
                probability,
            )

    session.commit()
    logger.info(
        "Stored predictions for match %d (%s vs %s)",
        match_id,
        home_team_name,
        away_team_name,
    )

    return prediction


def predict_upcoming_matches(
    session: Session,
    *,
    limit: int = 10,
    model: str = "",
) -> list[tuple[Match, MatchPrediction]]:
    """Predict outcomes for upcoming scheduled matches.

    Fetches the next ``limit`` scheduled matches that don't already have
    predictions with this model version, and generates predictions for each.

    Args:
        session: Active SQLAlchemy session.
        limit: Maximum number of matches to predict.
        model: Unused (kept for compatibility).

    Returns:
        List of (Match, MatchPrediction) tuples.
    """
    from sqlalchemy import and_, not_

    from src.db.models import MatchStatus

    # Find scheduled matches without existing predictions from this model
    already_predicted = (
        session.query(Prediction.match_id)
        .filter(Prediction.model_version == MODEL_VERSION)
        .distinct()
        .scalar_subquery()
    )

    matches = (
        session.query(Match)
        .filter(
            and_(
                Match.status == MatchStatus.SCHEDULED,
                not_(Match.id.in_(already_predicted)),
            )
        )
        .order_by(Match.kickoff)
        .limit(limit)
        .all()
    )

    results: list[tuple[Match, MatchPrediction]] = []

    for match in matches:
        try:
            prediction = predict_and_store(session, match.id)
            results.append((match, prediction))
        except (ValueError, Exception):
            logger.exception("Failed to predict match %d", match.id)
            continue

    logger.info("Predicted %d of %d upcoming matches", len(results), len(matches))
    return results
