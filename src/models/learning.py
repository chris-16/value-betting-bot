"""Learning loop — periodic model retraining and calibration updates.

Retrains the Poisson model on recent finished matches and refits the
probability calibrator on (prediction, outcome) pairs.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from src.db.models import Match, MatchStatus, ModelRun, Prediction
from src.models.calibration import ProbabilityCalibrator
from src.models.poisson import PoissonPredictor

logger = logging.getLogger(__name__)


def _compute_brier_score(
    predictions: list[tuple[float, int]],
) -> float:
    """Compute Brier score from (predicted_prob, actual_outcome) pairs."""
    if not predictions:
        return 1.0
    total = sum((p - a) ** 2 for p, a in predictions)
    return total / len(predictions)


def _compute_log_loss(
    predictions: list[tuple[float, int]],
) -> float:
    """Compute log loss from (predicted_prob, actual_outcome) pairs."""
    import math

    if not predictions:
        return 1.0
    eps = 1e-15
    total = 0.0
    for p, a in predictions:
        p = max(eps, min(1 - eps, p))
        total += -(a * math.log(p) + (1 - a) * math.log(1 - p))
    return total / len(predictions)


def retrain_model(session: Session) -> PoissonPredictor:
    """Retrain the Poisson model on all available data.

    1. Fetch finished matches → refit Poisson ratings (with decay weighting).
    2. Refit calibrator on (prediction, outcome) pairs.
    3. Save updated artifacts.
    4. Log Brier score + log loss to model_runs table.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        The refitted PoissonPredictor.
    """
    logger.info("Starting model retrain...")

    # 1. Refit Poisson model
    predictor = PoissonPredictor()
    predictor.fit(session)

    # 2. Build calibration data from historical predictions vs outcomes
    finished_matches = (
        session.query(Match)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
        )
        .all()
    )

    pred_outcome_pairs: list[tuple[float, int]] = []

    for match in finished_matches:
        preds = (
            session.query(Prediction)
            .filter(Prediction.match_id == match.id)
            .all()
        )

        if not preds:
            continue

        hg = match.home_goals or 0
        ag = match.away_goals or 0
        if hg > ag:
            actual = "home"
        elif hg == ag:
            actual = "draw"
        else:
            actual = "away"

        for pred in preds:
            outcome = 1 if pred.selection == actual else 0
            pred_outcome_pairs.append((float(pred.probability), outcome))

    # 3. Refit calibrator
    calibrator = ProbabilityCalibrator()
    if len(pred_outcome_pairs) >= 10:
        probs = [p for p, _ in pred_outcome_pairs]
        outcomes = [o for _, o in pred_outcome_pairs]
        calibrator.fit(probs, outcomes)
        calibrator.save()
        logger.info("Calibrator refitted on %d samples", len(pred_outcome_pairs))
    else:
        logger.info("Not enough data for calibration (%d pairs)", len(pred_outcome_pairs))

    # 4. Compute and log metrics
    brier = _compute_brier_score(pred_outcome_pairs) if pred_outcome_pairs else None
    log_loss = _compute_log_loss(pred_outcome_pairs) if pred_outcome_pairs else None

    try:
        n_teams = len(predictor.team_strengths)
        model_run = ModelRun(
            model_version="poisson-v1",
            trained_at=datetime.utcnow(),
            train_matches=len(finished_matches),
            brier_score=brier,
            log_loss=log_loss,
            notes=f"Retrained on {len(finished_matches)} matches, {n_teams} teams, {len(pred_outcome_pairs)} pred pairs",
        )
        session.add(model_run)
        session.commit()
        logger.info(
            "Model run logged: brier=%.4f, log_loss=%.4f, matches=%d",
            brier or 0,
            log_loss or 0,
            len(finished_matches),
        )
    except Exception:
        logger.exception("Failed to log model run")
        session.rollback()

    return predictor
